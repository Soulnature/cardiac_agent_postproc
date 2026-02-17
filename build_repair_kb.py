#!/usr/bin/env python
"""
build_repair_kb.py – Build a Repair Comparison Knowledge Base.

Offline script that generates labeled before→after examples for VLM training.

For each worst-case mask:
  1. Apply standalone repair operations (no agent orchestration)
  2. Compute Dice delta (requires GT)
  3. Generate a 3-panel diff overlay (BEFORE | AFTER | DIFF)
  4. Categorize:  improved (Δ > +0.02) / degraded (Δ < -0.02) / neutral
  5. Save to repair_kb/{improved,degraded,neutral}/

The resulting KB is used at runtime (GT-free) as few-shot VLM examples.
"""
import os, sys, yaml, re, glob
import numpy as np
import cv2

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cardiac_agent_postproc.io_utils import read_mask, read_image
from cardiac_agent_postproc.ops import (
    myo_bridge, lv_bridge, fill_holes_ops, fill_holes_ops_m2,
    smooth_morph, boundary_smoothing, morph_close_open,
    rv_lv_barrier, dilate_or_erode, topology_cleanup,
    remove_islands, morphological_gap_close, morph_close_large,
    convex_hull_fill, neighbor_class_repair,
    atlas_class_repair, atlas_guided_grow_prune,
)
from cardiac_agent_postproc.rqs import sobel_edges, valve_plane_y
from cardiac_agent_postproc.slice_consistency import (
    neighbor_centroid_align, neighbor_shape_prior, neighbor_area_consistency,
)
from cardiac_agent_postproc.vlm_guardrail import create_diff_overlay
from cardiac_agent_postproc.eval_metrics import dice_macro
from cardiac_agent_postproc.view_utils import infer_view_type
from cardiac_agent_postproc.atlas import build_atlas

# ── Config ───────────────────────────────────────────────────────────────
CONFIG_PATH  = "config/default.yaml"
REPAIR_KB_DIR = "repair_kb"
IMPROVED_THR  =  0.005   # lowered from 0.02 to capture more positives
DEGRADED_THR  = -0.02


def build_ops(mask, view, cfg, E, vy, neighbor_masks=None, target_img=None, atlas=None):
    """Build standalone ops (no executor/agent dependencies)."""
    ops = {
        "myo_bridge": lambda m: myo_bridge(m),
        "lv_bridge": lambda m: lv_bridge(m),
        "fill_holes": lambda m: fill_holes_ops(m, cfg),
        "fill_holes_m2": lambda m: fill_holes_ops_m2(m, cfg),
        "smooth_morph": lambda m: smooth_morph(m, kernel=3),
        "smooth_morph_k5": lambda m: smooth_morph(m, kernel=5),
        "smooth_contour": lambda m: boundary_smoothing(m, cfg, "contour"),
        "smooth_gaussian": lambda m: boundary_smoothing(m, cfg, "gaussian"),
        "close2_open3": lambda m: morph_close_open(m, cfg, close2_open3=True),
        "open2_close3": lambda m: morph_close_open(m, cfg, close2_open3=False),
        "rv_lv_barrier": lambda m: rv_lv_barrier(m, cfg),
        "rv_erode": lambda m: dilate_or_erode(m, cfg, 1, "erode"),
        "topology_cleanup": lambda m: topology_cleanup(m, view, cfg),
        "remove_islands": lambda m: remove_islands(m),
        "2_dilate": lambda m: dilate_or_erode(m, cfg, 2, "dilate"),
        "2_erode": lambda m: dilate_or_erode(m, cfg, 2, "erode"),
        "3_dilate": lambda m: dilate_or_erode(m, cfg, 3, "dilate"),
        "3_erode": lambda m: dilate_or_erode(m, cfg, 3, "erode"),
        "1_dilate": lambda m: dilate_or_erode(m, cfg, 1, "dilate"),
        "morphological_gap_close": lambda m: morphological_gap_close(m, cfg),
        "morph_close_large": lambda m: morph_close_large(m, cfg),
        "convex_hull_fill": lambda m: convex_hull_fill(m),
        "2_dilate_large": lambda m: dilate_or_erode(m, cfg, 2, "dilate_large"),
        "2_erode_large": lambda m: dilate_or_erode(m, cfg, 2, "erode_large"),
        "3_dilate_large": lambda m: dilate_or_erode(m, cfg, 3, "dilate_large"),
        "3_erode_large": lambda m: dilate_or_erode(m, cfg, 3, "erode_large"),
        "rv_erode_large": lambda m: dilate_or_erode(m, cfg, 1, "erode_large"),
    }

    # Neighbor-based ops — use ALL sibling slices (same patient+view)
    if neighbor_masks:
        nb = neighbor_masks
        ops["neighbor_centroid_align"] = lambda m: neighbor_centroid_align(m, nb)
        ops["neighbor_shape_prior"] = lambda m: neighbor_shape_prior(m, nb)
        ops["neighbor_area_consistency"] = lambda m: neighbor_area_consistency(m, nb)
        for nkey, nb_mask in neighbor_masks.items():
            ops[f"neighbor_rv_repair_{nkey}"] = lambda m, nb=nb_mask: neighbor_class_repair(m, nb, target_class=1)
            ops[f"neighbor_myo_repair_{nkey}"] = lambda m, nb=nb_mask: neighbor_class_repair(m, nb, target_class=2)
            ops[f"neighbor_lv_repair_{nkey}"] = lambda m, nb=nb_mask: neighbor_class_repair(m, nb, target_class=3)

    # Atlas-based ops
    if atlas is not None:
        for cls_id, cls_name in {1: "rv", 2: "myo", 3: "lv"}.items():
            ops[f"atlas_{cls_name}_repair"] = lambda m, c=cls_id: atlas_class_repair(m, atlas, target_class=c)
        ops["atlas_grow_prune"] = lambda m: atlas_guided_grow_prune(m, atlas)

    return ops


def _parse_stem_info(stem):
    """Extract patient_id and slice_index from stem."""
    m = re.match(r"(\d+)_original_lax_(\w+)_(\d+)", stem)
    if m:
        return m.group(1), f"lax_{m.group(2)}", int(m.group(3))
    return stem, "unknown", 0


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    source_dir = cfg["paths"]["source_dir"]

    # Create output dirs
    for cat in ("improved", "degraded", "neutral"):
        os.makedirs(os.path.join(REPAIR_KB_DIR, cat), exist_ok=True)

    # Build atlas
    atlas_path = os.path.join(source_dir, cfg["paths"]["atlas_path"])
    atlas = None
    try:
        atlas = build_atlas(source_dir, atlas_path)
        print(f"Atlas loaded: {list(atlas.keys()) if isinstance(atlas, dict) else type(atlas)}")
    except Exception as e:
        print(f"Atlas load failed (continuing without): {e}")

    # Discover all pred files (exclude step/iter intermediates)
    all_files = sorted(glob.glob(os.path.join(source_dir, "*_pred.png")))
    pred_files = [f for f in all_files if "_step" not in f and "_iter" not in f]
    print(f"Found {len(pred_files)} pred files (excluded {len(all_files)-len(pred_files)} intermediates)")

    # Get worst-case stems
    worst_dir = os.path.join(os.path.dirname(source_dir), "worst_cases")
    stem_re = re.compile(r"^\d+\.\d+_\w+_\w+_((\d+)_original_lax_\w+_\d+)\.png$")
    worst_stems = set()
    if os.path.isdir(worst_dir):
        for f in os.listdir(worst_dir):
            m = stem_re.match(f)
            if m:
                worst_stems.add(m.group(1))
    print(f"Worst-case stems: {len(worst_stems)}")

    # Build per-patient groups for neighbor masks
    patient_groups = {}
    for pf in pred_files:
        stem = os.path.basename(pf)[:-9]  # remove _pred.png
        pid, view_key, slice_idx = _parse_stem_info(stem)
        key = f"{pid}_{view_key}"
        if key not in patient_groups:
            patient_groups[key] = []
        patient_groups[key].append((stem, slice_idx, pf))

    stats_all = {"improved": 0, "degraded": 0, "neutral": 0, "skipped": 0, "error": 0}

    for stem in sorted(worst_stems):
        # Load pred, image, GT
        pred_path = os.path.join(source_dir, f"{stem}_pred.png")
        img_path = os.path.join(source_dir, f"{stem}_img.png")
        gt_path = os.path.join(source_dir, f"{stem}_gt.png")

        if not all(os.path.exists(p) for p in [pred_path, img_path, gt_path]):
            stats_all["skipped"] += 1
            continue

        mask = read_mask(pred_path)
        image = read_image(img_path)
        gt = read_mask(gt_path)
        view_type, _ = infer_view_type(stem, mask, rv_ratio_threshold=0.02)

        dice_before = dice_macro(mask, gt)[0]

        # Find neighbors: use ALL sibling slices (no distance limit)
        pid, view_key, slice_idx = _parse_stem_info(stem)
        group_key = f"{pid}_{view_key}"
        neighbor_masks = {}
        for ns, ni, np_ in patient_groups.get(group_key, []):
            if ns != stem:
                neighbor_masks[ni] = read_mask(np_)

        # Build ops
        E = sobel_edges(image, percentile=float(cfg["edge"]["sobel_percentile"]))
        vy = valve_plane_y(mask)
        op_fns = build_ops(mask, view_type, cfg, E, vy,
                           neighbor_masks=neighbor_masks, target_img=image, atlas=atlas)

        print(f"  {stem}: {len(op_fns)} ops, dice_before={dice_before:.4f}, neighbors={len(neighbor_masks)}")

        for op_name, op_fn in op_fns.items():
            try:
                result = op_fn(mask)
                if result is None:
                    continue
                if result.shape != mask.shape:
                    continue
                if np.array_equal(result, mask):
                    continue

                dice_after = dice_macro(result, gt)[0]
                delta = dice_after - dice_before

                if delta > IMPROVED_THR:
                    cat = "improved"
                elif delta < DEGRADED_THR:
                    cat = "degraded"
                else:
                    cat = "neutral"

                stats_all[cat] += 1

                fname = f"{delta:+.4f}_{op_name}_{stem}.png"
                out_path = os.path.join(REPAIR_KB_DIR, cat, fname)
                create_diff_overlay(img_path, mask, result, out_path)

            except Exception as e:
                stats_all["error"] += 1
                # Don't spam for common failures
                if "error" not in str(e).lower() or stats_all["error"] <= 5:
                    print(f"    {op_name} error: {e}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Repair KB built in {REPAIR_KB_DIR}/")
    for k, v in stats_all.items():
        print(f"  {k}: {v}")
    total = stats_all["improved"] + stats_all["degraded"] + stats_all["neutral"]
    print(f"  total examples: {total}")


if __name__ == "__main__":
    main()
