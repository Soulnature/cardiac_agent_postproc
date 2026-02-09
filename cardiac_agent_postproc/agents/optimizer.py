from __future__ import annotations
import os
import csv
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm

from ..io_utils import read_image, read_mask, write_mask
from ..view_utils import infer_view_type
from ..ops import (
    is_valid_candidate, myo_bridge, lv_bridge, smooth_morph,
    fill_holes_ops, fill_holes_ops_m2, rv_lv_barrier,
    boundary_smoothing, topology_cleanup, safe_topology_fix,
    morph_close_open, dilate_or_erode, valve_suppress,
    edge_snapping_proxy, remove_islands,
)
from ..rqs import sobel_edges, valve_plane_y
from ..vlm_guardrail import VisionGuardrail
from .diagnosis_agent import (
    DiagnosisAgent, Diagnosis, SpatialDiagnosis, compute_region_mask,
)
from .triage_agent import TriageResult


def _append_row_csv(path: str, header: List[str], row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


# Map operation names → callables
# Each returns a single mask (or None to skip)
def _build_op_fn(mask, view, cfg, E, vy):
    """Build a dict of op_name → callable(mask) -> mask."""
    return {
        "myo_bridge": lambda m: myo_bridge(m),
        "lv_bridge": lambda m: lv_bridge(m),
        "fill_holes": lambda m: fill_holes_ops(m, cfg),
        "fill_holes_m2": lambda m: fill_holes_ops_m2(m, cfg),
        "smooth_morph": lambda m: smooth_morph(m, kernel=3),
        "smooth_contour": lambda m: boundary_smoothing(m, cfg, "contour"),
        "smooth_gaussian": lambda m: boundary_smoothing(m, cfg, "gaussian"),
        "close2_open3": lambda m: morph_close_open(m, cfg, close2_open3=True),
        "open2_close3": lambda m: morph_close_open(m, cfg, close2_open3=False),
        "rv_lv_barrier": lambda m: rv_lv_barrier(m, cfg),
        "rv_erode": lambda m: dilate_or_erode(m, cfg, 1, "erode"),
        "topology_cleanup": lambda m: topology_cleanup(m, view, cfg),
        "safe_topology_fix": lambda m: safe_topology_fix(m, view, cfg),
        "valve_suppress": lambda m: topology_cleanup(valve_suppress(m, vy, cfg), view, cfg),
        "edge_snap_proxy": lambda m: topology_cleanup(edge_snapping_proxy(m, E, cfg), view, cfg),
        "2_dilate": lambda m: dilate_or_erode(m, cfg, 2, "dilate"),
        "2_erode": lambda m: dilate_or_erode(m, cfg, 2, "erode"),
        "3_dilate": lambda m: dilate_or_erode(m, cfg, 3, "dilate"),
        "3_erode": lambda m: dilate_or_erode(m, cfg, 3, "erode"),
        "remove_islands": lambda m: remove_islands(m),
    }


class OptimizerAgent:
    def __init__(self, cfg: dict, atlas: dict):
        self.cfg = cfg
        self.atlas = atlas
        self.vlm = VisionGuardrail(cfg)
        self.diagnosis = DiagnosisAgent(cfg)

    def optimize_folder(
        self,
        folder: str,
        is_target: bool = False,
        run_dir: str = None,
        triage_results: Dict[str, TriageResult] | None = None,
        diagnosis_results: Dict[str, List[Diagnosis]] | None = None,
        spatial_diagnoses: Dict[str, List[SpatialDiagnosis]] | None = None,
    ) -> dict:
        """
        AI Controller: apply spatially-targeted fixes based on VLM diagnosis.

        Flow per "needs_fix" sample:
        1. Get spatial diagnosis (pre-computed or on-the-fly via rules)
        2. For each diagnosis item:
           a. Compute region mask from (affected_class, direction)
           b. Apply recommended operation
           c. Region-constrain: only keep changes within region (unless global)
           d. Validate with is_valid_candidate()
        3. VLM verification: compare before vs after
           - If improved → accept
           - If worse → revert to original
        4. "good" samples: copy original, no modification
        """
        # --- File Discovery ---
        try:
            files = os.listdir(folder)
        except FileNotFoundError:
            return {"n": 0, "optimized": 0, "skipped": 0}

        preds = [f for f in files if f.endswith("_pred.png")]
        metas: Dict[str, Dict[str, str]] = {}
        source_dir = self.cfg.get("paths", {}).get("source_dir", "")

        for p in preds:
            stem = p[:-9]  # remove _pred.png
            img_name = f"{stem}_img.png"
            local_img = os.path.join(folder, img_name)
            if os.path.exists(local_img):
                img_path = local_img
            elif source_dir and os.path.exists(os.path.join(source_dir, img_name)):
                img_path = os.path.join(source_dir, img_name)
            else:
                continue
            metas[stem] = {
                "stem": stem,
                "pred": os.path.join(folder, p),
                "img": img_path,
            }

        if not metas:
            print(f"[{os.path.basename(folder)}] No paired pred/image files found.")
            return {"n": 0, "optimized": 0, "skipped": 0}

        cur_masks = {s: read_mask(m["pred"]) for s, m in metas.items()}
        pred0_masks = {k: v.copy() for k, v in cur_masks.items()}

        # --- Determine which stems need fixes ---
        if triage_results is not None:
            needs_fix_stems = [
                s for s in cur_masks
                if s in triage_results and triage_results[s].category != "good"
            ]
            good_stems = [s for s in cur_masks if s not in needs_fix_stems]
        else:
            needs_fix_stems = list(cur_masks.keys())
            good_stems = []

        n_total = len(cur_masks)
        n_fix = len(needs_fix_stems)
        n_good = len(good_stems)
        print(
            f"[{os.path.basename(folder)}] {n_total} samples: "
            f"{n_fix} needs_fix, {n_good} good (skip)"
        )

        if not needs_fix_stems:
            return {"n": n_total, "optimized": 0, "skipped": n_good}

        # --- CSV Log ---
        cfg = self.cfg
        log_csv = os.path.join(folder, "label_optimization_log.csv")
        header = [
            "filename", "view_type", "step", "op_name",
            "issue", "direction", "affected_class",
            "pixels_changed", "region_constrained",
        ]
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

        # Verification log
        out_dir_ = run_dir if run_dir else folder
        verify_csv = os.path.join(out_dir_, "vlm_verification.csv")
        v_header = ["stem", "improved", "reason", "n_ops_applied", "reverted"]
        if not os.path.exists(verify_csv):
            with open(verify_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=v_header).writeheader()

        # --- AI Controller: spatially-targeted fixes ---
        optimized_count = 0
        reverted_count = 0

        for stem in tqdm(needs_fix_stems, desc="AI controller", leave=False):
            fr = metas[stem]
            img = read_image(fr["img"])
            cur = cur_masks[stem]
            pred0 = pred0_masks[stem]
            view, _ = infer_view_type(
                fr["stem"], cur,
                rv_ratio_threshold=float(cfg["view_inference"]["rv_ratio_2ch_threshold"]),
            )

            E = sobel_edges(img, percentile=float(cfg["edge"]["sobel_percentile"]))
            vy = valve_plane_y(cur)

            # Get spatial diagnosis
            diagnoses: List[SpatialDiagnosis] = []
            if spatial_diagnoses and stem in spatial_diagnoses:
                diagnoses = spatial_diagnoses[stem]
            else:
                # Fall back: build spatial diagnoses on-the-fly
                triage_issues = None
                if triage_results and stem in triage_results:
                    triage_issues = triage_results[stem].issues
                diagnoses = self.diagnosis.diagnose_spatial(
                    cur, img, view,
                    img_path=fr["img"], stem=stem,
                    triage_issues=triage_issues,
                )

            if not diagnoses:
                continue

            diag_str = ", ".join(
                f"{d.issue}({d.affected_class},{d.direction})->{d.operation}"
                for d in diagnoses[:4]
            )
            print(f"[{stem}] Spatial Diagnosis: {diag_str}")

            # Build operation functions
            op_fns = _build_op_fn(cur, view, cfg, E, vy)

            # Sort diagnoses: structural ops first, cosmetic last
            # This prevents cosmetic changes from creating artifacts
            # that trigger structural ops incorrectly.
            _STRUCTURAL_OPS = {
                "topology_cleanup", "safe_topology_fix", "fill_holes",
                "fill_holes_m2", "rv_lv_barrier", "rv_erode",
                "myo_bridge", "lv_bridge", "valve_suppress",
                "remove_islands", "atlas_growprune_0", "atlas_growprune_1",
            }
            _COSMETIC_OPS = {
                "smooth_morph", "smooth_contour", "smooth_gaussian",
                "close2_open3", "open2_close3", "edge_snap_proxy",
                "2_dilate", "2_erode", "3_dilate", "3_erode",
            }
            # Ops safe to apply without VLM verification
            # These only fill holes or remove tiny islands — very low risk
            _SAFE_WITHOUT_VLM = {
                "fill_holes", "fill_holes_m2", "safe_topology_fix",
                "remove_islands",
            }

            def _op_priority(d: SpatialDiagnosis) -> int:
                if d.operation in _STRUCTURAL_OPS:
                    return 0
                if d.operation in _COSMETIC_OPS:
                    return 2
                return 1

            diagnoses = sorted(diagnoses, key=_op_priority)

            # Without VLM, restrict to safe-only operations
            vlm_available = is_target and self.vlm.enabled
            if not vlm_available:
                diagnoses = [
                    d for d in diagnoses
                    if d.operation in _SAFE_WITHOUT_VLM
                ]

            # Execute spatially-targeted ops
            step = 0
            ops_applied = 0
            total_px_changed = 0
            fg_original = int(np.count_nonzero(pred0))
            # Cumulative change budget: max 5% of original foreground
            # Without VLM verification, we must be ultra-conservative
            max_cumulative_change = max(50, int(fg_original * 0.05))

            for diag in diagnoses:
                op_name = diag.operation
                if op_name not in op_fns:
                    continue

                # Budget check: stop if we've changed too much already
                if total_px_changed >= max_cumulative_change:
                    print(
                        f"  [{stem}] BUDGET EXHAUSTED: "
                        f"{total_px_changed}px >= {max_cumulative_change}px limit"
                    )
                    break

                # 1. Compute region mask
                region = compute_region_mask(cur, diag.affected_class, diag.direction)

                # 2. Apply operation on full mask
                try:
                    candidate = op_fns[op_name](cur)
                except Exception as e:
                    print(f"  [{stem}] Op {op_name} failed: {e}")
                    continue

                if candidate is None:
                    continue

                # 3. Region-constrain: only keep changes within region
                if diag.direction != "global":
                    candidate = np.where(region, candidate, cur)

                # 4. Validate
                if not is_valid_candidate(candidate):
                    continue

                pixels_changed = int(np.sum(candidate != cur))
                if pixels_changed == 0:
                    continue

                # 5. Safety guards: prevent destructive changes
                fg_before = int(np.count_nonzero(cur))
                fg_after = int(np.count_nonzero(candidate))

                # Guard A: reject if operation removes > 20% of foreground
                if fg_before > 0:
                    fg_loss = (fg_before - fg_after) / fg_before
                    if fg_loss > 0.20:
                        print(
                            f"  [{stem}] REJECT {op_name}: "
                            f"removes {fg_loss*100:.0f}% foreground"
                        )
                        continue

                # Guard B: reject if too many pixels change relative to mask
                # (> 5% of foreground per-op = too aggressive without VLM)
                if fg_before > 0:
                    change_ratio = pixels_changed / fg_before
                    if change_ratio > 0.05:
                        print(
                            f"  [{stem}] REJECT {op_name}: "
                            f"changes {change_ratio*100:.0f}% of foreground"
                        )
                        continue

                # Guard C: per-class preservation — don't lose any class entirely
                for cls_val in [1, 2, 3]:
                    cls_before = int(np.sum(cur == cls_val))
                    cls_after = int(np.sum(candidate == cls_val))
                    if cls_before > 50 and cls_after < cls_before * 0.3:
                        print(
                            f"  [{stem}] REJECT {op_name}: "
                            f"class {cls_val} reduced from {cls_before} to {cls_after}"
                        )
                        candidate = None
                        break
                if candidate is None:
                    continue

                # Guard D: myo enclosure check — myo must still wrap LV
                myo_before = int(np.sum(cur == 2))
                myo_after = int(np.sum(candidate == 2))
                if myo_before > 0 and myo_after < myo_before * 0.7:
                    print(
                        f"  [{stem}] REJECT {op_name}: "
                        f"myo reduced from {myo_before} to {myo_after} "
                        f"({myo_after*100//max(1,myo_before)}%)"
                    )
                    continue

                # Guard E: touching check — reject if op creates
                # new LV-RV contact (myo barrier broken)
                lv_cand = (candidate == 3)
                rv_cand = (candidate == 1)
                lv_cur = (cur == 3)
                rv_cur = (cur == 1)
                if lv_cand.any() and rv_cand.any():
                    from scipy.ndimage import binary_dilation
                    touch_after = np.sum(
                        binary_dilation(lv_cand) & rv_cand
                    )
                    touch_before = np.sum(
                        binary_dilation(lv_cur) & rv_cur
                    )
                    if touch_after > touch_before + 10:
                        print(
                            f"  [{stem}] REJECT {op_name}: "
                            f"creates LV-RV contact ({touch_before}->{touch_after}px)"
                        )
                        continue

                # Accept the operation
                step += 1
                ops_applied += 1
                total_px_changed += pixels_changed
                cur = candidate
                cur_masks[stem] = cur

                _append_row_csv(log_csv, header, {
                    "filename": f"{stem}_pred.png",
                    "view_type": view,
                    "step": step,
                    "op_name": op_name,
                    "issue": diag.issue,
                    "direction": diag.direction,
                    "affected_class": diag.affected_class,
                    "pixels_changed": pixels_changed,
                    "region_constrained": diag.direction != "global",
                })

                print(
                    f"  [{stem}] Step {step}: {op_name} "
                    f"({diag.issue}, {diag.affected_class}@{diag.direction}) "
                    f"-> {pixels_changed}px"
                )

            # 5. VLM Verification: compare before vs after
            reverted = False
            if ops_applied > 0 and is_target and self.vlm.enabled:
                diag_summary = "; ".join(
                    f"{d.issue} ({d.affected_class}@{d.direction})"
                    for d in diagnoses[:4]
                )
                verify_result = self.vlm.verify_fix(
                    fr["img"], pred0, cur, stem,
                    diagnosis_summary=diag_summary,
                )

                if not verify_result["improved"]:
                    # Revert to original
                    print(
                        f"  [{stem}] VLM says fix is WORSE: {verify_result['reason']}. "
                        f"Reverting to original."
                    )
                    cur = pred0
                    cur_masks[stem] = cur
                    reverted = True
                    reverted_count += 1
                else:
                    print(
                        f"  [{stem}] VLM VERIFIED improvement: {verify_result['reason']}"
                    )

                # Log verification
                _append_row_csv(verify_csv, v_header, {
                    "stem": stem,
                    "improved": verify_result["improved"],
                    "reason": verify_result["reason"],
                    "n_ops_applied": ops_applied,
                    "reverted": reverted,
                })

            if ops_applied > 0 and not reverted:
                optimized_count += 1

        # --- Finalize: write all masks ---
        print(
            f"[{os.path.basename(folder)}] Finalizing: "
            f"{optimized_count} improved, {reverted_count} reverted"
        )

        for stem, mask in cur_masks.items():
            if is_target:
                final_path = os.path.join(folder, f"{stem}_pred.png")
                write_mask(final_path, mask)

        return {
            "n": n_total,
            "optimized": optimized_count,
            "skipped": n_good,
            "reverted": reverted_count,
        }
