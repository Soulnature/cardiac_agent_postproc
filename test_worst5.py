"""
Test script: Can diagnosis-driven optimization improve the 5 worst cases?

For each case:
1. Baseline Dice (pred vs gt)
2. Rules-based diagnosis → which issues detected, which ops proposed
3. Apply diagnosed ops sequentially → Dice after each step
4. Exhaustive search: try ALL available ops → find best possible improvement
5. Multi-step greedy: iteratively apply best op → find ceiling
"""
import os, sys, yaml
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(__file__))

from cardiac_agent_postproc.io_utils import read_mask, read_image
from cardiac_agent_postproc.view_utils import infer_view_type
from cardiac_agent_postproc.quality_model import extract_quality_features
from cardiac_agent_postproc.agents.diagnosis_agent import DiagnosisAgent, ISSUE_OPS_MAP
from cardiac_agent_postproc.ops import (
    is_valid_candidate, myo_bridge, lv_bridge, smooth_morph,
    fill_holes_ops, fill_holes_ops_m2, rv_lv_barrier,
    boundary_smoothing, topology_cleanup, safe_topology_fix,
    morph_close_open, dilate_or_erode, valve_suppress,
    edge_snapping_proxy, remove_islands,
)
from cardiac_agent_postproc.rqs import sobel_edges, valve_plane_y


DATA = "results/Input_MnM2/all_frames_export"
CASES = [
    "335_original_lax_4c_009",
    "307_original_lax_4c_024",
    "244_original_lax_4c_009",
    "244_original_lax_4c_000",
    "209_original_lax_4c_022",
]


def dice_macro(pred, gt):
    """Macro-averaged Dice over classes 1,2,3."""
    dices = []
    for c in [1, 2, 3]:
        p = (pred == c)
        g = (gt == c)
        inter = np.sum(p & g)
        union = np.sum(p) + np.sum(g)
        d = 2 * inter / max(union, 1)
        dices.append(d)
    return np.mean(dices), dices


def dice_per_class(pred, gt):
    """Returns dict of per-class Dice."""
    names = {1: "RV", 2: "Myo", 3: "LV"}
    result = {}
    for c in [1, 2, 3]:
        p = (pred == c)
        g = (gt == c)
        inter = np.sum(p & g)
        union = np.sum(p) + np.sum(g)
        result[names[c]] = 2 * inter / max(union, 1)
    return result


def build_all_ops(mask, view, cfg, E, vy):
    """Build dict of ALL available operations."""
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
        "1_dilate": lambda m: dilate_or_erode(m, cfg, 1, "dilate"),
        "remove_islands": lambda m: remove_islands(m),
        "smooth_morph_k5": lambda m: smooth_morph(m, kernel=5),
    }


def main():
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    diag_agent = DiagnosisAgent(cfg)

    print("=" * 80)
    print("TEST: Diagnosis-Driven Optimization on 5 Worst Cases")
    print("=" * 80)

    for stem in CASES:
        pred_path = os.path.join(DATA, f"{stem}_pred.png")
        gt_path = os.path.join(DATA, f"{stem}_gt.png")
        img_path = os.path.join(DATA, f"{stem}_img.png")

        pred = read_mask(pred_path)
        gt = read_mask(gt_path)
        img = read_image(img_path)

        view, _ = infer_view_type(
            stem, pred,
            rv_ratio_threshold=float(cfg["view_inference"]["rv_ratio_2ch_threshold"]),
        )
        E = sobel_edges(img, percentile=float(cfg["edge"]["sobel_percentile"]))
        vy = valve_plane_y(pred)

        baseline_dice, baseline_per = dice_macro(pred, gt)
        pc = dice_per_class(pred, gt)

        print(f"\n{'='*80}")
        print(f"CASE: {stem}  (view={view})")
        print(f"  Baseline Dice: {baseline_dice:.4f}  "
              f"[RV={pc['RV']:.3f}, Myo={pc['Myo']:.3f}, LV={pc['LV']:.3f}]")

        # ----- Step 1: Rules-based Diagnosis -----
        feats = extract_quality_features(pred, img, view)
        diagnoses = diag_agent.diagnose(pred, img, view)

        print(f"\n  [Diagnosis] {len(diagnoses)} issues detected:")
        for d in diagnoses:
            print(f"    - {d.issue} (conf={d.confidence:.2f}) → ops={d.ops}")

        # Key features
        print(f"  [Features] cc_myo={feats['cc_count_myo']:.0f}, "
              f"cc_lv={feats['cc_count_lv']:.0f}, cc_rv={feats['cc_count_rv']:.0f}, "
              f"touch={feats['touch_ratio']:.4f}, "
              f"thickness_cv={feats['thickness_cv']:.3f}")

        # ----- Step 2: Apply diagnosed ops sequentially -----
        print(f"\n  [Diagnosed Ops - Sequential Application]")
        cur = pred.copy()
        ops_plan = DiagnosisAgent.ops_from_diagnoses(diagnoses)
        all_op_fns = build_all_ops(cur, view, cfg, E, vy)

        for op_name in ops_plan:
            if op_name not in all_op_fns:
                print(f"    {op_name}: NOT AVAILABLE")
                continue
            try:
                result = all_op_fns[op_name](cur)
            except Exception as e:
                print(f"    {op_name}: FAILED ({e})")
                continue
            if result is None or not is_valid_candidate(result):
                print(f"    {op_name}: INVALID result")
                continue
            px = int(np.sum(result != cur))
            if px == 0:
                print(f"    {op_name}: no change")
                continue
            d_new, _ = dice_macro(result, gt)
            delta = d_new - dice_macro(cur, gt)[0]
            pc_new = dice_per_class(result, gt)
            cur = result
            # rebuild op fns with new mask state
            all_op_fns = build_all_ops(cur, view, cfg, E, vy)
            print(f"    {op_name}: {px}px changed → Dice={d_new:.4f} (Δ={delta:+.4f}) "
                  f"[RV={pc_new['RV']:.3f}, Myo={pc_new['Myo']:.3f}, LV={pc_new['LV']:.3f}]")

        diag_final, _ = dice_macro(cur, gt)
        print(f"  → Diagnosed ops result: {baseline_dice:.4f} → {diag_final:.4f} "
              f"(Δ={diag_final - baseline_dice:+.4f})")

        # ----- Step 3: Exhaustive single-op search from original -----
        print(f"\n  [Exhaustive Single-Op Search (from original pred)]")
        all_op_fns = build_all_ops(pred, view, cfg, E, vy)
        results = []
        for op_name, op_fn in all_op_fns.items():
            try:
                result = op_fn(pred)
            except Exception:
                continue
            if result is None or not is_valid_candidate(result):
                continue
            px = int(np.sum(result != pred))
            if px == 0:
                continue
            d_new, _ = dice_macro(result, gt)
            delta = d_new - baseline_dice
            results.append((op_name, d_new, delta, px))

        results.sort(key=lambda x: x[2], reverse=True)
        for op_name, d_new, delta, px in results[:5]:
            print(f"    {op_name}: Dice={d_new:.4f} (Δ={delta:+.4f}) [{px}px]")
        if results:
            best = results[0]
            print(f"  → Best single op: {best[0]} → Dice={best[1]:.4f} (Δ={best[2]:+.4f})")
        else:
            print(f"  → No ops produced change")

        # ----- Step 4: Greedy multi-step (find ceiling) -----
        print(f"\n  [Greedy Multi-Step Search (up to 10 steps)]")
        cur = pred.copy()
        for step in range(10):
            all_op_fns = build_all_ops(cur, view, cfg, E, vy)
            best_name, best_result, best_delta = None, None, -999
            cur_dice, _ = dice_macro(cur, gt)
            for op_name, op_fn in all_op_fns.items():
                try:
                    result = op_fn(cur)
                except Exception:
                    continue
                if result is None or not is_valid_candidate(result):
                    continue
                px = int(np.sum(result != cur))
                if px == 0:
                    continue
                d_new, _ = dice_macro(result, gt)
                delta = d_new - cur_dice
                if delta > best_delta:
                    best_name, best_result, best_delta = op_name, result, delta
            if best_name is None or best_delta <= 0:
                print(f"    Step {step+1}: no improvement found → stop")
                break
            cur = best_result
            d_new, _ = dice_macro(cur, gt)
            pc_new = dice_per_class(cur, gt)
            print(f"    Step {step+1}: {best_name} → Dice={d_new:.4f} (Δ={best_delta:+.4f}) "
                  f"[RV={pc_new['RV']:.3f}, Myo={pc_new['Myo']:.3f}, LV={pc_new['LV']:.3f}]")

        greedy_final, _ = dice_macro(cur, gt)
        print(f"  → Greedy ceiling: {baseline_dice:.4f} → {greedy_final:.4f} "
              f"(Δ={greedy_final - baseline_dice:+.4f})")

    print(f"\n{'='*80}")
    print("DONE")


if __name__ == "__main__":
    main()
