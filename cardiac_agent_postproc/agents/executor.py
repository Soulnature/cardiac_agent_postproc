"""
ExecutorAgent: Executes repair plans step-by-step.

Follows the PlannerAgent's plan, applies morphological operations,
uses VLM visual judgment to validate each step, and reports results.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi

from ..io_utils import read_image
from ..ops import (
    is_valid_candidate, myo_bridge, lv_bridge, smooth_morph,
    fill_holes_ops, fill_holes_ops_m2, rv_lv_barrier,
    boundary_smoothing, topology_cleanup, safe_topology_fix,
    morph_close_open, dilate_or_erode, valve_suppress,
    edge_snapping_proxy, remove_islands, generate_candidates,
    atlas_guided_grow_prune, atlas_class_repair, neighbor_class_repair,
    expand_myo_intensity_guided, expand_blood_pool_intensity,
    morphological_gap_close, morph_close_large, convex_hull_fill,
    erode_lv_expand_myo, erode_rv_expand_myo_conditional,
    erode_myo_expand_lv, erode_myo_expand_rv,
    reassign_secondary_lv_to_rv, reassign_secondary_rv_to_lv,
)
from ..rqs import compute_rqs, sobel_edges, valve_plane_y
from ..slice_consistency import (
    neighbor_centroid_align, neighbor_shape_prior, neighbor_area_consistency,
)
from ..vlm_guardrail import create_overlay
from .base_agent import BaseAgent
from .message_bus import CaseContext, MessageBus, MessageType
from .diagnosis_agent import compute_region_mask, normalize_direction_hint
from .visual_knowledge import VisualKnowledgeBase


from ..individual_atlas import hybrid_repair


HIGH_IMPACT_OPS = {
    "individual_atlas_transfer",
    "convex_hull_fill",
    "rv_lv_barrier",
    "expand_myo_intensity",
    "expand_lv_intensity",
    "expand_rv_intensity",
    "neighbor_rv_repair",
    "neighbor_myo_repair",
    "neighbor_lv_repair",
    "atlas_rv_repair",
    "atlas_myo_repair",
    "atlas_lv_repair",
}


def _is_high_impact_op(op_name: str) -> bool:
    """Identify operations that can cause large/global structural changes."""
    if not op_name:
        return False
    if op_name in HIGH_IMPACT_OPS:
        return True
    if op_name.startswith("atlas_") or op_name.startswith("neighbor_"):
        return True
    if op_name.endswith("_large"):
        return True
    return False


def _component_count(class_mask: np.ndarray) -> int:
    """Count connected components in a binary class mask."""
    if class_mask is None or int(np.count_nonzero(class_mask)) == 0:
        return 0
    _, n_comp = ndi.label(class_mask.astype(np.uint8))
    return int(n_comp)


_CLASS_NAME_BY_ID = {1: "rv", 2: "myo", 3: "lv"}
_CLASS_ID_BY_NAME = {"rv": 1, "myo": 2, "lv": 3}
_CONTACT_INCREASE_ALLOWED_OPS = {
    "rv_lv_barrier",
    "rv_lv_barrier_mild",
    "rv_lv_barrier_strong",
    "myo_bridge",
    "myo_bridge_mild",
    "myo_bridge_strong",
    "neighbor_myo_repair",
    "atlas_myo_repair",
    "morphological_gap_close",
    "convex_hull_fill",
}


def _required_class_ids_for_view(view_type: str) -> set[int]:
    v = str(view_type or "").strip().lower()
    if "2ch" in v:
        return {2, 3}
    return {1, 2, 3}


def _class_area(mask: np.ndarray, class_id: int) -> int:
    return int(np.count_nonzero(mask == int(class_id)))


def _class_centroid(mask: np.ndarray, class_id: int) -> Optional[tuple[float, float]]:
    ys, xs = np.where(mask == int(class_id))
    if ys.size == 0:
        return None
    return float(np.mean(ys)), float(np.mean(xs))


def _euclid(a: Optional[tuple[float, float]], b: Optional[tuple[float, float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    dy = float(a[0]) - float(b[0])
    dx = float(a[1]) - float(b[1])
    return float(np.sqrt(dy * dy + dx * dx))


def _rv_lv_contact_len(mask: np.ndarray) -> int:
    rv = (mask == 1)
    lv = (mask == 3)
    if int(np.count_nonzero(rv)) == 0 or int(np.count_nonzero(lv)) == 0:
        return 0
    rv_edge = rv & (~ndi.binary_erosion(rv, iterations=1))
    lv_dil = ndi.binary_dilation(lv, iterations=1)
    return int(np.count_nonzero(rv_edge & lv_dil))


def _lv_myo_coverage(mask: np.ndarray, valve_y: Optional[int], valve_margin_px: int = 6) -> float:
    lv = (mask == 3)
    myo = (mask == 2)
    if int(np.count_nonzero(lv)) == 0:
        return 1.0
    lv_edge = lv & (~ndi.binary_erosion(lv, iterations=1))
    if valve_y is not None:
        h = int(mask.shape[0])
        cut = int(max(0, min(h, int(round(float(valve_y))) + int(valve_margin_px))))
        valid = np.ones_like(lv_edge, dtype=bool)
        valid[:cut, :] = False
        filtered = lv_edge & valid
        if int(np.count_nonzero(filtered)) >= 12:
            lv_edge = filtered
    n_edge = int(np.count_nonzero(lv_edge))
    if n_edge == 0:
        return 1.0
    myo_band = ndi.binary_dilation(myo, iterations=1)
    covered = int(np.count_nonzero(lv_edge & myo_band))
    return float(covered) / float(max(1, n_edge))


def _neighbor_stats_for_class(
    neighbor_masks: Dict[int, np.ndarray],
    class_id: int,
    min_area_px: int = 25,
) -> Optional[Dict[str, Any]]:
    if not neighbor_masks:
        return None
    areas: List[float] = []
    ccs: List[float] = []
    centroids: List[tuple[float, float]] = []
    for nb in neighbor_masks.values():
        area = _class_area(nb, class_id)
        if area < int(min_area_px):
            continue
        areas.append(float(area))
        ccs.append(float(_component_count(nb == int(class_id))))
        c = _class_centroid(nb, class_id)
        if c is not None:
            centroids.append(c)
    if not areas:
        return None
    centroid_mean: Optional[tuple[float, float]] = None
    if centroids:
        ys = [c[0] for c in centroids]
        xs = [c[1] for c in centroids]
        centroid_mean = (float(np.mean(ys)), float(np.mean(xs)))
    return {
        "mean_area": float(np.mean(areas)),
        "mean_cc": float(np.mean(ccs)) if ccs else 0.0,
        "centroid": centroid_mean,
        "n": int(len(areas)),
    }


def _build_op_fn(mask, view, cfg, E, vy, atlas=None, neighbor_masks=None, neighbor_img_paths=None, target_img=None, diagnosis_issues=None):
    """Build a dict of op_name → callable(mask) -> mask."""
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
        "safe_topology_fix": lambda m: safe_topology_fix(m, view, cfg),
        "valve_suppress": lambda m: topology_cleanup(valve_suppress(m, vy, cfg), view, cfg),
        "edge_snap_proxy": lambda m: topology_cleanup(edge_snapping_proxy(m, E, cfg), view, cfg),
        "remove_islands": lambda m: remove_islands(m),
        "2_dilate": lambda m: dilate_or_erode(m, cfg, 2, "dilate"),
        "2_erode": lambda m: dilate_or_erode(m, cfg, 2, "erode"),
        "3_dilate": lambda m: dilate_or_erode(m, cfg, 3, "dilate"),
        "3_erode": lambda m: dilate_or_erode(m, cfg, 3, "erode"),
        "1_dilate": lambda m: dilate_or_erode(m, cfg, 1, "dilate"),

        # Intensity-guided robustness ops
        "expand_myo_intensity": lambda m: expand_myo_intensity_guided(m, target_img),
        "expand_blood_intensity": lambda m: expand_blood_pool_intensity(m, target_img, 1, cfg), # Default to RV? Or needs generic?
        # distinct ops for 1 (RV) and 3 (LV)
        "expand_rv_intensity": lambda m: expand_blood_pool_intensity(m, target_img, 1, cfg),
        "expand_lv_intensity": lambda m: expand_blood_pool_intensity(m, target_img, 3, cfg),
        
        # New Gap Closing
        "morphological_gap_close": lambda m: morphological_gap_close(m, cfg),
        "morph_close_large": lambda m: morph_close_large(m, cfg),
        "convex_hull_fill": lambda m: convex_hull_fill(m),
        "lv_split_to_rv": lambda m: reassign_secondary_lv_to_rv(m),
        "rv_split_to_lv": lambda m: reassign_secondary_rv_to_lv(m),
        "2_dilate_large": lambda m: dilate_or_erode(m, cfg, 2, "dilate_large"),
        "2_erode_large": lambda m: dilate_or_erode(m, cfg, 2, "erode_large"),
        "3_dilate_large": lambda m: dilate_or_erode(m, cfg, 3, "dilate_large"),
        "3_erode_large": lambda m: dilate_or_erode(m, cfg, 3, "erode_large"),
        "rv_erode_large": lambda m: dilate_or_erode(m, cfg, 1, "erode_large"),
        "3_erode_expand_myo": lambda m: erode_lv_expand_myo(m, cfg),
        "1_erode_expand_myo": lambda m: erode_rv_expand_myo_conditional(m, cfg),
        "2_erode_expand_lv": lambda m: erode_myo_expand_lv(m, cfg),
        "2_erode_expand_rv": lambda m: erode_myo_expand_rv(m, cfg),
    }

    # Inter-slice consistency ops (only available when neighbors exist)
    if neighbor_masks:
        nb = neighbor_masks  # capture for closures
        ops["neighbor_centroid_align"] = lambda m: neighbor_centroid_align(m, nb)
        ops["neighbor_shape_prior"] = lambda m: neighbor_shape_prior(m, nb)
        ops["neighbor_area_consistency"] = lambda m: neighbor_area_consistency(m, nb)

        # Individual Atlas Transfer (Hybrid Repair)
        # Select best neighbor (if multiple)
        # For now simply try the first one that works?
        # Or let hybrid_repair handle selection?
        # Let's simple wrapper here.
        def _individual_atlas_op(m):
            if not neighbor_img_paths:
                return m
            # Try to use first neighbor
            # In future: iterate all and find best registration
            idx = list(neighbor_masks.keys())[0]
            nb_mask = neighbor_masks[idx]
            nb_img_path = neighbor_img_paths.get(idx)
            if not nb_img_path or not os.path.exists(nb_img_path):
                return m
            nb_img = read_image(nb_img_path)
            
            # target_img passed from ctx
            if target_img is None:
                return m
                
            return hybrid_repair(m, nb_mask, target_img, nb_img, diagnosis_issues)

        ops["individual_atlas_transfer"] = _individual_atlas_op

        # Class-specific neighbor repair: use neighbor's class labels as donor
        first_nb_mask = neighbor_masks[list(neighbor_masks.keys())[0]]
        ops["neighbor_rv_repair"] = lambda m, nb=first_nb_mask: neighbor_class_repair(m, nb, target_class=1)
        ops["neighbor_myo_repair"] = lambda m, nb=first_nb_mask: neighbor_class_repair(m, nb, target_class=2)
        ops["neighbor_lv_repair"] = lambda m, nb=first_nb_mask: neighbor_class_repair(m, nb, target_class=3)

    return ops


def _add_atlas_ops(op_fns, mask, view, atlas, cfg):
    """Add atlas-dependent operations if atlas is available."""
    if atlas is None:
        return
    # atlas_guided_grow_prune returns List[np.ndarray]; extract first valid candidate
    def _atlas_first(m, z=None):
        candidates = atlas_guided_grow_prune(m, view, atlas, cfg, z_norm=z)
        return candidates[0] if candidates else None
    op_fns["atlas_growprune_0"] = lambda m: _atlas_first(m, z=None)
    op_fns["atlas_growprune_1"] = lambda m: _atlas_first(m, z=0.5)
    # Class-specific atlas repair: only fix the target class, leave others intact
    op_fns["atlas_rv_repair"] = lambda m: atlas_class_repair(m, view, atlas, cfg, target_class=1)
    op_fns["atlas_myo_repair"] = lambda m: atlas_class_repair(m, view, atlas, cfg, target_class=2)
    op_fns["atlas_lv_repair"] = lambda m: atlas_class_repair(m, view, atlas, cfg, target_class=3)


def _apply_local_op(mask: np.ndarray, op_fn, bbox: List[int]) -> np.ndarray:
    """
    Apply op_fn only within the bbox [ymin, xmin, ymax, xmax].
    bbox is in PIXELS (0-H, 0-W).
    """
    ymin, xmin, ymax, xmax = bbox
    H, W = mask.shape
    
    # Clip
    ymin = max(0, min(ymin, H))
    xmin = max(0, min(xmin, W))
    ymax = max(0, min(ymax, H))
    xmax = max(0, min(xmax, W))
    
    if ymax <= ymin or xmax <= xmin:
        return mask
        
    crop = mask[ymin:ymax, xmin:xmax].copy()
    
    # Apply op to crop
    # Note: Op must be robust to partial views
    try:
        processed_crop = op_fn(crop)
    except Exception:
        return mask
        
    if processed_crop is None:
        return mask
        
    result = mask.copy()
    result[ymin:ymax, xmin:xmax] = processed_crop
    return result


class ExecutorAgent(BaseAgent):
    """
    Executes repair plans step-by-step.

    For each operation in the plan:
    1. Apply the operation
    2. Validate the result (is_valid_candidate)
    3. VLM visual quality check (replaces RQS-based validation)
    4. Report via message bus
    5. If op fails or regresses, notify PlannerAgent
    """

    def __init__(
        self,
        cfg: dict,
        bus: MessageBus,
        atlas: dict = None,
        visual_kb=None,
        repair_kb=None,
    ):
        self._atlas = atlas
        super().__init__(cfg, bus, visual_kb=visual_kb, repair_kb=repair_kb)

    @property
    def name(self) -> str:
        return "executor"

    @property
    def system_prompt(self) -> str:
        fallback = (
            "You are the Executor Agent in a cardiac MRI segmentation repair system.\n\n"
            "Your role is to execute repair operations and evaluate their effects.\n"
            "After each operation, you report:\n"
            "- Whether the operation succeeded\n"
            "- How many pixels changed\n"
            "- VLM visual quality assessment (did it improve?)\n"
            "- Whether the result is anatomically valid\n\n"
            "If an operation fails or causes regression, you flag it.\n"
            "You ONLY execute — you don't decide what to do."
        )
        return self._prompt_with_fallback("executor_system.txt", fallback)

    def process_case(self, ctx: CaseContext) -> None:
        """
        Execute the repair plan for a case.
        Reads plan from bus messages, applies ops, updates ctx.
        Uses VLM for visual validation instead of RQS as primary quality signal.
        """
        # Get the plan from pending messages
        plan_msgs = self.receive_messages(
            msg_type=MessageType.REPAIR_PLAN, case_id=ctx.case_id
        )
        revision_msgs = self.receive_messages(
            msg_type=MessageType.PLAN_REVISION, case_id=ctx.case_id
        )

        # Use latest plan (revision takes priority)
        plan = []
        if revision_msgs:
            plan = revision_msgs[-1].content.get("plan", [])
        elif plan_msgs:
            plan = plan_msgs[-1].content.get("plan", [])

        if not plan:
            self.log(f"No plan for {ctx.stem}, skip execution")
            return

        self.log(f"Executing {len(plan)} ops for {ctx.stem}")

        assert ctx.current_mask is not None and ctx.image is not None

        # Prepare edge map and valve plane
        E = sobel_edges(ctx.image, percentile=float(self.cfg["edge"]["sobel_percentile"]))
        vy = valve_plane_y(ctx.current_mask)

        # Build operation functions (includes inter-slice ops if neighbors available)
        op_fns = _build_op_fn(
            ctx.current_mask, ctx.view_type, self.cfg, E, vy, self._atlas,
            neighbor_masks=ctx.neighbor_masks,
            neighbor_img_paths=ctx.neighbor_img_paths,
            target_img=ctx.image,
            diagnosis_issues=ctx.triage_issues,
        )
        _add_atlas_ops(op_fns, ctx.current_mask, ctx.view_type, self._atlas, self.cfg)

        # Compute initial RQS (kept as optional reporting metric)
        try:
            initial_rqs = compute_rqs(
                ctx.current_mask, ctx.image, ctx.view_type,
                ctx.original_mask, self._atlas, [], self.cfg
            )
            current_rqs = initial_rqs.total
        except Exception:
            current_rqs = 0.0

        # Load GT for Oracle Dice Logging (if available)
        gt_mask = None
        current_dice = 0.0
        if ctx.gt_path and os.path.exists(ctx.gt_path):
            try:
                from ..io_utils import read_mask
                from ..eval_metrics import dice_macro
                gt_mask = read_mask(ctx.gt_path)
                current_dice = dice_macro(ctx.current_mask, gt_mask)[0]
                self.log(f"  [Oracle] Initial Dice: {current_dice:.4f}")
            except Exception as e:
                self.log(f"  [Oracle] Failed to load GT: {e}")

        steps_applied = []
        ops_failed = []
        debug_dir = ""
        if ctx.img_path:
            if ctx.output_dir:
                debug_dir = os.path.join(ctx.output_dir, "debug_steps", ctx.case_id)
            else:
                debug_dir = os.path.join(os.path.dirname(ctx.img_path), "debug_steps", ctx.case_id)
            os.makedirs(debug_dir, exist_ok=True)

        # Initialize VLM baseline score (optional; some VLMs do not output numeric score)
        ctx.current_score = None
        ctx.score_history = []
        if ctx.img_path and self.visual_kb:
            try:
                step0_overlay = ctx.debug_img_path("_step0_overlay.png")
                # Create overlay of ORIGINAL (current_mask is original at start)
                if create_overlay(ctx.img_path, ctx.current_mask, step0_overlay):
                    shutil.copy(step0_overlay, os.path.join(debug_dir, "step_0_initial.png"))
                    v0 = self.judge_visual_quality(step0_overlay)
                    base_score = float(v0.get("score", 0) or 0)
                    if base_score > 0:
                        ctx.current_score = base_score
                        ctx.score_history.append(base_score)
                        self.log(f"  [Init] Baseline Score: {ctx.current_score:.2f}")
                    else:
                        self.log(
                            "  [Init] Baseline score unavailable; "
                            f"using qualitative gating (quality={v0.get('quality', 'unknown')})"
                        )
            except Exception as e:
                self.log(f"  [Init] Baseline VLM check failed: {e}")

        # Per-step VLM scoring and strict monotonic config
        ex_cfg = self.cfg.get("executor", {})
        vlm_per_step = bool(ex_cfg.get("vlm_per_step_scoring", False))
        vlm_score_tol = float(ex_cfg.get("vlm_score_tolerance", 0.3))
        vlm_min_conf = float(ex_cfg.get("vlm_score_min_confidence", 0.4))
        strict_monotonic_mode = bool(ex_cfg.get("strict_monotonic_mode", False))
        strict_min_score_delta = float(ex_cfg.get("strict_min_score_delta", 0.05))
        strict_step_min_conf = float(
            ex_cfg.get("strict_step_min_confidence", max(0.55, vlm_min_conf))
        )
        strict_reject_low_conf = bool(ex_cfg.get("strict_reject_low_confidence", True))
        strict_require_baseline = bool(ex_cfg.get("strict_require_baseline_score", True))
        enforce_directional_execution = bool(ex_cfg.get("enforce_directional_execution", True))
        split_relabel_require_bbox = bool(ex_cfg.get("split_relabel_require_bbox", True))
        split_relabel_topology_gate = bool(ex_cfg.get("split_relabel_topology_gate", True))
        split_relabel_max_change_ratio = float(
            ex_cfg.get("split_relabel_max_change_ratio", 0.06)
        )
        strict_no_baseline_block = bool(
            strict_monotonic_mode
            and vlm_per_step
            and strict_require_baseline
            and (ctx.current_score is None or float(ctx.current_score) <= 0.0)
        )
        if strict_no_baseline_block:
            self.log(
                "  [StrictMonotonic] Baseline VLM score unavailable; "
                "rejecting all planned steps for safety."
            )

        for step_idx, step in enumerate(plan):
            op_name = step.get("operation", "")
            direction = normalize_direction_hint(str(step.get("direction", "global")))
            affected_class = step.get("affected_class", "myo")
            bbox = step.get("bbox", None)
            if not (isinstance(bbox, list) and len(bbox) == 4):
                bbox = None
            policy_tag = str(step.get("policy_tag", "")).strip().lower()
            is_forced_atlas = (
                policy_tag == "forced_atlas_class_replace"
                and str(op_name).startswith("atlas_")
            )
            is_hard_split_relabel = op_name in {"lv_split_to_rv", "rv_split_to_lv"}

            if strict_no_baseline_block:
                reason = "strict_monotonic_requires_baseline_score"
                self.log(f"  [{step_idx}] {op_name}: REJECTED ({reason})")
                ops_failed.append({"operation": op_name, "reason": reason})
                continue

            if is_hard_split_relabel and split_relabel_require_bbox and bbox is None:
                reason = "split_relabel_requires_bbox"
                self.log(f"  [{step_idx}] {op_name}: REJECTED ({reason})")
                ops_failed.append({"operation": op_name, "reason": reason})
                continue

            # ── Smart Global Repair: Infer bbox from direction if missing ──
            if not bbox and direction != "global" and direction:
                try:
                    region_mask = compute_region_mask(ctx.current_mask, affected_class, direction)
                    focus_mask = region_mask

                    # If bbox is missing, tighten localization with class-aware focus.
                    # This keeps edits near the diagnosed structure instead of applying
                    # to an entire half-plane.
                    class_id_map = {"rv": 1, "myo": 2, "lv": 3}
                    class_id = class_id_map.get(str(affected_class).strip().lower())
                    if class_id is not None:
                        class_mask = (ctx.current_mask == class_id)
                        candidate = region_mask & class_mask
                        # Use class-focused region when it has enough pixels.
                        if int(np.count_nonzero(candidate)) >= 16:
                            focus_mask = candidate

                    # Get bbox of the True regions
                    rows = np.any(focus_mask, axis=1)
                    cols = np.any(focus_mask, axis=0)
                    if np.any(rows) and np.any(cols):
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        # Add a moderate margin to ensure boundary features are caught
                        margin = 20
                        H, W = ctx.current_mask.shape
                        ymin = max(0, ymin - margin)
                        xmin = max(0, xmin - margin)
                        # Slice upper bound is exclusive, so +1 + margin
                        ymax = min(H, ymax + 1 + margin)
                        xmax = min(W, xmax + 1 + margin)
                        
                        bbox = [int(ymin), int(xmin), int(ymax), int(xmax)]
                        self.log(f"  [SmartGlobal] Inferred bbox {bbox} from direction '{direction}'")
                except Exception as e:
                    self.log(f"  [SmartGlobal] Failed to infer bbox: {e}")

            if op_name not in op_fns:
                self.log(f"  [{step_idx}] {op_name}: NOT AVAILABLE")
                ops_failed.append({"operation": op_name, "reason": "not available"})
                continue

            try:
                if is_hard_split_relabel:
                    if op_name == "lv_split_to_rv":
                        result = reassign_secondary_lv_to_rv(ctx.current_mask, bbox=bbox)
                    else:
                        result = reassign_secondary_rv_to_lv(ctx.current_mask, bbox=bbox)
                    if bbox is not None:
                        self.log(f"  [{step_idx}] {op_name}: using diagnosis hint bbox {bbox}")
                else:
                    # Force GLOBAL application for neighbor/atlas ops (they need full mask context)
                    apply_bbox = bbox
                    if op_name.startswith("neighbor_") or op_name.startswith("atlas_"):
                        apply_bbox = None

                    if apply_bbox:
                        result = _apply_local_op(ctx.current_mask, op_fns[op_name], apply_bbox)
                        self.log(f"  [{step_idx}] {op_name} (local): Applied to {apply_bbox}")
                    else:
                        result = op_fns[op_name](ctx.current_mask)
            except Exception as e:
                self.log(f"  [{step_idx}] {op_name}: FAILED ({e})")
                ops_failed.append({"operation": op_name, "reason": str(e)})
                continue

            if result is None or not is_valid_candidate(result):
                self.log(f"  [{step_idx}] {op_name}: INVALID result")
                ops_failed.append({"operation": op_name, "reason": "invalid result"})
                continue

            # Apply diagnosis-constrained merge region:
            # - direction mask (if provided)
            # - bbox mask (if provided)
            # This enables targeted correction even when current pixel labels are wrong.
            merge_region = None
            if not is_hard_split_relabel:
                if direction != "global":
                    merge_region = compute_region_mask(ctx.current_mask, affected_class, direction)

                if bbox is not None:
                    H, W = ctx.current_mask.shape
                    ymin, xmin, ymax, xmax = [int(v) for v in bbox]
                    ymin = max(0, min(ymin, H))
                    xmin = max(0, min(xmin, W))
                    ymax = max(0, min(ymax, H))
                    xmax = max(0, min(xmax, W))
                    if ymax > ymin and xmax > xmin:
                        bbox_region = np.zeros_like(ctx.current_mask, dtype=bool)
                        bbox_region[ymin:ymax, xmin:xmax] = True
                        merge_region = bbox_region if merge_region is None else (merge_region & bbox_region)

            if merge_region is not None:
                merged = ctx.current_mask.copy()
                merged[merge_region] = result[merge_region]
                if is_valid_candidate(merged):
                    result = merged
            elif enforce_directional_execution and (direction != "global") and (not is_hard_split_relabel):
                reason = "directional_execution_region_missing"
                self.log(f"  [{step_idx}] {op_name}: REJECTED ({reason})")
                ops_failed.append({"operation": op_name, "reason": reason})
                continue

            # Check pixel change
            px_changed = int(np.sum(result != ctx.current_mask))
            if px_changed == 0:
                self.log(f"  [{step_idx}] {op_name}: no change")
                continue
            total_px = max(1, int(ctx.current_mask.size))
            change_ratio = float(px_changed) / float(total_px)
            is_high_impact = _is_high_impact_op(op_name)

            # Label-repair safeguards for split relabel ops:
            # require true fragmentation and enforce localized edits.
            if is_hard_split_relabel and split_relabel_topology_gate:
                source_class = 3 if op_name == "lv_split_to_rv" else 1
                before_cc = _component_count(ctx.current_mask == source_class)
                after_cc = _component_count(result == source_class)
                if before_cc <= 1:
                    reason = f"split_source_not_fragmented (cc_before={before_cc})"
                    self.log(f"  [{step_idx}] {op_name}: REJECTED ({reason})")
                    ops_failed.append({"operation": op_name, "reason": reason})
                    continue
                if after_cc >= before_cc:
                    reason = f"split_fragmentation_not_reduced ({before_cc}->{after_cc})"
                    self.log(f"  [{step_idx}] {op_name}: REJECTED ({reason})")
                    ops_failed.append({"operation": op_name, "reason": reason})
                    continue
                if change_ratio > split_relabel_max_change_ratio:
                    reason = (
                        f"split_change_ratio {change_ratio:.4f} exceeds "
                        f"max {split_relabel_max_change_ratio:.4f}"
                    )
                    self.log(f"  [{step_idx}] {op_name}: REJECTED ({reason})")
                    ops_failed.append({"operation": op_name, "reason": reason})
                    continue

            # Oracle Dice Check (Logging Only)
            oracle_delta = 0.0
            new_dice = 0.0
            if gt_mask is not None:
                try:
                    new_dice = dice_macro(result, gt_mask)[0]
                    oracle_delta = new_dice - current_dice
                    self.log(f"  [Oracle] {op_name}: Dice {current_dice:.4f} -> {new_dice:.4f} (Delta {oracle_delta:+.4f})")
                except Exception:
                    pass

            # Trust-region guard: reject oversized edits before VLM gating.
            max_high_ratio = float(ex_cfg.get("max_high_impact_change_ratio", 0.20))
            max_normal_ratio = float(ex_cfg.get("max_step_change_ratio", 0.12))
            forced_atlas_max_ratio = float(
                ex_cfg.get("forced_atlas_max_step_change_ratio", 0.45)
            )
            limit_ratio = max_high_ratio if is_high_impact else max_normal_ratio
            if is_forced_atlas:
                limit_ratio = max(limit_ratio, forced_atlas_max_ratio)
                self.log(
                    f"  [{step_idx}] {op_name}: forced-atlas relax "
                    f"change_ratio_limit -> {limit_ratio:.3f}"
                )
            if change_ratio > limit_ratio:
                reason = (
                    f"change_ratio {change_ratio:.4f} exceeds limit {limit_ratio:.4f} "
                    f"for {'high-impact' if is_high_impact else 'normal'} op"
                )
                self.log(f"  [{step_idx}] {op_name}: REJECTED ({reason})")
                ops_failed.append({"operation": op_name, "reason": reason})
                continue

            # ── Rule-based quality gate (GT-free) ──
            accepted = True
            gate_reason = ""
            _CLASS_IDS = {1: "RV", 2: "Myo", 3: "LV"}
            target_cls_name = step.get("affected_class", "").lower()
            op_lower = op_name.lower()
            is_erode = "erode" in op_lower or "prune" in op_lower or "suppress" in op_lower
            is_grow = "dilate" in op_lower or "expand" in op_lower or "grow" in op_lower
            is_repair = "repair" in op_lower or "atlas" in op_lower or "transfer" in op_lower
            nontarget_min_ratio = float(ex_cfg.get("nontarget_min_area_ratio", 0.70))
            nontarget_max_ratio = float(ex_cfg.get("nontarget_max_area_ratio", 1.50))
            if is_forced_atlas:
                nontarget_min_ratio = float(
                    ex_cfg.get("forced_atlas_nontarget_min_area_ratio", 0.55)
                )
                nontarget_max_ratio = float(
                    ex_cfg.get("forced_atlas_nontarget_max_area_ratio", 1.80)
                )

            # Get neighbor areas for reference (GT-free baseline)
            nb_areas = {}
            if ctx.neighbor_masks:
                nb_mask = list(ctx.neighbor_masks.values())[0]
                for cid in _CLASS_IDS:
                    nb_areas[cid] = int(np.count_nonzero(nb_mask == cid))

            for cid, cname in _CLASS_IDS.items():
                if not accepted:
                    break
                before_area = int(np.count_nonzero(ctx.current_mask == cid))
                after_area = int(np.count_nonzero(result == cid))

                # Gate 1: Class survival — reject if a present class disappears
                if before_area > 0 and after_area == 0:
                    accepted = False
                    gate_reason = f"{cname}_disappeared (was {before_area}px)"
                    break

                if before_area == 0:
                    continue

                area_ratio = after_area / before_area
                is_targeted = cname.lower() == target_cls_name

                # Gate 2: Area preservation for non-targeted classes
                # Repair ops are allowed larger changes on the targeted class
                # 'neighbor' ops are radical repairs -> exempt them from strict area checks
                is_neighbor_op = "neighbor" in op_lower
                if not is_targeted and not is_neighbor_op:
                    if area_ratio < nontarget_min_ratio:
                        accepted = False
                        gate_reason = (
                            f"{cname}_area_drop={area_ratio:.2f} "
                            f"({before_area}->{after_area}px, min={nontarget_min_ratio:.2f})"
                        )
                        break
                    if is_grow and area_ratio > nontarget_max_ratio:
                        accepted = False
                        gate_reason = (
                            f"{cname}_area_explode={area_ratio:.2f} "
                            f"({before_area}->{after_area}px, max={nontarget_max_ratio:.2f})"
                        )
                        break

                # Gate 3: Neighbor consistency — after the op, class area should
                # not exceed 2x the neighbor's area (prevents over-filling)
                # Exempt 'neighbor' ops because they explicitly copy from neighbor (so area will match)
                if (
                    nb_areas.get(cid, 0) > 0
                    and not is_repair
                    and not is_neighbor_op
                    and not is_forced_atlas
                ):
                    if after_area > nb_areas[cid] * 2.0:
                        accepted = False
                        gate_reason = (
                            f"{cname}_exceeds_neighbor: {after_area}px > "
                            f"2x neighbor {nb_areas[cid]}px"
                        )
                        break

            self.log(
                f"  [{step_idx}] {op_name}: "
                f"{'ACCEPTED' if accepted else 'REJECTED'} "
                f"(chg={px_changed}px"
                f"{f', dice_d={oracle_delta:+.4f}' if gt_mask is not None else ''}"
                f"{f', reason={gate_reason}' if not accepted else ''})"
            )

            # Save debug overlay
            vlm_step_verdict = {}
            if ctx.img_path:
                try:
                    step_overlay = ctx.debug_img_path(f"_step{step_idx}_overlay.png")
                    if create_overlay(ctx.img_path, result, step_overlay):
                        debug_path = os.path.join(debug_dir, f"step_{step_idx}_{op_name}.png")
                        shutil.copy(step_overlay, debug_path)
                        if not accepted:
                            shutil.move(debug_path, debug_path.replace(".png", "_REJECTED.png"))
                except Exception:
                    pass

            if not accepted:
                ops_failed.append({"operation": op_name, "reason": gate_reason})
                continue

            hard_gate_ok, hard_gate_reason, hard_gate_suggestions = self._run_hard_anatomy_gates(
                before_mask=ctx.current_mask,
                after_mask=result,
                op_name=op_name,
                affected_class=affected_class,
                direction=direction,
                view_type=ctx.view_type,
                neighbor_masks=ctx.neighbor_masks,
                valve_y=vy,
                ex_cfg=ex_cfg,
            )
            if not hard_gate_ok:
                self.log(f"  [{step_idx}] {op_name}: REJECTED ({hard_gate_reason})")
                fail_item: Dict[str, Any] = {"operation": op_name, "reason": hard_gate_reason}
                if hard_gate_suggestions:
                    fail_item["suggested_ops"] = [
                        s for s in hard_gate_suggestions if s in op_fns
                    ]
                ops_failed.append(fail_item)
                continue

            # ── Per-step VLM scoring gate ──
            vlm_step_score = 0.0
            vlm_step_score_prev = ctx.current_score or 0.0
            vlm_step_conf = 0.0
            vlm_step_has_confident_score = False
            vlm_score_rejected = False
            vlm_reject_reason = "VLM score gate failed"

            if vlm_per_step and ctx.img_path and self.visual_kb:
                try:
                    step_score_overlay = ctx.debug_img_path(f"_step{step_idx}_score.png")
                    if create_overlay(ctx.img_path, result, step_score_overlay):
                        vs = self.judge_visual_quality(step_score_overlay)
                        if isinstance(vs, dict):
                            vlm_step_verdict = dict(vs)
                        new_score = float(vs.get("score", 0) or 0)
                        new_conf = float(vs.get("confidence", 0) or 0)
                        vlm_step_score = new_score
                        vlm_step_conf = new_conf
                        conf_threshold = max(vlm_min_conf, strict_step_min_conf if strict_monotonic_mode else vlm_min_conf)
                        vlm_step_has_confident_score = bool(
                            new_score > 0 and new_conf >= conf_threshold
                        )

                        if strict_monotonic_mode and strict_reject_low_conf and not vlm_step_has_confident_score:
                            vlm_score_rejected = True
                            vlm_reject_reason = (
                                f"strict_low_confidence_score "
                                f"(score={new_score:.2f}, conf={new_conf:.2f}, min_conf={conf_threshold:.2f})"
                            )
                            self.log(f"  [{step_idx}] {op_name}: REJECTED ({vlm_reject_reason})")
                        elif (
                            vlm_step_has_confident_score
                            and ctx.current_score is not None
                            and ctx.current_score > 0
                            and strict_monotonic_mode
                            and new_score < ctx.current_score + strict_min_score_delta
                        ):
                            vlm_score_rejected = True
                            vlm_reject_reason = (
                                f"strict_non_monotonic "
                                f"({ctx.current_score:.2f}->{new_score:.2f}, "
                                f"need +{strict_min_score_delta:.2f}, conf={new_conf:.2f})"
                            )
                            self.log(f"  [{step_idx}] {op_name}: REJECTED ({vlm_reject_reason})")
                        elif (
                            vlm_step_has_confident_score
                            and ctx.current_score is not None
                            and ctx.current_score > 0
                            and (not strict_monotonic_mode)
                            and new_score < ctx.current_score - vlm_score_tol
                        ):
                            vlm_score_rejected = True
                            vlm_reject_reason = (
                                f"score_drop {ctx.current_score:.2f}->{new_score:.2f} "
                                f"(tol={vlm_score_tol:.2f}, conf={new_conf:.2f})"
                            )
                            self.log(
                                f"  [{step_idx}] {op_name}: VLM SCORE REJECTED "
                                f"({ctx.current_score:.1f} -> {new_score:.1f}, "
                                f"tol={vlm_score_tol}, conf={new_conf:.2f})"
                            )
                        elif vlm_step_has_confident_score:
                            # Valid score: update tracking
                            if (
                                strict_monotonic_mode
                                and (ctx.current_score is None or ctx.current_score <= 0)
                            ):
                                self.log(
                                    f"  [{step_idx}] {op_name}: strict baseline established "
                                    f"at {new_score:.1f} (conf={new_conf:.2f})"
                                )
                            else:
                                self.log(
                                    f"  [{step_idx}] {op_name}: VLM score "
                                    f"{ctx.current_score or 0:.1f} -> {new_score:.1f} "
                                    f"(conf={new_conf:.2f})"
                                )
                        else:
                            self.log(
                                f"  [{step_idx}] {op_name}: VLM score low-confidence "
                                f"(score={new_score:.2f}, conf={new_conf:.2f})"
                            )
                except Exception as e:
                    self.log(f"  [{step_idx}] {op_name}: VLM per-step scoring failed: {e}")
                    if strict_monotonic_mode:
                        vlm_score_rejected = True
                        vlm_reject_reason = f"strict_vlm_scoring_failed: {e}"

            if vlm_score_rejected:
                ops_failed.append({"operation": op_name, "reason": vlm_reject_reason})
                continue

            # Accept the step
            ctx.current_mask = result
            if gt_mask is not None:
                current_dice = new_dice

            # Update VLM score tracking
            if vlm_per_step and vlm_step_has_confident_score:
                ctx.current_score = vlm_step_score
                ctx.score_history.append(vlm_step_score)

            # RQS removed. Defaulting values for backward compatibility in logs/CSV.
            current_rqs = 0.0
            rqs_delta = 0.0

            step_record = {
                "name": op_name,
                "pixels_changed": px_changed,
                "rqs_before": 0.0,
                "rqs_after": 0.0,
                "rqs_delta": 0.0,
                "vlm_verdict": vlm_step_verdict.get("quality", "N/A"),
                "vlm_score": vlm_step_score,
                "vlm_score_prev": vlm_step_score_prev,
                "vlm_confidence": vlm_step_conf,
                "vlm_compare_verdict": "N/A",
                "vlm_compare_confidence": 0.0,
                "change_ratio": round(change_ratio, 6),
                "oracle_dice_delta": round(oracle_delta, 6) if gt_mask is not None else 0.0
            }
            ctx.applied_ops.append(step_record)
            steps_applied.append(step_record)

            # Rebuild op functions with updated mask state
            vy = valve_plane_y(ctx.current_mask)
            op_fns = _build_op_fn(
                ctx.current_mask, ctx.view_type, self.cfg, E, vy, self._atlas,
                neighbor_masks=ctx.neighbor_masks,
                neighbor_img_paths=ctx.neighbor_img_paths,
                target_img=ctx.image,
                diagnosis_issues=ctx.triage_issues,
            )
            _add_atlas_ops(op_fns, ctx.current_mask, ctx.view_type, self._atlas, self.cfg)

            self.log(
                f"  [{step_idx}] {op_name}: {px_changed}px, "
                f"VLM={vlm_step_verdict.get('quality', 'N/A')} "
            )

            # Send step update
            self.send_message(
                recipient="coordinator",
                msg_type=MessageType.STEP_COMPLETE,
                content=step_record,
                case_id=ctx.case_id,
            )

        # Report failures
        if ops_failed:
            self.send_message(
                recipient="planner",
                msg_type=MessageType.OP_FAILED,
                content={"failed_ops": ops_failed},
                case_id=ctx.case_id,
            )

        # Final summary
        self.log(
            f"  Done: {len(steps_applied)}/{len(plan)} ops applied"
        )

        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.TASK_RESULT,
            content={
                "steps_applied": len(steps_applied),
                "steps_total": len(plan),
                "rqs_final": 0.0,
                "failed_ops": ops_failed,
            },
            case_id=ctx.case_id,
        )

    @staticmethod
    def _gate_repair_hints(
        reason: str,
        affected_class: str,
        direction: str,
        view_type: str,
    ) -> List[str]:
        r = str(reason or "").lower()
        cls = str(affected_class or "myo").strip().lower()
        if "required_class" in r or "optional_class" in r:
            if cls == "rv":
                return ["neighbor_rv_repair", "atlas_rv_repair", "expand_rv_intensity", "1_dilate"]
            if cls == "lv":
                return ["neighbor_lv_repair", "atlas_lv_repair", "expand_lv_intensity", "3_dilate"]
            return ["neighbor_myo_repair", "atlas_myo_repair", "expand_myo_intensity", "2_dilate"]
        if "lv_myo_containment" in r:
            return ["myo_bridge", "morphological_gap_close", "2_dilate", "topology_cleanup"]
        if "rv_lv_contact_increase" in r:
            return ["rv_lv_barrier", "neighbor_myo_repair", "myo_bridge"]
        if "interslice_" in r:
            return ["neighbor_shape_prior", "neighbor_centroid_align", "neighbor_area_consistency", "smooth_morph"]
        if "split_" in r:
            return ["lv_split_to_rv", "rv_split_to_lv", "atlas_rv_repair", "atlas_lv_repair"]
        return []

    def _run_hard_anatomy_gates(
        self,
        before_mask: np.ndarray,
        after_mask: np.ndarray,
        op_name: str,
        affected_class: str,
        direction: str,
        view_type: str,
        neighbor_masks: Dict[int, np.ndarray],
        valve_y: Optional[int],
        ex_cfg: Dict[str, Any],
    ) -> tuple[bool, str, List[str]]:
        """
        Hard anatomy gates that enforce clinically meaningful constraints.
        Returns (accepted, reason, suggested_ops).
        """
        required_ids = _required_class_ids_for_view(view_type)
        optional_min_area = int(ex_cfg.get("gate_optional_disappear_min_area_px", 120))
        for cid in (1, 2, 3):
            before_area = _class_area(before_mask, cid)
            after_area = _class_area(after_mask, cid)
            cname = _CLASS_NAME_BY_ID.get(cid, str(cid))
            if cid in required_ids:
                if before_area > 0 and after_area == 0:
                    reason = f"required_class_{cname}_disappeared"
                    return False, reason, self._gate_repair_hints(reason, cname, direction, view_type)
            else:
                if before_area >= optional_min_area and after_area == 0:
                    reason = f"optional_class_{cname}_hard_drop ({before_area}->0)"
                    return False, reason, self._gate_repair_hints(reason, cname, direction, view_type)

        lv_myo_gate = bool(ex_cfg.get("gate_lv_myo_enable", True))
        if lv_myo_gate and 3 in required_ids:
            valve_margin = int(ex_cfg.get("gate_valve_exempt_margin_px", 6))
            min_cov = float(ex_cfg.get("gate_lv_myo_min_coverage", 0.70))
            allow_drop = float(ex_cfg.get("gate_lv_myo_allow_drop", 0.03))
            cov_before = _lv_myo_coverage(before_mask, valve_y=valve_y, valve_margin_px=valve_margin)
            cov_after = _lv_myo_coverage(after_mask, valve_y=valve_y, valve_margin_px=valve_margin)
            if cov_after < min_cov and cov_after < (cov_before - allow_drop):
                reason = f"lv_myo_containment_drop {cov_before:.3f}->{cov_after:.3f}"
                return False, reason, self._gate_repair_hints(reason, "myo", direction, view_type)

        contact_gate = bool(ex_cfg.get("gate_rv_lv_contact_enable", True))
        if contact_gate and str(op_name).strip() not in _CONTACT_INCREASE_ALLOWED_OPS:
            allow_inc = int(ex_cfg.get("gate_rv_lv_contact_allow_increase_px", 2))
            c_before = _rv_lv_contact_len(before_mask)
            c_after = _rv_lv_contact_len(after_mask)
            if c_after > (c_before + allow_inc):
                reason = f"rv_lv_contact_increase {c_before}->{c_after}"
                return False, reason, self._gate_repair_hints(reason, "myo", direction, view_type)

        interslice_gate = bool(ex_cfg.get("gate_interslice_enable", True))
        if interslice_gate and neighbor_masks:
            min_nb_area = int(ex_cfg.get("gate_interslice_min_neighbor_area_px", 25))
            area_dev_max = float(ex_cfg.get("gate_interslice_area_dev_max", 1.20))
            area_worsen_tol = float(ex_cfg.get("gate_interslice_area_dev_worsen_tol", 0.12))
            centroid_max = float(ex_cfg.get("gate_interslice_centroid_max_shift_px", 35.0))
            centroid_worsen = float(ex_cfg.get("gate_interslice_centroid_worsen_tol_px", 5.0))
            cc_dev_max = float(ex_cfg.get("gate_interslice_cc_max_dev", 2.0))
            cc_worsen = float(ex_cfg.get("gate_interslice_cc_worsen_tol", 1.0))

            cls_id = _CLASS_ID_BY_NAME.get(str(affected_class).strip().lower())
            class_ids = set(required_ids)
            if cls_id is not None:
                class_ids.add(int(cls_id))
            for cid in sorted(class_ids):
                nb_stats = _neighbor_stats_for_class(
                    neighbor_masks=neighbor_masks,
                    class_id=cid,
                    min_area_px=min_nb_area,
                )
                if not nb_stats:
                    continue

                cname = _CLASS_NAME_BY_ID.get(cid, str(cid))
                nb_area = float(nb_stats.get("mean_area", 0.0))
                before_area = float(_class_area(before_mask, cid))
                after_area = float(_class_area(after_mask, cid))
                if nb_area > 0:
                    area_dev_before = abs(before_area - nb_area) / max(1.0, nb_area)
                    area_dev_after = abs(after_area - nb_area) / max(1.0, nb_area)
                    if (
                        area_dev_after > area_dev_max
                        and area_dev_after > area_dev_before + area_worsen_tol
                    ):
                        reason = (
                            f"interslice_area_jump_{cname} "
                            f"{area_dev_before:.2f}->{area_dev_after:.2f}"
                        )
                        return (
                            False,
                            reason,
                            self._gate_repair_hints(reason, cname, direction, view_type),
                        )

                nb_centroid = nb_stats.get("centroid")
                before_centroid = _class_centroid(before_mask, cid)
                after_centroid = _class_centroid(after_mask, cid)
                dist_before = _euclid(before_centroid, nb_centroid)
                dist_after = _euclid(after_centroid, nb_centroid)
                if dist_after is not None:
                    db = dist_before if dist_before is not None else dist_after
                    if dist_after > centroid_max and dist_after > db + centroid_worsen:
                        reason = (
                            f"interslice_centroid_shift_{cname} "
                            f"{db:.1f}->{dist_after:.1f}px"
                        )
                        return (
                            False,
                            reason,
                            self._gate_repair_hints(reason, cname, direction, view_type),
                        )

                before_cc = float(_component_count(before_mask == cid))
                after_cc = float(_component_count(after_mask == cid))
                nb_cc = float(nb_stats.get("mean_cc", 0.0))
                cc_dev_before = abs(before_cc - nb_cc)
                cc_dev_after = abs(after_cc - nb_cc)
                if cc_dev_after > cc_dev_max and cc_dev_after > cc_dev_before + cc_worsen:
                    reason = (
                        f"interslice_cc_jump_{cname} "
                        f"{cc_dev_before:.1f}->{cc_dev_after:.1f}"
                    )
                    return (
                        False,
                        reason,
                        self._gate_repair_hints(reason, cname, direction, view_type),
                    )

        return True, "", []
