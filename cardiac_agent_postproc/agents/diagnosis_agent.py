"""
DiagnosisAgent: LLM-powered spatial defect diagnosis.

Combines rule-based feature analysis, VLM overlay inspection, and LLM
reasoning to produce prioritized, spatially-aware diagnoses.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi

from ..quality_model import extract_quality_features
from ..slice_consistency import detect_misalignment
from ..vlm_guardrail import create_overlay
from .base_agent import BaseAgent
from .message_bus import CaseContext, MessageBus, MessageType


# ---- Canonical issue → ops mapping ----

ISSUE_OPS_MAP: Dict[str, List[str]] = {
    # Structural breaks (high severity)
    # Prefer topology_cleanup/bridge first, convex_hull only if severe
    "broken_ring": ["neighbor_myo_repair", "morphological_gap_close", "myo_bridge_strong", "myo_bridge", "myo_bridge_mild", "topology_cleanup", "convex_hull_fill"],
    "disconnected_myo": ["neighbor_myo_repair", "morphological_gap_close", "myo_bridge_strong", "myo_bridge", "myo_bridge_mild", "topology_cleanup", "convex_hull_fill"],
    "fragmented_lv": ["atlas_rv_repair", "lv_split_to_rv", "lv_bridge_strong", "lv_bridge", "lv_bridge_mild", "safe_topology_fix", "topology_cleanup"],
    "fragmented_rv": ["atlas_lv_repair", "rv_split_to_lv", "safe_topology_fix", "topology_cleanup"],
    "touching": ["neighbor_myo_repair", "rv_lv_barrier_strong", "rv_lv_barrier", "rv_lv_barrier_mild", "myo_bridge", "rv_erode"],
    "rv_lv_touching": ["neighbor_myo_repair", "rv_lv_barrier_strong", "rv_lv_barrier", "rv_lv_barrier_mild", "myo_bridge", "rv_erode"],
    # Boundary position errors (directional dilate/erode)
    "myo_too_thin": ["2_dilate", "close2_open3", "expand_myo_intensity"],
    "myo_too_thick": ["2_erode_expand_lv", "2_erode", "smooth_morph", "2_erode_expand_rv"],
    "lv_too_large": ["3_erode_expand_myo", "3_erode", "smooth_morph"],
    "lv_too_small": ["3_dilate", "expand_lv_intensity"],
    "rv_too_large": ["1_erode_expand_myo", "rv_erode"],
    "rv_too_small": ["1_dilate", "expand_rv_intensity"],
    "too_thin": ["2_dilate", "close2_open3"],
    "boundary_error": ["3_erode", "3_dilate", "2_dilate", "2_erode"],
    # Holes and artifacts
    "holes": ["fill_holes", "fill_holes_m2", "morph_close_large", "safe_topology_fix", "expand_myo_intensity"],
    "noise_island": ["topology_cleanup_strong", "topology_cleanup_mild", "topology_cleanup", "safe_topology_fix", "remove_islands"],
    "valve_leak": ["valve_suppress", "safe_topology_fix"],
    "jagged_boundary": ["smooth_morph", "smooth_morph_k5", "smooth_contour"],
    "edge_misalignment": ["edge_snap_proxy", "smooth_contour", "smooth_morph"],
    "myo_thickness_var": ["smooth_morph", "smooth_morph_k5", "topology_cleanup"],
    # Missing structures — prefer neighbor-based repair, then global atlas
    "missing_structure": ["neighbor_rv_repair", "neighbor_myo_repair", "neighbor_lv_repair", "atlas_growprune_0"],
    "missing_myo": ["neighbor_myo_repair", "convex_hull_fill", "atlas_growprune_0"],
    "missing_lv": ["neighbor_lv_repair", "atlas_growprune_0", "expand_lv_intensity"],
    "missing_rv": ["neighbor_rv_repair", "atlas_growprune_0", "expand_rv_intensity"],
    # Inter-slice consistency issues
    "centroid_shift": ["neighbor_centroid_align", "neighbor_shape_prior"],
    "area_inconsistency": ["neighbor_area_consistency", "neighbor_shape_prior"],
    "slice_misalignment": ["neighbor_centroid_align", "neighbor_shape_prior", "neighbor_area_consistency"],
    # Severe structural issues requiring atlas-guided reconstruction
    "severe_misplacement": ["neighbor_rv_repair", "neighbor_lv_repair", "individual_atlas_transfer", "atlas_growprune_0"],
    "massive_missing": ["neighbor_rv_repair", "neighbor_myo_repair", "neighbor_lv_repair", "individual_atlas_transfer", "atlas_growprune_0"],
    # Semantic overrides for severe fragmentation
    "missing_lv_fragments": ["atlas_rv_repair", "lv_split_to_rv", "neighbor_lv_repair", "atlas_growprune_0"],
    "missing_rv_fragments": ["atlas_lv_repair", "rv_split_to_lv", "neighbor_rv_repair", "atlas_growprune_0"],
}

_CLASS_LABEL = {"rv": 1, "myo": 2, "lv": 3}


@dataclass
class SpatialDiagnosis:
    """A single spatial diagnosis — what's wrong, where, and how to fix it."""
    issue: str
    affected_class: str  # "rv", "myo", or "lv"
    direction: str       # "septal", "lateral", "apical", "basal", "global"
    severity: str        # "high", "medium", "low"
    operation: str       # recommended op name
    description: str
    confidence: float = 0.8
    source: str = "rules"  # "rules", "vlm", "llm"
    bbox: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax] in pixels


def normalize_direction_hint(direction: str, description: str = "") -> str:
    """
    Normalize free-form direction hints into canonical labels supported by the
    pipeline: septal|lateral|apical|basal|left|right|global.
    """
    canonical = {"septal", "lateral", "apical", "basal", "left", "right", "global"}
    raw = str(direction or "").strip().lower()
    if raw in canonical:
        return raw

    text = f"{raw} {str(description or '').strip().lower()}"
    for ch in (",", ";", ":", "/", "|", "-", "_", "(", ")", "[", "]", "{", "}"):
        text = text.replace(ch, " ")
    tokens = [tok for tok in text.split() if tok]

    token_to_direction = {
        "septal": "septal",
        "septum": "septal",
        "interventricular": "septal",
        "medial": "septal",
        "lateral": "lateral",
        "freewall": "lateral",
        "outer": "lateral",
        "apical": "apical",
        "apex": "apical",
        "inferior": "apical",
        "lower": "apical",
        "bottom": "apical",
        "caudal": "apical",
        "basal": "basal",
        "base": "basal",
        "superior": "basal",
        "upper": "basal",
        "top": "basal",
        "cranial": "basal",
        "left": "left",
        "right": "right",
        "global": "global",
        "whole": "global",
        "entire": "global",
        "all": "global",
    }

    for idx, tok in enumerate(tokens):
        mapped = token_to_direction.get(tok)
        if mapped:
            return mapped
        # Handle phrase "free wall".
        if tok == "free" and idx + 1 < len(tokens) and tokens[idx + 1] == "wall":
            return "lateral"

    return "global"


def compute_region_mask(
    mask: np.ndarray,
    affected_class: str,
    direction: str,
) -> np.ndarray:
    """
    Compute a boolean region mask (H, W) indicating where to apply an operation.
    Uses LV centroid as the anatomical reference point.
    """
    H, W = mask.shape
    direction = normalize_direction_hint(direction)
    region = np.ones((H, W), dtype=bool)

    if direction == "global":
        return region

    # Find LV centroid as reference
    lv_mask = (mask == 3)
    if lv_mask.any():
        lv_y, lv_x = np.where(lv_mask)
        cy, cx = float(lv_y.mean()), float(lv_x.mean())
    else:
        cy, cx = H / 2, W / 2

    # Find RV centroid for septal/lateral axis
    rv_mask = (mask == 1)
    if rv_mask.any():
        rv_y, rv_x = np.where(rv_mask)
        rv_cx = float(rv_x.mean())
    else:
        rv_cx = cx - W * 0.3  # default: RV is to the left

    if direction == "apical":
        region[:int(cy), :] = False  # keep bottom half
    elif direction == "basal":
        region[int(cy):, :] = False  # keep top half
    elif direction == "septal":
        # Septal = between LV and RV
        if rv_cx < cx:
            region[:, int(cx):] = False  # keep left side
        else:
            region[:, :int(cx)] = False  # keep right side
    elif direction == "lateral":
        # Lateral = opposite of septal
        if rv_cx < cx:
            region[:, :int(cx)] = False  # keep right side
        else:
            region[:, int(cx):] = False  # keep left side
    elif direction == "left":
        region[:, W // 2:] = False
    elif direction == "right":
        region[:, :W // 2] = False

    return region


class DiagnosisAgent(BaseAgent):
    """
    LLM-powered diagnosis agent.

    Pipeline:
    1. Rule-based feature analysis → initial issue list
    2. VLM overlay analysis → spatial defect detection (if enabled)
    3. LLM synthesis → prioritized, spatially-aware diagnosis plan
    """

    def __init__(self, cfg: dict, bus: MessageBus):
        self._diagnosis_prompt = ""
        diag_cfg = cfg.get("diagnosis", {})
        # Keep a diagnosis-specific gate without shadowing BaseAgent._vlm_enabled().
        self._diag_vlm_enabled = bool(diag_cfg.get("vlm_enabled", False))
        self._max_suggestions = int(diag_cfg.get("max_suggestions", 4))
        super().__init__(cfg, bus)

    @property
    def name(self) -> str:
        return "diagnosis"

    @property
    def system_prompt(self) -> str:
        fallback = (
            "You are the Diagnosis Agent in a cardiac MRI segmentation repair system.\n\n"
            "Your role is to identify specific defects in segmentation masks and suggest "
            "targeted, DIRECTIONAL repair operations.\n\n"
            "For each defect, specify:\n"
            "- issue: the defect type\n"
            "- affected_class: 'rv', 'myo', or 'lv'\n"
            "- direction: 'septal', 'lateral', 'apical', 'basal', or 'global'\n"
            "- severity: 'high', 'medium', or 'low'\n"
            "- operation: the SINGLE best repair operation\n"
            "- description: brief SPATIAL description (e.g. 'Myo too thin at lateral wall')\n\n"
            "CRITICAL: BE CONSERVATIVE. But do use intensity-guided operations (expand_myo_intensity) "
            "if the myocardium is significantly thin or has gaps that simple closing cannot fix. "
            "For minor issues (rough edges, small thickness variations), suggest 'smooth_morph' or '1_dilate'.\n\n"
            "Per-class directional ops: \n"
            "   - Standard (3px): 2_dilate, 2_erode, 3_dilate, 3_erode, 1_dilate\n"
            "   - Coupled: 2_erode_expand_lv, 2_erode_expand_rv, 3_erode_expand_myo, 1_erode_expand_myo\n"
            "   - RV: rv_erode, rv_erode_large (Use with CAUTION - only for obvious over-segmentation)\n"
            "Structural: convex_hull_fill (best for gaps - prefer over intensity), myo_bridge, lv_bridge, rv_lv_barrier.\n"
            "Atlas: atlas_growprune_0 (best for missing parts), atlas_growprune_1, individual_atlas_transfer.\n"
            "Cleanup: topology_cleanup, safe_topology_fix, fill_holes, morph_close_large (for big holes), remove_islands.\n"
            "Smoothing: smooth_morph (3px), smooth_morph_k5 (5px), smooth_contour, smooth_gaussian.\n"
            "Other: close2_open3, open2_close3, edge_snap_proxy, valve_suppress, "
            "neighbor_centroid_align, neighbor_shape_prior, neighbor_area_consistency.\n\n"
            "Respond with JSON. EXAMPLE:\n"
            '{"diagnoses": ['
            '  {"issue": "broken_ring", "affected_class": "myo", "direction": "septal", '
            '   "severity": "high", "operation": "convex_hull_fill", "confidence": 0.95, '
            '   "description": "Gap in septal myocardium", '
            '   "bbox_2d": [120, 140, 160, 180]}, '
            '  {"issue": "touching", "affected_class": "rv", "direction": "global", '
            '   "severity": "medium", "operation": "rv_lv_barrier", "confidence": 0.85, '
            '   "description": "RV contacting LV", '
            '   "bbox_2d": [200, 200, 250, 250]}'
            ']}\n'
            "IMPORTANT: For spatial defects (broken_ring, holes, extra blobs), YOU MUST PROVIDE 'bbox_2d' [ymin, xmin, ymax, xmax] (0-1000 scale). Missing bbox will cause repair failure."
        )
        return self._prompt_with_fallback("diagnosis_system.txt", fallback)

    def process_case(self, ctx: CaseContext) -> None:
        """
        Diagnose defects in a single case.
        Updates ctx.diagnoses with SpatialDiagnosis objects.
        """
        assert ctx.mask is not None and ctx.image is not None
        self.log(f"Diagnosing {ctx.stem}")

        # Step 1: Rule-based feature analysis
        feats = extract_quality_features(ctx.mask, ctx.image, ctx.view_type)
        hole_frac_myo = feats.get("hole_fraction_myo", feats.get("hole_frac_myo", 0))
        hole_frac_lv = feats.get("hole_fraction_lv", feats.get("hole_frac_lv", 0))
        rule_diagnoses = self._diagnose_rules(
            ctx.mask, feats, ctx.view_type, ctx.triage_issues,
            neighbor_masks=ctx.neighbor_masks,
        )

        # Step 2: VLM analysis (if enabled)
        vlm_diagnoses = []
        if self._diag_vlm_enabled and self._vlm_enabled() and ctx.img_path:
            vlm_diagnoses = self._diagnose_vlm(ctx)

        # Step 3: LLM synthesis — combine all signals
        all_signals = {
            "stem": ctx.stem,
            "view_type": ctx.view_type,
            "triage_issues": ctx.triage_issues,
            "rule_diagnoses": [
                {"issue": d.issue, "affected_class": d.affected_class,
                 "direction": d.direction, "severity": d.severity,
                 "operation": d.operation, "confidence": d.confidence}
                for d in rule_diagnoses
            ],
            "vlm_diagnoses": [
                {"issue": d.issue, "affected_class": d.affected_class,
                 "direction": d.direction, "severity": d.severity,
                 "operation": d.operation, "confidence": d.confidence,
                 "description": d.description}
                for d in vlm_diagnoses
            ],
            "key_features": {
                "cc_myo": feats.get("cc_count_myo", 0),
                "cc_lv": feats.get("cc_count_lv", 0),
                "cc_rv": feats.get("cc_count_rv", 0),
                "touch_ratio": round(feats.get("touch_ratio", 0), 4),
                "thickness_cv": round(feats.get("thickness_cv", 0), 3),
                "hole_frac_myo": round(hole_frac_myo, 4),
                "valve_leak": round(feats.get("valve_leak_ratio", 0), 4),
            },
            "verifier_feedback": ctx.verifier_feedback,  # Inject feedback from previous rounds
        }

        prompt = (
            f"Analyze the following signals for case {ctx.stem} and produce a "
            f"final prioritized diagnosis (max {self._max_suggestions} items).\n\n"
            f"Signals:\n{_format_dict(all_signals)}\n\n"
            f"Synthesize rule-based and VLM findings. Remove duplicates, "
            f"resolve conflicts, and rank by severity. For each diagnosis, "
            f"choose the single best operation.\n\n"
            f"CONTEXTUAL HINTS:\n"
            f"- Thickness CV is {feats.get('thickness_cv', 0):.3f}. "
            f"If > 0.5, 'myo_too_thin' is likely a False Positive artifact. "
            f"Prefer 'topology_cleanup' or 'smooth_morph' over 'dilate'."
        )

        result = self.think_json(prompt)
        llm_diagnoses = self._parse_llm_diagnoses(result, source="llm", img_shape=ctx.mask.shape)
        # Merge: prefer LLM synthesis, but append critical/high-confidence rules
        # that are truly missing. Keep LLM items at the front to avoid losing
        # their spatial hints due max_suggestions truncation.
        if llm_diagnoses:
            final_diagnoses = list(llm_diagnoses)
            existing_issues = {(d.issue, d.affected_class) for d in final_diagnoses}
            for rd in rule_diagnoses:
                if rd.severity in ("critical", "high") or rd.issue in (
                    "massive_missing",
                    "severe_misplacement",
                    "neighbor_rv_repair",
                    "neighbor_lv_repair",
                ):
                    if (rd.issue, rd.affected_class) not in existing_issues:
                        # Keep LLM first; add missing critical rules as fallback.
                        final_diagnoses.append(rd)
                        existing_issues.add((rd.issue, rd.affected_class))
        else:
            final_diagnoses = rule_diagnoses

        final_diagnoses = final_diagnoses[:self._max_suggestions]

        # Update context.
        # Preserve diagnosis bbox when available; otherwise infer a directional bbox
        # so downstream executor can localize high-impact ops (e.g. atlas repair).
        ctx.diagnoses = []
        for d in final_diagnoses:
            bbox = d.bbox
            if bbox is None and str(d.direction).strip().lower() != "global":
                bbox = self._infer_directional_bbox(
                    ctx.mask, d.affected_class, d.direction
                )
            if bbox is None:
                issue_lower = str(d.issue).strip().lower()
                if issue_lower in {
                    "fragmented_lv",
                    "missing_lv_fragments",
                    "fragmented_rv",
                    "missing_rv_fragments",
                }:
                    bbox = self._infer_secondary_component_bbox(
                        ctx.mask, d.affected_class
                    )
            ctx.diagnoses.append(
                {
                    "issue": d.issue,
                    "affected_class": d.affected_class,
                    "direction": d.direction,
                    "severity": d.severity,
                    "operation": d.operation,
                    "description": d.description,
                    "confidence": d.confidence,
                    "source": d.source,
                    "bbox": bbox,
                }
            )

        self.log(f"  → {len(final_diagnoses)} diagnoses: "
                 f"{[d.issue for d in final_diagnoses]}")

        # Send result to bus
        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.DIAGNOSIS_RESULT,
            content={"diagnoses": ctx.diagnoses},
            case_id=ctx.case_id,
        )

    @staticmethod
    def _region_to_bbox(region: np.ndarray, margin: int = 12) -> Optional[List[int]]:
        if region is None or not np.any(region):
            return None
        H, W = region.shape
        rows = np.any(region, axis=1)
        cols = np.any(region, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        ymin = max(0, int(ymin) - margin)
        xmin = max(0, int(xmin) - margin)
        ymax = min(H, int(ymax) + 1 + margin)
        xmax = min(W, int(xmax) + 1 + margin)
        if ymax <= ymin or xmax <= xmin:
            return None
        return [ymin, xmin, ymax, xmax]

    @classmethod
    def _infer_directional_bbox(
        cls,
        mask: np.ndarray,
        affected_class: str,
        direction: str,
    ) -> Optional[List[int]]:
        try:
            region = compute_region_mask(mask, affected_class, direction)
            # Tighten to neighborhood of affected structure when that class exists.
            class_id = _CLASS_LABEL.get(str(affected_class).strip().lower())
            if class_id is not None:
                class_mask = (mask == class_id)
                if np.any(class_mask):
                    class_band = ndi.binary_dilation(class_mask, iterations=12)
                    focused = region & class_band
                    if int(np.count_nonzero(focused)) >= 16:
                        region = focused
            return cls._region_to_bbox(region, margin=16)
        except Exception:
            return None

    @classmethod
    def _infer_secondary_component_bbox(
        cls,
        mask: np.ndarray,
        affected_class: str,
    ) -> Optional[List[int]]:
        """
        For fragmented/missing-fragment diagnoses, localize the largest
        non-primary connected component of the affected class.
        """
        try:
            class_id = _CLASS_LABEL.get(str(affected_class).strip().lower())
            if class_id is None:
                return None

            cls_mask = (mask == class_id)
            if int(np.count_nonzero(cls_mask)) == 0:
                return None

            labeled, n_comp = ndi.label(cls_mask.astype(np.uint8))
            if int(n_comp) <= 1:
                return None

            comps: List[tuple[int, int]] = []
            for comp_id in range(1, int(n_comp) + 1):
                size = int(np.count_nonzero(labeled == comp_id))
                if size > 0:
                    comps.append((size, comp_id))
            if len(comps) <= 1:
                return None

            comps.sort(reverse=True)  # primary component first
            _, sec_id = comps[1]  # largest secondary component
            sec_region = (labeled == sec_id)
            if int(np.count_nonzero(sec_region)) < 8:
                return None

            return cls._region_to_bbox(sec_region, margin=16)
        except Exception:
            return None

    def _diagnose_rules(
        self,
        mask: np.ndarray,
        feats: Dict[str, float],
        view_type: str,
        triage_issues: List[str],
        neighbor_masks: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[SpatialDiagnosis]:
        """Rule-based diagnosis using quality features."""
        diagnoses = []
        h, w = mask.shape
        total_px = h * w
        view = view_type if view_type in ("2ch", "3ch", "4ch") else "4ch"
        expected_area = {
            "4ch": {"rv": 0.015, "myo": 0.008, "lv": 0.012},
            "3ch": {"rv": 0.005, "myo": 0.008, "lv": 0.012},
            "2ch": {"rv": 0.0, "myo": 0.008, "lv": 0.012},
        }[view]
        triage_tokens = {
            str(issue).strip().lower()
            for issue in triage_issues
            if issue is not None
        }
        has_discmyo_prior = any(
            t in {"discmyo", "disc_myo", "disconnected_myo", "broken_ring"}
            for t in triage_tokens
        )
        has_lowdice_prior = any(
            t in {
                "lowdice",
                "low_dice",
                "low-dice",
                "boundary_error",
                "area_inconsistency",
            }
            for t in triage_tokens
        )

        # Disconnected myocardium
        cc_myo = int(feats.get("cc_count_myo", 1))
        has_discmyo, gap_px = self._has_meaningful_myo_break(
            mask,
            min_second_ratio=0.04 if has_discmyo_prior else 0.05,
            min_gap_px=2 if has_discmyo_prior else 3,
        )
        if cc_myo > 1 and has_discmyo:
            direction = self._infer_myo_break_direction(mask)
            severity = "high" if (gap_px >= 4 or cc_myo >= 3) else "medium"
            diagnoses.append(SpatialDiagnosis(
                issue="broken_ring", affected_class="myo",
                direction=direction, severity=severity,
                operation=ISSUE_OPS_MAP["broken_ring"][0],
                confidence=0.90 if severity == "high" else 0.78,
                description=f"Myocardium has a meaningful gap (~{gap_px}px) with {cc_myo} components",
                source="rules",
            ))
        elif cc_myo > 1 and has_discmyo_prior:
            direction = self._infer_myo_break_direction(mask)
            diagnoses.append(SpatialDiagnosis(
                issue="broken_ring", affected_class="myo",
                direction=direction, severity="medium",
                operation="myo_bridge", confidence=0.70,
                description=f"DiscMyo prior with split myocardium ({cc_myo} components) suggests a local ring bridge at {direction}",
                source="rules",
            ))
        elif cc_myo > 1:
            diagnoses.append(SpatialDiagnosis(
                issue="noise_island", affected_class="myo",
                direction="global", severity="low",
                operation="topology_cleanup", confidence=0.65,
                description=f"Myocardium has tiny extra components ({cc_myo} total) without a clear ring break",
                source="rules",
            ))

        # Fragmented LV
        if feats.get("cc_count_lv", 1) > 1:
            diagnoses.append(SpatialDiagnosis(
                issue="fragmented_lv", affected_class="lv",
                direction="global", severity="high",
                operation="lv_bridge", confidence=0.85,
                description=f"LV split into {int(feats['cc_count_lv'])} components",
                source="rules",
            ))

        # RV-LV touching
        touch_ratio = float(feats.get("touch_ratio", 0))
        touch_len = self._rv_lv_contact_len(mask)
        if touch_len >= 8 or touch_ratio > 0.03:
            diagnoses.append(SpatialDiagnosis(
                issue="touching", affected_class="rv",
                direction="septal", severity="high",
                operation="rv_lv_barrier", confidence=0.88,
                description=f"RV-LV direct contact (contact_len={touch_len}px, touch_ratio={touch_ratio:.4f})",
                source="rules",
            ))

        # Holes
        hole_frac_myo = feats.get("hole_fraction_myo", feats.get("hole_frac_myo", 0))
        hole_frac_lv = feats.get("hole_fraction_lv", feats.get("hole_frac_lv", 0))

        if hole_frac_myo > 0.02:
            diagnoses.append(SpatialDiagnosis(
                issue="holes", affected_class="myo",
                direction="global", severity="medium",
                operation="fill_holes", confidence=0.8,
                description="Holes inside myocardium region",
                source="rules",
            ))

        if hole_frac_lv > 0.02:
            diagnoses.append(SpatialDiagnosis(
                issue="holes", affected_class="lv",
                direction="global", severity="medium",
                operation="fill_holes", confidence=0.8,
                description="Holes inside LV region",
                source="rules",
            ))

        # Valve leak
        if self._has_basal_leak(mask) or feats.get("valve_leak_ratio", 0) > 0.08:
            diagnoses.append(SpatialDiagnosis(
                issue="valve_leak", affected_class="lv",
                direction="basal", severity="medium",
                operation="valve_suppress", confidence=0.75,
                description="Structure pixels above valve plane",
                source="rules",
            ))

        # Myo thickness analysis — directional
        thickness_mean = float(feats.get("thickness_mean", 0))
        thin_threshold = 4.2 if has_lowdice_prior else 3.8
        if 0 < thickness_mean < thin_threshold:
            # Myo is too thin somewhere specific — try to localize
            # CRITICAL SAFETY: Check for high variation (artifacts)
            if feats.get("thickness_cv", 0) > 0.5:
                diagnoses.append(SpatialDiagnosis(
                    issue="myo_thickness_var", affected_class="myo",
                    direction="global", severity="high",
                    operation="topology_cleanup", confidence=0.75,
                    description=f"High thickness var (cv={feats['thickness_cv']:.3f}) - risky to dilate",
                    source="rules",
                ))
            else:
                thin_dir = self._infer_thin_myo_direction(mask)
                diagnoses.append(SpatialDiagnosis(
                    issue="myo_too_thin", affected_class="myo",
                    direction=thin_dir,
                    severity="high" if thickness_mean < 3.0 else "medium",
                    operation="2_dilate",
                    confidence=0.74,
                    description=f"Myo appears thin (mean thickness={thickness_mean:.1f}px) at {thin_dir}",
                    source="rules",
                ))

        # LowDice-like directional area mismatch (moderate, non-missing)
        area_frac_lv = float(np.count_nonzero(mask == 3)) / max(1, total_px)
        area_frac_rv = float(np.count_nonzero(mask == 1)) / max(1, total_px)
        area_frac_myo = float(np.count_nonzero(mask == 2)) / max(1, total_px)

        exp_lv = expected_area["lv"]
        if exp_lv > 0:
            lv_small_threshold = 0.78 if has_lowdice_prior else 0.72
            lv_large_threshold = 1.38 if has_lowdice_prior else 1.45
            if 0.35 * exp_lv <= area_frac_lv < lv_small_threshold * exp_lv:
                diagnoses.append(SpatialDiagnosis(
                    issue="lv_too_small", affected_class="lv",
                    direction=self._infer_structure_bias_direction(mask, class_label=3, prefer="deficit"),
                    severity="high" if area_frac_lv < 0.55 * exp_lv else "medium",
                    operation="3_dilate", confidence=0.72,
                    description=f"LV area smaller than expected ({area_frac_lv:.4f} vs ~{exp_lv:.4f})",
                    source="rules",
                ))
            elif area_frac_lv > lv_large_threshold * exp_lv:
                diagnoses.append(SpatialDiagnosis(
                    issue="lv_too_large", affected_class="lv",
                    direction=self._infer_structure_bias_direction(mask, class_label=3, prefer="surplus"),
                    severity="high" if area_frac_lv > 1.65 * exp_lv else "medium",
                    operation=ISSUE_OPS_MAP["lv_too_large"][0], confidence=0.72,
                    description=f"LV area larger than expected ({area_frac_lv:.4f} vs ~{exp_lv:.4f})",
                    source="rules",
                ))

        exp_rv = expected_area["rv"]
        if exp_rv > 0:
            rv_small_threshold = 0.80 if has_lowdice_prior else 0.72
            rv_large_threshold = 1.40 if has_lowdice_prior else 1.50
            if 0.35 * exp_rv <= area_frac_rv < rv_small_threshold * exp_rv:
                diagnoses.append(SpatialDiagnosis(
                    issue="rv_too_small", affected_class="rv",
                    direction=self._infer_structure_bias_direction(mask, class_label=1, prefer="deficit"),
                    severity="high" if area_frac_rv < 0.55 * exp_rv else "medium",
                    operation="1_dilate", confidence=0.70,
                    description=f"RV area smaller than expected ({area_frac_rv:.4f} vs ~{exp_rv:.4f})",
                    source="rules",
                ))
            elif area_frac_rv > rv_large_threshold * exp_rv:
                diagnoses.append(SpatialDiagnosis(
                    issue="rv_too_large", affected_class="rv",
                    direction=self._infer_structure_bias_direction(mask, class_label=1, prefer="surplus"),
                    severity="high" if area_frac_rv > 1.65 * exp_rv else "medium",
                    operation=ISSUE_OPS_MAP["rv_too_large"][0], confidence=0.70,
                    description=f"RV area larger than expected ({area_frac_rv:.4f} vs ~{exp_rv:.4f})",
                    source="rules",
                ))

        exp_myo = expected_area["myo"]
        if exp_myo > 0:
            myo_small_threshold = 0.80 if has_lowdice_prior else 0.74
            myo_large_threshold = 1.35 if has_lowdice_prior else 1.45
            if 0.30 * exp_myo <= area_frac_myo < myo_small_threshold * exp_myo:
                # CRITICAL SAFETY: If thickness variation is high (CV > 0.5), 
                # "thin" is likely due to fragmentation/artifacts, not true thinness.
                # Dilation will likely worsen artifacts.
                if feats.get("thickness_cv", 0) > 0.5:
                    diagnoses.append(SpatialDiagnosis(
                        issue="myo_thickness_var", affected_class="myo",
                        direction="global", severity="high",
                        operation="topology_cleanup", confidence=0.75,
                        description=f"High thickness var (cv={feats['thickness_cv']:.3f}) - risky to dilate",
                        source="rules",
                    ))
                else:
                    diagnoses.append(SpatialDiagnosis(
                        issue="myo_too_thin", affected_class="myo",
                        direction=self._infer_thin_myo_direction(mask),
                        severity="high" if area_frac_myo < 0.58 * exp_myo else "medium",
                        operation="2_dilate", confidence=0.70,
                        description=f"Myo area appears under-segmented ({area_frac_myo:.4f} vs ~{exp_myo:.4f})",
                        source="rules",
                    ))
            elif area_frac_myo > myo_large_threshold * exp_myo:
                diagnoses.append(SpatialDiagnosis(
                    issue="myo_too_thick", affected_class="myo",
                    direction=self._infer_structure_bias_direction(mask, class_label=2, prefer="surplus"),
                    severity="high" if area_frac_myo > 1.70 * exp_myo else "medium",
                    operation="2_erode", confidence=0.68,
                    description=f"Myo area appears over-segmented ({area_frac_myo:.4f} vs ~{exp_myo:.4f})",
                    source="rules",
                ))

        # Thickness variation
        if feats.get("thickness_cv", 0) > 0.5:
            diagnoses.append(SpatialDiagnosis(
                issue="jagged_boundary", affected_class="myo",
                direction="global", severity="low",
                operation="smooth_morph", confidence=0.6,
                description=f"High myo thickness variation (cv={feats['thickness_cv']:.3f})",
                source="rules",
            ))

        # Inter-slice misalignment (only if no clear local boundary/structure defect)
        local_primary_issues = {
            "broken_ring",
            "touching",
            "fragmented_lv",
            "fragmented_rv",
            "lv_too_small",
            "lv_too_large",
            "rv_too_small",
            "rv_too_large",
            "myo_too_thin",
            "myo_too_thick",
            "valve_leak",
            "massive_missing",
            "severe_misplacement",
        }
        has_local_primary = any(
            (d.issue in local_primary_issues) and (d.severity in ("critical", "high", "medium"))
            for d in diagnoses
        )
        if neighbor_masks and len(neighbor_masks) >= 2 and not has_local_primary:
            misalign_issues = detect_misalignment(mask, neighbor_masks, sigma_threshold=3.8)
            for mi in misalign_issues:
                issue_type = mi["type"]  # 'centroid_shift' or 'area_inconsistency'
                cls_name = mi["class"].lower()  # 'rv', 'myo', 'lv'
                if issue_type == "centroid_shift":
                    score = max(float(mi.get("deviation_x", 0)), float(mi.get("deviation_y", 0)))
                else:
                    score = float(mi.get("deviation", 0))
                if score < 4.5:
                    continue
                ops = ISSUE_OPS_MAP.get(issue_type, ["neighbor_shape_prior"])
                desc_parts = [f"{mi['class']} {issue_type}"]
                if "deviation_x" in mi:
                    desc_parts.append(f"dx={mi['deviation_x']}σ, dy={mi['deviation_y']}σ")
                if "deviation" in mi:
                    desc_parts.append(f"area dev={mi['deviation']}σ")
                diagnoses.append(SpatialDiagnosis(
                    issue=issue_type,
                    affected_class=cls_name,
                    direction="global",
                    severity="medium" if score >= 6.0 else "low",
                    operation=ops[0],
                    confidence=0.60 if score >= 6.0 else 0.52,
                    description="; ".join(desc_parts),
                    source="rules",
                ))

        # Severe structural misplacement detection
        expected = expected_area
        
        preferred_op = "individual_atlas_transfer" if neighbor_masks else "atlas_growprune_0"

        for cls_name, cls_label in _CLASS_LABEL.items():
            area_px = int(np.count_nonzero(mask == cls_label))
            area_frac = area_px / max(1, total_px)
            min_expected = expected.get(cls_name, 0.0)

            if min_expected <= 0:
                continue  # This structure is not expected in this view

            # Determine best op for this specific class
            specific_neighbor_op = f"neighbor_{cls_name}_repair"
            best_op = specific_neighbor_op if neighbor_masks else "atlas_growprune_0"

            # Massive missing: structure almost absent where it should exist.
            # Keep this strict to avoid premature atlas usage.
            if area_px < 80 and min_expected > 0.005:
                diagnoses.append(SpatialDiagnosis(
                    issue="massive_missing",
                    affected_class=cls_name,
                    direction="global",
                    severity="critical",
                    operation=best_op,
                    confidence=0.95,
                    description=(
                        f"{cls_name.upper()} nearly absent: {area_px}px "
                        f"(expected ≥{int(min_expected * total_px)}px). "
                        f"Atlas-guided reconstruction required."
                    ),
                    source="rules",
                ))
            # Severe global misplacement: very undersized class.
            elif area_frac < 0.35 * min_expected:
                diagnoses.append(SpatialDiagnosis(
                    issue="severe_misplacement",
                    affected_class=cls_name,
                    direction="global",
                    severity="critical",
                    operation=best_op,
                    confidence=0.90,
                    description=(
                        f"{cls_name.upper()} severely undersized: "
                        f"{area_frac:.4f} vs expected ≥{min_expected:.4f}. "
                        f"Atlas-guided reconstruction required."
                    ),
                    source="rules",
                ))

        # Sort by severity (critical > high > medium > low)
        severity_order = {"critical": -1, "high": 0, "medium": 1, "low": 2}
        diagnoses.sort(key=lambda d: severity_order.get(d.severity, 3))

        return diagnoses

    def _diagnose_vlm(self, ctx: CaseContext) -> List[SpatialDiagnosis]:
        """VLM-based spatial diagnosis using overlay image."""
        try:
            overlay_path = ctx.debug_img_path("_diag_overlay.png")
            if not create_overlay(ctx.img_path, ctx.mask, overlay_path):
                return []

            user_text = self._read_prompt_file("diagnosis_prompt.txt")
            if not user_text:
                user_text = (
                    f"Analyze this cardiac segmentation overlay for case {ctx.stem} "
                    f"(view: {ctx.view_type}). Identify defects."
                )

            result = self.think_vision(
                user_prompt=user_text,
                image_path=overlay_path,
            )

            return self._parse_llm_diagnoses(result, source="vlm", img_shape=ctx.mask.shape)

        except Exception as e:
            self.log(f"VLM diagnosis failed: {e}")
            return []

    def _parse_llm_diagnoses(
        self, result: Dict[str, Any], source: str = "llm", img_shape: Tuple[int, int] = None
    ) -> List[SpatialDiagnosis]:
        """Parse LLM/VLM JSON response into SpatialDiagnosis objects."""
        diagnoses = []
        items = result.get("diagnoses", [])
        all_known_ops = set()
        for ops in ISSUE_OPS_MAP.values():
            all_known_ops.update(ops)

        for item in items:
            issue = str(item.get("issue", "unknown"))
            affected_class = str(item.get("affected_class", "myo")).strip().lower()
            class_alias = {
                "left ventricle": "lv",
                "right ventricle": "rv",
                "myocardium": "myo",
                "left_ventricle": "lv",
                "right_ventricle": "rv",
            }
            affected_class = class_alias.get(affected_class, affected_class)
            if affected_class not in _CLASS_LABEL:
                affected_class = "myo"

            description = str(item.get("description", ""))
            direction = normalize_direction_hint(
                str(item.get("direction", "global")),
                description=description,
            )
            severity = str(item.get("severity", "medium")).lower()
            if severity not in ("critical", "high", "medium", "low"):
                severity = "medium"
            operation = str(item.get("operation", ""))
            confidence = float(item.get("confidence", 0.7))

            bbox_2d = item.get("bbox_2d", None)
            bbox = None
            if bbox_2d and isinstance(bbox_2d, list) and len(bbox_2d) == 4 and img_shape:
                H, W = img_shape[0], img_shape[1]
                # Default assume 0-1000 scale normalization
                try:
                    ymin = int(float(bbox_2d[0]) / 1000.0 * H)
                    xmin = int(float(bbox_2d[1]) / 1000.0 * W)
                    ymax = int(float(bbox_2d[2]) / 1000.0 * H)
                    xmax = int(float(bbox_2d[3]) / 1000.0 * W)
                    # Add margin? No, trust VLM or executor adds margin. 
                    # Executor doesn't add margin in `_apply_local_op` unless we changed it.
                    # Best to add margin here or there. 
                    # Let's add 10px margin here to be safe
                    margin = 10
                    ymin = max(0, ymin - margin)
                    xmin = max(0, xmin - margin)
                    ymax = min(H, ymax + margin)
                    xmax = min(W, xmax + margin)
                    bbox = [ymin, xmin, ymax, xmax]
                except Exception:
                    bbox = None

            # Normalize issue names from potential Triage/LLM hallucinations
            issue_map = {
                "DiscMyo": "broken_ring",
                "DiscMyo-like": "broken_ring",
                "DiscLV": "fragmented_lv",
                "LowDice-like": "severe_misplacement",
                "myo_gap": "broken_ring",
                "myo_gap_septal_wall": "broken_ring",
                "structural_break": "broken_ring",
                "RV-LV direct contact": "rv_lv_touching",
                "rv_lv_contact": "rv_lv_touching",
                "lv_split": "fragmented_lv",
                "rv_split": "fragmented_rv",
                "lv_multi_component": "fragmented_lv",
                "rv_multi_component": "fragmented_rv",
            }
            if issue in issue_map:
                issue = issue_map[issue]
            issue_map_lower = {
                "lv split": "fragmented_lv",
                "rv split": "fragmented_rv",
                "lv multi component": "fragmented_lv",
                "rv multi component": "fragmented_rv",
                "multiple lv components": "fragmented_lv",
                "multiple rv components": "fragmented_rv",
            }
            issue_lower = issue.lower().strip()
            if issue_lower in issue_map_lower:
                issue = issue_map_lower[issue_lower]
            # Also handle generic "gap" or "discontinuity" in issue string
            issue_lower = issue.lower()
            if "myo" in issue_lower:
                if "gap" in issue_lower or "discontinuity" in issue_lower or "break" in issue_lower:
                    issue = "broken_ring"

            # Semantic Override for Severe Fragmentation -> Missing
            # This forces the Planner to use Atlas as per the new "Missing = Atlas" rule
            if issue == "fragmented_lv" and severity in ("high", "critical"):
                issue = "missing_lv_fragments"
                severity = "critical"
                if ("atlas" not in operation) and operation != "lv_split_to_rv":
                    operation = "atlas_rv_repair"
            elif issue == "fragmented_rv" and severity in ("high", "critical"):
                issue = "missing_rv_fragments"
                severity = "critical"
                if ("atlas" not in operation) and operation != "rv_split_to_lv":
                    operation = "atlas_lv_repair"

            # Validate operation is known
            if operation not in all_known_ops:
                # Fall back to first known op for this issue
                fallback = ISSUE_OPS_MAP.get(issue, [])
                operation = fallback[0] if fallback else "topology_cleanup"

            # Safety policy:
            # keep standard directional erode/dilate for medium/high; only
            # force-downgrade large/neighbor/atlas ops when severity is low/moderate.
            old_op = operation
            if operation.endswith("_large") and severity not in ("critical", "high"):
                smaller = operation[:-6]
                if smaller in all_known_ops:
                    operation = smaller
                else:
                    operation = self._conservative_issue_operation(issue, affected_class)

            if operation.startswith("atlas_") or operation == "individual_atlas_transfer":
                if severity != "critical":
                    operation = self._conservative_issue_operation(issue, affected_class)
            elif operation.startswith("neighbor_"):
                if severity not in ("critical", "high"):
                    operation = self._conservative_issue_operation(issue, affected_class)
            elif operation.startswith("expand_") and severity == "low":
                operation = self._conservative_issue_operation(issue, affected_class)

            if operation not in all_known_ops:
                operation = self._conservative_issue_operation(issue, affected_class)

            if old_op != operation:
                description += f" (Safety: {old_op} -> {operation})"

            if operation == "topology_cleanup" and severity not in ("critical", "high"):
                severity = "low"
            confidence = max(0.0, min(1.0, confidence))

            diagnoses.append(SpatialDiagnosis(
                issue=issue,
                affected_class=affected_class,
                direction=direction,
                severity=severity,
                operation=operation,
                description=description,
                confidence=confidence,
                source=source,
                bbox=bbox,
            ))

        return diagnoses

    @staticmethod
    def _conservative_issue_operation(issue: str, affected_class: str) -> str:
        """Pick a conservative, local fallback operation for the given issue."""
        blocked_prefixes = ("neighbor_", "atlas_")
        for op in ISSUE_OPS_MAP.get(issue, []):
            if op == "individual_atlas_transfer":
                continue
            if op.startswith(blocked_prefixes):
                continue
            if op.endswith("_large"):
                continue
            if op.startswith("expand_"):
                continue
            return op

        if affected_class == "myo":
            return "2_dilate"
        if affected_class in ("lv", "rv"):
            return "1_dilate"
        return "topology_cleanup"

    @staticmethod
    def _infer_structure_bias_direction(
        mask: np.ndarray,
        class_label: int,
        prefer: str = "surplus",
    ) -> str:
        """Infer directional surplus/deficit for a class via quadrant occupancy."""
        cls = (mask == class_label)
        lv = (mask == 3)
        if not cls.any() or not lv.any():
            return "global"

        H, W = mask.shape
        lv_y, lv_x = np.where(lv)
        cy, cx = float(lv_y.mean()), float(lv_x.mean())

        rv = (mask == 1)
        if rv.any():
            rv_cx = float(np.where(rv)[1].mean())
        else:
            rv_cx = cx - W * 0.3

        regions = {
            "apical": (slice(int(cy), H), slice(0, W)),
            "basal": (slice(0, int(cy)), slice(0, W)),
        }
        if rv_cx < cx:
            regions["septal"] = (slice(0, H), slice(0, int(cx)))
            regions["lateral"] = (slice(0, H), slice(int(cx), W))
        else:
            regions["septal"] = (slice(0, H), slice(int(cx), W))
            regions["lateral"] = (slice(0, H), slice(0, int(cx)))

        scores = {
            direction: int(cls[ys, xs].sum())
            for direction, (ys, xs) in regions.items()
        }
        if not scores or max(scores.values()) == 0:
            return "global"

        if prefer == "deficit":
            return min(scores, key=scores.get)
        return max(scores, key=scores.get)

    @staticmethod
    def _rv_lv_contact_len(mask: np.ndarray) -> int:
        """Approximate RV-LV direct contact length in pixels."""
        rv = (mask == 1).astype(np.uint8)
        lv = (mask == 3).astype(np.uint8)
        if rv.sum() == 0 or lv.sum() == 0:
            return 0
        ker = np.ones((3, 3), np.uint8)
        rv_edge = rv - cv2.erode(rv, ker, iterations=1)
        lv_dil = cv2.dilate(lv, ker, iterations=1)
        return int(np.count_nonzero((rv_edge > 0) & (lv_dil > 0)))

    @staticmethod
    def _has_basal_leak(
        mask: np.ndarray,
        top_frac: float = 0.08,
        min_protrusion_px: int = 6,
        min_leak_px: int = 35,
    ) -> bool:
        """Detect obvious leakage into top (basal) region."""
        fg = (mask > 0)
        if not fg.any():
            return False
        h, _ = mask.shape
        top_limit = int(h * top_frac)
        leak_px = int(np.count_nonzero(fg[:top_limit, :]))
        ys = np.where(fg)[0]
        y0 = int(ys.min()) if ys.size > 0 else h
        protrusion = max(0, top_limit - y0)
        return leak_px >= min_leak_px and protrusion >= min_protrusion_px

    @staticmethod
    def _has_meaningful_myo_break(
        mask: np.ndarray,
        min_second_ratio: float = 0.05,
        min_gap_px: int = 3,
    ) -> tuple[bool, int]:
        """
        Distinguish true ring break from tiny myo islands.
        Returns (is_meaningful_break, approx_gap_px).
        """
        myo = (mask == 2).astype(np.uint8)
        labels, n = ndi.label(myo)
        if n < 2:
            return False, 0

        areas = np.array([(labels == i).sum() for i in range(1, n + 1)], dtype=np.float32)
        if areas.size < 2:
            return False, 0
        order = np.argsort(areas)[::-1]
        total = float(areas.sum())
        if total <= 0:
            return False, 0
        second_ratio = float(areas[order[1]]) / total
        if second_ratio < min_second_ratio:
            return False, 0

        c1 = (labels == (order[0] + 1))
        c2 = (labels == (order[1] + 1))
        dt = ndi.distance_transform_edt(~c1)
        min_dist = float(dt[c2].min()) if c2.any() else 0.0
        gap_px = int(round(min_dist))
        return gap_px >= min_gap_px, gap_px

    @staticmethod
    def _infer_thin_myo_direction(mask: np.ndarray) -> str:
        """Infer where the myocardium is thinnest by comparing quadrant thicknesses."""
        myo = (mask == 2)
        lv = (mask == 3)
        if not myo.any() or not lv.any():
            return "global"

        H, W = mask.shape
        lv_y, lv_x = np.where(lv)
        cy, cx = float(lv_y.mean()), float(lv_x.mean())

        # Find RV centroid for septal/lateral axis
        rv = (mask == 1)
        if rv.any():
            rv_cx = float(np.where(rv)[1].mean())
        else:
            rv_cx = cx - W * 0.3

        # Define quadrants relative to LV centroid
        regions = {
            "apical": (slice(int(cy), H), slice(0, W)),
            "basal": (slice(0, int(cy)), slice(0, W)),
        }
        if rv_cx < cx:
            regions["septal"] = (slice(0, H), slice(0, int(cx)))
            regions["lateral"] = (slice(0, H), slice(int(cx), W))
        else:
            regions["septal"] = (slice(0, H), slice(int(cx), W))
            regions["lateral"] = (slice(0, H), slice(0, int(cx)))

        # Count myo pixels in each quadrant
        min_count = float("inf")
        min_dir = "global"
        for direction, (ys, xs) in regions.items():
            count = int(myo[ys, xs].sum())
            if count < min_count:
                min_count = count
                min_dir = direction

        return min_dir

    @staticmethod
    def _infer_myo_break_direction(mask: np.ndarray) -> str:
        """Infer where the myocardium break is relative to LV centroid."""
        myo = (mask == 2).astype(np.uint8)
        labels, n = ndi.label(myo)
        if n < 2:
            return "global"

        # Find the gap between the two largest components
        areas = [(labels == i).sum() for i in range(1, n + 1)]
        sorted_idx = np.argsort(areas)[::-1]
        c1 = (labels == sorted_idx[0] + 1)
        c2 = (labels == sorted_idx[1] + 1)

        # Gap center
        b1_y, b1_x = np.where(c1)
        b2_y, b2_x = np.where(c2)
        gap_y = (b1_y.mean() + b2_y.mean()) / 2
        gap_x = (b1_x.mean() + b2_x.mean()) / 2

        # LV centroid
        lv = (mask == 3)
        if lv.any():
            lv_y, lv_x = np.where(lv)
            cy, cx = lv_y.mean(), lv_x.mean()
        else:
            cy, cx = mask.shape[0] / 2, mask.shape[1] / 2

        # Determine direction
        dy = gap_y - cy
        dx = gap_x - cx

        if abs(dy) > abs(dx):
            return "apical" if dy > 0 else "basal"
        else:
            # Check if gap is on RV side (septal) or opposite (lateral)
            rv = (mask == 1)
            if rv.any():
                rv_x = np.where(rv)[1].mean()
                if (gap_x - cx) * (rv_x - cx) > 0:
                    return "septal"
                else:
                    return "lateral"
            return "lateral"

    @staticmethod
    def ops_from_diagnoses(diagnoses: List[Dict[str, Any]]) -> List[str]:
        """Extract unique op names from diagnosis list, preserving order."""
        seen = set()
        ops = []
        for d in diagnoses:
            op = d.get("operation", "")
            if op and op not in seen:
                ops.append(op)
                seen.add(op)
        return ops


def _format_dict(d: Dict[str, Any], indent: int = 2) -> str:
    """Format a dict as readable key: value lines."""
    import json
    return json.dumps(d, indent=indent, default=str)
