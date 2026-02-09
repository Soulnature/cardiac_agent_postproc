from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from scipy import ndimage as ndi

from ..quality_model import extract_quality_features
from ..vlm_guardrail import VisionGuardrail, create_overlay
from ..api_client import OpenAICompatClient
from ..settings import LLMSettings


# Canonical issue → ops mapping
ISSUE_OPS_MAP: Dict[str, List[str]] = {
    "disconnected_myo": ["myo_bridge", "close2_open3", "atlas_growprune_0"],
    "fragmented_lv": ["lv_bridge", "safe_topology_fix", "topology_cleanup"],
    "fragmented_rv": ["safe_topology_fix", "topology_cleanup"],
    "holes": ["fill_holes", "fill_holes_m2", "safe_topology_fix"],
    "rv_lv_touching": ["rv_lv_barrier", "rv_erode", "close2_open3"],
    "jagged_boundary": ["smooth_contour", "smooth_morph", "smooth_gaussian"],
    "islands_noise": ["topology_cleanup", "safe_topology_fix"],
    "valve_leak": ["valve_suppress", "safe_topology_fix"],
    "myo_thickness_var": ["smooth_morph", "close2_open3"],
    "missing_myo": ["atlas_growprune_0", "close2_open3"],
    "missing_lv": ["atlas_growprune_0", "close2_open3"],
    "missing_rv": ["atlas_growprune_0"],
    "missing_structure": ["atlas_growprune_0", "close2_open3"],
    "edge_misalignment": ["edge_snap_proxy", "smooth_contour", "smooth_morph"],
    "low_quality_score": ["topology_cleanup", "safe_topology_fix", "atlas_growprune_0"],
    "components": ["topology_cleanup", "safe_topology_fix"],
    "broken_ring": ["myo_bridge", "close2_open3", "atlas_growprune_0"],
    "too_thin": ["close2_open3", "atlas_growprune_0"],
    "noise_island": ["topology_cleanup", "safe_topology_fix", "remove_islands"],
    "touching": ["rv_lv_barrier", "rv_erode"],
}

# Class name → mask label
_CLASS_LABEL = {"rv": 1, "myo": 2, "lv": 3}


@dataclass
class Diagnosis:
    issue: str
    ops: List[str]
    confidence: float
    source: str = "rules"  # "rules" or "vlm"


@dataclass
class SpatialDiagnosis:
    issue: str              # e.g., "broken_ring", "too_thin", "noise_island"
    affected_class: str     # "rv" | "myo" | "lv"
    direction: str          # "septal" | "lateral" | "apical" | "basal" |
                            # "left" | "right" | "top" | "bottom" | "global"
    severity: str           # "high" | "medium" | "low"
    operation: str          # recommended op from ops.py
    description: str        # free-text from VLM
    confidence: float       # 0.0-1.0
    source: str             # "vlm" | "rules"


def compute_region_mask(
    mask: np.ndarray,
    affected_class: str,
    direction: str,
) -> np.ndarray:
    """
    Compute a boolean region mask (H, W) indicating where to apply an operation.

    Uses LV centroid as the anatomical reference point, with RV centroid
    for septal/lateral axis determination.

    Args:
        mask: segmentation mask with labels {0,1,2,3}
        affected_class: "rv" | "myo" | "lv"
        direction: anatomical direction or "global"

    Returns:
        Boolean array (H, W) - True where the operation should apply.
    """
    H, W = mask.shape

    if direction == "global":
        return np.ones((H, W), dtype=bool)

    # Compute centroids
    lv_mask = (mask == 3)
    rv_mask = (mask == 1)

    lv_ys, lv_xs = np.where(lv_mask)
    if len(lv_ys) == 0:
        # No LV found: fall back to image center
        lv_cy, lv_cx = H // 2, W // 2
    else:
        lv_cy, lv_cx = int(lv_ys.mean()), int(lv_xs.mean())

    rv_ys, rv_xs = np.where(rv_mask)
    if len(rv_ys) == 0:
        # No RV: assume RV is to the left of LV (common in 4ch view)
        rv_cy, rv_cx = lv_cy, max(0, lv_cx - W // 4)
    else:
        rv_cy, rv_cx = int(rv_ys.mean()), int(rv_xs.mean())

    # Get bounding box of the affected class for apical/basal
    class_label = _CLASS_LABEL.get(affected_class)
    class_mask = (mask == class_label) if class_label else (mask > 0)
    class_ys, class_xs = np.where(class_mask)

    if len(class_ys) == 0:
        # Class not present: use full mask extent
        all_ys, all_xs = np.where(mask > 0)
        if len(all_ys) == 0:
            return np.ones((H, W), dtype=bool)
        y_min, y_max = int(all_ys.min()), int(all_ys.max())
        x_min, x_max = int(all_xs.min()), int(all_xs.max())
    else:
        y_min, y_max = int(class_ys.min()), int(class_ys.max())
        x_min, x_max = int(class_xs.min()), int(class_xs.max())

    region = np.zeros((H, W), dtype=bool)
    yy, xx = np.mgrid[0:H, 0:W]

    if direction == "septal":
        # Between LV and RV: pixels on the RV-side of LV centroid
        # Vector from LV to RV
        dx = rv_cx - lv_cx
        dy = rv_cy - lv_cy
        norm = max(1.0, np.sqrt(dx**2 + dy**2))
        dx, dy = dx / norm, dy / norm
        # Dot product: positive = towards RV (septal side)
        dot = (xx - lv_cx) * dx + (yy - lv_cy) * dy
        region = dot > 0

    elif direction == "lateral":
        # Opposite side of LV from RV
        dx = rv_cx - lv_cx
        dy = rv_cy - lv_cy
        norm = max(1.0, np.sqrt(dx**2 + dy**2))
        dx, dy = dx / norm, dy / norm
        dot = (xx - lv_cx) * dx + (yy - lv_cy) * dy
        region = dot <= 0

    elif direction == "apical":
        # Bottom 30% of the structure bounding box
        h_range = y_max - y_min
        cutoff = y_max - int(0.3 * h_range)
        region = yy >= cutoff

    elif direction == "basal":
        # Top 30% of the structure bounding box
        h_range = y_max - y_min
        cutoff = y_min + int(0.3 * h_range)
        region = yy <= cutoff

    elif direction == "left":
        region = xx < lv_cx

    elif direction == "right":
        region = xx >= lv_cx

    elif direction == "top":
        region = yy < lv_cy

    elif direction == "bottom":
        region = yy >= lv_cy

    else:
        # Unknown direction: fall back to global
        region = np.ones((H, W), dtype=bool)

    return region


class DiagnosisAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        diag_cfg = cfg.get("diagnosis", {})
        self.enabled = bool(diag_cfg.get("enabled", True))
        self.quality_threshold = float(diag_cfg.get("quality_threshold", 0.8))
        self.max_suggestions = int(diag_cfg.get("max_suggestions", 6))
        self.vlm_enabled = bool(diag_cfg.get("vlm_enabled", False))
        self.spatial_enabled = bool(diag_cfg.get("spatial", False))

        # VLM client for diagnosis (reuse vision guardrail settings)
        self._vlm_client = None
        if self.vlm_enabled:
            vg_cfg = cfg.get("vision_guardrail", {})
            if vg_cfg.get("enabled", False):
                settings = LLMSettings()
                model = vg_cfg.get("model", settings.openai_model)
                api_key = vg_cfg.get("api_key", settings.openai_api_key)
                self._vlm_client = OpenAICompatClient(
                    base_url=settings.openai_base_url,
                    api_key=api_key,
                    model=model,
                    timeout=180.0,
                )

    # ------------------------------------------------------------------
    # Rules-based diagnosis (feature-driven, no RQS dependency)
    # ------------------------------------------------------------------

    def diagnose_rules(
        self,
        mask: np.ndarray,
        img: np.ndarray,
        view_type: str,
        triage_issues: List[str] | None = None,
    ) -> List[Diagnosis]:
        """Feature-based diagnosis using quality features directly."""
        feats = extract_quality_features(mask, img, view_type)
        suggestions: List[Diagnosis] = []

        # Use triage-detected issues as primary signal
        if triage_issues:
            for issue in triage_issues:
                ops = ISSUE_OPS_MAP.get(issue, ["topology_cleanup"])
                suggestions.append(Diagnosis(
                    issue=issue,
                    ops=ops,
                    confidence=0.7,
                    source="rules",
                ))

        # Additional feature-based checks
        if feats.get("cc_count_myo", 1.0) > 1:
            suggestions.append(Diagnosis(
                issue="disconnected_myo",
                ops=ISSUE_OPS_MAP["disconnected_myo"],
                confidence=min(1.0, (feats["cc_count_myo"] - 1) * 0.3),
                source="rules",
            ))

        if feats.get("touch_ratio", 0.0) > 0.01:
            suggestions.append(Diagnosis(
                issue="rv_lv_touching",
                ops=ISSUE_OPS_MAP["rv_lv_touching"],
                confidence=min(1.0, feats["touch_ratio"] / 0.05),
                source="rules",
            ))

        if feats.get("hole_frac_lv", 0.0) > 0.05 or feats.get("hole_frac_myo", 0.0) > 0.05:
            hole_val = max(feats.get("hole_frac_lv", 0.0), feats.get("hole_frac_myo", 0.0))
            suggestions.append(Diagnosis(
                issue="holes",
                ops=ISSUE_OPS_MAP["holes"],
                confidence=min(1.0, hole_val / 0.1),
                source="rules",
            ))

        if feats.get("cc_count_rv", 1.0) > 1 or feats.get("cc_count_lv", 1.0) > 1:
            suggestions.append(Diagnosis(
                issue="components",
                ops=ISSUE_OPS_MAP["components"],
                confidence=0.6,
                source="rules",
            ))

        if feats.get("boundary_over_area", 0.0) > 0.18:
            suggestions.append(Diagnosis(
                issue="jagged_boundary",
                ops=ISSUE_OPS_MAP["jagged_boundary"],
                confidence=min(1.0, (feats["boundary_over_area"] - 0.13) / 0.05),
                source="rules",
            ))

        if feats.get("rv_missing", 0.0) > 0.5 or feats.get("lv_missing", 0.0) > 0.5:
            suggestions.append(Diagnosis(
                issue="missing_structure",
                ops=ISSUE_OPS_MAP["missing_structure"],
                confidence=0.8,
                source="rules",
            ))

        if feats.get("valve_leak_ratio", 0.0) > 0.1:
            suggestions.append(Diagnosis(
                issue="valve_leak",
                ops=ISSUE_OPS_MAP["valve_leak"],
                confidence=min(1.0, feats["valve_leak_ratio"]),
                source="rules",
            ))

        if feats.get("thickness_cv", 0.0) > 0.5:
            suggestions.append(Diagnosis(
                issue="myo_thickness_var",
                ops=ISSUE_OPS_MAP["myo_thickness_var"],
                confidence=min(1.0, feats["thickness_cv"]),
                source="rules",
            ))

        return suggestions

    # ------------------------------------------------------------------
    # VLM-based spatial diagnosis
    # ------------------------------------------------------------------

    def diagnose_spatial_with_vlm(
        self,
        img_path: str,
        mask: np.ndarray,
        stem: str,
        view_type: str,
        triage_issues: List[str] | None = None,
    ) -> List[SpatialDiagnosis]:
        """Send overlay to VLM, get structured spatial diagnosis."""
        if self._vlm_client is None:
            return []

        # Create overlay image
        overlay_path = img_path.replace(".png", "_diag_overlay.png")
        if not create_overlay(img_path, mask, overlay_path):
            return []

        # Load spatial diagnosis prompt
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "prompts",
            "diagnosis_prompt.txt",
        )
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        else:
            system_prompt = self._default_spatial_diagnosis_prompt()

        issues_hint = ""
        if triage_issues:
            issues_hint = f" Pre-detected issues: {', '.join(triage_issues)}."

        user_text = (
            f"Diagnose segmentation defects for case {stem} (view: {view_type})."
            f"{issues_hint}"
            f" Provide spatial diagnosis with affected class, direction, and recommended operation."
        )

        try:
            resp = self._vlm_client.chat_vision_json(system_prompt, user_text, overlay_path)
        except Exception as e:
            print(f"[DiagnosisAgent] VLM spatial diagnosis failed for {stem}: {e}")
            return []
        return self._parse_spatial_vlm_diagnoses(resp)

    def _parse_spatial_vlm_diagnoses(self, resp: Dict[str, Any]) -> List[SpatialDiagnosis]:
        """Parse VLM JSON response into SpatialDiagnosis objects."""
        diagnoses: List[SpatialDiagnosis] = []
        items = resp.get("diagnoses", [])
        if not isinstance(items, list):
            return diagnoses

        valid_directions = {
            "septal", "lateral", "apical", "basal",
            "left", "right", "top", "bottom", "global",
        }
        valid_classes = {"rv", "myo", "lv"}
        valid_severities = {"high", "medium", "low"}

        for item in items:
            if not isinstance(item, dict):
                continue

            issue = item.get("issue", "unknown")
            affected_class = item.get("affected_class", "myo").lower()
            if affected_class not in valid_classes:
                affected_class = "myo"

            direction = item.get("direction", "global").lower()
            if direction not in valid_directions:
                direction = "global"

            severity = item.get("severity", "medium").lower()
            if severity not in valid_severities:
                severity = "medium"

            operation = item.get("operation", "")
            if operation not in _KNOWN_OPS:
                # Fall back to first op from issue mapping
                ops_list = ISSUE_OPS_MAP.get(issue, ["topology_cleanup"])
                operation = ops_list[0]

            confidence = float(item.get("confidence", 0.5))
            description = item.get("description", "")

            diagnoses.append(SpatialDiagnosis(
                issue=issue,
                affected_class=affected_class,
                direction=direction,
                severity=severity,
                operation=operation,
                description=description,
                confidence=confidence,
                source="vlm",
            ))

        return diagnoses

    # ------------------------------------------------------------------
    # Legacy VLM-based diagnosis (non-spatial)
    # ------------------------------------------------------------------

    def diagnose_with_vlm(
        self,
        img_path: str,
        mask: np.ndarray,
        stem: str,
        view_type: str,
        triage_issues: List[str] | None = None,
    ) -> List[Diagnosis]:
        """Send overlay to VLM, get structured diagnosis."""
        if self._vlm_client is None:
            return []

        # Create overlay image
        overlay_path = img_path.replace(".png", "_diag_overlay.png")
        if not create_overlay(img_path, mask, overlay_path):
            return []

        # Load diagnosis prompt
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "prompts",
            "diagnosis_prompt.txt",
        )
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        else:
            system_prompt = self._default_diagnosis_prompt()

        issues_hint = ""
        if triage_issues:
            issues_hint = f" Pre-detected issues: {', '.join(triage_issues)}."

        user_text = (
            f"Diagnose segmentation defects for case {stem} (view: {view_type})."
            f"{issues_hint}"
        )

        try:
            resp = self._vlm_client.chat_vision_json(system_prompt, user_text, overlay_path)
        except Exception as e:
            print(f"[DiagnosisAgent] VLM failed for {stem}: {e}")
            return []
        return self._parse_vlm_diagnoses(resp)

    def _parse_vlm_diagnoses(self, resp: Dict[str, Any]) -> List[Diagnosis]:
        """Parse VLM JSON response into Diagnosis objects."""
        diagnoses: List[Diagnosis] = []
        items = resp.get("diagnoses", [])
        if not isinstance(items, list):
            return diagnoses
        for item in items:
            if not isinstance(item, dict):
                continue
            issue = item.get("issue", "unknown")
            ops = item.get("operations", [])
            if isinstance(ops, str):
                ops = [ops]
            # Validate ops against known operations
            valid_ops = [op for op in ops if op in _KNOWN_OPS]
            if not valid_ops:
                valid_ops = ISSUE_OPS_MAP.get(issue, ["topology_cleanup"])
            confidence = float(item.get("confidence", 0.5))
            diagnoses.append(Diagnosis(
                issue=issue,
                ops=valid_ops,
                confidence=confidence,
                source="vlm",
            ))
        return diagnoses

    @staticmethod
    def _default_diagnosis_prompt() -> str:
        return (
            "You are a cardiac MRI segmentation quality expert.\n"
            "Color legend: Red=RV, Green=Myocardium, Blue=LV.\n"
            "Identify defects and suggest repair operations.\n"
            "Output JSON: {\"diagnoses\": [{\"issue\": str, \"operations\": [str], "
            "\"confidence\": float, \"description\": str}]}\n"
            "Known operations: topology_cleanup, safe_topology_fix, fill_holes, "
            "fill_holes_m2, close2_open3, open2_close3, rv_lv_barrier, rv_erode, "
            "valve_suppress, edge_snap_proxy, smooth_contour, smooth_gaussian, "
            "smooth_morph, atlas_growprune_0, atlas_growprune_1, myo_bridge, lv_bridge"
        )

    @staticmethod
    def _default_spatial_diagnosis_prompt() -> str:
        return (
            "You are a cardiac MRI segmentation quality expert.\n"
            "Color legend: Red=RV, Green=Myocardium, Blue=LV.\n\n"
            "Anatomical direction reference (4-chamber view):\n"
            "- septal: between LV and RV (interventricular septum)\n"
            "- lateral: outer wall of LV, opposite side from RV\n"
            "- apical: bottom tip of the heart\n"
            "- basal: top near the valve plane\n\n"
            "Identify defects with spatial location and suggest ONE repair operation per defect.\n\n"
            "Output JSON:\n"
            "{\"diagnoses\": [\n"
            "  {\"issue\": str, \"affected_class\": \"rv\"|\"myo\"|\"lv\",\n"
            "   \"direction\": \"septal\"|\"lateral\"|\"apical\"|\"basal\"|\"left\"|\"right\"|\"global\",\n"
            "   \"severity\": \"high\"|\"medium\"|\"low\",\n"
            "   \"operation\": str, \"confidence\": float, \"description\": str}\n"
            "]}\n\n"
            "Known operations: topology_cleanup, safe_topology_fix, fill_holes, "
            "fill_holes_m2, close2_open3, open2_close3, rv_lv_barrier, rv_erode, "
            "valve_suppress, edge_snap_proxy, smooth_contour, smooth_gaussian, "
            "smooth_morph, atlas_growprune_0, atlas_growprune_1, myo_bridge, lv_bridge, "
            "remove_islands"
        )

    # ------------------------------------------------------------------
    # Rules → SpatialDiagnosis conversion
    # ------------------------------------------------------------------

    def rules_to_spatial(
        self,
        mask: np.ndarray,
        img: np.ndarray,
        view_type: str,
        triage_issues: List[str] | None = None,
    ) -> List[SpatialDiagnosis]:
        """Convert rule-based diagnoses to SpatialDiagnosis with direction inference."""
        rule_diags = self.diagnose_rules(mask, img, view_type, triage_issues)
        spatial: List[SpatialDiagnosis] = []

        for diag in rule_diags:
            # Infer direction from issue type
            direction = "global"
            affected_class = "myo"

            if diag.issue in ("disconnected_myo", "broken_ring"):
                affected_class = "myo"
                direction = self._infer_myo_break_direction(mask)
            elif diag.issue in ("myo_thickness_var", "too_thin"):
                affected_class = "myo"
                # Thickness variance is cosmetic — localize or skip
                direction = self._infer_myo_break_direction(mask)
                if direction == "global":
                    # Skip: smooth_morph applied globally is too destructive
                    continue
            elif diag.issue in ("rv_lv_touching", "touching"):
                affected_class = "myo"
                direction = "septal"
            elif diag.issue in ("fragmented_lv", "missing_lv", "holes"):
                affected_class = "lv"
            elif diag.issue in ("fragmented_rv", "missing_rv"):
                affected_class = "rv"
            elif diag.issue == "valve_leak":
                affected_class = "myo"
                direction = "basal"
            elif diag.issue in ("islands_noise", "noise_island", "components"):
                affected_class = "rv"  # islands often in background near RV
            elif diag.issue == "jagged_boundary":
                affected_class = "myo"
                # Jagged boundary smoothing should NOT be global
                direction = self._infer_myo_break_direction(mask)
                if direction == "global":
                    continue
            elif diag.issue == "edge_misalignment":
                affected_class = "myo"
                # Edge alignment is cosmetic; skip if global
                direction = self._infer_myo_break_direction(mask)
                if direction == "global":
                    continue

            severity = "high" if diag.confidence >= 0.7 else "medium" if diag.confidence >= 0.4 else "low"

            spatial.append(SpatialDiagnosis(
                issue=diag.issue,
                affected_class=affected_class,
                direction=direction,
                severity=severity,
                operation=diag.ops[0] if diag.ops else "topology_cleanup",
                description=f"Rule-detected: {diag.issue}",
                confidence=diag.confidence,
                source="rules",
            ))

        return spatial

    @staticmethod
    def _infer_myo_break_direction(mask: np.ndarray) -> str:
        """Infer where the myocardium break is by checking gap location relative to LV."""
        myo = (mask == 2)
        lv = (mask == 3)

        lv_ys, lv_xs = np.where(lv)
        if len(lv_ys) == 0:
            return "global"

        lv_cy, lv_cx = int(lv_ys.mean()), int(lv_xs.mean())

        # Check if myo has gaps by looking at connected components
        labeled, n_cc = ndi.label(myo)
        if n_cc <= 1:
            return "global"

        # Find the gap: dilate myo and find where it differs from filled version
        filled = ndi.binary_fill_holes(ndi.binary_dilation(myo, iterations=2))
        gap = filled & ~ndi.binary_dilation(myo, iterations=1)

        gap_ys, gap_xs = np.where(gap)
        if len(gap_ys) == 0:
            return "global"

        gap_cy, gap_cx = int(gap_ys.mean()), int(gap_xs.mean())

        # Determine direction relative to LV centroid
        rv = (mask == 1)
        rv_ys, rv_xs = np.where(rv)
        if len(rv_ys) > 0:
            rv_cx = int(rv_xs.mean())
            rv_cy = int(rv_ys.mean())
            # Use LV→RV vector to define septal axis
            dx = rv_cx - lv_cx
            dy = rv_cy - lv_cy
            norm = max(1.0, np.sqrt(dx**2 + dy**2))
            # Project gap-LV vector onto septal axis
            gap_dx = gap_cx - lv_cx
            gap_dy = gap_cy - lv_cy
            dot = (gap_dx * dx + gap_dy * dy) / norm
            if dot > 0:
                return "septal"
            else:
                return "lateral"

        return "global"

    # ------------------------------------------------------------------
    # Combined spatial diagnosis entry point
    # ------------------------------------------------------------------

    def diagnose_spatial(
        self,
        mask: np.ndarray,
        img: np.ndarray,
        view_type: str,
        img_path: str = "",
        stem: str = "",
        triage_issues: List[str] | None = None,
    ) -> List[SpatialDiagnosis]:
        """
        Spatial diagnosis: VLM-based (preferred) with rules fallback.
        Returns list of SpatialDiagnosis sorted by confidence descending.
        """
        if not self.enabled:
            return []

        spatial_diags: List[SpatialDiagnosis] = []

        # Try VLM spatial diagnosis first (preferred)
        if self.vlm_enabled and self._vlm_client is not None and img_path:
            vlm_diags = self.diagnose_spatial_with_vlm(
                img_path, mask, stem, view_type, triage_issues
            )
            spatial_diags.extend(vlm_diags)

        # If VLM returned nothing or is disabled, use rules → spatial conversion
        if not spatial_diags:
            spatial_diags = self.rules_to_spatial(mask, img, view_type, triage_issues)

        # Sort by severity (high > medium > low) then confidence
        severity_order = {"high": 0, "medium": 1, "low": 2}
        spatial_diags.sort(
            key=lambda d: (severity_order.get(d.severity, 1), -d.confidence)
        )

        return spatial_diags[: self.max_suggestions]

    # ------------------------------------------------------------------
    # Combined diagnosis entry point (legacy, non-spatial)
    # ------------------------------------------------------------------

    def diagnose(
        self,
        mask: np.ndarray,
        img: np.ndarray,
        view_type: str,
        img_path: str = "",
        stem: str = "",
        triage_issues: List[str] | None = None,
        use_vlm: bool = False,
    ) -> List[Diagnosis]:
        """
        Combined diagnosis: rules + optional VLM.
        Deduplicates by issue, sorts by confidence descending.
        """
        if not self.enabled:
            return []

        suggestions: List[Diagnosis] = []

        # Rules-based
        suggestions.extend(
            self.diagnose_rules(mask, img, view_type, triage_issues)
        )

        # VLM-based (if enabled and requested)
        if use_vlm and self.vlm_enabled and img_path:
            vlm_diags = self.diagnose_with_vlm(
                img_path, mask, stem, view_type, triage_issues
            )
            suggestions.extend(vlm_diags)

        # Deduplicate by issue, keep highest confidence
        dedup: Dict[str, Diagnosis] = {}
        for s in suggestions:
            if s.issue not in dedup or s.confidence > dedup[s.issue].confidence:
                dedup[s.issue] = s
        result = sorted(dedup.values(), key=lambda d: d.confidence, reverse=True)
        return result[: self.max_suggestions]

    @staticmethod
    def ops_from_diagnoses(diagnoses: List[Diagnosis]) -> List[str]:
        ops: List[str] = []
        for diag in diagnoses:
            for op in diag.ops:
                if op not in ops:
                    ops.append(op)
        return ops

    @staticmethod
    def reorder_candidates(candidates: List[tuple[str, Any]], priority_ops: List[str]):
        if not priority_ops:
            return candidates

        def priority(name: str) -> int:
            for idx, op in enumerate(priority_ops):
                if name == op or name.startswith(op):
                    return idx
            return len(priority_ops)

        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda item: (priority(item[1][0]), item[0]))
        return [item[1] for item in indexed]


# Set of known ops for VLM response validation
_KNOWN_OPS = {
    "topology_cleanup",
    "safe_topology_fix",
    "fill_holes",
    "fill_holes_m2",
    "close2_open3",
    "open2_close3",
    "rv_lv_barrier",
    "rv_erode",
    "valve_suppress",
    "edge_snap_proxy",
    "smooth_contour",
    "smooth_gaussian",
    "smooth_morph",
    "atlas_growprune_0",
    "atlas_growprune_1",
    "myo_bridge",
    "lv_bridge",
    "2_dilate",
    "2_erode",
    "3_dilate",
    "3_erode",
    "remove_islands",
}
