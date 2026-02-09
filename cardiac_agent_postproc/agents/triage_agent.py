from __future__ import annotations
import os
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

from ..quality_model import extract_quality_features, QualityScorer
from ..vlm_guardrail import VisionGuardrail, create_overlay
from ..io_utils import read_image


@dataclass
class TriageResult:
    stem: str
    category: str  # "good" | "needs_fix"
    quality_score: float  # fast score (0-1)
    vlm_score: float  # VLM score if called, else -1
    issues: List[str] = field(default_factory=list)
    view_type: str = "4ch"


def _detect_issues(feats: Dict[str, float]) -> List[str]:
    """Detect anatomical issues from quality features."""
    issues: List[str] = []
    if feats.get("cc_count_myo", 1.0) > 1:
        issues.append("disconnected_myo")
    if feats.get("cc_count_lv", 1.0) > 1:
        issues.append("fragmented_lv")
    if feats.get("cc_count_rv", 1.0) > 1:
        issues.append("fragmented_rv")
    if feats.get("touch_ratio", 0.0) > 0.01:
        issues.append("rv_lv_touching")
    if feats.get("hole_frac_lv", 0.0) > 0.05:
        issues.append("holes")
    if feats.get("hole_frac_myo", 0.0) > 0.05:
        issues.append("holes")
    if feats.get("valve_leak_ratio", 0.0) > 0.1:
        issues.append("valve_leak")
    if feats.get("thickness_cv", 0.0) > 0.5:
        issues.append("myo_thickness_var")
    if feats.get("boundary_over_area", 0.0) > 0.18:
        issues.append("jagged_boundary")
    if feats.get("myo_missing", 0.0) > 0.5:
        issues.append("missing_myo")
    if feats.get("lv_missing", 0.0) > 0.5:
        issues.append("missing_lv")
    if feats.get("rv_missing", 0.0) > 0.5 and feats.get("view_is_4ch", 0.0) > 0.5:
        issues.append("missing_rv")
    # deduplicate
    return list(dict.fromkeys(issues))


def _heuristic_quality_score(feats: Dict[str, float]) -> float:
    """Compute a heuristic quality score (0-1) from geometric features."""
    score = 1.0
    # Penalize multiple components
    for key in ["cc_count_rv", "cc_count_myo", "cc_count_lv"]:
        cc = feats.get(key, 1.0)
        if cc > 1:
            score -= 0.08 * (cc - 1)
    # Penalize holes
    score -= 0.15 * feats.get("hole_frac_lv", 0.0)
    score -= 0.10 * feats.get("hole_frac_myo", 0.0)
    # Penalize touching
    score -= 0.20 * min(1.0, feats.get("touch_ratio", 0.0) / 0.05)
    # Penalize valve leak
    score -= 0.10 * min(1.0, feats.get("valve_leak_ratio", 0.0) / 0.2)
    # Penalize high thickness CV
    score -= 0.10 * min(1.0, max(0.0, feats.get("thickness_cv", 0.0) - 0.3) / 0.4)
    # Penalize jagged boundary
    score -= 0.10 * min(1.0, max(0.0, feats.get("boundary_over_area", 0.0) - 0.13) / 0.05)
    # Penalize missing structures
    score -= 0.15 * feats.get("myo_missing", 0.0)
    score -= 0.15 * feats.get("lv_missing", 0.0)
    # Reward compactness
    compact = feats.get("lv_compactness", 0.0)
    if compact > 0.7:
        score += 0.05
    return float(max(0.0, min(1.0, score)))


class TriageAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        triage_cfg = cfg.get("triage", {})
        self.enabled = bool(triage_cfg.get("enabled", True))
        self.good_threshold = float(triage_cfg.get("good_threshold", 0.95))
        self.bad_threshold = float(triage_cfg.get("bad_threshold", 0.70))
        self.vlm_enabled = bool(triage_cfg.get("vlm_enabled", True))
        self.vlm_accept_threshold = int(triage_cfg.get("vlm_accept_threshold", 70))
        self.vlm = VisionGuardrail(cfg) if self.vlm_enabled else None

        # Try to load trained quality scorer
        q_cfg = cfg.get("quality_model", {})
        self.scorer = None
        self._blend_mode = q_cfg.get("blend_mode", "hybrid")
        self._blend_alpha = float(q_cfg.get("blend_alpha", 0.5))
        if q_cfg.get("enabled", False):
            path = q_cfg.get("path", "quality_model.pkl")
            self.scorer = QualityScorer.load(path)

    def _fast_score(self, mask: np.ndarray, img: np.ndarray, view_type: str) -> tuple[float, Dict[str, float]]:
        """Tier 1: fast feature-based quality score with hybrid blending."""
        feats = extract_quality_features(mask, img, view_type)
        h_score = _heuristic_quality_score(feats)

        if self.scorer is None:
            return h_score, feats

        t_score, _ = self.scorer.score(mask, img, view_type)
        blend = self._blend_mode
        if blend == "replace":
            return t_score, feats
        if blend == "hybrid":
            alpha = self._blend_alpha
            return alpha * t_score + (1 - alpha) * h_score, feats
        return h_score, feats

    def _vlm_score(self, img_path: str, mask: np.ndarray, stem: str) -> float:
        """Tier 2: VLM-based score for borderline cases."""
        if self.vlm is None or not self.vlm.enabled:
            return -1.0
        report = self.vlm.check(img_path, mask, f"{stem}_triage")
        return float(report.get("score", 0))

    def triage_folder(
        self,
        folder: str,
        metas: Dict[str, Dict[str, str]],
        masks: Dict[str, np.ndarray],
        view_types: Dict[str, str],
    ) -> Dict[str, TriageResult]:
        """
        Classify every sample as "good" or "needs_fix".

        Tier 1 (fast): extract_quality_features -> heuristic/trained score
          - Score >= good_threshold (0.95) -> "good" (very confident, skip)
          - Score <= bad_threshold (0.70) -> "needs_fix" (obviously bad)
          - Everything else -> Tier 2

        Tier 2 (VLM): for ALL borderline samples, send overlay to VLM
          - VLM decides good vs needs_fix
          - Enabled by default

        Returns mapping stem -> TriageResult.
        """
        if not self.enabled:
            # If triage disabled, mark everything as needs_fix (process all)
            return {
                stem: TriageResult(
                    stem=stem,
                    category="needs_fix",
                    quality_score=0.0,
                    vlm_score=-1.0,
                    view_type=view_types.get(stem, "4ch"),
                )
                for stem in masks
            }

        results: Dict[str, TriageResult] = {}
        borderline_stems: List[str] = []

        # Tier 1: fast scoring
        for stem, mask in masks.items():
            fr = metas[stem]
            img = read_image(fr["img"])
            vt = view_types.get(stem, "4ch")

            q_score, feats = self._fast_score(mask, img, vt)
            issues = _detect_issues(feats)

            if q_score >= self.good_threshold:
                category = "good"
            elif q_score <= self.bad_threshold:
                category = "needs_fix"
            else:
                category = "borderline"
                borderline_stems.append(stem)

            results[stem] = TriageResult(
                stem=stem,
                category=category,
                quality_score=q_score,
                vlm_score=-1.0,
                issues=issues,
                view_type=vt,
            )

        # Tier 2: VLM for ALL borderline cases
        if borderline_stems:
            if self.vlm_enabled and self.vlm is not None:
                print(f"[Triage] VLM screening {len(borderline_stems)} borderline samples...")
                for stem in borderline_stems:
                    fr = metas[stem]
                    vlm_s = self._vlm_score(fr["img"], masks[stem], stem)
                    results[stem].vlm_score = vlm_s
                    if vlm_s >= self.vlm_accept_threshold:
                        results[stem].category = "good"
                    else:
                        results[stem].category = "needs_fix"
            else:
                # VLM disabled: treat borderline as needs_fix (conservative)
                print(f"[Triage] {len(borderline_stems)} borderline -> needs_fix (VLM disabled)")
                for stem in borderline_stems:
                    results[stem].category = "needs_fix"

        n_good = sum(1 for r in results.values() if r.category == "good")
        n_fix = sum(1 for r in results.values() if r.category == "needs_fix")
        print(f"[Triage] {len(results)} samples: {n_good} good, {n_fix} needs_fix")
        return results

    def save_triage_report(self, results: Dict[str, TriageResult], output_path: str) -> None:
        """Write triage results to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        header = ["stem", "category", "quality_score", "vlm_score", "issues", "view_type"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for stem in sorted(results.keys()):
                r = results[stem]
                w.writerow({
                    "stem": r.stem,
                    "category": r.category,
                    "quality_score": f"{r.quality_score:.4f}",
                    "vlm_score": f"{r.vlm_score:.1f}",
                    "issues": ";".join(r.issues),
                    "view_type": r.view_type,
                })
