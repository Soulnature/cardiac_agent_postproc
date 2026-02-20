"""
TriageAgent: VLM-powered quality triage for segmentation masks.

Combines optional fast classifier with VLM visual judgment using
few-shot examples from the visual knowledge base.
Can be challenged by other agents (e.g., Coordinator, Verifier).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..quality_model import extract_quality_features, QualityScorer
from ..vlm_guardrail import create_overlay
from ..io_utils import read_image
from .base_agent import BaseAgent
from .message_bus import AgentMessage, CaseContext, MessageBus, MessageType
from .visual_knowledge import VisualKnowledgeBase


# ---- Feature-based issue detection ----

# Severity weights for each issue type.
# Critical structural defects = 1.0, minor cosmetic issues = 0.2
ISSUE_SEVERITY: Dict[str, float] = {
    "disconnected_myo": 1.0,
    "fragmented_lv": 0.9,
    "fragmented_rv": 0.7,
    "rv_lv_touching": 0.9,
    "holes": 0.6,
    "valve_leak": 0.3,
    "myo_thickness_var": 0.2,
    "missing_myo": 1.0,
    "missing_lv": 1.0,
    "missing_rv": 0.8,
}


def _detect_issues(feats: Dict[str, float]) -> List[str]:
    """Detect anatomical issues from quality features."""
    hole_myo = feats.get("hole_fraction_myo", feats.get("hole_frac_myo", 0))
    hole_lv = feats.get("hole_fraction_lv", feats.get("hole_frac_lv", 0))
    issues = []
    if feats.get("cc_count_myo", 1) > 1:
        issues.append("disconnected_myo")
    if feats.get("cc_count_lv", 1) > 1:
        issues.append("fragmented_lv")
    if feats.get("cc_count_rv", 1) > 1:
        issues.append("fragmented_rv")
    if feats.get("touch_ratio", 0) > 0.01:
        issues.append("rv_lv_touching")
    if hole_myo > 0.02:
        issues.append("holes")
    if hole_lv > 0.02:
        issues.append("holes")
    if feats.get("valve_leak_ratio", 0) > 0.10:
        issues.append("valve_leak")
    if feats.get("thickness_cv", 0) > 0.9:
        issues.append("myo_thickness_var")
    if feats.get("myo_missing", 0) > 0.5:
        issues.append("missing_myo")
    if feats.get("lv_missing", 0) > 0.5:
        issues.append("missing_lv")
    if feats.get("rv_missing", 0) > 0.5 and feats.get("view_is_4ch", 0) > 0.5:
        issues.append("missing_rv")
    return list(set(issues))


def _issue_severity(issues: List[str]) -> float:
    """Compute aggregate severity score from issue list."""
    return sum(ISSUE_SEVERITY.get(iss, 0.5) for iss in issues)


def _canonicalize_issue(issue: Any) -> Optional[str]:
    """Normalize noisy VLM/LLM issue strings to canonical triage issue labels."""
    if issue is None:
        return None
    text = str(issue).strip().lower()
    if not text:
        return None
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    direct_map = {
        "disconnected myo": "disconnected_myo",
        "broken ring": "disconnected_myo",
        "discmyo": "disconnected_myo",
        "rv lv touching": "rv_lv_touching",
        "rv lv contact": "rv_lv_touching",
        "rv lv merge": "rv_lv_touching",
        "fragmented lv": "fragmented_lv",
        "fragmented rv": "fragmented_rv",
        "missing myo": "missing_myo",
        "missing lv": "missing_lv",
        "missing rv": "missing_rv",
        "valve leak": "valve_leak",
        "myo thickness var": "myo_thickness_var",
    }
    if text in direct_map:
        return direct_map[text]

    if text in ISSUE_SEVERITY:
        return text

    if "hole" in text:
        return "holes"

    if "valve" in text and ("leak" in text or "regurg" in text):
        return "valve_leak"

    if "missing" in text:
        if "myo" in text or "myocard" in text:
            return "missing_myo"
        if "lv" in text:
            return "missing_lv"
        if "rv" in text:
            return "missing_rv"

    if "fragment" in text:
        if "lv" in text:
            return "fragmented_lv"
        if "rv" in text:
            return "fragmented_rv"

    if (
        ("touch" in text or "contact" in text or "merge" in text or "overlap" in text)
        and "lv" in text
        and "rv" in text
    ):
        return "rv_lv_touching"

    if (
        "myo" in text or "myocard" in text or "ring" in text
    ) and (
        "gap" in text or "break" in text or "discontinu" in text or "incomplete" in text
    ):
        return "disconnected_myo"

    if "thickness" in text or "rough" in text or "jagged" in text:
        if "myo" in text or "myocard" in text:
            return "myo_thickness_var"

    return None


def _normalize_issues(issues: List[Any]) -> List[str]:
    """Canonicalize + deduplicate + sort issues by severity."""
    out = set()
    for issue in issues:
        canon = _canonicalize_issue(issue)
        if canon:
            out.add(canon)
    return sorted(out, key=lambda x: (-ISSUE_SEVERITY.get(x, 0.0), x))


class TriageAgent(BaseAgent):
    """
    LLM-powered triage agent.

    Pipeline:
    1. Fast path: trained classifier gives score + is_bad flag
    2. Feature-based issue detection
    3. LLM reasoning for borderline cases (synthesizes classifier + features)
    4. Can handle CHALLENGE messages from other agents
    """

    def __init__(
        self,
        cfg: dict,
        bus: MessageBus,
        visual_kb: Optional[VisualKnowledgeBase] = None,
    ):
        # Load classifier before super().__init__ (which calls self.name)
        self.cfg = cfg  # needed before super().__init__
        self._scorer: Optional[QualityScorer] = None

        qm_cfg = dict(cfg.get("quality_model", {}))
        triage_agent_cfg = cfg.get("agents", {}).get("triage", {})
        if isinstance(triage_agent_cfg, dict):
            q_override = triage_agent_cfg.get("quality_model", {})
            if isinstance(q_override, dict):
                qm_cfg.update(q_override)
        if qm_cfg.get("enabled", False):
            model_path = qm_cfg.get("path", "quality_model.pkl")
            if os.path.exists(model_path):
                try:
                    self._scorer = QualityScorer.load(model_path)
                except Exception as e:
                    print(f"[triage] Failed to load quality model: {e}")

        super().__init__(cfg, bus, visual_kb=visual_kb)

    @property
    def name(self) -> str:
        return "triage"

    @property
    def system_prompt(self) -> str:
        fallback = (
            "You are the Triage Agent in a multi-agent cardiac MRI segmentation repair system.\n\n"
            "Your role is to quickly assess whether a segmentation mask needs repair or is acceptable.\n"
            "You combine VLM visual judgment with quantitative features for robust triage.\n\n"
            "Decision framework:\n"
            "- If VLM judges 'good' with high confidence AND no issues → 'good'\n"
            "- If VLM judges 'bad' AND issues are detected → 'needs_fix'\n"
            "- For borderline VLM verdicts: reason about the features and make a judgment call\n\n"
            "You can be challenged by other agents. When challenged, re-examine your evidence.\n\n"
            "Always respond with JSON:\n"
            '{"category": "good" | "needs_fix", "confidence": 0.0-1.0, '
            '"issues": [...], "reasoning": "..."}'
        )
        prompt = self._prompt_with_fallback("triage_system.txt", fallback)
        # Inject knowledge from visual examples
        if self.visual_kb:
            prompt += "\n\n" + self.visual_kb.build_knowledge_prompt()
        return prompt

    def process_case(self, ctx: CaseContext) -> None:
        """
        Triage a single case: classify as good or needs_fix.
        Uses VLM visual judgment as primary signal, with optional classifier backup.
        Updates ctx.triage_category, ctx.triage_score, ctx.triage_issues.
        """
        assert ctx.mask is not None and ctx.image is not None
        self.log(f"Triaging {ctx.stem} (view={ctx.view_type})")

        # Step 1: Feature-based issue detection (always available)
        feats = extract_quality_features(ctx.mask, ctx.image, ctx.view_type)
        issues = _normalize_issues(_detect_issues(feats))
        agent_cfg = self._agent_cfg()
        vlm_n_refs = max(1, int(agent_cfg.get("vlm_n_refs", 2)))
        vlm_match_view = bool(agent_cfg.get("vlm_ref_match_view", True))
        vlm_four_way_refs = bool(agent_cfg.get("vlm_four_way_refs", False))
        raw_ref_categories = agent_cfg.get(
            "vlm_ref_categories",
            ["01_good", "02_good_low_dice", "03_bad_high_dice", "04_bad"],
        )
        vlm_ref_categories = (
            list(raw_ref_categories)
            if isinstance(raw_ref_categories, list)
            else ["01_good", "02_good_low_dice", "03_bad_high_dice", "04_bad"]
        )
        vlm_ref_seed = f"triage:{ctx.stem}"

        # Step 2: Optional fast classifier
        classifier_score = 0.5
        is_bad = False
        bad_prob = 0.0
        if self._scorer is not None:
            try:
                is_bad, bad_prob, _feats = self._scorer.classify(ctx.mask, ctx.image, ctx.view_type)
                classifier_score = 1.0 - bad_prob
            except Exception as e:
                self.log(f"Classifier failed: {e}")

        # Step 3: VLM visual judgment (primary decision signal)
        vlm_verdict = {}
        overlay_path = ""
        if ctx.img_path and self.visual_kb:
            try:
                overlay_path = ctx.debug_img_path("_triage_overlay.png")
                success = create_overlay(ctx.img_path, ctx.mask, overlay_path)
                if success:
                    vlm_verdict = self.judge_visual_quality(
                        overlay_path=overlay_path,
                        n_refs=vlm_n_refs,
                        view_type=ctx.view_type if vlm_match_view else None,
                        seed=vlm_ref_seed,
                        four_way_categories=(
                            vlm_ref_categories if vlm_four_way_refs else None
                        ),
                    )
            except Exception as e:
                self.log(f"VLM triage failed: {e}")

        # Step 4: Compute severity from feature issues
        severity = _issue_severity(issues)

        # Step 5: Decision logic — VLM score + severity-based gating
        # Default from features using severity scoring
        if severity >= 0.8:
            category = "needs_fix"
        elif severity < 0.5 and not is_bad:
            category = "good"
        else:
            category = "needs_fix"  # conservative default for moderate severity
        confidence = max(bad_prob, 0.6)
        reasoning = f"Feature severity={severity:.2f}, issues={issues}"

        if vlm_verdict:
            vlm_quality = vlm_verdict.get("quality", "borderline")
            vlm_conf = float(vlm_verdict.get("confidence", 0.5))
            vlm_score = float(vlm_verdict.get("score", 0))
            # Triage thresholds are configurable per model in agents.triage.*
            # Keeping strict defaults prevents over-accepting borderline masks.
            good_min_score = float(agent_cfg.get("vlm_good_min_score", 9.0))
            good_min_conf = float(agent_cfg.get("vlm_good_min_conf", 0.85))
            borderline_auto_good_max_severity = float(
                agent_cfg.get("borderline_auto_good_max_severity", 0.2)
            )

            # Use VLM numeric score to correct unreliable quality text
            if vlm_score >= 7:
                vlm_quality = "good"
            elif vlm_score <= 4:
                vlm_quality = "bad"

            if (
                vlm_quality == "good"
                and vlm_conf >= good_min_conf
                and vlm_score >= good_min_score
                and severity < 0.5
            ):
                category = "good"
                confidence = vlm_conf
                reasoning = (f"VLM Good (score={vlm_score}, conf={vlm_conf:.2f}). "
                             f"Feature severity={severity:.2f}, issues={issues}")

            elif vlm_quality == "bad":
                category = "needs_fix"
                confidence = vlm_conf
                vlm_issues = vlm_verdict.get("issues", [])
                issues = _normalize_issues(issues + list(vlm_issues))
                reasoning = f"VLM Bad (score={vlm_score})"

            else:
                # Borderline VLM → use severity-based rescue
                if severity < borderline_auto_good_max_severity and not is_bad:
                    category = "good"
                    confidence = 0.6
                    reasoning = (f"VLM borderline but low severity={severity:.2f}. "
                                 f"Minor issues only: {issues}")
                elif severity < 0.8:
                    # True borderline → LLM decides
                    category, confidence, reasoning, issues = self._llm_triage(
                        ctx, feats, issues, classifier_score, bad_prob, is_bad, vlm_verdict
                    )
                # severity >= 0.8 keeps default "needs_fix"

        # Final issue normalization after all signal merges.
        issues = _normalize_issues(issues)
        # Severity should reflect merged signals (features + VLM/LLM tags).
        severity = max(severity, _issue_severity(issues))

        # Calibrated triage score as "probability of being good" for fast-path gating.
        # Blend VLM and quality-model signals to improve ranking stability (AUC).
        triage_score = classifier_score
        vlm_good_prob: Optional[float] = None
        if vlm_verdict:
            vlm_score = float(vlm_verdict.get("score", 0))
            vlm_conf = float(vlm_verdict.get("confidence", 0.5))

            # Normalize VLM score (1..10 -> 0..1), then shrink toward 0.5 if confidence is low.
            score_norm = max(0.0, min(1.0, (vlm_score - 1.0) / 9.0))
            conf_scale = max(0.0, min(1.0, 0.5 + 0.5 * vlm_conf))
            vlm_good_prob = 0.5 + (score_norm - 0.5) * conf_scale
            vlm_good_prob = float(max(0.0, min(1.0, vlm_good_prob)))

            vlm_w = float(agent_cfg.get("vlm_good_prob_weight", 0.85))
            qm_w = float(agent_cfg.get("qm_good_prob_weight", 0.15 if self._scorer is not None else 0.0))
            if self._scorer is None:
                qm_w = 0.0
            denom = max(vlm_w + qm_w, 1e-6)
            triage_score = (vlm_w * vlm_good_prob + qm_w * classifier_score) / denom

        # Penalize high-severity anatomical defects so score is safer for fast-path gating.
        severity_penalty = min(0.25, 0.04 * severity)
        triage_score = float(max(0.0, min(1.0, triage_score - severity_penalty)))
        triage_bad_prob = float(1.0 - triage_score)

        # Final consistency pass: category should agree with calibrated score,
        # while still respecting hard structural defects.
        critical_issues = {
            "disconnected_myo",
            "missing_myo",
            "missing_lv",
            "rv_lv_touching",
            "fragmented_lv",
        }
        has_critical_issue = any(iss in critical_issues for iss in issues)
        score_good_floor = float(agent_cfg.get("score_good_floor", 0.72))
        score_bad_ceiling = float(agent_cfg.get("score_bad_ceiling", 0.40))

        if severity >= 0.8:
            category = "needs_fix"
        elif has_critical_issue and triage_score < 0.9:
            category = "needs_fix"
        elif triage_score <= score_bad_ceiling:
            category = "needs_fix"
        elif triage_score >= score_good_floor and not has_critical_issue:
            category = "good"

        confidence = float(max(0.0, min(1.0, confidence)))
        if category == "good":
            confidence = max(confidence, triage_score)
        else:
            confidence = max(confidence, triage_bad_prob)

        # Update case context
        ctx.triage_category = category
        ctx.triage_score = triage_score
        ctx.triage_issues = issues
        ctx.vlm_verdict_data = vlm_verdict # Store full dictionary
        ctx.vlm_quality = vlm_verdict.get("quality", "N/A") if vlm_verdict else "N/A"
        ctx.vlm_score = float(vlm_verdict.get("score", -1.0)) if vlm_verdict else -1.0
        ctx.vlm_confidence = float(vlm_verdict.get("confidence", 0.0)) if vlm_verdict else 0.0
        ctx.vlm_good_prob = float(vlm_good_prob) if vlm_good_prob is not None else -1.0
        ctx.qm_bad_prob = float(bad_prob)
        ctx.triage_bad_prob = triage_bad_prob
        ctx.triage_severity = float(severity)


        self.log(
            f"  → {category} (vlm={vlm_verdict.get('quality', 'N/A')}, "
            f"score={triage_score:.3f}, bad_prob={triage_bad_prob:.3f}, issues={issues})"
        )

        # Send result to bus
        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.TRIAGE_RESULT,
            content={
                "category": category,
                "score": triage_score,
                "bad_prob": triage_bad_prob,
                "confidence": confidence,
                "issues": issues,
                "reasoning": reasoning,
                "vlm_verdict": vlm_verdict,
            },
            case_id=ctx.case_id,
        )

    def _llm_triage(
        self, ctx, feats, issues, classifier_score, bad_prob, is_bad, vlm_verdict
    ):
        """LLM reasoning for borderline cases, combining all signals."""
        prompt = (
            f"Case: {ctx.stem} (view: {ctx.view_type})\n"
            f"Classifier quality score: {classifier_score:.3f} (bad_prob={bad_prob:.3f})\n"
            f"VLM visual judgment: {vlm_verdict if vlm_verdict else 'unavailable'}\n"
            f"Detected issues: {issues if issues else 'none'}\n"
            f"Key features:\n"
            f"  - cc_myo={feats.get('cc_count_myo', 0):.0f}, "
            f"cc_lv={feats.get('cc_count_lv', 0):.0f}, "
            f"cc_rv={feats.get('cc_count_rv', 0):.0f}\n"
            f"  - touch_ratio={feats.get('touch_ratio', 0):.4f}\n"
            f"  - thickness_cv={feats.get('thickness_cv', 0):.3f}\n"
            f"  - hole_frac_myo={feats.get('hole_fraction_myo', feats.get('hole_frac_myo', 0)):.4f}\n"
            f"  - valve_leak={feats.get('valve_leak_ratio', 0):.4f}\n\n"
            f"Should this case be classified as 'good' or 'needs_fix'? "
            f"Think step by step, then give your verdict."
        )
        result = self.think_json(prompt)
        category = result.get("category", "needs_fix" if is_bad else "good")
        confidence = result.get("confidence", bad_prob)
        reasoning = result.get("reasoning", "LLM decision")
        llm_issues = result.get("issues", [])
        if llm_issues:
            issues = _normalize_issues(issues + list(llm_issues))
        return category, confidence, reasoning, issues

    def handle_challenge(self, msg: AgentMessage, ctx: CaseContext) -> None:
        """Handle a CHALLENGE message from another agent."""
        self.log(f"Received challenge from {msg.sender}: {msg.content}")
        original_category = ctx.triage_category

        # Re-examine with the challenger's evidence
        prompt = (
            f"Another agent ({msg.sender}) is challenging your triage of case {ctx.stem}.\n"
            f"Your original verdict: {ctx.triage_category}\n"
            f"Their concern: {msg.content.get('reason', 'unspecified')}\n"
            f"Additional evidence: {msg.content.get('evidence', 'none')}\n\n"
            f"Re-examine your decision. Should you change your verdict?"
        )

        result = self.think_json(prompt)
        new_category = result.get("category", ctx.triage_category)

        if new_category != ctx.triage_category:
            self.log(f"  → Changed verdict: {ctx.triage_category} → {new_category}")
            ctx.triage_category = new_category
            if result.get("issues"):
                ctx.triage_issues = _normalize_issues(ctx.triage_issues + list(result["issues"]))

        changed = new_category != original_category

        self.send_message(
            recipient=msg.sender,
            msg_type=MessageType.CHALLENGE_RESPONSE,
            content={
                "original": original_category,
                "revised": new_category,
                "changed": changed,
                "reasoning": result.get("reasoning", ""),
            },
            case_id=ctx.case_id,
        )
