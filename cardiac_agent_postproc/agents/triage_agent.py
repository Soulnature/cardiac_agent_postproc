"""
TriageAgent: VLM-powered quality triage for segmentation masks.

Combines optional fast classifier with VLM visual judgment using
few-shot examples from the visual knowledge base.
Can be challenged by other agents (e.g., Coordinator, Verifier).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..quality_model import extract_quality_features, QualityScorer
from ..vlm_guardrail import create_overlay
from ..io_utils import read_image
from .base_agent import BaseAgent
from .message_bus import AgentMessage, CaseContext, MessageBus, MessageType
from .visual_knowledge import VisualKnowledgeBase


# ---- Feature-based issue detection (preserved from original) ----

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
    if feats.get("valve_leak_ratio", 0) > 0.03:
        issues.append("valve_leak")
    if feats.get("thickness_cv", 0) > 0.5:
        issues.append("myo_thickness_var")
    if feats.get("myo_missing", 0) > 0.5:
        issues.append("missing_myo")
    if feats.get("lv_missing", 0) > 0.5:
        issues.append("missing_lv")
    if feats.get("rv_missing", 0) > 0.5 and feats.get("view_is_4ch", 0) > 0.5:
        issues.append("missing_rv")
    return list(set(issues))


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
        prompt = (
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
        issues = _detect_issues(feats)

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
                    vlm_verdict = self.judge_visual_quality(overlay_path)
            except Exception as e:
                self.log(f"VLM triage failed: {e}")

        # Step 4: Decision logic — VLM-first, classifier-assisted
        if vlm_verdict and vlm_verdict.get("confidence", 0) > 0.6:
            # VLM is confident → use its verdict
            vlm_quality = vlm_verdict.get("quality", "borderline")
            if vlm_quality == "good" and not issues:
                category = "good"
                confidence = vlm_verdict["confidence"]
                reasoning = f"VLM: {vlm_verdict.get('reasoning', 'good quality')}"
            elif vlm_quality == "bad":
                category = "needs_fix"
                confidence = vlm_verdict["confidence"]
                vlm_issues = vlm_verdict.get("issues", [])
                issues = list(set(issues + vlm_issues))
                reasoning = f"VLM: {vlm_verdict.get('reasoning', 'bad quality')}"
            else:
                # Borderline VLM + features → LLM decides
                category, confidence, reasoning, issues = self._llm_triage(
                    ctx, feats, issues, classifier_score, bad_prob, is_bad, vlm_verdict
                )
        elif issues:
            # No VLM but issues detected → needs fix
            category = "needs_fix"
            confidence = max(bad_prob, 0.6)
            reasoning = f"Feature issues detected: {issues}"
        else:
            # Fallback to LLM reasoning
            category, confidence, reasoning, issues = self._llm_triage(
                ctx, feats, issues, classifier_score, bad_prob, is_bad, vlm_verdict
            )

        # Normalize triage score as "probability of being good" for fast-path gating.
        triage_score = classifier_score
        if vlm_verdict:
            vlm_quality = vlm_verdict.get("quality", "borderline")
            vlm_conf = float(vlm_verdict.get("confidence", 0.5))
            if vlm_quality == "good":
                triage_score = vlm_conf
            elif vlm_quality == "bad":
                triage_score = 1.0 - vlm_conf
            else:
                triage_score = 0.5
        triage_score = float(max(0.0, min(1.0, triage_score)))

        # Update case context
        ctx.triage_category = category
        ctx.triage_score = triage_score
        ctx.triage_issues = issues

        self.log(f"  → {category} (vlm={vlm_verdict.get('quality', 'N/A')}, issues={issues})")

        # Send result to bus
        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.TRIAGE_RESULT,
            content={
                "category": category,
                "score": triage_score,
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
            issues = list(set(issues + llm_issues))
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
                ctx.triage_issues = list(set(ctx.triage_issues + result["issues"]))

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
