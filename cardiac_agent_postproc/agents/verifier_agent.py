"""
VerifierAgent: VLM-powered quality verification and gating.

Validates fixes using VLM visual comparison (before/after overlays),
with optional GT metrics (Dice/HD95) for reporting. The VLM visual
judgment is the primary decision driver, replacing RQS.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..eval_metrics import dice_macro, hd95_macro
from ..vlm_guardrail import create_overlay, create_side_by_side_overlay, create_diff_overlay
from .base_agent import BaseAgent
from .message_bus import CaseContext, MessageBus, MessageType
from .visual_knowledge import VisualKnowledgeBase


class VerifierAgent(BaseAgent):
    """
    Verifies and gates repair results.

    Pipeline per case:
    1. VLM visual comparison (before vs after overlay) — PRIMARY decision signal
    2. Optionally compute GT metrics (Dice) for reporting
    3. LLM reasoning to approve or reject the fix
    4. Can challenge the TriageAgent if post-fix analysis reveals new issues
    """

    @property
    def name(self) -> str:
        return "verifier"

    @property
    def system_prompt(self) -> str:
        fallback = (
            "You are the Verifier Agent in a cardiac MRI segmentation repair system.\n\n"
            "Your role is to critically evaluate whether a repair actually improved the mask.\n"
            "You receive:\n"
            "1. VLM visual comparison of before vs after overlays (PRIMARY signal)\n"
            "2. Optionally, before/after Dice scores (if GT available)\n"
            "3. Summary of operations applied\n\n"
            "You must decide:\n"
            "- APPROVE: the fix improved the mask (or at least didn't worsen it)\n"
            "- REJECT: the fix made things worse — revert to original\n"
            "- NEEDS_MORE_WORK: partial improvement, but more fixes needed\n\n"
            "Be conservative: if unsure, lean towards REJECT rather than accepting bad fixes.\n\n"
            "Respond with JSON:\n"
            '{"verdict": "approve" | "reject" | "needs_more_work", '
            '"confidence": 0.0-1.0, "reasoning": "...", '
            '"remaining_issues": [...]}'
        )
        prompt = self._prompt_with_fallback("verifier_system.txt", fallback)
        if self.visual_kb:
            prompt += "\n\n" + self.visual_kb.build_knowledge_prompt()
        return prompt

    def process_case(self, ctx: CaseContext) -> None:
        """
        Verify the repair result for a case.
        Uses VLM visual comparison as primary quality signal.
        Updates ctx.verified, ctx.verifier_approved, ctx.verifier_feedback.
        """
        assert ctx.current_mask is not None
        self.log(f"Verifying {ctx.stem}")

        # Step 1: VLM visual comparison (PRIMARY decision signal)
        ex_cfg = self.cfg.get("executor", {})
        strict_monotonic_mode = bool(ex_cfg.get("strict_monotonic_mode", False))
        strict_round_min_compare_conf = float(
            ex_cfg.get("strict_round_min_compare_confidence", 0.60)
        )

        vlm_comparison = {}
        before_overlay = ""
        after_overlay = ""
        comparison_reference = "original"

        if ctx.img_path and ctx.original_mask is not None and self.visual_kb:
            (
                vlm_comparison,
                before_overlay,
                after_overlay,
                comparison_reference,
            ) = self._vlm_compare(ctx)

        # Step 2: VLM quality judgment on the AFTER image
        vlm_quality = {}
        if after_overlay and self.visual_kb:
            vlm_quality = self.judge_visual_quality(after_overlay)

        # Step 3: Optional GT metrics (for reporting, not decision driving)
        metrics = self._compute_metrics(ctx)

        # Step 4: LLM reasoning — synthesize all signals
        prompt = (
            f"Verify the repair of case {ctx.stem} (view: {ctx.view_type}).\n\n"
            f"**Operations applied**: {json.dumps([op['name'] for op in ctx.applied_ops])}\n"
            f"**Comparison reference**: {comparison_reference}\n"
        )

        if vlm_comparison:
            prompt += (
                f"**VLM Before/After Comparison**: {json.dumps(vlm_comparison)}\n"
            )

        if vlm_quality:
            prompt += (
                f"**VLM Quality of AFTER**: {json.dumps(vlm_quality)}\n"
            )

        if metrics.get("dice_before") is not None:
            prompt += (
                f"**Dice change**: {metrics['dice_before']:.4f} → "
                f"{metrics['dice_after']:.4f} (Δ={metrics['dice_delta']:+.4f})\n"
            )

        prompt += (
            f"\n**Pixels changed**: {sum(op.get('pixels_changed', 0) for op in ctx.applied_ops)}\n"
            f"**Rounds completed**: {ctx.rounds_completed + 1}\n\n"
            f"Based on the VLM visual assessment, should this fix be approved, rejected, "
            f"or does it need more work?"
        )

        result = self.think_json(prompt)
        verdict = str(result.get("verdict", "approve")).lower()
        if verdict not in ("approve", "reject", "needs_more_work"):
            verdict = "needs_more_work"
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")
        remaining = result.get("remaining_issues", [])
        if not isinstance(remaining, list):
            remaining = [str(remaining)]

        # Apply fallback logic if LLM fails
        if not result:
            # Fallback: use VLM comparison verdict
            cmp_v = str(vlm_comparison.get("verdict", "")).lower()
            if cmp_v == "improved":
                verdict = "approve"
                reasoning = "Fallback: VLM says improved"
            elif cmp_v == "worse":
                verdict = "reject"
                reasoning = "Fallback: VLM says worse"
            elif cmp_v == "same":
                verdict = "needs_more_work"
                reasoning = "Fallback: VLM says same; continue refining"
                remaining = ["no_clear_visual_improvement"]
            else:
                verdict = "reject"
                reasoning = "Fallback: uncertain, conservative reject"

        # Policy: if VLM comparison says "same", do not hard reject.
        cmp_v = str(vlm_comparison.get("verdict", "")).lower()
        if cmp_v == "same" and verdict == "reject":
            self.log("  → Policy: VLM comparison='same', downgrade reject -> needs_more_work")
            verdict = "needs_more_work"
            if not remaining:
                remaining = ["no_clear_visual_improvement"]
            reasoning = (
                f"{reasoning} "
                "(Policy: VLM says same, continue another refinement round instead of hard reject.)"
            ).strip()

        # Override: for severely degraded cases where the VLM comparison
        # confirms improvement, accept the repair even if absolute quality
        # is still "bad".  The absolute VLM quality will always be "bad" for
        # these cases (Dice < 0.5) because the starting point is so poor.
        initial_is_bad = getattr(ctx, "triage_category", "needs_fix") != "good"
        vlm_improved = (
            vlm_comparison.get("verdict") == "improved"
            and vlm_comparison.get("confidence", 0) >= 0.7
        )
        ops_applied = len(ctx.applied_ops) > 0
        
        # New Relaxation: If verdict is "needs_more_work" but VLM says improved, APPROVE it.
        # This allows incremental progress to be saved.
        if verdict == "needs_more_work" and vlm_improved:
             self.log(f"  → Override: upgrading 'needs_more_work' to 'approve' because VLM=improved")
             verdict = "approve"
             reasoning += " (Upgraded from needs_more_work due to VLM improvement)"

        # RELAXATION: Quantitative Override for Worst Cases
        # DISABLED per user request (Reference-Free Repair only)
        # dice_delta = metrics.get("dice_delta", 0.0)
        # if (cmp_v in ("same", "neutral") or verdict == "needs_more_work") and dice_delta > 0.015:
        #      self.log(f"  → Override: upgrading '{verdict}' to 'approve' because Dice improved (+{dice_delta:.4f}) despite VLM neutral")
        #      verdict = "approve"
        #      reasoning += f" (Quantitative Override: Dice +{dice_delta:.4f} > 0.015)"

        if initial_is_bad and vlm_improved and ops_applied and verdict != "approve":
            self.log(f"  → Override: VLM comparison=improved, accepting repair for bad case")
            verdict = "approve"
            reasoning = (
                f"Override: VLM comparison confirmed improvement "
                f"(conf={vlm_comparison.get('confidence', 0):.2f}), "
                f"{len(ctx.applied_ops)} ops applied. "
                f"Original: {reasoning}"
            )

        if strict_monotonic_mode:
            cmp_v = str(vlm_comparison.get("verdict", "")).lower()
            cmp_conf = float(vlm_comparison.get("confidence", 0.0) or 0.0)
            if not vlm_comparison:
                verdict = "reject"
                reasoning = (
                    "Strict monotonic mode: missing VLM comparison signal; "
                    "rejecting round for safety."
                )
            elif cmp_v != "improved":
                verdict = "reject"
                reasoning = (
                    f"Strict monotonic mode: comparison vs {comparison_reference} "
                    f"is '{cmp_v or 'unknown'}' (must be 'improved')."
                )
            elif cmp_conf < strict_round_min_compare_conf:
                verdict = "reject"
                reasoning = (
                    "Strict monotonic mode: improvement confidence too low "
                    f"({cmp_conf:.2f} < {strict_round_min_compare_conf:.2f})."
                )

        confidence = max(0.0, min(1.0, confidence))
        if verdict == "needs_more_work" and not remaining:
            remaining = ["residual_issue_needs_refinement"]

        # Update context
        ctx.verified = True
        ctx.verifier_approved = verdict in ("approve",)
        ctx.verifier_feedback = reasoning

        self.log(f"  → Verdict: {verdict} (confidence={confidence:.2f})")
        self.log(f"  → Reasoning: {reasoning[:100]}")

        # Send result to bus
        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.VERIFY_RESULT,
            content={
                "verdict": verdict,
                "confidence": confidence,
                "reasoning": reasoning,
                "remaining_issues": remaining,
                "metrics": metrics,
                "vlm_comparison": vlm_comparison,
                "vlm_quality": vlm_quality,
                "comparison_reference": comparison_reference,
            },
            case_id=ctx.case_id,
        )

        # If needs more work, notify coordinator
        if verdict == "needs_more_work" and remaining:
            self.send_message(
                recipient="coordinator",
                msg_type=MessageType.NEEDS_MORE_WORK,
                content={
                    "remaining_issues": remaining,
                    "reasoning": reasoning,
                },
                case_id=ctx.case_id,
            )

    def _compute_metrics(self, ctx: CaseContext) -> Dict[str, Any]:
        """Compute optional GT metrics for reporting (not decision driving)."""
        metrics: Dict[str, Any] = {}

        # Dice if GT available
        if ctx.gt_path and os.path.exists(ctx.gt_path):
            try:
                from ..io_utils import read_mask
                gt = read_mask(ctx.gt_path)

                dice_after, d1, d2, d3 = dice_macro(ctx.current_mask, gt)
                metrics["dice_after"] = dice_after
                metrics["dice_per_class_after"] = {"RV": d1, "Myo": d2, "LV": d3}

                if ctx.original_mask is not None:
                    dice_before, db1, db2, db3 = dice_macro(ctx.original_mask, gt)
                    metrics["dice_before"] = dice_before
                    metrics["dice_per_class_before"] = {"RV": db1, "Myo": db2, "LV": db3}
                    metrics["dice_delta"] = dice_after - dice_before
            except Exception as e:
                self.log(f"Dice computation failed: {e}")

        return metrics

    def _vlm_compare(self, ctx: CaseContext):
        """
        VLM-based visual comparison of before vs after.

        If repair_kb is available, uses the new diff-overlay approach
        (single 3-panel image). Otherwise falls back to the old
        separate-image comparison.

        Returns (comparison_result, before_overlay_path, after_overlay_path, reference_tag).
        """
        try:
            ex_cfg = self.cfg.get("executor", {})
            strict_monotonic_mode = bool(ex_cfg.get("strict_monotonic_mode", False))
            reference_mask = ctx.original_mask
            reference_tag = "original"
            if (
                strict_monotonic_mode
                and getattr(ctx, "best_intermediate_mask", None) is not None
                and not bool(getattr(ctx, "best_intermediate_is_original", True))
            ):
                reference_mask = ctx.best_intermediate_mask
                reference_tag = "best_intermediate"
            if reference_mask is None:
                reference_mask = ctx.original_mask if ctx.original_mask is not None else ctx.current_mask

            # Create individual overlays for before and after (always, for debugging)
            before_overlay = ctx.debug_img_path("_verify_before.png")
            after_overlay = ctx.debug_img_path("_verify_after.png")
            create_overlay(ctx.img_path, reference_mask, before_overlay)
            create_overlay(ctx.img_path, ctx.current_mask, after_overlay)

            # NEW: Use diff-overlay approach with repair KB
            if self.repair_kb is not None:
                diff_path = ctx.debug_img_path("_verify_diff.png")
                create_diff_overlay(
                    ctx.img_path, reference_mask, ctx.current_mask, diff_path
                )
                result = self.judge_repair_comparison(
                    diff_path, 
                    applied_ops=[op["name"] for op in ctx.applied_ops]
                )
                # Map verdict to old format for backward compat
                v = result.get("verdict", "neutral")
                mapped = {"improved": "improved", "degraded": "worse", "neutral": "same"}
                result["verdict"] = mapped.get(v, "same")
                result["reference"] = reference_tag
                return result, before_overlay, after_overlay, reference_tag

            # Fallback: old approach
            result = self.judge_visual_comparison(before_overlay, after_overlay)
            result["reference"] = reference_tag
            return result, before_overlay, after_overlay, reference_tag

        except Exception as e:
            self.log(f"VLM comparison failed: {e}")
            return {}, "", "", "unknown"

    def evaluate_batch(
        self,
        cases: Dict[str, CaseContext],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Evaluate all cases (GT-based metrics) and write reports.
        This is the final evaluation step.
        """
        self.log(f"Evaluating batch: {len(cases)} cases")

        results = []
        dice_list = []

        for case_id, ctx in cases.items():
            if not ctx.gt_path or not os.path.exists(ctx.gt_path):
                continue

            try:
                from ..io_utils import read_mask
                gt = read_mask(ctx.gt_path)
                pred = ctx.current_mask if ctx.current_mask is not None else ctx.mask

                dmac, d1, d2, d3 = dice_macro(pred, gt)
                hmac, h1, h2, h3 = hd95_macro(pred, gt)

                dice_list.append(dmac)
                ctx.dice_macro = dmac
                ctx.dice_per_class = {"RV": d1, "Myo": d2, "LV": d3}

                results.append({
                    "stem": ctx.stem,
                    "dice_macro": dmac,
                    "dice_rv": d1,
                    "dice_myo": d2,
                    "dice_lv": d3,
                    "hd95_macro": hmac,
                    "triage": ctx.triage_category,
                    "ops_applied": len(ctx.applied_ops),
                    "verified": ctx.verifier_approved,
                })
            except Exception as e:
                self.log(f"Eval failed for {ctx.stem}: {e}")

        if dice_list:
            dice_arr = np.array(dice_list)
            summary = {
                "n_cases": len(results),
                "mean_dice": float(np.mean(dice_arr)),
                "median_dice": float(np.median(dice_arr)),
                "min_dice": float(np.min(dice_arr)),
                "max_dice": float(np.max(dice_arr)),
            }
            self.log(
                f"  Batch eval: n={summary['n_cases']} "
                f"mean_dice={summary['mean_dice']:.4f} "
                f"median={summary['median_dice']:.4f}"
            )
            return summary

        return {"n_cases": 0}
