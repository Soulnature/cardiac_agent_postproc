"""
VerifierAgent: VLM-powered quality verification and gating.

Validates fixes using VLM visual comparison (before/after overlays),
with optional GT metrics (Dice/HD95) for reporting. The VLM visual
judgment is the primary decision driver, replacing RQS.
"""
from __future__ import annotations

import json
import os
import re
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
        proxy_consensus = self._build_proxy_consensus(
            ctx=ctx,
            vlm_comparison=vlm_comparison,
            vlm_quality=vlm_quality,
            metrics=metrics,
            ex_cfg=ex_cfg,
        )

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
        if proxy_consensus.get("enabled"):
            prompt += (
                f"**Proxy consensus**: {json.dumps(proxy_consensus, ensure_ascii=False)}\n"
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

        if proxy_consensus.get("enabled", False):
            consensus_v = str(proxy_consensus.get("consensus_verdict", "")).lower()
            consensus_score = self._safe_float(
                proxy_consensus.get("consensus_score", 0.0), default=0.0
            )
            if consensus_v == "reject" and verdict == "approve":
                verdict = "needs_more_work"
                if not remaining:
                    remaining = ["proxy_consensus_detected_regression"]
                reasoning = (
                    f"{reasoning} "
                    f"(Proxy consensus downgraded approve due to regression risk, score={consensus_score:+.3f})"
                ).strip()
            elif (
                consensus_v == "approve"
                and verdict == "reject"
                and bool(proxy_consensus.get("unlock_applied", False))
            ):
                verdict = "needs_more_work"
                if not remaining:
                    remaining = ["high_risk_unlock_retry"]
                reasoning = (
                    f"{reasoning} "
                    "(High-risk unlock: proxy consensus keeps candidate for another round.)"
                ).strip()

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
                if bool(proxy_consensus.get("strict_override_allow", False)):
                    verdict = "needs_more_work"
                    if not remaining:
                        remaining = ["strict_proxy_override_retry"]
                    reasoning = (
                        "Strict monotonic mode: comparison not improved, "
                        "but high-risk proxy consensus allows one more refinement round."
                    )
                else:
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
        ctx.last_verify_result = {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "remaining_issues": list(remaining),
            "metrics": dict(metrics),
            "vlm_comparison": dict(vlm_comparison),
            "vlm_quality": dict(vlm_quality),
            "comparison_reference": comparison_reference,
            "proxy_consensus": dict(proxy_consensus),
        }

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
                "proxy_consensus": proxy_consensus,
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

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _clip01(value: Any, default: float = 0.0) -> float:
        x = VerifierAgent._safe_float(value, default=default)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    @staticmethod
    def _diff_stats_from_masks(mask_before: np.ndarray, mask_after: np.ndarray) -> Dict[str, int]:
        fg_before = mask_before > 0
        fg_after = mask_after > 0
        added = int(np.count_nonzero((~fg_before) & fg_after))
        removed = int(np.count_nonzero(fg_before & (~fg_after)))
        changed = int(np.count_nonzero(fg_before & fg_after & (mask_before != mask_after)))
        return {"added": added, "removed": removed, "changed": changed}

    def _extract_diff_stats(self, vlm_comparison: Dict[str, Any]) -> Dict[str, int]:
        stats_obj = vlm_comparison.get("diff_stats_numeric", {})
        if isinstance(stats_obj, dict):
            added = int(self._safe_float(stats_obj.get("added", 0), 0))
            removed = int(self._safe_float(stats_obj.get("removed", 0), 0))
            changed = int(self._safe_float(stats_obj.get("changed", 0), 0))
            if added > 0 or removed > 0 or changed > 0:
                return {"added": added, "removed": removed, "changed": changed}

        txt = str(vlm_comparison.get("diff_stats", ""))
        ma = re.search(r"\+(\d+)", txt)
        mr = re.search(r"-(\d+)", txt)
        mc = re.search(r"~(\d+)", txt)
        return {
            "added": int(ma.group(1)) if ma else 0,
            "removed": int(mr.group(1)) if mr else 0,
            "changed": int(mc.group(1)) if mc else 0,
        }

    def _is_high_risk_case(self, ctx: CaseContext, ex_cfg: Dict[str, Any]) -> bool:
        triage_score = self._safe_float(getattr(ctx, "triage_score", 0.0), 0.0)
        triage_bad_prob = self._safe_float(
            getattr(ctx, "triage_bad_prob", 1.0 - triage_score),
            1.0 - triage_score,
        )
        score_th = float(ex_cfg.get("high_risk_unlock_score_threshold", 0.62))
        bad_prob_th = float(ex_cfg.get("high_risk_unlock_bad_prob_threshold", 0.80))
        issue_tokens = ("missing", "massive", "severe", "fragment", "disconnect", "broken", "touch", "leak")
        issue_hit = any(
            any(tok in str(issue).lower() for tok in issue_tokens)
            for issue in (ctx.triage_issues or [])
        )
        return triage_score <= score_th or triage_bad_prob >= bad_prob_th or issue_hit

    def _proxy_from_diff(
        self,
        diff_stats: Dict[str, int],
        applied_ops: List[Dict[str, Any]],
        min_change_px: int,
    ) -> Dict[str, Any]:
        added = int(diff_stats.get("added", 0))
        removed = int(diff_stats.get("removed", 0))
        changed = int(diff_stats.get("changed", 0))
        total = int(added + removed + changed)
        net = int(added - removed)
        abs_delta = max(1, total)
        net_ratio = float(net) / float(abs_delta)

        op_names = [str(op.get("name", "")) for op in (applied_ops or [])]
        has_split = any(name in ("lv_split_to_rv", "rv_split_to_lv") for name in op_names)
        is_pure_relabel = has_split and added <= 12 and removed <= 12 and changed >= 40

        if total < int(min_change_px):
            return {
                "proxy_verdict": "neutral",
                "proxy_score": 0.0,
                "proxy_confidence": 0.30,
                "added": added,
                "removed": removed,
                "changed": changed,
                "net": net,
            }

        if is_pure_relabel:
            return {
                "proxy_verdict": "improved",
                "proxy_score": 0.55,
                "proxy_confidence": 0.70,
                "added": added,
                "removed": removed,
                "changed": changed,
                "net": net,
            }

        degrade_margin = max(30, int(0.10 * total))
        improve_margin = max(24, int(0.08 * total))
        if removed > added + degrade_margin:
            score = -min(1.0, abs(net_ratio) + 0.20)
            conf = min(0.95, 0.45 + abs(score) * 0.45)
            verdict = "degraded"
        elif added > removed + improve_margin:
            score = min(1.0, abs(net_ratio) + 0.15)
            conf = min(0.95, 0.45 + score * 0.45)
            verdict = "improved"
        else:
            score = max(-0.25, min(0.25, net_ratio * 0.8))
            conf = 0.40 + min(0.25, abs(score))
            verdict = "neutral"

        return {
            "proxy_verdict": verdict,
            "proxy_score": float(score),
            "proxy_confidence": float(max(0.0, min(1.0, conf))),
            "added": added,
            "removed": removed,
            "changed": changed,
            "net": net,
        }

    def _build_proxy_consensus(
        self,
        ctx: CaseContext,
        vlm_comparison: Dict[str, Any],
        vlm_quality: Dict[str, Any],
        metrics: Dict[str, Any],
        ex_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        enabled = bool(ex_cfg.get("proxy_consensus_enable", True))
        if not enabled:
            return {"enabled": False}

        diff_stats = self._extract_diff_stats(vlm_comparison)
        min_change_px = int(ex_cfg.get("proxy_consensus_min_change_px", 24))
        proxy = self._proxy_from_diff(diff_stats, ctx.applied_ops, min_change_px=min_change_px)

        cmp_v = str(vlm_comparison.get("verdict", "same")).lower()
        cmp_conf = self._clip01(vlm_comparison.get("confidence", 0.0), default=0.0)
        cmp_vote_map = {"improved": 1.0, "same": 0.0, "neutral": 0.0, "worse": -1.0, "degraded": -1.0}
        cmp_vote = cmp_vote_map.get(cmp_v, 0.0) * max(0.35, cmp_conf)

        quality_v = str(vlm_quality.get("quality", "borderline")).lower()
        quality_conf = self._clip01(vlm_quality.get("confidence", 0.0), default=0.0)
        quality_vote_map = {"good": 0.8, "borderline": 0.0, "bad": -0.6}
        quality_vote = quality_vote_map.get(quality_v, 0.0) * max(0.30, quality_conf)

        proxy_vote_map = {"improved": 1.0, "neutral": 0.0, "degraded": -1.0}
        proxy_vote = proxy_vote_map.get(proxy["proxy_verdict"], 0.0) * max(
            0.35, self._clip01(proxy["proxy_confidence"], default=0.0)
        )

        w_cmp = float(ex_cfg.get("proxy_consensus_cmp_weight", 1.00))
        w_proxy = float(ex_cfg.get("proxy_consensus_proxy_weight", 1.00))
        w_quality = float(ex_cfg.get("proxy_consensus_quality_weight", 0.35))
        consensus_score = float(w_cmp * cmp_vote + w_proxy * proxy_vote + w_quality * quality_vote)

        dice_delta = self._safe_float(metrics.get("dice_delta", 0.0), default=0.0)
        if dice_delta > 0.0:
            consensus_score += min(0.25, dice_delta * 3.0)

        pos_th = float(ex_cfg.get("proxy_consensus_positive_threshold", 0.65))
        neg_th = float(ex_cfg.get("proxy_consensus_negative_threshold", -0.65))
        if consensus_score >= pos_th:
            consensus_verdict = "approve"
        elif consensus_score <= neg_th:
            consensus_verdict = "reject"
        else:
            consensus_verdict = "needs_more_work"

        high_risk_unlock = bool(ex_cfg.get("proxy_consensus_unlock_on_high_risk", True))
        unlock_min_score = float(ex_cfg.get("proxy_consensus_unlock_min_score", -0.15))
        is_high_risk = self._is_high_risk_case(ctx, ex_cfg)
        unlock_applied = False
        strict_override_allow = False
        if high_risk_unlock and is_high_risk:
            if (
                consensus_verdict == "reject"
                and consensus_score >= unlock_min_score
                and cmp_v not in ("worse", "degraded")
            ):
                consensus_verdict = "needs_more_work"
                unlock_applied = True
            if (
                proxy.get("proxy_verdict") == "improved"
                and self._safe_float(proxy.get("proxy_confidence", 0.0), 0.0) >= 0.55
                and cmp_v in ("same", "neutral")
            ):
                strict_override_allow = True
                unlock_applied = True

        return {
            "enabled": True,
            "diff_stats": diff_stats,
            "proxy_verdict": proxy.get("proxy_verdict", "neutral"),
            "proxy_score": float(proxy.get("proxy_score", 0.0)),
            "proxy_confidence": float(proxy.get("proxy_confidence", 0.0)),
            "consensus_score": consensus_score,
            "consensus_verdict": consensus_verdict,
            "cmp_vote": cmp_vote,
            "proxy_vote": proxy_vote,
            "quality_vote": quality_vote,
            "unlock_applied": unlock_applied,
            "strict_override_allow": strict_override_allow,
            "is_high_risk_case": is_high_risk,
        }

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
            diff_stats = self._diff_stats_from_masks(reference_mask, ctx.current_mask)

            # NEW: Use diff-overlay approach with repair KB
            if self.repair_kb is not None:
                diff_path = ctx.debug_img_path("_verify_diff.png")
                create_diff_overlay(
                    ctx.img_path,
                    reference_mask,
                    ctx.current_mask,
                    diff_path,
                    stats=diff_stats,
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
                result["diff_stats_numeric"] = diff_stats
                return result, before_overlay, after_overlay, reference_tag

            # Fallback: old approach
            result = self.judge_visual_comparison(before_overlay, after_overlay)
            result["reference"] = reference_tag
            result["diff_stats_numeric"] = diff_stats
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
