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

_CANONICAL_REMAINING_ISSUES = {
    "disconnected_myo",
    "rv_lv_touching",
    "fragmented_lv",
    "fragmented_rv",
    "holes",
    "valve_leak",
    "myo_thickness_var",
    "missing_myo",
    "missing_lv",
    "missing_rv",
}


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
            "2. Structured Top-K retrieved knowledge evidence IDs and metadata\n"
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
        return self._prompt_with_fallback("verifier_system.txt", fallback)

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
        strict_no_gt_online = self._strict_no_gt_online()

        vlm_comparison: Dict[str, Any] = {}
        before_overlay = ""
        after_overlay = ""
        comparison_reference = "original"

        if ctx.img_path and ctx.original_mask is not None:
            (
                vlm_comparison,
                before_overlay,
                after_overlay,
                comparison_reference,
            ) = self._vlm_compare(ctx)

        # Step 2: VLM quality judgment on the AFTER image
        vlm_quality: Dict[str, Any] = {}
        if after_overlay and self.visual_kb:
            vlm_quality = self.judge_visual_quality(
                overlay_path=after_overlay,
                n_refs=1,
                view_type=ctx.view_type,
                seed=f"verify:{ctx.stem}",
            )

        # Step 3: Optional GT metrics (for reporting, not decision driving)
        metrics = self._compute_metrics(ctx)
        proxy_consensus = self._build_proxy_consensus(
            ctx=ctx,
            vlm_comparison=vlm_comparison,
            vlm_quality=vlm_quality,
            metrics=metrics,
            ex_cfg=ex_cfg,
        )

        # Step 3b: Anatomy score signal (no-GT, view-specific area/shape improvement proxy)
        anatomy_signal = self._compute_anatomy_signal(ctx)
        anatomy_llm_payload: Dict[str, Any] = {}
        if self._anatomy_llm_scope_enabled("verifier"):
            anatomy_llm_payload = {
                "baseline": dict(getattr(ctx, "anatomy_baseline_summary", {}) or {}),
                "latest": dict(getattr(ctx, "anatomy_latest_summary", {}) or {}),
            }
            if isinstance(anatomy_signal, dict) and anatomy_signal.get("enabled"):
                anatomy_llm_payload["current_signal"] = dict(anatomy_signal)
            anatomy_llm_payload = {k: v for k, v in anatomy_llm_payload.items() if v}

        # Step 4: LLM reasoning text (reason only); final verdict is score-thresholded.
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
        if anatomy_signal.get("enabled"):
            prompt += (
                f"**Anatomy score (no-GT)**: before={anatomy_signal['score_before']:.1f} "
                f"after={anatomy_signal['score_after']:.1f} "
                f"Δ={anatomy_signal['delta']:+.1f}/100\n"
            )
            if anatomy_signal.get("violations_after"):
                prompt += (
                    f"**Anatomy violations after repair**: "
                    f"{anatomy_signal['violations_after']}\n"
                )
        if anatomy_llm_payload:
            prompt += (
                f"**Anatomy structured summary**: "
                f"{json.dumps(anatomy_llm_payload, ensure_ascii=False)}\n"
            )

        if (not strict_no_gt_online) and metrics.get("dice_before") is not None:
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
        llm_reasoning = str(
            result.get("reasoning", result.get("reason", ""))
        ).strip()
        llm_remaining = result.get("remaining_issues", [])
        if not isinstance(llm_remaining, list):
            llm_remaining = [str(llm_remaining)]
        llm_remaining = self._normalize_remaining_issues(llm_remaining)

        online_score = self._compute_online_score(
            vlm_comparison=vlm_comparison,
            vlm_quality=vlm_quality,
            proxy_consensus=proxy_consensus,
            anatomy_signal=anatomy_signal,
        )
        reject_th, approve_th = self._resolve_score_thresholds(ctx.view_type)
        verdict = self._map_score_to_verdict(online_score, reject_th, approve_th)
        remaining = list(llm_remaining if verdict == "needs_more_work" else [])
        confidence = max(
            self._clip01(result.get("confidence", 0.0), default=0.0),
            self._clip01(vlm_comparison.get("confidence", 0.0), default=0.0),
        )
        if confidence <= 0.0:
            confidence = self._clip01(vlm_quality.get("confidence", 0.0), default=0.0)

        reasoning = llm_reasoning or "Threshold decision from calibrated online score."
        reasoning = (
            f"{reasoning} "
            f"(Score policy: score={online_score:.1f}, "
            f"reject<{reject_th:.1f}, approve>={approve_th:.1f})"
        ).strip()

        if strict_monotonic_mode:
            cmp_v = str(vlm_comparison.get("verdict", "")).lower()
            cmp_conf = float(vlm_comparison.get("confidence", 0.0) or 0.0)
            strict_reason = ""
            if not vlm_comparison:
                verdict = "reject"
                strict_reason = (
                    "Strict monotonic mode: missing VLM comparison signal; "
                    "rejecting round for safety."
                )
            elif cmp_v != "improved":
                if bool(proxy_consensus.get("strict_override_allow", False)):
                    verdict = "needs_more_work"
                    if not remaining:
                        remaining = ["strict_proxy_override_retry"]
                    strict_reason = (
                        "Strict monotonic mode: comparison not improved, "
                        "but high-risk proxy consensus allows one more refinement round."
                    )
                else:
                    verdict = "reject"
                    strict_reason = (
                        f"Strict monotonic mode: comparison vs {comparison_reference} "
                        f"is '{cmp_v or 'unknown'}' (must be 'improved')."
                    )
            elif cmp_conf < strict_round_min_compare_conf:
                verdict = "reject"
                strict_reason = (
                    "Strict monotonic mode: improvement confidence too low "
                    f"({cmp_conf:.2f} < {strict_round_min_compare_conf:.2f})."
                )
            if strict_reason:
                reasoning = f"{reasoning} ({strict_reason})".strip()

        verdict, reasoning, remaining = self._apply_anatomy_violation_policy(
            ctx=ctx,
            verdict=verdict,
            reasoning=reasoning,
            remaining=remaining,
            anatomy_signal=anatomy_signal,
            vlm_comparison=vlm_comparison,
        )

        confidence = max(0.0, min(1.0, confidence))
        if verdict == "needs_more_work" and not remaining:
            remaining = ["residual_issue_needs_refinement"]

        matched_example_ids = self._collect_matched_example_ids(vlm_comparison, vlm_quality)

        # Update context
        ctx.verified = True
        ctx.verifier_approved = verdict in ("approve",)
        ctx.verifier_feedback = reasoning
        ctx.last_verify_result = {
            "verdict": verdict,
            "score": online_score,
            "confidence": confidence,
            "reasoning": reasoning,
            "reason": reasoning,
            "remaining_issues": list(remaining),
            "metrics": dict(metrics),
            "offline_oracle_metrics": dict(metrics),
            "vlm_comparison": dict(vlm_comparison),
            "vlm_quality": dict(vlm_quality),
            "comparison_reference": comparison_reference,
            "proxy_consensus": dict(proxy_consensus),
            "anatomy_signal": dict(anatomy_signal),
            "online_verdict": verdict,
            "online_score": float(online_score),
            "online_threshold_reject": float(reject_th),
            "online_threshold_approve": float(approve_th),
            "online_matched_example_ids": list(matched_example_ids),
            "strict_no_gt_online": bool(strict_no_gt_online),
        }

        self._write_verifier_payload(
            ctx=ctx,
            online_prompt=prompt,
            online_data={
                "vlm_comparison": dict(vlm_comparison),
                "vlm_quality": dict(vlm_quality),
                "proxy_consensus": dict(proxy_consensus),
                "anatomy_signal": dict(anatomy_signal),
                "online_score": float(online_score),
                "online_threshold_reject": float(reject_th),
                "online_threshold_approve": float(approve_th),
                "online_verdict": verdict,
                "online_matched_example_ids": list(matched_example_ids),
            },
            offline_oracle_metrics=dict(metrics),
        )

        self.log(
            f"  → Verdict: {verdict} "
            f"(score={online_score:.1f}, confidence={confidence:.2f})"
        )
        self.log(f"  → Reasoning: {reasoning[:100]}")

        # Send result to bus
        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.VERIFY_RESULT,
            content={
                "verdict": verdict,
                "score": online_score,
                "confidence": confidence,
                "reasoning": reasoning,
                "reason": reasoning,
                "remaining_issues": remaining,
                "metrics": metrics,
                "offline_oracle_metrics": metrics,
                "vlm_comparison": vlm_comparison,
                "vlm_quality": vlm_quality,
                "comparison_reference": comparison_reference,
                "proxy_consensus": proxy_consensus,
                "anatomy_signal": dict(anatomy_signal),
                "online_verdict": verdict,
                "online_score": float(online_score),
                "online_threshold_reject": float(reject_th),
                "online_threshold_approve": float(approve_th),
                "online_matched_example_ids": list(matched_example_ids),
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
    def _normalize_remaining_issue(issue: Any) -> Optional[str]:
        if issue is None:
            return None
        if isinstance(issue, dict):
            issue = issue.get("issue", issue.get("name", issue.get("label", "")))
        text = str(issue).strip().lower()
        if not text:
            return None
        compact = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        if not compact:
            return None
        for canon in sorted(_CANONICAL_REMAINING_ISSUES):
            if (
                compact == canon
                or compact.startswith(f"{canon}_")
                or compact.endswith(f"_{canon}")
                or f"_{canon}_" in f"_{compact}_"
            ):
                return canon
        if ("touch" in compact or "contact" in compact) and "rv" in compact and "lv" in compact:
            return "rv_lv_touching"
        if any(tok in compact for tok in ("discmyo", "ring_break", "myo_gap", "broken_ring")):
            return "disconnected_myo"
        if "hole" in compact:
            return "holes"
        if "valve" in compact and "leak" in compact:
            return "valve_leak"
        if "missing" in compact and "myo" in compact:
            return "missing_myo"
        if "missing" in compact and "lv" in compact:
            return "missing_lv"
        if "missing" in compact and "rv" in compact:
            return "missing_rv"
        if "fragment" in compact and "lv" in compact:
            return "fragmented_lv"
        if "fragment" in compact and "rv" in compact:
            return "fragmented_rv"
        if "thickness" in compact and "myo" in compact:
            return "myo_thickness_var"
        return None

    @classmethod
    def _normalize_remaining_issues(cls, items: List[Any]) -> List[str]:
        out: List[str] = []
        for raw in items:
            canon = cls._normalize_remaining_issue(raw)
            if canon and canon not in out:
                out.append(canon)
        return out

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

    def _strict_no_gt_online(self) -> bool:
        ma_cfg = self.cfg.get("multi_agent", {}) or {}
        return bool(ma_cfg.get("strict_no_gt_online", False))

    def _anatomy_llm_scope_enabled(self, scope_name: str) -> bool:
        ma_cfg = self.cfg.get("multi_agent", {}) or {}
        if not bool(ma_cfg.get("anatomy_llm_enable", True)):
            return False
        scope_raw = ma_cfg.get(
            "anatomy_llm_scope", ["diagnosis", "planner", "verifier"]
        )
        if isinstance(scope_raw, list):
            scope_set = {str(x).strip().lower() for x in scope_raw}
        elif isinstance(scope_raw, str):
            scope_set = {
                str(tok).strip().lower()
                for tok in re.split(r"[,\s]+", scope_raw)
                if str(tok).strip()
            }
        else:
            scope_set = {"diagnosis", "planner", "verifier"}
        return str(scope_name).strip().lower() in scope_set

    def _apply_anatomy_violation_policy(
        self,
        ctx: CaseContext,
        verdict: str,
        reasoning: str,
        remaining: List[str],
        anatomy_signal: Dict[str, Any],
        vlm_comparison: Dict[str, Any],
    ) -> tuple[str, str, List[str]]:
        """
        Deterministic safety policy:
        if post-repair hard anatomy violations increase and VLM does not show
        strong confident improvement, downgrade to reject/needs_more_work.
        """
        verifier_cfg = self.cfg.get("verifier", {}) or {}
        policy_enabled = bool(verifier_cfg.get("anatomy_hard_violation_reject", True))
        if not policy_enabled:
            return verdict, reasoning, remaining
        if not isinstance(anatomy_signal, dict) or not anatomy_signal.get("enabled"):
            return verdict, reasoning, remaining

        def _norm_view_token(v: Any) -> str:
            tok = re.sub(r"[^a-z0-9]", "", str(v or "").strip().lower())
            if tok in {"2c", "2ch"}:
                return "2ch"
            if tok in {"3c", "3ch"}:
                return "3ch"
            if tok in {"4c", "4ch"}:
                return "4ch"
            return tok

        baseline_summary = dict(getattr(ctx, "anatomy_baseline_summary", {}) or {})
        view_token = _norm_view_token(ctx.view_type)
        base_view_token = _norm_view_token(baseline_summary.get("view", ""))
        if base_view_token and view_token and base_view_token != view_token:
            return verdict, reasoning, remaining

        # Delta-based hard downgrade:
        # even without hard-violation tags, reject when anatomy score regresses
        # beyond threshold.
        delta_gate_enabled = bool(verifier_cfg.get("anatomy_delta_reject_enable", True))
        delta_reject_th = self._safe_float(
            verifier_cfg.get("anatomy_delta_reject_threshold", -1.0),
            -1.0,
        )
        delta_v = self._safe_float(anatomy_signal.get("delta", 0.0), 0.0)
        if delta_gate_enabled and np.isfinite(delta_v) and delta_v < delta_reject_th:
            if verdict in ("approve", "needs_more_work"):
                verdict = "reject"
            if not remaining:
                remaining = ["anatomy_score_regression"]
            delta_reason = (
                "Anatomy delta policy triggered: "
                f"delta={delta_v:+.2f} < threshold={delta_reject_th:+.2f}."
            )
            reasoning = f"{reasoning} ({delta_reason})".strip()

        before_v = {
            str(x).strip().lower()
            for x in (baseline_summary.get("hard_violations") or [])
            if str(x).strip()
        }
        after_v = {
            str(x).strip().lower()
            for x in (anatomy_signal.get("violations_after") or [])
            if str(x).strip()
        }
        if not after_v:
            return verdict, reasoning, remaining

        added = sorted(x for x in after_v if x not in before_v)
        count_increased = len(after_v) > len(before_v)
        if not added and not count_increased:
            return verdict, reasoning, remaining

        cmp_v = str(vlm_comparison.get("verdict", "")).lower()
        cmp_conf = self._safe_float(vlm_comparison.get("confidence", 0.0), 0.0)
        cmp_score = self._safe_float(vlm_comparison.get("score", 50.0), 50.0)
        strong_improve = (
            cmp_v == "improved"
            and cmp_conf >= 0.80
            and cmp_score >= 70.0
        )
        if strong_improve:
            return verdict, reasoning, remaining

        if verdict == "approve":
            verdict = "reject"
        elif verdict == "needs_more_work":
            verdict = "reject"
        if not remaining:
            remaining = ["anatomy_hard_violation_regression"]

        anatomy_reason = (
            "Anatomy hard-violation policy triggered: post-repair violations "
            f"increased (before={sorted(before_v)}, after={sorted(after_v)}, added={added}) "
            "without strong confident visual improvement."
        )
        reasoning = f"{reasoning} ({anatomy_reason})".strip()
        return verdict, reasoning, remaining

    @staticmethod
    def _normalize_view_key(view_type: Optional[str]) -> str:
        token = re.sub(r"[^a-z0-9]", "", str(view_type or "").strip().lower())
        if token in {"2c", "2ch"}:
            return "2CH"
        if token in {"3c", "3ch"}:
            return "3CH"
        if token in {"4c", "4ch"}:
            return "4CH"
        return token.upper() if token else "ANY"

    def _resolve_score_thresholds(self, view_type: Optional[str]) -> tuple[float, float]:
        base_cfg = self.cfg.get("verifier", {}) or {}
        ag_cfg = (
            (self.cfg.get("agents", {}) or {}).get("verifier", {}) or {}
        )
        base_thr = base_cfg.get("thresholds", {}) if isinstance(base_cfg, dict) else {}
        ag_thr = ag_cfg.get("thresholds", {}) if isinstance(ag_cfg, dict) else {}

        reject_th = float(base_thr.get("reject", 45.0))
        approve_th = float(base_thr.get("approve", 65.0))
        if isinstance(ag_thr, dict):
            if "reject" in ag_thr:
                reject_th = float(ag_thr.get("reject"))
            if "approve" in ag_thr:
                approve_th = float(ag_thr.get("approve"))

        view_key = self._normalize_view_key(view_type)
        per_view_base = base_thr.get("per_view", {}) if isinstance(base_thr, dict) else {}
        per_view_ag = ag_thr.get("per_view", {}) if isinstance(ag_thr, dict) else {}
        view_cfg = {}
        if isinstance(per_view_base, dict):
            view_cfg.update(per_view_base.get(view_key, {}) or {})
            view_cfg.update(per_view_base.get(view_key.lower(), {}) or {})
        if isinstance(per_view_ag, dict):
            view_cfg.update(per_view_ag.get(view_key, {}) or {})
            view_cfg.update(per_view_ag.get(view_key.lower(), {}) or {})
        if "reject" in view_cfg:
            reject_th = float(view_cfg["reject"])
        if "approve" in view_cfg:
            approve_th = float(view_cfg["approve"])
        if approve_th < reject_th:
            approve_th = reject_th
        return float(reject_th), float(approve_th)

    @staticmethod
    def _map_score_to_verdict(score: float, reject_th: float, approve_th: float) -> str:
        if score < reject_th:
            return "reject"
        if score < approve_th:
            return "needs_more_work"
        return "approve"

    def _compute_anatomy_signal(self, ctx: "CaseContext") -> Dict[str, Any]:
        """
        Compute anatomy score for the current (post-repair) mask and compare
        it to the baseline score stored in ctx.anatomy_score_before.

        Returns a dict with enabled=True and score_100 (0-100) if available,
        or enabled=False when anatomy stats are missing / computation fails.
        """
        if (
            getattr(ctx, "anatomy_stats", None) is None
            or getattr(ctx, "anatomy_score_before", -1.0) < 0
            or ctx.current_mask is None
            or ctx.image is None
        ):
            return {"enabled": False}
        try:
            from ..anatomy_score import (
                compute_anatomy_score,
                build_anatomy_llm_summary,
                _normalize_view,
            )
            view = _normalize_view(ctx.view_type or "")
            stats = ctx.anatomy_stats.get(view)
            if stats is None:
                return {"enabled": False}
            score_after, detail = compute_anatomy_score(ctx.current_mask, ctx.image, view, stats)
            ma_cfg = self.cfg.get("multi_agent", {}) or {}
            top_k = int(ma_cfg.get("anatomy_llm_top_k", 5))
            summary_after = build_anatomy_llm_summary(
                score=score_after,
                detail=detail,
                view_type=view,
                top_k=max(1, top_k),
            )
            ctx.anatomy_latest_summary = dict(summary_after)
            delta = float(score_after) - float(ctx.anatomy_score_before)
            # Map delta to 0-100 signal: delta=0 → 50, ±20pts → 0/100 (clipped)
            anatomy_score_100 = max(0.0, min(100.0, 50.0 + delta * 2.5))
            violations_after = detail.get("hard_violations", [])
            self.log(
                f"  Anatomy: before={ctx.anatomy_score_before:.1f} "
                f"after={score_after:.1f} Δ={delta:+.1f} → {anatomy_score_100:.0f}/100"
            )
            return {
                "enabled": True,
                "score_before": float(ctx.anatomy_score_before),
                "score_after": float(score_after),
                "delta": float(delta),
                "score_100": float(anatomy_score_100),
                "violations_after": violations_after,
                "summary_after": summary_after,
            }
        except Exception as e:
            self.log(f"  Anatomy signal error: {e}")
            return {"enabled": False}

    def _compute_online_score(
        self,
        vlm_comparison: Dict[str, Any],
        vlm_quality: Dict[str, Any],
        proxy_consensus: Dict[str, Any],
        anatomy_signal: Optional[Dict[str, Any]] = None,
    ) -> float:
        cmp_score = self._safe_float(vlm_comparison.get("score", 50.0), 50.0)
        cmp_conf = self._clip01(vlm_comparison.get("confidence", 0.0), default=0.0)
        cmp_v = str(vlm_comparison.get("verdict", "")).lower()
        if not np.isfinite(cmp_score):
            cmp_map = {"improved": 80.0, "same": 50.0, "neutral": 50.0, "worse": 20.0, "degraded": 20.0}
            cmp_score = cmp_map.get(cmp_v, 50.0)

        quality_score = self._safe_float(vlm_quality.get("score", 5.0), 5.0) * 10.0
        quality_score = max(0.0, min(100.0, quality_score))
        quality_conf = self._clip01(vlm_quality.get("confidence", 0.0), default=0.0)

        proxy_score = self._safe_float(proxy_consensus.get("proxy_score", 0.0), 0.0)
        proxy_score_100 = max(0.0, min(100.0, 50.0 + 50.0 * proxy_score))
        proxy_conf = self._clip01(proxy_consensus.get("proxy_confidence", 0.0), default=0.0)

        # Base weights
        w_cmp = 0.70 + 0.10 * cmp_conf
        w_quality = 0.15 + 0.05 * quality_conf
        w_proxy = 0.15 + 0.05 * proxy_conf

        # Anatomy signal: no-GT area/shape quality delta (most reliable proxy for Dice)
        w_anatomy = 0.0
        anatomy_score_100 = 50.0
        if anatomy_signal and anatomy_signal.get("enabled"):
            anatomy_score_100 = float(anatomy_signal.get("score_100", 50.0))
            w_anatomy = 0.20  # fixed weight — anatomy is our best GT-free Dice proxy

        w_sum = max(1e-6, w_cmp + w_quality + w_proxy + w_anatomy)
        score = (
            w_cmp * cmp_score
            + w_quality * quality_score
            + w_proxy * proxy_score_100
            + w_anatomy * anatomy_score_100
        ) / w_sum
        return float(max(0.0, min(100.0, score)))

    @staticmethod
    def _collect_matched_example_ids(
        vlm_comparison: Dict[str, Any],
        vlm_quality: Dict[str, Any],
    ) -> List[str]:
        out: List[str] = []
        for obj in (vlm_comparison, vlm_quality):
            ids = obj.get("matched_example_ids", [])
            if not isinstance(ids, list):
                continue
            for eid in ids:
                eid_s = str(eid).strip()
                if eid_s and eid_s not in out:
                    out.append(eid_s)
        return out

    def _write_verifier_payload(
        self,
        ctx: CaseContext,
        online_prompt: str,
        online_data: Dict[str, Any],
        offline_oracle_metrics: Dict[str, Any],
    ) -> None:
        payload = {
            "case_id": ctx.case_id,
            "stem": ctx.stem,
            "strict_no_gt_online": bool(self._strict_no_gt_online()),
            "anatomy_baseline_summary": dict(getattr(ctx, "anatomy_baseline_summary", {}) or {}),
            "anatomy_latest_summary": dict(getattr(ctx, "anatomy_latest_summary", {}) or {}),
            "online_prompt": online_prompt,
            "online_data": online_data,
            "offline_oracle_metrics": offline_oracle_metrics,
        }
        out_path = ctx.debug_img_path("_verifier_payload.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log(f"Failed to write verifier payload: {e}")

    @staticmethod
    def _normalize_defect_tag(tag: Any) -> str:
        txt = re.sub(r"[^a-z0-9]+", "_", str(tag or "").strip().lower())
        txt = re.sub(r"_+", "_", txt).strip("_")
        return txt

    def _collect_defect_tags(self, ctx: CaseContext) -> List[str]:
        tags: List[str] = []
        for issue in (ctx.triage_issues or []):
            t = self._normalize_defect_tag(issue)
            if t and t not in tags:
                tags.append(t)
        for diag in (ctx.diagnoses or []):
            if not isinstance(diag, dict):
                continue
            for key in ("issue", "issue_type", "label", "name"):
                if key not in diag:
                    continue
                t = self._normalize_defect_tag(diag.get(key))
                if t and t not in tags:
                    tags.append(t)
        for op in (ctx.applied_ops or []):
            t = self._normalize_defect_tag(op.get("name", ""))
            if t and t not in tags:
                tags.append(t)
        return tags

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

        if not self._strict_no_gt_online():
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
                verifier_cfg = self.cfg.get("verifier", {}) or {}
                agent_verifier_cfg = (
                    (self.cfg.get("agents", {}) or {}).get("verifier", {}) or {}
                )
                knowledge_top_k = int(
                    agent_verifier_cfg.get(
                        "knowledge_top_k",
                        verifier_cfg.get("knowledge_top_k", 2),
                    )
                )
                defect_tags = self._collect_defect_tags(ctx)
                strict_no_gt_online = self._strict_no_gt_online()
                result = self.judge_repair_comparison(
                    diff_path,
                    applied_ops=[op["name"] for op in ctx.applied_ops],
                    view_type=ctx.view_type,
                    defect_tags=defect_tags,
                    top_k_refs=max(1, knowledge_top_k),
                    use_confidence_prior=(not strict_no_gt_online),
                )
                # Map verdict to old format for backward compat
                v = result.get("verdict", "neutral")
                mapped = {"improved": "improved", "degraded": "worse", "neutral": "same"}
                result["raw_verdict"] = v
                result["verdict"] = mapped.get(v, "same")
                if "score" not in result:
                    cmp_conf = self._safe_float(result.get("confidence", 0.0), 0.0)
                    if result["verdict"] == "improved":
                        result["score"] = 50.0 + 50.0 * max(0.0, min(1.0, cmp_conf))
                    elif result["verdict"] in ("worse", "degraded"):
                        result["score"] = 50.0 - 50.0 * max(0.0, min(1.0, cmp_conf))
                    else:
                        result["score"] = 50.0
                result["reference"] = reference_tag
                result["diff_stats_numeric"] = diff_stats
                matched_ids = result.get("matched_example_ids", [])
                if not isinstance(matched_ids, list):
                    matched_ids = []
                trace_item = {
                    "stage": "repair_verification",
                    "comparison_reference": reference_tag,
                    "matched_example_ids": [str(x) for x in matched_ids],
                    "defect_tags": defect_tags,
                }
                if not hasattr(ctx, "knowledge_trace") or not isinstance(ctx.knowledge_trace, list):
                    ctx.knowledge_trace = []
                ctx.knowledge_trace.append(trace_item)
                return result, before_overlay, after_overlay, reference_tag

            # Fallback: old approach
            result = self.judge_visual_comparison(before_overlay, after_overlay)
            result["reference"] = reference_tag
            result["diff_stats_numeric"] = diff_stats
            if "matched_example_ids" not in result:
                result["matched_example_ids"] = []
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
