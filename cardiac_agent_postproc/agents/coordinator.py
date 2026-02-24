"""
CoordinatorAgent: Central coordinator for the multi-agent pipeline.

Manages the case-by-case workflow: Triage → Diagnosis → Plan → Execute → Verify,
with dynamic re-routing when the Verifier rejects a fix.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from .message_bus import CaseContext, MessageBus, MessageType


class CoordinatorAgent(BaseAgent):
    """
    LLM-powered coordinator that orchestrates the multi-agent pipeline.

    For each case:
    1. Dispatches to TriageAgent
    2. If needs_fix: Diagnosis → Plan → Execute → Verify loop (up to N rounds)
    3. Handles NEEDS_MORE_WORK and CHALLENGE flows
    4. Cross-case batch summary at the end
    """

    @property
    def name(self) -> str:
        return "coordinator"

    @property
    def system_prompt(self) -> str:
        fallback = (
            "You are the Coordinator Agent overseeing a cardiac MRI segmentation repair pipeline.\n\n"
            "You manage a team of specialist agents:\n"
            "- **TriageAgent**: quality classification (good / needs_fix)\n"
            "- **DiagnosisAgent**: spatial defect identification\n"
            "- **PlannerAgent**: repair operation sequencing\n"
            "- **ExecutorAgent**: operation execution + RQS monitoring\n"
            "- **VerifierAgent**: quality gating + approval/rejection\n\n"
            "Your decisions:\n"
            "1. Whether to trust Triage or challenge it\n"
            "2. Whether a rejected fix deserves another round\n"
            "3. When to give up on a difficult case\n"
            "4. Cross-case strategy adjustments\n\n"
            "Respond with JSON:\n"
            '{"action": "proceed" | "retry" | "skip" | "challenge_triage", '
            '"reasoning": "..."}'
        )
        return self._prompt_with_fallback("coordinator_system.txt", fallback)

    def process_case(self, ctx: CaseContext) -> None:
        """
        Not used directly — the Coordinator calls other agents.
        See orchestrate_case() instead.
        """
        pass

    def orchestrate_case(
        self,
        ctx: CaseContext,
        agents: Dict[str, BaseAgent],
        max_rounds: int = 3,
    ) -> str:
        """
        Run the full pipeline for a single case.
        Returns the final verdict: 'good', 'approved', 'rejected', 'gave_up'.
        """
        self.log(f"━━ Case {ctx.stem} ━━")
        ex_cfg = self.cfg.get("executor", {})
        strict_monotonic_mode = bool(ex_cfg.get("strict_monotonic_mode", False))
        try:
            # Step 1: Triage
            triage = agents["triage"]
            triage.process_case(ctx)
        except Exception as e:
            self.log(f"CRITICAL: Triage failed with error: {e}")
            import traceback
            self.log(traceback.format_exc())
            return "error"

        # Fast path for high-confidence good masks.
        fast_path_threshold = float(
            self.cfg.get("multi_agent", {}).get("fast_path_threshold", 1.01)
        )
        if (
            ctx.triage_score >= fast_path_threshold
            and not self._has_high_risk_issue(ctx.triage_issues)
        ):
            self.log(
                f"  Fast-path skip (score={ctx.triage_score:.3f} >= {fast_path_threshold:.3f})"
            )
            return "good"

        self.log(f"  [coordinator] triage category: {ctx.triage_category}, score: {ctx.triage_score:.3f}")
        if ctx.triage_category == "good":
            self.log(f"  Triaged as GOOD — skip optimization")
            return "good"

        # Load anatomy stats and compute no-GT baseline score for the original mask.
        self._init_anatomy_baseline(ctx)

        # Initialize best-intermediate baseline (original prediction).
        if ctx.original_mask is not None and ctx.best_intermediate_mask is None:
            ctx.best_intermediate_mask = ctx.original_mask.copy()
            ctx.best_intermediate_score = 0.0
            ctx.best_intermediate_round = 0
            ctx.best_intermediate_verdict = "baseline_original"
            ctx.best_intermediate_reason = "initial prediction baseline"
            ctx.best_intermediate_ops = []
            ctx.best_intermediate_is_original = True
            ctx.best_intermediate_online_score = float("nan")
            try:
                ctx.best_intermediate_anatomy_score = float(ctx.anatomy_score_before)
            except Exception:
                ctx.best_intermediate_anatomy_score = float("nan")
        if ctx.original_mask is not None and ctx.best_anatomy_intermediate_mask is None:
            ctx.best_anatomy_intermediate_mask = ctx.original_mask.copy()
            try:
                baseline_anatomy = float(ctx.anatomy_score_before)
            except Exception:
                baseline_anatomy = float("nan")
            if self._is_finite(baseline_anatomy) and baseline_anatomy < 0.0:
                baseline_anatomy = float("nan")
            ctx.best_anatomy_intermediate_score = baseline_anatomy
            ctx.best_anatomy_intermediate_delta = 0.0
            ctx.best_anatomy_intermediate_round = 0
            ctx.best_anatomy_intermediate_verdict = "baseline_original"
            ctx.best_anatomy_intermediate_reason = "initial prediction baseline"
            ctx.best_anatomy_intermediate_ops = []
            ctx.best_anatomy_intermediate_is_original = True
            ctx.best_anatomy_intermediate_online_score = float("nan")

        # Step 2-5: Diagnosis → Plan → Execute → Verify loop
        max_rounds = max(1, int(max_rounds))
        replan_on_failure = bool(
            self.cfg.get("multi_agent", {}).get("in_round_replan_on_failure", True)
        )
        max_replan_attempts = max(
            0,
            int(
                self.cfg.get("multi_agent", {}).get(
                    "max_replan_attempts_per_round", 1
                )
            ),
        )
        hard_split_ops = {"lv_split_to_rv", "rv_split_to_lv"}
        post_split_cleanup_pending = False
        post_split_cleanup_consumed = False
        round_num = 0
        while round_num < max_rounds or post_split_cleanup_pending:
            if post_split_cleanup_pending:
                self.log(
                    "  [PostSplit] Running forced post-process round after split relabel"
                )
                post_split_cleanup_pending = False
                post_split_cleanup_consumed = True

            round_total = max_rounds + (
                1 if (post_split_cleanup_consumed or post_split_cleanup_pending) else 0
            )
            self.log(f"  ── Round {round_num + 1}/{round_total} ──")

            # Diagnosis
            diagnosis = agents["diagnosis"]
            diagnosis.process_case(ctx)

            if not ctx.diagnoses:
                self.log(f"  No diagnoses — stop")
                break

            # Planning
            planner = agents["planner"]
            planner.process_case(ctx)

            # Execution
            executor = agents["executor"]
            prev_op_count = len(ctx.applied_ops or [])
            executor.process_case(ctx)
            new_ops = (ctx.applied_ops or [])[prev_op_count:]
            split_relabel_applied = any(
                str(op.get("name", "")).strip() in hard_split_ops for op in new_ops
            )

            # In-round adaptive recovery:
            # if executor reports hard-gate failures, ask planner for a revision and retry.
            if replan_on_failure and max_replan_attempts > 0:
                for attempt in range(max_replan_attempts):
                    failed_msgs = planner.receive_messages(
                        msg_type=MessageType.OP_FAILED,
                        case_id=ctx.case_id,
                    )
                    if not failed_msgs:
                        break
                    failed_payload = failed_msgs[-1].content or {}
                    failed_ops = failed_payload.get("failed_ops", [])
                    if not failed_ops:
                        break
                    self.log(
                        "  [Replan] Executor reported failed ops; "
                        f"attempt {attempt + 1}/{max_replan_attempts}"
                    )
                    planner.revise_plan(
                        ctx,
                        {
                            "stage": "executor_hard_gate",
                            "round": int(round_num + 1),
                            "failed_ops": failed_ops,
                            "applied_ops": [op.get("name", "") for op in (ctx.applied_ops or [])],
                        },
                    )
                    prev_retry_count = len(ctx.applied_ops or [])
                    executor.process_case(ctx)
                    retry_new_ops = (ctx.applied_ops or [])[prev_retry_count:]
                    if retry_new_ops:
                        split_relabel_applied = split_relabel_applied or any(
                            str(op.get("name", "")).strip() in hard_split_ops
                            for op in retry_new_ops
                        )
                    if not retry_new_ops:
                        # Avoid extra loops when revision still cannot pass hard gates.
                        break

            # Verification
            verifier = agents["verifier"]
            verifier.process_case(ctx)

            ctx.rounds_completed = round_num + 1

            # Check verdict
            verify_msgs = self.receive_messages(
                msg_type=MessageType.VERIFY_RESULT, case_id=ctx.case_id
            )
            if verify_msgs:
                verify_content = verify_msgs[-1].content
                verdict = verify_content.get("verdict", "approve")
                self._maybe_update_best_intermediate(ctx, verify_content, round_num)
                self._maybe_update_best_anatomy_intermediate(
                    ctx=ctx,
                    verify_content=verify_content,
                    round_num=round_num,
                )

                if verdict == "approve":
                    if split_relabel_applied and not post_split_cleanup_consumed:
                        post_split_cleanup_pending = True
                        self.log(
                            "  [PostSplit] Split relabel applied; "
                            "forcing one additional diagnosis/repair round"
                        )
                    else:
                        self.log(f"  ✓ Approved on round {round_num + 1}")
                        return "approved"

                elif verdict == "reject":
                    if split_relabel_applied and not post_split_cleanup_consumed:
                        post_split_cleanup_pending = True
                        self.log(
                            "  [PostSplit] Split relabel applied; defer reject and "
                            "run one post-process round on current mask"
                        )
                        # Keep current mask for one follow-up round to repair
                        # side effects introduced by split relabel.
                    else:
                        # Revert to a safe baseline (original or current best intermediate).
                        self.log(f"  ✗ Rejected — reverting to safe baseline")
                        action = "retry"
                        can_retry = (round_num + 1) < max_rounds
                        if can_retry:
                            action = self._should_retry(ctx, verify_content)
                        fallback_mask = ctx.original_mask
                        if (
                            strict_monotonic_mode
                            and ctx.best_intermediate_mask is not None
                        ):
                            fallback_mask = ctx.best_intermediate_mask
                        if fallback_mask is not None:
                            ctx.current_mask = fallback_mask.copy()
                        ctx.applied_ops = []
                        ctx.rqs_history = []

                        if can_retry:
                            if action == "skip":
                                self.log("  ↷ Skip further retries for this case")
                                break
                            self.log(f"  ⟳ Retrying with feedback (Round {round_num + 2})")

                elif verdict == "needs_more_work":
                    self.log(f"  ⟳ Needs more work — continuing")
                    # Handle NEEDS_MORE_WORK: get remaining issues
                    nmw_msgs = self.receive_messages(
                        msg_type=MessageType.NEEDS_MORE_WORK, case_id=ctx.case_id
                    )
                    if nmw_msgs:
                        remaining = nmw_msgs[-1].content.get("remaining_issues", [])
                        # Feed remaining issues into next diagnosis round
                        ctx.triage_issues = remaining
                    # Continue loop

            # Clear agent memories for next round (fresh reasoning)
            for agent in [diagnosis, planner]:
                agent.reset_memory()
            round_num += 1

        # Exhausted rounds
        if ctx.verifier_approved:
            return "approved"
        else:
            if (
                ctx.best_intermediate_mask is not None
                and not getattr(ctx, "best_intermediate_is_original", True)
            ):
                self.log(
                    "  Using best-intermediate fallback is available "
                    f"(round={ctx.best_intermediate_round}, "
                    f"score={ctx.best_intermediate_score:.3f})"
                )
            if strict_monotonic_mode and ctx.best_intermediate_mask is not None:
                ctx.current_mask = ctx.best_intermediate_mask.copy()
            self.log(f"  Exhausted {ctx.rounds_completed} rounds")
            return "gave_up"

    def _init_anatomy_baseline(self, ctx: CaseContext) -> None:
        """
        Load anatomy distribution stats and compute the no-GT anatomy score for the
        original (pre-repair) mask. Stores results in ctx.anatomy_stats and
        ctx.anatomy_score_before so DiagnosisAgent and VerifierAgent can use them.
        """
        stats_path = self.cfg.get("anatomy_stats_path")
        if not stats_path:
            return
        if ctx.original_mask is None or ctx.image is None:
            return
        try:
            from ..anatomy_score import (
                load_stats,
                compute_anatomy_score,
                build_anatomy_llm_summary,
                _normalize_view,
            )
            stats_dict = load_stats(stats_path)
            ctx.anatomy_stats = stats_dict
            view = _normalize_view(ctx.view_type or "")
            stats = stats_dict.get(view)
            if stats is None:
                self.log(f"  Anatomy stats: no entry for view='{view}'")
                return
            score, detail = compute_anatomy_score(ctx.original_mask, ctx.image, view, stats)
            ctx.anatomy_score_before = float(score)
            ma_cfg = self.cfg.get("multi_agent", {}) or {}
            summary_top_k = int(ma_cfg.get("anatomy_llm_top_k", 5))
            summary = build_anatomy_llm_summary(
                score=score,
                detail=detail,
                view_type=view,
                top_k=max(1, summary_top_k),
            )
            ctx.anatomy_score_current = float(score)
            ctx.anatomy_score_history = [float(score)]
            ctx.anatomy_baseline_summary = dict(summary)
            ctx.anatomy_latest_summary = dict(summary)
            self.log(f"  Anatomy baseline score: {score:.1f}/100 (view={view})")
            if summary.get("hard_violations"):
                self.log(
                    "  Anatomy baseline violations: "
                    f"{summary.get('hard_violations', [])}"
                )
        except Exception as e:
            self.log(f"  Anatomy baseline init failed: {e}")

    @staticmethod
    def _has_high_risk_issue(issues: List[str]) -> bool:
        risk_tokens = (
            "missing",
            "massive",
            "severe",
            "broken",
            "disconnect",
            "touch",
            "leak",
            "hole",
        )
        for issue in issues:
            low = str(issue).lower()
            if any(tok in low for tok in risk_tokens):
                return True
        return False

    @staticmethod
    def _clip01(value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            x = default
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    @staticmethod
    def _safe_float(value: Any, default: float = float("nan")) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _is_finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    def _strict_no_gt_online(self) -> bool:
        ma_cfg = self.cfg.get("multi_agent", {}) or {}
        return bool(ma_cfg.get("strict_no_gt_online", False))

    def _score_verifier_snapshot(self, verify_content: Dict[str, Any]) -> float:
        """
        No-GT quality score for an intermediate repaired mask.
        Higher is better.
        """
        verdict = str(verify_content.get("verdict", "needs_more_work")).lower()
        verdict_map = {"approve": 1.0, "needs_more_work": 0.2, "reject": -1.0}

        cmp_obj = verify_content.get("vlm_comparison", {}) or {}
        cmp_v = str(cmp_obj.get("verdict", "same")).lower()
        cmp_map = {
            "improved": 1.0,
            "same": 0.0,
            "neutral": 0.0,
            "worse": -1.0,
            "degraded": -1.0,
        }

        quality_obj = verify_content.get("vlm_quality", {}) or {}
        quality_v = str(quality_obj.get("quality", "borderline")).lower()
        quality_map = {"good": 1.0, "borderline": 0.0, "bad": -1.0}
        quality_score_raw = 0.0
        try:
            quality_score_raw = float(quality_obj.get("score", 5.0))
        except Exception:
            quality_score_raw = 5.0
        quality_score = max(-1.0, min(1.0, (quality_score_raw - 5.0) / 5.0))

        verifier_conf = self._clip01(verify_content.get("confidence", 0.5), default=0.5)
        cmp_conf = self._clip01(cmp_obj.get("confidence", 0.0), default=0.0)
        metrics_obj = verify_content.get("metrics", {}) or {}
        dice_delta = 0.0
        try:
            dice_delta = float(metrics_obj.get("dice_delta", 0.0))
        except Exception:
            dice_delta = 0.0
        proxy_obj = verify_content.get("proxy_consensus", {}) or {}
        proxy_score = 0.0
        proxy_conf = 0.0
        proxy_v = str(proxy_obj.get("proxy_verdict", "neutral")).lower()
        proxy_map = {"improved": 1.0, "neutral": 0.0, "degraded": -1.0}
        try:
            proxy_score = float(proxy_obj.get("proxy_score", 0.0))
        except Exception:
            proxy_score = 0.0
        proxy_conf = self._clip01(proxy_obj.get("proxy_confidence", 0.0), default=0.0)
        consensus_score = 0.0
        try:
            consensus_score = float(proxy_obj.get("consensus_score", 0.0))
        except Exception:
            consensus_score = 0.0
        # Canonical online score (0-100) is primary when available.
        online_score = verify_content.get("online_score", None)
        online_bonus = 0.0
        try:
            if online_score is not None:
                online_bonus = max(-1.0, min(1.0, (float(online_score) - 50.0) / 50.0))
        except Exception:
            online_bonus = 0.0

        # Cap GT contribution to avoid dominating no-GT signals.
        dice_bonus = 0.0
        if not self._strict_no_gt_online():
            dice_bonus = max(-0.5, min(0.5, dice_delta * 10.0))

        return float(
            verdict_map.get(verdict, 0.0)
            + 2.0 * cmp_map.get(cmp_v, 0.0)
            + 0.8 * quality_map.get(quality_v, 0.0)
            + 0.2 * quality_score
            + 0.3 * verifier_conf
            + 0.5 * cmp_conf
            + 0.6 * proxy_map.get(proxy_v, 0.0)
            + 0.3 * proxy_score
            + 0.2 * proxy_conf
            + 0.2 * consensus_score
            + 0.8 * online_bonus
            + dice_bonus
        )

    def _is_best_candidate(self, verify_content: Dict[str, Any]) -> bool:
        strict_no_gt = self._strict_no_gt_online()
        verdict = str(verify_content.get("verdict", "")).lower()
        cmp_obj = verify_content.get("vlm_comparison", {}) or {}
        cmp_v = str(cmp_obj.get("verdict", "same")).lower()
        quality_obj = verify_content.get("vlm_quality", {}) or {}
        quality_v = str(quality_obj.get("quality", "borderline")).lower()
        proxy_obj = verify_content.get("proxy_consensus", {}) or {}
        proxy_v = str(proxy_obj.get("proxy_verdict", "neutral")).lower()
        consensus_v = str(proxy_obj.get("consensus_verdict", "needs_more_work")).lower()
        metrics_obj = verify_content.get("metrics", {}) or {}
        dice_delta = 0.0
        try:
            dice_delta = float(metrics_obj.get("dice_delta", 0.0))
        except Exception:
            dice_delta = 0.0
        gt_delta_th = float(
            self.cfg.get("multi_agent", {}).get(
                "best_intermediate_gt_dice_delta_threshold", 0.01
            )
        )

        if cmp_v in ("worse", "degraded") and verdict != "approve":
            # In evaluation runs with GT, keep large true improvements even if VLM compare is noisy.
            if strict_no_gt or dice_delta <= gt_delta_th:
                return False

        has_online_gain = False
        try:
            has_online_gain = float(verify_content.get("online_score", 50.0)) >= 60.0
        except Exception:
            has_online_gain = False

        base_hit = (
            verdict == "approve"
            or cmp_v == "improved"
            or quality_v == "good"
            or proxy_v == "improved"
            or consensus_v == "approve"
        )
        if strict_no_gt:
            return base_hit or has_online_gain
        return base_hit or has_online_gain or (dice_delta > gt_delta_th)

    def _maybe_update_best_intermediate(
        self,
        ctx: CaseContext,
        verify_content: Dict[str, Any],
        round_num: int,
    ) -> None:
        """
        Cache the best no-GT intermediate repaired mask so it can be saved
        even if later rounds are rejected.
        """
        if ctx.current_mask is None:
            return
        ex_cfg = self.cfg.get("executor", {})
        strict_monotonic_mode = bool(ex_cfg.get("strict_monotonic_mode", False))
        strict_round_min_compare_conf = float(
            ex_cfg.get("strict_round_min_compare_confidence", 0.60)
        )
        strict_round_score_delta = float(ex_cfg.get("strict_round_score_delta", 1e-3))
        hard_relabel_ops = {"lv_split_to_rv", "rv_split_to_lv"}
        has_hard_relabel = any(
            str(op.get("name", "")).strip() in hard_relabel_ops
            for op in (ctx.applied_ops or [])
        )

        if strict_monotonic_mode:
            cmp_obj = verify_content.get("vlm_comparison", {}) or {}
            cmp_v = str(cmp_obj.get("verdict", "")).lower()
            cmp_conf = self._clip01(cmp_obj.get("confidence", 0.0), default=0.0)
            proxy_obj = verify_content.get("proxy_consensus", {}) or {}
            strict_proxy_override = bool(proxy_obj.get("strict_override_allow", False))
            if cmp_v != "improved":
                if not strict_proxy_override:
                    return
            elif cmp_conf < strict_round_min_compare_conf:
                return
            if not self._is_best_candidate(verify_content):
                return
        else:
            if not has_hard_relabel and not self._is_best_candidate(verify_content):
                return

        candidate_score = self._score_verifier_snapshot(verify_content)
        best_score = float(getattr(ctx, "best_intermediate_score", 0.0))
        # Hard relabel snapshots should be preserved once for fallback even when
        # verifier is conservative, because this path is user-forced policy.
        has_only_baseline = bool(getattr(ctx, "best_intermediate_is_original", False))
        if strict_monotonic_mode:
            if candidate_score <= best_score + strict_round_score_delta:
                return
        else:
            if has_hard_relabel and has_only_baseline:
                candidate_score = max(candidate_score, best_score + 1e-3)
            elif candidate_score <= best_score + 1e-6:
                return

        ctx.best_intermediate_mask = ctx.current_mask.copy()
        ctx.best_intermediate_score = candidate_score
        ctx.best_intermediate_round = int(round_num + 1)
        ctx.best_intermediate_verdict = str(
            verify_content.get("verdict", "needs_more_work")
        ).lower()
        ctx.best_intermediate_reason = str(verify_content.get("reasoning", ""))[:300]
        ctx.best_intermediate_ops = [dict(op) for op in (ctx.applied_ops or [])]
        ctx.best_intermediate_is_original = False
        try:
            ctx.best_intermediate_online_score = float(
                verify_content.get("online_score", float("nan"))
            )
        except Exception:
            ctx.best_intermediate_online_score = float("nan")
        anatomy_obj = verify_content.get("anatomy_signal", {}) or {}
        try:
            ctx.best_intermediate_anatomy_score = float(
                anatomy_obj.get("score_after", float("nan"))
            )
        except Exception:
            ctx.best_intermediate_anatomy_score = float("nan")
        self.log(
            "  ↺ Updated best intermediate "
            f"(round={ctx.best_intermediate_round}, score={candidate_score:.3f})"
        )

    def _maybe_update_best_anatomy_intermediate(
        self,
        ctx: CaseContext,
        verify_content: Dict[str, Any],
        round_num: int,
    ) -> None:
        """
        Track the intermediate snapshot with the highest anatomy score.
        Used by nonapproved final-pick policy (anatomy-priority fallback).
        """
        if ctx.current_mask is None:
            return

        anatomy_obj = verify_content.get("anatomy_signal", {}) or {}
        if not bool(anatomy_obj.get("enabled", False)):
            return

        score_after = self._safe_float(anatomy_obj.get("score_after", float("nan")))
        score_before = self._safe_float(
            getattr(ctx, "anatomy_score_before", float("nan"))
        )
        if self._is_finite(score_before) and score_before < 0.0:
            score_before = float("nan")
        if not self._is_finite(score_after):
            return
        delta = score_after - score_before if self._is_finite(score_before) else float("nan")

        best_score = self._safe_float(
            getattr(ctx, "best_anatomy_intermediate_score", float("nan"))
        )
        has_baseline_only = bool(
            getattr(ctx, "best_anatomy_intermediate_is_original", False)
        )
        should_update = False
        if not self._is_finite(best_score):
            should_update = True
        elif has_baseline_only and score_after >= (best_score - 1e-6):
            should_update = True
        elif score_after > (best_score + 1e-6):
            should_update = True

        if not should_update:
            return

        ctx.best_anatomy_intermediate_mask = ctx.current_mask.copy()
        ctx.best_anatomy_intermediate_score = float(score_after)
        ctx.best_anatomy_intermediate_delta = float(delta)
        ctx.best_anatomy_intermediate_round = int(round_num + 1)
        ctx.best_anatomy_intermediate_verdict = str(
            verify_content.get("verdict", "needs_more_work")
        ).lower()
        ctx.best_anatomy_intermediate_reason = str(
            verify_content.get("reasoning", "")
        )[:300]
        ctx.best_anatomy_intermediate_ops = [dict(op) for op in (ctx.applied_ops or [])]
        ctx.best_anatomy_intermediate_is_original = False
        ctx.best_anatomy_intermediate_online_score = self._safe_float(
            verify_content.get("online_score", float("nan"))
        )
        self.log(
            "  ↺ Updated best anatomy intermediate "
            f"(round={ctx.best_anatomy_intermediate_round}, "
            f"anatomy={score_after:.1f}, delta={delta:+.1f})"
        )

    def _should_retry(
        self, ctx: CaseContext, verify_content: Dict[str, Any]
    ) -> str:
        """Ask LLM whether to retry a rejected case."""
        prompt = (
            f"Case {ctx.stem} was rejected by the Verifier.\n"
            f"Reason: {verify_content.get('reasoning', 'unknown')}\n"
            f"Round: {ctx.rounds_completed}\n"
            f"Applied ops: {[op['name'] for op in ctx.applied_ops]}\n\n"
            f"Should we retry with a different approach, or skip this case?"
        )

        result = self.think_json(prompt)
        action = result.get("action", "retry")

        if action in ("skip", "give_up"):
            return "skip"
        return "retry"

    def generate_batch_summary(
        self,
        cases: Dict[str, CaseContext],
    ) -> Dict[str, Any]:
        """Generate a summary of the entire batch processing."""
        summary: Dict[str, Any] = {
            "total": len(cases),
            "good": 0,
            "approved": 0,
            "rejected": 0,
            "gave_up": 0,
            "ops_distribution": {},
        }

        for ctx in cases.values():
            if ctx.triage_category == "good":
                summary["good"] += 1
            elif ctx.verifier_approved:
                summary["approved"] += 1
            else:
                summary["gave_up"] += 1

            for op in ctx.applied_ops:
                op_name = op.get("name", "unknown")
                summary["ops_distribution"][op_name] = (
                    summary["ops_distribution"].get(op_name, 0) + 1
                )

        self.log(
            f"Batch: {summary['total']} cases — "
            f"{summary['good']} good, {summary['approved']} approved, "
            f"{summary['gave_up']} gave_up"
        )

        # LLM cross-case learning (if enabled)
        if self.cfg.get("multi_agent", {}).get("batch_summary", True):
            prompt = (
                f"Batch processing complete.\n"
                f"Good: {summary['good']}, Approved: {summary['approved']}, "
                f"Gave up: {summary['gave_up']}\n"
                f"Ops used: {json.dumps(summary['ops_distribution'])}\n\n"
                f"What patterns do you see? Any recommendations for future batches?"
            )
            result = self.think_json(prompt)
            summary["llm_analysis"] = result.get("reasoning", "")

        return summary
