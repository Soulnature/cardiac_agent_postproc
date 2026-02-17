"""
CoordinatorAgent: Central coordinator for the multi-agent pipeline.

Manages the case-by-case workflow: Triage → Diagnosis → Plan → Execute → Verify,
with dynamic re-routing when the Verifier rejects a fix.
"""
from __future__ import annotations

import json
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
        return (
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

        # Step 2-5: Diagnosis → Plan → Execute → Verify loop
        for round_num in range(max_rounds):
            self.log(f"  ── Round {round_num + 1}/{max_rounds} ──")

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
            executor.process_case(ctx)

            # Verification
            verifier = agents["verifier"]
            verifier.process_case(ctx)

            ctx.rounds_completed = round_num + 1

            # Check verdict
            verify_msgs = self.receive_messages(
                msg_type=MessageType.VERIFY_RESULT, case_id=ctx.case_id
            )
            if verify_msgs:
                verdict = verify_msgs[-1].content.get("verdict", "approve")

                if verdict == "approve":
                    self.log(f"  ✓ Approved on round {round_num + 1}")
                    return "approved"

                elif verdict == "reject":
                    # Revert to original mask
                    self.log(f"  ✗ Rejected — reverting to original")
                    action = "retry"
                    if round_num < max_rounds - 1:
                        action = self._should_retry(ctx, verify_msgs[-1].content)
                    ctx.current_mask = ctx.original_mask.copy()
                    ctx.applied_ops = []
                    ctx.rqs_history = []

                    if round_num < max_rounds - 1:
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

        # Exhausted rounds
        if ctx.verifier_approved:
            return "approved"
        else:
            self.log(f"  Exhausted {max_rounds} rounds")
            return "gave_up"

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
