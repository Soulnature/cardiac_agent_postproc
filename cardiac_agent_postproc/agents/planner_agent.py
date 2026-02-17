"""
PlannerAgent: LLM-powered repair planning.

Receives diagnoses from DiagnosisAgent and produces an optimized repair plan
— ordering operations to minimize conflicts and maximize improvement.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .base_agent import BaseAgent
from .message_bus import CaseContext, MessageBus, MessageType
from .diagnosis_agent import ISSUE_OPS_MAP


class PlannerAgent(BaseAgent):
    """
    Plans the optimal sequence of repair operations based on diagnoses.

    The planner reasons about:
    - Operation ordering (dependencies, conflicts)
    - Risk of regression (e.g., myo_bridge may cause touching)
    - Which ops to combine vs. skip
    """

    @property
    def name(self) -> str:
        return "planner"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Planner Agent in a cardiac MRI segmentation repair system.\n\n"
            "Your role is to create a minimal, safe, and optimal repair plan from a GIVEN set of diagnoses.\n"
            "CRITICAL: You MUST plan ONLY for the provided diagnoses. Do NOT invent new defects, do NOT escalate severity,\n"
            "and do NOT reinterpret anatomy beyond what the diagnoses state.\n\n"

            "You must consider:\n"
            "0. **Hard Constraints (No Hallucination + Plan Limit)**:\n"
            "   - Plan ONLY for issues explicitly present in the diagnoses list.\n"
            "   - Do NOT add new operations for unmentioned problems.\n"
            "   - Do NOT change directions from the diagnoses.\n"
            "   - Maximum 4 operations total. Prefer 1-2 operations when possible.\n\n"

            "1. **Class Constraint (Crucial)**:\n"
            "   - For 3ch and 4ch views, the final result MUST contain all 3 classes (LV, Myo, RV).\n"
            "   - For 2ch views, RV may be absent; require at least LV and Myo.\n"
            "   - A class is considered truly MISSING only if:\n"
            "       (a) the diagnosis explicitly states missing/nearly absent/massive_missing, OR\n"
            "       (b) quantitative hint indicates area < 30% expected, OR\n"
            "       (c) the class is completely absent.\n"
            "     Thin/small structures are NOT missing.\n"
            "   - If a class is truly missing, you MUST include an operation to restore it.\n\n"

            "2. **Non-overlapping, Class-Specific Operations**:\n"
            "   - Every operation must be targeted to exactly ONE class: LV, Myo, or RV.\n"
            "   - Use the 'affected_class' field to specify which class is modified.\n"
            "   - Avoid collateral damage: do NOT plan edits that are likely to erase another class.\n"
            "   - Avoid redundancy: do NOT repeat the same operation for the same class+direction.\n"
            "   - Avoid contradictions: NEVER schedule both dilate and erode for the same class in the same direction.\n\n"

            "3. **Severity levels**: critical > high > medium > low.\n"
            "   - CRITICAL does NOT automatically imply Atlas. Distinguish 'Broken' vs 'Missing':\n"
            "     * BROKEN/FRAGMENTED (Medium/High):\n"
            "       - If severity is MEDIUM: Prefer `convex_hull_fill` or `bridge` ops.\n"
            "       - If severity is HIGH or CRITICAL (e.g. `fragmented_lv`): Use `neighbor_*_repair` FIRST (e.g. `neighbor_rv_repair` for RV). Fall back to `atlas_growprune_0` only if neighbor is unavailable.\n"
            "     * MISSING / WRONG LOCATION / NEARLY ABSENT:\n"
            "       - ALWAYS prefer `neighbor_rv_repair` / `neighbor_myo_repair` / `neighbor_lv_repair` (class-specific, uses neighbor slice as donor).\n"
            "       - Fall back to `atlas_growprune_0` ONLY if neighbor repair is not available.\n"
            "       - These are HIGH RISK ops and must NOT be used for boundary mismatch, thin wall, jaggedness, or small gaps.\n\n"

            "4. **Operation Order Matters (Strict Order)**:\n"
            "   - (0) `neighbor_rv_repair` / `neighbor_myo_repair` / `neighbor_lv_repair` FIRST for missing/severely-fragmented classes (uses neighbor slice as shape donor).\n"
            "   - (1) `atlas_growprune_0` NEXT as fallback for missing structures if neighbor repair unavailable.\n"
            "   - (2) Structural topology fixes NEXT: `rv_lv_barrier`, `convex_hull_fill`, `myo_bridge`, `lv_bridge`.\n"
            "   - (3) Intensity expansion NEXT: `expand_myo_intensity`, `expand_lv_intensity`, `expand_rv_intensity`.\n"
            "   - (4) Directional boundary ops NEXT: `2_dilate`, `2_erode`, `3_dilate`, `3_erode`, `3_erode_expand_myo`, `rv_erode`, `1_erode_expand_myo`.\n"
            "   - (5) Cleanup ops LAST: `fill_holes`, `fill_holes_m2`, `remove_islands`, `topology_cleanup`,\n"
            "       `smooth_morph`, `smooth_morph_k5`, `valve_suppress`.\n"
            "   - If both structural and boundary issues exist, ALWAYS fix structural first.\n"
            "   - After a `neighbor_*_repair`, re-diagnosis will happen automatically for follow-up refinement.\n\n"

            "5. **Conflict Rules (Safety)**:\n"
            "   - If any diagnosis indicates RV and LV touch directly, schedule `rv_lv_barrier` BEFORE any dilation.\n"
            "   - Only schedule `valve_suppress` if a diagnosis explicitly states valve leak.\n"
            "   - Only schedule smoothing if a diagnosis explicitly states jagged boundary.\n\n"

            "6. **Risk Labeling**:\n"
            "   - risk='high' for `atlas_growprune_0`.\n"
            "   - risk='medium' for structural ops (`convex_hull_fill`, bridges, `rv_lv_barrier`) and expand_*_intensity.\n"
            "   - risk='low' for directional boundary ops and cleanup ops.\n\n"

            "Respond with JSON ONLY:\n"
            "{\"plan\": [{\"operation\": \"...\", \"affected_class\": \"rv|myo|lv\", "
            "\"direction\": \"septal|lateral|apical|basal|global\", "
            "\"reason\": \"must cite a provided diagnosis\", \"risk\": \"low|medium|high\"}], "
            "\"strategy\": \"brief overall approach\"}\n"
        )


    def process_case(self, ctx: CaseContext) -> None:
        """
        Create a repair plan from the case's diagnoses.
        Sends REPAIR_PLAN message to the bus.
        """
        if not ctx.diagnoses:
            self.log(f"No diagnoses for {ctx.stem}, skip planning")
            empty_plan = {"plan": [], "strategy": "No issues to fix"}
            self.send_message(
                recipient="coordinator",
                msg_type=MessageType.REPAIR_PLAN,
                content=empty_plan,
                case_id=ctx.case_id,
            )
            self.send_message(
                recipient="executor",
                msg_type=MessageType.REPAIR_PLAN,
                content=empty_plan,
                case_id=ctx.case_id,
            )
            return

        self.log(f"Planning repair for {ctx.stem} ({len(ctx.diagnoses)} diagnoses)")

        # Build the prompt with all available info
        # Load Repair Knowledge Base
        repair_kb = {}
        try:
            kb_path = "repair_kb.json"
            if os.path.exists(kb_path):
                with open(kb_path, "r") as f:
                    repair_kb = json.load(f)
        except Exception as e:
            self.log(f"Failed to load repair_kb.json: {e}")

        # Enrich diagnoses with KB suggestions
        rich_diagnoses = [dict(d) for d in ctx.diagnoses]
        for d in rich_diagnoses:
            issue = d.get("issue", "")
            # 1. Try exact match in KB
            kb_suggestions = repair_kb.get(issue, [])
            # 2. Try partial match in KB (e.g. "gap" -> "broken_ring")
            if not kb_suggestions:
                 for key, suggests in repair_kb.items():
                     if key in issue.lower() or issue.lower() in key:
                         kb_suggestions = suggests
                         break
            
            # Format suggestions for the prompt
            if kb_suggestions:
                best_ops = [s["op"] for s in kb_suggestions]
                d["proven_solutions"] = f"Use {best_ops} (High Confidence)"
            else:
                d["proven_solutions"] = "No specific history"

            # Fallback to hardcoded map if KB empty
            valid_ops = ISSUE_OPS_MAP.get(issue)
            if not valid_ops:
                if "gap" in issue.lower() and "myo" in issue.lower():
                     valid_ops = ISSUE_OPS_MAP.get("broken_ring", [])
            d["available_tools"] = valid_ops if valid_ops else "check 'Available operations'"

        diag_summary = json.dumps(rich_diagnoses, indent=2, default=str)

        # Include all known ops for reference
        all_ops = set()
        for ops_list in ISSUE_OPS_MAP.values():
            all_ops.update(ops_list)

        prompt = (
            f"Case: {ctx.stem} (view: {ctx.view_type})\n\n"
            f"Diagnoses (with Proven Solutions):\n{diag_summary}\n\n"
            f"Available operations: {sorted(all_ops)}\n\n"
            f"Previous ops applied this case: {[op['name'] for op in ctx.applied_ops] if ctx.applied_ops else 'none'}\n"
            f"Current round: {ctx.rounds_completed + 1}\n\n"
            f"Create an optimized repair plan. **Prioritize 'Proven Solutions' if available.**\n"
            f"Order operations carefully. If a fix might cause a side effect, add a mitigation step."
        )

        result = self.think_json(prompt)
        plan = result.get("plan", [])
        strategy = result.get("strategy", "")

        # Fallback: if LLM fails, use diagnoses directly in severity order
        if not plan:
            plan = self._fallback_plan(ctx.diagnoses)
            strategy = "Fallback: operations from diagnoses in severity order"
        plan = self._sanitize_plan(plan, ctx.diagnoses)

        self.log(f"  → Plan: {[step['operation'] for step in plan]}")
        self.log(f"  → Strategy: {strategy}")

        # Send plan to bus (both coordinator and executor need it)
        plan_content = {"plan": plan, "strategy": strategy}
        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.REPAIR_PLAN,
            content=plan_content,
            case_id=ctx.case_id,
        )
        self.send_message(
            recipient="executor",
            msg_type=MessageType.REPAIR_PLAN,
            content=plan_content,
            case_id=ctx.case_id,
        )

    def revise_plan(self, ctx: CaseContext, feedback: Dict[str, Any]) -> None:
        """
        Revise the plan based on Verifier or Executor feedback.
        Called when an op fails or the Verifier rejects the result.
        """
        self.log(f"Revising plan for {ctx.stem} based on feedback")

        prompt = (
            f"The current repair plan for {ctx.stem} needs revision.\n\n"
            f"Feedback: {json.dumps(feedback, indent=2, default=str)}\n"
            f"Previously applied ops: {[op['name'] for op in ctx.applied_ops]}\n"
            f"Current diagnoses: {json.dumps(ctx.diagnoses, indent=2, default=str)}\n\n"
            f"What should we try differently? Consider alternative operations or sequences."
        )

        result = self.think_json(prompt)
        plan = result.get("plan", [])
        strategy = result.get("strategy", "Revised plan based on feedback")
        plan = self._sanitize_plan(plan, ctx.diagnoses)

        revision_content = {"plan": plan, "strategy": strategy}
        self.send_message(
            recipient="coordinator",
            msg_type=MessageType.PLAN_REVISION,
            content=revision_content,
            case_id=ctx.case_id,
        )
        self.send_message(
            recipient="executor",
            msg_type=MessageType.PLAN_REVISION,
            content=revision_content,
            case_id=ctx.case_id,
        )

    def _fallback_plan(self, diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a simple plan from diagnoses when LLM fails."""
        severity_order = {"critical": -1, "high": 0, "medium": 1, "low": 2}
        sorted_diags = sorted(
            diagnoses,
            key=lambda d: severity_order.get(d.get("severity", "low"), 3),
        )

        plan = []
        seen_ops = set()
        for d in sorted_diags:
            op = d.get("operation", "")
            if op and op not in seen_ops:
                plan.append({
                    "operation": op,
                    "affected_class": d.get("affected_class", "myo"),
                    "direction": d.get("direction", "global"),
                    "reason": f"Fix {d.get('issue', 'unknown')}",
                    "risk": "unknown",
                })
                seen_ops.add(op)

        return plan

    @staticmethod
    def _is_atlas_op(op_name: str) -> bool:
        return op_name in {
            "atlas_growprune_0",
            "atlas_growprune_1",
            "individual_atlas_transfer",
            "neighbor_rv_repair",
            "neighbor_myo_repair",
            "neighbor_lv_repair",
            "atlas_rv_repair",
            "atlas_myo_repair",
            "atlas_lv_repair",
        }

    @staticmethod
    def _has_critical_missing_issue(diagnoses: List[Dict[str, Any]]) -> bool:
        missing_tokens = (
            "massive_missing",
            "missing_structure",
            "missing_myo",
            "missing_lv",
            "missing_rv",
            "severe_misplacement",
        )
        for d in diagnoses:
            issue = str(d.get("issue", "")).lower()
            sev = str(d.get("severity", "")).lower()
            if sev != "critical":
                continue
            if issue in missing_tokens or "missing" in issue:
                return True
        return False

    def _sanitize_plan(
        self,
        raw_plan: List[Dict[str, Any]],
        diagnoses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Enforce deterministic safeguards:
        1) deduplicate operations (keep first),
        2) atlas-first only when critical missing issue exists.
        """
        if not isinstance(raw_plan, list):
            return []

        deduped: List[Dict[str, Any]] = []
        seen_steps = set()
        for step in raw_plan:
            if not isinstance(step, dict):
                continue
            op = str(step.get("operation", "")).strip()
            if not op:
                continue
            item = dict(step)
            item["operation"] = op
            item.setdefault("affected_class", "myo")
            item.setdefault("direction", "global")
            item.setdefault("reason", "")
            item.setdefault("risk", "unknown")
            step_key = (
                item["operation"],
                str(item["affected_class"]).strip().lower(),
                str(item["direction"]).strip().lower(),
            )
            if step_key in seen_steps:
                continue
            deduped.append(item)
            seen_steps.add(step_key)

        enriched_plan = []
        for step in deduped:
            # Try to find matching diagnosis to attach bbox
            # Match by: affected_class AND (operation matches OR issue mentioned in reason)
            step_op = step["operation"]
            step_cls = step["affected_class"]
            step_dir = step.get("direction", "global")
            
            best_diag = None
            for d in diagnoses:
                d_op = d.get("operation", "")
                d_cls = d.get("affected_class", "")
                d_dir = d.get("direction", "")
                
                # Strict match on class
                if d_cls != step_cls:
                    continue

                # Weak match on op (step op should match diagnosis op)
                if d_op == step_op:
                    best_diag = d
                    break
                
                # Fallback: check if reason cites issue
                if step.get("reason") and d.get("issue") in step["reason"]:
                    best_diag = d
                    break
            
            if best_diag and best_diag.get("bbox"):
                step["bbox"] = best_diag["bbox"]
            
            # --- SAFETY CHECK REMOVED ---
            # ExecutorAgent now handles missing bboxes via "Smart Global" inference (computed from direction).
            
            # --- ANATOMICAL SAFEGUARD: VALVE PLANE PROTECTION ---
            # Basal region corresponds to the valve plane, which is naturally open.
            # Operations that enforce closure (convex_hull, bridge) here are destructive.
            # We explicitly block them for 'basal' direction.
            op_name_lower = step["operation"].lower()
            direction_lower = str(step.get("direction", "")).lower()
            
            if direction_lower == "basal" and any(x in op_name_lower for x in ["convex", "hull", "bridge", "close"]):
                 self.log(f"WARNING: Dropping anatomical violation '{step['operation']}' for basal/valve plane.")
                 continue

            enriched_plan.append(step)

        if not enriched_plan:
            return []

        deduped = enriched_plan # Update variable for following logic

        has_critical_missing = self._has_critical_missing_issue(diagnoses)
        atlas_steps = [s for s in deduped if self._is_atlas_op(s["operation"])]
        non_atlas_steps = [s for s in deduped if not self._is_atlas_op(s["operation"])]

        if not atlas_steps:
            return deduped

        if has_critical_missing:
            return atlas_steps + non_atlas_steps
        
        return non_atlas_steps + atlas_steps[:1] if non_atlas_steps else atlas_steps[:1]
