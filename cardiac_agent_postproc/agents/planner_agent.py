"""
PlannerAgent: LLM-powered repair planning.

Receives diagnoses from DiagnosisAgent and produces an optimized repair plan
— ordering operations to minimize conflicts and maximize improvement.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from .message_bus import CaseContext, MessageBus, MessageType
from .diagnosis_agent import ISSUE_OPS_MAP, normalize_direction_hint


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
        fallback = (
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
            "   - (4) Directional boundary ops NEXT: `2_dilate`, `2_erode`, `3_dilate`, `3_erode`, `2_erode_expand_lv`, `2_erode_expand_rv`, `3_erode_expand_myo`, `rv_erode`, `1_erode_expand_myo`.\n"
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
        return self._prompt_with_fallback("planner_system.txt", fallback)


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

        atlas_applied_before = self._has_prior_atlas_class_repair(ctx)
        if atlas_applied_before and int(getattr(ctx, "rounds_completed", 0)) >= 1:
            post_plan = self._build_post_atlas_cleanup_plan(ctx.diagnoses)
            plan = self._sanitize_plan(post_plan, ctx.diagnoses)
            strategy = (
                "Policy override: atlas class replacement already applied in prior "
                "round -> run post-processing cleanup in this round"
            )
            self.log(
                "  → Policy override active: post-atlas cleanup round "
                "(skip repeated atlas replacement)"
            )
        else:
            forced_atlas_plan = self._force_atlas_plan_if_needed(ctx, ctx.diagnoses)
            if forced_atlas_plan:
                plan = self._sanitize_plan(forced_atlas_plan, ctx.diagnoses)
                strategy = (
                    "Policy override: low triage score with near-missing class -> "
                    "force atlas replacement"
                )
                self.log(
                    "  → Policy override active: using atlas replacement for "
                    "low-score near-missing case"
                )

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

        suggested_plan = self._build_suggested_plan_from_feedback(feedback, ctx.diagnoses)
        if suggested_plan:
            strategy = "Deterministic fallback from executor hard-gate suggested_ops"
            revision_content = {"plan": suggested_plan, "strategy": strategy}
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
            return

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

    def _build_suggested_plan_from_feedback(
        self,
        feedback: Dict[str, Any],
        diagnoses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Build a deterministic fallback plan from executor-provided suggested_ops
        so we can recover within the same round without relying on LLM.
        """
        if not isinstance(feedback, dict):
            return []
        failed_ops = feedback.get("failed_ops", [])
        if not isinstance(failed_ops, list) or not failed_ops:
            return []

        all_known_ops = set()
        for ops in ISSUE_OPS_MAP.values():
            all_known_ops.update(ops)

        plan: List[Dict[str, Any]] = []
        seen = set()
        max_steps = 3

        def _find_diag_for_failed(op_name: str) -> Dict[str, Any]:
            op_name = str(op_name or "").strip()
            for d in diagnoses:
                if not isinstance(d, dict):
                    continue
                if str(d.get("operation", "")).strip() == op_name:
                    return d
            # fallback: first non-global direction diagnosis if available
            for d in diagnoses:
                if not isinstance(d, dict):
                    continue
                direction = normalize_direction_hint(str(d.get("direction", "global")))
                if direction != "global":
                    return d
            return diagnoses[0] if diagnoses else {}

        for fo in failed_ops:
            if not isinstance(fo, dict):
                continue
            failed_name = str(fo.get("operation", "")).strip()
            suggested_ops = fo.get("suggested_ops", [])
            if not isinstance(suggested_ops, list):
                continue

            diag = _find_diag_for_failed(failed_name)
            d_cls = str(diag.get("affected_class", "myo")).strip().lower() or "myo"
            d_dir = normalize_direction_hint(str(diag.get("direction", "global")))
            d_bbox = diag.get("bbox")
            if not (isinstance(d_bbox, list) and len(d_bbox) == 4):
                d_bbox = None

            for sop in suggested_ops:
                op_name = str(sop).strip()
                if (not op_name) or (op_name not in all_known_ops):
                    continue
                key = (op_name, d_cls, d_dir)
                if key in seen:
                    continue
                step: Dict[str, Any] = {
                    "operation": op_name,
                    "affected_class": d_cls,
                    "direction": d_dir,
                    "reason": (
                        f"Executor gate fallback for {failed_name}: "
                        f"{fo.get('reason', 'unknown')}"
                    ),
                    "risk": "medium",
                }
                if d_bbox is not None:
                    step["bbox"] = d_bbox
                plan.append(step)
                seen.add(key)
                if len(plan) >= max_steps:
                    break
            if len(plan) >= max_steps:
                break

        return self._sanitize_plan(plan, diagnoses)

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

    @staticmethod
    def _required_class_ids(view_type: str) -> List[int]:
        view = str(view_type).lower()
        if "2ch" in view or view == "2ch":
            return [2, 3]  # 2ch may not include RV
        return [1, 2, 3]

    @staticmethod
    def _class_name_from_id(class_id: int) -> str:
        return {1: "rv", 2: "myo", 3: "lv"}.get(class_id, "myo")

    @staticmethod
    def _severity_rank(severity: str) -> int:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return order.get(str(severity).strip().lower(), 9)

    @staticmethod
    def _source_rank(source: str) -> int:
        order = {"llm": 0, "vlm": 1, "rules": 2}
        return order.get(str(source).strip().lower(), 3)

    def _pick_spatial_hint_for_class(
        self,
        cls_name: str,
        diagnoses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Pick the most informative diagnosis as spatial hint for a target class.
        Priority: matching class + bbox + non-global direction + higher severity.
        Fallback: any diagnosis carrying bbox/non-global direction.
        """
        rows = [d for d in diagnoses if isinstance(d, dict)]
        primary = [
            d for d in rows
            if str(d.get("affected_class", "")).strip().lower() == cls_name
        ]
        fallback = rows if not primary else primary

        def _score(d: Dict[str, Any]) -> tuple[int, int, int]:
            has_bbox = 0 if d.get("bbox") else 1
            non_global = 0 if str(d.get("direction", "global")).strip().lower() != "global" else 1
            sev = self._severity_rank(str(d.get("severity", "medium")))
            return (has_bbox, non_global, sev)

        if not fallback:
            return {"direction": "global", "bbox": None, "issue": ""}

        best = sorted(fallback, key=_score)[0]
        direction = str(best.get("direction", "global")).strip().lower() or "global"
        bbox = best.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            bbox = None
        issue = str(best.get("issue", "")).strip()
        return {"direction": direction, "bbox": bbox, "issue": issue}

    @staticmethod
    def _has_prior_atlas_class_repair(ctx: CaseContext) -> bool:
        for op in (ctx.applied_ops or []):
            name = str(op.get("name", "")).strip()
            if name in {"atlas_rv_repair", "atlas_myo_repair", "atlas_lv_repair"}:
                return True
        return False

    def _build_post_atlas_cleanup_plan(
        self, diagnoses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        issue_set = {
            str(d.get("issue", "")).strip().lower()
            for d in diagnoses
            if isinstance(d, dict)
        }
        plan: List[Dict[str, Any]] = []

        if any(("touch" in issue) or ("overlap" in issue) for issue in issue_set):
            plan.append(
                {
                    "operation": "rv_lv_barrier",
                    "affected_class": "myo",
                    "direction": "septal",
                    "reason": "Post-atlas cleanup: re-separate RV and LV boundary",
                    "risk": "medium",
                }
            )

        if "fragmented_lv" in issue_set:
            plan.append(
                {
                    "operation": "lv_bridge",
                    "affected_class": "lv",
                    "direction": "global",
                    "reason": "Post-atlas cleanup: reconnect fragmented LV after class replacement",
                    "risk": "medium",
                }
            )

        if "disconnected_myo" in issue_set:
            plan.append(
                {
                    "operation": "myo_bridge",
                    "affected_class": "myo",
                    "direction": "global",
                    "reason": "Post-atlas cleanup: reconnect myocardium ring continuity",
                    "risk": "medium",
                }
            )

        plan.append(
            {
                "operation": "safe_topology_fix",
                "affected_class": "myo",
                "direction": "global",
                "reason": "Post-atlas cleanup: enforce single-component and hole-free topology",
                "risk": "low",
            }
        )
        plan.append(
            {
                "operation": "topology_cleanup",
                "affected_class": "myo",
                "direction": "global",
                "reason": "Post-atlas cleanup: enforce anatomical consistency constraints",
                "risk": "low",
            }
        )
        plan.append(
            {
                "operation": "remove_islands",
                "affected_class": "myo",
                "direction": "global",
                "reason": "Post-atlas cleanup: remove small isolated artifacts",
                "risk": "low",
            }
        )

        # Keep cleanup concise and deterministic.
        return plan[:3]

    def _detect_near_missing_classes(self, ctx: CaseContext) -> List[int]:
        mask = ctx.current_mask if ctx.current_mask is not None else ctx.mask
        if mask is None:
            return []

        planner_cfg = self.cfg.get("planner", {}) or {}
        min_pixels = int(planner_cfg.get("force_atlas_missing_min_pixels", 300))
        min_frac = float(planner_cfg.get("force_atlas_missing_min_frac", 0.005))
        total_px = max(1, int(mask.size))

        near_missing: List[int] = []
        for class_id in self._required_class_ids(ctx.view_type):
            area = int((mask == class_id).sum())
            area_frac = area / total_px
            if area == 0 or area < min_pixels or area_frac < min_frac:
                near_missing.append(class_id)
        return near_missing

    def _force_atlas_plan_if_needed(
        self,
        ctx: CaseContext,
        diagnoses: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        issue_set = {
            str(d.get("issue", "")).strip().lower()
            for d in diagnoses
            if isinstance(d, dict)
        }
        # Hard single-component relabel takes precedence over forced atlas.
        if {"fragmented_lv", "missing_lv_fragments", "fragmented_rv", "missing_rv_fragments"} & issue_set:
            return None

        planner_cfg = self.cfg.get("planner", {}) or {}
        low_score_th = float(planner_cfg.get("force_atlas_low_score_threshold", 0.25))
        triage_score = float(getattr(ctx, "triage_score", 1.0))
        if triage_score > low_score_th:
            return None

        near_missing = self._detect_near_missing_classes(ctx)
        if not near_missing:
            return None

        missing_names = [self._class_name_from_id(cid) for cid in near_missing]
        reason = (
            f"Policy override: triage_score={triage_score:.3f} <= {low_score_th:.3f}, "
            f"near-missing classes={missing_names}; force class-specific atlas replacement"
        )

        # Replace only the specified class regions (do not globally replace all classes).
        steps: List[Dict[str, Any]] = []
        for cls_name in missing_names:
            hint = self._pick_spatial_hint_for_class(cls_name, diagnoses)
            step: Dict[str, Any] = {
                "operation": f"atlas_{cls_name}_repair",
                "affected_class": cls_name,
                "direction": hint["direction"],
                "reason": reason,
                "risk": "high",
                "policy_tag": "forced_atlas_class_replace",
                "hint_issue": hint["issue"],
            }
            if hint["bbox"] is not None:
                step["bbox"] = hint["bbox"]
            steps.append(
                step
            )
        return steps

    def _pick_fragment_relabel_hint(
        self,
        diagnoses: List[Dict[str, Any]],
        target_class: str,
        issue_tokens: set[str],
    ) -> Dict[str, Any]:
        rows = [d for d in diagnoses if isinstance(d, dict)]
        if not rows:
            return {"bbox": None, "direction": "global", "issue": "", "source": ""}

        def _is_valid_bbox(val: Any) -> bool:
            return isinstance(val, list) and len(val) == 4

        focused_rows = []
        for d in rows:
            issue = str(d.get("issue", "")).strip().lower()
            cls = str(d.get("affected_class", "")).strip().lower()
            if issue in issue_tokens or cls == target_class:
                focused_rows.append(d)
        pool = focused_rows or rows

        def _score(d: Dict[str, Any]) -> tuple[int, int, int, int, int, int]:
            issue = str(d.get("issue", "")).strip().lower()
            cls = str(d.get("affected_class", "")).strip().lower()
            direction = str(d.get("direction", "global")).strip().lower() or "global"
            sev = self._severity_rank(str(d.get("severity", "medium")))
            src = self._source_rank(str(d.get("source", "")))
            match_issue = 0 if issue in issue_tokens else 1
            match_cls = 0 if cls == target_class else 1
            non_global = 0 if direction != "global" else 1
            has_bbox = 0 if _is_valid_bbox(d.get("bbox")) else 1
            return (match_issue, match_cls, src, non_global, has_bbox, sev)

        best = sorted(pool, key=_score)[0]
        bbox = best.get("bbox")
        if not _is_valid_bbox(bbox):
            bbox = None
        direction = normalize_direction_hint(
            str(best.get("direction", "global")),
            description=str(best.get("description", "")),
        )
        return {
            "bbox": bbox,
            "direction": direction,
            "issue": str(best.get("issue", "")).strip(),
            "source": str(best.get("source", "")).strip().lower(),
        }

    def _hard_fragment_relabel_steps(
        self,
        diagnoses: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Hard rule:
        if LLM diagnosis indicates LV/RV split-fragment issue, deterministically
        relabel secondary connected components before other operations.
        """
        rows = [d for d in diagnoses if isinstance(d, dict)]

        def _has_llm_issue(tokens: set[str], target_class: str) -> bool:
            for d in rows:
                issue = str(d.get("issue", "")).strip().lower()
                cls = str(d.get("affected_class", "")).strip().lower()
                src = str(d.get("source", "")).strip().lower()
                if issue in tokens and cls == target_class and src in {"llm", "vlm"}:
                    return True
            return False

        steps: List[Dict[str, Any]] = []

        lv_tokens = {"fragmented_lv", "missing_lv_fragments"}
        if _has_llm_issue(lv_tokens, target_class="lv"):
            hint = self._pick_fragment_relabel_hint(
                diagnoses,
                target_class="lv",
                issue_tokens=lv_tokens,
            )
            if hint["source"] in {"llm", "vlm"} and (
                hint["bbox"] is not None or hint["direction"] != "global"
            ):
                steps.append(
                    {
                        "operation": "lv_split_to_rv",
                        "affected_class": "lv",
                        "direction": hint["direction"],
                        "reason": (
                            "Hard rule: fragmented_lv/missing_lv_fragments diagnosed by LLM/VLM; "
                            "relabel LV component selected by diagnosis hint to RV"
                        ),
                        "risk": "medium",
                        "policy_tag": "hard_single_component_relabel",
                        "hint_issue": hint["issue"],
                        "hint_source": hint["source"],
                    }
                )
                if hint["bbox"] is not None:
                    steps[-1]["bbox"] = hint["bbox"]
            else:
                self.log("  [HardRule] Skip lv_split_to_rv: no LLM/VLM spatial hint")

        rv_tokens = {"fragmented_rv", "missing_rv_fragments"}
        if _has_llm_issue(rv_tokens, target_class="rv"):
            hint = self._pick_fragment_relabel_hint(
                diagnoses,
                target_class="rv",
                issue_tokens=rv_tokens,
            )
            if hint["source"] in {"llm", "vlm"} and (
                hint["bbox"] is not None or hint["direction"] != "global"
            ):
                steps.append(
                    {
                        "operation": "rv_split_to_lv",
                        "affected_class": "rv",
                        "direction": hint["direction"],
                        "reason": (
                            "Hard rule: fragmented_rv/missing_rv_fragments diagnosed by LLM/VLM; "
                            "relabel RV component selected by diagnosis hint to LV"
                        ),
                        "risk": "medium",
                        "policy_tag": "hard_single_component_relabel",
                        "hint_issue": hint["issue"],
                        "hint_source": hint["source"],
                    }
                )
                if hint["bbox"] is not None:
                    steps[-1]["bbox"] = hint["bbox"]
            else:
                self.log("  [HardRule] Skip rv_split_to_lv: no LLM/VLM spatial hint")
        return steps

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

        raw_steps: List[Dict[str, Any]] = []
        raw_steps.extend(self._hard_fragment_relabel_steps(diagnoses))
        raw_steps.extend(raw_plan)

        deduped: List[Dict[str, Any]] = []
        seen_steps = set()
        for step in raw_steps:
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

        issue_set = {
            str(d.get("issue", "")).strip().lower()
            for d in diagnoses
            if isinstance(d, dict)
        }
        has_lv_too_small = "lv_too_small" in issue_set
        has_rv_too_small = "rv_too_small" in issue_set
        has_myo_too_thick = "myo_too_thick" in issue_set

        enriched_plan = []
        for step in deduped:
            # Try to find matching diagnosis to attach bbox
            # Match by: affected_class AND (operation matches OR issue mentioned in reason)
            step_op = step["operation"]
            step_cls = str(step["affected_class"]).strip().lower()
            step_reason = str(step.get("reason", "")).strip().lower()
            step["direction"] = normalize_direction_hint(str(step.get("direction", "global")))

            def _has_bbox(v: Any) -> bool:
                return isinstance(v, list) and len(v) == 4

            best_diag = None
            diag_candidates = []
            for d in diagnoses:
                if not isinstance(d, dict):
                    continue
                d_op = d.get("operation", "")
                d_cls = str(d.get("affected_class", "")).strip().lower()
                d_dir = str(d.get("direction", "global")).strip().lower() or "global"
                d_issue = str(d.get("issue", "")).strip().lower()
                
                # Strict match on class
                if d_cls != step_cls:
                    continue

                # Weak match on op (step op should match diagnosis op)
                op_match = (d_op == step_op)
                issue_match = bool(step_reason and d_issue and (d_issue in step_reason))
                if not (op_match or issue_match):
                    continue

                score = (
                    0 if op_match else 1,
                    0 if issue_match else 1,
                    self._source_rank(str(d.get("source", ""))),
                    0 if d_dir != "global" else 1,
                    0 if _has_bbox(d.get("bbox")) else 1,
                    self._severity_rank(str(d.get("severity", "medium"))),
                )
                diag_candidates.append((score, d))

            if diag_candidates:
                best_diag = sorted(diag_candidates, key=lambda x: x[0])[0][1]

            step_has_bbox = _has_bbox(step.get("bbox"))
            if best_diag:
                if (not step_has_bbox) and _has_bbox(best_diag.get("bbox")):
                    step["bbox"] = best_diag["bbox"]
                step_dir = normalize_direction_hint(str(step.get("direction", "global")))
                diag_dir = normalize_direction_hint(
                    str(best_diag.get("direction", "global")),
                    description=str(best_diag.get("description", "")),
                )
                if diag_dir != "global" and step_dir != diag_dir:
                    step["direction"] = diag_dir

            # Coupled-op policy:
            # Prefer operations that reassign vacated boundary pixels to the
            # anatomically adjacent class instead of background.
            linked_issue = str(best_diag.get("issue", "")).strip().lower() if best_diag else ""
            original_op = step["operation"]
            direction_lower = str(step.get("direction", "")).strip().lower()

            if (
                step["operation"] == "3_erode"
                and linked_issue in {"lv_too_large", "boundary_error"}
                and (has_lv_too_small or has_myo_too_thick)
            ):
                step["operation"] = "3_erode_expand_myo"
            elif step["operation"] == "2_erode" and linked_issue in {"myo_too_thick", "myo_thickness_var"}:
                if has_lv_too_small:
                    step["operation"] = "2_erode_expand_lv"
                elif has_rv_too_small or direction_lower == "septal":
                    step["operation"] = "2_erode_expand_rv"
                else:
                    step["operation"] = "2_erode_expand_lv"

            if original_op != step["operation"]:
                note = f"Coupled op: {original_op} -> {step['operation']}"
                if step.get("reason"):
                    step["reason"] = f"{step['reason']} [{note}]"
                else:
                    step["reason"] = note
            
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
        final_plan: List[Dict[str, Any]] = []
        seen_final = set()
        for step in deduped:
            step_key = (
                str(step.get("operation", "")).strip(),
                str(step.get("affected_class", "")).strip().lower(),
                str(step.get("direction", "")).strip().lower(),
            )
            if step_key in seen_final:
                continue
            final_plan.append(step)
            seen_final.add(step_key)
        deduped = final_plan

        hard_first_ops = {"lv_split_to_rv", "rv_split_to_lv"}
        hard_steps = [s for s in deduped if s.get("operation") in hard_first_ops]
        normal_steps = [s for s in deduped if s.get("operation") not in hard_first_ops]

        # Hard-rule mode: when split-relabel op is present, run it directly.
        # Additional fixes can be handled in subsequent rounds after re-diagnosis.
        if hard_steps:
            return hard_steps

        if not normal_steps:
            return hard_steps

        has_critical_missing = self._has_critical_missing_issue(diagnoses)
        atlas_steps = [s for s in normal_steps if self._is_atlas_op(s["operation"])]
        non_atlas_steps = [s for s in normal_steps if not self._is_atlas_op(s["operation"])]

        if not atlas_steps:
            return hard_steps + normal_steps

        if has_critical_missing:
            return hard_steps + atlas_steps + non_atlas_steps

        ordered_normal = (
            non_atlas_steps + atlas_steps[:1]
            if non_atlas_steps
            else atlas_steps[:1]
        )
        return hard_steps + ordered_normal
