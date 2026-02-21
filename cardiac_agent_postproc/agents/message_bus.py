"""
MessageBus: Inter-agent communication system for the multi-agent pipeline.

Provides structured message passing, shared case context, and audit trailing
for agent-to-agent coordination and negotiation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    # Workflow
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"

    # Triage
    TRIAGE_RESULT = "triage_result"
    CHALLENGE = "challenge"
    CHALLENGE_RESPONSE = "challenge_response"

    # Diagnosis
    DIAGNOSIS_RESULT = "diagnosis_result"
    REQUEST_SECOND_OPINION = "request_second_opinion"
    SECOND_OPINION = "second_opinion"

    # Planning
    REPAIR_PLAN = "repair_plan"
    PLAN_REVISION = "plan_revision"

    # Execution
    STEP_COMPLETE = "step_complete"
    OP_FAILED = "op_failed"

    # Verification
    VERIFY_RESULT = "verify_result"
    NEEDS_MORE_WORK = "needs_more_work"
    DISAGREE_WITH_TRIAGE = "disagree_with_triage"

    # Coordination
    BATCH_SUMMARY = "batch_summary"
    STATUS_UPDATE = "status_update"


@dataclass
class AgentMessage:
    """A single message between agents."""
    sender: str
    recipient: str
    msg_type: MessageType
    content: Dict[str, Any]
    case_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary for logging."""
        return f"[{self.sender}→{self.recipient}] {self.msg_type.value} case={self.case_id}"


@dataclass
class CaseContext:
    """Shared mutable state for a single case being processed."""
    case_id: str
    stem: str
    img_path: str
    pred_path: str
    gt_path: str = ""

    # Loaded data (populated lazily)
    mask: Optional[np.ndarray] = None
    image: Optional[np.ndarray] = None
    view_type: str = "4ch"

    # Inter-slice context
    patient_id: str = ""         # e.g. "201"
    slice_index: int = -1        # e.g. 9 from "201_original_lax_4c_009"
    neighbor_masks: Dict[int, np.ndarray] = field(default_factory=dict)  # idx → mask
    neighbor_img_paths: Dict[int, str] = field(default_factory=dict)     # idx → image path

    # Pipeline state
    current_mask: Optional[np.ndarray] = None  # evolves during optimization
    original_mask: Optional[np.ndarray] = None  # immutable copy of initial pred

    # Triage
    triage_category: str = ""  # "good" | "needs_fix"
    triage_score: float = 0.0
    triage_issues: List[str] = field(default_factory=list)

    # Diagnosis
    diagnoses: List[Dict[str, Any]] = field(default_factory=list)

    # Execution
    applied_ops: List[Dict[str, Any]] = field(default_factory=list)
    rqs_history: List[float] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    current_score: Optional[float] = None
    rounds_completed: int = 0

    # Verification
    verified: bool = False
    verifier_approved: bool = False
    verifier_feedback: str = ""
    last_verify_result: Dict[str, Any] = field(default_factory=dict)

    # Executor diagnostics / fallbacks
    executor_high_risk_unlock: bool = False
    executor_high_risk_unlock_reason: str = ""
    executor_candidate_search_used: bool = False
    executor_candidate_search_pick: str = ""
    executor_candidate_search_score: float = 0.0

    # Best-intermediate fallback (for conservative final save policy)
    best_intermediate_mask: Optional[np.ndarray] = None
    best_intermediate_score: float = 0.0
    best_intermediate_round: int = 0
    best_intermediate_verdict: str = "baseline_original"
    best_intermediate_reason: str = ""
    best_intermediate_ops: List[Dict[str, Any]] = field(default_factory=list)
    best_intermediate_is_original: bool = True

    # Metrics (populated during evaluation if GT available)
    dice_macro: float = -1.0
    dice_per_class: Dict[str, float] = field(default_factory=dict)

    # Output directory for debug/intermediate images
    output_dir: str = ""

    def debug_img_path(self, suffix: str) -> str:
        """
        Build a path for a debug/intermediate image.
        If output_dir is set, writes to output_dir/<stem><suffix>.
        Otherwise falls back to img_path directory (legacy behavior).
        """
        import os
        basename = os.path.basename(self.img_path).replace(".png", suffix)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, basename)
        return self.img_path.replace(".png", suffix)


class MessageBus:
    """
    Central message bus for inter-agent communication.

    Thread-safe design (future-proofing for parallel agents).
    Maintains full audit trail per case.
    """

    def __init__(self):
        self._inbox: Dict[str, List[AgentMessage]] = {}  # agent_name -> pending messages
        self._history: List[AgentMessage] = []  # all messages ever sent
        self._cases: Dict[str, CaseContext] = {}  # case_id -> context

    # ---- Message Passing ----

    def send(self, msg: AgentMessage) -> None:
        """Post a message to a specific agent's inbox."""
        if msg.recipient not in self._inbox:
            self._inbox[msg.recipient] = []
        self._inbox[msg.recipient].append(msg)
        self._history.append(msg)

    def broadcast(self, msg: AgentMessage) -> None:
        """Send a message to all agents (recipient set to '*')."""
        msg.recipient = "*"
        for agent_name in self._inbox:
            self._inbox[agent_name].append(msg)
        self._history.append(msg)

    def receive(
        self,
        agent_name: str,
        msg_type: Optional[MessageType] = None,
        case_id: Optional[str] = None,
    ) -> List[AgentMessage]:
        """
        Pop pending messages for an agent, optionally filtered.
        Returns and removes matched messages from the inbox.
        """
        if agent_name not in self._inbox:
            self._inbox[agent_name] = []
            return []

        inbox = self._inbox[agent_name]
        matched = []
        remaining = []

        for m in inbox:
            match = True
            if msg_type and m.msg_type != msg_type:
                match = False
            if case_id and m.case_id != case_id:
                match = False
            if match:
                matched.append(m)
            else:
                remaining.append(m)

        self._inbox[agent_name] = remaining
        return matched

    def peek(
        self,
        agent_name: str,
        msg_type: Optional[MessageType] = None,
    ) -> List[AgentMessage]:
        """Look at pending messages without removing them."""
        if agent_name not in self._inbox:
            return []
        msgs = self._inbox[agent_name]
        if msg_type:
            return [m for m in msgs if m.msg_type == msg_type]
        return list(msgs)

    def register_agent(self, agent_name: str) -> None:
        """Register an agent so it can receive messages."""
        if agent_name not in self._inbox:
            self._inbox[agent_name] = []

    # ---- Case Context ----

    def register_case(self, ctx: CaseContext) -> None:
        """Register a case context for shared access."""
        self._cases[ctx.case_id] = ctx

    def get_case(self, case_id: str) -> Optional[CaseContext]:
        """Get the shared context for a case."""
        return self._cases.get(case_id)

    def all_cases(self) -> Dict[str, CaseContext]:
        """Get all registered case contexts."""
        return dict(self._cases)

    # ---- Audit Trail ----

    def get_case_history(self, case_id: str) -> List[AgentMessage]:
        """Get all messages related to a specific case."""
        return [m for m in self._history if m.case_id == case_id]

    def get_agent_history(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages sent by a specific agent."""
        return [m for m in self._history if m.sender == agent_name]

    def get_full_history(self) -> List[AgentMessage]:
        """Get the complete message history."""
        return list(self._history)

    def format_case_log(self, case_id: str) -> str:
        """Format the case history as a readable log string."""
        msgs = self.get_case_history(case_id)
        lines = [f"=== Case {case_id} — {len(msgs)} messages ==="]
        for m in msgs:
            lines.append(f"  {m.summary()}: {_truncate(str(m.content), 120)}")
        return "\n".join(lines)


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string with ellipsis."""
    return s if len(s) <= max_len else s[:max_len - 3] + "..."
