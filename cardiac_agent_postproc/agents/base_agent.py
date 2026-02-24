"""
BaseAgent: Abstract base class for all LLM-powered agents.

Every agent in the system inherits from BaseAgent.  It provides:
- An identity / system prompt (role)
- Access to the shared LLM client
- Conversation memory for multi-turn reasoning
- Standard think/act/respond_to interface
- Connection to the MessageBus for inter-agent communication
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..api_client import OpenAICompatClient
from ..settings import LLMSettings
from .message_bus import AgentMessage, CaseContext, MessageBus, MessageType
from .visual_knowledge import KnowledgeRecord, RepairComparisonKB, VisualKnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Rolling conversation memory for an agent."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    max_messages: int = 40  # keep last N messages to avoid context overflow

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        # Trim oldest (keep system prompt + recent)
        if len(self.messages) > self.max_messages:
            # Always keep the first message (system prompt) if it exists
            if self.messages and self.messages[0]["role"] == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_messages - 1):]
            else:
                self.messages = self.messages[-self.max_messages:]

    def to_messages(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def clear(self) -> None:
        """Clear memory, keeping only the system prompt if present."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def last_n(self, n: int) -> List[Dict[str, str]]:
        """Get the last N messages."""
        return self.messages[-n:]


class BaseAgent(ABC):
    """
    Abstract base class for all LLM-powered agents.

    Subclasses must implement:
    - name (property): unique agent identifier
    - system_prompt (property): the agent's persona/instructions
    - process_case(case_ctx): main processing logic for a single case
    """

    def __init__(
        self,
        cfg: dict,
        bus: MessageBus,
        visual_kb: Optional[VisualKnowledgeBase] = None,
        repair_kb: Optional[RepairComparisonKB] = None,
    ):
        self.cfg = cfg
        self.bus = bus
        self.memory = Memory()
        self.visual_kb = visual_kb
        self.repair_kb = repair_kb
        self._llm_client: Optional[OpenAICompatClient] = None
        self._vlm_client: Optional[OpenAICompatClient] = None

        # Register with bus
        self.bus.register_agent(self.name)

        # Initialize system prompt in memory
        self.memory.add("system", self.system_prompt)

    # ---- Identity (must be implemented by subclasses) ----

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name, e.g. 'triage', 'diagnosis'."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's persona / system instructions."""
        ...

    @abstractmethod
    def process_case(self, ctx: CaseContext) -> None:
        """
        Main entry point: process a single case.
        Should read from ctx, update ctx, and send messages via bus.
        """
        ...

    # ---- Prompt loading ----

    @staticmethod
    def _prompt_dirs() -> List[Path]:
        """Candidate prompt directories (cwd first, then project root)."""
        cwd_prompts = Path.cwd() / "prompts"
        project_prompts = Path(__file__).resolve().parents[2] / "prompts"
        dirs: List[Path] = []
        for p in (cwd_prompts, project_prompts):
            if p not in dirs:
                dirs.append(p)
        return dirs

    def _read_prompt_file(self, filename: str) -> Optional[str]:
        """
        Read a prompt text file from prompt directories.
        Returns None when file is missing or unreadable.
        """
        for prompt_dir in self._prompt_dirs():
            prompt_path = prompt_dir / filename
            if not prompt_path.exists():
                continue
            try:
                text = prompt_path.read_text(encoding="utf-8").strip()
                if text:
                    return text
                logger.warning("[%s] Prompt file is empty: %s", self.name, prompt_path)
                return None
            except Exception as e:
                logger.warning("[%s] Failed to read prompt file %s: %s", self.name, prompt_path, e)
                return None
        return None

    def _prompt_with_fallback(self, filename: str, fallback: str) -> str:
        """Load prompt from file, or use fallback text if file is missing."""
        loaded = self._read_prompt_file(filename)
        return loaded if loaded else fallback

    # ---- LLM Clients (lazy initialization) ----

    def _agent_cfg(self) -> Dict[str, Any]:
        agents_cfg = self.cfg.get("agents", {})
        if not isinstance(agents_cfg, dict):
            return {}
        agent_cfg = agents_cfg.get(self.name, {})
        return agent_cfg if isinstance(agent_cfg, dict) else {}

    def _llm_cfg(self) -> Dict[str, Any]:
        llm_cfg = dict(self.cfg.get("llm", {}))
        agent_cfg = self._agent_cfg()
        for k in ("provider", "base_url", "api_key", "model", "timeout", "temperature", "max_tokens"):
            if k in agent_cfg:
                llm_cfg[k] = agent_cfg[k]
        return llm_cfg

    def _vlm_cfg(self) -> Dict[str, Any]:
        vg_cfg = dict(self.cfg.get("vision_guardrail", {}))
        agent_cfg = self._agent_cfg()
        for k in ("provider", "base_url", "api_key", "model", "timeout", "image_detail", "temperature", "max_tokens"):
            if k in agent_cfg:
                vg_cfg[k] = agent_cfg[k]
        return vg_cfg

    def _llm_enabled(self) -> bool:
        llm_cfg = self._llm_cfg()
        if "enabled" in llm_cfg:
            return bool(llm_cfg.get("enabled"))
        return bool(self.cfg.get("llm", {}).get("enabled", True))

    def _vlm_enabled(self) -> bool:
        vg_cfg = self._vlm_cfg()
        if "enabled" in vg_cfg:
            return bool(vg_cfg.get("enabled"))
        return bool(self.cfg.get("vision_guardrail", {}).get("enabled", True))

    @property
    def llm(self) -> OpenAICompatClient:
        """Lazy-init LLM client from config."""
        if self._llm_client is None:
            llm_cfg = self._llm_cfg()
            settings = LLMSettings()
            self._llm_client = OpenAICompatClient(
                base_url=llm_cfg.get("base_url", settings.openai_base_url),
                api_key=llm_cfg.get("api_key", settings.openai_api_key),
                model=llm_cfg.get("model", settings.openai_model),
                timeout=float(llm_cfg.get("timeout", 180.0)),
                provider=llm_cfg.get("provider", settings.llm_provider),
            )
        return self._llm_client

    @property
    def vlm(self) -> OpenAICompatClient:
        """Lazy-init VLM client from config."""
        if self._vlm_client is None:
            vg_cfg = self._vlm_cfg()
            settings = LLMSettings()
            self._vlm_client = OpenAICompatClient(
                base_url=vg_cfg.get("base_url", settings.openai_base_url),
                api_key=vg_cfg.get("api_key", settings.openai_api_key),
                model=vg_cfg.get("model", settings.openai_model),
                timeout=float(vg_cfg.get("timeout", 180.0)),
                provider=vg_cfg.get("provider", settings.llm_provider),
            )
        return self._vlm_client

    # ---- Core Reasoning ----

    def think(self, user_prompt: str, temperature: float = 0.2) -> str:
        """
        Core reasoning method: sends the user prompt to the LLM
        with the agent's full memory context.

        Returns the raw LLM response text.
        Appends both the prompt and response to memory.
        """
        self.memory.add("user", user_prompt)

        if not self._llm_enabled():
            self.log("LLM disabled by config; skip text reasoning")
            self.memory.add("assistant", "{}")
            return "{}"

        llm_cfg = self._llm_cfg()
        temp = llm_cfg.get("temperature", temperature)
        max_tokens = llm_cfg.get("max_tokens", 8192)

        # Build messages from memory
        messages = self.memory.to_messages()

        try:
            response = self.llm.chat_json(
                system=messages[0]["content"] if messages and messages[0]["role"] == "system" else self.system_prompt,
                user=user_prompt,
                temperature=temp,
                max_tokens=max_tokens,
            )
            # chat_json returns a dict; convert to string for memory
            response_text = json.dumps(response) if isinstance(response, dict) else str(response)
        except Exception as e:
            print(f"[{self.name}] LLM call failed: {e}")
            response_text = "{}"

        self.memory.add("assistant", response_text)
        return response_text

    def think_json(self, user_prompt: str, temperature: float = 0.15) -> Dict[str, Any]:
        """
        Like think(), but returns parsed JSON.
        Falls back to empty dict on parse failure.
        """
        self.memory.add("user", user_prompt)

        if not self._llm_enabled():
            self.log("LLM disabled by config; skip JSON reasoning")
            self.memory.add("assistant", "{}")
            return {}

        llm_cfg = self._llm_cfg()
        temp = llm_cfg.get("temperature", temperature)
        max_tokens = llm_cfg.get("max_tokens", 8192)

        result = self.llm.chat_json(
            system=self.system_prompt,
            user=user_prompt,
            temperature=temp,
            max_tokens=max_tokens,
        )

        self.memory.add("assistant", json.dumps(result))
        return result

    def think_vision(
        self,
        user_prompt: str,
        image_path: str,
        temperature: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Multimodal reasoning: send text + image to VLM.
        Returns parsed JSON response.
        """
        self.memory.add("user", f"[IMAGE: {image_path}] {user_prompt}")

        if not self._vlm_enabled():
            self.log("VLM disabled by config; skip vision reasoning")
            self.memory.add("assistant", "{}")
            return {}

        vg_cfg = self._vlm_cfg()
        image_detail = vg_cfg.get("image_detail", "auto")

        result = self.vlm.chat_vision_json(
            system=self.system_prompt,
            user_text=user_prompt,
            image_path=image_path,
            temperature=temperature,
            image_detail=image_detail,
        )

        self.memory.add("assistant", json.dumps(result))
        return result

    # ---- VLM-Based Quality Judgment (replaces RQS) ----

    _JUDGE_PROMPT = (
        "You are a cardiac MRI segmentation quality judge.\n"
        "I will show REFERENCE and BAD reference overlays, then one TARGET overlay.\n\n"
        "Images labeled [REFERENCE: SCORE 10/10] are curated best cases representing perfect quality. "
        "Calibrate your scores relative to these anchors.\n\n"
        "All images are SINGLE-PANEL overlays on raw MRI.\n"
        "Color legend: RV=red/cyan, Myocardium ring=green/yellow, LV=blue/red-brown.\n\n"
        "Task: classify TARGET as 'good', 'borderline', or 'bad'.\n"
        "The REFERENCE images are 10/10; score the TARGET relative to them.\n"
        "Judge by morphology only (do NOT guess exact Dice from appearance).\n\n"
        "View-aware anatomy rules:\n"
        "- RV is often more trabeculated and crescent/triangular compared with LV.\n"
        "- LV is usually smoother and more compact than RV.\n"
        "- Interventricular septum separates RV and LV.\n"
        "- Myo is LV myocardium; do not require Myo to surround RV.\n"
        "- SAX mid-slice: LV should be enclosed or near-enclosed by Myo.\n"
        "- LAX/basal/apical slices: do NOT force strict 360-degree Myo closure.\n\n"
        "Scoring checklist (0-10 each):\n"
        "- ring_integrity: Is Myo continuity plausible for this view/slice?\n"
        "- separation: Are RV and LV separated by Myo (no direct merge)?\n"
        "- structure_presence: Are required structures present and coherent?\n"
        "- contour_quality: Are boundaries smooth with limited holes/noise?\n\n"
        "Overall score rules:\n"
        "- Provide a calibrated score in [1, 10] (one decimal allowed)\n"
        "- Use the full score range; do not collapse to only a few fixed values\n"
        "- Minor roughness with closed ring and clean separation is still often GOOD (typically 7.0-8.5)\n"
        "- Reserve 1-3 for severe structural/anatomical failure\n\n"
        "Quality mapping:\n"
        "- good: score >= 7.0 and no major structural violation\n"
        "- borderline: 4.5 <= score < 7.0\n"
        "- bad: score < 4.5 OR major violation (large Myo gap, RV/LV merge, missing critical structure)\n\n"
        "Respond in JSON only with keys quality, score, confidence, issues, subscores, reasoning.\n"
        '{"quality":"good"|"borderline"|"bad",'
        '"score":1-10,'
        '"confidence":0.0-1.0,'
        '"issues":["specific issues"],'
        '"subscores":{"ring_integrity":0-10,"separation":0-10,"structure_presence":0-10,"contour_quality":0-10},'
        '"reasoning":"brief explanation"}'
    )


    _COMPARE_PROMPT = (
        "You are a cardiac MRI segmentation quality judge.\n\n"
        "I will show you a BEFORE and AFTER overlay of the same case.\n"
        "Determine if the repair IMPROVED the segmentation.\n\n"
        "Each overlay is a single-panel segmentation overlay on MRI.\n"
        "Color legend: Cyan=RV, Yellow ring=Myo, Red-brown=LV\n\n"
        "Use view-aware anatomy judgment (SAX closure stricter, LAX/basal/apical more conservative).\n\n"
        "CRITICAL: Count as IMPROVED if:\n"
        "- A structural defect is fixed (e.g., a gap in the yellow ring is closed).\n"
        "- A missing part is restored.\n"
        "- Gross errors (like giant blobs) are reduced.\n\n"
        "ACCEPTABLE TRADE-OFFS (still IMPROVED):\n"
        "- Fixing a gap introduces slight roughness or minor noise.\n"
        "- Restoring a missing part makes the boundary slightly incorrect.\n"
        "- Better anatomy is worth a few incorrect pixels.\n\n"
        "Respond in JSON only with keys verdict, confidence, before_quality, after_quality, changes_observed, reasoning.\n"
        '{"verdict": "improved"|"same"|"worse", '
        '"confidence": 0.0-1.0, '
        '"before_quality": "good"|"bad"|"borderline", '
        '"after_quality": "good"|"bad"|"borderline", '
        '"changes_observed": ["list of changes"], '
        '"reasoning": "brief explanation"}'
    )

    def judge_visual_quality(
        self,
        overlay_path: str,
        n_refs: int = 1,
        view_type: Optional[str] = None,
        seed: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Use VLM to judge segmentation quality by comparing the target overlay
        against few-shot good/bad reference examples.

        Returns:
            {"quality": "good"|"bad"|"borderline",
             "confidence": float,
             "issues": [...],
             "reasoning": str}
        """
        if self.visual_kb is None or not overlay_path:
            logger.warning("judge_visual_quality called without visual_kb or overlay_path")
            return {"quality": "borderline", "confidence": 0.0, "issues": [], "reasoning": "no visual knowledge base"}
        if not self._vlm_enabled():
            return {
                "quality": "borderline",
                "confidence": 0.0,
                "issues": [],
                "reasoning": "vlm disabled by config",
            }

        # Build image list with labels
        image_paths = []
        image_labels = []
        strict_no_gt_online = bool(
            (self.cfg.get("multi_agent", {}) or {}).get("strict_no_gt_online", False)
        )

        n_each = max(1, n_refs // 2)
        good_refs, bad_refs = self.visual_kb.get_few_shot_set(
            n_each, n_each, view_type=view_type, seed=seed
        )
        for i, ex in enumerate(good_refs):
            if strict_no_gt_online:
                image_labels.append(f"[REFERENCE: GOOD #{i+1}]")
            else:
                image_labels.append(f"[REFERENCE: SCORE 10/10 #{i+1}] Dice={ex.dice:.3f}")
            image_paths.append(ex.path)

        for i, ex in enumerate(bad_refs):
            if strict_no_gt_online:
                image_labels.append(f"[REFERENCE: BAD #{i+1}]")
            else:
                image_labels.append(
                    f"[BAD EXAMPLE {i+1}] Dice={ex.dice:.3f}, Issue={ex.issue_type}"
                )
            image_paths.append(ex.path)

        image_labels.append("[TARGET - EVALUATE THIS]")
        image_paths.append(overlay_path)

        # Add knowledge context to prompt
        knowledge_text = ""
        if not strict_no_gt_online:
            knowledge_text = self.visual_kb.build_knowledge_prompt()

        # Inject view-specific rules so VLM doesn't penalise absent RV in 2CH
        view_addendum = ""
        if view_type and "2ch" in str(view_type).lower():
            view_addendum = (
                "\n\nIMPORTANT — THIS IS A 2CH (2-CHAMBER LONG-AXIS) VIEW:\n"
                "In 2CH, the RIGHT VENTRICLE (RV) is anatomically ABSENT. "
                "Do NOT penalise for missing RV — its absence is correct and expected.\n"
                "Required structures for 2CH: Myocardium (green/yellow) and LV (blue) ONLY.\n"
                "- structure_presence: judge based on Myo + LV only; RV absence is NORMAL.\n"
                "- separation: no RV/LV separation is needed; judge the Myo-LV boundary quality.\n"
                "- ring_integrity: Myo does NOT form a closed ring in 2CH (LAX view); "
                "partial coverage along LV wall is acceptable and correct.\n"
                "Do NOT list 'missing RV', 'missing RV label', or 'no RV' as issues in 2CH.\n"
            )

        base_prompt = self._JUDGE_PROMPT + view_addendum
        full_prompt = f"{knowledge_text}\n\n{base_prompt}" if knowledge_text else base_prompt

        self.log(
            f"VLM judge: {len(good_refs)} good + {len(bad_refs)} bad refs "
            f"(view={view_type or 'any'}) -> evaluating {overlay_path}"
        )

        vg_cfg = self._vlm_cfg()
        image_detail = vg_cfg.get("image_detail", "auto")

        result = self.vlm.chat_vision_multi_json(
            system="You are an expert cardiac MRI segmentation quality assessor.",
            user_text=full_prompt,
            image_paths=image_paths,
            image_labels=image_labels,
            temperature=0.1,
            image_detail=image_detail,
        )
        print(f"DEBUG: Raw VLM Response: {result}", flush=True)

        # Normalize result
        quality = result.get("quality", "borderline")
        if quality not in ("good", "bad", "borderline"):
            quality = "borderline"
        result["quality"] = quality
        score_val = result.get("score", 5.0)
        try:
            score_val = float(score_val)
        except Exception:
            score_val = 5.0
        result["score"] = max(1.0, min(10.0, score_val))
        result.setdefault("confidence", 0.5)
        result.setdefault("issues", [])
        result.setdefault("reasoning", "")
        result.setdefault("reason", str(result.get("reasoning", "")))
        matched_example_ids = [
            str(getattr(ex, "example_id", "")).strip()
            for ex in (good_refs + bad_refs)
            if str(getattr(ex, "example_id", "")).strip()
        ]
        if isinstance(result.get("matched_example_ids"), list):
            raw_ids = [str(x).strip() for x in result["matched_example_ids"] if str(x).strip()]
            if raw_ids:
                matched_example_ids = raw_ids
        result["matched_example_ids"] = matched_example_ids

        self.log(f"VLM verdict: {result['quality']} (conf={result['confidence']:.2f})")
        return result

    def judge_visual_comparison(
        self, before_path: str, after_path: str
    ) -> Dict[str, Any]:
        """
        Use VLM to compare before/after overlays and determine if repair improved quality.

        Returns:
            {"verdict": "improved"|"same"|"worse",
             "confidence": float,
             "before_quality": str, "after_quality": str,
             "changes_observed": [...],
             "reasoning": str}
        """
        if not self._vlm_enabled():
            return {
                "verdict": "same",
                "confidence": 0.0,
                "before_quality": "borderline",
                "after_quality": "borderline",
                "changes_observed": [],
                "reasoning": "vlm disabled by config",
            }

        image_paths = [before_path, after_path]
        image_labels = ["[BEFORE repair]", "[AFTER repair]"]

        # Optionally add a good reference for calibration
        if self.visual_kb and self.visual_kb.good_examples:
            good_ref = self.visual_kb.get_good_examples(1)[0]
            image_paths.insert(0, good_ref.path)
            image_labels.insert(0, f"[REFERENCE: SCORE 10/10] Dice={good_ref.dice:.3f}")

        self.log(f"VLM comparison: before={before_path} vs after={after_path}")

        vg_cfg = self._vlm_cfg()
        image_detail = vg_cfg.get("image_detail", "auto")

        result = self.vlm.chat_vision_multi_json(
            system="You are an expert cardiac MRI segmentation quality assessor.",
            user_text=self._COMPARE_PROMPT,
            image_paths=image_paths,
            image_labels=image_labels,
            temperature=0.1,
            image_detail=image_detail,
        )

        # Normalize
        verdict = result.get("verdict", "same")
        if verdict not in ("improved", "same", "worse"):
            verdict = "same"
        result["verdict"] = verdict
        result.setdefault("confidence", 0.5)
        result.setdefault("changes_observed", [])
        result.setdefault("reasoning", "")

        self.log(f"VLM comparison verdict: {result['verdict']} (conf={result['confidence']:.2f})")
        return result

    # ---- Repair Comparison (diff-overlay based) ----

    _REPAIR_COMPARE_PROMPT = (
        "You are a cardiac MRI segmentation repair quality judge.\n\n"
        "TASK:\n"
        "1) Read the DIFF panel header text: DIFF +N -M ~K\n"
        "2) Verify with morphology in BEFORE/AFTER/DIFF panels\n"
        "3) Return calibrated relative-improvement score.\n\n"
        "DIFF header meaning:\n"
        "+N = pixels ADDED, -M = pixels REMOVED, ~K = class-changed pixels.\n\n"
        "Output contract (JSON only, no extra keys required):\n"
        "{\n"
        '  "verdict": "improved" | "degraded" | "neutral",\n'
        '  "score": 0-100,\n'
        '  "quality": "better" | "same" | "worse",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "matched_example_ids": ["..."],\n'
        '  "reason": "brief evidence-based explanation",\n'
        '  "diff_stats": "+N -M ~K"\n'
        "}\n\n"
        "Scoring semantics:\n"
        "- 0-39: confident degradation\n"
        "- 40-59: uncertain/neutral\n"
        "- 60-100: confident improvement\n"
    )

    @staticmethod
    def _default_relative_score(verdict: str, confidence: float) -> float:
        conf = max(0.0, min(1.0, float(confidence)))
        v = str(verdict).lower()
        if v == "improved":
            return 50.0 + 50.0 * conf
        if v == "degraded":
            return 50.0 - 50.0 * conf
        return 50.0

    @staticmethod
    def _quality_from_verdict(verdict: str) -> str:
        v = str(verdict).lower()
        if v == "improved":
            return "better"
        if v == "degraded":
            return "worse"
        return "same"

    @staticmethod
    def _render_evidence_table(
        records: Sequence[KnowledgeRecord],
        include_prior: bool = True,
    ) -> str:
        if not records:
            return "No retrieval evidence."
        lines = [
            "Evidence table (Top-K retrieved examples):",
            "id | outcome | view | defects | actions | prior",
        ]
        for rec in records:
            prior_txt = f"{float(rec.confidence_prior):.3f}" if include_prior else "N/A"
            lines.append(
                f"{rec.example_id} | {rec.expected_outcome} | {rec.view_type} | "
                f"{','.join(rec.defect_tags) or '-'} | "
                f"{','.join(rec.action_tags) or '-'} | "
                f"{prior_txt}"
            )
        return "\n".join(lines)

    def judge_repair_comparison(
        self,
        diff_overlay_path: str,
        applied_ops: Optional[List[str]] = None,
        view_type: Optional[str] = None,
        defect_tags: Optional[List[str]] = None,
        top_k_refs: int = 2,
        use_confidence_prior: bool = True,
    ) -> Dict[str, Any]:
        """
        VLM comparison on one diff-overlay with typed, deterministic KB retrieval.

        Returns:
            {
              "verdict": "improved"|"degraded"|"neutral",
              "score": 0-100,
              "quality": "better"|"same"|"worse",
              "confidence": 0-1,
              "matched_example_ids": [...],
              "reason": "...",
              "diff_stats": "+... -... ~..."
            }
        """
        if not self._vlm_enabled():
            return {
                "verdict": "neutral",
                "score": 50.0,
                "quality": "same",
                "confidence": 0.0,
                "matched_example_ids": [],
                "reason": "vlm disabled by config",
                "reasoning": "vlm disabled by config",
            }

        image_paths: List[str] = []
        image_labels: List[str] = []
        selected_records: List[KnowledgeRecord] = []
        selected_ids: List[str] = []
        strict_no_gt_online = bool(
            (self.cfg.get("multi_agent", {}) or {}).get("strict_no_gt_online", False)
        )

        tags = list(defect_tags or [])
        if applied_ops:
            tags.extend([str(op) for op in applied_ops if str(op).strip()])

        if self.repair_kb and self.repair_kb.total > 0:
            selected_records = self.repair_kb.retrieve(
                stage="repair_verification",
                view_type=view_type,
                defect_tags=tags,
                top_k=max(1, int(top_k_refs)),
                use_confidence_prior=(
                    bool(use_confidence_prior) and (not strict_no_gt_online)
                ),
            )
            selected_ids = [rec.example_id for rec in selected_records]
            ref_examples = self.repair_kb.get_examples_by_ids(selected_ids)
            for ex in ref_examples:
                label = f"[REFERENCE {ex.example_id}] outcome={ex.category}"
                if not strict_no_gt_online:
                    label += f" delta={ex.dice_delta:+.3f}"
                image_labels.append(label)
                image_paths.append(ex.path)

        image_labels.append("[TARGET - EVALUATE THIS REPAIR]")
        image_paths.append(diff_overlay_path)

        evidence_table = self._render_evidence_table(
            selected_records,
            include_prior=(not strict_no_gt_online),
        )
        prompt = (
            f"{self._REPAIR_COMPARE_PROMPT}\n"
            f"{evidence_table}\n\n"
            "Use matched_example_ids from the evidence table only.\n"
        )

        self.log(
            f"VLM repair comparison: {diff_overlay_path} "
            f"(references={len(image_paths)-1})"
        )

        vg_cfg = self._vlm_cfg()
        image_detail = vg_cfg.get("image_detail", "auto")
        result = self.vlm.chat_vision_multi_json(
            system="You are a cardiac MRI segmentation repair quality judge.",
            user_text=prompt,
            image_paths=image_paths,
            image_labels=image_labels,
            temperature=0.1,
            image_detail=image_detail,
        )

        verdict = str(result.get("verdict", "neutral")).lower()
        if verdict not in ("improved", "degraded", "neutral"):
            verdict = "neutral"
        result["verdict"] = verdict

        try:
            confidence = float(result.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        result["confidence"] = confidence

        score = result.get("score")
        try:
            score_val = float(score)
        except Exception:
            score_val = self._default_relative_score(verdict, confidence)
        score_val = max(0.0, min(100.0, score_val))
        result["score"] = score_val
        result["quality"] = str(result.get("quality", self._quality_from_verdict(verdict)))
        if result["quality"] not in ("better", "same", "worse"):
            result["quality"] = self._quality_from_verdict(verdict)

        reason = result.get("reason", result.get("reasoning", ""))
        if not isinstance(reason, str):
            reason = str(reason)
        result["reason"] = reason
        result["reasoning"] = reason

        matched_ids = result.get("matched_example_ids", [])
        if not isinstance(matched_ids, list):
            matched_ids = []
        matched_ids = [str(x).strip() for x in matched_ids if str(x).strip()]
        valid_ids = set(selected_ids)
        matched_ids = [x for x in matched_ids if x in valid_ids]
        if not matched_ids:
            matched_ids = list(selected_ids)
        result["matched_example_ids"] = matched_ids
        result["selected_example_ids"] = list(selected_ids)

        if "diff_stats" in result:
            self.log(f"VLM read stats: {result['diff_stats']}")
            stats_str = str(result["diff_stats"])
            match_add = re.search(r"\+(\d+)", stats_str)
            match_rem = re.search(r"-(\d+)", stats_str)
            match_chg = re.search(r"~(\d+)", stats_str)

            if match_add and match_rem:
                n_add = int(match_add.group(1))
                n_rem = int(match_rem.group(1))
                n_chg = int(match_chg.group(1)) if match_chg else 0

                calc_verdict = "neutral"
                is_swapper = False
                swapper_ops = [
                    "erode_expand_myo",
                    "neighbor_repair",
                    "neighbor_rv_repair",
                    "neighbor_myo_repair",
                    "atlas",
                    "1_erode_expand_myo",
                    "3_erode_expand_myo",
                    "lv_split_to_rv",
                    "rv_split_to_lv",
                ]
                if applied_ops:
                    for op in applied_ops:
                        op_name = str(op)
                        if any(s in op_name for s in swapper_ops):
                            is_swapper = True
                            break

                is_pure_relabel = (
                    is_swapper and n_add <= 10 and n_rem <= 10 and n_chg > 0
                )

                total_delta = n_add + n_rem + n_chg
                # Suppress severe_removal when a class-swapping op produced substantial
                # relabeling (n_chg ≥ 30): erode-expand legitimately removes one class
                # while converting pixels to another, so large n_rem is expected.
                severe_removal = (
                    (n_rem > n_add + 50)
                    and (not is_pure_relabel)
                    and not (is_swapper and n_chg >= 30)
                )
                severe_relabel_noise = (
                    (not is_pure_relabel)
                    and n_chg > (n_add + (500 if is_swapper else 200))
                )
                strong_relabel_fix = is_pure_relabel and n_chg >= 120
                strong_additive_fix = (
                    (not is_pure_relabel)
                    and n_add >= max(120, 3 * n_rem)
                    and n_rem <= 20
                    and n_chg <= max(40, int(0.5 * n_add))
                )

                if severe_removal or severe_relabel_noise:
                    calc_verdict = "degraded"
                elif strong_relabel_fix or strong_additive_fix:
                    calc_verdict = "improved"
                elif total_delta < 30:
                    calc_verdict = "neutral"

                current_verdict = str(result.get("verdict", "neutral"))
                override_verdict = current_verdict
                # Conservative override policy:
                # - Always allow override to degraded on strong regression evidence.
                # - Pull optimistic improved back to neutral when evidence is weak.
                # - Never flip degraded directly to improved via heuristics.
                if calc_verdict == "degraded":
                    override_verdict = "degraded"
                elif calc_verdict == "neutral" and current_verdict == "improved":
                    override_verdict = "neutral"
                elif (
                    calc_verdict == "improved"
                    and current_verdict == "neutral"
                ):
                    override_verdict = "improved"

                # Rescue swapper-op false-positive degraded verdicts:
                # When a class-swapping op (erode_expand_myo, split_relabel, etc.)
                # produces substantial relabeling (n_chg ≥ 30) with few or no
                # outright removals, VLM's "degraded" is almost certainly a false
                # positive — it sees class-changed pixels and mistakes them for
                # misclassification. Demote to "neutral" so the proxy consensus
                # and anatomy signals can drive the final verdict.
                if (
                    is_swapper
                    and n_chg >= 30
                    and n_rem <= n_chg * 0.5
                    and override_verdict == "degraded"
                ):
                    override_verdict = "neutral"

                if override_verdict != current_verdict:
                    self.log(
                        f"LOGIC OVERRIDE: {current_verdict} -> {override_verdict} "
                        f"(based on +{n_add} -{n_rem} ~{n_chg}, swapper={is_swapper})"
                    )
                    result["original_verdict"] = current_verdict
                    result["verdict"] = override_verdict
                    result["quality"] = self._quality_from_verdict(override_verdict)
                    result["score"] = self._default_relative_score(
                        override_verdict, result["confidence"]
                    )
                    result["reason"] = (
                        f"{result.get('reason', '')} "
                        "[Auto-Correction: logic override based on diff stats]"
                    ).strip()
                    result["reasoning"] = result["reason"]

        self.log(
            f"VLM repair verdict: {result['verdict']} "
            f"(score={result['score']:.1f}, conf={result['confidence']:.2f})"
        )
        return result


    def send_message(
        self,
        recipient: str,
        msg_type: MessageType,
        content: Dict[str, Any],
        case_id: str = "",
    ) -> None:
        """Send a message to another agent via the bus."""
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            msg_type=msg_type,
            content=content,
            case_id=case_id,
        )
        self.bus.send(msg)
        print(f"  {msg.summary()}", flush=True)

    def receive_messages(
        self,
        msg_type: Optional[MessageType] = None,
        case_id: Optional[str] = None,
    ) -> List[AgentMessage]:
        """Receive pending messages from the bus."""
        return self.bus.receive(self.name, msg_type=msg_type, case_id=case_id)

    def respond_to(self, msg: AgentMessage) -> Optional[AgentMessage]:
        """
        Respond to an incoming message from another agent.
        Default: use LLM to generate a response.
        Subclasses can override for custom behavior.
        """
        prompt = (
            f"You received a {msg.msg_type.value} message from {msg.sender}:\n"
            f"{json.dumps(msg.content, indent=2)}\n\n"
            f"Provide your response as JSON."
        )
        response = self.think_json(prompt)
        print(f"DEBUG: Raw VLM Response: {response}", flush=True)

        reply = AgentMessage(
            sender=self.name,
            recipient=msg.sender,
            msg_type=MessageType.CHALLENGE_RESPONSE,
            content=response,
            case_id=msg.case_id,
        )
        self.bus.send(reply)
        return reply

    # ---- Utilities ----

    def log(self, message: str) -> None:
        """Print a log line with agent name prefix."""
        print(f"[{self.name}] {message}", flush=True)

    def reset_memory(self) -> None:
        """Clear conversation memory (keep system prompt)."""
        self.memory.clear()
