"""
Visual Knowledge Base: learns good/bad segmentation patterns from annotated examples.

Loads overlay images from results/Input_MnM2/{best_cases,worst_cases} and provides
few-shot references and text summaries for VLM-based quality judgment.
"""
from __future__ import annotations

import os
import re
import random
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------- Filename pattern ----------
# MnM2: "0.373_4C_LowDice_335_original_lax_4c_009.png"
# UKB:  "0.953_4C_Best_3834376_2_original_lax_4c_024.png"
# Group 2 (view) and group 3 (issue) use [^_]+ to stop at the first underscore,
# preventing the greedy \w+ from consuming underscores and misparsing UKB stems.
# Group 4 (stem) uses \d+.*_original_lax_\w+_\d+ to tolerate extra _{scan_id}
# segments present in UKB patient IDs.
_FNAME_RE = re.compile(
    r"^(\d+\.\d+)_([^_]+)_([^_]+)_(\d+.*_original_lax_\w+_\d+)\.png$"
)


@dataclass(frozen=True)
class KnowledgeRecord:
    """
    Canonical typed knowledge item used for deterministic retrieval.
    """

    example_id: str
    stage: str
    view_type: str
    defect_tags: Tuple[str, ...]
    action_tags: Tuple[str, ...]
    expected_outcome: str
    confidence_prior: float
    artifact_path: str
    source: str

    def compact(self) -> Dict[str, object]:
        return {
            "example_id": self.example_id,
            "stage": self.stage,
            "view_type": self.view_type,
            "defect_tags": list(self.defect_tags),
            "action_tags": list(self.action_tags),
            "expected_outcome": self.expected_outcome,
            "confidence_prior": float(self.confidence_prior),
            "artifact_path": self.artifact_path,
            "source": self.source,
        }


def _norm_view(view: Optional[str]) -> str:
    if not view:
        return "ANY"
    token = re.sub(r"[^a-z0-9]", "", str(view).strip().lower())
    if token in {"2c", "2ch"}:
        return "2CH"
    if token in {"3c", "3ch"}:
        return "3CH"
    if token in {"4c", "4ch"}:
        return "4CH"
    if "sax" in token:
        return "SAX"
    return token.upper() if token else "ANY"


def _norm_tag(tag: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", str(tag).strip().lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return out


def _norm_tags(tags: Sequence[str]) -> Tuple[str, ...]:
    uniq = sorted({_norm_tag(t) for t in tags if _norm_tag(t)})
    return tuple(uniq)


def _parse_view_from_stem(stem: str) -> str:
    s = str(stem).lower()
    if "_lax_2c_" in s:
        return "2CH"
    if "_lax_3c_" in s:
        return "3CH"
    if "_lax_4c_" in s:
        return "4CH"
    return "ANY"


def _default_retrieval(
    records: Sequence[KnowledgeRecord],
    stage: str,
    view_type: Optional[str],
    defect_tags: Sequence[str],
    top_k: int,
    use_confidence_prior: bool = True,
) -> List[KnowledgeRecord]:
    stage_q = _norm_tag(stage) or "any_stage"
    view_q = _norm_view(view_type)
    defect_q = set(_norm_tags(defect_tags))

    top_k = max(1, int(top_k))

    stage_exact = [r for r in records if _norm_tag(r.stage) == stage_q]
    pool = stage_exact if stage_exact else list(records)

    scored: List[Tuple[float, int, int, str, KnowledgeRecord]] = []
    for rec in pool:
        rec_tags = set(rec.defect_tags)
        overlap = len(defect_q.intersection(rec_tags))
        view_match = int(view_q == "ANY" or rec.view_type == "ANY" or rec.view_type == view_q)
        prior = float(rec.confidence_prior) if use_confidence_prior else 0.0
        score = float(overlap) + prior
        scored.append((-score, -overlap, -view_match, rec.example_id, rec))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [item[-1] for item in scored[:top_k]]


@dataclass
class VisualExample:
    """A single annotated overlay image."""
    path: str          # absolute path to the overlay PNG
    dice: float        # e.g. 0.373
    view: str          # e.g. "4C"
    issue_type: str    # "LowDice" | "DiscMyo" | "Best"
    stem: str          # case stem e.g. "335_original_lax_4c_009"
    category: str      # "good" | "bad"
    example_id: str = ""


@dataclass
class VisualKnowledgeBase:
    """
    Collection of curated good/bad segmentation examples.

    Used by agents to:
    1. Select few-shot reference images for VLM prompts
    2. Generate text summaries of quality patterns
    """
    good_examples: List[VisualExample] = field(default_factory=list)
    bad_examples: List[VisualExample] = field(default_factory=list)

    # Index by issue type for targeted lookups
    _by_issue: Dict[str, List[VisualExample]] = field(
        default_factory=dict, repr=False
    )
    records: List[KnowledgeRecord] = field(default_factory=list)
    _record_by_id: Dict[str, KnowledgeRecord] = field(default_factory=dict, repr=False)
    _legacy_warned: ClassVar[bool] = False

    @classmethod
    def load(cls, results_dir: str) -> "VisualKnowledgeBase":
        """
        Load examples from a results directory containing
        best_cases/ and worst_cases/ subdirectories.
        """
        kb = cls()

        best_dir = os.path.join(results_dir, "best_cases")
        worst_dir = os.path.join(results_dir, "worst_cases")

        # Load best (good) examples
        if os.path.isdir(best_dir):
            for fname in sorted(os.listdir(best_dir)):
                ex = cls._parse_filename(fname, best_dir, category="good")
                if ex:
                    kb.good_examples.append(ex)

        # Load worst (bad) examples
        if os.path.isdir(worst_dir):
            for fname in sorted(os.listdir(worst_dir)):
                ex = cls._parse_filename(fname, worst_dir, category="bad")
                if ex:
                    kb.bad_examples.append(ex)
                    issue = ex.issue_type
                    if issue not in kb._by_issue:
                        kb._by_issue[issue] = []
                    kb._by_issue[issue].append(ex)

        # Normalize reference format for VLM triage:
        # generate single-panel overlays (same style as runtime target overlay)
        # from all_frames_export/{stem}_img.png + {stem}_pred.png when available.
        kb._materialize_single_panel_references(results_dir)
        kb._materialize_records()

        logger.info(
            "VisualKnowledgeBase loaded: %d good, %d bad (%s)",
            len(kb.good_examples),
            len(kb.bad_examples),
            ", ".join(f"{k}={len(v)}" for k, v in kb._by_issue.items()),
        )
        return kb

    def _materialize_records(self) -> None:
        records: List[KnowledgeRecord] = []
        rec_by_id: Dict[str, KnowledgeRecord] = {}
        for ex in self.good_examples + self.bad_examples:
            rec = self._record_from_example(ex)
            ex.example_id = rec.example_id
            records.append(rec)
            rec_by_id[rec.example_id] = rec
        self.records = records
        self._record_by_id = rec_by_id

    @staticmethod
    def _issue_tags(issue_type: str, category: str) -> Tuple[str, ...]:
        issue = str(issue_type or "").strip().lower()
        tags = [category]
        if issue == "best":
            tags.extend(["good_anatomy", "closed_ring", "clean_contour"])
        elif issue == "discmyo":
            tags.extend(["disconnected_myo", "myo_gap", "ring_break"])
        elif issue == "lowdice":
            tags.extend(["low_dice", "global_shape_error", "structure_misalignment"])
        else:
            tags.extend([issue, "quality_pattern"])
        return _norm_tags(tags)

    @staticmethod
    def _confidence_prior(dice: float, category: str) -> float:
        d = max(0.0, min(1.0, float(dice)))
        if category == "good":
            return max(0.05, d)
        return max(0.05, 1.0 - d)

    def _record_from_example(self, ex: VisualExample) -> KnowledgeRecord:
        view = _norm_view(ex.view)
        issue = _norm_tag(ex.issue_type or "unknown")
        eid = f"vkb:{ex.category}:{view}:{issue}:{ex.stem}"
        return KnowledgeRecord(
            example_id=eid,
            stage="quality_assessment",
            view_type=view,
            defect_tags=self._issue_tags(ex.issue_type, ex.category),
            action_tags=("inspect_overlay",),
            expected_outcome="good" if ex.category == "good" else "bad",
            confidence_prior=self._confidence_prior(ex.dice, ex.category),
            artifact_path=ex.path,
            source=f"legacy_visual_kb:{ex.category}",
        )

    def retrieve(
        self,
        stage: str,
        view_type: Optional[str],
        defect_tags: Sequence[str],
        top_k: int = 4,
        use_confidence_prior: bool = True,
    ) -> List[KnowledgeRecord]:
        """
        Deterministic retrieval of typed knowledge examples.

        Ranking: tag-overlap + confidence-prior, tie-break by example_id.
        """
        return _default_retrieval(
            self.records,
            stage=stage,
            view_type=view_type,
            defect_tags=defect_tags,
            top_k=top_k,
            use_confidence_prior=use_confidence_prior,
        )

    def get_record(self, example_id: str) -> Optional[KnowledgeRecord]:
        return self._record_by_id.get(example_id)

    def _materialize_single_panel_references(self, results_dir: str) -> None:
        """
        Best-effort conversion of KB examples to single-panel overlays.

        Source assumptions:
          results_dir/
            all_frames_export/{stem}_img.png
            all_frames_export/{stem}_pred.png

        If conversion fails for an example, keep the original path unchanged.
        """
        source_dir = os.path.join(results_dir, "all_frames_export")
        if not os.path.isdir(source_dir):
            logger.warning(
                "VisualKnowledgeBase: source directory missing for single-panel refs: %s",
                source_dir,
            )
            return

        cache_dir = os.path.join(results_dir, "single_panel_refs")
        os.makedirs(cache_dir, exist_ok=True)

        # Local imports to avoid circular import issues at module load time.
        import cv2
        from ..vlm_guardrail import create_overlay

        converted = 0
        total = 0
        for ex in self.good_examples + self.bad_examples:
            total += 1
            out_path = os.path.join(cache_dir, f"{ex.stem}_overlay.png")
            if os.path.exists(out_path):
                ex.path = out_path
                converted += 1
                continue

            img_path = os.path.join(source_dir, f"{ex.stem}_img.png")
            pred_path = os.path.join(source_dir, f"{ex.stem}_pred.png")
            if not (os.path.exists(img_path) and os.path.exists(pred_path)):
                continue

            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred_mask is None:
                continue

            try:
                ok = create_overlay(img_path, pred_mask, out_path)
            except Exception:
                ok = False
            if ok:
                ex.path = out_path
                converted += 1

        logger.info(
            "VisualKnowledgeBase single-panel refs: %d/%d ready (cache=%s)",
            converted,
            total,
            cache_dir,
        )

    @staticmethod
    def _parse_filename(
        fname: str, directory: str, category: str
    ) -> Optional[VisualExample]:
        """Parse an overlay filename into a VisualExample."""
        m = _FNAME_RE.match(fname)
        if not m:
            return None
        if not VisualKnowledgeBase._legacy_warned:
            logger.warning(
                "VisualKnowledgeBase.load: detected legacy filename KB format; "
                "auto-normalizing to typed records."
            )
            VisualKnowledgeBase._legacy_warned = True
        dice_str, view, issue_type, stem = m.groups()
        return VisualExample(
            path=os.path.join(directory, fname),
            dice=float(dice_str),
            view=view,
            issue_type=issue_type,
            stem=stem,
            category=category,
        )

    # ---------- Querying ----------

    def get_good_examples(self, n: int = 2) -> List[VisualExample]:
        """Return n representative good examples (random subset)."""
        return self.get_good_examples_for_view(n=n)

    @staticmethod
    def _normalize_view(view: Optional[str]) -> str:
        """Normalize raw view tags (e.g. 4c, 4ch, 4CH) to a canonical token."""
        out = _norm_view(view)
        return "" if out == "ANY" else out

    @staticmethod
    def _select_examples(
        pool: List[VisualExample],
        n: int,
        seed: Optional[str],
        salt: str,
    ) -> List[VisualExample]:
        """Sample examples. Deterministic when seed is provided."""
        if len(pool) <= n:
            return list(pool)
        if seed is None:
            return random.sample(pool, n)

        ranked = sorted(
            pool,
            key=lambda ex: hashlib.sha256(
                f"{seed}|{salt}|{ex.stem}|{ex.path}".encode("utf-8")
            ).hexdigest(),
        )
        return ranked[:n]

    def get_good_examples_for_view(
        self,
        n: int = 2,
        view_type: Optional[str] = None,
        seed: Optional[str] = None,
    ) -> List[VisualExample]:
        """Return n good examples; prefer same-view examples when available."""
        pool = list(self.good_examples)
        if view_type:
            wanted = self._normalize_view(view_type)
            same_view = [
                ex for ex in pool if self._normalize_view(ex.view) == wanted
            ]
            if same_view:
                primary = self._select_examples(
                    same_view,
                    min(n, len(same_view)),
                    seed,
                    salt=f"good:primary:{wanted}",
                )
                if len(primary) >= n:
                    return primary[:n]

                seen = {ex.path for ex in primary}
                fallback_pool = [ex for ex in pool if ex.path not in seen]
                fallback = self._select_examples(
                    fallback_pool,
                    n - len(primary),
                    seed,
                    salt=f"good:fallback:{wanted}",
                )
                return primary + fallback
        return self._select_examples(pool, n, seed, salt=f"good:{view_type or ''}")

    def get_bad_examples(
        self,
        n: int = 2,
        issue_type: Optional[str] = None,
        view_type: Optional[str] = None,
        seed: Optional[str] = None,
    ) -> List[VisualExample]:
        """Return n bad examples, optionally filtered by issue type/view."""
        pool = (
            self._by_issue.get(issue_type, self.bad_examples)
            if issue_type
            else self.bad_examples
        )
        pool = list(pool)
        if view_type:
            wanted = self._normalize_view(view_type)
            same_view = [
                ex for ex in pool if self._normalize_view(ex.view) == wanted
            ]
            if same_view:
                primary = self._select_examples(
                    same_view,
                    min(n, len(same_view)),
                    seed,
                    salt=f"bad:primary:{issue_type or ''}:{wanted}",
                )
                if len(primary) >= n:
                    return primary[:n]

                seen = {ex.path for ex in primary}
                fallback_pool = [ex for ex in pool if ex.path not in seen]
                fallback = self._select_examples(
                    fallback_pool,
                    n - len(primary),
                    seed,
                    salt=f"bad:fallback:{issue_type or ''}:{wanted}",
                )
                return primary + fallback
        return self._select_examples(pool, n, seed, salt=f"bad:{issue_type or ''}:{view_type or ''}")

    def get_few_shot_set(
        self,
        n_good: int = 2,
        n_bad: int = 2,
        view_type: Optional[str] = None,
        seed: Optional[str] = None,
    ) -> Tuple[List[VisualExample], List[VisualExample]]:
        """Return a balanced set of good and bad examples for VLM few-shot."""
        return (
            self.get_good_examples_for_view(n=n_good, view_type=view_type, seed=seed),
            self.get_bad_examples(n=n_bad, view_type=view_type, seed=seed),
        )

    @property
    def issue_types(self) -> List[str]:
        """All distinct issue types found in bad examples."""
        return list(self._by_issue.keys())

    # ---------- Prompt generation ----------

    def build_knowledge_prompt(self) -> str:
        """
        Generate a text summary of visual quality patterns learned from examples.
        This is injected into agent system prompts.
        """
        # Compute stats
        good_dice = [e.dice for e in self.good_examples]
        bad_dice = [e.dice for e in self.bad_examples]
        good_range = (
            f"{min(good_dice):.3f}–{max(good_dice):.3f}" if good_dice else "N/A"
        )
        bad_range = (
            f"{min(bad_dice):.3f}–{max(bad_dice):.3f}" if bad_dice else "N/A"
        )

        lines = [
            "## Visual Quality Knowledge (learned from annotated examples)",
            "",
            f"Training set: {len(self.good_examples)} good cases "
            f"(Dice {good_range}) and "
            f"{len(self.bad_examples)} bad cases "
            f"(Dice {bad_range}).",
            "Reference format: single-panel segmentation overlays (no GT/error-map panels at inference time).",
            "",
            "### Score Calibration Anchors",
            "- Good reference images represent PERFECT segmentation quality (score 10/10).",
            "- A TARGET matching reference quality → score 8.5-10.",
            "- Significant deviations lower the score proportionally.",
            "",
            "### Anatomy Priors (evidence-backed, weak prior only)",
            "- RV blood pool is typically crescent/triangular and more trabeculated than LV.",
            "- LV blood pool is typically smoother and more compact than RV.",
            "- Interventricular septum separates RV and LV; use it as septal orientation anchor.",
            "- Myocardium label corresponds to LV myocardium (between LV endocardial/epicardial boundaries).",
            "- SAX mid-slice: LV should be enclosed or near-enclosed by Myo.",
            "- LAX (2CH/3CH/4CH), basal valve-plane, and extreme apical slices: do not force strict 360-degree Myo closure.",
            "",
            "### Good Segmentation Characteristics",
            "- Myocardium (green/yellow): SAX mid-slice should be complete/near-closed around LV; LAX focuses on plausible continuity",
            "- LV (blue/red-brown): single solid region; in SAX usually enclosed by Myo, in LAX near-boundary alignment is the key check",
            "- RV (red/cyan): located adjacent to septum, separated from LV by Myo",
            "- All three structures present, no fragmentation",
            "- Boundaries are smooth and follow anatomical contours",
            "",
            "### Bad: Disconnected Myocardium (DiscMyo)",
            "- Myo ring has a visible GAP — usually at the septal or lateral wall",
            "- RV and LV may directly touch through the gap",
            f"- {len(self._by_issue.get('DiscMyo', []))} examples, "
            f"Dice range: {self._dice_range('DiscMyo')}",
            "",
            "### Bad: Low Overall Dice (LowDice)",
            "- Structures poorly positioned, wrong shape, or missing regions",
            "- May have fragmented components or entirely wrong boundaries",
            f"- {len(self._by_issue.get('LowDice', []))} examples, "
            f"Dice range: {self._dice_range('LowDice')}",
            "",
            "### Decision Boundary",
            "- Morphology-first judgment (do not infer exact Dice value from appearance).",
            "- Good: complete/closed Myo ring, RV-LV separation, required structures present; minor roughness allowed.",
            "- Bad: obvious Myo gap, RV-LV merge, missing structure, or severe fragmentation.",
            "- Borderline: localized moderate defects without gross anatomical collapse.",
        ]
        return "\n".join(lines)

    def _dice_range(self, issue_type: str) -> str:
        examples = self._by_issue.get(issue_type, [])
        if not examples:
            return "N/A"
        dices = [e.dice for e in examples]
        return f"{min(dices):.3f}–{max(dices):.3f}"


# =====================================================================
# Repair Comparison Knowledge Base
# =====================================================================

# Filename pattern for repair KB:
# e.g. "+0.0512_myo_bridge_335_original_lax_4c_009.png"
_REPAIR_FNAME_RE = re.compile(
    r"^([+-]\d+\.\d+)_(.+?)_(\d+_original_lax_\w+_\d+)\.png$"
)


@dataclass
class RepairExample:
    """A single labeled repair before→after diff overlay."""
    path: str             # absolute path to the 3-panel diff overlay
    dice_delta: float     # Dice change caused by this repair
    op_name: str          # operation that was applied
    stem: str             # case stem
    category: str         # "improved" | "degraded" | "neutral"
    example_id: str = ""
    view_type: str = ""
    defect_tags: Tuple[str, ...] = field(default_factory=tuple)
    action_tags: Tuple[str, ...] = field(default_factory=tuple)
    confidence_prior: float = 0.5
    source: str = "legacy_repair_kb"


class RepairComparisonKB:
    """
    Knowledge base of labeled repair examples.

    Loads 3-panel diff overlays (BEFORE | AFTER | DIFF) that were
    pre-generated by ``build_repair_kb.py`` and labeled by Dice delta.

    Used at runtime (GT-free) to provide few-shot examples showing
    the VLM what improved vs degraded repairs look like.
    """

    def __init__(self):
        self.improved: List[RepairExample] = []
        self.degraded: List[RepairExample] = []
        self.neutral:  List[RepairExample] = []
        self.records: List[KnowledgeRecord] = []
        self._record_by_id: Dict[str, KnowledgeRecord] = {}
        self._example_by_id: Dict[str, RepairExample] = {}

    _legacy_warned: ClassVar[bool] = False

    @classmethod
    def load(cls, kb_dir: str) -> "RepairComparisonKB":
        """Load from a repair_kb/ directory with improved/degraded/neutral sub-dirs."""
        kb = cls()

        for cat, store in [
            ("improved", kb.improved),
            ("degraded", kb.degraded),
            ("neutral",  kb.neutral),
        ]:
            cat_dir = os.path.join(kb_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                m = _REPAIR_FNAME_RE.match(fname)
                if not m:
                    continue
                if not cls._legacy_warned:
                    logger.warning(
                        "RepairComparisonKB.load: detected legacy filename KB format; "
                        "auto-normalizing to typed records."
                    )
                    cls._legacy_warned = True
                delta_str, op_name, stem = m.groups()
                ex = RepairExample(
                    path=os.path.join(cat_dir, fname),
                    dice_delta=float(delta_str),
                    op_name=op_name,
                    stem=stem,
                    category=cat,
                )
                kb._fill_repair_schema(ex)
                store.append(ex)

        kb._materialize_records()
        logger.info(
            "RepairComparisonKB loaded: %d improved, %d degraded, %d neutral",
            len(kb.improved), len(kb.degraded), len(kb.neutral),
        )
        return kb

    @staticmethod
    def _op_tags(op_name: str) -> Tuple[str, ...]:
        raw = str(op_name or "").strip().lower()
        toks = [t for t in re.split(r"[^a-z0-9]+", raw) if t]
        tags: List[str] = list(toks)
        if "atlas" in toks:
            tags.append("atlas_guided")
        if "neighbor" in toks:
            tags.append("neighbor_guided")
        if "bridge" in toks:
            tags.append("gap_repair")
        if "fill" in toks:
            tags.append("hole_repair")
        for cls_tok in ("rv", "myo", "lv"):
            if cls_tok in toks:
                tags.append(f"target_{cls_tok}")
        return _norm_tags(tags)

    @staticmethod
    def _defect_tags_from_repair(ex: RepairExample) -> Tuple[str, ...]:
        tags: List[str] = [ex.category]
        tags.extend(list(RepairComparisonKB._op_tags(ex.op_name)))
        if ex.category == "improved":
            tags.append("positive_repair")
        elif ex.category == "degraded":
            tags.append("regression_risk")
        else:
            tags.append("neutral_change")
        return _norm_tags(tags)

    @staticmethod
    def _confidence_from_delta(delta: float, category: str) -> float:
        mag = min(1.0, max(0.0, abs(float(delta)) / 0.10))
        if category == "neutral":
            return max(0.05, 0.20 * (1.0 - mag))
        return max(0.05, 0.25 + 0.75 * mag)

    def _fill_repair_schema(self, ex: RepairExample) -> None:
        ex.view_type = _parse_view_from_stem(ex.stem)
        ex.action_tags = self._op_tags(ex.op_name)
        ex.defect_tags = self._defect_tags_from_repair(ex)
        ex.confidence_prior = self._confidence_from_delta(ex.dice_delta, ex.category)
        safe_op = _norm_tag(ex.op_name or "op")
        ex.example_id = f"rkb:{ex.category}:{ex.view_type}:{safe_op}:{ex.stem}"
        ex.source = f"legacy_repair_kb:{ex.category}"

    def _materialize_records(self) -> None:
        records: List[KnowledgeRecord] = []
        by_id: Dict[str, KnowledgeRecord] = {}
        ex_by_id: Dict[str, RepairExample] = {}
        for ex in self.improved + self.degraded + self.neutral:
            rec = KnowledgeRecord(
                example_id=ex.example_id,
                stage="repair_verification",
                view_type=ex.view_type or "ANY",
                defect_tags=tuple(ex.defect_tags),
                action_tags=tuple(ex.action_tags),
                expected_outcome=ex.category,
                confidence_prior=float(ex.confidence_prior),
                artifact_path=ex.path,
                source=ex.source,
            )
            records.append(rec)
            by_id[rec.example_id] = rec
            ex_by_id[rec.example_id] = ex
        self.records = records
        self._record_by_id = by_id
        self._example_by_id = ex_by_id

    def retrieve(
        self,
        stage: str,
        view_type: Optional[str],
        defect_tags: Sequence[str],
        top_k: int = 4,
        use_confidence_prior: bool = True,
    ) -> List[KnowledgeRecord]:
        """
        Deterministic retrieval for repair-verification evidence.
        """
        return _default_retrieval(
            self.records,
            stage=stage,
            view_type=view_type,
            defect_tags=defect_tags,
            top_k=top_k,
            use_confidence_prior=use_confidence_prior,
        )

    def get_examples_by_ids(self, example_ids: Sequence[str]) -> List[RepairExample]:
        out: List[RepairExample] = []
        for eid in example_ids:
            ex = self._example_by_id.get(str(eid))
            if ex is not None:
                out.append(ex)
        return out

    @property
    def total(self) -> int:
        return len(self.improved) + len(self.degraded) + len(self.neutral)

    def get_few_shot_repair(
        self, n_improved: int = 2, n_degraded: int = 2
    ) -> Tuple[List[RepairExample], List[RepairExample]]:
        """
        Return balanced few-shot examples of improved and degraded repairs.
        Prioritizes examples with the most extreme Dice deltas.
        """
        imp_sorted = sorted(self.improved, key=lambda x: -x.dice_delta)
        deg_sorted = sorted(self.degraded, key=lambda x: x.dice_delta)

        imp = imp_sorted[:n_improved]
        deg = deg_sorted[:n_degraded]
        return imp, deg

    def build_repair_knowledge_prompt(self) -> str:
        """Generate a text summary of repair quality patterns."""
        lines = [
            "## Repair Quality Judgment Knowledge",
            "",
            f"Training set: {len(self.improved)} improved repairs, "
            f"{len(self.degraded)} degraded repairs.",
            "",
            "### How to read the DIFF panel:",
            "- Green pixels = ADDED by repair (new foreground)",
            "- Red pixels = REMOVED by repair (lost foreground)",
            "- Yellow pixels = CLASS CHANGED (label swap)",
            "- Gray pixels = UNCHANGED foreground",
            "",
            "### Signs of IMPROVED repair:",
            "- Green fills a gap in the Myo ring → structure restored",
            "- Green restores a missing RV/LV region → anatomy improved",
            "- Red removes small spurious islands → cleaner result",
            "- Changes are focused on the defective area",
            "",
            "### Signs of DEGRADED repair:",
            "- Large red areas (removing correct structure)",
            "- Green extends far beyond anatomical boundaries",
            "- Yellow indicates massive class confusion",
            "- Changes affect areas that were already correct",
            "",
            "### Decision Rules:",
            "- IMPROVED: targeted fix in defective area, minimal collateral damage",
            "- DEGRADED: widespread changes, or correct areas destroyed",
            "- NEUTRAL: very small or meaningless changes",
        ]
        return "\n".join(lines)
