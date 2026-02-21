"""
Visual Knowledge Base: learns good/bad segmentation patterns from annotated examples.

Loads overlay images from results/Input_MnM2/{best_cases,worst_cases} and provides
few-shot references and text summaries for VLM-based quality judgment.
"""
from __future__ import annotations

import csv
import os
import re
import random
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------- Filename pattern ----------
# e.g. "0.373_4C_LowDice_335_original_lax_4c_009.png"
# e.g. "0.967_4C_Best_215_original_lax_4c_000.png"
_FNAME_RE = re.compile(
    r"^(\d+\.\d+)_(\w+)_(\w+)_(\d+_original_lax_\w+_\d+)\.png$"
)


@dataclass
class VisualExample:
    """A single annotated overlay image."""
    path: str          # absolute path to the overlay PNG
    dice: float        # e.g. 0.373
    view: str          # e.g. "4C"
    issue_type: str    # "LowDice" | "DiscMyo" | "Best"
    stem: str          # case stem e.g. "335_original_lax_4c_009"
    category: str      # "good" | "bad"


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
    four_way_examples: Dict[str, List[VisualExample]] = field(default_factory=dict)

    # Index by issue type for targeted lookups
    _by_issue: Dict[str, List[VisualExample]] = field(
        default_factory=dict, repr=False
    )

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
        kb._load_four_way_collections(results_dir)

        logger.info(
            "VisualKnowledgeBase loaded: %d good, %d bad (%s); 4-way refs: %s",
            len(kb.good_examples),
            len(kb.bad_examples),
            ", ".join(f"{k}={len(v)}" for k, v in kb._by_issue.items()),
            ", ".join(f"{k}={len(v)}" for k, v in sorted(kb.four_way_examples.items())) or "none",
        )
        return kb

    @staticmethod
    def _resolve_existing_path(path_value: str, results_dir: str) -> str:
        """
        Resolve a manifest path that may be absolute or workspace-relative.
        Returns the first existing path candidate; otherwise returns cwd-relative.
        """
        raw = Path(path_value)
        if raw.is_absolute():
            return str(raw)

        candidates = [
            (Path.cwd() / raw),
            (Path(results_dir) / raw),
            (Path(results_dir).parent / raw),
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand.resolve())
        return str((Path.cwd() / raw).resolve())

    def _load_four_way_collections(self, results_dir: str) -> None:
        """
        Load four-category reference pools from triage_4way_collections_gold manifest.

        Expected manifest:
          {results_dir}/triage_4way_collections_gold/triage_4way_manifest.csv
        """
        manifest_path = os.path.join(
            results_dir, "triage_4way_collections_gold", "triage_4way_manifest.csv"
        )
        if not os.path.exists(manifest_path):
            logger.info("VisualKnowledgeBase 4-way manifest missing: %s", manifest_path)
            return

        loaded = 0
        seen = set()
        try:
            with open(manifest_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    category = (row.get("category") or "").strip()
                    stem = (row.get("stem") or "").strip()
                    overlay_rel = (
                        (row.get("category_overlay_path") or "")
                        or (row.get("overlay_path") or "")
                    ).strip()
                    if not category or not stem or not overlay_rel:
                        continue

                    overlay_path = self._resolve_existing_path(overlay_rel, results_dir)
                    if not os.path.exists(overlay_path):
                        continue

                    dedup_key = (category, stem, overlay_path)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    try:
                        dice = float((row.get("mean_dice") or "0").strip())
                    except ValueError:
                        dice = 0.0

                    ex = VisualExample(
                        path=overlay_path,
                        dice=dice,
                        view=(row.get("view") or "").strip(),
                        issue_type=(row.get("issue_type") or "").strip(),
                        stem=stem,
                        category=category,
                    )
                    self.four_way_examples.setdefault(category, []).append(ex)
                    loaded += 1
        except Exception as e:
            logger.warning("Failed loading 4-way references from %s: %s", manifest_path, e)
            return

        for examples in self.four_way_examples.values():
            examples.sort(key=lambda ex: ex.stem)
        logger.info(
            "VisualKnowledgeBase 4-way refs loaded: %d examples across %d categories",
            loaded,
            len(self.four_way_examples),
        )

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
        if not view:
            return ""
        token = re.sub(r"[^a-z0-9]", "", str(view).strip().lower())
        if token in {"2c", "2ch"}:
            return "2CH"
        if token in {"3c", "3ch"}:
            return "3CH"
        if token in {"4c", "4ch"}:
            return "4CH"
        if "sax" in token:
            return "SAX"
        return token.upper()

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

    def get_four_way_set(
        self,
        categories: List[str],
        view_type: Optional[str] = None,
        seed: Optional[str] = None,
    ) -> List[VisualExample]:
        """
        Return exactly one reference per requested category when possible.
        If view_type is set, prefer same-view references; otherwise fallback to any.
        """
        refs: List[VisualExample] = []
        wanted = self._normalize_view(view_type) if view_type else ""
        for category in categories:
            pool = list(self.four_way_examples.get(category, []))
            if not pool:
                continue

            selected_pool = pool
            salt = f"four_way:{category}:any"
            if wanted:
                same_view = [
                    ex for ex in pool if self._normalize_view(ex.view) == wanted
                ]
                if same_view:
                    selected_pool = same_view
                    salt = f"four_way:{category}:primary:{wanted}"
                else:
                    salt = f"four_way:{category}:fallback:{wanted}"

            refs.extend(self._select_examples(selected_pool, 1, seed, salt=salt))
        return refs

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
            "### Good Segmentation Characteristics",
            "- Myocardium (green/yellow): forms a COMPLETE, CLOSED ring around LV",
            "- LV (blue/red-brown): single solid region, fully enclosed by Myo ring",
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
                delta_str, op_name, stem = m.groups()
                store.append(RepairExample(
                    path=os.path.join(cat_dir, fname),
                    dice_delta=float(delta_str),
                    op_name=op_name,
                    stem=stem,
                    category=cat,
                ))

        logger.info(
            "RepairComparisonKB loaded: %d improved, %d degraded, %d neutral",
            len(kb.improved), len(kb.degraded), len(kb.neutral),
        )
        return kb

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
