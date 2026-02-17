"""
Visual Knowledge Base: learns good/bad segmentation patterns from annotated examples.

Loads overlay images from results/Input_MnM2/{best_cases,worst_cases} and provides
few-shot references and text summaries for VLM-based quality judgment.
"""
from __future__ import annotations

import os
import re
import random
import logging
from dataclasses import dataclass, field
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

        logger.info(
            "VisualKnowledgeBase loaded: %d good, %d bad (%s)",
            len(kb.good_examples),
            len(kb.bad_examples),
            ", ".join(f"{k}={len(v)}" for k, v in kb._by_issue.items()),
        )
        return kb

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
        if len(self.good_examples) <= n:
            return list(self.good_examples)
        return random.sample(self.good_examples, n)

    def get_bad_examples(
        self, n: int = 2, issue_type: Optional[str] = None
    ) -> List[VisualExample]:
        """Return n bad examples, optionally filtered by issue type."""
        pool = (
            self._by_issue.get(issue_type, self.bad_examples)
            if issue_type
            else self.bad_examples
        )
        if len(pool) <= n:
            return list(pool)
        return random.sample(pool, n)

    def get_few_shot_set(
        self, n_good: int = 2, n_bad: int = 2
    ) -> Tuple[List[VisualExample], List[VisualExample]]:
        """Return a balanced set of good and bad examples for VLM few-shot."""
        return self.get_good_examples(n_good), self.get_bad_examples(n_bad)

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
            "",
            "### Good Segmentation Characteristics",
            "- Myocardium (green/yellow): forms a COMPLETE, CLOSED ring around LV",
            "- LV (blue/red-brown): single solid region, fully enclosed by Myo ring",
            "- RV (red/cyan): located adjacent to septum, separated from LV by Myo",
            "- All three structures present, no fragmentation",
            "- Boundaries are smooth and follow anatomical contours",
            "- Error overlay shows minimal colored regions",
            "",
            "### Bad: Disconnected Myocardium (DiscMyo)",
            "- Myo ring has a visible GAP — usually at the septal or lateral wall",
            "- RV and LV may directly touch through the gap",
            "- Error map shows class confusion (yellow) and false negatives (blue) at gap",
            f"- {len(self._by_issue.get('DiscMyo', []))} examples, "
            f"Dice range: {self._dice_range('DiscMyo')}",
            "",
            "### Bad: Low Overall Dice (LowDice)",
            "- Structures poorly positioned, wrong shape, or missing regions",
            "- May have fragmented components or entirely wrong boundaries",
            "- Large red (false positive) and blue (false negative) error regions",
            f"- {len(self._by_issue.get('LowDice', []))} examples, "
            f"Dice range: {self._dice_range('LowDice')}",
            "",
            "### Decision Boundary",
            "- Good: Dice ≥ 0.95, complete Myo ring, all structures present",
            "- Bad: Dice < 0.90, OR broken Myo ring, OR missing structure",
            "- Borderline (0.90–0.95): needs careful visual inspection",
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

