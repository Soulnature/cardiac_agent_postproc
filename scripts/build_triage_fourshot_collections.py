#!/usr/bin/env python3
"""
Build a 4-way triage reference collection for few-shot experiments.

Categories:
1) good
2) bad
3) bad_high_dice  (bad anatomy but high Dice)
4) other

Default rule set:
- good: mean_dice >= 0.93 and was_disc_c2 == False
- bad_high_dice: was_disc_c2 == True and mean_dice > 0.90
- bad: mean_dice < 0.90 and not bad_high_dice
- other: all remaining cases
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Make local package importable when running script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from cardiac_agent_postproc.vlm_guardrail import create_overlay


@dataclass(frozen=True)
class CategoryRule:
    good_min_dice: float = 0.93
    bad_max_dice: float = 0.90
    bad_high_dice_min: float = 0.90


def _classify(mean_dice: float, was_disc_c2: bool, rule: CategoryRule) -> str:
    if (mean_dice >= rule.good_min_dice) and (not was_disc_c2):
        return "good"
    if was_disc_c2 and (mean_dice > rule.bad_high_dice_min):
        return "bad_high_dice"
    if mean_dice < rule.bad_max_dice:
        return "bad"
    return "other"


def _classify_gold_case(gold_label: str, mean_dice: float, bad_high_dice_min: float) -> str:
    """
    Classification when using curated gold folders:
    - best_cases   -> good
    - worst_cases  -> bad_high_dice if Dice > threshold else bad
    """
    if gold_label == "good":
        return "good"
    if gold_label == "bad":
        return "bad_high_dice" if mean_dice > bad_high_dice_min else "bad"
    return "other"


def _safe_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _link_or_copy(src: Path, dst: Path, force_copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if force_copy:
        shutil.copy2(src, dst)
        return
    rel_src = os.path.relpath(src, dst.parent)
    try:
        dst.symlink_to(rel_src)
    except OSError:
        shutil.copy2(src, dst)


def _ensure_overlay(
    stem: str,
    single_panel_refs_dir: Path,
    all_frames_export_dir: Path,
    output_overlays_dir: Path,
    force_copy: bool,
    generate_missing_overlays: bool,
) -> str:
    prebuilt = single_panel_refs_dir / f"{stem}_overlay.png"
    out_overlay = output_overlays_dir / f"{stem}_overlay.png"

    if prebuilt.exists():
        _link_or_copy(prebuilt, out_overlay, force_copy=force_copy)
        return str(out_overlay)

    if not generate_missing_overlays:
        return ""

    img_path = all_frames_export_dir / f"{stem}_img.png"
    pred_path = all_frames_export_dir / f"{stem}_pred.png"
    if not (img_path.exists() and pred_path.exists()):
        return ""

    import cv2  # local import to keep startup fast

    pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if pred_mask is None:
        return ""

    output_overlays_dir.mkdir(parents=True, exist_ok=True)
    ok = create_overlay(str(img_path), pred_mask, str(out_overlay))
    return str(out_overlay) if ok else ""


def _load_gold_cases(results_dir: Path) -> pd.DataFrame:
    """
    Load curated labels from best_cases/ and worst_cases filenames.

    Filename format:
      <dice>_<view>_<issue_type>_<stem>.png
    """
    best_dir = results_dir / "best_cases"
    worst_dir = results_dir / "worst_cases"
    if not best_dir.exists() or not worst_dir.exists():
        raise FileNotFoundError(
            f"gold folders missing: best_cases={best_dir.exists()}, worst_cases={worst_dir.exists()}"
        )

    pattern = re.compile(
        r"^(?P<dice>\d+\.\d+)_(?P<view>\w+)_(?P<issue>\w+)_(?P<stem>\d+_original_lax_\w+_\d+)\.png$"
    )
    rows: List[Dict[str, object]] = []
    for folder, gold_label in ((best_dir, "good"), (worst_dir, "bad")):
        for p in sorted(folder.glob("*.png")):
            m = pattern.match(p.name)
            if not m:
                continue
            issue = m.group("issue")
            was_disc_c2 = issue.lower() == "discmyo"
            rows.append(
                {
                    "id": m.group("stem"),
                    "mean_dice": float(m.group("dice")),
                    "was_disc_c2": was_disc_c2,
                    "view": m.group("view"),
                    "gold_label": gold_label,
                    "issue_type": issue,
                    "source_file": str(p),
                }
            )
    if not rows:
        raise ValueError("No parseable PNG files found in best_cases/worst_cases.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 4-way triage few-shot collections")
    parser.add_argument(
        "--results_dir",
        default="results/Input_MnM2",
        help="Directory containing metrics_report.csv, all_frames_export/, single_panel_refs/",
    )
    parser.add_argument(
        "--output_dir",
        default="results/Input_MnM2/triage_4way_collections",
        help="Output directory for manifests and category folders",
    )
    parser.add_argument(
        "--source_mode",
        choices=("metrics", "gold_cases"),
        default="metrics",
        help="Use all metrics rows or only curated best_cases/worst_cases",
    )
    parser.add_argument("--good_min_dice", type=float, default=0.93)
    parser.add_argument("--bad_max_dice", type=float, default=0.90)
    parser.add_argument("--bad_high_dice_min", type=float, default=0.90)
    parser.add_argument(
        "--force_copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    parser.add_argument(
        "--no_generate_missing_overlays",
        action="store_true",
        help="Do not generate overlays for stems missing in single_panel_refs",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    metrics_csv = results_dir / "metrics_report.csv"
    all_frames_export_dir = results_dir / "all_frames_export"
    single_panel_refs_dir = results_dir / "single_panel_refs"
    output_overlays_dir = output_dir / "overlays"
    categories_root = output_dir / "categories"

    if not all_frames_export_dir.exists():
        raise FileNotFoundError(f"all_frames_export dir not found: {all_frames_export_dir}")

    if args.source_mode == "metrics" and not metrics_csv.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_csv}")

    rule = CategoryRule(
        good_min_dice=args.good_min_dice,
        bad_max_dice=args.bad_max_dice,
        bad_high_dice_min=args.bad_high_dice_min,
    )
    generate_missing = not args.no_generate_missing_overlays

    output_dir.mkdir(parents=True, exist_ok=True)
    output_overlays_dir.mkdir(parents=True, exist_ok=True)
    for cat in ("good", "bad", "bad_high_dice", "other"):
        (categories_root / cat).mkdir(parents=True, exist_ok=True)

    if args.source_mode == "metrics":
        df = pd.read_csv(metrics_csv)
        required_cols = {"id", "mean_dice", "was_disc_c2", "view"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"metrics_report.csv missing columns: {sorted(missing)}")
    else:
        df = _load_gold_cases(results_dir)

    records: List[Dict[str, object]] = []
    for row in df.sort_values("id").itertuples(index=False):
        stem = str(getattr(row, "id"))
        mean_dice = float(getattr(row, "mean_dice"))
        was_disc_c2 = _safe_bool(getattr(row, "was_disc_c2"))
        view = str(getattr(row, "view"))
        if args.source_mode == "metrics":
            category = _classify(mean_dice, was_disc_c2, rule)
            gold_label = ""
            issue_type = ""
        else:
            gold_label = str(getattr(row, "gold_label", ""))
            issue_type = str(getattr(row, "issue_type", ""))
            category = _classify_gold_case(
                gold_label=gold_label,
                mean_dice=mean_dice,
                bad_high_dice_min=rule.bad_high_dice_min,
            )

        overlay_path = _ensure_overlay(
            stem=stem,
            single_panel_refs_dir=single_panel_refs_dir,
            all_frames_export_dir=all_frames_export_dir,
            output_overlays_dir=output_overlays_dir,
            force_copy=args.force_copy,
            generate_missing_overlays=generate_missing,
        )

        cat_overlay_path = ""
        if overlay_path:
            src_overlay = Path(overlay_path)
            dst_overlay = categories_root / category / f"{stem}_overlay.png"
            _link_or_copy(src_overlay, dst_overlay, force_copy=args.force_copy)
            cat_overlay_path = str(dst_overlay)

        record = {
            "stem": stem,
            "category": category,
            "gold_label": gold_label,
            "issue_type": issue_type,
            "mean_dice": mean_dice,
            "was_disc_c2": was_disc_c2,
            "view": view,
            "img_path": str(all_frames_export_dir / f"{stem}_img.png"),
            "pred_path": str(all_frames_export_dir / f"{stem}_pred.png"),
            "gt_path": str(all_frames_export_dir / f"{stem}_gt.png"),
            "overlay_path": overlay_path,
            "category_overlay_path": cat_overlay_path,
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    manifest_csv = output_dir / "triage_4way_manifest.csv"
    out_df.to_csv(manifest_csv, index=False)

    summary = (
        out_df.groupby("category", dropna=False)
        .agg(
            n=("stem", "count"),
            dice_mean=("mean_dice", "mean"),
            dice_min=("mean_dice", "min"),
            dice_max=("mean_dice", "max"),
            disc_rate=("was_disc_c2", "mean"),
        )
        .reset_index()
        .sort_values("category")
    )
    summary_csv = output_dir / "triage_4way_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"Saved manifest: {manifest_csv}")
    print(f"Saved summary : {summary_csv}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
