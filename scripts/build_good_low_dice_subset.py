#!/usr/bin/env python3
"""
Build a "good low-Dice" subset from metrics_report.csv.

Default:
- Dice range: [0.90, 0.95]
- Exclude disc cases (was_disc_c2=True)
- Select top-k hardest good cases by lowest Dice
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from cardiac_agent_postproc.vlm_guardrail import create_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Build good low-Dice subset")
    parser.add_argument("--results_dir", default="results/Input_MnM2")
    parser.add_argument(
        "--output_dir",
        default="results/Input_MnM2/good_low_dice_subset",
        help="Output directory for subset files",
    )
    parser.add_argument("--min_dice", type=float, default=0.90)
    parser.add_argument("--max_dice", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--include_disc",
        action="store_true",
        help="Include was_disc_c2 cases (default: excluded)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    metrics_csv = results_dir / "metrics_report.csv"
    all_frames_export = results_dir / "all_frames_export"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")
    if not all_frames_export.exists():
        raise FileNotFoundError(f"Missing dir: {all_frames_export}")

    df = pd.read_csv(metrics_csv)
    need_cols = {"id", "mean_dice", "was_disc_c2", "view"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"metrics_report.csv missing columns: {sorted(miss)}")

    sub = df[(df["mean_dice"] >= args.min_dice) & (df["mean_dice"] <= args.max_dice)].copy()
    if not args.include_disc:
        sub = sub[~sub["was_disc_c2"].astype(bool)].copy()
    sub = sub.sort_values(["mean_dice", "id"], ascending=[True, True]).head(args.top_k)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    category_dir = output_dir / "categories" / "good_low_dice"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    category_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for row in sub.itertuples(index=False):
        stem = str(row.id)
        img_path = all_frames_export / f"{stem}_img.png"
        pred_path = all_frames_export / f"{stem}_pred.png"
        gt_path = all_frames_export / f"{stem}_gt.png"
        overlay_path = overlays_dir / f"{stem}_overlay.png"

        ok = False
        if img_path.exists() and pred_path.exists():
            import cv2

            pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if pred_mask is not None:
                ok = create_overlay(str(img_path), pred_mask, str(overlay_path))
        if not ok:
            overlay_str = ""
            category_overlay_str = ""
        else:
            category_overlay = category_dir / f"{stem}_overlay.png"
            if category_overlay.exists() or category_overlay.is_symlink():
                category_overlay.unlink()
            rel_src = os.path.relpath(overlay_path, category_overlay.parent)
            try:
                category_overlay.symlink_to(rel_src)
            except OSError:
                import shutil

                shutil.copy2(overlay_path, category_overlay)
            overlay_str = str(overlay_path)
            category_overlay_str = str(category_overlay)

        rows.append(
            {
                "stem": stem,
                "category": "good_low_dice",
                "mean_dice": float(row.mean_dice),
                "was_disc_c2": bool(row.was_disc_c2),
                "view": str(row.view),
                "img_path": str(img_path),
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "overlay_path": overlay_str,
                "category_overlay_path": category_overlay_str,
            }
        )

    out_df = pd.DataFrame(rows)
    manifest = output_dir / "good_low_dice_manifest.csv"
    summary = output_dir / "good_low_dice_summary.csv"
    out_df.to_csv(manifest, index=False)

    if not out_df.empty:
        s = (
            out_df.groupby("category", dropna=False)
            .agg(
                n=("stem", "count"),
                dice_mean=("mean_dice", "mean"),
                dice_min=("mean_dice", "min"),
                dice_max=("mean_dice", "max"),
                disc_rate=("was_disc_c2", "mean"),
            )
            .reset_index()
        )
    else:
        s = pd.DataFrame(
            [
                {
                    "category": "good_low_dice",
                    "n": 0,
                    "dice_mean": float("nan"),
                    "dice_min": float("nan"),
                    "dice_max": float("nan"),
                    "disc_rate": float("nan"),
                }
            ]
        )
    s.to_csv(summary, index=False)

    print(f"Saved manifest: {manifest}")
    print(f"Saved summary : {summary}")
    print(out_df[["stem", "mean_dice"]].to_string(index=False))


if __name__ == "__main__":
    main()
