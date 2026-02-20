#!/usr/bin/env python3
"""
Plot MICCAI-style three-group boxplots from a triage CSV.

Expected labels after normalization:
  good / borderline / bad

Usage:
  python scripts/plot_triage_boxplot_miccai.py \
      --input_csv batch_eval_results/triage_only_gpt4o_grouping_all_cases.csv \
      --output_plot batch_eval_results/triage_only_gpt4o_dice_boxplot_miccai.png \
      --stats_csv batch_eval_results/triage_only_gpt4o_dice_boxplot_miccai_stats.csv
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import pandas as pd
import seaborn as sns

# Make Matplotlib cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUP_ORDER = ["good", "borderline", "bad"]
GROUP_PALETTE = {
    "good": "#2A9D8F",
    "borderline": "#E9C46A",
    "bad": "#E76F51",
}


def _normalize_group_label(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"good", "ok", "acceptable"}:
        return "good"
    if s in {"borderline", "middle", "mid", "uncertain"}:
        return "borderline"
    if s in {"bad", "needs_fix", "need_fix", "poor", "error"}:
        return "bad"
    return None


def _choose_group_column(df: pd.DataFrame, group_col: str) -> str:
    if group_col != "auto":
        if group_col not in df.columns:
            raise ValueError(f"group column '{group_col}' not found in CSV.")
        return group_col

    for candidate in ["VLM_Quality", "TriageCategory3", "Category3", "TriageCategory", "Category"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No suitable group column found. Please pass --group_col explicitly "
        "(e.g., VLM_Quality or TriageCategory)."
    )


def derive_three_groups(df: pd.DataFrame, group_col: str) -> Tuple[pd.Series, str]:
    selected_col = _choose_group_column(df, group_col)
    group3 = df[selected_col].map(_normalize_group_label)

    # If only 2-group triage is available, use VLM quality to split bad -> borderline/bad.
    if (
        selected_col in {"TriageCategory", "Category"}
        and "borderline" not in set(group3.dropna().unique())
        and "VLM_Quality" in df.columns
    ):
        vlm_group = df["VLM_Quality"].map(_normalize_group_label)
        bad_mask = group3 == "bad"
        group3.loc[bad_mask] = vlm_group.loc[bad_mask].where(
            vlm_group.loc[bad_mask].isin(["borderline", "bad"]),
            "bad",
        )

    return group3, selected_col


def build_stats(plot_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    grouped = plot_df.groupby("Group3")[score_col]
    stats = pd.DataFrame(
        {
            "n": grouped.count(),
            "mean": grouped.mean(),
            "median": grouped.median(),
            "std": grouped.std(),
            "q1": grouped.quantile(0.25),
            "q3": grouped.quantile(0.75),
            "min": grouped.min(),
            "max": grouped.max(),
        }
    ).reset_index()
    stats["Group3"] = pd.Categorical(stats["Group3"], categories=GROUP_ORDER, ordered=True)
    stats = stats.sort_values("Group3").reset_index(drop=True)
    return stats


def apply_miccai_style() -> None:
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_miccai_boxplot(
    plot_df: pd.DataFrame,
    score_col: str,
    output_plot: str,
    title: str,
) -> None:
    apply_miccai_style()
    order = [g for g in GROUP_ORDER if g in set(plot_df["Group3"])]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)
    sns.boxplot(
        data=plot_df,
        x="Group3",
        y=score_col,
        hue="Group3",
        order=order,
        hue_order=order,
        palette=GROUP_PALETTE,
        dodge=False,
        width=0.56,
        showfliers=False,
        linewidth=1.25,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.stripplot(
        data=plot_df,
        x="Group3",
        y=score_col,
        order=order,
        color="#1F1F1F",
        size=2.6,
        alpha=0.42,
        jitter=0.2,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Quality Group")
    ax.set_ylabel(score_col.replace("_", " "))
    ax.grid(axis="x", visible=False)

    y0, y1 = ax.get_ylim()
    span = max(y1 - y0, 1e-6)
    for i, cat in enumerate(order):
        mean_val = float(plot_df.loc[plot_df["Group3"] == cat, score_col].mean())
        ax.text(i, y1 - 0.03 * span, f"{mean_val:.3f}", ha="center", va="top", fontsize=10)

    out_dir = os.path.dirname(output_plot)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw MICCAI-style boxplot for triage CSV (good/borderline/bad)."
    )
    parser.add_argument("--input_csv", required=True, help="Input CSV path")
    parser.add_argument(
        "--output_plot",
        required=True,
        help="Output plot path (e.g., *.png or *.pdf)",
    )
    parser.add_argument(
        "--stats_csv",
        default=None,
        help="Optional output CSV for group statistics",
    )
    parser.add_argument(
        "--group_col",
        default="auto",
        help="Group column name or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--score_col",
        default="Dice_Mean",
        help="Metric column to plot (default: Dice_Mean)",
    )
    parser.add_argument(
        "--title",
        default="Dice Distribution by Triage Group",
        help="Figure title",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.score_col not in df.columns:
        raise ValueError(f"score column '{args.score_col}' not found in CSV.")

    group3, selected_col = derive_three_groups(df, args.group_col)
    plot_df = df.copy()
    plot_df["Group3"] = group3
    plot_df[args.score_col] = pd.to_numeric(plot_df[args.score_col], errors="coerce")
    plot_df = plot_df[plot_df["Group3"].isin(GROUP_ORDER) & plot_df[args.score_col].notna()].copy()

    if plot_df.empty:
        raise ValueError("No valid rows after group/score filtering.")

    stats = build_stats(plot_df, args.score_col)
    if args.stats_csv:
        stats_dir = os.path.dirname(args.stats_csv)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        stats.to_csv(args.stats_csv, index=False)

    plot_miccai_boxplot(plot_df, args.score_col, args.output_plot, args.title)

    print(f"Input CSV: {args.input_csv}")
    print(f"Selected group column: {selected_col}")
    print(f"Score column: {args.score_col}")
    print(f"Saved plot: {args.output_plot}")
    if args.stats_csv:
        print(f"Saved stats CSV: {args.stats_csv}")
    print("\nGroup counts:")
    print(plot_df["Group3"].value_counts().reindex(GROUP_ORDER, fill_value=0).to_string())
    print("\nGroup means:")
    print(plot_df.groupby("Group3")[args.score_col].mean().reindex(GROUP_ORDER).to_string())


if __name__ == "__main__":
    main()
