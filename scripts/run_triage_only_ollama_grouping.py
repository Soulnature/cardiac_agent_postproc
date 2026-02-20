#!/usr/bin/env python3
"""
Run triage-only grouping with local Ollama (ministral-3:14b) and visualize Dice differences.

This script follows the triage-only output style used in run_batch_repair.py:
  Stem, TriageCategory, TriageScore, VLM_Quality, VLM_Confidence, Issues

It also appends Dice metrics and generates a group comparison plot.

Usage:
    conda activate FDA
    python scripts/run_triage_only_ollama_grouping.py \
        --config config/ollama_ministral_triage.yaml \
        --output_csv batch_eval_results/triage_only_ollama_grouping_all_cases.csv \
        --output_plot batch_eval_results/triage_only_ollama_dice_diff_all_cases.png \
        --stats_csv batch_eval_results/triage_only_ollama_dice_stats_all_cases.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd
import seaborn as sns
import yaml

# Make Matplotlib cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cardiac_agent_postproc.agents.message_bus import MessageBus
from cardiac_agent_postproc.agents.triage_agent import TriageAgent
from cardiac_agent_postproc.agents.visual_knowledge import VisualKnowledgeBase
from cardiac_agent_postproc.eval_metrics import dice_macro
from cardiac_agent_postproc.io_utils import read_mask
from cardiac_agent_postproc.orchestrator import discover_cases


def load_cfg(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_dice_metrics(pred_mask, gt_path: str) -> Dict[str, float]:
    if not gt_path or not os.path.exists(gt_path):
        return {"Dice_Mean": -1.0, "Dice_RV": -1.0, "Dice_Myo": -1.0, "Dice_LV": -1.0}
    gt = read_mask(gt_path)
    d_mean, d_rv, d_myo, d_lv = dice_macro(pred_mask, gt)
    return {
        "Dice_Mean": float(d_mean),
        "Dice_RV": float(d_rv),
        "Dice_Myo": float(d_myo),
        "Dice_LV": float(d_lv),
    }


def save_group_dice_plot(df: pd.DataFrame, output_plot: str, stats_csv: str) -> None:
    plot_df = df[df["TriageCategory"].isin(["good", "needs_fix"])].copy()
    plot_df = plot_df[plot_df["Dice_Mean"] >= 0].copy()

    stats = (
        plot_df.groupby("TriageCategory")["Dice_Mean"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "mean": "Dice_Mean_mean",
                "median": "Dice_Mean_median",
                "std": "Dice_Mean_std",
                "min": "Dice_Mean_min",
                "max": "Dice_Mean_max",
            }
        )
    )
    stats.to_csv(stats_csv, index=False)

    if plot_df.empty:
        print("No valid Dice rows found for plotting.")
        return

    sns.set_theme(style="whitegrid")
    order = [c for c in ["good", "needs_fix"] if c in set(plot_df["TriageCategory"])]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    sns.boxplot(
        data=plot_df,
        x="TriageCategory",
        y="Dice_Mean",
        order=order,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="TriageCategory",
        y="Dice_Mean",
        order=order,
        color="black",
        size=3,
        alpha=0.45,
        jitter=0.2,
        ax=ax,
    )

    ax.set_title("Dice Distribution by Triage-Only Group (Ollama ministral-3:14b)")
    ax.set_xlabel("Triage Group")
    ax.set_ylabel("Dice_Mean")
    ax.grid(axis="y", alpha=0.3)

    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for i, cat in enumerate(order):
        mean_val = float(plot_df.loc[plot_df["TriageCategory"] == cat, "Dice_Mean"].mean())
        ax.text(i, y1 - 0.03 * span, f"{mean_val:.3f}", ha="center", va="top", fontsize=9)

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_plot, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Triage-only grouping with local Ollama (ministral-3:14b) + Dice visualization"
    )
    parser.add_argument(
        "--config",
        default="config/ollama_ministral_triage.yaml",
        help="YAML config path (default: config/ollama_ministral_triage.yaml)",
    )
    parser.add_argument(
        "--output_csv",
        default="batch_eval_results/triage_only_ollama_grouping.csv",
        help="Output CSV path for triage-only grouping",
    )
    parser.add_argument(
        "--output_plot",
        default="batch_eval_results/triage_only_ollama_dice_diff.png",
        help="Output plot path for Dice group comparison",
    )
    parser.add_argument(
        "--stats_csv",
        default="batch_eval_results/triage_only_ollama_dice_stats.csv",
        help="Output CSV path for Dice summary stats by group",
    )
    parser.add_argument(
        "--debug_dir",
        default="batch_eval_results/triage_only_ollama_debug",
        help="Directory for triage overlay debug images",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of cases")
    parser.add_argument("--case_contains", type=str, default=None, help="Optional stem substring filter")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    source_dir = cfg["paths"]["source_dir"]
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.debug_dir, exist_ok=True)

    # Print config summary for verification
    llm_cfg = cfg.get("llm", {})
    vlm_cfg = cfg.get("vision_guardrail", {})
    triage_agent_cfg = cfg.get("agents", {}).get("triage", {})
    print(f"LLM: {llm_cfg.get('model', 'N/A')} @ {llm_cfg.get('base_url', 'N/A')}")
    print(f"VLM: {vlm_cfg.get('model', 'N/A')} @ {vlm_cfg.get('base_url', 'N/A')}")
    print(f"Triage Agent Override: {triage_agent_cfg.get('model', 'N/A')} @ {triage_agent_cfg.get('base_url', 'N/A')}")

    # Same visual KB loading strategy as run_batch_repair.py
    vk_dir = os.path.dirname(source_dir)
    visual_kb = VisualKnowledgeBase.load(vk_dir) if os.path.isdir(vk_dir) else None
    if visual_kb is None:
        print("Warning: visual KB not loaded. VLM triage may degrade.")

    bus = MessageBus()
    triage_agent = TriageAgent(cfg, bus, visual_kb=visual_kb)

    all_cases = discover_cases(source_dir, "", "")
    all_cases = sorted(all_cases, key=lambda c: c.stem)

    if args.case_contains:
        all_cases = [c for c in all_cases if args.case_contains in c.stem]
    if args.limit is not None:
        all_cases = all_cases[: args.limit]

    print(f"Source dir: {source_dir}")
    print(f"Total cases to process: {len(all_cases)}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Output Plot: {args.output_plot}")

    rows: List[Dict[str, Any]] = []

    # Resume: skip cases already in output CSV
    processed_stems = set()
    if os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
            processed_stems = set(existing_df["Stem"].tolist())
            rows = existing_df.to_dict("records")
            print(f"Resuming: found {len(processed_stems)} already processed cases.")
        except Exception as e:
            print(f"Warning: could not read existing CSV for resume: {e}")

    for ctx in tqdm(all_cases, desc="Triage-only Ollama"):
        if ctx.stem in processed_stems:
            continue

        t0 = time.time()
        ctx.output_dir = args.debug_dir
        if ctx.mask is not None and ctx.original_mask is None:
            ctx.original_mask = ctx.mask.copy()
        if ctx.mask is not None and ctx.current_mask is None:
            ctx.current_mask = ctx.mask.copy()

        gt_path = os.path.join(source_dir, f"{ctx.stem}_gt.png")
        if os.path.exists(gt_path):
            ctx.gt_path = gt_path
        dice = compute_dice_metrics(ctx.mask, ctx.gt_path) if ctx.mask is not None else {
            "Dice_Mean": -1.0,
            "Dice_RV": -1.0,
            "Dice_Myo": -1.0,
            "Dice_LV": -1.0,
        }

        try:
            bus.register_case(ctx)
            triage_agent.process_case(ctx)
            vlm_data = getattr(ctx, "vlm_verdict_data", {}) or {}
            row = {
                "Stem": ctx.stem,
                "TriageCategory": ctx.triage_category,
                "TriageScore": float(ctx.triage_score),
                "TriageBadProb": float(getattr(ctx, "triage_bad_prob", 1.0 - float(ctx.triage_score))),
                "TriageSeverity": float(getattr(ctx, "triage_severity", -1.0)),
                "VLM_Quality": getattr(ctx, "vlm_quality", "N/A"),
                "VLM_Score": float(getattr(ctx, "vlm_score", vlm_data.get("score", -1.0))),
                "VLM_GoodProb": float(getattr(ctx, "vlm_good_prob", -1.0)),
                "VLM_Confidence": float(vlm_data.get("confidence", 0.0)),
                "QM_BadProb": float(getattr(ctx, "qm_bad_prob", 0.0)),
                "Issues": json.dumps(ctx.triage_issues, ensure_ascii=False),
                "Dice_Mean": dice["Dice_Mean"],
                "Dice_RV": dice["Dice_RV"],
                "Dice_Myo": dice["Dice_Myo"],
                "Dice_LV": dice["Dice_LV"],
                "Time_s": round(time.time() - t0, 2),
            }
        except Exception as e:
            row = {
                "Stem": ctx.stem,
                "TriageCategory": "error",
                "TriageScore": 0.0,
                "TriageBadProb": 1.0,
                "TriageSeverity": -1.0,
                "VLM_Quality": "error",
                "VLM_Score": -1.0,
                "VLM_GoodProb": -1.0,
                "VLM_Confidence": 0.0,
                "QM_BadProb": 0.0,
                "Issues": json.dumps([f"exception: {e}"], ensure_ascii=False),
                "Dice_Mean": dice["Dice_Mean"],
                "Dice_RV": dice["Dice_RV"],
                "Dice_Myo": dice["Dice_Myo"],
                "Dice_LV": dice["Dice_LV"],
                "Time_s": round(time.time() - t0, 2),
            }

        rows.append(row)

        # Incremental save every 10 cases
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved grouping CSV: {args.output_csv}")

    save_group_dice_plot(out_df, args.output_plot, args.stats_csv)
    print(f"Saved Dice plot: {args.output_plot}")
    print(f"Saved Dice stats: {args.stats_csv}")

    if not out_df.empty:
        print("\nGroup counts:")
        print(out_df["TriageCategory"].value_counts().to_string())
        valid = out_df[out_df["TriageCategory"].isin(["good", "needs_fix"]) & (out_df["Dice_Mean"] >= 0)]
        if not valid.empty:
            print("\nDice mean by group:")
            print(valid.groupby("TriageCategory")["Dice_Mean"].mean().to_string())


if __name__ == "__main__":
    main()
