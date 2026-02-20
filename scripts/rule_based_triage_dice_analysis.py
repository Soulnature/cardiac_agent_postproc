from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cardiac_agent_postproc.agents.triage_agent import _detect_issues
from cardiac_agent_postproc.eval_metrics import dice_macro
from cardiac_agent_postproc.io_utils import read_image, read_mask
from cardiac_agent_postproc.quality_model import extract_quality_features


@dataclass
class CaseRuleResult:
    stem: str
    view_type: str
    dice_mean: float
    dice_rv: float
    dice_myo: float
    dice_lv: float
    issue_count: int
    issues: List[str]
    rule_category: str


def _infer_view_type(stem: str) -> str:
    s = stem.lower()
    if "2ch" in s:
        return "2ch"
    if "3ch" in s:
        return "3ch"
    if "4ch" in s:
        return "4ch"
    return "4ch"


def _iter_stems(source_dir: str) -> Sequence[str]:
    for fn in sorted(os.listdir(source_dir)):
        if fn.endswith("_pred.png"):
            yield fn[:-9]


def _collect_results(source_dir: str) -> List[CaseRuleResult]:
    records: List[CaseRuleResult] = []
    stems = list(_iter_stems(source_dir))
    total = len(stems)

    for i, stem in enumerate(stems, start=1):
        pred_path = os.path.join(source_dir, f"{stem}_pred.png")
        gt_path = os.path.join(source_dir, f"{stem}_gt.png")
        img_path = os.path.join(source_dir, f"{stem}_img.png")

        if not (os.path.exists(pred_path) and os.path.exists(gt_path) and os.path.exists(img_path)):
            continue

        try:
            pred = read_mask(pred_path)
            gt = read_mask(gt_path)
            img = read_image(img_path)
            view_type = _infer_view_type(stem)

            feats = extract_quality_features(pred, img, view_type)
            issues = sorted(_detect_issues(feats))
            rule_category = "needs_fix" if issues else "good"

            d_mean, d_rv, d_myo, d_lv = dice_macro(pred, gt)
            records.append(
                CaseRuleResult(
                    stem=stem,
                    view_type=view_type,
                    dice_mean=d_mean,
                    dice_rv=d_rv,
                    dice_myo=d_myo,
                    dice_lv=d_lv,
                    issue_count=len(issues),
                    issues=issues,
                    rule_category=rule_category,
                )
            )
        except Exception as e:
            print(f"[WARN] Failed on {stem}: {e}")

        if i % 50 == 0 or i == total:
            print(f"[INFO] Processed {i}/{total}")

    return records


def _to_dataframe(records: List[CaseRuleResult]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in records:
        rows.append(
            {
                "Stem": r.stem,
                "ViewType": r.view_type,
                "Dice_Mean": r.dice_mean,
                "Dice_RV": r.dice_rv,
                "Dice_Myo": r.dice_myo,
                "Dice_LV": r.dice_lv,
                "IssueCount": r.issue_count,
                "Issues": json.dumps(r.issues),
                "RuleCategory": r.rule_category,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["IssueCountBin"] = pd.cut(
        df["IssueCount"],
        bins=[-0.1, 0.5, 1.5, 2.5, 999],
        labels=["0", "1", "2", "3+"],
    )
    return df


def _classification_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    gt_bad = df["Dice_Mean"] < threshold
    pred_bad = df["RuleCategory"] == "needs_fix"
    tp = int((gt_bad & pred_bad).sum())
    tn = int((~gt_bad & ~pred_bad).sum())
    fp = int((~gt_bad & pred_bad).sum())
    fn = int((gt_bad & ~pred_bad).sum())
    total = len(df)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "threshold": threshold,
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def _plot_dice_by_rule_category(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="RuleCategory",
        y="Dice_Mean",
        order=["good", "needs_fix"],
        showfliers=False,
    )
    sns.stripplot(
        data=df.sample(min(len(df), 200), random_state=0),
        x="RuleCategory",
        y="Dice_Mean",
        order=["good", "needs_fix"],
        alpha=0.4,
        jitter=0.15,
        size=3,
        color="black",
    )
    plt.title("Dice by Rule-Based Triage Group")
    plt.xlabel("Rule Group")
    plt.ylabel("Dice (mean)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_dice_by_issue_count(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="IssueCountBin",
        y="Dice_Mean",
        order=["0", "1", "2", "3+"],
        showfliers=False,
    )
    plt.title("Dice by Number of Rule Issues")
    plt.xlabel("Issue Count Bin")
    plt.ylabel("Dice (mean)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_top_issue_effect(df: pd.DataFrame, out_path: str, min_cases: int = 5) -> None:
    issue_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        issues = json.loads(row["Issues"])
        for issue in issues:
            issue_rows.append({"Issue": issue, "Dice_Mean": row["Dice_Mean"]})
    issue_df = pd.DataFrame(issue_rows)
    if issue_df.empty:
        return

    counts = issue_df["Issue"].value_counts()
    keep = counts[counts >= min_cases].index.tolist()
    if not keep:
        return

    mean_dice = (
        issue_df[issue_df["Issue"].isin(keep)]
        .groupby("Issue", as_index=False)["Dice_Mean"]
        .mean()
        .sort_values("Dice_Mean")
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=mean_dice, x="Dice_Mean", y="Issue", orient="h")
    plt.title(f"Mean Dice for Frequent Rule Issues (n >= {min_cases})")
    plt.xlabel("Mean Dice")
    plt.ylabel("Issue")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--source_dir", default="")
    parser.add_argument("--output_dir", default="batch_eval_results/rule_based_triage")
    parser.add_argument("--dice_thresholds", default="0.85,0.90,0.92")
    parser.add_argument("--min_issue_cases", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    source_dir = args.source_dir or cfg["paths"]["source_dir"]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] source_dir={source_dir}")
    print("[INFO] Running rule-only triage grouping...")
    records = _collect_results(source_dir)
    df = _to_dataframe(records)
    if df.empty:
        print("[ERROR] No valid cases found.")
        return

    csv_path = os.path.join(out_dir, "rule_triage_results.csv")
    df.to_csv(csv_path, index=False)

    group_stats = (
        df.groupby("RuleCategory", as_index=False)["Dice_Mean"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    group_stats_path = os.path.join(out_dir, "group_stats.csv")
    group_stats.to_csv(group_stats_path, index=False)

    issue_bin_stats = (
        df.groupby("IssueCountBin", as_index=False)["Dice_Mean"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    issue_bin_stats_path = os.path.join(out_dir, "issue_count_stats.csv")
    issue_bin_stats.to_csv(issue_bin_stats_path, index=False)

    thresholds = [float(x.strip()) for x in args.dice_thresholds.split(",") if x.strip()]
    metric_rows = [_classification_metrics(df, th) for th in thresholds]
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = os.path.join(out_dir, "rule_triage_metrics_by_threshold.csv")
    metrics_df.to_csv(metrics_path, index=False)

    _plot_dice_by_rule_category(df, os.path.join(out_dir, "dice_by_rule_category.png"))
    _plot_dice_by_issue_count(df, os.path.join(out_dir, "dice_by_issue_count.png"))
    _plot_top_issue_effect(
        df,
        os.path.join(out_dir, "dice_by_frequent_issue.png"),
        min_cases=args.min_issue_cases,
    )

    report_path = os.path.join(out_dir, "summary.txt")
    good_mean = float(df[df["RuleCategory"] == "good"]["Dice_Mean"].mean())
    fix_mean = float(df[df["RuleCategory"] == "needs_fix"]["Dice_Mean"].mean())
    mean_gap = good_mean - fix_mean
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Rule-only triage Dice analysis\n")
        f.write("=============================\n")
        f.write(f"Total cases: {len(df)}\n")
        f.write(f"good count: {(df['RuleCategory'] == 'good').sum()}\n")
        f.write(f"needs_fix count: {(df['RuleCategory'] == 'needs_fix').sum()}\n")
        f.write(f"mean Dice(good): {good_mean:.4f}\n")
        f.write(f"mean Dice(needs_fix): {fix_mean:.4f}\n")
        f.write(f"mean gap: {mean_gap:.4f}\n\n")
        f.write("Classification metrics by Dice threshold:\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n")

    print(f"[DONE] Results saved to {out_dir}")
    print(f"[DONE] Main plot: {os.path.join(out_dir, 'dice_by_rule_category.png')}")
    print(f"[DONE] Summary: {report_path}")


if __name__ == "__main__":
    main()
