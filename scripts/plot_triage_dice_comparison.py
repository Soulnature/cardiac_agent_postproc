"""
MICCAI-style figure: Dice comparison between good vs needs_fix across views.
Saves PNG + PDF to the experiment's figures/ directory.
"""
import csv
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats


# ── colour palette (colorblind-safe) ────────────────────────────────────────
C_GOOD = "#4393C3"      # blue
C_FIX  = "#D6604D"      # red-orange
ALPHA_BOX = 0.18
ALPHA_VIOLIN = 0.30

# ── significance thresholds ──────────────────────────────────────────────────
def pval_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load_data(triage_csv):
    rows = list(csv.DictReader(open(triage_csv)))
    data = defaultdict(lambda: defaultdict(list))   # data[view][cat:metric]

    for r in rows:
        cat  = r.get("TriageCategory", "unknown")
        stem = r.get("Stem", "")
        if   "_lax_2c_" in stem: view = "2CH"
        elif "_lax_3c_" in stem: view = "3CH"
        elif "_lax_4c_" in stem: view = "4CH"
        else: continue
        for k in ["Dice_Mean", "Dice_RV", "Dice_Myo", "Dice_LV"]:
            try:
                v = float(r.get(k, "nan"))
                if v == v:
                    # For 2CH, recompute Mean Dice from LV+Myo only (RV absent)
                    if k == "Dice_Mean" and view == "2CH":
                        lv  = float(r.get("Dice_LV",  "nan"))
                        myo = float(r.get("Dice_Myo", "nan"))
                        if lv == lv and myo == myo:
                            data[view][f"{cat}:{k}"].append((lv + myo) / 2.0)
                    else:
                        data[view][f"{cat}:{k}"].append(v)
            except Exception:
                pass
    return data


def draw_sig_bracket(ax, x1, x2, y, p, dy=0.008, lw=1.0, fs=7.5):
    stars = pval_stars(p)
    col = "#333333" if stars != "ns" else "#888888"
    ax.plot([x1, x1, x2, x2], [y, y+dy, y+dy, y], lw=lw, color=col)
    ax.text((x1+x2)/2, y+dy+0.002, stars,
            ha="center", va="bottom", fontsize=fs,
            color=col, fontweight="bold" if stars != "ns" else "normal")


def make_figure(data, out_dir):
    views   = ["2CH", "3CH", "4CH"]
    metrics = [
        ("Dice_Mean", "Mean Dice"),
        ("Dice_RV",   "RV Dice"),
        ("Dice_Myo",  "Myo Dice"),
        ("Dice_LV",   "LV Dice"),
    ]
    cats = [("good", C_GOOD, "Good"), ("needs_fix", C_FIX, "Needs Fix")]

    n_rows = len(metrics)
    n_cols = len(views)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5, 7.8),
        dpi=300,
        sharey="row",
    )
    fig.subplots_adjust(
        left=0.10, right=0.97,
        top=0.91,  bottom=0.08,
        hspace=0.52, wspace=0.18,
    )

    # ── column headers (views) ───────────────────────────────────────────────
    for ci, view in enumerate(views):
        axes[0, ci].set_title(view, fontsize=9, fontweight="bold", pad=6)

    # ── per-cell violin + box ────────────────────────────────────────────────
    for ri, (mkey, mlabel) in enumerate(metrics):
        for ci, view in enumerate(views):
            ax = axes[ri, ci]

            # skip RV for 2CH (all zeros, not meaningful)
            if view == "2CH" and mkey == "Dice_RV":
                ax.text(0.5, 0.5, "RV absent\nin 2CH",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=7.5, color="#aaaaaa", style="italic")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue

            g_vals = np.array(data[view].get(f"good:{mkey}", []))
            f_vals = np.array(data[view].get(f"needs_fix:{mkey}", []))

            positions = [1, 2]
            all_vals  = [g_vals, f_vals]
            colors    = [C_GOOD, C_FIX]

            # ── violin ──────────────────────────────────────────────────────
            for pos, vals, col in zip(positions, all_vals, colors):
                if len(vals) < 3:
                    continue
                try:
                    vp = ax.violinplot(
                        vals, positions=[pos],
                        widths=0.55, showmedians=False,
                        showextrema=False,
                    )
                    for pc in vp["bodies"]:
                        pc.set_facecolor(col)
                        pc.set_alpha(ALPHA_VIOLIN)
                        pc.set_edgecolor("none")
                except Exception:
                    pass

            # ── box plot ─────────────────────────────────────────────────────
            bp = ax.boxplot(
                all_vals,
                positions=positions,
                widths=0.32,
                patch_artist=True,
                notch=False,
                showfliers=True,
                medianprops=dict(color="white", linewidth=1.8),
                whiskerprops=dict(linewidth=0.9, color="#444444"),
                capprops=dict(linewidth=0.9, color="#444444"),
                flierprops=dict(
                    marker="o", markersize=2.0,
                    markerfacecolor="#888888",
                    markeredgecolor="none", alpha=0.6,
                ),
                boxprops=dict(linewidth=0.8),
            )
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.75)

            # ── individual data points (jittered) ────────────────────────────
            rng = np.random.default_rng(42)
            for pos, vals, col in zip(positions, all_vals, colors):
                jitter = rng.uniform(-0.10, 0.10, size=len(vals))
                ax.scatter(
                    pos + jitter, vals,
                    s=4, color=col, alpha=0.45,
                    linewidths=0, zorder=3,
                )

            # ── significance bracket ─────────────────────────────────────────
            if len(g_vals) > 1 and len(f_vals) > 1:
                _, p = stats.ttest_ind(g_vals, f_vals, equal_var=False)
                ymax = max(
                    np.percentile(g_vals, 95),
                    np.percentile(f_vals, 95),
                )
                yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
                draw_sig_bracket(ax, 1, 2, ymax + 0.01, p,
                                 dy=0.006, lw=0.8, fs=7.0)

            # ── axis styling ─────────────────────────────────────────────────
            ax.set_xticks([1, 2])
            ax.set_xticklabels(
                [f"Good\n(n={len(g_vals)})", f"Fix\n(n={len(f_vals)})"],
                fontsize=6.5,
            )
            ax.tick_params(axis="y", labelsize=6.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.spines["left"].set_linewidth(0.6)
            ax.tick_params(width=0.6, length=3)
            ax.set_xlim(0.4, 2.6)

            # fine-tune y-axis per metric
            y_cfg = {
                "Dice_Mean": (0.50, 1.00),
                "Dice_RV":   (0.40, 1.00),
                "Dice_Myo":  (0.68, 0.96),
                "Dice_LV":   (0.82, 1.00),
            }
            ylo, yhi = y_cfg.get(mkey, (0.50, 1.00))
            ax.set_ylim(ylo, yhi + 0.06)
            ax.yaxis.set_major_locator(
                matplotlib.ticker.MultipleLocator(0.05)
            )

        # ── row ylabel ───────────────────────────────────────────────────────
        axes[ri, 0].set_ylabel(mlabel, fontsize=8, labelpad=4)

    # ── global title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Dice Score Comparison: Good vs. Needs-Fix Triage Groups\n"
        "UKB Dataset — ministral-3:14b (local atlas, 2-way VLM refs)  |  2CH Mean = (LV+Myo)/2",
        fontsize=8.0, fontweight="bold", y=0.975,
    )

    # ── legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=C_GOOD, alpha=0.75, label="Good"),
        mpatches.Patch(facecolor=C_FIX,  alpha=0.75, label="Needs Fix"),
        Line2D([0], [0], color="#333333", lw=0.8, label="*  p<0.05"),
        Line2D([0], [0], color="#333333", lw=0.8,
               linestyle="--", label="** p<0.01 / *** p<0.001"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.53, 0.005),
    )

    # ── save ─────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"triage_dice_comparison.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triage_csv",
        default="results/exp_ukb_ollama_ministral3_20260222/triage/triage_results.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="results/exp_ukb_ollama_ministral3_20260222/figures",
    )
    args = parser.parse_args()

    data = load_data(args.triage_csv)
    make_figure(data, args.out_dir)
