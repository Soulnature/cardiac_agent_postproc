"""
MICCAI-style Triage Comparison Figure
Compares: Ollama (outputs_ollama_all_cases) vs GPT-4o (exp_mnm2_gpt4o_allcases_20260220_bg_after_preflight)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = "/data484_5/xzhao14/cardiac_agent_postproc/results"
EXPS = {
    "Ollama": f"{BASE}/outputs_ollama_all_cases",
    "GPT-4o": f"{BASE}/exp_mnm2_gpt4o_allcases_20260220_bg_after_preflight",
}
OUT = f"{BASE}/triage_comparison_miccai.pdf"
OUT_PNG = f"{BASE}/triage_comparison_miccai.png"

# ─── MICCAI colour palette ────────────────────────────────────────────────────
C_OLLAMA  = "#1f77b4"   # steel blue
C_GPT4O   = "#d62728"   # crimson
C_GOOD    = "#2ca02c"   # green
C_FIX     = "#ff7f0e"   # orange
C_LIGHT1  = "#aec7e8"
C_LIGHT2  = "#ffbb78"
GREY      = "#555555"

# ─── Typography ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# ─── Load data ────────────────────────────────────────────────────────────────
def load(exp_dir):
    metrics  = pd.read_csv(f"{exp_dir}/triage/triage_curated_binary_metrics.csv")
    metrics  = dict(zip(metrics["metric"], metrics["value"]))
    eval_df  = pd.read_csv(f"{exp_dir}/triage/triage_curated_eval_rows.csv")
    dice_grp = pd.read_csv(f"{exp_dir}/triage/triage_dice_group_stats.csv")
    triage   = pd.read_csv(f"{exp_dir}/triage/triage_results.csv")
    return metrics, eval_df, dice_grp, triage

m_ol, ev_ol, dg_ol, tr_ol = load(EXPS["Ollama"])
m_gp, ev_gp, dg_gp, tr_gp = load(EXPS["GPT-4o"])

# ─── Layout ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
gs  = gridspec.GridSpec(3, 4, figure=fig,
                        wspace=0.45, hspace=0.58,
                        left=0.06, right=0.97, top=0.93, bottom=0.07)

# Row 0: A(metrics bar) | B(ROC) | C(confusion Ollama) | D(confusion GPT4o)
# Row 1: E(Dice violin Ollama) | F(Dice violin GPT4o) | G(Dice per-class bar) | H(sample dist)
# Row 2: I(score distribution) | J(Dice scatter) | K(FN breakdown) | (empty)

ax_A  = fig.add_subplot(gs[0, 0])
ax_B  = fig.add_subplot(gs[0, 1])
ax_C  = fig.add_subplot(gs[0, 2])
ax_D  = fig.add_subplot(gs[0, 3])
ax_E  = fig.add_subplot(gs[1, 0])
ax_F  = fig.add_subplot(gs[1, 1])
ax_G  = fig.add_subplot(gs[1, 2])
ax_H  = fig.add_subplot(gs[1, 3])
ax_I  = fig.add_subplot(gs[2, 0])
ax_J  = fig.add_subplot(gs[2, 1])
ax_K  = fig.add_subplot(gs[2, 2])
ax_L  = fig.add_subplot(gs[2, 3])

# ─── Helper ──────────────────────────────────────────────────────────────────
def panel_label(ax, letter, x=-0.18, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

def spine_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Classification metrics bar chart
# ─────────────────────────────────────────────────────────────────────────────
labels  = ["AUC", "F1", "Accuracy", "Precision\n(needs_fix)", "Recall\n(needs_fix)", "Recall\n(good)"]
keys    = ["AUC", "F1_needs_fix", "ACC", "Precision_needs_fix", "Recall_needs_fix", "Recall_good"]
v_ol    = [m_ol[k] for k in keys]
v_gp    = [m_gp[k] for k in keys]

x = np.arange(len(labels))
w = 0.35
bars1 = ax_A.bar(x - w/2, v_ol, w, color=C_OLLAMA, alpha=0.85, edgecolor="white", linewidth=0.5, label="Ollama")
bars2 = ax_A.bar(x + w/2, v_gp, w, color=C_GPT4O,  alpha=0.85, edgecolor="white", linewidth=0.5, label="GPT-4o")

for b, v in zip(bars1, v_ol):
    ax_A.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
              f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=C_OLLAMA, fontweight="bold")
for b, v in zip(bars2, v_gp):
    ax_A.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
              f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=C_GPT4O,  fontweight="bold")

ax_A.set_xticks(x); ax_A.set_xticklabels(labels, fontsize=6.5, rotation=0)
ax_A.set_ylim(0.5, 1.13); ax_A.set_ylabel("Score")
ax_A.set_title("(A) Triage Classification Metrics\n(annotated subset, n=49)", fontweight="bold")
ax_A.legend(loc="lower right", framealpha=0.9, edgecolor="none")
ax_A.axhline(0.9, color="gray", lw=0.6, ls="--", alpha=0.5)
spine_style(ax_A); panel_label(ax_A, "A")

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: ROC curves
# ─────────────────────────────────────────────────────────────────────────────
for ev, label, color, ls in [
        (ev_ol, "Ollama",  C_OLLAMA, "-"),
        (ev_gp, "GPT-4o",  C_GPT4O,  "--"),
]:
    fpr, tpr, _ = roc_curve(ev["y_true"], ev["y_score"])
    roc_auc = auc(fpr, tpr)
    ax_B.plot(fpr, tpr, color=color, lw=1.8, ls=ls,
              label=f"{label} (AUC={roc_auc:.3f})")

ax_B.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.4)
ax_B.fill_between(*roc_curve(ev_ol["y_true"], ev_ol["y_score"])[:2],
                  alpha=0.08, color=C_OLLAMA)
ax_B.fill_between(*roc_curve(ev_gp["y_true"], ev_gp["y_score"])[:2],
                  alpha=0.08, color=C_GPT4O)
ax_B.set_xlabel("False Positive Rate"); ax_B.set_ylabel("True Positive Rate")
ax_B.set_title("(B) ROC Curve\n(needs_fix detection)", fontweight="bold")
ax_B.legend(loc="lower right", framealpha=0.9, edgecolor="none")
ax_B.set_xlim(-0.02, 1.02); ax_B.set_ylim(-0.02, 1.05)
spine_style(ax_B); panel_label(ax_B, "B")

# ─────────────────────────────────────────────────────────────────────────────
# Panels C & D: Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────
def draw_cm(ax, ev, title, color_tp):
    cm = confusion_matrix(ev["y_true"], ev["y_pred"])
    # rows: true 0(good), 1(needs_fix)  cols: pred 0,1
    labels_cm = ["good", "needs_fix"]
    tp_pos = [(0,0),(1,1)]
    base_c = "#f5f5f5"
    cmap = np.array([[base_c, "#fee8d5"], ["#d4e9f7", "#c8e6c9"]])
    cmap[0][0] = "#c8e6c9" if color_tp == "green" else "#e8f5e9"
    cmap[1][1] = "#c8e6c9"

    for i in range(2):
        for j in range(2):
            face = "#c8e6c9" if i==j else "#ffcdd2"
            ax.add_patch(FancyBboxPatch((j-0.42, i-0.42), 0.84, 0.84,
                         boxstyle="round,pad=0.03", fc=face, ec="white", lw=1.2))
            v = cm[i,j]
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if v > cm.max()*0.3 else GREY)
            pct = 100.*v/cm[i,:].sum()
            ax.text(j, i+0.26, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=6, color=GREY)

    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.55, 1.55)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred: good","Pred: needs_fix"], fontsize=7)
    ax.set_yticks([0,1]); ax.set_yticklabels(["True: good","True: needs_fix"], fontsize=7)
    ax.set_title(title, fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)

draw_cm(ax_C, ev_ol, "(C) Confusion Matrix\nOllama", "green")
draw_cm(ax_D, ev_gp, "(D) Confusion Matrix\nGPT-4o", "green")
panel_label(ax_C, "C"); panel_label(ax_D, "D")

# ─────────────────────────────────────────────────────────────────────────────
# Panels E & F: Dice violin plots by triage category
# ─────────────────────────────────────────────────────────────────────────────
def dice_violin(ax, triage_df, title, c_good, c_fix):
    good_dice = triage_df.loc[triage_df["TriageCategory"]=="good", "Dice_Mean"].dropna()
    fix_dice  = triage_df.loc[triage_df["TriageCategory"]=="needs_fix", "Dice_Mean"].dropna()

    parts = ax.violinplot([good_dice, fix_dice], positions=[0,1],
                          showmedians=True, showextrema=True, widths=0.6)
    parts["bodies"][0].set_facecolor(c_good); parts["bodies"][0].set_alpha(0.6)
    parts["bodies"][1].set_facecolor(c_fix);  parts["bodies"][1].set_alpha(0.6)
    for key in ["cmedians","cbars","cmins","cmaxes"]:
        if key in parts:
            parts[key].set_color(GREY); parts[key].set_linewidth(1.2)

    # overlay box quartiles
    for i, data in enumerate([good_dice, fix_dice]):
        q1, med, q3 = np.percentile(data, [25,50,75])
        ax.plot([i-0.1, i+0.1], [med, med], "k-", lw=1.5)
        ax.add_patch(FancyBboxPatch((i-0.12, q1), 0.24, q3-q1,
                     boxstyle="round,pad=0.005", fc="none", ec="black", lw=1.0))

    # jitter
    np.random.seed(42)
    for i, data in enumerate([good_dice, fix_dice]):
        jitter = np.random.uniform(-0.18, 0.18, size=len(data))
        ax.scatter(i + jitter, data, s=4, alpha=0.35,
                   color=[c_good, c_fix][i], zorder=2)

    # significance
    stat, p = stats.mannwhitneyu(good_dice, fix_dice, alternative="two-sided")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
    ax.plot([0, 1], [0.99, 0.99], "k-", lw=0.8)
    ax.text(0.5, 0.995, sig, ha="center", va="bottom", fontsize=9, color="black")
    ax.text(0.5, 1.005, f"p={p:.2e}", ha="center", va="bottom", fontsize=6, color=GREY)

    ax.set_xticks([0,1])
    ax.set_xticklabels([f"Good\n(n={len(good_dice)})", f"Needs Fix\n(n={len(fix_dice)})"])
    ax.set_ylabel("Dice (Mean)")
    ax.set_ylim(0.30, 1.03)
    ax.set_title(title, fontweight="bold")
    spine_style(ax)

dice_violin(ax_E, tr_ol, "(E) Dice by Triage Group\nOllama", C_GOOD, C_FIX)
dice_violin(ax_F, tr_gp, "(F) Dice by Triage Group\nGPT-4o", C_GOOD, C_FIX)
panel_label(ax_E, "E"); panel_label(ax_F, "F")

# ─────────────────────────────────────────────────────────────────────────────
# Panel G: Per-class Dice comparison (RV, Myo, LV) by triage group & model
# ─────────────────────────────────────────────────────────────────────────────
classes  = ["RV", "Myo", "LV"]
dcols    = ["Dice_RV", "Dice_Myo", "Dice_LV"]

means_ol_good = [tr_ol.loc[tr_ol.TriageCategory=="good", c].mean() for c in dcols]
means_ol_fix  = [tr_ol.loc[tr_ol.TriageCategory=="needs_fix", c].mean() for c in dcols]
means_gp_good = [tr_gp.loc[tr_gp.TriageCategory=="good", c].mean() for c in dcols]
means_gp_fix  = [tr_gp.loc[tr_gp.TriageCategory=="needs_fix", c].mean() for c in dcols]

x = np.arange(len(classes))
w = 0.2
ax_G.bar(x - 1.5*w, means_ol_good, w, color=C_OLLAMA, alpha=0.85, label="Ollama / Good", hatch="")
ax_G.bar(x - 0.5*w, means_ol_fix,  w, color=C_OLLAMA, alpha=0.50, label="Ollama / Needs Fix", hatch="//")
ax_G.bar(x + 0.5*w, means_gp_good, w, color=C_GPT4O,  alpha=0.85, label="GPT-4o / Good", hatch="")
ax_G.bar(x + 1.5*w, means_gp_fix,  w, color=C_GPT4O,  alpha=0.50, label="GPT-4o / Needs Fix", hatch="//")

for xpos, val in zip(x - 1.5*w, means_ol_good):
    ax_G.text(xpos, val+0.003, f"{val:.3f}", ha="center", fontsize=5.5, color=C_OLLAMA)
for xpos, val in zip(x - 0.5*w, means_ol_fix):
    ax_G.text(xpos, val+0.003, f"{val:.3f}", ha="center", fontsize=5.5, color=C_OLLAMA, alpha=0.7)
for xpos, val in zip(x + 0.5*w, means_gp_good):
    ax_G.text(xpos, val+0.003, f"{val:.3f}", ha="center", fontsize=5.5, color=C_GPT4O)
for xpos, val in zip(x + 1.5*w, means_gp_fix):
    ax_G.text(xpos, val+0.003, f"{val:.3f}", ha="center", fontsize=5.5, color=C_GPT4O, alpha=0.7)

ax_G.set_xticks(x); ax_G.set_xticklabels(classes)
ax_G.set_ylim(0.75, 1.02); ax_G.set_ylabel("Mean Dice")
ax_G.set_title("(G) Per-class Dice by\nTriage Group & Model", fontweight="bold")
ax_G.legend(loc="lower right", fontsize=6, framealpha=0.9, edgecolor="none", ncol=2)
spine_style(ax_G); panel_label(ax_G, "G")

# ─────────────────────────────────────────────────────────────────────────────
# Panel H: Sample distribution stacked bar
# ─────────────────────────────────────────────────────────────────────────────
n_total = 320
n_ol_good = (tr_ol["TriageCategory"] == "good").sum()
n_ol_fix  = (tr_ol["TriageCategory"] == "needs_fix").sum()
n_gp_good = (tr_gp["TriageCategory"] == "good").sum()
n_gp_fix  = (tr_gp["TriageCategory"] == "needs_fix").sum()

models = ["Ollama", "GPT-4o"]
goods  = [n_ol_good, n_gp_good]
fixes  = [n_ol_fix,  n_gp_fix]
pct_g  = [g/n_total*100 for g in goods]
pct_f  = [f/n_total*100 for f in fixes]

ax_H.bar(models, pct_f,  color=C_FIX,  alpha=0.85, label="Needs Fix", edgecolor="white", lw=0.5)
ax_H.bar(models, pct_g,  color=C_GOOD, alpha=0.85, label="Good",
         bottom=pct_f, edgecolor="white", lw=0.5)

for i, (m, g, f, pg, pf) in enumerate(zip(models, goods, fixes, pct_g, pct_f)):
    ax_H.text(i, pf/2,       f"{f}\n({pf:.1f}%)", ha="center", va="center",
              fontsize=7, color="white", fontweight="bold")
    ax_H.text(i, pf + pg/2,  f"{g}\n({pg:.1f}%)", ha="center", va="center",
              fontsize=7, color="white", fontweight="bold")

ax_H.set_ylim(0, 115)
ax_H.set_ylabel("Percentage (%)")
ax_H.set_title("(H) Triage Sample Distribution\n(N=320 total)", fontweight="bold")
ax_H.legend(loc="upper right", framealpha=0.9, edgecolor="none")
spine_style(ax_H); panel_label(ax_H, "H")

# ─────────────────────────────────────────────────────────────────────────────
# Panel I: TriageScore / BadProb distributions (KDE)
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde

for data, color, label, ls in [
        (tr_ol["TriageBadProb"], C_OLLAMA, "Ollama", "-"),
        (tr_gp["TriageBadProb"], C_GPT4O,  "GPT-4o", "--"),
]:
    xgrid = np.linspace(0, 1, 300)
    kde   = gaussian_kde(data.dropna(), bw_method=0.15)
    ax_I.plot(xgrid, kde(xgrid), color=color, lw=1.8, ls=ls, label=label)
    ax_I.fill_between(xgrid, kde(xgrid), alpha=0.12, color=color)

ax_I.axvline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7, label="Decision threshold")
ax_I.set_xlabel("Triage Bad Probability")
ax_I.set_ylabel("Density")
ax_I.set_title("(I) Triage Score Distribution\n(all 320 cases)", fontweight="bold")
ax_I.legend(framealpha=0.9, edgecolor="none")
spine_style(ax_I); panel_label(ax_I, "I")

# ─────────────────────────────────────────────────────────────────────────────
# Panel J: Scatter – Dice_Mean vs TriageBadProb, coloured by true label
# ─────────────────────────────────────────────────────────────────────────────
for ev, marker, alpha, label_sfx in [(ev_ol, "o", 0.6, "Ollama"), (ev_gp, "s", 0.6, "GPT-4o")]:
    for ytrue, color, lbl in [(0, C_GOOD, "Good"), (1, C_FIX, "Needs Fix")]:
        sub = ev[ev["y_true"] == ytrue]
        ax_J.scatter(sub["y_score"], sub["Dice_Mean"],
                     s=18, c=color, marker=marker, alpha=alpha,
                     label=f"{label_sfx}/{lbl}" if label_sfx=="Ollama" else None,
                     edgecolors="none")

# distinguish markers with legend
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_GOOD,  markersize=6, label="True: Good"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_FIX,   markersize=6, label="True: Needs Fix"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",   markersize=6, label="Ollama"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="gray",   markersize=6, label="GPT-4o"),
]
ax_J.legend(handles=legend_handles, fontsize=6, framealpha=0.9, edgecolor="none", ncol=2)
ax_J.axvline(0.5, color="gray", lw=0.8, ls=":", alpha=0.6)
ax_J.set_xlabel("Triage Bad Probability")
ax_J.set_ylabel("Dice Mean")
ax_J.set_title("(J) Dice vs. Triage Score\n(annotated subset, n=49)", fontweight="bold")
spine_style(ax_J); panel_label(ax_J, "J")

# ─────────────────────────────────────────────────────────────────────────────
# Panel K: FN analysis – Dice of False Negatives
# ─────────────────────────────────────────────────────────────────────────────
fn_ol = ev_ol[(ev_ol["y_true"]==1) & (ev_ol["y_pred"]==0)]  # FN Ollama (should be 0)
fn_gp = ev_gp[(ev_gp["y_true"]==1) & (ev_gp["y_pred"]==0)]  # FN GPT-4o (12 samples)

tp_ol = ev_ol[(ev_ol["y_true"]==1) & (ev_ol["y_pred"]==1)]
tp_gp = ev_gp[(ev_gp["y_true"]==1) & (ev_gp["y_pred"]==1)]

# Box plot: Dice distribution of FN vs TP for GPT-4o
data_to_plot = [tp_gp["Dice_Mean"].values, fn_gp["Dice_Mean"].values]
bp = ax_K.boxplot(data_to_plot, positions=[0, 1], widths=0.5,
                  patch_artist=True, notch=False,
                  medianprops=dict(color="black", lw=1.5),
                  whiskerprops=dict(lw=1.0),
                  capprops=dict(lw=1.0))
bp["boxes"][0].set_facecolor(C_GPT4O); bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor("#ff9999"); bp["boxes"][1].set_alpha(0.7)

# jitter
for i, data in enumerate(data_to_plot):
    jx = np.random.uniform(-0.18, 0.18, len(data))
    ax_K.scatter(i + jx, data, s=20, zorder=3,
                 color=[C_GPT4O, "#cc0000"][i], alpha=0.7, edgecolors="white", lw=0.3)

ax_K.set_xticks([0,1])
ax_K.set_xticklabels([f"TP (n={len(tp_gp)})\n(correctly flagged)",
                       f"FN (n={len(fn_gp)})\n(missed, GPT-4o)"])
ax_K.set_ylabel("Dice Mean")
ax_K.set_title("(K) GPT-4o False Negatives:\nDice of missed needs_fix", fontweight="bold")
ax_K.set_ylim(0.35, 1.0)

# annotate with Dice means
ax_K.axhline(fn_gp["Dice_Mean"].mean(), color="#cc0000", lw=0.8, ls="--", alpha=0.7)
ax_K.text(1.28, fn_gp["Dice_Mean"].mean(), f"mean={fn_gp['Dice_Mean'].mean():.3f}",
          va="center", fontsize=6.5, color="#cc0000")
ax_K.text(1.28, tp_gp["Dice_Mean"].mean(), f"mean={tp_gp['Dice_Mean'].mean():.3f}",
          va="center", fontsize=6.5, color=C_GPT4O)
spine_style(ax_K); panel_label(ax_K, "K")

# ─────────────────────────────────────────────────────────────────────────────
# Panel L: Summary table
# ─────────────────────────────────────────────────────────────────────────────
ax_L.axis("off")
rows = [
    ["Metric", "Ollama", "GPT-4o", "Winner"],
    ["AUC",         f"{m_ol['AUC']:.3f}",              f"{m_gp['AUC']:.3f}",              "Ollama ★"],
    ["F1",          f"{m_ol['F1_needs_fix']:.3f}",     f"{m_gp['F1_needs_fix']:.3f}",     "Ollama ★"],
    ["Accuracy",    f"{m_ol['ACC']:.3f}",               f"{m_gp['ACC']:.3f}",              "Ollama ★"],
    ["Recall(fix)", f"{m_ol['Recall_needs_fix']:.3f}", f"{m_gp['Recall_needs_fix']:.3f}", "Ollama ★"],
    ["Precision",   f"{m_ol['Precision_needs_fix']:.3f}", f"{m_gp['Precision_needs_fix']:.3f}", "GPT-4o ★"],
    ["Recall(good)","0.900",                            "1.000",                           "GPT-4o ★"],
    ["FN count",    "0",                                "12",                              "Ollama ★"],
    ["# good",      "53 (16.6%)",                       "238 (74.4%)",                     "—"],
    ["# needs_fix", "267 (83.4%)",                      "82 (25.6%)",                      "—"],
    ["Dice(good)",  f"{dg_ol.loc[dg_ol.TriageCategory=='good','mean'].values[0]:.4f}",
                    f"{dg_gp.loc[dg_gp.TriageCategory=='good','mean'].values[0]:.4f}",    "Ollama ★"],
    ["Dice(fix)",   f"{dg_ol.loc[dg_ol.TriageCategory=='needs_fix','mean'].values[0]:.4f}",
                    f"{dg_gp.loc[dg_gp.TriageCategory=='needs_fix','mean'].values[0]:.4f}", "Ollama ★"],
]

col_widths = [0.30, 0.22, 0.22, 0.26]
col_x      = [0.02, 0.32, 0.54, 0.76]
row_h      = 0.074

for ri, row in enumerate(rows):
    y = 1.0 - ri * row_h
    bg = "#e8f4fd" if ri == 0 else ("#f9f9f9" if ri % 2 == 0 else "white")
    ax_L.add_patch(plt.Rectangle((0, y - row_h), 1, row_h,
                                  fc=bg, ec="none", transform=ax_L.transAxes, zorder=0))
    for ci, (text, cx) in enumerate(zip(row, col_x)):
        fw = "bold" if ri == 0 else "normal"
        fc = C_OLLAMA if "Ollama ★" in text else (C_GPT4O if "GPT-4o ★" in text else "black")
        fc = "black" if ri == 0 else fc
        ax_L.text(cx, y - row_h/2, text,
                  transform=ax_L.transAxes,
                  ha="left", va="center",
                  fontsize=6.8, fontweight=fw, color=fc)

ax_L.set_title("(L) Summary Comparison Table", fontweight="bold", fontsize=9)
panel_label(ax_L, "L")

# ─────────────────────────────────────────────────────────────────────────────
# Figure title
# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Triage Classification Comparison: Ollama vs. GPT-4o  ·  "
    "Cardiac MRI Segmentation Post-processing  ·  MnM-2 Dataset (N=320)",
    fontsize=11, fontweight="bold", y=0.995
)

plt.savefig(OUT,     dpi=300, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"Saved: {OUT_PNG}")
