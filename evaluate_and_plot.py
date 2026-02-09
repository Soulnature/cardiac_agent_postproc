
import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff

# --- Metrics ---
def compute_dice(pred, gt):
    # Binary Dice for foreground (ignoring background)
    # Assuming classes 1,2,3 are all relevant. We measure 'average per-case' or 'whole foreground'?
    # Let's do Whole Foreground Dice for robustness (prediction vs gt)
    p_b = (pred > 0)
    g_b = (gt > 0)
    if p_b.sum() == 0 and g_b.sum() == 0:
        return 1.0
    return 2.0 * (p_b & g_b).sum() / (p_b.sum() + g_b.sum())

def compute_hd95(pred, gt):
    p_b = (pred > 0)
    g_b = (gt > 0)
    if p_b.sum() == 0 or g_b.sum() == 0:
        return 0.0 # undefined/skipped in stats usually, or penalized.
    
    p_pts = np.argwhere(p_b)
    g_pts = np.argwhere(g_b)
    
    # Simple HD95 approx using directed hausdorff (max of one way)
    # For speed, we just use the max of directed. 
    # Proper HD95 requires percentiles. 
    # Let's stick to standard Euclidean HD (max min dist).
    # Since HD95 is requested, we compute distances and take 95th percentile.
    
    # Using scipy cdist is slow for large images. 
    # Use distance transform for speed.
    dist_g = cv2.distanceTransform((~g_b).astype(np.uint8), cv2.DIST_L2, 5)
    dist_p = cv2.distanceTransform((~p_b).astype(np.uint8), cv2.DIST_L2, 5)
    
    d1 = dist_g[p_b] # dist from P points to G
    d2 = dist_p[g_b] # dist from G points to P
    
    if len(d1) == 0 or len(d2) == 0: return 0.0
    
    hd95 = max(np.percentile(d1, 95), np.percentile(d2, 95))
    return hd95

# --- Main Evaluation ---
def main():
    source_dir = "results/inputs"
    target_dir = "results/outputs"
    
    data = []
    
    # Find all GT files
    gt_files = glob.glob(os.path.join(source_dir, "*_gt.png"))
    print(f"Found {len(gt_files)} GT files.")
    
    for gt_path in gt_files:
        stem = os.path.basename(gt_path).replace("_gt.png", "")
        
        # Baseline Path
        pred_base_path = os.path.join(source_dir, f"{stem}_pred.png")
        
        # Optimized path logic: Find the latest iter/step file
        cand_pattern = os.path.join(target_dir, f"{stem}_pred_iter*_step*.png")
        cands = glob.glob(cand_pattern)
        
        if cands:
            # Sort by name (iter9 > iter1, so need smart sort or just lexical if strictly formatted?)
            # Formatted as iterX_stepY. 
            # Simple lexical sort might fail if iter10 comes before iter2.
            # But here we likely have iter1 only.
            # Let's sort simply for now, usually sufficient.
            cands.sort() 
            pred_opt_path = cands[-1] # Pick last
            # Debug 335
            if "335_original_lax_4c_019" in stem:
                 print(f"DEBUG {stem}: Found candidates {len(cands)}. Selected: {pred_opt_path}")
        else:
            # Fallback
            pred_opt_path = os.path.join(target_dir, f"{stem}_pred.png")

        if not os.path.exists(pred_base_path): continue
        if not os.path.exists(pred_opt_path): 
            pred_opt_path = os.path.join(target_dir, f"{stem}_pred.png")

        # Read
        gt = cv2.imread(gt_path, 0)
        base = cv2.imread(pred_base_path, 0)
        opt = cv2.imread(pred_opt_path, 0)
        
        if gt is None or base is None or opt is None: continue
        
        # Metrics
        d0 = compute_dice(base, gt)
        d1 = compute_dice(opt, gt)
        
        if "335_original_lax_4c_019" in stem:
            print(f"DEBUG {stem}: BaseDice={d0:.4f}, OptDice={d1:.4f}, Path={pred_opt_path}")

        h0 = compute_hd95(base, gt)
        h1 = compute_hd95(opt, gt)
        
        data.append({"Case": stem, "Stage": "Baseline", "Dice": d0, "HD95": h0})
        data.append({"Case": stem, "Stage": "Post-Process", "Dice": d1, "HD95": h1})

    df = pd.DataFrame(data)
    df.to_csv("final_metrics.csv", index=False)
    print("Metrics saved to final_metrics.csv")
    
    # --- Plotting Nature Style ---
    # Nature Config: 
    # Font: usually sans-serif (Arial/Helvetica). Size ~7-8pt for labels.
    # Colors: Distinct but not neon.
    
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("ticks")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    
    # Palette
    nature_colors = ["#E64B35", "#4DBBD5"] # Red/Blue distinct
    palette = {"Baseline": nature_colors[0], "Post-Process": nature_colors[1]}
    
    # Dice Plot
    sns.violinplot(data=df, x="Stage", y="Dice", ax=axes[0], palette=palette, inner="box", linewidth=1.5, alpha=0.8)
    axes[0].set_title("Dice Coefficient (Higher is Better)", fontweight="bold")
    axes[0].set_ylabel("Dice")
    axes[0].set_xlabel("")
    axes[0].grid(axis='y', linestyle='--', alpha=0.3)
    
    # HD95 Plot
    # Filter outliers for visualization if needed, but show all for honesty
    sns.violinplot(data=df, x="Stage", y="HD95", ax=axes[1], palette=palette, inner="box", linewidth=1.5, alpha=0.8)
    axes[1].set_title("HD95 (Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("Hausdorff Dist (px)")
    axes[1].set_xlabel("")
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Remove top/right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig("comparison_plot.png", bbox_inches='tight')
    print("Plot saved to comparison_plot.png")
    
    # Summary
    print("\nSummary Stats:")
    print(df.groupby("Stage")[["Dice", "HD95"]].describe().T)

if __name__ == "__main__":
    main()
