
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Settings
DATA_DIR = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export"
OUTPUT_DIR = "batch_eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def compute_hd95(pred, gt):
    """
    Computes Hausdorff Distance 95% (HD95) for binary masks.
    Returns float (pixels).
    """
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.nan # Undefined/Infinite

    # Extract contours to get boundary points
    contours_p, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_g, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_p or not contours_g:
         return 0.0 # Should be caught by sum check, but safe guard

    # Stack all contour points
    if len(contours_p) == 0: return np.nan
    if len(contours_g) == 0: return np.nan

    pts_p = np.vstack([c.reshape(-1, 2) for c in contours_p])
    pts_g = np.vstack([c.reshape(-1, 2) for c in contours_g])
    
    # Use cdist
    d = cdist(pts_p, pts_g)
    
    min_d_p2g = d.min(axis=1) 
    min_d_g2p = d.min(axis=0) 
    
    hd95_p2g = np.percentile(min_d_p2g, 95)
    hd95_g2p = np.percentile(min_d_g2p, 95)
    
    return max(hd95_p2g, hd95_g2p)

def compute_metrics(pred, gt):
    """
    Computes Dice AND HD95 for RV (1), Myo (2), LV (3).
    Returns dictionary with per-class and mean metrics.
    """
    metrics = {}
    classes = {"RV": 1, "Myo": 2, "LV": 3}
    
    dice_sum = 0
    hd95_sum = 0
    hd95_count = 0
    
    for name, label in classes.items():
        # Binary masks
        p = (pred == label).astype(np.float32)
        g = (gt == label).astype(np.float32)
        
        # Dice
        d = dice_coef(g, p)
        metrics[f"Dice_{name}"] = d
        dice_sum += d
        
        # HD95
        # Pass binary mask (0/1)
        h = compute_hd95(p, g)
        metrics[f"HD95_{name}"] = h
        if not np.isnan(h):
            hd95_sum += h
            hd95_count += 1
        
    metrics["Dice_Mean"] = dice_sum / 3.0
    metrics["HD95_Mean"] = hd95_sum / max(1, hd95_count)
    return metrics

def visualize_case(ax_img, ax_gt, ax_pred, ax_overlay, img, gt, pred, title):
    """
    Draws Image, GT, Pred, and Error Overlay on provided axes.
    """
    # 1. Image
    ax_img.imshow(img, cmap="gray")
    ax_img.set_title(f"{title}\nImage", fontsize=10)
    ax_img.axis("off")
    
    # 2. GT
    ax_gt.imshow(gt, cmap="viridis", vmin=0, vmax=3)
    ax_gt.set_title("Ground Truth", fontsize=10)
    ax_gt.axis("off")
    
    # 3. Pred
    ax_pred.imshow(pred, cmap="viridis", vmin=0, vmax=3)
    ax_pred.set_title("Prediction", fontsize=10)
    ax_pred.axis("off")
    
    # 4. Overlay Error Map
    # Green = TP (Correct), Red = FP (Extra), Blue = FN (Missed)
    h, w = gt.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # TP (Correct Class): Green
    tp_mask = (gt == pred) & (gt > 0)
    
    # FP (Pred > 0 but GT == 0): Red
    fp_mask = (pred > 0) & (gt == 0)
    
    # FN (GT > 0 but Pred == 0): Blue
    fn_mask = (gt > 0) & (pred == 0)
    
    # Class Mismatch (GT > 0, Pred > 0, but different): Yellow
    mismatch_mask = (gt > 0) & (pred > 0) & (gt != pred)
    
    # Apply colors
    overlay = img_rgb.copy()
    overlay[tp_mask] = [0, 255, 0]    # Green
    overlay[fp_mask] = [255, 0, 0]    # Red
    overlay[fn_mask] = [0, 0, 255]    # Blue
    overlay[mismatch_mask] = [255, 255, 0] # Yellow
    
    # Blend
    cv2.addWeighted(overlay, 0.4, img_rgb, 0.6, 0, img_rgb)
    
    ax_overlay.imshow(img_rgb)
    ax_overlay.set_title("Error Map\n(G=TP, R=FP, B=FN, Y=Mis)", fontsize=8)
    ax_overlay.axis("off")


def main():
    print(f"Scanning {DATA_DIR}...")
    files = os.listdir(DATA_DIR)
    
    # Find pred files
    pred_files = [f for f in files if f.endswith("_pred.png")]
    print(f"Found {len(pred_files)} predictions.")
    
    results = []
    
    for f in tqdm(pred_files):
        stem = f.replace("_pred.png", "")
        gt_name = f"{stem}_gt.png"
        img_name = f"{stem}_img.png"
        
        pred_path = os.path.join(DATA_DIR, f)
        gt_path = os.path.join(DATA_DIR, gt_name)
        img_path = os.path.join(DATA_DIR, img_name)
        
        if not os.path.exists(gt_path):
            continue
            
        # Load
        pred_raw = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        gt_raw = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        
        # Handle multi-channel masks (if saved as RGB)
        if gt_raw.ndim == 3: gt_raw = gt_raw[:,:,0]
        if pred_raw.ndim == 3: pred_raw = pred_raw[:,:,0]
        
        # Heuristic check for scaling
        if pred_raw.max() > 3:
            pred_raw = np.round(pred_raw / 85).astype(np.uint8)
        if gt_raw.max() > 3:
            gt_raw = np.round(gt_raw / 85).astype(np.uint8)
            
        # Compute Metrics
        m = compute_metrics(pred_raw, gt_raw)
        
        # Extract Group (First 3 digits of stem)
        group_id = stem.split("_")[0] # e.g. "201" from "201_original..."
        
        row = {
            "Stem": stem,
            "Group": group_id,
            "Image Path": img_path,
            "GT Path": gt_path,
            "Pred Path": pred_path
        }
        # Flatten metrics
        for k, v in m.items():
            row[k] = v
            
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # 1. Save Metrics
    csv_path = os.path.join(OUTPUT_DIR, "batch_evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # 2. Overall Stats
    print("\n=== Overall Statistics ===")
    print(df[["Dice_Mean", "HD95_Mean"]].describe())
    
    # 3. Visualization of Worst Cases (Lowest Mean Dice)
    df_sorted = df.sort_values("Dice_Mean", ascending=True)
    top_n = 10
    worst_cases = df_sorted.head(top_n)
    
    print(f"\nGenerating visualization for Top {top_n} Worst Cases...")
    
    fig, axes = plt.subplots(top_n, 4, figsize=(16, 4*top_n))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for i, (_, row) in enumerate(worst_cases.iterrows()):
        img = cv2.imread(row["Image Path"], cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(row["GT Path"], cv2.IMREAD_UNCHANGED)
        pred = cv2.imread(row["Pred Path"], cv2.IMREAD_UNCHANGED)
        
        # Handle multi-channel masks (if saved as RGB)
        if gt.ndim == 3: gt = gt[:,:,0]
        if pred.ndim == 3: pred = pred[:,:,0]
        
        # Renormalize if needed
        if gt.max() > 3: gt = np.round(gt / 85).astype(np.uint8)
        if pred.max() > 3: pred = np.round(pred / 85).astype(np.uint8)
            
        ax_row = axes[i]
        title = f"{row['Stem']}\nDice: {row['Dice_Mean']:.3f} | HD {row['HD95_Mean']:.1f}"
        visualize_case(ax_row[0], ax_row[1], ax_row[2], ax_row[3], img, gt, pred, title)
        
    viz_path = os.path.join(OUTPUT_DIR, "batch_evaluation_worst_cases.png")
    plt.savefig(viz_path, bbox_inches="tight")
    plt.close()
    print(f"Worst cases visualization saved to {viz_path}")
    
    # 4. Group Comparison (Dice)
    print("\nGenerating Group Comparison (Dice)...")
    plt.figure(figsize=(15, 6))
    group_order = df.groupby("Group")["Dice_Mean"].median().sort_values().index
    sns.boxplot(x="Group", y="Dice_Mean", data=df, order=group_order)
    plt.title("Dice Score Distribution by Group (Patient ID)")
    plt.xticks(rotation=45)
    plt.ylabel("Mean Dice")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, "batch_evaluation_groups_dice.png"), bbox_inches="tight")
    plt.close()
    
    # 5. Group Comparison (HD95)
    print("\nGenerating Group Comparison (HD95)...")
    plt.figure(figsize=(15, 6))
    group_order_hd = df.groupby("Group")["HD95_Mean"].median().sort_values().index
    sns.boxplot(x="Group", y="HD95_Mean", data=df, order=group_order_hd)
    plt.title("HD95 Distribution by Group (Lower is Better)")
    plt.xticks(rotation=45)
    plt.ylabel("Mean HD95 (px)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(OUTPUT_DIR, "batch_evaluation_groups_hd95.png"), bbox_inches="tight")
    plt.close()
    
    # 6. Summary Text
    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
        f.write("Batch Evaluation Summary\n")
        f.write("========================\n\n")
        f.write(f"Total Cases: {len(df)}\n")
        f.write(f"Average Mean Dice: {df['Dice_Mean'].mean():.4f} +/- {df['Dice_Mean'].std():.4f}\n")
        f.write(f"Average Mean HD95: {df['HD95_Mean'].mean():.4f} +/- {df['HD95_Mean'].std():.4f}\n")
        f.write(f"Average RV Dice: {df['Dice_RV'].mean():.4f}\n")
        f.write(f"Average Myo Dice: {df['Dice_Myo'].mean():.4f}\n")
        f.write(f"Average LV Dice: {df['Dice_LV'].mean():.4f}\n\n")
        f.write("Worst 5 Cases (by Dice):\n")
        for i, row in df_sorted.head(5).iterrows():
            f.write(f"  {row['Stem']}: Dice {row['Dice_Mean']:.4f}, HD95 {row['HD95_Mean']:.2f}\n")

if __name__ == "__main__":
    main()
