
import os
import glob
import numpy as np
import cv2
import pandas as pd

def compute_multiclass_dice(pred, gt):
    dices = []
    
    # Normalize to 0-3
    if pred.max() > 3:
        pred = (pred / 85).astype(np.uint8)
    if gt.max() > 3:
        gt = (gt / 85).astype(np.uint8)
        
    for cls in [1, 2, 3]:
        p_c = (pred == cls)
        g_c = (gt == cls)
        
        if p_c.sum() == 0 and g_c.sum() == 0:
            dices.append(1.0)
            continue
            
        inter = (p_c & g_c).sum()
        union = p_c.sum() + g_c.sum()
        dice = 2.0 * inter / union if union > 0 else 0.0
        dices.append(dice)
        
    return np.mean(dices), dices

def main():
    source_dir = "results/inputs"
    target_dir = "results/outputs"
    
    # Load Optimization Log to find the Best Output File
    log_path = "results/outputs/label_optimization_log.csv"
    opt_map = {}
    if os.path.exists(log_path):
        import csv
        with open(log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stem = row["filename"].replace("_pred.png", "")
                opt_map[stem] = row["out_path"].strip()

    gt_files = glob.glob(os.path.join(source_dir, "*_gt.png"))
    print(f"Evaluating {len(gt_files)} cases with Multiclass Dice...")
    
    results = []
    
    for gt_path in gt_files:
        stem = os.path.basename(gt_path).replace("_gt.png", "")
        
        pred_base_path = os.path.join(source_dir, f"{stem}_pred.png")
        
        # Determine Optimized Path
        # Priority: Glob for latest candidate in output dir
        cand_pattern = os.path.join(target_dir, f"{stem}_pred_iter*_step*.png")
        cands = glob.glob(cand_pattern)
        
        pred_opt_path = os.path.join(target_dir, f"{stem}_pred.png") # Default
        
        if cands:
            cands.sort()
            pred_opt_path = cands[-1]
            # Debug 335
            if "335_original_lax_4c_019" in stem:
                 print(f"DEBUG {stem}: Found {len(cands)} candidates. using {pred_opt_path}")
        
        # Read
        gt = cv2.imread(gt_path, 0)
        base = cv2.imread(pred_base_path, 0)
        opt = cv2.imread(pred_opt_path, 0)
        
        if gt is None or base is None or opt is None:
            continue
            
        md0, ds0 = compute_multiclass_dice(base, gt)
        md1, ds1 = compute_multiclass_dice(opt, gt)
        
        results.append({
            "Case": stem,
            "Base_Mean": md0,
            "Opt_Mean": md1,
            "Diff": md1 - md0,
            "Opt_Path": pred_opt_path
        })

    df = pd.DataFrame(results)
    
    print("\n=== Multiclass Evaluation Summary ===")
    print(df[["Base_Mean", "Opt_Mean"]].describe())
    
    print("\nTop Improvements:")
    print(df.sort_values("Diff", ascending=False).head(5)[["Case", "Base_Mean", "Opt_Mean", "Diff"]])
    
    print("\nTop Regressions:")
    print(df.sort_values("Diff", ascending=True).head(5)[["Case", "Base_Mean", "Opt_Mean", "Diff"]])

    # Save
    df.to_csv("analysis_multiclass.csv", index=False)

if __name__ == "__main__":
    main()
