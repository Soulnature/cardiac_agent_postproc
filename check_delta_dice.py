
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

def dice_score(pred, gt):
    # Overall FG Dice (1+2+3) vs (1+2+3)
    p_fg = (pred > 0)
    g_fg = (gt > 0)
    if p_fg.sum() + g_fg.sum() == 0: return 1.0
    return 2.0 * (p_fg * g_fg).sum() / (p_fg.sum() + g_fg.sum())

def compute_delta():
    run_dir = "results/outputs/run_20260205_212251_v0"
    base_dir = "results/inputs"
    log_path = os.path.join(run_dir, "label_optimization_log.csv")
    
    if not os.path.exists(log_path):
        print("Log not found")
        return

    df = pd.read_csv(log_path)
    # Get unique filenames that were actually modified
    # In log, 'filename' column stores the stem+ext e.g. "xxx.png".
    # Warning: The log might contain intermediate steps. We just need the unique filenames.
    target_files = df["filename"].unique().tolist()
    
    print(f"Found {len(target_files)} modified files in log.")
    
    deltas = []
    improved = 0
    worsened = 0
    same = 0
    
    for f in tqdm(target_files):
        stem = f.replace("_pred.png", "")
        gt_name = f"{stem}_gt.png"
        
        opt_path = os.path.join(run_dir, f)
        base_path = os.path.join(base_dir, f)
        gt_path = os.path.join(run_dir, gt_name) # Run dir has GT copy
        
        if not os.path.exists(opt_path) or not os.path.exists(base_path) or not os.path.exists(gt_path):
            continue
             
        opt = cv2.imread(opt_path, 0)
        base = cv2.imread(base_path, 0)
        gt = cv2.imread(gt_path, 0)
        
        if opt is None or base is None or gt is None: continue

        d_opt = dice_score(opt, gt)
        d_base = dice_score(base, gt)
        
        delta = d_opt - d_base
        deltas.append(delta)
        
        if delta > 0.001: improved += 1
        elif delta < -0.001: worsened += 1
        else: same += 1

    if not deltas:
        print("No valid triplets found.")
        return

    avg_delta = np.mean(deltas)
    print(f"\nAnalyzed {len(deltas)} samples:")
    print(f"Average Dice Delta: {avg_delta:+.5f}")
    print(f"Improved: {improved}")
    print(f"Worsened: {worsened}")
    print(f"Unchanged: {same}")

if __name__ == "__main__":
    compute_delta()
