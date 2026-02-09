
import numpy as np
import cv2
import glob
import os

def compute_dice(pred, gt):
    if np.max(pred) > 3:
        pred = np.round(pred / 85).astype(np.uint8)
    if np.max(gt) > 3:
        gt = np.round(gt / 85).astype(np.uint8)

    dice_scores = []
    for cls in [1, 2, 3]:
        p = (pred == cls).astype(np.float32)
        g = (gt == cls).astype(np.float32)
        union = np.sum(p) + np.sum(g)
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append(2.0 * np.sum(p * g) / union)
    return np.mean(dice_scores)

base_dir = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export"
out_dir = "/data484_5/xzhao14/cardiac_agent_postproc/results/outputs/run_20260206_113302_v0"

cases = ["210_original_lax_4c_011", "244_original_lax_4c_009"]

for c in cases:
    gt_path = os.path.join(base_dir, f"{c}_gt.png")
    pred0_path = os.path.join(base_dir, f"{c}_pred.png")
    
    # Check for optimized iter files
    iters = glob.glob(os.path.join(base_dir, f"{c}_pred_iter*.png"))
    if not iters:
        print(f"[{c}] No optimization files found.")
        continue
    
    iters.sort()
    last_iter = iters[-1]
    
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred0 = cv2.imread(pred0_path, cv2.IMREAD_GRAYSCALE)
    opt = cv2.imread(last_iter, cv2.IMREAD_GRAYSCALE)
    
    if gt is None: continue
    
    d0 = compute_dice(pred0, gt)
    d1 = compute_dice(opt, gt)
    
    print(f"[{c}] Baseline: {d0:.4f} -> Optimized: {d1:.4f} (Delta: {d1-d0:.4f})")
