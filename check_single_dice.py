
import numpy as np
import cv2

def compute_dice(pred, gt):
    # Normalize 0-255 -> 0-3
    if np.max(pred) > 3:
        pred = np.round(pred / 85).astype(np.uint8)
    if np.max(gt) > 3:
        gt = np.round(gt / 85).astype(np.uint8)

    dice_scores = []
    # Classes 1, 2, 3
    for cls in [1, 2, 3]:
        p = (pred == cls).astype(np.float32)
        g = (gt == cls).astype(np.float32)
        intersection = np.sum(p * g)
        union = np.sum(p) + np.sum(g)
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append(2.0 * intersection / union)
    return np.mean(dice_scores), dice_scores

# Paths
gt_path = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export/335_original_lax_4c_009_gt.png"
base_path = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export/335_original_lax_4c_009_pred.png"
opt_path = "/data484_5/xzhao14/cardiac_agent_postproc/results/outputs/run_20260207_114501_v0/335_original_lax_4c_009_pred.png"

# Read
gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
base = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
opt = cv2.imread(opt_path, cv2.IMREAD_GRAYSCALE)

if gt is None or base is None or opt is None:
    print("Error reading files")
    exit(1)

d_base, d_base_cls = compute_dice(base, gt)
d_opt, d_opt_cls = compute_dice(opt, gt)

print(f"CASE 335 SLICE 009 EVALUATION")
print(f"Baseline Mean Dice: {d_base:.4f} (Classes: {d_base_cls})")
print(f"Optimized Mean Dice: {d_opt:.4f} (Classes: {d_opt_cls})")
print(f"Improvement: {d_opt - d_base:.4f}")
