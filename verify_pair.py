
import cv2
import numpy as np
import os

def compute_multiclass_dice(pred, gt):
    dices = []
    
    # Normalize GT to 0-3 if needed
    if gt.max() > 3:
        gt = (gt / 85).astype(np.uint8)
        
    # Normalize Pred to 0-3 if needed
    if pred.max() > 3:
        pred = (pred / 85).astype(np.uint8)
        
    print(f"Normalized Unique - Pred: {np.unique(pred)}, GT: {np.unique(gt)}")
        
    for cls in [1, 2, 3]:
        p_c = (pred == cls)
        g_c = (gt == cls)
        
        inter = (p_c & g_c).sum()
        union = p_c.sum() + g_c.sum()
        dice = 2.0 * inter / union if union > 0 else 0.0
        dices.append(dice)
        print(f"  Class {cls}: Dice={dice:.4f} (Inter={inter}, Union={union})")
        
    return np.mean(dices)

def verify():
    base_dir = "results/inputs"
    stem = "201_original_lax_4c_000"
    
    pred_path = os.path.join(base_dir, f"{stem}_pred.png")
    gt_path = os.path.join(base_dir, f"{stem}_gt.png")
    
    print(f"Checking pair:\n  Pred: {pred_path}\n  GT:   {gt_path}\n")
    
    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        print("Files not found!")
        return
        
    # Read UNCHANGED to see raw values
    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    
    # Handle RGBA/RGB GT
    if len(gt.shape) == 3:
        print(f"GT has {gt.shape[2]} channels. Converting to grayscale.")
        gt = gt[:,:,0]
        
    # Handle RGBA/RGB Pred
    if len(pred.shape) == 3:
        print(f"Pred has {pred.shape[2]} channels. Converting to grayscale.")
        pred = pred[:,:,0]
    
    dice = compute_multiclass_dice(pred, gt)
    print(f"\nMean Multiclass Dice: {dice:.4f}")

if __name__ == "__main__":
    verify()
