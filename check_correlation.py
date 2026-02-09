
import pandas as pd
import numpy as np
import cv2
import os

def dice_score(pred, gt):
    if pred.sum() + gt.sum() == 0: return 1.0
    return 2.0 * (pred * gt).sum() / (pred.sum() + gt.sum())

def calc_correlation():
    feedback_path = "results/outputs/run_20260205_204949_v0/vision_feedback.csv"
    if not os.path.exists(feedback_path):
        print("No feedback csv found")
        return

    df = pd.read_csv(feedback_path)
    dices = []
    scores = []
    
    print("\nCorrect Correlation Check (Vision Score vs Dice):")
    
    found = 0
    for idx, row in df.iterrows():
        stem = row["stem"]
        score = row["score"]
        
        # Paths
        base = "results/outputs/run_20260205_204949_v0"
        pred_path = os.path.join(base, f"{stem}_pred.png")
        gt_path = os.path.join(base, f"{stem}_gt.png")
        
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            p = cv2.imread(pred_path, 0)
            g = cv2.imread(gt_path, 0)
            
            # Map back to 0-3 (p is 0,85,170,255)
            # Or just threshold > 0 for overall FG Dice
            # Let's do Macro Dice across classes 1,2,3
            p_cls = np.zeros_like(p)
            p_cls[p == 85] = 1
            p_cls[p == 170] = 2
            p_cls[p == 255] = 3
            
            # GT is likely already 0-3 or 0-255?
            # Check unique values of GT
            uni = np.unique(g)
            if 255 in uni:
                 # assume scaled
                 g_cls = np.zeros_like(g)
                 g_cls[g == 85] = 1
                 g_cls[g == 170] = 2
                 g_cls[g == 255] = 3
            else:
                 g_cls = g
            
            # Calc Dice per class
            ds = []
            for c in [1, 2, 3]:
                d = dice_score(p_cls==c, g_cls==c)
                ds.append(d)
            macro_dice = np.mean(ds)
            
            dices.append(macro_dice)
            scores.append(score)
            found += 1
            if found < 5:
                print(f"  [{stem}] Score={score} -> Dice={macro_dice:.3f}")

    if len(dices) > 1:
        corr = np.corrcoef(scores, dices)[0, 1]
        print(f"\nAnalyzed {len(dices)} samples.")
        print(f"Correlation (Pearson): {corr:.4f}")
        
        # Buckets
        high_score_dices = [d for s,d in zip(scores, dices) if s >= 80]
        low_score_dices = [d for s,d in zip(scores, dices) if s < 50]
        
        if high_score_dices:
            print(f"Avg Dice for High Score (>=80): {np.mean(high_score_dices):.3f}")
        if low_score_dices:
            print(f"Avg Dice for Low Score (<50): {np.mean(low_score_dices):.3f}")

if __name__ == "__main__":
    calc_correlation()
