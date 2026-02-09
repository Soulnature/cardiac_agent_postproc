
import os
import glob
import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

def compute_multiclass_dice(pred, gt):
    dices = []
    # Normalize to 0-3
    if pred.max() > 3: pred = (pred / 85).astype(np.uint8)
    if gt.max() > 3: gt = (gt / 85).astype(np.uint8)
        
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
    return np.mean(dices)

def main():
    log_path = "results/outputs/label_optimization_log.csv"
    if not os.path.exists(log_path):
        print("No log found.")
        return

    # Fix shifted columns by disabling automatic index detection
    df = pd.read_csv(log_path, index_col=False)
    print(f"Loaded {len(df)} log entries.")
    
    # RQS Feature Columns (Exclude metadata and targets)
    feature_cols = [c for c in df.columns if c.startswith('P_') or c.startswith('R_') or c in ['mean_conf', 'atlas_match_score']]
    print(f"Features: {feature_cols}")
    
    # Add True Dice column
    true_dices = []
    valid_indices = []
    
    # Pre-cache GT paths
    gt_map = {}
    for f in glob.glob("results/inputs/*_gt.png"):
        stem = os.path.basename(f).replace("_gt.png", "")
        gt_map[stem] = f
        
    print(f"GT Map Size: {len(gt_map)}")
    
    print("Computing True Dice for Training Data...")
    for i, (idx, row) in enumerate(df.iterrows()):
        # Robust check for header repeats or malformed lines
        if str(row['filename']) == 'filename' or str(row['view_type']) not in ['2ch', '4ch']:
             continue

        stem = str(row['filename']).replace("_pred.png", "")
        # FIX: Log points to 'inputs', but files are in 'outputs'
        pred_path = str(row['out_path']).strip().replace("results/inputs", "results/outputs")
        
        # Debug first 10
        if i < 10:
             print(f"DEBUG {i}: Stem={stem} | InGT={stem in gt_map} | Path={pred_path} | Exists={os.path.exists(pred_path)}")

        # Check GT
        if stem not in gt_map:
            # Only print first few errors
            if i < 100: print(f"DEBUG: No GT for '{stem}'")
            continue
        # Check Pred
        try:
            if not os.path.exists(pred_path):
                if i < 3000 and i % 500 == 0:
                     print(f"DEBUG: Missing pred {pred_path}")
                continue
        except Exception as e:
            print(f"Path Error: {e}")
            continue
            
        gt = cv2.imread(gt_map[stem], 0)
        pred = cv2.imread(pred_path, 0)
        
        if gt is None or pred is None: 
            continue
            
        d = compute_multiclass_dice(pred, gt)
        true_dices.append(d)
        valid_indices.append(idx)
        
        if idx % 100 == 0:
            print(f"Processed {idx}...")

    # Robustness check
    if len(valid_indices) < 50:
        print("Not enough data points.")
        return

    df_train = df.iloc[valid_indices].copy()
    df_train['True_Dice'] = true_dices
    
    X = df_train[feature_cols].fillna(0)
    y = df_train['True_Dice']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Linear Regression (Interpretability)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("\n=== Linear Regression ===")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print("Coefficients:")
    coeffs = pd.Series(lr.coef_, index=feature_cols).sort_values(ascending=False)
    print(coeffs)
    
    # Model 2: Random Forest (Non-linear)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("\n=== Random Forest ===")
    print(f"R2 Score: {r2_score(y_test, y_pred_rf):.4f}")
    print("Feature Importance:")
    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(fi.head(10))
    
    # Save best model
    joblib.dump(rf, "surrogate_model.pkl")
    print("Saved surrogate_model.pkl")
    
    # Save correlation for plotting
    df_train[['True_Dice', 'RQS_total'] + feature_cols].corr()['True_Dice'].sort_values().to_csv("feature_correlation.csv")
    print("Saved feature_correlation.csv")

if __name__ == "__main__":
    main()
