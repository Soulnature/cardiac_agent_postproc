
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import numpy as np

CSV_PATH = "batch_eval_results/triage_audit_results.csv"
OUTPUT_DIR = "batch_eval_results"

def main():
    # Coalesce chunked results if main file is missing
    if not os.path.exists(CSV_PATH):
        print(f"Main output file {CSV_PATH} not found. Checking for chunks...")
        chunk_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("triage_audit_results_") and f.endswith(".csv")]
        if not chunk_files:
            print("No results found.")
            return
        
        dfs = []
        for cf in chunk_files:
            dfs.append(pd.read_csv(os.path.join(OUTPUT_DIR, cf)))
        df = pd.concat(dfs)
        df.to_csv(CSV_PATH, index=False)
        print(f"Merged {len(chunk_files)} chunks into {CSV_PATH}.")
    else:
        df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} triage records.")
    
    # --- Metrics ---
    # Define Ground Truth "Bad"
    # Let's say Bad is Dice < 0.85 (Need repair)
    BAD_THRESHOLD_DICE = 0.85
    
    # System Flags
    # TriageCategory: "needs_fix" vs "good"
    df["IsBadGT"] = df["Dice_Mean"] < BAD_THRESHOLD_DICE
    df["IsFlaggedSystem"] = df["TriageCategory"] == "needs_fix"
    
    # 1. Confusion Matrix
    tp = len(df[df["IsBadGT"] & df["IsFlaggedSystem"]]) # Correctly flagged bad
    tn = len(df[~df["IsBadGT"] & ~df["IsFlaggedSystem"]]) # Correctly called good
    fp = len(df[~df["IsBadGT"] & df["IsFlaggedSystem"]]) # Good case but flagged (Over-sensitive)
    fn = len(df[df["IsBadGT"] & ~df["IsFlaggedSystem"]]) # Bad case but missed (Dangerous)
    
    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== Triage Accuracy Analysis ===")
    print(f"Ground Truth Bad Threshold: Dice < {BAD_THRESHOLD_DICE}")
    print(f"Total Cases: {total}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Recall:    {recall:.2%} (Ability to find bad cases)")
    print(f"Precision: {precision:.2%} (Trustworthiness of 'needs_fix')")
    print(f"F1 Score:  {f1:.3f}")
    print("-" * 30)
    print(f"True Positives (Hit):  {tp}")
    print(f"False Negatives (Miss): {fn}")
    print(f"False Positives (Over): {fp}")
    print(f"True Negatives (Cool): {tn}")
    
    # 2. Group Differences (Dice & HD95)
    print("\n=== Group Performance Stats ===")
    
    fix_group = df[df["TriageCategory"] == "needs_fix"]
    good_group = df[df["TriageCategory"] == "good"]
    
    print(f"Group 'Needs Fix' (n={len(fix_group)}):")
    print(f"  Dice: {fix_group['Dice_Mean'].mean():.3f} +/- {fix_group['Dice_Mean'].std():.3f}")
    if "HD95_Mean" in df.columns:
        print(f"  HD95: {fix_group['HD95_Mean'].mean():.2f} +/- {fix_group['HD95_Mean'].std():.2f}")
        
    print(f"Group 'Good' (n={len(good_group)}):")
    print(f"  Dice: {good_group['Dice_Mean'].mean():.3f} +/- {good_group['Dice_Mean'].std():.3f}")
    if "HD95_Mean" in df.columns:
        print(f"  HD95: {good_group['HD95_Mean'].mean():.2f} +/- {good_group['HD95_Mean'].std():.2f}")

    # 3. Visualization
    # Boxplot of Dice by Triage Category
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="TriageCategory", y="Dice_Mean", data=df, order=["good", "needs_fix"])
    plt.axhline(BAD_THRESHOLD_DICE, color='r', linestyle='--', label="Bad Threshold")
    plt.title("Dice Score Distribution by Triage Decision")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "triage_dice_boxplot.png"))
    plt.close()
    
    if "HD95_Mean" in df.columns:
        plt.figure(figsize=(8, 6))
        # Filter outliers for visualization?
        sns.boxplot(x="TriageCategory", y="HD95_Mean", data=df, order=["good", "needs_fix"], showfliers=False)
        plt.title("HD95 Distribution by Triage Decision (Outliers Hidden)")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "triage_hd95_boxplot.png"))
        plt.close()

    # Save summary report
    with open(os.path.join(OUTPUT_DIR, "triage_analysis_report.txt"), "w") as f:
        f.write("Triage Accuracy Audit Report\n")
        f.write("============================\n\n")
        f.write(f"Total Cases Checked: {total}\n")
        f.write(f"Recall (Sensitivity): {recall:.2%}\n")
        f.write(f"Precision: {precision:.2%}\n")
        f.write(f"False Negatives (Risk): {fn}\n\n")
        
        f.write("Performance Split:\n")
        f.write(f"  'Needs Fix' Group: Mean Dice {fix_group['Dice_Mean'].mean():.3f}\n")
        f.write(f"  'Good' Group:      Mean Dice {good_group['Dice_Mean'].mean():.3f}\n\n")
        
        if fn > 0:
            f.write("Missed Cases (Dice < 0.85 but called Good):\n")
            missed = df[df["IsBadGT"] & ~df["IsFlaggedSystem"]].sort_values("Dice_Mean")
            for _, row in missed.iterrows():
                f.write(f"  {row['Stem']}: Dice {row['Dice_Mean']:.3f}\n")

if __name__ == "__main__":
    main()
