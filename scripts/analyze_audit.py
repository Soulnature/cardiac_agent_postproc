
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

CSV_PATH = "batch_eval_results/diagnosis_audit_results.csv"
OUTPUT_DIR = "batch_eval_results"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} audit records.")
    
    # --- Metrics ---
    # Define Ground Truth "Bad"
    # User's request implies we want to know if system correctly identifies "bad" cases.
    # Let's say Bad is Dice < 0.85 (Need repair)
    BAD_THRESHOLD = 0.85
    
    # Define System "Flagged"
    # System flags if MaxSeverity is 'high' or 'critical'
    # Severity Rank: critical=4, high=3, medium=2, low=1, none=0
    severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0, "unknown": 0}
    df["SeverityRank"] = df["MaxSeverity"].apply(lambda x: severity_map.get(str(x).lower(), 0))
    
    df["IsBadGT"] = df["Dice_Mean"] < BAD_THRESHOLD
    df["IsFlaggedSystem"] = df["SeverityRank"] >= 3 # High or Critical
    
    # Confusion Matrix
    tp = len(df[df["IsBadGT"] & df["IsFlaggedSystem"]])
    tn = len(df[~df["IsBadGT"] & ~df["IsFlaggedSystem"]])
    fp = len(df[~df["IsBadGT"] & df["IsFlaggedSystem"]]) # Good case but flagged as High severity (False Positive / Over-sensitive)
    fn = len(df[df["IsBadGT"] & ~df["IsFlaggedSystem"]]) # Bad case but NOT flagged (False Negative / Missed)
    
    accuracy = (tp + tn) / len(df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\n=== Diagnosis Accuracy Analysis ===")
    print(f"Ground Truth Bad Threshold: Dice < {BAD_THRESHOLD}")
    print(f"System Flag Threshold: Severity >= High")
    print(f"Total Cases: {len(df)}")
    print(f"True Positives (Correctly Flagged Bad): {tp}")
    print(f"True Negatives (Correctly Ignored Good): {tn}")
    print(f"False Positives (Good but Flagged): {fp}")
    print(f"False Negatives (Bad but Ignored): {fn}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Recall (Sensitivity): {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    
    # --- Plots ---
    
    # 1. Swarmplot: Dice vs Severity
    plt.figure(figsize=(10, 6))
    order = ["critical", "high", "medium", "low", "none"]
    # Filter order to existing
    exist_order = [o for o in order if o in df["MaxSeverity"].unique()]
    
    sns.swarmplot(x="MaxSeverity", y="Dice_Mean", data=df, order=exist_order, hue="GroupLabel", size=8)
    plt.axhline(BAD_THRESHOLD, color='r', linestyle='--', label=f"Bad Threshold ({BAD_THRESHOLD})")
    plt.title("Dice Score vs System Diagnosis Severity")
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "audit_dice_vs_severity.png"))
    plt.close()
    
    # 2. HD95 vs Severity
    if "HD95_Mean" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.swarmplot(x="MaxSeverity", y="HD95_Mean", data=df, order=exist_order, hue="GroupLabel", size=8)
        plt.title("HD95 vs System Diagnosis Severity")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "audit_hd95_vs_severity.png"))
        plt.close()

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "audit_analysis_report.txt"), "w") as f:
        f.write("Diagnosis Audit Analysis\n")
        f.write("========================\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Recall: {recall:.2%}\n")
        f.write(f"False Negatives: {fn}\n")
        
        if fn > 0:
            f.write("\nMissed Cases (False Negatives):\n")
            missed = df[df["IsBadGT"] & ~df["IsFlaggedSystem"]]
            for _, row in missed.iterrows():
                f.write(f"  {row['Stem']} (Dice: {row['Dice_Mean']:.3f}, Sev: {row['MaxSeverity']})\n")

if __name__ == "__main__":
    main()
