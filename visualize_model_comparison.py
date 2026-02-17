import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODELS = ["Ministral-3b", "Gemini-2.5-Flash", "GPT-4o"]
# Metrics to track per model
results = {
    "Ministral-3b": {"case_244": {"dice_delta": 0.0077, "bbox": False, "verdict": "degraded"}}, # From memory/logs
    "Gemini-2.5-Flash": {"case_244": {"dice_delta": 0.0050, "bbox": False, "verdict": "bad"}},    # From recent Gemini run
    "GPT-4o": {"case_244": {"dice_delta": 0.0108, "bbox": False, "verdict": "bad"}}          # From recent GPT-4o run
}

# Add Case 201/202 if available in logs (they were skipped by FastPath, so data is null)
# We will focus on Case 244 as it's the only one that underwent full repair.

data = []
for model, cases in results.items():
    res = cases["case_244"]
    data.append({
        "Model": model, 
        "Dice Delta": res["dice_delta"],
        "BBox Generated": res["bbox"],
        "Final Verdict": res["verdict"]
    })

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Model", y="Dice Delta", palette="viridis")
plt.title("Dice Improvement by Model (Case 244)")
plt.ylabel("Dice Score Increase")
plt.axhline(0, color='black', linewidth=0.8)

# Add annotations
for index, row in df.iterrows():
    plt.text(index, row["Dice Delta"] + 0.0005, f"{row['Dice Delta']:.4f}", color='black', ha="center")
    plt.text(index, row["Dice Delta"] / 2, f"BBox: {row['BBox Generated']}\nVerdict: {row['Final Verdict']}", 
             color='white', ha="center", fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison_case_244.png")
print("Comparison chart saved to model_comparison_case_244.png")
