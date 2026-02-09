
import os
import glob
import re
import cv2
import pandas as pd
import numpy as np
from cardiac_agent_postproc.rqs import compute_rqs
from cardiac_agent_postproc.vlm_guardrail import VisionGuardrail
# from cardiac_agent_postproc.api_client import KimiClient

import yaml

# Initialize components
with open("config/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    
# Ensure Guardrail is enabled for this test
if "vision_guardrail" not in cfg:
    cfg["vision_guardrail"] = {}
cfg["vision_guardrail"]["enabled"] = True
cfg["vision_guardrail"]["model"] = "moonshotai/kimi-k2.5"

guardrail = VisionGuardrail(cfg)

WORST_CASE_DIR = "/data484_2/mferrari2/results_segmentation/MnM2/worst_cases"
INPUTS_DIR = "results/inputs"

def parse_worst_cases():
    files = glob.glob(os.path.join(WORST_CASE_DIR, "*.png"))
    data = []
    for fpath in files:
        fname = os.path.basename(fpath)
        match = re.search(r"(\d{3}_original_lax_\w+_\d{3})", fname)
        if not match:
            print(f"Skipping {fname} (No ID match)")
            continue
        stem = match.group(1)
        parts = fname.split(stem)
        prefix = parts[0]
        meta_parts = prefix.split('_')
        dice_str = meta_parts[0]
        try:
            dice = float(dice_str)
        except:
            dice = -1.0
        error_type = meta_parts[2] if len(meta_parts) > 2 else "Unknown"
        data.append({
            "filename": fname,
            "stem": stem,
            "dice": dice,
            "error_type": error_type
        })
    return data

def analyze():
    cases = parse_worst_cases()
    print(f"Found {len(cases)} worst cases.")
    
    results = []
    
    # Process only top 5 for quick calibration
    cases = cases[:5] 
    for case in cases:
        stem = case["stem"]
        print(f"\nScanning {stem} (Dice={case['dice']}, Type={case['error_type']})...")
        
        # Locate Mask and Image
        mask_path = os.path.join(INPUTS_DIR, f"{stem}_pred.png")
        img_path = os.path.join(INPUTS_DIR, f"{stem}_img.png")
        
        if not os.path.exists(mask_path):
            print(f"  Missing mask in {INPUTS_DIR}")
            continue
            
        mask = cv2.imread(mask_path, 0)
        img = cv2.imread(img_path, 0)
        
        # Normalize Mask if needed (0,85,170,255 -> 0,1,2,3)
        if mask.max() > 3:
            mask = mask // 85
        
        # 1. Compute RQS
        view = "4ch" if "4c" in stem else "2ch"
        # compute_rqs(mask, img, view_type, pred0, atlas, siblings, cfg)
        rqs_res = compute_rqs(mask, img, view, mask, None, [], cfg)
        rqs = rqs_res.total
        print(f"  RQS: {rqs:.2f}")
        
        # 2. Compute Vision Score
        # guardrail.check(img_path, mask_arr, stem)
        verdict = guardrail.check(img_path, mask, stem)
        print(f"  Verdict: {verdict}")
        
        results.append({
            "id": stem,
            "dice": case["dice"],
            "error_type": case["error_type"],
            "rqs": rqs,
            "vision_score": verdict["score"],
            "accepted": verdict["accepted"],
            "reason": verdict["reason"]
        })

    # Save summary
    df = pd.DataFrame(results)
    df.to_csv("worst_case_analysis.csv", index=False)
    print("\nAnalysis Complete. Saved to worst_case_analysis.csv")
    
    # Print simple stats
    if not df.empty:
        groundly_bad = df[df["vision_score"] < 50]
        print(f"Detection Rate (Vision < 50): {len(groundly_bad)} / {len(df)}")
        
        rqs_bad = df[df["rqs"] < 50]
        print(f"Detection Rate (RQS < 50): {len(rqs_bad)} / {len(df)}")

if __name__ == "__main__":
    analyze()
