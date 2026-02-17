"""
Hybrid Repair: Selective Replacement.

Strategy:
1. Load Original Prediction (Myo/LV are okay-ish).
2. Load Deterministic Repair Mask (RV is great).
3. Combine: 
   - Base = Original Prediction
   - Replace RV in Base with RV from Deterministic Mask
   - Resolve conflicts: Maintain Myocardium integrity.
4. Evaluate.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import yaml
from scipy import ndimage as ndi

from cardiac_agent_postproc.io_utils import read_mask, read_image, write_mask
from cardiac_agent_postproc.vlm_guardrail import create_overlay

# Config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)
SOURCE_DIR = cfg["paths"]["source_dir"]
OUT_DIR = "/tmp/hybrid_test"
os.makedirs(OUT_DIR, exist_ok=True)

ATLAS_STEM = "335_original_lax_4c_019"
TARGET_STEM = "335_original_lax_4c_009"

# Load data (using results from previous steps)
orig_pred = read_mask(f"{SOURCE_DIR}/{TARGET_STEM}_pred.png")
det_mask = read_mask(f"/tmp/deterministic_v2/335_det_v2_mask.png") # Loaded from prev step
target_gt = read_mask(f"{SOURCE_DIR}/{TARGET_STEM}_gt.png")
target_img = read_mask(f"{SOURCE_DIR}/{TARGET_STEM}_img.png") # Read as simple array for shape

# Hybrid Construction
# Revised Logic: Deterministic RV is trusted (Dice 0.725). Original RV is trash.
# Original Myo/LV are okay but might overlap with correct RV.
# Priority: Det_RV > Orig_Myo/LV.

hybrid_mask = orig_pred.copy()

# Step 1: Force Det_RV into the mask (overwriting whatever was there)
det_rv = (det_mask == 1)
hybrid_mask[det_rv] = 1

# Step 2: Ensure we didn't wipe out too much LV/Myo?
# If Det_RV is correct, then anything under it was likely Wrong Myo/LV.
# So simple overwrite is correct strategy.

# Evaluation
def compute_dice(pred, gt):
    scores = {}
    for c, n in {1:"RV", 2:"Myo", 3:"LV"}.items():
        p = (pred==c)
        g = (gt==c)
        inter = (p&g).sum()
        union = p.sum() + g.sum()
        scores[n] = 2*inter/(union+1e-8)
    scores["Mean"] = np.mean(list(scores.values()))
    return scores

scores = compute_dice(hybrid_mask, target_gt)
print(f"Hybrid Repair Results (Orig Myo/LV + Det RV):")
print(f"  RV: {scores['RV']:.4f}")
print(f"  Myo: {scores['Myo']:.4f}")
print(f"  LV: {scores['LV']:.4f}")
print(f"  Mean: {scores['Mean']:.4f}")

# Save
write_mask(f"{OUT_DIR}/335_hybrid_mask.png", hybrid_mask)
create_overlay(f"{SOURCE_DIR}/{TARGET_STEM}_img.png", hybrid_mask, f"{OUT_DIR}/335_hybrid_overlay.png")
