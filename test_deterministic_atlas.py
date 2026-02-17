"""
Deterministic Individual Atlas Repair.

Strategy:
1. Load Good Neighbor Mask (slice 019, 'atlas')
2. Load Bad Prediction (slice 009, 'anchor')
3. Alignment:
   - Calculate LV centroid of Atlas (019).
   - Calculate LV centroid of Anchor (009 prediction).
   - Shift Atlas to match Anchor's LV centroid.
4. Scaling:
   - Scale the Atlas by factor 0.8 (ED -> ES heuristic).
5. Diagnostic Adjustment (Optional):
   - Refine based on diagnosis? (For now, just scaling).
6. Evaluate Dice vs GT.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import yaml
from scipy import ndimage as ndi

from cardiac_agent_postproc.io_utils import read_mask, write_mask
from cardiac_agent_postproc.vlm_guardrail import create_overlay

# Config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)
SOURCE_DIR = cfg["paths"]["source_dir"]
OUT_DIR = "/tmp/deterministic_test"
os.makedirs(OUT_DIR, exist_ok=True)

# Cases
ATLAS_STEM = "335_original_lax_4c_019" # Good neighbor (Dice 0.85, ED approx)
TARGET_STEM = "335_original_lax_4c_009" # Bad slice (Dice 0.37, ES approx)

# Load data
atlas_mask = read_mask(f"{SOURCE_DIR}/{ATLAS_STEM}_pred.png")
target_pred = read_mask(f"{SOURCE_DIR}/{TARGET_STEM}_pred.png")
target_gt = read_mask(f"{SOURCE_DIR}/{TARGET_STEM}_gt.png")
target_img = f"{SOURCE_DIR}/{TARGET_STEM}_img.png"

# Helper: Centroid
def get_centroid(mask, label=3):
    coords = np.argwhere(mask == label)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0) # (y, x)

# 1. Alignment
atlas_lv_c = get_centroid(atlas_mask, label=3)
target_lv_c = get_centroid(target_pred, label=3)

print("=== Alignment ===")
print(f"Atlas LV Centroid (019): {atlas_lv_c}")
print(f"Target Pred LV Centroid (009): {target_lv_c}")

if atlas_lv_c is None or target_lv_c is None:
    print("Error: LV missing in atlas or target prediction. Cannot align.")
    sys.exit(1)

shift = target_lv_c - atlas_lv_c
print(f"Shift Vector (y,x): {shift}")

# Apply shift
from scipy.ndimage import shift as scipy_shift
# We need to shift the whole mask (nearest interpolation to keep integers)
shifted_mask = scipy_shift(atlas_mask, shift, order=0, mode='constant', cval=0)

# 2. Scaling
# Scaling around the new centroid (target_lv_c)
print("\n=== Scaling ===")
SCALE_FACTOR = 0.8 # Heuristic for ED->ES contraction
print(f"Scaling factor: {SCALE_FACTOR}")

def scale_mask(mask, center, scale):
    h, w = mask.shape
    cy, cx = center
    
    # Create coordinate grid
    y, x = np.indices((h, w))
    
    # Center coordinates
    y = y - cy
    x = x - cx
    
    # Scale inverse (to map output pixels back to input)
    y = y / scale
    x = x / scale
    
    # Un-center
    y = y + cy
    x = x + cx
    
    # Map back using nearest neighbor
    # map_coordinates expects (row_coords, col_coords)
    from scipy.ndimage import map_coordinates
    coords = np.array([y.flatten(), x.flatten()])
    
    mapped = map_coordinates(mask, coords, order=0, mode='constant', cval=0)
    return mapped.reshape((h, w))

scaled_mask = scale_mask(shifted_mask, target_lv_c, SCALE_FACTOR)

# 3. Evaluate
print("\n=== Evaluation ===")
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

# Original
orig_scores = compute_dice(target_pred, target_gt)
print(f"Original Prediction Dice:")
print(f"  RV: {orig_scores['RV']:.4f}")
print(f"  Myo: {orig_scores['Myo']:.4f}")
print(f"  LV: {orig_scores['LV']:.4f}")
print(f"  Mean: {orig_scores['Mean']:.4f}")

# Deterministic
det_scores = compute_dice(scaled_mask, target_gt)
print(f"\nDeterministic Repair (Individual Atlas + Align + Scale 0.8):")
print(f"  RV: {det_scores['RV']:.4f} (Δ={det_scores['RV']-orig_scores['RV']:+.4f})")
print(f"  Myo: {det_scores['Myo']:.4f} (Δ={det_scores['Myo']-orig_scores['Myo']:+.4f})")
print(f"  LV: {det_scores['LV']:.4f} (Δ={det_scores['LV']-orig_scores['LV']:+.4f})")
print(f"  Mean: {det_scores['Mean']:.4f} (Δ={det_scores['Mean']-orig_scores['Mean']:+.4f})")

# Save
write_mask(f"{OUT_DIR}/335_deterministic_mask.png", scaled_mask)
create_overlay(target_img, scaled_mask, f"{OUT_DIR}/335_deterministic_overlay.png")
print(f"\nResults saved to {OUT_DIR}")
