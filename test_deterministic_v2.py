"""
Deterministic Individual Atlas Repair v2 (Image Registration).

Strategy:
1. Load Good Neighbor Image (019) and Target Image (009).
2. Register images using Phase Correlation to find translation shift.
   - This relies on image anatomy (GT) rather than bad predictions.
3. Apply shift to Good Neighbor Mask.
4. Scale mask by 0.7 (based on area ratio 1163/1830 ~ 0.64).
5. Evaluate.
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
OUT_DIR = "/tmp/deterministic_v2"
os.makedirs(OUT_DIR, exist_ok=True)

ATLAS_STEM = "335_original_lax_4c_019" # Good neighbor
TARGET_STEM = "335_original_lax_4c_009" # Bad slice

# Load
atlas_img = read_image(f"{SOURCE_DIR}/{ATLAS_STEM}_img.png")
target_img = read_image(f"{SOURCE_DIR}/{TARGET_STEM}_img.png")
atlas_mask = read_mask(f"{SOURCE_DIR}/{ATLAS_STEM}_pred.png")
target_gt = read_mask(f"{SOURCE_DIR}/{TARGET_STEM}_gt.png")

# 1. Registration (Phase Correlation)
# Ensure grayscale and float32
def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY if img.shape[2]==4 else cv2.COLOR_RGB2GRAY)
    return img

src_gray = to_gray(atlas_img).astype(np.float32)
dst_gray = to_gray(target_img).astype(np.float32)

# Hann window to reduce edge effects
def create_hann_window(shape):
    hy = np.hanning(shape[0])
    hx = np.hanning(shape[1])
    return np.outer(hy, hx)

window = create_hann_window(src_gray.shape)
shift, response = cv2.phaseCorrelate(src_gray * window, dst_gray * window)
print(f"Detected Shift (x, y): {shift}, Response: {response}")

# Shift is (dx, dy)
dx, dy = shift

# 2. Apply shift
# We shift the mask by (dy, dx)
shifted_mask = ndi.shift(atlas_mask, (dy, dx), order=0, mode='constant', cval=0)

# 3. Scaling
SCALE_FACTOR = 0.7
print(f"Scaling factor: {SCALE_FACTOR}")

def scale_mask_image_center(mask, scale, shift_x=0, shift_y=0):
    """Scale around image center"""
    h, w = mask.shape
    cy, cx = h//2, w//2
    # Grid
    y, x = np.indices((h, w))
    y = (y - cy) / scale + cy
    x = (x - cx) / scale + cx
    
    # Map
    coords = np.array([y.flatten(), x.flatten()])
    from scipy.ndimage import map_coordinates
    mapped = map_coordinates(mask, coords, order=0, mode='constant', cval=0)
    return mapped.reshape((h, w))

# Calculate center of mass of the shifted mask to use as scaling center
# Better to scale around the mask centroid than image center
coords = np.argwhere(shifted_mask > 0)
if len(coords) > 0:
    cy, cx = coords.mean(axis=0)
    print(f"Scaling around centroid: {cy:.1f}, {cx:.1f}")
else:
    cy, cx = 128, 128
    
scaled_mask = np.zeros_like(shifted_mask)

# Scale
# Using affine transform for cleaner code
M = cv2.getRotationMatrix2D((cx, cy), 0, SCALE_FACTOR)
scaled_mask_cv = cv2.warpAffine(shifted_mask, M, (shifted_mask.shape[1], shifted_mask.shape[0]), flags=cv2.INTER_NEAREST)


# 4. Evaluate
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

# Original prediction results (hardcoded from prev runs)
print(f"Original Prediction Mean: 0.3729 (RV 0.056)")

scores = compute_dice(scaled_mask_cv, target_gt)
print(f"\nDeterministic v2 (PhaseCorr + Scale 0.7):")
print(f"  RV: {scores['RV']:.4f}")
print(f"  Myo: {scores['Myo']:.4f}")
print(f"  LV: {scores['LV']:.4f}")
print(f"  Mean: {scores['Mean']:.4f}")

# Save
write_mask(f"{OUT_DIR}/335_det_v2_mask.png", scaled_mask_cv)
create_overlay(f"{SOURCE_DIR}/{TARGET_STEM}_img.png", scaled_mask_cv, f"{OUT_DIR}/335_det_v2_overlay.png")
print(f"\nResults saved to {OUT_DIR}")
