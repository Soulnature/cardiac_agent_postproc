import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())

from cardiac_agent_postproc.quality_model import extract_quality_features
from cardiac_agent_postproc.agents.triage_agent import _detect_issues

def test_features(stem):
    print(f"Testing features for {stem}...")
    root = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export"
    img_path = f"{root}/{stem}_img.png"
    mask_path = f"{root}/{stem}_pred.png"
    
    if not os.path.exists(mask_path):
        print(f"Mask file not found: {mask_path}")
        return

    # Load mask and dummy image
    mask = np.array(Image.open(mask_path).convert("L"))
    print(f"Mask unique values: {np.unique(mask)}")
    # We can pass None for image if extract_quality_features handles it, or load the image.
    # It seems extract_quality_features might use image for some features.
    img = np.array(Image.open(img_path)) if os.path.exists(img_path) else np.zeros_like(mask)
    
    # Use the standalone function with 3 args: mask, img, view_type
    # Assuming view_type=4ch for now as in the case name
    features = extract_quality_features(mask, img, "4ch")
    print("Features extracted:")
    for k, v in features.items():
        print(f"  {k}: {v}")
        
    # Check specifically for structural flags
    issues = _detect_issues(features)
    print(f"Issues detected: {issues}")
    print("Features extracted:")
    for k, v in features.items():
        print(f"  {k}: {v}")
        
    # Check specifically for structural flags
    issues = TriageAgent._detect_issues(features)
    print(f"Issues detected: {issues}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_features(sys.argv[1])
    else:
        # Default to the 'good' case that VLM failed on
        test_features("201_original_lax_4c_000")
