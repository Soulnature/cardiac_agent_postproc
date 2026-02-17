
import os
import sys
import yaml
import cv2
import json
import numpy as np
from google import genai
from google.genai import types

# Config
API_KEY = "AIzaSyC7Lvp9jdP7BuD2si5Ql8V7B5SFCp0VAVc"
CASE_STEM = "244_original_lax_4c_000"
SOURCE_DIR = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export"
OUT_DIR = "/tmp/gemini_bbox_test"
os.makedirs(OUT_DIR, exist_ok=True)

# Paths
img_path = os.path.join(SOURCE_DIR, f"{CASE_STEM}_img.png")
pred_path = os.path.join(SOURCE_DIR, f"{CASE_STEM}_pred.png")

# Helper to create overlay (green only for Myo to highlight gaps)
def create_highlight_overlay(img_path, mask_path, out_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"Mask unique values: {np.unique(mask)}")
    
    # Resize mask to match image if needed
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Handle 0-255 scaling if present
    if mask.max() > 3:
        # Assuming [0, 85, 170, 255] mapping
        print("Normalizing mask from 0-255 to 0-3")
        mask = np.round(mask / 85).astype(np.uint8)

    overlay = img.copy()
    
    # Color coding: Myo (label 2) = Green
    green_mask = (mask == 2)
    overlay[green_mask] = [0, 255, 0] # BGR
    
    if np.count_nonzero(green_mask) == 0:
        print("WARNING: No Myocardium pixels found after normalization!")
    else:
        print(f"Myocardium pixels: {np.count_nonzero(green_mask)}")
    
    # Blend
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, overlay)
    cv2.imwrite(out_path, overlay)
    return overlay

overlay_path = os.path.join(OUT_DIR, f"{CASE_STEM}_myo_overlay.png")
create_highlight_overlay(img_path, pred_path, overlay_path)
print(f"Created overlay: {overlay_path}")

# Client
client = genai.Client(api_key=API_KEY)

# Prompt
prompt = """
You are a cardiac findings expert.
I will show you a cardiac MRI segmentation overlay.
- The Green region is the Myocardium (heart muscle).
- Ideally, the Green region should be a CLOSED RING (donut shape).
- However, there is a GAP or BREAK in the ring.

Task:
1. Identify the location of the GAP/BREAK in the green ring.
2. Return the Bounding Box of this gap in [ymin, xmin, ymax, xmax] format (0-1000 scale).

Output JSON ONLY:
{
  "gap_found": true,
  "box_2d": [ymin, xmin, ymax, xmax],
  "description": "Short text describing location (e.g., septal wall)"
}
"""

print(f"Sending to gemini-2.5-flash...")
with open(overlay_path, "rb") as f:
    img_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[prompt, types.Part.from_bytes(data=img_bytes, mime_type="image/png")],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0
    ),
)

print("\nResponse:")
print(response.text)

# Visualization
try:
    data = json.loads(response.text)
    if data.get("gap_found"):
        box = data["box_2d"]
        h, w = 256, 256 # MnM2 size
        ymin, xmin, ymax, xmax = box
        
        # Normalize 0-1000 to pixels
        y1 = int(ymin / 1000 * h)
        x1 = int(xmin / 1000 * w)
        y2 = int(ymax / 1000 * h)
        x2 = int(xmax / 1000 * w)
        
        print(f"Drawing box: [{x1}, {y1}, {x2}, {y2}]")
        
        vis_img = cv2.imread(overlay_path)
        # Resize to 256x256 if not already (MnM2 are often cropped)
        if vis_img.shape[:2] != (h, w):
             vis_img = cv2.resize(vis_img, (w, h))
             
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_img, "GAP", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        out_vis = os.path.join(OUT_DIR, f"{CASE_STEM}_bbox_result.png")
        cv2.imwrite(out_vis, vis_img)
        print(f"Saved visualization to {out_vis}")
except Exception as e:
    print(f"Error parsing/drawing: {e}")
