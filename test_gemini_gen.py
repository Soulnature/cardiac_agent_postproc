"""
Proof of concept: Use Gemini Nano Banana (gemini-2.5-flash-image) to
generate a segmentation mask for a cardiac MRI image.

Approach: Send the MRI image + good example overlays as references,
ask Gemini to generate a corrected segmentation overlay, then parse
the generated image back into a label mask.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import yaml
from PIL import Image
from google import genai
from google.genai import types

from cardiac_agent_postproc.io_utils import read_mask, read_image, write_mask
from cardiac_agent_postproc.vlm_guardrail import create_overlay

# Config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

API_KEY = cfg["llm"]["api_key"]
SOURCE_DIR = cfg["paths"]["source_dir"]
OUT_DIR = "/tmp/gemini_gen_test"
os.makedirs(OUT_DIR, exist_ok=True)

# Target case (worst case)
TARGET_STEM = "335_original_lax_4c_009"

# Good reference cases (same view type 4ch, high quality)
# Pick 2-3 good cases to show as examples
GOOD_STEMS = [
    "201_original_lax_4c_000",
    "201_original_lax_4c_009",
    "281_original_lax_4c_000",
]

client = genai.Client(api_key=API_KEY)

# Create overlays for good reference cases
print("=== Preparing reference overlays ===")
ref_overlays = []
for stem in GOOD_STEMS:
    img_path = os.path.join(SOURCE_DIR, f"{stem}_img.png")
    gt_path = os.path.join(SOURCE_DIR, f"{stem}_gt.png")
    if not os.path.exists(gt_path):
        print(f"  Skipping {stem} (no GT)")
        continue
    overlay_path = os.path.join(OUT_DIR, f"{stem}_ref_overlay.png")
    create_overlay(img_path, read_mask(gt_path), overlay_path)
    ref_overlays.append((stem, overlay_path))
    print(f"  {stem}: overlay created")

# Target MRI image
target_img_path = os.path.join(SOURCE_DIR, f"{TARGET_STEM}_img.png")
target_pred_path = os.path.join(SOURCE_DIR, f"{TARGET_STEM}_pred.png")
target_gt_path = os.path.join(SOURCE_DIR, f"{TARGET_STEM}_gt.png")

# Also create overlay of the bad prediction for reference
bad_overlay_path = os.path.join(OUT_DIR, f"{TARGET_STEM}_bad_overlay.png")
create_overlay(target_img_path, read_mask(target_pred_path), bad_overlay_path)

# Create GT overlay for comparison
if os.path.exists(target_gt_path):
    gt_overlay_path = os.path.join(OUT_DIR, f"{TARGET_STEM}_gt_overlay.png")
    create_overlay(target_img_path, read_mask(target_gt_path), gt_overlay_path)

print(f"\n=== Generating segmentation for {TARGET_STEM} ===")

# Build the prompt with reference images
contents = []

# System instruction as first text
prompt_text = (
    "You are an expert cardiac MRI segmentation system. "
    "I will show you reference examples of correct cardiac segmentation overlays, "
    "then ask you to generate a segmentation overlay for a new image.\n\n"
    "In the segmentation overlay:\n"
    "- RED = Right Ventricle (RV)\n"
    "- GREEN = Myocardium (Myo)\n"
    "- BLUE = Left Ventricle (LV)\n"
    "- The overlay is semi-transparent over the grayscale MRI image\n\n"
    "Here are examples of CORRECT segmentations for 4-chamber (4ch) cardiac MRI views:\n"
)
contents.append(prompt_text)

# Add reference overlays
for i, (stem, overlay_path) in enumerate(ref_overlays):
    contents.append(f"\nReference example {i+1} ({stem}) - correct segmentation:")
    contents.append(types.Part.from_bytes(
        data=open(overlay_path, "rb").read(),
        mime_type="image/png",
    ))

# Add the target image
contents.append(
    "\n\nNow, here is a NEW cardiac MRI image that needs segmentation. "
    "The current automated prediction is very poor (shown below):\n"
)
contents.append(f"Bad prediction overlay:")
contents.append(types.Part.from_bytes(
    data=open(bad_overlay_path, "rb").read(),
    mime_type="image/png",
))

contents.append(f"\nOriginal MRI image (no overlay):")
contents.append(types.Part.from_bytes(
    data=open(target_img_path, "rb").read(),
    mime_type="image/png",
))

contents.append(
    "\n\nPlease generate a CORRECTED segmentation overlay for this MRI image. "
    "The overlay should have the same format as the reference examples: "
    "semi-transparent colored regions (RED=RV, GREEN=Myo, BLUE=LV) "
    "overlaid on the grayscale MRI image. "
    "Make sure the anatomy is correct: LV should be enclosed by Myocardium ring, "
    "RV should be adjacent to the Myo on the right side. "
    "Generate the overlay image at 256x256 resolution."
)

print(f"  Sending {len(ref_overlays)} reference images + target to Gemini...")
print(f"  Model: gemini-2.5-flash-image")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    # Save generated image
    for i, part in enumerate(response.parts):
        if part.text is not None:
            print(f"  Text response: {part.text[:200]}")
        elif part.inline_data is not None:
            gen_path = os.path.join(OUT_DIR, f"{TARGET_STEM}_gemini_gen.png")
            img = part.as_image()
            img.save(gen_path)
            print(f"  Generated image saved: {gen_path}")
            print(f"  Size: {img.size}")

            # Try to parse the overlay back into a label mask
            gen_arr = np.array(img)
            print(f"  Array shape: {gen_arr.shape}, dtype: {gen_arr.dtype}")

            # If it's RGB, extract R/G/B channels to identify structures
            if len(gen_arr.shape) == 3 and gen_arr.shape[2] >= 3:
                r, g, b = gen_arr[:,:,0], gen_arr[:,:,1], gen_arr[:,:,2]

                # Simple color-based parsing:
                # RV (red): high R, low G, low B
                # Myo (green): low R, high G, low B
                # LV (blue): low R, low G, high B
                rv_mask = (r > 100) & (r > g + 30) & (r > b + 30)
                myo_mask = (g > 100) & (g > r + 30) & (g > b + 30)
                lv_mask = (b > 100) & (b > r + 30) & (b > g + 30)

                label_mask = np.zeros((gen_arr.shape[0], gen_arr.shape[1]), dtype=np.uint8)
                label_mask[rv_mask] = 1
                label_mask[myo_mask] = 2
                label_mask[lv_mask] = 3

                # Resize to 256x256 if needed
                if label_mask.shape != (256, 256):
                    label_mask = cv2.resize(label_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

                print(f"  Parsed labels: {np.unique(label_mask)}")
                for c, name in {1: "RV", 2: "Myo", 3: "LV"}.items():
                    area = np.count_nonzero(label_mask == c)
                    print(f"    {name}: {area}px")

                # Save parsed mask
                mask_path = os.path.join(OUT_DIR, f"{TARGET_STEM}_gemini_mask.png")
                write_mask(mask_path, label_mask)
                print(f"  Parsed mask saved: {mask_path}")

                # Compute Dice vs GT
                gt = read_mask(target_gt_path)
                for c, name in {1: "RV", 2: "Myo", 3: "LV"}.items():
                    p = (label_mask == c)
                    gm = (gt == c)
                    inter = (p & gm).sum()
                    union = p.sum() + gm.sum()
                    dice = 2 * inter / union if union > 0 else 1.0
                    print(f"    Dice {name}: {dice:.4f}")

                mean_dice = np.mean([
                    2 * ((label_mask == c) & (gt == c)).sum() / ((label_mask == c).sum() + (gt == c).sum() + 1e-8)
                    for c in [1, 2, 3]
                ])
                print(f"    Mean Dice: {mean_dice:.4f}")

                # Create overlay of parsed mask
                parsed_overlay = os.path.join(OUT_DIR, f"{TARGET_STEM}_gemini_overlay.png")
                create_overlay(target_img_path, label_mask, parsed_overlay)
                print(f"  Parsed overlay saved: {parsed_overlay}")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DONE ===")
