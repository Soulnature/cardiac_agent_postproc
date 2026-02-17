"""
Individual-Level Atlas + Nano Banana Editing.

Flow:
1. Find good neighbor slice from SAME patient (individual atlas)
2. Run diagnosis on the bad slice
3. Send to Gemini: good neighbor overlay + bad overlay + diagnostic text
4. Ask Gemini to edit the bad mask guided by the good neighbor
5. Parse & evaluate
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import yaml
from google import genai
from google.genai import types

from cardiac_agent_postproc.io_utils import read_mask, read_image, write_mask
from cardiac_agent_postproc.vlm_guardrail import create_overlay

# Config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

API_KEY = cfg["llm"]["api_key"]
SOURCE_DIR = cfg["paths"]["source_dir"]
OUT_DIR = "/tmp/gemini_individual_test"
os.makedirs(OUT_DIR, exist_ok=True)

client = genai.Client(api_key=API_KEY)

# ── Patient 335: bad slice vs good slice ──
BAD_STEM = "335_original_lax_4c_009"
GOOD_STEM = "335_original_lax_4c_019"  # Same patient, Dice=0.851

bad_mask = read_mask(f"{SOURCE_DIR}/{BAD_STEM}_pred.png")
good_mask = read_mask(f"{SOURCE_DIR}/{GOOD_STEM}_pred.png")
gt = read_mask(f"{SOURCE_DIR}/{BAD_STEM}_gt.png")

print("=== Individual Atlas Info ===")
print(f"Bad slice ({BAD_STEM}):")
for c, n in {1:"RV",2:"Myo",3:"LV"}.items():
    print(f"  {n}: {np.count_nonzero(bad_mask==c)}px")

print(f"\nGood neighbor ({GOOD_STEM}):")
for c, n in {1:"RV",2:"Myo",3:"LV"}.items():
    print(f"  {n}: {np.count_nonzero(good_mask==c)}px")

# ── Create overlays ──
bad_overlay = f"{OUT_DIR}/{BAD_STEM}_bad_overlay.png"
good_overlay = f"{OUT_DIR}/{GOOD_STEM}_good_overlay.png"
create_overlay(f"{SOURCE_DIR}/{BAD_STEM}_img.png", bad_mask, bad_overlay)
create_overlay(f"{SOURCE_DIR}/{GOOD_STEM}_img.png", good_mask, good_overlay)

# Also create pure color mask of the good neighbor
def mask_to_rgb(m):
    h, w = m.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[m == 1] = [255, 0, 0]
    rgb[m == 2] = [0, 255, 0]
    rgb[m == 3] = [0, 0, 255]
    return rgb

good_color = mask_to_rgb(good_mask)
good_color_path = f"{OUT_DIR}/{GOOD_STEM}_color.png"
cv2.imwrite(good_color_path, cv2.cvtColor(good_color, cv2.COLOR_RGB2BGR))

bad_color = mask_to_rgb(bad_mask)
bad_color_path = f"{OUT_DIR}/{BAD_STEM}_color.png"
cv2.imwrite(bad_color_path, cv2.cvtColor(bad_color, cv2.COLOR_RGB2BGR))

# ── Diagnosis text ──
h, w = bad_mask.shape
total_px = h * w
diagnosis = []
for c, name in {1:"RV", 2:"Myo", 3:"LV"}.items():
    bad_area = np.count_nonzero(bad_mask == c)
    good_area = np.count_nonzero(good_mask == c)
    ratio = bad_area / (good_area + 1e-8)
    diagnosis.append(
        f"- {name}: bad_slice={bad_area}px, good_neighbor={good_area}px, "
        f"ratio={ratio:.2f}"
    )
    if ratio < 0.3:
        diagnosis.append(
            f"  → CRITICAL: {name} is {100*(1-ratio):.0f}% smaller than neighbor. "
            f"Needs major expansion."
        )
    elif ratio > 2.0:
        diagnosis.append(
            f"  → WARNING: {name} is {100*(ratio-1):.0f}% larger than neighbor."
        )

diagnosis_text = "\n".join(diagnosis)
print(f"\n=== Diagnosis ===\n{diagnosis_text}")

# ── Send to Gemini ──
print("\n=== Gemini Individual Atlas Editing ===")

contents = []
contents.append(
    "You are a cardiac MRI segmentation correction system.\n\n"
    "I have TWO slices from the SAME patient's 4-chamber cardiac MRI:\n"
    "1. A GOOD slice with correct segmentation (from a different cardiac phase)\n"
    "2. A BAD slice with incorrect segmentation that needs fixing\n\n"
    "Both slices show the same heart from the same angle. "
    "The structures (RV, Myo, LV) should be in roughly similar positions and proportions.\n\n"
    "Colors: RED=RV (Right Ventricle), GREEN=Myo (Myocardium), BLUE=LV (Left Ventricle)\n\n"
)

# Show good neighbor
contents.append(
    "GOOD slice (same patient, correct segmentation) — use this as reference:\n"
    "MRI image with overlay:"
)
contents.append(types.Part.from_bytes(
    data=open(good_overlay, "rb").read(), mime_type="image/png"
))
contents.append("Pure segmentation mask:")
contents.append(types.Part.from_bytes(
    data=open(good_color_path, "rb").read(), mime_type="image/png"
))

# Show bad slice
contents.append(
    "\n\nBAD slice (same patient, INCORRECT segmentation — needs fixing):\n"
    "MRI image with overlay:"
)
contents.append(types.Part.from_bytes(
    data=open(bad_overlay, "rb").read(), mime_type="image/png"
))
contents.append("Pure segmentation mask:")
contents.append(types.Part.from_bytes(
    data=open(bad_color_path, "rb").read(), mime_type="image/png"
))

contents.append(
    f"\n\n**Diagnosis (comparing bad slice to good neighbor):**\n{diagnosis_text}\n\n"
    "Please generate the CORRECTED pure segmentation mask for the BAD slice.\n"
    "Rules:\n"
    "1. Use the GOOD neighbor as a reference for structure proportions\n"
    "2. RV should be a crescent shape, similar proportion to the good neighbor\n"
    "3. Myo should be a complete ring around LV\n"
    "4. Output ONLY the pure color mask (RED/GREEN/BLUE on BLACK background)\n"
    "5. Output size MUST be 256x256 pixels\n"
    "6. Keep structures aligned with the BAD slice's MRI anatomy (not copied from good slice)\n"
)

print("  Sending 4 images (2 overlays + 2 masks) + diagnosis text...")
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=contents,
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    ),
)

# Parse
for part in response.parts:
    if part.text:
        print(f"  Text: {part.text[:200]}")
    elif part.inline_data:
        gen_path = f"{OUT_DIR}/{BAD_STEM}_individual_gen.png"
        with open(gen_path, "wb") as f:
            f.write(part.inline_data.data)

        gen = cv2.imread(gen_path)
        gen_rgb = cv2.cvtColor(gen, cv2.COLOR_BGR2RGB)
        print(f"  Generated: {gen_rgb.shape}")

        if gen_rgb.shape[:2] != (256, 256):
            gen_rgb = cv2.resize(gen_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)

        r, g, b = gen_rgb[:,:,0].astype(float), gen_rgb[:,:,1].astype(float), gen_rgb[:,:,2].astype(float)
        total = r + g + b
        label = np.zeros((256, 256), dtype=np.uint8)
        nonblack = total > 50
        label[nonblack & (r >= g) & (r >= b)] = 1
        label[nonblack & (g > r) & (g >= b)] = 2
        label[nonblack & (b > r) & (b > g)] = 3

        print(f"\n=== Results ===")
        for c, n in {1:"RV", 2:"Myo", 3:"LV"}.items():
            area = np.count_nonzero(label==c)
            gt_area = np.count_nonzero(gt==c)
            print(f"  {n}: generated={area}px, GT={gt_area}px")

        print(f"\nDice vs GT:")
        for c, n in {1:"RV", 2:"Myo", 3:"LV"}.items():
            p, gm = (label==c), (gt==c)
            dice = 2*(p&gm).sum()/(p.sum()+gm.sum()+1e-8)
            orig_dice = 2*((bad_mask==c)&(gt==c)).sum()/((bad_mask==c).sum()+(gt==c).sum()+1e-8)
            delta = dice - orig_dice
            sign = "+" if delta >= 0 else ""
            print(f"  {n}: {dice:.4f} (orig={orig_dice:.4f}, Δ={sign}{delta:.4f})")

        mean = np.mean([2*((label==c)&(gt==c)).sum()/((label==c).sum()+(gt==c).sum()+1e-8) for c in [1,2,3]])
        orig_mean = np.mean([2*((bad_mask==c)&(gt==c)).sum()/((bad_mask==c).sum()+(gt==c).sum()+1e-8) for c in [1,2,3]])
        delta_m = mean - orig_mean
        sign_m = "+" if delta_m >= 0 else ""
        print(f"  Mean: {mean:.4f} (orig={orig_mean:.4f}, Δ={sign_m}{delta_m:.4f})")

        # Save
        write_mask(f"{OUT_DIR}/{BAD_STEM}_gen_mask.png", label)
        create_overlay(
            f"{SOURCE_DIR}/{BAD_STEM}_img.png", label,
            f"{OUT_DIR}/{BAD_STEM}_gen_overlay.png"
        )
        print(f"\nResults saved to {OUT_DIR}")

print("\n=== DONE ===")
