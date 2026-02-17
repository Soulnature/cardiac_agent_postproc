"""
Diagnosis-Guided Mask Editing with Gemini Nano Banana.

Flow:
1. Run diagnosis agent → get diagnostic text describing issues
2. Create overlay of the predicted mask on the MRI image  
3. Send overlay + diagnostic text to Gemini image editing
4. Parse generated image back to label mask at 256x256
5. Compute Dice vs GT
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
from cardiac_agent_postproc.view_utils import infer_view_type
from cardiac_agent_postproc.vlm_guardrail import create_overlay
from cardiac_agent_postproc.quality_model import extract_quality_features
from cardiac_agent_postproc.agents.diagnosis_agent import DiagnosisAgent, ISSUE_OPS_MAP
from cardiac_agent_postproc.agents import MessageBus, CaseContext

# ── Config ──
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["quality_model"]["enabled"] = False

API_KEY = "AIzaSyC7Lvp9jdP7BuD2si5Ql8V7B5SFCp0VAVc" # Hardcoded for test
SOURCE_DIR = cfg["paths"]["source_dir"]
OUT_DIR = "/tmp/gemini_edit_test_307"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STEM = "307_original_lax_4c_024"
# GOOD_STEMS should be valid ones. 201 and 281 are good in MnM2.
GOOD_STEMS = ["201_original_lax_4c_000", "281_original_lax_4c_000"]

# ── Step 1: Run Diagnosis ──
print("=" * 60)
print("STEP 1: Diagnosis")
print("=" * 60)

img_path = os.path.join(SOURCE_DIR, f"{TARGET_STEM}_img.png")
pred_path = os.path.join(SOURCE_DIR, f"{TARGET_STEM}_pred.png")
gt_path = os.path.join(SOURCE_DIR, f"{TARGET_STEM}_gt.png")

mask = read_mask(pred_path)
image = read_image(img_path)
gt = read_mask(gt_path)
view_type, _ = infer_view_type(TARGET_STEM, mask)

# Extract features for diagnosis
feats = extract_quality_features(mask, image, view_type)

# Build diagnosis text from rule-based analysis
diag_texts = []
h, w = mask.shape
total_px = h * w

# Area analysis
for c, name in {1: "RV", 2: "Myo", 3: "LV"}.items():
    area = np.count_nonzero(mask == c)
    frac = area / total_px
    diag_texts.append(f"- {name}: {area}px ({frac:.3f} of image)")

# Structural issues
if feats.get("cc_count_myo", 1) > 1:
    diag_texts.append(f"- ISSUE: Myocardium is disconnected ({int(feats['cc_count_myo'])} fragments)")
if feats.get("cc_count_lv", 1) > 1:
    diag_texts.append(f"- ISSUE: LV is fragmented ({int(feats['cc_count_lv'])} pieces)")
if feats.get("touch_ratio", 0) > 0.01:
    diag_texts.append(f"- ISSUE: RV and LV are directly touching (should be separated by Myocardium)")
if feats.get("hole_fraction_myo", 0) > 0.02:
    diag_texts.append(f"- ISSUE: Myocardium has holes/gaps")

# Severe area issues
EXPECTED_4CH = {"RV": (0.015, 0.06), "Myo": (0.008, 0.04), "LV": (0.012, 0.06)}
for name, (lo, hi) in EXPECTED_4CH.items():
    c = {"RV": 1, "Myo": 2, "LV": 3}[name]
    area_frac = np.count_nonzero(mask == c) / total_px
    if area_frac < lo * 0.3:
        diag_texts.append(
            f"- CRITICAL: {name} is severely undersized ({area_frac:.4f}, "
            f"expected {lo:.3f}-{hi:.3f}). Needs major reconstruction."
        )

# Topology
from scipy import ndimage as ndi
for c, name in {2: "Myo", 3: "LV"}.items():
    Mc = (mask == c)
    if Mc.sum() > 0:
        filled = ndi.binary_fill_holes(Mc)
        hole_frac = (filled & ~Mc).sum() / Mc.sum()
        if hole_frac > 0.05:
            diag_texts.append(f"- ISSUE: {name} has large holes ({hole_frac:.1%} of area)")

diagnostic_text = "\n".join(diag_texts)
print(f"View: {view_type}")
print(f"Diagnostic findings:\n{diagnostic_text}\n")

# ── Step 2: Create overlays ──
print("=" * 60)
print("STEP 2: Prepare images")
print("=" * 60)

# Bad prediction overlay
bad_overlay_path = os.path.join(OUT_DIR, f"{TARGET_STEM}_bad_overlay.png")
create_overlay(img_path, mask, bad_overlay_path)

# Good reference overlays (from GT of other cases)
ref_overlay_paths = []
for stem in GOOD_STEMS:
    ref_img = os.path.join(SOURCE_DIR, f"{stem}_img.png")
    ref_gt = os.path.join(SOURCE_DIR, f"{stem}_gt.png")
    if os.path.exists(ref_gt):
        ref_overlay = os.path.join(OUT_DIR, f"{stem}_ref_overlay.png")
        create_overlay(ref_img, read_mask(ref_gt), ref_overlay)
        ref_overlay_paths.append((stem, ref_overlay))
        print(f"  Reference: {stem}")

print(f"  Bad overlay: {bad_overlay_path}")

# ── Step 3: Send to Gemini for guided editing ──
print("\n" + "=" * 60)
print("STEP 3: Gemini-guided mask editing")
print("=" * 60)

client = genai.Client(api_key=API_KEY)

contents = []

# Instruction with diagnostic context
contents.append(
    "You are a cardiac MRI segmentation editor. "
    "I will show you reference examples of CORRECT cardiac segmentation overlays, "
    "then show you a POOR segmentation that needs correction.\n\n"
    "In the overlays:\n"
    "- RED = Right Ventricle (RV)\n"
    "- GREEN = Myocardium (Myo)\n"
    "- BLUE = Left Ventricle (LV)\n\n"
    "Here are correctly segmented 4-chamber cardiac MRI examples for reference:\n"
)

# Reference overlays  
for i, (stem, path) in enumerate(ref_overlay_paths):
    contents.append(f"\nCorrect example {i+1}:")
    contents.append(types.Part.from_bytes(data=open(path, "rb").read(), mime_type="image/png"))

# Now the bad prediction with diagnostic text
contents.append(
    "\n\nBelow is a POORLY segmented cardiac MRI that needs correction. "
    f"The image is a {view_type} (4-chamber) view.\n\n"
    f"**Automated diagnosis found these issues:**\n{diagnostic_text}\n\n"
    "Here is the current (incorrect) segmentation overlay:"
)
contents.append(types.Part.from_bytes(data=open(bad_overlay_path, "rb").read(), mime_type="image/png"))

contents.append(
    "\n\nPlease EDIT this overlay to fix the identified issues. Specifically:\n"
    "1. Keep the same MRI background image — do NOT change the underlying anatomy\n"
    "2. Correct the colored regions based on the diagnosis above\n"
    "3. Make the RV (red) an appropriately sized crescent shape, similar to the references\n"
    "4. Make the Myo (green) a complete ring surrounding the LV\n"
    "5. Keep the LV (blue) as a filled cavity inside the Myo ring\n"
    "6. The output MUST be exactly 256x256 pixels\n"
    "7. Use the same semi-transparent overlay style as the input\n\n"
    "Generate the CORRECTED segmentation overlay image now."
)

print(f"  Sending to gemini-2.5-flash...")
print(f"  {len(ref_overlay_paths)} refs + 1 bad overlay + diagnosis text")

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=contents,
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    ),
)

# ── Step 4: Parse response ──
print("\n" + "=" * 60)
print("STEP 4: Parse generated image")
print("=" * 60)

gen_mask = None
for part in response.parts:
    if part.text is not None:
        print(f"  Gemini text: {part.text[:300]}")
    elif part.inline_data is not None:
        gen_path = os.path.join(OUT_DIR, f"{TARGET_STEM}_gemini_edited.png")
        raw_bytes = part.inline_data.data
        with open(gen_path, "wb") as f:
            f.write(raw_bytes)
        print(f"  Saved: {gen_path}")

        # Parse overlay → label mask
        gen_img = cv2.imread(gen_path)
        gen_rgb = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        orig_size = gen_rgb.shape[:2]
        print(f"  Generated size: {orig_size}")

        # Resize to 256x256
        if gen_rgb.shape[:2] != (256, 256):
            gen_rgb = cv2.resize(gen_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

        r = gen_rgb[:, :, 0].astype(float)
        g = gen_rgb[:, :, 1].astype(float)
        b = gen_rgb[:, :, 2].astype(float)

        gen_mask = np.zeros((256, 256), dtype=np.uint8)

        # Color dominance parsing — identify pixels where one channel
        # is significantly stronger than the others (overlay region)
        max_ch = np.maximum(np.maximum(r, g), b)
        min_ch = np.minimum(np.minimum(r, g), b)
        saturation = max_ch - min_ch  # color "purity"

        colored = saturation > 25  # enough color saturation to be an overlay

        rv_mask = colored & (r > g) & (r > b)
        myo_mask = colored & (g > r) & (g > b)
        lv_mask = colored & (b > r) & (b > g)

        gen_mask[rv_mask] = 1
        gen_mask[myo_mask] = 2
        gen_mask[lv_mask] = 3

        for c, name in {1: "RV", 2: "Myo", 3: "LV"}.items():
            area = np.count_nonzero(gen_mask == c)
            print(f"    {name}: {area}px")

# ── Step 5: Evaluate ──
print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)

def dice_per_class(pred, gt):
    scores = {}
    for c, name in {1: "RV", 2: "Myo", 3: "LV"}.items():
        p, gm = (pred == c), (gt == c)
        inter = (p & gm).sum()
        union = p.sum() + gm.sum()
        scores[name] = 2 * inter / union if union > 0 else 1.0
    scores["mean"] = np.mean([scores["RV"], scores["Myo"], scores["LV"]])
    return scores

# Original prediction
orig_dice = dice_per_class(mask, gt)
print(f"Original prediction Dice: {orig_dice['mean']:.4f}")
for k in ["RV", "Myo", "LV"]:
    print(f"  {k}: {orig_dice[k]:.4f}")

if gen_mask is not None:
    gen_dice = dice_per_class(gen_mask, gt)
    print(f"\nGemini-edited Dice: {gen_dice['mean']:.4f}")
    for k in ["RV", "Myo", "LV"]:
        delta = gen_dice[k] - orig_dice[k]
        sign = "+" if delta >= 0 else ""
        print(f"  {k}: {gen_dice[k]:.4f}  (Δ={sign}{delta:.4f})")

    delta_mean = gen_dice["mean"] - orig_dice["mean"]
    sign = "+" if delta_mean >= 0 else ""
    print(f"\n  Mean: {orig_dice['mean']:.4f} → {gen_dice['mean']:.4f}  (Δ={sign}{delta_mean:.4f})")

    # Save results
    write_mask(os.path.join(OUT_DIR, f"{TARGET_STEM}_gemini_mask.png"), gen_mask)
    create_overlay(img_path, gen_mask, os.path.join(OUT_DIR, f"{TARGET_STEM}_gemini_overlay.png"))
    print(f"\nResults saved to {OUT_DIR}")
else:
    print("ERROR: No image generated")

print("\n=== DONE ===")
