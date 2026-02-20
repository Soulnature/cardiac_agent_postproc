#!/usr/bin/env python3
"""
Fast triage: send ONLY 1 overlay image per case to VLM (no reference images).
Much faster than the full pipeline (~5-15s/case depending on API).

Usage:
    conda activate FDA
    python scripts/fast_triage_all.py --config config/gpt4o_case307.yaml
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cardiac_agent_postproc.api_client import OpenAICompatClient
from cardiac_agent_postproc.settings import LLMSettings
from cardiac_agent_postproc.vlm_guardrail import create_overlay

# Directories with pre-built overlays from previous pipeline runs
EXISTING_OVERLAY_DIRS = [
    "batch_repair_results/debug_images",
]

JUDGE_PROMPT = (
    "You are an image segmentation quality judge.\n\n"
    "This overlay image has 4 panels: Raw input, Ground Truth (GT), Prediction (Pred), Error map.\n"
    "Color legend: Cyan=Region A, Yellow ring=Region B (ring-shaped), Red-brown=Region C\n\n"
    "Classify this segmentation as 'good', 'borderline', or 'bad'.\n"
    "Compare the Pred panel against the GT panel.\n\n"
    "CRITERIA for 'good' (NO FIX NEEDED):\n"
    "- All 3 regions (A, B, C) are clearly present and correctly shaped\n"
    "- The ring structure (Region B) is COMPLETE and CLOSED (no visible gaps)\n"
    "- Regions are not merged or fragmented\n"
    "- Minor imperfections are OK (slight roughness, small thickness variation)\n"
    "- Error map shows only small colored regions\n\n"
    "CRITERIA for 'borderline':\n"
    "- Small gaps in the ring structure (mostly intact)\n"
    "- Moderate class confusion in localized areas\n"
    "- One region is slightly fragmented but recognizable\n\n"
    "CRITERIA for 'bad' (NEEDS FIX):\n"
    "- Ring structure has LARGE gaps or is completely missing\n"
    "- Massive merging of regions (giant blob, no separation)\n"
    "- Non-standard shapes (squares, exploded pixels, noise)\n"
    "- A region is entirely absent or severely fragmented\n"
    "- Error map shows large red/blue regions\n\n"
    "SCORE SCALE (1-10):\n"
    "  1-3: Bad, major structural errors → quality='bad'\n"
    "  4-6: Borderline, noticeable issues → quality='borderline'\n"
    "  7-8: Good, minor issues only → quality='good'\n"
    "  9-10: Excellent, near-perfect → quality='good'\n\n"
    "IMPORTANT: Use your score to set quality. Score >= 7 means 'good'.\n\n"
    "Respond in JSON:\n"
    '{"quality": "good"|"bad"|"borderline", '
    '"score": 1-10, '
    '"confidence": 0.0-1.0, '
    '"issues": ["list of specific issues found"], '
    '"reasoning": "brief explanation"}'
)


def discover_cases(source_dir):
    """Find all case stems in source directory."""
    cases = {}
    for img_file in sorted(glob.glob(os.path.join(source_dir, "*_img.png"))):
        stem = os.path.basename(img_file).replace("_img.png", "")
        pred_file = os.path.join(source_dir, f"{stem}_pred.png")
        gt_file = os.path.join(source_dir, f"{stem}_gt.png")
        if os.path.exists(pred_file):
            cases[stem] = {
                "img": img_file,
                "pred": pred_file,
                "gt": gt_file if os.path.exists(gt_file) else None,
            }
    return cases


def compute_dice_from_masks(pred_path, gt_path):
    """Compute Dice score between prediction and ground truth PNG masks."""
    if gt_path is None or not os.path.exists(gt_path):
        return -1.0
    from PIL import Image
    pred = np.array(Image.open(pred_path).convert("L"))
    gt = np.array(Image.open(gt_path).convert("L"))
    pred_bin = pred > 0
    gt_bin = gt > 0
    intersection = np.sum(pred_bin & gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def find_overlay(stem, new_overlay_dir):
    """Find existing overlay from previous runs or return path for new one."""
    # Check existing overlay directories
    for d in EXISTING_OVERLAY_DIRS:
        pattern = os.path.join(d, f"{stem}*triage_overlay*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0], True

    # Also check with _img_ prefix (pipeline convention)
    for d in EXISTING_OVERLAY_DIRS:
        pattern = os.path.join(d, f"{stem}_img_triage_overlay.png")
        if os.path.exists(pattern):
            return pattern, True

    # No existing overlay, return new path
    return os.path.join(new_overlay_dir, f"{stem}_overlay.png"), False


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    source_dir = cfg["paths"]["source_dir"]
    print(f"Source dir: {source_dir}")

    vg_cfg = cfg.get("vision_guardrail", {})
    settings = LLMSettings()
    vlm = OpenAICompatClient(
        base_url=vg_cfg.get("base_url", settings.openai_base_url),
        api_key=vg_cfg.get("api_key", settings.openai_api_key),
        model=vg_cfg.get("model", settings.openai_model),
        timeout=float(vg_cfg.get("timeout", 180.0)),
        provider=vg_cfg.get("provider", settings.llm_provider),
    )
    model_name = vg_cfg.get("model", "unknown").replace("/", "_")
    output_csv = f"batch_eval_results/fast_triage_{model_name}.csv"
    print(f"VLM model: {model_name}")
    print(f"Output: {output_csv}")

    cases = discover_cases(source_dir)
    print(f"Found {len(cases)} cases")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    new_overlay_dir = "batch_eval_results/fast_triage_overlays"
    os.makedirs(new_overlay_dir, exist_ok=True)

    results = []
    total = len(cases)
    reused = 0
    created = 0

    for i, (stem, files) in enumerate(cases.items()):
        t0 = time.time()

        # Dice
        dice = compute_dice_from_masks(files["pred"], files["gt"])

        # Find or create overlay
        overlay_path, exists = find_overlay(stem, new_overlay_dir)
        if exists:
            reused += 1
        else:
            from PIL import Image
            pred_mask = np.array(Image.open(files["pred"]).convert("L"))
            success = create_overlay(files["img"], pred_mask, overlay_path)
            if not success:
                print(f"[{i+1}/{total}] {stem}: overlay failed, skipping")
                continue
            created += 1

        # Call VLM with retry (GPT-4o may rate-limit or fail intermittently)
        result = {}
        for attempt in range(3):
            try:
                result = vlm.chat_vision_json(
                    system="You are an expert image segmentation quality assessor.",
                    user_text=JUDGE_PROMPT,
                    image_path=overlay_path,
                    temperature=0.1,
                    image_detail="auto",
                )
                if result and result.get("score", 0) > 0:
                    break  # success
                # Empty result, retry
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"[{i+1}/{total}] {stem}: VLM attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)

        if not result or result.get("score", 0) == 0:
            result = {"quality": "error", "score": 0, "confidence": 0,
                      "issues": ["API call failed after retries"], "reasoning": "VLM call failed"}

        quality = result.get("quality", "borderline")
        score = float(result.get("score", 0))
        confidence = float(result.get("confidence", 0.5))
        issues = result.get("issues", [])
        reasoning = result.get("reasoning", "")

        # Map score to category
        if score >= 7:
            category = "good"
        elif score <= 4:
            category = "needs_fix"
        else:
            category = "borderline"

        elapsed = time.time() - t0
        print(f"[{i+1}/{total}] {stem}: Dice={dice:.3f} | score={score} q={quality} → {category} ({elapsed:.1f}s)")

        results.append({
            "Stem": stem,
            "Dice": f"{dice:.4f}",
            "VLM_Quality": quality,
            "VLM_Score": score,
            "VLM_Confidence": f"{confidence:.2f}",
            "Category": category,
            "Issues": str(issues),
            "Reasoning": reasoning,
            "Time_s": f"{elapsed:.1f}",
        })

        # Save incrementally every 20 cases
        if len(results) % 20 == 0:
            _write_csv(results, output_csv)

    _write_csv(results, output_csv)
    print(f"\nDone! Results saved to {output_csv}")
    print(f"Overlays: {reused} reused, {created} new")

    goods = sum(1 for r in results if r["Category"] == "good")
    bads = sum(1 for r in results if r["Category"] == "needs_fix")
    borders = sum(1 for r in results if r["Category"] == "borderline")
    print(f"Summary: {goods} good, {borders} borderline, {bads} needs_fix (total {len(results)})")


def _write_csv(results, output_csv):
    if not results:
        return
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast single-image VLM triage")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML file")
    args = parser.parse_args()
    main(args.config)
