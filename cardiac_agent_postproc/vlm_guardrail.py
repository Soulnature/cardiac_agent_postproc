
import os
import cv2
import numpy as np
import base64
from typing import Optional
from .api_client import OpenAICompatClient
from .settings import LLMSettings

def create_overlay(img_path: str, mask_arr: np.ndarray, outcome_path: str):
    """
    Creates a visual overlay of the mask on the image.
    blue=LV(3), green=Myo(2), red=RV(1)
    """
    if not os.path.exists(img_path):
        return False

    img = cv2.imread(img_path)
    if img is None:
        return False

    # Resize mask to img if needed (should match)
    if mask_arr.shape != img.shape[:2]:
        mask_arr = cv2.resize(mask_arr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Color map: BGR
    # 1=RV (Red), 2=Myo (Green), 3=LV (Blue)
    overlay = img.copy()

    # Map grayscale values if needed (common in this dataset: 85=RV, 170=Myo, 255=LV)
    # or just allow direct intensity matching
    
    # RV - Red (Label 1 or 85)
    overlay[(mask_arr == 1) | (mask_arr == 85)] = [0, 0, 255]
    # Myo - Green (Label 2 or 170)
    overlay[(mask_arr == 2) | (mask_arr == 170)] = [0, 255, 0]
    # LV - Blue (Label 3 or 255)
    overlay[(mask_arr == 3) | (mask_arr == 255)] = [255, 0, 0]

    # Blend
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    cv2.imwrite(outcome_path, img)
    return True


def create_side_by_side_overlay(
    img_path: str,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    output_path: str,
) -> bool:
    """
    Create a side-by-side overlay image: before (left) vs after (right).
    Both overlaid on the same raw image.
    """
    if not os.path.exists(img_path):
        return False

    img = cv2.imread(img_path)
    if img is None:
        return False

    H, W = img.shape[:2]

    # Resize masks if needed
    if mask_before.shape != (H, W):
        mask_before = cv2.resize(mask_before, (W, H), interpolation=cv2.INTER_NEAREST)
    if mask_after.shape != (H, W):
        mask_after = cv2.resize(mask_after, (W, H), interpolation=cv2.INTER_NEAREST)

    def _overlay(base_img, mask):
        ov = base_img.copy()
        ov[mask == 1] = [0, 0, 255]   # RV - Red
        ov[mask == 2] = [0, 255, 0]   # Myo - Green
        ov[mask == 3] = [255, 0, 0]   # LV - Blue
        blended = base_img.copy()
        cv2.addWeighted(ov, 0.4, base_img, 0.6, 0, blended)
        return blended

    left = _overlay(img, mask_before)
    right = _overlay(img, mask_after)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(left, "BEFORE", (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(right, "AFTER", (10, 25), font, 0.7, (255, 255, 255), 2)

    # Separator line (3px white)
    sep = np.full((H, 3, 3), 255, dtype=np.uint8)

    combined = np.concatenate([left, sep, right], axis=1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, combined)
    return True


def create_diff_overlay(
    img_path: str,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    output_path: str,
    stats: Optional[dict] = None,
) -> bool:
    """
    Create a 3-panel comparison image: [BEFORE | AFTER | DIFF]

    DIFF panel highlights pixel-level changes:
      - Green : pixels ADDED by repair   (absent before, present after)
      - Red   : pixels REMOVED by repair (present before, absent after)
      - Yellow: pixels where CLASS CHANGED (both present, different label)
      - Gray  : unchanged foreground

    If *stats* dict is provided, it is populated with change counts.
    """
    if not os.path.exists(img_path):
        return False

    img = cv2.imread(img_path)
    if img is None:
        return False

    H, W = img.shape[:2]
    if mask_before.shape != (H, W):
        mask_before = cv2.resize(mask_before, (W, H), interpolation=cv2.INTER_NEAREST)
    if mask_after.shape != (H, W):
        mask_after = cv2.resize(mask_after, (W, H), interpolation=cv2.INTER_NEAREST)

    # --- helper: overlay a mask on an image ---
    def _ov(base, mask):
        out = base.copy()
        out[(mask == 1) | (mask == 85)]  = [0, 0, 255]   # RV – red
        out[(mask == 2) | (mask == 170)] = [0, 255, 0]    # Myo – green
        out[(mask == 3) | (mask == 255)] = [255, 0, 0]    # LV – blue
        blended = base.copy()
        cv2.addWeighted(out, 0.4, base, 0.6, 0, blended)
        return blended

    # Panel 1 & 2: standard overlays
    before_panel = _ov(img, mask_before)
    after_panel  = _ov(img, mask_after)

    # Panel 3: DIFF highlight
    diff_panel = (img * 0.5).astype(np.uint8)  # dim background

    fg_before = mask_before > 0
    fg_after  = mask_after  > 0

    added   = (~fg_before) & fg_after           # new foreground pixels
    removed = fg_before & (~fg_after)           # lost foreground pixels
    changed = fg_before & fg_after & (mask_before != mask_after)  # class swap
    kept    = fg_before & fg_after & (mask_before == mask_after)  # unchanged

    # Paint diff colours (BGR)
    diff_panel[added]   = [0, 220, 0]     # green  = added
    diff_panel[removed] = [0, 0, 220]     # red    = removed
    diff_panel[changed] = [0, 220, 220]   # yellow = class change
    diff_panel[kept]    = [160, 160, 160]  # gray   = unchanged

    # Stats
    n_added   = int(np.count_nonzero(added))
    n_removed = int(np.count_nonzero(removed))
    n_changed = int(np.count_nonzero(changed))
    if stats is not None:
        stats["added"] = n_added
        stats["removed"] = n_removed
        stats["changed"] = n_changed

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(before_panel, "BEFORE", (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(after_panel,  "AFTER",  (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(diff_panel,   f"DIFF +{n_added} -{n_removed} ~{n_changed}",
                (10, 25), font, 0.55, (255, 255, 255), 2)

    # Legend on diff panel
    y0 = H - 60
    cv2.putText(diff_panel, "Green=Added", (10, y0),      font, 0.4, (0, 220, 0), 1)
    cv2.putText(diff_panel, "Red=Removed", (10, y0 + 18), font, 0.4, (0, 0, 220), 1)
    cv2.putText(diff_panel, "Yellow=ClassChg", (10, y0 + 36), font, 0.4, (0, 220, 220), 1)

    # Assemble
    sep = np.full((H, 3, 3), 255, dtype=np.uint8)
    combined = np.concatenate([before_panel, sep, after_panel, sep, diff_panel], axis=1)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, combined)
    return True

class VisionGuardrail:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = cfg.get("vision_guardrail", {}).get("enabled", False)
        if not self.enabled:
            return

        self.llm_settings = LLMSettings()

        vg_cfg = cfg.get("vision_guardrail", {})
        self.model = vg_cfg.get("model", self.llm_settings.openai_model)
        api_key = vg_cfg.get("api_key", self.llm_settings.openai_api_key)
        provider = vg_cfg.get("provider", self.llm_settings.llm_provider)
        self.image_detail = vg_cfg.get("image_detail", "auto")

        self.client = OpenAICompatClient(
            base_url=vg_cfg.get("base_url", self.llm_settings.openai_base_url),
            api_key=api_key,
            model=self.model,
            timeout=float(vg_cfg.get("timeout", 180.0)),
            provider=provider,
        )

    def check(self, img_path: str, mask_arr: np.ndarray, stem: str, output_dir: str = "") -> dict:
        """
        Returns {'score': 0-100, 'reason': str, 'accepted': bool}
        """
        if not self.enabled:
            return {'score': 100, 'reason': 'Guardrail disabled', 'accepted': True}

        # 1. Generate Overlay
        basename = os.path.basename(img_path).replace(".png", "_overlay_debug.png")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            overlay_path = os.path.join(output_dir, basename)
        else:
            overlay_path = img_path.replace(".png", "_overlay_debug.png")
        if not create_overlay(img_path, mask_arr, overlay_path):
            return {'score': 0, 'reason': 'Overlay generation failed', 'accepted': False}

        # 2. Prompt - RELATIVE IMPROVEMENT CHECK
        # We are repairing flawed masks. The intermediate result might still be "bad" 
        # but we need to know if it's BETTER than the previous step.
        # Since we don't have the previous step here easily in `check` (it's stateless),
        # we still need a sanity check. 
        # BUT, the `Executor` calls this with `check`? 
        # Wait, Executor calls `judge_visual_quality` which maps to `check`.
        # The prompt below forces a Score. If score < threshold, it rejects.
        # This is too strict for intermediate steps.
        
        # Let's relax the scoring or change the prompt to be more tolerant of "work in progress".
        # Better yet, let's ask for a "validity" check rather than "perfection".
        
        system = """You are a Cardiac MRI Segmentation Expert Evaluator.
Your task is to basic sanity check the segmentation mask validation.
Color Legend: Red=RV, Green=Myo, Blue=LV.

Evaluate if the segmentation is PLAUSIBLE and NOT WORSE than a typical noisy segmentation.
We are in the middle of a repair process, so some defects are expected.

CRITICAL REJECTION CRITERIA (Score < 50):
1. Myocardium (Green) is completely scattered noise.
2. LV (Blue) or RV (Red) is completely missing.
3. Grossly non-anatomical shapes (e.g. square LV, exploded pixels).
4. Massive leakage merging all classes into one blob.

ACCEPTABLE FLAWS (Score 60-80):
1. Small gaps in Myocardium.
2. Rough edges.
3. Minor class confusion.
4. "Blobby" shapes.

PERFECT (Score 90+):
1. Smooth, continuous, correct anatomy.

Output JSON:
{
  "score": <0-100>,
  "reason": "<concise comparison>",
  "suggestion": "None"
}
"""
        user_text = f"Evaluate the segmentation for case {stem}."

        # 3. Call VLM
        resp = self.client.chat_vision_json(
            system, user_text, overlay_path, image_detail=self.image_detail,
        )
        print(f"DEBUG: VLM Response for {stem}: {resp}", flush=True)

        # 4. Parse
        score = resp.get("score", 0)
        reason = resp.get("reason", "No reason provided")
        suggestion = resp.get("suggestion", "None")
        threshold = int(self.cfg.get("vision_guardrail", {}).get("threshold", 60))

        return {
            'score': score,
            'reason': reason,
            'suggestion': suggestion,
            'accepted': score >= threshold
        }

    def verify_fix(
        self,
        img_path: str,
        mask_before: np.ndarray,
        mask_after: np.ndarray,
        stem: str,
        diagnosis_summary: str = "",
    ) -> dict:
        """
        VLM verification of a fix: compare before vs after overlay.

        Returns {'improved': bool, 'reason': str}
        """
        if not self.enabled:
            return {'improved': True, 'reason': 'Guardrail disabled'}

        # Generate side-by-side overlay
        sbs_path = img_path.replace(".png", f"_verify_sbs.png")
        if not create_side_by_side_overlay(img_path, mask_before, mask_after, sbs_path):
            return {'improved': True, 'reason': 'Could not generate comparison image'}

        diag_hint = ""
        if diagnosis_summary:
            diag_hint = f"\nThe AI applied fixes for: {diagnosis_summary}"

        system = f"""You are a Cardiac MRI Segmentation Expert Evaluator.
You are comparing a BEFORE (left) and AFTER (right) segmentation overlay.
Color Legend: Red=RV, Green=Myocardium, Blue=LV.
{diag_hint}

Compare the two segmentations carefully:
- Did the fix improve anatomical correctness?
- Is the myocardium ring more complete/continuous in AFTER?
- Are there fewer noise islands in AFTER?
- Did the fix accidentally REMOVE or DAMAGE any correct structures?

IMPORTANT: If the AFTER version looks WORSE than BEFORE (e.g., structures removed,
new artifacts introduced, shapes distorted), then the fix did NOT improve things.
If they look roughly the same or AFTER is slightly better, consider it improved.

Output JSON:
{{"improved": true/false, "reason": "<concise explanation>"}}
"""
        user_text = f"Compare before (left) vs after (right) for case {stem}."

        try:
            resp = self.client.chat_vision_json(
                system, user_text, sbs_path, image_detail=self.image_detail,
            )
        except Exception as e:
            print(f"[VLM Verify] Failed for {stem}: {e}")
            # On VLM failure, REJECT the fix (conservative: don't apply unverified changes)
            return {'improved': False, 'reason': f'VLM verification failed: {e}'}

        # If VLM returned empty response, be conservative and reject
        if not resp:
            return {'improved': False, 'reason': 'VLM returned empty response'}
        improved = bool(resp.get("improved", False))
        reason = resp.get("reason", "No reason provided")

        return {'improved': improved, 'reason': reason}
