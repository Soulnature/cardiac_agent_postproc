
import os
import cv2
import numpy as np
import base64
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

    # RV - Red
    overlay[mask_arr == 1] = [0, 0, 255]
    # Myo - Green
    overlay[mask_arr == 2] = [0, 255, 0]
    # LV - Blue
    overlay[mask_arr == 3] = [255, 0, 0]

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

        self.client = OpenAICompatClient(
            base_url=self.llm_settings.openai_base_url,
            api_key=api_key,
            model=self.model,
            timeout=180.0,
        )

    def check(self, img_path: str, mask_arr: np.ndarray, stem: str) -> dict:
        """
        Returns {'score': 0-100, 'reason': str, 'accepted': bool}
        """
        if not self.enabled:
            return {'score': 100, 'reason': 'Guardrail disabled', 'accepted': True}

        # 1. Generate Overlay
        overlay_path = img_path.replace(".png", "_overlay_debug.png")
        if not create_overlay(img_path, mask_arr, overlay_path):
            return {'score': 0, 'reason': 'Overlay generation failed', 'accepted': False}

        # 2. Prompt
        system = """You are a Cardiac MRI Segmentation Expert Evaluator.
Your task is to CRITICALLY evaluate the quality of the segmentation mask overlay.
Color Legend:
- Red: Right Ventricle (RV)
- Green: Myocardium (Myo)
- Blue: Left Ventricle (LV)

STRICT Scoring Criteria:

Score 90-100 (Perfect):
1. Blue LV is perfectly elliptical/circular.
2. Green Myocardium is a CONTINUOUS, UNIFORM ring around LV.
3. No gaps, no leaks, no noise.

Score 60-80 (Acceptable but Flawed):
1. Myocardium has variable thickness or looks jagged.
2. RV shape is irregular.
3. Minor edge roughness.

Score < 50 (REJECT - Bad):
1. ANY disconnection in the Green Myocardium ring (Check carefully!).
2. Missing LV or RV.
3. "Islands" of noise in background.
4. "Exploded" or non-anatomical shapes.
5. If the Myocardium is extremely thin or broken, score 0.

Output JSON:
{
  "score": <0-100>,
  "reason": "<concise explanation focusing on defects>",
  "suggestion": "<actionable fix>"
}
"""
        user_text = f"Evaluate the segmentation for case {stem}."

        # 3. Call VLM
        resp = self.client.chat_vision_json(system, user_text, overlay_path)

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
            resp = self.client.chat_vision_json(system, user_text, sbs_path)
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
