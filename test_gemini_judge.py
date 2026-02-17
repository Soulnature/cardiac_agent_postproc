import os
import json
from google import genai
from google.genai import types

# ── CONFIG ──
# User provided key
API_KEY = "AIzaSyC7Lvp9jdP7BuD2si5Ql8V7B5SFCp0VAVc"
MODEL_NAME = "gemini-2.5-flash"

# Case 335 (The tricky one: +1261 added, -92 removed)
IMG_PATH = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/repaired_frames/debug_images/335_original_lax_4c_009_img_verify_diff.png"

def test_gemini_judge():
    if not os.path.exists(IMG_PATH):
        print(f"Error: Image not found at {IMG_PATH}")
        return

    print(f"Testing Gemini ({MODEL_NAME}) on Case 335...")
    print(f"Image: {IMG_PATH}")

    client = genai.Client(api_key=API_KEY)

    # Prompt: similar to original semantic prompt (without quantitative hand-holding)
    # We want to see if Gemini can figure it out visually.
    prompt = (
        "You are a cardiac MRI segmentation repair quality judge.\n\n"
        "TASK: Determine if the repair IMPROVED or DEGRADED the segmentation based on pixel statistics and visual evidence.\n\n"
        "The image has 3 panels:\n"
        "  LEFT = BEFORE repair\n"
        "  MIDDLE = AFTER repair\n"
        "  RIGHT = DIFF panel showing pixel-level changes.\n\n"
        "  +N = pixels ADDED (green)\n"
        "  -N = pixels REMOVED (red)\n"
        "  ~N = pixels CLASS CHANGED (yellow)\n\n"
        "Respond ONLY in JSON:\n"
        '{"verdict": "improved"|"degraded"|"neutral", '
        '"confidence": 0.0-1.0, '
        '"reasoning": "visual explanation"}'
    )

    try:
        # Load image
        img_part = types.Part.from_bytes(
            data=open(IMG_PATH, "rb").read(),
            mime_type="image/png"
        )
        
        print("Sending request to Gemini...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, img_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        if response.text:
            print("\n" + "="*40)
            print("GEMINI RESPONSE:")
            print("="*40)
            print(response.text)
            
            # Parse
            try:
                data = json.loads(response.text)
                verdict = data.get("verdict", "unknown").lower()
                print(f"\nVerdict: {verdict.upper()}")
                if verdict == "improved":
                    print("SUCCESS! Gemini identified the improvement correctly.")
                else:
                    print("FAILURE. Gemini failed to identify improvement (Semantic Bias?).")
            except:
                print("Could not parse JSON.")
        else:
            print("Empty response.")

    except Exception as e:
        print(f"Error calling Gemini: {e}")

if __name__ == "__main__":
    test_gemini_judge()
