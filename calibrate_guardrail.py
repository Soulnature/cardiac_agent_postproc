
import os
import glob
import pandas as pd
from tqdm import tqdm
from cardiac_agent_postproc.vlm_guardrail import VisionGuardrail
from cardiac_agent_postproc.api_client import OpenAICompatClient
from cardiac_agent_postproc.settings import LLMSettings
import yaml

def calibrate():
    # Load Config
    with open("config/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Initialize VLM Client manually (since we bypass full Guardrail logic)
    # Actually, we can use VisionGuardrail helper if we hack it, 
    # but easier to just use the client directly to send pre-made overlay.
    
    llm_settings = LLMSettings()
    model = cfg.get("vision_guardrail", {}).get("model", "nvidia/vila-b1-2.8b-sft")
    
    client = OpenAICompatClient(
        base_url=llm_settings.openai_base_url,
        api_key=llm_settings.openai_api_key,
        model=model,
        timeout=180.0
    )
    
    # Prompt (Same as VLM Guardrail)
    SYSTEM_PROMPT = """You are a Senior Cardiac MRI Radiologist. Evaluate the segmentation quality.
SCORING (0-100):
- 90-100: Perfect. Smooth, uniform, anatomical.
- 75-89: Good. Minor roughness or slight thickness variation. ACCEPTABLE.
- 60-74: Fair. Visible jagged edges, non-uniform thickness. REJECT.
- <60: Poor. Disconnection, holes, massive islands. REJECT.

CRITICAL CHECKS:
1. Is the Myocardium (Green) a Continuous Ring? (Any break = Score 0).
2. Is the Wall Thickness Uniform? (Lumpy/Wobbly = Score < 70).
3. Are Edges Smooth? (Jagged/Pixelated = Score < 75).

Return JSON:
{
  "score": <int>,
  "reason": "<string>",
  "suggestion": "<keywords: smooth, dilate, remove islands, connect>"
}"""

    input_dir = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2"
    
    results = []
    
    # Process Best and Worst
    for category in ["best_cases", "worst_cases"]:
        folder = os.path.join(input_dir, category)
        files = glob.glob(os.path.join(folder, "*.png"))[:3] # Limit to 3 for fast check
        
        print(f"Processing {category}: {len(files)} files...")
        
        for img_path in tqdm(files):
            stem = os.path.basename(img_path)
            
            user_text = f"Evaluate this cardiac segmentation overlay. Stem: {stem}"
           
            print(f"[{category}] Sending {stem}...") 
            try:
                resp = client.chat_vision_json(SYSTEM_PROMPT, user_text, img_path)
                score = resp.get("score", 0)
                reason = resp.get("reason", "Error")
                print(f"[{category}] {stem} -> Score: {score}")
                resp = client.chat_vision_json(SYSTEM_PROMPT, user_text, img_path)
                score = resp.get("score", 0)
                reason = resp.get("reason", "Error")
                
                results.append({
                    "category": category,
                    "filename": stem,
                    "kimi_score": score,
                    "kimi_reason": reason,
                    "kimi_suggestion": resp.get("suggestion", "")
                })
            except Exception as e:
                print(f"Error on {stem}: {e}")
                results.append({
                    "category": category,
                    "filename": stem,
                    "kimi_score": -1,
                    "kimi_reason": str(e),
                    "kimi_suggestion": "Error"
                })

    # Save
    df = pd.DataFrame(results)
    df.to_csv("calibration_log.csv", index=False)
    print("Calibration complete. Saved to calibration_log.csv")
    
    # Summary
    print("\n=== Calibration Summary ===")
    print(df.groupby("category")["kimi_score"].describe())

if __name__ == "__main__":
    calibrate()
