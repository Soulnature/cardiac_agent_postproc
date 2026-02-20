
import os
import sys
import yaml
import glob
import re
from cardiac_agent_postproc.agents.message_bus import MessageBus
from cardiac_agent_postproc.agents.triage_agent import TriageAgent
from cardiac_agent_postproc.io_utils import read_mask
import cv2
import pandas as pd
from tqdm import tqdm

# Add parent dir to path
# We are in scripts/ so we need to go up one level
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Also print sys.path to be sure
print(f"DEBUG: sys.path: {sys.path}", flush=True)

from cardiac_agent_postproc.agents.message_bus import CaseContext

CONFIG_PATH = "config/default.yaml"
BEST_CASES_DIR = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/best_cases"
OUTPUT_CSV = "batch_eval_results/best_cases_audit.csv"

# Regex to extract stem from the filename like "0.959_4C_Best_274_original_lax_4c_009.png"
# We need "274_original_lax_4c_009"
STEM_REGEX = re.compile(r"Best_(\d+_original_lax_\w+_\d+)\.png")

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_best_cases_stems():
    stems = []
    files = glob.glob(os.path.join(BEST_CASES_DIR, "*.png"))
    for f in files:
        fname = os.path.basename(f)
        match = STEM_REGEX.search(fname)
        if match:
            stems.append(match.group(1))
        else:
            print(f"Warning: Could not parse stem from {fname}")
    return stems

def main():
    print("Loading config...")
    cfg = load_config()
    
    print(f"Discovering best cases in {BEST_CASES_DIR}...")
    stems = get_best_cases_stems()
    print(f"Found {len(stems)} best cases: {stems}")
    
    # Initialize Triage Agent
    # We use a real bus to capture messages if needed, but TriageAgent writes to ctx too
    bus = MessageBus()
    # Ensure VLM is enabled in config for this test if not already
    cfg["diagnosis"]["vlm_enabled"] = True 
    
    agent = TriageAgent(cfg, bus)
    
    # We need to find the original images/masks for these stems.
    # They should be in the source_dir defined in config.
    source_dir = cfg["paths"]["source_dir"]
    
    results = []
    
    for stem in tqdm(stems, desc="Auditing Best Cases"):
        # Construct paths
        # Assuming standard naming convention in source_dir
        pred_path = os.path.join(source_dir, f"{stem}_pred.png")
        gt_path = os.path.join(source_dir, f"{stem}_gt.png")
        img_path = os.path.join(source_dir, f"{stem}.png")
        
        if not os.path.exists(pred_path):
            print(f"Error: Pred file not found for {stem} at {pred_path}")
            continue
            
        ctx = CaseContext(
            case_id=stem,
            stem=stem,
            img_path=img_path,
            pred_path=pred_path,
            gt_path=gt_path,
            output_dir=os.path.join("batch_eval_results", "best_cases_debug")
        )
        
        os.makedirs(ctx.output_dir, exist_ok=True)
        
        try:
            ctx.mask = read_mask(pred_path)
            ctx.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if "2ch" in stem: ctx.view_type = "2ch"
            elif "3ch" in stem: ctx.view_type = "3ch"
            else: ctx.view_type = "4ch"
            
            print(f"Triaging {stem}...")
            agent.process_case(ctx)
            
            # Result
            print(f"  -> {ctx.triage_category} (Score: {ctx.triage_score:.4f})")
            
            results.append({
                "Stem": stem,
                "TriageCategory": ctx.triage_category,
                "TriageScore": ctx.triage_score,
                "Issues": ctx.triage_issues
            })
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Best cases audit saved to {OUTPUT_CSV}")
    
    # Quick Summary
    good_count = len(df[df["TriageCategory"] == "good"])
    print(f"\nSummary:")
    print(f"Total: {len(df)}")
    print(f"Good: {good_count}")
    print(f"Needs Fix: {len(df) - good_count}")

if __name__ == "__main__":
    main()
