
"""
Audit script for Triage Agent.
Runs TriageAgent on ALL cases to classify them as 'good' or 'needs_fix'.
Saves Triage Decision + Ground Truth Metrics (Dice, HD95) for analysis.
"""
import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from dataclasses import dataclass, field
import argparse
import time

# Ensure we can import from cardiac_agent_postproc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cardiac_agent_postproc.agents.triage_agent import TriageAgent
from cardiac_agent_postproc.agents.message_bus import CaseContext, MessageType, AgentMessage
from cardiac_agent_postproc.io_utils import read_mask

# --- Mocks ---
class MockBus:
    def __init__(self):
        self.inbox = []
        self.cases = {}

    def send(self, msg):
        self.inbox.append(msg)

    def register_case(self, ctx):
        self.cases[ctx.case_id] = ctx

    def register_agent(self, name):
        pass

# --- Setup ---
CONFIG_PATH = "config/default.yaml"
METRICS_PATH = "batch_eval_results/batch_evaluation_metrics.csv"
OUTPUT_CSV = "batch_eval_results/triage_audit_results.csv"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_all_cases(metrics_path):
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return None
    
    df = pd.read_csv(metrics_path)
    # Ensure GroupLabel exists for later analysis
    if "Group" in df.columns:
        df["GroupLabel"] = df["Group"]
    else:
        df["GroupLabel"] = "Unknown"
        
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0, help="Chunk index (0-based)")
    parser.add_argument("--total_chunks", type=int, default=1, help="Total number of chunks")
    args = parser.parse_args()

    print("Loading config...")
    cfg = load_config()
    
    # Ensure VLM is enabled
    # We want to audit the full capability including VLM
    if "diagnosis" in cfg:
        cfg["diagnosis"]["vlm_enabled"] = True
    if "multi_agent" in cfg:
        cfg["multi_agent"]["vlm_enabled"] = True
    # Also set root level just in case
    cfg["vlm_enabled"] = True
        
    print("Loading metrics...", flush=True)
    all_df = get_all_cases(METRICS_PATH)
    if all_df is None:
        return
    
    # Chunking
    total_cases = len(all_df)
    chunk_size = int(np.ceil(total_cases / args.total_chunks))
    start_idx = args.chunk * chunk_size
    end_idx = min((args.chunk + 1) * chunk_size, total_cases)
    
    sample_df = all_df.iloc[start_idx:end_idx]
    
    print(f"Selected {len(sample_df)} cases for audit (Chunk {args.chunk}/{args.total_chunks}: {start_idx}-{end_idx}).")
    
    # Output file specific to chunk
    chunk_output_csv = OUTPUT_CSV.replace(".csv", f"_{args.chunk}.csv")
    
    # Initialize Agent
    bus = MockBus()
    # TriageAgent needs 'triage' config, usually part of 'agents' in cfg
    # It inherits BaseAgent which reads cfg
    agent = TriageAgent(cfg, bus)
    
    results = []
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc=f"Chunk {args.chunk}"):
        stem = row["Stem"]
        img_path = row["Image Path"]
        gt_path = row["GT Path"]
        pred_path = row["Pred Path"]
        dice_mean = row.get("Dice_Mean", 0.0)
        hd95_mean = row.get("HD95_Mean", -1.0)
        
        ctx = CaseContext(
            case_id=stem,
            stem=stem,
            img_path=img_path,
            pred_path=pred_path,
            gt_path=gt_path,
            output_dir=os.path.join("batch_eval_results", "audit_debug")
        )
        
        try:
            ctx.mask = read_mask(pred_path)
            ctx.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if "2ch" in stem: ctx.view_type = "2ch"
            elif "3ch" in stem: ctx.view_type = "3ch"
            else: ctx.view_type = "4ch"
                
            # Run Triage Agent
            agent.process_case(ctx)
            
            # Retrieve Result from Context and Message History
            # TriageAgent updates ctx.triage_category, ctx.triage_score, ctx.triage_issues
            # AND ctx.vlm_quality (newly added)
            
            vlm_quality = getattr(ctx, "vlm_quality", "N/A")
            vlm_data = getattr(ctx, "vlm_verdict_data", {}) or {}
            vlm_conf = vlm_data.get("confidence", 0.0)

            res_row = {
                "Stem": stem,
                "GroupLabel": row["GroupLabel"],
                "Dice_Mean": dice_mean,
                "HD95_Mean": hd95_mean,
                "TriageCategory": ctx.triage_category,
                "TriageScore": ctx.triage_score,
                "VLM_Quality": vlm_quality,
                "VLM_Confidence": vlm_conf,
                "TriageIssues": json.dumps(ctx.triage_issues)
            }
            results.append(res_row)
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            
    # Save Chunk Results
    out_df = pd.DataFrame(results)
    out_df.to_csv(chunk_output_csv, index=False)
    print(f"Audit chunk results saved to {chunk_output_csv}")

if __name__ == "__main__":
    main()
