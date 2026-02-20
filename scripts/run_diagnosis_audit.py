
"""
Audit script for Diagnosis Agent.
Runs DiagnosisAgent on a stratified sample of 30 cases (10 Low, 10 Mid, 10 High Dice).
Captures the output diagnoses and saves to CSV for analysis.
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
import time

# Ensure we can import from cardiac_agent_postproc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cardiac_agent_postproc.agents.diagnosis_agent import DiagnosisAgent
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
OUTPUT_CSV = "batch_eval_results/diagnosis_audit_results.csv"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def get_all_cases(metrics_path):
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        return None
    
    df = pd.read_csv(metrics_path)
    df["GroupLabel"] = "All" # No stratification labels needed if running all, or we can keep group IDs? 
    # Use Patient Group (first 3 digits) as label?
    # The existing script uses "GroupLabel" for the plot hue.
    # Let's use the "Group" column from the CSV if it exists, or create it.
    if "Group" in df.columns:
        df["GroupLabel"] = df["Group"]
    else:
        df["GroupLabel"] = "Unknown"
        
    return df

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0, help="Chunk index (0-based)")
    parser.add_argument("--total_chunks", type=int, default=1, help="Total number of chunks")
    args = parser.parse_args()

    print("Loading config...")
    cfg = load_config()
    
    # Ensure VLM is enabled
    if not cfg["diagnosis"].get("vlm_enabled", False):
        print("Enabling VLM for audit...")
        cfg["diagnosis"]["vlm_enabled"] = True
        
    print("Loading metrics...")
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
    agent = DiagnosisAgent(cfg, bus)
    
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
                
            agent.process_case(ctx)
            
            diagnoses = ctx.diagnoses
            
            num_diagnoses = len(diagnoses)
            max_severity = "none"
            top_issue = "none"
            top_op = "none"
            
            severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
            current_max_rank = 0
            
            for d in diagnoses:
                sev = d.get("severity", "medium").lower()
                rank = severity_rank.get(sev, 1)
                
                if rank > current_max_rank:
                    current_max_rank = rank
                    max_severity = sev
                    top_issue = d.get("issue", "unknown")
                    top_op = d.get("operation", "unknown")
            
            res_row = {
                "Stem": stem,
                "GroupLabel": row["GroupLabel"],
                "Dice_Mean": dice_mean,
                "HD95_Mean": hd95_mean,
                "NumDiagnoses": num_diagnoses,
                "MaxSeverity": max_severity,
                "TopIssue": top_issue,
                "TopOp": top_op,
                "Diagnoses": json.dumps(diagnoses) 
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
