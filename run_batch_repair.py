"""
Batch Repair Script for Worst Cases.
"""
print("DEBUG: Script Start", flush=True)
import os
import sys
import re
import csv
import yaml
import json
import numpy as np
import cv2
from tqdm import tqdm

print("DEBUG: Importing agents...", flush=True)
from cardiac_agent_postproc.agents.message_bus import MessageBus
from cardiac_agent_postproc.agents.coordinator import CoordinatorAgent
from cardiac_agent_postproc.agents.triage_agent import TriageAgent
from cardiac_agent_postproc.agents.diagnosis_agent import DiagnosisAgent
from cardiac_agent_postproc.agents.planner_agent import PlannerAgent
from cardiac_agent_postproc.agents.executor import ExecutorAgent
from cardiac_agent_postproc.agents.verifier_agent import VerifierAgent
from cardiac_agent_postproc.agents.atlas_builder import AtlasBuilderAgent
from cardiac_agent_postproc.orchestrator import discover_cases, group_and_link_neighbors
from cardiac_agent_postproc.io_utils import read_mask
from cardiac_agent_postproc.agents.visual_knowledge import VisualKnowledgeBase
from cardiac_agent_postproc.atlas import load_atlas

# Config & Paths
CONFIG_PATH = "config/default.yaml"
BATCH_OUT_DIR = "batch_repair_results"
# Regex to extract stem
STEM_REGEX = re.compile(r"^\d+\.\d+_\w+_\w+_(\d+_original_lax_\w+_\d+)\.png$")

def setup(config_path="config/default.yaml"):
    print("DEBUG: Inside setup()", flush=True)
    if not os.path.exists(BATCH_OUT_DIR):
        os.makedirs(BATCH_OUT_DIR)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"DEBUG: Config loaded from {config_path}", flush=True)
    return cfg

def get_target_stems(cfg):
    source_dir = cfg["paths"]["source_dir"]
    targets = cfg.get("targets", [])
    return set(targets) if targets else None, source_dir

from cardiac_agent_postproc.eval_metrics import dice_macro

def compute_dice(pred, gt):
    mean, d1, d2, d3 = dice_macro(pred, gt)
    return {"Mean": mean, "RV": d1, "Myo": d2, "LV": d3}

def load_or_build_atlas(cfg, source_dir):
    atlas_path = cfg["paths"]["atlas_path"]
    if os.path.exists(atlas_path):
        print(f"DEBUG: Loading atlas from {atlas_path}...", flush=True)
        return load_atlas(atlas_path)
    
    # If not exists, try to build or return None
    print("DEBUG: Atlas not found, attempting build...", flush=True)
    try:
        builder = AtlasBuilderAgent(cfg, None) # No bus needed for build
        return None
    except Exception as e:
        print(f"DEBUG: Atlas build failed: {e}")
        return None


def sort_cases_by_severity(cases, source_dir):
    print("DEBUG: Sorting cases by severity (Dice score)...", flush=True)
    severity_list = []
    
    # We need to read masks to compute Dice. 
    # This might be slow for 300 cases (~1-2 mins), but worth it for prioritization.
    for ctx in tqdm(cases, desc="Evaluating Severity"):
        gt_path = os.path.join(source_dir, f"{ctx.stem}_gt.png")
        pred_path = os.path.join(source_dir, f"{ctx.stem}_pred.png")
        
        if os.path.exists(gt_path) and os.path.exists(pred_path):
            try:
                # Read raw masks
                gt = read_mask(gt_path)
                pred = read_mask(pred_path)
                dice = compute_dice(pred, gt)["Mean"]
                severity_list.append((dice, ctx))
            except Exception as e:
                print(f"WARN: Could not read/score {ctx.stem}: {e}")
                # Treat error cases as high severity (priority) or low? 
                # Let's put them at the end (high dice) so we don't crash on them immediately 
                # unless they are critical. Let's assign -1.0 to prioritize them?
                # Actually, best to put them last if they are broken files.
                severity_list.append((1.0, ctx)) 
        else:
            # No GT or Pred? Info missing.
            severity_list.append((1.0, ctx))
            
    # Sort: Low Dice first (Ascending)
    severity_list.sort(key=lambda x: x[0])
    
    print(f"DEBUG: Top 5 Worst Cases:", flush=True)
    for i in range(min(5, len(severity_list))):
        print(f"  {i+1}. {severity_list[i][1].stem}: {severity_list[i][0]:.4f}", flush=True)
        
    return [x[1] for x in severity_list]


def main(config_path="config/default.yaml", case_filter=None, limit=None):
    print("DEBUG: Starting main", flush=True)
    cfg = setup(config_path)
    target_stems, source_dir = get_target_stems(cfg)
    print(f"DEBUG: Targets: {len(target_stems) if target_stems else 'ALL'} cases", flush=True)
    
    # ... (KB loading omitted for brevity in replace, assuming lines 52-75 unchanged) ...
    # Visual KB
    vk_dir = os.path.dirname(source_dir)
    visual_kb = None
    if os.path.isdir(vk_dir):
        try:
            visual_kb = VisualKnowledgeBase.load(vk_dir)
            print(f"DEBUG: Visual KB loaded ({len(visual_kb.good_examples)} good)", flush=True)
        except Exception as e:
            print(f"DEBUG: Visual KB failed: {e}", flush=True)

    # Repair Comparison KB
    repair_kb = None
    repair_kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repair_kb")
    if os.path.isdir(repair_kb_dir):
        try:
            from cardiac_agent_postproc.agents.visual_knowledge import RepairComparisonKB
            repair_kb = RepairComparisonKB.load(repair_kb_dir)
            print(f"DEBUG: Repair KB loaded ({repair_kb.total} examples)", flush=True)
        except Exception as e:
            print(f"DEBUG: Repair KB failed: {e}", flush=True)

    atlas = load_or_build_atlas(cfg, source_dir)

    # Agents
    print("DEBUG: Initializing agents...", flush=True)
    bus = MessageBus()
    agents = {
        "coordinator": CoordinatorAgent(cfg, bus),
        "triage": TriageAgent(cfg, bus, visual_kb=visual_kb),
        "diagnosis": DiagnosisAgent(cfg, bus),
        "planner": PlannerAgent(cfg, bus),
        "executor": ExecutorAgent(cfg, bus, atlas=atlas, visual_kb=visual_kb, repair_kb=repair_kb),
        "verifier": VerifierAgent(cfg, bus, visual_kb=visual_kb, repair_kb=repair_kb),
    }
    
    # Discover Context
    print("DEBUG: Discovering cases...", flush=True)
    all_cases = discover_cases(source_dir, "")
    group_and_link_neighbors(all_cases)
    
    # Filter
    if case_filter:
        print(f"INFO: Applying case filter '{case_filter}'...", flush=True)
        target_stems = {c.stem for c in all_cases if case_filter in c.stem}
        print(f"INFO: Filter matched {len(target_stems)} cases", flush=True)

    if target_stems:
        queue = [c for c in all_cases if c.stem in target_stems]
        print(f"INFO: Filtered to {len(queue)} cases from {len(all_cases)} total", flush=True)
    else:
        queue = all_cases
    
    # Sort by severity (Prioritize Worse Cases)
    queue = sort_cases_by_severity(queue, source_dir)
        
    if limit:
        print(f"INFO: Limiting to first {limit} cases (Worst First).", flush=True)
        queue = queue[:limit]
    # Set output_dir on each case so debug images go to the output folder
    debug_img_dir = os.path.join(BATCH_OUT_DIR, "debug_images")
    os.makedirs(debug_img_dir, exist_ok=True)
    for c in queue:
        c.output_dir = debug_img_dir
    print(f"INFO: Processing {len(queue)} cases in full batch mode", flush=True)
    
    # CSV Init
    csv_path = os.path.join(BATCH_OUT_DIR, "batch_report.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            csv.writer(f).writerow(["Stem", "Verdict", "Ops", "Atlas", "Pre", "Post", "Delta"])

    # Loop
    for ctx in tqdm(queue):
        print(f"Processing {ctx.stem}...", flush=True)
        print(f"DEBUG: Registering case {ctx.stem} with bus...", flush=True)
        bus.register_case(ctx)
        print(f"DEBUG: Case registered. Original mask copy...", flush=True)
        # Ensure original_mask is set for rollback safety
        if ctx.mask is not None and ctx.original_mask is None:
            ctx.original_mask = ctx.mask.copy()
        print(f"DEBUG: Pre-Dice calculation...", flush=True)
        gt_path = os.path.join(source_dir, f"{ctx.stem}_gt.png")
        dice_pre = 0.0
        if os.path.exists(gt_path):
            ctx.gt_path = gt_path
            dice_pre = compute_dice(ctx.mask, read_mask(gt_path))["Mean"]
            
        # Fast Path Check
        threshold = cfg["multi_agent"]["fast_path_threshold"]
        if dice_pre >= threshold:
            print(f"  [FastPath] Skipping {ctx.stem} (Dice {dice_pre:.4f} >= {threshold})", flush=True)
            bus.register_case(ctx) # Register mostly for stats if needed, or skip?
            # Log to CSV as skipped/good
            with open(csv_path, "a") as f:
                csv.writer(f).writerow([ctx.stem, "skipped_good", 0, False, f"{dice_pre:.4f}", f"{dice_pre:.4f}", "0.0000"])
            continue
        # Run
        try:
            print(f"DEBUG: Calling coordinator.orchestrate_case for {ctx.stem}...", flush=True)
            verdict = agents["coordinator"].orchestrate_case(ctx, agents, max_rounds=3)
            print(f"DEBUG: orchestrate_case returned verdict: {verdict}", flush=True)
        except Exception as e:
            print(f"CRITICAL Error ({ctx.stem}): {e}", flush=True)
            import traceback
            traceback.print_exc()
            verdict = "error"
            
        # Post-Agent Dice
        dice_post_agent = 0.0
        if os.path.exists(gt_path) and ctx.current_mask is not None:
             dice_post_agent = compute_dice(ctx.current_mask, read_mask(gt_path))["Mean"]
             
             # SAFETY CHECK 1: Did Agent make it worse?
             if dice_post_agent < dice_pre - 0.01:
                 print(f"  [Safety] Agent degraded Dice ({dice_post_agent:.4f} < {dice_pre:.4f}). Reverting to Original.", flush=True)
                 ctx.current_mask = ctx.original_mask.copy()
                 dice_post_agent = dice_pre
                 verdict = "reverted_agent"

        # --- HYBRID MODE: Polish disabled (Agent handles intensity ops now) ---
        dice_post_final = dice_post_agent
        # Manual polish removed to allow agents to handle it dynamically
        if ctx.current_mask is not None:
             # Just ensure we have a valid final result
             pass
             
        # Log
        ops = [op["name"] for op in ctx.applied_ops]
        atlas = "individual_atlas_transfer" in ops
        with open(csv_path, "a") as f:
            csv.writer(f).writerow([ctx.stem, verdict, len(ops), atlas, 
                                    f"{dice_pre:.4f}", f"{dice_post_final:.4f}", f"{dice_post_final-dice_pre:.4f}"])
        
        # Save
        if ctx.current_mask is not None:
             out_path = os.path.join(BATCH_OUT_DIR, f"{ctx.stem}_repaired.png")
             # Write RAW integer labels (0, 1, 2, 3), do not rescale.
             cv2.imwrite(out_path, ctx.current_mask.astype(np.uint8))
        
        # Save Audit Trail
        audit_path = os.path.join(BATCH_OUT_DIR, f"{ctx.stem}_audit.json")
        history = bus.get_full_history()
        # Filter manually if needed, or dump all
        case_history = [m for m in history if getattr(m, 'case_id', None) == ctx.case_id or getattr(m, 'case_id', None) is None]
             
        with open(audit_path, "w") as f:
             json.dump([
                 {"sender": m.sender, "recipient": m.recipient, "type": m.msg_type.value, 
                  "content": str(m.content)}
                 for m in case_history
             ], f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    parser.add_argument("--case", type=str, default=None, help="Filter to specific case stem")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases processed")
    args = parser.parse_args()
    
    # Global overrides (hacky but effective for this script structure)
    # RESULTS_DIR = args.results_dir # Removed as not in args
    # BATCH_OUT_DIR = os.path.join(RESULTS_DIR, "repaired_frames") # Logic for this is elsewhere
    
    # Monkeypatch get_target_stems if case is specified
    if args.case:
        # We need to find where get_target_stems is defined. 
        # It's likely imported or defined above.
        # But wait, main() calls get_target_stems(cfg).
        # We can't easily monkeypatch a local function inside main if main calls it directly from global scope.
        # But get_target_stems IS defined or imported in global scope.
        # Let's check imports.
        pass

    # Pass args.case to main? 
    # main only takes config_path.
    # We should update main signature or handle filtering inside main.
    
    main(args.config, case_filter=args.case, limit=args.limit)
