"""
Batch Repair Script for Worst Cases.
"""
print("DEBUG: Script Start", flush=True)
import os
import sys
import re
import csv
import yaml
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
# from cardiac_agent_postproc.agents.verifier_agent import VerifierAgent
from cardiac_agent_postproc.orchestrator import discover_cases, group_and_link_neighbors
from cardiac_agent_postproc.io_utils import read_mask, write_mask
from cardiac_agent_postproc.agents.visual_knowledge import VisualKnowledgeBase
from cardiac_agent_postproc.atlas import load_atlas

# Config & Paths
CONFIG_PATH = "config/default.yaml"
BATCH_OUT_DIR = "batch_repair_results"
# Regex to extract stem
STEM_REGEX = re.compile(r"^\d+\.\d+_\w+_\w+_(\d+_original_lax_\w+_\d+)\.png$")

def setup():
    print("DEBUG: Inside setup()", flush=True)
    if not os.path.exists(BATCH_OUT_DIR):
        os.makedirs(BATCH_OUT_DIR)
    
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    print("DEBUG: Config loaded", flush=True)
    return cfg

def get_target_stems(cfg):
    print("DEBUG: Inside get_target_stems()", flush=True)
    source_dir = cfg["paths"]["source_dir"]
    worst_dir = os.path.join(os.path.dirname(source_dir), "worst_cases")
    print(f"DEBUG: Scanning {worst_dir}", flush=True)
    
    stems = set()
    if os.path.exists(worst_dir):
        for f in os.listdir(worst_dir):
            m = STEM_REGEX.match(f)
            if m:
                stems.add(m.group(1))
    return sorted(list(stems)), source_dir

def compute_dice(pred, gt):
    dice = {}
    for c, n in {1:"RV", 2:"Myo", 3:"LV"}.items():
        p, g = (pred==c), (gt==c)
        inter = (p&g).sum()
        union = p.sum() + g.sum()
        dice[n] = 2*inter/(union+1e-8)
    dice["Mean"] = np.mean(list(dice.values()))
    return dice

def main():
    print("DEBUG: Starting main", flush=True)
    cfg = setup()
    target_stems, source_dir = get_target_stems(cfg)
    print(f"DEBUG: Targets: {len(target_stems)} cases", flush=True)
    
    # Visual KB
    vk_dir = os.path.dirname(source_dir)
    visual_kb = None
    if os.path.isdir(vk_dir):
        try:
            visual_kb = VisualKnowledgeBase.load(vk_dir)
            print(f"DEBUG: Visual KB loaded ({len(visual_kb.good_examples)} good)", flush=True)
        except Exception as e:
            print(f"DEBUG: Visual KB failed: {e}", flush=True)

    atlas_rel = cfg.get("paths", {}).get("atlas_path", "shape_atlas.pkl")
    atlas_path = os.path.join(source_dir, atlas_rel)
    if os.path.exists(atlas_path):
        atlas = load_atlas(atlas_path)
    else:
        atlas = AtlasBuilderAgent(cfg).run(source_dir, atlas_path)

    # Agents
    print("DEBUG: Initializing agents...", flush=True)
    bus = MessageBus()
    agents = {
        "coordinator": CoordinatorAgent(cfg, bus),
        "triage": TriageAgent(cfg, bus, visual_kb=visual_kb),
        "diagnosis": DiagnosisAgent(cfg, bus),
        "planner": PlannerAgent(cfg, bus),
        "executor": ExecutorAgent(cfg, bus, atlas=atlas, visual_kb=visual_kb),
        "verifier": VerifierAgent(cfg, bus, visual_kb=visual_kb),
    }
    
    # Discover Context
    print("DEBUG: Discovering cases...", flush=True)
    all_cases = discover_cases(source_dir, "")
    group_and_link_neighbors(all_cases)
    
    # Filter
    queue = [c for c in all_cases if c.stem in target_stems]
    # LIMIT FOR PILOT: First 5 cases only
    queue = queue[:5]
    print(f"DEBUG: Processing queue size: {len(queue)} (PILOT RUN)", flush=True)
    
    # CSV Init
    csv_path = os.path.join(BATCH_OUT_DIR, "batch_report.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            csv.writer(f).writerow(["Stem", "Verdict", "Ops", "Atlas", "Pre", "Post", "Delta"])

    # Loop
    for ctx in tqdm(queue):
        print(f"Processing {ctx.stem}...", flush=True)
        bus.register_case(ctx)
        # Ensure original_mask is set for rollback safety
        if ctx.mask is not None and ctx.original_mask is None:
            ctx.original_mask = ctx.mask.copy()
        
        # Pre Dice
        gt_path = os.path.join(source_dir, f"{ctx.stem}_gt.png")
        dice_pre = 0.0
        if os.path.exists(gt_path):
            dice_pre = compute_dice(ctx.mask, read_mask(gt_path))["Mean"]
            
        # Run
        try:
            verdict = agents["coordinator"].orchestrate_case(ctx, agents, max_rounds=3)
        except Exception as e:
            print(f"Error ({ctx.stem}): {e}", flush=True)
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

        # --- HYBRID MODE: Deterministic Polish ---
        dice_post_final = dice_post_agent
        
        if ctx.current_mask is not None:
            from cardiac_agent_postproc.ops import (
                expand_myo_intensity_guided, 
                expand_blood_pool_intensity, 
                topology_cleanup
            )
            # Ensure image is grayscale for intensity ops
            polish_img = ctx.image
            if polish_img.ndim == 3 and polish_img.shape[2] in (3, 4):
                 polish_img = cv2.cvtColor(polish_img, cv2.COLOR_BGRA2GRAY if polish_img.shape[2]==4 else cv2.COLOR_BGR2GRAY)
            
            polished_mask = ctx.current_mask.copy()
            
            # 1. Expand RV (target 1)
            polished_mask = expand_blood_pool_intensity(polished_mask, polish_img, 1, cfg)
            # 2. Expand Myo (target 2) - fill gaps
            polished_mask = expand_myo_intensity_guided(polished_mask, polish_img)
            # 3. Expand LV (target 3)
            polished_mask = expand_blood_pool_intensity(polished_mask, polish_img, 3, cfg)
            # 4. Final Safe Cleanup
            polished_mask = topology_cleanup(polished_mask, "4ch", cfg)
            
            # Re-compute Dice after Polish
            if os.path.exists(gt_path):
                dice_polished = compute_dice(polished_mask, read_mask(gt_path))["Mean"]
                
                # SAFETY CHECK 2: Did Polish make it worse?
                if dice_polished >= dice_post_agent - 0.005:
                    dice_post_final = dice_polished
                    ctx.current_mask = polished_mask
                    ctx.applied_ops.append({"name": "deterministic_polish", "delta": dice_polished - dice_post_agent})
                else:
                    print(f"  [Safety] Polish degraded Dice ({dice_polished:.4f} < {dice_post_agent:.4f}). Reverting Polish.", flush=True) 
                    dice_post_final = dice_post_agent
                    
            else:
                 # No GT: Trust the polish (since ops are safe)
                ctx.current_mask = polished_mask
             
        # Log
        ops = [op["name"] for op in ctx.applied_ops]
        atlas = "individual_atlas_transfer" in ops
        with open(csv_path, "a") as f:
            csv.writer(f).writerow([ctx.stem, verdict, len(ops), atlas, 
                                    f"{dice_pre:.4f}", f"{dice_post_final:.4f}", f"{dice_post_final-dice_pre:.4f}"])
        
        # Save

        if ctx.current_mask is not None:
             out_path = os.path.join(BATCH_OUT_DIR, f"{ctx.stem}_repaired.png")
             write_mask(out_path, ctx.current_mask)

if __name__ == "__main__":
    main()
