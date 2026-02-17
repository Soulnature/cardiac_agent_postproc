"""
Test Full Pipeline on Case 335 (Hybrid Repair).
"""
import os
import sys
import yaml
import shutil
from cardiac_agent_postproc.agents.message_bus import MessageBus
from cardiac_agent_postproc.agents.coordinator import CoordinatorAgent
from cardiac_agent_postproc.agents.triage_agent import TriageAgent
from cardiac_agent_postproc.agents.diagnosis_agent import DiagnosisAgent
from cardiac_agent_postproc.agents.planner_agent import PlannerAgent
from cardiac_agent_postproc.agents.executor import ExecutorAgent
from cardiac_agent_postproc.agents.verifier_agent import VerifierAgent
from cardiac_agent_postproc.agents.atlas_builder import AtlasBuilderAgent
from cardiac_agent_postproc.orchestrator import discover_cases, group_and_link_neighbors, _update_neighbor_masks
from cardiac_agent_postproc.io_utils import read_mask
from cardiac_agent_postproc.agents.visual_knowledge import VisualKnowledgeBase
from cardiac_agent_postproc.atlas import load_atlas

# Config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)

# Mock output dir
TEST_DIR = "test_results_335"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

SOURCE_DIR = cfg["paths"]["source_dir"]
cfg["paths"]["source_dir"] = SOURCE_DIR # Ensure it points to real data

# Copy 335 files to TEST_DIR
print(f"Copying 335 files from {SOURCE_DIR} to {TEST_DIR}...")
import glob
for f in glob.glob(f"{SOURCE_DIR}/335_*"):
    shutil.copy(f, TEST_DIR)

# Discover cases
print("Discovering cases...")
cases = discover_cases(TEST_DIR, "") # source_dir_cfg irrelevant since we copied
# Filter for 335
cases = [c for c in cases if "335_original_lax_4c" in c.stem]
print(f"Found {len(cases)} cases for 335: {[c.stem for c in cases]}")

# Group (important for neighbors)
groups = group_and_link_neighbors(cases)
print(f"Grouped into {len(groups)} groups.")
for c in cases:
    print(f"  {c.stem}: {len(c.neighbor_masks)} neighbors, {len(c.neighbor_img_paths)} neighbor images")

# Setup Visual KB
visual_kb = None
vk_dir = os.path.dirname(SOURCE_DIR)
if os.path.isdir(vk_dir):
    try:
        visual_kb = VisualKnowledgeBase.load(vk_dir)
        print(f"Visual KB loaded: {len(visual_kb.good_examples)} good, {len(visual_kb.bad_examples)} bad")
    except Exception as e:
        print(f"Visual KB load failed: {e}")

# Setup Agents
atlas_rel = cfg.get("paths", {}).get("atlas_path", "shape_atlas.pkl")
atlas_path = os.path.join(SOURCE_DIR, atlas_rel)
if os.path.exists(atlas_path):
    atlas = load_atlas(atlas_path)
else:
    atlas = AtlasBuilderAgent(cfg).run(SOURCE_DIR, atlas_path)

bus = MessageBus()
agents = {
    "coordinator": CoordinatorAgent(cfg, bus),
    "triage": TriageAgent(cfg, bus, visual_kb=visual_kb),
    "diagnosis": DiagnosisAgent(cfg, bus),
    "planner": PlannerAgent(cfg, bus),
    "executor": ExecutorAgent(cfg, bus, atlas=atlas, visual_kb=visual_kb),
    "verifier": VerifierAgent(cfg, bus, visual_kb=visual_kb),
}

coordinator = agents["coordinator"]

# Run pipeline
print("\n=== Running Pipeline ===")
results = {}
for ctx in cases:
    if "009" not in ctx.stem:
        continue # Focused on the bad slice
        
    print(f"\nProcessing {ctx.stem}...")
    bus.register_case(ctx)
    verdict = coordinator.orchestrate_case(ctx, agents, max_rounds=3)
    results[ctx.stem] = verdict
    print(f"Verdict: {verdict}")
    
    # Save result
    if ctx.current_mask is not None:
        out_path = f"{TEST_DIR}/{ctx.stem}_pred.png"
        import cv2
        cv2.imwrite(out_path, ctx.current_mask)
        
        # Evaluate Dice
        gt = read_mask(f"{SOURCE_DIR}/{ctx.stem}_gt.png")
        import numpy as np
        dice = {}
        for c, n in {1:"RV", 2:"Myo", 3:"LV"}.items():
            p, g = (ctx.current_mask==c), (gt==c)
            dice[n] = 2*(p&g).sum()/(p.sum()+g.sum()+1e-8)
        dice["Mean"] = np.mean(list(dice.values()))
        print(f"Final Dice: {dice}")

        # Check applied ops
        print("Applied Ops:")
        for op in ctx.applied_ops:
            print(f"  - {op['name']} (px changed: {op['pixels_changed']})")

print("\nDONE")
