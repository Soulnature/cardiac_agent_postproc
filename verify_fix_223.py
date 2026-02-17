"""
Verification Script for Case 223 (Intensity-Guided Repair).
"""
import os
import sys
import logging
import csv
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
PROJECT_ROOT = "/data484_5/xzhao14/cardiac_agent_postproc"
sys.path.insert(0, PROJECT_ROOT)

import yaml
import logging
import numpy as np
from cardiac_agent_postproc.agents.coordinator import CoordinatorAgent
from cardiac_agent_postproc.agents.message_bus import CaseContext, MessageBus
from cardiac_agent_postproc.agents.atlas_builder import AtlasBuilderAgent
from cardiac_agent_postproc.io_utils import read_image, read_mask, write_mask
from cardiac_agent_postproc.view_utils import infer_view_type
from cardiac_agent_postproc.atlas import load_atlas

def load_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)
from cardiac_agent_postproc.agents.visual_knowledge import VisualKnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify_223")

def compute_dice_score(pred, gt):
    dice = []
    for c in [1, 2, 3]:
        p = (pred == c)
        g = (gt == c)
        inter = (p & g).sum()
        union = p.sum() + g.sum()
        dice.append(2 * inter / (union + 1e-8))
    return float(np.mean(dice))

def main():
    cfg = load_config()
    
    # Force enable VLM
    if "vision_guardrail" not in cfg:
        cfg["vision_guardrail"] = {}
    cfg["vision_guardrail"]["enabled"] = True
    
    # Initialize agents
    vk_dir = "/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2"
    try:
        visual_kb = VisualKnowledgeBase.load(vk_dir)
        logger.info(f"Visual KB loaded: {len(visual_kb.good_examples)} good examples")
    except Exception as e:
        logger.warning(f"Visual KB load failed: {e}")
        visual_kb = None
    
    bus = MessageBus()
    from cardiac_agent_postproc.agents.triage_agent import TriageAgent
    from cardiac_agent_postproc.agents.diagnosis_agent import DiagnosisAgent
    from cardiac_agent_postproc.agents.planner_agent import PlannerAgent
    from cardiac_agent_postproc.agents.executor import ExecutorAgent
    from cardiac_agent_postproc.agents.verifier_agent import VerifierAgent

    source_dir = cfg["paths"]["source_dir"]
    atlas_rel = cfg.get("paths", {}).get("atlas_path", "shape_atlas.pkl")
    atlas_path = os.path.join(source_dir, atlas_rel)
    if os.path.exists(atlas_path):
        atlas = load_atlas(atlas_path)
    else:
        atlas = AtlasBuilderAgent(cfg).run(source_dir, atlas_path)

    agents = {
        "coordinator": CoordinatorAgent(cfg, bus),
        "triage": TriageAgent(cfg, bus, visual_kb=visual_kb),
        "diagnosis": DiagnosisAgent(cfg, bus),
        "planner": PlannerAgent(cfg, bus),
        "executor": ExecutorAgent(cfg, bus, atlas=atlas, visual_kb=visual_kb),
        "verifier": VerifierAgent(cfg, bus, visual_kb=visual_kb),
    }

    # Target Case 223
    base_dir = Path("/data484_5/xzhao14/cardiac_agent_postproc/results/Input_MnM2/all_frames_export")
    case_stem = "223_original_lax_4c_024"
    
    img_path = base_dir / f"{case_stem}_img.png"
    pred_path = base_dir / f"{case_stem}_pred.png"
    gt_path = base_dir / f"{case_stem}_gt.png"
    
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        return

    logger.info(f"Processing single case: {case_stem}")
    
    # Load data
    img = read_image(str(img_path))
    mask = read_mask(str(pred_path))
    gt_mask = read_mask(str(gt_path))
    
    # Create Context
    view_type, _ = infer_view_type(case_stem, mask, rv_ratio_threshold=0.02)
    ctx = CaseContext(
        case_id=case_stem,
        stem=case_stem,
        img_path=str(img_path),
        pred_path=str(pred_path),
        gt_path=str(gt_path),
        view_type=view_type,
        original_mask=mask.copy(),
        image=img,
        mask=mask.copy(),
    )
    ctx.current_mask = mask.copy()

    # Initial Dice
    initial_dice = compute_dice_score(mask, gt_mask)
    logger.info(f"Initial Dice: {initial_dice:.4f}")
    
    # Run Pipeline
    bus.register_case(ctx)
    result = agents["coordinator"].orchestrate_case(ctx, agents, max_rounds=3)
    
    final_mask = ctx.current_mask
    final_dice = compute_dice_score(final_mask, gt_mask)
    
    logger.info(f"Verdict: {result}")
    logger.info(f"Final Dice: {final_dice:.4f}")
    logger.info(f"Delta: {final_dice - initial_dice:.4f}")
    
    ops = [op["name"] for op in ctx.applied_ops]
    logger.info(f"Ops: {ops}")
    
    # Save result
    out_dir = Path("verify_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{case_stem}_fixed.png"
    write_mask(str(out_path), final_mask)
    
    if final_dice > 0.8:
        print("SUCCESS: Fix verified (Dice > 0.8)")
    else:
        print(f"FAILURE: Dice {final_dice:.4f} still low")

if __name__ == "__main__":
    main()
