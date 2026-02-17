"""
End-to-end pipeline test with local Ollama (ministral-3:14b).

Runs 2 cases through the full multi-agent loop:
  Triage → Diagnosis → Plan → Execute → Verify

Usage:
    python test_pipeline_ollama.py
    python test_pipeline_ollama.py --cases 5
    python test_pipeline_ollama.py --stem 210_original_lax_4c_029
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from cardiac_agent_postproc.agents import (
    MessageBus, CaseContext, CoordinatorAgent, TriageAgent,
    DiagnosisAgent, PlannerAgent, ExecutorAgent, VerifierAgent,
)
from cardiac_agent_postproc.agents.atlas_builder import AtlasBuilderAgent
from cardiac_agent_postproc.agents.visual_knowledge import VisualKnowledgeBase
from cardiac_agent_postproc.api_client import OpenAICompatClient
from cardiac_agent_postproc.atlas import load_atlas
from cardiac_agent_postproc.io_utils import read_mask, read_image, write_mask
from cardiac_agent_postproc.orchestrator import discover_cases, group_and_link_neighbors
from cardiac_agent_postproc.view_utils import infer_view_type


def compute_dice(pred, gt):
    """Compute per-class and mean Dice."""
    scores = {}
    for c, name in {1: "RV", 2: "Myo", 3: "LV"}.items():
        p, g = (pred == c), (gt == c)
        inter = (p & g).sum()
        union = p.sum() + g.sum()
        scores[name] = float(2 * inter / (union + 1e-8))
    scores["Mean"] = float(np.mean(list(scores.values())))
    return scores


def main():
    parser = argparse.ArgumentParser(description="E2E pipeline test with Ollama")
    parser.add_argument("--config", default="config/ollama_local.yaml")
    parser.add_argument("--cases", type=int, default=2, help="Number of cases to process")
    parser.add_argument("--stem", default=None, help="Run a specific case stem")
    parser.add_argument("--max-rounds", type=int, default=2, help="Max repair rounds per case")
    parser.add_argument("--out-dir", default="test_results_ollama", help="Output directory")
    args = parser.parse_args()

    # ---- Load config ----
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Config: {args.config}")
    print(f"LLM: {cfg['llm']['provider']} / {cfg['llm']['model']}")
    print(f"VLM: {cfg['vision_guardrail']['provider']} / {cfg['vision_guardrail']['model']}")

    # ---- Warmup ----
    llm_cfg = cfg["llm"]
    client = OpenAICompatClient(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=llm_cfg["model"],
        timeout=float(llm_cfg.get("timeout", 300)),
        provider=llm_cfg["provider"],
    )
    if not client.warmup():
        print("ERROR: Ollama warmup failed. Is the server running?")
        sys.exit(1)

    # ---- Discover cases ----
    source_dir = cfg["paths"]["source_dir"]
    cases = discover_cases(source_dir, source_dir)
    groups = group_and_link_neighbors(cases)
    print(f"Discovered {len(cases)} cases in {len(groups)} patient-views")

    # Filter
    if args.stem:
        cases = [c for c in cases if c.stem == args.stem]
        if not cases:
            print(f"ERROR: stem '{args.stem}' not found")
            sys.exit(1)
    cases = cases[:args.cases]
    print(f"Processing {len(cases)} case(s)")

    # ---- Visual KB ----
    vk_dir = os.path.dirname(source_dir)
    visual_kb = None
    if os.path.isdir(vk_dir):
        try:
            visual_kb = VisualKnowledgeBase.load(vk_dir)
            print(f"Visual KB: {len(visual_kb.good_examples)} good, {len(visual_kb.bad_examples)} bad")
        except Exception as e:
            print(f"Visual KB load failed (non-fatal): {e}")

    # ---- Create agents ----
    atlas_rel = cfg.get("paths", {}).get("atlas_path", "shape_atlas.pkl")
    atlas_path = os.path.join(source_dir, atlas_rel)
    if os.path.exists(atlas_path):
        atlas = load_atlas(atlas_path)
    else:
        atlas = AtlasBuilderAgent(cfg).run(source_dir, atlas_path)

    bus = MessageBus()
    agents = {
        "coordinator": CoordinatorAgent(cfg, bus),
        "triage": TriageAgent(cfg, bus, visual_kb=visual_kb),
        "diagnosis": DiagnosisAgent(cfg, bus),
        "planner": PlannerAgent(cfg, bus),
        "executor": ExecutorAgent(cfg, bus, atlas=atlas, visual_kb=visual_kb),
        "verifier": VerifierAgent(cfg, bus, visual_kb=visual_kb),
    }

    # ---- Output dir ----
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Process each case ----
    results = []
    for i, ctx in enumerate(cases):
        print(f"\n{'='*70}")
        print(f"CASE {i+1}/{len(cases)}: {ctx.stem} (view: {ctx.view_type})")
        print(f"{'='*70}")

        bus.register_case(ctx)

        # Pre-repair Dice (if GT available)
        gt_path = os.path.join(source_dir, f"{ctx.stem}_gt.png")
        dice_pre = None
        if os.path.exists(gt_path):
            gt = read_mask(gt_path)
            dice_pre = compute_dice(ctx.mask, gt)
            print(f"  Pre-repair Dice: {dice_pre}")

        # Run the multi-agent pipeline
        t0 = time.time()
        try:
            verdict = agents["coordinator"].orchestrate_case(
                ctx, agents, max_rounds=args.max_rounds,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            verdict = "error"
        elapsed = time.time() - t0

        # Post-repair Dice
        dice_post = None
        if os.path.exists(gt_path) and ctx.current_mask is not None:
            dice_post = compute_dice(ctx.current_mask, gt)
            print(f"  Post-repair Dice: {dice_post}")

        # Summary
        ops = [op.get("name", "?") for op in ctx.applied_ops]
        result = {
            "stem": ctx.stem,
            "view": ctx.view_type,
            "verdict": verdict,
            "triage": ctx.triage_category,
            "ops": ops,
            "rounds": ctx.rounds_completed,
            "elapsed_s": round(elapsed, 1),
            "dice_pre": dice_pre,
            "dice_post": dice_post,
        }
        results.append(result)

        delta_str = ""
        if dice_pre and dice_post:
            delta = dice_post["Mean"] - dice_pre["Mean"]
            delta_str = f"  Dice delta: {delta:+.4f}"

        print(f"\n  Verdict: {verdict}")
        print(f"  Triage: {ctx.triage_category}")
        print(f"  Ops applied: {ops}")
        print(f"  Rounds: {ctx.rounds_completed}")
        print(f"  Time: {elapsed:.1f}s")
        if delta_str:
            print(delta_str)

        # Save repaired mask
        if ctx.current_mask is not None:
            out_path = os.path.join(args.out_dir, f"{ctx.stem}_repaired.png")
            write_mask(out_path, ctx.current_mask)

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print("PIPELINE TEST SUMMARY")
    print(f"{'='*70}")

    n_good = sum(1 for r in results if r["verdict"] == "good")
    n_approved = sum(1 for r in results if r["verdict"] == "approved")
    n_gave_up = sum(1 for r in results if r["verdict"] == "gave_up")
    n_error = sum(1 for r in results if r["verdict"] == "error")

    print(f"  Total cases: {len(results)}")
    print(f"  Good (skip): {n_good}")
    print(f"  Approved:    {n_approved}")
    print(f"  Gave up:     {n_gave_up}")
    print(f"  Errors:      {n_error}")

    if any(r["dice_pre"] for r in results):
        pre_dices = [r["dice_pre"]["Mean"] for r in results if r["dice_pre"]]
        post_dices = [r["dice_post"]["Mean"] for r in results if r["dice_post"]]
        if pre_dices and post_dices:
            print(f"  Mean Dice pre:  {np.mean(pre_dices):.4f}")
            print(f"  Mean Dice post: {np.mean(post_dices):.4f}")
            print(f"  Mean Dice Δ:    {np.mean(post_dices) - np.mean(pre_dices):+.4f}")

    # Save results
    report_path = os.path.join(args.out_dir, "pipeline_test_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # Exit code
    if n_error > 0:
        print("\nSome cases had errors!")
        sys.exit(1)
    else:
        print("\nAll cases processed successfully!")


if __name__ == "__main__":
    main()
