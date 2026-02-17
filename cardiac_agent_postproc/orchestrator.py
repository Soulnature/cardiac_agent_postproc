"""
Orchestrator: Multi-agent pipeline for cardiac segmentation repair.

This module replaces the old procedural pipeline with a dynamic, LLM-driven
multi-agent workflow orchestrated by the CoordinatorAgent.
"""
from __future__ import annotations

import os
import re
import copy
import json
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from .agents import (
    MessageBus, CaseContext, CoordinatorAgent, TriageAgent,
    DiagnosisAgent, PlannerAgent, ExecutorAgent, VerifierAgent,
    AtlasBuilderAgent,
)
from .agents.visual_knowledge import VisualKnowledgeBase, RepairComparisonKB
from .api_client import OpenAICompatClient
from .io_utils import read_mask, read_image, write_mask
from .settings import LLMSettings
from .view_utils import infer_view_type


def ensure_runs_dir(base_dir: str, subdir: str) -> str:
    run_dir = os.path.join(base_dir, subdir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _parse_stem_info(stem: str) -> Tuple[str, str, int]:
    """
    Parse patient_id, view_key, and slice_index from a stem.
    Example: '201_original_lax_4c_009' → ('201', '201_original_lax_4c', 9)
    The view_key groups slices from the same patient+view.
    """
    # Last segment after _ is the slice index (e.g. '009')
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        view_key = parts[0]  # e.g. '201_original_lax_4c'
        slice_index = int(parts[1])
    else:
        view_key = stem
        slice_index = 0

    # Patient ID is the first numeric segment
    match = re.match(r"^(\d+)", stem)
    patient_id = match.group(1) if match else stem[:3]

    return patient_id, view_key, slice_index


def _summarize_verdicts(verdicts: Dict[str, str]) -> Dict[str, int]:
    """Count verdict categories for compact run summaries."""
    summary = {
        "total": len(verdicts),
        "good": 0,
        "approved": 0,
        "gave_up": 0,
        "rejected": 0,
    }
    for v in verdicts.values():
        if v in summary:
            summary[v] += 1
    return summary


def discover_cases(
    folder: str,
    source_dir: str = "",
    gt_dir: str = "",
) -> List[CaseContext]:
    """
    Discover pred/img pairs and build CaseContext objects.
    Parses patient_id and slice_index from filenames.
    """
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        return []

    preds = [f for f in files if f.endswith("_pred.png")]
    cases = []

    for p in preds:
        stem = p[:-9]  # remove _pred.png
        img_name = f"{stem}_img.png"
        pred_path = os.path.join(folder, p)

        # Find image
        local_img = os.path.join(folder, img_name)
        if os.path.exists(local_img):
            img_path = local_img
        elif source_dir and os.path.exists(os.path.join(source_dir, img_name)):
            img_path = os.path.join(source_dir, img_name)
        else:
            continue

        # Find GT (optional)
        gt_path = ""
        if gt_dir:
            # Prefer *_gt.png; keep *_pred.png fallback for legacy datasets.
            for gt_name in (f"{stem}_gt.png", f"{stem}_pred.png"):
                cand = os.path.join(gt_dir, gt_name)
                if os.path.exists(cand):
                    gt_path = cand
                    break

        # Load mask and infer view
        mask = read_mask(pred_path)
        image = read_image(img_path)
        view_type, _ = infer_view_type(stem, mask, rv_ratio_threshold=0.02)

        # Parse patient/slice info
        patient_id, _view_key, slice_index = _parse_stem_info(stem)

        ctx = CaseContext(
            case_id=stem,
            stem=stem,
            img_path=img_path,
            pred_path=pred_path,
            gt_path=gt_path,
            mask=mask,
            image=image,
            view_type=view_type,
            current_mask=mask.copy(),
            original_mask=mask.copy(),
            patient_id=patient_id,
            slice_index=slice_index,
        )
        cases.append(ctx)

    return cases


def group_and_link_neighbors(cases: List[CaseContext]) -> Dict[str, List[CaseContext]]:
    """
    Group cases by patient+view and populate neighbor_masks on each case.
    Returns groups dict (view_key → list of CaseContext sorted by slice_index).
    """
    # Group by view_key (patient_id + view, e.g. '201_original_lax_4c')
    groups: Dict[str, List[CaseContext]] = defaultdict(list)
    for ctx in cases:
        _pid, view_key, _si = _parse_stem_info(ctx.stem)
        groups[view_key].append(ctx)

    # Sort each group by slice_index and populate neighbor_masks
    for view_key, group in groups.items():
        group.sort(key=lambda c: c.slice_index)

        for ctx in group:
            ctx.neighbor_masks = {}
            ctx.neighbor_img_paths = {}
            for other_ctx in group:
                if other_ctx.slice_index != ctx.slice_index:
                    ctx.neighbor_masks[other_ctx.slice_index] = other_ctx.mask.copy()
                    ctx.neighbor_img_paths[other_ctx.slice_index] = other_ctx.img_path

    n_with_neighbors = sum(1 for c in cases if len(c.neighbor_masks) > 0)
    print(f"Slice groups: {len(groups)} patient-views, {n_with_neighbors}/{len(cases)} cases have neighbors")

    return dict(groups)


def _update_neighbor_masks(ctx: CaseContext, all_cases: List[CaseContext]) -> None:
    """
    After processing a case, update the neighbor_masks of its sibling slices
    so they see the repaired mask instead of the original.
    """
    if ctx.current_mask is None:
        return
    _, view_key, _ = _parse_stem_info(ctx.stem)
    for other in all_cases:
        if other.stem == ctx.stem:
            continue
        _, other_vk, _ = _parse_stem_info(other.stem)
        if other_vk == view_key:
            other.neighbor_masks[ctx.slice_index] = ctx.current_mask.copy()


class Orchestrator:
    """
    Multi-agent orchestrator.

    Creates and wires together all LLM agents, then processes
    cases through the CoordinatorAgent's dynamic pipeline.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run_once(self, cfg: dict, revision_tag: str = "v0") -> Dict[str, Any]:
        """
        Run the multi-agent pipeline:
        1. Build atlas
        2. Set up agents and message bus
        3. Process each case via CoordinatorAgent
        4. Final batch evaluation
        """
        source = cfg["paths"]["source_dir"]
        target = cfg["paths"]["target_dir"]
        runs_sub = cfg["paths"]["runs_subdir"]
        gt_dir = cfg.get("paths", {}).get("gt_dir", "")

        source_runs = ensure_runs_dir(source, runs_sub)
        target_runs = ensure_runs_dir(target, runs_sub)

        # Step 0: Warmup LLM if using Ollama (preload model into GPU)
        llm_cfg = cfg.get("llm", {})
        if llm_cfg.get("provider", "") == "ollama" and llm_cfg.get("enabled", True):
            settings = LLMSettings()
            warmup_client = OpenAICompatClient(
                base_url=llm_cfg.get("base_url", settings.openai_base_url),
                api_key=llm_cfg.get("api_key", settings.openai_api_key),
                model=llm_cfg.get("model", settings.openai_model),
                timeout=float(llm_cfg.get("timeout", 180.0)),
                provider="ollama",
            )
            warmup_client.warmup()

        # Step 1: Build atlas
        atlas_path = os.path.join(source, cfg["paths"]["atlas_path"])
        atlas_agent = AtlasBuilderAgent(cfg)
        atlas = atlas_agent.run(source, atlas_path)

        # Step 2: Load visual knowledge base for VLM-based quality judgment
        results_dir = cfg.get("paths", {}).get(
            "visual_examples_dir",
            os.path.join(os.path.dirname(source), "results", "Input_MnM2"),
        )
        visual_kb = None
        if os.path.isdir(results_dir):
            visual_kb = VisualKnowledgeBase.load(results_dir)
            print(f"Visual KB: {len(visual_kb.good_examples)} good, {len(visual_kb.bad_examples)} bad examples")
        else:
            print(f"Warning: visual examples dir not found: {results_dir}")

        # Step 2b: Load repair comparison KB (if built)
        repair_kb_dir = cfg.get("paths", {}).get(
            "repair_kb_dir",
            os.path.join(os.path.dirname(source), "repair_kb"),
        )
        repair_kb = None
        if os.path.isdir(repair_kb_dir):
            repair_kb = RepairComparisonKB.load(repair_kb_dir)
            print(f"Repair KB: {repair_kb.total} examples "
                  f"({len(repair_kb.improved)} improved, {len(repair_kb.degraded)} degraded)")
        else:
            print(f"Info: No repair KB found at {repair_kb_dir} (using old VLM comparison)")

        # Step 3: Create MessageBus and agents
        bus = MessageBus()

        multi_cfg = cfg.get("multi_agent", {})
        max_rounds = int(multi_cfg.get("max_rounds_per_case", 3))

        agents = {
            "coordinator": CoordinatorAgent(cfg, bus),
            "triage": TriageAgent(cfg, bus, visual_kb=visual_kb),
            "diagnosis": DiagnosisAgent(cfg, bus),
            "planner": PlannerAgent(cfg, bus),
            "executor": ExecutorAgent(cfg, bus, atlas=atlas, visual_kb=visual_kb, repair_kb=repair_kb),
            "verifier": VerifierAgent(cfg, bus, visual_kb=visual_kb, repair_kb=repair_kb),
        }

        coordinator = agents["coordinator"]
        source_dir_cfg = cfg.get("paths", {}).get("source_dir", "")

        # --- SOURCE pipeline ---
        print("=== SOURCE: Multi-Agent Pipeline ===")
        src_cases = discover_cases(source, source_dir_cfg)
        group_and_link_neighbors(src_cases)
        src_verdicts = {}

        for ctx in src_cases:
            bus.register_case(ctx)
            verdict = coordinator.orchestrate_case(ctx, agents, max_rounds=max_rounds)
            src_verdicts[ctx.stem] = verdict

            # After processing, update neighbor_masks for siblings
            # so later slices in the same group get repaired masks
            _update_neighbor_masks(ctx, src_cases)

            # Save optimized mask if approved
            if verdict in ("approved", "good") and ctx.current_mask is not None:
                out_path = os.path.join(source_runs, f"{ctx.stem}_pred.png")
                write_mask(out_path, ctx.current_mask)

        # --- TARGET pipeline ---
        print("=== TARGET: Multi-Agent Pipeline ===")
        tgt_cases = discover_cases(target, source_dir_cfg, gt_dir=gt_dir)
        group_and_link_neighbors(tgt_cases)
        tgt_verdicts = {}

        for ctx in tgt_cases:
            bus.register_case(ctx)
            verdict = coordinator.orchestrate_case(ctx, agents, max_rounds=max_rounds)
            tgt_verdicts[ctx.stem] = verdict

            # Update neighbor_masks for siblings
            _update_neighbor_masks(ctx, tgt_cases)

            # Save optimized mask if approved
            if verdict in ("approved",) and ctx.current_mask is not None:
                out_path = os.path.join(target_runs, f"{ctx.stem}_pred.png")
                write_mask(out_path, ctx.current_mask)

        # --- Batch evaluation ---
        verifier = agents["verifier"]
        tgt_case_dict = {c.case_id: c for c in tgt_cases}
        eval_summary = verifier.evaluate_batch(tgt_case_dict, target_runs)

        # --- Batch summary ---
        all_cases = {c.case_id: c for c in src_cases + tgt_cases}
        batch_summary = coordinator.generate_batch_summary(all_cases)

        # --- Save message audit trail ---
        audit_path = os.path.join(target_runs, "agent_audit_trail.json")
        history = bus.get_full_history()
        with open(audit_path, "w") as f:
            json.dump(
                [{"sender": m.sender, "recipient": m.recipient,
                  "msg_type": m.msg_type.value, "case_id": m.case_id,
                  "content": str(m.content)[:300]}
                 for m in history],
                f, indent=2,
            )
        print(f"Audit trail: {len(history)} messages → {audit_path}")

        return {
            "src_verdicts": src_verdicts,
            "src_summary": _summarize_verdicts(src_verdicts),
            "tgt_verdicts": tgt_verdicts,
            "tgt_summary": _summarize_verdicts(tgt_verdicts),
            "tgt_eval": eval_summary,
            "batch_summary": batch_summary,
        }

    def _setup_run_dir(self, base_target: str, run_name: str, source_dir: str) -> str:
        run_dir = os.path.join(base_target, run_name)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        print(f"[{run_name}] Initializing workspace from {source_dir}...")
        for item in os.listdir(source_dir):
            s = os.path.join(source_dir, item)
            d = os.path.join(run_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        return run_dir

    def run_with_revisions(self) -> Dict[str, Any]:
        """
        Run the pipeline. The multi-agent system handles revisions internally
        via the CoordinatorAgent's retry loop.
        """
        cfg0 = copy.deepcopy(self.cfg)
        results = {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_target = cfg0["paths"]["target_dir"]

        run_name = f"run_{ts}_v0"
        print(f"=== RUN {run_name} ===")

        run_dir = self._setup_run_dir(base_target, run_name, cfg0["paths"]["source_dir"])
        cfg0["paths"]["target_dir"] = run_dir

        results["v0"] = self.run_once(cfg0, revision_tag="v0")

        return results
