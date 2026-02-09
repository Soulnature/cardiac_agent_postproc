from __future__ import annotations
import os, copy, json, shutil
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

from .agents.atlas_builder import AtlasBuilderAgent
from .agents.optimizer import OptimizerAgent
from .agents.evaluator import EvaluatorAgent
from .agents.triage_agent import TriageAgent, TriageResult
from .agents.diagnosis_agent import DiagnosisAgent, SpatialDiagnosis
from .io_utils import read_mask, read_image
from .view_utils import infer_view_type


def ensure_runs_dir(base_dir: str, subdir: str) -> str:
    run_dir = os.path.join(base_dir, subdir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def discover_files(
    folder: str, source_dir: str = ""
) -> tuple[Dict[str, Dict[str, str]], Dict[str, np.ndarray], Dict[str, str]]:
    """
    Discover pred/img pairs in folder, returning metas, masks, and view_types.
    Reusable by triage, diagnosis, and optimizer.
    """
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        return {}, {}, {}

    preds = [f for f in files if f.endswith("_pred.png")]
    metas: Dict[str, Dict[str, str]] = {}

    for p in preds:
        stem = p[:-9]  # remove _pred.png
        img_name = f"{stem}_img.png"
        local_img = os.path.join(folder, img_name)
        if os.path.exists(local_img):
            img_path = local_img
        elif source_dir and os.path.exists(os.path.join(source_dir, img_name)):
            img_path = os.path.join(source_dir, img_name)
        else:
            continue
        metas[stem] = {
            "stem": stem,
            "pred": os.path.join(folder, p),
            "img": img_path,
        }

    masks: Dict[str, np.ndarray] = {}
    view_types: Dict[str, str] = {}

    for stem, fr in metas.items():
        mask = read_mask(fr["pred"])
        masks[stem] = mask
        vt, _ = infer_view_type(stem, mask, rv_ratio_threshold=0.02)
        view_types[stem] = vt

    return metas, masks, view_types


class Orchestrator:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run_once(self, cfg: dict, revision_tag: str = "v0") -> Dict[str, Any]:
        """
        Simplified pipeline:
        1. Build atlas (shape reference)
        2. VLM Screening (enhanced triage)
        3. VLM Spatial Diagnosis (for needs_fix cases)
        4. AI Controller (spatially-targeted fixes with VLM verification)
        5. Evaluate (GT, unchanged)
        """
        source = cfg["paths"]["source_dir"]
        target = cfg["paths"]["target_dir"]
        runs_sub = cfg["paths"]["runs_subdir"]

        source_runs = ensure_runs_dir(source, runs_sub)
        target_runs = ensure_runs_dir(target, runs_sub)

        # Step 1: Build atlas (unchanged)
        atlas_path = os.path.join(source, cfg["paths"]["atlas_path"])
        atlas_agent = AtlasBuilderAgent(cfg)
        atlas = atlas_agent.run(source, atlas_path)

        source_dir_cfg = cfg.get("paths", {}).get("source_dir", "")

        # --- SOURCE pipeline ---
        print("=== SOURCE: Triage -> Spatial Diagnosis -> AI Controller ===")
        src_metas, src_masks, src_views = discover_files(source, source_dir_cfg)

        # Step 2: VLM Screening (triage)
        triage_agent = TriageAgent(cfg)
        src_triage = triage_agent.triage_folder(source, src_metas, src_masks, src_views)

        # Step 3: VLM Spatial Diagnosis (only for needs_fix cases)
        diag_agent = DiagnosisAgent(cfg)
        src_spatial_diags: Dict[str, List[SpatialDiagnosis]] = {}
        for stem, tr in src_triage.items():
            if tr.category != "good":
                img = read_image(src_metas[stem]["img"])
                src_spatial_diags[stem] = diag_agent.diagnose_spatial(
                    src_masks[stem], img, tr.view_type,
                    img_path=src_metas[stem]["img"],
                    stem=stem,
                    triage_issues=tr.issues,
                )

        # Step 4: AI Controller (spatially-targeted optimization)
        opt_agent_src = OptimizerAgent(cfg, atlas)
        src_summary = opt_agent_src.optimize_folder(
            source, is_target=False, run_dir=source_runs,
            triage_results=src_triage,
            spatial_diagnoses=src_spatial_diags,
        )

        # --- TARGET pipeline ---
        print("=== TARGET: Triage -> Spatial Diagnosis -> AI Controller ===")
        tgt_metas, tgt_masks, tgt_views = discover_files(target, source_dir_cfg)

        # Step 5: VLM Screening (triage)
        tgt_triage = triage_agent.triage_folder(target, tgt_metas, tgt_masks, tgt_views)

        # Step 6: VLM Spatial Diagnosis (only for needs_fix cases)
        tgt_spatial_diags: Dict[str, List[SpatialDiagnosis]] = {}
        for stem, tr in tgt_triage.items():
            if tr.category != "good":
                img = read_image(tgt_metas[stem]["img"])
                tgt_spatial_diags[stem] = diag_agent.diagnose_spatial(
                    tgt_masks[stem], img, tr.view_type,
                    img_path=tgt_metas[stem]["img"],
                    stem=stem,
                    triage_issues=tr.issues,
                )

        # Step 7: AI Controller (spatially-targeted optimization with VLM verification)
        opt_agent_tgt = OptimizerAgent(cfg, atlas)
        tgt_summary = opt_agent_tgt.optimize_folder(
            target, is_target=True, run_dir=target_runs,
            triage_results=tgt_triage,
            spatial_diagnoses=tgt_spatial_diags,
        )

        # Step 8: Save triage reports
        triage_agent.save_triage_report(
            src_triage,
            os.path.join(source_runs, "triage_report.csv"),
        )
        triage_agent.save_triage_report(
            tgt_triage,
            os.path.join(target_runs, "triage_report.csv"),
        )

        # Step 9: Evaluation (TARGET only, unchanged)
        eval_agent = EvaluatorAgent(cfg)
        eval_summary = eval_agent.evaluate_target(
            target, outer_iter=int(cfg["solver"].get("max_outer", 10))
        )

        return {
            "src_summary": src_summary,
            "tgt_summary": tgt_summary,
            "tgt_eval": eval_summary,
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
        Run the pipeline. Revision loop is disabled by default since the new
        VLM-driven pipeline handles fixes in a single pass.
        """
        cfg0 = copy.deepcopy(self.cfg)
        results = {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_target = cfg0["paths"]["target_dir"]

        # Single run (no revision loop needed with VLM-driven pipeline)
        run_name = f"run_{ts}_v0"
        print(f"=== RUN {run_name} ===")

        run_dir = self._setup_run_dir(base_target, run_name, cfg0["paths"]["source_dir"])
        cfg0["paths"]["target_dir"] = run_dir

        results["v0"] = self.run_once(cfg0, revision_tag="v0")

        if not self.cfg.get("revision", {}).get("enabled", False):
            return results

        # Optional revision loop (re-enable later with better strategy)
        # For now, the VLM-driven pipeline is designed as single-pass
        print("[Revision] Revision loop disabled for VLM-driven pipeline.")
        return results
