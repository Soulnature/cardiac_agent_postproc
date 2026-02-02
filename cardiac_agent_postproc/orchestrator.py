from __future__ import annotations
import os, copy, json, shutil
from typing import Dict, Any
import numpy as np

from .agents.atlas_builder import AtlasBuilderAgent
from .agents.optimizer import OptimizerAgent
from .agents.evaluator import EvaluatorAgent
from .agents.revision_controller import RevisionControllerAgent

def ensure_runs_dir(base_dir: str, subdir: str) -> str:
    run_dir = os.path.join(base_dir, subdir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

class Orchestrator:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run_once(self, cfg: dict, revision_tag: str="v0") -> Dict[str, Any]:
        source = cfg["paths"]["source_dir"]
        target = cfg["paths"]["target_dir"]
        runs_sub = cfg["paths"]["runs_subdir"]

        source_runs = ensure_runs_dir(source, runs_sub)
        target_runs = ensure_runs_dir(target, runs_sub)

        atlas_path = os.path.join(source, cfg["paths"]["atlas_path"])
        atlas_agent = AtlasBuilderAgent(cfg)
        atlas = atlas_agent.run(source, atlas_path)

        # Phase 1: optimize SOURCE (RQS-only)
        opt_agent_src = OptimizerAgent(cfg, atlas)
        src_summary = opt_agent_src.optimize_folder(source, is_target=False, run_dir=source_runs)

        # Phase 2: optimize TARGET (RQS-only but will be evaluated)
        opt_agent_tgt = OptimizerAgent(cfg, atlas)
        tgt_summary = opt_agent_tgt.optimize_folder(target, is_target=True, run_dir=target_runs)

        # Evaluation (TARGET only)
        eval_agent = EvaluatorAgent(cfg)
        eval_summary = eval_agent.evaluate_target(target, outer_iter=int(cfg["solver"]["max_outer"]))

        return {"src_rqs": src_summary, "tgt_rqs": tgt_summary, "tgt_eval": eval_summary}

    def run_with_revisions(self) -> Dict[str, Any]:
        cfg0 = copy.deepcopy(self.cfg)
        results = {}
        best = None

        # baseline run
        print("=== RUN v0 ===")
        results["v0"] = self.run_once(cfg0, revision_tag="v0")

        if not self.cfg["revision"]["enabled"]:
            return results

        # track stall
        stall = 0
        prev = results["v0"]["tgt_eval"]
        prev_min_dice = prev["min_dice"]
        prev_max_hd95 = prev["max_hd95"]
        prev_min_rqs = results["v0"]["tgt_rqs"]["min_rqs"]

        for rev in range(1, int(self.cfg["revision"]["max_revisions"])+1):
            # stopping success
            cur_eval = results[f"v{rev-1}"]["tgt_eval"]
            cur_rqs = results[f"v{rev-1}"]["tgt_rqs"]
            if (cur_eval["min_dice"] >= float(self.cfg["revision"]["stop_min_dice"])
                and cur_eval["max_hd95"] < float(self.cfg["revision"]["stop_max_hd95"])
                and cur_rqs["min_rqs"] >= float(self.cfg["revision"]["stop_min_rqs"])):
                print(f"Stop early: success criteria met at v{rev-1}")
                break

            # propose ONE change
            rc = RevisionControllerAgent(cfg0)
            proposal = rc.propose_change(cfg0["paths"]["target_dir"])
            change = proposal["change"]
            print(f"[REVISION] Applying change: {change}")

            cfg1 = copy.deepcopy(cfg0)
            cfg1 = rc.apply_change(cfg1, change)

            # persist cfg for this revision
            target = cfg1["paths"]["target_dir"]
            run_cfg_path = os.path.join(target, cfg1["paths"]["runs_subdir"], f"params_v{rev}.json")
            os.makedirs(os.path.dirname(run_cfg_path), exist_ok=True)
            with open(run_cfg_path, "w", encoding="utf-8") as f:
                json.dump({"change": change, "cfg": cfg1}, f, indent=2)

            print(f"=== RUN v{rev} ===")
            results[f"v{rev}"] = self.run_once(cfg1, revision_tag=f"v{rev}")

            # stall check
            cur_eval2 = results[f"v{rev}"]["tgt_eval"]
            cur_min_dice = cur_eval2["min_dice"]
            cur_max_hd95 = cur_eval2["max_hd95"]
            cur_min_rqs = results[f"v{rev}"]["tgt_rqs"]["min_rqs"]

            d_dice = cur_min_dice - prev_min_dice
            d_hd = prev_max_hd95 - cur_max_hd95
            d_rqs = cur_min_rqs - prev_min_rqs

            if (d_dice < float(self.cfg["revision"]["stall_min_dice_delta"])
                and d_hd < float(self.cfg["revision"]["stall_max_hd95_delta"])
                and d_rqs < float(self.cfg["revision"]["stall_min_rqs_delta"])):
                stall += 1
            else:
                stall = 0

            prev_min_dice, prev_max_hd95, prev_min_rqs = cur_min_dice, cur_max_hd95, cur_min_rqs

            if stall >= int(self.cfg["revision"]["stall_rounds"]):
                print(f"Stop: improvement stalled for {stall} consecutive revisions.")
                break

            cfg0 = cfg1

        return results
