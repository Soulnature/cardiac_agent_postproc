from __future__ import annotations
import os, json, csv, shutil
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from ..settings import LLMSettings
from ..api_client import OpenAICompatClient

SYSTEM_PROMPT = """You are a strict ML engineer. You will propose exactly ONE parameter change to improve alignment between RQS and TARGET Dice/HD95.
Return JSON only with this schema:
{
  "change": {
     "path": "dot.separated.yaml.path",
     "value": <number or string>,
     "rationale": "one paragraph"
  }
}
Constraints:
- You MUST NOT propose using GT inside inner-loop optimization.
- Prefer small, safe changes to weights/thresholds (e.g., reduce P_size_drift weight, adjust edge percentile).
"""

class RevisionControllerAgent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.llm_settings = LLMSettings()
        self.client = None
        if cfg["llm"]["enabled"] or self.llm_settings.llm_enabled:
            self.client = OpenAICompatClient(
                base_url=self.llm_settings.openai_base_url,
                api_key=self.llm_settings.openai_api_key,
                model=self.llm_settings.openai_model,
            )

    def _load_target_reports(self, target_dir: str) -> pd.DataFrame:
        path = os.path.join(target_dir, "target_dice_report.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _load_rqs_log(self, folder: str) -> pd.DataFrame:
        path = os.path.join(folder, "label_optimization_log.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _worst_union(self, df_eval: pd.DataFrame, df_log: pd.DataFrame, k: int=20) -> dict:
        # compute worst lists at latest outer_iter
        out = {}
        if "outer_iter" in df_eval.columns:
            last_outer = int(df_eval["outer_iter"].max())
            de = df_eval[df_eval["outer_iter"]==last_outer].copy()
        else:
            de = df_eval.copy()
            last_outer = -1
        # worst by dice, worst by hd95
        bottom_dice = de.nsmallest(k, "dice_macro")["filename"].tolist()
        top_hd = de.nlargest(k, "hd95_macro")["filename"].tolist()

        # worst by RQS: use latest outer_iter in log
        last_outer_log = int(df_log["outer_iter"].max()) if "outer_iter" in df_log.columns else -1
        dl = df_log[df_log["outer_iter"]==last_outer_log].copy() if last_outer_log>=0 else df_log
        # take last record per filename
        dl = dl.sort_values(["filename","inner_step"]).groupby("filename").tail(1)
        bottom_rqs = dl.nsmallest(k, "RQS_total")["filename"].tolist()

        union = sorted(set(bottom_dice) | set(top_hd) | set(bottom_rqs))
        out.update({
            "last_outer_eval": last_outer,
            "last_outer_log": last_outer_log,
            "bottom_dice": bottom_dice,
            "top_hd95": top_hd,
            "bottom_rqs": bottom_rqs,
            "union": union,
            "overlap_count": len(set(bottom_dice) & set(top_hd) & set(bottom_rqs)),
        })
        return out

    def _dominant_penalties(self, df_log: pd.DataFrame, offenders: List[str]) -> List[Tuple[str,float]]:
        last_outer = int(df_log["outer_iter"].max())
        dl = df_log[df_log["outer_iter"]==last_outer].copy()
        dl = dl.sort_values(["filename","inner_step"]).groupby("filename").tail(1)
        dl = dl[dl["filename"].isin(offenders)]
        pen_cols = [c for c in dl.columns if c.startswith("P_")]
        means = {c: float(dl[c].mean()) for c in pen_cols}
        top3 = sorted(means.items(), key=lambda x: -x[1])[:3]
        return top3

    def propose_change(self, target_dir: str) -> Dict[str, Any]:
        df_eval = self._load_target_reports(target_dir)
        df_log = self._load_rqs_log(target_dir)

        worst = self._worst_union(df_eval, df_log, k=20)
        offenders = worst["union"]
        top3 = self._dominant_penalties(df_log, offenders)

        print(f"[REVISION] overlap(bottomRQS & bottomDice & topHD95) = {worst['overlap_count']}")
        print(f"[REVISION] offenders union (n={len(offenders)}): {offenders[:30]}{'...' if len(offenders)>30 else ''}")
        print(f"[REVISION] dominant penalties (mean over offenders): {top3}")

        # If LLM disabled, use a deterministic heuristic proposal
        if self.client is None:
            # heuristic: if boundary roughness dominates -> increase smoothing; if size drift dominates -> reduce its weight; else reduce shape prior.
            lead = top3[0][0] if top3 else "P_size_drift"
            if lead == "P_boundary_roughness":
                return {"change":{"path":"ops.smooth.gaussian_sigma","value":1.2,"rationale":"High HD95 proxy suggests jagged boundaries; modestly increase Gaussian smoothing strength."}}
            if lead == "P_size_drift":
                return {"change":{"path":"reward_weights.P_size_drift","value":0.8,"rationale":"If size drift dominates offenders, it may over-penalize necessary corrections and harm Dice; reduce weight slightly."}}
            if lead == "P_edge_misalignment":
                return {"change":{"path":"edge.sobel_percentile","value":88,"rationale":"Edge term dominates; increase edge threshold to focus on stronger edges and reduce noise-driven boundary pulling."}}
            return {"change":{"path":"reward_weights.P_shape_prior","value":0.85,"rationale":"Shape prior dominates offenders; reduce its weight to avoid over-constraining patient-specific morphology."}}

        user = {
            "last_outer_eval": worst["last_outer_eval"],
            "last_outer_log": worst["last_outer_log"],
            "offenders": offenders,
            "dominant_penalties": top3,
            "current_config_snippet": {
                "reward_weights": self.cfg["reward_weights"],
                "edge": self.cfg["edge"],
                "ops.smooth": self.cfg["ops"]["smooth"],
                "shape_atlas": self.cfg["shape_atlas"],
            }
        }
        payload = json.dumps(user, indent=2)
        out = self.client.chat_json(SYSTEM_PROMPT, payload,
                                   temperature=float(self.cfg["llm"]["temperature"]),
                                   max_tokens=int(self.cfg["llm"]["max_tokens"]))
        return out

    @staticmethod
    def apply_change(cfg: dict, change: dict) -> dict:
        # change: {"path":"a.b.c","value":...}
        path = change["path"]
        val = change["value"]
        keys = path.split(".")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur:
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = val
        return cfg
