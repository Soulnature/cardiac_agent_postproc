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
        llm_cfg = cfg.get("llm", {})
        if llm_cfg.get("enabled", False) or self.llm_settings.llm_enabled:
            # Use config-specified model/key, fallback to env
            model = llm_cfg.get("model", self.llm_settings.openai_model)
            api_key = llm_cfg.get("api_key", self.llm_settings.openai_api_key)
            self.client = OpenAICompatClient(
                base_url=self.llm_settings.openai_base_url,
                api_key=api_key,
                model=model,
            )
        
        # Load system prompt from file if available, else use default
        self.system_prompt = SYSTEM_PROMPT
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "system_prompt.txt")
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()

    def _load_target_reports(self, target_dir: str) -> pd.DataFrame:
        path = os.path.join(target_dir, "target_dice_report.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def _load_rqs_log(self, folder: str) -> pd.DataFrame:
        path = os.path.join(folder, "label_optimization_log.csv")
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path, on_bad_lines="skip")
        # Support new quality_score format: map to RQS_total for compatibility
        if "RQS_total" not in df.columns and "quality_score" in df.columns:
            df["RQS_total"] = pd.to_numeric(df["quality_score"], errors="coerce") * 100.0
        elif "RQS_total" in df.columns:
            df["RQS_total"] = pd.to_numeric(df["RQS_total"], errors="coerce")
        if "outer_iter" not in df.columns:
            df["outer_iter"] = 1
        else:
            df["outer_iter"] = pd.to_numeric(df["outer_iter"], errors="coerce")
        if "inner_step" not in df.columns:
            df["inner_step"] = 1
        return df

    def _worst_union(self, df_eval: pd.DataFrame, df_log: pd.DataFrame, k: int=20) -> dict:
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

        # worst by RQS (legacy) or by pixels_changed (new diagnosis-driven log)
        bottom_rqs = []
        last_outer_log = -1
        if not df_log.empty:
            if "RQS_total" in df_log.columns:
                if "outer_iter" in df_log.columns:
                    m = df_log["outer_iter"].max()
                    last_outer_log = int(m) if pd.notna(m) else -1
                dl = df_log[df_log["outer_iter"]==last_outer_log].copy() if last_outer_log>=0 else df_log
                if "inner_step" in dl.columns:
                    dl = dl.sort_values(["filename","inner_step"]).groupby("filename").tail(1)
                bottom_rqs = dl.nsmallest(k, "RQS_total")["filename"].tolist()
            elif "pixels_changed" in df_log.columns:
                # New format: files with most pixels changed are most modified
                per_file = df_log.groupby("filename")["pixels_changed"].sum().reset_index()
                bottom_rqs = per_file.nlargest(k, "pixels_changed")["filename"].tolist()

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
        if df_log.empty:
            return []

        # Legacy format with P_ columns
        pen_cols = [c for c in df_log.columns if c.startswith("P_")]
        if pen_cols:
            if "outer_iter" not in df_log.columns:
                last_outer_log = -1
            else:
                m = df_log["outer_iter"].max()
                last_outer_log = int(m) if pd.notna(m) else -1
            dl = df_log[df_log["outer_iter"]==last_outer_log].copy() if last_outer_log>=0 else df_log
            if "inner_step" in dl.columns:
                dl = dl.sort_values(["filename","inner_step"]).groupby("filename").tail(1)
            dl = dl[dl["filename"].isin(offenders)]
            means = {c: float(dl[c].mean()) for c in pen_cols}
            top3 = sorted(means.items(), key=lambda x: -x[1])[:3]
            return top3

        # New diagnosis-driven format: aggregate issues as pseudo-penalties
        if "issue" in df_log.columns:
            dl = df_log[df_log["filename"].isin(offenders)]
            issue_counts = dl.groupby("issue")["pixels_changed"].sum()
            top3 = [(f"issue_{k}", float(v)) for k, v in issue_counts.nlargest(3).items()]
            return top3

        return []

    def _load_vision_feedback(self, target_dir: str) -> List[str]:
        path = os.path.join(target_dir, "vision_feedback.csv")
        metrics = []
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Filter for rejected cases with non-empty suggestions
                rejected = df[df["accepted"].astype(str).str.lower() == "false"]
                if not rejected.empty and "suggestion" in rejected.columns:
                     # Take top 10 unique suggestions to avoid token overflow
                     suggs = rejected["suggestion"].dropna().unique().tolist()
                     metrics = suggs[:10]
            except Exception as e:
                print(f"[REVISION] Failed to load vision_feedback: {e}")
        return metrics

    def propose_change(self, target_dir: str) -> Dict[str, Any]:
        df_eval = self._load_target_reports(target_dir)
        df_log = self._load_rqs_log(target_dir)

        worst = self._worst_union(df_eval, df_log, k=20)
        offenders = worst["union"]
        top3 = self._dominant_penalties(df_log, offenders)
        
        visual_suggestions = self._load_vision_feedback(target_dir)

        print(f"[REVISION] overlap(bottomRQS & bottomDice & topHD95) = {worst['overlap_count']}")
        print(f"[REVISION] offenders union (n={len(offenders)}): {offenders[:30]}{'...' if len(offenders)>30 else ''}")
        print(f"[REVISION] dominant penalties (mean over offenders): {top3}")
        if visual_suggestions:
            print(f"[REVISION] Visual Suggestions (n={len(visual_suggestions)})")

        # If LLM disabled, use heuristic (ignoring visual for simple heuristic)
        if self.client is None:
            return self._heuristic_proposal(top3)

        user = {
            "last_outer_eval": worst["last_outer_eval"],
            "last_outer_log": worst["last_outer_log"],
            "offenders": offenders,
            "dominant_penalties": top3,
            "visual_guardrail_suggestions": visual_suggestions,
            "current_config_snippet": {
                "reward_weights": self.cfg["reward_weights"],
                "edge": self.cfg["edge"],
                "ops.smooth": self.cfg["ops"]["smooth"],
                "shape_atlas": self.cfg["shape_atlas"],
                "solver": self.cfg["solver"],
            }
        }
        payload = json.dumps(user, indent=2)
        
        try:
            out = self.client.chat_json(self.system_prompt, payload,
                                       temperature=float(self.cfg["llm"]["temperature"]),
                                       max_tokens=int(self.cfg["llm"]["max_tokens"]))
            return out
        except Exception as e:
            print(f"[REVISION] LLM call failed: {e}. Falling back to heuristics.")
            return self._heuristic_proposal(top3)

    @staticmethod
    def _heuristic_proposal(top3: List[Tuple[str,float]]) -> Dict[str, Any]:
        """Simple heuristic: reduce the weight of the dominant penalty by 20%."""
        if not top3:
            return {"change": {"path": "ops.smooth.morph_r", "value": 3, "rationale": "No penalties found; try larger smoothing radius."}}
        name, val = top3[0]
        if name.startswith("issue_"):
            # New diagnosis-driven format — suggest smoothing tweak
            return {"change": {"path": "ops.smooth.morph_r", "value": 3, "rationale": f"Dominant issue: {name} ({val:.0f}px). Try larger smoothing."}}
        # Legacy P_ penalty — reduce weight by 20%
        current = val
        new_val = round(current * 0.8, 2)
        return {"change": {"path": f"reward_weights.{name}", "value": new_val, "rationale": f"Reduce dominant penalty {name} from {current} to {new_val}."}}

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
