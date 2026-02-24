#!/usr/bin/env python3
"""
Calibrate verifier online score thresholds from historical repair CSV files.

Example:
  python scripts/calibrate_verifier_scores.py \
    --input_csv results/exp_mnm2/repair/need_fix_diagnosis_repair_results.csv \
    --out_csv results/exp_mnm2/summary/verifier_score_calibration.csv \
    --out_json results/exp_mnm2/summary/verifier_threshold_recommendation.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _is_finite(x: float) -> bool:
    return isinstance(x, (float, int)) and math.isfinite(float(x))


def _score_bin_table(df: pd.DataFrame, score_col: str, delta_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["score_num"] = pd.to_numeric(work[score_col], errors="coerce")
    work["delta_num"] = pd.to_numeric(work[delta_col], errors="coerce")
    work = work[np.isfinite(work["score_num"].to_numpy()) & np.isfinite(work["delta_num"].to_numpy())]
    if work.empty:
        return pd.DataFrame()

    bins = np.arange(0, 110, 10)
    labels = [f"{int(bins[i])}-{int(bins[i+1]-1)}" for i in range(len(bins) - 1)]
    work["score_bin"] = pd.cut(
        work["score_num"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    out = (
        work.groupby("score_bin", observed=False)
        .agg(
            n=("delta_num", "count"),
            mean_delta=("delta_num", "mean"),
            median_delta=("delta_num", "median"),
            improved_rate=("delta_num", lambda s: float((s > 0).mean()) if len(s) else float("nan")),
            degradation_rate=("delta_num", lambda s: float((s < 0).mean()) if len(s) else float("nan")),
        )
        .reset_index()
    )
    out["score_bin"] = out["score_bin"].astype(str)
    return out


def _recommend_thresholds(calib_df: pd.DataFrame, default_reject: float, default_approve: float) -> Dict[str, Any]:
    if calib_df.empty:
        return {
            "recommended_reject": float(default_reject),
            "recommended_approve": float(default_approve),
            "reason": "no usable rows",
        }

    approve = float(default_approve)
    for _, row in calib_df.iterrows():
        n = int(row.get("n", 0) or 0)
        if n < 3:
            continue
        mean_delta = _safe_float(row.get("mean_delta"))
        deg_rate = _safe_float(row.get("degradation_rate"))
        m = re.match(r"^(\d+)-(\d+)$", str(row.get("score_bin", "")))
        if not m:
            continue
        lo = float(m.group(1))
        if _is_finite(mean_delta) and _is_finite(deg_rate) and mean_delta > 0 and deg_rate <= 0.20:
            approve = lo
            break

    reject = float(default_reject)
    for _, row in calib_df.iterrows():
        n = int(row.get("n", 0) or 0)
        if n < 3:
            continue
        mean_delta = _safe_float(row.get("mean_delta"))
        m = re.match(r"^(\d+)-(\d+)$", str(row.get("score_bin", "")))
        if not m:
            continue
        hi = float(m.group(2))
        if hi >= approve:
            break
        if _is_finite(mean_delta) and mean_delta < 0:
            reject = hi + 1.0

    if approve < reject:
        approve = reject
    return {
        "recommended_reject": float(reject),
        "recommended_approve": float(approve),
        "reason": "bin reliability heuristic",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate verifier score thresholds from historical repair CSV files.")
    p.add_argument(
        "--input_csv",
        action="append",
        required=True,
        help="Repair result CSV path. Repeatable.",
    )
    p.add_argument("--score_col", default="online_score", help="Verifier score column.")
    p.add_argument("--delta_col", default="DiceDelta_Mean", help="Observed Dice delta column.")
    p.add_argument("--default_reject", type=float, default=45.0)
    p.add_argument("--default_approve", type=float, default=65.0)
    p.add_argument("--out_csv", required=True, help="Output reliability-table CSV path.")
    p.add_argument("--out_json", required=True, help="Output threshold recommendation JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames: List[pd.DataFrame] = []
    for path_str in args.input_csv:
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p}")
        df = pd.read_csv(p)
        if args.score_col not in df.columns or args.delta_col not in df.columns:
            raise ValueError(
                f"CSV missing required columns: {p} "
                f"(need {args.score_col}, {args.delta_col})"
            )
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    calib_df = _score_bin_table(merged, score_col=args.score_col, delta_col=args.delta_col)
    reco = _recommend_thresholds(
        calib_df,
        default_reject=float(args.default_reject),
        default_approve=float(args.default_approve),
    )
    reco["score_col"] = args.score_col
    reco["delta_col"] = args.delta_col
    reco["n_rows"] = int(len(merged))

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    calib_df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(reco, f, ensure_ascii=False, indent=2)

    print(f"Saved calibration CSV: {out_csv}")
    print(f"Saved recommendation JSON: {out_json}")
    print(json.dumps(reco, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
