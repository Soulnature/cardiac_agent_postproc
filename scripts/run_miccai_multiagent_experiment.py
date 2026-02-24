#!/usr/bin/env python3
"""
Unified MICCAI-style experiment runner for multi-agent post-processing.

Expected input layout (source_root):
  source_root/
    all_frames_export/
    best_cases/
    worst_cases/

Main outputs (target_dir):
  target_dir/
    triage/   -> triage outputs + curated-label classification metrics
    repair/   -> diagnosis + repaired labels + Dice improvement stats
    figures/  -> MICCAI-style plots (PNG + PDF)
    summary/  -> run summary JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, ttest_rel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

# Make Matplotlib cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cardiac_agent_postproc.agents import (  # noqa: E402
    AtlasBuilderAgent,
    CoordinatorAgent,
    DiagnosisAgent,
    ExecutorAgent,
    MessageBus,
    PlannerAgent,
    TriageAgent,
    VerifierAgent,
)
from cardiac_agent_postproc.agents.executor import (  # noqa: E402
    _add_atlas_ops,
    _build_op_fn,
)
from cardiac_agent_postproc.agents.visual_knowledge import (  # noqa: E402
    RepairComparisonKB,
    VisualKnowledgeBase,
)
from cardiac_agent_postproc.api_client import OpenAICompatClient  # noqa: E402
from cardiac_agent_postproc.atlas import build_atlas, load_atlas, save_atlas  # noqa: E402
from cardiac_agent_postproc.eval_metrics import dice_macro  # noqa: E402
from cardiac_agent_postproc.io_utils import read_image, read_mask, write_mask  # noqa: E402
from cardiac_agent_postproc.orchestrator import (  # noqa: E402
    discover_cases,
    group_and_link_neighbors,
)
from cardiac_agent_postproc.quality_model import (  # noqa: E402
    QUALITY_FEATURE_NAMES,
    extract_quality_features,
)
from cardiac_agent_postproc.rqs import sobel_edges, valve_plane_y  # noqa: E402
from cardiac_agent_postproc.settings import LLMSettings  # noqa: E402
from cardiac_agent_postproc.view_utils import infer_view_type  # noqa: E402
from cardiac_agent_postproc.vlm_guardrail import (  # noqa: E402
    create_diff_overlay,
    create_overlay,
)

STEM_REGEX = re.compile(r"(\d+_original_lax_[a-z0-9]+_\d+)", re.IGNORECASE)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _is_finite(x: float) -> bool:
    return isinstance(x, (float, int)) and math.isfinite(float(x))


def _parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    s = value.strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _normalize_view_for_thresholds(view_type: Optional[str]) -> str:
    token = re.sub(r"[^a-z0-9]", "", str(view_type or "").strip().lower())
    if token in {"2c", "2ch"}:
        return "2CH"
    if token in {"3c", "3ch"}:
        return "3CH"
    if token in {"4c", "4ch"}:
        return "4CH"
    return token.upper() if token else "ANY"


def _verifier_thresholds_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    verifier_cfg = cfg.get("verifier", {}) if isinstance(cfg.get("verifier"), dict) else {}
    agent_cfg = (
        cfg.get("agents", {}).get("verifier", {})
        if isinstance(cfg.get("agents", {}).get("verifier"), dict)
        else {}
    )
    base_thr = verifier_cfg.get("thresholds", {}) if isinstance(verifier_cfg.get("thresholds"), dict) else {}
    ag_thr = agent_cfg.get("thresholds", {}) if isinstance(agent_cfg.get("thresholds"), dict) else {}

    reject_th = float(base_thr.get("reject", 45.0))
    approve_th = float(base_thr.get("approve", 65.0))
    if "reject" in ag_thr:
        reject_th = float(ag_thr["reject"])
    if "approve" in ag_thr:
        approve_th = float(ag_thr["approve"])
    if approve_th < reject_th:
        approve_th = reject_th

    per_view: Dict[str, Dict[str, float]] = {}
    for view in ["2CH", "3CH", "4CH"]:
        view_cfg: Dict[str, Any] = {}
        base_per = base_thr.get("per_view", {}) if isinstance(base_thr, dict) else {}
        ag_per = ag_thr.get("per_view", {}) if isinstance(ag_thr, dict) else {}
        if isinstance(base_per, dict):
            view_cfg.update(base_per.get(view, {}) or {})
            view_cfg.update(base_per.get(view.lower(), {}) or {})
        if isinstance(ag_per, dict):
            view_cfg.update(ag_per.get(view, {}) or {})
            view_cfg.update(ag_per.get(view.lower(), {}) or {})
        v_reject = float(view_cfg.get("reject", reject_th))
        v_approve = float(view_cfg.get("approve", approve_th))
        if v_approve < v_reject:
            v_approve = v_reject
        per_view[view] = {"reject": v_reject, "approve": v_approve}

    return {
        "reject": float(reject_th),
        "approve": float(approve_th),
        "per_view": per_view,
    }


def _score_bin_table(
    repair_df: pd.DataFrame,
    score_col: str = "online_score",
) -> pd.DataFrame:
    if repair_df.empty or score_col not in repair_df.columns:
        return pd.DataFrame()

    work = repair_df.copy()
    work["score_num"] = pd.to_numeric(work[score_col], errors="coerce")
    work["dice_delta_num"] = pd.to_numeric(work["DiceDelta_Mean"], errors="coerce")
    work = work[np.isfinite(work["score_num"].to_numpy()) & np.isfinite(work["dice_delta_num"].to_numpy())]
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

    grp = (
        work.groupby("score_bin", observed=False)
        .agg(
            n=("dice_delta_num", "count"),
            mean_delta=("dice_delta_num", "mean"),
            median_delta=("dice_delta_num", "median"),
            improved_rate=("dice_delta_num", lambda s: float((s > 0).mean()) if len(s) else float("nan")),
            degradation_rate=("dice_delta_num", lambda s: float((s < 0).mean()) if len(s) else float("nan")),
        )
        .reset_index()
    )
    grp["score_bin"] = grp["score_bin"].astype(str)
    return grp


def _recommend_thresholds_from_bins(
    calib_df: pd.DataFrame,
    default_reject: float,
    default_approve: float,
) -> Dict[str, Any]:
    if calib_df.empty:
        return {
            "recommended_reject": float(default_reject),
            "recommended_approve": float(default_approve),
            "reason": "no calibration rows",
        }

    # approve: first bin with positive mean delta and low degradation risk.
    approve = float(default_approve)
    for _, row in calib_df.iterrows():
        n = int(row.get("n", 0) or 0)
        if n < 3:
            continue
        mean_delta = _safe_float(row.get("mean_delta", float("nan")))
        deg_rate = _safe_float(row.get("degradation_rate", float("nan")))
        bin_txt = str(row.get("score_bin", ""))
        m = re.match(r"^(\d+)-(\d+)$", bin_txt)
        if not m:
            continue
        lo = float(m.group(1))
        if _is_finite(mean_delta) and _is_finite(deg_rate) and mean_delta > 0 and deg_rate <= 0.20:
            approve = lo
            break

    # reject: last clearly negative bin before approve.
    reject = float(default_reject)
    for _, row in calib_df.iterrows():
        n = int(row.get("n", 0) or 0)
        if n < 3:
            continue
        mean_delta = _safe_float(row.get("mean_delta", float("nan")))
        bin_txt = str(row.get("score_bin", ""))
        m = re.match(r"^(\d+)-(\d+)$", bin_txt)
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


def _view_key(stem: str) -> str:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def _update_neighbor_masks_for_group(ctx, groups_by_view: Dict[str, List[Any]]) -> None:
    if ctx.current_mask is None:
        return
    key = _view_key(ctx.stem)
    siblings = groups_by_view.get(key, [])
    for other in siblings:
        if other.stem == ctx.stem:
            continue
        if ctx.slice_index >= 0:
            other.neighbor_masks[ctx.slice_index] = ctx.current_mask.copy()


def _apply_miccai_style() -> None:
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_fig(fig: plt.Figure, out_base: Path) -> Dict[str, str]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _compute_dice(mask: Optional[np.ndarray], gt_path: str) -> Dict[str, float]:
    if mask is None or not gt_path or not os.path.exists(gt_path):
        return {
            "Dice_Mean": float("nan"),
            "Dice_RV": float("nan"),
            "Dice_Myo": float("nan"),
            "Dice_LV": float("nan"),
        }
    gt = read_mask(gt_path)
    d_mean, d_rv, d_myo, d_lv = dice_macro(mask, gt)
    return {
        "Dice_Mean": float(d_mean),
        "Dice_RV": float(d_rv),
        "Dice_Myo": float(d_myo),
        "Dice_LV": float(d_lv),
    }


def _compute_anatomy_score_for_mask(ctx: Any, mask: Optional[np.ndarray]) -> float:
    """
    Compute anatomy no-GT score (view-aware) for a candidate mask.
    Returns NaN when prerequisites are missing.
    """
    if mask is None:
        return float("nan")
    if getattr(ctx, "image", None) is None:
        return float("nan")
    stats_dict = getattr(ctx, "anatomy_stats", None)
    if not isinstance(stats_dict, dict):
        return float("nan")
    try:
        from cardiac_agent_postproc.anatomy_score import compute_anatomy_score, _normalize_view

        view = _normalize_view(getattr(ctx, "view_type", "") or "")
        stats = stats_dict.get(view)
        if stats is None:
            return float("nan")
        score, _ = compute_anatomy_score(mask, ctx.image, view, stats)
        return float(score)
    except Exception:
        return float("nan")


def _blend_approved_pick_score(
    llm_score: float,
    anatomy_score: float,
    cfg: Dict[str, Any],
) -> float:
    """
    Final no-GT score used for approved-case mask selection.
    Blends verifier online score (LLM/VLM-driven) and anatomy score.
    """
    ma_cfg = cfg.get("multi_agent", {}) if isinstance(cfg, dict) else {}
    w_llm = _safe_float(ma_cfg.get("approved_pick_llm_weight", 0.60), 0.60)
    w_anatomy = _safe_float(ma_cfg.get("approved_pick_anatomy_weight", 0.40), 0.40)
    w_llm = max(0.0, w_llm)
    w_anatomy = max(0.0, w_anatomy)
    w_sum = max(1e-6, w_llm + w_anatomy)

    llm = llm_score if _is_finite(llm_score) else 50.0
    anat = anatomy_score if _is_finite(anatomy_score) else 50.0
    llm = float(max(0.0, min(100.0, llm)))
    anat = float(max(0.0, min(100.0, anat)))
    return float((w_llm * llm + w_anatomy * anat) / w_sum)


def _estimate_original_online_score(ctx: Any) -> float:
    """
    Estimate a no-GT online quality score for the original mask from triage score.
    """
    triage_score = _safe_float(getattr(ctx, "triage_score", float("nan")))
    if not _is_finite(triage_score):
        return 50.0
    if triage_score <= 1.0:
        return float(max(0.0, min(100.0, triage_score * 100.0)))
    return float(max(0.0, min(100.0, triage_score)))


def _select_approved_final_mask(
    ctx: Any,
    cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """
    For approved cases, choose between current_mask and best_intermediate_mask
    using blended no-GT score = f(LLM online score, anatomy score).
    """
    current_mask = (
        ctx.current_mask.copy() if getattr(ctx, "current_mask", None) is not None else None
    )
    best_mask = None
    if (
        getattr(ctx, "best_intermediate_mask", None) is not None
        and not bool(getattr(ctx, "best_intermediate_is_original", True))
    ):
        best_mask = ctx.best_intermediate_mask.copy()

    current_online = _safe_float(
        (getattr(ctx, "last_verify_result", {}) or {}).get("online_score", float("nan"))
    )
    current_anatomy = _compute_anatomy_score_for_mask(ctx, current_mask)
    current_blended = _blend_approved_pick_score(
        llm_score=current_online,
        anatomy_score=current_anatomy,
        cfg=cfg,
    )

    best_online = _safe_float(
        getattr(ctx, "best_intermediate_online_score", float("nan"))
    )
    if (not _is_finite(best_online)) and (
        int(getattr(ctx, "best_intermediate_round", 0))
        == int(getattr(ctx, "rounds_completed", 0))
    ):
        best_online = current_online
    best_anatomy = _safe_float(
        getattr(ctx, "best_intermediate_anatomy_score", float("nan"))
    )
    if not _is_finite(best_anatomy):
        best_anatomy = _compute_anatomy_score_for_mask(ctx, best_mask)
    best_blended = _blend_approved_pick_score(
        llm_score=best_online,
        anatomy_score=best_anatomy,
        cfg=cfg,
    )

    margin = _safe_float(
        (cfg.get("multi_agent", {}) or {}).get("approved_pick_min_margin", 1e-6),
        1e-6,
    )
    pick_best = best_mask is not None and (best_blended > (current_blended + margin))
    if pick_best:
        selected_mask = best_mask
        saved_mask_type = "best_intermediate"
        selected_source = "best_intermediate"
    else:
        selected_mask = current_mask
        saved_mask_type = "repaired"
        selected_source = "current"

    meta = {
        "selected_source": selected_source,
        "current_online_score": current_online,
        "current_anatomy_score": current_anatomy,
        "current_blended_score": current_blended,
        "best_online_score": best_online,
        "best_anatomy_score": best_anatomy,
        "best_blended_score": best_blended,
    }
    return selected_mask, saved_mask_type, meta


def _select_nonapproved_final_mask(
    ctx: Any,
    cfg: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], str, Dict[str, Any]]:
    """
    For non-approved outcomes (gave_up/reject/needs_more_work),
    prioritize the intermediate with the largest anatomy-score improvement.
    """
    original_mask = None
    if getattr(ctx, "original_mask", None) is not None:
        original_mask = ctx.original_mask.copy()
    elif getattr(ctx, "mask", None) is not None:
        original_mask = ctx.mask.copy()

    best_mask = None
    best_source = "none"
    if (
        getattr(ctx, "best_anatomy_intermediate_mask", None) is not None
        and not bool(getattr(ctx, "best_anatomy_intermediate_is_original", True))
    ):
        best_mask = ctx.best_anatomy_intermediate_mask.copy()
        best_source = "best_anatomy_intermediate"
    if (
        best_mask is None
        and getattr(ctx, "best_intermediate_mask", None) is not None
        and not bool(getattr(ctx, "best_intermediate_is_original", True))
    ):
        best_mask = ctx.best_intermediate_mask.copy()
        best_source = "best_intermediate"

    original_online = _estimate_original_online_score(ctx)
    original_anatomy = _safe_float(getattr(ctx, "anatomy_score_before", float("nan")))
    if not _is_finite(original_anatomy):
        original_anatomy = _compute_anatomy_score_for_mask(ctx, original_mask)
    original_blended = _blend_approved_pick_score(
        llm_score=original_online,
        anatomy_score=original_anatomy,
        cfg=cfg,
    )

    best_online = _safe_float(
        getattr(ctx, "best_anatomy_intermediate_online_score", float("nan"))
    )
    if not _is_finite(best_online):
        best_online = _safe_float(
            getattr(ctx, "best_intermediate_online_score", float("nan"))
        )
    best_anatomy = _safe_float(
        getattr(ctx, "best_anatomy_intermediate_score", float("nan"))
    )
    if not _is_finite(best_anatomy):
        best_anatomy = _safe_float(
            getattr(ctx, "best_intermediate_anatomy_score", float("nan"))
        )
    if not _is_finite(best_anatomy):
        best_anatomy = _compute_anatomy_score_for_mask(ctx, best_mask)
    best_blended = _blend_approved_pick_score(
        llm_score=best_online,
        anatomy_score=best_anatomy,
        cfg=cfg,
    )

    ma_cfg = cfg.get("multi_agent", {}) if isinstance(cfg, dict) else {}
    min_margin = _safe_float(ma_cfg.get("nonapproved_pick_min_margin", 0.25), 0.25)
    min_online_gain = _safe_float(ma_cfg.get("nonapproved_pick_min_online_gain", 0.0), 0.0)
    min_anatomy_gain = _safe_float(ma_cfg.get("nonapproved_pick_min_anatomy_gain", 0.0), 0.0)

    online_gain_ok = _is_finite(best_online) and (
        best_online >= (original_online + min_online_gain)
    )
    if _is_finite(best_anatomy) and _is_finite(original_anatomy):
        anatomy_gain_ok = best_anatomy >= (original_anatomy + min_anatomy_gain)
    else:
        anatomy_gain_ok = _is_finite(best_anatomy)
    blended_gain_ok = _is_finite(best_blended) and _is_finite(original_blended) and (
        best_blended > (original_blended + min_margin)
    )

    # User policy: nonapproved path prioritizes anatomy-improvement maximum.
    pick_best = best_mask is not None and anatomy_gain_ok

    if pick_best:
        selected_mask = best_mask
        saved_mask_type = "best_intermediate"
        selected_source = f"{best_source}_nonapproved_anatomy_priority"
    else:
        selected_mask = original_mask
        saved_mask_type = "original"
        selected_source = "original_nonapproved"

    meta = {
        "selected_source": selected_source,
        "current_online_score": original_online,
        "current_anatomy_score": original_anatomy,
        "current_blended_score": original_blended,
        "best_online_score": best_online,
        "best_anatomy_score": best_anatomy,
        "best_blended_score": best_blended,
        "nonapproved_online_gain_ok": bool(online_gain_ok),
        "nonapproved_anatomy_gain_ok": bool(anatomy_gain_ok),
        "nonapproved_blended_gain_ok": bool(blended_gain_ok),
        "nonapproved_policy": "anatomy_priority",
    }
    return selected_mask, saved_mask_type, meta


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_source_root(source_root: Path) -> Tuple[Path, Path, Path]:
    all_frames = source_root / "all_frames_export"
    best_cases = source_root / "best_cases"
    worst_cases = source_root / "worst_cases"
    missing = [p for p in [all_frames, best_cases, worst_cases] if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"source_root must contain all_frames_export, best_cases, worst_cases. Missing: {missing_str}"
        )
    return all_frames, best_cases, worst_cases


def _known_stems_from_all_frames(all_frames_dir: Path) -> List[str]:
    stems = []
    for name in os.listdir(all_frames_dir):
        if not name.endswith("_pred.png"):
            continue
        stem = name[:-9]
        img = all_frames_dir / f"{stem}_img.png"
        if img.exists():
            stems.append(stem)
    return sorted(set(stems))


def _match_stem_from_filename(filename: str, known_stems: List[str]) -> Optional[str]:
    m = STEM_REGEX.search(filename)
    if m:
        stem = m.group(1)
        if stem in known_stems:
            return stem
    hits = [s for s in known_stems if s in filename]
    if not hits:
        return None
    # Prefer the most specific/longest match if multiple are found.
    hits.sort(key=len, reverse=True)
    return hits[0]


def _load_curated_labels(
    best_cases_dir: Path,
    worst_cases_dir: Path,
    known_stems: List[str],
) -> Tuple[Dict[str, str], Dict[str, int], List[str]]:
    labels: Dict[str, str] = {}
    unresolved: List[str] = []
    counts = {"best_files": 0, "worst_files": 0, "resolved_best": 0, "resolved_worst": 0}

    def ingest(folder: Path, label: str, count_key: str) -> None:
        for fname in sorted(os.listdir(folder)):
            fpath = folder / fname
            if not fpath.is_file():
                continue
            counts[count_key] += 1
            stem = _match_stem_from_filename(fname, known_stems)
            if stem is None:
                unresolved.append(str(fpath))
                continue
            # Keep worst-case label if conflicts appear.
            if stem in labels and labels[stem] != label:
                labels[stem] = "needs_fix"
            else:
                labels[stem] = label
            if label == "good":
                counts["resolved_best"] += 1
            else:
                counts["resolved_worst"] += 1

    ingest(best_cases_dir, "good", "best_files")
    ingest(worst_cases_dir, "needs_fix", "worst_files")
    return labels, counts, unresolved


def _prepare_output_dirs(target_dir: Path) -> Dict[str, Path]:
    out = {
        "root": target_dir,
        "triage": target_dir / "triage",
        "repair": target_dir / "repair",
        "figures": target_dir / "figures",
        "summary": target_dir / "summary",
        "triage_debug": target_dir / "triage" / "debug_overlays",
        "repair_debug": target_dir / "repair" / "debug_overlays",
        "repaired_labels": target_dir / "repair" / "repaired_labels",
        "artifacts": target_dir / "artifacts",
        "artifacts_atlas": target_dir / "artifacts" / "atlas",
        "artifacts_models": target_dir / "artifacts" / "models",
        "artifacts_repair_kb": target_dir / "artifacts" / "repair_kb",
        "artifacts_configs": target_dir / "artifacts" / "config",
        "artifacts_runtime": target_dir / "artifacts" / "runtime",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _clear_dir(dir_path: Path) -> None:
    if not dir_path.exists():
        return
    for child in dir_path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def _safe_link_or_copy(src: Path, dst: Path, copy_mode: str = "symlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst, ignore_errors=True)
        else:
            dst.unlink()
    if copy_mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return

    rel = os.path.relpath(src, dst.parent)
    try:
        dst.symlink_to(rel, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _extract_dice_prefix(filename: str) -> float:
    m = re.match(r"^(\d+\.\d+)_", filename)
    if not m:
        return float("nan")
    try:
        return float(m.group(1))
    except Exception:
        return float("nan")


def _parse_issue_from_curated_filename(filename: str) -> str:
    # Expected common format:
    #   <dice>_<view>_<issue>_<stem>.png
    m = re.match(
        r"^\d+\.\d+_[^_]+_([^_]+)_\d+_original_lax_[a-z0-9]+_\d+\.png$",
        filename,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    return ""


def _infer_view_from_stem(stem: str) -> str:
    s = stem.lower()
    if "_lax_4c_" in s:
        return "4C"
    if "_lax_3c_" in s:
        return "3C"
    if "_lax_2c_" in s:
        return "2C"
    return ""


def _materialize_overlay_for_stem(
    stem: str,
    source_root: Path,
    all_frames_dir: Path,
    overlays_dir: Path,
    copy_mode: str = "symlink",
) -> str:
    overlays_dir.mkdir(parents=True, exist_ok=True)
    out_overlay = overlays_dir / f"{stem}_overlay.png"

    single_panel = source_root / "single_panel_refs" / f"{stem}_overlay.png"
    if single_panel.exists():
        _safe_link_or_copy(single_panel, out_overlay, copy_mode=copy_mode)
        return str(out_overlay)

    img_path = all_frames_dir / f"{stem}_img.png"
    pred_path = all_frames_dir / f"{stem}_pred.png"
    if not (img_path.exists() and pred_path.exists()):
        return ""
    try:
        pred = read_mask(str(pred_path))
        ok = create_overlay(str(img_path), pred, str(out_overlay))
        return str(out_overlay) if ok else ""
    except Exception:
        return ""


def _repair_kb_counts(kb_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cat in ["improved", "degraded", "neutral"]:
        cat_dir = kb_dir / cat
        counts[cat] = (
            len([p for p in cat_dir.glob("*.png") if p.is_file()]) if cat_dir.is_dir() else 0
        )
    counts["total"] = counts["improved"] + counts["degraded"] + counts["neutral"]
    return counts


def _repair_kb_has_any(kb_dir: Path) -> bool:
    counts = _repair_kb_counts(kb_dir)
    if counts["total"] > 0:
        return True
    return (kb_dir / "repair_kb_build_summary.json").exists()


def _repair_kb_usable(kb_dir: Path) -> bool:
    counts = _repair_kb_counts(kb_dir)
    return counts["improved"] >= 1 and counts["degraded"] >= 1


def _sanitize_mask(mask: Any) -> Optional[np.ndarray]:
    if not isinstance(mask, np.ndarray):
        return None
    if mask.ndim != 2:
        return None
    out = mask.astype(np.int32, copy=False)
    out = np.clip(out, 0, 3)
    return out


def _normalize_issue_key(issue: Any) -> str:
    token = re.sub(r"[^a-z0-9_]+", "_", str(issue or "").strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _collect_case_dice_scores(cases: List[Any]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ctx in cases:
        if not ctx.gt_path or not os.path.exists(ctx.gt_path):
            continue
        try:
            d = _compute_dice(ctx.mask, ctx.gt_path)["Dice_Mean"]
            if _is_finite(d):
                scores[ctx.stem] = float(d)
        except Exception:
            continue
    return scores


def _select_repair_kb_stems(
    cases: List[Any],
    curated_labels: Dict[str, str],
    max_cases: int,
    min_dice: float = 0.70,
    max_dice: float = 0.90,
    dice_scores: Optional[Dict[str, float]] = None,
) -> List[str]:
    max_cases = max(1, int(max_cases))
    by_stem = {c.stem: c for c in cases}
    if dice_scores is None:
        dice_scores = _collect_case_dice_scores(cases)

    scored: List[Tuple[float, str]] = [
        (float(d), stem) for stem, d in dice_scores.items() if stem in by_stem
    ]
    scored.sort(key=lambda x: x[0])

    if not scored:
        return []

    curated_bad = {
        s
        for s, lab in curated_labels.items()
        if lab == "needs_fix" and s in by_stem and by_stem[s].gt_path
    }
    banded = [(d, stem) for d, stem in scored if float(min_dice) < d < float(max_dice)]

    if banded:
        selected: List[str] = []
        banded_curated = [(d, stem) for d, stem in banded if stem in curated_bad]
        selected.extend([stem for _, stem in banded_curated[:max_cases]])
        if len(selected) < max_cases:
            for _d, stem in banded:
                if stem in selected:
                    continue
                selected.append(stem)
                if len(selected) >= max_cases:
                    break
        return selected[:max_cases]

    curated_scored = [(d, stem) for d, stem in scored if stem in curated_bad]
    if curated_scored:
        return [stem for _, stem in curated_scored[:max_cases]]

    return [stem for _, stem in scored[:max_cases]]


def _infer_case_issues_for_repair_kb(ctx: Any, diagnosis_agent: DiagnosisAgent) -> List[str]:
    issues: List[str] = []
    try:
        feats = extract_quality_features(ctx.mask, ctx.image, ctx.view_type)
        rule_diags = diagnosis_agent._diagnose_rules(
            ctx.mask,
            feats,
            ctx.view_type,
            triage_issues=["lowdice"],
            neighbor_masks=ctx.neighbor_masks,
        )
        for d in rule_diags:
            issue = _normalize_issue_key(getattr(d, "issue", ""))
            if issue:
                issues.append(issue)
    except Exception:
        issues = []

    if not issues:
        return ["generic_quality_issue"]

    dedup: List[str] = []
    seen = set()
    for it in issues:
        if it in seen:
            continue
        seen.add(it)
        dedup.append(it)
    return dedup


def _new_op_stat_bucket() -> Dict[str, Any]:
    return {
        "trials": 0,
        "improved": 0,
        "degraded": 0,
        "neutral": 0,
        "delta_sum": 0.0,
        "deltas": [],
    }


def _accumulate_issue_op_trial(
    collector: Dict[str, Dict[str, Dict[str, Any]]],
    issue: str,
    op_name: str,
    delta: float,
    category: str,
) -> None:
    issue_key = _normalize_issue_key(issue)
    if not issue_key:
        return
    op_key = str(op_name).strip()
    if not op_key:
        return
    by_op = collector.setdefault(issue_key, {})
    row = by_op.setdefault(op_key, _new_op_stat_bucket())
    row["trials"] += 1
    row["delta_sum"] += float(delta)
    row["deltas"].append(float(delta))
    if category in {"improved", "degraded", "neutral"}:
        row[category] += 1


def _finalize_issue_op_rankings(
    collector: Dict[str, Dict[str, Dict[str, Any]]],
    top_k: int = 12,
) -> Dict[str, Dict[str, Any]]:
    finalized: Dict[str, Dict[str, Any]] = {}
    for issue, op_rows in collector.items():
        ranked: List[Dict[str, Any]] = []
        for op_name, row in op_rows.items():
            trials = int(row.get("trials", 0))
            if trials <= 0:
                continue
            deltas = row.get("deltas", [])
            mean_delta = float(row.get("delta_sum", 0.0)) / float(trials)
            median_delta = float(np.median(deltas)) if deltas else mean_delta
            improved = int(row.get("improved", 0))
            degraded = int(row.get("degraded", 0))
            neutral = int(row.get("neutral", 0))
            ranked.append(
                {
                    "op": op_name,
                    "support": trials,
                    "mean_delta": mean_delta,
                    "median_delta": median_delta,
                    "improved_count": improved,
                    "degraded_count": degraded,
                    "neutral_count": neutral,
                    "improved_rate": float(improved) / float(trials),
                    "degraded_rate": float(degraded) / float(trials),
                    "neutral_rate": float(neutral) / float(trials),
                }
            )
        ranked.sort(
            key=lambda r: (
                -float(r["mean_delta"]),
                -float(r["improved_rate"]),
                float(r["degraded_rate"]),
                -int(r["support"]),
                str(r["op"]),
            )
        )
        finalized[issue] = {
            "ops": ranked[: max(1, int(top_k))],
            "n_ops_total": int(len(ranked)),
            "n_trials_total": int(sum(int(x["support"]) for x in ranked)),
        }
    return finalized


def _build_repair_kb_dataset_signature(
    cases: List[Any],
    curated_labels: Dict[str, str],
    max_cases: int,
    min_dice: float,
    max_dice: float,
) -> Dict[str, Any]:
    dice_scores = _collect_case_dice_scores(cases)
    selected_stems = _select_repair_kb_stems(
        cases,
        curated_labels,
        max_cases=max_cases,
        min_dice=min_dice,
        max_dice=max_dice,
        dice_scores=dice_scores,
    )

    by_stem = {c.stem: c for c in cases}
    records: List[Dict[str, Any]] = []
    for stem in selected_stems:
        ctx = by_stem.get(stem)
        if ctx is None:
            continue
        row: Dict[str, Any] = {
            "stem": stem,
            "dice": round(float(dice_scores[stem]), 6) if stem in dice_scores else None,
        }
        for key, path_str in [("pred", getattr(ctx, "pred_path", "")), ("gt", getattr(ctx, "gt_path", ""))]:
            if not path_str:
                continue
            p = Path(path_str)
            if not p.exists():
                continue
            st = p.stat()
            row[f"{key}_size"] = int(st.st_size)
            row[f"{key}_mtime_ns"] = int(st.st_mtime_ns)
        records.append(row)

    payload = {
        "schema_version": 1,
        "case_count": int(len(cases)),
        "gt_case_count": int(len(dice_scores)),
        "max_cases": int(max_cases),
        "min_dice": float(min_dice),
        "max_dice": float(max_dice),
        "selected_stems": [str(x) for x in selected_stems],
        "selected_records": records,
        "all_stems_sha16": hashlib.sha256(
            "|".join(sorted(str(c.stem) for c in cases)).encode("utf-8")
        ).hexdigest()[:16],
    }
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    signature = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return {"signature": signature, "payload": payload}


def _read_existing_repair_kb_signature(kb_dir: Path) -> Dict[str, Any]:
    sig_path = kb_dir / "repair_kb_dataset_signature.json"
    if not sig_path.exists():
        return {}
    try:
        with sig_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_repair_kb_signature(kb_dir: Path, signature_info: Dict[str, Any]) -> None:
    sig_path = kb_dir / "repair_kb_dataset_signature.json"
    payload = {
        "schema_version": 1,
        "signature": str(signature_info.get("signature", "")),
        "payload": signature_info.get("payload", {}),
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with sig_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _build_repair_kb_from_cases(
    cases: List[Any],
    curated_labels: Dict[str, str],
    kb_dir: Path,
    cfg: Dict[str, Any],
    max_cases: int = 10,
    max_ops: int = 18,
    improved_thr: float = 0.005,
    degraded_thr: float = -0.02,
    min_dice: float = 0.70,
    max_dice: float = 0.90,
    atlas: Optional[dict] = None,
) -> Dict[str, Any]:
    kb_dir.mkdir(parents=True, exist_ok=True)
    _clear_dir(kb_dir)
    for cat in ["improved", "degraded", "neutral"]:
        (kb_dir / cat).mkdir(parents=True, exist_ok=True)

    dice_scores = _collect_case_dice_scores(cases)
    selected_stems = _select_repair_kb_stems(
        cases,
        curated_labels,
        max_cases=max_cases,
        min_dice=min_dice,
        max_dice=max_dice,
        dice_scores=dice_scores,
    )
    by_stem = {c.stem: c for c in cases}
    groups = group_and_link_neighbors(cases)
    del groups  # group_and_link_neighbors mutates case contexts with neighbor maps

    # Rule-only diagnosis for case-level issue tags used by dynamic Planner KB.
    diagnosis_agent = DiagnosisAgent(cfg=cfg, bus=MessageBus())
    issue_op_acc: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    view_issue_op_acc: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(dict)

    op_priority = [
        "myo_bridge",
        "lv_bridge",
        "fill_holes",
        "fill_holes_m2",
        "smooth_morph",
        "smooth_contour",
        "remove_islands",
        "2_dilate",
        "2_erode",
        "3_dilate",
        "3_erode",
        "convex_hull_fill",
        "neighbor_shape_prior",
        "neighbor_area_consistency",
        "neighbor_rv_repair",
        "neighbor_myo_repair",
        "neighbor_lv_repair",
        "atlas_growprune_0",
        "atlas_rv_repair",
        "atlas_myo_repair",
        "atlas_lv_repair",
    ]

    build_stats = {
        "selected_case_count": int(len(selected_stems)),
        "op_trials": 0,
        "saved_examples": 0,
        "skipped_cases": 0,
        "errors": 0,
        "issue_stats_entries": 0,
    }

    for stem in selected_stems:
        ctx = by_stem.get(stem)
        if ctx is None:
            continue
        if not ctx.gt_path or not os.path.exists(ctx.gt_path):
            build_stats["skipped_cases"] += 1
            continue
        try:
            gt = read_mask(ctx.gt_path)
            before = _sanitize_mask(ctx.mask)
            if before is None:
                build_stats["skipped_cases"] += 1
                continue
            dice_before = float(dice_macro(before, gt)[0])
            case_issues = _infer_case_issues_for_repair_kb(ctx, diagnosis_agent=diagnosis_agent)
            view_key = _normalize_view_for_thresholds(ctx.view_type)
            sobel_pct = float(cfg.get("edge", {}).get("sobel_percentile", 85.0))
            E = sobel_edges(ctx.image, percentile=sobel_pct)
            vy = valve_plane_y(before)

            op_fns = _build_op_fn(
                before,
                ctx.view_type,
                cfg,
                E,
                vy,
                atlas=atlas,
                neighbor_masks=ctx.neighbor_masks,
                neighbor_img_paths=ctx.neighbor_img_paths,
                target_img=ctx.image,
                diagnosis_issues=[],
            )
            if atlas is not None:
                _add_atlas_ops(op_fns, before, ctx.view_type, atlas, cfg)

            ordered = [name for name in op_priority if name in op_fns]
            ordered.extend([name for name in sorted(op_fns.keys()) if name not in ordered])
            ordered = ordered[: max(1, int(max_ops))]

            for op_name in ordered:
                op_fn = op_fns.get(op_name)
                if op_fn is None:
                    continue
                build_stats["op_trials"] += 1
                try:
                    cand_raw = op_fn(before.copy())
                    if isinstance(cand_raw, list):
                        cand_raw = cand_raw[0] if cand_raw else None
                    cand = _sanitize_mask(cand_raw)
                    if cand is None:
                        continue
                    if cand.shape != before.shape:
                        continue
                    if np.array_equal(cand, before):
                        continue

                    dice_after = float(dice_macro(cand, gt)[0])
                    delta = dice_after - dice_before
                    if delta > improved_thr:
                        cat = "improved"
                    elif delta < degraded_thr:
                        cat = "degraded"
                    else:
                        cat = "neutral"

                    _accumulate_issue_op_trial(issue_op_acc, "global", op_name, delta, cat)
                    for issue in case_issues:
                        _accumulate_issue_op_trial(issue_op_acc, issue, op_name, delta, cat)
                        per_view_issue = view_issue_op_acc.setdefault(view_key, {})
                        _accumulate_issue_op_trial(per_view_issue, issue, op_name, delta, cat)

                    safe_op = re.sub(r"[^A-Za-z0-9_.-]+", "_", op_name)
                    fname = f"{delta:+.4f}_{safe_op}_{stem}.png"
                    out_path = kb_dir / cat / fname
                    ok = create_diff_overlay(ctx.img_path, before, cand, str(out_path))
                    if ok:
                        build_stats["saved_examples"] += 1
                except Exception:
                    build_stats["errors"] += 1
                    continue
        except Exception:
            build_stats["errors"] += 1
            continue

    all_issue_stats = _finalize_issue_op_rankings(issue_op_acc, top_k=12)
    global_ops = all_issue_stats.get("global", {}).get("ops", [])
    issue_stats = {
        issue: payload
        for issue, payload in all_issue_stats.items()
        if issue != "global"
    }
    view_issue_stats: Dict[str, Any] = {}
    for view_key, per_view_collector in view_issue_op_acc.items():
        view_issue_stats[str(view_key)] = _finalize_issue_op_rankings(
            per_view_collector, top_k=12
        )

    issue_stats_payload = {
        "schema_version": 1,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection": {
            "selected_stems": selected_stems,
            "selected_case_count": int(len(selected_stems)),
            "max_cases": int(max_cases),
        },
        "thresholds": {
            "min_dice": float(min_dice),
            "max_dice": float(max_dice),
            "improved_thr": float(improved_thr),
            "degraded_thr": float(degraded_thr),
        },
        "global_ops": global_ops,
        "issue_stats": issue_stats,
        "view_issue_stats": view_issue_stats,
    }
    with (kb_dir / "issue_op_stats.json").open("w", encoding="utf-8") as f:
        json.dump(issue_stats_payload, f, ensure_ascii=False, indent=2)

    issue_rows: List[Dict[str, Any]] = []
    for issue, payload in issue_stats.items():
        for rank, row in enumerate(payload.get("ops", []), start=1):
            issue_rows.append(
                {
                    "view_type": "ANY",
                    "issue": issue,
                    "rank": int(rank),
                    **row,
                }
            )
    for view_key, per_view_payload in view_issue_stats.items():
        for issue, payload in per_view_payload.items():
            for rank, row in enumerate(payload.get("ops", []), start=1):
                issue_rows.append(
                    {
                        "view_type": str(view_key),
                        "issue": issue,
                        "rank": int(rank),
                        **row,
                    }
                )
    if issue_rows:
        pd.DataFrame(issue_rows).to_csv(kb_dir / "issue_op_stats.csv", index=False)

    build_stats["issue_stats_entries"] = int(len(issue_rows))
    counts = _repair_kb_counts(kb_dir)
    build_info = {
        **build_stats,
        "counts": counts,
        "selected_stems": selected_stems,
        "dice_scores_selected": {
            stem: round(float(dice_scores.get(stem, float("nan"))), 6)
            for stem in selected_stems
            if stem in dice_scores
        },
        "min_dice": float(min_dice),
        "max_dice": float(max_dice),
        "issue_op_stats_path": str(kb_dir / "issue_op_stats.json"),
        "usable": _repair_kb_usable(kb_dir),
    }
    with (kb_dir / "repair_kb_build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(build_info, f, ensure_ascii=False, indent=2)
    return build_info


def _prepare_repair_kb(
    mode: str,
    kb_dir: Path,
    cases: List[Any],
    curated_labels: Dict[str, str],
    cfg: Dict[str, Any],
    max_cases: int = 10,
    max_ops: int = 18,
    improved_thr: float = 0.005,
    degraded_thr: float = -0.02,
    min_dice: float = 0.70,
    max_dice: float = 0.90,
    atlas: Optional[dict] = None,
    skip_if_exists: bool = False,
) -> Dict[str, Any]:
    kb_dir.mkdir(parents=True, exist_ok=True)
    existing_counts = _repair_kb_counts(kb_dir)
    has_any_existing = _repair_kb_has_any(kb_dir)
    has_existing = _repair_kb_usable(kb_dir)
    current_signature = _build_repair_kb_dataset_signature(
        cases=cases,
        curated_labels=curated_labels,
        max_cases=max_cases,
        min_dice=min_dice,
        max_dice=max_dice,
    )
    existing_signature_data = _read_existing_repair_kb_signature(kb_dir)
    existing_signature = str(existing_signature_data.get("signature", ""))
    signature_match = bool(existing_signature) and existing_signature == str(
        current_signature.get("signature", "")
    )

    info: Dict[str, Any] = {
        "mode": mode,
        "path": str(kb_dir),
        "used_existing": False,
        "built": False,
        "counts_before": existing_counts,
        "counts_after": existing_counts,
        "usable_after": has_existing,
        "has_any_before": has_any_existing,
        "skip_if_exists": bool(skip_if_exists),
        "dataset_signature": str(current_signature.get("signature", "")),
        "existing_dataset_signature": existing_signature,
        "signature_match": bool(signature_match),
        "dataset_changed": bool(has_any_existing and not signature_match),
        "min_dice": float(min_dice),
        "max_dice": float(max_dice),
    }

    if mode == "disable":
        return info

    if mode == "reuse":
        info["used_existing"] = has_existing
        info["usable_after"] = has_existing
        return info

    if mode == "build_if_missing":
        if skip_if_exists and has_any_existing and signature_match:
            info["used_existing"] = True
            info["usable_after"] = has_existing
            return info
        if has_existing and signature_match:
            info["used_existing"] = True
            info["usable_after"] = True
            return info

    build_info = _build_repair_kb_from_cases(
        cases=cases,
        curated_labels=curated_labels,
        kb_dir=kb_dir,
        cfg=cfg,
        max_cases=max_cases,
        max_ops=max_ops,
        improved_thr=improved_thr,
        degraded_thr=degraded_thr,
        min_dice=min_dice,
        max_dice=max_dice,
        atlas=atlas,
    )
    _write_repair_kb_signature(kb_dir, current_signature)
    info["built"] = True
    info["build_info"] = build_info
    info["counts_after"] = _repair_kb_counts(kb_dir)
    info["usable_after"] = _repair_kb_usable(kb_dir)
    info["signature_match_after_build"] = True
    return info


def _collect_quality_training_table(
    all_frames_dir: Path,
    curated_labels: Dict[str, str],
    bad_dice_threshold: float = 0.90,
    label_mode: str = "curated_only",
    max_cases: Optional[int] = None,
) -> pd.DataFrame:
    stems = _known_stems_from_all_frames(all_frames_dir)
    rows: List[Dict[str, Any]] = []

    if label_mode not in {"curated_only", "dice_with_curated_override"}:
        raise ValueError(
            f"Unsupported quality label mode: {label_mode}. "
            "Expected curated_only or dice_with_curated_override."
        )

    for stem in stems:
        img_path = all_frames_dir / f"{stem}_img.png"
        pred_path = all_frames_dir / f"{stem}_pred.png"
        gt_path = all_frames_dir / f"{stem}_gt.png"
        if not (img_path.exists() and pred_path.exists()):
            continue
        try:
            pred = read_mask(str(pred_path))
            img = read_image(str(img_path))
            view_type, _ = infer_view_type(stem, pred, rv_ratio_threshold=0.02)
            feats = extract_quality_features(pred, img, view_type)

            curated = curated_labels.get(stem)
            d_mean = float("nan")
            label_source = "unknown"

            if label_mode == "curated_only":
                if curated not in {"good", "needs_fix"}:
                    continue
                y_bad = 1 if curated == "needs_fix" else 0
                label_source = "curated"
                if gt_path.exists():
                    try:
                        gt = read_mask(str(gt_path))
                        d_mean = float(dice_macro(pred, gt)[0])
                    except Exception:
                        d_mean = float("nan")
            else:
                if not gt_path.exists():
                    continue
                gt = read_mask(str(gt_path))
                d_mean = float(dice_macro(pred, gt)[0])
                y_bad = int(d_mean < bad_dice_threshold)
                label_source = "dice"
                if curated == "good":
                    y_bad = 0
                    label_source = "curated"
                elif curated == "needs_fix":
                    y_bad = 1
                    label_source = "curated"

            row = {
                "stem": stem,
                "dice_mean": d_mean,
                "y_bad": y_bad,
                "label_source": label_source,
            }
            for key in QUALITY_FEATURE_NAMES:
                row[key] = float(feats.get(key, 0.0))
            rows.append(row)
        except Exception:
            continue

    if max_cases is not None:
        rows = rows[: max(1, int(max_cases))]
    return pd.DataFrame(rows)


def _train_quality_model(
    train_df: pd.DataFrame,
    output_path: Path,
    bad_dice_threshold: float = 0.90,
    classifier_threshold: float = 0.5,
) -> Dict[str, Any]:
    if train_df.empty:
        raise RuntimeError("No training rows available for quality model.")
    if len(train_df) < 8:
        raise RuntimeError(f"Insufficient training rows for quality model: {len(train_df)}")

    X = train_df[QUALITY_FEATURE_NAMES].astype(np.float32).to_numpy()
    y_bad = train_df["y_bad"].astype(np.int32).to_numpy()
    if "dice_mean" in train_df.columns:
        y_dice_raw = train_df["dice_mean"].astype(np.float32).to_numpy()
    else:
        y_dice_raw = np.full(len(train_df), np.nan, dtype=np.float32)
    y_dice = np.where(np.isfinite(y_dice_raw), y_dice_raw, 1.0 - y_bad.astype(np.float32))
    y_dice = np.clip(y_dice, 0.0, 1.0).astype(np.float32)
    n_dice_available = int(np.isfinite(y_dice_raw).sum())
    n_dice_missing = int(len(y_dice_raw) - n_dice_available)
    reg_target_mode = "true_dice" if n_dice_missing == 0 else "dice_plus_curated_fallback"

    reg = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    reg.fit(X, y_dice)
    pred_dice = reg.predict(X)
    rmse = float(math.sqrt(mean_squared_error(y_dice, pred_dice)))

    clf = None
    use_classifier = False
    clf_auc = float("nan")
    class_values = np.unique(y_bad)
    if len(class_values) >= 2:
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=2,
        )
        clf.fit(X, y_bad)
        try:
            bad_prob = clf.predict_proba(X)[:, 1]
            if len(np.unique(y_bad)) >= 2:
                clf_auc = float(roc_auc_score(y_bad, bad_prob))
            use_classifier = True
        except Exception:
            use_classifier = False

    payload = {
        "model": reg,
        "classifier": clf,
        "feature_names": list(QUALITY_FEATURE_NAMES),
        "threshold": float(classifier_threshold if use_classifier else bad_dice_threshold),
        "use_classifier": bool(use_classifier),
        "meta": {
            "n_samples": int(len(train_df)),
            "n_bad": int((y_bad == 1).sum()),
            "n_good": int((y_bad == 0).sum()),
            "dice_bad_threshold": float(bad_dice_threshold),
            "classifier_threshold": float(classifier_threshold),
            "reg_rmse_train": rmse,
            "clf_auc_train": clf_auc,
            "n_dice_available": n_dice_available,
            "n_dice_missing": n_dice_missing,
            "reg_target_mode": reg_target_mode,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, str(output_path))
    return payload["meta"]


def _prepare_quality_model(
    mode: str,
    model_path: Path,
    all_frames_dir: Path,
    curated_labels: Dict[str, str],
    bad_dice_threshold: float = 0.90,
    classifier_threshold: float = 0.5,
    label_mode: str = "curated_only",
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    exists_before = model_path.exists()
    info: Dict[str, Any] = {
        "mode": mode,
        "label_mode": label_mode,
        "path": str(model_path),
        "exists_before": bool(exists_before),
        "used_existing": False,
        "trained": False,
        "enabled": mode != "disable",
    }

    if mode == "disable":
        return info

    if mode == "reuse":
        info["used_existing"] = bool(exists_before)
        info["exists_after"] = bool(model_path.exists())
        return info

    if mode == "train_if_missing" and exists_before:
        info["used_existing"] = True
        info["exists_after"] = True
        return info

    train_df = _collect_quality_training_table(
        all_frames_dir=all_frames_dir,
        curated_labels=curated_labels,
        bad_dice_threshold=bad_dice_threshold,
        label_mode=label_mode,
        max_cases=max_cases,
    )
    info["n_training_rows"] = int(len(train_df))
    if train_df.empty:
        if label_mode == "curated_only":
            info["train_error"] = (
                "No usable training rows for curated_only mode "
                "(need *_img.png/*_pred.png and resolved curated labels)."
            )
        else:
            info["train_error"] = (
                "No usable training rows for dice_with_curated_override mode "
                "(need *_img.png/*_pred.png/*_gt.png)."
            )
        info["exists_after"] = bool(model_path.exists())
        return info

    info["n_bad_labels"] = int((train_df["y_bad"] == 1).sum())
    info["n_good_labels"] = int((train_df["y_bad"] == 0).sum())
    try:
        metrics = _train_quality_model(
            train_df=train_df,
            output_path=model_path,
            bad_dice_threshold=bad_dice_threshold,
            classifier_threshold=classifier_threshold,
        )
        info["trained"] = True
        info["train_metrics"] = metrics
    except Exception as e:
        info["train_error"] = str(e)

    info["exists_after"] = bool(model_path.exists())
    return info


def _warmup_if_needed(cfg: Dict[str, Any]) -> None:
    llm_cfg = cfg.get("llm", {}) or {}
    if llm_cfg.get("provider") != "ollama" or not llm_cfg.get("enabled", True):
        return
    settings = LLMSettings()
    warmup_client = OpenAICompatClient(
        base_url=llm_cfg.get("base_url", settings.openai_base_url),
        api_key=llm_cfg.get("api_key", settings.openai_api_key),
        model=llm_cfg.get("model", settings.openai_model),
        timeout=float(llm_cfg.get("timeout", 180.0)),
        provider="ollama",
    )
    try:
        warmup_client.warmup()
    except Exception as e:
        print(f"Warning: LLM warmup failed, continue without warmup. Error: {e}")


def _build_agent_stack(
    cfg: Dict[str, Any],
    source_root: Path,
    all_frames_dir: Path,
    atlas_mode: str,
    atlas_path: Path,
) -> Tuple[MessageBus, Dict[str, Any]]:
    _warmup_if_needed(cfg)

    atlas = None
    if atlas_mode == "disable_global":
        print("Atlas mode: disable_global (no global atlas)")
    elif atlas_mode == "reuse_global":
        if atlas_path.exists():
            atlas = load_atlas(str(atlas_path))
            print(f"Atlas mode: reuse_global (loaded {atlas_path})")
        else:
            atlas_agent = AtlasBuilderAgent(cfg)
            atlas = atlas_agent.run(str(all_frames_dir), str(atlas_path))
            print(f"Atlas mode: reuse_global fallback -> rebuilt {atlas_path}")
    elif atlas_mode == "local_per_sample":
        # No global atlas; local atlas is built per case in repair stage.
        atlas = None
        print("Atlas mode: local_per_sample (global atlas disabled, per-case atlas enabled)")
    else:
        atlas_agent = AtlasBuilderAgent(cfg)
        atlas = atlas_agent.run(str(all_frames_dir), str(atlas_path))
        print(f"Atlas mode: rebuild_global (rebuilt {atlas_path})")

    visual_kb = None
    if source_root.is_dir():
        visual_kb = VisualKnowledgeBase.load(str(source_root))
        print(
            f"Visual KB loaded: good={len(visual_kb.good_examples)}, "
            f"bad={len(visual_kb.bad_examples)}"
        )

    default_repair_kb_dir = Path(__file__).resolve().parents[1] / "repair_kb"
    repair_kb_dir = Path(
        cfg.get("paths", {}).get("repair_kb_dir", str(default_repair_kb_dir))
    )
    repair_kb = None
    if repair_kb_dir.is_dir():
        repair_kb = RepairComparisonKB.load(str(repair_kb_dir))
        print(
            f"Repair KB loaded: total={repair_kb.total}, "
            f"improved={len(repair_kb.improved)}, degraded={len(repair_kb.degraded)}"
        )
    else:
        print(f"Repair KB not found: {repair_kb_dir} (fallback comparison mode)")

    bus = MessageBus()
    agents = {
        "coordinator": CoordinatorAgent(cfg, bus),
        "triage": TriageAgent(cfg, bus, visual_kb=visual_kb),
        "diagnosis": DiagnosisAgent(cfg, bus),
        "planner": PlannerAgent(cfg, bus),
        "executor": ExecutorAgent(cfg, bus, atlas=atlas, visual_kb=visual_kb, repair_kb=repair_kb),
        "verifier": VerifierAgent(cfg, bus, visual_kb=visual_kb, repair_kb=repair_kb),
    }
    return bus, agents


def _build_local_atlas_for_case(
    ctx: Any,
    groups_by_view: Dict[str, List[Any]],
    cfg: Dict[str, Any],
    out_path: Path,
) -> Tuple[Optional[dict], List[str]]:
    key = _view_key(ctx.stem)
    group = groups_by_view.get(key, [])
    # Local per-sample atlas policy:
    # - never include the current slice itself
    # - use nearest neighboring slices as donors (up to 2)
    donor_candidates = [c for c in group if c.stem != ctx.stem]
    if int(getattr(ctx, "slice_index", -1)) >= 0:
        center_idx = int(getattr(ctx, "slice_index", -1))

        def _dist_key(c: Any) -> Tuple[int, int]:
            si = int(getattr(c, "slice_index", -1))
            if si < 0:
                return (10**9, 10**9)
            return (abs(si - center_idx), si)

        donor_candidates = sorted(donor_candidates, key=_dist_key)
    else:
        donor_candidates = sorted(donor_candidates, key=lambda c: str(c.stem))
    donors = donor_candidates[:2]

    masks: List[np.ndarray] = []
    views: List[str] = []
    zs: List[float] = []
    donor_stems: List[str] = []

    for c in donors:
        # Keep donor atlas stable across batch execution: always prefer
        # original prediction instead of progressively repaired masks.
        m = c.original_mask if c.original_mask is not None else c.mask
        if m is None:
            continue
        masks.append(m)
        views.append(c.view_type if getattr(c, "view_type", None) else "4ch")
        z = float(getattr(c, "slice_index", -1))
        z_norm = max(0.0, min(1.0, (z if z >= 0 else 0.0) / 64.0))
        zs.append(z_norm)
        donor_stems.append(str(c.stem))

    # Allow 1 donor for sparse groups (still excludes self).
    if len(masks) < 1:
        return None, donor_stems

    atlas = build_atlas(masks, views, zs, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_atlas(str(out_path), atlas)
    return atlas, donor_stems


def _filter_cases(
    cases: List[Any],
    case_contains: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Any]:
    rows = sorted(cases, key=lambda c: c.stem)
    if case_contains:
        rows = [c for c in rows if case_contains in c.stem]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _run_triage_stage(
    cases: List[Any],
    triage_agent: TriageAgent,
    bus: MessageBus,
    triage_debug_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ctx in tqdm(cases, desc="Stage1 Triage"):
        t0 = time.time()
        ctx.output_dir = str(triage_debug_dir)
        triage_agent.reset_memory()
        bus.register_case(ctx)

        try:
            triage_agent.process_case(ctx)
            dice = _compute_dice(ctx.mask, ctx.gt_path)
            vlm_data = getattr(ctx, "vlm_verdict_data", {}) or {}
            row = {
                "Stem": ctx.stem,
                "TriageCategory": ctx.triage_category,
                "TriageScore": float(ctx.triage_score),
                "TriageBadProb": float(getattr(ctx, "triage_bad_prob", 1.0 - float(ctx.triage_score))),
                "TriageSeverity": float(getattr(ctx, "triage_severity", float("nan"))),
                "VLM_Quality": getattr(ctx, "vlm_quality", "N/A"),
                "VLM_Score": float(getattr(ctx, "vlm_score", vlm_data.get("score", float("nan")))),
                "VLM_GoodProb": float(getattr(ctx, "vlm_good_prob", float("nan"))),
                "VLM_Confidence": float(vlm_data.get("confidence", float("nan"))),
                "QM_BadProb": float(getattr(ctx, "qm_bad_prob", float("nan"))),
                "Issues": json.dumps(ctx.triage_issues, ensure_ascii=False),
                "Dice_Mean": dice["Dice_Mean"],
                "Dice_RV": dice["Dice_RV"],
                "Dice_Myo": dice["Dice_Myo"],
                "Dice_LV": dice["Dice_LV"],
                "Time_s": round(time.time() - t0, 3),
            }
        except Exception as e:
            row = {
                "Stem": ctx.stem,
                "TriageCategory": "error",
                "TriageScore": 0.0,
                "TriageBadProb": 1.0,
                "TriageSeverity": float("nan"),
                "VLM_Quality": "error",
                "VLM_Score": float("nan"),
                "VLM_GoodProb": float("nan"),
                "VLM_Confidence": float("nan"),
                "QM_BadProb": float("nan"),
                "Issues": json.dumps([f"exception: {e}"], ensure_ascii=False),
                "Dice_Mean": float("nan"),
                "Dice_RV": float("nan"),
                "Dice_Myo": float("nan"),
                "Dice_LV": float("nan"),
                "Time_s": round(time.time() - t0, 3),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _reset_agents_memory(agents: Dict[str, Any]) -> None:
    for agent in agents.values():
        if hasattr(agent, "reset_memory"):
            agent.reset_memory()


def _evaluate_triage_curated(
    triage_df: pd.DataFrame,
    curated_labels: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if triage_df.empty:
        return pd.DataFrame(), {"n_eval_overlap": 0}

    label_df = pd.DataFrame(
        [{"Stem": stem, "TrueLabel": label} for stem, label in curated_labels.items()]
    )
    merged = label_df.merge(triage_df, on="Stem", how="inner")
    if merged.empty:
        return merged, {"n_eval_overlap": 0}

    merged["y_true"] = (merged["TrueLabel"] == "needs_fix").astype(int)
    merged["y_pred"] = (merged["TriageCategory"] == "needs_fix").astype(int)

    if "TriageBadProb" in merged.columns:
        merged["y_score"] = pd.to_numeric(merged["TriageBadProb"], errors="coerce")
    else:
        merged["y_score"] = 1.0 - pd.to_numeric(merged["TriageScore"], errors="coerce")

    eval_rows = merged[merged["y_pred"].isin([0, 1])].copy()
    n_eval = len(eval_rows)
    if n_eval == 0:
        return merged, {"n_eval_overlap": 0}

    y_true = eval_rows["y_true"].astype(int).to_numpy()
    y_pred = eval_rows["y_pred"].astype(int).to_numpy()
    y_score = eval_rows["y_score"].astype(float).to_numpy()

    metrics: Dict[str, Any] = {
        "n_eval_overlap": int(n_eval),
        "n_labels_total": int(len(curated_labels)),
        "n_good_true": int((y_true == 0).sum()),
        "n_needs_fix_true": int((y_true == 1).sum()),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Precision_needs_fix": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall_needs_fix": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall_good": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "F1_needs_fix": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

    finite_mask = np.isfinite(y_score)
    if finite_mask.sum() >= 2 and len(np.unique(y_true[finite_mask])) >= 2:
        metrics["AUC"] = float(roc_auc_score(y_true[finite_mask], y_score[finite_mask]))
    else:
        metrics["AUC"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    metrics.update({"TP": tp, "FP": fp, "TN": tn, "FN": fn})

    return eval_rows, metrics


def _triage_group_dice_stats(triage_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    valid = triage_df[
        triage_df["TriageCategory"].isin(["good", "needs_fix"])
        & pd.to_numeric(triage_df["Dice_Mean"], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame(), {}

    valid["Dice_Mean"] = pd.to_numeric(valid["Dice_Mean"], errors="coerce")
    stats = (
        valid.groupby("TriageCategory")["Dice_Mean"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )

    summary: Dict[str, Any] = {}
    g_good = valid.loc[valid["TriageCategory"] == "good", "Dice_Mean"].to_numpy()
    g_bad = valid.loc[valid["TriageCategory"] == "needs_fix", "Dice_Mean"].to_numpy()
    if len(g_good) > 0 and len(g_bad) > 0:
        summary["dice_mean_good"] = float(np.mean(g_good))
        summary["dice_mean_needs_fix"] = float(np.mean(g_bad))
        summary["dice_mean_diff_needs_fix_minus_good"] = float(np.mean(g_bad) - np.mean(g_good))
        try:
            _, p_mwu = mannwhitneyu(g_good, g_bad, alternative="two-sided")
            summary["dice_group_mannwhitney_p"] = float(p_mwu)
        except Exception:
            summary["dice_group_mannwhitney_p"] = float("nan")
    return stats, summary


def _reset_case_for_repair(ctx: Any, repair_debug_dir: Path) -> None:
    if ctx.original_mask is None and ctx.mask is not None:
        ctx.original_mask = ctx.mask.copy()
    if ctx.original_mask is not None:
        ctx.current_mask = ctx.original_mask.copy()
    elif ctx.mask is not None:
        ctx.current_mask = ctx.mask.copy()
    ctx.applied_ops = []
    ctx.rqs_history = []
    ctx.diagnoses = []
    ctx.rounds_completed = 0
    ctx.verified = False
    ctx.verifier_approved = False
    ctx.verifier_feedback = ""
    ctx.last_verify_result = {}
    ctx.knowledge_trace = []
    ctx.executor_high_risk_unlock = False
    ctx.executor_high_risk_unlock_reason = ""
    ctx.executor_candidate_search_used = False
    ctx.executor_candidate_search_pick = ""
    ctx.executor_candidate_search_score = 0.0
    baseline = ctx.original_mask if ctx.original_mask is not None else ctx.mask
    ctx.best_intermediate_mask = baseline.copy() if baseline is not None else None
    ctx.best_intermediate_score = 0.0
    ctx.best_intermediate_round = 0
    ctx.best_intermediate_verdict = "baseline_original"
    ctx.best_intermediate_reason = "initial prediction baseline"
    ctx.best_intermediate_ops = []
    ctx.best_intermediate_is_original = True
    ctx.best_intermediate_online_score = float("nan")
    ctx.best_intermediate_anatomy_score = float("nan")
    baseline_anatomy = _safe_float(getattr(ctx, "anatomy_score_before", float("nan")))
    if _is_finite(baseline_anatomy) and baseline_anatomy < 0.0:
        baseline_anatomy = float("nan")
    ctx.best_anatomy_intermediate_mask = baseline.copy() if baseline is not None else None
    ctx.best_anatomy_intermediate_score = baseline_anatomy
    ctx.best_anatomy_intermediate_delta = 0.0 if _is_finite(baseline_anatomy) else float("nan")
    ctx.best_anatomy_intermediate_round = 0
    ctx.best_anatomy_intermediate_verdict = "baseline_original"
    ctx.best_anatomy_intermediate_reason = "initial prediction baseline"
    ctx.best_anatomy_intermediate_ops = []
    ctx.best_anatomy_intermediate_is_original = True
    ctx.best_anatomy_intermediate_online_score = float("nan")
    ctx.output_dir = str(repair_debug_dir)


def _build_repair_queue(
    triage_df: pd.DataFrame,
    curated_labels: Dict[str, str],
    mode: str,
) -> List[str]:
    if mode == "worst_cases":
        return sorted([stem for stem, lab in curated_labels.items() if lab == "needs_fix"])
    if triage_df.empty:
        return []
    return sorted(
        triage_df.loc[triage_df["TriageCategory"] == "needs_fix", "Stem"].astype(str).tolist()
    )


def _run_repair_stage(
    cases: List[Any],
    repair_stems: List[str],
    agents: Dict[str, Any],
    bus: MessageBus,
    repair_debug_dir: Path,
    repaired_labels_dir: Path,
    max_rounds: int,
    atlas_mode: str,
    cfg: Dict[str, Any],
    local_atlas_dir: Optional[Path] = None,
    atlas_source_cases: Optional[List[Any]] = None,
) -> pd.DataFrame:
    case_by_stem = {c.stem: c for c in cases}
    queue = [case_by_stem[s] for s in repair_stems if s in case_by_stem]

    if not queue:
        return pd.DataFrame()

    share_repair_to_neighbors = bool(
        cfg.get("multi_agent", {}).get("share_repair_to_neighbors", False)
    )
    atlas_cases = atlas_source_cases if atlas_source_cases is not None else cases
    groups = group_and_link_neighbors(atlas_cases)
    rows: List[Dict[str, Any]] = []
    coordinator: CoordinatorAgent = agents["coordinator"]
    executor: ExecutorAgent = agents["executor"]
    global_atlas = executor._atlas

    for ctx in tqdm(queue, desc="Stage2 Repair"):
        _reset_agents_memory(agents)
        _reset_case_for_repair(ctx, repair_debug_dir)
        bus.register_case(ctx)

        atlas_used = "global"
        atlas_path_used = ""
        if atlas_mode == "local_per_sample":
            atlas_used = "local_per_sample"
            if local_atlas_dir is not None:
                local_path = local_atlas_dir / f"{ctx.stem}_atlas.pkl"
                local_atlas, donor_stems = _build_local_atlas_for_case(ctx, groups, cfg, local_path)
                if local_atlas is not None:
                    executor._atlas = local_atlas
                    atlas_path_used = str(local_path)
                    if donor_stems:
                        print(f"[local_atlas] {ctx.stem}: donor_stems={donor_stems}")
                else:
                    executor._atlas = None
                    atlas_path_used = ""
                    print(
                        f"[local_atlas] {ctx.stem}: unavailable "
                        f"(donor_stems={donor_stems})"
                    )
        elif atlas_mode == "disable_global":
            atlas_used = "disabled"
            executor._atlas = None
        else:
            executor._atlas = global_atlas

        pre = _compute_dice(ctx.original_mask if ctx.original_mask is not None else ctx.mask, ctx.gt_path)
        verdict = coordinator.orchestrate_case(ctx, agents, max_rounds=max_rounds)
        pick_meta: Dict[str, Any] = {
            "selected_source": "n/a",
            "current_online_score": float("nan"),
            "current_anatomy_score": float("nan"),
            "current_blended_score": float("nan"),
            "best_online_score": float("nan"),
            "best_anatomy_score": float("nan"),
            "best_blended_score": float("nan"),
        }

        if verdict == "approved" and ctx.current_mask is not None:
            final_mask, saved_mask_type, pick_meta = _select_approved_final_mask(
                ctx=ctx,
                cfg=cfg,
            )
        else:
            final_mask, saved_mask_type, pick_meta = _select_nonapproved_final_mask(
                ctx=ctx,
                cfg=cfg,
            )

        if final_mask is not None:
            ctx.current_mask = final_mask.copy()
        if share_repair_to_neighbors:
            _update_neighbor_masks_for_group(ctx, groups)
        out_pred_path = repaired_labels_dir / f"{ctx.stem}_pred.png"
        if final_mask is not None:
            write_mask(str(out_pred_path), final_mask)

        post = _compute_dice(final_mask, ctx.gt_path)
        delta = {
            "DiceDelta_Mean": (
                post["Dice_Mean"] - pre["Dice_Mean"]
                if _is_finite(pre["Dice_Mean"]) and _is_finite(post["Dice_Mean"])
                else float("nan")
            ),
            "DiceDelta_RV": (
                post["Dice_RV"] - pre["Dice_RV"]
                if _is_finite(pre["Dice_RV"]) and _is_finite(post["Dice_RV"])
                else float("nan")
            ),
            "DiceDelta_Myo": (
                post["Dice_Myo"] - pre["Dice_Myo"]
                if _is_finite(pre["Dice_Myo"]) and _is_finite(post["Dice_Myo"])
                else float("nan")
            ),
            "DiceDelta_LV": (
                post["Dice_LV"] - pre["Dice_LV"]
                if _is_finite(pre["Dice_LV"]) and _is_finite(post["Dice_LV"])
                else float("nan")
            ),
        }

        last_verify = getattr(ctx, "last_verify_result", {}) or {}
        last_cmp = last_verify.get("vlm_comparison", {}) or {}
        last_proxy = last_verify.get("proxy_consensus", {}) or {}
        online_matched_ids = last_verify.get("online_matched_example_ids", [])
        if not isinstance(online_matched_ids, list):
            online_matched_ids = []
        row = {
            "Stem": ctx.stem,
            "FinalVerdict": verdict,
            "SavedMaskType": saved_mask_type,
            "SavedPredPath": str(out_pred_path),
            "BestIntermediateUsed": bool(saved_mask_type == "best_intermediate"),
            "BestIntermediateRound": int(getattr(ctx, "best_intermediate_round", 0)),
            "BestIntermediateScore": _safe_float(
                getattr(ctx, "best_intermediate_score", float("nan"))
            ),
            "BestIntermediateVerdict": str(
                getattr(ctx, "best_intermediate_verdict", "")
            ),
            "BestIntermediateReason": str(
                getattr(ctx, "best_intermediate_reason", "")
            ),
            "BestIntermediateOps": json.dumps(
                [
                    op.get("name", "")
                    for op in (getattr(ctx, "best_intermediate_ops", []) or [])
                ],
                ensure_ascii=False,
            ),
            "BestIntermediateOnlineScore": _safe_float(
                getattr(ctx, "best_intermediate_online_score", float("nan"))
            ),
            "BestIntermediateAnatomyScore": _safe_float(
                getattr(ctx, "best_intermediate_anatomy_score", float("nan"))
            ),
            "FinalPickSource": str(pick_meta.get("selected_source", "n/a")),
            "FinalPickNonApprovedPolicy": str(
                pick_meta.get("nonapproved_policy", "")
            ),
            "FinalPickCurrentOnlineScore": _safe_float(
                pick_meta.get("current_online_score", float("nan"))
            ),
            "FinalPickCurrentAnatomyScore": _safe_float(
                pick_meta.get("current_anatomy_score", float("nan"))
            ),
            "FinalPickCurrentBlendedScore": _safe_float(
                pick_meta.get("current_blended_score", float("nan"))
            ),
            "FinalPickBestOnlineScore": _safe_float(
                pick_meta.get("best_online_score", float("nan"))
            ),
            "FinalPickBestAnatomyScore": _safe_float(
                pick_meta.get("best_anatomy_score", float("nan"))
            ),
            "FinalPickBestBlendedScore": _safe_float(
                pick_meta.get("best_blended_score", float("nan"))
            ),
            "FinalPickNonApprovedOnlineGainOK": bool(
                pick_meta.get("nonapproved_online_gain_ok", False)
            ),
            "FinalPickNonApprovedAnatomyGainOK": bool(
                pick_meta.get("nonapproved_anatomy_gain_ok", False)
            ),
            "FinalPickNonApprovedBlendedGainOK": bool(
                pick_meta.get("nonapproved_blended_gain_ok", False)
            ),
            "AtlasMode": atlas_mode,
            "AtlasUsed": atlas_used,
            "AtlasPathUsed": atlas_path_used,
            "TriageCategory_FinalRun": getattr(ctx, "triage_category", ""),
            "TriageScore_FinalRun": _safe_float(getattr(ctx, "triage_score", float("nan"))),
            "TriageBadProb_FinalRun": _safe_float(getattr(ctx, "triage_bad_prob", float("nan"))),
            "NumDiagnoses": len(ctx.diagnoses),
            "Diagnoses": json.dumps(ctx.diagnoses, ensure_ascii=False),
            "OpsCount": len(ctx.applied_ops),
            "AppliedOps": json.dumps([op.get("name", "") for op in ctx.applied_ops], ensure_ascii=False),
            "RoundsCompleted": int(getattr(ctx, "rounds_completed", 0)),
            "VerifyVerdict_Raw": str(last_verify.get("verdict", "")),
            "VerifyScore_Raw": _safe_float(last_verify.get("score", float("nan"))),
            "VerifyConfidence_Raw": _safe_float(last_verify.get("confidence", float("nan"))),
            "VerifyCompareVerdict": str(last_cmp.get("verdict", "")),
            "VerifyCompareConfidence": _safe_float(last_cmp.get("confidence", float("nan"))),
            "VerifyMatchedExampleIDs": json.dumps(online_matched_ids, ensure_ascii=False),
            "VerifyCompareReference": str(last_verify.get("comparison_reference", "")),
            "VerifyProxyVerdict": str(last_proxy.get("proxy_verdict", "")),
            "VerifyProxyScore": _safe_float(last_proxy.get("proxy_score", float("nan"))),
            "VerifyProxyConfidence": _safe_float(last_proxy.get("proxy_confidence", float("nan"))),
            "VerifyConsensusScore": _safe_float(last_proxy.get("consensus_score", float("nan"))),
            "VerifyConsensusVerdict": str(last_proxy.get("consensus_verdict", "")),
            "VerifyUnlockApplied": bool(last_proxy.get("unlock_applied", False)),
            "online_verdict": str(last_verify.get("online_verdict", "")),
            "online_score": _safe_float(last_verify.get("online_score", float("nan"))),
            "online_confidence": _safe_float(last_verify.get("confidence", float("nan"))),
            "online_threshold_reject": _safe_float(
                last_verify.get("online_threshold_reject", float("nan"))
            ),
            "online_threshold_approve": _safe_float(
                last_verify.get("online_threshold_approve", float("nan"))
            ),
            "online_compare_verdict": str(last_cmp.get("verdict", "")),
            "online_compare_confidence": _safe_float(last_cmp.get("confidence", float("nan"))),
            "online_quality_label": str((last_verify.get("vlm_quality", {}) or {}).get("quality", "")),
            "online_quality_score": _safe_float(
                (last_verify.get("vlm_quality", {}) or {}).get("score", float("nan"))
            ),
            "online_matched_example_ids": json.dumps(online_matched_ids, ensure_ascii=False),
            "online_prompt_strict_no_gt": bool(last_verify.get("strict_no_gt_online", False)),
            "online_knowledge_trace": json.dumps(getattr(ctx, "knowledge_trace", []), ensure_ascii=False),
            "ExecHighRiskUnlock": bool(getattr(ctx, "executor_high_risk_unlock", False)),
            "ExecHighRiskUnlockReason": str(
                getattr(ctx, "executor_high_risk_unlock_reason", "")
            ),
            "ExecCandidateSearchUsed": bool(
                getattr(ctx, "executor_candidate_search_used", False)
            ),
            "ExecCandidateSearchPick": str(
                getattr(ctx, "executor_candidate_search_pick", "")
            ),
            "ExecCandidateSearchScore": _safe_float(
                getattr(ctx, "executor_candidate_search_score", float("nan"))
            ),
            "offline_oracle_pre_dice_mean": pre["Dice_Mean"],
            "offline_oracle_pre_dice_rv": pre["Dice_RV"],
            "offline_oracle_pre_dice_myo": pre["Dice_Myo"],
            "offline_oracle_pre_dice_lv": pre["Dice_LV"],
            "offline_oracle_post_dice_mean": post["Dice_Mean"],
            "offline_oracle_post_dice_rv": post["Dice_RV"],
            "offline_oracle_post_dice_myo": post["Dice_Myo"],
            "offline_oracle_post_dice_lv": post["Dice_LV"],
            "offline_oracle_dice_delta_mean": delta["DiceDelta_Mean"],
            "offline_oracle_dice_delta_rv": delta["DiceDelta_RV"],
            "offline_oracle_dice_delta_myo": delta["DiceDelta_Myo"],
            "offline_oracle_dice_delta_lv": delta["DiceDelta_LV"],
            "PreDice_Mean": pre["Dice_Mean"],
            "PreDice_RV": pre["Dice_RV"],
            "PreDice_Myo": pre["Dice_Myo"],
            "PreDice_LV": pre["Dice_LV"],
            "PostDice_Mean": post["Dice_Mean"],
            "PostDice_RV": post["Dice_RV"],
            "PostDice_Myo": post["Dice_Myo"],
            "PostDice_LV": post["Dice_LV"],
            **delta,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _summarize_repair(repair_df: pd.DataFrame) -> Dict[str, Any]:
    if repair_df.empty:
        return {"n_cases": 0}

    valid = repair_df[
        pd.to_numeric(repair_df["PreDice_Mean"], errors="coerce").notna()
        & pd.to_numeric(repair_df["PostDice_Mean"], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return {"n_cases": int(len(repair_df)), "n_valid_dice": 0}

    pre = valid["PreDice_Mean"].astype(float).to_numpy()
    post = valid["PostDice_Mean"].astype(float).to_numpy()
    delta = post - pre

    summary: Dict[str, Any] = {
        "n_cases": int(len(repair_df)),
        "n_valid_dice": int(len(valid)),
        "n_approved": int((repair_df["FinalVerdict"] == "approved").sum()),
        "n_needs_more_work": int((repair_df["FinalVerdict"] == "needs_more_work").sum()),
        "n_rejected": int((repair_df["FinalVerdict"] == "rejected").sum()),
        "n_gave_up": int((repair_df["FinalVerdict"] == "gave_up").sum()),
        "n_saved_repaired": int(
            repair_df["SavedMaskType"].isin(["repaired", "best_intermediate"]).sum()
        ),
        "pre_dice_mean": float(np.mean(pre)),
        "post_dice_mean": float(np.mean(post)),
        "delta_dice_mean": float(np.mean(delta)),
        "delta_dice_median": float(np.median(delta)),
        "improved_count": int((delta > 0).sum()),
        "worsened_count": int((delta < 0).sum()),
        "unchanged_count": int((delta == 0).sum()),
        "improved_ratio": float(np.mean(delta > 0)),
    }

    approved = valid[valid["FinalVerdict"] == "approved"].copy()
    if not approved.empty:
        approved_delta = approved["PostDice_Mean"].astype(float) - approved["PreDice_Mean"].astype(float)
        summary["approved_degradation_rate"] = float((approved_delta < 0).mean())
        summary["approved_delta_dice_mean"] = float(np.mean(approved_delta))
    else:
        summary["approved_degradation_rate"] = float("nan")
        summary["approved_delta_dice_mean"] = float("nan")

    if "online_score" in valid.columns:
        score_bins_df = _score_bin_table(valid, score_col="online_score")
        if not score_bins_df.empty:
            summary["gain_distribution_by_score_bin"] = score_bins_df.to_dict(orient="records")
        else:
            summary["gain_distribution_by_score_bin"] = []

    try:
        stat = ttest_rel(post, pre, alternative="greater")
        summary["paired_ttest_p_post_gt_pre"] = float(stat.pvalue)
    except Exception:
        summary["paired_ttest_p_post_gt_pre"] = float("nan")

    return summary


def _plot_triage_roc(eval_rows: pd.DataFrame, out_base: Path) -> Optional[Dict[str, str]]:
    if eval_rows.empty or "y_true" not in eval_rows.columns or "y_score" not in eval_rows.columns:
        return None
    y_true = eval_rows["y_true"].astype(int).to_numpy()
    y_score = pd.to_numeric(eval_rows["y_score"], errors="coerce").to_numpy()
    ok = np.isfinite(y_score)
    y_true = y_true[ok]
    y_score = y_score[ok]
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    _apply_miccai_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.0, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", lw=1.2, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Triage Classification ROC (Curated Best/Worst)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.35, linestyle="--")
    return _save_fig(fig, out_base)


def _plot_triage_dice_groups(triage_df: pd.DataFrame, out_base: Path) -> Optional[Dict[str, str]]:
    valid = triage_df[
        triage_df["TriageCategory"].isin(["good", "needs_fix"])
        & pd.to_numeric(triage_df["Dice_Mean"], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return None
    valid["Dice_Mean"] = pd.to_numeric(valid["Dice_Mean"], errors="coerce")
    order = ["good", "needs_fix"]
    palette = {"good": "#2A9D8F", "needs_fix": "#E76F51"}

    _apply_miccai_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sns.boxplot(
        data=valid,
        x="TriageCategory",
        y="Dice_Mean",
        hue="TriageCategory",
        order=order,
        hue_order=order,
        palette=palette,
        dodge=False,
        width=0.56,
        showfliers=False,
        linewidth=1.2,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.stripplot(
        data=valid,
        x="TriageCategory",
        y="Dice_Mean",
        order=order,
        color="#1F1F1F",
        size=2.5,
        alpha=0.35,
        jitter=0.18,
        ax=ax,
    )

    good_vals = valid.loc[valid["TriageCategory"] == "good", "Dice_Mean"].to_numpy()
    bad_vals = valid.loc[valid["TriageCategory"] == "needs_fix", "Dice_Mean"].to_numpy()
    p_val = float("nan")
    if len(good_vals) > 0 and len(bad_vals) > 0:
        try:
            _, p_val = mannwhitneyu(good_vals, bad_vals, alternative="two-sided")
        except Exception:
            p_val = float("nan")

    ax.set_xlabel("Triage Group")
    ax.set_ylabel("Dice Mean")
    ax.set_title("Dice Distribution by Triage Group")
    if _is_finite(p_val):
        ax.text(
            0.98,
            0.98,
            f"Mann-Whitney p={p_val:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    return _save_fig(fig, out_base)


def _plot_repair_before_after(repair_df: pd.DataFrame, out_base: Path) -> Optional[Dict[str, str]]:
    if repair_df.empty:
        return None
    valid = repair_df[
        pd.to_numeric(repair_df["PreDice_Mean"], errors="coerce").notna()
        & pd.to_numeric(repair_df["PostDice_Mean"], errors="coerce").notna()
    ].copy()
    if valid.empty:
        return None
    valid["PreDice_Mean"] = pd.to_numeric(valid["PreDice_Mean"], errors="coerce")
    valid["PostDice_Mean"] = pd.to_numeric(valid["PostDice_Mean"], errors="coerce")

    long_df = pd.concat(
        [
            pd.DataFrame({"Stage": "Before", "Dice_Mean": valid["PreDice_Mean"]}),
            pd.DataFrame({"Stage": "After", "Dice_Mean": valid["PostDice_Mean"]}),
        ],
        ignore_index=True,
    )

    pre = valid["PreDice_Mean"].to_numpy()
    post = valid["PostDice_Mean"].to_numpy()
    p_val = float("nan")
    if len(pre) > 1:
        try:
            p_val = float(ttest_rel(post, pre, alternative="greater").pvalue)
        except Exception:
            p_val = float("nan")
    delta_mean = float(np.mean(post - pre))

    _apply_miccai_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    sns.boxplot(
        data=long_df,
        x="Stage",
        y="Dice_Mean",
        hue="Stage",
        order=["Before", "After"],
        hue_order=["Before", "After"],
        palette={"Before": "#8DA0CB", "After": "#66C2A5"},
        dodge=False,
        width=0.56,
        showfliers=False,
        linewidth=1.2,
        ax=ax,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    for _, row in valid.iterrows():
        ax.plot(
            [0, 1],
            [row["PreDice_Mean"], row["PostDice_Mean"]],
            color="#6c6c6c",
            alpha=0.22,
            linewidth=0.6,
            zorder=1,
        )
    sns.stripplot(
        data=long_df,
        x="Stage",
        y="Dice_Mean",
        order=["Before", "After"],
        color="#1F1F1F",
        size=2.5,
        alpha=0.35,
        jitter=0.14,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Dice Mean")
    ax.set_title("Need-Fix Cases: Dice Before vs After Repair")
    txt = f"Mean ΔDice={delta_mean:+.4f}"
    if _is_finite(p_val):
        txt += f"\nPaired t-test p={p_val:.2e}"
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9)
    return _save_fig(fig, out_base)


def _plot_repair_delta_hist(repair_df: pd.DataFrame, out_base: Path) -> Optional[Dict[str, str]]:
    if repair_df.empty or "DiceDelta_Mean" not in repair_df.columns:
        return None
    delta = pd.to_numeric(repair_df["DiceDelta_Mean"], errors="coerce")
    delta = delta[np.isfinite(delta.to_numpy())]
    if delta.empty:
        return None

    _apply_miccai_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.hist(delta.to_numpy(), bins=24, color="#4C78A8", alpha=0.85, edgecolor="white")
    ax.axvline(0.0, color="#D62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Dice Improvement (Post - Pre)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Dice Improvement on Need-Fix Cases")
    ax.text(
        0.98,
        0.98,
        f"Mean={float(delta.mean()):+.4f}\nMedian={float(delta.median()):+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    return _save_fig(fig, out_base)


def _to_gray_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def _make_change_overlay(
    gray_img: np.ndarray,
    gt: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """
    Build RGB overlay:
      - Green: corrected pixels (before wrong -> after correct)
      - Red: regressed pixels (before correct -> after wrong)
      - Yellow: changed but still wrong
    """
    if gray_img.dtype != np.uint8:
        g = gray_img.astype(np.float32)
        if g.max() <= 1.0:
            g = g * 255.0
        gray_img = np.clip(g, 0, 255).astype(np.uint8)
    base = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    before_correct = (before == gt)
    after_correct = (after == gt)
    corrected = (~before_correct) & after_correct
    regressed = before_correct & (~after_correct)
    changed_wrong = (before != after) & (~after_correct) & (~regressed)

    overlay = base.copy()
    overlay[corrected] = [0, 200, 0]
    overlay[regressed] = [220, 0, 0]
    overlay[changed_wrong] = [240, 190, 40]
    blended = cv2.addWeighted(overlay, 0.45, base, 0.55, 0)
    return blended, int(corrected.sum()), int(regressed.sum())


def _plot_top_improved_cases(
    repair_df: pd.DataFrame,
    all_frames_dir: Path,
    out_dir: Path,
    top_k: int = 8,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if repair_df.empty:
        return {"n_selected": 0, "message": "repair_df is empty"}

    work = repair_df.copy()
    work["DiceDelta_Mean"] = pd.to_numeric(work["DiceDelta_Mean"], errors="coerce")
    work = work[np.isfinite(work["DiceDelta_Mean"].to_numpy())]
    work = work[work["DiceDelta_Mean"] > float(min_delta)]
    if work.empty:
        return {"n_selected": 0, "message": f"no cases with DiceDelta_Mean > {min_delta}"}

    k = max(1, int(top_k))
    selected = work.sort_values("DiceDelta_Mean", ascending=False).head(k).copy()
    selected_csv = out_dir / "top_improved_cases.csv"
    selected.to_csv(selected_csv, index=False)

    panel_paths: List[str] = []
    for _, row in selected.iterrows():
        stem = str(row["Stem"])
        img_path = all_frames_dir / f"{stem}_img.png"
        gt_path = all_frames_dir / f"{stem}_gt.png"
        before_path = all_frames_dir / f"{stem}_pred.png"
        after_path = Path(str(row.get("SavedPredPath", "")))
        if not (img_path.exists() and gt_path.exists() and before_path.exists() and after_path.exists()):
            continue

        img = _to_gray_image(read_image(str(img_path)))
        gt = read_mask(str(gt_path))
        before = read_mask(str(before_path))
        after = read_mask(str(after_path))
        change_overlay, corrected_px, regressed_px = _make_change_overlay(img, gt, before, after)

        pre_dice = _safe_float(row.get("PreDice_Mean", float("nan")))
        post_dice = _safe_float(row.get("PostDice_Mean", float("nan")))
        delta = _safe_float(row.get("DiceDelta_Mean", float("nan")))
        verdict = str(row.get("FinalVerdict", ""))

        _apply_miccai_style()
        fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.6))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="viridis", vmin=0, vmax=3)
        axes[1].set_title("GT")
        axes[1].axis("off")

        axes[2].imshow(before, cmap="viridis", vmin=0, vmax=3)
        axes[2].set_title(f"Before\nDice={pre_dice:.3f}")
        axes[2].axis("off")

        axes[3].imshow(after, cmap="viridis", vmin=0, vmax=3)
        axes[3].set_title(f"After\nDice={post_dice:.3f}")
        axes[3].axis("off")

        axes[4].imshow(change_overlay)
        axes[4].set_title(
            "Change Overlay\n"
            f"Corrected={corrected_px}, Regressed={regressed_px}"
        )
        axes[4].axis("off")

        fig.suptitle(
            f"{stem} | ΔDice={delta:+.4f} | Verdict={verdict}",
            y=1.02,
            fontsize=12,
        )
        out_base = out_dir / f"{stem}_top_improved_panel"
        paths = _save_fig(fig, out_base)
        panel_paths.append(paths["png"])

    # Summary bar chart for selected top-k improvements
    if not selected.empty:
        plot_df = selected.copy()
        plot_df["StemShort"] = plot_df["Stem"].astype(str).str.replace("_original_lax_", "_", regex=False)
        plot_df = plot_df.sort_values("DiceDelta_Mean", ascending=True)

        _apply_miccai_style()
        fig, ax = plt.subplots(figsize=(10.0, max(4.8, 0.42 * len(plot_df))))
        sns.barplot(
            data=plot_df,
            x="DiceDelta_Mean",
            y="StemShort",
            color="#2A9D8F",
            ax=ax,
        )
        ax.axvline(0.0, color="#D62728", linestyle="--", linewidth=1.1)
        ax.set_xlabel("Dice Improvement (Post - Pre)")
        ax.set_ylabel("Case")
        ax.set_title(f"Top-{len(plot_df)} Most Improved Cases")
        top_bar_paths = _save_fig(fig, out_dir / "top_improved_cases_bar")
    else:
        top_bar_paths = {}

    return {
        "n_selected": int(len(selected)),
        "selected_csv": str(selected_csv),
        "panel_pngs": panel_paths,
        "bar_plot": top_bar_paths,
    }


def _save_metrics_dict(metrics: Dict[str, Any], out_csv: Path) -> None:
    rows = [{"metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full MICCAI-style multi-agent post-processing experiment."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config path (e.g., config/azure_openai_medrag.yaml)",
    )
    parser.add_argument(
        "--source_root",
        required=True,
        help="Directory containing all_frames_export, best_cases, worst_cases",
    )
    parser.add_argument(
        "--target_dir",
        required=True,
        help="Output directory for triage/repair/figures/summary",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit after filtering (for smoke test)",
    )
    parser.add_argument(
        "--case_contains",
        type=str,
        default=None,
        help="Optional stem substring filter",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=None,
        help="Override multi_agent.max_rounds_per_case",
    )
    parser.add_argument(
        "--repair_subset",
        choices=["triage_need_fix", "worst_cases"],
        default="triage_need_fix",
        help="Need-fix set for Stage-2 repair",
    )
    parser.add_argument(
        "--skip_repair",
        action="store_true",
        help="Run Stage-1 triage + metrics + figures only",
    )
    parser.add_argument(
        "--triage_csv",
        default=None,
        help=(
            "Path to an existing triage_results.csv from a previous run. "
            "When provided, Stage-1 triage is skipped and repair runs directly "
            "using the supplied triage results."
        ),
    )
    parser.add_argument(
        "--atlas_mode",
        choices=["rebuild_global", "reuse_global", "disable_global", "local_per_sample"],
        default="rebuild_global",
        help="Atlas strategy: rebuild/reuse/disable global atlas, or build per-sample local atlas",
    )
    parser.add_argument(
        "--atlas_filename",
        default="global_shape_atlas.pkl",
        help="Atlas filename under artifacts/atlas/",
    )
    parser.add_argument(
        "--triage_n_refs",
        type=int,
        default=None,
        help="Override agents.triage.vlm_n_refs",
    )
    parser.add_argument(
        "--strict_no_gt_online",
        type=str,
        default=None,
        help="Strictly isolate online decision flow from GT/oracle metrics (true/false).",
    )
    parser.add_argument(
        "--llm_enabled",
        type=str,
        default=None,
        help="Global LLM enabled override (true/false)",
    )
    parser.add_argument(
        "--vision_enabled",
        type=str,
        default=None,
        help="Global VLM/vision enabled override (true/false)",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Directory for all intermediate artifacts (default: <target_dir>/artifacts)",
    )
    parser.add_argument(
        "--knowledge_mode",
        choices=["custom", "auto_skip", "force_rebuild"],
        default="custom",
        help=(
            "One-switch knowledge policy. "
            "auto_skip: generate once and skip if exists; "
            "force_rebuild: rebuild all knowledge each run; "
            "custom: use detailed quality/repair mode args."
        ),
    )
    parser.add_argument(
        "--quality_model_mode",
        choices=["train_if_missing", "retrain", "reuse", "disable"],
        default="train_if_missing",
        help="Quality model mode: train_if_missing/retrain/reuse/disable",
    )
    parser.add_argument(
        "--quality_model_output",
        type=str,
        default=None,
        help="Output path for dataset-specific quality_model.pkl (default: <artifacts>/models/quality_model.pkl)",
    )
    parser.add_argument(
        "--quality_bad_dice_threshold",
        type=float,
        default=0.90,
        help="Dice threshold used to derive bad labels when training quality model",
    )
    parser.add_argument(
        "--quality_classifier_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifier mode in quality model",
    )
    parser.add_argument(
        "--quality_label_mode",
        choices=["curated_only", "dice_with_curated_override"],
        default="curated_only",
        help=(
            "Label source mode for quality-model training: "
            "curated_only (default) or dice_with_curated_override (legacy)."
        ),
    )
    parser.add_argument(
        "--quality_train_max_cases",
        type=int,
        default=None,
        help="Optional cap on number of samples used to train quality model",
    )
    parser.add_argument(
        "--repair_kb_mode",
        choices=["build_if_missing", "rebuild", "reuse", "disable"],
        default="build_if_missing",
        help="Repair KB mode: build_if_missing/rebuild/reuse/disable",
    )
    parser.add_argument(
        "--repair_kb_dir",
        type=str,
        default=None,
        help="Repair KB directory (default: <artifacts>/repair_kb)",
    )
    parser.add_argument(
        "--repair_kb_max_cases",
        type=int,
        default=10,
        help="Max number of cases used to auto-build repair KB",
    )
    parser.add_argument(
        "--repair_kb_min_dice",
        type=float,
        default=0.70,
        help="Lower Dice bound (exclusive) when selecting cases for repair KB build",
    )
    parser.add_argument(
        "--repair_kb_max_dice",
        type=float,
        default=0.90,
        help="Upper Dice bound (exclusive) when selecting cases for repair KB build",
    )
    parser.add_argument(
        "--repair_kb_max_ops",
        type=int,
        default=18,
        help="Max operations sampled per case when auto-building repair KB",
    )
    parser.add_argument(
        "--repair_kb_improved_thr",
        type=float,
        default=0.005,
        help="Delta Dice threshold for labeling repair example as improved",
    )
    parser.add_argument(
        "--repair_kb_degraded_thr",
        type=float,
        default=-0.02,
        help="Delta Dice threshold for labeling repair example as degraded",
    )
    parser.add_argument(
        "--top_improved_k",
        type=int,
        default=8,
        help="Number of top improved cases to visualize (recommended: 5-10)",
    )
    parser.add_argument(
        "--top_improved_min_delta",
        type=float,
        default=0.0,
        help="Minimum Dice delta for top-improved visualization",
    )
    parser.add_argument(
        "--save_baseline_snapshot",
        action="store_true",
        help="Persist artifacts/runtime/baseline_metrics.json for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.repair_kb_min_dice) >= float(args.repair_kb_max_dice):
        raise ValueError(
            "repair_kb_min_dice must be smaller than repair_kb_max_dice "
            f"(got {args.repair_kb_min_dice} >= {args.repair_kb_max_dice})"
        )
    cfg_path = Path(args.config).resolve()
    source_root = Path(args.source_root).resolve()
    target_dir = Path(args.target_dir).resolve()

    cfg = _load_yaml(cfg_path)
    all_frames_dir, best_cases_dir, worst_cases_dir = _validate_source_root(source_root)
    output_dirs = _prepare_output_dirs(target_dir)

    llm_enabled = _parse_optional_bool(args.llm_enabled)
    vision_enabled = _parse_optional_bool(args.vision_enabled)
    strict_no_gt_online = _parse_optional_bool(args.strict_no_gt_online)

    # High-level knowledge policy (optional one-switch control).
    effective_quality_mode = args.quality_model_mode
    effective_repair_kb_mode = args.repair_kb_mode
    repair_kb_skip_if_exists = False
    if args.knowledge_mode == "auto_skip":
        effective_quality_mode = "train_if_missing"
        effective_repair_kb_mode = "build_if_missing"
        repair_kb_skip_if_exists = True
    elif args.knowledge_mode == "force_rebuild":
        effective_quality_mode = "retrain"
        effective_repair_kb_mode = "rebuild"
        repair_kb_skip_if_exists = False

    # Bind config paths to this experiment.
    cfg.setdefault("paths", {})
    cfg["paths"]["source_dir"] = str(all_frames_dir)
    cfg["paths"]["target_dir"] = str(target_dir)
    cfg["paths"]["gt_dir"] = str(all_frames_dir)
    if args.artifacts_dir:
        artifacts_root = Path(args.artifacts_dir).resolve()
    else:
        artifacts_root = output_dirs["artifacts"]
    output_dirs["artifacts"] = artifacts_root
    output_dirs["artifacts_atlas"] = artifacts_root / "atlas"
    output_dirs["artifacts_models"] = artifacts_root / "models"
    output_dirs["artifacts_repair_kb"] = artifacts_root / "repair_kb"
    output_dirs["artifacts_configs"] = artifacts_root / "config"
    output_dirs["artifacts_runtime"] = artifacts_root / "runtime"
    for p in [
        output_dirs["artifacts"],
        output_dirs["artifacts_atlas"],
        output_dirs["artifacts_models"],
        output_dirs["artifacts_repair_kb"],
        output_dirs["artifacts_configs"],
        output_dirs["artifacts_runtime"],
    ]:
        p.mkdir(parents=True, exist_ok=True)

    known_stems = _known_stems_from_all_frames(all_frames_dir)
    curated_labels, curated_counts, unresolved_files = _load_curated_labels(
        best_cases_dir, worst_cases_dir, known_stems
    )

    # Expose all visual resources from artifacts root so best/worst/all_frames are consistent.
    artifacts_visual_root = output_dirs["artifacts"]
    for name in ["best_cases", "worst_cases", "all_frames_export", "single_panel_refs"]:
        src = source_root / name
        dst = artifacts_visual_root / name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst, ignore_errors=True)
            else:
                dst.unlink()
        if src.exists():
            rel = os.path.relpath(src, artifacts_visual_root)
            try:
                dst.symlink_to(rel, target_is_directory=src.is_dir())
            except OSError:
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
    cfg["paths"]["visual_examples_dir"] = str(artifacts_visual_root)

    # Discover all cases once (reuse for prep + experiment filtering).
    all_cases_full = discover_cases(str(all_frames_dir), str(all_frames_dir), gt_dir=str(all_frames_dir))

    # Auto prepare quality model (dataset-specific).
    quality_model_path = (
        Path(args.quality_model_output).resolve()
        if args.quality_model_output
        else (output_dirs["artifacts_models"] / "quality_model.pkl")
    )
    quality_info = _prepare_quality_model(
        mode=effective_quality_mode,
        model_path=quality_model_path,
        all_frames_dir=all_frames_dir,
        curated_labels=curated_labels,
        bad_dice_threshold=float(args.quality_bad_dice_threshold),
        classifier_threshold=float(args.quality_classifier_threshold),
        label_mode=str(args.quality_label_mode),
        max_cases=args.quality_train_max_cases,
    )

    # Auto prepare repair KB.
    repair_kb_dir = (
        Path(args.repair_kb_dir).resolve()
        if args.repair_kb_dir
        else output_dirs["artifacts_repair_kb"]
    )
    repair_kb_info = _prepare_repair_kb(
        mode=effective_repair_kb_mode,
        kb_dir=repair_kb_dir,
        cases=all_cases_full,
        curated_labels=curated_labels,
        cfg=cfg,
        max_cases=int(args.repair_kb_max_cases),
        max_ops=int(args.repair_kb_max_ops),
        improved_thr=float(args.repair_kb_improved_thr),
        degraded_thr=float(args.repair_kb_degraded_thr),
        min_dice=float(args.repair_kb_min_dice),
        max_dice=float(args.repair_kb_max_dice),
        atlas=None,
        skip_if_exists=repair_kb_skip_if_exists,
    )

    atlas_path = output_dirs["artifacts_atlas"] / args.atlas_filename
    cfg["paths"]["atlas_path"] = str(atlas_path)
    cfg["paths"]["repair_kb_dir"] = str(repair_kb_dir)
    cfg["quality_model"] = cfg.get("quality_model", {})
    cfg["quality_model"]["path"] = str(quality_model_path)
    cfg["quality_model"]["enabled"] = (
        effective_quality_mode != "disable" and bool(quality_info.get("exists_after", False))
    )

    # Auto-build or reuse anatomy stats from current dataset predictions.
    _anatomy_stats_pkl = output_dirs["artifacts_models"] / "anatomy_stats.pkl"
    if _anatomy_stats_pkl.exists():
        print(f"Anatomy stats: reusing cached {_anatomy_stats_pkl}")
    else:
        from cardiac_agent_postproc.anatomy_score import (
            fit_source_stats as _fit_anatomy,
            save_stats as _save_anatomy,
        )
        _anatomy_dict = {}
        for _view in ["2c", "3c", "4c"]:
            try:
                _s = _fit_anatomy(str(all_frames_dir), view_type=_view)
                _anatomy_dict[_view] = _s
                print(f"  Anatomy stats {_view}: n={_s.n_samples}")
            except ValueError as _e:
                print(f"  Anatomy stats {_view}: insufficient samples — {_e}")
        if _anatomy_dict:
            _save_anatomy(_anatomy_dict, str(_anatomy_stats_pkl))
            print(f"Anatomy stats saved → {_anatomy_stats_pkl}")
        else:
            _anatomy_stats_pkl = None
    if _anatomy_stats_pkl is not None:
        cfg["anatomy_stats_path"] = str(_anatomy_stats_pkl)

    cfg.setdefault("llm", {})
    cfg.setdefault("vision_guardrail", {})
    cfg.setdefault("triage", {})
    cfg.setdefault("diagnosis", {})
    cfg.setdefault("verifier", {})
    cfg.setdefault("agents", {})
    cfg["agents"].setdefault("triage", {})
    cfg["agents"]["triage"].setdefault("quality_model", {})
    cfg["agents"]["triage"]["quality_model"]["path"] = str(quality_model_path)
    cfg["agents"]["triage"]["quality_model"]["enabled"] = bool(cfg["quality_model"]["enabled"])
    cfg["agents"].setdefault("diagnosis", {})
    cfg["agents"].setdefault("planner", {})
    cfg["agents"].setdefault("verifier", {})
    cfg.setdefault("multi_agent", {})
    cfg["verifier"].setdefault("thresholds", {})
    cfg["verifier"]["thresholds"].setdefault("reject", 45.0)
    cfg["verifier"]["thresholds"].setdefault("approve", 65.0)
    cfg["verifier"]["thresholds"].setdefault("per_view", {})
    cfg["verifier"]["thresholds"]["per_view"].setdefault(
        "2CH", {"reject": 43.0, "approve": 63.0}
    )
    cfg["verifier"]["thresholds"]["per_view"].setdefault(
        "3CH", {"reject": 45.0, "approve": 65.0}
    )
    cfg["verifier"]["thresholds"]["per_view"].setdefault(
        "4CH", {"reject": 47.0, "approve": 67.0}
    )
    cfg.setdefault("agents", {}).setdefault("verifier", {})
    cfg["agents"]["verifier"].setdefault("knowledge_top_k", 2)

    if llm_enabled is not None:
        cfg["llm"]["enabled"] = bool(llm_enabled)
    if vision_enabled is not None:
        cfg["vision_guardrail"]["enabled"] = bool(vision_enabled)
        cfg["diagnosis"]["vlm_enabled"] = bool(vision_enabled)
        cfg["triage"]["vlm_enabled"] = bool(vision_enabled)
        cfg["agents"]["diagnosis"]["vlm_enabled"] = bool(vision_enabled)
    if args.triage_n_refs is not None:
        cfg["agents"]["triage"]["vlm_n_refs"] = int(args.triage_n_refs)
    if strict_no_gt_online is not None:
        cfg["multi_agent"]["strict_no_gt_online"] = bool(strict_no_gt_online)
    else:
        cfg["multi_agent"].setdefault("strict_no_gt_online", False)
    verifier_thresholds = _verifier_thresholds_snapshot(cfg)

    # Persist resolved runtime config for reproducibility.
    resolved_cfg_path = output_dirs["artifacts_configs"] / "resolved_runtime_config.yaml"
    with resolved_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    runtime_args_path = output_dirs["artifacts_runtime"] / "runtime_args.json"
    with runtime_args_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    if args.max_rounds is not None:
        cfg.setdefault("multi_agent", {})
        cfg["multi_agent"]["max_rounds_per_case"] = int(args.max_rounds)
    max_rounds = int(cfg.get("multi_agent", {}).get("max_rounds_per_case", 2))

    print(f"Config: {cfg_path}")
    print(f"Source root: {source_root}")
    print(f"All frames: {all_frames_dir}")
    print(f"Known stems in all_frames_export: {len(known_stems)}")
    print(
        "Curated labels resolved: "
        f"best={curated_counts['resolved_best']}/{curated_counts['best_files']}, "
        f"worst={curated_counts['resolved_worst']}/{curated_counts['worst_files']}"
    )
    print(
        "Knowledge policy: "
        f"mode={args.knowledge_mode}, "
        f"quality={effective_quality_mode}, repair_kb={effective_repair_kb_mode}, "
        f"repair_skip_if_exists={repair_kb_skip_if_exists}"
    )
    print(
        "Verifier online policy: "
        f"strict_no_gt_online={cfg.get('multi_agent', {}).get('strict_no_gt_online', False)}, "
        f"thresholds={verifier_thresholds}"
    )
    if unresolved_files:
        print(f"Warning: {len(unresolved_files)} curated files could not map to stems.")
    print(
        "Quality model: "
        f"path={quality_model_path}, "
        f"label_mode={args.quality_label_mode}, "
        f"trained={quality_info.get('trained', False)}, "
        f"used_existing={quality_info.get('used_existing', False)}, "
        f"enabled={cfg.get('quality_model', {}).get('enabled', False)}"
    )
    if quality_info.get("train_error"):
        print(f"Warning: quality model training issue: {quality_info.get('train_error')}")
    print(
        "Repair KB: "
        f"path={repair_kb_dir}, "
        f"built={repair_kb_info.get('built', False)}, "
        f"used_existing={repair_kb_info.get('used_existing', False)}, "
        f"usable={repair_kb_info.get('usable_after', False)}, "
        f"dice_band=({args.repair_kb_min_dice:.2f},{args.repair_kb_max_dice:.2f}), "
        f"signature_match={repair_kb_info.get('signature_match', False)}"
    )

    bus, agents = _build_agent_stack(
        cfg=cfg,
        source_root=Path(cfg["paths"]["visual_examples_dir"]),
        all_frames_dir=all_frames_dir,
        atlas_mode=args.atlas_mode,
        atlas_path=atlas_path,
    )
    all_cases = list(all_cases_full)
    all_cases = _filter_cases(all_cases, case_contains=args.case_contains, limit=args.limit)
    group_and_link_neighbors(all_cases)
    print(f"Cases to process: {len(all_cases)}")

    # Stage 1: Triage + metrics.
    triage_csv = output_dirs["triage"] / "triage_results.csv"
    if args.triage_csv:
        # Skip triage and reuse supplied CSV
        print(f"[Stage 1] Skipping triage — loading existing results from: {args.triage_csv}")
        triage_df = pd.read_csv(args.triage_csv)
        import shutil as _shutil
        _shutil.copy(args.triage_csv, triage_csv)
        print(f"Copied triage results to: {triage_csv}")
    else:
        triage_df = _run_triage_stage(
            cases=all_cases,
            triage_agent=agents["triage"],
            bus=bus,
            triage_debug_dir=output_dirs["triage_debug"],
        )
        triage_df.to_csv(triage_csv, index=False)
        print(f"Saved triage results: {triage_csv}")

    eval_rows, triage_metrics = _evaluate_triage_curated(triage_df, curated_labels)
    eval_csv = output_dirs["triage"] / "triage_curated_eval_rows.csv"
    eval_rows.to_csv(eval_csv, index=False)
    triage_metrics_csv = output_dirs["triage"] / "triage_curated_binary_metrics.csv"
    _save_metrics_dict(triage_metrics, triage_metrics_csv)

    group_stats_df, group_summary = _triage_group_dice_stats(triage_df)
    group_stats_csv = output_dirs["triage"] / "triage_dice_group_stats.csv"
    group_stats_df.to_csv(group_stats_csv, index=False)
    group_summary_csv = output_dirs["triage"] / "triage_dice_group_summary.csv"
    _save_metrics_dict(group_summary, group_summary_csv)

    # Stage 2: Repair on need-fix set.
    repair_df = pd.DataFrame()
    repair_summary: Dict[str, Any] = {"n_cases": 0, "skipped": True}
    verifier_calib_csv = output_dirs["summary"] / "verifier_score_calibration.csv"
    verifier_thresh_json = output_dirs["summary"] / "verifier_threshold_recommendation.json"
    verifier_calib_df = pd.DataFrame()
    verifier_thresh_reco: Dict[str, Any] = {}
    if not args.skip_repair:
        repair_stems = _build_repair_queue(triage_df, curated_labels, mode=args.repair_subset)
        print(f"Repair subset ({args.repair_subset}): {len(repair_stems)} stems")
        repair_df = _run_repair_stage(
            cases=all_cases,
            repair_stems=repair_stems,
            agents=agents,
            bus=bus,
            repair_debug_dir=output_dirs["repair_debug"],
            repaired_labels_dir=output_dirs["repaired_labels"],
            max_rounds=max_rounds,
            atlas_mode=args.atlas_mode,
            cfg=cfg,
            local_atlas_dir=output_dirs["artifacts_atlas"] / "per_sample" if args.atlas_mode == "local_per_sample" else None,
            atlas_source_cases=list(all_cases_full) if args.atlas_mode == "local_per_sample" else None,
        )
        repair_csv = output_dirs["repair"] / "need_fix_diagnosis_repair_results.csv"
        repair_df.to_csv(repair_csv, index=False)
        repair_summary = _summarize_repair(repair_df)
        repair_summary_csv = output_dirs["repair"] / "repair_summary_metrics.csv"
        _save_metrics_dict(repair_summary, repair_summary_csv)
        print(f"Saved repair results: {repair_csv}")

        # Verifier score calibration artifacts.
        verifier_calib_df = _score_bin_table(repair_df, score_col="online_score")
        verifier_calib_df.to_csv(verifier_calib_csv, index=False)
        verifier_thresh_reco = _recommend_thresholds_from_bins(
            verifier_calib_df,
            default_reject=float(verifier_thresholds.get("reject", 45.0)),
            default_approve=float(verifier_thresholds.get("approve", 65.0)),
        )
        verifier_thresh_reco["thresholds_used"] = verifier_thresholds
        verifier_thresh_reco["score_column"] = "online_score"
        verifier_thresh_reco["calibration_csv"] = str(verifier_calib_csv)
        with verifier_thresh_json.open("w", encoding="utf-8") as f:
            json.dump(verifier_thresh_reco, f, ensure_ascii=False, indent=2)

        # Optional baseline snapshot artifact for strict before/after comparability.
        if args.save_baseline_snapshot:
            baseline_payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "strict_no_gt_online": bool(cfg.get("multi_agent", {}).get("strict_no_gt_online", False)),
                "repair_summary": repair_summary,
                "verifier_thresholds": verifier_thresholds,
            }
            baseline_path = output_dirs["artifacts_runtime"] / "baseline_metrics.json"
            with baseline_path.open("w", encoding="utf-8") as f:
                json.dump(baseline_payload, f, ensure_ascii=False, indent=2)
            print(f"Saved baseline snapshot: {baseline_path}")

    # Stage 3: MICCAI-style figures.
    figure_paths: Dict[str, Any] = {}
    roc_paths = _plot_triage_roc(eval_rows, output_dirs["figures"] / "fig1_triage_roc")
    if roc_paths:
        figure_paths["fig1_triage_roc"] = roc_paths
    g_paths = _plot_triage_dice_groups(triage_df, output_dirs["figures"] / "fig2_triage_dice_groups")
    if g_paths:
        figure_paths["fig2_triage_dice_groups"] = g_paths
    ba_paths = _plot_repair_before_after(
        repair_df, output_dirs["figures"] / "fig3_need_fix_before_after_dice"
    )
    if ba_paths:
        figure_paths["fig3_need_fix_before_after_dice"] = ba_paths
    dh_paths = _plot_repair_delta_hist(
        repair_df, output_dirs["figures"] / "fig4_need_fix_dice_delta_hist"
    )
    if dh_paths:
        figure_paths["fig4_need_fix_dice_delta_hist"] = dh_paths
    top_imp = _plot_top_improved_cases(
        repair_df=repair_df,
        all_frames_dir=all_frames_dir,
        out_dir=output_dirs["figures"] / "top_improved_cases",
        top_k=args.top_improved_k,
        min_delta=args.top_improved_min_delta,
    )
    figure_paths["top_improved_cases"] = top_imp

    summary = {
        "config_path": str(cfg_path),
        "source_root": str(source_root),
        "all_frames_dir": str(all_frames_dir),
        "target_dir": str(target_dir),
        "n_cases_processed": int(len(all_cases)),
        "curated_label_counts": curated_counts,
        "unresolved_curated_files": unresolved_files,
        "triage_outputs": {
            "triage_csv": str(triage_csv),
            "triage_curated_eval_rows_csv": str(eval_csv),
            "triage_curated_binary_metrics_csv": str(triage_metrics_csv),
            "triage_dice_group_stats_csv": str(group_stats_csv),
            "triage_dice_group_summary_csv": str(group_summary_csv),
            "metrics": triage_metrics,
            "group_summary": group_summary,
        },
        "repair_outputs": {
            "repair_subset_mode": args.repair_subset,
            "skip_repair": bool(args.skip_repair),
            "metrics": repair_summary,
            "verifier_thresholds_used": verifier_thresholds,
            "verifier_score_calibration_csv": str(verifier_calib_csv),
            "verifier_threshold_recommendation_json": str(verifier_thresh_json),
            "verifier_threshold_recommendation": verifier_thresh_reco,
            "gain_distribution_by_score_bin": (
                verifier_calib_df.to_dict(orient="records")
                if not verifier_calib_df.empty
                else []
            ),
        },
        "preparation": {
            "quality_model": quality_info,
            "repair_kb": repair_kb_info,
        },
        "runtime_overrides": {
            "atlas_mode": args.atlas_mode,
            "atlas_path": str(atlas_path),
            "llm_enabled": cfg.get("llm", {}).get("enabled", True),
            "vision_enabled": cfg.get("vision_guardrail", {}).get("enabled", True),
            "strict_no_gt_online": bool(cfg.get("multi_agent", {}).get("strict_no_gt_online", False)),
            "verifier_thresholds": verifier_thresholds,
            "triage_n_refs": cfg.get("agents", {}).get("triage", {}).get("vlm_n_refs"),
            "knowledge_mode": args.knowledge_mode,
            "quality_model_mode": effective_quality_mode,
            "quality_label_mode": args.quality_label_mode,
            "quality_model_path": str(quality_model_path),
            "repair_kb_mode": effective_repair_kb_mode,
            "repair_kb_skip_if_exists": repair_kb_skip_if_exists,
            "repair_kb_dir": str(repair_kb_dir),
            "artifacts_dir": str(output_dirs["artifacts"]),
            "save_baseline_snapshot": bool(args.save_baseline_snapshot),
            "baseline_metrics_json": str(output_dirs["artifacts_runtime"] / "baseline_metrics.json"),
            "resolved_runtime_config": str(resolved_cfg_path),
            "runtime_args_json": str(runtime_args_path),
        },
        "figures": figure_paths,
    }
    summary_path = output_dirs["summary"] / "experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Experiment Summary ===")
    print(f"Triage CSV: {triage_csv}")
    print(f"Triage curated metrics: {triage_metrics_csv}")
    print(f"Artifacts dir: {output_dirs['artifacts']}")
    print(f"Atlas path: {atlas_path}")
    print(f"Quality model path: {quality_model_path}")
    print(f"Repair KB dir: {repair_kb_dir}")
    print(f"Strict no-GT online: {cfg.get('multi_agent', {}).get('strict_no_gt_online', False)}")
    print(f"Verifier thresholds: {verifier_thresholds}")
    print(f"Verifier calibration CSV: {verifier_calib_csv}")
    print(f"Verifier threshold recommendation: {verifier_thresh_json}")
    if not args.skip_repair:
        print(f"Repair cases: {repair_summary.get('n_cases', 0)}")
        print(f"Repair mean Dice delta: {repair_summary.get('delta_dice_mean', float('nan')):+.4f}")
        print(
            "Top improved visualization: "
            f"{figure_paths.get('top_improved_cases', {}).get('n_selected', 0)} cases"
        )
    print(f"Figures saved in: {output_dirs['figures']}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
