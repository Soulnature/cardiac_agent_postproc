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
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
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
        "artifacts_refs4": target_dir / "artifacts" / "triage_4way_collections_gold",
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


def _build_four_way_ref_artifacts_from_curated(
    source_root: Path,
    all_frames_dir: Path,
    refs4_artifact_dir: Path,
    known_stems: List[str],
    copy_mode: str = "symlink",
    good_low_dice_max: float = 0.93,
    bad_high_dice_min: float = 0.90,
) -> Dict[str, Any]:
    overlays_dir = refs4_artifact_dir / "overlays"
    categories_root = refs4_artifact_dir / "categories"
    for name in ["01_good", "02_good_low_dice", "03_bad_high_dice", "04_bad"]:
        (categories_root / name).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    category_counter: Counter[str] = Counter()

    def _cat_from_label(label: str, mean_dice: float) -> str:
        if label == "good":
            if _is_finite(mean_dice) and mean_dice < good_low_dice_max:
                return "02_good_low_dice"
            return "01_good"
        if _is_finite(mean_dice) and mean_dice > bad_high_dice_min:
            return "03_bad_high_dice"
        return "04_bad"

    for folder_name, gold_label in [("best_cases", "good"), ("worst_cases", "bad")]:
        folder = source_root / folder_name
        if not folder.is_dir():
            continue
        for item in sorted(folder.iterdir()):
            if not item.is_file():
                continue
            stem = _match_stem_from_filename(item.name, known_stems)
            if stem is None:
                unresolved.append(str(item))
                continue

            mean_dice = _extract_dice_prefix(item.name)
            category = _cat_from_label(gold_label, mean_dice)
            issue_type = _parse_issue_from_curated_filename(item.name)
            overlay_path = _materialize_overlay_for_stem(
                stem=stem,
                source_root=source_root,
                all_frames_dir=all_frames_dir,
                overlays_dir=overlays_dir,
                copy_mode=copy_mode,
            )
            if not overlay_path:
                unresolved.append(str(item))
                continue
            cat_overlay = categories_root / category / f"{stem}_overlay.png"
            _safe_link_or_copy(Path(overlay_path), cat_overlay, copy_mode=copy_mode)

            view = _infer_view_from_stem(stem)
            rows.append(
                {
                    "stem": stem,
                    "category": category,
                    "gold_label": gold_label,
                    "issue_type": issue_type,
                    "mean_dice": float(mean_dice) if _is_finite(mean_dice) else float("nan"),
                    "was_disc_c2": bool("disc" in issue_type.lower()),
                    "view": view,
                    "img_path": str(all_frames_dir / f"{stem}_img.png"),
                    "pred_path": str(all_frames_dir / f"{stem}_pred.png"),
                    "gt_path": str(all_frames_dir / f"{stem}_gt.png"),
                    "overlay_path": str(overlay_path),
                    "category_overlay_path": str(cat_overlay),
                }
            )
            category_counter[category] += 1

    manifest_path = refs4_artifact_dir / "triage_4way_manifest.csv"
    summary_path = refs4_artifact_dir / "triage_4way_summary.csv"
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)

    if not df.empty:
        summary = (
            df.groupby("category", dropna=False)["stem"]
            .count()
            .reset_index()
            .rename(columns={"stem": "n"})
            .sort_values("category")
        )
    else:
        summary = pd.DataFrame(columns=["category", "n"])
    summary.to_csv(summary_path, index=False)

    return {
        "generated_manifest": True,
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "n_rows": int(len(df)),
        "category_counts": {k: int(v) for k, v in sorted(category_counter.items())},
        "unresolved_files": unresolved,
    }


def _prepare_four_way_ref_artifacts(
    source_root: Path,
    all_frames_dir: Path,
    known_stems: List[str],
    refs4_artifact_dir: Path,
    copy_mode: str = "symlink",
    build_mode: str = "build_if_missing",
    good_low_dice_max: float = 0.93,
    bad_high_dice_min: float = 0.90,
) -> Dict[str, Any]:
    """
    Prepare 4-way reference folders under artifacts.
    Priority:
      1) copy/symlink existing source_root/triage_4way_collections_gold
      2) if missing (or forced), generate from best_cases/worst_cases
      3) fallback: create empty 4-category folder skeleton.
    """
    refs4_artifact_dir.mkdir(parents=True, exist_ok=True)
    src = source_root / "triage_4way_collections_gold"
    used_existing = False
    generated = False
    generated_info: Dict[str, Any] = {}

    # Clean previous contents for deterministic reruns.
    _clear_dir(refs4_artifact_dir)

    src_manifest = src / "triage_4way_manifest.csv"
    can_reuse_existing = src.exists() and src.is_dir() and src_manifest.exists()

    if build_mode in {"reuse", "build_if_missing"} and can_reuse_existing:
        used_existing = True
        if copy_mode == "copy":
            shutil.copytree(src, refs4_artifact_dir, dirs_exist_ok=True)
        else:
            # Symlink all first-level children for lightweight artifact materialization.
            for child in src.iterdir():
                dst = refs4_artifact_dir / child.name
                _safe_link_or_copy(child, dst, copy_mode=copy_mode)

    need_generate = build_mode == "rebuild" or (
        build_mode == "build_if_missing" and not used_existing
    )
    if need_generate:
        generated_info = _build_four_way_ref_artifacts_from_curated(
            source_root=source_root,
            all_frames_dir=all_frames_dir,
            refs4_artifact_dir=refs4_artifact_dir,
            known_stems=known_stems,
            copy_mode=copy_mode,
            good_low_dice_max=good_low_dice_max,
            bad_high_dice_min=bad_high_dice_min,
        )
        generated = bool(generated_info.get("n_rows", 0) > 0)

    categories_root = refs4_artifact_dir / "categories"
    for name in ["01_good", "02_good_low_dice", "03_bad_high_dice", "04_bad"]:
        (categories_root / name).mkdir(parents=True, exist_ok=True)

    manifest_path = refs4_artifact_dir / "triage_4way_manifest.csv"
    if not manifest_path.exists():
        pd.DataFrame(
            columns=[
                "stem",
                "category",
                "gold_label",
                "issue_type",
                "mean_dice",
                "was_disc_c2",
                "view",
                "img_path",
                "pred_path",
                "gt_path",
                "overlay_path",
                "category_overlay_path",
            ]
        ).to_csv(manifest_path, index=False)

    return {
        "path": str(refs4_artifact_dir),
        "categories_root": str(categories_root),
        "used_existing_collections": used_existing,
        "generated_from_curated": generated,
        "generated_info": generated_info,
        "build_mode": build_mode,
        "copy_mode": copy_mode,
        "manifest_path": str(manifest_path),
    }


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


def _select_repair_kb_stems(
    cases: List[Any],
    curated_labels: Dict[str, str],
    max_cases: int,
) -> List[str]:
    by_stem = {c.stem: c for c in cases}
    curated_bad = [
        s for s, lab in curated_labels.items() if lab == "needs_fix" and s in by_stem and by_stem[s].gt_path
    ]
    if curated_bad:
        return sorted(curated_bad)[:max(1, max_cases)]

    scored: List[Tuple[float, str]] = []
    for ctx in cases:
        if not ctx.gt_path or not os.path.exists(ctx.gt_path):
            continue
        try:
            d = _compute_dice(ctx.mask, ctx.gt_path)["Dice_Mean"]
            if _is_finite(d):
                scored.append((float(d), ctx.stem))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0])
    return [stem for _, stem in scored[: max(1, max_cases)]]


def _build_repair_kb_from_cases(
    cases: List[Any],
    curated_labels: Dict[str, str],
    kb_dir: Path,
    cfg: Dict[str, Any],
    max_cases: int = 24,
    max_ops: int = 18,
    improved_thr: float = 0.005,
    degraded_thr: float = -0.02,
    atlas: Optional[dict] = None,
) -> Dict[str, Any]:
    kb_dir.mkdir(parents=True, exist_ok=True)
    _clear_dir(kb_dir)
    for cat in ["improved", "degraded", "neutral"]:
        (kb_dir / cat).mkdir(parents=True, exist_ok=True)

    selected_stems = _select_repair_kb_stems(cases, curated_labels, max_cases=max_cases)
    by_stem = {c.stem: c for c in cases}
    groups = group_and_link_neighbors(cases)
    del groups  # group_and_link_neighbors mutates case contexts with neighbor maps

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

    counts = _repair_kb_counts(kb_dir)
    build_info = {
        **build_stats,
        "counts": counts,
        "selected_stems": selected_stems,
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
    max_cases: int = 24,
    max_ops: int = 18,
    improved_thr: float = 0.005,
    degraded_thr: float = -0.02,
    atlas: Optional[dict] = None,
    skip_if_exists: bool = False,
) -> Dict[str, Any]:
    kb_dir.mkdir(parents=True, exist_ok=True)
    existing_counts = _repair_kb_counts(kb_dir)
    has_any_existing = _repair_kb_has_any(kb_dir)
    has_existing = _repair_kb_usable(kb_dir)

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
    }

    if mode == "disable":
        return info

    if mode == "reuse":
        info["used_existing"] = has_existing
        info["usable_after"] = has_existing
        return info

    if mode == "build_if_missing":
        if skip_if_exists and has_any_existing:
            info["used_existing"] = True
            info["usable_after"] = has_existing
            return info
        if has_existing:
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
        atlas=atlas,
    )
    info["built"] = True
    info["build_info"] = build_info
    info["counts_after"] = _repair_kb_counts(kb_dir)
    info["usable_after"] = _repair_kb_usable(kb_dir)
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
        m = c.current_mask if c.current_mask is not None else (c.original_mask if c.original_mask is not None else c.mask)
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

    atlas_cases = atlas_source_cases if atlas_source_cases is not None else cases
    groups = group_and_link_neighbors(atlas_cases)
    rows: List[Dict[str, Any]] = []
    coordinator: CoordinatorAgent = agents["coordinator"]
    executor: ExecutorAgent = agents["executor"]
    global_atlas = executor._atlas

    for ctx in tqdm(queue, desc="Stage2 Repair"):
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

        if verdict == "approved" and ctx.current_mask is not None:
            final_mask = ctx.current_mask.copy()
            saved_mask_type = "repaired"
        elif (
            getattr(ctx, "best_intermediate_mask", None) is not None
            and not bool(getattr(ctx, "best_intermediate_is_original", True))
        ):
            final_mask = ctx.best_intermediate_mask.copy()
            saved_mask_type = "best_intermediate"
        else:
            fallback = ctx.original_mask if ctx.original_mask is not None else ctx.mask
            final_mask = fallback.copy() if fallback is not None else None
            saved_mask_type = "original"

        if final_mask is not None:
            ctx.current_mask = final_mask.copy()
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
            "VerifyConfidence_Raw": _safe_float(last_verify.get("confidence", float("nan"))),
            "VerifyCompareVerdict": str(last_cmp.get("verdict", "")),
            "VerifyCompareConfidence": _safe_float(last_cmp.get("confidence", float("nan"))),
            "VerifyCompareReference": str(last_verify.get("comparison_reference", "")),
            "VerifyProxyVerdict": str(last_proxy.get("proxy_verdict", "")),
            "VerifyProxyScore": _safe_float(last_proxy.get("proxy_score", float("nan"))),
            "VerifyProxyConfidence": _safe_float(last_proxy.get("proxy_confidence", float("nan"))),
            "VerifyConsensusScore": _safe_float(last_proxy.get("consensus_score", float("nan"))),
            "VerifyConsensusVerdict": str(last_proxy.get("consensus_verdict", "")),
            "VerifyUnlockApplied": bool(last_proxy.get("unlock_applied", False)),
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
        "--triage_four_way_refs",
        type=str,
        default=None,
        help="Override agents.triage.vlm_four_way_refs (true/false)",
    )
    parser.add_argument(
        "--triage_ref_categories",
        type=str,
        default=None,
        help="Comma-separated triage ref categories, e.g. 01_good,02_good_low_dice,03_bad_high_dice,04_bad",
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
            "custom: use detailed refs4/quality/repair mode args."
        ),
    )
    parser.add_argument(
        "--refs4_build_mode",
        choices=["reuse", "build_if_missing", "rebuild"],
        default="build_if_missing",
        help="4-way triage refs mode: reuse existing, build if missing, or force rebuild from curated folders",
    )
    parser.add_argument(
        "--refs4_good_low_dice_max",
        type=float,
        default=0.93,
        help="Dice upper bound for 02_good_low_dice when auto-building 4-way refs",
    )
    parser.add_argument(
        "--refs4_bad_high_dice_min",
        type=float,
        default=0.90,
        help="Dice lower bound for 03_bad_high_dice when auto-building 4-way refs",
    )
    parser.add_argument(
        "--refs_copy_mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize 4-way ref folders into artifacts",
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
        default=24,
        help="Max number of cases used to auto-build repair KB",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    source_root = Path(args.source_root).resolve()
    target_dir = Path(args.target_dir).resolve()

    cfg = _load_yaml(cfg_path)
    all_frames_dir, best_cases_dir, worst_cases_dir = _validate_source_root(source_root)
    output_dirs = _prepare_output_dirs(target_dir)

    llm_enabled = _parse_optional_bool(args.llm_enabled)
    vision_enabled = _parse_optional_bool(args.vision_enabled)
    triage_four_way_refs = _parse_optional_bool(args.triage_four_way_refs)

    # High-level knowledge policy (optional one-switch control).
    effective_refs4_mode = args.refs4_build_mode
    effective_quality_mode = args.quality_model_mode
    effective_repair_kb_mode = args.repair_kb_mode
    repair_kb_skip_if_exists = False
    if args.knowledge_mode == "auto_skip":
        effective_refs4_mode = "build_if_missing"
        effective_quality_mode = "train_if_missing"
        effective_repair_kb_mode = "build_if_missing"
        repair_kb_skip_if_exists = True
    elif args.knowledge_mode == "force_rebuild":
        effective_refs4_mode = "rebuild"
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
    output_dirs["artifacts_refs4"] = artifacts_root / "triage_4way_collections_gold"
    output_dirs["artifacts_models"] = artifacts_root / "models"
    output_dirs["artifacts_repair_kb"] = artifacts_root / "repair_kb"
    output_dirs["artifacts_configs"] = artifacts_root / "config"
    output_dirs["artifacts_runtime"] = artifacts_root / "runtime"
    for p in [
        output_dirs["artifacts"],
        output_dirs["artifacts_atlas"],
        output_dirs["artifacts_refs4"],
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

    # Materialize 4-way refs into artifacts and point VisualKnowledge loading there.
    refs_info = _prepare_four_way_ref_artifacts(
        source_root=source_root,
        all_frames_dir=all_frames_dir,
        known_stems=known_stems,
        refs4_artifact_dir=output_dirs["artifacts_refs4"],
        copy_mode=args.refs_copy_mode,
        build_mode=effective_refs4_mode,
        good_low_dice_max=float(args.refs4_good_low_dice_max),
        bad_high_dice_min=float(args.refs4_bad_high_dice_min),
    )
    cfg["paths"]["visual_examples_dir"] = str(source_root)
    manifest_ready = Path(refs_info.get("manifest_path", "")).exists()
    if manifest_ready:
        # Expose all visual resources from artifacts root so refs/best/worst/all_frames are consistent.
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

    cfg.setdefault("llm", {})
    cfg.setdefault("vision_guardrail", {})
    cfg.setdefault("triage", {})
    cfg.setdefault("diagnosis", {})
    cfg.setdefault("agents", {})
    cfg["agents"].setdefault("triage", {})
    cfg["agents"]["triage"].setdefault("quality_model", {})
    cfg["agents"]["triage"]["quality_model"]["path"] = str(quality_model_path)
    cfg["agents"]["triage"]["quality_model"]["enabled"] = bool(cfg["quality_model"]["enabled"])
    cfg["agents"].setdefault("diagnosis", {})
    cfg["agents"].setdefault("planner", {})
    cfg["agents"].setdefault("verifier", {})

    if llm_enabled is not None:
        cfg["llm"]["enabled"] = bool(llm_enabled)
    if vision_enabled is not None:
        cfg["vision_guardrail"]["enabled"] = bool(vision_enabled)
        cfg["diagnosis"]["vlm_enabled"] = bool(vision_enabled)
        cfg["triage"]["vlm_enabled"] = bool(vision_enabled)
        cfg["agents"]["diagnosis"]["vlm_enabled"] = bool(vision_enabled)
    if args.triage_n_refs is not None:
        cfg["agents"]["triage"]["vlm_n_refs"] = int(args.triage_n_refs)
    if triage_four_way_refs is not None:
        cfg["agents"]["triage"]["vlm_four_way_refs"] = bool(triage_four_way_refs)
    if args.triage_ref_categories:
        categories = [x.strip() for x in args.triage_ref_categories.split(",") if x.strip()]
        if categories:
            cfg["agents"]["triage"]["vlm_ref_categories"] = categories

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
        f"mode={args.knowledge_mode}, refs4={effective_refs4_mode}, "
        f"quality={effective_quality_mode}, repair_kb={effective_repair_kb_mode}, "
        f"repair_skip_if_exists={repair_kb_skip_if_exists}"
    )
    if unresolved_files:
        print(f"Warning: {len(unresolved_files)} curated files could not map to stems.")
    print(
        "4-way refs: "
        f"existing={refs_info.get('used_existing_collections', False)}, "
        f"generated={refs_info.get('generated_from_curated', False)}, "
        f"manifest={refs_info.get('manifest_path', '')}"
    )
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
        f"usable={repair_kb_info.get('usable_after', False)}"
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
    triage_df = _run_triage_stage(
        cases=all_cases,
        triage_agent=agents["triage"],
        bus=bus,
        triage_debug_dir=output_dirs["triage_debug"],
    )
    triage_csv = output_dirs["triage"] / "triage_results.csv"
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
        },
        "preparation": {
            "refs4": refs_info,
            "quality_model": quality_info,
            "repair_kb": repair_kb_info,
        },
        "runtime_overrides": {
            "atlas_mode": args.atlas_mode,
            "atlas_path": str(atlas_path),
            "llm_enabled": cfg.get("llm", {}).get("enabled", True),
            "vision_enabled": cfg.get("vision_guardrail", {}).get("enabled", True),
            "triage_n_refs": cfg.get("agents", {}).get("triage", {}).get("vlm_n_refs"),
            "triage_four_way_refs": cfg.get("agents", {}).get("triage", {}).get("vlm_four_way_refs"),
            "triage_ref_categories": cfg.get("agents", {}).get("triage", {}).get("vlm_ref_categories"),
            "knowledge_mode": args.knowledge_mode,
            "refs4_build_mode": effective_refs4_mode,
            "quality_model_mode": effective_quality_mode,
            "quality_label_mode": args.quality_label_mode,
            "quality_model_path": str(quality_model_path),
            "repair_kb_mode": effective_repair_kb_mode,
            "repair_kb_skip_if_exists": repair_kb_skip_if_exists,
            "repair_kb_dir": str(repair_kb_dir),
            "artifacts_dir": str(output_dirs["artifacts"]),
            "refs4_info": refs_info,
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
