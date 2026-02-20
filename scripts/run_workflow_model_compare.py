#!/usr/bin/env python3
"""
Run the same MICCAI workflow across multiple LLM/VLM provider configs.

Default model set:
  - ollama_ministral -> config/ollama_ministral_triage.yaml
  - azure_openai     -> config/azure_openai_medrag.yaml
  - openai           -> config/openai.yaml
  - gemini           -> config/gemini.yaml

Example:
  python scripts/run_workflow_model_compare.py \
    --source_root results/Input_MnM2 \
    --output_root results/exp_model_compare_mnm2 \
    --limit 5 \
    --knowledge_mode auto_skip \
    --atlas_mode reuse_global
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml

DEFAULT_MODEL_CONFIGS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("ollama_ministral", "config/ollama_ministral_triage.yaml"),
        ("azure_openai", "config/azure_openai_medrag.yaml"),
        ("openai", "config/openai.yaml"),
        ("gemini", "config/gemini.yaml"),
    ]
)

SUMMARY_COLUMNS: List[str] = [
    "model_alias",
    "status",
    "exit_code",
    "elapsed_sec",
    "config_path",
    "llm_provider",
    "llm_model",
    "n_cases_processed",
    "triage_good_count",
    "triage_needs_fix_count",
    "triage_acc",
    "triage_f1_needs_fix",
    "repair_n_cases",
    "repair_n_approved",
    "repair_improved_count",
    "repair_worsened_count",
    "repair_pre_dice_mean",
    "repair_post_dice_mean",
    "repair_delta_dice_mean",
    "target_dir",
    "run_log",
    "summary_json",
]


def _parse_name_path(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH format, got: {text}")
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Invalid NAME=PATH value: {text}")
    return name, path


def _default_output_root() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results/exp_model_compare_{ts}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/run_miccai_multiagent_experiment.py on multiple provider configs "
            "and write a unified comparison summary."
        )
    )
    parser.add_argument(
        "--source_root",
        required=True,
        help="Dataset root containing all_frames_export/best_cases/worst_cases",
    )
    parser.add_argument(
        "--output_root",
        default=_default_output_root(),
        help="Root dir for per-model outputs and comparison summary",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_CONFIGS.keys()),
        help="Comma-separated model aliases to run",
    )
    parser.add_argument(
        "--model_config",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override/add model config mapping, repeatable",
    )
    parser.add_argument(
        "--runner_script",
        default="scripts/run_miccai_multiagent_experiment.py",
        help="Runner script path",
    )
    parser.add_argument(
        "--shared_artifacts_dir",
        default=None,
        help=(
            "Shared artifacts dir reused across models. "
            "Default: <output_root>/artifacts_shared"
        ),
    )
    parser.add_argument(
        "--knowledge_mode",
        choices=["custom", "auto_skip", "force_rebuild"],
        default="auto_skip",
        help="Knowledge policy for all model runs",
    )
    parser.add_argument(
        "--atlas_mode",
        choices=["rebuild_global", "reuse_global", "disable_global", "local_per_sample"],
        default="reuse_global",
        help="Atlas policy for all model runs",
    )
    parser.add_argument(
        "--repair_subset",
        choices=["triage_need_fix", "worst_cases"],
        default="triage_need_fix",
    )
    parser.add_argument("--skip_repair", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case_contains", type=str, default=None)
    parser.add_argument("--max_rounds", type=int, default=None)
    parser.add_argument("--llm_enabled", type=str, default=None)
    parser.add_argument("--vision_enabled", type=str, default=None)
    parser.add_argument("--triage_n_refs", type=int, default=None)
    parser.add_argument("--triage_four_way_refs", type=str, default=None)
    parser.add_argument("--triage_ref_categories", type=str, default=None)
    parser.add_argument("--top_improved_k", type=int, default=8)
    parser.add_argument("--top_improved_min_delta", type=float, default=0.0)
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any model run fails",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "runner_extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to runner script. Prefix with -- to separate.",
    )
    return parser.parse_args()


def _resolve_path(p: str, base: Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _safe_float(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    return None


def _safe_int(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return int(x)
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def _extract_triage_counts(triage_csv: Path) -> Tuple[int, int]:
    good = 0
    needs_fix = 0
    if not triage_csv.exists():
        return good, needs_fix
    with triage_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = (row.get("TriageCategory") or "").strip()
            if category == "good":
                good += 1
            elif category == "needs_fix":
                needs_fix += 1
    return good, needs_fix


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _write_csv(rows: Sequence[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in SUMMARY_COLUMNS})


def _write_md(rows: Sequence[Dict[str, Any]], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(SUMMARY_COLUMNS) + " |")
    lines.append("| " + " | ".join(["---"] * len(SUMMARY_COLUMNS)) + " |")
    for row in rows:
        vals = [_fmt_cell(row.get(col)) for col in SUMMARY_COLUMNS]
        lines.append("| " + " | ".join(vals) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(rows: Sequence[Dict[str, Any]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def _build_run_cmd(
    python_exe: str,
    runner_script: Path,
    config_path: Path,
    source_root: Path,
    target_dir: Path,
    artifacts_dir: Path,
    args: argparse.Namespace,
    runner_extra: Sequence[str],
) -> List[str]:
    cmd = [
        python_exe,
        str(runner_script),
        "--config",
        str(config_path),
        "--source_root",
        str(source_root),
        "--target_dir",
        str(target_dir),
        "--artifacts_dir",
        str(artifacts_dir),
        "--knowledge_mode",
        args.knowledge_mode,
        "--atlas_mode",
        args.atlas_mode,
        "--repair_subset",
        args.repair_subset,
        "--top_improved_k",
        str(args.top_improved_k),
        "--top_improved_min_delta",
        str(args.top_improved_min_delta),
    ]

    def add_opt(flag: str, value: Any) -> None:
        if value is None:
            return
        cmd.extend([flag, str(value)])

    if args.skip_repair:
        cmd.append("--skip_repair")
    add_opt("--limit", args.limit)
    add_opt("--case_contains", args.case_contains)
    add_opt("--max_rounds", args.max_rounds)
    add_opt("--llm_enabled", args.llm_enabled)
    add_opt("--vision_enabled", args.vision_enabled)
    add_opt("--triage_n_refs", args.triage_n_refs)
    add_opt("--triage_four_way_refs", args.triage_four_way_refs)
    add_opt("--triage_ref_categories", args.triage_ref_categories)
    cmd.extend(runner_extra)
    return cmd


def _run_with_tee(cmd: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        return proc.wait()


def _extract_result_row(
    model_alias: str,
    config_path: Path,
    llm_provider: str,
    llm_model: str,
    target_dir: Path,
    log_path: Path,
    exit_code: int | None,
    elapsed_sec: float | None,
) -> Dict[str, Any]:
    summary_path = target_dir / "summary" / "experiment_summary.json"
    row: Dict[str, Any] = {
        "model_alias": model_alias,
        "status": "ok" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "elapsed_sec": _safe_float(elapsed_sec),
        "config_path": str(config_path),
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "n_cases_processed": None,
        "triage_good_count": None,
        "triage_needs_fix_count": None,
        "triage_acc": None,
        "triage_f1_needs_fix": None,
        "repair_n_cases": None,
        "repair_n_approved": None,
        "repair_improved_count": None,
        "repair_worsened_count": None,
        "repair_pre_dice_mean": None,
        "repair_post_dice_mean": None,
        "repair_delta_dice_mean": None,
        "target_dir": str(target_dir),
        "run_log": str(log_path),
        "summary_json": str(summary_path),
    }
    if not summary_path.exists():
        return row

    summary = _load_json(summary_path)
    triage_outputs = summary.get("triage_outputs", {}) or {}
    triage_metrics = triage_outputs.get("metrics", {}) or {}
    repair_outputs = summary.get("repair_outputs", {}) or {}
    repair_metrics = repair_outputs.get("metrics", {}) or {}

    triage_csv = triage_outputs.get("triage_csv")
    triage_csv_path = Path(triage_csv) if triage_csv else (target_dir / "triage" / "triage_results.csv")
    triage_good, triage_needs_fix = _extract_triage_counts(triage_csv_path)

    row["n_cases_processed"] = _safe_int(summary.get("n_cases_processed"))
    row["triage_good_count"] = triage_good
    row["triage_needs_fix_count"] = triage_needs_fix
    row["triage_acc"] = _safe_float(triage_metrics.get("ACC"))
    row["triage_f1_needs_fix"] = _safe_float(triage_metrics.get("F1_needs_fix"))
    row["repair_n_cases"] = _safe_int(repair_metrics.get("n_cases"))
    row["repair_n_approved"] = _safe_int(repair_metrics.get("n_approved"))
    row["repair_improved_count"] = _safe_int(repair_metrics.get("improved_count"))
    row["repair_worsened_count"] = _safe_int(repair_metrics.get("worsened_count"))
    row["repair_pre_dice_mean"] = _safe_float(repair_metrics.get("pre_dice_mean"))
    row["repair_post_dice_mean"] = _safe_float(repair_metrics.get("post_dice_mean"))
    row["repair_delta_dice_mean"] = _safe_float(repair_metrics.get("delta_dice_mean"))
    return row


def _apply_model_overrides(model_configs: OrderedDict[str, str], overrides: Iterable[str]) -> None:
    for item in overrides:
        name, path = _parse_name_path(item)
        model_configs[name] = path


def _parse_model_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    source_root = _resolve_path(args.source_root, Path.cwd())
    output_root = _resolve_path(args.output_root, Path.cwd())
    runner_script = _resolve_path(args.runner_script, Path.cwd())
    if not runner_script.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script}")

    model_configs = OrderedDict(DEFAULT_MODEL_CONFIGS)
    _apply_model_overrides(model_configs, args.model_config)
    selected_models = _parse_model_list(args.models)
    if not selected_models:
        raise ValueError("No models selected. Use --models.")

    unknown = [m for m in selected_models if m not in model_configs]
    if unknown:
        raise ValueError(
            f"Unknown model alias(es): {unknown}. "
            f"Known aliases: {list(model_configs.keys())}"
        )

    if args.runner_extra and args.runner_extra[0] == "--":
        runner_extra = args.runner_extra[1:]
    else:
        runner_extra = list(args.runner_extra)

    if args.shared_artifacts_dir:
        shared_artifacts_dir = _resolve_path(args.shared_artifacts_dir, Path.cwd())
    else:
        shared_artifacts_dir = output_root / "artifacts_shared"

    output_root.mkdir(parents=True, exist_ok=True)
    shared_artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_plan_path = output_root / "run_plan.json"
    plan_payload = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "runner_script": str(runner_script),
        "shared_artifacts_dir": str(shared_artifacts_dir),
        "selected_models": selected_models,
        "model_configs": {k: model_configs[k] for k in selected_models},
        "runner_extra": runner_extra,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with run_plan_path.open("w", encoding="utf-8") as f:
        json.dump(plan_payload, f, ensure_ascii=False, indent=2)

    all_rows: List[Dict[str, Any]] = []

    print("==== Workflow Model Compare ====")
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Shared artifacts: {shared_artifacts_dir}")
    print(f"Selected models: {', '.join(selected_models)}")
    print("")

    for idx, model_alias in enumerate(selected_models, start=1):
        cfg_path = _resolve_path(model_configs[model_alias], Path.cwd())
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found for {model_alias}: {cfg_path}")

        cfg = _load_yaml(cfg_path)
        llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
        llm_provider = str(llm_cfg.get("provider", ""))
        llm_model = str(llm_cfg.get("model", ""))

        target_dir = output_root / model_alias
        log_path = target_dir / "run.log"
        cmd = _build_run_cmd(
            python_exe=sys.executable,
            runner_script=runner_script,
            config_path=cfg_path,
            source_root=source_root,
            target_dir=target_dir,
            artifacts_dir=shared_artifacts_dir,
            args=args,
            runner_extra=runner_extra,
        )

        print(f"[{idx}/{len(selected_models)}] {model_alias}")
        print(f"  config: {cfg_path}")
        print(f"  llm: provider={llm_provider} model={llm_model}")
        print(f"  target: {target_dir}")
        print(f"  cmd: {' '.join(cmd)}")

        if args.dry_run:
            row = _extract_result_row(
                model_alias=model_alias,
                config_path=cfg_path,
                llm_provider=llm_provider,
                llm_model=llm_model,
                target_dir=target_dir,
                log_path=log_path,
                exit_code=None,
                elapsed_sec=None,
            )
            row["status"] = "dry_run"
            all_rows.append(row)
            print("  dry_run=true, skipped\n")
            continue

        t0 = time.time()
        exit_code = _run_with_tee(cmd, cwd=repo_root, log_path=log_path)
        elapsed = time.time() - t0

        row = _extract_result_row(
            model_alias=model_alias,
            config_path=cfg_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
            target_dir=target_dir,
            log_path=log_path,
            exit_code=exit_code,
            elapsed_sec=elapsed,
        )
        all_rows.append(row)

        print(f"  exit_code={exit_code}, elapsed_sec={elapsed:.2f}")
        if exit_code != 0 and args.stop_on_error:
            print("  stop_on_error=true, aborting remaining models.")
            break
        print("")

    summary_csv = output_root / "model_comparison_summary.csv"
    summary_md = output_root / "model_comparison_summary.md"
    summary_json = output_root / "model_comparison_summary.json"
    _write_csv(all_rows, summary_csv)
    _write_md(all_rows, summary_md)
    _write_json(all_rows, summary_json)

    print("==== Completed ====")
    print(f"Plan: {run_plan_path}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary MD: {summary_md}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
