# Cardiac Agent Post-Processing

Multi-agent LLM system for cardiac MRI segmentation repair.

## Overview

This project repairs cardiac segmentation masks through a coordinated agent pipeline:

```
Triage -> Diagnosis -> Planning -> Execution -> Verification
  ^                                              |
  +---------------- retry loop ------------------+
```

The design focuses on conservative edits, structured decision logs, and robust fallbacks.

## Core Agents

| Agent | Responsibility |
|---|---|
| Coordinator | Orchestrates per-case rounds and retry policy |
| Triage | Classifies `good` vs `needs_fix` and prioritizes cases |
| Diagnosis | Produces spatial defect hypotheses with issue-level metadata |
| Planner | Builds operation sequence from diagnoses |
| Executor | Applies morphology/atlas/neighbor operations with safety gates |
| Verifier | Approves, rejects, or requests another round |

## Repository Layout

```text
cardiac_agent_postproc/
├── cardiac_agent_postproc/      # package code (agents, ops, API client, utils)
├── scripts/                     # experiment runners and analysis tools
├── config/                      # runtime configs (OpenAI/Azure/Gemini/Ollama variants)
├── prompts/                     # system and task prompts for each agent
├── results/                     # run outputs
└── WORKLOG.md                   # development session log
```

## Installation

```bash
pip install -e .
```

Optional:

```bash
pip install -r requirements.txt
```

## API Configuration

Model/provider settings are read from YAML under `config/`.

For OpenAI-compatible endpoints, set API key in environment before running:

```bash
export AZURE_OPENAI_API_KEY="your_api_key"
```

Notes:
- Most project runners read `AZURE_OPENAI_API_KEY` by default.
- Some helper scripts also accept `OPENAI_API_KEY`.

## Quick Start

Run the package entry pipeline:

```bash
python -m cardiac_agent_postproc.run --config config/default.yaml
```

## Main Experiment Runner (MICCAI Style)

Expected source root:

```text
<source_root>/
├── all_frames_export/   # *_img.png, *_pred.png, *_gt.png
├── best_cases/          # curated good references
└── worst_cases/         # curated difficult cases
```

Run:

```bash
python scripts/run_miccai_multiagent_experiment.py \
  --config config/azure_openai_medrag.yaml \
  --source_root results/Input_MnM2 \
  --target_dir results/exp_mnm2_run \
  --repair_subset triage_need_fix \
  --atlas_mode local_per_sample \
  --knowledge_mode auto_skip
```

### Single-case preflight

```bash
python scripts/run_miccai_multiagent_experiment.py \
  --config config/azure_openai_medrag.yaml \
  --source_root results/Input_MnM2 \
  --target_dir results/exp_case335_preflight \
  --limit 1 \
  --case_contains 335_original_lax_4c_009 \
  --repair_subset worst_cases \
  --atlas_mode local_per_sample \
  --knowledge_mode auto_skip
```

### Background full run

```bash
mkdir -p results/exp_mnm2_allcases_bg
setsid python scripts/run_miccai_multiagent_experiment.py \
  --config config/azure_openai_medrag.yaml \
  --source_root results/Input_MnM2 \
  --target_dir results/exp_mnm2_allcases_bg \
  --repair_subset triage_need_fix \
  --atlas_mode local_per_sample \
  --knowledge_mode auto_skip \
  > results/exp_mnm2_allcases_bg/run.log 2>&1 < /dev/null &
```

## Outputs

Each run writes:

- `triage/`: triage CSVs and curated-label metrics
- `repair/`: diagnosis, operation traces, repaired masks, Dice deltas
- `figures/`: summary plots and top-improved visualizations
- `summary/`: run-level summary JSON
- `artifacts/`: atlas, quality model, repair KB, runtime config snapshots

## Model Comparison

Run multiple model configs in one command:

```bash
python scripts/run_workflow_model_compare.py \
  --source_root results/Input_MnM2 \
  --output_root results/exp_model_compare_mnm2 \
  --knowledge_mode auto_skip \
  --atlas_mode reuse_global
```

Comparison summary files:

- `results/exp_model_compare_mnm2/model_comparison_summary.csv`
- `results/exp_model_compare_mnm2/model_comparison_summary.md`
- `results/exp_model_compare_mnm2/model_comparison_summary.json`

## Useful Config Files

- `config/azure_openai_medrag.yaml`
- `config/openai.yaml`
- `config/openai_official_gpt52_medrag.yaml`
- `config/gemini.yaml`
- `config/ollama_ministral_triage.yaml`

## Development Notes

- Python: `>=3.10`
- Formatting: `black`, `isort`
- Main logic entry points:
  - `cardiac_agent_postproc/orchestrator.py`
  - `scripts/run_miccai_multiagent_experiment.py`
