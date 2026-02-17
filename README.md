# Cardiac Agent Post-Processing

Multi-agent LLM system for cardiac MRI segmentation label repair.

## Overview

A team of 6 LLM-powered agents work together to automatically assess, diagnose, plan,
repair, and verify cardiac segmentation masks — without requiring ground truth labels.

```
Triage → Diagnosis → Planning → Execution → Verification
  ↑                                              │
  └──────── retry with new strategy ─────────────┘
```

### Agents

| Agent | Role | LLM/VLM |
|-------|------|---------|
| **Coordinator** | Orchestrates pipeline, manages retries | LLM |
| **Triage** | Fast quality classification (good/needs_fix) | LLM + classifier |
| **Diagnosis** | Spatial defect identification | LLM + VLM + rules |
| **Planner** | Operation sequencing with conflict avoidance | LLM |
| **Executor** | Applies morphological ops, monitors RQS | LLM |
| **Verifier** | Quality gating, approve/reject fixes | LLM + VLM + metrics |

### Key Design Principles

1. **No-GT Optimization**: Uses Reward Quality Score (RQS) — a proxy metric combining topology, shape, and consistency terms — to optimize without ground truth
2. **Agent Communication**: Structured message bus with full audit trail
3. **Conservative Gating**: Verifier must approve fixes before saving; rejects preserve original
4. **Dynamic Retry**: Rejected cases get re-diagnosed and re-planned (up to N rounds)

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Configure LLM

Copy `.env.example` to `.env` and set your API credentials:

```bash
LLM_ENABLED=true
OPENAI_BASE_URL=https://your-api-endpoint/v1
OPENAI_API_KEY=your-key
OPENAI_MODEL=your-model
```

### 3. Run

```bash
python -m cardiac_agent_postproc.run --config config/default.yaml
```

### 4. Output

- Optimized masks: `<target_dir>/runs/`
- Audit trail: `<target_dir>/runs/agent_audit_trail.json`

## Configuration

See `config/default.yaml` for all settings. Key multi-agent parameters:

```yaml
multi_agent:
  max_rounds_per_case: 3       # max retry cycles per case
  fast_path_threshold: 0.95    # classifier score above this → skip optimization
  agent_temperature: 0.15      # LLM creativity level
  enable_negotiation: true     # agents can challenge each other
  batch_summary: true          # cross-case analysis after batch
```

## Folder Structure

```
source_data/
├── case001_img.png          # raw MRI image
├── case001_pred.png         # predicted segmentation mask
├── atlas.pkl                # probabilistic shape atlas
└── runs/                    # outputs go here

target_data/
├── case001_img.png
├── case001_pred.png
├── case001_gt.png           # ground truth (optional, for evaluation)
└── runs/
```
