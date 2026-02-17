# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent pipeline for anatomy-aware post-processing of cardiac MRI segmentation masks. Uses a **No-GT (No Ground Truth) inner-loop** optimization approach: a rule-based Reward Quality Score (RQS) drives optimization using only raw image edges and a source-learned shape atlas. Target ground truth is used **only** for evaluation (Dice + HD95), never for optimization decisions.

**Class map**: 0=Background, 1=RV blood pool, 2=LV myocardium, 3=LV blood pool.

## Commands

```bash
# Run full pipeline
python -m cardiac_agent_postproc.run --config config/default.yaml

# Quick smoke test (small dataset, no revision loop)
python -m cardiac_agent_postproc.run --config config/sample.yaml

# Run all tests
pytest

# Run a single test file or class
pytest test_io.py
pytest test_io.py::TestRoundTrip -v

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Pipeline Flow (Orchestrator.run_with_revisions)

```
run.py → Orchestrator(cfg)
  └→ run_with_revisions()
      ├→ run_once(cfg, "v0")                    # Baseline
      │   ├→ AtlasBuilderAgent.run()            # Build shape atlas from SOURCE
      │   ├→ OptimizerAgent.optimize_folder()   # Optimize SOURCE (RQS-only)
      │   ├→ OptimizerAgent.optimize_folder()   # Optimize TARGET (RQS-only, NO GT)
      │   └→ EvaluatorAgent.evaluate_target()   # Compute Dice/HD95 using GT
      │
      └→ Revision loop (v1..vN):
          ├→ RevisionControllerAgent.propose_change()  # Analyze worst cases → propose 1 param change
          ├→ apply_change(cfg, change)                  # Update config in-memory
          └→ run_once(cfg_updated, f"v{rev}")           # Re-run full pipeline
```

### Key Agents

- **AtlasBuilderAgent** (`agents/atlas_builder.py`): Builds `shape_atlas.pkl` from SOURCE predictions filtered by RQS threshold. Uses KMeans clustering per view type.
- **OptimizerAgent** (`agents/optimizer.py`, ~560 lines, most complex): Outer loop selects worst-K files by RQS; inner loop generates mutation candidates (`ops.py`) and greedily selects best RQS improvement. Includes optional vision guardrail pre-screening (Phase 5) and quality model blending.
- **EvaluatorAgent** (`agents/evaluator.py`): GT-only evaluation computing per-file and aggregate Dice/HD95 metrics.
- **RevisionControllerAgent** (`agents/revision_controller.py`): Loads evaluation CSVs, identifies worst cases (bottom-RQS ∪ bottom-Dice ∪ top-HD95), extracts dominant penalties, and calls LLM (or heuristic fallback) to propose one parameter change.
- **DiagnosisAgent** (`agents/diagnosis_agent.py`): Detects issues (holes, touching, components) and reorders mutation candidates accordingly.

### Supporting Modules

- **rqs.py**: Computes RQS from ~15 penalty/reward terms (P_touch, P_components, P_atlas_mismatch, R_enclosure, etc.)
- **ops.py**: Generates mutation candidates (topology_cleanup, fill_holes, smooth_morph, atlas_growprune, rv_lv_barrier, etc.) with `is_valid_candidate()` guard
- **atlas.py**: Shape atlas building (KMeans), alignment (affine), and matching
- **api_client.py**: `OpenAICompatClient` wrapping any OpenAI-compatible LLM (NVIDIA, OpenAI, local). Methods: `chat_json()`, `chat_vision_json()`
- **io_utils.py**: Mask I/O with encoding (0..3 ↔ 0,85,170,255), frame listing
- **view_utils.py**: Infers 2ch vs 4ch view type from filename or RV ratio
- **vlm_guardrail.py**: Vision-Language Model pre-screening (Phase 5, optional)
- **quality_model.py**: Offline-trained quality scorer with 30 geometric/intensity features. `QualityScorer` supports both regressor (`score()`) and classifier (`classify()`) modes

### Configuration

- **YAML configs** in `config/`: `default.yaml` (production), `sample.yaml` (quick test)
- **Environment** via `.env`: `LLM_ENABLED`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`
- Settings loaded through `pydantic-settings` in `settings.py`
- Config keys use lowercase with underscores, accessed via dot-notation paths (e.g., `reward_weights.P_touch`)

### Critical Invariant

The OptimizerAgent **must never read TARGET ground truth** during optimization. The `is_target` flag controls this. RQS scoring uses only: predicted masks, raw image edges, and the source-learned shape atlas. GT is loaded only in EvaluatorAgent for Dice/HD95 computation.

## Code Style

- Black-style formatting (4-space indent, trailing commas)
- `snake_case` for functions/variables, `PascalCase` for classes
- Type hints on public APIs
- Stateless helper functions; pass config dicts rather than module-level globals
- Conventional commits: `feat:`, `fix:`, `docs:`, etc. (subject ≤72 chars)

## Data Format

Each frame directory contains `*_img.png` (raw image), `*_pred.png` (predicted mask, values {0,1,2,3}), and `*_gt.png` (ground truth, TARGET evaluation only).

## Key Outputs

- `label_optimization_log.csv`: Per-iteration RQS trace with all penalty terms
- `label_optimization_summary.csv`: Per-file final metrics
- `target_dice_report.csv`: Per-file Dice/HD95 evaluation
- `target_dice_summary.csv`: Aggregate statistics
- `*_pred_optimized.png`: Final optimized masks
- `shape_atlas.pkl`: Cached shape atlas
- `quality_model.pkl`: Trained quality model (regressor + classifier + threshold)

## Current Progress / Known Issues

### Triage Classifier Retraining (completed 2026-02-10)

**Problem**: Old triage flagged 318/320 samples as `needs_fix` because `thickness_cv > 0.5` triggered for 99% of samples, VLM returned 0 for all, and quality model R²=0.22. This caused the optimizer to degrade 94 good samples.

**Changes made**:

| File | Change |
|------|--------|
| `train_quality_model.py` | Rewritten: reads `final_metrics.csv`, trains 3 classifier configs (RF, GBC) + regressor with `RepeatedStratifiedKFold(5,3)`, auto-selects best by F1, calibrates probability threshold |
| `quality_model.py` | `QualityScorer` extended: `classifier`, `threshold`, `use_classifier` fields; `classify()` method; `has_classifier` property; backward-compatible `load()` |
| `agents/triage_agent.py` | Fixed `_detect_issues()`: `thickness_cv > 1.0` (was 0.5), `boundary_over_area > 0.22` (was 0.18). Added `mode: "classifier"` path in `triage_folder()`. VLM default off. |
| `config/default.yaml` | `triage.mode: "classifier"`, `triage.vlm_enabled: false`, `quality_model.blend_mode: "replace"`, `quality_model.bad_threshold: 0.93` |

**Training results** (`quality_model.pkl`, RF_balanced classifier):
- CV needs_fix: **50 / 320** (down from 318)
- CV recall(bad, Dice<0.93): 0.64 (catches 9/14; 5 missed are borderline Dice 0.92–0.93)
- CV precision(bad): 0.18
- Regressor CV R² ≈ −0.03 (features lack discriminative power for exact Dice prediction)

**Limitation**: The 30 geometric features cannot cleanly separate bad (Dice<0.93) from good samples — regressor CV R² is near zero. Achieving recall≥0.93 AND needs_fix≤50 simultaneously is not possible with current features. The chosen operating point prioritizes **not degrading good samples** over catching every bad sample.

**Retrain command**:
```bash
python train_quality_model.py --data-dir results/Input_MnM2/all_frames_export --metrics-csv final_metrics.csv --output quality_model.pkl --min-recall 0.64
```

### TODO / Next Steps
- Run full pipeline with new triage (`python -m cardiac_agent_postproc.run --config config/default.yaml`) and verify good samples are no longer degraded
- Consider adding new features (e.g. atlas distance, RQS components) to improve classifier discrimination
- Consider lowering `--bad-threshold` to 0.90 to focus on truly broken samples (currently 0.93)
