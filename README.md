# Cardiac MRI Segmentation Post-processing Agent System (No-GT Optimization + GT Evaluation)

This project implements a multi-agent pipeline for anatomy-aware post-processing of cardiac MRI segmentation masks.

**Key design**:
- Inner-loop optimization is **NO-GT**: only rule-based reward `RQS` + raw image edges + (SOURCE-learned) shape atlas.
- TARGET ground truth is used **ONLY** for evaluation (Dice + HD95), never for optimization decisions.
- An outer **revision loop** optionally uses TARGET evaluation to propose ONE change per revision and rerun.

## Folder structure expected (both SOURCE_DIR and TARGET_DIR)
For each frame:
- `*_img.png`  : raw image (grayscale or RGB)
- `*_pred.png` : predicted mask (values in {0,1,2,3})
- `*_gt.png`   : ground truth mask (values in {0,1,2,3})  (TARGET evaluation only; SOURCE atlas only)

Class map:
- 0 background
- 1 RV blood pool
- 2 LV myocardium (LM)
- 3 LV blood pool (LV)

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure
Copy `.env.example` to `.env` and adjust paths + LLM endpoint if you want the RevisionController to use an LLM.
Edit `config/default.yaml`:
- `paths.source_dir`
- `paths.target_dir`

### 3) Run end-to-end
```bash
python -m cardiac_agent_postproc.run --config config/default.yaml
```

Outputs:
- `SOURCE_DIR/shape_atlas.pkl` (cached)
- `SOURCE_DIR/label_optimization_summary.csv`
- `TARGET_DIR/label_optimization_summary.csv`
- `TARGET_DIR/target_dice_report.csv`
- `TARGET_DIR/target_dice_summary.csv`
- Per-mask intermediates and optimized predictions in the same folder (never overwrites original *_pred.png)

## Agents
- `AtlasBuilderAgent`: builds multi-template probabilistic atlas from SOURCE GT.
- `OptimizerAgent`: optimizes masks using RQS only; writes intermediates + logs.
- `EvaluatorAgent`: computes Dice/HD95 on TARGET using GT (reporting only).
- `RevisionControllerAgent`: reads reports + logs and proposes ONE parameter change per revision.

## Notes
- The implementation follows your spec closely; parameters are externalized in YAML for safe revisioning.
- To enforce No-GT, OptimizerAgent never loads `*_gt.png` from TARGET and will raise if such paths are provided.
