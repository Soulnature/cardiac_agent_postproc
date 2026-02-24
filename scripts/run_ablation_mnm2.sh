#!/usr/bin/env bash
# =============================================================================
# Ablation experiments on MnM2 dataset (ollama ministral-3:14b)
#
# Variants:
#   full          - baseline (all components, max_rounds=2)
#   no_vlm        - vision_enabled=false (rules/RQS only)
#   no_repair_kb  - repair KB disabled
#   no_local_atlas- global atlas (disable local per-sample atlas)
#   rounds_1      - max_rounds=1
#   rounds_2      - max_rounds=2  (same as full, for reference)
#   rounds_3      - max_rounds=3
#
# Usage:
#   bash scripts/run_ablation_mnm2.sh [variant...]
#   bash scripts/run_ablation_mnm2.sh              # run all
#   bash scripts/run_ablation_mnm2.sh full no_vlm  # run specific variants
# =============================================================================
set -euo pipefail

SCRIPT="scripts/run_miccai_multiagent_experiment.py"
CONFIG="config/ollama_ministral_triage.yaml"
SOURCE="results/Input_MnM2"
# Reuse quality model and repair KB pre-built from the GPT-4o full run
# to avoid redundant training; override per-variant where needed.
SHARED_ARTIFACTS="results/exp_mnm2_gpt4o_allcases_20260220_bg_after_preflight/artifacts"
DATE="20260222"
LOG_DIR="results/ablation_logs"

mkdir -p "$LOG_DIR"

# ---------- common base flags ------------------------------------------------
BASE_FLAGS=(
    --config         "$CONFIG"
    --source_root    "$SOURCE"
    --repair_subset  triage_need_fix
    --atlas_mode     local_per_sample
    --knowledge_mode auto_skip          # reuse existing artifacts by default
    --artifacts_dir  "$SHARED_ARTIFACTS"
)

# ---------- variant definitions ----------------------------------------------
declare -A VARIANT_EXTRA
declare -A VARIANT_ARTIFACTS   # empty = use SHARED_ARTIFACTS

VARIANT_EXTRA[full]=""
VARIANT_ARTIFACTS[full]=""

VARIANT_EXTRA[no_vlm]="--vision_enabled false --llm_enabled false"
VARIANT_ARTIFACTS[no_vlm]=""

VARIANT_EXTRA[no_repair_kb]="--repair_kb_mode disable"
VARIANT_ARTIFACTS[no_repair_kb]=""

VARIANT_EXTRA[no_local_atlas]="--atlas_mode disable_global"
VARIANT_ARTIFACTS[no_local_atlas]=""

VARIANT_EXTRA[rounds_1]="--max_rounds 1"
VARIANT_ARTIFACTS[rounds_1]=""

VARIANT_EXTRA[rounds_2]="--max_rounds 2"
VARIANT_ARTIFACTS[rounds_2]=""

VARIANT_EXTRA[rounds_3]="--max_rounds 3"
VARIANT_ARTIFACTS[rounds_3]=""

# ---------- which variants to run -------------------------------------------
ALL_VARIANTS=(full no_vlm no_repair_kb no_local_atlas rounds_1 rounds_2 rounds_3)

if [[ $# -gt 0 ]]; then
    VARIANTS=("$@")
else
    VARIANTS=("${ALL_VARIANTS[@]}")
fi

# ---------- run each variant sequentially ------------------------------------
echo "===== MnM2 Ablation Run ($(date)) ====="
echo "Variants: ${VARIANTS[*]}"
echo ""

for VARIANT in "${VARIANTS[@]}"; do
    if [[ -z "${VARIANT_EXTRA[$VARIANT]+x}" ]]; then
        echo "ERROR: unknown variant '$VARIANT'. Available: ${ALL_VARIANTS[*]}"
        exit 1
    fi

    TARGET="results/exp_mnm2_ablation_${VARIANT}_${DATE}"
    LOG="$LOG_DIR/ablation_${VARIANT}_${DATE}.log"

    echo "----------------------------------------------------------------------"
    echo "Variant : $VARIANT"
    echo "Output  : $TARGET"
    echo "Log     : $LOG"
    echo "Start   : $(date)"

    # Build artifact flag: variant-specific dir takes precedence over shared
    ARTIFACT_FLAG=()
    if [[ -n "${VARIANT_ARTIFACTS[$VARIANT]}" ]]; then
        ARTIFACT_FLAG=(--artifacts_dir "${VARIANT_ARTIFACTS[$VARIANT]}")
    else
        ARTIFACT_FLAG=(--artifacts_dir "$SHARED_ARTIFACTS")
    fi

    # Parse extra flags into array (handles empty string safely)
    EXTRA=()
    if [[ -n "${VARIANT_EXTRA[$VARIANT]}" ]]; then
        read -ra EXTRA <<< "${VARIANT_EXTRA[$VARIANT]}"
    fi

    CMD=(
        python -u "$SCRIPT"
        "${BASE_FLAGS[@]}"
        "${ARTIFACT_FLAG[@]}"
        --target_dir "$TARGET"
        "${EXTRA[@]}"
    )

    echo "CMD: ${CMD[*]}"
    echo ""

    # Run and tee: live output goes to log, summary tail to stdout
    "${CMD[@]}" 2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo "✓ $VARIANT done ($(date))"
    else
        echo ""
        echo "✗ $VARIANT FAILED (exit=$EXIT_CODE) — continuing to next variant"
    fi
    echo ""
done

echo "===== All ablation variants finished ($(date)) ====="

# ---------- quick summary table ----------------------------------------------
echo ""
echo "===== Summary (Repair mean Dice delta per variant) ====="
python3 - << 'PYEOF'
import json, os, glob

date = "20260222"
variants = ["full", "no_vlm", "no_repair_kb", "no_local_atlas",
            "rounds_1", "rounds_2", "rounds_3"]

print(f"{'Variant':<20} {'N_repair':>8} {'DeltaDice_mean':>15} {'Improved':>9} {'Degraded':>9}")
print("-" * 65)
for v in variants:
    summary = f"results/exp_mnm2_ablation_{v}_{date}/summary/experiment_summary.json"
    if not os.path.exists(summary):
        print(f"{v:<20} {'(no data)':>8}")
        continue
    with open(summary) as f:
        d = json.load(f)
    m = d.get("repair_outputs", {}).get("metrics", {})
    n  = m.get("n_cases", "-")
    dd = m.get("delta_dice_mean", float("nan"))
    imp = m.get("n_improved", "-")
    deg = m.get("n_degraded", "-")
    print(f"{v:<20} {str(n):>8} {dd:>+15.4f} {str(imp):>9} {str(deg):>9}")
PYEOF
