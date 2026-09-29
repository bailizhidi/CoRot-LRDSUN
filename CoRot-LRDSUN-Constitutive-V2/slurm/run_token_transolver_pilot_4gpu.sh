#!/usr/bin/env bash
# Two-epoch V2 pilot training.  This script is generated after the real DDP
# smoke passes; it is intentionally not submitted automatically.
#
# Example:
#   sbatch --gpus=4 -p gpu_5090 slurm/run_token_transolver_pilot_4gpu.sh

#SBATCH --job-name=corot-v2-token-pilot
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "${V2_DIR}/slurm/common.sh"
cd "${V2_DIR}"

CACHE_DIR="${CACHE_DIR:-${V1_DIR}/prepared_cache_prod_v1}"
WARM_START="${WARM_START:-${V1_DIR}/outputs/finetune_5step/run_20260920_143444_1610485/best_val_5step.pt}"
OUT_DIR="${OUT_DIR:-${V2_DIR}/outputs/token_transolver_pilot_${SLURM_JOB_ID:-manual}}"

[[ -d "${CACHE_DIR}" ]] || { echo "Missing CACHE_DIR=${CACHE_DIR}"; exit 2; }
[[ -f "${CACHE_DIR}/stats.json" ]] || { echo "Missing stats.json under ${CACHE_DIR}"; exit 2; }
[[ -f "${WARM_START}" ]] || { echo "Missing WARM_START=${WARM_START}"; exit 2; }
mkdir -p "${OUT_DIR}"

echo "V2_DIR=${V2_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "WARM_START=${WARM_START}"
echo "OUT_DIR=${OUT_DIR}"
echo "EPOCHS=${EPOCHS:-2} WINDOWS_PER_SAMPLE=${WINDOWS_PER_SAMPLE:-1}"
echo "CENTERS_PER_WINDOW=${CENTERS_PER_WINDOW:-1024}"
echo "VAL_WINDOWS_PER_SAMPLE=${VAL_WINDOWS_PER_SAMPLE:-1} MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-0}"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${V2_DIR}/train_token_transolver.py" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUT_DIR}" \
  --warm-start "${WARM_START}" \
  --pilot \
  --epochs "${EPOCHS:-2}" \
  --windows-per-sample "${WINDOWS_PER_SAMPLE:-1}" \
  --centers-per-window "${CENTERS_PER_WINDOW:-1024}" \
  --val-windows-per-sample "${VAL_WINDOWS_PER_SAMPLE:-1}" \
  --max-val-samples "${MAX_VAL_SAMPLES:-0}" \
  --hidden-dim "${HIDDEN_DIM:-192}" \
  --token-dim "${TOKEN_DIM:-192}" \
  --attention-depth "${ATTENTION_DEPTH:-2}" \
  --attention-heads "${ATTENTION_HEADS:-4}"
