#!/usr/bin/env bash
# Formal V2 training wrapper.  Do not submit until token pipeline validation is accepted.
#SBATCH --job-name=ysy_v2_trans_c1024
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "${V2_DIR}/slurm/common.sh"
cd "${V2_DIR}"

# Match the V1 4-rank 32-CPU contract: six host threads per rank.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

WARM_START="${WARM_START:-${V1_DIR}/outputs/finetune_5step/run_20260920_143444_1610485/best_val_5step.pt}"
[[ -f "${WARM_START}" ]] || { echo "Missing WARM_START=${WARM_START}"; exit 2; }
[[ -d "${CACHE_DIR}" ]] || { echo "Missing CACHE_DIR=${CACHE_DIR}"; exit 2; }

OUT_DIR="${OUT_DIR:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step}"
RESUME="${RESUME:-}"
if [[ -e "${OUT_DIR}/last.pt" || -e "${OUT_DIR}/best_val.pt" ]] && [[ -z "${RESUME}" ]]; then
  echo "Refusing to overwrite existing formal output: ${OUT_DIR}" >&2
  exit 3
fi
mkdir -p "${OUT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME}" ]]; then
  [[ -f "${RESUME}" ]] || { echo "Missing RESUME=${RESUME}" >&2; exit 4; }
  RESUME_ARGS+=(--resume "${RESUME}")
fi

torchrun --standalone --nproc_per_node=4 \
  "${V2_DIR}/train_token_transolver.py" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUT_DIR}" \
  --warm-start "${WARM_START}" \
  --epochs "${EPOCHS:-30}" \
  --windows-per-sample 15 \
  --centers-per-window "${CENTERS_PER_WINDOW:-1024}" \
  --val-windows-per-sample 18 \
  --max-val-samples 0 \
  --hidden-dim "${HIDDEN_DIM:-192}" \
  --token-dim "${TOKEN_DIM:-192}" \
  --attention-depth "${ATTENTION_DEPTH:-2}" \
  --attention-heads "${ATTENTION_HEADS:-4}" \
  --lr 1e-5 \
  --min-lr 1e-6 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --w-delta8 1.0 \
  --w-peeq 1.0 \
  --w-gate 0.20 \
  --w-le 0.25 \
  --w-mises 0.25 \
  --early-stop-start 18 \
  --early-stop-patience 8 \
  "${RESUME_ARGS[@]}"
