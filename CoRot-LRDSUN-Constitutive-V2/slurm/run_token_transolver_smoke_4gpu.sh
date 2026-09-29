#!/usr/bin/env bash
# V2 real-data DDP smoke: one forward/loss/backward pass per rank.
# It intentionally creates no optimizer and saves no checkpoint.
#SBATCH --job-name=corot-v2-token-smoke
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "${V2_DIR}/slurm/common.sh"
cd "${V2_DIR}"

OUT_DIR="${OUT_DIR:-${V2_DIR}/outputs/token_smoke_${SLURM_JOB_ID:-manual}}"
mkdir -p "${OUT_DIR}"

WARM_START="${WARM_START:-${V1_DIR}/outputs/finetune_5step/run_20260920_143444_1610485/best_val_5step.pt}"
[[ -d "${CACHE_DIR}" && -f "${CACHE_DIR}/stats.json" ]] || { echo "Missing CACHE_DIR=${CACHE_DIR}"; exit 2; }
[[ -f "${WARM_START}" ]] || { echo "Missing WARM_START=${WARM_START}"; exit 2; }

torchrun --standalone --nproc_per_node=4 \
  "${V2_DIR}/test_ddp_token_pipeline.py" \
  --cache-dir "${CACHE_DIR}" \
  --checkpoint "${WARM_START}" \
  --sample-id 119 \
  --time 0 \
  --centers-per-rank "${CENTERS_PER_RANK:-1024}" \
  --hidden-dim "${HIDDEN_DIM:-192}" \
  --token-dim "${TOKEN_DIM:-192}" \
  | tee "${OUT_DIR}/ddp_smoke.json"
