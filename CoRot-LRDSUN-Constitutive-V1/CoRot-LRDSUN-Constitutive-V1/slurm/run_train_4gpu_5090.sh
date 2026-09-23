#!/bin/bash
#SBATCH --job-name=ysy_corot-constit-v1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
OUT_ROOT="${PROJECT_DIR}/outputs/formal"; mkdir -p "${OUT_ROOT}"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"; mkdir -p "${RUN_DIR}"
RESUME_ARGS=(); [[ -n "${RESUME_CKPT:-}" ]] && RESUME_ARGS=(--resume "${RESUME_CKPT}")
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${PROJECT_DIR}/train_ddp.py" \
  --cache-dir "${CACHE_DIR}" --output-dir "${RUN_DIR}" \
  --epochs 120 --batch-size 2048 --eval-batch-size 2048 \
  --transitions-per-sample 15 --transition-sampling cyclic --centers-per-transition 1024 \
  --val-transitions-per-sample 18 --val-centers-per-transition 1024 \
  --hidden-dim 192 --lr 2e-4 --min-lr 2e-6 --weight-decay 0.01 \
  --grad-clip 1.0 --early-stop-start 30 --early-stop-patience 20 \
  --w-delta8 1.0 --w-peeq 1.0 --w-gate 0.20 --w-le 0.25 --w-mises 0.25 \
  "${RESUME_ARGS[@]}"
printf '%s\n' "${RUN_DIR}" > "${PROJECT_DIR}/formal_latest.txt"
echo "RUN_DIR=${RUN_DIR}"
