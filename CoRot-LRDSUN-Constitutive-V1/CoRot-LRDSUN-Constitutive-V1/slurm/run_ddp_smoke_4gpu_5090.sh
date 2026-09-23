#!/bin/bash
#SBATCH --job-name=ysy_corot-constit-smoke
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
OUT="${PROJECT_DIR}/outputs/smoke_${SLURM_JOB_ID}"; mkdir -p "${OUT}"
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${PROJECT_DIR}/train_ddp.py" --cache-dir "${CACHE_DIR}" --output-dir "${OUT}" \
  --epochs 2 --batch-size 2048 --eval-batch-size 2048 --hidden-dim 192 --smoke
