#!/bin/bash
#SBATCH --job-name=ysy_corot-constit-stats
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "Refresh paired-surface normalization statistics"
echo "1024 geometric centers -> 2048 material states"
echo "============================================================"

"${PYTHON}" compute_stats.py \
  --cache-dir "${CACHE_DIR}" \
  --transitions-per-sample 12 \
  --centers-per-transition 1024

echo "============================================================"
echo "Run preflight"
echo "============================================================"

"${PYTHON}" preflight.py \
  --cache-dir "${CACHE_DIR}" \
  --device cuda

echo "============================================================"
echo "STATS + PREFLIGHT COMPLETE"
echo "============================================================"
