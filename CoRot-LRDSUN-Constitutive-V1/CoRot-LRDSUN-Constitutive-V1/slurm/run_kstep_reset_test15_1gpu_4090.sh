#!/bin/bash
#SBATCH --job-name=ysy_corot-kreset
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

cd "${PROJECT_DIR}"

RUN_DIR="${PROJECT_DIR}/outputs/formal/run_20260904_001957_1546153"

OUT_DIR="${RUN_DIR}/kstep_reset_test15"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "============================================================"
echo "K-step reset diagnostic"
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

nvidia-smi

"${PYTHON}" -u evaluate_kstep_reset.py \
    --cache-dir "${CACHE_DIR}" \
    --checkpoint "${RUN_DIR}/best_val.pt" \
    --output-dir "${OUT_DIR}" \
    --split test \
    --ks 1,2,5,10,20,30,60,180 \
    --batch-size 8192 \
    --device cuda

echo "============================================================"
echo "RESULT"
echo "============================================================"

cat "${OUT_DIR}/kstep_global_metrics.csv"

echo "============================================================"
echo "DONE"
echo "============================================================"
