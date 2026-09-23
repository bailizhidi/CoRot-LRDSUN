#!/bin/bash
#SBATCH --job-name=ysy_corot-vtu119-4090
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

cd "${PROJECT_DIR}"

RUN_DIR="${PROJECT_DIR}/outputs/formal/run_20260904_001957_1546153"
OUT_DIR="${RUN_DIR}/vtu_rollout_gpu_sample119"

# 4090 accompanies 6 CPU cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

echo "============================================================"
echo "GPU VTU EXPORT - sample119"
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "CPUS=${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

nvidia-smi

echo "============================================================"
echo "Start autoregressive rollout + VTU export"
echo "============================================================"

"${PYTHON}" -u export_rollout_vtu.py \
    --cache-dir "${CACHE_DIR}" \
    --checkpoint "${RUN_DIR}/best_val.pt" \
    --output-dir "${OUT_DIR}" \
    --split test \
    --device cuda \
    --batch-size 8192 \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --frame-stride 1 \
    --fields all \
    --overwrite

echo "============================================================"
echo "Create archive"
echo "============================================================"

rm -f "${RUN_DIR}/vtu_rollout_gpu_sample119.tar.gz"

tar -czf "${RUN_DIR}/vtu_rollout_gpu_sample119.tar.gz" \
    -C "${RUN_DIR}" \
    vtu_rollout_gpu_sample119

echo "============================================================"
echo "GPU VTU EXPORT COMPLETE"
echo
echo "PVD:"
find "${OUT_DIR}" -name "*.pvd" -print
echo
echo "ARCHIVE:"
ls -lh "${RUN_DIR}/vtu_rollout_gpu_sample119.tar.gz"
echo "============================================================"
