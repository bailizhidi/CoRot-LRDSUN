#!/bin/bash
#SBATCH --job-name=ysy_corot-v2-kreset
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"

cd "${PROJECT_DIR}"

# ============================================================
# V2: CoRot-LRDSUN-2Step
# ============================================================
RUN_DIR="$(cat "${PROJECT_DIR}/finetune_2step_latest.txt")"
CHECKPOINT="${RUN_DIR}/best_val_2step.pt"

OUT_DIR="${RUN_DIR}/kstep_reset_test15"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "============================================================"
echo "V2 K-step reset diagnostic"
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "OUT_DIR=${OUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found:"
    echo "${CHECKPOINT}"
    exit 1
fi

nvidia-smi

"${PYTHON}" -u evaluate_kstep_reset.py \
    --cache-dir "${CACHE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${OUT_DIR}" \
    --split test \
    --ks 1,2,5,10,20,30,60,180 \
    --batch-size 8192 \
    --device cuda

echo
echo "============================================================"
echo "V2 K-STEP GLOBAL RESULTS"
echo "============================================================"

cat "${OUT_DIR}/kstep_global_metrics.csv"

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
