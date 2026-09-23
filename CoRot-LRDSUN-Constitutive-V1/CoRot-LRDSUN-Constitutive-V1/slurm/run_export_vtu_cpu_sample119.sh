#!/bin/bash
#SBATCH --job-name=ysy_corot-vtu119
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1
CACHE_DIR=${PROJECT_DIR}/prepared_cache_prod_v1
RUN_DIR=${PROJECT_DIR}/outputs/formal/run_20260904_001957_1546153
OUT_DIR=${RUN_DIR}/vtu_rollout_cpu_sample119

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "CPU VTU EXPORT"
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "============================================================"

# ------------------------------------------------------------
# Force CPU-only execution
# ------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------
source /data/home/scxk573/run/lsz/ysy/physicsnemo/.venv/bin/activate 2>/dev/null || true

which python
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda visible:", torch.cuda.is_available())
PY

echo "============================================================"
echo "Export sample119"
echo "============================================================"

python -u export_rollout_vtu_cpu.py \
    --cache-dir "${CACHE_DIR}" \
    --checkpoint "${RUN_DIR}/best_val.pt" \
    --output-dir "${OUT_DIR}" \
    --split test \
    --sample-ids 119 \
    --batch-size 2048 \
    --threads ${SLURM_CPUS_PER_TASK:-16} \
    --frame-stride 1 \
    --fields all \
    --overwrite

echo "============================================================"
echo "Create archive"
echo "============================================================"

cd "${RUN_DIR}"

tar -czf vtu_rollout_cpu_sample119.tar.gz \
    vtu_rollout_cpu_sample119

echo "============================================================"
echo "VTU EXPORT COMPLETE"
echo "PVD:"
find "${OUT_DIR}" -name "*.pvd" -print

echo
echo "ARCHIVE:"
ls -lh "${RUN_DIR}/vtu_rollout_cpu_sample119.tar.gz"

echo "============================================================"
