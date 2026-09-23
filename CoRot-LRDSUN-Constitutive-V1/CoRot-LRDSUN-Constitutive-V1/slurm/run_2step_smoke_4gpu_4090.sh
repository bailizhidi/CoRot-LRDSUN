#!/bin/bash
#SBATCH --job-name=ysy_corot-2step-smoke
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
cd "${PROJECT_DIR}"

# 4 x RTX 4090 -> 24 CPUs total -> 6 CPU threads per DDP rank.
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

V1_RUN="${PROJECT_DIR}/outputs/formal/run_20260904_001957_1546153"
WARM_START="${V1_RUN}/best_val.pt"
OUT="${PROJECT_DIR}/outputs/finetune_2step_smoke/run_${SLURM_JOB_ID}"
mkdir -p "${OUT}"

echo "============================================================"
echo "2-step differentiable fine-tune SMOKE"
echo "WARM_START=${WARM_START}"
echo "OUT=${OUT}"
echo "============================================================"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${PROJECT_DIR}/train_2step_finetune_ddp.py" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUT}" \
  --warm-start "${WARM_START}" \
  --epochs 2 \
  --batch-size 2048 \
  --eval-batch-size 2048 \
  --windows-per-sample 2 \
  --centers-per-window 1024 \
  --val-windows-per-sample 2 \
  --val-centers-per-window 512 \
  --hidden-dim 192 \
  --lr 2e-5 \
  --min-lr 2e-6 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --step2-weight 0.5 \
  --early-stop-start 18 \
  --early-stop-patience 8 \
  --w-delta8 1.0 \
  --w-peeq 1.0 \
  --w-gate 0.20 \
  --w-le 0.25 \
  --w-mises 0.25 \
  --smoke

echo "SMOKE_OUT=${OUT}"
