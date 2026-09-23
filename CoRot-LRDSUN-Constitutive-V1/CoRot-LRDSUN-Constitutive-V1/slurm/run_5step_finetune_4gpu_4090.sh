#!/bin/bash
#SBATCH --job-name=ysy_corot-5step-ft
#SBATCH --cpus-per-task=32
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

V2_RUN="$(cat "${PROJECT_DIR}/finetune_2step_latest.txt")"
WARM_START="${V2_RUN}/best_val_2step.pt"
OUT_ROOT="${PROJECT_DIR}/outputs/finetune_5step"
RUN_DIR="${OUT_ROOT}/run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

if [[ ! -f "${WARM_START}" ]]; then
  echo "ERROR: V2 checkpoint not found: ${WARM_START}"
  exit 1
fi

echo "============================================================"
echo "5-step differentiable fine-tune FORMAL"
echo "V2_RUN=${V2_RUN}"
echo "WARM_START=${WARM_START}"
echo "RUN_DIR=${RUN_DIR}"
echo "============================================================"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${PROJECT_DIR}/train_5step_finetune_ddp.py" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${RUN_DIR}" \
  --warm-start "${WARM_START}" \
  --epochs 30 \
  --batch-size 2048 \
  --eval-batch-size 2048 \
  --windows-per-sample 15 \
  --centers-per-window 1024 \
  --val-windows-per-sample 18 \
  --val-centers-per-window 1024 \
  --hidden-dim 192 \
  --lr 1e-5 \
  --min-lr 1e-6 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --step-weights 1.0,0.75,0.50,0.35,0.25 \
  --early-stop-start 18 \
  --early-stop-patience 8 \
  --w-delta8 1.0 \
  --w-peeq 1.0 \
  --w-gate 0.20 \
  --w-le 0.25 \
  --w-mises 0.25

printf '%s\n' "${RUN_DIR}" > "${PROJECT_DIR}/finetune_5step_latest.txt"
echo "RUN_DIR=${RUN_DIR}"
