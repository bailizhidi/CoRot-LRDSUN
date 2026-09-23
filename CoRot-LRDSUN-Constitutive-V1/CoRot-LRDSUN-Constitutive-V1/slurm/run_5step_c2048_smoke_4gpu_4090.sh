#!/bin/bash
#SBATCH --job-name=ysy_corot-5s-c2048-smk
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
OUT="${PROJECT_DIR}/outputs/finetune_5step_c2048_smoke/run_${SLURM_JOB_ID}"
mkdir -p "${OUT}"

if [[ ! -f "${WARM_START}" ]]; then
  echo "ERROR: V2 checkpoint not found: ${WARM_START}"
  exit 1
fi

echo "============================================================"
echo "5-step C2048 differentiable fine-tune SMOKE"
echo "V2_RUN=${V2_RUN}"
echo "WARM_START=${WARM_START}"
echo "OUT=${OUT}"
echo "============================================================"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${PROJECT_DIR}/train_5step_finetune_ddp.py" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUT}" \
  --warm-start "${WARM_START}" \
  --epochs 2 \
  --batch-size 4096 \
  --eval-batch-size 2048 \
  --windows-per-sample 2 \
  --centers-per-window 2048 \
  --val-windows-per-sample 2 \
  --val-centers-per-window 512 \
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
  --w-mises 0.25 \
  --smoke

echo "SMOKE_OUT=${OUT}"
