#!/bin/bash
#SBATCH --job-name=ysy_corot-5step-eval
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
RUN_DIR="${RUN_DIR:-$(cat "${PROJECT_DIR}/finetune_5step_latest.txt")}"; CKPT="${CKPT:-${RUN_DIR}/best_val_5step.pt}"
[[ -f "${CKPT}" ]] || { echo "MISSING CKPT=${CKPT}"; exit 2; }
"${PYTHON}" "${PROJECT_DIR}/evaluate_teacher_forcing.py" --cache-dir "${CACHE_DIR}" --checkpoint "${CKPT}" --split test --output "${RUN_DIR}/test_teacher_forcing.json" --batch-size 1024
"${PYTHON}" "${PROJECT_DIR}/evaluate_rollout.py" --cache-dir "${CACHE_DIR}" --checkpoint "${CKPT}" --split test --output-dir "${RUN_DIR}/test_rollout" --batch-size 1024
