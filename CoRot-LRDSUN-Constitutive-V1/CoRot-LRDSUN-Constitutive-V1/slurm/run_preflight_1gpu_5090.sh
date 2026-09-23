#!/bin/bash
#SBATCH --job-name=ysy_corot-constit-check
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
[[ -f "${CACHE_DIR}/stats.json" ]] || { echo "MISSING stats.json"; exit 2; }
"${PYTHON}" "${PROJECT_DIR}/preflight.py" --cache-dir "${CACHE_DIR}" --device cuda
