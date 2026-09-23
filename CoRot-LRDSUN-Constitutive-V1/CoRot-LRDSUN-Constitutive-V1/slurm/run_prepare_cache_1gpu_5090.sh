#!/bin/bash
#SBATCH --job-name=ysy_corot-constit-prep
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
[[ -d "${RAW_DATA_DIR}" ]] || { echo "MISSING RAW_DATA_DIR=${RAW_DATA_DIR}"; exit 2; }
"${PYTHON}" "${PROJECT_DIR}/scan_dataset.py" --dataset-dir "${RAW_DATA_DIR}" --expect-samples 150
"${PYTHON}" "${PROJECT_DIR}/prepare_training_cache.py" --dataset-dir "${RAW_DATA_DIR}" --cache-dir "${CACHE_DIR}" --workers 4
"${PYTHON}" "${PROJECT_DIR}/compute_stats.py" --cache-dir "${CACHE_DIR}" --transitions-per-sample 12 --centers-per-transition 512
"${PYTHON}" "${PROJECT_DIR}/preflight.py" --cache-dir "${CACHE_DIR}" --device cuda
