#!/usr/bin/env bash
# Step 1.5 real V1-cache/checkpoint integration smoke.
# No optimizer is created and no optimizer.step() is called.
# Submit explicitly, for example:
#   sbatch --gpus=1 -p gpu_4090 slurm/run_real_token_pipeline_smoke_1gpu.sh

#SBATCH --job-name=corot-v2-real-smoke
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Slurm executes a submitted script from /var/spool/slurmd.  Prefer the
# submit directory so this remains the real V2 project path after spooling.
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
V1_PROJECT_DIR="${V1_PROJECT_DIR:-${V2_DIR}/../CoRot-LRDSUN-Constitutive-V1}"

# Reuse the V1 Slurm environment only for the already-tested Python runtime and
# read-only cache path.  No V1 source or checkpoint is modified.
source "${V1_PROJECT_DIR}/slurm/common.sh"

export PYTHONPATH="${V2_DIR}:${PYTHONPATH:-}"
cd "${V2_DIR}"

CACHE_DIR="${CACHE_DIR:-${V1_PROJECT_DIR}/prepared_cache_prod_v1}"
CHECKPOINT="${CHECKPOINT:-${V1_PROJECT_DIR}/outputs/finetune_5step/run_20260920_143444_1610485/best_val_5step.pt}"
SAMPLE_ID="${SAMPLE_ID:-119}"
TIME_INDEX="${TIME_INDEX:-0}"
CENTER_COUNTS="${CENTER_COUNTS:-1024}"
OUT_DIR="${OUT_DIR:-${V2_DIR}/logs/real_token_smoke_${SLURM_JOB_ID:-manual}}"
mkdir -p "${OUT_DIR}"

echo "V2_DIR=${V2_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SAMPLE_ID=${SAMPLE_ID} TIME_INDEX=${TIME_INDEX} CENTER_COUNTS=${CENTER_COUNTS}"

"${PYTHON}" "${V2_DIR}/test_real_token_pipeline.py" \
  --cache-dir "${CACHE_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --sample-id "${SAMPLE_ID}" \
  --time "${TIME_INDEX}" \
  --centers ${CENTER_COUNTS} \
  --device cuda \
  | tee "${OUT_DIR}/result.json"
