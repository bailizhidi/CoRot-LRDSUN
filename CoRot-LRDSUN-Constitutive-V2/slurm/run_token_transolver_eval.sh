#!/usr/bin/env bash
# V2 evaluator wrapper.  Evaluation is token-global: no V1 surface chunks.
#SBATCH --job-name=corot-v2-token-eval
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
source "${V2_DIR}/slurm/common.sh"
cd "${V2_DIR}"

CHECKPOINT="${CHECKPOINT:-}"
[[ -f "${CHECKPOINT}" ]] || { echo "Set CHECKPOINT to a V2 checkpoint"; exit 2; }
OUT_DIR="${OUT_DIR:-${V2_DIR}/outputs/token_eval_${SLURM_JOB_ID:-manual}}"
mkdir -p "${OUT_DIR}"

"${PYTHON}" "${V2_DIR}/evaluate_token_teacher_forcing.py" \
  --cache-dir "${CACHE_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT:-test}" \
  --output "${OUT_DIR}/teacher_forcing.json"

"${PYTHON}" "${V2_DIR}/evaluate_token_rollout.py" \
  --cache-dir "${CACHE_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --split "${SPLIT:-test}" \
  --output-dir "${OUT_DIR}/rollout"
