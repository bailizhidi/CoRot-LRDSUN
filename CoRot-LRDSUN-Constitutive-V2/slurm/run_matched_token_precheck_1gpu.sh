#!/usr/bin/env bash
# Matched-token sample119 one-step sanity check.
#SBATCH --job-name=ysy_v2_matched_pre
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
V2_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${V2_DIR}/slurm/common.sh"
cd "${V2_DIR}"
CHECKPOINT="${CHECKPOINT:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step/best_val.pt}"
OUT_DIR="${OUT_DIR:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step/test15_matched_tokens}"
"${PYTHON}" "${V2_DIR}/evaluate_matched_tokens.py" \
  --cache-dir "${CACHE_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_DIR}" \
  --precheck
