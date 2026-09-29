#!/usr/bin/env bash
# First-step teacher-forcing versus rollout equivalence audit.
#SBATCH --job-name=ysy_v2_firststep
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
V2_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${V2_DIR}/slurm/common.sh"
cd "${V2_DIR}"

CHECKPOINT="${CHECKPOINT:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step/best_val.pt}"
OUT="${OUT:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step/test15/first_step_equivalence_audit.json}"
[[ -f "${CHECKPOINT}" ]] || { echo "MISSING CHECKPOINT=${CHECKPOINT}" >&2; exit 2; }

"${PYTHON}" "${V2_DIR}/audit_first_step_equivalence.py" \
  --cache-dir "${CACHE_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUT}" \
  --sample-id 119 \
  --centers 1024
