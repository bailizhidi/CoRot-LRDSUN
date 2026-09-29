#!/usr/bin/env bash
# Corrected final Test15 evaluation after the first-step gates pass.
#SBATCH --job-name=ysy_v2_test15_fix
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
V2_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${V2_DIR}/slurm/common.sh"
cd "${V2_DIR}"
CHECKPOINT="${CHECKPOINT:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step/best_val.pt}"
V1_RESULT_DIR="${V1_RESULT_DIR:-${V1_DIR}/outputs/finetune_5step/run_20260920_143444_1610485}"
OUT_DIR="${OUT_DIR:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step/test15_surface_order_fixed}"
OLD_DIR="${OLD_DIR:-${V2_DIR}/outputs/formal_token_transolver_c1024_5step/test15}"
[[ -f "${CHECKPOINT}" ]] || { echo "MISSING CHECKPOINT=${CHECKPOINT}" >&2; exit 2; }
[[ -f "${V1_RESULT_DIR}/test_teacher_forcing.json" ]] || { echo "MISSING V1 teacher-forcing result" >&2; exit 2; }
[[ -f "${V1_RESULT_DIR}/test_rollout/rollout_summary.json" ]] || { echo "MISSING V1 rollout result" >&2; exit 2; }
mkdir -p "${OUT_DIR}"
if [[ -f "${OLD_DIR}/sdpa_equivalence_check.json" ]]; then cp -f "${OLD_DIR}/sdpa_equivalence_check.json" "${OUT_DIR}/sdpa_equivalence_check.json"; fi
"${PYTHON}" "${V2_DIR}/evaluate_test15.py" \
  --cache-dir "${CACHE_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --v1-result-dir "${V1_RESULT_DIR}" \
  --output-dir "${OUT_DIR}"
