#!/usr/bin/env bash
# Shared environment for the independent V2 branch.
set -euo pipefail

# Slurm runs a submitted script from /var/spool/slurmd.  The submit directory
# is the V2 project when wrappers are submitted from that project directory.
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"

# Reuse the V1 Slurm bootstrap, which extracts the tested PyTorch environment
# on the compute node.  The cluster layout stores V1 directly beside V2; the
# local checkout keeps an additional bundle directory, so accept both forms.
V1_HINT="${V1_DIR:-${V2_DIR}/../CoRot-LRDSUN-Constitutive-V1}"
if [[ -f "${V1_HINT}/slurm/common.sh" ]]; then
  V1_DIR="${V1_HINT}"
elif [[ -f "${V1_HINT}/CoRot-LRDSUN-Constitutive-V1/slurm/common.sh" ]]; then
  V1_DIR="${V1_HINT}/CoRot-LRDSUN-Constitutive-V1"
else
  echo "Missing V1 Slurm environment under ${V1_HINT}" >&2
  exit 2
fi
source "${V1_DIR}/slurm/common.sh"

# V1 common.sh sets PROJECT_DIR, CACHE_DIR, PYTHON, and PYTHONPATH.  Add V2
# after it so the V2 package is imported while the V1 runtime remains intact.
V1_DIR="${PROJECT_DIR:-${V1_DIR}}"
CACHE_DIR="${CACHE_DIR:-${V1_DIR}/prepared_cache_prod_v1}"
export PYTHONPATH="${V2_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-6}"

echo "============================================================"
echo "V2_DIR=${V2_DIR}"
echo "V1_DIR=${V1_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "PYTHON=${PYTHON}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
"${PYTHON}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
echo "============================================================"
