#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

PHYSICSNEMO_ROOT="${PHYSICSNEMO_ROOT:-/data/home/scxk573/run/lsz/ysy/physicsnemo}"
PROJECT_DIR="${PROJECT_DIR:-${PHYSICSNEMO_ROOT}/my_experiments/CoRot-LRDSUN-Constitutive-V1}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${PHYSICSNEMO_ROOT}/data/dataset_corot_constitutive_prod_v1}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/prepared_cache_prod_v1}"
VENV_TAR="${VENV_TAR:-/data/run01/scxk573/lsz/codes/physi-ai/venv_with_vtk.tar}"

WORKDIR="/dev/shm/${USER}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
VIRTUAL_ENV="${WORKDIR}/.venv"
PYTHON="${VIRTUAL_ENV}/bin/python3"
cleanup(){ status=$?; rm -rf "${WORKDIR}"; exit "${status}"; }
trap cleanup EXIT
mkdir -p "${WORKDIR}" "${PHYSICSNEMO_ROOT}/logs" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/outputs"
[[ -f "${VENV_TAR}" ]] || { echo "MISSING VENV_TAR=${VENV_TAR}"; exit 2; }
[[ -d "${PROJECT_DIR}" ]] || { echo "MISSING PROJECT_DIR=${PROJECT_DIR}"; exit 2; }
tar -xf "${VENV_TAR}" -C "${WORKDIR}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}:${PHYSICSNEMO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
unset NCCL_ASYNC_ERROR_HANDLING || true
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_RAS_ENABLE=0
export NCCL_DEBUG=WARN

echo "============================================================"
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "RAW_DATA_DIR=${RAW_DATA_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
"${PYTHON}" - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), f'{p.total_memory/1024**3:.2f} GiB')
PY
echo "============================================================"
