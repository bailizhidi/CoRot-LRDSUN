#!/usr/bin/env bash
#SBATCH --job-name=ysy_v2_trans_c1024
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
V2_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec "${V2_DIR}/slurm/run_token_transolver_train_4gpu.sh" "$@"
