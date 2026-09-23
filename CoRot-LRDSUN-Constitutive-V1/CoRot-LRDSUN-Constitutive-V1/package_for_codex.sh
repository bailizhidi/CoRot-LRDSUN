#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# package_for_codex.sh
#
# Purpose:
#   Create a compact source/context bundle for Codex to inspect
#   CoRot-LRDSUN-Constitutive-V1 without copying large datasets,
#   checkpoints, outputs, or prepared caches.
#
# Usage:
#   bash package_for_codex.sh
#
# Optional:
#   PROJECT_DIR=/path/to/project \
#   RAW_DATA_DIR=/path/to/raw_dataset \
#   bash package_for_codex.sh
# ============================================================

PROJECT_DIR="${PROJECT_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/my_experiments/CoRot-LRDSUN-Constitutive-V1}"
RAW_DATA_DIR="${RAW_DATA_DIR:-/data/home/scxk573/run/lsz/ysy/physicsnemo/data/dataset_corot_constitutive_prod_v1}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/prepared_cache_prod_v1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="CoRot-LRDSUN-Constitutive-V1_codex_${STAMP}"
WORK_ROOT="${PROJECT_DIR}/../${BUNDLE_NAME}"
SRC_DST="${WORK_ROOT}/CoRot-LRDSUN-Constitutive-V1"
CTX_DST="${WORK_ROOT}/codex_context"
ARCHIVE="${PROJECT_DIR}/../${BUNDLE_NAME}.tar.gz"

# Representative experiment identifiers
C1024_JOB_ID="${C1024_JOB_ID:-1610485}"
C2048_JOB_ID="${C2048_JOB_ID:-1619336}"
C1024_RUN_TAG="${C1024_RUN_TAG:-run_20260920_143444_1610485}"
C2048_RUN_TAG="${C2048_RUN_TAG:-run_20260922_192848_1619336}"

echo "============================================================"
echo "Codex packaging"
echo "PROJECT_DIR = ${PROJECT_DIR}"
echo "RAW_DATA_DIR = ${RAW_DATA_DIR}"
echo "CACHE_DIR = ${CACHE_DIR}"
echo "WORK_ROOT = ${WORK_ROOT}"
echo "ARCHIVE = ${ARCHIVE}"
echo "============================================================"

if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "[ERROR] PROJECT_DIR does not exist: ${PROJECT_DIR}" >&2
    exit 1
fi

rm -rf "${WORK_ROOT}"
mkdir -p "${SRC_DST}" "${CTX_DST}/cache_metadata" "${CTX_DST}/representative_logs"

# ------------------------------------------------------------
# 1. Copy compact project source
# ------------------------------------------------------------
echo "[1/8] Copying source/configuration files..."

# Prefer rsync if available because its exclusions are reliable.
if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude='outputs/' \
      --exclude='logs/' \
      --exclude='prepared_cache_prod_v1/' \
      --exclude='__pycache__/' \
      --exclude='.pytest_cache/' \
      --exclude='.mypy_cache/' \
      --exclude='.git/' \
      --exclude='*.pt' \
      --exclude='*.pth' \
      --exclude='*.ckpt' \
      --exclude='*.npz' \
      --exclude='*.npy' \
      --exclude='*.h5' \
      --exclude='*.hdf5' \
      --exclude='*.odb' \
      --exclude='*.sim' \
      --exclude='*.stt' \
      --exclude='*.prt' \
      --exclude='*.res' \
      --exclude='*.mdl' \
      --exclude='*.pac' \
      --exclude='*.abq' \
      --exclude='*.sel' \
      --exclude='*.lck' \
      --exclude='*.tar' \
      --exclude='*.tar.gz' \
      --exclude='*.tar.xz' \
      --exclude='*.zip' \
      "${PROJECT_DIR}/" "${SRC_DST}/"
else
    echo "[WARN] rsync not found; using tar fallback."
    (
      cd "${PROJECT_DIR}"
      tar \
        --exclude='./outputs' \
        --exclude='./logs' \
        --exclude='./prepared_cache_prod_v1' \
        --exclude='./__pycache__' \
        --exclude='*/__pycache__' \
        --exclude='./.git' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.ckpt' \
        --exclude='*.npz' \
        --exclude='*.npy' \
        --exclude='*.h5' \
        --exclude='*.hdf5' \
        --exclude='*.odb' \
        --exclude='*.tar' \
        --exclude='*.tar.gz' \
        --exclude='*.tar.xz' \
        --exclude='*.zip' \
        -cf - .
    ) | (
      cd "${SRC_DST}"
      tar -xf -
    )
fi

# ------------------------------------------------------------
# 2. Project tree
# ------------------------------------------------------------
echo "[2/8] Generating project tree..."

(
  cd "${PROJECT_DIR}"
  find . \
    -path './outputs' -prune -o \
    -path './logs' -prune -o \
    -path './prepared_cache_prod_v1' -prune -o \
    -path './.git' -prune -o \
    -path '*/__pycache__' -prune -o \
    -type f \
    ! -name '*.pt' \
    ! -name '*.pth' \
    ! -name '*.ckpt' \
    ! -name '*.npz' \
    ! -name '*.npy' \
    ! -name '*.h5' \
    ! -name '*.hdf5' \
    ! -name '*.odb' \
    -print | sort
) > "${CTX_DST}/PROJECT_TREE.txt"

# ------------------------------------------------------------
# 3. Environment snapshot
# ------------------------------------------------------------
echo "[3/8] Recording environment..."

{
  echo "DATE=$(date -Is 2>/dev/null || date)"
  echo "HOSTNAME=$(hostname)"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "RAW_DATA_DIR=${RAW_DATA_DIR}"
  echo "CACHE_DIR=${CACHE_DIR}"
  echo
  echo "=== Python ==="
  command -v python || true
  python --version 2>&1 || true
  echo
  echo "=== PyTorch / CUDA ==="
  python - <<'PY' 2>/dev/null || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(i, torch.cuda.get_device_name(i), f"{p.total_memory/1024**3:.2f} GiB")
except Exception as e:
    print("torch probe failed:", repr(e))
PY
  echo
  echo "=== Git ==="
  if [[ -d "${PROJECT_DIR}/.git" ]]; then
      git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || true
      git -C "${PROJECT_DIR}" status --short 2>/dev/null || true
  else
      echo "No .git directory inside project."
  fi
} > "${CTX_DST}/ENVIRONMENT.txt"

# ------------------------------------------------------------
# Helper: inspect a data file without copying its full contents
# ------------------------------------------------------------
inspect_file() {
    local INPUT_FILE="$1"
    local OUTPUT_FILE="$2"

    python - "${INPUT_FILE}" > "${OUTPUT_FILE}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
print("FILE:", path)
print("SUFFIX:", path.suffix)
print("=" * 110)

def show_value(name, x):
    shape = getattr(x, "shape", None)
    dtype = getattr(x, "dtype", None)
    if shape is not None:
        print(f"{name:40s} shape={str(tuple(shape)):24s} dtype={str(dtype):16s}")
    else:
        t = type(x).__name__
        s = repr(x)
        if len(s) > 250:
            s = s[:247] + "..."
        print(f"{name:40s} type={t:24s} value={s}")

suffix = path.suffix.lower()

try:
    if suffix == ".npz":
        import numpy as np
        d = np.load(path, allow_pickle=True)
        for k in d.files:
            show_value(k, d[k])

    elif suffix == ".npy":
        import numpy as np
        x = np.load(path, allow_pickle=True)
        show_value(path.name, x)

    elif suffix in {".pt", ".pth", ".ckpt"}:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)

        def walk(prefix, value, depth=0):
            if depth > 3:
                print(prefix, "<max depth reached>")
                return
            if isinstance(value, dict):
                print(prefix or "<root>", f"dict[{len(value)}]")
                for k, v in list(value.items())[:200]:
                    walk(f"{prefix}.{k}" if prefix else str(k), v, depth + 1)
            elif isinstance(value, (list, tuple)):
                print(prefix, f"{type(value).__name__}[{len(value)}]")
                for i, v in enumerate(value[:20]):
                    walk(f"{prefix}[{i}]", v, depth + 1)
            else:
                show_value(prefix or "<root>", value)

        walk("", obj)

    elif suffix in {".json"}:
        import json
        obj = json.loads(path.read_text())
        print(json.dumps(obj, indent=2)[:50000])

    elif suffix in {".txt", ".yaml", ".yml", ".csv"}:
        print(path.read_text(errors="replace")[:50000])

    else:
        print("No specialized inspector for this suffix.")
        print("File size:", path.stat().st_size, "bytes")

except Exception as e:
    print("INSPECTION_FAILED:", repr(e))
PY
}

# ------------------------------------------------------------
# 4. Raw-data schema
# ------------------------------------------------------------
echo "[4/8] Generating raw-data schema..."

RAW_SAMPLE=""
if [[ -d "${RAW_DATA_DIR}" ]]; then
    RAW_SAMPLE="$(find "${RAW_DATA_DIR}" -type f \( -name '*.npz' -o -name '*.npy' \) | sort | head -n 1 || true)"
fi

if [[ -n "${RAW_SAMPLE}" ]]; then
    inspect_file "${RAW_SAMPLE}" "${CTX_DST}/DATA_SCHEMA.txt"
else
    {
      echo "No representative .npz/.npy file found."
      echo "RAW_DATA_DIR=${RAW_DATA_DIR}"
      if [[ -d "${RAW_DATA_DIR}" ]]; then
          echo
          echo "First files:"
          find "${RAW_DATA_DIR}" -maxdepth 2 -type f | sort | head -n 50
      fi
    } > "${CTX_DST}/DATA_SCHEMA.txt"
fi

# ------------------------------------------------------------
# 5. Prepared-cache schema + metadata
# ------------------------------------------------------------
echo "[5/8] Generating prepared-cache schema..."

CACHE_SAMPLE=""
if [[ -d "${CACHE_DIR}" ]]; then
    CACHE_SAMPLE="$(find "${CACHE_DIR}" -type f \
      \( -name '*.npz' -o -name '*.npy' -o -name '*.pt' -o -name '*.pth' \) \
      | sort | head -n 1 || true)"

    # Copy only lightweight metadata/config files.
    while IFS= read -r f; do
        rel="${f#${CACHE_DIR}/}"
        dst="${CTX_DST}/cache_metadata/${rel}"
        mkdir -p "$(dirname "${dst}")"
        cp -p "${f}" "${dst}"
    done < <(
        find "${CACHE_DIR}" -maxdepth 3 -type f \
          \( -name '*.json' -o -name '*.txt' -o -name '*.yaml' -o -name '*.yml' -o -name '*.csv' \) \
          -size -10M | sort
    )
fi

if [[ -n "${CACHE_SAMPLE}" ]]; then
    inspect_file "${CACHE_SAMPLE}" "${CTX_DST}/CACHE_SCHEMA.txt"
else
    {
      echo "No representative .npz/.npy/.pt/.pth file found."
      echo "CACHE_DIR=${CACHE_DIR}"
      if [[ -d "${CACHE_DIR}" ]]; then
          echo
          echo "First files:"
          find "${CACHE_DIR}" -maxdepth 2 -type f | sort | head -n 80
      fi
    } > "${CTX_DST}/CACHE_SCHEMA.txt"
fi

# ------------------------------------------------------------
# 6. Representative logs
# ------------------------------------------------------------
echo "[6/8] Looking for representative C1024/C2048 logs..."

collect_log() {
    local LABEL="$1"
    local JOB_ID="$2"
    local RUN_TAG="$3"
    local FOUND=""

    # Search filenames first.
    FOUND="$(find "${PROJECT_DIR}" -type f \
      \( -path '*/logs/*' -o -name '*.out' -o -name '*.log' -o -name '*.txt' \) \
      \( -name "*${JOB_ID}*" -o -name "*${RUN_TAG}*" \) \
      2>/dev/null | sort | head -n 1 || true)"

    # If filename search fails, grep text logs for job/run identifiers.
    if [[ -z "${FOUND}" ]] && command -v grep >/dev/null 2>&1; then
        FOUND="$(grep -RIl \
          --include='*.out' --include='*.log' --include='*.txt' \
          -e "${JOB_ID}" -e "${RUN_TAG}" \
          "${PROJECT_DIR}/logs" "${PROJECT_DIR}" \
          2>/dev/null | head -n 1 || true)"
    fi

    if [[ -n "${FOUND}" && -f "${FOUND}" ]]; then
        cp -p "${FOUND}" "${CTX_DST}/representative_logs/${LABEL}_$(basename "${FOUND}")"
        echo "${LABEL}: ${FOUND}" >> "${CTX_DST}/representative_logs/LOG_INDEX.txt"
    else
        echo "${LABEL}: NOT FOUND automatically (job=${JOB_ID}, run=${RUN_TAG})" \
          >> "${CTX_DST}/representative_logs/LOG_INDEX.txt"
    fi
}

: > "${CTX_DST}/representative_logs/LOG_INDEX.txt"
collect_log "C1024_5STEP" "${C1024_JOB_ID}" "${C1024_RUN_TAG}"
collect_log "C2048_5STEP" "${C2048_JOB_ID}" "${C2048_RUN_TAG}"

# ------------------------------------------------------------
# 7. Human-readable Codex context
# ------------------------------------------------------------
echo "[7/8] Writing Codex context note..."

cat > "${CTX_DST}/README_FOR_CODEX.md" <<'EOF'
# CoRot-LRDSUN-Constitutive-V1: context for code audit

## Goal

This repository implements a constitutive surrogate for tube bending.

Current high-level pipeline:

Geometry / process
→ displacement trajectory U_t
→ CoRot-LRDSUN constitutive model
→ S_t, PE_t, PEEQ_t, LE_t

The immediate research task is NOT to replace the baseline blindly.

We want to investigate a new architecture:

local subgraph
→ CoRot objective local representation
→ local encoder
→ one latent token per material state
→ Transolver / Physics-Attention across tokens from the SAME FE sample and SAME time step
→ center material-state update.

## Current recursive state

z_t = [S4, PE4, PEEQ]

S4 consists of four local stress components.
PE4 consists of four local plastic-strain components.
PEEQ is scalar.

LE is auxiliary/non-recursive.
Mises is computed analytically from the predicted stress tensor components.

## Important current contracts

- local neighborhood is CoRot/objective;
- center material state is recursive;
- neighbors provide kinematics, not their ground-truth material state;
- shell surfaces use paired SPOS/SNEG material states;
- geometric-center sampling is expanded into two material states;
- the current production model supports differentiable multi-step unrolling;
- the current 5-step weights are:
  [1.0, 0.75, 0.5, 0.35, 0.25].

## Important comparison

Baseline 5-step C1024:
- 1024 geometric centers/window
- 2048 paired material states/window
- batch size 2048

Spatial-data experiment C2048:
- 2048 geometric centers/window
- 4096 paired material states/window
- batch size 4096

The intended Transolver extension must NOT mix tokens from different FE samples
inside the same attention sequence.

## First Codex task

Before editing code, identify:

1. exact local CoRot construction;
2. local-neighbor encoder;
3. local attention/pooling;
4. recursive-state encoder;
5. context encoder;
6. fusion point before constitutive trunk;
7. output heads;
8. SPOS/SNEG representation;
9. centers/batch semantics;
10. differentiable 5-step unrolling;
11. tensor shapes;
12. checkpoint loading/saving.

Then propose the smallest safe insertion point for a Transolver-style block.

Do not overwrite the existing production baseline.
EOF

# ------------------------------------------------------------
# 8. Manifest + archive
# ------------------------------------------------------------
echo "[8/8] Creating manifest and archive..."

{
  echo "Bundle: ${BUNDLE_NAME}"
  echo "Created: $(date -Is 2>/dev/null || date)"
  echo
  echo "=== Included source size ==="
  du -sh "${SRC_DST}" 2>/dev/null || true
  echo
  echo "=== Context size ==="
  du -sh "${CTX_DST}" 2>/dev/null || true
  echo
  echo "=== Files ==="
  find "${WORK_ROOT}" -type f | sed "s#${WORK_ROOT}/##" | sort
} > "${CTX_DST}/BUNDLE_MANIFEST.txt"

tar -C "$(dirname "${WORK_ROOT}")" -czf "${ARCHIVE}" "$(basename "${WORK_ROOT}")"

echo
echo "============================================================"
echo "DONE"
echo "Bundle directory:"
echo "  ${WORK_ROOT}"
echo
echo "Archive:"
echo "  ${ARCHIVE}"
echo
ls -lh "${ARCHIVE}"
echo
echo "Sanity-check archive contents:"
tar -tzf "${ARCHIVE}" | head -n 40
echo "============================================================"
