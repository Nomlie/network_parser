#!/bin/bash
# Shared CHPC module + conda activation for PBS job scripts.
# Source this after #PBS directives (and after set -euo pipefail).
# Safe to source from an interactive shell that already has conda.

ulimit -s unlimited

# Load CHPC software stack when modules are available (compute nodes).
if command -v module >/dev/null 2>&1; then
  module load chpc/BIOMODULES 2>/dev/null || true
  module load anaconda/3 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

# Activate networkparser conda env (required for package + deps).
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME:-networkparser}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
  : # already inside a conda env
else
  echo "WARNING: conda not found; relying on PYTHON_BIN=${PYTHON_BIN:-python}" >&2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "Starting job at $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo "PWD: $(pwd)"
echo "Python: $(command -v python || true) ($(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo n/a))"
if command -v free >/dev/null 2>&1; then
  echo "Free memory: $(free -h | head -n 2 | tr -s ' ')"
fi
if command -v nproc >/dev/null 2>&1; then
  echo "nproc: $(nproc)"
fi
