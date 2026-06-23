#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-memco}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${ROOT}/data/alfworld}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT}/requirements-full-freeze.txt"

export ALFWORLD_DATA
alfworld-download --force-download --force

echo
echo "MemCo environment is ready."
echo "  conda env: ${ENV_NAME}"
echo "  ALFWORLD_DATA=${ALFWORLD_DATA}"
echo "  ALFWorld game root: ${ALFWORLD_DATA}/json_2.1.1"
