#!/usr/bin/env bash
# Start a local vLLM OpenAI-compatible embedding server.
#
# Usage:
#   bash scripts/launch_qwen3_embedding_api.sh
#   MODEL_PATH=/path/to/Qwen3-Embedding-0.6B bash scripts/launch_qwen3_embedding_api.sh
#   CUDA_VISIBLE_DEVICES=1 PORT=8001 bash scripts/launch_qwen3_embedding_api.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Default to the first visible GPU; override CUDA_VISIBLE_DEVICES explicitly on
# multi-GPU machines when targeting a specific device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DEFAULT_MODEL_PATH="/Model/Qwen3-Embedding-0.6B"
if [[ ! -e "${DEFAULT_MODEL_PATH}" ]]; then
  DEFAULT_MODEL_PATH="/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-embedding-api}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PID_FILE="${PID_DIR}/qwen3_embedding_api.pid"
LOG_FILE="${LOG_DIR}/qwen3_embedding_api.log"
BASE_URL="http://127.0.0.1:${PORT}/v1"

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "ERROR: MODEL_PATH not found: ${MODEL_PATH}"
  echo "Set MODEL_PATH to an existing embedding model directory before launching."
  exit 1
fi

is_embedding_server_pid() {
  local pid="$1"
  local cmdline
  cmdline="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  [[ "${cmdline}" == *"vllm.entrypoints.openai.api_server"* ]] && [[ "${cmdline}" == *"${SERVED_MODEL_NAME}"* ]]
}

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    if is_embedding_server_pid "${OLD_PID}"; then
      echo "qwen3 embedding api already running with pid ${OLD_PID}"
      echo "log: ${LOG_FILE}"
      echo "base url: ${BASE_URL}"
      exit 0
    fi
    echo "stale pid file points to a different process: ${OLD_PID}"
  fi
  rm -f "${PID_FILE}"
fi

nohup "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --runner pooling \
  --convert embed \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "started qwen3 embedding api"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "base url: ${BASE_URL}"
echo "model id: ${SERVED_MODEL_NAME}"
echo "cuda: ${CUDA_VISIBLE_DEVICES}"
