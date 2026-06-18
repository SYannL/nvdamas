#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen32b-api}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PYTHON_BIN="${PYTHON_BIN:-/bigdata/siyan/envs/miniconda3/envs/nvdamas/bin/python}"

PID_FILE="${PID_DIR}/qwen32_api.pid"
LOG_FILE="${LOG_DIR}/qwen32_api.log"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    echo "qwen32 api already running with pid ${OLD_PID}"
    echo "log: ${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

nohup "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "started qwen32 api"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "base url: http://127.0.0.1:${PORT}/v1/chat/completions"
echo "model id: ${SERVED_MODEL_NAME}"
echo "python: ${PYTHON_BIN}"
