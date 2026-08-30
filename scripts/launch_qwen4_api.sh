#!/usr/bin/env bash
# Start a local vLLM OpenAI-compatible server for Qwen3-4B.
#
# Usage:
#   bash scripts/launch_qwen4_api.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8004 bash scripts/launch_qwen4_api.sh
#   MODEL_PATH=/path/to/Qwen3-4B bash scripts/launch_qwen4_api.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# Keep CUDA indices aligned with nvidia-smi on mixed-GPU hosts. GPU 0 is the
# RTX 3090; the 96 GB GPU 1 is reserved for the Qwen3-32B server.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen4b-api}"
PORT="${PORT:-8004}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PID_FILE="${PID_DIR}/qwen4_api.pid"
LOG_FILE="${LOG_DIR}/qwen4_api.log"
CHAT_TEMPLATE_FILE="${LOG_DIR}/qwen4_api_nothink.jinja"
BASE_URL="http://127.0.0.1:${PORT}/v1"

is_qwen4_server_pid() {
  local pid="$1"
  local cmdline
  cmdline="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
  [[ "${cmdline}" == *"vllm.entrypoints.openai.api_server"* ]] && [[ "${cmdline}" == *"${SERVED_MODEL_NAME}"* ]]
}

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    if is_qwen4_server_pid "${OLD_PID}"; then
      echo "qwen4 api process already running with pid ${OLD_PID}"
      echo "log: ${LOG_FILE}"
      echo "base url: ${BASE_URL}"
      if command -v curl >/dev/null 2>&1 && curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
        echo "health: ${BASE_URL}/models OK"
      else
        echo "health: ${BASE_URL}/models not ready; check log or wait for model loading"
      fi
      exit 0
    fi
    echo "stale pid file points to a different process: ${OLD_PID}"
  fi
  rm -f "${PID_FILE}"
fi

if command -v curl >/dev/null 2>&1 && curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "qwen4 api appears to be reachable at ${BASE_URL} without a pid file"
  echo "model id should be: ${SERVED_MODEL_NAME}"
  exit 0
fi

# Qwen3 enables thinking in its chat template by default.  vLLM 0.10.x only
# accepts enable_thinking per request, so create a server-wide template that
# fixes it to false while preserving the model's own template verbatim.
"${PYTHON_BIN}" - "${MODEL_PATH}" "${CHAT_TEMPLATE_FILE}" <<'PY'
import pathlib
import sys

from transformers import AutoTokenizer

model_path, output_path = sys.argv[1:]
template = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True
).chat_template
if not template:
    raise RuntimeError(f"No chat template found for {model_path}")
pathlib.Path(output_path).write_text(
    "{% set enable_thinking = false %}\n" + template,
    encoding="utf-8",
)
PY

nohup "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --chat-template "${CHAT_TEMPLATE_FILE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "started qwen4 api"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "base url: ${BASE_URL}"
echo "model id: ${SERVED_MODEL_NAME}"
echo "cuda: ${CUDA_VISIBLE_DEVICES}"
