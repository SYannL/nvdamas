#!/usr/bin/env bash
# Start a local vLLM OpenAI-compatible server for Qwen3-4B.
#
# Usage:
#   bash launch/launch_qwen4b_api.sh
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 bash launch/launch_qwen4b_api.sh
#   MODEL_PATH=/path/to/Qwen3-4B bash launch/launch_qwen4b_api.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/pids"
mkdir -p "${LOG_DIR}" "${PID_DIR}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/model/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen4b-api}"
PORT="${PORT:-8004}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PYTHON_BIN="${PYTHON_BIN:-/bigdata/siyan/envs/miniconda3/envs/nvdamas/bin/python}"

PID_FILE="${PID_DIR}/qwen4b_api.pid"
LOG_FILE="${LOG_DIR}/qwen4b_api.log"

if [[ ! -f "${MODEL_PATH}/config.json" ]] || ! compgen -G "${MODEL_PATH}/model*.safetensors" >/dev/null; then
  echo "Qwen3-4B model files not found at: ${MODEL_PATH}" >&2
  echo "Download with:" >&2
  echo "  /bigdata/siyan/envs/miniconda3/envs/nvdamas/bin/python -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-4B --local-dir ${MODEL_PATH} --cache-dir /bigdata/siyan/hf_cache" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    echo "qwen4b api already running with pid ${OLD_PID}"
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
echo "started qwen4b api"
echo "pid: $(cat "${PID_FILE}")"
echo "log: ${LOG_FILE}"
echo "base url: http://127.0.0.1:${PORT}/v1/chat/completions"
echo "model id: ${SERVED_MODEL_NAME}"
echo "model path: ${MODEL_PATH}"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES}"
echo "python: ${PYTHON_BIN}"
