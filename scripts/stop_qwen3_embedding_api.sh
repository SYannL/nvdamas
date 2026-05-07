#!/usr/bin/env bash
# Stop the embedding server started by scripts/launch_qwen3_embedding_api.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/pids/qwen3_embedding_api.pid"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-embedding-api}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "qwen3 embedding api pid file not found: ${PID_FILE}"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
CMDLINE="$(ps -p "${PID}" -o args= 2>/dev/null || true)"

if [[ -z "${CMDLINE}" ]]; then
  echo "process ${PID} is not running"
  rm -f "${PID_FILE}"
  exit 0
fi

if [[ "${CMDLINE}" != *"vllm.entrypoints.openai.api_server"* || "${CMDLINE}" != *"${SERVED_MODEL_NAME}"* ]]; then
  echo "pid ${PID} is not the ${SERVED_MODEL_NAME} vLLM server; removing stale pid file only"
  rm -f "${PID_FILE}"
  exit 0
fi

kill "${PID}"
rm -f "${PID_FILE}"
echo "stopped qwen3 embedding api pid ${PID}"
