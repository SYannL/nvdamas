#!/usr/bin/env bash
# Stop the qwen32b-api vLLM server started by scripts/launch_qwen32_api.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/pids/qwen32_api.pid"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen32b-api}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "qwen32 api pid file not found: ${PID_FILE}"
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
echo "stopped qwen32 api pid ${PID}"
