#!/usr/bin/env bash
# Stop the qwen32b-api vLLM server started by scripts/launch_qwen32_api.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/pids/qwen32_api.pid"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen32b-api}"

cmdline_for_pid() {
  local pid="$1"
  if [[ -r "/proc/${pid}/cmdline" ]]; then
    tr '\0' ' ' <"/proc/${pid}/cmdline"
    return 0
  fi
  if command -v ps >/dev/null 2>&1; then
    ps -p "${pid}" -o args= 2>/dev/null || true
  fi
}

find_vllm_pid() {
  local pid cmd
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    cmd="$(cmdline_for_pid "${pid}")"
    if [[ "${cmd}" == *"vllm.entrypoints.openai.api_server"* && "${cmd}" == *"${SERVED_MODEL_NAME}"* ]]; then
      echo "${pid}"
      return 0
    fi
  done
  return 1
}

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
else
  PID="$(find_vllm_pid || true)"
  if [[ -z "${PID}" ]]; then
    echo "qwen32 api pid file not found and no ${SERVED_MODEL_NAME} vLLM process found: ${PID_FILE}"
    exit 0
  fi
  echo "qwen32 api pid file not found; found ${SERVED_MODEL_NAME} pid ${PID} from /proc"
fi

CMDLINE="$(cmdline_for_pid "${PID}")"

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
