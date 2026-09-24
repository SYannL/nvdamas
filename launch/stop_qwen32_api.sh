#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="${ROOT_DIR}/pids/qwen32_api.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "qwen32 api pid file not found: ${PID_FILE}"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" >/dev/null 2>&1; then
  kill "${PID}"
  echo "stopped qwen32 api pid ${PID}"
else
  echo "process ${PID} is not running"
fi

rm -f "${PID_FILE}"
