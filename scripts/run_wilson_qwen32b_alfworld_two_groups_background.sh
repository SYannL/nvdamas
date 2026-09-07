#!/usr/bin/env bash
# Start the two Wilson/Qwen32B/ALFWorld all-ones experiments as one detached,
# sequential background job. The worker waits for the API before group 1 and
# starts group 2 only after group 1 exits successfully.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
RUN_TIMEZONE="${RUN_TIMEZONE:-America/Vancouver}"
API_WAIT_SECONDS="${API_WAIT_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-10}"

if [[ "${NV_MEMCO_TWO_GROUP_WORKER:-0}" != "1" ]]; then
  BATCH_TS="${BATCH_TS:-$(TZ="${RUN_TIMEZONE}" date +%Y%m%d_%H%M%S)}"
  BATCH_ID="${BATCH_ID:-wilson_qwen32b_alfworld_two_groups_${BATCH_TS}}"
  CONTROL_DIR="${ROOT}/logs/memco_wilson_two_groups/${BATCH_ID}"
  mkdir -p "${CONTROL_DIR}"

  MASTER_LOG="${CONTROL_DIR}/controller.log"
  PID_FILE="${CONTROL_DIR}/controller.pid"
  STATUS_FILE="${CONTROL_DIR}/status.txt"

  nohup env \
    NV_MEMCO_TWO_GROUP_WORKER=1 \
    BATCH_TS="${BATCH_TS}" \
    BATCH_ID="${BATCH_ID}" \
    CONTROL_DIR="${CONTROL_DIR}" \
    OPENAI_API_BASE="${OPENAI_API_BASE}" \
    OPENAI_API_KEY="${OPENAI_API_KEY}" \
    RUN_TIMEZONE="${RUN_TIMEZONE}" \
    API_WAIT_SECONDS="${API_WAIT_SECONDS}" \
    POLL_SECONDS="${POLL_SECONDS}" \
    bash "${BASH_SOURCE[0]}" \
    >"${MASTER_LOG}" 2>&1 </dev/null &
  CONTROLLER_PID="$!"
  echo "${CONTROLLER_PID}" >"${PID_FILE}"
  printf 'STARTED pid=%s batch=%s\n' "${CONTROLLER_PID}" "${BATCH_ID}" >"${STATUS_FILE}"

  echo "[two-groups] controller_pid=${CONTROLLER_PID}"
  echo "[two-groups] batch_id=${BATCH_ID}"
  echo "[two-groups] status=${STATUS_FILE}"
  echo "[two-groups] controller_log=${MASTER_LOG}"
  exit 0
fi

: "${BATCH_TS:?worker requires BATCH_TS}"
: "${BATCH_ID:?worker requires BATCH_ID}"
: "${CONTROL_DIR:?worker requires CONTROL_DIR}"

STATUS_FILE="${CONTROL_DIR}/status.txt"
GROUP1_RUN_ID="alfworld_memco_wilson_qwen32b_eq22_27_allones_${BATCH_TS}"
GROUP2_RUN_ID="alfworld_memco_wilson_qwen32b_eq22_27_a3_2_activation_allones_${BATCH_TS}"
GROUP1_LOG="${CONTROL_DIR}/group1_eq22_27_allones.log"
GROUP2_LOG="${CONTROL_DIR}/group2_eq22_27_a3_2_activation_allones.log"

on_exit() {
  exit_code="$?"
  if [[ "${exit_code}" -ne 0 ]]; then
    printf 'FAILED exit_code=%s batch=%s\n' "${exit_code}" "${BATCH_ID}" >"${STATUS_FILE}"
  fi
}
trap on_exit EXIT

printf 'WAITING_FOR_API batch=%s api=%s\n' "${BATCH_ID}" "${OPENAI_API_BASE}" >"${STATUS_FILE}"
waited=0
until curl -fsS "${OPENAI_API_BASE}/models" >/dev/null 2>&1; do
  if (( waited >= API_WAIT_SECONDS )); then
    echo "[two-groups] API did not become ready within ${API_WAIT_SECONDS}s: ${OPENAI_API_BASE}" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
  waited=$((waited + POLL_SECONDS))
done

printf 'RUNNING_GROUP1 run_id=%s\n' "${GROUP1_RUN_ID}" >"${STATUS_FILE}"
echo "[two-groups] starting group 1: ${GROUP1_RUN_ID}"
FOREGROUND=1 \
RUN_ID="${GROUP1_RUN_ID}" \
bash scripts/run_memco_wilson_qwen32b_alfworld_allones.sh \
  >"${GROUP1_LOG}" 2>&1

printf 'RUNNING_GROUP2 run_id=%s\n' "${GROUP2_RUN_ID}" >"${STATUS_FILE}"
echo "[two-groups] group 1 complete; starting group 2: ${GROUP2_RUN_ID}"
FOREGROUND=1 \
RUN_ID="${GROUP2_RUN_ID}" \
bash scripts/run_memco_wilson_qwen32b_alfworld_allones_a3_2_activation.sh \
  >"${GROUP2_LOG}" 2>&1

printf 'COMPLETE group1=%s group2=%s\n' "${GROUP1_RUN_ID}" "${GROUP2_RUN_ID}" >"${STATUS_FILE}"
echo "[two-groups] both groups complete"
