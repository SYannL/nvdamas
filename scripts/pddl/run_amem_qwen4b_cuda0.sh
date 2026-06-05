#!/usr/bin/env bash
# PDDL multi-domain collab eval with A-Mem on qwen4b-api via local OpenAI-compatible server.
#
# Usage:
#   bash scripts/pddl/run_amem_qwen4b_cuda0.sh
#   RUN_ID=my_run MAX_TRAIN=5 MAX_EVAL=10 bash scripts/pddl/run_amem_qwen4b_cuda0.sh
#   FOREGROUND=1 bash scripts/pddl/run_amem_qwen4b_cuda0.sh

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "/workspace/nvdamas" ]; then
  cd /workspace/nvdamas
else
  cd "${SCRIPT_ROOT}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8004/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-pddl_amem_qwen4b_cuda0_${TS}}"
MODEL="${MODEL:-qwen4b-api}"
MAX_TRIALS="${MAX_TRIALS:-30}"
BATCH_SIZE="${BATCH_SIZE:-10}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOG_DIR="${LOG_DIR:-logs/pddl_amem_qwen4b_cuda0}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_ID}.pid}"

if [ -z "${PYTHON_BIN}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

CMD=(
  "${PYTHON_BIN}" scripts/eval_collab_multidomain_global.py
  --dataset_family pddl
  --pddl_domains gripper,blockworld,barman,tyreworld
  --pddl_train_jsonl data/pddl/pddl_domain_gripper.jsonl,data/pddl/pddl_domain_blockworld.jsonl,data/pddl/pddl_domain_barman.jsonl,data/pddl/pddl_domain_tyreworld.jsonl
  --pddl_test_jsonl data/pddl/test.jsonl
  --mas_type autogen
  --mas_memory amem
  --reasoning io
  --model "${MODEL}"
  --run_id "${RUN_ID}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --tool_mode search
  --reset_memory
)

if [ -n "${MAX_TRAIN:-}" ]; then
  CMD+=(--max_train "${MAX_TRAIN}")
fi

if [ -n "${MAX_EVAL:-}" ]; then
  CMD+=(--max_eval "${MAX_EVAL}")
fi

mkdir -p "${LOG_DIR}"

echo "RUN_ID=${RUN_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "${CMD[@]}"

if [ "${FOREGROUND:-0}" = "1" ]; then
  exec "${CMD[@]}" >"${LOG_FILE}" 2>&1
fi

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_FILE}"
echo "Started background job PID=${PID}"
echo "Log: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"
echo "Tail with: tail -f ${LOG_FILE}"
