#!/usr/bin/env bash
# Launch FEVER MemCo evaluation.
#
# Usage:
#   bash scripts/run_memco_qwen4b_fever.sh
#   MODEL=qwen32b-api OPENAI_API_BASE=http://127.0.0.1:8000/v1 bash scripts/run_memco_qwen4b_fever.sh
#   RUN_ID=my_run bash scripts/run_memco_qwen4b_fever.sh
#   MAX_TRAIN=10 MAX_EVAL=20 bash scripts/run_memco_qwen4b_fever.sh

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "/workspace/nvdamas" ]; then
  cd /workspace/nvdamas
else
  cd "${SCRIPT_ROOT}"
fi
ROOT="$(pwd)"

MODEL="${MODEL:-qwen4b-api}"

if [ -z "${OPENAI_API_BASE:-}" ]; then
  case "$MODEL" in
    qwen*|*qwen*)
      export OPENAI_API_BASE="${QWEN_OPENAI_API_BASE:-http://127.0.0.1:8004/v1}"
      ;;
    gpt*|o[0-9]*|o[0-9]-*)
      export OPENAI_API_BASE="https://api.openai.com/v1"
      ;;
    *)
      export OPENAI_API_BASE="https://api.anthropic.com/v1/"
      ;;
  esac
else
  export OPENAI_API_BASE
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [[ "$MODEL" == qwen* || "$MODEL" == *qwen* ]]; then
  export OPENAI_API_KEY="dummy"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
MAX_TRIALS="${MAX_TRIALS:-30}"
BATCH_SIZE="${BATCH_SIZE:-10}"
TOOL_MODE="${TOOL_MODE:-search}"
MEMCO_ROUTER="${MEMCO_ROUTER:-textloss}"
MEMCO_SETTINGS="${MEMCO_SETTINGS:-local_plus_global}"
MEMCO_PROMOTION_THRESHOLD="${MEMCO_PROMOTION_THRESHOLD:-0.35}"
MODEL_SAFE="$(printf '%s' "$MODEL" | tr '/:' '__')"
RUN_ID="${RUN_ID:-fever_memco_${MODEL_SAFE}_${TS}}"
LOG_FILE="${LOG_FILE:-${ROOT}/L_${RUN_ID}.log}"
PID_DIR="${PID_DIR:-${ROOT}/logs/memco_fever_pids}"
PID_FILE="${PID_DIR}/${RUN_ID}.pid"
mkdir -p "${PID_DIR}"

CMD=(
  "${PYTHON_BIN}" scripts/eval_collab_multidomain_global.py
  --dataset_family fever
  --fever_domains A_film_tv,B_music
  --fever_train_jsonl data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl
  --fever_test_jsonl data/fever/fever_ab_test_v3.jsonl
  --mas_type autogen
  --mas_memory memco
  --reasoning io
  --model "${MODEL}"
  --tool_mode "${TOOL_MODE}"
  --run_id "${RUN_ID}"
  --memco_dynamic_graph
  --memco_settings "${MEMCO_SETTINGS}"
  --memco_router "${MEMCO_ROUTER}"
  --memco_promotion_threshold "${MEMCO_PROMOTION_THRESHOLD}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --reset_memory
)

if [ -n "${MAX_TRAIN:-}" ]; then
  CMD+=(--max_train "${MAX_TRAIN}")
fi
if [ -n "${MAX_EVAL:-}" ]; then
  CMD+=(--max_eval "${MAX_EVAL}")
fi

echo "[fever] RUN_ID=${RUN_ID}"
echo "[fever] LOG=${LOG_FILE}"
echo "[fever] OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "[fever] MODEL=${MODEL}"
echo "[fever] ${CMD[*]}"

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_FILE}"
echo "[fever] PID=${PID} PID_FILE=${PID_FILE}"
echo "Tail with:"
echo "  tail -f ${LOG_FILE}"
