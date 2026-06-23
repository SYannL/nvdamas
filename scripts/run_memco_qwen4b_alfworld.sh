#!/usr/bin/env bash
# Launch ALFWorld MemCo evaluation with qwen4b.
#
# Usage:
#   bash scripts/run_memco_qwen4b_alfworld.sh
#   MODEL=qwen32b-api QWEN_OPENAI_API_BASE=http://127.0.0.1:8000/v1 bash scripts/run_memco_qwen4b_alfworld.sh
#   MAX_TRAIN=5 MAX_EVAL=10 bash scripts/run_memco_qwen4b_alfworld.sh

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
BATCH_SIZE="${BATCH_SIZE:-5}"
ALFWORLD_DOMAINS="${ALFWORLD_DOMAINS:-bathroom,bedroom,kitchen,living}"
ALFWORLD_SUBSET_DIR="${ALFWORLD_SUBSET_DIR:-data/alfworld/collab_subsets/v3_s}"
ALFWORLD_EVAL_SPLIT="${ALFWORLD_EVAL_SPLIT:-valid_seen,valid_unseen}"
ALFWORLD_GAME_ROOT="${ALFWORLD_GAME_ROOT:-${ROOT}/data/alfworld/json_2.1.1}"
MEMCO_ROUTER="${MEMCO_ROUTER:-textloss}"
MEMCO_SETTINGS="${MEMCO_SETTINGS:-local_plus_global}"
MEMCO_PROMOTION_THRESHOLD="${MEMCO_PROMOTION_THRESHOLD:-0.35}"
MODEL_SAFE="$(printf '%s' "$MODEL" | tr '/:' '__')"
RUN_ID="${RUN_ID:-alfworld_memco_${MODEL_SAFE}_${TS}}"
LOG_FILE="${LOG_FILE:-${ROOT}/L_${RUN_ID}.log}"
PID_DIR="${PID_DIR:-${ROOT}/logs/memco_alfworld_pids}"
PID_FILE="${PID_DIR}/${RUN_ID}.pid"
mkdir -p "${PID_DIR}"

CMD=(
  "${PYTHON_BIN}" scripts/eval_collab_multidomain_global.py
  --dataset_family alfworld
  --alfworld_domains "${ALFWORLD_DOMAINS}"
  --alfworld_subset_dir "${ALFWORLD_SUBSET_DIR}"
  --alfworld_eval_split "${ALFWORLD_EVAL_SPLIT}"
  --alfworld_game_root "${ALFWORLD_GAME_ROOT}"
  --mas_type autogen
  --mas_memory memco
  --reasoning io
  --model "${MODEL}"
  --run_id "${RUN_ID}"
  --memco_dynamic_graph
  --memco_router "${MEMCO_ROUTER}"
  --memco_settings "${MEMCO_SETTINGS}"
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

echo "[alfworld] RUN_ID=${RUN_ID}"
echo "[alfworld] LOG=${LOG_FILE}"
echo "[alfworld] OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "[alfworld] MODEL=${MODEL}"
echo "[alfworld] SUBSET=${ALFWORLD_SUBSET_DIR}"
echo "[alfworld] ${CMD[*]}"

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_FILE}"
echo "[alfworld] PID=${PID} PID_FILE=${PID_FILE}"
echo "Tail with:"
echo "  tail -f ${LOG_FILE}"
