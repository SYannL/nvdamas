#!/usr/bin/env bash
# Launch FEVER MemRL evaluation against a local qwen32b-api OpenAI-compatible endpoint.
#
# Usage:
#   bash scripts/medmcqa/run_memrl_qwen32b_api_fever.sh
#   MAX_TRAIN=10 MAX_EVAL=20 bash scripts/medmcqa/run_memrl_qwen32b_api_fever.sh
#   OPENAI_API_BASE=http://127.0.0.1:8000/v1 bash scripts/medmcqa/run_memrl_qwen32b_api_fever.sh

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "/workspace/nvdamas" ]; then
  cd /workspace/nvdamas
else
  cd "${SCRIPT_ROOT}"
fi
ROOT="$(pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# MemRL uses a separate embedding provider.
export OPENAI_EMBEDDING_API_BASE="${OPENAI_EMBEDDING_API_BASE:-http://127.0.0.1:8001/v1}"
export OPENAI_EMBEDDING_API_KEY="${OPENAI_EMBEDDING_API_KEY:-${OPENAI_API_KEY}}"
export MEMRL_EMBEDDING_MODEL="${MEMRL_EMBEDDING_MODEL:-qwen3-embedding-api}"

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
MODEL="${MODEL:-qwen32b-api}"
MAX_TRIALS="${MAX_TRIALS:-30}"
BATCH_SIZE="${BATCH_SIZE:-10}"
RUN_ID="${RUN_ID:-fever_memrl_qwen32b_api_${TS}}"
LOG_FILE="${LOG_FILE:-${ROOT}/L_${RUN_ID}.log}"
PID_DIR="${PID_DIR:-${ROOT}/logs/memrl_qwen32b_api_fever_pids}"
PID_FILE="${PID_DIR}/${RUN_ID}.pid"
mkdir -p "${PID_DIR}"

if command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "${OPENAI_API_BASE%/}/models" >/dev/null 2>&1; then
    echo "WARN: cannot reach ${OPENAI_API_BASE%/}/models"
    echo "      Start qwen32b-api first, or set OPENAI_API_BASE to the running server."
  fi
fi

CMD=(
  "${PYTHON_BIN}" scripts/medmcqa/eval_collab_multidomain_global.py
  --dataset_family fever
  --fever_domains A_film_tv,B_music
  --fever_train_jsonl data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl
  --fever_test_jsonl data/fever/fever_ab_test_v3.jsonl
  --mas_type autogen
  --mas_memory memrl
  --reasoning io
  --model "${MODEL}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --tool_mode search
  --reset_memory
  --run_id "${RUN_ID}"
)

if [ -n "${MAX_TRAIN:-}" ]; then
  CMD+=(--max_train "${MAX_TRAIN}")
fi
if [ -n "${MAX_EVAL:-}" ]; then
  CMD+=(--max_eval "${MAX_EVAL}")
fi

echo "[fever] RUN_ID=${RUN_ID}"
echo "[fever] LOG=${LOG_FILE}"
echo "[fever] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[fever] OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "[fever] OPENAI_EMBEDDING_API_BASE=${OPENAI_EMBEDDING_API_BASE}"
echo "[fever] MEMRL_EMBEDDING_MODEL=${MEMRL_EMBEDDING_MODEL}"
echo "[fever] ${CMD[*]}"

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_FILE}"
echo "[fever] PID=${PID} PID_FILE=${PID_FILE}"
echo "Tail with:"
echo "  tail -f ${LOG_FILE}"
