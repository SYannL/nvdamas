#!/usr/bin/env bash
# Launch PDDL GraphMemory3 evaluation with gpt-4o-mini.
#
# Usage:
#   OPENAI_API_KEY=... bash scripts/medmcqa/run_gm3_gpt4omini_pddl.sh
#   RUN_ID=my_run OPENAI_API_KEY=... bash scripts/medmcqa/run_gm3_gpt4omini_pddl.sh
#   MAX_TRAIN=10 MAX_EVAL=20 OPENAI_API_KEY=... bash scripts/medmcqa/run_gm3_gpt4omini_pddl.sh

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "/workspace/nvdamas" ]; then
  cd /workspace/nvdamas
else
  cd "${SCRIPT_ROOT}"
fi
ROOT="$(pwd)"

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

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
MODEL="${MODEL:-gpt-4o-mini}"
MAX_TRIALS="${MAX_TRIALS:-30}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GM3_ROUTER="${GM3_ROUTER:-textloss}"
GM3_SETTINGS="${GM3_SETTINGS:-local_plus_global}"
GM3_PROMOTION_THRESHOLD="${GM3_PROMOTION_THRESHOLD:-0.35}"
RUN_ID="${RUN_ID:-pddl_gm3_gpt4omini_${TS}}"
LOG_FILE="${LOG_FILE:-${ROOT}/L_${RUN_ID}.log}"
PID_DIR="${PID_DIR:-${ROOT}/logs/gm3_gpt4omini_pddl_pids}"
PID_FILE="${PID_DIR}/${RUN_ID}.pid"
mkdir -p "${PID_DIR}"

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "WARN: OPENAI_API_KEY is empty. Set it before running gpt-4o-mini."
fi

CMD=(
  "${PYTHON_BIN}" scripts/medmcqa/eval_collab_multidomain_global.py
  --dataset_family pddl
  --pddl_domains gripper,blockworld,barman,tyreworld
  --pddl_train_jsonl data/pddl/pddl_domain_gripper.jsonl,data/pddl/pddl_domain_blockworld.jsonl,data/pddl/pddl_domain_barman.jsonl,data/pddl/pddl_domain_tyreworld.jsonl
  --pddl_test_jsonl data/pddl/test.jsonl
  --mas_type autogen
  --mas_memory graph_memory3
  --reasoning io
  --model "${MODEL}"
  --run_id "${RUN_ID}"
  --gm3_dynamic_graph
  --gm3_settings "${GM3_SETTINGS}"
  --gm3_router "${GM3_ROUTER}"
  --gm3_promotion_threshold "${GM3_PROMOTION_THRESHOLD}"
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

echo "[pddl] RUN_ID=${RUN_ID}"
echo "[pddl] LOG=${LOG_FILE}"
echo "[pddl] OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "[pddl] MODEL=${MODEL}"
echo "[pddl] ${CMD[*]}"

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_FILE}"
echo "[pddl] PID=${PID} PID_FILE=${PID_FILE}"
echo "Tail with:"
echo "  tail -f ${LOG_FILE}"
