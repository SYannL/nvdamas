#!/usr/bin/env bash
# Launch PDDL and FEVER MemSkill evaluations against a local qwen32b-api
# OpenAI-compatible endpoint.
#
# Usage:
#   bash scripts/medmcqa/run_memskill_qwen32b_api_fever_pddl.sh
#   MAX_TRAIN=10 MAX_EVAL=20 bash scripts/medmcqa/run_memskill_qwen32b_api_fever_pddl.sh
#   OPENAI_API_BASE=http://127.0.0.1:8004/v1 bash scripts/medmcqa/run_memskill_qwen32b_api_fever_pddl.sh
#   MEMSKILL_CHECKPOINT_PATH=Models/memskill/alfworld_controller.pt bash scripts/medmcqa/run_memskill_qwen32b_api_fever_pddl.sh

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "/workspace/nvdamas" ]; then
  cd /workspace/nvdamas
else
  cd "${SCRIPT_ROOT}"
fi
ROOT="$(pwd)"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${MEMSKILL_CUDA_VISIBLE_DEVICES:-1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

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
MEMSKILL_CONTROLLER="${MEMSKILL_CONTROLLER:-ppo}"
MEMSKILL_CHECKPOINT_PATH="${MEMSKILL_CHECKPOINT_PATH:-Models/memskill/alfworld_controller.pt}"
MEMSKILL_PPO_DEVICE="${MEMSKILL_PPO_DEVICE:-cuda:0}"
MEMSKILL_PPO_CONTROLLER_SOURCE="${MEMSKILL_PPO_CONTROLLER_SOURCE:-internal}"
MEMSKILL_REQUIRE_PPO="${MEMSKILL_REQUIRE_PPO:-1}"
MEMSKILL_USE_FLASH_ATTN="${MEMSKILL_USE_FLASH_ATTN:-0}"
PID_DIR="${PID_DIR:-${ROOT}/logs/memskill_qwen32b_api_fever_pddl_pids}"
mkdir -p "${PID_DIR}"

if command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "${OPENAI_API_BASE%/}/models" >/dev/null 2>&1; then
    echo "WARN: cannot reach ${OPENAI_API_BASE%/}/models"
    echo "      Start qwen32b-api first, or set OPENAI_API_BASE to the running server."
  fi
fi

if [ "${MEMSKILL_CONTROLLER}" = "ppo" ] && [ "${MEMSKILL_REQUIRE_PPO}" = "1" ] && [ ! -f "${MEMSKILL_CHECKPOINT_PATH}" ]; then
  echo "WARN: MEMSKILL_CHECKPOINT_PATH not found: ${MEMSKILL_CHECKPOINT_PATH}"
  echo "      This run will fail because MEMSKILL_REQUIRE_PPO=1."
fi

COMMON_ARGS=(
  --mas_type autogen
  --mas_memory memskill
  --reasoning io
  --model "${MODEL}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --tool_mode search
  --reset_memory
)

MEMSKILL_ARGS=(
  --memskill_finalize_local
  --memskill_controller "${MEMSKILL_CONTROLLER}"
  --memskill_checkpoint_path "${MEMSKILL_CHECKPOINT_PATH}"
  --memskill_ppo_device "${MEMSKILL_PPO_DEVICE}"
  --memskill_ppo_controller_source "${MEMSKILL_PPO_CONTROLLER_SOURCE}"
)
if [ "${MEMSKILL_USE_FLASH_ATTN}" = "1" ]; then
  MEMSKILL_ARGS+=(--memskill_use_flash_attn)
else
  MEMSKILL_ARGS+=(--no-memskill_use_flash_attn)
fi
if [ "${MEMSKILL_REQUIRE_PPO}" = "1" ]; then
  MEMSKILL_ARGS+=(--memskill_require_ppo)
fi

LIMIT_ARGS=()
if [ -n "${MAX_TRAIN:-}" ]; then
  LIMIT_ARGS+=(--max_train "${MAX_TRAIN}")
fi
if [ -n "${MAX_EVAL:-}" ]; then
  LIMIT_ARGS+=(--max_eval "${MAX_EVAL}")
fi

start_job() {
  local name="$1"
  local run_id="$2"
  local log_file="${ROOT}/L_${run_id}.log"
  local pid_file="${PID_DIR}/${run_id}.pid"
  shift 2

  local cmd=(
    "${PYTHON_BIN}" scripts/medmcqa/eval_collab_multidomain_global.py
    "$@"
    "${COMMON_ARGS[@]}"
    "${MEMSKILL_ARGS[@]}"
    --run_id "${run_id}"
    "${LIMIT_ARGS[@]}"
  )

  echo "[$name] RUN_ID=${run_id}"
  echo "[$name] LOG=${log_file}"
  echo "[$name] OPENAI_API_BASE=${OPENAI_API_BASE}"
  echo "[$name] CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}"
  echo "[$name] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[$name] ${cmd[*]}"
  nohup "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${pid_file}"
  echo "[$name] PID=${pid} PID_FILE=${pid_file}"
}

start_job "pddl" "pddl_memskill_qwen32b_api_${TS}" \
  --dataset_family pddl \
  --pddl_domains gripper,blockworld,barman,tyreworld \
  --pddl_train_jsonl data/pddl/pddl_domain_gripper.jsonl,data/pddl/pddl_domain_blockworld.jsonl,data/pddl/pddl_domain_barman.jsonl,data/pddl/pddl_domain_tyreworld.jsonl \
  --pddl_test_jsonl data/pddl/test.jsonl

start_job "fever" "fever_memskill_qwen32b_api_${TS}" \
  --dataset_family fever \
  --fever_domains A_film_tv,B_music \
  --fever_train_jsonl data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl \
  --fever_test_jsonl data/fever/fever_ab_test_v3.jsonl

echo "Started 2 background jobs."
echo "Logs:"
echo "  tail -f ${ROOT}/L_pddl_memskill_qwen32b_api_${TS}.log"
echo "  tail -f ${ROOT}/L_fever_memskill_qwen32b_api_${TS}.log"
