#!/usr/bin/env bash
# Launch PDDL, FEVER, and ALFWorld A-Mem evaluations as three independent
# background processes on CUDA 0 against the local OpenAI-compatible endpoint.
#
# Usage:
#   bash scripts/run_amem_qwen4b_cuda0_parallel.sh
#   MAX_TRAIN=10 MAX_EVAL=20 bash scripts/run_amem_qwen4b_cuda0_parallel.sh

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -d "/workspace/nvdamas" ]; then
  cd /workspace/nvdamas
else
  cd "${SCRIPT_ROOT}"
fi
ROOT="$(pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8004/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

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
MODEL="${MODEL:-qwen4b-api}"
MAX_TRIALS="${MAX_TRIALS:-30}"
BATCH_SIZE="${BATCH_SIZE:-10}"
PID_DIR="${PID_DIR:-${ROOT}/logs/amem_qwen4b_cuda0_parallel_pids}"
mkdir -p "${PID_DIR}"

COMMON_ARGS=(
  --mas_type autogen
  --mas_memory amem
  --reasoning io
  --model "${MODEL}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --tool_mode search
  --reset_memory
)

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
    "${PYTHON_BIN}" scripts/eval_collab_multidomain_global.py
    "$@"
    "${COMMON_ARGS[@]}"
    --run_id "${run_id}"
    "${LIMIT_ARGS[@]}"
  )

  echo "[$name] RUN_ID=${run_id}"
  echo "[$name] LOG=${log_file}"
  echo "[$name] ${cmd[*]}"
  nohup "${cmd[@]}" >"${log_file}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${pid_file}"
  echo "[$name] PID=${pid} PID_FILE=${pid_file}"
}

start_job "pddl" "pddl_amem_qwen4b_cuda0_${TS}" \
  --dataset_family pddl \
  --pddl_domains gripper,blockworld,barman,tyreworld \
  --pddl_train_jsonl data/pddl/pddl_domain_gripper.jsonl,data/pddl/pddl_domain_blockworld.jsonl,data/pddl/pddl_domain_barman.jsonl,data/pddl/pddl_domain_tyreworld.jsonl \
  --pddl_test_jsonl data/pddl/test.jsonl

start_job "fever" "fever_amem_qwen4b_cuda0_${TS}" \
  --dataset_family fever \
  --fever_domains A_film_tv,B_music \
  --fever_train_jsonl data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl \
  --fever_test_jsonl data/fever/fever_ab_test_v3.jsonl

start_job "alfworld" "alfworld_amem_qwen4b_cuda0_${TS}" \
  --dataset_family alfworld \
  --alfworld_domains bathroom,bedroom,kitchen,living \
  --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
  --alfworld_eval_split valid_seen,valid_unseen \
  --alfworld_game_root "${ALFWORLD_DATA:-${ROOT}/data/alfworld}/json_2.1.1"

echo "Started 3 background jobs."
echo "Logs:"
echo "  tail -f ${ROOT}/L_pddl_amem_qwen4b_cuda0_${TS}.log"
echo "  tail -f ${ROOT}/L_fever_amem_qwen4b_cuda0_${TS}.log"
echo "  tail -f ${ROOT}/L_alfworld_amem_qwen4b_cuda0_${TS}.log"
