#!/usr/bin/env bash
# Run the Wilson promotion branch with the local Qwen3-4B API.
#
# Supported adapter-backed datasets:
#   alfworld, fever, pddl, pddl_2, scienceworld, bfcl_mt
#
# Dataset defaults preserve the existing MemCo/data protocol.  Wilson only
# replaces the promotion decision.
#
# Examples:
#   DATASET=alfworld bash scripts/run_memco_wilson_qwen4b.sh
#   DATASET=fever MAX_TRAIN=5 MAX_EVAL=10 bash scripts/run_memco_wilson_qwen4b.sh
#   DATASET=pddl PROMOTION_POLICY=shadow bash scripts/run_memco_wilson_qwen4b.sh
#   DATASET=scienceworld FOREGROUND=1 bash scripts/run_memco_wilson_qwen4b.sh
#   DATASET=bfcl_mt MAX_TRIALS=96 bash scripts/run_memco_wilson_qwen4b.sh  # optional long budget

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /workspace/nvdamas ]]; then
  cd /workspace/nvdamas
else
  cd "${SCRIPT_ROOT}"
fi
ROOT="$(pwd)"

DATASET="${DATASET:-alfworld}"
MODEL="${MODEL:-qwen4b-api}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8004/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
# Qwen inference is served by the already-running API.  Episode workers only
# need a small sentence encoder; keep it on CPU so isolated workers cannot
# exhaust the model GPU.  Other launchers retain EmbeddingFunc's auto behavior.
export NV_DAMAS_EMBEDDING_DEVICE="${NV_DAMAS_EMBEDDING_DEVICE:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PROMOTION_POLICY="${PROMOTION_POLICY:-wilson}"
WILSON_ALPHA="${WILSON_ALPHA:-0.05}"
WILSON_THRESHOLD="${WILSON_THRESHOLD:-0.5}"
MEMCO_SETTINGS="${MEMCO_SETTINGS:-local_plus_global}"
MEMCO_ROUTER="${MEMCO_ROUTER:-textloss}"
MAX_TRIALS="${MAX_TRIALS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
FOREGROUND="${FOREGROUND:-0}"
DRY_RUN="${DRY_RUN:-0}"

case "${PROMOTION_POLICY}" in
  legacy|shadow|wilson) ;;
  *)
    echo "PROMOTION_POLICY must be legacy, shadow, or wilson: ${PROMOTION_POLICY}" >&2
    exit 2
    ;;
esac

case "${DATASET}" in
  alfworld|fever|pddl|pddl_2|scienceworld|bfcl_mt) ;;
  *)
    echo "Unsupported DATASET=${DATASET}. Choose alfworld, fever, pddl, pddl_2, scienceworld, or bfcl_mt." >&2
    echo "AmaBench is not listed because the current MemCo backend has no AmaBench-specific adapter." >&2
    exit 2
    ;;
esac

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  curl -fsS "${OPENAI_API_BASE}/models" >/dev/null || {
    echo "Qwen API is not ready at ${OPENAI_API_BASE}." >&2
    echo "Start it on the RTX 6000 with: CUDA_VISIBLE_DEVICES=1 GPU_MEM_UTIL=0.80 bash scripts/launch_qwen4b_api.sh" >&2
    exit 1
  }
fi

RUN_TIMEZONE="${RUN_TIMEZONE:-America/Vancouver}"
RUN_TS="${RUN_TS:-$(TZ="${RUN_TIMEZONE}" date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${DATASET}_memco_wilson_qwen4b_${RUN_TS}}"
LOG_FILE="${LOG_FILE:-${ROOT}/L_${RUN_ID}.log}"
PID_DIR="${PID_DIR:-${ROOT}/logs/memco_wilson_pids}"
PID_FILE="${PID_DIR}/${RUN_ID}.pid"
mkdir -p "${PID_DIR}"

DATASET_ARGS=()
case "${DATASET}" in
  alfworld)
    MAX_TRIALS="${MAX_TRIALS:-30}"
    # Matches scripts/run_memco_qwen4b_alfworld.sh.
    BATCH_SIZE="${BATCH_SIZE:-5}"
    DATASET_ARGS=(
      --alfworld_domains "${ALFWORLD_DOMAINS:-bathroom,bedroom,kitchen,living}"
      --alfworld_subset_dir "${ALFWORLD_SUBSET_DIR:-data/alfworld/collab_subsets/v3_s}"
      --alfworld_eval_split "${ALFWORLD_EVAL_SPLIT:-valid_seen,valid_unseen}"
      --alfworld_game_root "${ALFWORLD_GAME_ROOT:-${ROOT}/data/alfworld/json_2.1.1}"
    )
    ;;
  fever)
    MAX_TRIALS="${MAX_TRIALS:-30}"
    # Matches scripts/run_memco_qwen4b_fever.sh.
    BATCH_SIZE="${BATCH_SIZE:-10}"
    DATASET_ARGS=(
      --fever_domains "${FEVER_DOMAINS:-A_film_tv,B_music}"
      --fever_train_jsonl "${FEVER_TRAIN_JSONL:-data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl}"
      --fever_test_jsonl "${FEVER_TEST_JSONL:-data/fever/fever_ab_test_v3.jsonl}"
      --tool_mode "${TOOL_MODE:-search}"
    )
    ;;
  pddl|pddl_2)
    MAX_TRIALS="${MAX_TRIALS:-30}"
    # Matches the completed Qwen4B PDDL baseline report.  The current
    # standalone launcher defaults to 1; pass BATCH_SIZE=1 to reproduce it.
    # pddl_2 uses the same tasks and budget but reports partial progress too.
    BATCH_SIZE="${BATCH_SIZE:-10}"
    DATASET_ARGS=(
      --pddl_domains "${PDDL_DOMAINS:-gripper,blockworld,barman,tyreworld}"
      --pddl_train_jsonl "${PDDL_TRAIN_JSONL:-data/pddl/pddl_domain_gripper.jsonl,data/pddl/pddl_domain_blockworld.jsonl,data/pddl/pddl_domain_barman.jsonl,data/pddl/pddl_domain_tyreworld.jsonl}"
      --pddl_test_jsonl "${PDDL_TEST_JSONL:-data/pddl/test.jsonl}"
    )
    ;;
  scienceworld)
    # Matches the completed ScienceWorld collaborative protocol.
    MAX_TRIALS="${MAX_TRIALS:-50}"
    BATCH_SIZE="${BATCH_SIZE:-10}"
    DATASET_ARGS=(
      --scienceworld_domains "${SCIENCEWORLD_DOMAINS:-conductivity,friction,genetics,melting_point}"
      --scienceworld_subset_dir "${SCIENCEWORLD_SUBSET_DIR:-data/ScienceWorld/collab_subsets/v3_family_mixed}"
      --scienceworld_test_json "${SCIENCEWORLD_TEST_JSON:-data/ScienceWorld/collab_subsets/v3_family_mixed/merged__test.json}"
    )
    ;;
  bfcl_mt)
    # Matches the established four-family BFCL protocol.  Override
    # MAX_TRIALS=96 only for the optional long-budget setting.
    MAX_TRIALS="${MAX_TRIALS:-30}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    DATASET_ARGS=(
      --bfcl_use_family_collab_split
      --bfcl_family_domains "${BFCL_FAMILY_DOMAINS:-trading,travel,vehicle,fs}"
      --bfcl_train_limit_per_domain "${BFCL_TRAIN_LIMIT_PER_DOMAIN:-5}"
      --tool_mode "${TOOL_MODE:-search}"
    )
    ;;
esac

COMMON=(
  "${PYTHON_BIN}" scripts/eval_collab_multidomain_global.py
  --dataset_family "${DATASET}"
  --mas_type autogen
  --mas_memory memco
  --reasoning io
  --model "${MODEL}"
  --run_id "${RUN_ID}"
  --memco_dynamic_graph
  --memco_settings "${MEMCO_SETTINGS}"
  --memco_router "${MEMCO_ROUTER}"
  --memco_promotion_policy "${PROMOTION_POLICY}"
  --memco_wilson_alpha "${WILSON_ALPHA}"
  --memco_wilson_threshold "${WILSON_THRESHOLD}"
  --max_trials "${MAX_TRIALS}"
  --reset_memory
)

CMD=("${COMMON[@]}" "${DATASET_ARGS[@]}" --batch_size "${BATCH_SIZE}")
if [[ -n "${MAX_TRAIN:-}" ]]; then
  CMD+=(--max_train "${MAX_TRAIN}")
fi
if [[ -n "${MAX_EVAL:-}" ]]; then
  CMD+=(--max_eval "${MAX_EVAL}")
fi

printf '[wilson] RUN_TS=%s TZ=%s DATASET=%s POLICY=%s RUN_ID=%s\n' \
  "${RUN_TS}" "${RUN_TIMEZONE}" "${DATASET}" "${PROMOTION_POLICY}" "${RUN_ID}"
printf '[wilson] API=%s MODEL=%s\n' "${OPENAI_API_BASE}" "${MODEL}"
printf '[wilson] eval_cuda=%s embedding_device=%s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${NV_DAMAS_EMBEDDING_DEVICE}"
printf '[wilson] max_trials=%s batch_size=%s eval=local_plus_global_per_domain\n' \
  "${MAX_TRIALS}" "${BATCH_SIZE}"
printf '[wilson] alpha=%s threshold=%s (source coverage is diagnostic only)\n' \
  "${WILSON_ALPHA}" "${WILSON_THRESHOLD}"
printf '[wilson] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi
if [[ "${FOREGROUND}" == "1" ]]; then
  exec "${CMD[@]}"
fi

nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" >"${PID_FILE}"
echo "[wilson] PID=${PID}"
echo "[wilson] log=${LOG_FILE}"
echo "[wilson] tail -f ${LOG_FILE}"
