#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

EXPERIMENT="${1:-}"
if [[ -z "${EXPERIMENT}" ]]; then
  echo "Usage: SOURCE_RUN_ID=<main-run> bash scripts/ablations/run_alfworld_frozen.sh <experiment>" >&2
  echo "Experiments: adaptive_reuse, local_only, global_only, empirical_rate, positive_only, fixed_l3g3, fixed_l5g5, lambda025, lambda050" >&2
  exit 2
fi
: "${SOURCE_RUN_ID:?Set SOURCE_RUN_ID to a completed main ALFWorld MemCo run}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-qwen32b-api}"
MODEL_DB_NAME="${MODEL_DB_NAME:-${MODEL}}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_TRIALS="${MAX_TRIALS:-30}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-alfworld_${EXPERIMENT}_seed${SEED}_${RUN_TS}}"
DRY_RUN="${DRY_RUN:-0}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export NV_DAMAS_EMBEDDING_DEVICE="${NV_DAMAS_EMBEDDING_DEVICE:-cpu}"
export NV_MEMCO_PAPER_EQ22_27_ALL_ONES=1
export NV_MEMCO_PAPER_A3_2_ACTIVATION_ALL_ONES=0
export PYTHONHASHSEED="${SEED}"
export WILSON_ABLATION_SOURCE_RUN_ID="${SOURCE_RUN_ID}"

SOURCE_MEMORY="${ROOT}/.db/${MODEL_DB_NAME}/alfworld_collab_eval/${SOURCE_RUN_ID}/autogen/memory/memco"
TARGET_MEMORY="${ROOT}/.db/${MODEL_DB_NAME}/alfworld_collab_eval/${RUN_ID}/autogen/memory/memco"

stage_memory() {
  local domain
  for domain in bathroom bedroom kitchen living; do
    test -s "${SOURCE_MEMORY}/local/${domain}/memco/local_${domain}.json" || {
      echo "Missing source local memory for ${domain}: ${SOURCE_MEMORY}" >&2
      return 1
    }
  done
  test -s "${SOURCE_MEMORY}/global/memco/global_memory.json" || {
    echo "Missing source global memory: ${SOURCE_MEMORY}" >&2
    return 1
  }
  if [[ -e "${TARGET_MEMORY}" ]]; then
    echo "Refusing to overwrite existing target: ${TARGET_MEMORY}" >&2
    return 1
  fi
  for domain in bathroom bedroom kitchen living; do
    mkdir -p "${TARGET_MEMORY}/local/${domain}/memco"
    cp -a "${SOURCE_MEMORY}/local/${domain}/memco/local_${domain}.json" \
      "${TARGET_MEMORY}/local/${domain}/memco/"
  done
  mkdir -p "${TARGET_MEMORY}/global/memco"
  cp -a "${SOURCE_MEMORY}/global/memco/." "${TARGET_MEMORY}/global/memco/"
  printf '%s\n' "${SOURCE_RUN_ID}" >"${TARGET_MEMORY}/FROZEN_SOURCE_RUN_ID"
}

POLICY=reuse
SETTING=local_plus_global
THRESHOLD=0.35
export NV_MEMCO_FIXED_TOPK_USE_ALL=0

case "${EXPERIMENT}" in
  adaptive_reuse) ;;
  local_only) SETTING=local_only ;;
  global_only) SETTING=global_only ;;
  empirical_rate) POLICY=empirical_rate ;;
  positive_only) POLICY=positive_only ;;
  fixed_l3g3)
    export NV_MEMCO_FIXED_TOPK_USE_ALL=1
    export NV_MEMCO_FIXED_LOCAL_TOP_K=3
    export NV_MEMCO_FIXED_GLOBAL_TOP_K=3
    ;;
  fixed_l5g5)
    export NV_MEMCO_FIXED_TOPK_USE_ALL=1
    export NV_MEMCO_FIXED_LOCAL_TOP_K=5
    export NV_MEMCO_FIXED_GLOBAL_TOP_K=5
    ;;
  lambda025) POLICY=wilson; THRESHOLD=0.25 ;;
  lambda050) POLICY=wilson; THRESHOLD=0.50 ;;
  *)
    echo "Unsupported experiment: ${EXPERIMENT}" >&2
    exit 2
    ;;
esac

CMD=(
  "${PYTHON_BIN}" -u scripts/ablations/frozen_memory_eval.py
  --ablation-promotion-policy "${POLICY}"
  --dataset_family alfworld
  --alfworld_domains bathroom,bedroom,kitchen,living
  --alfworld_subset_dir data/alfworld/collab_subsets/v3_s
  --alfworld_eval_split valid_seen,valid_unseen
  --alfworld_game_root "${ALFWORLD_GAME_ROOT:-${ROOT}/data/alfworld/json_2.1.1}"
  --mas_type autogen
  --mas_memory memco
  --reasoning io
  --model "${MODEL}"
  --run_id "${RUN_ID}"
  --seed "${SEED}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --tool_mode search
  --eval_only
  --memco_dynamic_graph
  --memco_settings "${SETTING}"
  --memco_router textloss
  --memco_promotion_policy wilson
  --memco_wilson_alpha 0.05
  --memco_wilson_threshold "${THRESHOLD}"
)

if [[ -n "${MAX_EVAL:-}" ]]; then
  CMD+=(--max_eval "${MAX_EVAL}")
fi

printf 'SOURCE_RUN_ID=%s\nRUN_ID=%s\n' "${SOURCE_RUN_ID}" "${RUN_ID}"
printf 'COMMAND='
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

stage_memory
exec "${CMD[@]}"
