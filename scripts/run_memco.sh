#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATASET="${1:-}"
if [[ -z "${DATASET}" ]]; then
  echo "Usage: bash scripts/run_memco.sh {alfworld|pddl|fever|scienceworld}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-qwen32b-api}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-10}"
WILSON_ALPHA="${WILSON_ALPHA:-0.05}"
WILSON_THRESHOLD="${WILSON_THRESHOLD:-0.35}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${DATASET}_memco_wilson_seed${SEED}_${RUN_TS}}"
DRY_RUN="${DRY_RUN:-0}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export NV_DAMAS_EMBEDDING_DEVICE="${NV_DAMAS_EMBEDDING_DEVICE:-cpu}"
export NV_MEMCO_PAPER_EQ22_27_ALL_ONES=1
export NV_MEMCO_PAPER_A3_2_ACTIVATION_ALL_ONES=0
export PYTHONHASHSEED="${SEED}"

case "${DATASET}" in
  alfworld)
    MAX_TRIALS="${MAX_TRIALS:-30}"
    DATASET_ARGS=(
      --dataset_family alfworld
      --alfworld_domains bathroom,bedroom,kitchen,living
      --alfworld_subset_dir data/alfworld/collab_subsets/v3_s
      --alfworld_eval_split valid_seen,valid_unseen
      --alfworld_game_root "${ALFWORLD_GAME_ROOT:-${ROOT}/data/alfworld/json_2.1.1}"
    )
    ;;
  pddl)
    MAX_TRIALS="${MAX_TRIALS:-30}"
    DATASET_ARGS=(
      --dataset_family pddl
      --pddl_domains gripper,blockworld,barman,tyreworld
      --pddl_train_jsonl data/pddl/pddl_domain_gripper.jsonl,data/pddl/pddl_domain_blockworld.jsonl,data/pddl/pddl_domain_barman.jsonl,data/pddl/pddl_domain_tyreworld.jsonl
      --pddl_test_jsonl data/pddl/test.jsonl
    )
    ;;
  fever)
    MAX_TRIALS="${MAX_TRIALS:-12}"
    DATASET_ARGS=(
      --dataset_family fever
      --fever_domains A_film_tv,B_music
      --fever_train_jsonl data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl
      --fever_test_jsonl data/fever/fever_ab_test_v3.jsonl
    )
    ;;
  scienceworld)
    MAX_TRIALS="${MAX_TRIALS:-50}"
    DATASET_ARGS=(
      --dataset_family scienceworld
      --sw_domains 1,2,3,4,5,6,7,8,9,10
      --sw_subset_dir data/ScienceWorld/collab_subsets/v4_id_grouped
      --sw_test_json data/ScienceWorld/collab_subsets/v4_id_grouped/merged__test.json
    )
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

CMD=(
  "${PYTHON_BIN}" -u scripts/eval_collab_multidomain_global.py
  "${DATASET_ARGS[@]}"
  --mas_type autogen
  --mas_memory memco
  --reasoning io
  --model "${MODEL}"
  --run_id "${RUN_ID}"
  --seed "${SEED}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --tool_mode search
  --memco_dynamic_graph
  --memco_settings local_plus_global
  --memco_router textloss
  --memco_promotion_policy wilson
  --memco_wilson_alpha "${WILSON_ALPHA}"
  --memco_wilson_threshold "${WILSON_THRESHOLD}"
  --reset_memory
)

if [[ -n "${MAX_TRAIN:-}" ]]; then
  CMD+=(--max_train "${MAX_TRAIN}")
fi
if [[ -n "${MAX_EVAL:-}" ]]; then
  CMD+=(--max_eval "${MAX_EVAL}")
fi

printf 'RUN_ID=%s\n' "${RUN_ID}"
printf 'COMMAND='
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

exec "${CMD[@]}"
