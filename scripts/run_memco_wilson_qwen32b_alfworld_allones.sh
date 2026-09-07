#!/usr/bin/env bash
# Prepare/run the Qwen3-32B ALFWorld Wilson experiment.
#
# The fixed parameters claimed in paper equations (22), (24), (25), (26), and
# (27) are pinned to 1. Wilson acceptance itself remains unchanged.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export NV_DAMAS_EMBEDDING_DEVICE="${NV_DAMAS_EMBEDDING_DEVICE:-cpu}"

# Activates only the parameters listed in the accompanying manifest.
export NV_MEMCO_PAPER_EQ22_27_ALL_ONES=1

MODEL="${MODEL:-qwen32b-api}"
WILSON_ALPHA="${WILSON_ALPHA:-0.05}"
WILSON_THRESHOLD="${WILSON_THRESHOLD:-0.5}"
MAX_TRIALS="${MAX_TRIALS:-30}"
BATCH_SIZE="${BATCH_SIZE:-5}"
FOREGROUND="${FOREGROUND:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  curl -fsS "${OPENAI_API_BASE}/models" >/dev/null || {
    echo "Qwen32B API is not ready at ${OPENAI_API_BASE}." >&2
    echo "Start it with: MODEL_PATH=${ROOT}/model/Qwen3-32B bash scripts/launch_qwen32_api.sh" >&2
    exit 1
  }
fi

RUN_TIMEZONE="${RUN_TIMEZONE:-America/Vancouver}"
RUN_TS="${RUN_TS:-$(TZ="${RUN_TIMEZONE}" date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-alfworld_memco_wilson_qwen32b_eq22_27_allones_${RUN_TS}}"
LOG_FILE="${LOG_FILE:-${ROOT}/L_${RUN_ID}.log}"
PID_DIR="${PID_DIR:-${ROOT}/logs/memco_wilson_pids}"
PID_FILE="${PID_DIR}/${RUN_ID}.pid"
mkdir -p "${PID_DIR}"

CMD=(
  "${PYTHON_BIN}" scripts/eval_collab_multidomain_global.py
  --dataset_family alfworld
  --alfworld_domains "${ALFWORLD_DOMAINS:-bathroom,bedroom,kitchen,living}"
  --alfworld_subset_dir "${ALFWORLD_SUBSET_DIR:-data/alfworld/collab_subsets/v3_s}"
  --alfworld_eval_split "${ALFWORLD_EVAL_SPLIT:-valid_seen,valid_unseen}"
  --alfworld_game_root "${ALFWORLD_GAME_ROOT:-${ROOT}/data/alfworld/json_2.1.1}"
  --mas_type autogen
  --mas_memory memco
  --reasoning io
  --model "${MODEL}"
  --run_id "${RUN_ID}"
  --memco_dynamic_graph
  --memco_settings local_plus_global
  --memco_router textloss
  --memco_promotion_policy wilson
  --memco_wilson_alpha "${WILSON_ALPHA}"
  --memco_wilson_threshold "${WILSON_THRESHOLD}"
  --max_trials "${MAX_TRIALS}"
  --batch_size "${BATCH_SIZE}"
  --reset_memory
)

if [[ -n "${MAX_TRAIN:-}" ]]; then
  CMD+=(--max_train "${MAX_TRAIN}")
fi
if [[ -n "${MAX_EVAL:-}" ]]; then
  CMD+=(--max_eval "${MAX_EVAL}")
fi

echo "[wilson-32b-allones] run_id=${RUN_ID} api=${OPENAI_API_BASE} model=${MODEL}"
echo "[wilson-32b-allones] alpha=${WILSON_ALPHA} threshold=${WILSON_THRESHOLD}"
echo "[wilson-32b-allones] paper equations 22,24,25,26,27 parameters=1"
if [[ "${NV_MEMCO_PAPER_A3_2_ACTIVATION_ALL_ONES:-0}" == "1" ]]; then
  echo "[wilson-32b-allones] A3.2 section activation bounds=1"
  echo "[wilson-32b-allones] manifest=${ROOT}/configs/wilson_qwen32b_alfworld_eq22_27_a3_2_activation_allones.json"
else
  echo "[wilson-32b-allones] A3.2 section activation bounds=paper defaults"
  echo "[wilson-32b-allones] manifest=${ROOT}/configs/wilson_qwen32b_alfworld_eq22_27_allones.json"
fi
printf '[wilson-32b-allones] command:'
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
echo "[wilson-32b-allones] PID=${PID}"
echo "[wilson-32b-allones] log=${LOG_FILE}"
