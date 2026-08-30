#!/usr/bin/env bash
# One-command deployment and reproduction for Qwen3-4B on FEVER and PDDL.
#
# Default matrix: 2 datasets x 6 memory methods x 2 repeats = 24 runs.
# ALFWorld assets are neither downloaded nor referenced.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:-all}"
case "${MODE}" in
  all|setup|services|run|smoke) ;;
  *)
    echo "Usage: bash scripts/quickstart_qwen4b_repro.sh [all|setup|services|run|smoke]" >&2
    exit 2
    ;;
esac

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.11}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv-qwen4b}"
PYTHON_BIN="${VENV_DIR}/bin/python"
HF_BIN="${VENV_DIR}/bin/hf"
HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.cache/matplotlib}"
export NLTK_DATA="${NLTK_DATA:-${ROOT}/.cache/nltk}"

GPU_ID="${GPU_ID:-0}"
QWEN4_PORT="${QWEN4_PORT:-8004}"
EMBED_PORT="${EMBED_PORT:-8001}"
QWEN4_MODEL="${QWEN4_MODEL:-${ROOT}/model/Qwen3-4B}"
EMBED_MODEL="${EMBED_MODEL:-${ROOT}/model/Qwen3-Embedding-0.6B}"
QWEN4_GPU_MEM_UTIL="${QWEN4_GPU_MEM_UTIL:-0.65}"
EMBED_GPU_MEM_UTIL="${EMBED_GPU_MEM_UTIL:-0.18}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

REPEATS="${REPEATS:-2}"
METHODS_CSV="${METHODS:-memco,g-memory,amem,empty,memskill,memrl}"
DATASETS_CSV="${DATASETS:-fever,pddl}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_EVAL="${MAX_EVAL:-}"

log() {
  printf '[qwen4b-repro] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

setup_runtime() {
  require_command "${PYTHON_BOOTSTRAP}"
  require_command curl
  require_command sha256sum

  mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${NLTK_DATA}"

  if [[ ! -x "${PYTHON_BIN}" ]]; then
    log "Creating Python environment: ${VENV_DIR}"
    "${PYTHON_BOOTSTRAP}" -m venv "${VENV_DIR}"
  fi

  local requirement_hash installed_hash
  requirement_hash="$(sha256sum requirements-qwen4b-repro.txt | awk '{print $1}')"
  installed_hash="$(cat "${VENV_DIR}/.requirements-qwen4b-repro.sha256" 2>/dev/null || true)"
  if [[ "${requirement_hash}" != "${installed_hash}" ]]; then
    log "Installing FEVER/PDDL runtime dependencies"
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    "${PYTHON_BIN}" -m pip install -r requirements-qwen4b-repro.txt
    printf '%s\n' "${requirement_hash}" > "${VENV_DIR}/.requirements-qwen4b-repro.sha256"
  else
    log "Python dependencies already match"
  fi

  mkdir -p "${QWEN4_MODEL}" "${EMBED_MODEL}" "${HF_HOME}"
  if [[ ! -f "${QWEN4_MODEL}/config.json" ]]; then
    log "Downloading Qwen/Qwen3-4B to ${QWEN4_MODEL}"
    HF_HOME="${HF_HOME}" "${HF_BIN}" download Qwen/Qwen3-4B --local-dir "${QWEN4_MODEL}"
  else
    log "Qwen3-4B already present"
  fi
  if [[ ! -f "${EMBED_MODEL}/config.json" ]]; then
    log "Downloading Qwen/Qwen3-Embedding-0.6B to ${EMBED_MODEL}"
    HF_HOME="${HF_HOME}" "${HF_BIN}" download Qwen/Qwen3-Embedding-0.6B --local-dir "${EMBED_MODEL}"
  else
    log "Qwen3-Embedding-0.6B already present"
  fi

  "${PYTHON_BIN}" -m nltk.downloader -q punkt punkt_tab
  log "Setup complete; no ALFWorld assets were downloaded"
}

wait_for_model() {
  local port="$1"
  local expected="$2"
  local label="$3"
  local attempts="${SERVICE_WAIT_ATTEMPTS:-120}"
  local interval="${SERVICE_WAIT_INTERVAL:-10}"
  local model_id=""

  for ((i = 1; i <= attempts; i++)); do
    model_id="$(curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" 2>/dev/null \
      | "${PYTHON_BIN}" -c 'import json,sys; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || true)"
    if [[ "${model_id}" == "${expected}" ]]; then
      log "${label} ready on port ${port}: ${model_id}"
      return 0
    fi
    if ((i == 1 || i % 6 == 0)); then
      log "Waiting for ${label} (${i}/${attempts}); inspect logs under ${ROOT}/logs"
    fi
    sleep "${interval}"
  done

  echo "Timed out waiting for ${label} on port ${port}" >&2
  return 1
}

start_services() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Runtime missing. Run setup first." >&2
    exit 1
  fi
  if [[ ! -f "${QWEN4_MODEL}/config.json" || ! -f "${EMBED_MODEL}/config.json" ]]; then
    echo "Model files missing. Run setup first." >&2
    exit 1
  fi

  log "Starting Qwen3-4B generation service on GPU ${GPU_ID}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MODEL_PATH="${QWEN4_MODEL}" \
  SERVED_MODEL_NAME=qwen4b-api \
  PORT="${QWEN4_PORT}" \
  GPU_MEM_UTIL="${QWEN4_GPU_MEM_UTIL}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/launch_qwen4_api.sh

  wait_for_model "${QWEN4_PORT}" qwen4b-api Qwen3-4B

  log "Starting Qwen3-Embedding-0.6B service on GPU ${GPU_ID}"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MODEL_PATH="${EMBED_MODEL}" \
  SERVED_MODEL_NAME=qwen3-embedding-api \
  PORT="${EMBED_PORT}" \
  GPU_MEM_UTIL="${EMBED_GPU_MEM_UTIL}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  PYTHON_BIN="${PYTHON_BIN}" \
    bash scripts/launch_qwen3_embedding_api.sh

  wait_for_model "${EMBED_PORT}" qwen3-embedding-api Qwen3-Embedding-0.6B

  local embedding_dim
  embedding_dim="$(curl -fsS --max-time 30 "http://127.0.0.1:${EMBED_PORT}/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer dummy' \
    -d '{"model":"qwen3-embedding-api","input":["deployment health check"]}' \
    | "${PYTHON_BIN}" -c 'import json,sys; print(len(json.load(sys.stdin)["data"][0]["embedding"]))')"
  log "Embedding request succeeded; dimension=${embedding_dim}"
}

preflight_data() {
  local path
  local required=(
    data/fever/fever_ab_train_A_v3.jsonl
    data/fever/fever_ab_train_B_v3.jsonl
    data/fever/fever_ab_test_v3.jsonl
    data/pddl/pddl_domain_gripper.jsonl
    data/pddl/pddl_domain_blockworld.jsonl
    data/pddl/pddl_domain_barman.jsonl
    data/pddl/pddl_domain_tyreworld.jsonl
    data/pddl/test.jsonl
  )
  for path in "${required[@]}"; do
    if [[ ! -s "${path}" ]]; then
      echo "Required reproduction data missing: ${path}" >&2
      exit 1
    fi
  done
}

run_matrix() {
  preflight_data
  wait_for_model "${QWEN4_PORT}" qwen4b-api Qwen3-4B
  wait_for_model "${EMBED_PORT}" qwen3-embedding-api Qwen3-Embedding-0.6B

  export OPENAI_API_BASE="http://127.0.0.1:${QWEN4_PORT}/v1"
  export OPENAI_API_KEY=dummy
  export OPENAI_EMBEDDING_API_BASE="http://127.0.0.1:${EMBED_PORT}/v1"
  export OPENAI_EMBEDDING_API_KEY=dummy
  export OPENAI_EMBEDDING_MODEL=qwen3-embedding-api
  export MEMRL_EMBEDDING_MODEL=qwen3-embedding-api

  # The API services already own the GPU. Local retrieval encoders run on CPU.
  export CUDA_VISIBLE_DEVICES=""

  local stamp output_dir master_log total group_index
  stamp="$(date +%Y%m%d_%H%M%S)"
  output_dir="${ROOT}/logs/qwen4b_repro/${stamp}"
  master_log="${output_dir}/matrix.log"
  mkdir -p "${output_dir}"

  IFS=',' read -r -a methods <<< "${METHODS_CSV}"
  IFS=',' read -r -a datasets <<< "${DATASETS_CSV}"
  total=$((${#methods[@]} * ${#datasets[@]} * REPEATS))
  group_index=0

  {
    echo "Qwen3-4B FEVER/PDDL reproduction matrix"
    echo "datasets=${DATASETS_CSV}"
    echo "methods=${METHODS_CSV}"
    echo "repeats=${REPEATS}"
    echo "total_runs=${total}"
    echo "generation_model=qwen4b-api"
    echo "embedding_model=Qwen3-Embedding-0.6B"
    echo "start_time=$(date --iso-8601=seconds)"
  } | tee "${master_log}"

  set +e
  local dataset method repeat label run_id run_log status
  local -a dataset_args method_args limit_args cmd
  for dataset in "${datasets[@]}"; do
    case "${dataset}" in
      fever)
        dataset_args=(
          --dataset_family fever
          --fever_domains A_film_tv,B_music
          --fever_train_jsonl data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl
          --fever_test_jsonl data/fever/fever_ab_test_v3.jsonl
        )
        ;;
      pddl)
        dataset_args=(
          --dataset_family pddl
          --pddl_domains gripper,blockworld,barman,tyreworld
          --pddl_train_jsonl data/pddl/pddl_domain_gripper.jsonl,data/pddl/pddl_domain_blockworld.jsonl,data/pddl/pddl_domain_barman.jsonl,data/pddl/pddl_domain_tyreworld.jsonl
          --pddl_test_jsonl data/pddl/test.jsonl
        )
        ;;
      *)
        echo "Unsupported dataset in DATASETS: ${dataset}" | tee -a "${master_log}"
        continue
        ;;
    esac

    for method in "${methods[@]}"; do
      method_args=()
      label="${method}"
      case "${method}" in
        memco)
          method_args=(
            --memco_dynamic_graph
            --memco_settings local_plus_global
            --memco_router textloss
            --memco_promotion_threshold 0.35
          )
          ;;
        g-memory) label=gmemory ;;
        amem|empty|memrl) ;;
        memskill)
          method_args=(
            --memskill_controller llm
            --memskill_finalize_local
            --memskill_state_encoder "${EMBED_MODEL}"
            --memskill_op_encoder "${EMBED_MODEL}"
            --memskill_ppo_device cpu
            --no-memskill_use_flash_attn
          )
          ;;
        *)
          echo "Unsupported method in METHODS: ${method}" | tee -a "${master_log}"
          continue
          ;;
      esac

      for ((repeat = 1; repeat <= REPEATS; repeat++)); do
        group_index=$((group_index + 1))
        run_id="${dataset^^}_${label}_rep_4b_06b_r${repeat}_${stamp}"
        run_log="${output_dir}/${run_id}.log"
        limit_args=()
        [[ -n "${MAX_TRAIN}" ]] && limit_args+=(--max_train "${MAX_TRAIN}")
        [[ -n "${MAX_EVAL}" ]] && limit_args+=(--max_eval "${MAX_EVAL}")

        cmd=(
          "${PYTHON_BIN}" -u scripts/eval_collab_multidomain_global.py
          "${dataset_args[@]}"
          --mas_type autogen
          --mas_memory "${method}"
          --reasoning io
          --model qwen4b-api
          --max_trials 30
          --batch_size 10
          --tool_mode search
          --run_id "${run_id}"
          --reset_memory
          "${method_args[@]}"
          "${limit_args[@]}"
        )

        {
          echo
          echo "============================================================"
          echo "START group=${group_index}/${total} dataset=${dataset} method=${label} repeat=${repeat}/${REPEATS}"
          echo "RUN_ID=${run_id}"
          echo "RUN_LOG=${run_log}"
          echo "START_TIME=$(date --iso-8601=seconds)"
          printf 'COMMAND='
          printf '%q ' "${cmd[@]}"
          echo
          echo "============================================================"
        } | tee -a "${master_log}"

        "${cmd[@]}" > "${run_log}" 2>&1
        status=$?

        {
          echo "END group=${group_index}/${total} dataset=${dataset} method=${label} repeat=${repeat}/${REPEATS} exit_status=${status}"
          echo "END_TIME=$(date --iso-8601=seconds)"
          if [[ "${status}" -eq 0 ]]; then
            echo "RESULT=completed"
          else
            echo "RESULT=failed; continuing with next run"
            echo "LAST_LOG_LINES:"
            tail -n 20 "${run_log}"
          fi
          echo "============================================================"
        } | tee -a "${master_log}"
      done
    done
  done
  set -e

  {
    echo
    echo "MATRIX_FINISHED=$(date --iso-8601=seconds)"
    echo "MASTER_LOG=${master_log}"
  } | tee -a "${master_log}"
}

case "${MODE}" in
  setup)
    setup_runtime
    ;;
  services)
    start_services
    ;;
  run)
    run_matrix
    ;;
  smoke)
    setup_runtime
    start_services
    REPEATS=1
    METHODS_CSV=memco
    DATASETS_CSV=fever
    MAX_TRAIN=2
    MAX_EVAL=2
    run_matrix
    ;;
  all)
    setup_runtime
    start_services
    run_matrix
    ;;
esac
