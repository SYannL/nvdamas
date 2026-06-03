#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-claude-haiku-4-5}"

if [ -z "${OPENAI_API_BASE:-}" ]; then
    case "$MODEL" in
        gemini*|*gemini*)
            export OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/"
            ;;
        deepseek*|*deepseek*)
            export OPENAI_API_BASE="https://openrouter.ai/api/v1"
            ;;
        gpt*|o[0-9]*|o[0-9]-*)
            export OPENAI_API_BASE="https://api.openai.com/v1"
            ;;
        qwen*|*qwen*)
            export OPENAI_API_BASE="${QWEN_OPENAI_API_BASE:-http://127.0.0.1:8000/v1}"
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

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is not set."
    echo "Example:"
    echo "  export OPENAI_API_KEY='your_api_key'"
    exit 1
fi

MAS_MEMORY="${MAS_MEMORY:-g-memory}"
MAS_TYPE="${MAS_TYPE:-autogen}"
REASONING="${REASONING:-io}"
MAX_TRIALS="${MAX_TRIALS:-30}"
BATCH_SIZE="${BATCH_SIZE:-10}"
FEVER_DOMAINS="${FEVER_DOMAINS:-A_film_tv,B_music}"
FEVER_TRAIN_JSONL="${FEVER_TRAIN_JSONL:-data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl}"
FEVER_TEST_JSONL="${FEVER_TEST_JSONL:-data/fever/fever_ab_test_v3.jsonl}"
RESET_MEMORY="${RESET_MEMORY:-1}"
EVAL_ONLY="${EVAL_ONLY:-0}"
TOOL_MODE="${TOOL_MODE:-}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_EVAL="${MAX_EVAL:-}"
GM2_DYNAMIC_GRAPH="${GM2_DYNAMIC_GRAPH:-0}"
GM2_REPO_ROOT="${GM2_REPO_ROOT:-}"
GM2_RETRIEVAL_MODE="${GM2_RETRIEVAL_MODE:-graph_policy}"
GM2_SETTINGS="${GM2_SETTINGS:-local_plus_global}"
GM3_DYNAMIC_GRAPH="${GM3_DYNAMIC_GRAPH:-1}"
GM3_REPO_ROOT="${GM3_REPO_ROOT:-}"
GM3_ROUTER="${GM3_ROUTER:-textloss}"
GM3_SETTINGS="${GM3_SETTINGS:-local_plus_global}"
GM3_PROMOTION_THRESHOLD="${GM3_PROMOTION_THRESHOLD:-0.35}"
MEMSKILL_CONTROLLER="${MEMSKILL_CONTROLLER:-llm}"
MEMSKILL_CHECKPOINT_PATH="${MEMSKILL_CHECKPOINT_PATH:-/home/xenial/scratch/nvdamas/Models/memskill/alfworld_controller.pt}"
MEMSKILL_OPERATION_BANK_PATH="${MEMSKILL_OPERATION_BANK_PATH:-}"
MEMSKILL_REQUIRE_PPO="${MEMSKILL_REQUIRE_PPO:-0}"
MEMSKILL_PPO_DEVICE="${MEMSKILL_PPO_DEVICE:-cpu}"
MEMSKILL_PPO_REPO_PATH="${MEMSKILL_PPO_REPO_PATH:-}"
MEMSKILL_FINALIZE_LOCAL="${MEMSKILL_FINALIZE_LOCAL:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
MODEL_SAFE="$(printf '%s' "$MODEL" | tr '/:' '__')"
RUN="${RUN:-fever_${MAS_MEMORY}_${MODEL_SAFE}_${TS}}"

cmd=(
    python -u scripts/medmcqa/eval_collab_multidomain_global.py
    --dataset_family fever
    --fever_domains "$FEVER_DOMAINS"
    --fever_train_jsonl "$FEVER_TRAIN_JSONL"
    --fever_test_jsonl "$FEVER_TEST_JSONL"
    --mas_type "$MAS_TYPE"
    --mas_memory "$MAS_MEMORY"
    --reasoning "$REASONING"
    --model "$MODEL"
    --max_trials "$MAX_TRIALS"
    --batch_size "$BATCH_SIZE"
    --run_id "$RUN"
)

if [ "$RESET_MEMORY" = "1" ]; then
    cmd+=(--reset_memory)
fi

if [ "$EVAL_ONLY" = "1" ]; then
    cmd+=(--eval_only)
fi

if [ -n "$MAX_EVAL" ]; then
    cmd+=(--max_eval "$MAX_EVAL")
fi

if [ -n "$MAX_TRAIN" ]; then
    cmd+=(--max_train "$MAX_TRAIN")
fi

if [ -n "$TOOL_MODE" ]; then
    cmd+=(--tool_mode "$TOOL_MODE")
elif [ "$MAS_MEMORY" = "amem" ] || [ "$MAS_MEMORY" = "memrl" ]; then
    cmd+=(--tool_mode search)
fi

if [ "$MAS_MEMORY" = "graph_memory2" ]; then
    if [ "$GM2_DYNAMIC_GRAPH" = "1" ]; then
        cmd+=(--gm2_dynamic_graph)
    fi
    if [ -n "$GM2_REPO_ROOT" ]; then
        cmd+=(--gm2_repo_root "$GM2_REPO_ROOT")
    fi
    cmd+=(--gm2_retrieval_mode "$GM2_RETRIEVAL_MODE")
    cmd+=(--gm2_settings "$GM2_SETTINGS")
fi

if [ "$MAS_MEMORY" = "graph_memory3" ]; then
    if [ "$GM3_DYNAMIC_GRAPH" = "1" ]; then
        cmd+=(--gm3_dynamic_graph)
    fi
    if [ -n "$GM3_REPO_ROOT" ]; then
        cmd+=(--gm3_repo_root "$GM3_REPO_ROOT")
    fi
    cmd+=(--gm3_router "$GM3_ROUTER")
    cmd+=(--gm3_settings "$GM3_SETTINGS")
    cmd+=(--gm3_promotion_threshold "$GM3_PROMOTION_THRESHOLD")
fi

if [ "$MAS_MEMORY" = "memskill" ]; then
    cmd+=(--memskill_controller "$MEMSKILL_CONTROLLER")
    if [ -n "$MEMSKILL_CHECKPOINT_PATH" ]; then
        cmd+=(--memskill_checkpoint_path "$MEMSKILL_CHECKPOINT_PATH")
    fi
    if [ -n "$MEMSKILL_OPERATION_BANK_PATH" ]; then
        cmd+=(--memskill_operation_bank_path "$MEMSKILL_OPERATION_BANK_PATH")
    fi
    if [ "$MEMSKILL_REQUIRE_PPO" = "1" ]; then
        cmd+=(--memskill_require_ppo)
    fi
    if [ -n "$MEMSKILL_PPO_DEVICE" ]; then
        cmd+=(--memskill_ppo_device "$MEMSKILL_PPO_DEVICE")
    fi
    if [ -n "$MEMSKILL_PPO_REPO_PATH" ]; then
        cmd+=(--memskill_ppo_repo_path "$MEMSKILL_PPO_REPO_PATH")
    fi
    if [ "$MEMSKILL_FINALIZE_LOCAL" = "1" ]; then
        cmd+=(--memskill_finalize_local)
    fi
fi

printf 'RUN=%s\n' "$RUN"
printf 'MODEL=%s\n' "$MODEL"
printf 'MAS_MEMORY=%s\n' "$MAS_MEMORY"
printf 'OPENAI_API_BASE=%s\n' "$OPENAI_API_BASE"

"${cmd[@]}"
