#!/usr/bin/env bash
# Two-phase MT-Mind2Web + G-Memory:
#  1) Train on full train-turn list (mtmind2web_train_eval.jsonl), write FINAL_DATASET_METRICS to log + stdout.
#  2) Evaluate on test_task with the same memory (--mtmind2web_shared_memory + same RUN_ID).
#
# Usage:
#   ./scripts/mtmind2web/run_gmemory_train_then_test.sh
#   MODEL=gpt-4o-mini RUN_ID=my_exp ./scripts/mtmind2web/run_gmemory_train_then_test.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-gpt-4o-mini}"
RUN_ID="${RUN_ID:-mtmind2web_gmem_$(date +%Y%m%d_%H%M%S)}"

echo "=== Phase 1: TRAIN (G-Memory, shared root mtmind2web) RUN_ID=${RUN_ID} ==="
.venv/bin/python tasks/run.py \
  --task mtmind2web_train \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model "${MODEL}" \
  --max_trials 1 \
  --mtmind2web_shared_memory \
  --run_id "${RUN_ID}"

echo ""
echo "=== Phase 2: TEST test_task (reuse memory above) RUN_ID=${RUN_ID} ==="
.venv/bin/python tasks/run.py \
  --task mtmind2web_test_task \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model "${MODEL}" \
  --max_trials 1 \
  --mtmind2web_shared_memory \
  --run_id "${RUN_ID}"

echo ""
echo "Done. RUN_ID=${RUN_ID}"
echo "Train log:   logs/mtmind2web_train/autogen/memory/g-memory/${MODEL}/${RUN_ID}/total_task.log"
echo "Test log:    logs/mtmind2web_test_task/autogen/memory/g-memory/${MODEL}/${RUN_ID}/total_task.log"
echo "Shared G-Memory dir: .db/${MODEL}/mtmind2web/autogen/memory/g-memory/${RUN_ID}/"
