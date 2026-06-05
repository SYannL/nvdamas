#!/bin/bash
# ALFWorld collab eval smoke test with small subset (new insight strategy).
# Uses --max_train 4 --max_eval 2 for quick validation.

set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ -f ".env" ]; then
    export $(grep -v '^#' ".env" | xargs)
fi

python scripts/eval_collab_domain_adaptation.py \
    --dataset_family alfworld \
    --alfworld_group_a kitchen_statechange \
    --alfworld_group_b home_search \
    --alfworld_subset_dir data/alfworld/collab_subsets/v2 \
    --max_train 10 \
    --max_eval 6 \
    --mas_type autogen \
    --mas_memory selectivemem \
    --reasoning io \
    --model gpt-4o-mini \
    --batch_size 2 \
    --scenarios all
