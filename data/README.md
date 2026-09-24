# Benchmark subsets

This directory contains only the deterministic split metadata used by the
paper. Full benchmark runtimes and large upstream datasets are intentionally
not redistributed.

## ALFWorld

The included `alfworld/collab_subsets/v3_s` files contain 100 training tasks
per room domain and the deterministic official `valid_seen` and
`valid_unseen` selections described in the paper. Install ALFWorld and place
its official `json_2.1.1` directory at `data/alfworld/json_2.1.1`, or pass a
different location through `ALFWORLD_GAME_ROOT`.

## PDDL

The four training files contain 20 Gripper, 10 Blockworld, 20 Barman, and 10
Tyreworld tasks. `pddl/test.jsonl` is the merged 60-task evaluation set. The
runtime used by this artifact is under `tasks/envs/pddl_env`.

## FEVER

The two training partitions contain Film/TV and Music claims. The merged test
set contains 60 balanced examples. FEVER interaction uses the Wikipedia
search interface and therefore requires network access at evaluation time.

## ScienceWorld

The `v4_id_grouped` split groups official task identifiers by their first
component into ten task-family domains. It contains up to ten official train
variations and two official test variations per task template, for a merged
60-task test set. The ScienceWorld runtime is installed from PyPI by the
fully pinned environment.

Users remain responsible for complying with each benchmark's upstream terms.
