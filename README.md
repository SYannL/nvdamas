# G-Memory Multidomain Evaluation

This repository contains the evaluation code for running a shared memory-enabled multi-agent system across multiple domains and benchmarks. The main reviewer-facing entry point is:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py
```

The script follows the same high-level protocol for every benchmark: train local memory on each source domain, merge local memories into a shared global memory, and evaluate each target split with local plus global retrieval.

![Framework overview](assets/framework.png)

## Supported Benchmarks

The multidomain script currently supports:

| Benchmark | `--dataset_family` | Domain split |
| --- | --- | --- |
| ALFWorld | `alfworld` | scene domains: bathroom, bedroom, kitchen, living |
| ScienceWorld | `scienceworld` | initial-room domains |
| ScienceWorld-2 | `scienceworld_2` | mixed science-family domains |
| FEVER | `fever` | claim-topic domains |
| PDDL | `pddl` or `pddl_2` | planning-game domains |
| BFCL multi-turn | `bfcl_mt` | API-family domains |
| AmaBench | `amabench` | task-category domains |

## Basic Usage

All runs use the same core command shape:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family <benchmark> \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id <run_name> \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

Use `--eval_only` with the same `--run_id` to evaluate an existing global memory without retraining. Do not combine `--eval_only` and `--reset_memory`.

## Benchmark Commands

ALFWorld:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family alfworld \
  --alfworld_domains bathroom,bedroom,kitchen,living \
  --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
  --alfworld_eval_split valid_seen,valid_unseen \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id alfworld_gmemory \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

ScienceWorld:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family scienceworld \
  --sw_domains art_studio,bathroom,greenhouse,hallway,kitchen,living_room \
  --sw_subset_dir data/ScienceWorld/collab_subsets/v2_room \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id scienceworld_gmemory \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

ScienceWorld-2:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family scienceworld_2 \
  --sw2_domains conductivity,melting_point,friction,genetics \
  --sw2_subset_dir data/ScienceWorld/collab_subsets/v3_family_mixed \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id scienceworld2_gmemory \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

FEVER:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family fever \
  --fever_domains A_film_tv,B_music \
  --fever_train_jsonl data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl \
  --fever_test_jsonl data/fever/fever_ab_test_v3.jsonl \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id fever_gmemory \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

PDDL:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family pddl \
  --pddl_domains gripper,blockworld,barman,tyreworld \
  --pddl_test_jsonl data/pddl/test.jsonl \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id pddl_gmemory \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

BFCL multi-turn:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family bfcl_mt \
  --bfcl_use_family_collab_split \
  --bfcl_family_domains trading,travel,vehicle,fs \
  --mas_type autogen \
  --mas_memory g-memory \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id bfcl_mt_gmemory \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

## Outputs

Each run writes a structured report and a Markdown summary under the configured log directory. At completion, the script prints the exact `report_json` and `report_md` paths, followed by per-domain accuracy, reward, step count, task count, and wall-clock summaries.

## Notes for Reviewers

The commands above are intentionally explicit so that each benchmark can be reproduced from the same entry point. Benchmark-specific arguments only define domain splits and data files; the memory training, global merge, and evaluation flow remain shared across benchmarks.
