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

All benchmarks use the same multidomain protocol. The script first builds a local memory for each source domain, then consolidates local memories into a shared global memory, and finally evaluates with both local and global memory available to the agents. This keeps the evaluation mechanism fixed while only changing the benchmark loader and data paths.

The core command shape is:

```bash
python scripts/medmcqa/eval_collab_multidomain_global.py \
  --dataset_family <benchmark> \
  --mas_type autogen \
  --mas_memory graph_memory3 \
  --reasoning io \
  --model gpt-4o-mini \
  --run_id <run_name> \
  --max_trials 30 \
  --batch_size 1 \
  --tool_mode search \
  --reset_memory
```

The benchmark-specific arguments only select the dataset family and the corresponding local data files. For example, use `--dataset_family alfworld`, `scienceworld`, `scienceworld_2`, `fever`, `pddl`, `bfcl_mt`, or `amabench` depending on the benchmark.

To evaluate an existing memory without retraining, rerun the command with the same `--run_id` and replace `--reset_memory` with `--eval_only`.

## Mechanism

The evaluation is organized around local-to-global transfer. During local training, each domain produces its own memory artifacts from task trajectories and agent feedback. During global construction, reusable information from those local memories is promoted into a shared memory store. During evaluation, the agent can retrieve from the current local domain and the shared global memory, allowing the same policy interface to use both domain-specific and cross-domain experience.

This design makes the comparison across benchmarks controlled: the MAS backend, model, reasoning mode, memory implementation, step budget, batching behavior, and report generation are shared. Dataset adapters provide only the environment-specific task interface.

## Outputs

Each run writes a structured report and a Markdown summary under the configured log directory. At completion, the script prints the exact `report_json` and `report_md` paths, followed by per-domain accuracy, reward, step count, task count, and wall-clock summaries.

## Notes for Reviewers

The commands above are intentionally explicit so that each benchmark can be reproduced from the same entry point. Benchmark-specific arguments only define domain splits and data files; the memory training, global merge, and evaluation flow remain shared across benchmarks.
