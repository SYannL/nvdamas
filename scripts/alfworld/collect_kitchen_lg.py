#!/usr/bin/env python3
"""
Run ALFWorld kitchen_statechange tasks until N successes and export GL artifacts.

Outputs:
  <out_dir>/local_instance_graph.json
  <out_dir>/local_instance_graph.png
  <out_dir>/local_category_graph.json
  <out_dir>/local_category_graph.png
  <out_dir>/run_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import importlib.util

import sys

# Allow importing `prompts` and other task helpers from the `tasks/` directory
# even when `tasks/` itself is not a Python package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT / "tasks"))
from prompts import get_dataset_system_prompt, get_task_few_shots


def _load_eval_helpers(repo_root: Path):
    """
    Load helper functions from scripts/eval_collab_domain_adaptation.py without requiring
    scripts/ to be a Python package.
    """
    module_path = repo_root / "scripts" / "eval_collab_domain_adaptation.py"
    spec = importlib.util.spec_from_file_location("nvdamas_eval_collab_domain_adaptation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--mas_type", type=str, default="autogen", choices=["autogen", "macnet", "dylan", "strategy"])
    parser.add_argument("--mas_memory", type=str, default="selectivemem")
    parser.add_argument("--reasoning", type=str, default="io")
    parser.add_argument("--subset_dir", type=str, default="data/alfworld/collab_subsets/v2")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid_unseen"])
    parser.add_argument("--max_trials", type=int, default=50)
    parser.add_argument("--target_successes", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="reports/selectivemem_kitchen_gl")
    parser.add_argument("--max_tasks", type=int, default=50)
    args = parser.parse_args()

    repo_root = _REPO_ROOT
    out_dir = repo_root / args.out_dir / time.strftime("%Y%m%d_%H%M%S")
    ensure_dir(str(out_dir))

    helpers = _load_eval_helpers(repo_root)

    # Working/log dirs under .db/.logs so memory module can persist snapshots.
    run_id = f"collect_kitchen_gl_{time.strftime('%Y%m%d_%H%M%S')}"
    model_type = args.model.replace("/", "_")
    working_dir = str(repo_root / ".db" / model_type / "collect_kitchen_gl" / run_id)
    log_dir = str(repo_root / "logs" / "collect_kitchen_gl" / run_id)
    ensure_dir(working_dir)
    ensure_dir(log_dir)

    # Build a TaskManager for ALFWorld, then override tasks to kitchen_statechange subset.
    manager = helpers.build_task_manager(
        task="alfworld",
        mas_type=args.mas_type,
        memory_type=args.mas_memory,
        max_steps=args.max_trials,
        working_dir=working_dir,
        log_dir=log_dir,
    )
    manager.mem_config["viz_output_dir"] = str(out_dir)

    subset_tasks = helpers.load_alfworld_subset_file(Path(args.subset_dir), "kitchen_statechange", args.split)
    manager.tasks = subset_tasks[: max(1, min(len(subset_tasks), args.max_tasks))]

    mem = helpers.build_mas(manager, reasoning=args.reasoning, mas_memory=args.mas_memory, llm_type=args.model)

    successes = 0
    attempted = 0
    succeeded_task_ids: list[int] = []
    last_report_attempted = 0
    last_report_successes = 0

    manager.recorder.dataset_begin()
    for task_id, task_config in enumerate(manager.tasks):
        attempted += 1
        manager.recorder.task_begin(task_id, task_config)
        try:
            task_main, task_description = manager.mas.env.set_env(task_config)
            few_shots = get_task_few_shots(
                dataset=manager.task_name, task_config=task_config, few_shots_num=1
            )
            task_config.update(
                task_main=task_main,
                task_description=task_description,
                few_shots=few_shots,
            )

            # IMPORTANT: inject ALFWorld action-format constraints into both agents,
            # otherwise the LLM may output natural-language plans that the env can't execute.
            task_instruction = get_dataset_system_prompt(
                manager.task_name, task_config=task_config
            )
            for agent in manager.mas.agents_team.values():
                manager.recorder.log(agent.add_task_instruction(task_instruction))

            reward, success = manager.mas.schedule(task_config)
        except BaseException as exc:
            # Skip failures in env/loader; goal is to collect successes.
            manager.recorder.log(f"[collect] task {task_id} exception: {type(exc).__name__}: {exc}")
            manager.recorder.task_end(0.0, False)
            continue

        manager.recorder.task_end(float(reward), bool(success))
        if success:
            successes += 1
            succeeded_task_ids.append(task_id)
            # Force persist GL snapshot for the latest success
            try:
                mem.persist_entity_graph()
            except Exception:
                pass

        if successes >= args.target_successes:
            break

        # Report success rate every 10 attempted tasks.
        if attempted % 10 == 0:
            # last_report_successes/last_report_attempted are "since last report"
            window_attempted = attempted - last_report_attempted
            window_successes = successes - last_report_successes
            # Avoid division by zero; should not happen because attempted increments by 1.
            rate = (window_successes / window_attempted) if window_attempted else 0.0
            print(
                f"[collect] success rate last {window_attempted} tasks: "
                f"{window_successes}/{window_attempted} ({rate:.2%}); "
                f"total: {successes}/{attempted}",
                flush=True,
            )
            last_report_attempted = attempted
            last_report_successes = successes

    manager.recorder.dataset_end()

    # Copy the persisted GL snapshot from memory dir to out_dir
    mem_dir = Path(mem.persist_dir)
    gl_path = mem_dir / "local_instance_graph.json"
    cat_gl_path = mem_dir / "local_category_graph.json"
    if not gl_path.exists():
        # It's possible that no task is actually solved under the corrected win-based success criterion.
        # In that case, we still want the run to terminate cleanly so you can inspect success-rate logs.
        meta = {
            "model": args.model,
            "mas_type": args.mas_type,
            "mas_memory": args.mas_memory,
            "reasoning": args.reasoning,
            "subset_dir": args.subset_dir,
            "split": args.split,
            "max_trials": args.max_trials,
            "target_successes": args.target_successes,
            "attempted": attempted,
            "successes": successes,
            "succeeded_task_ids": succeeded_task_ids,
            "working_dir": working_dir,
            "memory_persist_dir": str(mem_dir),
            "out_dir": str(out_dir),
            "gl_json": "",
            "gl_png": "",
            "gl_category_json": "",
            "gl_category_png": "",
        }
        (out_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[collect] warning: Missing GL snapshot (no successful tasks). "
            f"successes={successes}/{attempted}",
            flush=True,
        )
        return

    out_gl = out_dir / "local_instance_graph.json"
    out_gl.write_text(gl_path.read_text(encoding="utf-8"), encoding="utf-8")
    out_cat_gl = out_dir / "local_category_graph.json"
    if cat_gl_path.exists():
        out_cat_gl.write_text(cat_gl_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Visualize
    out_png = out_dir / "local_instance_graph.png"
    out_cat_png = out_dir / "local_category_graph.png"
    viz_script = repo_root / "scripts" / "alfworld" / "visualize_local_instance_graph.py"
    os.system(
        f"python \"{viz_script}\" --input \"{out_gl}\" --output \"{out_png}\""
    )
    if out_cat_gl.exists():
        os.system(
            f"python \"{viz_script}\" --input \"{out_cat_gl}\" --output \"{out_cat_png}\""
        )

    meta = {
        "model": args.model,
        "mas_type": args.mas_type,
        "mas_memory": args.mas_memory,
        "reasoning": args.reasoning,
        "subset_dir": args.subset_dir,
        "split": args.split,
        "max_trials": args.max_trials,
        "target_successes": args.target_successes,
        "attempted": attempted,
        "successes": successes,
        "succeeded_task_ids": succeeded_task_ids,
        "working_dir": working_dir,
        "memory_persist_dir": str(mem_dir),
        "out_dir": str(out_dir),
        "gl_json": str(out_gl),
        "gl_png": str(out_png),
        "gl_category_json": str(out_cat_gl) if out_cat_gl.exists() else "",
        "gl_category_png": str(out_cat_png) if out_cat_gl.exists() else "",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[collect] success. successes={successes}/{attempted}")
    print(f"[collect] GL JSON: {out_gl}")
    print(f"[collect] GL PNG : {out_png}")
    if out_cat_gl.exists():
        print(f"[collect] GL(category) JSON: {out_cat_gl}")
        print(f"[collect] GL(category) PNG : {out_cat_png}")


if __name__ == "__main__":
    main()

