"""
多数据集评测基线：与 ``eval_collab_multidomain_global.py`` 使用相同的数据加载与调用链
（ALFWorld / PDDL / FEVER），固定 ``autogen`` + ``gpt-4o-mini`` + ``reasoning=io``，
``mas_memory=empty``（无持久化记忆、无训练阶段），对测试集样本依次跑 ``run_tasks``。

PDDL/FEVER 与 multidomain 脚本一致；ALFWorld 对每个 eval split 合并各 domain 子集后
整批评测一次（empty 下不按 domain 重复挂载 local+global）。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tasks.utils import get_model_type

from scripts.medmcqa.eval_collab_domain_adaptation import (
    build_isolated_subprocess_args,
    build_mas,
    build_task_manager,
    collab_report_run_dir,
    compute_metrics,
    ensure_dir,
    run_tasks,
)
from scripts.medmcqa.eval_collab_multidomain_global import (
    build_fever_task,
    dedupe_tasks,
    load_jsonl_rows,
    merge_eval_split,
    normalize_pddl_test_jsonl_rows,
    parse_domains,
    parse_eval_splits,
    _raise_if_legacy_alfworld_gamefiles,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ALFWorld / PDDL / FEVER：与 multidomain_global 相同载入方式，empty memory 无训练基线。"
    )
    p.add_argument(
        "--dataset_family",
        type=str,
        choices=["alfworld", "pddl", "fever", "all"],
        default="all",
        help="all 表示按顺序跑 alfworld → pddl → fever。",
    )
    p.add_argument("--alfworld_domains", type=str, default="bathroom,bedroom,kitchen,living")
    p.add_argument("--alfworld_subset_dir", type=str, default="data/alfworld/collab_subsets/v3_s")
    p.add_argument("--alfworld_merged_eval_dir", type=str, default="")
    p.add_argument("--alfworld_game_root", type=str, default="")
    p.add_argument("--alfworld_eval_split", type=str, default="valid_seen,valid_unseen")
    p.add_argument("--fever_domains", type=str, default="A_film_tv,B_music")
    p.add_argument("--fever_test_jsonl", type=str, default="data/fever/fever_ab_test_v3.jsonl")
    p.add_argument("--pddl_domains", type=str, default="gripper,blockworld,barman,tyreworld")
    p.add_argument("--pddl_test_jsonl", type=str, default="data/pddl/test.jsonl")
    p.add_argument("--max_eval", type=int, default=None, help="每个 split 最多评测前 N 条；默认全量。")
    p.add_argument("--max_trials", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=10, help="仅写入报告元数据，与 multidomain 对齐。")
    p.add_argument("--tool_mode", choices=["search"], default="search")
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    return p


def _load_merged_eval_alfworld(
    repo_root: Path, args: argparse.Namespace
) -> tuple[dict[str, list[dict]], list[dict[str, Any]], Path, list[str]]:
    subset_dir = (repo_root / args.alfworld_subset_dir).resolve()
    domains = parse_domains(args.alfworld_domains)
    eval_splits = parse_eval_splits(args.alfworld_eval_split)
    merged_eval_dir = (
        Path(args.alfworld_merged_eval_dir).expanduser().resolve()
        if str(args.alfworld_merged_eval_dir or "").strip()
        else subset_dir
    )
    merged_eval_tasks: dict[str, list[dict]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for split_name in eval_splits:
        _path, rows, meta = merge_eval_split(subset_dir, merged_eval_dir, domains, split_name)
        if args.max_eval is not None:
            rows = rows[: int(args.max_eval)]
        _raise_if_legacy_alfworld_gamefiles(rows, where=f"eval split={split_name}")
        merged_eval_tasks[split_name] = rows
        manifest_rows.append(meta)
    return merged_eval_tasks, manifest_rows, merged_eval_dir, domains


def _load_merged_eval_pddl(repo_root: Path, args: argparse.Namespace) -> tuple[list[dict], dict[str, Any]]:
    domains = parse_domains(args.pddl_domains)
    if len(domains) < 1:
        raise ValueError("dataset_family=pddl 需要至少 1 个 --pddl_domains。")
    test_rel = str(args.pddl_test_jsonl or "").strip() or "data/pddl/test.jsonl"
    test_path = (repo_root / test_rel).resolve()
    if not test_path.is_file():
        raise FileNotFoundError(f"PDDL 测试集不存在: {test_path}")
    raw_test = load_jsonl_rows(test_path)
    merged_rows = normalize_pddl_test_jsonl_rows(raw_test)
    if not merged_rows:
        raise ValueError(
            f"PDDL 测试集未解析出有效任务: {test_path}。"
            "请检查 additional_info.subtask / game_name / problem_index。"
        )
    rows = dedupe_tasks(merged_rows)
    if args.max_eval is not None:
        rows = rows[: int(args.max_eval)]
    meta = {
        "split": "test",
        "num_tasks_raw": len(merged_rows),
        "num_tasks_dedup": len(rows),
        "source_files": [str(test_path)],
        "pddl_domains": domains,
        "pddl_test_jsonl": test_rel,
    }
    return rows, meta


def _load_merged_eval_fever(repo_root: Path, args: argparse.Namespace) -> tuple[list[dict], dict[str, Any]]:
    eval_path = (repo_root / args.fever_test_jsonl).resolve()
    if not eval_path.exists():
        raise FileNotFoundError(f"FEVER 测试文件不存在: {eval_path}")
    eval_rows_raw = load_jsonl_rows(eval_path)
    rows = [task for task in (build_fever_task(x) for x in eval_rows_raw) if task]
    rows = dedupe_tasks(rows)
    if args.max_eval is not None:
        rows = rows[: int(args.max_eval)]
    meta = {
        "split": "test",
        "num_tasks_raw": len(eval_rows_raw),
        "num_tasks_dedup": len(rows),
        "source_files": [str(eval_path)],
        "fever_domains_arg": parse_domains(args.fever_domains),
    }
    return rows, meta


def _run_eval_split(
    *,
    dataset_family: str,
    split_name: str,
    eval_tasks: list[dict],
    args: argparse.Namespace,
    memory_working_dir: str,
    log_eval: str,
    report_base: str,
) -> dict[str, Any]:
    eval_task_name = {"alfworld": "alfworld", "pddl": "pddl", "fever": "fever"}[dataset_family]
    ensure_dir(memory_working_dir)
    ensure_dir(log_eval)

    manager = build_task_manager(
        eval_task_name,
        "autogen",
        "empty",
        int(args.max_trials),
        memory_working_dir,
        log_eval,
        alfworld_game_root=str(args.alfworld_game_root or "") if dataset_family == "alfworld" else "",
    )
    manager.tasks = copy.deepcopy(eval_tasks)
    manager.mas_config.update({"silent_mas": True, "insights_topk": 1})
    manager.mem_config.update({"freeze_memory": True})

    build_mas(manager, "io", "empty", str(args.model))

    isolated_sp = build_isolated_subprocess_args(
        dataset_family=dataset_family,
        reasoning="io",
        model=str(args.model),
        max_trials=int(args.max_trials),
        alfworld_game_root=str(getattr(args, "alfworld_game_root", "") or ""),
    )
    t0 = time.perf_counter()
    rewards, _, _saved, skipped, per_task_mt = run_tasks(
        manager, 0, len(manager.tasks), str(args.tool_mode), alfworld_subprocess_args=isolated_sp
    )
    wall = time.perf_counter() - t0
    metrics = compute_metrics(rewards, per_task_mt)
    return {
        "dataset_family": dataset_family,
        "split": split_name,
        "memory_type": "empty",
        "mas_type": "autogen",
        "model": str(args.model),
        "report_base": report_base,
        "memory_working_dir": memory_working_dir,
        "log_eval": log_eval,
        **metrics,
        "num_tasks": len(manager.tasks),
        "num_completed": len(rewards),
        "num_skipped": len(skipped),
        "num_success": sum(1 for r in rewards if r > 0),
        "wall_time_sec": wall,
    }


def _run_one_family(repo_root: Path, args: argparse.Namespace, dataset_family: str) -> dict[str, Any]:
    model_type = get_model_type(str(args.model))
    run_id = str(args.run_id or time.strftime("%Y%m%d_%H%M%S"))
    eval_namespace = f"{dataset_family}_collab_eval"
    mas_memory = "empty"
    base_dir = os.path.join(".db", model_type, eval_namespace, run_id, "autogen", "memory", mas_memory)
    local_root = os.path.join(base_dir, "local")
    memory_working_dir = os.path.join(local_root, "_nomem_empty_baseline")
    log_base = os.path.join("./logs", eval_namespace, run_id, "autogen", "memory", mas_memory, model_type)
    report_ts = time.strftime("%Y%m%d_%H%M%S")
    report_base = collab_report_run_dir(dataset_family=dataset_family, mas_memory=mas_memory, report_ts=report_ts)
    ensure_dir(local_root)
    ensure_dir(log_base)
    ensure_dir(report_base)

    eval_summaries: list[dict[str, Any]] = []
    merge_manifest: dict[str, Any] = {}

    if dataset_family == "alfworld":
        merged_eval_tasks, manifest_rows, merged_eval_dir, domains = _load_merged_eval_alfworld(repo_root, args)
        subset_dir = (repo_root / args.alfworld_subset_dir).resolve()
        merge_manifest = {
            "subset_dir": str(subset_dir),
            "merged_eval_dir": str(merged_eval_dir),
            "domains": domains,
            "eval_splits": list(merged_eval_tasks.keys()),
            "rows": manifest_rows,
        }
        for split_name, rows in merged_eval_tasks.items():
            log_eval = os.path.join(log_base, "eval", "empty_merged", split_name)
            eval_summaries.append(
                _run_eval_split(
                    dataset_family=dataset_family,
                    split_name=split_name,
                    eval_tasks=rows,
                    args=args,
                    memory_working_dir=memory_working_dir,
                    log_eval=log_eval,
                    report_base=report_base,
                )
            )
    elif dataset_family == "pddl":
        rows, meta = _load_merged_eval_pddl(repo_root, args)
        merge_manifest = {"rows": [meta], "pddl_domains": parse_domains(args.pddl_domains)}
        log_eval = os.path.join(log_base, "eval", "empty_merged", "test")
        eval_summaries.append(
            _run_eval_split(
                dataset_family=dataset_family,
                split_name="test",
                eval_tasks=rows,
                args=args,
                memory_working_dir=memory_working_dir,
                log_eval=log_eval,
                report_base=report_base,
            )
        )
    else:
        rows, meta = _load_merged_eval_fever(repo_root, args)
        merge_manifest = {"rows": [meta], "fever_domains": parse_domains(args.fever_domains)}
        log_eval = os.path.join(log_base, "eval", "empty_merged", "test")
        eval_summaries.append(
            _run_eval_split(
                dataset_family=dataset_family,
                split_name="test",
                eval_tasks=rows,
                args=args,
                memory_working_dir=memory_working_dir,
                log_eval=log_eval,
                report_base=report_base,
            )
        )

    manifest_path = os.path.join(report_base, "merged_eval_manifest_nomem.json")
    with open(manifest_path, "w", encoding="utf-8") as writer:
        json.dump(
            {"dataset_family": dataset_family, "run_id": run_id, **merge_manifest},
            writer,
            ensure_ascii=False,
            indent=2,
        )

    out = {
        "dataset_family": dataset_family,
        "run_id": run_id,
        "batch_size": int(args.batch_size),
        "memory_type": mas_memory,
        "mas_type": "autogen",
        "model": str(args.model),
        "reasoning": "io",
        "train_results": [],
        "eval_results": eval_summaries,
        "merged_eval_manifest_path": manifest_path,
        "note": "无训练；mas_memory=empty；每 split 一批次顺序评测。",
    }
    json_path = os.path.join(report_base, f"{dataset_family}_nomem_empty_baseline.json")
    with open(json_path, "w", encoding="utf-8") as writer:
        json.dump(out, writer, ensure_ascii=False, indent=2)

    print("", flush=True)
    print(f"=== {dataset_family} | empty memory baseline | summary ===", flush=True)
    for row in eval_summaries:
        print(
            f"  split={row['split']} | accuracy={float(row.get('accuracy', 0.0)):.4f} "
            f"avg_reward={float(row.get('avg_reward', 0.0)):.4f} "
            f"avg_steps={float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
            f"tasks={int(row.get('num_tasks', 0))} completed={int(row.get('num_completed', 0))} "
            f"skipped={int(row.get('num_skipped', 0))} success={int(row.get('num_success', 0))} | "
            f"wall_s={float(row.get('wall_time_sec', 0.0)):.2f}",
            flush=True,
        )
    print(json.dumps({"report_json": json_path}, ensure_ascii=False, indent=2), flush=True)
    return out


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    families: list[str]
    if args.dataset_family == "all":
        families = ["alfworld", "pddl", "fever"]
    else:
        families = [args.dataset_family]

    all_outputs: list[dict[str, Any]] = []
    for fam in families:
        if fam == "alfworld":
            subset_dir = (repo_root / args.alfworld_subset_dir).resolve()
            if not subset_dir.exists():
                raise FileNotFoundError(f"subset_dir 不存在: {subset_dir}")
        elif fam == "pddl":
            if not (repo_root / "data" / "pddl").exists():
                raise FileNotFoundError(f"data/pddl 不存在: {repo_root / 'data' / 'pddl'}")
        all_outputs.append(_run_one_family(repo_root, args, fam))

    if len(all_outputs) > 1:
        combo_ts = time.strftime("%Y%m%d_%H%M%S")
        combo_dir = collab_report_run_dir(dataset_family="all_nomem", mas_memory="empty", report_ts=combo_ts)
        ensure_dir(combo_dir)
        combo_path = os.path.join(combo_dir, "all_families_nomem_empty_baseline.json")
        with open(combo_path, "w", encoding="utf-8") as writer:
            json.dump({"runs": all_outputs}, writer, ensure_ascii=False, indent=2)
        print(json.dumps({"combined_report_json": combo_path}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
