from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tasks.utils import get_model_type

from scripts.medmcqa.eval_collab_domain_adaptation import (
    CONFIG,
    GPTChat,
    EmbeddingFunc,
    build_mas,
    build_task_manager,
    compute_metrics,
    ensure_dir,
    module_map,
    rebuild_graph_memory2_global_from_locals,
    rebuild_selectivemem_global_from_locals,
    reset_graph_memory2_artifacts_once,
    run_tasks,
)


def parse_csv(value: str) -> list[str]:
    rows = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not rows:
        raise ValueError("参数不能为空。")
    return list(dict.fromkeys(rows))


def parse_domains(value: str) -> list[str]:
    return parse_csv(value)


def parse_eval_splits(value: str) -> list[str]:
    splits = parse_csv(value)
    allowed = {"valid_seen", "valid_unseen"}
    invalid = [s for s in splits if s not in allowed]
    if invalid:
        raise ValueError(f"不支持的 split: {invalid}，仅支持 {sorted(allowed)}")
    return splits


def load_subset_file(subset_dir: Path, domain: str, split_name: str) -> list[dict]:
    subset_path = subset_dir / f"{domain}__{split_name}.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"缺少子集文件: {subset_path}")
    with subset_path.open("r", encoding="utf-8") as reader:
        data = json.load(reader)
    if not isinstance(data, list):
        raise ValueError(f"子集文件格式错误（应为 list）: {subset_path}")
    return data


def load_amabench_subset_file(subset_dir: Path, domain: str, split_name: str) -> list[dict]:
    subset_path = subset_dir / f"{domain}__{split_name}.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing AmaBench subset file: {subset_path}")
    with subset_path.open("r", encoding="utf-8") as reader:
        data = json.load(reader)
    if not isinstance(data, list):
        raise ValueError(f"AmaBench subset must be a list: {subset_path}")
    return data


def _format_traj_lines(traj: list[dict], max_turns: int = 0) -> str:
    rows = traj
    if max_turns and max_turns > 0:
        rows = traj[:max_turns]
    lines: list[str] = []
    for item in rows:
        t = item.get("turn_idx")
        action = str(item.get("action", "")).strip()
        obs = str(item.get("observation", "")).strip()
        lines.append(f"[Turn {t}] Action: {action}\nObservation: {obs}")
    return "\n".join(lines)


def build_amabench_prompt(*, episode: dict, qa: dict, qa_index: int, total_qas: int, max_turns: int = 0) -> str:
    trajectory = _format_traj_lines(episode.get("trajectory") or [], max_turns=max_turns)
    return (
        "You are an offline trajectory QA assistant.\n"
        "Read the task and trajectory, then answer the single question strictly based on the episode evidence.\n\n"
        f"Episode ID: {episode.get('episode_id')}\n"
        f"Domain: {episode.get('domain')}\n"
        f"Task: {episode.get('task')}\n"
        f"Task Type: {episode.get('task_type')}\n\n"
        f"Trajectory:\n{trajectory}\n\n"
        f"Question ({qa_index + 1}/{total_qas}): {qa.get('question')}\n\n"
        "Return only one concise final answer string (no extra explanation)."
    )


def flatten_amabench_episodes_to_tasks(
    episodes: list[dict],
    *,
    max_turns: int = 0,
) -> list[dict]:
    tasks: list[dict] = []
    for ep in episodes:
        qas = ep.get("qa_pairs") or []
        if not isinstance(qas, list):
            continue
        for qidx, qa in enumerate(qas):
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question:
                continue
            task_text = build_amabench_prompt(
                episode=ep,
                qa=qa,
                qa_index=qidx,
                total_qas=len(qas),
                max_turns=max_turns,
            )
            tasks.append(
                {
                    "task": task_text,
                    "answer": answer,
                    "env_name": "huskyqa",
                    "_episode_id": ep.get("episode_id"),
                    "_domain": ep.get("domain"),
                    "_qa_idx": qidx,
                    "_qa_question": question,
                }
            )
    return tasks


def _extract_finish_answer(task_trajectory: str) -> str:
    text = str(task_trajectory or "")
    matches = re.findall(r"Finish\[(.*?)\]", text, flags=re.DOTALL)
    if not matches:
        return ""
    return str(matches[-1]).strip()


def build_amabench_episode_outputs(
    tasks: list[dict],
    saved_messages: list[Any],
) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, Any]] = {}
    for idx, task in enumerate(tasks):
        ep_id = task.get("_episode_id")
        qa_idx = int(task.get("_qa_idx", 0))
        rec = grouped.setdefault(
            ep_id,
            {"episode_id": ep_id, "answer_list": [], "reasoning_trace": []},
        )
        while len(rec["answer_list"]) <= qa_idx:
            rec["answer_list"].append("")
            rec["reasoning_trace"].append("")

        msg = saved_messages[idx] if idx < len(saved_messages) else None
        if isinstance(msg, dict):
            traj = str(msg.get("task_trajectory", ""))
        else:
            traj = str(getattr(msg, "task_trajectory", "") if msg is not None else "")
        rec["answer_list"][qa_idx] = _extract_finish_answer(traj)
        rec["reasoning_trace"][qa_idx] = traj
    return list(grouped.values())


def dedupe_tasks(tasks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in tasks:
        gamefile = ((row.get("env_kwargs") or {}).get("gamefile") or "").strip()
        key = gamefile or json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def merge_eval_split(
    subset_dir: Path,
    merged_dir: Path,
    domains: list[str],
    split_name: str,
) -> tuple[str, list[dict], dict[str, Any]]:
    merged_rows: list[dict] = []
    source_files: list[str] = []
    for domain in domains:
        src = subset_dir / f"{domain}__{split_name}.json"
        rows = load_subset_file(subset_dir, domain, split_name)
        merged_rows.extend(copy.deepcopy(rows))
        source_files.append(str(src))

    deduped = dedupe_tasks(merged_rows)
    merged_dir.mkdir(parents=True, exist_ok=True)
    out_path = merged_dir / f"merged__{split_name}.json"
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(deduped, writer, ensure_ascii=False, indent=2)
    meta = {
        "split": split_name,
        "output_file": str(out_path),
        "num_tasks_raw": len(merged_rows),
        "num_tasks_dedup": len(deduped),
        "source_files": source_files,
    }
    return str(out_path), deduped, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ALFWorld 多 domain 协作训练（local）+ global-only 评测（seen/unseen）。"
    )
    parser.add_argument("--dataset_family", type=str, choices=["alfworld", "amabench"], default="alfworld")
    parser.add_argument(
        "--alfworld_domains",
        type=str,
        default="bathroom,bedroom,kitchen,living",
        help="逗号分隔 domain，如 bathroom,bedroom,kitchen,living",
    )
    parser.add_argument("--alfworld_subset_dir", type=str, default="data/alfworld/collab_subsets/v3_s")
    parser.add_argument(
        "--alfworld_merged_eval_dir",
        type=str,
        default="",
        help="合并后的评测集目录；为空则写入 subset_dir。",
    )
    parser.add_argument("--alfworld_game_root", type=str, default="")
    parser.add_argument(
        "--alfworld_eval_split",
        type=str,
        default="valid_seen,valid_unseen",
        help="仅支持 valid_seen,valid_unseen，可逗号分隔。",
    )
    parser.add_argument("--amabench_subset_dir", type=str, default="data/amabench/collab_subsets/v3_s")
    parser.add_argument(
        "--amabench_domains",
        type=str,
        default="EMBODIED_AI,Game,OPENWORLD_QA,SOFTWARE,TEXT2SQL,WEB",
        help="AmaBench domain 列表，逗号分隔。",
    )
    parser.add_argument(
        "--amabench_eval_split",
        type=str,
        default="test",
        choices=["test"],
        help="AmaBench 仅支持 test split（作为 unseen）。",
    )
    parser.add_argument(
        "--amabench_max_traj_turns",
        type=int,
        default=0,
        help="AmaBench prompt 中最多保留的轨迹步数；0 表示全量。",
    )
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_eval", type=int, default=None)
    parser.add_argument("--mas_type", type=str, choices=["autogen", "macnet", "dylan", "strategy"], default="autogen")
    parser.add_argument("--mas_memory", type=str, default="graph_memory2")
    parser.add_argument("--reasoning", type=str, default="io")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_trials", type=int, default=30)
    parser.add_argument("--tool_mode", choices=["search"], default="search")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--reset_memory", action="store_true")
    parser.add_argument("--gm2_dynamic_graph", action="store_true")
    parser.add_argument("--gm2_repo_root", type=str, default="")
    parser.add_argument(
        "--gm2_retrieval_mode",
        type=str,
        default="graph_policy",
    )
    parser.add_argument(
        "--gm2_settings",
        type=str,
        default="local_plus_global",
        choices=["base", "local_only", "global_only", "local_plus_global"],
    )
    parser.add_argument("--gm2_promotion_threshold", type=float, default=0.35)
    args = parser.parse_args()

    if args.reset_memory and args.eval_only:
        raise ValueError("--reset_memory 不能和 --eval_only 同时使用。")

    repo_root = Path(__file__).resolve().parents[2]
    if args.dataset_family == "alfworld":
        subset_dir = (repo_root / args.alfworld_subset_dir).resolve()
        domains = parse_domains(args.alfworld_domains)
        eval_splits = parse_eval_splits(args.alfworld_eval_split)
        train_task_name = "alfworld"
    else:
        subset_dir = (repo_root / args.amabench_subset_dir).resolve()
        domains = parse_domains(args.amabench_domains)
        eval_splits = [args.amabench_eval_split]
        train_task_name = "huskyqa"
    merged_eval_dir = (
        Path(args.alfworld_merged_eval_dir).expanduser().resolve()
        if args.alfworld_merged_eval_dir
        else subset_dir
    )

    if not subset_dir.exists():
        raise FileNotFoundError(f"subset_dir 不存在: {subset_dir}")

    model_type = get_model_type(args.model)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    # Keep output layout aligned with existing collab script.
    eval_namespace = f"{args.dataset_family}_collab_eval"
    base_dir = os.path.join(".db", model_type, eval_namespace, run_id, args.mas_type, "memory", args.mas_memory)
    local_root = os.path.join(base_dir, "local")
    global_dir = os.path.join(base_dir, "global")
    log_base = os.path.join("./logs", eval_namespace, run_id, args.mas_type, "memory", args.mas_memory, model_type)
    report_base = os.path.join("./reports", "collab", time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(local_root)
    ensure_dir(global_dir)
    ensure_dir(log_base)
    ensure_dir(report_base)

    train_tasks_by_domain: dict[str, list[dict]] = {}
    for domain in domains:
        if args.dataset_family == "alfworld":
            rows = load_subset_file(subset_dir, domain, "train")
        else:
            episodes = load_amabench_subset_file(subset_dir, domain, "train")
            rows = flatten_amabench_episodes_to_tasks(
                episodes,
                max_turns=int(args.amabench_max_traj_turns or 0),
            )
        if args.max_train is not None:
            rows = rows[: int(args.max_train)]
        train_tasks_by_domain[domain] = rows

    merged_eval_tasks: dict[str, list[dict]] = {}
    merge_manifest_rows: list[dict[str, Any]] = []
    for split_name in eval_splits:
        if args.dataset_family == "alfworld":
            _path, rows, meta = merge_eval_split(subset_dir, merged_eval_dir, domains, split_name)
        else:
            merged_rows: list[dict] = []
            source_files: list[str] = []
            for domain in domains:
                src = subset_dir / f"{domain}__{split_name}.json"
                episodes = load_amabench_subset_file(subset_dir, domain, split_name)
                tasks = flatten_amabench_episodes_to_tasks(
                    episodes,
                    max_turns=int(args.amabench_max_traj_turns or 0),
                )
                merged_rows.extend(copy.deepcopy(tasks))
                source_files.append(str(src))
            rows = dedupe_tasks(merged_rows)
            merged_eval_dir.mkdir(parents=True, exist_ok=True)
            out_path = merged_eval_dir / f"merged__{split_name}.json"
            with out_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)
            meta = {
                "split": split_name,
                "output_file": str(out_path),
                "num_tasks_raw": len(merged_rows),
                "num_tasks_dedup": len(rows),
                "source_files": source_files,
            }
        if args.max_eval is not None:
            rows = rows[: int(args.max_eval)]
        merged_eval_tasks[split_name] = rows
        merge_manifest_rows.append(meta)

    merge_manifest = {
        "subset_dir": str(subset_dir),
        "merged_eval_dir": str(merged_eval_dir),
        "domains": domains,
        "eval_splits": eval_splits,
        "rows": merge_manifest_rows,
    }
    merge_manifest_path = os.path.join(report_base, "merged_eval_manifest.json")
    with open(merge_manifest_path, "w", encoding="utf-8") as writer:
        json.dump(merge_manifest, writer, ensure_ascii=False, indent=2)

    gm2_common = {
        "gm2_dynamic_graph": bool(args.gm2_dynamic_graph),
        "gm2_repo_root": str(args.gm2_repo_root or "").strip(),
        "gm2_retrieval_mode": str(args.gm2_retrieval_mode or "graph_policy").strip(),
        "gm2_promotion_threshold": float(args.gm2_promotion_threshold),
        "gm2_shared_global_dir": global_dir,
    }

    local_dirs: list[str] = []
    if args.reset_memory and args.mas_memory == "graph_memory2" and args.gm2_dynamic_graph:
        reset_graph_memory2_artifacts_once(
            memory_dirs=[os.path.join(local_root, d) for d in domains] + [global_dir],
            owner_scenes=domains + ["global"],
        )

    train_results: list[dict[str, Any]] = []
    saved_messages_by_domain: dict[str, list[Any]] = {}
    if not args.eval_only:
        for domain in domains:
            task_name = train_task_name
            local_dir = os.path.join(local_root, domain)
            local_dirs.append(local_dir)
            ensure_dir(local_dir)
            log_dir = os.path.join(log_base, "train_local", domain)
            ensure_dir(log_dir)

            manager = build_task_manager(
                task_name,
                args.mas_type,
                args.mas_memory,
                args.max_trials,
                local_dir,
                log_dir,
                alfworld_game_root=args.alfworld_game_root,
            )
            manager.tasks = copy.deepcopy(train_tasks_by_domain[domain])
            manager.mas_config.update({"silent_mas": True, "insights_topk": 1})
            manager.mem_config.update(gm2_common)
            if args.mas_memory == "graph_memory2":
                manager.mem_config.update(
                    gm2_owner_scene=domain,
                    gm2_settings="local_only",
                    gm2_freeze_memory=False,
                )
            build_mas(manager, args.reasoning, args.mas_memory, args.model)
            alfworld_sp = (
                {
                    "reasoning": args.reasoning,
                    "model": args.model,
                    "max_trials": args.max_trials,
                    "alfworld_game_root": args.alfworld_game_root,
                }
                if args.dataset_family == "alfworld"
                else None
            )
            t0 = time.perf_counter()
            rewards, _, saved_messages, skipped, per_task_mt = run_tasks(
                manager, 0, len(manager.tasks), args.tool_mode, alfworld_subprocess_args=alfworld_sp
            )
            saved_messages_by_domain[domain] = list(saved_messages or [])
            wall = time.perf_counter() - t0
            persist_fn = getattr(getattr(manager.mas, "meta_memory", None), "persist_entity_graph", None)
            if callable(persist_fn):
                persist_fn()
            train_results.append(
                {
                    "domain": domain,
                    **compute_metrics(rewards, per_task_mt),
                    "num_tasks": len(manager.tasks),
                    "num_completed": len(rewards),
                    "num_skipped": len(skipped),
                    "num_success": sum(1 for r in rewards if r > 0),
                    "wall_time_sec": wall,
                }
            )
    else:
        local_dirs = [os.path.join(local_root, d) for d in domains]

    if args.mas_memory == "graph_memory2":
        if not args.gm2_dynamic_graph:
            raise ValueError("多 domain global 构建目前要求 --gm2_dynamic_graph。")
        rebuild_graph_memory2_global_from_locals(
            local_dirs=local_dirs,
            global_dir=global_dir,
            promotion_threshold=float(args.gm2_promotion_threshold),
            gm2_repo_root=args.gm2_repo_root,
        )
    elif args.mas_memory == "selectivemem":
        rebuild_selectivemem_global_from_locals(
            local_dirs=local_dirs,
            global_dir=global_dir,
            model_name=args.model,
            snapshot_tag=None,
        )
    elif args.mas_memory == "empty":
        pass
    else:
        if args.eval_only:
            # eval_only assumes global memory artifacts already exist under global_dir.
            pass
        else:
            _reasoning_cls, global_mem_cls = module_map(args.reasoning, args.mas_memory)
            global_mem = global_mem_cls(
                namespace=args.mas_memory,
                global_config={"working_dir": global_dir},
                llm_model=GPTChat(model_name=args.model),
                embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            )
            try:
                from mas.memory.common import MASMessage  # type: ignore
            except Exception:
                MASMessage = None  # type: ignore

            def _coerce_message(x: Any) -> Any:
                if MASMessage is None:
                    return x
                if isinstance(x, MASMessage):
                    return x
                if isinstance(x, dict):
                    return MASMessage.from_dict(x)
                return x

            for domain in domains:
                for msg in saved_messages_by_domain.get(domain, []):
                    m = _coerce_message(msg)
                    add_peer = getattr(global_mem, "add_memory_from_peer", None)
                    if callable(add_peer):
                        add_peer(m, source_id=domain)
                    else:
                        global_mem.add_memory(m)
            persist_fn = getattr(global_mem, "persist_entity_graph", None)
            if callable(persist_fn):
                persist_fn()

    eval_results: list[dict[str, Any]] = []

    amabench_episode_outputs: list[dict[str, Any]] = []

    def eval_one(
        *,
        split_name: str,
        memory_scope: str,
        memory_dir: str,
        owner_scene: str,
    ) -> dict[str, Any]:
        eval_tasks = copy.deepcopy(merged_eval_tasks[split_name])
        log_eval = os.path.join(log_base, "eval", memory_scope, split_name)
        ensure_dir(log_eval)
        manager = build_task_manager(
            train_task_name,
            args.mas_type,
            args.mas_memory,
            args.max_trials,
            memory_dir,
            log_eval,
            alfworld_game_root=args.alfworld_game_root if args.dataset_family == "alfworld" else "",
        )
        manager.tasks = eval_tasks
        manager.mas_config.update({"silent_mas": True, "insights_topk": 1})
        manager.mem_config.update({"freeze_memory": True, **gm2_common})
        if args.mas_memory == "graph_memory2":
            gm2_settings = "global_only" if memory_scope == "global" else "local_only"
            manager.mem_config.update(
                gm2_owner_scene=owner_scene,
                gm2_settings=gm2_settings,
                gm2_freeze_memory=True,
            )
        build_mas(manager, args.reasoning, args.mas_memory, args.model)
        alfworld_sp = (
            {
                "reasoning": args.reasoning,
                "model": args.model,
                "max_trials": args.max_trials,
                "alfworld_game_root": args.alfworld_game_root,
            }
            if args.dataset_family == "alfworld"
            else None
        )
        t0 = time.perf_counter()
        rewards, _, saved_messages, skipped, per_task_mt = run_tasks(
            manager, 0, len(manager.tasks), args.tool_mode, alfworld_subprocess_args=alfworld_sp
        )
        wall = time.perf_counter() - t0
        if args.dataset_family == "amabench":
            amabench_episode_outputs.extend(
                [
                    {
                        "split": split_name,
                        "memory_scope": memory_scope,
                        **row,
                    }
                    for row in build_amabench_episode_outputs(manager.tasks, saved_messages)
                ]
            )
        return {
            "split": split_name,
            "memory_scope": memory_scope,
            "memory_dir": memory_dir,
            **compute_metrics(rewards, per_task_mt),
            "num_tasks": len(manager.tasks),
            "num_completed": len(rewards),
            "num_skipped": len(skipped),
            "num_success": sum(1 for r in rewards if r > 0),
            "wall_time_sec": wall,
        }

    for split_name in eval_splits:
        for domain in domains:
            local_dir = os.path.join(local_root, domain)
            eval_results.append(
                eval_one(
                    split_name=split_name,
                    memory_scope=f"local:{domain}",
                    memory_dir=local_dir,
                    owner_scene=domain,
                )
            )
        eval_results.append(
            eval_one(
                split_name=split_name,
                memory_scope="global",
                memory_dir=global_dir,
                owner_scene="global",
            )
        )

    output = {
        "dataset_family": args.dataset_family,
        "run_id": run_id,
        "domains": domains,
        "global_only_eval": False,
        "expected_eval_result_count": len(eval_splits) * (len(domains) + 1),
        "memory_type": args.mas_memory,
        "merged_eval_manifest_path": merge_manifest_path,
        "train_results": train_results,
        "eval_results": eval_results,
    }
    if args.dataset_family == "amabench":
        output["amabench_episode_outputs"] = amabench_episode_outputs

    json_path = os.path.join(report_base, "alfworld_multidomain_global_eval.json")
    md_path = os.path.join(report_base, "alfworld_multidomain_global_eval.md")
    with open(json_path, "w", encoding="utf-8") as writer:
        json.dump(output, writer, ensure_ascii=False, indent=2)

    if args.dataset_family == "amabench":
        epi_jsonl = os.path.join(report_base, "amabench_episode_outputs.jsonl")
        with open(epi_jsonl, "w", encoding="utf-8") as writer:
            for row in amabench_episode_outputs:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(md_path, "w", encoding="utf-8") as writer:
        writer.write("# ALFWorld Multi-domain Global-only Eval\n\n")
        writer.write("## Eval Results\n\n")
        writer.write("| Split | Memory Scope | Accuracy | Avg Reward | Tasks | Completed | Skipped | Success | Wall Time(s) |\n")
        writer.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in eval_results:
            writer.write(
                f"| {row['split']} | {row['memory_scope']} | {float(row.get('accuracy', 0.0)):.4f} | "
                f"{float(row.get('avg_reward', 0.0)):.4f} | {int(row.get('num_tasks', 0))} | "
                f"{int(row.get('num_completed', 0))} | {int(row.get('num_skipped', 0))} | "
                f"{int(row.get('num_success', 0))} | {float(row.get('wall_time_sec', 0.0)):.2f} |\n"
            )
        if train_results:
            writer.write("\n## Train Results (Per Domain Local)\n\n")
            writer.write("| Domain | Accuracy | Avg Reward | Tasks | Completed | Skipped | Success | Wall Time(s) |\n")
            writer.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in train_results:
                writer.write(
                    f"| {row['domain']} | {float(row.get('accuracy', 0.0)):.4f} | "
                    f"{float(row.get('avg_reward', 0.0)):.4f} | {int(row.get('num_tasks', 0))} | "
                    f"{int(row.get('num_completed', 0))} | {int(row.get('num_skipped', 0))} | "
                    f"{int(row.get('num_success', 0))} | {float(row.get('wall_time_sec', 0.0)):.2f} |\n"
                )

    print(json.dumps({"report_json": json_path, "report_md": md_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
