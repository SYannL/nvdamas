from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tasks.utils import get_model_type
from scripts.memskill.memskill_assets import prepare_memskill_assets

from scripts.medmcqa.eval_collab_domain_adaptation import (
    CONFIG,
    GPTChat,
    EmbeddingFunc,
    build_isolated_subprocess_args,
    build_mas,
    build_task_manager,
    collab_report_run_dir,
    compute_metrics,
    ensure_dir,
    module_map,
    rebuild_graph_memory2_global_from_locals,
    rebuild_selectivemem_global_from_locals,
    reset_graph_memory2_artifacts_once,
    memrl_collab_global_ready,
    run_tasks,
)

# GraphMemory3 reuses GM2 persistence / shared global layout; treat like GM2 in this script.
_GM_GRAPH_MEMORY = frozenset({"graph_memory2", "graph_memory3"})
GM2_RETRIEVAL_MODE_CHOICES: tuple[str, ...] = (
    "lightweight",
    "query",
    "phasee_compat",
    "phasee_policy",
    "phasee_action",
    "hybrid_policy",
    "hybrid_repair",
    "lightweight_repair",
    "graph_policy",
    "graph_policy_rerank",
    "graph_policy_feedback",
    "graph_policy_candidate",
    "graph_policy_quality",
)
GRAPH_MEMORY_SETTINGS_CHOICES: tuple[str, ...] = ("base", "local_only", "global_only", "local_plus_global")
GM3_ROUTER_CHOICES: tuple[str, ...] = ("textloss",)


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


def _warn_legacy_graph_arg(memory_type: str, legacy_flag: str, new_flag: str, value: object) -> None:
    print(
        f"[config] {memory_type} received legacy {legacy_flag}={value!r}; prefer {new_flag}.",
        flush=True,
    )


def _resolve_graph_memory_common(args: argparse.Namespace, *, shared_global_dir: str) -> dict[str, Any]:
    if args.mas_memory == "graph_memory2":
        config: dict[str, Any] = {
            "gm2_dynamic_graph": bool(args.gm2_dynamic_graph),
            "gm2_repo_root": str(args.gm2_repo_root or "").strip(),
            "gm2_retrieval_mode": str(args.gm2_retrieval_mode or "graph_policy").strip(),
            "gm2_settings": str(args.gm2_settings or "local_plus_global").strip(),
            "gm2_promotion_threshold": float(args.gm2_promotion_threshold),
            "gm2_shared_global_dir": shared_global_dir,
        }
        if bool(args.gm2_enable_overlay):
            config["gm2_enable_overlay"] = True
        return config

    if args.mas_memory != "graph_memory3":
        return {}

    gm3_dynamic_graph = bool(args.gm3_dynamic_graph)
    if not gm3_dynamic_graph and bool(args.gm2_dynamic_graph):
        gm3_dynamic_graph = True
        _warn_legacy_graph_arg(args.mas_memory, "--gm2_dynamic_graph", "--gm3_dynamic_graph", True)

    gm3_repo_root = str(args.gm3_repo_root or "").strip()
    if not gm3_repo_root and str(args.gm2_repo_root or "").strip():
        gm3_repo_root = str(args.gm2_repo_root).strip()
        _warn_legacy_graph_arg(args.mas_memory, "--gm2_repo_root", "--gm3_repo_root", gm3_repo_root)

    gm3_router = str(args.gm3_router or "").strip().lower()
    legacy_mode = str(args.gm2_retrieval_mode or "graph_policy").strip().lower()
    if not gm3_router and legacy_mode != "graph_policy":
        gm3_router = "textloss"
        _warn_legacy_graph_arg(args.mas_memory, "--gm2_retrieval_mode", "--gm3_router", legacy_mode)
    gm3_router = gm3_router or "textloss"

    gm3_settings = str(args.gm3_settings or "").strip()
    legacy_settings = str(args.gm2_settings or "local_plus_global").strip()
    if not gm3_settings and legacy_settings != "local_plus_global":
        gm3_settings = legacy_settings
        _warn_legacy_graph_arg(args.mas_memory, "--gm2_settings", "--gm3_settings", legacy_settings)
    gm3_settings = gm3_settings or "local_plus_global"

    gm3_enable_overlay = bool(args.gm3_enable_overlay)
    if not gm3_enable_overlay and bool(args.gm2_enable_overlay):
        gm3_enable_overlay = True
        _warn_legacy_graph_arg(args.mas_memory, "--gm2_enable_overlay", "--gm3_enable_overlay", True)

    gm3_promotion_threshold = args.gm3_promotion_threshold
    if gm3_promotion_threshold is None and float(args.gm2_promotion_threshold) != 0.35:
        gm3_promotion_threshold = float(args.gm2_promotion_threshold)
        _warn_legacy_graph_arg(
            args.mas_memory,
            "--gm2_promotion_threshold",
            "--gm3_promotion_threshold",
            gm3_promotion_threshold,
        )
    if gm3_promotion_threshold is None:
        gm3_promotion_threshold = 0.35

    config = {
        "gm3_dynamic_graph": gm3_dynamic_graph,
        "gm3_repo_root": gm3_repo_root,
        "gm3_router": gm3_router,
        "gm3_settings": gm3_settings,
        "gm3_promotion_threshold": float(gm3_promotion_threshold),
        "gm3_shared_global_dir": shared_global_dir,
        "gm3_use_textgrad": bool(args.gm3_use_textgrad),
        "gm3_textgrad_engine": str(args.gm3_textgrad_engine or "").strip(),
    }
    if gm3_enable_overlay:
        config["gm3_enable_overlay"] = True
    return config


def _resolve_memskill_common(args: argparse.Namespace) -> dict[str, Any]:
    if args.mas_memory != "memskill":
        return {}
    config: dict[str, Any] = {}
    if args.memskill_finalize_local:
        config.update(
            {
                "memskill_finalize_local": True,
                "memskill_collect_replay": True,
                "memskill_finalize_rebuild": True,
            }
        )
    if args.memskill_controller:
        config["memskill_controller"] = args.memskill_controller
    if args.memskill_operation_bank_path:
        config["memskill_operation_bank_path"] = args.memskill_operation_bank_path
    if args.memskill_checkpoint_path:
        config["memskill_checkpoint_path"] = args.memskill_checkpoint_path
    if args.memskill_ppo_repo_path:
        config["memskill_ppo_repo_path"] = args.memskill_ppo_repo_path
    if args.memskill_state_encoder:
        config["memskill_state_encoder"] = args.memskill_state_encoder
    if args.memskill_op_encoder:
        config["memskill_op_encoder"] = args.memskill_op_encoder
    if args.memskill_ppo_device:
        config["memskill_ppo_device"] = args.memskill_ppo_device
    if args.memskill_action_top_k is not None:
        config["memskill_action_top_k"] = args.memskill_action_top_k
    if args.memskill_require_ppo:
        config["memskill_require_ppo"] = True
    if args.memskill_retrieve_top_k is not None:
        config["memskill_retrieve_top_k"] = args.memskill_retrieve_top_k
    if args.memskill_construction_top_k is not None:
        config["memskill_construction_top_k"] = args.memskill_construction_top_k
    if args.memskill_chunk_chars is not None:
        config["memskill_chunk_chars"] = args.memskill_chunk_chars
    if args.memskill_train_controller:
        config["memskill_train_controller"] = True
    return config


def _coerce_saved_message_rows(rows: list[Any]) -> list[Any]:
    try:
        from mas.memory.common import MASMessage  # type: ignore
    except Exception:
        MASMessage = None  # type: ignore

    out: list[Any] = []
    for row in rows or []:
        if MASMessage is not None and isinstance(row, MASMessage):
            out.append(row)
        elif MASMessage is not None and isinstance(row, dict):
            try:
                out.append(MASMessage.from_dict(row))
            except Exception:
                continue
        else:
            out.append(row)
    return out


def finalize_memskill_local_memory(
    *,
    args: argparse.Namespace,
    domain: str,
    local_dir: str,
    saved_messages: list[Any],
    memskill_common: dict[str, Any],
) -> dict[str, Any] | None:
    if args.mas_memory != "memskill" or not args.memskill_finalize_local:
        return None
    _reasoning_cls, mem_cls = module_map(args.reasoning, args.mas_memory)
    local_mem = mem_cls(
        namespace=args.mas_memory,
        global_config={
            "working_dir": local_dir,
            **memskill_common,
            "memskill_finalize_local": True,
            "memskill_collect_replay": True,
        },
        llm_model=GPTChat(model_name=args.model),
        embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
    )
    finalize = getattr(local_mem, "finalize_training", None)
    if not callable(finalize):
        return None
    return finalize(
        saved_messages=_coerce_saved_message_rows(saved_messages),
        source_id=domain,
        train_controller=bool(args.memskill_train_controller),
        rebuild_memory=True,
    )


def merge_memskill_global_from_local_memories(
    *,
    args: argparse.Namespace,
    domains: list[str],
    local_root: str,
    global_mem: Any,
    memskill_common: dict[str, Any],
) -> bool:
    if args.mas_memory != "memskill" or not args.memskill_finalize_local:
        return False
    add_items = getattr(global_mem, "add_memory_items_from_peer", None)
    if not callable(add_items):
        return False
    for domain in domains:
        local_dir = os.path.join(local_root, domain)
        _reasoning_cls, mem_cls = module_map(args.reasoning, args.mas_memory)
        local_mem = mem_cls(
            namespace=args.mas_memory,
            global_config={
                "working_dir": local_dir,
                **memskill_common,
                "freeze_memory": True,
            },
            llm_model=GPTChat(model_name=args.model),
            embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        )
        export_items = getattr(local_mem, "export_memory_items", None)
        if callable(export_items):
            add_items(export_items(), source_id=domain)
        add_failures = getattr(global_mem, "add_failure_cases_from_peer", None)
        export_failures = getattr(local_mem, "export_failure_cases", None)
        if callable(add_failures) and callable(export_failures):
            add_failures(export_failures(), source_id=domain)
    return True


def pddl_domain_train_jsonl(repo_root: Path, domain: str, override_path: Path | None) -> Path:
    if override_path is not None:
        return override_path
    return (repo_root / "data" / "pddl" / f"pddl_domain_{domain}.jsonl").resolve()


def load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pddl_row_game_name(row: dict) -> str:
    gn = row.get("game_name")
    if gn is not None and str(gn).strip():
        return str(gn).strip().lower()
    add = row.get("additional_info")
    if isinstance(add, dict):
        st = add.get("subtask")
        if st is not None and str(st).strip():
            return str(st).strip().lower()
    return ""


def normalize_pddl_test_jsonl_rows(rows: list[dict]) -> list[dict]:
    """
    ``data/pddl/test.jsonl`` 风格：可有 ``additional_info.subtask``，可无顶层 ``game_name``/``problem_index``。
    无 ``problem_index`` 时按文件顺序对每个 ``game_name`` 递增编号，与 pddlgym 题序一致。
    """
    per_game_next: dict[str, int] = {}
    out: list[dict] = []
    for raw in rows:
        row = copy.deepcopy(raw)
        gn = _pddl_row_game_name(row)
        if not gn:
            continue
        row["game_name"] = gn
        if row.get("problem_index") is not None:
            try:
                row["problem_index"] = int(row["problem_index"])
            except (TypeError, ValueError):
                idx = per_game_next.get(gn, 0)
                row["problem_index"] = idx
                per_game_next[gn] = idx + 1
        else:
            idx = per_game_next.get(gn, 0)
            row["problem_index"] = idx
            per_game_next[gn] = idx + 1
        if "difficulty" not in row or row.get("difficulty") is None:
            row["difficulty"] = ""
        else:
            row["difficulty"] = str(row["difficulty"])
        row.setdefault("env_name", "pddl")
        row.setdefault("task_type", "pddl")
        out.append(row)
    return out


def build_fever_task(row: dict) -> dict:
    claim = str(row.get("claim", "")).strip()
    label = str(row.get("label", "")).strip()
    if not claim or not label:
        return {}
    task = {
        "task": claim,
        "answer": label,
        "env_name": "fever",
    }
    domain = str(row.get("ab_domain", "")).strip()
    if domain:
        task["ab_domain"] = domain
    return task


def split_fever_train_by_domain(rows: list[dict], domains: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {domain: [] for domain in domains}
    domain_set = set(domains)
    for row in rows:
        domain = str(row.get("ab_domain", "")).strip()
        if domain not in domain_set:
            continue
        task = build_fever_task(row)
        if task:
            grouped[domain].append(task)
    return grouped


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


# Repo layout: games live under data/alfworld/alfworld_official_042/json_2.1.1/...
# Legacy lists used data/alfworld/json_2.1.1/... which breaks once data is only materialized under 042.
_LEGACY_ALFWORLD_GAMEPATH = re.compile(r"(?:^|/)data/alfworld/json_2\.1\.1/")


def _raise_if_legacy_alfworld_gamefiles(tasks: list[dict], *, where: str) -> None:
    for i, row in enumerate(tasks):
        g = str((row.get("env_kwargs") or {}).get("gamefile") or "").replace("\\", "/").strip()
        if not g:
            continue
        if _LEGACY_ALFWORLD_GAMEPATH.search(g):
            raise ValueError(
                f"ALFWorld gamefile 仍使用旧路径（缺少 alfworld_official_042）。where={where} i={i} gamefile={g!r}。"
                "请将子集 JSON 中的路径改为 data/alfworld/alfworld_official_042/json_2.1.1/...，"
                "或对 data/alfworld/json_2.1.1 建立指向 alfworld_official_042/json_2.1.1 的符号链接。"
            )


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


def _dataset_eval_title(dataset_family: str) -> str:
    return {
        "alfworld": "ALFWorld",
        "fever": "FEVER",
        "pddl": "PDDL",
        "amabench": "AMA-Bench",
    }.get(dataset_family, dataset_family.replace("_", " ").title())


def _report_file_stem(dataset_family: str) -> str:
    safe_family = re.sub(r"[^a-z0-9_]+", "_", str(dataset_family or "").strip().lower()).strip("_")
    return f"{safe_family or 'dataset'}_multidomain_global_eval"


def _print_multidomain_run_summary(
    *,
    dataset_title: str,
    eval_results: list[dict[str, Any]],
    train_results: list[dict[str, Any]],
) -> None:
    """Same stdout layout for all dataset_family (FEVER / PDDL / ALFWorld / …)."""
    print("", flush=True)
    print(f"=== {dataset_title} Multi-domain Global Eval | summary ===", flush=True)
    print("-- Eval --", flush=True)
    for row in eval_results:
        print(
            f"  split={row['split']} scope={row['memory_scope']} | "
            f"accuracy={float(row.get('accuracy', 0.0)):.4f} "
            f"avg_reward={float(row.get('avg_reward', 0.0)):.4f} "
            f"avg_steps={float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
            f"tasks={int(row.get('num_tasks', 0))} "
            f"completed={int(row.get('num_completed', 0))} "
            f"skipped={int(row.get('num_skipped', 0))} "
            f"success={int(row.get('num_success', 0))} | "
            f"wall_s={float(row.get('wall_time_sec', 0.0)):.2f}",
            flush=True,
        )
    if train_results:
        print("-- Train (per-domain local) --", flush=True)
        for row in train_results:
            print(
                f"  domain={row['domain']} | "
                f"accuracy={float(row.get('accuracy', 0.0)):.4f} "
                f"avg_reward={float(row.get('avg_reward', 0.0)):.4f} "
                f"avg_steps={float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
                f"tasks={int(row.get('num_tasks', 0))} "
                f"completed={int(row.get('num_completed', 0))} "
                f"skipped={int(row.get('num_skipped', 0))} "
                f"success={int(row.get('num_success', 0))} | "
                f"wall_s={float(row.get('wall_time_sec', 0.0)):.2f}",
                flush=True,
            )


_ALFWORLD_MEMRL_EXAMPLE = r"""
ALFWorld + memrl（与同 pipeline 下 graph_memory3 一样：显式子集、训练/评测 split、max_train/max_eval 等；
memrl 不需要 --gm2_* / --gm3_*；依赖见仓库根 requirements-memrl.txt）:

  cd /path/to/nvdamasgm && python scripts/medmcqa/eval_collab_multidomain_global.py \
    --dataset_family alfworld \
    --alfworld_domains bathroom,bedroom,kitchen,living \
    --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
    --alfworld_game_root "" \
    --alfworld_eval_split valid_seen,valid_unseen \
    --mas_type autogen \
    --mas_memory memrl \
    --reasoning io \
    --model gpt-4o-mini \
    --run_id "${RUN_ID}" \
    --max_trials 30 \
    --max_train 10000 \
    --max_eval 10000 \
    --batch_size 1 \
    --tool_mode search \
    --reset_memory

仅评测（同一 RUN_ID；需已有 global/memrl/mem_cubes；不要与 --reset_memory 同用）:

  cd /path/to/nvdamasgm && python scripts/medmcqa/eval_collab_multidomain_global.py \
    --dataset_family alfworld \
    --alfworld_domains bathroom,bedroom,kitchen,living \
    --alfworld_subset_dir data/alfworld/collab_subsets/v3_s \
    --alfworld_game_root "" \
    --alfworld_eval_split valid_seen,valid_unseen \
    --mas_type autogen \
    --mas_memory memrl \
    --reasoning io \
    --model gpt-4o-mini \
    --run_id "${RUN_ID}" \
    --max_trials 30 \
    --max_train 10000 \
    --max_eval 10000 \
    --batch_size 1 \
    --tool_mode search \
    --eval_only
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多 domain 协作训练（local）+ global 评测：ALFWorld / AmaBench / FEVER / PDDL。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ALFWORLD_MEMRL_EXAMPLE,
    )
    parser.add_argument(
        "--dataset_family",
        type=str,
        choices=["alfworld", "amabench", "fever", "pddl"],
        default="alfworld",
    )
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
        "--fever_domains",
        type=str,
        default="A_film_tv,B_music",
        help="FEVER domain（ab_domain）列表，逗号分隔。",
    )
    parser.add_argument(
        "--fever_train_jsonl",
        type=str,
        default="data/fever/fever_ab_train_A_v3.jsonl,data/fever/fever_ab_train_B_v3.jsonl",
        help="FEVER 训练 JSONL；可用逗号分隔多个文件。",
    )
    parser.add_argument("--fever_test_jsonl", type=str, default="data/fever/fever_ab_test_v3.jsonl")
    parser.add_argument(
        "--pddl_domains",
        type=str,
        default="gripper,blockworld,barman,tyreworld",
        help="PDDL：各游戏一个 local 分区名，与 data/pddl/pddl_domain_<name>.jsonl 对应。",
    )
    parser.add_argument(
        "--pddl_train_jsonl",
        type=str,
        default="",
        help="可选：与 --pddl_domains 等长的逗号分隔训练 JSONL；留空则用默认 data/pddl/pddl_domain_<domain>.jsonl。",
    )
    parser.add_argument(
        "--pddl_test_jsonl",
        type=str,
        default="data/pddl/test.jsonl",
        help=(
            "PDDL 评测 JSONL（相对仓库根）；默认 data/pddl/test.jsonl。"
            "载入完整列表；每个 --pddl_domains 均在「该域 local + 共享 global」下跑同一批任务（不按 game_name 拆分到各域）。"
        ),
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
    parser.add_argument(
        "--mas_memory",
        type=str,
        default="graph_memory2",
        help=(
            "记忆类型：graph_memory2 / graph_memory3 / g-memory / selectivemem / memrl / empty 等。"
            "memrl 走与 g-memory 相同的多域 global 合并（add_memory_from_peer），无需 gm2/gm3 开关。"
        ),
    )
    parser.add_argument("--reasoning", type=str, default="io")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_trials", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)
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
        choices=GM2_RETRIEVAL_MODE_CHOICES,
    )
    parser.add_argument(
        "--gm2_settings",
        type=str,
        default="local_plus_global",
        choices=GRAPH_MEMORY_SETTINGS_CHOICES,
    )
    parser.add_argument("--gm2_enable_overlay", action="store_true")
    parser.add_argument("--gm2_promotion_threshold", type=float, default=0.35)
    parser.add_argument("--gm3_dynamic_graph", action="store_true")
    parser.add_argument("--gm3_repo_root", type=str, default="")
    parser.add_argument(
        "--gm3_router",
        type=str,
        default="",
        choices=GM3_ROUTER_CHOICES,
    )
    parser.add_argument(
        "--gm3_settings",
        type=str,
        default="",
        choices=GRAPH_MEMORY_SETTINGS_CHOICES,
    )
    parser.add_argument("--gm3_enable_overlay", action="store_true")
    parser.add_argument("--gm3_promotion_threshold", type=float, default=None)
    parser.add_argument("--gm3_use_textgrad", action="store_true")
    parser.add_argument("--gm3_textgrad_engine", type=str, default="")
    parser.add_argument(
        "--memskill_finalize_local",
        action="store_true",
        help=(
            "MemSkill only: after each local/domain train split, rebuild that local memory from "
            "its saved train episodes; global memory then merges finalized local memory items."
        ),
    )
    parser.add_argument(
        "--memskill_train_controller",
        action="store_true",
        help=(
            "MemSkill only: request controller training during finalize. This is opt-in and "
            "currently requires an external/configured PPO adapter or checkpoint."
        ),
    )
    parser.add_argument(
        "--memskill_controller",
        type=str,
        default="",
        choices=("", "llm", "heuristic", "ppo"),
        help="MemSkill controller selector. Empty keeps the memory backend default.",
    )
    parser.add_argument("--memskill_operation_bank_path", type=str, default="")
    parser.add_argument("--memskill_checkpoint_path", type=str, default="")
    parser.add_argument(
        "--memskill_models_dir",
        type=str,
        default="Models/memskill",
        help="MemSkill assets directory. Relative paths are resolved from the nvdamas repo root.",
    )
    parser.add_argument(
        "--memskill_auto_download_checkpoint",
        action="store_true",
        help="Download an official MemSkill PPO checkpoint from Hugging Face if the checkpoint path is missing.",
    )
    parser.add_argument("--memskill_hf_repo_id", type=str, default="XaiverZ/MemSkill")
    parser.add_argument(
        "--memskill_hf_filename",
        type=str,
        default="",
        help="Optional exact checkpoint filename inside the Hugging Face repo. Empty scans downloaded files.",
    )
    parser.add_argument(
        "--memskill_force_checkpoint",
        action="store_true",
        help="Overwrite existing Models/memskill checkpoint when downloading/training.",
    )
    parser.add_argument(
        "--memskill_train_ppo_from_scratch",
        action="store_true",
        help=(
            "Run the original MemSkill ALFWorld PPO training loop before nvdamas eval, "
            "then copy the newest controller checkpoint into --memskill_checkpoint_path."
        ),
    )
    parser.add_argument(
        "--memskill_ppo_offline_data",
        type=str,
        default="",
        help="Original MemSkill ALFWorld offline expert trajectory JSON. Empty uses <repo>/data/alfworld_train_offline.json.",
    )
    parser.add_argument(
        "--memskill_generate_offline_data",
        action="store_true",
        help="Generate --memskill_ppo_offline_data via original MemSkill alfworld_replay.py before training.",
    )
    parser.add_argument("--memskill_offline_split", type=str, default="train")
    parser.add_argument("--memskill_force_offline_data", action="store_true")
    parser.add_argument("--memskill_ppo_save_dir", type=str, default="")
    parser.add_argument("--memskill_ppo_model", type=str, default="")
    parser.add_argument("--memskill_ppo_designer_model", type=str, default="")
    parser.add_argument("--memskill_ppo_api_base", type=str, default="")
    parser.add_argument("--memskill_ppo_api_key", type=str, default="")
    parser.add_argument("--memskill_ppo_train_device", type=str, default="")
    parser.add_argument("--memskill_ppo_inner_epochs", type=int, default=100)
    parser.add_argument("--memskill_ppo_outer_epochs", type=int, default=10)
    parser.add_argument("--memskill_ppo_batch_size", type=int, default=16)
    parser.add_argument("--memskill_ppo_encode_batch_size", type=int, default=8)
    parser.add_argument("--memskill_ppo_epochs", type=int, default=2)
    parser.add_argument("--memskill_ppo_mem_top_k", type=int, default=20)
    parser.add_argument("--memskill_ppo_mem_top_k_eval", type=int, default=20)
    parser.add_argument("--memskill_ppo_reward_metric", type=str, default="llm_judge", choices=("f1", "llm_judge"))
    parser.add_argument("--memskill_ppo_enable_designer", action="store_true")
    parser.add_argument(
        "--memskill_ppo_train_backend",
        type=str,
        default="embedded",
        choices=("embedded", "subprocess"),
        help=(
            "How to run from-scratch MemSkill PPO training. 'embedded' imports the original "
            "MemSkill trainer into the nvdamas process; 'subprocess' calls original main.py."
        ),
    )
    parser.add_argument(
        "--memskill_ppo_wandb_mode",
        type=str,
        default="disabled",
        help="WANDB_MODE used for MemSkill PPO training. Default disables network logging.",
    )
    parser.add_argument(
        "--memskill_ppo_alfworld_eval_query_source",
        type=str,
        default="objective",
        choices=("objective", "first_observation"),
    )
    parser.add_argument("--memskill_ppo_pair_a_min", type=int, default=40)
    parser.add_argument("--memskill_ppo_pair_a_max", type=int, default=60)
    parser.add_argument("--memskill_ppo_pair_b_size", type=int, default=5)
    parser.add_argument("--memskill_ppo_pair_same_type_prob", type=float, default=1.0)
    parser.add_argument("--memskill_ppo_pair_chunk_size", type=int, default=2048)
    parser.add_argument("--memskill_ppo_pair_max_steps", type=int, default=50)
    parser.add_argument("--memskill_ppo_pair_b_workers", type=int, default=80)
    parser.add_argument("--memskill_ppo_designer_freq", type=int, default=1)
    parser.add_argument("--memskill_ppo_new_action_bias_steps", type=int, default=25)
    parser.add_argument("--memskill_ppo_stage_reward_fraction", type=float, default=0.25)
    parser.add_argument("--memskill_ppo_designer_reflection_cycles", type=int, default=3)
    parser.add_argument("--memskill_ppo_designer_max_changes", type=int, default=3)
    parser.add_argument("--memskill_ppo_designer_failure_window_epochs", type=int, default=100)
    parser.add_argument("--memskill_ppo_designer_failure_pool_size", type=int, default=2000)
    parser.add_argument("--memskill_ppo_designer_new_skill_hint", action="store_true")
    parser.add_argument(
        "--memskill_ppo_extra_args",
        type=str,
        default="",
        help="Extra arguments appended to original MemSkill main.py, parsed with shlex.",
    )
    parser.add_argument(
        "--memskill_ppo_repo_path",
        type=str,
        default="/workspace/MemSkill-main",
        help="MemSkill PPO adapter: path to the original MemSkill repo containing src/controller.py.",
    )
    parser.add_argument(
        "--memskill_state_encoder",
        type=str,
        default="",
        help="MemSkill PPO adapter: override state encoder from checkpoint config.",
    )
    parser.add_argument(
        "--memskill_op_encoder",
        type=str,
        default="",
        help="MemSkill PPO adapter: override operation encoder from checkpoint config.",
    )
    parser.add_argument(
        "--memskill_ppo_retriever",
        type=str,
        default="contriever",
        choices=("contriever", "dpr", "dragon"),
        help="Retriever used during PPO training (memory retrieval in ALFWorld pair episodes). Default: contriever.",
    )
    parser.add_argument(
        "--memskill_ppo_device",
        type=str,
        default="",
        help="MemSkill PPO adapter device, e.g. cuda:0 or cpu. Empty uses checkpoint/default.",
    )
    parser.add_argument(
        "--memskill_action_top_k",
        type=int,
        default=None,
        help="MemSkill PPO adapter: override number of skills selected per step.",
    )
    parser.add_argument(
        "--memskill_require_ppo",
        action="store_true",
        help="Fail instead of falling back to LLM selection if PPO checkpoint loading/selection fails.",
    )
    parser.add_argument("--memskill_retrieve_top_k", type=int, default=None)
    parser.add_argument("--memskill_construction_top_k", type=int, default=None)
    parser.add_argument(
        "--memskill_chunk_chars",
        type=int,
        default=None,
        help="MemSkill construction chunk size in chars. None keeps backend default; 0 means no split.",
    )
    args = parser.parse_args()

    if args.reset_memory and args.eval_only:
        raise ValueError("--reset_memory 不能和 --eval_only 同时使用。")

    repo_root = Path(__file__).resolve().parents[2]
    if args.dataset_family == "alfworld":
        subset_dir = (repo_root / args.alfworld_subset_dir).resolve()
        domains = parse_domains(args.alfworld_domains)
        eval_splits = parse_eval_splits(args.alfworld_eval_split)
        train_task_name = "alfworld"
        eval_task_name = "alfworld"
    elif args.dataset_family == "amabench":
        subset_dir = (repo_root / args.amabench_subset_dir).resolve()
        domains = parse_domains(args.amabench_domains)
        eval_splits = [args.amabench_eval_split]
        train_task_name = "huskyqa"
        eval_task_name = "huskyqa"
    elif args.dataset_family == "pddl":
        subset_dir = (repo_root / "data" / "pddl").resolve()
        domains = parse_domains(args.pddl_domains)
        if len(domains) < 1:
            raise ValueError("dataset_family=pddl 需要至少 1 个 --pddl_domains。")
        eval_splits = ["test"]
        train_task_name = "pddl"
        eval_task_name = "pddl"
    else:
        subset_dir = repo_root / "data" / "fever"
        domains = parse_domains(args.fever_domains)
        eval_splits = ["test"]
        train_task_name = "fever"
        eval_task_name = "fever"
    merged_eval_dir = (
        Path(args.alfworld_merged_eval_dir).expanduser().resolve()
        if args.alfworld_merged_eval_dir
        else subset_dir
    )

    if not subset_dir.exists():
        raise FileNotFoundError(f"subset_dir 不存在: {subset_dir}")

    memskill_asset_report: dict[str, Any] | None = None
    if args.mas_memory == "memskill":
        memskill_assets = prepare_memskill_assets(args, repo_root=repo_root)
        memskill_asset_report = memskill_assets.to_dict()
        training = (memskill_asset_report.get("ppo_training") or {}).get("training") if memskill_asset_report else None
        download = memskill_asset_report.get("hf_download") if memskill_asset_report else None
        failed = []
        if isinstance(download, dict) and download.get("requested") and download.get("status") == "failed":
            failed.append(f"Hugging Face checkpoint download failed: {download.get('error') or download.get('status')}")
        if isinstance(training, dict) and training.get("requested") and training.get("status") == "failed":
            failed.append(f"MemSkill PPO training failed: {training.get('error') or training.get('status')}")
        if failed:
            raise RuntimeError("; ".join(failed))

    model_type = get_model_type(args.model)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    # Keep output layout aligned with existing collab script.
    eval_namespace = f"{args.dataset_family}_collab_eval"
    base_dir = os.path.join(".db", model_type, eval_namespace, run_id, args.mas_type, "memory", args.mas_memory)
    local_root = os.path.join(base_dir, "local")
    global_dir = os.path.join(base_dir, "global")
    log_base = os.path.join("./logs", eval_namespace, run_id, args.mas_type, "memory", args.mas_memory, model_type)
    report_ts = time.strftime("%Y%m%d_%H%M%S")
    report_base = collab_report_run_dir(
        dataset_family=args.dataset_family,
        mas_memory=args.mas_memory,
        report_ts=report_ts,
    )
    ensure_dir(local_root)
    ensure_dir(global_dir)
    ensure_dir(log_base)
    ensure_dir(report_base)

    if args.eval_only and args.mas_memory == "memrl":
        if not memrl_collab_global_ready(global_dir):
            raise ValueError(
                "eval_only + mas_memory=memrl 需要已有 global 记忆。"
                f" 未检测到 {os.path.abspath(os.path.join(global_dir, 'memrl', 'mem_cubes'))}；"
                "请先跑完整训练阶段生成 global，或从其它 run 拷贝对应 global 目录。"
            )

    train_tasks_by_domain: dict[str, list[dict]] = {}
    for domain in domains:
        if args.dataset_family == "alfworld":
            rows = load_subset_file(subset_dir, domain, "train")
        elif args.dataset_family == "amabench":
            episodes = load_amabench_subset_file(subset_dir, domain, "train")
            rows = flatten_amabench_episodes_to_tasks(
                episodes,
                max_turns=int(args.amabench_max_traj_turns or 0),
            )
        else:
            rows = []
        if args.dataset_family not in ("fever", "pddl"):
            if args.max_train is not None:
                rows = rows[: int(args.max_train)]
            train_tasks_by_domain[domain] = rows

    if args.dataset_family == "alfworld":
        for dom, trows in train_tasks_by_domain.items():
            _raise_if_legacy_alfworld_gamefiles(trows, where=f"train {dom}")

    if args.dataset_family == "fever":
        fever_train_rows: list[dict] = []
        fever_train_paths = [(repo_root / item).resolve() for item in parse_csv(args.fever_train_jsonl)]
        missing_paths = [str(path) for path in fever_train_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"FEVER 训练文件不存在: {missing_paths}")
        for fever_train_path in fever_train_paths:
            fever_train_rows.extend(load_jsonl_rows(fever_train_path))
        fever_by_domain = split_fever_train_by_domain(fever_train_rows, domains)
        for domain in domains:
            rows = fever_by_domain.get(domain, [])
            if args.max_train is not None:
                rows = rows[: int(args.max_train)]
            train_tasks_by_domain[domain] = rows

    if args.dataset_family == "pddl":
        override_paths: list[Path | None] = []
        if str(args.pddl_train_jsonl or "").strip():
            override_paths = [(repo_root / item).resolve() for item in parse_csv(args.pddl_train_jsonl)]
            if len(override_paths) != len(domains):
                raise ValueError("--pddl_train_jsonl 逗号项数须与 --pddl_domains 一致。")
        else:
            override_paths = [None] * len(domains)
        for i, domain in enumerate(domains):
            path = pddl_domain_train_jsonl(repo_root, domain, override_paths[i])
            if not path.exists():
                raise FileNotFoundError(
                    f"PDDL 训练文件不存在: {path}（可先运行 scripts/pddl/split_pddl_by_gamename.py）"
                )
            rows = [copy.deepcopy(r) for r in load_jsonl_rows(path)]
            for row in rows:
                row.setdefault("env_name", "pddl")
                row.setdefault("task_type", "pddl")
            if args.max_train is not None:
                rows = rows[: int(args.max_train)]
            train_tasks_by_domain[domain] = rows

    merged_eval_tasks: dict[str, list[dict]] = {}
    merge_manifest_rows: list[dict[str, Any]] = []
    for split_name in eval_splits:
        if args.dataset_family == "alfworld":
            _path, rows, meta = merge_eval_split(subset_dir, merged_eval_dir, domains, split_name)
        elif args.dataset_family == "amabench":
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
        elif args.dataset_family == "pddl":
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
            source_files = [str(test_path)]
            rows = dedupe_tasks(merged_rows)
            merged_eval_dir.mkdir(parents=True, exist_ok=True)
            out_path = merged_eval_dir / "merged__test.json"
            with out_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)
            meta = {
                "split": split_name,
                "output_file": str(out_path),
                "num_tasks_raw": len(merged_rows),
                "num_tasks_dedup": len(rows),
                "source_files": source_files,
                "pddl_domains": domains,
                "pddl_test_jsonl": test_rel,
            }
        else:
            eval_path = (repo_root / args.fever_test_jsonl).resolve()
            if not eval_path.exists():
                raise FileNotFoundError(f"FEVER 测试文件不存在: {eval_path}")
            eval_rows_raw = load_jsonl_rows(eval_path)
            rows = [task for task in (build_fever_task(x) for x in eval_rows_raw) if task]
            rows = dedupe_tasks(rows)
            merged_eval_dir.mkdir(parents=True, exist_ok=True)
            out_path = merged_eval_dir / "merged__test.json"
            with out_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)
            meta = {
                "split": split_name,
                "output_file": str(out_path),
                "num_tasks_raw": len(eval_rows_raw),
                "num_tasks_dedup": len(rows),
                "source_files": [str(eval_path)],
            }
        if args.max_eval is not None:
            rows = rows[: int(args.max_eval)]
        if args.dataset_family == "alfworld":
            _raise_if_legacy_alfworld_gamefiles(rows, where=f"eval split={split_name}")
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

    graph_memory_common = _resolve_graph_memory_common(args, shared_global_dir=global_dir)
    memskill_common = _resolve_memskill_common(args)
    graph_memory_prefix = "gm3" if args.mas_memory == "graph_memory3" else "gm2"
    graph_dynamic_graph = bool(graph_memory_common.get(f"{graph_memory_prefix}_dynamic_graph", False))
    graph_settings_value = str(
        graph_memory_common.get(f"{graph_memory_prefix}_settings", "local_plus_global") or "local_plus_global"
    )
    graph_promotion_threshold = float(
        graph_memory_common.get(f"{graph_memory_prefix}_promotion_threshold", 0.35) or 0.35
    )
    graph_repo_root = str(graph_memory_common.get(f"{graph_memory_prefix}_repo_root", "") or "").strip()

    def apply_gm_graph_scene_config(
        manager,
        owner_scene: str,
        *,
        settings: str,
        freeze: bool,
    ) -> None:
        if args.mas_memory not in _GM_GRAPH_MEMORY:
            return
        manager.mem_config.update(
            {
                f"{graph_memory_prefix}_owner_scene": owner_scene,
                f"{graph_memory_prefix}_settings": str(settings or "local_plus_global"),
                f"{graph_memory_prefix}_freeze_memory": bool(freeze),
            }
        )

    local_dirs: list[str] = []
    if args.reset_memory and args.mas_memory in _GM_GRAPH_MEMORY and graph_dynamic_graph:
        reset_graph_memory2_artifacts_once(
            memory_dirs=[os.path.join(local_root, d) for d in domains] + [global_dir],
            owner_scenes=domains + ["global"],
            memory_namespace=args.mas_memory,
        )
    elif args.reset_memory and args.mas_memory == "memrl":
        for d in domains:
            p = os.path.join(local_root, d)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            ensure_dir(p)
        if os.path.isdir(global_dir):
            shutil.rmtree(global_dir, ignore_errors=True)
        ensure_dir(global_dir)

    train_results: list[dict[str, Any]] = []
    saved_messages_by_domain: dict[str, list[Any]] = {}
    if not args.eval_only:
        for domain_idx, domain in enumerate(domains):
            if args.dataset_family == "pddl":
                task_name = f"pddl_domain_{domains[domain_idx]}"
            else:
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
            manager.mem_config.update(graph_memory_common)
            manager.mem_config.update(memskill_common)
            train_graph_settings = graph_settings_value
            apply_gm_graph_scene_config(
                manager,
                domain,
                settings=train_graph_settings,
                freeze=False,
            )
            build_mas(manager, args.reasoning, args.mas_memory, args.model)
            isolated_sp = build_isolated_subprocess_args(
                dataset_family=args.dataset_family,
                reasoning=args.reasoning,
                model=args.model,
                max_trials=args.max_trials,
                alfworld_game_root=getattr(args, "alfworld_game_root", "") or "",
            )
            t0 = time.perf_counter()
            rewards, _, saved_messages, skipped, per_task_mt = run_tasks(
                manager, 0, len(manager.tasks), args.tool_mode, alfworld_subprocess_args=isolated_sp
            )
            saved_messages_by_domain[domain] = list(saved_messages or [])
            finalize_report = finalize_memskill_local_memory(
                args=args,
                domain=domain,
                local_dir=local_dir,
                saved_messages=saved_messages_by_domain[domain],
                memskill_common=memskill_common,
            )
            wall = time.perf_counter() - t0
            persist_fn = getattr(getattr(manager.mas, "meta_memory", None), "persist_entity_graph", None)
            if callable(persist_fn):
                persist_fn()
            train_results.append(
                {
                    "domain": domain,
                    "graph_memory_settings": train_graph_settings if args.mas_memory in _GM_GRAPH_MEMORY else None,
                    **compute_metrics(rewards, per_task_mt),
                    "num_tasks": len(manager.tasks),
                    "num_completed": len(rewards),
                    "num_skipped": len(skipped),
                    "num_success": sum(1 for r in rewards if r > 0),
                    "wall_time_sec": wall,
                    "memskill_finalize": finalize_report,
                }
            )
    else:
        local_dirs = [os.path.join(local_root, d) for d in domains]

    if args.mas_memory in _GM_GRAPH_MEMORY:
        if not graph_dynamic_graph:
            raise ValueError("多 domain global 构建目前要求启用对应 memory 的 dynamic_graph 开关。")
        rebuild_graph_memory2_global_from_locals(
            local_dirs=local_dirs,
            global_dir=global_dir,
            promotion_threshold=graph_promotion_threshold,
            gm2_repo_root=graph_repo_root,
            memory_namespace=args.mas_memory,
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
                global_config={"working_dir": global_dir, **memskill_common},
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

            merged_from_local = merge_memskill_global_from_local_memories(
                args=args,
                domains=domains,
                local_root=local_root,
                global_mem=global_mem,
                memskill_common=memskill_common,
            )
            if not merged_from_local:
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
            eval_task_name,
            args.mas_type,
            args.mas_memory,
            args.max_trials,
            memory_dir,
            log_eval,
            alfworld_game_root=args.alfworld_game_root if args.dataset_family == "alfworld" else "",
        )
        manager.tasks = eval_tasks
        manager.mas_config.update({"silent_mas": True, "insights_topk": 1})
        manager.mem_config.update({"freeze_memory": True, **graph_memory_common})
        manager.mem_config.update(memskill_common)
        graph_memory_settings = graph_settings_value
        if args.mas_memory in _GM_GRAPH_MEMORY:
            apply_gm_graph_scene_config(
                manager,
                owner_scene,
                settings=graph_memory_settings,
                freeze=True,
            )
        eval_mem = build_mas(manager, args.reasoning, args.mas_memory, args.model)
        if args.mas_memory not in _GM_GRAPH_MEMORY:
            # Non-GM2 memories use local memory as base and explicitly attach global retriever.
            _reasoning_cls, global_mem_cls = module_map(args.reasoning, args.mas_memory)
            global_retriever = global_mem_cls(
                namespace=args.mas_memory,
                global_config={
                    "working_dir": global_dir,
                    "freeze_memory": True,
                    **memskill_common,
                },
                llm_model=GPTChat(model_name=args.model),
                embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            )
            set_global = getattr(eval_mem, "set_global_retriever", None)
            if callable(set_global):
                set_global(global_retriever)
        isolated_sp = build_isolated_subprocess_args(
            dataset_family=args.dataset_family,
            reasoning=args.reasoning,
            model=args.model,
            max_trials=args.max_trials,
            alfworld_game_root=getattr(args, "alfworld_game_root", "") or "",
        )
        t0 = time.perf_counter()
        rewards, _, saved_messages, skipped, per_task_mt = run_tasks(
            manager, 0, len(manager.tasks), args.tool_mode, alfworld_subprocess_args=isolated_sp
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
            "graph_memory_settings": graph_memory_settings if args.mas_memory in _GM_GRAPH_MEMORY else None,
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
                    memory_scope=f"local+global:{domain}",
                    memory_dir=local_dir,
                    owner_scene=domain,
                )
            )

    output = {
        "dataset_family": args.dataset_family,
        "run_id": run_id,
        "batch_size": int(args.batch_size),
        "domains": domains,
        "global_only_eval": False,
        "eval_injection_mode": "local_plus_global_per_domain",
        "expected_eval_result_count": len(eval_splits) * len(domains),
        "memory_type": args.mas_memory,
        "merged_eval_manifest_path": merge_manifest_path,
        "memskill_assets": memskill_asset_report,
        "train_results": train_results,
        "eval_results": eval_results,
    }
    if args.dataset_family == "amabench":
        output["amabench_episode_outputs"] = amabench_episode_outputs

    dataset_title = _dataset_eval_title(args.dataset_family)
    report_stem = _report_file_stem(args.dataset_family)
    json_path = os.path.join(report_base, f"{report_stem}.json")
    md_path = os.path.join(report_base, f"{report_stem}.md")
    with open(json_path, "w", encoding="utf-8") as writer:
        json.dump(output, writer, ensure_ascii=False, indent=2)

    if args.dataset_family == "amabench":
        epi_jsonl = os.path.join(report_base, "amabench_episode_outputs.jsonl")
        with open(epi_jsonl, "w", encoding="utf-8") as writer:
            for row in amabench_episode_outputs:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(md_path, "w", encoding="utf-8") as writer:
        writer.write(f"# {dataset_title} Multi-domain Global Eval\n\n")
        writer.write("## Eval Results\n\n")
        writer.write("| Split | Memory Scope | Accuracy | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Success | Wall Time(s) |\n")
        writer.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in eval_results:
            writer.write(
                f"| {row['split']} | {row['memory_scope']} | {float(row.get('accuracy', 0.0)):.4f} | "
                f"{float(row.get('avg_reward', 0.0)):.4f} | {float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
                f"{int(row.get('num_tasks', 0))} | "
                f"{int(row.get('num_completed', 0))} | {int(row.get('num_skipped', 0))} | "
                f"{int(row.get('num_success', 0))} | {float(row.get('wall_time_sec', 0.0)):.2f} |\n"
            )
        if train_results:
            writer.write("\n## Train Results (Per Domain Local)\n\n")
            writer.write("| Domain | Accuracy | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Success | Wall Time(s) |\n")
            writer.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in train_results:
                writer.write(
                    f"| {row['domain']} | {float(row.get('accuracy', 0.0)):.4f} | "
                    f"{float(row.get('avg_reward', 0.0)):.4f} | {float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
                    f"{int(row.get('num_tasks', 0))} | "
                    f"{int(row.get('num_completed', 0))} | {int(row.get('num_skipped', 0))} | "
                    f"{int(row.get('num_success', 0))} | {float(row.get('wall_time_sec', 0.0)):.2f} |\n"
                )

    print(json.dumps({"report_json": json_path, "report_md": md_path}, ensure_ascii=False, indent=2))
    _print_multidomain_run_summary(
        dataset_title=dataset_title,
        eval_results=eval_results,
        train_results=train_results,
    )


if __name__ == "__main__":
    main()
