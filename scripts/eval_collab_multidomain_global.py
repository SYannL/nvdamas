from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.utils import get_model_type

from scripts.eval_collab_domain_adaptation import (
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
    rebuild_memco_global_from_locals,
    rebuild_selectivemem_global_from_locals,
    reset_memco_artifacts_once,
    memrl_collab_global_ready,
    run_tasks,
)

_GM_GRAPH_MEMORY = frozenset({"memco"})
GRAPH_MEMORY_SETTINGS_CHOICES: tuple[str, ...] = ("base", "local_only", "global_only", "local_plus_global")
MEMCO_ROUTER_CHOICES: tuple[str, ...] = ("textloss",)

# BFCL family collab：默认路径与当前仓库内 data/gorilla 的 family split 输出一致；
# 可通过 CLI 覆盖（与 FEVER 多文件参数同模式）。
_BFCL_FAM_COLLAB_DIR = "data/gorilla"
_BFCL_FAM_PA = f"{_BFCL_FAM_COLLAB_DIR}/possible_answer"
_BFCL_FAM_STEM = "BFCL_v4_multi_turn_base"
_BFCL_FAM_DEFAULT_TRAIN_QUESTIONS = ",".join(
    f"{_BFCL_FAM_COLLAB_DIR}/{_BFCL_FAM_STEM}__family_{slug}__train.json"
    for slug in ("trading", "travel", "vehicle", "fs")
)
_BFCL_FAM_DEFAULT_TRAIN_ANSWERS = ",".join(
    f"{_BFCL_FAM_PA}/{_BFCL_FAM_STEM}__family_{slug}__train.json"
    for slug in ("trading", "travel", "vehicle", "fs")
)
_BFCL_FAM_DEFAULT_TEST_QUESTIONS = f"{_BFCL_FAM_COLLAB_DIR}/{_BFCL_FAM_STEM}__family__test_all.json"
_BFCL_FAM_DEFAULT_TEST_ANSWERS = f"{_BFCL_FAM_PA}/{_BFCL_FAM_STEM}__family__test_all.json"


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


def parse_bfcl_eval_splits(value: str) -> list[str]:
    splits = parse_csv(value)
    return splits or ["eval"]


def bfcl_involved_classes_domain(row: dict) -> str:
    """Stable signature: sorted API class names, joined by '+', for memory / local grouping."""
    ic = row.get("involved_classes")
    if not isinstance(ic, list) or not ic:
        return "unknown"
    parts = sorted({str(x).strip() for x in ic if str(x).strip()})
    return "+".join(parts) if parts else "unknown"


def bfcl_family_from_signature(sig: str) -> str:
    """四族划分（与 scripts/gorilla/split_bfcl_mt_four_families.py 一致）。"""
    if sig.startswith("GorillaFileSystem") or "GorillaFileSystem+" in sig or "+GorillaFileSystem" in sig:
        return "fs"
    if "TravelAPI" in sig:
        return "travel"
    if "VehicleControlAPI" in sig:
        return "vehicle"
    if "TradingBot" in sig:
        return "trading"
    return "other"


def bfcl_family_domain(row: dict) -> str:
    return bfcl_family_from_signature(bfcl_involved_classes_domain(row))


def discover_bfcl_domains(rows: list[dict]) -> list[str]:
    keys = {bfcl_involved_classes_domain(r) for r in rows}
    keys.discard("")
    return sorted(keys)


def bfcl_split_train_test(
    rows: list[dict],
    *,
    train_ratio: float,
    domains_filter: frozenset[str] | None,
) -> tuple[list[dict], list[dict]]:
    """
    Per involved_classes domain: sort by id, first floor(train_ratio * n) -> train, rest -> test.
    Deterministic, no randomness. Single-sample domains go entirely to train.
    """
    tr = float(train_ratio)
    if tr <= 0 or tr >= 1:
        raise ValueError("bfcl_train_ratio 必须在 (0, 1) 内，例如 0.8 表示每类前 80% 训练、后 20% 测试。")
    work = list(rows)
    if domains_filter is not None:
        work = [r for r in work if bfcl_involved_classes_domain(r) in domains_filter]
    by_dom: dict[str, list[dict]] = defaultdict(list)
    for r in work:
        by_dom[bfcl_involved_classes_domain(r)].append(r)
    train_all: list[dict] = []
    test_all: list[dict] = []
    for dom in sorted(by_dom.keys()):
        group = sorted(by_dom[dom], key=lambda x: str(x.get("id", "")))
        n = len(group)
        if n <= 0:
            continue
        if n == 1:
            train_all.extend(group)
            continue
        n_train = int(tr * n)
        if n_train <= 0:
            n_train = 1
        if n_train >= n:
            n_train = n - 1
        train_all.extend(group[:n_train])
        test_all.extend(group[n_train:])
    return train_all, test_all


def load_bfcl_question_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_bfcl_answers_by_id(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            bid = str(obj.get("id", "")).strip()
            if bid:
                by_id[bid] = obj
    return by_id


def load_bfcl_family_collab_from_paths(
    *,
    family_slugs: list[str],
    train_question_paths: list[Path],
    train_answer_paths: list[Path],
    test_question_path: Path,
    test_answer_path: Path,
    require_test: bool = True,
) -> tuple[dict[str, dict], dict[str, list[dict]], list[dict], list[str]]:
    """
    按显式路径加载：train 与 --bfcl_family_domains 逐项对齐；合并 test 题 + 答案。
    返回 (answers_by_id, train_by_family, test_rows, source_paths)。
    require_test=False 时跳过测试集文件与行（与 --skip_eval 仅训不评配合）。
    """
    n = len(family_slugs)
    if len(train_question_paths) != n or len(train_answer_paths) != n:
        raise ValueError(
            f"BFCL family：--bfcl_family_domains 共 {n} 项，须与 "
            f"--bfcl_family_train_questions（{len(train_question_paths)} 个文件）及 "
            f"--bfcl_family_train_answers（{len(train_answer_paths)} 个文件）一一对应。"
        )
    answers_by_id: dict[str, dict] = {}
    train_by_family: dict[str, list[dict]] = {}
    source_paths: list[str] = []
    for slug, qf, af in zip(family_slugs, train_question_paths, train_answer_paths):
        if not qf.is_file():
            raise FileNotFoundError(f"BFCL family 训练题不存在: {qf}")
        if not af.is_file():
            raise FileNotFoundError(f"BFCL family 训练答案不存在: {af}")
        source_paths.append(str(qf.resolve()))
        source_paths.append(str(af.resolve()))
        answers_by_id.update(load_bfcl_answers_by_id(af))
        rows = load_bfcl_question_rows(qf)
        for r in rows:
            got = bfcl_family_domain(r)
            if got != slug:
                raise ValueError(
                    f"BFCL family 文件与签名不一致: 该 train 文件对应 domain={slug!r} 但 id={r.get('id')!r} 归为 {got!r}。"
                )
            rid = str(r.get("id", "")).strip()
            if rid not in answers_by_id:
                raise KeyError(f"family train 题 id={rid!r} 在对应答案文件中不存在: {af}")
        train_by_family[slug] = rows

    if not require_test:
        return answers_by_id, train_by_family, [], source_paths

    if not test_question_path.is_file():
        raise FileNotFoundError(f"BFCL family 测试题不存在: {test_question_path}")
    if not test_answer_path.is_file():
        raise FileNotFoundError(f"BFCL family 测试答案不存在: {test_answer_path}")
    source_paths.append(str(test_question_path.resolve()))
    source_paths.append(str(test_answer_path.resolve()))
    answers_by_id.update(load_bfcl_answers_by_id(test_answer_path))
    test_rows = load_bfcl_question_rows(test_question_path)
    for r in test_rows:
        rid = str(r.get("id", "")).strip()
        if rid not in answers_by_id:
            raise KeyError(f"family test 题 id={rid!r} 无 ground_truth 答案条目: {test_answer_path}")
    return answers_by_id, train_by_family, test_rows, source_paths


def build_bfcl_mt_task(row: dict, answers_by_id: dict[str, dict], memory_domain: str) -> dict:
    bid = str(row.get("id", "")).strip()
    if not bid:
        raise ValueError("BFCL 行缺少 id")
    ans = answers_by_id.get(bid)
    if not ans or "ground_truth" not in ans:
        raise KeyError(f"BFCL possible_answer 缺少 id={bid} 的 ground_truth")
    entry = {k: row[k] for k in ("id", "question", "initial_config", "involved_classes") if k in row}
    return {
        "env_name": "bfcl_mt",
        "bfcl_id": bid,
        "bfcl_shard_domain": str(memory_domain).strip(),
        "bfcl_entry": entry,
        "bfcl_ground_truth": ans["ground_truth"],
        "memco_domain": "bfcl_mt",
    }


def _resolve_graph_memory_common(args: argparse.Namespace, *, shared_global_dir: str) -> dict[str, Any]:
    if args.mas_memory != "memco":
        return {}

    memco_policy = _resolve_memco_dataset_policy(args)
    memco_router = str(args.memco_router or "").strip().lower() or "textloss"
    memco_settings = str(args.memco_settings or "").strip() or "local_plus_global"
    memco_promotion_threshold = float(args.memco_promotion_threshold) if args.memco_promotion_threshold is not None else 0.35

    config: dict[str, Any] = {
        "memco_dynamic_graph": bool(args.memco_dynamic_graph),
        "memco_repo_root": str(args.memco_repo_root or "").strip(),
        "memco_router": memco_router,
        "memco_settings": memco_settings,
        "memco_promotion_threshold": memco_promotion_threshold,
        "memco_shared_global_dir": shared_global_dir,
        "memco_use_textgrad": bool(args.memco_use_textgrad),
        "memco_textgrad_engine": str(args.memco_textgrad_engine or "").strip(),
        "memco_dataset_policy": memco_policy,
    }
    if memco_policy == "fever":
        config["memco_fever_policy"] = "adaptive_cache"
    elif memco_policy == "pddl":
        config["memco_pddl_policy"] = "action_guard"
    if bool(args.memco_enable_overlay):
        config["memco_enable_overlay"] = True
    return config


def _resolve_memco_dataset_policy(args: argparse.Namespace) -> str:
    family = str(getattr(args, "dataset_family", "") or "").strip().lower()
    memory = str(getattr(args, "mas_memory", "") or "").strip().lower()
    if memory != "memco":
        return ""
    if family == "fever":
        return "fever"
    if family in {"pddl", "pddl_2"}:
        return "pddl"
    if family == "alfworld":
        return "alfworld"
    return "default"


def _use_memco_fever_policy(args: argparse.Namespace) -> bool:
    return _resolve_memco_dataset_policy(args) == "fever"


def _use_memco_pddl_policy(args: argparse.Namespace) -> bool:
    return _resolve_memco_dataset_policy(args) == "pddl"


def _resolve_memskill_common(args: argparse.Namespace) -> dict[str, Any]:
    if args.mas_memory != "memskill":
        return {}

    config: dict[str, Any] = {}

    def add_str(arg_name: str, config_name: str | None = None) -> None:
        value = str(getattr(args, arg_name, "") or "").strip()
        if value:
            config[config_name or arg_name] = value

    def add_int(arg_name: str, config_name: str | None = None) -> None:
        value = getattr(args, arg_name, None)
        if value is not None:
            config[config_name or arg_name] = int(value)

    add_str("memskill_controller")
    add_str("memskill_checkpoint_path")
    add_str("memskill_operation_bank_path")
    add_str("memskill_ppo_repo_path")
    add_str("memskill_ppo_device")
    add_str("memskill_ppo_controller_source")
    add_str("memskill_state_encoder")
    add_str("memskill_op_encoder")
    add_int("memskill_max_ops")
    add_int("memskill_top_k")
    add_int("memskill_retrieve_top_k")
    add_int("memskill_action_top_k")

    if bool(getattr(args, "memskill_finalize_local", False)):
        config["memskill_finalize_local"] = True
        config["memskill_collect_replay"] = True
    if bool(getattr(args, "memskill_require_ppo", False)):
        config["memskill_require_ppo"] = True
        config["require_ppo"] = True
    if bool(getattr(args, "memskill_train_controller", False)):
        config["memskill_train_controller"] = True
    if bool(getattr(args, "memskill_skip_noop", False)):
        config["memskill_skip_noop"] = True
    if bool(getattr(args, "memskill_expose_skill_notes", False)):
        config["memskill_expose_skill_notes"] = True
    if getattr(args, "memskill_finalize_rebuild", None) is not None:
        config["memskill_finalize_rebuild"] = bool(args.memskill_finalize_rebuild)
    if getattr(args, "memskill_use_flash_attn", None) is not None:
        config["memskill_use_flash_attn"] = bool(args.memskill_use_flash_attn)
    return config


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
        "memco_domain": "fever",
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
    premerged_src = subset_dir / f"merged__{split_name}.json"
    domain_sources = [subset_dir / f"{domain}__{split_name}.json" for domain in domains]
    if not all(src.exists() for src in domain_sources) and premerged_src.exists():
        with premerged_src.open("r", encoding="utf-8") as reader:
            data = json.load(reader)
        if not isinstance(data, list):
            raise ValueError(f"merged eval 文件格式错误（应为 list）: {premerged_src}")
        rows = dedupe_tasks(copy.deepcopy(data))
        merged_dir.mkdir(parents=True, exist_ok=True)
        out_path = merged_dir / f"merged__{split_name}.json"
        if out_path.resolve() != premerged_src.resolve():
            with out_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)
        meta = {
            "split": split_name,
            "output_file": str(out_path),
            "num_tasks_raw": len(data),
            "num_tasks_dedup": len(rows),
            "source_files": [str(premerged_src)],
            "source_mode": "premerged",
        }
        return str(out_path), rows, meta

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
        "source_mode": "per_domain",
    }
    return str(out_path), deduped, meta


def build_scienceworld_task(row: dict) -> dict:
    """Normalize a ScienceWorld task row for the collab eval pipeline."""
    task = dict(row)
    task.setdefault("env_name", "scienceworld")
    task.setdefault("memco_domain", "scienceworld")
    return task


def build_scienceworld_family_task(row: dict, *, domain: str | None = None, split: str | None = None) -> dict:
    """Normalize current ScienceWorld family-domain rows for the collab eval pipeline."""
    task = build_scienceworld_task(row)
    domain_value = str(
        task.get("scienceworld_domain")
        or task.get("sw_domain")
        or task.get("subset_group")
        or domain
        or ""
    ).strip()
    family_value = str(
        task.get("scienceworld_family")
        or task.get("sw_family")
        or domain_value
        or task.get("sw_task_name", "")
    ).strip()
    task["env_name"] = "scienceworld"
    task["task_type"] = "scienceworld"
    task["dataset_family"] = "scienceworld"
    task["memco_domain"] = "scienceworld"
    if domain_value:
        task["scienceworld_domain"] = domain_value
    if family_value:
        task["scienceworld_family"] = family_value
    if split:
        task.setdefault("scienceworld_split", split)
    return task


def _dataset_eval_title(dataset_family: str) -> str:
    return {
        "alfworld": "ALFWorld",
        "fever": "FEVER",
        "pddl": "PDDL",
        "pddl_2": "PDDL_2",
        "amabench": "AMA-Bench",
        "bfcl_mt": "BFCL multi_turn_base",
        "scienceworld": "ScienceWorld",
    }.get(dataset_family, dataset_family.replace("_", " ").title())


def _report_file_stem(dataset_family: str) -> str:
    safe_family = re.sub(r"[^a-z0-9_]+", "_", str(dataset_family or "").strip().lower()).strip("_")
    return f"{safe_family or 'dataset'}_multidomain_global_eval"


def compute_family_metrics(
    *,
    dataset_family: str,
    rewards: list[float],
    dones: list[bool],
    per_task_mt: list[dict[str, float]] | None,
) -> dict[str, float]:
    metrics = compute_metrics(rewards, per_task_mt)
    if dataset_family not in {"pddl_2", "scienceworld"}:
        return metrics
    total = len(dones) if len(dones) == len(rewards) else len(rewards)
    if dataset_family == "scienceworld" and per_task_mt is not None and len(per_task_mt) == len(rewards):
        partial_success = sum(
            1
            for row in per_task_mt
            if float(row.get("best_score", row.get("final_score", 0.0)) or 0.0) > 0
        )
        # ScienceWorld subprocess crashes are appended as failed attempts. If the worker
        # crashed before saving task metrics, count that episode as score 0 rather than
        # dropping it from score averages.
        final_scores = [
            max(float((row or {}).get("final_score", 0.0) or 0.0), 0.0)
            for row in per_task_mt
        ]
        best_scores = [
            max(float((row or {}).get("best_score", (row or {}).get("final_score", 0.0)) or 0.0), 0.0)
            for row in per_task_mt
        ]
        metrics["avg_final_score"] = sum(final_scores) / total if total else 0.0
        metrics["global_avg_score"] = metrics["avg_final_score"]
        metrics["avg_best_score"] = sum(best_scores) / total if total else 0.0
        metrics["global_avg_best_score"] = metrics["avg_best_score"]
    else:
        partial_success = sum(1 for r in rewards if r > 0)
    full_success = sum(1 for d in dones if d)
    metrics["partial_progress_rate"] = partial_success / total if total else 0.0
    metrics["num_partial_success"] = float(partial_success)
    metrics["full_success_rate"] = full_success / total if total else 0.0
    metrics["accuracy"] = metrics["full_success_rate"]
    if per_task_mt is not None and len(per_task_mt) == len(dones):
        success_steps = [
            float(row["trajectory_steps"])
            for row, done in zip(per_task_mt, dones)
            if done and isinstance(row, dict) and "trajectory_steps" in row
        ]
        if success_steps:
            metrics["avg_trajectory_steps_success"] = sum(success_steps) / len(success_steps)
    return metrics


def _count_partial_success(
    *,
    dataset_family: str,
    rewards: list[float],
    per_task_mt: list[dict[str, float]] | None,
) -> int:
    if dataset_family == "scienceworld" and per_task_mt is not None and len(per_task_mt) == len(rewards):
        return sum(
            1
            for row in per_task_mt
            if float(row.get("best_score", row.get("final_score", 0.0)) or 0.0) > 0
        )
    return sum(1 for r in rewards if r > 0)


def _weighted_global_avg_score(rows: list[dict[str, Any]]) -> float | None:
    weighted_sum = 0.0
    total_weight = 0
    for row in rows:
        if "global_avg_score" not in row and "avg_final_score" not in row:
            continue
        try:
            score = float(row.get("global_avg_score", row.get("avg_final_score", 0.0)) or 0.0)
            weight = int(row.get("num_completed", 0) or 0)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weighted_sum += score * weight
        total_weight += weight
    return (weighted_sum / total_weight) if total_weight > 0 else None


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
        if dataset_title in {"PDDL_2", "ScienceWorld"}:
            print(
                f"  split={row['split']} scope={row['memory_scope']} | "
                f"full_success_rate={float(row.get('full_success_rate', row.get('accuracy', 0.0))):.4f} "
                f"partial_progress_rate={float(row.get('partial_progress_rate', 0.0)):.4f} "
                f"global_avg_score={float(row.get('global_avg_score', row.get('avg_final_score', 0.0))):.4f} "
                f"avg_best_score={float(row.get('global_avg_best_score', row.get('avg_best_score', 0.0))):.4f} "
                f"avg_reward={float(row.get('avg_reward', 0.0)):.4f} "
                f"avg_steps={float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
                f"tasks={int(row.get('num_tasks', 0))} "
                f"completed={int(row.get('num_completed', 0))} "
                f"skipped={int(row.get('num_skipped', 0))} "
                f"full_success={int(row.get('num_success', 0))} "
                f"partial_progress={int(row.get('num_partial_success', 0) or 0)} | "
                f"wall_s={float(row.get('wall_time_sec', 0.0)):.2f}",
                flush=True,
            )
        else:
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
            if dataset_title in {"PDDL_2", "ScienceWorld"}:
                print(
                    f"  domain={row['domain']} | "
                    f"full_success_rate={float(row.get('full_success_rate', row.get('accuracy', 0.0))):.4f} "
                    f"partial_progress_rate={float(row.get('partial_progress_rate', 0.0)):.4f} "
                    f"global_avg_score={float(row.get('global_avg_score', row.get('avg_final_score', 0.0))):.4f} "
                    f"avg_best_score={float(row.get('global_avg_best_score', row.get('avg_best_score', 0.0))):.4f} "
                    f"avg_reward={float(row.get('avg_reward', 0.0)):.4f} "
                    f"avg_steps={float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
                    f"tasks={int(row.get('num_tasks', 0))} "
                    f"completed={int(row.get('num_completed', 0))} "
                    f"skipped={int(row.get('num_skipped', 0))} "
                    f"full_success={int(row.get('num_success', 0))} "
                    f"partial_progress={int(row.get('num_partial_success', 0) or 0)} | "
                    f"wall_s={float(row.get('wall_time_sec', 0.0)):.2f}",
                    flush=True,
                )
            else:
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
ALFWorld + memrl（与同 pipeline 下 memco 一样：显式子集、训练/评测 split、max_train/max_eval 等；
memrl 不需要 --memco_*；依赖见仓库根 requirements-memrl.txt）:

  cd /path/to/nvdamasgm && python scripts/eval_collab_multidomain_global.py \
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

  cd /path/to/nvdamasgm && python scripts/eval_collab_multidomain_global.py \
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

_BFCL_MT_FAMILY_GMEMORY_EXAMPLE = r"""
BFCL 四族 + g-memory（默认 train/test 路径即 argparse 默认值；可先跑 scripts/gorilla/split_bfcl_mt_four_families.py）:

  cd /path/to/nvdamasgm && python scripts/eval_collab_multidomain_global.py \
    --dataset_family bfcl_mt \
    --bfcl_use_family_collab_split \
    --mas_type autogen \
    --mas_memory g-memory \
    --reasoning io \
    --model gpt-4o-mini \
    --run_id "${RUN_ID}" \
    --max_trials 30 \
    --bfcl_train_limit_per_domain 5 \
    --skip_eval \
    --batch_size 1 \
    --tool_mode search \
    --reset_memory

更长任务可调高步数（例如 --max_trials 96）。仅训每域 5 条且暂不评时用上面 --bfcl_train_limit_per_domain 与 --skip_eval。

自定义文件时与 FEVER 一样传路径，例如:
  --bfcl_family_domains trading,travel,vehicle,fs \
  --bfcl_family_train_questions 'path/a.json,path/b.json,...' \
  --bfcl_family_train_answers 'ans/a.json,...' \
  --bfcl_family_test_questions path/test_q.json \
  --bfcl_family_test_answers path/test_a.json
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多 domain 协作训练（local）+ global 评测：ALFWorld / AmaBench / FEVER / PDDL / BFCL multi_turn_base。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ALFWORLD_MEMRL_EXAMPLE + _BFCL_MT_FAMILY_GMEMORY_EXAMPLE,
    )
    parser.add_argument(
        "--dataset_family",
        type=str,
        choices=[
            "alfworld",
            "amabench",
            "fever",
            "pddl",
            "pddl_2",
            "bfcl_mt",
            "scienceworld",
        ],
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
        "--eval_global_only_once",
        action="store_true",
        help=(
            "Graph memory eval: only run the merged test split once against shared global memory. "
            "This avoids repeating the same mixed test set once per local domain."
        ),
    )
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
        "--sw_domains",
        "--scienceworld_domains",
        dest="sw_domains",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="ScienceWorld：task id major-number domain 列表，逗号分隔。",
    )
    parser.add_argument(
        "--sw_subset_dir",
        "--scienceworld_subset_dir",
        dest="sw_subset_dir",
        type=str,
        default="data/ScienceWorld/collab_subsets/v4_id_grouped",
        help="ScienceWorld collab subset 目录（含 {domain}__train.json 和 merged__test.json）。",
    )
    parser.add_argument(
        "--sw_test_json",
        "--scienceworld_test_json",
        dest="sw_test_json",
        type=str,
        default="",
        help="ScienceWorld 测试集 JSON；留空则用 sw_subset_dir/merged__test.json。",
    )
    parser.add_argument(
        "--bfcl_domains",
        type=str,
        default="auto",
        help=(
            "BFCL：按 involved_classes 稳定签名分组（排序后 API 名用 '+' 连接，如 GorillaFileSystem+TwitterAPI）。"
            "填 auto 或留空则从数据中自动列出全部签名；否则为逗号子集，须与数据中的签名完全一致。"
        ),
    )
    parser.add_argument(
        "--bfcl_train_ratio",
        type=float,
        default=0.8,
        help="BFCL：每个签名组内按 id 字典序划分；前该比例进训练集，余下进测试集（确定性，非随机）。",
    )
    parser.add_argument(
        "--bfcl_questions_jsonl",
        type=str,
        default="data/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_base.json",
    )
    parser.add_argument(
        "--bfcl_answers_jsonl",
        type=str,
        default="data/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json",
    )
    parser.add_argument(
        "--bfcl_eval_split",
        type=str,
        default="eval",
        help="BFCL 仅评测集；逗号多项时对同一批任务重复跑（报告多行），默认 eval。",
    )
    parser.add_argument(
        "--bfcl_use_family_collab_split",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "使用「多文件」四族训练 + 单文件合并测试（路径由 --bfcl_family_train_* / --bfcl_family_test_* 指定，"
            "与 FEVER 的 fever_train_jsonl / fever_test_jsonl 同模式；默认指向 split_bfcl_mt_four_families.py 产出）。"
            "开启后忽略 --bfcl_train_ratio；local domain 为 --bfcl_family_domains。"
        ),
    )
    parser.add_argument(
        "--bfcl_family_domains",
        type=str,
        default="trading,travel,vehicle,fs",
        help="local 域名列表（逗号分隔，顺序须与下面 train 文件列表逐项对齐）。",
    )
    parser.add_argument(
        "--bfcl_family_train_questions",
        type=str,
        default=_BFCL_FAM_DEFAULT_TRAIN_QUESTIONS,
        help="BFCL family：各域训练题目 JSONL 路径，逗号分隔，项数须与 --bfcl_family_domains 一致（相对仓库根）。",
    )
    parser.add_argument(
        "--bfcl_family_train_answers",
        type=str,
        default=_BFCL_FAM_DEFAULT_TRAIN_ANSWERS,
        help="BFCL family：与各域训练题一一对应的 possible_answer JSONL，逗号分隔、顺序与 train_questions 一致。",
    )
    parser.add_argument(
        "--bfcl_family_test_questions",
        type=str,
        default=_BFCL_FAM_DEFAULT_TEST_QUESTIONS,
        help="BFCL family：合并测试集题目 JSONL（单路径，相对仓库根）。",
    )
    parser.add_argument(
        "--bfcl_family_test_answers",
        type=str,
        default=_BFCL_FAM_DEFAULT_TEST_ANSWERS,
        help="BFCL family：合并测试集 possible_answer JSONL（单路径）。",
    )
    parser.add_argument(
        "--bfcl_train_limit_per_domain",
        type=int,
        default=5,
        help=(
            "仅 dataset_family=bfcl_mt：每域训练题条数上限；<=0 表示不按域截断（仍可用 --max_train）。"
            "默认 5 用于低成本试跑；全量训练可传 0 或显式大数。"
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
        default="memco",
        help=(
            "记忆类型：memco / amem / g-memory / selectivemem / memrl / empty 等。"
            "memrl 走与 g-memory 相同的多域 global 合并（add_memory_from_peer），无需 memco 开关。"
        ),
    )
    parser.add_argument("--reasoning", type=str, default="io")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--max_trials",
        type=int,
        default=30,
        help="每 task 最大步数。",
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--tool_mode", choices=["search"], default="search")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="不跑合并评测阶段（eval_splits 置空；BFCL family 可与 require_test=False 配合不再读测试集文件）。",
    )
    parser.add_argument("--reset_memory", action="store_true")
    parser.add_argument("--memco_dynamic_graph", action="store_true")
    parser.add_argument("--memco_repo_root", type=str, default="")
    parser.add_argument(
        "--memco_router",
        type=str,
        default="",
        choices=MEMCO_ROUTER_CHOICES,
    )
    parser.add_argument(
        "--memco_settings",
        type=str,
        default="",
        choices=GRAPH_MEMORY_SETTINGS_CHOICES,
    )
    parser.add_argument("--memco_enable_overlay", action="store_true")
    parser.add_argument("--memco_promotion_threshold", type=float, default=None)
    parser.add_argument("--memco_use_textgrad", action="store_true")
    parser.add_argument("--memco_textgrad_engine", type=str, default="")
    parser.add_argument("--memskill_finalize_local", action="store_true")
    parser.add_argument("--memskill_controller", type=str, default="")
    parser.add_argument("--memskill_checkpoint_path", type=str, default="")
    parser.add_argument("--memskill_operation_bank_path", type=str, default="")
    parser.add_argument("--memskill_ppo_repo_path", type=str, default="")
    parser.add_argument("--memskill_ppo_device", type=str, default="")
    parser.add_argument("--memskill_ppo_controller_source", type=str, default="")
    parser.add_argument("--memskill_state_encoder", type=str, default="")
    parser.add_argument("--memskill_op_encoder", type=str, default="")
    parser.add_argument("--memskill_max_ops", type=int, default=None)
    parser.add_argument("--memskill_top_k", type=int, default=None)
    parser.add_argument("--memskill_retrieve_top_k", type=int, default=None)
    parser.add_argument("--memskill_action_top_k", type=int, default=None)
    parser.add_argument("--memskill_require_ppo", action="store_true")
    parser.add_argument("--memskill_train_controller", action="store_true")
    parser.add_argument("--memskill_skip_noop", action="store_true")
    parser.add_argument("--memskill_expose_skill_notes", action="store_true")
    parser.add_argument("--memskill_finalize_rebuild", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--memskill_use_flash_attn", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    if args.reset_memory and args.eval_only:
        raise ValueError("--reset_memory 不能和 --eval_only 同时使用。")

    if args.dataset_family == "bfcl_mt" and int(args.max_trials) < 20:
        print(
            "[bfcl_mt] 提示: 官方 BFCL 多轮每步预算常为 20；子进程/步数仍不足时可显式增大 --max_trials（如 30、64、96）。",
            flush=True,
        )

    if args.dataset_family == "bfcl_mt":
        if not args.bfcl_use_family_collab_split and not (0.0 < float(args.bfcl_train_ratio) < 1.0):
            raise ValueError("--bfcl_train_ratio 必须在 0 与 1 之间（不含端点），例如 0.8。")

    repo_root = Path(__file__).resolve().parents[1]
    bfcl_runtime: dict[str, Any] = {}
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
    elif args.dataset_family in {"pddl", "pddl_2"}:
        subset_dir = (repo_root / "data" / "pddl").resolve()
        domains = parse_domains(args.pddl_domains)
        if len(domains) < 1:
            raise ValueError("dataset_family=pddl/pddl_2 需要至少 1 个 --pddl_domains。")
        eval_splits = ["test"]
        train_task_name = args.dataset_family
        eval_task_name = args.dataset_family
    elif args.dataset_family == "fever":
        subset_dir = repo_root / "data" / "fever"
        domains = parse_domains(args.fever_domains)
        eval_splits = ["test"]
        train_task_name = "fever"
        eval_task_name = "fever"
    elif args.dataset_family == "scienceworld":
        subset_dir = (repo_root / args.sw_subset_dir).resolve()
        domains = parse_domains(args.sw_domains)
        if len(domains) < 1:
            raise ValueError("dataset_family=scienceworld 需要至少 1 个 --scienceworld_domains。")
        eval_splits = ["test"]
        train_task_name = "scienceworld"
        eval_task_name = "scienceworld"
    elif args.dataset_family == "bfcl_mt":
        bfcl_data_root = (
            repo_root / "data" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
        ).resolve()
        if args.bfcl_use_family_collab_split:
            family_slugs = parse_domains(args.bfcl_family_domains)
            if len(family_slugs) != len(set(family_slugs)):
                raise ValueError("--bfcl_family_domains 含重复项。")
            if len(family_slugs) < 1:
                raise ValueError("--bfcl_family_domains 至少 1 项。")
            train_q_paths = [(repo_root / p.strip()).resolve() for p in parse_csv(args.bfcl_family_train_questions)]
            train_a_paths = [(repo_root / p.strip()).resolve() for p in parse_csv(args.bfcl_family_train_answers)]
            test_q = (repo_root / str(args.bfcl_family_test_questions).strip()).resolve()
            test_a = (repo_root / str(args.bfcl_family_test_answers).strip()).resolve()
            if len(train_q_paths) != len(family_slugs) or len(train_a_paths) != len(family_slugs):
                raise ValueError(
                    f"--bfcl_family_train_questions / train_answers 各须有 {len(family_slugs)} 个逗号分隔路径，"
                    f"与 --bfcl_family_domains 对齐；当前为 {len(train_q_paths)} / {len(train_a_paths)}。"
                )
            require_test = not bool(getattr(args, "skip_eval", False))
            answers_by_id, train_by_fam, test_rows, src_paths = load_bfcl_family_collab_from_paths(
                family_slugs=family_slugs,
                train_question_paths=train_q_paths,
                train_answer_paths=train_a_paths,
                test_question_path=test_q,
                test_answer_path=test_a,
                require_test=require_test,
            )
            if require_test and not test_rows:
                raise ValueError("BFCL family_collab：合并测试集为空。")
            subset_dir = (test_q if require_test else train_q_paths[0]).parent.resolve()
            domains = list(family_slugs)
            n_tr_each = {s: len(train_by_fam[s]) for s in family_slugs}
            meta_q = test_q if require_test else train_q_paths[0]
            meta_a = test_a if require_test else train_a_paths[0]
            bfcl_runtime.update(
                {
                    "answers_by_id": answers_by_id,
                    "train_rows_by_family": train_by_fam,
                    "test_rows": test_rows,
                    "qpath": meta_q,
                    "apath": meta_a,
                    "bfcl_source_paths": src_paths,
                    "family_collab": True,
                    "bfcl_train_ratio": None,
                    "bfcl_family_train_question_paths": [str(p) for p in train_q_paths],
                    "bfcl_family_train_answer_paths": [str(p) for p in train_a_paths],
                    "bfcl_family_test_question_path": str(test_q),
                    "bfcl_family_test_answer_path": str(test_a),
                }
            )
            print(
                f"[bfcl_mt] family_collab: locals={domains} train_counts={n_tr_each} test_all={len(test_rows)}",
                flush=True,
            )
        else:
            subset_dir = bfcl_data_root
            qpath = (repo_root / str(args.bfcl_questions_jsonl).strip()).resolve()
            apath = (repo_root / str(args.bfcl_answers_jsonl).strip()).resolve()
            if not qpath.is_file():
                raise FileNotFoundError(f"BFCL 题目文件不存在: {qpath}")
            if not apath.is_file():
                raise FileNotFoundError(f"BFCL possible_answer 文件不存在: {apath}")
            all_rows = load_bfcl_question_rows(qpath)
            answers_by_id = load_bfcl_answers_by_id(apath)
            filtered = [r for r in all_rows if str(r.get("id", "")).strip() in answers_by_id]
            if not filtered:
                raise ValueError("BFCL：题目与 possible_answer 按 id 无交集。")
            discovered = discover_bfcl_domains(filtered)
            raw_dom = str(args.bfcl_domains or "").strip()
            if not raw_dom or raw_dom.lower() == "auto":
                domains = list(discovered)
                print(f"[bfcl_mt] auto domains by involved_classes ({len(domains)}): {domains}", flush=True)
            else:
                domains = parse_csv(raw_dom)
                unknown = sorted(set(domains) - set(discovered))
                if unknown:
                    raise ValueError(
                        f"--bfcl_domains 含未知 involved_classes 签名 {unknown}；数据中为: {discovered}"
                    )
            train_rows, test_rows = bfcl_split_train_test(
                filtered,
                train_ratio=float(args.bfcl_train_ratio),
                domains_filter=frozenset(domains),
            )
            if not test_rows and not bool(getattr(args, "skip_eval", False)):
                raise ValueError(
                    "BFCL：按 id 顺序划分后测试集为空（常见原因：各类仅 1 条样本）。"
                    "请扩充数据或接受部分签名仅参与训练；仅训练可传 --skip_eval。"
                )
            bfcl_runtime.update(
                {
                    "answers_by_id": answers_by_id,
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                    "qpath": qpath,
                    "apath": apath,
                    "bfcl_train_ratio": float(args.bfcl_train_ratio),
                    "family_collab": False,
                }
            )
        eval_splits = parse_bfcl_eval_splits(str(args.bfcl_eval_split or "eval"))
        train_task_name = "bfcl_mt"
        eval_task_name = "bfcl_mt"
    else:
        raise ValueError(f"不支持的 dataset_family: {args.dataset_family}")

    if getattr(args, "skip_eval", False):
        eval_splits = []

    memco_dataset_policy = _resolve_memco_dataset_policy(args)
    if memco_dataset_policy:
        os.environ["NV_MEMCO_DATASET_POLICY"] = memco_dataset_policy
    else:
        os.environ.pop("NV_MEMCO_DATASET_POLICY", None)
    if _use_memco_fever_policy(args):
        os.environ["NV_MEMCO_FEVER_POLICY"] = "adaptive_cache"
        cache_env = str(os.environ.get("FEVER_WIKI_CACHE", "") or "").strip()
        if not cache_env or cache_env.lower() in {"0", "false", "off", "none"}:
            os.environ["FEVER_WIKI_CACHE"] = os.path.join(os.getcwd(), ".cache", "fever_wiki_cache.json")
    else:
        os.environ.pop("NV_MEMCO_FEVER_POLICY", None)
    if _use_memco_pddl_policy(args):
        os.environ["NV_MEMCO_PDDL_POLICY"] = "action_guard"
    else:
        os.environ.pop("NV_MEMCO_PDDL_POLICY", None)

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
        elif args.dataset_family == "scienceworld":
            rows = [
                build_scienceworld_family_task(r, domain=domain, split="train")
                for r in load_subset_file(subset_dir, domain, "train")
            ]
        else:
            rows = []
        if args.dataset_family not in ("fever", "pddl", "pddl_2", "bfcl_mt"):
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

    if args.dataset_family in {"pddl", "pddl_2"}:
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
                if args.dataset_family == "pddl_2":
                    row["env_name"] = "pddl_2"
                    row["task_type"] = "pddl_2"
                else:
                    row.setdefault("env_name", "pddl")
                    row.setdefault("task_type", "pddl")
            if args.max_train is not None:
                rows = rows[: int(args.max_train)]
            train_tasks_by_domain[domain] = rows

    if args.dataset_family == "bfcl_mt":
        answers_by_id = bfcl_runtime["answers_by_id"]
        lim = int(getattr(args, "bfcl_train_limit_per_domain", 0) or 0)
        if bfcl_runtime.get("family_collab"):
            for domain in domains:
                rows = list(bfcl_runtime["train_rows_by_family"][domain])
                if lim > 0:
                    rows = rows[:lim]
                if args.max_train is not None:
                    rows = rows[: int(args.max_train)]
                train_tasks_by_domain[domain] = [
                    build_bfcl_mt_task(r, answers_by_id, domain) for r in rows
                ]
        else:
            train_rows = bfcl_runtime["train_rows"]
            for domain in domains:
                rows = [r for r in train_rows if bfcl_involved_classes_domain(r) == domain]
                if lim > 0:
                    rows = rows[:lim]
                if args.max_train is not None:
                    rows = rows[: int(args.max_train)]
                train_tasks_by_domain[domain] = [
                    build_bfcl_mt_task(r, answers_by_id, bfcl_involved_classes_domain(r)) for r in rows
                ]

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
        elif args.dataset_family in {"pddl", "pddl_2"}:
            test_rel = str(args.pddl_test_jsonl or "").strip() or "data/pddl/test.jsonl"
            test_path = (repo_root / test_rel).resolve()
            if not test_path.is_file():
                raise FileNotFoundError(f"PDDL 测试集不存在: {test_path}")
            raw_test = load_jsonl_rows(test_path)
            merged_rows = normalize_pddl_test_jsonl_rows(raw_test)
            for row in merged_rows:
                row["env_name"] = args.dataset_family
                row["task_type"] = args.dataset_family
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
        elif args.dataset_family == "fever":
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
        elif args.dataset_family == "bfcl_mt":
            answers_by_id = bfcl_runtime["answers_by_id"]
            test_rows = bfcl_runtime["test_rows"]
            qpath = bfcl_runtime["qpath"]
            apath = bfcl_runtime["apath"]
            if bfcl_runtime.get("family_collab"):
                rows = [
                    build_bfcl_mt_task(r, answers_by_id, bfcl_family_domain(r)) for r in test_rows
                ]
                src_list = list(bfcl_runtime.get("bfcl_source_paths") or [str(qpath), str(apath)])
                split_note = "four_family_train_files_plus_test_all"
                ratio_meta = bfcl_runtime.get("bfcl_train_ratio")
            else:
                rows = [
                    build_bfcl_mt_task(r, answers_by_id, bfcl_involved_classes_domain(r)) for r in test_rows
                ]
                src_list = [str(qpath), str(apath)]
                split_note = "per_involved_classes_sorted_id"
                ratio_meta = bfcl_runtime.get("bfcl_train_ratio")
            rows = dedupe_tasks(rows)
            merged_eval_dir.mkdir(parents=True, exist_ok=True)
            out_path = merged_eval_dir / f"merged__{split_name}.json"
            with out_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)
            meta = {
                "split": split_name,
                "output_file": str(out_path),
                "num_tasks_raw": len(test_rows),
                "num_tasks_dedup": len(rows),
                "source_files": src_list,
                "bfcl_domains": domains,
                "bfcl_train_ratio": ratio_meta,
                "bfcl_split": split_note,
            }
        elif args.dataset_family == "scienceworld":
            sw_test_rel = str(args.sw_test_json or "").strip()
            using_external_sw_test = bool(sw_test_rel)
            if sw_test_rel:
                sw_test_path = (repo_root / sw_test_rel).resolve()
            else:
                sw_test_path = (subset_dir / "merged__test.json").resolve()
            if not sw_test_path.is_file():
                raise FileNotFoundError(f"ScienceWorld 测试集不存在: {sw_test_path}")
            with sw_test_path.open("r", encoding="utf-8") as reader:
                raw_test = json.load(reader)
            rows = [
                build_scienceworld_family_task(r, split="test")
                for r in (raw_test if isinstance(raw_test, list) else [])
            ]
            rows = dedupe_tasks(rows)
            # Do not overwrite the curated subset when a one-off external eval JSON is used.
            sw_merged_eval_dir = Path(report_base) if using_external_sw_test else merged_eval_dir
            sw_merged_eval_dir.mkdir(parents=True, exist_ok=True)
            out_path = sw_merged_eval_dir / "merged__test.json"
            with out_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)
            meta = {
                "split": split_name,
                "output_file": str(out_path),
                "num_tasks_raw": len(raw_test) if isinstance(raw_test, list) else 0,
                "num_tasks_dedup": len(rows),
                "source_files": [str(sw_test_path)],
                "scienceworld_domains": domains,
                "scienceworld_subset_dir": str(subset_dir),
            }
        else:
            raise ValueError(f"merged eval 不支持的 dataset_family: {args.dataset_family}")
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
    graph_memory_prefix = "memco"
    graph_dynamic_graph = bool(graph_memory_common.get(f"{graph_memory_prefix}_dynamic_graph", False))
    graph_settings_value = str(
        graph_memory_common.get(f"{graph_memory_prefix}_settings", "local_plus_global") or "local_plus_global"
    )
    graph_promotion_threshold = float(
        graph_memory_common.get(f"{graph_memory_prefix}_promotion_threshold", 0.35) or 0.35
    )

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
        reset_memco_artifacts_once(
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
            if args.dataset_family in {"pddl", "pddl_2"}:
                prefix = "pddl_2_domain" if args.dataset_family == "pddl_2" else "pddl_domain"
                task_name = f"{prefix}_{domains[domain_idx]}"
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
            manager.mem_config.update({**graph_memory_common, **memskill_common})
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
            rewards, dones, saved_messages, skipped, per_task_mt = run_tasks(
                manager, 0, len(manager.tasks), args.tool_mode, alfworld_subprocess_args=isolated_sp
            )
            saved_messages_by_domain[domain] = list(saved_messages or [])
            wall = time.perf_counter() - t0
            persist_fn = getattr(getattr(manager.mas, "meta_memory", None), "persist_entity_graph", None)
            if callable(persist_fn):
                persist_fn()
            if args.mas_memory == "memskill" and bool(getattr(args, "memskill_finalize_local", False)):
                finalize_fn = getattr(getattr(manager.mas, "meta_memory", None), "finalize_training", None)
                if callable(finalize_fn):
                    finalize_report = finalize_fn(
                        saved_messages=saved_messages,
                        source_id=domain,
                        train_controller=bool(getattr(args, "memskill_train_controller", False)),
                    )
                else:
                    finalize_report = {"error": "memskill finalize_training not available"}
            else:
                finalize_report = None
            train_results.append(
                {
                    "domain": domain,
                    "graph_memory_settings": train_graph_settings if args.mas_memory in _GM_GRAPH_MEMORY else None,
                    "memskill_finalize": finalize_report,
                    **compute_family_metrics(
                        dataset_family=args.dataset_family,
                        rewards=rewards,
                        dones=dones,
                        per_task_mt=per_task_mt,
                    ),
                    "num_tasks": len(manager.tasks),
                    "num_completed": len(rewards),
                    "num_skipped": len(skipped),
                    "num_success": sum(1 for d in dones if d) if args.dataset_family in {"pddl_2", "scienceworld"} else sum(1 for r in rewards if r > 0),
                    "num_partial_success": _count_partial_success(
                        dataset_family=args.dataset_family,
                        rewards=rewards,
                        per_task_mt=per_task_mt,
                    ) if args.dataset_family in {"pddl_2", "scienceworld"} else None,
                    "wall_time_sec": wall,
                }
            )
    else:
        local_dirs = [os.path.join(local_root, d) for d in domains]

    if args.mas_memory in _GM_GRAPH_MEMORY:
        if not graph_dynamic_graph:
            raise ValueError("多 domain global 构建目前要求启用对应 memory 的 dynamic_graph 开关。")
        rebuild_memco_global_from_locals(
            local_dirs=local_dirs,
            global_dir=global_dir,
            promotion_threshold=graph_promotion_threshold,
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
        graph_settings_override: str | None = None,
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
        manager.mem_config.update({"freeze_memory": True, **graph_memory_common, **memskill_common})
        graph_memory_settings = str(graph_settings_override or graph_settings_value)
        if args.mas_memory in _GM_GRAPH_MEMORY:
            apply_gm_graph_scene_config(
                manager,
                owner_scene,
                settings=graph_memory_settings,
                freeze=True,
            )
        eval_mem = build_mas(manager, args.reasoning, args.mas_memory, args.model)
        if args.mas_memory not in _GM_GRAPH_MEMORY:
            # Non-MemCo memories use local memory as base and explicitly attach global retriever.
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
        if isolated_sp is not None and args.mas_memory not in _GM_GRAPH_MEMORY:
            isolated_sp["use_global_retriever"] = True
            isolated_sp["global_dir"] = global_dir
        t0 = time.perf_counter()
        rewards, dones, saved_messages, skipped, per_task_mt = run_tasks(
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
            **compute_family_metrics(
                dataset_family=args.dataset_family,
                rewards=rewards,
                dones=dones,
                per_task_mt=per_task_mt,
            ),
            "num_tasks": len(manager.tasks),
            "num_completed": len(rewards),
            "num_skipped": len(skipped),
            "num_success": sum(1 for d in dones if d) if args.dataset_family in {"pddl_2", "scienceworld"} else sum(1 for r in rewards if r > 0),
            "num_partial_success": _count_partial_success(
                dataset_family=args.dataset_family,
                rewards=rewards,
                per_task_mt=per_task_mt,
            ) if args.dataset_family in {"pddl_2", "scienceworld"} else None,
            "wall_time_sec": wall,
        }

    if args.eval_global_only_once:
        for split_name in eval_splits:
            eval_results.append(
                eval_one(
                    split_name=split_name,
                    memory_scope="global_only",
                    memory_dir=global_dir,
                    owner_scene="global",
                    graph_settings_override="global_only",
                )
            )
    else:
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

    global_avg_score = _weighted_global_avg_score(eval_results) if args.dataset_family in {"scienceworld"} else None
    output = {
        "dataset_family": args.dataset_family,
        "run_id": run_id,
        "batch_size": int(args.batch_size),
        "domains": domains,
        "global_only_eval": bool(args.eval_global_only_once),
        "eval_injection_mode": "global_only_once" if args.eval_global_only_once else "local_plus_global_per_domain",
        "expected_eval_result_count": len(eval_splits) if args.eval_global_only_once else len(eval_splits) * len(domains),
        "memory_type": args.mas_memory,
        "merged_eval_manifest_path": merge_manifest_path,
        "train_results": train_results,
        "eval_results": eval_results,
    }
    if global_avg_score is not None:
        output["global_avg_score"] = global_avg_score
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
        if global_avg_score is not None:
            writer.write(f"Global average score: {global_avg_score:.4f}\n\n")
        writer.write("## Eval Results\n\n")
        _use_partial_table = args.dataset_family in {"pddl_2", "scienceworld"}
        _include_score_col = args.dataset_family in {"scienceworld"}
        if _use_partial_table:
            if _include_score_col:
                writer.write("| Split | Memory Scope | Full Success Rate | Partial Progress Rate | Global Avg Score | Avg Best Score | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Full Success | Partial Progress | Wall Time(s) |\n")
                writer.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            else:
                writer.write("| Split | Memory Scope | Full Success Rate | Partial Progress Rate | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Full Success | Partial Progress | Wall Time(s) |\n")
                writer.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        else:
            writer.write("| Split | Memory Scope | Accuracy | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Success | Wall Time(s) |\n")
            writer.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in eval_results:
            if _use_partial_table:
                score_cell = (
                    f"{float(row.get('global_avg_score', row.get('avg_final_score', 0.0))):.4f} | "
                    f"{float(row.get('global_avg_best_score', row.get('avg_best_score', 0.0))):.4f} | "
                    if _include_score_col
                    else ""
                )
                writer.write(
                    f"| {row['split']} | {row['memory_scope']} | {float(row.get('full_success_rate', row.get('accuracy', 0.0))):.4f} | "
                    f"{float(row.get('partial_progress_rate', 0.0)):.4f} | {score_cell}"
                    f"{float(row.get('avg_reward', 0.0)):.4f} | "
                    f"{float(row.get('avg_trajectory_steps', 0.0)):.2f} | {int(row.get('num_tasks', 0))} | "
                    f"{int(row.get('num_completed', 0))} | {int(row.get('num_skipped', 0))} | "
                    f"{int(row.get('num_success', 0))} | {int(row.get('num_partial_success', 0) or 0)} | "
                    f"{float(row.get('wall_time_sec', 0.0)):.2f} |\n"
                )
            else:
                writer.write(
                    f"| {row['split']} | {row['memory_scope']} | {float(row.get('accuracy', 0.0)):.4f} | "
                    f"{float(row.get('avg_reward', 0.0)):.4f} | {float(row.get('avg_trajectory_steps', 0.0)):.2f} | "
                    f"{int(row.get('num_tasks', 0))} | "
                    f"{int(row.get('num_completed', 0))} | {int(row.get('num_skipped', 0))} | "
                    f"{int(row.get('num_success', 0))} | {float(row.get('wall_time_sec', 0.0)):.2f} |\n"
                )
        if train_results:
            writer.write("\n## Train Results (Per Domain Local)\n\n")
            if _use_partial_table:
                if _include_score_col:
                    writer.write("| Domain | Full Success Rate | Partial Progress Rate | Global Avg Score | Avg Best Score | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Full Success | Partial Progress | Wall Time(s) |\n")
                    writer.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                else:
                    writer.write("| Domain | Full Success Rate | Partial Progress Rate | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Full Success | Partial Progress | Wall Time(s) |\n")
                    writer.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            else:
                writer.write("| Domain | Accuracy | Avg Reward | Avg Steps | Tasks | Completed | Skipped | Success | Wall Time(s) |\n")
                writer.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in train_results:
                if _use_partial_table:
                    score_cell = (
                        f"{float(row.get('global_avg_score', row.get('avg_final_score', 0.0))):.4f} | "
                        f"{float(row.get('global_avg_best_score', row.get('avg_best_score', 0.0))):.4f} | "
                        if _include_score_col
                        else ""
                    )
                    writer.write(
                        f"| {row['domain']} | {float(row.get('full_success_rate', row.get('accuracy', 0.0))):.4f} | "
                        f"{float(row.get('partial_progress_rate', 0.0)):.4f} | {score_cell}"
                        f"{float(row.get('avg_reward', 0.0)):.4f} | "
                        f"{float(row.get('avg_trajectory_steps', 0.0)):.2f} | {int(row.get('num_tasks', 0))} | "
                        f"{int(row.get('num_completed', 0))} | {int(row.get('num_skipped', 0))} | "
                        f"{int(row.get('num_success', 0))} | {int(row.get('num_partial_success', 0) or 0)} | "
                        f"{float(row.get('wall_time_sec', 0.0)):.2f} |\n"
                    )
                else:
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
