import os
import sys
import json
import tempfile
import time
import random
import re
import copy
import traceback
import subprocess
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from pathlib import Path

import yaml
import networkx as nx
from networkx.readwrite import json_graph

# Keep terminal output clean: we only want progress prints (and success rate).
os.environ.setdefault("NV_DAMAS_CONSOLE_LOG", "0")


def _print_train_progress(
    *,
    phase: str,
    a_done: int,
    b_done: int,
    total: int,
    gg_done: int,
    gg_total: int,
    a_succ: int | None = None,
    a_att: int | None = None,
    b_succ: int | None = None,
    b_att: int | None = None,
    a_lg_succ: int | None = None,
    a_lg_att: int | None = None,
    a_lgg_succ: int | None = None,
    a_lgg_att: int | None = None,
    b_lg_succ: int | None = None,
    b_lg_att: int | None = None,
    b_lgg_succ: int | None = None,
    b_lgg_att: int | None = None,
    endline: bool = False,
) -> None:
    """
    Single-line progress display:
    A: a/total | B: b/total | GG: g/gtotal | phase=...
    """
    total = max(1, int(total))
    gg_total = max(1, int(gg_total))
    msg = (
        f"A {a_done}/{total} | "
        f"B {b_done}/{total} | "
        f"GG {gg_done}/{gg_total} | "
        f"{phase}"
    )

    # Append 4-group success counters when provided.
    if a_succ is not None and a_att is not None and b_succ is not None and b_att is not None:
        msg += f" | A_succ {a_succ}/{a_att} | B_succ {b_succ}/{b_att}"

    if (
        a_lg_succ is not None
        and a_lg_att is not None
        and a_lgg_succ is not None
        and a_lgg_att is not None
        and b_lg_succ is not None
        and b_lg_att is not None
        and b_lgg_succ is not None
        and b_lgg_att is not None
    ):
        msg += (
            f" | A_LG {a_lg_succ}/{a_lg_att} "
            f"| A_LGG {a_lgg_succ}/{a_lgg_att} "
            f"| B_LG {b_lg_succ}/{b_lg_att} "
            f"| B_LGG {b_lgg_succ}/{b_lgg_att}"
        )
    if endline:
        print(msg, flush=True)
    else:
        # overwrite current line
        print("\r" + msg, end="", flush=True)

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mas.llm import GPTChat
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase, MemoryBankGraphMASMemory, SelectiveMemMASMemory
from mas.utils import EmbeddingFunc

from tasks.envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from tasks.mas_workflow import get_mas
from tasks.prompts import get_dataset_system_prompt, get_task_few_shots
from tasks.prompts.medmcqa_prompt import (
    medmcqa_few_shots_no_search,
    medmcqa_solver_system_prompt_no_search,
    medmcqa_few_shots_force_search,
    medmcqa_solver_system_prompt_force_search,
    medmcqa_few_shots_smart_search,
    medmcqa_solver_system_prompt_smart_search,
)
from tasks.utils import get_model_type
from mas.timing_profile import (
    append as timing_profile_append,
    enabled as timing_profile_enabled,
    print_timing_report,
)


with open("tasks/configs.yaml") as reader:
    CONFIG: dict = yaml.safe_load(reader)


@dataclass
class TaskManager:
    task_name: str
    mas_type: str
    memory_type: str
    tasks: list[dict]
    env: BaseEnv
    recorder: BaseRecorder
    mas: Any
    mas_config: dict = field(default_factory=dict)
    mem_config: dict = field(default_factory=dict)


def build_task_manager(
    task: str,
    mas_type: str,
    memory_type: str,
    max_steps: int,
    working_dir: str,
    log_dir: str,
) -> TaskManager:
    with open(CONFIG.get(task).get("env_config_path")) as reader:
        config = yaml.safe_load(reader)

    env: BaseEnv = get_env(task, config, max_steps)
    recorder: BaseRecorder = get_recorder(task, working_dir=log_dir, namespace="total_task")
    tasks: list[dict] = get_task(task, env_config=config)
    mas_workflow = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    task_manager = TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        mas_config=mas_config,
    )
    task_manager.mem_config.update(
        working_dir=working_dir,
        task_name=task,
        viz_output_dir=log_dir,
    )
    return task_manager


def _snapshot_env_mt_metrics(env: Any) -> dict[str, float]:
    """MT-Mind2Web (and similar) envs expose ``latest_metrics`` after each turn / task."""
    raw = getattr(env, "latest_metrics", None) or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k in ("ele_acc", "op_f1", "ssr", "tsr"):
        if k not in raw:
            continue
        try:
            out[k] = float(raw[k])
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def build_mas(
    task_manager: TaskManager,
    reasoning: str,
    mas_memory: str,
    llm_type: str,
) -> MASMemoryBase:
    embed_func = EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    llm_model = GPTChat(model_name=llm_type)
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func,
    )

    task_manager.mas.add_observer(task_manager.recorder)
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)
    return mas_memory_module


def _run_one_alfworld_task_worker(args_json_path: str, task_config_path: str, result_path: str) -> None:
    """Run a single ALFWorld task in isolation. Used by subprocess to survive SIGSEGV.
    Exits 0 on success, 1 on caught exception. On segfault the process dies (e.g. 139).
    """
    wall_worker0 = time.perf_counter()
    with open(args_json_path, "r", encoding="utf-8") as f:
        args = json.load(f)
    with open(task_config_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

    task_name = args["task_name"]
    working_dir = args["working_dir"]
    log_dir = args["log_dir"]
    mas_type = args["mas_type"]
    mas_memory = args["mas_memory"]
    reasoning = args["reasoning"]
    model = args["model"]
    max_trials = args["max_trials"]
    tool_mode = args["tool_mode"]
    mem_config_override = args.get("mem_config_override", {})
    mas_config_override = args.get("mas_config_override", {})
    prof_worker = timing_profile_enabled(global_config=mem_config_override)

    reward: float = 0.0
    done: bool = False
    last_saved: Any = None
    got_outcome = False
    post_error: str | None = None

    seg: dict[str, Any] = {}
    try:
        t0 = time.perf_counter()
        manager = build_task_manager(task_name, mas_type, mas_memory, max_trials, working_dir, log_dir)
        manager.tasks = [task_config]
        manager.mem_config.update(mem_config_override)
        manager.mas_config.update(mas_config_override)
        eval_mem = build_mas(manager, reasoning, mas_memory, model)
        seg["build_manager_and_mas_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if args.get("use_global_insights") and args.get("global_dir") and mas_memory == "selectivemem":
            global_retriever = SelectiveMemMASMemory(
                namespace=mas_memory,
                global_config={
                    "working_dir": args["global_dir"],
                    "freeze_memory": True,
                    "insights_only": True,
                },
                llm_model=GPTChat(model_name=model),
                embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            )
            set_global = getattr(eval_mem, "set_global_retriever", None)
            if callable(set_global):
                eval_mem.set_global_retriever(global_retriever)
        seg["global_retriever_attach_s"] = time.perf_counter() - t0

        task_id = 0
        task_manager = manager
        t0 = time.perf_counter()
        task_manager.recorder.dataset_begin()
        task_manager.recorder.task_begin(task_id, task_config)
        task_main, task_description = task_manager.mas.env.set_env(task_config)
        few_shots_num = CONFIG.get(task_manager.task_name).get("few_shots_num", 0)
        if tool_mode == "no_search":
            task_config["disable_search_tools"] = True
            few_shots = medmcqa_few_shots_no_search[:few_shots_num]
        elif tool_mode == "force_search":
            task_config["force_search_mode"] = True
            task_config["auto_first_search"] = True
            few_shots = medmcqa_few_shots_force_search[:few_shots_num]
        elif tool_mode == "smart_search":
            few_shots = medmcqa_few_shots_smart_search[:few_shots_num]
        else:
            few_shots = get_task_few_shots(
                dataset=task_manager.task_name, task_config=task_config, few_shots_num=few_shots_num
            )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
        task_config["env_ref"] = manager.env

        if tool_mode == "no_search":
            task_instruction = medmcqa_solver_system_prompt_no_search
        elif tool_mode == "force_search":
            task_instruction = medmcqa_solver_system_prompt_force_search
        elif tool_mode == "smart_search":
            task_instruction = medmcqa_solver_system_prompt_smart_search
        else:
            task_instruction = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
        for agent in task_manager.mas.agents_team.values():
            task_manager.recorder.log(f"------------ MAS Agent: {agent.name} ------------")
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))
        seg["prep_env_few_shots_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        reward, done = task_manager.mas.schedule(task_config)
        seg["schedule_s"] = time.perf_counter() - t0
        got_outcome = True

        # Post-processing: never let failures here invalidate a successful outcome.
        t0 = time.perf_counter()
        try:
            task_manager.recorder.task_end(reward, done)
        except Exception as e:
            post_error = f"recorder.task_end: {type(e).__name__}: {e}"
        try:
            last_saved = getattr(task_manager.mas.meta_memory, "last_saved_message", None)
        except Exception as e:
            post_error = post_error or f"last_saved_message: {type(e).__name__}: {e}"
        try:
            task_manager.recorder.dataset_end()
        except Exception as e:
            post_error = post_error or f"recorder.dataset_end: {type(e).__name__}: {e}"
        seg["post_recorder_s"] = time.perf_counter() - t0

    except Exception as e:
        # If we already have a valid outcome, treat this as non-fatal and keep exit_code=0
        # so parent counts success correctly. Otherwise, re-raise to preserve skip behavior.
        if not got_outcome:
            raise
        post_error = post_error or f"post_exception: {type(e).__name__}: {e}"
    finally:
        if prof_worker:
            seg.setdefault("kind", "alfworld_worker_task")
            seg["task_name"] = task_name
            seg["episode_index_1based"] = mem_config_override.get("episode_index_1based")
            seg["worker_wall_s"] = time.perf_counter() - wall_worker0
            seg["worker_post_error"] = post_error
            seg["got_outcome"] = bool(got_outcome)
            timing_profile_append(log_dir, "timing_worker_task.jsonl", seg)

    # `last_saved` may be a MASMessage instance (not JSON-serializable). Keep worker robust:
    # parent process only needs reward/done; saved_message is best-effort for debugging.
    saved_payload: Any = None
    try:
        if last_saved is None:
            saved_payload = None
        elif isinstance(last_saved, dict):
            saved_payload = last_saved
        else:
            # Avoid importing MASMessage at top-level if not needed in some envs.
            try:
                from mas.memory.common import MASMessage  # type: ignore

                if isinstance(last_saved, MASMessage):
                    saved_payload = MASMessage.to_dict(last_saved)
                else:
                    saved_payload = str(last_saved)
            except Exception:
                saved_payload = str(last_saved)
    except Exception:
        saved_payload = None

    mt_metrics = _snapshot_env_mt_metrics(task_manager.mas.env)

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reward": float(reward),
                "done": bool(done),
                "saved_message": saved_payload,
                "worker_post_error": post_error,
                "latest_metrics": mt_metrics,
            },
            f,
            ensure_ascii=False,
        )


def run_tasks(
    task_manager: TaskManager,
    start: int,
    end: int,
    tool_mode: str,
    alfworld_subprocess_args: dict[str, Any] | None = None,
    progress_hook: Callable[[int, int, int, int], None] | None = None,
) -> tuple[list[float], list[bool], list, list[dict[str, Any]], list[dict[str, float]]]:
    task_manager.recorder.dataset_begin()
    total_tasks = len(task_manager.tasks)
    rewards: list[float] = []
    dones: list[bool] = []
    saved_messages = []
    skipped_tasks: list[dict[str, Any]] = []
    per_task_mt: list[dict[str, float]] = []
    start_time = time.time()
    successes = 0
    # For ALFWorld, some tasks can hard-crash in TextWorld PDDL parsing ("tw-pddl", KeyError 'val1').
    # User requested: do not count those as attempted samples; exclude them from denominators.
    effective_total = max(0, int(end) - int(start))
    attempted = 0  # non-PDDL attempted count
    done_count = 0  # number of non-PDDL tasks processed (success/fail/other-skip)

    script_path = os.path.abspath(__file__)
    use_subprocess = alfworld_subprocess_args is not None

    def _is_pddl_crash(gamefile: Any, err: Any) -> bool:
        gf = str(gamefile or "")
        em = str(err or "")
        return (
            "tw-pddl" in gf
            or "game.tw-pddl" in gf
            or "KeyError: 'val1'" in em
            or "textworld" in em.lower() and "pddl" in em.lower()
        )

    for task_id in range(start, end):
        task_config = copy.deepcopy(task_manager.tasks[task_id])
        task_manager.mem_config["episode_index_1based"] = task_id + 1
        task_manager.mem_config["episode_index_0based"] = task_id
        progress_msg = f"[{task_manager.task_name}] Progress: {task_id + 1}/{total_tasks}"

        if use_subprocess:
            # Run in subprocess to survive SIGSEGV (e.g. TextWorld PDDL parse crash)
            worker_args = {
                "task_name": task_manager.task_name,
                "working_dir": task_manager.mem_config["working_dir"],
                "log_dir": getattr(task_manager.recorder, "working_dir", task_manager.mem_config["working_dir"]),
                "mas_type": task_manager.mas_type,
                "mas_memory": task_manager.memory_type,
                "reasoning": alfworld_subprocess_args["reasoning"],
                "model": alfworld_subprocess_args["model"],
                "max_trials": alfworld_subprocess_args["max_trials"],
                "tool_mode": tool_mode,
                "mem_config_override": {k: v for k, v in task_manager.mem_config.items() if k not in ("working_dir", "task_name")},
                "mas_config_override": dict(task_manager.mas_config),
            }
            if alfworld_subprocess_args.get("use_global_insights") and alfworld_subprocess_args.get("global_dir"):
                worker_args["use_global_insights"] = True
                worker_args["global_dir"] = alfworld_subprocess_args["global_dir"]
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(worker_args, f, ensure_ascii=False)
                args_path = f.name
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(task_config, f, ensure_ascii=False)
                task_path = f.name
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                result_path = f.name
            try:
                t_sub0 = time.perf_counter()
                proc = subprocess.run(
                    [sys.executable, script_path, "--alfworld-worker", args_path, task_path, result_path],
                    capture_output=True,
                    timeout=600,
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(script_path))),
                )
                sub_wall_s = time.perf_counter() - t_sub0
                if timing_profile_enabled(global_config=task_manager.mem_config):
                    _plog = getattr(task_manager.recorder, "working_dir", None) or task_manager.mem_config.get(
                        "working_dir", "."
                    )
                    timing_profile_append(
                        _plog,
                        "timing_parent_subprocess.jsonl",
                        {
                            "kind": "alfworld_subprocess_wall",
                            "task_name": task_manager.task_name,
                            "task_index_1based": task_id + 1,
                            "returncode": proc.returncode,
                            "wall_s": sub_wall_s,
                            "task_outcome": "ok" if proc.returncode == 0 else "subprocess_nonzero_exit",
                        },
                    )
                if proc.returncode != 0:
                    stderr_text = (proc.stderr or b"").decode(errors="replace")
                    stdout_text = (proc.stdout or b"").decode(errors="replace")
                    # Prefer tail: real traceback usually appears at the end.
                    tail = (stderr_text or stdout_text)[-4000:]
                    # Also persist full stderr/stdout for inspection.
                    try:
                        log_dir = getattr(task_manager.recorder, "working_dir", None) or task_manager.mem_config.get("working_dir")  # type: ignore[assignment]
                        if log_dir:
                            os.makedirs(log_dir, exist_ok=True)
                            crash_path = os.path.join(
                                log_dir,
                                f"subprocess_crash_task{task_id+1:04d}.log",
                            )
                            with open(crash_path, "w", encoding="utf-8") as f:
                                f.write("=== STDERR ===\n")
                                f.write(stderr_text)
                                f.write("\n\n=== STDOUT ===\n")
                                f.write(stdout_text)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Subprocess exited {proc.returncode} (crash/exception): "
                        f"{tail}"
                    )
                with open(result_path, "r", encoding="utf-8") as f:
                    res = json.load(f)
                reward, done = res["reward"], res["done"]
                last_saved = res.get("saved_message")
                rewards.append(float(reward))
                dones.append(bool(done))
                lm = res.get("latest_metrics")
                per_task_mt.append(_mt_metrics_from_saved(lm) if isinstance(lm, dict) else {})
                if last_saved is not None:
                    saved_messages.append(last_saved)
                elapsed = time.time() - start_time
                if float(reward) > 0:
                    successes += 1
                attempted += 1
                done_count += 1

                if progress_hook is not None:
                    progress_hook(done_count, effective_total, successes, attempted)
                else:
                    # Print progress (and success rate) every 10 tasks.
                    if (attempted % 10 == 0) or (task_id == end - 1):
                        rate = (successes / attempted) if attempted else 0.0
                        print(
                            f"{progress_msg} success_rate={rate:.2%} ({successes}/{attempted})",
                            flush=True,
                        )
                # Worker already logged task_begin/task_end to same log_dir
            except subprocess.TimeoutExpired as exc:
                # subprocess.run 超时：此前未写入父进程计时，这里补一条，保证每个 task 都有墙钟记录
                sub_wall_s = time.perf_counter() - t_sub0
                if timing_profile_enabled(global_config=task_manager.mem_config):
                    _plog = getattr(task_manager.recorder, "working_dir", None) or task_manager.mem_config.get(
                        "working_dir", "."
                    )
                    timing_profile_append(
                        _plog,
                        "timing_parent_subprocess.jsonl",
                        {
                            "kind": "alfworld_subprocess_wall",
                            "task_name": task_manager.task_name,
                            "task_index_1based": task_id + 1,
                            "returncode": None,
                            "wall_s": sub_wall_s,
                            "task_outcome": "timeout",
                        },
                    )
                gamefile = (task_config.get("env_kwargs") or {}).get("gamefile")
                pddl_crash = _is_pddl_crash(gamefile, exc)
                skip_info = {
                    "task_id": task_id,
                    "task_index_1based": task_id + 1,
                    "task_name": task_manager.task_name,
                    "env_name": task_config.get("env_name"),
                    "gamefile": gamefile,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "pddl_crash": bool(pddl_crash),
                }
                skipped_tasks.append(skip_info)
                # Include the crash/exception snippet in logs; otherwise we only see the gamefile.
                err_preview = (skip_info.get("error_message") or "")[:500]
                task_manager.recorder.log(
                    f"{progress_msg} Task {task_id + 1} skipped (subprocess crash/error): {gamefile}. "
                    f"{skip_info['error_type']}: {err_preview}. Continuing."
                )
                if pddl_crash:
                    # Exclude from denominators (requested).
                    effective_total = max(0, effective_total - 1)
                else:
                    attempted += 1
                    done_count += 1
                    # Keep reward/done arrays aligned with "attempted" semantics in downstream metrics.
                    rewards.append(0.0)
                    dones.append(False)
                    per_task_mt.append({})
                if progress_hook is not None:
                    progress_hook(done_count, effective_total, successes, attempted)
                else:
                    if (attempted % 10 == 0) or (task_id == end - 1):
                        rate = (successes / attempted) if attempted else 0.0
                        print(
                            f"{progress_msg} success_rate={rate:.2%} ({successes}/{attempted})",
                            flush=True,
                        )
            except RuntimeError as exc:
                # 子进程非 0 退出等：父进程已在上面写过 timing（含 subprocess_nonzero_exit）
                gamefile = (task_config.get("env_kwargs") or {}).get("gamefile")
                pddl_crash = _is_pddl_crash(gamefile, exc)
                skip_info = {
                    "task_id": task_id,
                    "task_index_1based": task_id + 1,
                    "task_name": task_manager.task_name,
                    "env_name": task_config.get("env_name"),
                    "gamefile": gamefile,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "pddl_crash": bool(pddl_crash),
                }
                skipped_tasks.append(skip_info)
                err_preview = (skip_info.get("error_message") or "")[:500]
                task_manager.recorder.log(
                    f"{progress_msg} Task {task_id + 1} skipped (subprocess crash/error): {gamefile}. "
                    f"{skip_info['error_type']}: {err_preview}. Continuing."
                )
                if pddl_crash:
                    effective_total = max(0, effective_total - 1)
                else:
                    attempted += 1
                    done_count += 1
                    rewards.append(0.0)
                    dones.append(False)
                    per_task_mt.append({})
                if progress_hook is not None:
                    progress_hook(done_count, effective_total, successes, attempted)
                else:
                    if (attempted % 10 == 0) or (task_id == end - 1):
                        rate = (successes / attempted) if attempted else 0.0
                        print(
                            f"{progress_msg} success_rate={rate:.2%} ({successes}/{attempted})",
                            flush=True,
                        )
            finally:
                for p in (args_path, task_path, result_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            sys.stdout.flush()
            continue

        t_inline0 = time.perf_counter()
        try:
            task_manager.recorder.task_begin(task_id, task_config)
            task_main, task_description = task_manager.mas.env.set_env(task_config)
            few_shots_num = CONFIG.get(task_manager.task_name).get("few_shots_num", 0)
            if tool_mode == "no_search":
                task_config["disable_search_tools"] = True
                few_shots = medmcqa_few_shots_no_search[:few_shots_num]
            elif tool_mode == "force_search":
                task_config["force_search_mode"] = True
                task_config["auto_first_search"] = True  # system runs first Search; model only reasons and Finish (or second Search)
                few_shots = medmcqa_few_shots_force_search[:few_shots_num]
            elif tool_mode == "smart_search":
                few_shots = medmcqa_few_shots_smart_search[:few_shots_num]
            else:
                few_shots = get_task_few_shots(
                    dataset=task_manager.task_name,
                    task_config=task_config,
                    few_shots_num=few_shots_num,
                )
            task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
            task_config["env_ref"] = task_manager.env

            if tool_mode == "no_search":
                task_instruction = medmcqa_solver_system_prompt_no_search
            elif tool_mode == "force_search":
                task_instruction = medmcqa_solver_system_prompt_force_search
            elif tool_mode == "smart_search":
                task_instruction = medmcqa_solver_system_prompt_smart_search
            else:
                task_instruction = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
            for agent in task_manager.mas.agents_team.values():
                task_manager.recorder.log(f"------------ MAS Agent: {agent.name} ------------")
                task_manager.recorder.log(agent.add_task_instruction(task_instruction))

            reward, done = task_manager.mas.schedule(task_config)
            task_manager.recorder.task_end(reward, done)
            rewards.append(float(reward))
            dones.append(bool(done))
            per_task_mt.append(_snapshot_env_mt_metrics(task_manager.mas.env))

            if float(reward) > 0:
                successes += 1
            attempted += 1
            done_count += 1

            last_saved = getattr(task_manager.mas.meta_memory, "last_saved_message", None)
            if last_saved is not None:
                saved_messages.append(last_saved)

            elapsed = time.time() - start_time
            if progress_hook is not None:
                progress_hook(done_count, effective_total, successes, attempted)
            else:
                if (attempted % 10 == 0) or (task_id == end - 1):
                    rate = (successes / attempted) if attempted else 0.0
                    print(
                        f"{progress_msg} success_rate={rate:.2%} ({successes}/{attempted})",
                        flush=True,
                    )
            if timing_profile_enabled(global_config=task_manager.mem_config):
                _ilog = getattr(task_manager.recorder, "working_dir", None) or task_manager.mem_config.get(
                    "working_dir", "."
                )
                timing_profile_append(
                    _ilog,
                    "timing_parent_inline_task.jsonl",
                    {
                        "kind": "run_tasks_inline_wall",
                        "task_name": task_manager.task_name,
                        "task_index_1based": task_id + 1,
                        "wall_s": time.perf_counter() - t_inline0,
                        "task_outcome": "ok",
                    },
                )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            if timing_profile_enabled(global_config=task_manager.mem_config):
                _ilog = getattr(task_manager.recorder, "working_dir", None) or task_manager.mem_config.get(
                    "working_dir", "."
                )
                timing_profile_append(
                    _ilog,
                    "timing_parent_inline_task.jsonl",
                    {
                        "kind": "run_tasks_inline_wall",
                        "task_name": task_manager.task_name,
                        "task_index_1based": task_id + 1,
                        "wall_s": time.perf_counter() - t_inline0,
                        "task_outcome": "error",
                        "error_type": type(exc).__name__,
                    },
                )
            gamefile = (task_config.get("env_kwargs") or {}).get("gamefile")
            pddl_crash = _is_pddl_crash(gamefile, exc)
            skip_info = {
                "task_id": task_id,
                "task_index_1based": task_id + 1,
                "task_name": task_manager.task_name,
                "env_name": task_config.get("env_name"),
                "gamefile": gamefile,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "pddl_crash": bool(pddl_crash),
            }
            skipped_tasks.append(skip_info)
            # Short message for known ALFWorld grammar/load errors (including PDDL parse crash) so logs stay readable
            is_load_error = (
                "FailedToken" in type(exc).__name__
                or "FailedCut" in type(exc).__name__
                or type(exc).__name__ == "IndexError"
                or "expecting" in str(exc)
                or "pop from empty list" in str(exc)
                or "left recursion" in str(exc).lower()
                or "grammar" in str(exc).lower()
                or "load" in str(exc).lower()
            )
            if is_load_error:
                task_manager.recorder.log(
                    f"{progress_msg} Task {task_id + 1} skipped (game load/grammar error): {gamefile}. Continuing."
                )
                task_manager.recorder.log(f"  error_type={skip_info['error_type']}, error_message={str(exc)[:200]}")
            else:
                task_manager.recorder.log(
                    f"{progress_msg} Task {task_id + 1} skipped due to "
                    f"{skip_info['error_type']}: {skip_info['error_message']}"
                )
                task_manager.recorder.log(traceback.format_exc())

            if pddl_crash:
                effective_total = max(0, effective_total - 1)
            else:
                attempted += 1
                done_count += 1
                rewards.append(0.0)
                dones.append(False)
                per_task_mt.append({})
            if progress_hook is not None:
                progress_hook(done_count, effective_total, successes, attempted)
            else:
                if (attempted % 10 == 0) or (task_id == end - 1):
                    rate = (successes / attempted) if attempted else 0.0
                    print(
                        f"{progress_msg} success_rate={rate:.2%} ({successes}/{attempted})",
                        flush=True,
                    )
            continue

    task_manager.recorder.dataset_end()
    return rewards, dones, saved_messages, skipped_tasks, per_task_mt


def _mt_metrics_from_saved(d: Any) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    out: dict[str, float] = {}
    for k in ("ele_acc", "op_f1", "ssr", "tsr"):
        if k not in d:
            continue
        try:
            out[k] = float(d[k])
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def compute_metrics(
    rewards: list[float],
    per_task_mt: list[dict[str, float]] | None = None,
) -> dict[str, float]:
    if not rewards:
        out: dict[str, float] = {"accuracy": 0.0, "avg_reward": 0.0}
    else:
        correct = sum(1 for r in rewards if r > 0)
        out = {
            "accuracy": correct / len(rewards),
            "avg_reward": sum(rewards) / len(rewards),
        }
    if (
        per_task_mt is not None
        and len(per_task_mt) == len(rewards)
        and any(any(k in row for k in ("ele_acc", "op_f1", "ssr", "tsr")) for row in per_task_mt)
    ):
        for k in ("ele_acc", "op_f1", "ssr", "tsr"):
            vals = [float(row.get(k, 0.0)) for row in per_task_mt]
            out[f"avg_{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _dump_memory_nl_report(
    *,
    output_path: str,
    mas_memory: str,
    local_a_dir: str,
    local_b_dir: str,
    global_dir: str,
    task_a_label: str,
    task_b_label: str,
) -> None:
    """
    Export a human-readable natural-language memory report for local/global memories.
    """
    lines: list[str] = []
    lines.append("# Natural Language Memory Snapshot")
    lines.append("")
    lines.append(f"- memory_type: {mas_memory}")
    lines.append(f"- local_a: {task_a_label}")
    lines.append(f"- local_b: {task_b_label}")
    lines.append("")

    def _append_gmemory(scope_name: str, base_dir: str) -> None:
        p = os.path.join(base_dir, mas_memory, "insights.json")
        data = _safe_load_json(p)
        lines.append(f"## {scope_name}")
        lines.append(f"- source: `{p}`")
        if not isinstance(data, list) or len(data) == 0:
            lines.append("- no insights found")
            lines.append("")
            return
        lines.append(f"- total_rules: {len(data)}")
        lines.append("")
        for i, item in enumerate(data[:8], start=1):
            if not isinstance(item, dict):
                continue
            rule = str(item.get("rule", "")).strip()
            score = item.get("score")
            if rule:
                lines.append(f"{i}. ({score}) {rule}" if score is not None else f"{i}. {rule}")
        lines.append("")

    def _append_selectivemem(scope_name: str, base_dir: str) -> None:
        p = os.path.join(base_dir, mas_memory, "merged_instance_graph.json")
        data = _safe_load_json(p)
        lines.append(f"## {scope_name}")
        lines.append(f"- source: `{p}`")
        if not isinstance(data, dict):
            lines.append("- no graph found")
            lines.append("")
            return
        nodes = data.get("nodes") or []
        links = data.get("links") or data.get("edges") or []
        lines.append(f"- nodes: {len(nodes)}")
        lines.append(f"- edges: {len(links)}")
        if not links:
            lines.append("- no relation sentences available")
            lines.append("")
            return
        lines.append("")
        lines.append("Top relation sentences:")
        for i, e in enumerate(links[:12], start=1):
            if not isinstance(e, dict):
                continue
            u = e.get("source", e.get("from", "?"))
            v = e.get("target", e.get("to", "?"))
            rel = e.get("relation", e.get("label", "related_to"))
            lines.append(f"{i}. {u} --{rel}--> {v}")
        lines.append("")

    if mas_memory == "selectivemem":
        _append_selectivemem(f"Local A ({task_a_label})", local_a_dir)
        _append_selectivemem(f"Local B ({task_b_label})", local_b_dir)
        _append_selectivemem("Global", global_dir)
    else:
        _append_gmemory(f"Local A ({task_a_label})", local_a_dir)
        _append_gmemory(f"Local B ({task_b_label})", local_b_dir)
        _append_gmemory("Global", global_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


ALFWORLD_TASK_TYPE_TO_FEWSHOT = {
    "pick_and_place_simple": "put",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_two_obj_and_place": "puttwo",
}

ALFWORLD_TASK_TYPE_TO_ENV_NAME = {
    "pick_and_place_simple": "pick_and_place",
    "pick_clean_then_place_in_recep": "pick_clean_then_place",
    "pick_heat_then_place_in_recep": "pick_heat_then_place",
    "pick_cool_then_place_in_recep": "pick_cool_then_place",
    "look_at_obj_in_light": "look_at_obj",
    "pick_two_obj_and_place": "pick_two_obj",
}

ALFWORLD_GROUP_SPECS = {
    "kitchen_statechange": {
        "scene_domains": {"kitchen"},
        "task_types": (
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ),
    },
    "home_search": {
        "scene_domains": {"living", "bedroom", "bathroom"},
        "task_types": (
            "pick_and_place_simple",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
        ),
    },
    # Scene-only groups for fixed subsets (e.g., collab_subsets/v3_ss/{kitchen,bathroom}__*.json).
    # These groups are used for loading pre-materialized tasks; task_types are not used for filtering in that mode.
    "kitchen": {
        "scene_domains": {"kitchen"},
        "task_types": (
            "pick_and_place_simple",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ),
    },
    "bathroom": {
        "scene_domains": {"bathroom"},
        "task_types": (
            "pick_and_place_simple",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ),
    },
}


def alfworld_scene_domain(scene_id: int) -> str:
    if 1 <= scene_id <= 30:
        return "kitchen"
    if 201 <= scene_id <= 230:
        return "living"
    if 301 <= scene_id <= 330:
        return "bedroom"
    if 401 <= scene_id <= 430:
        return "bathroom"
    return "other"


def parse_alfworld_gamefile(gamefile_relpath: str) -> tuple[str, int, str]:
    task_dir = Path(gamefile_relpath).parts[-3]
    task_type = task_dir.split("-")[0]
    scene_id = int(task_dir.rsplit("-", 1)[-1])
    return task_type, scene_id, alfworld_scene_domain(scene_id)


def extract_alfworld_goal_instruction(gamefile_abspath: Path) -> str:
    with gamefile_abspath.open("r", encoding="utf-8") as reader:
        data = json.load(reader)

    grammar = data.get("grammar", "")
    match = re.search(r'"task"\s*:\s*\[\s*\{\s*"rhs"\s*:\s*"([^"]+)"', grammar)
    if match:
        return match.group(1)

    traj_path = gamefile_abspath.with_name("traj_data.json")
    if traj_path.exists():
        with traj_path.open("r", encoding="utf-8") as reader:
            traj_data = json.load(reader)
        annotations = (traj_data.get("turk_annotations") or {}).get("anns") or []
        if annotations:
            task_desc = annotations[0].get("task_desc", "").strip()
            if task_desc:
                if not task_desc.lower().startswith("your task is to:"):
                    task_desc = f"Your task is to: {task_desc}"
                if not task_desc.endswith("."):
                    task_desc += "."
                return task_desc

    raise ValueError(f"Failed to extract ALFWorld goal instruction from {gamefile_abspath}")


def build_alfworld_subset(
    repo_root: Path,
    split_name: str,
    group_name: str,
    per_type_count: int,
    seed: int,
) -> list[dict]:
    if group_name not in ALFWORLD_GROUP_SPECS:
        raise ValueError(f"Unsupported ALFWorld group: {group_name}")

    group_spec = ALFWORLD_GROUP_SPECS[group_name]
    split_root = repo_root / "data" / "alfworld" / "json_2.1.1" / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"ALFWorld split directory not found: {split_root}")

    grouped_candidates: dict[str, list[dict]] = {task_type: [] for task_type in group_spec["task_types"]}
    for gamefile_path in split_root.glob("**/game.tw-pddl"):
        gamefile_relpath = str(gamefile_path.relative_to(repo_root))
        task_type, _scene_id, scene_domain = parse_alfworld_gamefile(gamefile_relpath)
        if task_type not in grouped_candidates:
            continue
        if scene_domain not in group_spec["scene_domains"]:
            continue

        goal_instruction = extract_alfworld_goal_instruction(gamefile_path)
        grouped_candidates[task_type].append(
            {
                "goal_instruction": goal_instruction,
                "env_kwargs": {
                    "config": "alfworld",
                    "gamefile": gamefile_relpath,
                },
                "task_type": ALFWORLD_TASK_TYPE_TO_FEWSHOT[task_type],
                "env_name": ALFWORLD_TASK_TYPE_TO_ENV_NAME[task_type],
            }
        )

    selected_tasks: list[dict] = []
    for task_type in group_spec["task_types"]:
        candidates = sorted(grouped_candidates[task_type], key=lambda row: row["env_kwargs"]["gamefile"])
        rng = random.Random(f"{seed}:{split_name}:{group_name}:{task_type}")
        rng.shuffle(candidates)
        if len(candidates) < per_type_count:
            raise ValueError(
                f"Not enough ALFWorld tasks for {group_name}/{split_name}/{task_type}: "
                f"need {per_type_count}, got {len(candidates)}"
            )
        selected_tasks.extend(copy.deepcopy(candidates[:per_type_count]))

    random.Random(f"{seed}:{split_name}:{group_name}:all").shuffle(selected_tasks)
    return selected_tasks


def prepare_alfworld_collab_tasks(
    repo_root: Path,
    group_a: str,
    group_b: str,
    build_per_type: int,
    test_per_type: int,
    seed: int,
) -> dict[str, Any]:
    if group_a == group_b:
        raise ValueError("ALFWorld collaborative evaluation requires two different groups.")

    return {
        "task_a_name": group_a,
        "task_b_name": group_b,
        "task_a": build_alfworld_subset(repo_root, "train", group_a, build_per_type, seed),
        "task_b": build_alfworld_subset(repo_root, "train", group_b, build_per_type, seed + 1),
        "task_a_test": build_alfworld_subset(repo_root, "valid_unseen", group_a, test_per_type, seed + 100),
        "task_b_test": build_alfworld_subset(repo_root, "valid_unseen", group_b, test_per_type, seed + 101),
    }


def load_alfworld_subset_file(subset_dir: Path, group_name: str, split_name: str) -> list[dict]:
    subset_path = subset_dir / f"{group_name}__{split_name}.json"
    if not subset_path.exists():
        raise FileNotFoundError(
            f"Missing fixed ALFWorld subset file: {subset_path}. "
            f"Please run scripts/alfworld/materialize_collab_subsets.py first."
        )
    with subset_path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def load_alfworld_collab_tasks(
    subset_dir: Path,
    group_a: str,
    group_b: str,
    *,
    load_eval_splits: bool = True,
) -> dict[str, Any]:
    if group_a == group_b:
        raise ValueError("ALFWorld collaborative evaluation requires two different groups.")

    out = {
        "task_a_name": group_a,
        "task_b_name": group_b,
        "task_a": load_alfworld_subset_file(subset_dir, group_a, "train"),
        "task_b": load_alfworld_subset_file(subset_dir, group_b, "train"),
    }
    if load_eval_splits:
        out["task_a_test"] = load_alfworld_subset_file(subset_dir, group_a, "valid_unseen")
        out["task_b_test"] = load_alfworld_subset_file(subset_dir, group_b, "valid_unseen")
    else:
        out["task_a_test"] = []
        out["task_b_test"] = []
    return out


def global_graph_exists(global_dir: str, mas_memory: str) -> bool:
    """
    Return True if a global memory graph already exists.

    - For memgraph (MemoryBankGraphMASMemory): check entity_graph.json under <global_dir>/<mas_memory>/.
    - For selectivemem (SelectiveMemMASMemory): check global_experience_graph.json under <global_dir>/<mas_memory>/.
    """
    graph_dir = os.path.join(global_dir, mas_memory)
    if mas_memory == "selectivemem":
        graph_path = os.path.join(graph_dir, "global_experience_graph.json")
    else:
        graph_path = os.path.join(graph_dir, "entity_graph.json")
    if not os.path.isfile(graph_path):
        return False
    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data.get("nodes") or []
        return len(nodes) > 0
    except (json.JSONDecodeError, OSError):
        return False


def rebuild_selectivemem_global_from_locals(
    *,
    local_dirs: list[str],
    global_dir: str,
    model_name: str,
    snapshot_tag: str | None = None,
) -> None:
    """
    Build latest global GG for SelectiveMem by merging persisted local memories.
    No task re-training is required.
    """
    global_mem = SelectiveMemMASMemory(
        namespace="selectivemem",
        global_config={
            "working_dir": global_dir,
            "insights_only": True,
        },
        llm_model=GPTChat(model_name=model_name),
        embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
    )

    global_mem.merged_lg = nx.MultiDiGraph()
    merged_fingerprints: list[dict[str, Any]] = []
    seen_fp: set[str] = set()

    for wd in local_dirs:
        local_mem = SelectiveMemMASMemory(
            namespace="selectivemem",
            global_config={
                "working_dir": wd,
                "freeze_memory": True,
                "insights_only": True,
            },
            llm_model=GPTChat(model_name=model_name),
            embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
        )

        # Merge merged_lg nodes + provenance
        for node, data in local_mem.merged_lg.nodes(data=True):
            nid = str(node)
            nd = data or {}
            if nid not in global_mem.merged_lg:
                global_mem.merged_lg.add_node(
                    nid,
                    label=nd.get("label"),
                    category=nd.get("category"),
                    provenance=[],
                )
            for prov in nd.get("provenance") or []:
                if isinstance(prov, dict):
                    global_mem._append_merged_node_provenance(nid, dict(prov))

        # Merge merged_lg edges + provenance
        for u, v, _k, ed in local_mem.merged_lg.edges(keys=True, data=True):
            rel = str((ed or {}).get("relation") or "").strip()
            if not rel:
                continue
            provs = (ed or {}).get("provenance") or []
            if not provs:
                provs = [{}]
            for prov in provs:
                if isinstance(prov, dict):
                    global_mem._merge_or_add_merged_edge(str(u), str(v), rel, dict(prov))

        # Merge trajectory fingerprints (dedupe)
        for rec in getattr(local_mem, "_fingerprint_records", []) or []:
            tid = str(rec.get("trajectory_id") or "")
            q = str(rec.get("query_text") or "")
            es = rec.get("edge_sequence") or []
            fp_key = json.dumps([tid, q, es], ensure_ascii=False, sort_keys=True)
            if fp_key in seen_fp:
                continue
            seen_fp.add(fp_key)
            merged_fingerprints.append(rec)

    global_mem._fingerprint_records = merged_fingerprints
    global_mem._persist_merged_lg_best_effort()
    try:
        with open(global_mem._fingerprint_path, "w", encoding="utf-8") as f:
            for rec in merged_fingerprints:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass

    # Keep per-merge snapshots for audit (e.g., 100 tasks with batch_size=10 => 10 GG snapshots).
    if snapshot_tag:
        try:
            snap_dir = os.path.join(global_mem.persist_dir, "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            snap_gg = os.path.join(snap_dir, f"merged_instance_graph.{snapshot_tag}.json")
            snap_fp = os.path.join(snap_dir, f"trajectory_fingerprints.{snapshot_tag}.jsonl")
            # Copy current GG artifacts to snapshot files.
            with open(global_mem._merged_lg_path, "r", encoding="utf-8") as rf:
                merged_text = rf.read()
            with open(snap_gg, "w", encoding="utf-8") as wf:
                wf.write(merged_text)
            with open(global_mem._fingerprint_path, "r", encoding="utf-8") as rf:
                fp_text = rf.read()
            with open(snap_fp, "w", encoding="utf-8") as wf:
                wf.write(fp_text)
        except Exception:
            pass


def main() -> None:
    random.seed(42)

    import argparse

    parser = argparse.ArgumentParser(description="Collaborative domain adaptation evaluation for memory-based MAS.")
    parser.add_argument("--dataset_family", type=str, choices=["medmcqa", "alfworld", "mtmind2web", "scienceworld"], default="medmcqa")
    parser.add_argument("--task_a", type=str, default="medmcqa_physio_150_build")
    parser.add_argument("--task_b", type=str, default="medmcqa_pharma_150_build")
    parser.add_argument("--task_a_test", type=str, default="medmcqa_physio_20_test")
    parser.add_argument("--task_b_test", type=str, default="medmcqa_pharma_20_test")
    parser.add_argument("--alfworld_group_a", type=str, choices=sorted(ALFWORLD_GROUP_SPECS.keys()), default="kitchen_statechange")
    parser.add_argument("--alfworld_group_b", type=str, choices=sorted(ALFWORLD_GROUP_SPECS.keys()), default="home_search")
    parser.add_argument(
        "--alfworld_subset_dir",
        type=str,
        default="data/alfworld/collab_subsets/v1",
        help="Directory containing fixed ALFWorld collaborative subset JSON files.",
    )
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Limit train tasks per group for quick smoke test (e.g. 4). Ignored if not set.",
    )
    parser.add_argument(
        "--max_eval",
        type=int,
        default=None,
        help="Limit eval tasks per group for quick smoke test (e.g. 2). Ignored if not set.",
    )
    parser.add_argument("--mas_type", type=str, choices=["autogen", "macnet", "dylan", "strategy"], default="autogen")
    parser.add_argument("--mas_memory", type=str, default="memgraph")
    parser.add_argument("--reasoning", type=str, default="io")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125", help="Agent/reasoning model (same as LLM-only baseline). Search uses gpt-4o-mini in env.")
    parser.add_argument("--max_trials", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--tool_mode", choices=["search", "no_search", "force_search", "smart_search"], default="search")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="all",
        help="Comma-separated scenario names or indices 1-6, e.g. '1,2,3' or 'inner_baseline_A_on_A,inner_ours_A_on_A'. Use 'all' to run all 6.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip all training (local A/B and global); only run the 6 eval scenarios. Requires existing memory at --run_id.",
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="none",
        choices=["none", "physio5"],
        help="Optional quick test mode. 'physio5': for inner_ours_A_on_A only, evaluate MedMCQA physio test on the first 5 questions using the existing memory graph.",
    )
    parser.add_argument(
        "--profile_timing",
        action="store_true",
        help="Append fine-grained timing JSONL under log_dir / persist_dir (NV_DAMAS_PROFILE=1 + mem_config.profile_timing).",
    )
    parser.add_argument(
        "--mt_task_a_train_jsonl",
        type=str,
        default="data/MT-Mind2Web/mtmind2web_entertainment_websites_train_eval.jsonl",
        help="MT-Mind2Web A-domain train jsonl (used when --dataset_family mtmind2web).",
    )
    parser.add_argument(
        "--mt_task_b_train_jsonl",
        type=str,
        default="data/MT-Mind2Web/mtmind2web_shopping_websites_train_eval.jsonl",
        help="MT-Mind2Web B-domain train jsonl (used when --dataset_family mtmind2web).",
    )
    parser.add_argument(
        "--mt_task_a_test_jsonl",
        type=str,
        default="data/MT-Mind2Web/mtmind2web_entertainment_websites_test_eval.jsonl",
        help="MT-Mind2Web A-domain test jsonl (used when --dataset_family mtmind2web).",
    )
    parser.add_argument(
        "--mt_task_b_test_jsonl",
        type=str,
        default="data/MT-Mind2Web/mtmind2web_shopping_websites_test_eval.jsonl",
        help="MT-Mind2Web B-domain test jsonl (used when --dataset_family mtmind2web).",
    )
    parser.add_argument(
        "--sw_task_a_train_jsonl",
        type=str,
        default="data/scienceworld/scienceworld_domain_a_train.jsonl",
        help="ScienceWorld A-domain train jsonl (used when --dataset_family scienceworld).",
    )
    parser.add_argument(
        "--sw_task_b_train_jsonl",
        type=str,
        default="data/scienceworld/scienceworld_domain_b_train.jsonl",
        help="ScienceWorld B-domain train jsonl (used when --dataset_family scienceworld).",
    )
    parser.add_argument(
        "--sw_task_a_test_jsonl",
        type=str,
        default="data/scienceworld/scienceworld_domain_a_test.jsonl",
        help="ScienceWorld A-domain test jsonl (used when --dataset_family scienceworld).",
    )
    parser.add_argument(
        "--sw_task_b_test_jsonl",
        type=str,
        default="data/scienceworld/scienceworld_domain_b_test.jsonl",
        help="ScienceWorld B-domain test jsonl (used when --dataset_family scienceworld).",
    )
    args = parser.parse_args()

    mem_profile_extras: dict[str, Any] = {}
    if getattr(args, "profile_timing", False):
        os.environ["NV_DAMAS_PROFILE"] = "1"
        mem_profile_extras["profile_timing"] = True

    if args.dataset_family == "alfworld" and args.tool_mode != "search":
        raise ValueError("ALFWorld collaborative evaluation only supports --tool_mode search.")

    repo_root = Path(__file__).resolve().parents[2]
    model_type: str = get_model_type(args.model)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    eval_namespace = f"{args.dataset_family}_collab_eval"
    base_dir = os.path.join("./.db", model_type, eval_namespace, run_id, args.mas_type, "memory", args.mas_memory)
    local_dir = os.path.join(base_dir, "local")
    dual_local_dir = os.path.join(base_dir, "dual_local")
    global_dir = os.path.join(base_dir, "global")
    report_dir = os.path.join("./reports")

    log_base = os.path.join("./logs", eval_namespace, run_id, args.mas_type, "memory", args.mas_memory, model_type)

    ensure_dir(local_dir)
    ensure_dir(dual_local_dir)
    ensure_dir(global_dir)
    ensure_dir(report_dir)

    if args.dataset_family == "alfworld":
        load_eval_splits = not (args.max_eval is not None and int(args.max_eval) <= 0)
        alfworld_tasks = load_alfworld_collab_tasks(
            subset_dir=repo_root / args.alfworld_subset_dir,
            group_a=args.alfworld_group_a,
            group_b=args.alfworld_group_b,
            load_eval_splits=load_eval_splits,
        )
        task_a_name = "alfworld"
        task_b_name = "alfworld"
        task_a_label = alfworld_tasks["task_a_name"]
        task_b_label = alfworld_tasks["task_b_name"]
        task_a_train_tasks = alfworld_tasks["task_a"]
        task_b_train_tasks = alfworld_tasks["task_b"]
        task_a_eval_name = "alfworld"
        task_b_eval_name = "alfworld"
        task_a_eval_tasks = alfworld_tasks["task_a_test"]
        task_b_eval_tasks = alfworld_tasks["task_b_test"]

        if args.max_train is not None:
            task_a_train_tasks = task_a_train_tasks[: args.max_train]
            task_b_train_tasks = task_b_train_tasks[: args.max_train]
        if args.max_eval is not None:
            task_a_eval_tasks = task_a_eval_tasks[: args.max_eval]
            task_b_eval_tasks = task_b_eval_tasks[: args.max_eval]
    elif args.dataset_family == "mtmind2web":
        task_a_name = "mtmind2web_train"
        task_b_name = "mtmind2web_train"
        task_a_label = "entertainment"
        task_b_label = "shopping"
        task_a_train_tasks = load_jsonl_rows(args.mt_task_a_train_jsonl)
        task_b_train_tasks = load_jsonl_rows(args.mt_task_b_train_jsonl)
        task_a_eval_name = "mtmind2web_test_task"
        task_b_eval_name = "mtmind2web_test_task"
        task_a_eval_tasks = load_jsonl_rows(args.mt_task_a_test_jsonl)
        task_b_eval_tasks = load_jsonl_rows(args.mt_task_b_test_jsonl)

        if args.max_train is not None:
            task_a_train_tasks = task_a_train_tasks[: args.max_train]
            task_b_train_tasks = task_b_train_tasks[: args.max_train]
        if args.max_eval is not None:
            task_a_eval_tasks = task_a_eval_tasks[: args.max_eval]
            task_b_eval_tasks = task_b_eval_tasks[: args.max_eval]
    elif args.dataset_family == "scienceworld":
        task_a_name = "scienceworld_train"
        task_b_name = "scienceworld_train"
        task_a_label = "scienceworld_a"
        task_b_label = "scienceworld_b"
        task_a_train_tasks = load_jsonl_rows(args.sw_task_a_train_jsonl)
        task_b_train_tasks = load_jsonl_rows(args.sw_task_b_train_jsonl)
        task_a_eval_name = "scienceworld_test"
        task_b_eval_name = "scienceworld_test"
        task_a_eval_tasks = load_jsonl_rows(args.sw_task_a_test_jsonl)
        task_b_eval_tasks = load_jsonl_rows(args.sw_task_b_test_jsonl)

        if args.max_train is not None:
            task_a_train_tasks = task_a_train_tasks[: args.max_train]
            task_b_train_tasks = task_b_train_tasks[: args.max_train]
        if args.max_eval is not None:
            task_a_eval_tasks = task_a_eval_tasks[: args.max_eval]
            task_b_eval_tasks = task_b_eval_tasks[: args.max_eval]
    else:
        task_a_name = args.task_a
        task_b_name = args.task_b
        task_a_label = args.task_a
        task_b_label = args.task_b
        task_a_train_tasks = None
        task_b_train_tasks = None
        task_a_eval_name = args.task_a_test
        task_b_eval_name = args.task_b_test
        task_a_eval_tasks = None
        task_b_eval_tasks = None

    local_a_dir = os.path.join(local_dir, task_a_label)
    local_b_dir = os.path.join(local_dir, task_b_label)
    ensure_dir(local_a_dir)
    ensure_dir(local_b_dir)

    def build_manager(task_name: str, working_dir: str, log_dir: str, tasks_override: list[dict] | None = None) -> TaskManager:
        manager = build_task_manager(task_name, args.mas_type, args.mas_memory, args.max_trials, working_dir, log_dir)
        if tasks_override is not None:
            manager.tasks = copy.deepcopy(tasks_override)
        return manager

    def apply_collab_mem_tags(mgr: TaskManager, side: str, scene: str, phase: str) -> None:
        """Tag SelectiveMem viz JSON (LG_viz / GG_viz) with A/B, scene, train phase."""
        if args.dataset_family != "alfworld":
            return
        mgr.mem_config.update(collab_side=side, collab_scene=scene, train_phase=phase)

    # ALFWorld: persist entity graph every 3 tasks; use raw task for retrieval; insights_topk=1
    alfworld_mem_overrides = {"entity_graph_persist_every": 3} if args.dataset_family == "alfworld" else {}
    alfworld_mas_overrides = {"insights_topk": 1} if args.dataset_family == "alfworld" else {}
    train_mem_extras = {**alfworld_mem_overrides, **mem_profile_extras}
    train_metrics: dict[str, Any] = {
        "dataset_family": args.dataset_family,
        "run_id": run_id,
        "train_batches": [],
        "train_summary": {},
    }

    if not args.eval_only:
        # ---- Train local memories ----
        log_train_local_a = os.path.join(log_base, "train_local", task_a_label)
        log_train_local_b = os.path.join(log_base, "train_local", task_b_label)

        a_manager = build_manager(task_a_name, local_a_dir, log_train_local_a, task_a_train_tasks)
        b_manager = build_manager(task_b_name, local_b_dir, log_train_local_b, task_b_train_tasks)
        apply_collab_mem_tags(a_manager, "A", task_a_label, "pass1_local")
        apply_collab_mem_tags(b_manager, "B", task_b_label, "pass1_local")
        a_manager.mem_config.update(train_mem_extras)
        b_manager.mem_config.update(train_mem_extras)
        a_manager.mas_config.update(alfworld_mas_overrides)
        b_manager.mas_config.update(alfworld_mas_overrides)
        a_manager.mas_config.update({"silent_mas": True})
        b_manager.mas_config.update({"silent_mas": True})
        build_mas(a_manager, args.reasoning, args.mas_memory, args.model)
        build_mas(b_manager, args.reasoning, args.mas_memory, args.model)

        alfworld_sp = {"reasoning": args.reasoning, "model": args.model, "max_trials": args.max_trials} if args.dataset_family == "alfworld" else None
        if args.mas_memory == "selectivemem" and args.dataset_family == "mtmind2web":
            # MT-Mind2Web SelectiveMem collaborative flow aligned with g-memory:
            # batch-wise local A/B -> rebuild global -> global-only A/B on the same batch.
            total_local = min(len(a_manager.tasks), len(b_manager.tasks))
            batch_size = max(1, int(args.batch_size))
            num_batches = (total_local + batch_size - 1) // batch_size if total_local > 0 else 0
            local_a_succ = 0
            local_a_att = 0
            local_b_succ = 0
            local_b_att = 0
            global_a_succ = 0
            global_a_att = 0
            global_b_succ = 0
            global_b_att = 0
            all_local_a_rewards: list[float] = []
            all_local_b_rewards: list[float] = []
            all_global_a_rewards: list[float] = []
            all_global_b_rewards: list[float] = []
            all_local_a_mt: list[dict[str, float]] = []
            all_local_b_mt: list[dict[str, float]] = []
            all_global_a_mt: list[dict[str, float]] = []
            all_global_b_mt: list[dict[str, float]] = []

            for start in range(0, total_local, batch_size):
                end = min(start + batch_size, total_local)
                batch_idx = (start // batch_size) + 1

                # Step 1: local A/B in parallel on the same batch.
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_a_local = executor.submit(
                        run_tasks, a_manager, start, end, args.tool_mode, alfworld_sp
                    )
                    fut_b_local = executor.submit(
                        run_tasks, b_manager, start, end, args.tool_mode, alfworld_sp
                    )
                    a_local_rewards, _, _, _, a_local_mt = fut_a_local.result()
                    b_local_rewards, _, _, _, b_local_mt = fut_b_local.result()

                local_a_succ += sum(1 for r in a_local_rewards if r > 0)
                local_a_att += len(a_local_rewards)
                local_b_succ += sum(1 for r in b_local_rewards if r > 0)
                local_b_att += len(b_local_rewards)
                all_local_a_rewards.extend(a_local_rewards)
                all_local_b_rewards.extend(b_local_rewards)
                all_local_a_mt.extend(a_local_mt)
                all_local_b_mt.extend(b_local_mt)

                # Persist local memories and rebuild GG from locals for this batch.
                for mgr in (a_manager, b_manager):
                    persist_fn = getattr(getattr(mgr.mas, "meta_memory", None), "persist_entity_graph", None)
                    if callable(persist_fn):
                        try:
                            persist_fn()
                        except Exception:
                            pass
                snap_tag = f"batch_{batch_idx:03d}_{start + 1:04d}-{end:04d}"
                rebuild_selectivemem_global_from_locals(
                    local_dirs=[local_a_dir, local_b_dir],
                    global_dir=global_dir,
                    model_name=args.model,
                    snapshot_tag=snap_tag,
                )

                # Step 2/3: global-only A/B in parallel on the same batch.
                global_retriever = SelectiveMemMASMemory(
                    namespace=args.mas_memory,
                    global_config={
                        "working_dir": global_dir,
                        "freeze_memory": True,
                        "insights_only": True,
                        **train_mem_extras,
                    },
                    llm_model=GPTChat(model_name=args.model),
                    embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
                )

                probe_a_dir = os.path.join(dual_local_dir, f"{task_a_label}_global_only")
                probe_b_dir = os.path.join(dual_local_dir, f"{task_b_label}_global_only")
                ensure_dir(probe_a_dir)
                ensure_dir(probe_b_dir)
                log_train_global_only = os.path.join(log_base, "train_global_only")
                a_manager_global = build_manager(task_a_name, probe_a_dir, log_train_global_only, task_a_train_tasks)
                b_manager_global = build_manager(task_b_name, probe_b_dir, log_train_global_only, task_b_train_tasks)
                a_manager_global.mem_config.update({**train_mem_extras, "freeze_memory": True})
                b_manager_global.mem_config.update({**train_mem_extras, "freeze_memory": True})
                a_manager_global.mas_config.update(alfworld_mas_overrides)
                b_manager_global.mas_config.update(alfworld_mas_overrides)
                a_manager_global.mas_config.update({"silent_mas": True})
                b_manager_global.mas_config.update({"silent_mas": True})
                a_mem_global = build_mas(a_manager_global, args.reasoning, args.mas_memory, args.model)
                b_mem_global = build_mas(b_manager_global, args.reasoning, args.mas_memory, args.model)
                set_global_a = getattr(a_mem_global, "set_global_retriever", None)
                if callable(set_global_a):
                    set_global_a(global_retriever)
                set_global_b = getattr(b_mem_global, "set_global_retriever", None)
                if callable(set_global_b):
                    set_global_b(global_retriever)

                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_a_global = executor.submit(
                        run_tasks, a_manager_global, start, end, args.tool_mode, alfworld_sp
                    )
                    fut_b_global = executor.submit(
                        run_tasks, b_manager_global, start, end, args.tool_mode, alfworld_sp
                    )
                    a_global_rewards, _, _, _, a_global_mt = fut_a_global.result()
                    b_global_rewards, _, _, _, b_global_mt = fut_b_global.result()

                global_a_succ += sum(1 for r in a_global_rewards if r > 0)
                global_a_att += len(a_global_rewards)
                global_b_succ += sum(1 for r in b_global_rewards if r > 0)
                global_b_att += len(b_global_rewards)
                all_global_a_rewards.extend(a_global_rewards)
                all_global_b_rewards.extend(b_global_rewards)
                all_global_a_mt.extend(a_global_mt)
                all_global_b_mt.extend(b_global_mt)

                batch_row = {
                    "batch_idx": batch_idx,
                    "start": start,
                    "end": end,
                    "local_A": {
                        **compute_metrics(a_local_rewards, a_local_mt),
                        "num_tasks": len(a_local_rewards),
                        "num_success": sum(1 for r in a_local_rewards if r > 0),
                    },
                    "local_B": {
                        **compute_metrics(b_local_rewards, b_local_mt),
                        "num_tasks": len(b_local_rewards),
                        "num_success": sum(1 for r in b_local_rewards if r > 0),
                    },
                    "global_A": {
                        **compute_metrics(a_global_rewards, a_global_mt),
                        "num_tasks": len(a_global_rewards),
                        "num_success": sum(1 for r in a_global_rewards if r > 0),
                    },
                    "global_B": {
                        **compute_metrics(b_global_rewards, b_global_mt),
                        "num_tasks": len(b_global_rewards),
                        "num_success": sum(1 for r in b_global_rewards if r > 0),
                    },
                    "cumulative": {
                        "local_A": {"num_success": local_a_succ, "num_tasks": local_a_att},
                        "local_B": {"num_success": local_b_succ, "num_tasks": local_b_att},
                        "global_A": {"num_success": global_a_succ, "num_tasks": global_a_att},
                        "global_B": {"num_success": global_b_succ, "num_tasks": global_b_att},
                    },
                }
                train_metrics["train_batches"].append(batch_row)

                print(
                    f"\rA {end}/{total_local} | B {end}/{total_local} | "
                    f"GG {batch_idx}/{num_batches} | train(mt selective batch) | "
                    f"A_local {local_a_succ}/{local_a_att} | B_local {local_b_succ}/{local_b_att} | "
                    f"A_global {global_a_succ}/{global_a_att} | B_global {global_b_succ}/{global_b_att}",
                    end="" if end < total_local else "\n",
                    flush=True,
                )

            train_metrics["train_summary"] = {
                "local_A": {
                    **compute_metrics(all_local_a_rewards, all_local_a_mt),
                    "num_tasks": len(all_local_a_rewards),
                    "num_success": sum(1 for r in all_local_a_rewards if r > 0),
                },
                "local_B": {
                    **compute_metrics(all_local_b_rewards, all_local_b_mt),
                    "num_tasks": len(all_local_b_rewards),
                    "num_success": sum(1 for r in all_local_b_rewards if r > 0),
                },
                "global_A": {
                    **compute_metrics(all_global_a_rewards, all_global_a_mt),
                    "num_tasks": len(all_global_a_rewards),
                    "num_success": sum(1 for r in all_global_a_rewards if r > 0),
                },
                "global_B": {
                    **compute_metrics(all_global_b_rewards, all_global_b_mt),
                    "num_tasks": len(all_global_b_rewards),
                    "num_success": sum(1 for r in all_global_b_rewards if r > 0),
                },
                "num_batches": num_batches,
                "batch_size": batch_size,
            }
        elif args.mas_memory == "selectivemem":
            # Run local training in batches:
            # - Pass 1 (LG-only): build/update local LG
            # - Merge GG once per batch (from persisted locals) + snapshot
            # - Pass 2 (LG+GG): rerun same batch with latest GG attached as global retriever
            total_local = min(len(a_manager.tasks), len(b_manager.tasks))
            # Total denominator excludes PDDL-crash tasks. We approximate this as
            # the maximum number of excluded tasks observed across A/B runs so far.
            total_effective = int(total_local)
            bs = max(1, int(args.batch_size))
            num_batches = (total_local + bs - 1) // bs
            # Track 4 results over time:
            # A_LG: pass1 (local LG only, using local_instance_graph)
            # A_LGG: pass2 (local LG + global GG, using merged global graph rebuilt per batch)
            # B_LG / B_LGG: same for scenario B.
            a1_succ = 0
            a1_att = 0
            b1_succ = 0
            b1_att = 0
            a2_succ = 0
            a2_att = 0
            b2_succ = 0
            b2_att = 0
            for start in range(0, total_local, bs):
                end = min(start + bs, total_local)
                batch_idx = (start // bs) + 1
                # Track how many PDDL-crash tasks were excluded so far in this batch run (per side).
                a1_removed = 0
                b1_removed = 0
                a2_removed = 0
                b2_removed = 0
                _print_train_progress(
                    phase=f"train(pass1, batch {batch_idx}/{num_batches})",
                    a_done=start,
                    b_done=start,
                    total=total_effective,
                    gg_done=batch_idx - 1,
                    gg_total=num_batches,
                    a_lg_succ=a1_succ,
                    a_lg_att=start,
                    a_lgg_succ=a2_succ,
                    a_lgg_att=a2_att,
                    b_lg_succ=b1_succ,
                    b_lg_att=start,
                    b_lgg_succ=b2_succ,
                    b_lgg_att=b2_att,
                )

                # Pass 1: LG-only (task-granularity progress)
                a_done = start
                b_done = start

                def _hook_a_pass1(task_idx_1based: int, _total: int, _succ: int, _attempted: int) -> None:
                    nonlocal a_done, b_done, a1_succ, a1_att, a2_succ, a2_att, b1_succ, b1_att, b2_succ, b2_att, total_effective, a1_removed, b1_removed, a2_removed, b2_removed
                    a_done = max(a_done, int(task_idx_1based))
                    a1_succ = int(_succ)
                    a1_att = int(_attempted)
                    a1_removed = max(0, (end - start) - int(_total))
                    total_effective = max(0, int(total_local) - max(a1_removed, b1_removed, a2_removed, b2_removed))
                    _print_train_progress(
                        phase=f"train(pass1, batch {batch_idx}/{num_batches})",
                        a_done=a_done,
                        b_done=b_done,
                        total=total_effective,
                        gg_done=batch_idx - 1,
                        gg_total=num_batches,
                        a_lg_succ=a1_succ,
                        a_lg_att=a1_att,
                        a_lgg_succ=a2_succ,
                        a_lgg_att=a2_att,
                        b_lg_succ=b1_succ,
                        b_lg_att=b1_att,
                        b_lgg_succ=b2_succ,
                        b_lgg_att=b2_att,
                    )

                def _hook_b_pass1(task_idx_1based: int, _total: int, _succ: int, _attempted: int) -> None:
                    nonlocal a_done, b_done, a1_succ, a1_att, a2_succ, a2_att, b1_succ, b1_att, b2_succ, b2_att, total_effective, a1_removed, b1_removed, a2_removed, b2_removed
                    b_done = max(b_done, int(task_idx_1based))
                    b1_succ = int(_succ)
                    b1_att = int(_attempted)
                    b1_removed = max(0, (end - start) - int(_total))
                    total_effective = max(0, int(total_local) - max(a1_removed, b1_removed, a2_removed, b2_removed))
                    _print_train_progress(
                        phase=f"train(pass1, batch {batch_idx}/{num_batches})",
                        a_done=a_done,
                        b_done=b_done,
                        total=total_effective,
                        gg_done=batch_idx - 1,
                        gg_total=num_batches,
                        a_lg_succ=a1_succ,
                        a_lg_att=a1_att,
                        a_lgg_succ=a2_succ,
                        a_lgg_att=a2_att,
                        b_lg_succ=b1_succ,
                        b_lg_att=b1_att,
                        b_lgg_succ=b2_succ,
                        b_lgg_att=b2_att,
                    )

                run_tasks(
                    a_manager,
                    start,
                    end,
                    args.tool_mode,
                    alfworld_subprocess_args=alfworld_sp,
                    progress_hook=_hook_a_pass1,
                )
                run_tasks(
                    b_manager,
                    start,
                    end,
                    args.tool_mode,
                    alfworld_subprocess_args=alfworld_sp,
                    progress_hook=_hook_b_pass1,
                )
                # Batch-end persist locals
                for mgr in (a_manager, b_manager):
                    persist_fn = getattr(getattr(mgr.mas, "meta_memory", None), "persist_entity_graph", None)
                    if callable(persist_fn):
                        try:
                            persist_fn()
                        except Exception:
                            pass
                # Rebuild latest GG from local persisted memories and snapshot it.
                snap_tag = f"batch_{(start // bs) + 1:03d}_{start + 1:04d}-{end:04d}"
                _print_train_progress(
                    phase=f"merge_GG(batch {batch_idx}/{num_batches})",
                    a_done=end,
                    b_done=end,
                    total=total_effective,
                    gg_done=batch_idx,
                    gg_total=num_batches,
                )
                t_merge = time.perf_counter()
                rebuild_selectivemem_global_from_locals(
                    local_dirs=[local_a_dir, local_b_dir],
                    global_dir=global_dir,
                    model_name=args.model,
                    snapshot_tag=snap_tag,
                )
                if mem_profile_extras.get("profile_timing"):
                    timing_profile_append(
                        global_dir,
                        "timing_merge_gg.jsonl",
                        {
                            "kind": "rebuild_selectivemem_global_from_locals",
                            "batch_idx": batch_idx,
                            "num_batches": num_batches,
                            "snap_tag": snap_tag,
                            "wall_s": time.perf_counter() - t_merge,
                        },
                    )

                # Pass 2: LG + GG (use the GG rebuilt above; do NOT rebuild again in this batch)
                if alfworld_sp is not None:
                    alfworld_sp_gg = dict(alfworld_sp)
                    alfworld_sp_gg["use_global_insights"] = True
                    alfworld_sp_gg["global_dir"] = global_dir
                else:
                    # Non-ALFWorld path currently does not wire global_dir into run_tasks() directly.
                    # We keep behavior unchanged.
                    alfworld_sp_gg = None

                log_train_local_a_gg = os.path.join(log_base, "train_local_gg", task_a_label)
                log_train_local_b_gg = os.path.join(log_base, "train_local_gg", task_b_label)
                a_manager_gg = build_manager(task_a_name, local_a_dir, log_train_local_a_gg, task_a_train_tasks)
                b_manager_gg = build_manager(task_b_name, local_b_dir, log_train_local_b_gg, task_b_train_tasks)
                apply_collab_mem_tags(a_manager_gg, "A", task_a_label, "pass2_lgg")
                apply_collab_mem_tags(b_manager_gg, "B", task_b_label, "pass2_lgg")
                a_manager_gg.mem_config.update(train_mem_extras)
                b_manager_gg.mem_config.update(train_mem_extras)
                a_manager_gg.mas_config.update(alfworld_mas_overrides)
                b_manager_gg.mas_config.update(alfworld_mas_overrides)
                a_manager_gg.mas_config.update({"silent_mas": True})
                b_manager_gg.mas_config.update({"silent_mas": True})
                build_mas(a_manager_gg, args.reasoning, args.mas_memory, args.model)
                build_mas(b_manager_gg, args.reasoning, args.mas_memory, args.model)

                # Pass 2: LG+GG (task-granularity progress)
                a2_done = start
                b2_done = start

                def _hook_a_pass2(task_idx_1based: int, _total: int, _succ: int, _attempted: int) -> None:
                    nonlocal a2_done, b2_done, a1_succ, a1_att, a2_succ, a2_att, b1_succ, b1_att, b2_succ, b2_att, total_effective, a1_removed, b1_removed, a2_removed, b2_removed
                    a2_done = max(a2_done, int(task_idx_1based))
                    a2_succ = int(_succ)
                    a2_att = int(_attempted)
                    a2_removed = max(0, (end - start) - int(_total))
                    total_effective = max(0, int(total_local) - max(a1_removed, b1_removed, a2_removed, b2_removed))
                    _print_train_progress(
                        phase=f"train(pass2, batch {batch_idx}/{num_batches})",
                        a_done=a2_done,
                        b_done=b2_done,
                        total=total_effective,
                        gg_done=batch_idx,
                        gg_total=num_batches,
                        a_lg_succ=a1_succ,
                        a_lg_att=a1_att,
                        a_lgg_succ=a2_succ,
                        a_lgg_att=a2_att,
                        b_lg_succ=b1_succ,
                        b_lg_att=b1_att,
                        b_lgg_succ=b2_succ,
                        b_lgg_att=b2_att,
                    )

                def _hook_b_pass2(task_idx_1based: int, _total: int, _succ: int, _attempted: int) -> None:
                    nonlocal a2_done, b2_done, a1_succ, a1_att, a2_succ, a2_att, b1_succ, b1_att, b2_succ, b2_att, total_effective, a1_removed, b1_removed, a2_removed, b2_removed
                    b2_done = max(b2_done, int(task_idx_1based))
                    b2_succ = int(_succ)
                    b2_att = int(_attempted)
                    b2_removed = max(0, (end - start) - int(_total))
                    total_effective = max(0, int(total_local) - max(a1_removed, b1_removed, a2_removed, b2_removed))
                    _print_train_progress(
                        phase=f"train(pass2, batch {batch_idx}/{num_batches})",
                        a_done=a2_done,
                        b_done=b2_done,
                        total=total_effective,
                        gg_done=batch_idx,
                        gg_total=num_batches,
                        a_lg_succ=a1_succ,
                        a_lg_att=a1_att,
                        a_lgg_succ=a2_succ,
                        a_lgg_att=a2_att,
                        b_lg_succ=b1_succ,
                        b_lg_att=b1_att,
                        b_lgg_succ=b2_succ,
                        b_lgg_att=b2_att,
                    )

                run_tasks(
                    a_manager_gg,
                    start,
                    end,
                    args.tool_mode,
                    alfworld_subprocess_args=alfworld_sp_gg,
                    progress_hook=_hook_a_pass2,
                )
                run_tasks(
                    b_manager_gg,
                    start,
                    end,
                    args.tool_mode,
                    alfworld_subprocess_args=alfworld_sp_gg,
                    progress_hook=_hook_b_pass2,
                )
                _print_train_progress(
                    phase=f"train(pass2, batch {batch_idx}/{num_batches})",
                    a_done=a2_done,
                    b_done=b2_done,
                    total=total_effective,
                    gg_done=batch_idx,
                    gg_total=num_batches,
                    a_lg_succ=a1_succ,
                    a_lg_att=a1_att,
                    a_lgg_succ=a2_succ,
                    a_lgg_att=a2_att,
                    b_lg_succ=b1_succ,
                    b_lg_att=b1_att,
                    b_lgg_succ=b2_succ,
                    b_lgg_att=b2_att,
                    endline=True,
                )
        else:
            # Non-selectivemem memory path (e.g., g-memory).
            # For MT-Mind2Web collaborative setup, run per-batch:
            # 1) A/B local parallel -> 2) merge into global -> 3) A/B global-only parallel.
            if args.dataset_family == "mtmind2web":
                from mas.memory.common import MASMessage  # type: ignore

                _reasoning_cls, global_mem_cls = module_map(args.reasoning, args.mas_memory)
                global_mem = global_mem_cls(
                    namespace=args.mas_memory,
                    global_config={
                        "working_dir": global_dir,
                        **train_mem_extras,
                    },
                    llm_model=GPTChat(model_name=args.model),
                    embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
                )

                def _coerce_mas_message(x: Any) -> MASMessage:
                    if isinstance(x, MASMessage):
                        return x
                    if isinstance(x, dict):
                        return MASMessage.from_dict(x)
                    raise TypeError(f"Unsupported peer message type for global merge: {type(x)}")

                def _merge_peer_msgs(msgs: list[Any], source_id: str) -> None:
                    for msg in msgs:
                        mm = _coerce_mas_message(msg)
                        if hasattr(global_mem, "add_memory_from_peer"):
                            global_mem.add_memory_from_peer(mm, source_id=source_id)
                        else:
                            global_mem.add_memory(mm)

                total_local = min(len(a_manager.tasks), len(b_manager.tasks))
                batch_size = max(1, int(args.batch_size))
                local_a_succ = 0
                local_a_att = 0
                local_b_succ = 0
                local_b_att = 0
                global_a_succ = 0
                global_a_att = 0
                global_b_succ = 0
                global_b_att = 0
                all_local_a_rewards: list[float] = []
                all_local_b_rewards: list[float] = []
                all_global_a_rewards: list[float] = []
                all_global_b_rewards: list[float] = []
                all_local_a_mt: list[dict[str, float]] = []
                all_local_b_mt: list[dict[str, float]] = []
                all_global_a_mt: list[dict[str, float]] = []
                all_global_b_mt: list[dict[str, float]] = []

                for start in range(0, total_local, batch_size):
                    end = min(start + batch_size, total_local)
                    batch_idx = (start // batch_size) + 1
                    num_batches = (total_local + batch_size - 1) // batch_size

                    # Step 1: local A/B in parallel on the same batch.
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        fut_a_local = executor.submit(
                            run_tasks, a_manager, start, end, args.tool_mode, alfworld_sp
                        )
                        fut_b_local = executor.submit(
                            run_tasks, b_manager, start, end, args.tool_mode, alfworld_sp
                        )
                        a_local_rewards, _, a_msgs, _, a_local_mt = fut_a_local.result()
                        b_local_rewards, _, b_msgs, _, b_local_mt = fut_b_local.result()

                    local_a_succ += sum(1 for r in a_local_rewards if r > 0)
                    local_a_att += len(a_local_rewards)
                    local_b_succ += sum(1 for r in b_local_rewards if r > 0)
                    local_b_att += len(b_local_rewards)
                    all_local_a_rewards.extend(a_local_rewards)
                    all_local_b_rewards.extend(b_local_rewards)
                    all_local_a_mt.extend(a_local_mt)
                    all_local_b_mt.extend(b_local_mt)

                    # Step 2: update global memory from this local batch.
                    _merge_peer_msgs(a_msgs, source_id="A")
                    _merge_peer_msgs(b_msgs, source_id="B")
                    persist_fn = getattr(global_mem, "persist_entity_graph", None)
                    if callable(persist_fn):
                        persist_fn()

                    # Step 3: run global-only A/B in parallel for the same batch.
                    probe_a_dir = os.path.join(dual_local_dir, f"{task_a_label}_global_only")
                    probe_b_dir = os.path.join(dual_local_dir, f"{task_b_label}_global_only")
                    ensure_dir(probe_a_dir)
                    ensure_dir(probe_b_dir)
                    log_train_global_only = os.path.join(log_base, "train_global_only")
                    a_manager_global = build_manager(task_a_name, probe_a_dir, log_train_global_only, task_a_train_tasks)
                    b_manager_global = build_manager(task_b_name, probe_b_dir, log_train_global_only, task_b_train_tasks)
                    a_manager_global.mem_config.update({**train_mem_extras, "freeze_memory": True})
                    b_manager_global.mem_config.update({**train_mem_extras, "freeze_memory": True})
                    a_manager_global.mas_config.update(alfworld_mas_overrides)
                    b_manager_global.mas_config.update(alfworld_mas_overrides)
                    a_manager_global.mas_config.update({"silent_mas": True})
                    b_manager_global.mas_config.update({"silent_mas": True})
                    a_mem_global = build_mas(a_manager_global, args.reasoning, args.mas_memory, args.model)
                    b_mem_global = build_mas(b_manager_global, args.reasoning, args.mas_memory, args.model)
                    set_global_a = getattr(a_mem_global, "set_global_retriever", None)
                    if callable(set_global_a):
                        set_global_a(global_mem)
                    set_global_b = getattr(b_mem_global, "set_global_retriever", None)
                    if callable(set_global_b):
                        set_global_b(global_mem)

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        fut_a_global = executor.submit(
                            run_tasks, a_manager_global, start, end, args.tool_mode, alfworld_sp
                        )
                        fut_b_global = executor.submit(
                            run_tasks, b_manager_global, start, end, args.tool_mode, alfworld_sp
                        )
                        a_global_rewards, _, _, _, a_global_mt = fut_a_global.result()
                        b_global_rewards, _, _, _, b_global_mt = fut_b_global.result()

                    global_a_succ += sum(1 for r in a_global_rewards if r > 0)
                    global_a_att += len(a_global_rewards)
                    global_b_succ += sum(1 for r in b_global_rewards if r > 0)
                    global_b_att += len(b_global_rewards)
                    all_global_a_rewards.extend(a_global_rewards)
                    all_global_b_rewards.extend(b_global_rewards)
                    all_global_a_mt.extend(a_global_mt)
                    all_global_b_mt.extend(b_global_mt)

                    batch_row = {
                        "batch_idx": batch_idx,
                        "start": start,
                        "end": end,
                        "local_A": {
                            **compute_metrics(a_local_rewards, a_local_mt),
                            "num_tasks": len(a_local_rewards),
                            "num_success": sum(1 for r in a_local_rewards if r > 0),
                        },
                        "local_B": {
                            **compute_metrics(b_local_rewards, b_local_mt),
                            "num_tasks": len(b_local_rewards),
                            "num_success": sum(1 for r in b_local_rewards if r > 0),
                        },
                        "global_A": {
                            **compute_metrics(a_global_rewards, a_global_mt),
                            "num_tasks": len(a_global_rewards),
                            "num_success": sum(1 for r in a_global_rewards if r > 0),
                        },
                        "global_B": {
                            **compute_metrics(b_global_rewards, b_global_mt),
                            "num_tasks": len(b_global_rewards),
                            "num_success": sum(1 for r in b_global_rewards if r > 0),
                        },
                        "cumulative": {
                            "local_A": {"num_success": local_a_succ, "num_tasks": local_a_att},
                            "local_B": {"num_success": local_b_succ, "num_tasks": local_b_att},
                            "global_A": {"num_success": global_a_succ, "num_tasks": global_a_att},
                            "global_B": {"num_success": global_b_succ, "num_tasks": global_b_att},
                        },
                    }
                    train_metrics["train_batches"].append(batch_row)

                    print(
                        f"\rA {end}/{total_local} | B {end}/{total_local} | "
                        f"GG {batch_idx}/{num_batches} | train(mt batch) | "
                        f"A_local {local_a_succ}/{local_a_att} | B_local {local_b_succ}/{local_b_att} | "
                        f"A_global {global_a_succ}/{global_a_att} | B_global {global_b_succ}/{global_b_att}",
                        end="" if end < total_local else "\n",
                        flush=True,
                    )

                persist_fn = getattr(global_mem, "persist_entity_graph", None)
                if callable(persist_fn):
                    persist_fn()
                train_metrics["train_summary"] = {
                    "local_A": {
                        **compute_metrics(all_local_a_rewards, all_local_a_mt),
                        "num_tasks": len(all_local_a_rewards),
                        "num_success": sum(1 for r in all_local_a_rewards if r > 0),
                    },
                    "local_B": {
                        **compute_metrics(all_local_b_rewards, all_local_b_mt),
                        "num_tasks": len(all_local_b_rewards),
                        "num_success": sum(1 for r in all_local_b_rewards if r > 0),
                    },
                    "global_A": {
                        **compute_metrics(all_global_a_rewards, all_global_a_mt),
                        "num_tasks": len(all_global_a_rewards),
                        "num_success": sum(1 for r in all_global_a_rewards if r > 0),
                    },
                    "global_B": {
                        **compute_metrics(all_global_b_rewards, all_global_b_mt),
                        "num_tasks": len(all_global_b_rewards),
                        "num_success": sum(1 for r in all_global_b_rewards if r > 0),
                    },
                    "num_batches": (total_local + batch_size - 1) // batch_size if total_local > 0 else 0,
                    "batch_size": batch_size,
                }
            else:
                # Legacy non-selectivemem behavior for medmcqa/alfworld.
                total_local = min(len(a_manager.tasks), len(b_manager.tasks))
                total_effective = int(total_local)
                a_done = 0
                b_done = 0
                a_succ = 0
                a_att = 0
                b_succ = 0
                b_att = 0
                a_removed = 0
                b_removed = 0

                def _hook_a(task_idx_1based: int, _total: int, _succ: int, _attempted: int) -> None:
                    nonlocal a_done, b_done, a_succ, a_att, b_succ, b_att, total_effective, a_removed, b_removed
                    a_done = max(a_done, int(task_idx_1based))
                    a_succ = int(_succ)
                    a_att = int(_attempted)
                    a_removed = max(0, int(total_local) - int(_total))
                    total_effective = max(0, int(total_local) - max(a_removed, b_removed))
                    _print_train_progress(
                        phase="train(g-memory)",
                        a_done=a_done,
                        b_done=b_done,
                        total=total_effective,
                        gg_done=0,
                        gg_total=1,
                        a_succ=a_succ,
                        a_att=a_att,
                        b_succ=b_succ,
                        b_att=b_att,
                    )

                def _hook_b(task_idx_1based: int, _total: int, _succ: int, _attempted: int) -> None:
                    nonlocal a_done, b_done, a_succ, a_att, b_succ, b_att, total_effective, a_removed, b_removed
                    b_done = max(b_done, int(task_idx_1based))
                    b_succ = int(_succ)
                    b_att = int(_attempted)
                    b_removed = max(0, int(total_local) - int(_total))
                    total_effective = max(0, int(total_local) - max(a_removed, b_removed))
                    _print_train_progress(
                        phase="train(g-memory)",
                        a_done=a_done,
                        b_done=b_done,
                        total=total_effective,
                        gg_done=0,
                        gg_total=1,
                        a_succ=a_succ,
                        a_att=a_att,
                        b_succ=b_succ,
                        b_att=b_att,
                    )

                run_tasks(
                    a_manager,
                    0,
                    total_local,
                    args.tool_mode,
                    alfworld_subprocess_args=alfworld_sp,
                    progress_hook=_hook_a,
                )
                run_tasks(
                    b_manager,
                    0,
                    total_local,
                    args.tool_mode,
                    alfworld_subprocess_args=alfworld_sp,
                    progress_hook=_hook_b,
                )
                _print_train_progress(
                    phase="train(g-memory)",
                    a_done=a_done,
                    b_done=b_done,
                    total=total_effective,
                    gg_done=0,
                    gg_total=1,
                    a_succ=a_succ,
                    a_att=a_att,
                    b_succ=b_succ,
                    b_att=b_att,
                    endline=True,
                )

        # Force persist local entity graphs
        for mgr, tag in [(a_manager, task_a_label), (b_manager, task_b_label)]:
            persist_fn = getattr(getattr(mgr.mas, "meta_memory", None), "persist_entity_graph", None)
            if callable(persist_fn):
                persist_fn()
                # Persisted artifact is still saved to disk; keep terminal quiet.

    # ---- Build global memory ----
    # SelectiveMem: GG is rebuilt during local training in batches (with snapshots),
    # so we only need a final rebuild here for safety (no snapshot).
    if args.mas_memory == "selectivemem":
        rebuild_selectivemem_global_from_locals(
            local_dirs=[local_a_dir, local_b_dir],
            global_dir=global_dir,
            model_name=args.model,
            snapshot_tag=None,
        )
    else:
        if args.dataset_family == "mtmind2web":
            # MT branch has already updated global memory incrementally
            # after each local batch in the training phase.
            pass
        elif args.eval_only or global_graph_exists(global_dir, args.mas_memory):
            if args.eval_only:
                pass
            else:
                pass
        else:
            dual_local_a_dir = os.path.join(dual_local_dir, task_a_label)
            dual_local_b_dir = os.path.join(dual_local_dir, task_b_label)
            ensure_dir(dual_local_a_dir)
            ensure_dir(dual_local_b_dir)

            log_train_global = os.path.join(log_base, "train_global")
            a_manager_dual = build_manager(task_a_name, dual_local_a_dir, log_train_global, task_a_train_tasks)
            b_manager_dual = build_manager(task_b_name, dual_local_b_dir, log_train_global, task_b_train_tasks)
            apply_collab_mem_tags(a_manager_dual, "A", task_a_label, "dual_global_peer")
            apply_collab_mem_tags(b_manager_dual, "B", task_b_label, "dual_global_peer")
            a_manager_dual.mas_config.update(alfworld_mas_overrides)
            b_manager_dual.mas_config.update(alfworld_mas_overrides)
            a_manager_dual.mas_config.update({"silent_mas": True})
            b_manager_dual.mas_config.update({"silent_mas": True})
            a_mem_dual = build_mas(a_manager_dual, args.reasoning, args.mas_memory, args.model)
            b_mem_dual = build_mas(b_manager_dual, args.reasoning, args.mas_memory, args.model)

            _reasoning_cls, global_mem_cls = module_map(args.reasoning, args.mas_memory)
            global_mem = global_mem_cls(
                namespace=args.mas_memory,
                global_config={
                    "working_dir": global_dir,
                    **train_mem_extras,
                },
                llm_model=GPTChat(model_name=args.model),
                embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            )

            set_global = getattr(a_mem_dual, "set_global_retriever", None)
            if callable(set_global):
                a_mem_dual.set_global_retriever(global_mem)
            set_global = getattr(b_mem_dual, "set_global_retriever", None)
            if callable(set_global):
                b_mem_dual.set_global_retriever(global_mem)

            total = min(len(a_manager_dual.tasks), len(b_manager_dual.tasks))
            batch_size = max(1, args.batch_size)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                alfworld_sp = {"reasoning": args.reasoning, "model": args.model, "max_trials": args.max_trials} if args.dataset_family == "alfworld" else None
                with ThreadPoolExecutor(max_workers=2) as executor:
                    fut_a = executor.submit(run_tasks, a_manager_dual, start, end, args.tool_mode, alfworld_sp)
                    fut_b = executor.submit(run_tasks, b_manager_dual, start, end, args.tool_mode, alfworld_sp)
                    _, _, a_msgs, _, _ = fut_a.result()
                    _, _, b_msgs, _, _ = fut_b.result()

                # In alfworld subprocess mode, `run_tasks` returns `saved_message` which may be a dict
                # (serialized MASMessage) instead of a MASMessage instance. Convert back before merge.
                from mas.memory.common import MASMessage  # type: ignore

                def _coerce_mas_message(x: Any) -> MASMessage:
                    if isinstance(x, MASMessage):
                        return x
                    if isinstance(x, dict):
                        return MASMessage.from_dict(x)
                    raise TypeError(f"Unsupported peer message type for global merge: {type(x)}")

                for msg in a_msgs:
                    if hasattr(global_mem, "add_memory_from_peer"):
                        global_mem.add_memory_from_peer(_coerce_mas_message(msg), source_id="A")
                    else:
                        global_mem.add_memory(_coerce_mas_message(msg))
                for msg in b_msgs:
                    if hasattr(global_mem, "add_memory_from_peer"):
                        global_mem.add_memory_from_peer(_coerce_mas_message(msg), source_id="B")
                    else:
                        global_mem.add_memory(_coerce_mas_message(msg))

                # ALFWorld: sync persist every batch
                if alfworld_mem_overrides:
                    persist_fn = getattr(global_mem, "persist_entity_graph", None)
                    if callable(persist_fn):
                        persist_fn()

            persist_fn = getattr(global_mem, "persist_entity_graph", None)
            if callable(persist_fn):
                persist_fn()

    # ---- Evaluation (freeze memory) ----
    if args.dataset_family == "mtmind2web":
        # MT collaborative setting requested by user:
        # 2 local-only results + 2 local+global results, all on test sets.
        EVAL_SCENARIOS = [
            ("local_A_on_A_test", task_a_eval_name, local_a_dir, task_a_eval_tasks, False),
            ("local_B_on_B_test", task_b_eval_name, local_b_dir, task_b_eval_tasks, False),
            ("global_A_on_A_test", task_a_eval_name, local_a_dir, task_a_eval_tasks, True),
            ("global_B_on_B_test", task_b_eval_name, local_b_dir, task_b_eval_tasks, True),
        ]
    else:
        # 6 tasks: inner/cross × baseline/ours (no redundant cross_ours, since they duplicate inner_ours)
        EVAL_SCENARIOS = [
            ("inner_baseline_A_on_A", task_a_eval_name, local_a_dir, task_a_eval_tasks, False),
            ("inner_baseline_B_on_B", task_b_eval_name, local_b_dir, task_b_eval_tasks, False),
            # Ours: use local retrieval as base, then augment with global abstract insights.
            ("inner_ours_A_on_A", task_a_eval_name, local_a_dir, task_a_eval_tasks, True),
            ("inner_ours_B_on_B", task_b_eval_name, local_b_dir, task_b_eval_tasks, True),
            ("cross_baseline_A_on_B", task_b_eval_name, local_a_dir, task_b_eval_tasks, False),
            ("cross_baseline_B_on_A", task_a_eval_name, local_b_dir, task_a_eval_tasks, False),
        ]

    raw = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    # When max_eval=0, skip evaluation entirely (v3_ss may not provide valid_unseen files).
    if args.max_eval is not None and int(args.max_eval) <= 0:
        raw = []
    if not raw:
        raw = ["all"]
    if args.max_eval is not None and int(args.max_eval) <= 0:
        selected_indices = []
    elif "all" in raw:
        selected_indices = list(range(len(EVAL_SCENARIOS)))
    else:
        name_to_idx = {sc[0]: i for i, sc in enumerate(EVAL_SCENARIOS)}
        selected_indices = []
        for s in raw:
            if s.isdigit():
                idx = int(s) - 1
                if 0 <= idx < len(EVAL_SCENARIOS):
                    selected_indices.append(idx)
            elif s in name_to_idx:
                selected_indices.append(name_to_idx[s])
        selected_indices = sorted(set(selected_indices))

    def eval_on(
        task_name: str,
        memory_dir: str,
        tag: str,
        tasks_override: list[dict] | None = None,
        use_global_insights: bool = False,
    ) -> dict[str, float]:
        log_eval = os.path.join(log_base, "eval", tag)
        manager = build_manager(task_name, memory_dir, log_eval, tasks_override)
        if args.dataset_family == "alfworld" and args.mas_memory == "selectivemem":
            mem_norm = memory_dir.replace("\\", "/")
            if task_a_label in mem_norm:
                apply_collab_mem_tags(manager, "A", task_a_label, "eval")
            elif task_b_label in mem_norm:
                apply_collab_mem_tags(manager, "B", task_b_label, "eval")
        manager.mem_config.update({**mem_profile_extras, "freeze_memory": True})
        manager.mas_config.update(alfworld_mas_overrides)
        manager.mas_config.update({"silent_mas": True})
        eval_mem = build_mas(manager, args.reasoning, args.mas_memory, args.model)
        if use_global_insights and args.mas_memory == "selectivemem":
            global_retriever = SelectiveMemMASMemory(
                namespace=args.mas_memory,
                global_config={
                    "working_dir": global_dir,
                    "freeze_memory": True,
                    "insights_only": True,
                },
                llm_model=GPTChat(model_name=args.model),
                embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            )
            set_global = getattr(eval_mem, "set_global_retriever", None)
            if callable(set_global):
                eval_mem.set_global_retriever(global_retriever)
        elif use_global_insights:
            _reasoning_cls, global_mem_cls = module_map(args.reasoning, args.mas_memory)
            global_retriever = global_mem_cls(
                namespace=args.mas_memory,
                global_config={
                    "working_dir": global_dir,
                    "freeze_memory": True,
                    **mem_profile_extras,
                },
                llm_model=GPTChat(model_name=args.model),
                embedding_func=EmbeddingFunc(CONFIG.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")),
            )
            set_global = getattr(eval_mem, "set_global_retriever", None)
            if callable(set_global):
                eval_mem.set_global_retriever(global_retriever)
        # Optional quick-test mode: restrict to first 5 physio test questions on inner_ours_A_on_A
        if (
            args.dataset_family == "medmcqa"
            and args.test_mode == "physio5"
            and tag == "inner_ours_A_on_A"
            and task_name == args.task_a_test
        ):
            end = min(5, len(manager.tasks))
        else:
            end = len(manager.tasks)
        alfworld_sp = {"reasoning": args.reasoning, "model": args.model, "max_trials": args.max_trials} if args.dataset_family == "alfworld" else None
        if alfworld_sp and use_global_insights and args.mas_memory == "selectivemem":
            alfworld_sp["use_global_insights"] = True
            alfworld_sp["global_dir"] = global_dir
        rewards, _, _, skipped_tasks, eval_mt = run_tasks(
            manager, 0, end, args.tool_mode, alfworld_subprocess_args=alfworld_sp
        )
        if skipped_tasks:
            load_err_count = sum(
                1 for s in skipped_tasks
                if "FailedToken" in s.get("error_type", "")
                or "FailedCut" in s.get("error_type", "")
                or s.get("error_type") == "IndexError"
                or "expecting" in s.get("error_message", "")
                or "pop from empty list" in s.get("error_message", "")
                or "left recursion" in s.get("error_message", "").lower()
                or ("subprocess" in s.get("error_message", "").lower() and ("139" in s.get("error_message", "") or "crash" in s.get("error_message", "").lower()))
            )
            if load_err_count:
                manager.recorder.log(
                    f"[EVAL] {load_err_count}/{len(skipped_tasks)} skips were game load/grammar errors. "
                    "Re-run materialize_collab_subsets with is_valid_gamefile to exclude broken games."
                )
        metrics = compute_metrics(rewards, eval_mt)
        metrics["num_tasks"] = end
        metrics["num_completed"] = len(rewards)
        metrics["num_skipped"] = len(skipped_tasks)
        metrics["skipped_tasks"] = skipped_tasks
        return metrics

    results = []
    for idx in selected_indices:
        scenario, task_name, memory_dir, tasks_override, use_global_insights = EVAL_SCENARIOS[idx]
        results.append(
            {
                "scenario": scenario,
                **eval_on(
                    task_name,
                    memory_dir,
                    scenario,
                    tasks_override=tasks_override,
                    use_global_insights=use_global_insights,
                ),
            }
        )

    # Always save results to a new timestamped folder under reports/collab
    report_ts = time.strftime("%Y%m%d_%H%M%S")
    collab_dir = os.path.join(report_dir, "collab", report_ts)
    ensure_dir(collab_dir)

    output_json = os.path.join(collab_dir, f"{args.dataset_family}_collab_domain_eval.json")
    output_md = os.path.join(collab_dir, f"{args.dataset_family}_collab_domain_eval.md")
    train_output_json = os.path.join(collab_dir, f"{args.dataset_family}_collab_train_metrics.json")
    train_output_md = os.path.join(collab_dir, f"{args.dataset_family}_collab_train_metrics.md")
    memory_nl_md = os.path.join(collab_dir, f"{args.dataset_family}_collab_memory_nl.md")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.dataset_family} Collaborative Domain Adaptation Evaluation\n\n")
        show_mt_eval = any("avg_ele_acc" in it for it in results)
        if show_mt_eval:
            f.write(
                "| Scenario | Accuracy | Avg Reward | avg_Ele.Acc | avg_Op.F1 | avg_SSR | avg_TSR | "
                "Requested | Completed | Skipped |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for item in results:
                f.write(
                    f"| {item['scenario']} | {item['accuracy']:.4f} | {item['avg_reward']:.4f} | "
                    f"{float(item.get('avg_ele_acc', 0.0)):.4f} | {float(item.get('avg_op_f1', 0.0)):.4f} | "
                    f"{float(item.get('avg_ssr', 0.0)):.4f} | {float(item.get('avg_tsr', 0.0)):.4f} | "
                    f"{item['num_tasks']} | {item['num_completed']} | {item['num_skipped']} |\n"
                )
        else:
            f.write("| Scenario | Accuracy | Avg Reward | Requested | Completed | Skipped |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for item in results:
                f.write(
                    f"| {item['scenario']} | {item['accuracy']:.4f} | {item['avg_reward']:.4f} | "
                    f"{item['num_tasks']} | {item['num_completed']} | {item['num_skipped']} |\n"
                )

    # Persist all train metrics (batch-level + summary), so both train/test are recorded.
    with open(train_output_json, "w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

    with open(train_output_md, "w", encoding="utf-8") as f:
        f.write(f"# {args.dataset_family} Collaborative Training Metrics\n\n")
        summary = train_metrics.get("train_summary") or {}
        if summary:
            f.write("## Summary\n\n")
            show_mt_train = any(
                "avg_ele_acc" in (summary.get(k) or {}) for k in ("local_A", "local_B", "global_A", "global_B")
            )
            if show_mt_train:
                f.write(
                    "| Split | Accuracy | Avg Reward | avg_Ele.Acc | avg_Op.F1 | avg_SSR | avg_TSR | Tasks | Success |\n"
                )
                f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                for key in ("local_A", "local_B", "global_A", "global_B"):
                    row = summary.get(key) or {}
                    f.write(
                        f"| {key} | {float(row.get('accuracy', 0.0)):.4f} | {float(row.get('avg_reward', 0.0)):.4f} | "
                        f"{float(row.get('avg_ele_acc', 0.0)):.4f} | {float(row.get('avg_op_f1', 0.0)):.4f} | "
                        f"{float(row.get('avg_ssr', 0.0)):.4f} | {float(row.get('avg_tsr', 0.0)):.4f} | "
                        f"{int(row.get('num_tasks', 0))} | {int(row.get('num_success', 0))} |\n"
                    )
            else:
                f.write("| Split | Accuracy | Avg Reward | Tasks | Success |\n")
                f.write("|---|---:|---:|---:|---:|\n")
                for key in ("local_A", "local_B", "global_A", "global_B"):
                    row = summary.get(key) or {}
                    f.write(
                        f"| {key} | {float(row.get('accuracy', 0.0)):.4f} | {float(row.get('avg_reward', 0.0)):.4f} | "
                        f"{int(row.get('num_tasks', 0))} | {int(row.get('num_success', 0))} |\n"
                    )
            f.write("\n")
            f.write(
                f"- batch_size: {int(summary.get('batch_size', 0))}\n"
                f"- num_batches: {int(summary.get('num_batches', 0))}\n\n"
            )

        batches = train_metrics.get("train_batches") or []
        if batches:
            f.write("## Batch Details\n\n")
            f.write("| Batch | Range | local_A | local_B | global_A | global_B |\n")
            f.write("|---:|---:|---:|---:|---:|---:|\n")
            for row in batches:
                s = int(row.get("start", 0))
                e = int(row.get("end", 0))
                la = row.get("local_A", {})
                lb = row.get("local_B", {})
                ga = row.get("global_A", {})
                gb = row.get("global_B", {})
                f.write(
                    f"| {int(row.get('batch_idx', 0))} | {s}-{e} | "
                    f"{float(la.get('accuracy', 0.0)):.4f} ({int(la.get('num_success', 0))}/{int(la.get('num_tasks', 0))}) | "
                    f"{float(lb.get('accuracy', 0.0)):.4f} ({int(lb.get('num_success', 0))}/{int(lb.get('num_tasks', 0))}) | "
                    f"{float(ga.get('accuracy', 0.0)):.4f} ({int(ga.get('num_success', 0))}/{int(ga.get('num_tasks', 0))}) | "
                    f"{float(gb.get('accuracy', 0.0)):.4f} ({int(gb.get('num_success', 0))}/{int(gb.get('num_tasks', 0))}) |\n"
                )

    _dump_memory_nl_report(
        output_path=memory_nl_md,
        mas_memory=args.mas_memory,
        local_a_dir=local_a_dir,
        local_b_dir=local_b_dir,
        global_dir=global_dir,
        task_a_label=task_a_label,
        task_b_label=task_b_label,
    )

    # Results are written to disk; keep terminal output quiet.
    if getattr(args, "profile_timing", False):
        print_timing_report(
            log_base,
            global_dir,
            banner=(
                f"NV_DAMAS 时间剖析汇总 | run_id={run_id} | dataset={args.dataset_family} | "
                f"memory={args.mas_memory} | model_type={model_type}"
            ),
        )


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--alfworld-worker":
        try:
            _run_one_alfworld_task_worker(sys.argv[2], sys.argv[3], sys.argv[4])
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
    main()
