from __future__ import annotations

import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import shutil
import yaml
from dataclasses import dataclass, field
import argparse
import json
import random
import time
import traceback

import mas
from mas.agents import Agent
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase, GMemory
from mas.llm import LLMCallable, GPTChat, get_price
from mas.mas import MetaMAS
from mas.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type


with open('tasks/configs.yaml') as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None
LOG_DIR: str = None


def load_alfworld_tasks_json(path: str) -> list[dict]:
    """
    Load ALFWorld task list from a JSON file (e.g. collab_subsets/v3_s/kitchen__train.json).

    Each entry must include env_kwargs.gamefile, env_name, task_type, and task or goal_instruction
    (same fields as materialized collab subset rows).
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"alfworld tasks json not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {abs_path}")
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Task entry {i} is not an object")
        ek = row.get("env_kwargs") or {}
        if not ek.get("gamefile"):
            raise ValueError(f"Task entry {i} missing env_kwargs.gamefile")
        if row.get("env_name") is None:
            raise ValueError(f"Task entry {i} missing env_name")
        if row.get("task_type") is None:
            raise ValueError(f"Task entry {i} missing task_type")
        if not row.get("task") and not row.get("goal_instruction"):
            raise ValueError(f"Task entry {i} needs task or goal_instruction")
    return data


def _export_alfworld_history_if_enabled(
    task_manager: TaskManager,
    task_config: dict,
    *,
    status_override: str | None = None,
) -> str | None:
    if task_manager.task_name != "alfworld":
        return None
    export_enabled = bool(task_manager.mem_config.get("export_gm2_history", False))
    if not export_enabled:
        return None
    history_dir = str(task_manager.mem_config.get("gm2_history_dir", "") or "").strip()
    if not history_dir:
        return None
    env = getattr(task_manager, "env", None)
    if env is None or not hasattr(env, "export_gm2_history"):
        return None
    model_id = str(task_config.get("model_type") or task_manager.mas_config.get("model_type", "") or "")
    try:
        return env.export_gm2_history(history_dir, model_id=model_id, status_override=status_override)
    except Exception as exc:
        task_manager.recorder.log(f"[gm2_history_export] failed: {type(exc).__name__}: {exc}")
        return None


@dataclass
class TaskManager:
    task_name: str              # task name
    mas_type: str               # type of mas
    memory_type: str            # memory type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    mas: MetaMAS                # multi-agent system
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs


def build_task(
    task: str,
    mas_type: str,
    memory_type: str,
    max_steps: int,
    *,
    alfworld_game_root: str = "",
) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)
    if task == "alfworld" and str(alfworld_game_root or "").strip():
        config["external_game_root"] = os.path.abspath(str(alfworld_game_root).strip())

    recorder: BaseRecorder = get_recorder(task, working_dir=LOG_DIR, namespace='total_task')
    recorder.log(f"[startup] task={task} mas_type={mas_type} memory_type={memory_type} log_dir={os.path.abspath(LOG_DIR)}")
    if task == "alfworld" and config.get("external_game_root"):
        recorder.log(f"[startup] alfworld_game_root={config['external_game_root']}")
        print(f"ALFWorld game root remap: {config['external_game_root']}", flush=True)
    env: BaseEnv = get_env(task, config, max_steps)
    tasks: list[dict] = get_task(task, env_config=config)
    mas_workflow: MetaMAS = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        mas_config=mas_config
    )   

def build_mas(
    task_manager: TaskManager,
    reasoning: str = None,
    mas_memory: str = None,
    llm_type: str = None,
) -> None:
    
    embed_func = EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")) 
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    llm_model: LLMCallable = GPTChat(model_name=llm_type)
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func 
    )
    
    task_manager.mas.add_observer(task_manager.recorder)  
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)

def run_task(task_manager: TaskManager) -> None:

    task_manager.recorder.dataset_begin()
    total_tasks = len(task_manager.tasks)
    start_time = time.time()
    # Cumulative success count for ALFWorld (aligned with env.feedback success / schedule final_done).
    alf_ok = 0

    for task_id, task_config in enumerate(task_manager.tasks):
        # Text progress: k / n
        progress_msg = f"[Progress: {task_id + 1}/{total_tasks}]"
        print(f"\n{progress_msg} Start task {task_id + 1}...", flush=True)
        task_manager.recorder.log(progress_msg)
        
        task_manager.recorder.task_begin(task_id, task_config)  

        try:
            task_main, task_description = task_manager.mas.env.set_env(task_config)
            few_shots: list[str] = get_task_few_shots(
                dataset=task_manager.task_name,
                task_config=task_config,
                few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0),
            )
            task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)
            task_config["env_ref"] = task_manager.env
            task_config["task_index"] = task_id
            task_config["task_index_1based"] = task_id + 1
            task_config["mas_memory_type"] = task_manager.memory_type
            task_config["model_type"] = task_manager.mas_config.get("model_type", "")
            if task_manager.memory_type in ("g-memory", "selectivemem"):
                _ma_dir = os.path.join(task_manager.recorder.working_dir, "memory_augmentation")
                os.makedirs(_ma_dir, exist_ok=True)
                task_config["memory_augment_log_dir"] = _ma_dir

            task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
            for agent in task_manager.mas.agents_team.values():
                task_manager.recorder.log(f"------------ MAS Agent: {agent.name} ------------")
                task_manager.recorder.log(agent.add_task_instruction(task_instruction))

            reward, done = task_manager.mas.schedule(task_config)
            task_manager.recorder.task_end(reward, done)
            exported = _export_alfworld_history_if_enabled(task_manager, task_config)
            if exported:
                task_manager.recorder.log(f"[gm2_history_export] saved: {exported}")
        except Exception as e:
            # Robustness: ALFWorld/TextWorld occasionally crashes on some gamefiles (e.g. PDDL parser issues).
            # We skip the task but keep the run alive, logging full traceback for later inspection.
            tb = traceback.format_exc()
            gamefile = ((task_config.get("env_kwargs") or {}).get("gamefile")) if isinstance(task_config, dict) else None
            gamefile_s = f" gamefile={gamefile}" if gamefile else ""
            task_manager.recorder.log(
                f"{progress_msg} Task {task_id + 1} skipped (env/MAS crash): {type(e).__name__}: {e}{gamefile_s}\n{tb}"
            )
            if task_manager.task_name == "alfworld":
                print(
                    f"{progress_msg} Task {task_id + 1} skipped (crash): {type(e).__name__}: {e}{gamefile_s}",
                    flush=True,
                )
            # Treat skipped task as failure for running accuracy accounting.
            reward, done = 0.0, False
            task_manager.recorder.task_end(reward, done)
            exported = _export_alfworld_history_if_enabled(task_manager, task_config, status_override="fail")
            if exported:
                task_manager.recorder.log(f"[gm2_history_export] saved after crash: {exported}")
            # Continue to next task.
            continue
        
        elapsed = time.time() - start_time
        n_done = task_id + 1
        if task_manager.task_name == "alfworld":
            ok = bool(done)
            if ok:
                alf_ok += 1
            running_acc = alf_ok / n_done if n_done else 0.0
            print(
                f"{progress_msg} Task {n_done} done | "
                f"this_task={'SUCCESS' if ok else 'FAIL'} | "
                f"running_acc={alf_ok}/{n_done} ({running_acc:.1%}) | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(f"{progress_msg} Task {n_done} finished (elapsed: {elapsed:.1f}s)", flush=True)
    
    task_manager.recorder.dataset_end()
    
    # Print total running time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    if hours > 0:
        time_str = f"{hours}h{minutes}m{seconds:.1f}s"
    elif minutes > 0:
        time_str = f"{minutes}m{seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"

    print(f"\n{'='*60}")
    print("All tasks finished!")
    print(f"Total tasks: {total_tasks}")
    if task_manager.task_name == "alfworld" and total_tasks > 0:
        print(
            f"Final accuracy: {alf_ok}/{total_tasks} ({alf_ok / total_tasks:.1%})",
            flush=True,
        )
    print(f"Total running time: {time_str} ({total_time:.2f}s)")
    print(f"{'='*60}\n", flush=True)
    task_manager.recorder.log(f"\n{'='*60}")
    task_manager.recorder.log(f"All tasks finished! Total tasks: {total_tasks}, total running time: {time_str}")
    task_manager.recorder.log(f"{'='*60}\n")



if __name__ == '__main__':
    # settings
    random.seed(42)

    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'alfworld',
            'fever',
            'fever_aca_test',
            'fever_oral_test',
            'pddl',
            'huskyqa',
            'huskyqa_aca_test',
            'medmcqa_test',
            'medmcqa_physio_30',
            'medmcqa_pharma_30',
            'medmcqa_physio_3',
            'medmcqa_pharma_3',
            'medmcqa_anatomy_20',
            'medmcqa_surgery_20',
            'mtmind2web_train',
            'mtmind2web_test_task',
            'mtmind2web_test_website',
            'mtmind2web_test_subdomain',
            'scienceworld_train',
            'scienceworld_dev',
            'scienceworld_test',
            'scienceworld_domain_a_train',
            'scienceworld_domain_b_train',
            'scienceworld_domain_a_test',
            'scienceworld_domain_b_test',
        ],
    )
    parser.add_argument('--mas_type', type=str, choices=['autogen', 'macnet', 'dylan', 'strategy'])
    parser.add_argument('--mas_memory', type=str, default='g-memory', help='Specify mas memory module')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=50, help='max number of steps')
    parser.add_argument('--mas_trials', type=int, dest='max_trials', help='(deprecated) use --max_trials instead. max number of steps')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help=(
            'Print full MAS step logs and recorder output to the terminal. '
            'Default is quiet: only progress lines (and per-task accuracy for ALFWorld).'
        ),
    )
    parser.add_argument(
        '--alfworld_tasks_json',
        type=str,
        default=None,
        help=(
            'ALFWorld only: path to a JSON array of task dicts (e.g. '
            'data/alfworld/collab_subsets/v3_s/kitchen__train.json). '
            'When set, replaces the default alfworld_tasks_suffix list.'
        ),
    )
    parser.add_argument(
        '--alfworld_max_tasks',
        type=int,
        default=None,
        help='ALFWorld only: after loading --alfworld_tasks_json, run at most this many tasks (head slice).',
    )
    parser.add_argument(
        '--run_id',
        type=str,
        default=None,
        help='Optional run identifier to isolate memory/log outputs (creates subfolders under .db and logs).',
    )
    parser.add_argument(
        '--reset_memory',
        action='store_true',
        help='Delete this run_id memory directory under .db before starting; use for fresh train/build runs.',
    )
    parser.add_argument(
        '--max_tasks',
        type=int,
        default=None,
        help='Run at most this many tasks for any dataset (head slice).',
    )
    parser.add_argument(
        '--mtmind2web_shared_memory',
        action='store_true',
        help=(
            'MT-Mind2Web only: store/load memory under .db/.../mtmind2web/... (same --run_id) '
            'so mtmind2web_train and mtmind2web_test_* runs share one G-Memory working_dir.'
        ),
    )
    parser.add_argument(
        '--export_gm2_history',
        action='store_true',
        help='ALFWorld only: export gm2-compatible history_*.json files while running the original project pipeline.',
    )
    parser.add_argument(
        '--gm2_history_dir',
        type=str,
        default='',
        help='ALFWorld only: directory for exported gm2-compatible history files. Defaults to <log_dir>/game_trajectory.',
    )
    parser.add_argument(
        '--alfworld_game_root',
        type=str,
        default='',
        help=(
            'ALFWorld only: external json_2.1.1 root to remap subset gamefile paths, '
            'for example /bigdata/.../run_alf/ALFWORLD_DATA/json_2.1.1'
        ),
    )
    parser.add_argument(
        '--gm2_memory_dir',
        type=str,
        default='',
        help='GraphMemory2 only: optional external directory for prebuilt/static gm2 memory artifacts.',
    )
    parser.add_argument(
        '--gm2_repo_root',
        type=str,
        default='',
        help='Deprecated: GraphMemory2 graph modules are vendored under gm2_backend.',
    )
    parser.add_argument(
        '--gm2_dynamic_graph',
        action='store_true',
        help='GraphMemory2 only: dynamically build real local/global graph memory from episodes in this run.',
    )
    parser.add_argument(
        '--gm2_retrieval_mode',
        type=str,
        default='lightweight',
        choices=[
            'lightweight',
            'query',
            'phasee_compat',
            'phasee_policy',
            'phasee_action',
            'hybrid_policy',
            'hybrid_repair',
            'lightweight_repair',
            'graph_policy',
            'graph_policy_rerank',
            'graph_policy_feedback',
            'graph_policy_candidate',
            'graph_policy_quality',
        ],
        help='GraphMemory2 only: retrieval mode for external graph memory artifacts.',
    )
    parser.add_argument(
        '--gm2_settings',
        type=str,
        default='local_only',
        choices=['base', 'local_only', 'global_only', 'local_plus_global'],
        help='GraphMemory2 only: which external graph memories to use.',
    )
    parser.add_argument(
        '--gm2_owner_scene',
        type=str,
        default='',
        help='GraphMemory2 only: local memory scene name, e.g. kitchen for local_kitchen.json.',
    )
    parser.add_argument(
        '--gm2_freeze_memory',
        action='store_true',
        help='GraphMemory2 only: disable cross-task memory updates and use committed memory in read-only mode.',
    )
    parser.add_argument(
        '--gm2_enable_overlay',
        action='store_true',
        help='GraphMemory2 only: enable per-episode overlay updates during a task.',
    )
    parser.add_argument(
        '--gm2_promotion_threshold',
        type=float,
        default=0.35,
        help='GraphMemory2 only: placeholder threshold for local-to-global promotion logic.',
    )
    parser.add_argument(
        '--gm3_use_textgrad',
        action='store_true',
        help='GraphMemory3 only: optimize the injected memory prompt with a TextGrad/TextLoss judge-rewrite loop.',
    )
    parser.add_argument(
        '--gm3_textgrad_engine',
        type=str,
        default='',
        help='GraphMemory3 only: TextGrad engine name, e.g. experimental:openai/qwen32b-api.',
    )

    args = parser.parse_args()

    task: str = args.task
    mas_type: str = args.mas_type
    max_trials: int = args.max_trials
    model_type: str = args.model
    mas_memory_type: str = args.mas_memory
    reasoning_type: str = args.reasoning
    
    # dir (MT-Mind2Web: optional unified memory root so train then test reuses G-Memory)
    mem_task_key = task
    if args.mtmind2web_shared_memory and task.startswith('mtmind2web_'):
        mem_task_key = 'mtmind2web'
    WORKING_DIR = os.path.join('./.db', get_model_type(model_type), mem_task_key, mas_type, 'memory', f'{mas_memory_type}')
    LOG_DIR = os.path.join('./logs', task, mas_type, 'memory', f'{mas_memory_type}', get_model_type(model_type))
    if args.run_id:
        WORKING_DIR = os.path.join(WORKING_DIR, args.run_id)
        LOG_DIR = os.path.join(LOG_DIR, args.run_id)
    if args.reset_memory and os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if args.mtmind2web_shared_memory and task.startswith('mtmind2web_'):
        print(
            f"MT-Mind2Web shared G-Memory working_dir: {os.path.abspath(WORKING_DIR)}",
            flush=True,
        )
    
    # run tasks
    task_configs: TaskManager = build_task(
        task,
        mas_type,
        mas_memory_type,
        max_trials,
        alfworld_game_root=args.alfworld_game_root,
    )
    if task == 'alfworld' and args.alfworld_tasks_json:
        loaded = load_alfworld_tasks_json(args.alfworld_tasks_json)
        if args.alfworld_max_tasks is not None:
            n = max(0, int(args.alfworld_max_tasks))
            loaded = loaded[:n]
        task_configs.tasks = loaded
        subset_msg = (
            f"[alfworld_tasks_json] loaded {len(loaded)} tasks from "
            f"{os.path.abspath(args.alfworld_tasks_json)}"
        )
        print(subset_msg, flush=True)
        task_configs.recorder.log(subset_msg)
        if loaded:
            first_gamefile = ((loaded[0].get("env_kwargs") or {}).get("gamefile")) if isinstance(loaded[0], dict) else None
            if first_gamefile:
                first_msg = f"[alfworld_tasks_json] first gamefile: {first_gamefile}"
                print(first_msg, flush=True)
                task_configs.recorder.log(first_msg)
    if args.max_tasks is not None:
        n_any = max(0, int(args.max_tasks))
        task_configs.tasks = task_configs.tasks[:n_any]
    task_configs.mas_config['successful_topk'] = args.successful_topk
    task_configs.mas_config['failed_topk'] = args.failed_topk
    task_configs.mas_config['insights_topk'] = args.insights_topk
    task_configs.mas_config['threshold'] = args.threshold
    task_configs.mas_config['use_projector'] = args.use_projector
    task_configs.mas_config['model_type'] = model_type
    quiet_mode = not args.verbose
    task_configs.mas_config['quiet'] = quiet_mode
    if quiet_mode:
        task_configs.recorder.set_quiet_console(True)
    _mem_task_name = task_configs.task_name
    if args.mtmind2web_shared_memory and task.startswith('mtmind2web_'):
        _mem_task_name = 'mtmind2web'
    task_configs.mem_config.update(
        working_dir=WORKING_DIR,
        hop=args.hop,
        task_name=_mem_task_name,
    )
    if mas_memory_type in {"graph_memory2", "graph_memory3"}:
        task_configs.mem_config.update(
            gm2_memory_dir=str(args.gm2_memory_dir or "").strip(),
            gm2_repo_root=str(args.gm2_repo_root or "").strip(),
            gm2_dynamic_graph=bool(args.gm2_dynamic_graph),
            gm2_retrieval_mode=str(args.gm2_retrieval_mode or "lightweight").strip(),
            gm2_settings=str(args.gm2_settings or "local_only").strip(),
            gm2_owner_scene=str(args.gm2_owner_scene or "").strip(),
            gm2_freeze_memory=bool(args.gm2_freeze_memory),
            gm2_promotion_threshold=float(args.gm2_promotion_threshold),
        )
        if mas_memory_type == "graph_memory3":
            task_configs.mem_config.update(
                gm3_use_textgrad=bool(args.gm3_use_textgrad),
                gm3_textgrad_engine=str(args.gm3_textgrad_engine or "").strip(),
            )
        if args.gm2_enable_overlay:
            task_configs.mem_config["gm2_enable_overlay"] = True
    if task == "alfworld":
        history_dir = args.gm2_history_dir.strip() or os.path.join(LOG_DIR, "game_trajectory")
        task_configs.mem_config.update(
            export_gm2_history=bool(args.export_gm2_history),
            gm2_history_dir=history_dir,
        )
        if args.export_gm2_history:
            os.makedirs(history_dir, exist_ok=True)
            print(f"GM2-compatible histories will be exported to: {os.path.abspath(history_dir)}", flush=True)

    build_mas(task_configs, reasoning_type, mas_memory_type, model_type)

    def _flush_selectivemem_graphs_to_disk() -> None:
        """Ensure SelectiveMem LG/GG JSON snapshots are written after the last task."""
        if task_configs.task_name != "alfworld" or mas_memory_type != "selectivemem":
            return
        mm = getattr(task_configs.mas, "meta_memory", None)
        if mm is None:
            return
        fn = getattr(mm, "persist_entity_graph", None)
        if callable(fn):
            fn()

    # Show basic task configuration (full banner only with --verbose)
    total_tasks = len(task_configs.tasks)
    if quiet_mode:
        print(
            f"\nRunning {total_tasks} tasks (quiet: progress + per-task accuracy only)\n",
            flush=True,
        )
    else:
        print(f"\n{'='*60}")
        print("Start running tasks")
        print(f"Task type: {task}")
        print(f"MAS type: {mas_type}")
        print(f"Memory type: {mas_memory_type}")
        print(f"Model: {model_type}")
        print(f"Total tasks: {total_tasks}")
        print(f"{'='*60}\n", flush=True)
    
    run_task(task_configs)
    _flush_selectivemem_graphs_to_disk()

    if mas_memory_type == "g-memory":
        _mm = getattr(task_configs.mas, "meta_memory", None)
        if isinstance(_mm, GMemory) and hasattr(_mm, "export_insight_query_provenance"):
            prov_path = _mm.export_insight_query_provenance()
            print(f"Saved insight–query provenance: {prov_path}\n", flush=True)

    # postprocess
    completion_tokens, prompt_tokens, _ = get_price()
    price = completion_tokens*15/1000000+prompt_tokens*5/1000000
    print(f"\nAPI usage statistics:")
    print(f"  completion_tokens: {completion_tokens}")
    print(f"  prompt_tokens: {prompt_tokens}")
    print(f"  estimated cost: ${price:.4f}\n", flush=True)
    task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price=${price:.4f}')
