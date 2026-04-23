"""
Anatomy + Surgery 双任务共享全局记忆：解剖学与外科知识会交叉（手术依赖解剖结构），
两路运行的记忆可相互检索与复用。
"""
import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import random
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import yaml

from mas.llm import GPTChat
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase, GMemory, GMemoryGraphMASMemory
from mas.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type


with open('tasks/configs.yaml') as reader:
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


def build_task_manager(task: str, mas_type: str, memory_type: str, max_steps: int, working_dir: str, log_dir: str) -> TaskManager:
    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)

    env: BaseEnv = get_env(task, config, max_steps)
    recorder: BaseRecorder = get_recorder(task, working_dir=log_dir, namespace='total_task')
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
        mas_config=mas_config
    )
    task_manager.mem_config.update(working_dir=working_dir, task_name=task)
    return task_manager


def build_mas(task_manager: TaskManager, reasoning: str, mas_memory: str, llm_type: str) -> MASMemoryBase:
    embed_func = EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"))
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    llm_model = GPTChat(model_name=llm_type)
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func
    )

    task_manager.mas.add_observer(task_manager.recorder)
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)
    return mas_memory_module


def run_batch(task_manager: TaskManager, start: int, end: int) -> list:
    task_manager.recorder.dataset_begin()
    total_tasks = len(task_manager.tasks)
    saved_messages = []
    start_time = time.time()

    for task_id in range(start, end):
        task_config = task_manager.tasks[task_id]
        progress_msg = f"[{task_manager.task_name}] 进度: {task_id + 1}/{total_tasks}"
        print(f"\n{progress_msg} 开始任务 {task_id + 1}...", flush=True)
        task_manager.recorder.log(progress_msg)

        task_manager.recorder.task_begin(task_id, task_config)
        task_main, task_description = task_manager.mas.env.set_env(task_config)
        few_shots: list[str] = get_task_few_shots(
            dataset=task_manager.task_name,
            task_config=task_config,
            few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
        )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)

        task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
        for agent in task_manager.mas.agents_team.values():
            task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))

        reward, done = task_manager.mas.schedule(task_config)
        task_manager.recorder.task_end(reward, done)

        last_saved = getattr(task_manager.mas.meta_memory, "last_saved_message", None)
        if last_saved is not None:
            saved_messages.append(last_saved)

        elapsed = time.time() - start_time
        print(f"{progress_msg} 任务 {task_id + 1} 完成 (已用时: {elapsed:.1f}秒)", flush=True)

    task_manager.recorder.dataset_end()
    return saved_messages


def main() -> None:
    random.seed(42)

    parser = argparse.ArgumentParser(description='Run dual MedMCQA Anatomy+Surgery with shared global memory.')
    parser.add_argument('--mas_type', type=str, choices=['autogen', 'macnet', 'dylan', 'strategy'], default='autogen')
    parser.add_argument('--mas_memory', type=str, default='g-memory')
    parser.add_argument('--reasoning', type=str, default='io')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125')
    parser.add_argument('--max_trials', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    model_type: str = get_model_type(args.model)
    base_dir = os.path.join('./.db', model_type, 'medmcqa_dual_anatomy_surgery', args.mas_type, 'memory', args.mas_memory)
    global_dir = os.path.join(base_dir, 'global')
    anatomy_dir = os.path.join(base_dir, 'anatomy')
    surgery_dir = os.path.join(base_dir, 'surgery')
    log_dir = os.path.join('./logs', 'medmcqa_dual_anatomy_surgery', args.mas_type, 'memory', args.mas_memory, model_type)
    os.makedirs(global_dir, exist_ok=True)
    os.makedirs(anatomy_dir, exist_ok=True)
    os.makedirs(surgery_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    anatomy_manager = build_task_manager('medmcqa_anatomy_20', args.mas_type, args.mas_memory, args.max_trials, anatomy_dir, log_dir)
    surgery_manager = build_task_manager('medmcqa_surgery_20', args.mas_type, args.mas_memory, args.max_trials, surgery_dir, log_dir)

    anatomy_manager.mem_config.update(hop=1)
    surgery_manager.mem_config.update(hop=1)

    anatomy_mem = build_mas(anatomy_manager, args.reasoning, args.mas_memory, args.model)
    surgery_mem = build_mas(surgery_manager, args.reasoning, args.mas_memory, args.model)

    # Shared global memory retriever
    global_mem = None
    if args.mas_memory == "g-memory":
        global_mem = GMemory(
            namespace='g-memory',
            global_config={
                'working_dir': global_dir,
                'hop': 1,
                'start_insights_threshold': 5,
                'rounds_per_insights': 5,
                'insights_point_num': 5,
            },
            llm_model=GPTChat(model_name=args.model),
            embedding_func=EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"))
        )
    elif args.mas_memory == "gmemgraph":
        global_mem = GMemoryGraphMASMemory(
            namespace='gmemgraph',
            global_config={
                'working_dir': global_dir,
                'hop': 1,
                'start_insights_threshold': 5,
                'rounds_per_insights': 5,
                'insights_point_num': 5,
                'entity_graph_persist_every': 10,
            },
            llm_model=GPTChat(model_name=args.model),
            embedding_func=EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"))
        )

    if global_mem is not None:
        set_global = getattr(anatomy_mem, "set_global_retriever", None)
        if callable(set_global):
            anatomy_mem.set_global_retriever(global_mem)
        set_global = getattr(surgery_mem, "set_global_retriever", None)
        if callable(set_global):
            surgery_mem.set_global_retriever(global_mem)

    total = min(len(anatomy_manager.tasks), len(surgery_manager.tasks))
    batch_size = max(1, args.batch_size)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_anatomy = executor.submit(run_batch, anatomy_manager, start, end)
            fut_surgery = executor.submit(run_batch, surgery_manager, start, end)
            anatomy_messages = fut_anatomy.result()
            surgery_messages = fut_surgery.result()

        if global_mem is not None:
            # Merge new nodes into global memory
            for msg in anatomy_messages:
                if msg is None:
                    continue
                global_mem.add_memory_with_source(msg, source_id="Anatomy", raw=True)
            for msg in surgery_messages:
                if msg is None:
                    continue
                global_mem.add_memory_with_source(msg, source_id="Surgery", raw=True)

            print(f"\n[SYNC] 已合并批次 {start + 1}-{end} 的节点到 global graph。\n", flush=True)


if __name__ == '__main__':
    main()
