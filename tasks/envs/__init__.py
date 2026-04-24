import importlib
import json
import os

from .base_env import BaseEnv, BaseRecorder
# from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes
from .fever_env import FeverEnv, FeverRecorder
from .huskyqa_env import HuskyQAEnv, HuskyQARecorder
from .medmcqa_env import MedMCQAEnv, MedMCQARecorder
from .mtmind2web_env import MTMind2WebEnv, MTMind2WebRecorder
from .scienceworld_env import ScienceWorldEnv, ScienceWorldRecorder
try:
    from .pddl_env.pddl_env import PDDLEnv, PDDLRecorder, get_all_environment_configs
except ImportError:
    PDDLEnv = None
    PDDLRecorder = None
    get_all_environment_configs = None

# Lazy import for alfworld to avoid dependency issues.
ALFWORLD_IMPORT_ERROR = None
try:
    _alfworld_env_module = importlib.import_module("tasks.envs.alfworld_env")
    AlfworldEnv = _alfworld_env_module.AlfworldEnv
    AlfworldRecorder = _alfworld_env_module.AlfworldRecorder
    get_env_name_from_gamefile = _alfworld_env_module.get_env_name_from_gamefile
    prefixes = _alfworld_env_module.prefixes
except Exception as exc:
    ALFWORLD_IMPORT_ERROR = exc
    AlfworldEnv = None
    AlfworldRecorder = None
    get_env_name_from_gamefile = None
    prefixes = {}

TASKS_PATH = {
    'alfworld': 'data/alfworld/alfworld_tasks_suffix.json',
    'fever': 'data/fever/fever_dev.jsonl',
    'fever_aca_test': 'data/fever/fever_dev_test_aca.jsonl',
    'fever_oral_test': 'data/fever/fever_dev_test_oral.jsonl',
    'pddl': 'data/pddl/test.jsonl',
    'huskyqa': 'data/HuskyQA/huskyQA.json',
    'huskyqa_aca_test': 'data/HuskyQA/huskyQA_aca_test.json',
    'medmcqa_test': 'data/medmcqa/medmcqa_test.jsonl',
    'medmcqa_physio_30': 'data/medmcqa/medmcqa_physio_30.jsonl',
    'medmcqa_pharma_30': 'data/medmcqa/medmcqa_pharma_30.jsonl',
    'medmcqa_physio_3': 'data/medmcqa/medmcqa_physio_3.jsonl',
    'medmcqa_pharma_3': 'data/medmcqa/medmcqa_pharma_3.jsonl',
    'medmcqa_physio_150_build': 'data/medmcqa/medmcqa_physio_150_build.jsonl',
    'medmcqa_pharma_150_build': 'data/medmcqa/medmcqa_pharma_150_build.jsonl',
    'medmcqa_physio_20_test': 'data/medmcqa/medmcqa_physio_20_test.jsonl',
    'medmcqa_pharma_20_test': 'data/medmcqa/medmcqa_pharma_20_test.jsonl',
    'medmcqa_anatomy_20': 'data/medmcqa/medmcqa_anatomy_20.jsonl',
    'medmcqa_surgery_20': 'data/medmcqa/medmcqa_surgery_20.jsonl',
    # 'mtmind2web_train': 'data/MT-Mind2Web/mtmind2web_train_eval.jsonl',
    # 'mtmind2web_test_task': 'data/MT-Mind2Web/mtmind2web_test_task_eval.jsonl',
    # 'mtmind2web_test_website': 'data/MT-Mind2Web/mtmind2web_test_website_eval.jsonl',
    # 'mtmind2web_test_subdomain': 'data/MT-Mind2Web/mtmind2web_test_subdomain_eval.jsonl',
    'scienceworld_train': 'data/scienceworld/scienceworld_train.jsonl',
    'scienceworld_dev': 'data/scienceworld/scienceworld_dev.jsonl',
    'scienceworld_test': 'data/scienceworld/scienceworld_test.jsonl',
    'scienceworld_domain_a_train': 'data/scienceworld/scienceworld_domain_a_train.jsonl',
    'scienceworld_domain_b_train': 'data/scienceworld/scienceworld_domain_b_train.jsonl',
    'scienceworld_domain_a_test': 'data/scienceworld/scienceworld_domain_a_test.jsonl',
    'scienceworld_domain_b_test': 'data/scienceworld/scienceworld_domain_b_test.jsonl',
}


def _load_jsonl_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_jsonl_rows_if_exists(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return _load_jsonl_rows(path)

## Tasks
if AlfworldEnv is not None:
    alfworld_tasks: list[dict] = [
        {
            'task': f'{row["goal"]}',
            'env_kwargs': {
                'config': 'alfworld',
                "gamefile": row["gamefile"],
            },
            'task_type': prefixes[get_env_name_from_gamefile(row["gamefile"])],
            'env_name': get_env_name_from_gamefile(row["gamefile"])
        } for row in json.load(open(TASKS_PATH['alfworld'], "r")) 
    ]
else:
    alfworld_tasks: list[dict] = []

with open(TASKS_PATH['fever'], 'r') as f:
    fever_tasks = [
        {
            'task': row['claim'],
            'answer': row['label'],
            'env_name': 'fever',
        }
        for row in (json.loads(line) for line in f) 
    ][:100]

with open(TASKS_PATH['fever_aca_test'], 'r') as f:
    fever_aca_test_tasks = [
        {
            'task': row['claim'],
            'answer': row['label'],
            'env_name': 'fever',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['fever_oral_test'], 'r') as f:
    fever_oral_test_tasks = [
        {
            'task': row['claim'],
            'answer': row['label'],
            'env_name': 'fever',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['huskyqa'], 'r') as f:
    huskyqa_tasks = [
        {
            'task': row['question'],
            'answer': row['answer'],
            'env_name': 'huskyqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['huskyqa_aca_test'], 'r') as f:
    huskyqa_aca_test_tasks = [
        {
            'task': row['question'],
            'answer': row['answer'],
            'env_name': 'huskyqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_test'], 'r') as f:
    medmcqa_test_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_physio_30'], 'r') as f:
    medmcqa_physio_30_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_pharma_30'], 'r') as f:
    medmcqa_pharma_30_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_physio_3'], 'r') as f:
    medmcqa_physio_3_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f if line.strip())
    ]

with open(TASKS_PATH['medmcqa_pharma_3'], 'r') as f:
    medmcqa_pharma_3_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f if line.strip())
    ]

with open(TASKS_PATH['medmcqa_physio_150_build'], 'r') as f:
    medmcqa_physio_150_build_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_pharma_150_build'], 'r') as f:
    medmcqa_pharma_150_build_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_physio_20_test'], 'r') as f:
    medmcqa_physio_20_test_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_pharma_20_test'], 'r') as f:
    medmcqa_pharma_20_test_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_anatomy_20'], 'r') as f:
    medmcqa_anatomy_20_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

with open(TASKS_PATH['medmcqa_surgery_20'], 'r') as f:
    medmcqa_surgery_20_tasks = [
        {
            'task': row['question'],
            'options': {
                'A': row['opa'],
                'B': row['opb'],
                'C': row['opc'],
                'D': row['opd'],
            },
            'answer_key': "ABCD"[int(row['cop']) - 1],
            'answer': row[f"op{'abcd'[int(row['cop']) - 1]}"],
            'subject_name': row.get('subject_name'),
            'env_name': 'medmcqa',
        }
        for row in (json.loads(line) for line in f)
    ]

# mtmind2web_train_tasks = _load_jsonl_rows(TASKS_PATH['mtmind2web_train'])
# mtmind2web_test_task_tasks = _load_jsonl_rows(TASKS_PATH['mtmind2web_test_task'])
# mtmind2web_test_website_tasks = _load_jsonl_rows(TASKS_PATH['mtmind2web_test_website'])
# mtmind2web_test_subdomain_tasks = _load_jsonl_rows(TASKS_PATH['mtmind2web_test_subdomain'])
# scienceworld_train_tasks = _load_jsonl_rows(TASKS_PATH['scienceworld_train'])
# scienceworld_dev_tasks = _load_jsonl_rows(TASKS_PATH['scienceworld_dev'])
# scienceworld_test_tasks = _load_jsonl_rows(TASKS_PATH['scienceworld_test'])

TASK_NAMES = ["barman", "blockworld", "gripper", "tyreworld"]
if get_all_environment_configs is not None:
    pddl_tasks: list[dict] = get_all_environment_configs(TASK_NAMES, TASKS_PATH['pddl'])
else:
    pddl_tasks: list[dict] = []


TASK_DATA = {
    'alfworld': alfworld_tasks,
    'fever': fever_tasks,
    'fever_aca_test': fever_aca_test_tasks,
    'fever_oral_test': fever_oral_test_tasks,
    'pddl': pddl_tasks,
    'huskyqa': huskyqa_tasks,
    'huskyqa_aca_test': huskyqa_aca_test_tasks,
    'medmcqa_test': medmcqa_test_tasks,
    'medmcqa_physio_30': medmcqa_physio_30_tasks,
    'medmcqa_pharma_30': medmcqa_pharma_30_tasks,
    'medmcqa_physio_3': medmcqa_physio_3_tasks,
    'medmcqa_pharma_3': medmcqa_pharma_3_tasks,
    'medmcqa_physio_150_build': medmcqa_physio_150_build_tasks,
    'medmcqa_pharma_150_build': medmcqa_pharma_150_build_tasks,
    'medmcqa_physio_20_test': medmcqa_physio_20_test_tasks,
    'medmcqa_pharma_20_test': medmcqa_pharma_20_test_tasks,
    'medmcqa_anatomy_20': medmcqa_anatomy_20_tasks,
    'medmcqa_surgery_20': medmcqa_surgery_20_tasks,
    # 'mtmind2web_train': mtmind2web_train_tasks,
    # 'mtmind2web_test_task': mtmind2web_test_task_tasks,
    # 'mtmind2web_test_website': mtmind2web_test_website_tasks,
    # 'mtmind2web_test_subdomain': mtmind2web_test_subdomain_tasks,
    # 'scienceworld_train': scienceworld_train_tasks,
    # 'scienceworld_dev': scienceworld_dev_tasks,
    # 'scienceworld_test': scienceworld_test_tasks,
}

ENVS = {
    'fever': FeverEnv,
    'fever_aca_test': FeverEnv,
    'fever_oral_test': FeverEnv,
    'pddl': PDDLEnv,
    'huskyqa': HuskyQAEnv,
    'huskyqa_aca_test': HuskyQAEnv,
    'medmcqa_test': MedMCQAEnv,
    'medmcqa_physio_30': MedMCQAEnv,
    'medmcqa_pharma_30': MedMCQAEnv,
    'medmcqa_physio_3': MedMCQAEnv,
    'medmcqa_pharma_3': MedMCQAEnv,
    'medmcqa_physio_150_build': MedMCQAEnv,
    'medmcqa_pharma_150_build': MedMCQAEnv,
    'medmcqa_physio_20_test': MedMCQAEnv,
    'medmcqa_pharma_20_test': MedMCQAEnv,
    'medmcqa_anatomy_20': MedMCQAEnv,
    'medmcqa_surgery_20': MedMCQAEnv,
    # 'mtmind2web_train': MTMind2WebEnv,
    # 'mtmind2web_test_task': MTMind2WebEnv,
    # 'mtmind2web_test_website': MTMind2WebEnv,
    # 'mtmind2web_test_subdomain': MTMind2WebEnv,
    # 'scienceworld_train': ScienceWorldEnv,
    # 'scienceworld_dev': ScienceWorldEnv,
    # 'scienceworld_test': ScienceWorldEnv,
}
if AlfworldEnv is not None:
    ENVS['alfworld'] = AlfworldEnv

RECORDERS = {
    'fever': FeverRecorder,
    'fever_aca_test': FeverRecorder,
    'fever_oral_test': FeverRecorder,
    'pddl': PDDLRecorder,
    'huskyqa': HuskyQARecorder,
    'huskyqa_aca_test': HuskyQARecorder,
    'medmcqa_test': MedMCQARecorder,
    'medmcqa_physio_30': MedMCQARecorder,
    'medmcqa_pharma_30': MedMCQARecorder,
    'medmcqa_physio_3': MedMCQARecorder,
    'medmcqa_pharma_3': MedMCQARecorder,
    'medmcqa_physio_150_build': MedMCQARecorder,
    'medmcqa_pharma_150_build': MedMCQARecorder,
    'medmcqa_physio_20_test': MedMCQARecorder,
    'medmcqa_pharma_20_test': MedMCQARecorder,
    'medmcqa_anatomy_20': MedMCQARecorder,
    'medmcqa_surgery_20': MedMCQARecorder,
    # 'mtmind2web_train': MTMind2WebRecorder,
    # 'mtmind2web_test_task': MTMind2WebRecorder,
    # 'mtmind2web_test_website': MTMind2WebRecorder,
    # 'mtmind2web_test_subdomain': MTMind2WebRecorder,
    # 'scienceworld_train': ScienceWorldRecorder,
    # 'scienceworld_dev': ScienceWorldRecorder,
    # 'scienceworld_test': ScienceWorldRecorder,
}
if AlfworldRecorder is not None:
    RECORDERS['alfworld'] = AlfworldRecorder


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    
    if ENVS.get(task) is None:
        if task == 'alfworld':
            raise ImportError(
                f'ALFWorld task is not available. This is likely because the alfworld package or its dependencies are not installed.\n'
                f'To fix this, you need to install alfworld and its dependencies. However, alfworld requires fast-downward-textworld\n'
                f'which needs C++ development tools. Please install Xcode Command Line Tools first:\n'
                f'  xcode-select --install\n'
                f'Then install alfworld:\n'
                f'  pip install alfworld\n'
                f'Or use uv:\n'
                f'  uv pip install alfworld\n'
                f'Underlying import error: {ALFWORLD_IMPORT_ERROR!r}'
            )
        if task == 'pddl':
            raise ImportError(
                'PDDL task is not available. This is likely because gym/pddlgym dependencies are missing.\n'
                'Please install gym/pddlgym or use a task that does not require it.'
            )
        raise ValueError(f'Unsupported task type: {task}')
    
    return ENVS.get(task)(env_config, max_trials)

def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    
    if RECORDERS.get(task) is None:
        if task == 'alfworld':
            raise ImportError(
                f'ALFWorld task is not available. This is likely because the alfworld package or its dependencies are not installed.\n'
                f'To fix this, you need to install alfworld and its dependencies. However, alfworld requires fast-downward-textworld\n'
                f'which needs C++ development tools. Please install Xcode Command Line Tools first:\n'
                f'  xcode-select --install\n'
                f'Then install alfworld:\n'
                f'  pip install alfworld\n'
                f'Or use uv:\n'
                f'  uv pip install alfworld\n'
                f'Underlying import error: {ALFWORLD_IMPORT_ERROR!r}'
            )
        if task == 'pddl':
            raise ImportError(
                'PDDL task recorder is not available. This is likely because gym/pddlgym dependencies are missing.\n'
                'Please install gym/pddlgym or use a task that does not require it.'
            )
        raise ValueError(f'Unsupported task type: {task}')
    
    return RECORDERS.get(task)(working_dir=working_dir, namespace=namespace)

def get_task(task: str, env_config: dict | None = None) -> list[dict]:

    if TASK_DATA.get(task) is None:
        if task == 'alfworld':
            raise ImportError(
                f'ALFWorld task is not available. This is likely because the alfworld package or its dependencies are not installed.\n'
                f'To fix this, you need to install alfworld and its dependencies. However, alfworld requires fast-downward-textworld\n'
                f'which needs C++ development tools. Please install Xcode Command Line Tools first:\n'
                f'  xcode-select --install\n'
                f'Then install alfworld:\n'
                f'  pip install alfworld\n'
                f'Or use uv:\n'
                f'  uv pip install alfworld\n'
                f'Underlying import error: {ALFWORLD_IMPORT_ERROR!r}'
            )
        if task == 'pddl':
            raise ImportError(
                'PDDL task list is not available. This is likely because gym/pddlgym dependencies are missing.\n'
                'Please install gym/pddlgym or use a task that does not require it.'
            )
        raise ValueError(f'Unsupported task type: {task}')
    
    task_list = TASK_DATA.get(task)
    if task == 'alfworld' and len(task_list) == 0:
        raise ImportError(
            f'ALFWorld tasks list is empty. This is likely because the alfworld package failed to import.\n'
            f'Please check the error message above and install the required dependencies.'
        )
    if task == 'pddl' and len(task_list) == 0 and get_all_environment_configs is None:
        raise ImportError(
            'PDDL task list is empty. This is likely because gym/pddlgym dependencies are missing.\n'
            'Please install gym/pddlgym or use a task that does not require it.'
        )
    
    if task == 'medmcqa_test':
        subject_name = None
        if env_config is not None:
            subject_name = env_config.get('subject_name')
        if subject_name and str(subject_name).lower() != "all":
            task_list = [
                item for item in task_list
                if item.get('subject_name') == subject_name
            ]

    return task_list