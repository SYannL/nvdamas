import os
import re
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Union, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Keep ALFWorld runtime assets inside the repo so imports do not depend on ~/.cache.
os.environ.setdefault("ALFWORLD_DATA", str(REPO_ROOT / "data" / "alfworld" / "_runtime_cache"))

import alfworld


def _load_alfred_tw_env() -> type:
    alfworld_root = Path(alfworld.__file__).resolve().parent
    module_path = alfworld_root / "agents" / "environment" / "alfred_tw_env.py"
    spec = spec_from_file_location("nvdamas_alfred_tw_env", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AlfredTWEnv from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AlfredTWEnv


AlfredTWEnv = _load_alfred_tw_env()

from .base_env import BaseEnv, BaseRecorder

prefixes = {  # tasks: task_type
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def get_env_name_from_gamefile(gamefile: str) -> Union[str, None]:

    for k in prefixes.keys():
        if k in gamefile:
            return k
    return None


class AlfworldEnv(BaseEnv):
    def __init__(
        self, 
        env_config: dict[str, Any], 
        max_trials: int = 50
    ): 
        self.env_config = env_config
        env_type = self.env_config['env']['type']
        if env_type != 'AlfredTWEnv':
            raise ValueError(
                f'Unsupported ALFWorld env type: {env_type}. '
                'This project currently supports text-only AlfredTWEnv.'
            )
        self.main_env = AlfredTWEnv(self.env_config, train_eval=self.env_config['split'])
        
        self.max_trials: int = max_trials
        self.reset()
    
    def set_env(self, configs: dict) -> tuple[str, str]:  
        self.gamefile = configs['env_kwargs']['gamefile']
        self.env_name: str = configs['env_name']
        self.main_env.game_files = [self.gamefile]

        self.reset()
        task = configs.get('task')
        if not task or 'Your task is to:' not in task:
            goal_instruction = configs.get('goal_instruction')
            if not goal_instruction:
                raise ValueError('Missing `task` or `goal_instruction` for ALFWorld task config.')
            task = f'{self.initial_observation}\n\n{goal_instruction}'.strip()
        return self._parse_task_main(task), self._parse_task_description(task)

    def reset(self):

        self.done = False
        self.won = False
        self.env = self.main_env.init_env(batch_size=1)
        observation, _ = self.env.reset()
        self.initial_observation = self._normalize_observation(observation[0]) if observation else ''

    def step(self, action: str) -> tuple[str, float, bool]:

        action = self.process_action(action)
        observation, reward, done, info = self.env.step([action])
        observation = self._normalize_observation(observation[0])

        self.done = done[0]
        # TextWorld/ALFWorld exposes a win signal via `info['won']`.
        # `done` can be True when the max-step budget is exhausted, which is not success.
        try:
            self.won = bool(info['won'][0])
        except Exception:
            self.won = False

        if 'think:' in action:
            observation = 'OK.' 
            processed_reward = -1
            self.won = False
        elif observation == 'Nothing happens.':
            processed_reward = -1
            self.won = False
        else:
            processed_reward = 1 if self.won else 0
        
        return observation, processed_reward, self.done
    
    def feedback(self) -> tuple[float, bool, str]:
        # Only count true wins as success.
        success = getattr(self, "won", False)
        reward = 1.0 if success else 0.0
        message = "You successfully finished this task!" if success else "You failed the task."
        
        return reward, success, message
    
    @staticmethod
    def process_action(action: str) -> str:
        action = AlfworldEnv._extract_action(action)
        # The solver may output numbered/bulleted steps like "1. take X from Y"
        # or "- think: ...". Strip those prefixes so TextWorld can parse the command.
        action = re.sub(r'^\s*\d+\.\s*', '', action)
        action = re.sub(r'^\s*[-*]\s*', '', action)
        action = action.replace('>', '').replace('OK.', '').replace('OK', '').strip()
        action = action.rstrip(".。").strip()

        return action

    @staticmethod
    def _extract_action(text: str) -> str:
        """Extract the first valid ALFWorld command from verbose/Qwen-style output."""
        text = str(text or "").strip()
        thought = AlfworldEnv._extract_thought(text)
        # Qwen3 may emit native reasoning tags before the actual answer.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

        commands = (
            "take ",
            "go to ",
            "open ",
            "put ",
            "clean ",
            "heat ",
            "cool ",
            "use ",
            "think:",
        )
        candidates: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip().lstrip(">").strip()
            line = re.sub(r"^\s*\d+\.\s*", "", line)
            line = re.sub(r"^\s*[-*]\s*", "", line)
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith(commands):
                candidates.append(line)

        # Prefer an executable environment action over an internal thought.
        for candidate in candidates:
            if not candidate.lower().startswith("think:"):
                return candidate
        if candidates:
            return candidates[0]

        # MCMA/ALFWorld trajectories support a legal ReAct-style thought action.
        # If Qwen only produced native thinking text and no executable command,
        # normalize it to `think:` instead of introducing a `look` action that
        # some validators/postprocessors do not expect.
        if thought:
            return f"think: {thought}"
        return "think: I need to choose one valid next action."

    @staticmethod
    def _extract_thought(text: str, limit: int = 220) -> str:
        text = str(text or "").strip()
        match = re.search(r"<think>(.*?)(?:</think>|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            thought = match.group(1)
        else:
            thought = text
        thought = re.sub(r"</?think>", "", thought, flags=re.IGNORECASE)
        thought = re.sub(r"\s+", " ", thought).strip(" >\t\r\n")
        if not thought or thought.lower() in {"think", "/think"}:
            return ""
        if len(thought) > limit:
            thought = thought[:limit].rsplit(" ", 1)[0].rstrip(".,;:")
        return thought

    @staticmethod
    def _normalize_observation(observation: str) -> str:
        if observation.startswith('You arrive at loc '):
            observation = observation[observation.find('. ') + 2:]
        return observation
    
    def _parse_task_main(self, task: str):
        return self.env_name + '-' + re.search(r'Your task is to:\s*(.+)', task, re.DOTALL).group(1).strip()
    @staticmethod
    def _parse_task_description(task: str) -> str:
        return task.split('___')[0]
            

@dataclass
class AlfworldRecorder(BaseRecorder):   
    
    def __post_init__(self):
        
        super().__post_init__()
        self.task = 'alfworld'
        self.counts = [0] * 6
        self.results = [0] * 6

    def task_begin(self, task_id, task_config):
        super().task_begin(task_id, task_config)
        
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)
    
    def task_end(self, reward: float, done: bool):
        gamefile: str = self.current_task_config['env_kwargs']['gamefile']
        env_name = get_env_name_from_gamefile(gamefile)
        if env_name is None:
            raise ValueError('Format of the task config is wrong.')

        for i, (k, v) in enumerate(prefixes.items()):
            if env_name == k:
                self.results[i] += done
                self.counts[i] += 1
                break
        
        message = f'success: {done}, ave success: {sum(self.results) / sum(self.counts)}'
        self.log(message)
        self.log("rs: " + str(self.results))
        self.log("cnts: " + str(self.counts))
    
    
