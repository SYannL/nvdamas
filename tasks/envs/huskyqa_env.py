import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .base_env import BaseEnv, BaseRecorder
from .utils import match_exactly


class HuskyQAEnv(BaseEnv):
    def __init__(
        self,
        env_config: dict[str, Any],
        max_trials: int = 7,
    ) -> None:
        self.env_config = env_config
        self.max_trials: int = max_trials
        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get("answer") is None:
            raise ValueError("Please provide the answer for the question.")
        if configs.get("task") is None:
            raise ValueError("The configs dict should have the `task` attribute.")
        self.config = configs

        task_value = self.config.get("task")
        task: str = f"Question: {task_value}"
        return task, task

    def reset(self) -> None:
        self.current_task: str = None
        self.reward: float = 0

    def step(self, action: str) -> tuple[str, float, bool]:
        action = self.process_action(action)

        if self._parse_action_type(action) == "thought":
            return "OK.", 0, False

        action_type, argument = self._parse_action(action)
        if action_type == "Finish":
            if self.success_fn(argument):
                observation = "Answer is CORRECT"
                self.reward = 1
                return observation, 1, True
            observation = "Answer is INCORRECT"
            return observation, 0, True

        observation = (
            "Invalid Action. Valid Action is Finish[<answer>]."
        )
        return observation, -1, False

    @staticmethod
    def _parse_action_type(action: str) -> Literal["action", "thought"]:
        return "thought" if "thought" in action.lower() else "action"

    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip().replace("<", "").split("\n")[0]
        action = action.replace(">", "").replace("OK.", "").replace("OK", "").strip()

        if HuskyQAEnv._parse_action_type(action) == "thought":
            return action
        if ":" in action:
            action = action.split(":", 1)[1].strip()
        return action

    @staticmethod
    def _parse_action(string: str) -> tuple[Optional[str], Optional[str]]:
        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, string)
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        return None, None

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        cleaned = text.replace(",", "")
        numbers = re.findall(r"[-+]?\d*\.?\d+", cleaned)
        if len(numbers) == 1:
            return float(numbers[0])
        return None

    def success_fn(self, agent_ans: str) -> bool:
        if agent_ans is None:
            return False
        key = self.config.get("answer")
        if key is None:
            return False

        num_ans = self._extract_number(str(agent_ans))
        num_key = self._extract_number(str(key))
        if num_ans is not None and num_key is not None:
            tolerance = max(1e-6, 1e-4 * abs(num_key))
            return abs(num_ans - num_key) <= tolerance

        return match_exactly(str(agent_ans), str(key))

    def feedback(self) -> tuple[float, bool, str]:
        feedback: str = (
            "You successfully finished this task."
            if self.reward == 1
            else "You failed the task."
        )
        done = self.reward == 1
        return self.reward, done, feedback


@dataclass
class HuskyQARecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = "huskyqa"
        self.counts = 0
        self.dones = 0
        self.rewards = 0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        message: str = f"---------- Task: {task_id} ----------"
        self.log(message)

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.dones += done
        self.counts += 1

        accuracy = self.rewards / self.counts
        done_rate = self.dones / self.counts
        message = (
            f"reward: {reward}, ave reward: {accuracy}.\n"
            f"done: {done}, ave done: {done_rate}"
        )
        self.log(message)
