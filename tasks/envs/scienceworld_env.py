import re
from dataclasses import dataclass
from typing import Any

from .base_env import BaseEnv, BaseRecorder

try:
    from scienceworld import ScienceWorldEnv as _ScienceWorldRuntimeEnv
except Exception as exc:  # pragma: no cover - handled by env registry
    _SCIENCEWORLD_IMPORT_ERROR = exc
    _ScienceWorldRuntimeEnv = None
else:
    _SCIENCEWORLD_IMPORT_ERROR = None


class ScienceWorldEnv(BaseEnv):
    def __init__(self, env_config: dict[str, Any], max_trials: int = 50):
        if _ScienceWorldRuntimeEnv is None:
            raise ImportError(
                "ScienceWorld is not available. Install local package first: "
                'pip install -e "data/ScienceWorld". '
                f"Underlying import error: {_SCIENCEWORLD_IMPORT_ERROR!r}"
            )
        self.env_config = env_config
        self.max_trials = int(max_trials)
        self.env = None
        self.task_name = ""
        self.variation_idx = 0
        self.simplification = ""
        self.last_observation = ""
        self.last_reward = 0.0
        self.last_score = 0.0
        self.last_completed = False
        self.has_positive_progress = False

    def set_env(self, configs: dict) -> tuple[str, str]:
        task_name = str(
            configs.get("sw_task")
            or configs.get("task_name")
            or configs.get("task_id")
            or self.env_config.get("default_task")
            or "1-1"
        )
        variation_idx = int(configs.get("variation_idx", self.env_config.get("default_variation_idx", 0)))
        simplification = str(configs.get("simplification_str", self.env_config.get("simplification_str", "")))

        if self.env is not None:
            self.env.close()
            self.env = None

        self.env = _ScienceWorldRuntimeEnv(envStepLimit=self.max_trials + 1)
        self.env.load(
            taskName=task_name,
            variationIdx=variation_idx,
            simplificationStr=simplification,
            generateGoldPath=False,
        )
        observation, info = self.env.reset()

        self.task_name = getattr(self.env, "taskName", task_name)
        self.variation_idx = variation_idx
        self.simplification = simplification
        self.last_observation = observation or ""
        self.last_score = float(info.get("score", 0))
        self.last_reward = 0.0
        self.last_completed = False
        self.has_positive_progress = self.last_score > 0

        task_main = f"scienceworld-{self.task_name}-v{self.variation_idx}"
        task_description = (
            f"{self.env.get_task_description()}\n\n"
            f"Initial observation:\n{self.last_observation}"
        )
        return task_main, task_description

    @classmethod
    def process_action(cls, action: str) -> str:
        text = (action or "").strip()
        if not text:
            return ""
        if "```" in text:
            parts = [p.strip() for p in text.split("```") if p.strip()]
            if parts:
                text = parts[0]
        first_line = text.splitlines()[0].strip()
        first_line = re.sub(r"^Action\s+\d+:\s*", "", first_line, flags=re.IGNORECASE)
        first_line = first_line.strip("`").strip().strip('"').strip("'")
        first_line = re.sub(r"[\[\]\(\)\{\}<>]", "", first_line)
        first_line = re.sub(r"\s+", " ", first_line).strip()
        wait_match = re.match(r"^wait\s+(\d+)\s+seconds?$", first_line, flags=re.IGNORECASE)
        if wait_match:
            return f"wait{wait_match.group(1)}"
        return first_line

    def step(self, action: str) -> tuple[str, float, bool]:
        if self.env is None:
            raise RuntimeError("ScienceWorld environment is not initialized. Call set_env first.")

        env_action = self.process_action(action)
        observation, reward, completed, step_info = self.env.step(env_action)
        self.last_observation = observation or ""
        self.last_reward = float(reward)
        self.last_score = float(step_info.get("score", self.last_score))
        self.last_completed = bool(completed)
        if self.last_reward > 0 or self.last_score > 0:
            self.has_positive_progress = True
        return self.last_observation, self.last_reward, self.last_completed

    def feedback(self) -> tuple[float, bool, str]:
        # ScienceWorld true completion may happen with negative/zero score in some failures.
        success = bool(self.last_completed and self.last_score > 0)
        reward = 1.0 if success else 0.0
        feedback = (
            f"ScienceWorld episode finished. completed={self.last_completed}, score={self.last_score}."
            if self.last_completed
            else f"ScienceWorld episode not finished. score={self.last_score}."
        )
        return reward, success, feedback

    @property
    def memory_success_label(self) -> bool:
        """ScienceWorld-specific memory success: any positive progress counts."""
        return bool(self.has_positive_progress)


@dataclass
class ScienceWorldRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = "scienceworld"
        self.counts = 0
        self.successes = 0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        self.log(f"---------- Task: {task_id} ----------")

    def task_end(self, reward: float, done: bool):
        self.counts += 1
        self.successes += int(bool(done))
        rate = self.successes / self.counts if self.counts else 0.0
        self.log(f"reward: {reward}, success: {done}, running_success_rate: {rate:.3f}")
