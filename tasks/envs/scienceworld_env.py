import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
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
        self.best_score = 0.0
        self.last_completed = False
        self.has_positive_progress = False
        self.sw_task: str = ""
        self.sw_task_name: str = ""
        self.sw_scene_room: str = ""
        self.sw_task_desc: str = ""
        self._task_config: dict = {}
        self._step_count: int = 0
        self.current_history: list = []
        self.last_admissible_commands: list = []

    def _valid_action_limit(self) -> int:
        try:
            return max(0, int(self.env_config.get("valid_action_prompt_limit", 120)))
        except Exception:
            return 120

    @staticmethod
    def _action_overlap_score(action: str, *contexts: str) -> tuple[int, int, int, str]:
        action_text = str(action or "").lower()
        action_words = {tok for tok in re.findall(r"[a-z0-9]+", action_text) if len(tok) > 2}
        context_words: set[str] = set()
        for ctx in contexts:
            context_words.update(tok for tok in re.findall(r"[a-z0-9]+", str(ctx or "").lower()) if len(tok) > 2)
        navigation_prefixes = ("open ", "go to ")
        good_prefixes = (
            "look ", "read ", "take ", "put ", "place ", "focus on ", "activate ",
            "deactivate ", "connect wire", "disconnect wire", "measure ", "examine ",
            "use ", "wait",
        )
        noisy_prefixes = (
            "pour ", "move ", "connect ", "disconnect ", "mix ",
        )
        action_priority = 0
        if any(action_text.startswith(prefix) or action_text == prefix.strip() for prefix in navigation_prefixes):
            action_priority = 5
        elif any(action_text.startswith(prefix) or action_text == prefix.strip() for prefix in good_prefixes):
            action_priority = 3
        if action_text in {"look around", "inventory", "task"}:
            action_priority = 2
        if any(action_text.startswith(prefix) for prefix in noisy_prefixes):
            action_priority = -2
        if action_text in {"reset task", "stop", "focus on agent"}:
            action_priority = -4

        overlap = len(action_words & context_words)
        focus_target_bonus = 1 if action_text.startswith("focus on ") and overlap else 0
        return (action_priority, focus_target_bonus, overlap, action_text)

    def _rank_valid_actions_for_prompt(self, actions: list[str]) -> list[str]:
        if not actions:
            return []
        if all(str(a).strip().isdigit() for a in actions):
            return actions
        context = " ".join(
            [
                str(getattr(self, "sw_task_desc", "") or ""),
                str(getattr(self, "last_observation", "") or ""),
            ]
        )
        observation_only = {"look around", "inventory", "task"}
        primary = [a for a in actions if self._normalize_action_for_match(a) not in observation_only]
        secondary = [a for a in actions if self._normalize_action_for_match(a) in observation_only]
        primary = sorted(
            primary,
            key=lambda a: self._action_overlap_score(a, context),
            reverse=True,
        )
        return primary + secondary

    def _format_valid_actions_block(self) -> str:
        actions = [str(a).strip() for a in self.last_admissible_commands if str(a).strip()]
        if not actions:
            return "Valid actions: <unavailable>"
        ranked = self._rank_valid_actions_for_prompt(actions)
        limit = self._valid_action_limit()
        shown = ranked if limit <= 0 else ranked[:limit]
        lines = [
            f"Valid actions from the ScienceWorld engine ({len(actions)} total; choose exactly one shown command):"
        ]
        lines.extend(f"- {action}" for action in shown)
        omitted = max(0, len(actions) - len(shown))
        if omitted:
            lines.append(f"- <{omitted} additional low-priority valid actions omitted>")
        return "\n".join(lines)

    def _format_step_context(self, observation: str, *, reward: float | None = None) -> str:
        parts = [str(observation or "").strip()]
        score_line = f"Current ScienceWorld score: {self.last_score:g}."
        if reward is not None:
            score_line += f" Last reward delta: {float(reward):g}."
        parts.append(score_line)
        parts.append(self._format_valid_actions_block())
        return "\n\n".join(part for part in parts if part)

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
        self.best_score = self.last_score
        self.last_reward = 0.0
        self.last_completed = False
        self.has_positive_progress = self.last_score > 0

        self.sw_task = task_name
        self.sw_task_name = str(getattr(self.env, "taskName", "") or configs.get("sw_task_name", task_name))
        self.sw_scene_room = str(configs.get("sw_scene_room", ""))
        self.sw_task_desc = str(configs.get("sw_task_desc", "") or self.env.get_task_description())
        self._task_config = dict(configs)
        self._step_count = 0
        self._refresh_admissible_commands()
        self.current_history = [{
            "Step": 0,
            "Observation": self.last_observation,
            "Score": self.last_score,
            "Done": self.last_completed,
            "Admissible Commands": list(self.last_admissible_commands),
        }]

        task_main = f"scienceworld-{self.task_name}-v{self.variation_idx}"
        task_description = (
            f"{self.env.get_task_description()}\n\n"
            f"Initial observation:\n{self._format_step_context(self.last_observation)}"
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
        first_line = re.sub(r"^\s*[-*]\s*", "", first_line)
        first_line = re.sub(r"^Action\s+\d+:\s*", "", first_line, flags=re.IGNORECASE)
        first_line = re.sub(r"^(?:Action|Command)\s*:\s*", "", first_line, flags=re.IGNORECASE)
        first_line = first_line.strip("`").strip().strip('"').strip("'")
        first_line = re.sub(r"[\[\]\(\)\{\}<>]", "", first_line)
        first_line = re.sub(r"\s+", " ", first_line).strip()
        wait_match = re.match(r"^wait\s+(\d+)\s+seconds?$", first_line, flags=re.IGNORECASE)
        if wait_match:
            return f"wait{wait_match.group(1)}"
        return first_line

    def _refresh_admissible_commands(self) -> None:
        """Refresh current valid ScienceWorld actions from the engine."""
        actions: list[str] = []
        if self.env is None:
            self.last_admissible_commands = []
            return
        try:
            raw_actions = self.env.get_valid_action_object_combinations()
        except Exception:
            raw_actions = []
        for action in raw_actions or []:
            text = str(action or "").strip()
            if text:
                actions.append(text)
        self.last_admissible_commands = actions

    @staticmethod
    def _normalize_action_for_match(action: str) -> str:
        return re.sub(r"\s+", " ", str(action or "").strip()).lower()

    def _repair_action_to_admissible(self, action: str) -> tuple[str, dict[str, Any] | None]:
        """Project a proposed action to the closest current valid engine action."""
        self._refresh_admissible_commands()
        candidates = list(self.last_admissible_commands)
        if not action or not candidates:
            return action, None

        norm_action = self._normalize_action_for_match(action)
        by_norm = {self._normalize_action_for_match(candidate): candidate for candidate in candidates}
        exact = by_norm.get(norm_action)
        if exact is not None:
            return exact, None

        best_action = ""
        best_score = -1.0
        for candidate in candidates:
            score = SequenceMatcher(None, norm_action, self._normalize_action_for_match(candidate)).ratio()
            if score > best_score:
                best_action = candidate
                best_score = score
        if not best_action:
            return action, None
        return best_action, {
            "original_action": action,
            "repaired_action": best_action,
            "match_score": best_score,
            "num_candidates": len(candidates),
        }

    def step(self, action: str) -> tuple[str, float, bool]:
        if self.env is None:
            raise RuntimeError("ScienceWorld environment is not initialized. Call set_env first.")

        proposed_action = self.process_action(action)
        env_action, repair_info = self._repair_action_to_admissible(proposed_action)
        observation, reward, completed, step_info = self.env.step(env_action)
        self.last_observation = observation or ""
        self.last_reward = float(reward)
        self.last_score = float(step_info.get("score", self.last_score))
        self.best_score = max(float(getattr(self, "best_score", 0.0) or 0.0), self.last_score)
        self.last_completed = bool(completed)
        if self.last_reward > 0 or self.last_score > 0 or self.best_score > 0:
            self.has_positive_progress = True
        self._step_count += 1
        self._refresh_admissible_commands()
        self.current_history.append({
            "Step": self._step_count,
            "Action": env_action,
            "Observation": self.last_observation,
            "Reward": self.last_reward,
            "Score": self.last_score,
            "Done": self.last_completed,
            "Admissible Commands": list(self.last_admissible_commands),
            "Action Repair": repair_info,
        })
        return self._format_step_context(self.last_observation, reward=self.last_reward), self.last_reward, self.last_completed

    def has_exportable_history(self) -> bool:
        return bool(getattr(self, "current_history", None))

    def export_memco_history(
        self,
        output_dir: str,
        *,
        model_id: str = "",
        status_override: str | None = None,
    ) -> str | None:
        if not self.has_exportable_history():
            return None
        final_score = float(getattr(self, "last_score", 0.0) or 0.0)
        best_score = float(getattr(self, "best_score", final_score) or 0.0)
        success = bool(getattr(self, "last_completed", False) and final_score > 0)
        status = status_override or ("success" if success else "fail")
        sw_task = str(getattr(self, "sw_task", "") or "unknown")
        sw_task_name = str(getattr(self, "sw_task_name", "") or sw_task)
        room = str(getattr(self, "sw_scene_room", "") or "unknown")
        variation_idx = int(getattr(self, "variation_idx", 0) or 0)
        payload = {
            "last_updated": time.strftime("%Y%m%d_%H%M%S"),
            "sw_task": sw_task,
            "sw_task_name": sw_task_name,
            "sw_scene_room": room,
            "variation_idx": variation_idx,
            "sw_task_desc": str(getattr(self, "sw_task_desc", "") or ""),
            "game_task": str(getattr(self, "sw_task_desc", "") or ""),
            "status": status,
            "step_count": max(len(self.current_history) - 1, 0),
            "final_score": final_score,
            "best_score": best_score,
            "had_positive_progress": bool(getattr(self, "has_positive_progress", False)),
            "history": list(self.current_history),
            "model_id": model_id,
            "memco_domain": "scienceworld",
            "task_config": dict(getattr(self, "_task_config", {}) or {}),
        }
        os.makedirs(os.path.abspath(output_dir), exist_ok=True)
        safe_room = re.sub(r"[^a-z0-9]+", "_", str(room).lower()).strip("_") or "unknown"
        safe_task = re.sub(r"[^a-z0-9]+", "_", str(sw_task_name).lower()).strip("_") or "task"
        out_path = os.path.join(
            os.path.abspath(output_dir),
            f"history_scienceworld_{safe_room}_{safe_task}_v{variation_idx}_{status}.json",
        )
        with open(out_path, "w", encoding="utf-8") as writer:
            import json as _json
            _json.dump(payload, writer, ensure_ascii=False, indent=2)
        return out_path

    def feedback(self) -> tuple[float, bool, str]:
        # ScienceWorld true completion may happen with negative/zero score in some failures.
        success = bool(self.last_completed and self.last_score > 0)
        reward = 1.0 if success else 0.0
        feedback = (
            f"ScienceWorld episode finished. completed={self.last_completed}, score={self.last_score}, best_score={self.best_score}."
            if self.last_completed
            else f"ScienceWorld episode not finished. score={self.last_score}, best_score={self.best_score}."
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
