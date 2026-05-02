from typing import Any, Literal
import re
from dataclasses import dataclass
import json
import os

from .base_env import BaseEnv, BaseRecorder
from .utils import LangChainWiki, match_exactly

class FeverEnv(BaseEnv):
    def __init__(
        self, 
        env_config: dict[str, Any], 
        max_trials: int = 7
    ) -> None:
        
        self.env_config = env_config
        self.gm3_domain = "fever"
        self.explorer = LangChainWiki()
        self.max_trials: int = max_trials
        self.config: dict[str, Any] = {}
        self.claim: str = ""
        self.ab_domain: str = ""
        self.current_history: list[dict[str, Any]] = []
        self.last_admissible_commands: list[str] = []
        
        self.reset()
        
    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get('answer') is None:
            raise ValueError('Please provide the answer for the question.')
        if configs.get('task') is None:
            raise ValueError('The configs dict should have the `task` attribute.')
        self.config = dict(configs or {})
        self.claim = str(self.config.get('task') or "")
        self.ab_domain = str(self.config.get("ab_domain", "") or "")
        
        task_value = self.claim
        task: str = f'Claim: {task_value}'
        self.current_task = task
        return task, task
    
    def reset(self) -> None:
        claim = str(getattr(self, "claim", "") or (getattr(self, "config", {}) or {}).get("task", "") or "")
        self.current_task = f"Claim: {claim}" if claim else None
        self.reward: float = 0
        self.steps: int = 0
        self.last_admissible_commands = self._default_admissible_commands(claim)
        self.current_history = [
            {
                "Step": 0,
                "Observation": f"Claim: {claim}" if claim else "",
                "Claim": claim,
                "Admissible Commands": list(self.last_admissible_commands),
                "Score": float(self.reward),
                "Reward": 0.0,
                "Done": False,
            }
        ]
    
    def step(self, action: str) -> tuple[str, float, bool]:

        action: str = self.process_action(action)
        self.steps += 1

        if self._parse_action_type(action) == 'thought':
            observation = 'OK.'
            self._append_gm3_history(
                action=action,
                observation=observation,
                reward=0,
                done=False,
                action_type="thought",
                argument="",
            )
            return observation, 0, False
        
        action_type, argument = self._parse_action(action)

        if action_type == 'Finish':
            if self.success_fn(argument):
                observation = 'Answer is CORRECT'
                self.reward = 1
                self._append_gm3_history(
                    action=action,
                    observation=observation,
                    reward=1,
                    done=True,
                    action_type=action_type,
                    argument=argument,
                )
                return observation, 1, True

            else: 
                observation = f'Answer is INCORRECT'
                self._append_gm3_history(
                    action=action,
                    observation=observation,
                    reward=0,
                    done=True,
                    action_type=action_type,
                    argument=argument,
                )
                return observation, 0, True

        elif action_type == 'Search':
            while True:
                try:
                    observation = self.explorer.search(argument).strip('\n').strip()
                    self.summary = observation
                    break
                except Exception as e:
                    print(e)
                    observation = f'Cannot find corresponding pages.'
                    break
        elif action_type == 'Lookup':
            try:
                observation = self.explorer.lookup(argument).strip('\n').strip()
            except ValueError:
                observation = f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        else:
            observation = 'Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
        
        if 'Invalid Action' in observation:
            processed_reward = -1
        else:
            processed_reward = 0
        self._append_gm3_history(
            action=action,
            observation=observation,
            reward=processed_reward,
            done=False,
            action_type=str(action_type or ""),
            argument=str(argument or ""),
        )
        return observation, processed_reward, False
    
    @staticmethod
    def _parse_action_type(action: str) -> Literal['action', 'thought']:
        if 'thought' in action.lower():
            return 'thought'
        else:
            return 'action'
    
    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip().replace('<', '').split('\n')[0]
        action = action.replace('>', '').replace('OK.', '').replace('OK', '').strip()
        
        if FeverEnv._parse_action_type(action) == 'thought':
            return action
        if ':' in action:
            action = action.split(':')[1].strip()

        return action

    @staticmethod
    def _parse_action(string: str) -> tuple[str, str]:

        pattern = r'^(\w+)\[(.+)\]$'
        match = re.match(pattern, string)
        
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        else:
            return None, None
        
    def success_fn(self, agent_ans: str) -> bool:
        return match_exactly(agent_ans, self.config.get('answer'))
    
    def feedback(self) -> tuple[float, bool, str]:
        
        feedback: str = 'You successfully finished this task.' if self.reward == 1 else 'You failed the task.'
        done = self.reward == 1

        return self.reward, done, feedback

    @staticmethod
    def _default_admissible_commands(claim: str = "") -> list[str]:
        anchor = FeverEnv._claim_anchor(claim)
        commands = [
            f"Search[{anchor or '<topic>'}]",
            f"Lookup[{anchor or '<keyword>'}]",
            "Finish[SUPPORTS]",
            "Finish[REFUTES]",
            "Finish[NOT ENOUGH INFO]",
        ]
        return commands

    @staticmethod
    def _claim_anchor(claim: str) -> str:
        text = str(claim or "").strip()
        match = re.search(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4}", text)
        if match:
            return match.group(0)
        words = [w for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 2]
        return words[0] if words else ""

    def _append_gm3_history(
        self,
        *,
        action: str,
        observation: str,
        reward: float,
        done: bool,
        action_type: str,
        argument: str,
    ) -> None:
        claim = str(getattr(self, "claim", "") or (getattr(self, "config", {}) or {}).get("task", "") or "")
        self.last_admissible_commands = self._default_admissible_commands(claim)
        self.current_history.append(
            {
                "Step": int(self.steps),
                "Action": str(action or ""),
                "Action Type": str(action_type or ""),
                "Argument": str(argument or ""),
                "Observation": str(observation or ""),
                "Claim": claim,
                "Admissible Commands": list(self.last_admissible_commands),
                "Score": float(self.reward),
                "Reward": float(reward),
                "Done": bool(done),
            }
        )

    def has_exportable_history(self) -> bool:
        return bool(getattr(self, "current_history", None))

    def export_gm2_history(
        self,
        output_dir: str,
        *,
        model_id: str = "",
        status_override: str | None = None,
    ) -> str | None:
        if not self.has_exportable_history():
            return None
        claim = str(getattr(self, "claim", "") or (getattr(self, "config", {}) or {}).get("task", "") or "")
        final_done = bool(self.current_history[-1].get("Done", False)) if self.current_history else False
        final_score = float(self.reward or 0.0)
        status = status_override or ("success" if final_done and final_score > 0 else "fail")
        domain = str(getattr(self, "ab_domain", "") or (getattr(self, "config", {}) or {}).get("ab_domain", "") or "default")
        payload = {
            "last_updated": __import__("time").strftime("%Y%m%d_%H%M%S"),
            "game_file": "",
            "game_name": "fever",
            "game_index": str((getattr(self, "config", {}) or {}).get("task_id", "")),
            "game_task": f"Claim: {claim}",
            "claim": claim,
            "answer": str((getattr(self, "config", {}) or {}).get("answer", "")),
            "ab_domain": domain,
            "status": status,
            "step_count": max(len(self.current_history) - 1, 0),
            "final_score": final_score,
            "history": list(self.current_history),
            "model_id": model_id,
            "gm3_domain": self.gm3_domain,
            "task_family": f"fever:{domain}",
            "task_config": dict(getattr(self, "config", {}) or {}),
        }
        out_dir = os.path.abspath(output_dir)
        os.makedirs(out_dir, exist_ok=True)
        digest = abs(hash(claim)) % 10_000_000
        safe_domain = domain.replace("/", "_")
        out_path = os.path.join(out_dir, f"history_fever_{safe_domain}_{digest}_{status}.json")
        with open(out_path, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False, indent=2)
        return out_path
    


@dataclass
class FeverRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = 'hotpotqa'
        self.counts = 0
        self.dones = 0
        self.rewards = 0
    
    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        
        message: str = f'---------- Task: {task_id} ----------'
        self.log(message)
    
    def task_end(self, reward: float, done: bool):
        
        self.rewards += reward
        self.dones += done
        self.counts += 1

        message = f'reward: {reward}, ave reward: {self.rewards / self.counts}.\ndone: {done}, ave done: {self.dones / self.counts}'
        self.log(message)
