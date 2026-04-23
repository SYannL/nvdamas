import sys
import re
from dataclasses import dataclass

from mas.agents import Agent
from mas.memory.common import MASMessage, AgentMessage
from mas.mas import MetaMAS
from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.memory import MASMemoryBase, GMemory
from mas.agents import Env

from .strategy_prompt import (
    STRATEGY_PROMPT,
    build_strategy_judge_prompt,
    build_critic_judge_prompt,
)
from ..format import format_task_prompt_with_insights, format_task_context


def _sanitize_action_text(action: str) -> str:
    """Normalize model action text for stricter env parsing."""
    if not action:
        return action
    cleaned = action.strip()
    while cleaned and cleaned[-1] in "。．.!！?？;；,，":
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _preview(text: str, limit: int = 220) -> str:
    if text is None:
        return ""
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def _parse_strategy_judge(out: str) -> tuple[bool, str]:
    """Parse Strategy output: (ok, hint). ok=True means no hint needed."""
    if not out:
        return True, ""
    s = out.strip().upper()
    if s == "OK":
        return True, ""
    m = re.search(r"HINT\s*:\s*(.+)", out, re.IGNORECASE | re.DOTALL)
    if m:
        return False, m.group(1).strip()
    return True, ""


def _parse_critic_judge(out: str) -> str | None:
    """Parse Critic output: None means OK (use executor action), else replacement action."""
    if not out:
        return None
    s = out.strip().upper()
    if s == "OK":
        return None
    # Treat non-empty non-OK as replacement action
    cleaned = _sanitize_action_text(out.strip())
    if cleaned:
        return cleaned
    return None


@dataclass
class StrategyMAS(MetaMAS):
    """
    Executor = ReAct primary (one action per step).
    Strategy = judgment only (ok, hint).
    Critic = judgment + replacement when stuck.
    """

    def __post_init__(self):
        self.observers = []
        self.reasoning_config = ReasoningConfig(temperature=0, stop_strs=["\n"])
        self._successful_topk: int = 1
        self._failed_topk: int = 1
        self._insights_topk: int = 3
        self._threshold: float = 0.0
        self._use_projector: bool = False
        self._max_strategy_retries: int = 1

    def build_system(
        self,
        reasoning: ReasoningBase,
        mas_memory: MASMemoryBase,
        env: Env,
        config: dict,
    ):
        self._silent_mas: bool = bool(
            config.get("silent_mas", False) or config.get("quiet", False)
        )
        self._successful_topk = config.get("successful_topk", 1)
        self._failed_topk = config.get("failed_topk", 1)
        self._insights_topk = config.get("insights_topk", 3)
        self._threshold = config.get("threshold", 0)
        self._use_projector = config.get("use_projector", False)
        self._max_strategy_retries = int(config.get("max_strategy_retries", 1))

        self.notify_observers(f"Successful Topk   : {self._successful_topk}")
        self.notify_observers(f"Failed Topk       : {self._failed_topk}")
        self.notify_observers(f"Insights Topk     : {self._insights_topk}")
        self.notify_observers(f"Retrieve Threshold: {self._threshold}")
        self.notify_observers(f"Use Role Projector: {self._use_projector}")
        self.notify_observers(f"Max Strategy Retries: {self._max_strategy_retries}")

        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning module must be an instance of ReasoningBase")
        if not isinstance(mas_memory, MASMemoryBase):
            raise TypeError("mas_memory module must be an instance of MASMemoryBase")

        executor_agent = Agent(
            name="executor",
            role="executor",
            system_instruction=STRATEGY_PROMPT.executor_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )
        strategy_agent = Agent(
            name="strategy",
            role="strategy",
            system_instruction=STRATEGY_PROMPT.strategy_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )
        critic_agent = Agent(
            name="critic",
            role="critic",
            system_instruction=STRATEGY_PROMPT.critic_system_prompt,
            reasoning_module=reasoning,
            memory_module=None,
        )

        self.hire([executor_agent, strategy_agent, critic_agent])
        self.set_env(env)
        self.meta_memory = mas_memory

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, message: str):
        for observer in self.observers:
            observer.log(message)

    def schedule(self, task_config: dict) -> tuple[float, bool]:
        if task_config.get("task_main") is None:
            raise ValueError("Missing required keys `task_main` in task_config")
        if task_config.get("task_description") is None:
            raise ValueError("Missing required keys `task_description` in task_config")

        task_main: str = task_config.get("task_main")
        task_description: str = task_config.get("task_description")
        few_shots: list[str] = task_config.get("few_shots", [])

        env: Env = self.env
        executor: Agent = self.get_agent("executor")
        strategy: Agent = self.get_agent("strategy")
        critic: Agent = self.get_agent("critic")
        if executor is None or strategy is None or critic is None:
            raise RuntimeError("StrategyMAS missing required agents.")

        env.reset()
        self.meta_memory.init_task_context(task_main, task_description)

        successful_trajectories: list[MASMessage]
        insights: list[dict]
        successful_trajectories, _, insights = self.meta_memory.retrieve_memory(
            query_task=task_main,
            successful_topk=self._successful_topk,
            failed_topk=self._failed_topk,
            insight_topk=self._insights_topk,
            threshold=self._threshold,
        )
        successful_shots: list[str] = [
            format_task_context(
                traj.task_description,
                traj.task_trajectory,
                traj.get_extra_field("key_steps"),
                source=traj.get_extra_field("source_id"),
            )
            for traj in successful_trajectories
        ]
        raw_rules: list[str] = [insight for insight in insights]
        roles_rules: dict[str, list[str]] = self._project_insights(raw_rules)

        prompt_for_log: str = format_task_prompt_with_insights(
            few_shots=few_shots,
            memory_few_shots=successful_shots,
            insights=roles_rules.get(executor.profile, raw_rules),
            task_description=self.meta_memory.summarize(),
        )
        self.notify_observers("=== Prompt With Memory/Insights ===")
        self.notify_observers(prompt_for_log)

        if task_config.get("auto_first_search"):
            first_search_display = "Search[（已自动执行当前题目检索）]"
            observation, _r, _d = env.step("Search[.]")
            self.meta_memory.move_memory_state(first_search_display, observation)
            self.notify_observers(
                f"Act 1: {first_search_display}\nObs 1: {observation[:500]}{'...' if len(observation) > 500 else ''}"
            )

        action_history: list[str] = []

        for i in range(env.max_trials):
            if not self._silent_mas:
                print(f"  [MAS] step {i + 1}/{env.max_trials} ...", flush=True)
                sys.stdout.flush()

            base_user_prompt: str = format_task_prompt_with_insights(
                few_shots=few_shots,
                memory_few_shots=successful_shots,
                insights=roles_rules.get(executor.profile, raw_rules),
                task_description=self.meta_memory.summarize(),
            )
            trajectory_summary: str = self.meta_memory.summarize()

            # ----- Executor ReAct + Strategy judgment (optional retry with hint) -----
            action: str = ""
            context_with_hint: str = base_user_prompt
            for attempt in range(self._max_strategy_retries + 1):
                tries = 0
                while tries < 3:
                    try:
                        action_raw: str = executor.response(context_with_hint, self.reasoning_config)
                        self.notify_observers(
                            f"[detail][step {i + 1}] executor.output.try_{tries + 1}: {action_raw}"
                        )
                        if not action_raw:
                            tries += 1
                            continue
                        action = _sanitize_action_text(action_raw)
                        action = env.process_action(action)
                        self.notify_observers(
                            f"[detail][step {i + 1}] executor.processed: {action}"
                        )
                        break
                    except Exception as e:
                        self.notify_observers(
                            f"[detail][step {i + 1}] executor.error.try_{tries + 1}: {type(e).__name__}: {e}"
                        )
                        if not self._silent_mas:
                            print(f"Error during executor: {e}")
                    tries += 1

                if not action:
                    break

                # Strategy judges
                strategy_prompt: str = build_strategy_judge_prompt(
                    base_user_prompt, action, trajectory_summary
                )
                tries = 0
                strategy_ok, hint = True, ""
                while tries < 3:
                    try:
                        strategy_out = strategy.response(strategy_prompt, self.reasoning_config)
                        self.notify_observers(
                            f"[detail][step {i + 1}] strategy.judge: {strategy_out}"
                        )
                        strategy_ok, hint = _parse_strategy_judge(strategy_out)
                        break
                    except Exception as e:
                        self.notify_observers(
                            f"[detail][step {i + 1}] strategy.error: {type(e).__name__}: {e}"
                        )
                        strategy_ok = True
                    tries += 1

                if strategy_ok or not hint:
                    break
                context_with_hint = base_user_prompt + "\n\n[Strategy hint] " + hint
                self.notify_observers(
                    f"[detail][step {i + 1}] strategy.retry with hint: {hint}"
                )

            # ----- Critic when stuck -----
            name: str = executor.name
            system_instruction = executor.system_instruction
            user_instruction: str = context_with_hint

            if action and self._executor_stuck(action, action_history):
                self.notify_observers(
                    f"[detail][step {i + 1}] critic.triggered: stuck"
                )
                critic_prompt: str = build_critic_judge_prompt(
                    base_user_prompt, action, action_history[-3:] if len(action_history) >= 3 else action_history
                )
                tries = 0
                while tries < 3:
                    try:
                        critic_out = critic.response(critic_prompt, self.reasoning_config)
                        self.notify_observers(
                            f"[detail][step {i + 1}] critic.output: {critic_out}"
                        )
                        replacement = _parse_critic_judge(critic_out)
                        if replacement:
                            action = replacement
                            name = critic.name
                            system_instruction = critic.system_instruction
                            user_instruction = critic_prompt
                        break
                    except Exception as e:
                        self.notify_observers(
                            f"[detail][step {i + 1}] critic.error: {type(e).__name__}: {e}"
                        )
                    tries += 1
            else:
                self.notify_observers(f"[detail][step {i + 1}] critic.triggered: False")

            agent_message: AgentMessage = AgentMessage(
                agent_name=name,
                system_instruction=system_instruction,
                user_instruction=user_instruction,
                message=action,
            )
            self.meta_memory.add_agent_node(agent_message, upstream_agent_ids=[])

            if not action:
                action = "look"
            observation, reward, done = env.step(action)
            action_history.append(action)
            action_preview = (action[:60] + "..") if len(action) > 60 else action
            if not self._silent_mas:
                print(f"  [MAS] step {i + 1} done: {action_preview}", flush=True)
                sys.stdout.flush()

            step_message: str = f"Act {i + 1}: {action}\nObs {i + 1}: {observation}"
            self.notify_observers(step_message)
            self.notify_observers(
                f"[detail][step {i + 1}] env.done={done}, env.won={getattr(env, 'won', None)}, step_reward={reward}"
            )
            self.meta_memory.move_memory_state(action, observation, reward=reward)

            if done:
                break

        final_reward, final_done, final_feedback = self.env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.save_task_context(label=final_done, feedback=final_feedback)
        self.meta_memory.backward(final_done)
        return final_reward, final_done

    def _executor_stuck(self, current: str, history: list[str]) -> bool:
        return (
            len(history) >= 2
            and current == history[-1]
            and current == history[-2]
        )

    def _project_insights(self, insights: list[str]) -> dict[str, list[str]]:
        roles_rules: dict[str, list[str]] = {}
        roles = set(agent.profile for agent in self.agents_team.values())
        if not self._use_projector or not isinstance(self.meta_memory, GMemory):
            for role in roles:
                roles_rules[role] = insights
        else:
            for role in roles:
                roles_rules[role] = self.meta_memory.project_insights(insights, role)
        for role, role_insights in roles_rules.items():
            roles_rules[role] = role_insights[: self._insights_topk]
        return roles_rules
