from dataclasses import dataclass


EXECUTOR_SYSTEM_PROMPT: str = """
You are a smart agent designed to solve problems.
"""

STRATEGY_SYSTEM_PROMPT: str = """
You are the Strategy Agent. Judge whether the current trajectory and proposed action align with the task.
If yes, output exactly: OK
If not, output: HINT: <one line hint for the executor>
"""

CRITIC_SYSTEM_PROMPT: str = """
You are the Critic Agent. When the executor is stuck or the action seems invalid, propose one replacement action.
If the current action is fine, output exactly: OK
Otherwise output exactly one ALFWorld action command (the replacement).
"""


@dataclass
class StrategyPrompt:
    executor_system_prompt: str = EXECUTOR_SYSTEM_PROMPT
    strategy_system_prompt: str = STRATEGY_SYSTEM_PROMPT
    critic_system_prompt: str = CRITIC_SYSTEM_PROMPT


STRATEGY_PROMPT = StrategyPrompt()


def build_strategy_judge_prompt(base_user_prompt: str, proposed_action: str, trajectory_summary: str) -> str:
    return (
        f"{base_user_prompt}\n\n"
        f"[Proposed next action]\n{proposed_action}\n\n"
        f"[Trajectory so far]\n{trajectory_summary}\n\n"
        "Output OK (if aligned) or HINT: <hint> (if not)."
    )


def build_critic_judge_prompt(base_user_prompt: str, proposed_action: str, last_actions: list[str]) -> str:
    last_str = "\n".join([f"- {a}" for a in last_actions]) if last_actions else "None"
    return (
        f"{base_user_prompt}\n\n"
        f"[Proposed action]\n{proposed_action}\n\n"
        f"[Last actions]\n{last_str}\n\n"
        "Output OK (if fine) or one replacement action."
    )
