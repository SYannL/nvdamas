import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mas  # noqa: F401  # load .env via mas.__init__
from mas.llm import GPTChat, Message
from scienceworld import ScienceWorldEnv

def _parse_action(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""
    if "```" in text:
        parts = [p.strip() for p in text.split("```") if p.strip()]
        if parts:
            text = parts[0]
    first_line = text.splitlines()[0].strip()
    return first_line.strip("`").strip().strip('"').strip("'")


def _build_prompt(
    task_name: str,
    task_description: str,
    step_idx: int,
    max_steps: int,
    observation: str,
    score: int,
    valid_actions: list[str],
    action_history: list[str],
) -> str:
    action_history_preview = action_history[-5:] if action_history else []
    actions_text = "\n".join(f"- {a}" for a in valid_actions)
    history_text = "\n".join(f"- {a}" for a in action_history_preview) if action_history_preview else "- <none>"
    is_ambiguous_resolution = bool(valid_actions) and all(a.isdigit() for a in valid_actions)
    if is_ambiguous_resolution:
        action_instruction = (
            "The environment is asking you to resolve an ambiguous request.\n"
            "Choose exactly one option index from the valid actions list below.\n"
            "Return ONE number only (e.g., 0). No explanation."
        )
    else:
        action_instruction = (
            "Choose exactly one action from the valid actions list below.\n"
            "Return ONE executable action line only. No explanation."
        )
    return (
        f"You are solving a ScienceWorld task.\n"
        f"Task name: {task_name}\n"
        f"Task description: {task_description}\n"
        f"Step: {step_idx}/{max_steps}\n"
        f"Current score: {score}\n\n"
        f"Recent action history (latest up to 5):\n{history_text}\n\n"
        f"Observation:\n{observation}\n\n"
        f"{action_instruction}\n\n"
        f"Valid actions:\n{actions_text}\n"
    )


@dataclass
class SmokeResult:
    run_dir: str
    task: str
    variation: int
    steps_taken: int
    final_score: int
    completed: bool


def run_episode(args: argparse.Namespace) -> SmokeResult:
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "trajectory.jsonl"
    summary_path = run_dir / "summary.json"

    planner = GPTChat(model_name=args.model)
    env = ScienceWorldEnv(envStepLimit=args.max_steps + 1)

    try:
        env.load(
            taskName=args.task,
            variationIdx=args.variation,
            simplificationStr=args.simplification,
            generateGoldPath=False,
        )
        observation, info = env.reset()
        task_name = env.taskName
        task_description = env.get_task_description()
        action_history: list[str] = []
        completed = False
        final_score = int(info.get("score", 0))
        steps_taken = 0

        with trajectory_path.open("w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(
                    {
                        "event": "episode_start",
                        "task": task_name,
                        "variation": args.variation,
                        "simplification": args.simplification,
                        "seed": args.seed,
                        "model": args.model,
                        "initial_observation": observation,
                        "task_description": task_description,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            for step_idx in range(1, args.max_steps + 1):
                valid_actions = env.get_valid_action_object_combinations()
                is_ambiguous_resolution = bool(valid_actions) and all(a.isdigit() for a in valid_actions)
                prompt = _build_prompt(
                    task_name=task_name,
                    task_description=task_description,
                    step_idx=step_idx,
                    max_steps=args.max_steps,
                    observation=observation,
                    score=final_score,
                    valid_actions=valid_actions,
                    action_history=action_history,
                )
                llm_response = planner(
                    [
                        Message(
                            role="system",
                            content="You are a precise planner for text-game environments.",
                        ),
                        Message(role="user", content=prompt),
                    ],
                    temperature=0.0,
                    max_tokens=64,
                    stop_strs=None,
                    num_comps=1,
                )
                proposed_action = _parse_action(llm_response)
                chosen_action = proposed_action

                next_observation, reward, completed, step_info = env.step(chosen_action)
                final_score = int(step_info.get("score", 0))
                action_history.append(chosen_action)
                steps_taken = step_idx

                writer.write(
                    json.dumps(
                        {
                            "event": "step",
                            "step": step_idx,
                            "prompt": prompt,
                            "is_ambiguous_resolution": is_ambiguous_resolution,
                            "llm_response": llm_response,
                            "proposed_action": proposed_action,
                            "chosen_action": chosen_action,
                            "reward": reward,
                            "score": final_score,
                            "completed": completed,
                            "observation": next_observation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                observation = next_observation
                if completed:
                    break

            writer.write(
                json.dumps(
                    {
                        "event": "episode_end",
                        "steps_taken": steps_taken,
                        "final_score": final_score,
                        "completed": completed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        summary = {
            "task": task_name,
            "variation": args.variation,
            "simplification": args.simplification,
            "seed": args.seed,
            "model": args.model,
            "max_steps": args.max_steps,
            "steps_taken": steps_taken,
            "final_score": final_score,
            "completed": completed,
            "trajectory_path": str(trajectory_path),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return SmokeResult(
            run_dir=str(run_dir),
            task=task_name,
            variation=args.variation,
            steps_taken=steps_taken,
            final_score=final_score,
            completed=completed,
        )
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case ScienceWorld smoke episode with OpenAI LLM.")
    parser.add_argument("--task", type=str, default="1-1", help="Task id or task name, e.g. 1-1 or boil.")
    parser.add_argument("--variation", type=int, default=0, help="Variation index.")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps in the episode.")
    parser.add_argument("--simplification", type=str, default="", help="ScienceWorld simplification string.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model name.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs/scienceworld_smoke",
        help="Directory to save trajectory and summary outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic fallback action selection.")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    os.makedirs(cli_args.output_dir, exist_ok=True)
    result = run_episode(cli_args)
    print(
        "ScienceWorld smoke run finished | "
        f"task={result.task} variation={result.variation} "
        f"steps={result.steps_taken} score={result.final_score} completed={result.completed} "
        f"run_dir={result.run_dir}"
    )
