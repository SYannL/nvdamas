import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Any

import yaml

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mas.llm import GPTChat, Message
from tasks.envs import get_task


with open("tasks/configs.yaml") as reader:
    CONFIG: dict = yaml.safe_load(reader)


SYSTEM_PROMPT = (
    "You are a medical multiple-choice QA model. "
    "Answer with only the single letter A, B, C, or D."
)


@dataclass
class EvalResult:
    dataset: str
    accuracy: float
    avg_reward: float
    num_tasks: int


def build_prompt(task: dict[str, Any]) -> str:
    subject = task.get("subject_name")
    subject_line = f"Subject: {subject}\n" if subject else ""
    options = task.get("options", {})
    options_block = "\n".join([f"{key}. {value}" for key, value in options.items()])
    return f"{subject_line}Question: {task.get('task')}\nOptions:\n{options_block}\nAnswer:"


def extract_choice(text: str) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned in {"A", "B", "C", "D"}:
        return cleaned
    if len(cleaned) > 0:
        for ch in cleaned:
            if ch in {"A", "B", "C", "D"}:
                return ch
    return None


def eval_dataset(
    llm: GPTChat,
    dataset: str,
    output_dir: str,
) -> EvalResult:
    tasks = get_task(dataset)
    if not tasks:
        return EvalResult(dataset=dataset, accuracy=0.0, avg_reward=0.0, num_tasks=0)

    correct = 0
    rewards = []
    output_path = os.path.join(output_dir, f"{dataset}.jsonl")
    total = len(tasks)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, task in enumerate(tasks):
            print(f"[{dataset}] {idx + 1}/{total}", flush=True)
            prompt = build_prompt(task)
            response = llm([Message("system", SYSTEM_PROMPT), Message("user", prompt)], temperature=0)
            pred = extract_choice(response)
            answer_key = task.get("answer_key")
            reward = 1.0 if pred == answer_key else 0.0
            if reward > 0:
                correct += 1
            rewards.append(reward)

            record = {
                "dataset": dataset,
                "index": idx,
                "question": task.get("task"),
                "options": task.get("options"),
                "answer_key": answer_key,
                "prediction": pred,
                "raw_response": response,
                "reward": reward,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    accuracy = correct / total
    avg_reward = sum(rewards) / total
    return EvalResult(dataset=dataset, accuracy=accuracy, avg_reward=avg_reward, num_tasks=total)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LLM-only baseline evaluation (no tools).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("reports", "llm_baseline", run_id)
    ensure_dir(output_dir)

    llm = GPTChat(model_name=args.model)

    datasets = [
        # "medmcqa_physio_30",
        # "medmcqa_pharma_30",
        # "medmcqa_anatomy_20",
        # "medmcqa_surgery_20",
        # "medmcqa_test",
        "medmcqa_physio_150_build",
        "medmcqa_pharma_150_build",
        # "medmcqa_physio_20_test",
        # "medmcqa_pharma_20_test",
    ]

    results: list[dict[str, Any]] = []
    for index, dataset in enumerate(datasets, start=1):
        print(f"[dataset] {index}/{len(datasets)} - {dataset}", flush=True)
        res = eval_dataset(llm, dataset, output_dir)
        results.append(
            {
                "dataset": res.dataset,
                "accuracy": res.accuracy,
                "avg_reward": res.avg_reward,
                "num_tasks": res.num_tasks,
            }
        )

    summary_json = os.path.join(output_dir, "summary.json")
    summary_md = os.path.join(output_dir, "summary.md")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# LLM-only Baseline (no tools)\n\n")
        f.write("| Dataset | Accuracy | Avg Reward | Num Tasks |\n")
        f.write("|---|---:|---:|---:|\n")
        for item in results:
            f.write(
                f"| {item['dataset']} | {item['accuracy']:.4f} | {item['avg_reward']:.4f} | {item['num_tasks']} |\n"
            )

    print(f"[ok] summary_json: {summary_json}")
    print(f"[ok] summary_md: {summary_md}")


if __name__ == "__main__":
    main()
