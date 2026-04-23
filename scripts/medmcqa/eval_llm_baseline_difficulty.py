"""
LLM-only 评估 + 难度排序：对 physio/pharma 各 150 题用 4o-mini 做题并输出 1–100 难度分，
再按「先做对的从易到难、再做错的从易到难」排序，输出每题的难度、是否正确、排序 index。
"""
import os
import re
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_script_dir, "../..")))
sys.path.insert(0, _script_dir)

from mas.llm import GPTChat, Message
from tasks.envs import get_task

from prompts_llm_difficulty import SYSTEM_PROMPT_DIFFICULTY, build_user_prompt


def extract_answer(text: str) -> str | None:
    if not text:
        return None
    # Answer: A  or  Answer: A.
    m = re.search(r"Answer\s*:\s*([A-Da-d])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for ch in text.strip():
        if ch in "ABCD":
            return ch
    return None


def extract_difficulty(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"Difficulty\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        return max(1, min(100, val))
    return None


@dataclass
class QuestionResult:
    original_index: int
    difficulty: int | None
    correct: bool
    prediction: str | None
    answer_key: str
    raw_response: str = ""


def eval_dataset_with_difficulty(
    llm: GPTChat,
    dataset: str,
    output_dir: str,
) -> list[QuestionResult]:
    tasks = get_task(dataset)
    if not tasks:
        return []

    results: list[QuestionResult] = []
    total = len(tasks)
    raw_path = os.path.join(output_dir, f"{dataset}_raw.jsonl")

    with open(raw_path, "w", encoding="utf-8") as f:
        for idx, task in enumerate(tasks):
            print(f"[{dataset}] {idx + 1}/{total}", flush=True)
            user_prompt = build_user_prompt(task)
            response = llm(
                [
                    Message("system", SYSTEM_PROMPT_DIFFICULTY),
                    Message("user", user_prompt),
                ],
                temperature=0,
            )
            pred = extract_choice(response)
            diff = extract_difficulty(response)
            answer_key = task.get("answer_key", "")
            correct = pred == answer_key
            if diff is None:
                diff = 50
            results.append(
                QuestionResult(
                    original_index=idx,
                    difficulty=diff,
                    correct=correct,
                    prediction=pred,
                    answer_key=answer_key,
                    raw_response=response,
                )
            )
            f.write(
                json.dumps(
                    {
                        "dataset": dataset,
                        "index": idx,
                        "question": task.get("task"),
                        "answer_key": answer_key,
                        "prediction": pred,
                        "difficulty": diff,
                        "correct": correct,
                        "raw_response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return results


def extract_choice(text: str) -> str | None:
    out = extract_answer(text)
    return out


def compute_sorted_order(results: list[QuestionResult]) -> list[dict[str, Any]]:
    """排序：先做对的按难度从低到高，再做错的按难度从低到高；赋予 sorted_index 0..N-1。"""
    correct_list = [(i, r) for i, r in enumerate(results) if r.correct]
    wrong_list = [(i, r) for i, r in enumerate(results) if not r.correct]
    correct_list.sort(key=lambda x: (x[1].difficulty or 0, x[0]))
    wrong_list.sort(key=lambda x: (x[1].difficulty or 0, x[0]))

    ordered: list[tuple[int, QuestionResult, int]] = []
    for pos, (orig_i, r) in enumerate(correct_list):
        ordered.append((orig_i, r, pos))
    for pos, (orig_i, r) in enumerate(wrong_list):
        ordered.append((orig_i, r, len(correct_list) + pos))

    out = []
    for orig_i, r, sorted_index in ordered:
        out.append(
            {
                "original_index": orig_i,
                "difficulty": r.difficulty,
                "correct": r.correct,
                "sorted_index": sorted_index,
                "prediction": r.prediction,
                "answer_key": r.answer_key,
            }
        )
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-only 评估 + 难度排序（150×2 题，输出难度与排序 index）"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("reports", "llm_baseline_difficulty", run_id)
    ensure_dir(output_dir)

    llm = GPTChat(model_name=args.model)

    datasets = [
        "medmcqa_physio_150_build",
        "medmcqa_pharma_150_build",
    ]

    for dataset in datasets:
        print(f"[dataset] {dataset}", flush=True)
        results = eval_dataset_with_difficulty(llm, dataset, output_dir)
        sorted_rows = compute_sorted_order(results)

        # 按 sorted_index 排序后的完整列表（含题目难度、是否正确、排序 index）
        out_path = os.path.join(output_dir, f"{dataset}_sorted.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in sorted_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 仅保留：original_index, difficulty, correct, sorted_index（最终要求的排序 index 输出）
        summary_path = os.path.join(output_dir, f"{dataset}_sorted_index.jsonl")
        with open(summary_path, "w", encoding="utf-8") as f:
            for row in sorted_rows:
                f.write(
                    json.dumps(
                        {
                            "original_index": row["original_index"],
                            "difficulty": row["difficulty"],
                            "correct": row["correct"],
                            "sorted_index": row["sorted_index"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        n_correct = sum(1 for r in results if r.correct)
        print(f"  accuracy: {n_correct}/{len(results)} = {n_correct/len(results):.4f}")
        print(f"  [ok] {out_path}")
        print(f"  [ok] {summary_path}")

    print(f"\n[ok] all under {output_dir}")


if __name__ == "__main__":
    main()
