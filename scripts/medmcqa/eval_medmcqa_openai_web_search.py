#!/usr/bin/env python
"""
Evaluate MedMCQA with OpenAI Responses API + built-in web_search tool
on medmcqa_physio_30 / medmcqa_pharma_30 using a forced-search setting,
and compare against the existing LLM-only baseline.

Note:
- Replace the hard-coded API key in this file with your own key before running.
"""

import os
import sys
import json
import time
import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

# Ensure we can run directly from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.envs import get_task  # noqa: E402
from mas.llm import GPTChat, Message  # noqa: E402


# 回答阶段沿用 LLM-only baseline 的 system prompt
ANSWER_SYSTEM_PROMPT = (
    "You are a medical multiple-choice QA model. "
    "Answer with only the single letter A, B, C, or D."
)

# 搜索阶段使用 4o-mini + web_search 工具
SEARCH_MODEL = "gpt-4o-mini"
# 回答阶段使用与 baseline 一致的 3.5 模型
ANSWER_MODEL = "gpt-3.5-turbo-0125"


def build_question_prompt(task: Dict[str, Any]) -> str:
    subject = task.get("subject_name")
    subject_line = f"Subject: {subject}\n" if subject else ""
    q = task["task"]
    options = task["options"]
    options_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{subject_line}Question: {q}\nOptions:\n{options_block}"


def extract_choice(text: str) -> Optional[str]:
    if not text:
        return None
    for ch in text:
        if ch in {"A", "B", "C", "D"}:
            return ch
    # Fallback: scan line by line
    for line in text.splitlines():
        line = line.strip()
        if line in {"A", "B", "C", "D"}:
            return line
    return None


async def search_evidence(
    client: AsyncOpenAI,
    task: Dict[str, Any],
) -> str:
    """
    Use gpt-4o-mini + web_search tool to obtain web evidence for a single
    question and summarize it. This only returns evidence, not the final choice.
    """
    prompt = build_question_prompt(task)
    search_system_prompt = (
        "You are a medical research assistant. "
        "Use the web_search tool to gather up-to-date factual information relevant to the question. "
        "Then summarize the key evidence in 3-6 short bullet points. "
        "Do NOT choose an option; only provide evidence."
    )

    response = await client.responses.create(
        model=SEARCH_MODEL,
        tools=[{"type": "web_search"}],
        input=f"{search_system_prompt}\n\n{prompt}",
    )

    text = getattr(response, "output_text", None)
    if not text:
        text = str(response)
    return text


def answer_with_35(
    llm_35: GPTChat,
    task: Dict[str, Any],
    evidence: str,
) -> str:
    """
    Use the same 3.5 model and system prompt as the LLM-only baseline,
    but augment the user prompt with external evidence from web search.
    """
    q = build_question_prompt(task)
    user_prompt = (
        f"{q}\n\n"
        f"Here is some external evidence from web search:\n"
        f"{evidence}\n\n"
        f"Using the above information, choose the single best option.\n"
        f"Answer:"
    )
    resp = llm_35(
        [Message("system", ANSWER_SYSTEM_PROMPT), Message("user", user_prompt)],
        temperature=0,
    )
    choice = extract_choice(resp)
    return choice or ""


async def eval_dataset(
    dataset: str,
    search_client: AsyncOpenAI,
    llm_35: GPTChat,
    output_dir: str,
) -> None:
    tasks: List[Dict[str, Any]] = get_task(dataset)
    if not tasks:
        print(f"[{dataset}] no tasks found")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset}.jsonl")

    correct = 0
    records: List[Dict[str, Any]] = []

    start = time.time()
    for idx, task in enumerate(tasks):
        print(f"[{dataset}] {idx+1}/{len(tasks)}", flush=True)
        try:
            evidence = await search_evidence(search_client, task)
            pred = answer_with_35(llm_35, task, evidence=evidence)
        except Exception as exc:
            print(f"  [error] request failed: {exc}")
            pred = ""

        answer_key = task.get("answer_key")
        reward = 1.0 if pred == answer_key else 0.0
        if reward > 0:
            correct += 1

        records.append(
            {
                "dataset": dataset,
                "index": idx,
                "question": task["task"],
                "options": task["options"],
                "answer_key": answer_key,
                "prediction": pred,
                "reward": reward,
            }
        )

    acc = correct / len(tasks)
    elapsed = time.time() - start
    print(f"[{dataset}] accuracy = {acc:.4f} ({correct}/{len(tasks)}), time = {elapsed:.1f}s")

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{dataset}] saved to {out_path}")


async def main() -> None:
    # TODO: Replace this with your own key
    api_key = "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"
    if not api_key or api_key == "OPENAI_API_KEY_PLACEHOLDER":
        raise RuntimeError("Please replace api_key in this script with your own OpenAI API Key.")

    # Search client: 4o-mini + web_search
    search_client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key=api_key,
    )
    # Answer client: reuse existing GPTChat wrapper with ANSWER_MODEL
    llm_35 = GPTChat(model_name=ANSWER_MODEL)

    output_dir = os.path.join("reports", "openai_web_search")
    # Example: run on selected datasets; adjust as needed.
    for dataset in ["medmcqa_pharma_150_build"]:
        await eval_dataset(dataset, search_client, llm_35, output_dir)


if __name__ == "__main__":
    asyncio.run(main())

