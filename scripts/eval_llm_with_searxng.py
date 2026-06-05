#!/usr/bin/env python
import os
import sys
import json
from typing import Any, Dict, List

# Ensure this script can be run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mas.llm import GPTChat, Message
from tasks.envs import get_task
from tasks.envs.utils import StructuredSearchParams, SearxngSearchClient


SYSTEM_FOR_SEARCH = """You are a professional medical information retrieval assistant.
Given the following multiple-choice question, produce a JSON object for Web search with fields:
- keywords: core search keywords in English (short, ideally 2-6 words)
- constraints: conditions to exclude or constrain results (may be empty)
- time_range: time range (e.g. "past_5_years" or "2015-2026", may be empty)
- source_type: preferred source type (e.g. "textbook", "guideline", "wikipedia", may be empty)
- reformulated_queries: 3 different but related English search queries to cover diverse phrasings

Strict requirements:
- Output exactly one JSON object and nothing else.
- All strings in the JSON must be valid JSON strings (double-quoted)."""


SYSTEM_FOR_ANSWER = """You are an assistant for medical multiple-choice questions.
You will receive a question, four options, and 3 short summaries obtained from Web search.
Read these summaries carefully, combine them with your own knowledge, and choose the best option.

Requirements:
- Output only a single uppercase letter: A, B, C, or D.
- Do not output any explanation."""


def build_question_text(task: Dict[str, Any]) -> str:
    question = task["task"]
    options = task["options"]
    options_block = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question: {question}\nOptions:\n{options_block}"


def ask_for_search_params(llm: GPTChat, task: Dict[str, Any]) -> StructuredSearchParams:
    q_text = build_question_text(task)
    messages = [
        Message("system", SYSTEM_FOR_SEARCH),
        Message("user", q_text),
    ]
    resp = llm(messages, temperature=0, max_tokens=400)
    start = resp.find("{")
    end = resp.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return a valid JSON object: {resp}")
    json_str = resp[start : end + 1]
    data = json.loads(json_str)

    return StructuredSearchParams(
        keywords=data.get("keywords", ""),
        constraints=data.get("constraints", ""),
        time_range=data.get("time_range", ""),
        source_type=data.get("source_type", ""),
        reformulated_queries=data.get("reformulated_queries") or [],
    )


def ask_for_answer(llm: GPTChat, task: Dict[str, Any], search_output: Dict[str, Any]) -> str:
    q_text = build_question_text(task)
    snippets = search_output.get("top_results", [])
    snippet_block_lines: List[str] = []
    for i, item in enumerate(snippets, start=1):
        title = item.get("title") or ""
        url = item.get("url") or ""
        snippet = item.get("snippet") or ""
        snippet_block_lines.append(
            f"[Result {i}]\nTitle: {title}\nURL: {url}\nSnippet: {snippet}\n"
        )
    snippet_block = "\n".join(snippet_block_lines) if snippets else "No web results."

    user_prompt = (
        f"{q_text}\n\n"
        f"=== Web Search Results (Top 3, already reranked) ===\n"
        f"{snippet_block}\n\n"
        f"Based on the question and the search results above, choose the single best option and output only A/B/C/D:"
    )

    messages = [
        Message("system", SYSTEM_FOR_ANSWER),
        Message("user", user_prompt),
    ]
    resp = llm(messages, temperature=0, max_tokens=10)
    for ch in resp:
        if ch in {"A", "B", "C", "D"}:
            return ch
    return ""


def eval_dataset_with_search(
    dataset: str,
    model_name: str = "gpt-4o-mini",
) -> None:
    llm = GPTChat(model_name=model_name)
    tasks = get_task(dataset)
    if not tasks:
        print(f"No tasks for dataset={dataset}")
        return

    search_client = SearxngSearchClient()
    correct = 0
    records: List[Dict[str, Any]] = []

    for idx, task in enumerate(tasks):
        print(f"[{dataset}] {idx+1}/{len(tasks)}", flush=True)
        answer_key = task.get("answer_key")

        try:
            params = ask_for_search_params(llm, task)
        except Exception as e:
            print(f"  [warn] search params error: {e}")
            pred = ""
        else:
            search_output = search_client.structured_search(params, top_k=3)
            if search_output.get("status") == "OK":
                pred = ask_for_answer(llm, task, search_output)
            else:
                print("  [warn] search returned I_NEED_MORE_INFO")
                pred = ""

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
    print(f"[{dataset}] accuracy = {acc:.4f} ({correct}/{len(tasks)})")

    out_path = f"reports/llm_with_searxng_{dataset}.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] saved to {out_path}")


if __name__ == "__main__":
    # Quick smoke test on small subsets
    eval_dataset_with_search("medmcqa_physio_3", model_name="gpt-4o-mini")
    eval_dataset_with_search("medmcqa_pharma_3", model_name="gpt-4o-mini")

