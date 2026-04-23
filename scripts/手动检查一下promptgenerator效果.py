#!/usr/bin/env python
"""
Standalone test for the meta-LLM two-stage insight summarization.
No ALFWorld or other heavy deps needed—just the LLM calls.
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mas.llm import GPTChat, Message
from mas.memory.mas_memory.insight_prompts import (
    META_SYSTEM,
    META_USER_TASK,
    META_USER_CHAIN,
    SUMMARY_SYSTEM,
    SUMMARY_USER_TASK,
    SUMMARY_USER_CHAIN,
)


def run_two_stage_task_summary(llm_model, task: str, trajectory: str) -> tuple[str, str]:
    """Stage 1: meta decides what to summarize. Stage 2: generate summary."""
    meta_prompt = META_USER_TASK.format(task=task, trajectory=trajectory)
    guidance = llm_model(
        [Message("system", META_SYSTEM), Message("user", meta_prompt)],
        temperature=0,
        max_tokens=256,
    ).strip()

    summary_prompt = SUMMARY_USER_TASK.format(
        guidance=guidance, task=task, trajectory=trajectory
    )
    summary = llm_model(
        [Message("system", SUMMARY_SYSTEM), Message("user", summary_prompt)],
        temperature=0,
        max_tokens=256,
    ).strip()

    return guidance, summary


def run_two_stage_chain_summary(llm_model, steps: str) -> tuple[str, str]:
    """Stage 1: meta decides what to summarize. Stage 2: generate summary."""
    meta_prompt = META_USER_CHAIN.format(steps=steps)
    guidance = llm_model(
        [Message("system", META_SYSTEM), Message("user", meta_prompt)],
        temperature=0,
        max_tokens=256,
    ).strip()

    summary_prompt = SUMMARY_USER_CHAIN.format(guidance=guidance, steps=steps)
    summary = llm_model(
        [Message("system", SUMMARY_SYSTEM), Message("user", summary_prompt)],
        temperature=0,
        max_tokens=256,
    ).strip()

    return guidance, summary


def main():
    model = "gpt-4o-mini"
    llm = GPTChat(model_name=model)

    print("=" * 70)
    print("Test 1: ALFWorld-style (embodied task)")
    print("=" * 70)

    task1 = "Examine the alarmclock with the desklamp."
    traj1 = (
        "> go to alarmclock 1\n"
        "> take alarmclock 1\n"
        "> go to desklamp 1\n"
        "> turn on desklamp 1\n"
        "> look at alarmclock 1\n"
        "OK."
    )

    g1, s1 = run_two_stage_task_summary(llm, task1, traj1)
    print("\n[Stage 1 - Meta guidance]:")
    print(g1)
    print("\n[Stage 2 - Summary]:")
    print(s1)

    print("\n" + "=" * 70)
    print("Test 2: MedMCQA-style (knowledge QA)")
    print("=" * 70)

    task2 = (
        "A 45-year-old male presents with chest pain. "
        "Question: What is the most likely diagnosis?"
    )
    traj2 = (
        "> thought: Chest pain in a 45-year-old male suggests possible cardiac etiology.\n"
        "> search: acute coronary syndrome risk factors\n"
        "> thought: ST elevation and troponin elevation support STEMI.\n"
        "> answer: STEMI (ST-elevation myocardial infarction)"
    )

    g2, s2 = run_two_stage_task_summary(llm, task2, traj2)
    print("\n[Stage 1 - Meta guidance]:")
    print(g2)
    print("\n[Stage 2 - Summary]:")
    print(s2)

    print("\n" + "=" * 70)
    print("Test 3: Chain summary (steps only)")
    print("=" * 70)

    steps3 = (
        "1. go to cabinet 3\n"
        "2. open cabinet 3\n"
        "3. take bowl 1\n"
        "4. go to sinkbasin 1\n"
        "5. put bowl 1 in sinkbasin 1\n"
        "6. clean bowl 1 with soapbar 1\n"
        "7. take bowl 1\n"
        "8. go to shelf 2\n"
        "9. put bowl 1 in shelf 2"
    )

    g3, s3 = run_two_stage_chain_summary(llm, steps3)
    print("\n[Stage 1 - Meta guidance]:")
    print(g3)
    print("\n[Stage 2 - Summary]:")
    print(s3)

    print("\n" + "=" * 70)
    print("Done. Meta-LLM adapts summarization focus per task type.")
    print("=" * 70)


if __name__ == "__main__":
    main()
