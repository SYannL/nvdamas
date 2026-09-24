# -*- coding: utf-8 -*-
"""
Generic insight summarization via meta-LLM: the LLM first decides what should be
summarized for similar future tasks, then generates the prompt for the actual summary.

No dataset-specific branches. Works universally across ALFWorld, MedMCQA, PDDL, etc.
"""

# ---------------------------------------------------------------------------
# Stage 1: Meta-prompt — LLM decides what aspects to summarize
# ---------------------------------------------------------------------------

META_SYSTEM = (
    "You analyze completed tasks and their solution trajectories. Your job is to decide "
    "what aspects of this solution would be most valuable to summarize for helping an agent "
    "solve similar tasks in the future."
)

META_USER_TASK = (
    "Given the task and its successful trajectory below, determine what should be summarized "
    "for similar future tasks. Consider:\n"
    "- What type of task is this (e.g., embodied navigation, QA, planning, verification)?\n"
    "- What would an agent need to remember: procedural steps, preconditions, failure patterns, "
    "domain facts, reasoning patterns, or something else?\n"
    "- What format or focus would make the summary actionable?\n\n"
    "Output 2-4 concrete instructions (one per line) describing what the summary should capture. "
    "Be specific to this task type. Example formats:\n"
    "- 'Focus on action order and preconditions (e.g., go to X before take X)'\n"
    "- 'Extract the key domain facts that support the answer'\n"
    "- 'Capture what to do when observations indicate failure (e.g., Nothing happens)'\n\n"
    "Task:\n{task}\n\nTrajectory:\n{trajectory}"
)

META_USER_CHAIN = (
    "Given the action/reasoning steps below from a completed solution, determine what should "
    "be summarized for similar future tasks. Output 2-4 concrete instructions (one per line) "
    "describing what the summary should capture. Be specific to the task type implied by the steps.\n\n"
    "Steps:\n{steps}"
)


# ---------------------------------------------------------------------------
# Stage 2: Summary prompt — uses meta-guidance to generate the insight
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = (
    "You write concise, actionable summaries of task solutions. Follow the summarization "
    "guidance you are given precisely."
)

SUMMARY_USER_TASK = (
    "Summarize the solution below following these instructions:\n\n{guidance}\n\n"
    "Task:\n{task}\n\nTrajectory:\n{trajectory}\n\n"
    "Write a 2-4 sentence summary."
)

SUMMARY_USER_CHAIN = (
    "Summarize the steps below following these instructions:\n\n{guidance}\n\n"
    "Steps:\n{steps}\n\n"
    "Write a 2-4 sentence summary."
)
