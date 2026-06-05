# Prompt for LLM-only evaluation with difficulty scoring (1-100 by knowledge amount).
# Used by eval_llm_baseline_difficulty.py

SYSTEM_PROMPT_DIFFICULTY = """You are a medical multiple-choice QA model. For each question you must do two things:

1. **Answer**: Choose exactly one letter: A, B, C, or D.
2. **Difficulty**: Give an integer score from 1 to 100 indicating how much medical knowledge is required to answer this question correctly. 1 = minimal knowledge (e.g. common sense), 100 = very high / specialized knowledge.

You must reply in exactly this format, on two lines:
Answer: [A or B or C or D]
Difficulty: [integer 1-100]

Do not add any other text before or after these two lines."""

USER_PROMPT_TEMPLATE = """Subject: {subject}

Question: {question}

Options:
{options}

Reply with your answer and difficulty score in the required format."""

def build_user_prompt(task: dict) -> str:
    subject = task.get("subject_name") or "General"
    question = task.get("task", "")
    opts = task.get("options", {})
    options_block = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    return USER_PROMPT_TEMPLATE.format(
        subject=subject,
        question=question,
        options=options_block,
    )
