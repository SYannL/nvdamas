medmcqa_few_shots = [
    """Question: Which vitamin is supplied from only animal source:
Options:
A. Vitamin C
B. Vitamin B7
C. Vitamin B12
D. Vitamin D
Thought 1: I should verify the source of Vitamin B12.
Action 1: Search[Vitamin B12]
Observation 1: • Vitamin B12 is found in animal-derived foods. • Plant-based diets typically require B12 supplementation.
Thought 2: The correct option is C.
Action 2: Finish[C]""",
]

medmcqa_solver_system_prompt = """
Solve a medical single-choice question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Search[topic], which runs a web search over the full question and returns 3-6 bullet points of relevant evidence.
(2) Lookup[keyword], which returns the same evidence as the last Search (use only to re-view prior evidence; the keyword is optional).
(3) Finish[option], which returns the answer and finishes the task. The option should be a single letter from A, B, C, or D.
Only use Search/Lookup when you are uncertain about the answer. If you are confident, answer directly with Finish.
"""

medmcqa_few_shots_no_search = [
    """Question: Which vitamin is supplied from only animal source:
Options:
A. Vitamin C
B. Vitamin B7
C. Vitamin B12
D. Vitamin D
Thought 1: I can answer directly based on domain knowledge.
Action 1: Finish[C]""",
]

medmcqa_solver_system_prompt_no_search = """
Solve a medical single-choice question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be one type:
(1) Finish[option], which returns the answer and finishes the task. The option should be a single letter from A, B, C, or D.
Do not use Search or Lookup actions.
"""

medmcqa_few_shots_force_search = []  # No few-shot: first Search is done by system; model only reasons and Finish (or second Search).

medmcqa_solver_system_prompt_force_search = """
Solve a medical single-choice question with Thought and Action steps.

**First Search already done for you**: The system has run one Search for the current question. In the task below you will see "Act 1: Search[...]" and "Obs 1: <evidence>". Do NOT output the first Search yourself.

Your turn: (1) Use Thought to summarize Obs 1 and reason toward the answer. (2) Then either Finish[option] (one letter A/B/C/D) if evidence is sufficient, or Search[query] for a second search if needed; after that, reason and Finish[option].
Valid Actions: Search[...], Finish[...]. Every response MUST start with "Thought:" or "Action:". Before Finish[option], use one Thought to summarize evidence, judge relevance, and state the chosen option.
Tool-use policy (web search):
- Use Search to search the full question; pass the current task's question text (or key part) as the query. Always use the question from the current "Your Turn" task, not any example.
- Use at most 2 Searches per task. Prefer 1 Search if the first result is sufficient; use a second Search only when the first result is unclear, contradictory, or irrelevant. For the second Search, combine the question and the specific confusion or missing point (e.g. "question part + what remains unclear").
- Do not use Lookup.
- If the retrieved evidence is contradictory or irrelevant after up to 2 Searches, default to internal knowledge and choose the best-supported option.
- Never output free text outside Thought/Action/Observation format.
Strict output rules:
- Every response MUST start with either "Thought:" or "Action:".
- Valid Actions are ONLY: Search[...], Finish[...]. Do not output any other text as an Action.
- After each Search, use at most ONE Thought to summarize the key evidence; after the last Search (or if skipping a second Search), then Finish[...].
Final decision step:
- Before the final Finish[option], use one Thought to explicitly:
  (a) summarize the external evidence you obtained from Search,
  (b) judge whether this evidence is reliable and relevant to the current question,
  (c) combine it with your own prior medical knowledge,
  and then decide the best option. If the external evidence conflicts with strong prior knowledge or looks off-topic, you may choose to ignore it.
"""

medmcqa_few_shots_smart_search = [
    """Question: Which vitamin is supplied from only animal source:
Options:
A. Vitamin C
B. Vitamin B7
C. Vitamin B12
D. Vitamin D
Thought 1: I know this is a high-confidence fact (B12 is animal-derived). No external lookup needed.
Action 1: Finish[C]""",
]

medmcqa_solver_system_prompt_smart_search = """
Solve a medical single-choice question answering task with interleaving Thought, Action, Observation steps.
Thought can reason about the current situation, and Action can be three types:
(1) Search[topic], which runs a web search over the full question and returns 3-6 bullet points of relevant evidence.
(2) Lookup[keyword], which returns the same evidence as the last Search (use only to re-view prior evidence; the keyword is optional).
(3) Finish[option], which returns the answer and finishes the task. The option should be a single letter from A, B, C, or D.

Decision Framework (strict):
1) Classify the question type and your confidence.
   - High confidence: directly answer with Finish.
   - Medium/low confidence: use Search to verify a key fact via web search.
2) Only Search when the external evidence is likely to change the answer.
3) Use at most 1 Search and optionally 1 Lookup per task. Lookup returns the same evidence as Search; if Search is irrelevant, stop and answer based on your knowledge.
4) Never output free text outside Thought/Action/Observation format.
5) Before the final Finish[option], use one Thought to:
   (a) summarize the external evidence you obtained from Search/Lookup,
   (b) judge whether this evidence is reliable and relevant,
   (c) combine it with your own prior medical knowledge,
   and then commit to the best option. If the external evidence looks noisy or contradicts strong prior knowledge, you may rely more on your own knowledge.
"""
