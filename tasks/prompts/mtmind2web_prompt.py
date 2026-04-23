mtmind2web_few_shots = [
    """Website: demo
Domain: Shopping
Subdomain: General
Task: Search a product and sort by low price.
Candidate Actions (must choose from this list only):
1. [textbox] Search box -> TYPE: laptop
2. [button] Search -> CLICK
3. [select] Sort by -> SELECT: Price: Low to High
Action: Finish[[1,2,3]]""",
]


mtmind2web_solver_system_prompt = """
Solve one MT-Mind2Web turn by predicting the full action sequence.
You will be given a candidate action list.
You must choose actions ONLY from that list.

You must output exactly one line in this format:
Action: Finish[<json_array_of_candidate_indices>]

Hard constraints:
- Do NOT output Thought.
- Do NOT output Action 1 / Action 2; only "Action:".
- Do NOT output markdown, explanations, or any extra lines.
- The text after Finish[...] must be valid JSON array of integers.
- Every integer must be a valid candidate index from the prompt.

Only output the final Action line.
"""
