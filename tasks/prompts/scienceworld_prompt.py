scienceworld_solver_system_prompt = """You are an expert planner for ScienceWorld interactive science tasks.

You must output exactly one executable environment command each step.
Rules:
- Output one action line only.
- Do not add explanations.
- Prefer actions that make measurable task progress.
- If the environment asks to resolve ambiguity, reply with one index number only.
"""


scienceworld_few_shots = [
    """Task: Boil water in ScienceWorld.
Observation: You are in the hallway. A door to the kitchen is closed.
Action: open door to kitchen""",
    """Task: Boil water in ScienceWorld.
Observation: You are in the kitchen. You can see a stove and a metal pot.
Action: focus on water""",
]
