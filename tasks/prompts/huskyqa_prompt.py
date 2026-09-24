huskyqa_few_shots = [
    """Question: If a plane can carry 200 passengers and needs to transport 1,000 people, how many full flights are needed?
Thought 1: Each full flight carries 200 passengers.
Thought 2: 1,000 / 200 = 5, so 5 full flights are needed.
Action 1: Finish[5 full flights]""",
    """Question: If the total population is 2,400,000 and it is split equally into 3 regions, how many people are in each region?
Thought 1: Split the total equally across 3 regions.
Thought 2: 2,400,000 / 3 = 800,000.
Action 1: Finish[800,000 people]""",
]

huskyqa_solver_system_prompt = """
Solve a quantitative question answering task. You may use short Thought steps to reason.
Always respond with the Action format: Finish[answer].
The answer should be concise and include units if present in the question (e.g., people, flights, percent, dollars).
Use only the numbers explicitly stated in the question; do not introduce outside estimates or facts.
"""
