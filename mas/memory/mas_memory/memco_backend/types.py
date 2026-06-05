from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemCoEpisodeState:
    task_main: str
    task_description: str = ""
    recent_actions: list[str] = field(default_factory=list)
    recent_observations: list[str] = field(default_factory=list)
    step_count: int = 0


@dataclass
class MemCoPromptPayload:
    reference_cases: list[str] = field(default_factory=list)
    execution_patterns: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    planner_notes: list[str] = field(default_factory=list)
    action_constraints: list[str] = field(default_factory=list)
    repair_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "reference_cases": list(self.reference_cases),
            "execution_patterns": list(self.execution_patterns),
            "insights": list(self.insights),
            "planner_notes": list(self.planner_notes),
            "action_constraints": list(self.action_constraints),
            "repair_hints": list(self.repair_hints),
        }
