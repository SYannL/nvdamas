from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .graph_types import CanonicalAction, Domain, StateSummary
from .pddl_adapter import PDDLAdapter, _normalize


class BfclAdapter(PDDLAdapter):
    """GraphMemory adapter for BFCL multi-turn tool episodes (Exec / FinishTurn strings)."""

    domain_name = "bfcl_mt"

    def canonicalize_action(self, raw_action: str) -> CanonicalAction:
        base = super().canonicalize_action(raw_action)
        return CanonicalAction(Domain.BFCL, base.family, base.verb, base.slots, surface_form=base.surface_form)

    def derive_scene_id(self, game_name: str | None = None, history_path: str | None = None) -> str:
        if game_name:
            return f"bfcl_mt:{_normalize(game_name)}"
        if history_path:
            stem = Path(history_path).stem
            match = re.search(r"history_bfcl_mt_([^_]+)_", stem)
            if match:
                return f"bfcl_mt:{_normalize(match.group(1))}"
        return "bfcl_mt:unknown"

    def infer_task_family(self, goal: str, game_name: str | None = None) -> str:
        token = _normalize(game_name or "") or "task"
        return f"bfcl_mt:{token}"

    def build_state(
        self,
        *,
        scene_id: str,
        observation: str,
        admissible_commands: list[str] | tuple[str, ...] = (),
        goal: str = "",
        goal_literals: list[str] | None = None,
        current_literals: list[str] | None = None,
    ) -> StateSummary:
        st = super().build_state(
            scene_id=scene_id,
            observation=observation,
            admissible_commands=admissible_commands,
            goal=goal,
            goal_literals=goal_literals,
            current_literals=current_literals,
        )
        return replace(st, domain=Domain.BFCL)

    def _state_from_record(self, record: dict[str, Any], *, scene_id: str, goal: str, goal_literals: list[str]) -> StateSummary:
        st = super()._state_from_record(record, scene_id=scene_id, goal=goal, goal_literals=goal_literals)
        return replace(st, domain=Domain.BFCL)
