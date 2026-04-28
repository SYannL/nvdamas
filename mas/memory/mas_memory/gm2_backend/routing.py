from __future__ import annotations

from ...common import MASMessage
from .overlay import GM2OnlineEpisodeBuilder
from .types import GM2PromptPayload


def _message_to_execution_pattern(message: MASMessage) -> str:
    description = str(message.task_description or "").strip()
    trajectory = str(message.task_trajectory or "").strip()
    blocks: list[str] = []
    if description:
        blocks.append("### Task description:\n" + description)
    if trajectory:
        blocks.append("### Detailed trajectory:\n" + trajectory)
    return "\n\n".join(blocks).strip()


def build_gm2_prompt_payload(
    *,
    successful_messages: list[MASMessage],
    failed_messages: list[MASMessage],
    overlay_builder: GM2OnlineEpisodeBuilder | None,
    stored_insights: list[str],
) -> GM2PromptPayload:
    execution_patterns = [
        _message_to_execution_pattern(message)
        for message in successful_messages
        if str(message.task_trajectory or "").strip()
    ]
    planner_notes = overlay_builder.planner_notes() if overlay_builder is not None else []
    repair_hints: list[str] = []
    if failed_messages:
        repair_hints.append(
            "[GM2Repair] Similar failed cases exist; avoid repeating actions that produced no progress."
        )
    return GM2PromptPayload(
        reference_cases=[],
        execution_patterns=execution_patterns,
        insights=list(stored_insights),
        planner_notes=planner_notes,
        action_constraints=[],
        repair_hints=repair_hints,
    )
