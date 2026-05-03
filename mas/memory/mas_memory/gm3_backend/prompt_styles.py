from __future__ import annotations

import re
from typing import Any


class BasePromptStyle:
    """Dataset-facing language and transfer boundary for GM3 prompt injection."""

    name = "alfworld"

    def phase_items(self, renderer: Any, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        progress = str(getattr(query, "progress_state", "") or "")
        held = int(getattr(query, "held_relevant_count", 0) or 0)
        visible = bool(getattr(query, "goal_object_matches_visible", False))
        items: list[str] = []
        if held > 0:
            if tool and progress in {"carry_target", "process_target"}:
                items.append(
                    f"held target_object={target}; next phase is process with tool={tool} "
                    f"if not already processed, otherwise deliver to destination={destination}."
                )
            else:
                items.append(f"held target_object={target}; next phase is delivery to destination={destination}, not more source search.")
        elif visible:
            items.append(f"target_object={target} is visible; take the matching target before broad search.")
        elif progress.startswith("search"):
            items.append(f"search target_object={target}; if graph priority gives an admissible queue, execute the next queued action.")
        return items

    def keep_global_text(self, renderer: Any, *, text: str, norm_text: str, task_family: str) -> bool:
        return True

    def keep_local_artifact_text(
        self,
        renderer: Any,
        *,
        domain: str,
        task_family: str,
        text: str,
        norm_text: str,
    ) -> bool:
        return True

    def phase_label(self, renderer: Any, *, query: Any, macro: str) -> str:
        labels = {
            "process_held_target": "process held target",
            "deliver_held_target": "deliver held target",
            "acquire_visible_target": "acquire visible target",
            "search_additional_target": "search additional target",
            "search_target": "search target",
            "continue_task": "continue task",
        }
        return labels.get(macro, macro.replace("_", " "))

    def state_summary(
        self,
        renderer: Any,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        parts = [
            f"target={target or 'unknown'}",
            f"tool={tool or 'none'}",
            f"destination={destination or 'unknown'}",
        ]
        if held:
            parts.append("held=" + ", ".join(renderer._gm3_base(x) for x in held[:2] if str(x).strip()))
        elif visible:
            visible_bases = [renderer._gm3_base(x) for x in visible[:3] if str(x).strip()]
            if visible_bases:
                parts.append("visible=" + ", ".join(visible_bases))
        if exhausted:
            counts = renderer._gm3_exhausted_base_counts(exhausted)
            if counts:
                top = ", ".join(
                    f"{base}x{count}" for base, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
                )
                parts.append("searched=" + top)
        return "; ".join(part for part in parts if part)

    def next_priority_line(
        self,
        renderer: Any,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = renderer._gm3_base(str(goal_roles.get("object", "") or ""))
        tool = renderer._gm3_base(str(goal_roles.get("tool", "") or ""))
        destination = renderer._gm3_base(str(goal_roles.get("destination", "") or ""))
        progress = str(getattr(query, "progress_state", "") or "")
        held_count = int(getattr(query, "held_relevant_count", 0) or 0)
        visible_match = bool(getattr(query, "goal_object_matches_visible", False))

        if held_count > 0:
            actions: list[str] = []
            if tool and progress in {"carry_target", "process_target"}:
                actions = renderer._gm3_tool_priority_actions(target=target, tool=tool, admissible=admissible)
            if not actions and destination:
                actions = renderer._gm3_destination_priority_actions(target=target, destination=destination, admissible=admissible)
            if actions:
                return actions[0]
            return "use the admissible process or delivery action for the held target; ignore source search."

        if visible_match:
            actions = renderer._gm3_take_priority_actions(target=target, admissible=admissible)
            if actions:
                return actions[0]
            return "take the visible matching target before broad search."

        for item in priority_items[:2]:
            text = str(item or "").strip()
            if text:
                return text
        if progress.startswith("search"):
            return "search for the target using current admissible actions; avoid repeating exhausted source types."
        return "follow the current observation and admissible actions."


class ALFWorldPromptStyle(BasePromptStyle):
    name = "alfworld"


class PDDLPromptStyle(BasePromptStyle):
    name = "pddl"

    def phase_items(self, renderer: Any, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        progress = str(getattr(query, "progress_state", "") or "")
        return [
            f"PDDL graph stage={progress or 'unknown'}; prefer current valid actions that previous local/global traces linked to goal-literal progress."
        ]

    def phase_label(self, renderer: Any, *, query: Any, macro: str) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        labels = {
            "need_plan": "choose valid operator",
            "goal_progress": "advance goal literals",
            "invalid_action": "recover from invalid operator",
            "done": "finished",
        }
        return labels.get(progress, progress.replace("_", " ") or "solve planning goal")

    def state_summary(
        self,
        renderer: Any,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        goal = renderer._gm3_clean(str(getattr(query, "goal", "") or "")).strip()
        progress = str(getattr(query, "progress_state", "") or "unknown")
        parts = [f"goal={renderer._gm3_shorten(goal, 120) or 'unknown'}", f"stage={progress}"]
        if visible:
            parts.append("state=" + "; ".join(renderer._gm3_base(x) for x in visible[:3] if str(x).strip()))
        if exhausted:
            parts.append("tried=" + ", ".join(renderer._gm3_base(x) for x in exhausted[:3] if str(x).strip()))
        return "; ".join(parts)

    def next_priority_line(
        self,
        renderer: Any,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        for item in priority_items[:2]:
            text = str(item or "").strip()
            if text:
                return text
        if admissible:
            return "choose a currently valid operator that advances unsatisfied goal literals; do not invent actions."
        return "continue planning from current predicates and goal literals."


class FeverPromptStyle(BasePromptStyle):
    name = "fever"

    def phase_items(self, renderer: Any, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        progress = str(getattr(query, "progress_state", "") or "")
        return [
            f"FEVER graph stage={progress or 'unknown'}; use remembered Search -> Lookup -> Finish workflow/failure evidence as prompt guidance only."
        ]

    def keep_global_text(self, renderer: Any, *, text: str, norm_text: str, task_family: str) -> bool:
        return _is_fever_transferable_hint(norm_text)

    def keep_local_artifact_text(
        self,
        renderer: Any,
        *,
        domain: str,
        task_family: str,
        text: str,
        norm_text: str,
    ) -> bool:
        return _is_fever_transferable_hint(norm_text)

    def phase_label(self, renderer: Any, *, query: Any, macro: str) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        labels = {
            "need_search": "choose focused evidence search",
            "need_lookup_or_finish": "inspect evidence or decide label",
            "ready_finish": "decide FEVER label from evidence",
            "done": "finished",
            "search_failed": "recover from failed evidence search",
            "invalid_action": "recover from invalid FEVER action",
        }
        return labels.get(progress, progress.replace("_", " ") or "verify claim")

    def state_summary(
        self,
        renderer: Any,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        goal = renderer._gm3_clean(str(getattr(query, "goal", "") or "")).removeprefix("Verify claim:").strip()
        progress = str(getattr(query, "progress_state", "") or "unknown")
        evidence = [renderer._gm3_base(x) for x in visible[:2] if str(x).strip()]
        parts = [f"claim={renderer._gm3_shorten(goal, 110) or 'unknown'}", f"stage={progress}"]
        if evidence:
            parts.append("evidence=" + "; ".join(evidence))
        if exhausted:
            parts.append("searched=" + ", ".join(renderer._gm3_base(x) for x in exhausted[:2] if str(x).strip()))
        return "; ".join(parts)

    def next_priority_line(
        self,
        renderer: Any,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        lookup_hint = _fever_lookup_hint(renderer, query)
        search_hint = _fever_search_hint(renderer, query, admissible)
        if progress == "need_search":
            if search_hint:
                return f"start with {search_hint}; use Search on the claim's primary entity before deciding the label."
            return "choose a focused Search[...] query from the claim; do not infer the label before evidence."
        if progress in {"need_lookup_or_finish", "ready_finish"}:
            if lookup_hint:
                return f"try Lookup[{lookup_hint}] if the current evidence does not directly settle the claim; finish only after evidence justifies the label."
            return "use Lookup[...] when evidence is insufficient; Finish[...] only when the evidence supports, refutes, or is missing."
        if progress in {"search_failed", "invalid_action"}:
            return "try a narrower evidence query or finish NOT ENOUGH INFO only after evidence search fails."
        return "verify the claim using Search, Lookup, then an evidence-grounded Finish label."


def _fever_search_hint(renderer: Any, query: Any, admissible: list[str]) -> str:
    goal_roles = getattr(query, "goal_roles", {}) or {}
    anchor = renderer._gm3_base(str(goal_roles.get("object", "") or ""))
    if not anchor:
        return ""
    for action in admissible:
        if str(action).lower().startswith("search[") and anchor in renderer._gm3_base(action):
            return str(action)
    return f"Search[{anchor.replace('_', ' ').title()}]"


def _fever_lookup_hint(renderer: Any, query: Any) -> str:
    goal = renderer._gm3_clean(str(getattr(query, "goal", "") or "")).removeprefix("Verify claim:").strip()
    if not goal:
        return ""
    goal_roles = getattr(query, "goal_roles", {}) or {}
    anchor = renderer._gm3_base(str(goal_roles.get("object", "") or "")).replace("_", " ")
    text = renderer._gm3_base(goal).replace("_", " ")
    if anchor:
        text = text.replace(anchor, " ")
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(
        r"\b(?:claim|has|have|had|is|are|was|were|a|an|the|to|of|in|on|by|for|with|and|or|no|not)\b",
        " ",
        text,
    )
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words[:5]).strip()


def _is_fever_label_only_hint(norm_text: str) -> bool:
    text = re.sub(r"[^a-z0-9]+", " ", str(norm_text or "").lower()).strip()
    if "fever" not in text:
        return False
    padded = f" {text} "
    has_finish = (
        " finish supports " in padded
        or " finish refutes " in padded
        or " finish not enough info " in padded
    )
    has_evidence_action = (
        " via search " in padded
        or " via lookup " in padded
        or " action search " in padded
        or " action lookup " in padded
        or " prefer search " in padded
        or " prefer lookup " in padded
    )
    return bool(has_finish and not has_evidence_action)


def _is_fever_transferable_hint(norm_text: str) -> bool:
    text = re.sub(r"[^a-z0-9]+", " ", str(norm_text or "").lower()).strip()
    if "fever" not in text:
        return True
    if _is_fever_label_only_hint(text):
        return False
    padded = f" {text} "
    if any(
        marker in padded
        for marker in (
            " fever evidence workflow ",
            " fever lookup workflow ",
            " fever recovery workflow ",
            " fever failure avoidance ",
            " evidence strategy ",
            " lookup strategy ",
            " no results recovery ",
            " premature finish ",
        )
    ):
        return True
    if " no results " in padded or " failure " in padded or " avoid " in padded or " reformulat" in text:
        return True
    if " current admissible grounding " in padded and (" search " in padded or " lookup " in padded):
        return True
    # Generic phase-only chunks like "need_lookup_or_finish -> done" add almost no
    # evidence value, and old entity-specific Search[...] examples bias the claim.
    return False


def prompt_style_for_env(env_name: str) -> BasePromptStyle:
    env = str(env_name or "").strip().lower()
    if env.startswith("fever"):
        return FeverPromptStyle()
    if env.startswith("pddl"):
        return PDDLPromptStyle()
    return ALFWorldPromptStyle()


def prompt_style_for_query(query: Any, task_family: str = "") -> BasePromptStyle:
    scene_id = str(getattr(query, "scene_id", "") or "").strip().lower()
    family = str(task_family or getattr(query, "task_family", "") or "").strip().lower()
    if scene_id.startswith("fever:") or family.startswith("fever"):
        return FeverPromptStyle()
    if scene_id.startswith("pddl:") or family.startswith("pddl"):
        return PDDLPromptStyle()
    return ALFWorldPromptStyle()
