from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .graph_types import (
    ActionFamily,
    CanonicalAction,
    Domain,
    EpisodeRecord,
    EpisodeStep,
    StateSummary,
    StepFeedback,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


def _short_text(text: str, *, limit: int = 500) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())[:limit]


_SCIENCEWORLD_FAMILY_ALIASES = {
    "test_conductivity": "conductivity",
    "test_conductivity_of_unknown_substances": "conductivity",
    "measure_melting_point_known_substance": "melting_point",
    "measure_melting_point_unknown_substance": "melting_point",
    "inclined_plane_friction_named_surfaces": "friction",
    "inclined_plane_friction_unnamed_surfaces": "friction",
    "mendelian_genetics_known_plant": "genetics",
    "mendelian_genetics_unknown_plant": "genetics",
}


# ScienceWorld action verb families for natural-language science commands.
_NAVIGATE_VERBS = frozenset({"go", "move", "walk", "travel", "enter", "exit", "leave"})
_INSPECT_VERBS = frozenset({"look", "examine", "read", "check", "observe", "inspect", "measure", "focus", "describe"})
_MANIPULATE_VERBS = frozenset({
    "pick", "take", "grab", "get", "put", "place", "drop", "throw", "heat", "cool",
    "boil", "melt", "freeze", "burn", "pour", "mix", "combine", "use", "connect",
    "disconnect", "turn", "push", "pull", "open", "close", "activate", "deactivate",
    "eat", "drink", "cut", "break", "crush", "wash", "clean", "fill", "empty",
})
_CONTROL_VERBS = frozenset({"wait", "stop", "end", "finish", "start", "reset"})


def _action_family(verb: str) -> ActionFamily:
    v = verb.lower()
    if v in _NAVIGATE_VERBS:
        return ActionFamily.NAVIGATE
    if v in _INSPECT_VERBS:
        return ActionFamily.INSPECT
    if v in _MANIPULATE_VERBS:
        return ActionFamily.MANIPULATE
    if v in _CONTROL_VERBS:
        return ActionFamily.CONTROL
    return ActionFamily.OTHER


def _extract_room_objects(observation: str) -> list[str]:
    """Extract item/object tokens from a ScienceWorld observation string."""
    text = str(observation or "")
    # Find sequences that look like object descriptions: "a/an thermometer", "[object]", etc.
    found: list[str] = []
    for match in re.finditer(r"\b(?:a|an|the|some)\s+([a-z][a-z0-9 \-]{1,30}?)(?:\s*[,\.\(\)\[\]]|\s+(?:is|are|in|on|at|with|that|which)\b)", text.lower()):
        phrase = match.group(1).strip()
        token = _normalize(phrase)
        if token and len(token) > 2:
            found.append(token)
    # Also grab tokens from "You see: ..." style observations
    see_match = re.search(r"you see\s*:?\s*(.+?)(?:\n|$)", text, re.I)
    if see_match:
        for phrase in re.split(r",|;", see_match.group(1)):
            token = _normalize(phrase.strip())
            if token and len(token) > 2:
                found.append(token)
    return list(dict.fromkeys(found))[:20]


class ScienceWorldAdapter:
    """GraphMemory adapter for ScienceWorld interactive science tasks.

    Local memory is keyed by room (sw_scene_room), which clusters tasks that
    share the same starting environment layout.  Global memory aggregates
    transferable science procedure patterns across rooms.
    """

    domain_name = "scienceworld"

    # ---------- action canonicalization ----------

    def canonicalize_action(self, raw_action: str) -> CanonicalAction:
        surface = str(raw_action or "").strip()
        text = re.sub(r"\s+", " ", surface.lower().strip().strip("."))
        tokens = re.findall(r"[a-z0-9\-]+", text)
        if not tokens:
            return CanonicalAction(Domain.SCIENCEWORLD, ActionFamily.OTHER, "unknown", surface_form=surface)
        verb = tokens[0]
        family = _action_family(verb)

        # Strip leading particles / prepositions from remainder to get clean args.
        # e.g. "pick up thermometer" → remainder after stripping "up" = ["thermometer"]
        # e.g. "move to kitchen" → remainder after stripping "to" = ["kitchen"]
        _PARTICLES = {"up", "down", "out", "off", "away", "back", "the", "a", "an"}
        _NAV_PREPS = {"to", "into", "through", "toward"}
        remainder = tokens[1:]

        if family == ActionFamily.NAVIGATE:
            # Drop leading preposition ("to", "into", ...)
            while remainder and remainder[0] in _NAV_PREPS:
                remainder = remainder[1:]
            dest = _normalize(" ".join(remainder)[:40]) if remainder else ""
            slots = {"destination": dest} if dest else {}
            return CanonicalAction(Domain.SCIENCEWORLD, family, verb, slots, surface_form=surface)

        if family == ActionFamily.INSPECT:
            # Drop leading prepositions
            while remainder and remainder[0] in _NAV_PREPS:
                remainder = remainder[1:]
            target = _normalize(" ".join(remainder)[:40]) if remainder else ""
            slots = {"target": target} if target else {}
            return CanonicalAction(Domain.SCIENCEWORLD, family, verb, slots, surface_form=surface)

        if family == ActionFamily.MANIPULATE:
            # Drop leading particles ("pick up X" → drop "up")
            while remainder and remainder[0] in _PARTICLES:
                remainder = remainder[1:]
            # Split at preposition: "put X in/on Y"
            prep_idx = next(
                (i for i, t in enumerate(remainder) if t in {"in", "on", "into", "onto", "from", "with", "to"}),
                None,
            )
            if prep_idx is not None:
                obj = _normalize(" ".join(remainder[:prep_idx])[:40])
                loc = _normalize(" ".join(remainder[prep_idx + 1:])[:40])
                slots = {k: v for k, v in [("object", obj), ("location", loc)] if v}
            else:
                obj = _normalize(" ".join(remainder)[:40])
                slots = {"object": obj} if obj else {}
            return CanonicalAction(Domain.SCIENCEWORLD, family, verb, slots, surface_form=surface)

        # Control / other
        return CanonicalAction(Domain.SCIENCEWORLD, family, verb, surface_form=surface)

    # ---------- scene / layout id ----------

    def derive_scene_id(self, room: str | None = None, task_name: str | None = None, history_path: str | None = None) -> str:
        """Local memory key = scienceworld:{room} so same-room tasks share experience."""
        if room:
            return f"scienceworld:{_normalize(room)}"
        if task_name:
            return f"scienceworld:{_normalize(task_name)}"
        if history_path:
            stem = Path(history_path).stem
            match = re.search(r"history_scienceworld_([^_]+)_", stem)
            if match:
                return f"scienceworld:{_normalize(match.group(1))}"
        return "scienceworld:unknown"

    def derive_layout_id(
        self,
        room: str | None = None,
        variation_idx: Any = None,
        task_name: str | None = None,
        history_path: str | None = None,
    ) -> str:
        scene_id = self.derive_scene_id(room, task_name, history_path)
        suffix = str(variation_idx) if variation_idx is not None else "unknown"
        return f"{scene_id}:{suffix}"

    def infer_task_family(self, task_name: str | None = None, room: str | None = None) -> str:
        """Task family groups by sw_task_name for fine-grained procedure memory."""
        if task_name:
            return f"scienceworld:{_normalize(task_name)}"
        if room:
            return f"scienceworld:{_normalize(room)}"
        return "scienceworld:task"

    def scienceworld_family(self, task_config: dict | None, *, task_name: str = "") -> str:
        task_config = task_config or {}
        for key in ("scienceworld_family", "sw_family", "transfer_family"):
            value = str(task_config.get(key, "") or "").strip()
            if value:
                return _normalize(value)
        token = _normalize(task_name or str(task_config.get("sw_task_name", "") or ""))
        if token in _SCIENCEWORLD_FAMILY_ALIASES:
            return _SCIENCEWORLD_FAMILY_ALIASES[token]
        domain = str(task_config.get("scienceworld_domain", "") or task_config.get("sw_domain", "") or "").strip()
        if domain:
            return _normalize(domain)
        return ""

    def scienceworld_domain(self, task_config: dict | None, *, task_name: str = "") -> str:
        task_config = task_config or {}
        for key in ("scienceworld_domain", "sw_domain", "transfer_domain", "subset_group"):
            value = str(task_config.get(key, "") or "").strip()
            if value:
                return _normalize(value)
        if str(task_config.get("dataset_family", "") or "").strip().lower() == "scienceworld":
            return self.scienceworld_family(task_config, task_name=task_name)
        return ""

    # ---------- state / query building ----------

    def build_state(
        self,
        *,
        scene_id: str,
        observation: str,
        admissible_commands: list[str] | tuple[str, ...] = (),
        score: float = 0.0,
        has_positive_progress: bool = False,
        last_completed: bool = False,
        step_count: int = 0,
    ) -> StateSummary:
        objects = tuple(dict.fromkeys(_extract_room_objects(observation)))[:20]
        verbs = tuple(dict.fromkeys(self.canonicalize_action(cmd).verb for cmd in admissible_commands if str(cmd).strip()))
        progress = self._progress_state(
            score=score,
            has_positive_progress=has_positive_progress,
            last_completed=last_completed,
            step_count=step_count,
        )
        return StateSummary(
            domain=Domain.SCIENCEWORLD,
            scene_id=scene_id,
            visible_objects=objects,
            admissible_verbs=verbs,
            workflow_stage=progress,
            raw_observation=_short_text(observation, limit=500),
        )

    def build_query(self, *, env_ref: Any, task_config: dict, query_task: str, MemoryQuery: Any, CandidateType: Any):
        # Goal / task description
        goal = str(
            task_config.get("sw_task_desc", "")
            or getattr(env_ref, "task_description", "")
            or query_task
            or ""
        ).strip()
        if not goal:
            return None

        # Task identity
        sw_task = str(task_config.get("sw_task", "") or getattr(env_ref, "task_name", "") or "").strip()
        sw_task_name = str(task_config.get("sw_task_name", "") or _normalize(sw_task)).strip()
        room = str(task_config.get("sw_scene_room", "") or getattr(env_ref, "sw_scene_room", "") or "").strip()
        variation_idx = task_config.get("variation_idx", getattr(env_ref, "variation_idx", 0))

        sw_domain = self.scienceworld_domain(task_config, task_name=sw_task_name)
        sw_family = self.scienceworld_family(task_config, task_name=sw_task_name)
        scene_id = f"scienceworld:{sw_domain}" if sw_domain else self.derive_scene_id(room, sw_task_name)
        task_family = f"scienceworld:{sw_family}" if sw_family else self.infer_task_family(sw_task_name, room)

        # State from env_ref
        observation = str(getattr(env_ref, "last_observation", "") or "")
        score = float(getattr(env_ref, "last_score", 0.0) or 0.0)
        has_positive_progress = bool(getattr(env_ref, "has_positive_progress", False))
        last_completed = bool(getattr(env_ref, "last_completed", False))
        step_count = int(getattr(env_ref, "step_count", 0) or 0)
        admissible = [str(cmd) for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()]

        state = self.build_state(
            scene_id=scene_id,
            observation=observation,
            admissible_commands=admissible,
            score=score,
            has_positive_progress=has_positive_progress,
            last_completed=last_completed,
            step_count=step_count,
        )
        progress = self._progress_state(
            score=score,
            has_positive_progress=has_positive_progress,
            last_completed=last_completed,
            step_count=step_count,
        )

        desired_types = [CandidateType.PRECONDITION, CandidateType.WORKFLOW]
        # When no progress has been made after several steps, add failure hints.
        if step_count > 5 and not has_positive_progress:
            desired_types.append(CandidateType.FAILURE)

        keywords = tuple(
            dict.fromkeys(
                filter(None, [
                    f"domain=scienceworld",
                    f"room={_normalize(room)}" if room else "",
                    f"task={_normalize(sw_task_name)}" if sw_task_name else "",
                    f"scienceworld_domain={sw_domain}" if sw_domain else "",
                    f"scienceworld_family={sw_family}" if sw_family else "",
                    f"progress={progress}",
                    f"score={score:.2f}",
                    *[_normalize(obj) for obj in state.visible_objects[:8]],
                ])
            )
        )

        return MemoryQuery(
            goal=goal,
            scene_id=scene_id,
            current_stage=state.workflow_stage,
            progress_state=progress,
            task_family=task_family,
            goal_roles={
                "task": _normalize(sw_task_name) if sw_task_name else "science_task",
                "room": _normalize(room) if room else "unknown",
                "scienceworld_domain": sw_domain,
                "scienceworld_family": sw_family,
            },
            required_count=1,
            placed_relevant_count=1 if progress in {"near_completion", "goal_satisfied"} else 0,
            remaining_relevant_count=0 if progress == "goal_satisfied" else 1,
            destination_reached=progress == "goal_satisfied",
            goal_object_matches_visible=has_positive_progress,
            admissible_actions=tuple(self.canonicalize_action(cmd) for cmd in admissible),
            desired_types=tuple(desired_types),
            failure_label=self._failure_label_from_observation(observation, score=score, step_count=step_count),
            keywords=tuple(str(k) for k in keywords if str(k).strip()),
            belief={
                "sw_task": sw_task,
                "sw_task_name": sw_task_name,
                "room": room,
                "variation_idx": variation_idx,
                "score": score,
                "has_positive_progress": has_positive_progress,
                "last_completed": last_completed,
                "step_count": step_count,
                "observation": observation[:600],
            },
            dynamic_context={
                "visible_objects": list(state.visible_objects),
                "layout_id": self.derive_layout_id(room, variation_idx, sw_task_name),
                "task_config_env_name": str(task_config.get("env_name", "scienceworld")),
                "memco_domain": self.domain_name,
                "sw_task": sw_task,
                "sw_task_name": sw_task_name,
                "room": room,
                "scienceworld_domain": sw_domain,
                "scienceworld_family": sw_family,
            },
        )

    def episode_from_history(self, history_path: str, agent_id: str) -> EpisodeRecord:
        with open(history_path, "r", encoding="utf-8") as reader:
            payload = json.load(reader)
        task_config = payload.get("task_config") if isinstance(payload.get("task_config"), dict) else {}
        sw_task = str(payload.get("sw_task") or task_config.get("sw_task") or "unknown")
        sw_task_name = str(payload.get("sw_task_name") or task_config.get("sw_task_name") or _normalize(sw_task))
        room = str(payload.get("sw_scene_room") or task_config.get("sw_scene_room") or "unknown")
        variation_idx = payload.get("variation_idx", task_config.get("variation_idx", 0))
        goal = str(payload.get("sw_task_desc") or task_config.get("sw_task_desc") or payload.get("game_task") or "")
        sw_domain = self.scienceworld_domain(task_config, task_name=sw_task_name)
        sw_family = self.scienceworld_family(task_config, task_name=sw_task_name)
        scene_id = f"scienceworld:{sw_domain}" if sw_domain else self.derive_scene_id(room, sw_task_name, history_path)
        task_family = f"scienceworld:{sw_family}" if sw_family else self.infer_task_family(sw_task_name, room)
        task_id = f"{sw_task_name}-v{variation_idx}"
        history = [row for row in payload.get("history", []) if isinstance(row, dict)]
        episode = EpisodeRecord(
            agent_id=agent_id,
            scene_id=scene_id,
            task_id=task_id,
            goal=goal,
            metadata={
                "status": payload.get("status", ""),
                "final_score": float(payload.get("final_score", 0.0) or 0.0),
                "layout_id": self.derive_layout_id(room, variation_idx, sw_task_name, history_path),
                "memco_domain": self.domain_name,
                "task_family": task_family,
                "sw_task": sw_task,
                "sw_task_name": sw_task_name,
                "room": room,
                "scienceworld_domain": sw_domain,
                "scienceworld_family": sw_family,
            },
        )
        previous = history[0] if history else {}
        prev_state = self._state_from_record(previous, scene_id=scene_id)
        for record in history[1:]:
            action_text = str(record.get("Action") or "").strip()
            if not action_text:
                continue
            next_state = self._state_from_record(record, scene_id=scene_id)
            failure_label = self._failure_label_from_record(record)
            reward = self._float_value(record.get("Reward", 0.0))
            score = self._float_value(record.get("Score", 0.0))
            prev_score = self._float_value(previous.get("Score", 0.0))
            done = bool(record.get("Done", False))
            observation = str(record.get("Observation") or "")
            observation_only_action = self._is_observation_only_action(action_text)
            success = (
                failure_label is None
                and not observation_only_action
                and (
                    reward > 0
                    or score > prev_score
                    or (done and score > 0)
                    or self._observation_indicates_progress(observation)
                )
            )
            episode.steps.append(
                EpisodeStep(
                    step_idx=len(episode.steps),
                    state=prev_state,
                    action=self.canonicalize_action(action_text),
                    next_state=next_state,
                    feedback=StepFeedback(
                        success=bool(success),
                        score=score,
                        done=done,
                        failure_label=failure_label,
                        state_delta=self._state_delta(prev_state, next_state),
                    ),
                    subgoal=str(next_state.workflow_stage or ""),
                )
            )
            prev_state = next_state
        return episode

    # ---------- internal helpers ----------

    def _state_from_record(self, record: dict[str, Any], *, scene_id: str) -> StateSummary:
        observation = str(record.get("Observation") or "")
        score = float(record.get("Score", 0.0) or 0.0)
        step_count = int(record.get("Step", 0) or 0)
        done = bool(record.get("Done", False))
        has_positive = score > 0
        progress = self._progress_state(
            score=score,
            has_positive_progress=has_positive,
            last_completed=done,
            step_count=step_count,
        )
        objects = tuple(dict.fromkeys(_extract_room_objects(observation)))[:20]
        verbs = tuple(
            dict.fromkeys(
                self.canonicalize_action(cmd).verb
                for cmd in (record.get("Admissible Commands") or [])
                if str(cmd).strip()
            )
        )
        return StateSummary(
            domain=Domain.SCIENCEWORLD,
            scene_id=scene_id,
            visible_objects=objects,
            admissible_verbs=verbs,
            workflow_stage=progress,
            raw_observation=_short_text(observation, limit=500),
        )

    @staticmethod
    def _progress_state(
        *,
        score: float,
        has_positive_progress: bool,
        last_completed: bool,
        step_count: int,
    ) -> str:
        if last_completed and score > 0:
            return "goal_satisfied"
        if score >= 0.5:
            return "near_completion"
        if has_positive_progress or score > 0:
            return "partial_progress"
        if step_count <= 0:
            return "initial_planning"
        return "exploring"

    @staticmethod
    def _failure_label_from_observation(observation: str, *, score: float = 0.0, step_count: int = 0) -> str | None:
        obs = str(observation or "").lower()
        if (
            "invalid action" in obs
            or "no known action matches" in obs
            or "i don't understand" in obs
            or "i'm not sure" in obs
            or "not sure how to" in obs
            or "don't know how" in obs
        ):
            return "invalid_action"
        if (
            "nothing happens" in obs
            or "can't do that" in obs
            or "cannot" in obs
            or "can't" in obs
            or "already open" in obs
            or "already closed" in obs
        ):
            return "action_has_no_effect"
        if "the door is not open" in obs or "is not open" in obs:
            return "precondition_missing"
        return None

    @staticmethod
    def _failure_label_from_record(record: dict[str, Any]) -> str | None:
        obs = str(record.get("Observation") or "").lower()
        if (
            "invalid action" in obs
            or "no known action matches" in obs
            or "i don't understand" in obs
            or "i'm not sure" in obs
            or "not sure how to" in obs
            or "don't know how" in obs
        ):
            return "invalid_action"
        if (
            "nothing happens" in obs
            or "cannot" in obs
            or "can't" in obs
            or "already open" in obs
            or "already closed" in obs
        ):
            return "action_has_no_effect"
        if "the door is not open" in obs or "is not open" in obs:
            return "precondition_missing"
        score = float(record.get("Score", 0.0) or 0.0)
        if bool(record.get("Done", False)) and score <= 0:
            return "task_failed"
        return None

    @staticmethod
    def _float_value(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _observation_indicates_progress(observation: str) -> bool:
        obs = str(observation or "").lower()
        if not obs:
            return False
        if ScienceWorldAdapter._failure_label_from_observation(obs) is not None:
            return False
        progress_markers = (
            "the door is now open",
            "is now open",
            "you move to",
            "you are now in",
            "you activate",
            "you deactivate",
            "you connect",
            "you disconnect",
            "you turn on",
            "you turn off",
        )
        return any(marker in obs for marker in progress_markers)

    @staticmethod
    def _is_observation_only_action(action: str) -> bool:
        text = re.sub(r"\s+", " ", str(action or "").strip().lower())
        return text in {"look", "look around", "look room", "look at room", "inventory"}

    @staticmethod
    def _state_delta(prev_state: StateSummary, next_state: StateSummary) -> tuple[str, ...]:
        prev = set(prev_state.visible_objects)
        nxt = set(next_state.visible_objects)
        added = sorted(nxt - prev)[:6]
        removed = sorted(prev - nxt)[:4]
        delta = [f"obj+={item}" for item in added] + [f"obj-={item}" for item in removed]
        if prev_state.workflow_stage != next_state.workflow_stage:
            delta.append(f"stage={prev_state.workflow_stage}->{next_state.workflow_stage}")
        return tuple(delta[:10])
