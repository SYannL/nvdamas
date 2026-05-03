from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .graph_types import (
    ActionFamily,
    CanonicalAction,
    CandidateType,
    Domain,
    EpisodeRecord,
    EpisodeStep,
    StateSummary,
    StepFeedback,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


def _short_text(text: str, *, limit: int = 220) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())[:limit]


def _extract_bracket_arg(action: str) -> tuple[str, str]:
    match = re.match(r"^\s*([A-Za-z_]+)\[(.*)\]\s*$", str(action or "").strip())
    if not match:
        return "", ""
    return match.group(1), match.group(2).strip()


class FeverAdapter:
    """GraphMemory adapter for FEVER search/lookup/finish trajectories."""

    domain_name = "fever"

    def canonicalize_action(self, raw_action: str) -> CanonicalAction:
        surface = str(raw_action or "").strip()
        action_type, argument = _extract_bracket_arg(surface)
        lowered = surface.lower()
        if "thought" in lowered and not action_type:
            return CanonicalAction(Domain.FEVER, ActionFamily.OTHER, "thought", surface_form=surface)
        if action_type.lower() == "search":
            return CanonicalAction(
                Domain.FEVER,
                ActionFamily.SEARCH,
                "search",
                {"query": _normalize(argument)},
                surface_form=surface,
            )
        if action_type.lower() == "lookup":
            return CanonicalAction(
                Domain.FEVER,
                ActionFamily.INSPECT,
                "lookup",
                {"keyword": _normalize(argument)},
                surface_form=surface,
            )
        if action_type.lower() == "finish":
            return CanonicalAction(
                Domain.FEVER,
                ActionFamily.CONTROL,
                "finish",
                {"label": _normalize(argument)},
                surface_form=surface,
            )
        verb = _normalize(surface.split(" ", 1)[0] if surface else "unknown") or "unknown"
        return CanonicalAction(Domain.FEVER, ActionFamily.OTHER, verb, surface_form=surface)

    def derive_scene_id(self, domain: str | None = None, history_path: str | None = None) -> str:
        if domain:
            return f"fever:{_normalize(domain)}"
        if history_path:
            stem = Path(history_path).stem
            match = re.search(r"history_fever_([^_]+)_", stem)
            if match:
                return f"fever:{_normalize(match.group(1))}"
        return "fever:default"

    def derive_layout_id(self, domain: str | None = None, task_id: Any = None, history_path: str | None = None) -> str:
        return f"{self.derive_scene_id(domain, history_path)}:{task_id if task_id is not None else 'unknown'}"

    def infer_task_family(self, claim: str, domain: str | None = None) -> str:
        if domain:
            return f"fever:{_normalize(domain)}"
        return "fever:claim_verification"

    def goal_slots(self, goal: str) -> dict[str, str]:
        entity = self._claim_anchor(goal)
        return {"object": entity} if entity else {}

    def required_goal_count(self, goal: str) -> int:
        return 1

    def build_state(
        self,
        *,
        scene_id: str,
        claim: str,
        observation: str,
        admissible_commands: list[str] | tuple[str, ...] = (),
        stage: str = "",
    ) -> StateSummary:
        evidence_tokens = tuple(dict.fromkeys(_normalize(x) for x in self._evidence_phrases(observation) if _normalize(x)))[:16]
        verbs = tuple(dict.fromkeys(self.canonicalize_action(cmd).verb for cmd in admissible_commands if str(cmd).strip()))
        return StateSummary(
            domain=Domain.FEVER,
            scene_id=scene_id,
            visible_objects=evidence_tokens,
            admissible_verbs=verbs,
            workflow_stage=stage or "need_search",
            raw_observation=_short_text(observation, limit=600),
        )

    def build_query(self, *, env_ref: Any, task_config: dict, query_task: str, MemoryQuery: Any, CandidateType: Any):
        claim = str(getattr(env_ref, "claim", "") or getattr(env_ref, "current_task", "") or task_config.get("task", "") or query_task or "").strip()
        claim = claim.removeprefix("Claim:").strip()
        if not claim:
            return None
        domain = str(task_config.get("ab_domain", "") or getattr(env_ref, "ab_domain", "") or "default")
        scene_id = self.derive_scene_id(domain)
        history = [row for row in (getattr(env_ref, "current_history", []) or []) if isinstance(row, dict)]
        current = history[-1] if history else {}
        observation = str(current.get("Observation") or "")
        admissible = [str(cmd) for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()]
        progress_state = self._progress_state(history, done=bool(current.get("Done", False)))
        state = self.build_state(
            scene_id=scene_id,
            claim=claim,
            observation=observation,
            admissible_commands=admissible,
            stage=progress_state,
        )
        env_config = getattr(env_ref, "config", {}) if hasattr(env_ref, "config") else {}
        label = str(task_config.get("answer", "") or env_config.get("answer", "") or "")
        desired_types = [CandidateType.PRECONDITION, CandidateType.WORKFLOW]
        if progress_state in {"search_failed", "invalid_action"}:
            desired_types.append(CandidateType.FAILURE)
        entity = self._claim_anchor(claim)
        keywords = tuple(
            dict.fromkeys(
                [
                    f"domain={domain}",
                    f"progress={progress_state}",
                    f"label={_normalize(label)}" if label else "",
                    *(_normalize(x) for x in self._claim_keywords(claim)[:10]),
                    *(_normalize(x) for x in self._evidence_phrases(observation)[:8]),
                ]
            )
        )
        return MemoryQuery(
            goal=f"Verify claim: {claim}",
            scene_id=scene_id,
            current_stage=state.workflow_stage,
            progress_state=progress_state,
            task_family=self.infer_task_family(claim, domain),
            goal_roles={"object": entity} if entity else {},
            required_count=1,
            placed_relevant_count=1 if progress_state in {"ready_finish", "done"} else 0,
            remaining_relevant_count=0 if progress_state == "done" else 1,
            destination_reached=progress_state == "done",
            goal_object_matches_visible=progress_state in {"need_lookup_or_finish", "ready_finish", "done"},
            admissible_actions=tuple(self.canonicalize_action(cmd) for cmd in admissible),
            desired_types=tuple(desired_types),
            failure_label=self._last_failure(history),
            keywords=tuple(k for k in keywords if k),
            belief={
                "claim": claim,
                "answer": label,
                "history_len": len(history),
                "last_action": str(current.get("Action", "") or ""),
            },
            dynamic_context={
                "visible_objects": list(state.visible_objects),
                "layout_id": self.derive_layout_id(domain, task_config.get("task_id", "")),
                "task_config_env_name": str(task_config.get("env_name", "")),
                "gm3_domain": self.domain_name,
            },
        )

    def episode_from_history(self, history_path: str, agent_id: str) -> EpisodeRecord:
        with open(history_path, "r", encoding="utf-8") as reader:
            payload = json.load(reader)
        claim = str(payload.get("claim") or payload.get("game_task") or "").removeprefix("Claim:").strip()
        domain = str(payload.get("ab_domain") or payload.get("domain") or "default")
        task_id = str(payload.get("task_id") or payload.get("game_index") or Path(history_path).stem)
        scene_id = self.derive_scene_id(domain, history_path)
        history = [row for row in payload.get("history", []) if isinstance(row, dict)]
        episode = EpisodeRecord(
            agent_id=agent_id,
            scene_id=scene_id,
            task_id=task_id,
            goal=f"Verify claim: {claim}",
            metadata={
                "status": payload.get("status", ""),
                "final_score": float(payload.get("final_score", 0.0) or 0.0),
                "layout_id": self.derive_layout_id(domain, task_id, history_path),
                "gm3_domain": self.domain_name,
                "task_family": self.infer_task_family(claim, domain),
            },
        )
        previous = history[0] if history else {}
        prev_state = self._state_from_record(previous, scene_id=scene_id, claim=claim, history_prefix=history[:1])
        for idx, record in enumerate(history[1:], start=1):
            action_text = str(record.get("Action") or "").strip()
            if not action_text:
                continue
            next_state = self._state_from_record(record, scene_id=scene_id, claim=claim, history_prefix=history[: idx + 1])
            failure_label = self._failure_label(record)
            success = failure_label is None and not str(record.get("Action Type", "")).lower() == "thought"
            episode.steps.append(
                EpisodeStep(
                    step_idx=len(episode.steps),
                    state=prev_state,
                    action=self.canonicalize_action(action_text),
                    next_state=next_state,
                    feedback=StepFeedback(
                        success=bool(success),
                        score=float(record.get("Score", 0.0) or 0.0),
                        done=bool(record.get("Done", False)),
                        failure_label=failure_label,
                        state_delta=self._state_delta(prev_state, next_state),
                    ),
                    subgoal=str(next_state.workflow_stage or ""),
                )
            )
            prev_state = next_state
        return episode

    def _state_from_record(
        self,
        record: dict[str, Any],
        *,
        scene_id: str,
        claim: str,
        history_prefix: list[dict[str, Any]],
    ) -> StateSummary:
        return self.build_state(
            scene_id=scene_id,
            claim=claim,
            observation=str(record.get("Observation") or ""),
            admissible_commands=[str(x) for x in (record.get("Admissible Commands") or [])],
            stage=self._progress_state(history_prefix, done=bool(record.get("Done", False))),
        )

    @staticmethod
    def _progress_state(history: list[dict[str, Any]], *, done: bool = False) -> str:
        if done:
            return "done"
        actions = [str(row.get("Action", "") or "") for row in history if str(row.get("Action", "") or "").strip()]
        observations = [str(row.get("Observation", "") or "") for row in history if str(row.get("Observation", "") or "").strip()]
        if any("invalid action" in obs.lower() for obs in observations[-2:]):
            return "invalid_action"
        if any(action.lower().startswith("lookup[") for action in actions):
            return "ready_finish"
        if any(action.lower().startswith("search[") for action in actions):
            if observations and "cannot find" in observations[-1].lower():
                return "search_failed"
            return "need_lookup_or_finish"
        return "need_search"

    @staticmethod
    def _last_failure(history: list[dict[str, Any]]) -> str | None:
        for row in reversed(history):
            failure = FeverAdapter._failure_label(row)
            if failure:
                return failure
        return None

    @staticmethod
    def _failure_label(record: dict[str, Any]) -> str | None:
        obs = str(record.get("Observation") or "").lower()
        if "invalid action" in obs:
            return "invalid_action"
        if "cannot find" in obs:
            return "search_not_found"
        if "last page searched was not found" in obs:
            return "lookup_without_page"
        if bool(record.get("Done", False)) and float(record.get("Score", 0.0) or 0.0) <= 0:
            return "wrong_finish_label"
        return None

    @staticmethod
    def _state_delta(prev_state: StateSummary, next_state: StateSummary) -> tuple[str, ...]:
        prev = set(prev_state.visible_objects)
        nxt = set(next_state.visible_objects)
        deltas = [f"evidence+={item}" for item in sorted(nxt - prev)[:6]]
        if prev_state.workflow_stage != next_state.workflow_stage:
            deltas.append(f"stage={next_state.workflow_stage}")
        return tuple(deltas)

    @staticmethod
    def _claim_anchor(claim: str) -> str:
        text = str(claim or "").strip()
        text = re.sub(r"^\s*(?:verify\s+claim|claim)\s*:\s*", "", text, flags=re.I).strip()
        # FEVER search should usually start from the claim subject.  The older
        # title-case regex collapsed names with lowercase connectors, e.g.
        # "Off the Wall" -> "Off", which produces weak Search hints.
        predicate = re.search(
            r"\b(?:is|are|was|were|has|have|had|does|do|did|worked|appeared|released|formed|created|directed|starred)\b",
            text,
            flags=re.I,
        )
        if predicate:
            subject = text[: predicate.start()].strip(" .,:;\"'")
            if 1 <= len(re.findall(r"[A-Za-z0-9]+", subject)) <= 8:
                return _normalize(subject)
        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]\.)(?:\s+(?:the|of|and|in|on|for|to|a|an|[A-Z][A-Za-z0-9]*|[A-Z]\.)){0,5}",
            text,
        )
        if candidates:
            return _normalize(candidates[0])
        words = FeverAdapter._claim_keywords(claim)
        return _normalize(words[0]) if words else ""

    @staticmethod
    def _claim_keywords(claim: str) -> list[str]:
        stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "and", "or", "to", "by", "for"}
        return [word for word in re.findall(r"[A-Za-z0-9]+", str(claim or "").lower()) if word not in stop and len(word) > 2]

    @staticmethod
    def _evidence_phrases(observation: str) -> list[str]:
        text = str(observation or "")
        if not text:
            return []
        pieces = re.split(r"\.\s+|;\s+|\n+", text)
        return [piece.strip(" .") for piece in pieces if piece.strip(" .")][:12]
