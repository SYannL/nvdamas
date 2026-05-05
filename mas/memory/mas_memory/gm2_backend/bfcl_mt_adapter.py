from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from .graph_types import ActionFamily, CanonicalAction, Domain, EpisodeRecord, EpisodeStep, StateSummary, StepFeedback
from .pddl_adapter import PDDLAdapter, _normalize


class BfclAdapter(PDDLAdapter):
    """GraphMemory adapter for BFCL multi-turn tool episodes (Exec / FinishTurn strings)."""

    domain_name = "bfcl_mt"

    def canonicalize_action(self, raw_action: str) -> CanonicalAction:
        surface = str(raw_action or "").strip()
        text = re.sub(r"^Exec\[(.*)\]$", r"\1", surface)
        low = text.lower().strip()
        if low.startswith("finishturn"):
            return CanonicalAction(Domain.BFCL, ActionFamily.CONTROL, "finish_turn", surface_form=surface)
        if low.startswith("execmany[") or low.startswith("["):
            return CanonicalAction(Domain.BFCL, ActionFamily.MANIPULATE, "multi_call", surface_form=surface)
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)\s*$", text, flags=re.S)
        if match:
            name = match.group(1).split(".")[-1]
            args = match.group(2)
            slots = {
                key: _normalize(value.strip().strip("\"'")[:80])
                for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\)]+)", args)
            }
            verb = _normalize(name)
            family = self._action_family_for_verb(verb)
            return CanonicalAction(Domain.BFCL, family, verb, slots, surface_form=surface)
        return CanonicalAction(Domain.BFCL, ActionFamily.OTHER, _normalize(surface.split("(", 1)[0] or "unknown"), surface_form=surface)

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

    def derive_layout_id(self, game_name: str | None = None, problem_index: Any = None, history_path: str | None = None) -> str:
        scene_id = self.derive_scene_id(game_name, history_path)
        return f"{scene_id}:{problem_index if problem_index is not None else 'unknown'}"

    def build_query(self, *, env_ref: Any, task_config: dict, query_task: str, MemoryQuery: Any, CandidateType: Any):
        game_name = str(
            getattr(env_ref, "game_name", "")
            or task_config.get("bfcl_shard_domain", "")
            or task_config.get("game_name", "")
            or "default"
        ).strip()
        goal = str(getattr(env_ref, "goal_instruction", "") or query_task or "").strip()
        if not goal:
            return None

        entry = task_config.get("bfcl_entry") or getattr(env_ref, "_test_entry", {}) or {}
        initial_config = entry.get("initial_config") or task_config.get("initial_config") or {}
        involved = [
            str(x).strip()
            for x in (
                getattr(env_ref, "goal_literals_text", None)
                or entry.get("involved_classes")
                or task_config.get("goal_literals")
                or []
            )
            if str(x).strip()
        ]
        history = [row for row in (getattr(env_ref, "current_history", []) or []) if isinstance(row, dict)]
        current = history[-1] if history else {}
        admissible = [str(cmd) for cmd in (getattr(env_ref, "last_admissible_commands", []) or []) if str(cmd).strip()]
        observation = str(
            current.get("Observation")
            or (getattr(env_ref, "states", [""])[-1] if getattr(env_ref, "states", None) else "")
            or ""
        )
        current_literals = [
            str(x).strip()
            for x in (getattr(env_ref, "current_literals_text", None) or current.get("Current Literals") or [])
            if str(x).strip()
        ]
        scene_id = self.derive_scene_id(game_name)
        state = self.build_state(
            scene_id=scene_id,
            observation=observation,
            admissible_commands=admissible,
            goal=goal,
            goal_literals=involved,
            current_literals=current_literals,
        )

        last_action = str(current.get("Action", "") or "").strip()
        recent_actions = [
            str(row.get("Action", "") or "").strip()
            for row in history
            if str(row.get("Action", "") or "").strip()
        ]
        repeat_count = 0
        if last_action:
            for action in reversed(recent_actions):
                if self._action_key(action) != self._action_key(last_action):
                    break
                repeat_count += 1
        progress_state = self._progress_state(
            observation=observation,
            last_action=last_action,
            done=bool(getattr(env_ref, "done", False) or current.get("Done", False)),
            reward=float(getattr(env_ref, "reward", 0.0) or current.get("Reward", 0.0) or 0.0),
        )
        desired_types = [CandidateType.PRECONDITION, CandidateType.WORKFLOW]
        if progress_state in {"invalid_action", "tool_error"} or repeat_count >= 2:
            desired_types.append(CandidateType.FAILURE)

        api_tokens = [_normalize(x) for x in involved if _normalize(x)]
        session_state = self._session_state(initial_config=initial_config, involved_classes=involved)
        turn_texts = [str(x) for x in (getattr(env_ref, "_turn_texts", []) or [])]
        turn_index = int(getattr(env_ref, "_turn_idx", 0) or 0)
        current_turn_text = turn_texts[turn_index] if 0 <= turn_index < len(turn_texts) else ""
        keywords = tuple(
            dict.fromkeys(
                [
                    f"domain={game_name}",
                    f"progress={progress_state}",
                    *[f"api={token}" for token in api_tokens[:8]],
                    *self._action_keywords(last_action),
                ]
            )
        )
        return MemoryQuery(
            goal=goal,
            scene_id=scene_id,
            current_stage=progress_state,
            progress_state=progress_state,
            task_family=self.infer_task_family(goal, game_name),
            goal_roles={"object": _normalize(game_name or "bfcl"), "api": "_".join(api_tokens[:3])},
            required_count=1,
            placed_relevant_count=1 if progress_state in {"ready_finish", "done"} else 0,
            remaining_relevant_count=0 if progress_state == "done" else 1,
            destination_reached=progress_state == "done",
            goal_object_matches_visible=progress_state in {"tool_result_observed", "ready_finish", "done"},
            admissible_actions=tuple(self.canonicalize_action(cmd) for cmd in admissible),
            desired_types=tuple(desired_types),
            failure_label=self._failure_label_from_observation(observation, repeat_count=repeat_count),
            keywords=tuple(str(k) for k in keywords if str(k).strip()),
            belief={
                "involved_classes": involved,
                "history_len": len(history),
                "turn_index": turn_index,
                "turn_count": len(turn_texts),
                "current_user_turn": current_turn_text[:800],
                "session_state": session_state,
                "last_action": last_action,
                "last_action_repeat_count": repeat_count,
                "last_observation": observation[:800],
            },
            dynamic_context={
                "visible_objects": list(state.visible_objects),
                "layout_id": self.derive_layout_id(game_name, task_config.get("bfcl_id") or entry.get("id", "")),
                "task_config_env_name": str(task_config.get("env_name", "")),
                "gm3_domain": self.domain_name,
                "last_action": last_action,
                "last_action_repeat_count": repeat_count,
                "involved_classes": involved,
                "session_state": session_state,
            },
        )

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
        action = str(record.get("Action") or "")
        observation = str(record.get("Observation") or "")
        progress = self._progress_state(
            observation=observation,
            last_action=action,
            done=bool(record.get("Done", False)),
            reward=float(record.get("Reward", record.get("Score", 0.0)) or 0.0),
        )
        visible = tuple(dict.fromkeys(_normalize(x) for x in (record.get("Current Literals") or []) if _normalize(x)))[:20]
        if not visible:
            visible = tuple(dict.fromkeys(_normalize(x) for x in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", observation)[:16] if _normalize(x)))
        verbs = tuple(
            dict.fromkeys(
                self.canonicalize_action(cmd).verb
                for cmd in (record.get("Admissible Commands") or [])
                if str(cmd).strip()
            )
        )
        return StateSummary(
            domain=Domain.BFCL,
            scene_id=scene_id,
            visible_objects=visible,
            admissible_verbs=verbs,
            workflow_stage=progress,
            raw_observation=re.sub(r"\s+", " ", observation.strip())[:500],
        )

    def episode_from_history(self, history_path: str, agent_id: str) -> EpisodeRecord:
        with open(history_path, "r", encoding="utf-8") as reader:
            payload = json.load(reader)
        task_config = payload.get("task_config") if isinstance(payload.get("task_config"), dict) else {}
        game_name = str(payload.get("game_name") or task_config.get("bfcl_shard_domain") or "default")
        task_id = str(payload.get("game_index") or task_config.get("bfcl_id") or Path(history_path).stem)
        goal = str(payload.get("game_task") or f"BFCL task {task_id}")
        scene_id = self.derive_scene_id(game_name, history_path)
        history = [row for row in payload.get("history", []) if isinstance(row, dict)]
        episode = EpisodeRecord(
            agent_id=agent_id,
            scene_id=scene_id,
            task_id=task_id,
            goal=goal,
            metadata={
                "status": payload.get("status", ""),
                "final_score": float(payload.get("final_score", 0.0) or 0.0),
                "layout_id": self.derive_layout_id(game_name, task_id, history_path),
                "gm3_domain": self.domain_name,
                "task_family": self.infer_task_family(goal, game_name),
            },
        )
        goal_literals = [str(x) for x in (payload.get("goal_literals") or []) if str(x).strip()]
        previous = history[0] if history else {}
        prev_state = self._state_from_record(previous, scene_id=scene_id, goal=goal, goal_literals=goal_literals)
        for record in history[1:]:
            action_text = str(record.get("Action") or "").strip()
            if not action_text:
                continue
            next_state = self._state_from_record(record, scene_id=scene_id, goal=goal, goal_literals=goal_literals)
            failure_label = self._failure_label_from_record(record)
            success = failure_label is None
            episode.steps.append(
                EpisodeStep(
                    step_idx=len(episode.steps),
                    state=prev_state,
                    action=self.canonicalize_action(action_text),
                    next_state=next_state,
                    feedback=StepFeedback(
                        success=success,
                        score=float(record.get("Score", 0.0) or 0.0),
                        done=bool(record.get("Done", False)),
                        failure_label=failure_label,
                        state_delta=self._bfcl_state_delta(prev_state, next_state),
                    ),
                    subgoal=str(next_state.workflow_stage or ""),
                )
            )
            prev_state = next_state
        return episode

    @staticmethod
    def _action_key(action: str) -> str:
        text = str(action or "").strip()
        text = re.sub(r"^Exec\[(.*)\]$", r"\1", text)
        return _normalize(text)

    @staticmethod
    def _action_keywords(action: str) -> list[str]:
        text = str(action or "")
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        return [f"action={_normalize(token)}" for token in tokens[:6] if _normalize(token)]

    @staticmethod
    def _failure_label_from_observation(observation: str, *, repeat_count: int = 0) -> str | None:
        low = str(observation or "").lower()
        if repeat_count >= 2:
            return "repeated_tool_call"
        if "invalid action" in low:
            return "invalid_action_format"
        if "error during execution" in low or low.startswith("error:") or '"error"' in low:
            return "tool_execution_error"
        if "fail (bfcl checker)" in low:
            return "checker_failed"
        return None

    @classmethod
    def _progress_state(cls, *, observation: str, last_action: str, done: bool, reward: float) -> str:
        low_obs = str(observation or "").lower()
        low_action = str(last_action or "").lower().strip()
        if done:
            return "done" if reward > 0 else "checker_failed"
        if "invalid action" in low_obs:
            return "invalid_action"
        if "error during execution" in low_obs or low_obs.startswith("error:") or '"error"' in low_obs:
            return "tool_error"
        if low_action.startswith("finishturn"):
            return "need_tool_call"
        if last_action:
            return "tool_result_observed"
        return "need_tool_call"

    @staticmethod
    def _action_family_for_verb(verb: str) -> ActionFamily:
        if verb in {"finish_turn", "login", "logout"} or verb.endswith("_login") or verb.endswith("_logout"):
            return ActionFamily.CONTROL
        if verb.startswith(("get_", "list_", "search_", "lookup_", "check_", "filter_")):
            return ActionFamily.INSPECT
        if verb.startswith(("add_", "remove_", "place_", "fund_", "withdraw_", "book_", "cancel_", "update_", "create_", "delete_", "send_")):
            return ActionFamily.MANIPULATE
        return ActionFamily.OTHER

    @classmethod
    def _failure_label_from_record(cls, record: dict[str, Any]) -> str | None:
        observation = str(record.get("Observation") or "")
        action = str(record.get("Action") or "")
        low_obs = observation.lower()
        low_action = action.lower()
        label = cls._failure_label_from_observation(observation)
        if label:
            return label
        if "already logged in" in low_obs and ("login" in low_action or "auth" in low_action):
            return "redundant_auth"
        if bool(record.get("Done", False)) and "fail (bfcl checker)" in low_obs:
            return "checker_failed"
        return None

    @staticmethod
    def _bfcl_state_delta(prev_state: StateSummary, next_state: StateSummary) -> tuple[str, ...]:
        delta: list[str] = []
        if prev_state.workflow_stage != next_state.workflow_stage:
            delta.append(f"stage={prev_state.workflow_stage}->{next_state.workflow_stage}")
        prev_visible = set(prev_state.visible_objects)
        for item in next_state.visible_objects:
            if item not in prev_visible:
                delta.append(f"observed={item}")
        return tuple(delta[:8])

    @classmethod
    def _session_state(cls, *, initial_config: Any, involved_classes: list[str]) -> dict[str, Any]:
        if not isinstance(initial_config, dict):
            return {}
        state: dict[str, Any] = {}
        for class_name in involved_classes:
            cfg = initial_config.get(class_name)
            if not isinstance(cfg, dict):
                continue
            key = _normalize(class_name)
            auth_value = cls._first_bool(cfg, ("authenticated", "logged_in", "is_authenticated", "login_status"))
            if auth_value is not None:
                state[f"{key}.authenticated"] = bool(auth_value)
            account = cfg.get("account_info")
            if isinstance(account, dict):
                if "balance" in account:
                    state[f"{key}.balance"] = account.get("balance")
                if "account_id" in account:
                    state[f"{key}.account_id"] = account.get("account_id")
            if isinstance(cfg.get("watch_list"), list):
                state[f"{key}.watch_list"] = list(cfg.get("watch_list") or [])[:8]
        return state

    @staticmethod
    def _first_bool(mapping: dict[str, Any], keys: tuple[str, ...]) -> bool | None:
        for key in keys:
            if key not in mapping:
                continue
            value = mapping.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return value.lower() == "true"
        return None
