from __future__ import annotations

import json
import re
import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .graph_memory2 import GraphMemory2MASMemory
from .gm2_backend import rank_messages_for_query
from .gm3_backend.prompt_styles import prompt_style_for_query


@dataclass
class GraphMemory3MASMemory(GraphMemory2MASMemory):
    """GraphMemory3: prompt-only local/global graph routing with text-loss pruning.

    GM3 reuses the shared graph construction/persistence backend, but keeps its
    prompt routing and dataset-facing language independent from GraphMemory2:

    - local graph grounds current state and admissible actions;
    - global graph contributes transferable workflow/phase knowledge;
    - source priors remain weak search context;
    - a small text-loss router prunes noisy memory before prompt injection.
    - dataset styles render ALFWorld/PDDL/FEVER memory without cross-task wording.

    The class is prompt-only by default. It does not override concrete actions
    through the GM2 action hook, so the non-memory nvdamas workflow remains
    unchanged.
    """

    _gm3_debug_trace_path: Path | None = field(default=None, init=False, repr=False)
    _gm3_last_prompt_signature: str = field(default="", init=False, repr=False)
    _gm3_textgrad_seen_route_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _gm3_textgrad_route_key_hits: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _gm3_textgrad_prompt_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _gm3_textgrad_disabled_reason: str = field(default="", init=False, repr=False)
    _gm3_textgrad_calls_this_episode: int = field(default=0, init=False, repr=False)
    _gm3_search_bias_queue: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _gm3_runtime_blocked_actions: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        router = str(self._graph_config_value("router", "textloss") or "textloss").strip().lower()
        if router not in {"textloss"}:
            print(
                f"[graph_memory3] unsupported gm3_router `{router}`; falling back to `textloss`.",
                flush=True,
            )
            router = "textloss"
        self._external_retrieval_mode = "graph_memory3_textloss" if router == "textloss" else f"graph_memory3_{router}"
        self._gm3_debug_trace_path = Path(self.persist_dir) / "gm3_debug_trace.jsonl"
        self._gm3_last_prompt_signature = ""
        self._gm3_use_textgrad = bool(
            self.global_config.get("gm3_use_textgrad", False)
            or str(os.getenv("NV_GM3_USE_TEXTGRAD", "")).strip() in {"1", "true", "True", "yes"}
        )
        self._gm3_textgrad_engine = str(
            self.global_config.get("gm3_textgrad_engine", "")
            or os.getenv("NV_GM3_TEXTGRAD_ENGINE", "")
            or ""
        ).strip()
        self._gm3_textgrad_max_iters = max(
            0,
            int(
                self.global_config.get("gm3_textgrad_max_iters", 1)
                or os.getenv("NV_GM3_TEXTGRAD_MAX_ITERS", "1")
                or 1
            ),
        )
        self._gm3_textgrad_pass_threshold = float(
            self.global_config.get("gm3_textgrad_pass_threshold", 0.82)
            or os.getenv("NV_GM3_TEXTGRAD_PASS_THRESHOLD", "0.82")
            or 0.82
        )
        self._gm3_textgrad_max_calls_per_episode = max(
            0,
            int(
                self.global_config.get("gm3_textgrad_max_calls_per_episode", 2)
                or os.getenv("NV_GM3_TEXTGRAD_MAX_CALLS_PER_EPISODE", "2")
                or 2
            ),
        )

        # configure search priority weights for local and global source priors.  
        # A higher local weight biases toward local prior suggestions.  
        # A higher global weight increases influence of transferable patterns, which
        # is useful in unseen scenes. Users can override via env vars or global_config.
        try:
            self._gm3_local_weight = float(
                self.global_config.get("gm3_local_weight", os.getenv("NV_GM3_LOCAL_WEIGHT", "1.0"))
            )
        except Exception:
            self._gm3_local_weight = 1.0
        try:
            self._gm3_global_weight = float(
                self.global_config.get("gm3_global_weight", os.getenv("NV_GM3_GLOBAL_WEIGHT", "0.65"))
            )
        except Exception:
            self._gm3_global_weight = 0.65

        # minimum support and confidence thresholds for filtering memory artifacts; used by LocalGraphMaintainer
        try:
            self._gm3_min_artifact_support = int(
                self.global_config.get("gm3_min_artifact_support", os.getenv("NV_GM3_MIN_ARTIFACT_SUPPORT", "2"))
            )
        except Exception:
            self._gm3_min_artifact_support = 2
        try:
            self._gm3_min_artifact_confidence = float(
                self.global_config.get("gm3_min_artifact_confidence", os.getenv("NV_GM3_MIN_ARTIFACT_CONFIDENCE", "0.4"))
            )
        except Exception:
            self._gm3_min_artifact_confidence = 0.4
        # logistic parameter controlling the steepness of the support factor curve in source role scoring
        try:
            self._gm3_support_logistic_k = float(
                self.global_config.get("gm3_support_logistic_k", os.getenv("NV_GM3_SUPPORT_LOGISTIC_K", "0.35"))
            )
        except Exception:
            self._gm3_support_logistic_k = 0.35
        self._gm3_global_exclude_owner = bool(
            self.global_config.get("gm3_global_exclude_owner", True)
            and str(os.getenv("NV_GM3_GLOBAL_EXCLUDE_OWNER", "1")).strip() not in {"0", "false", "False", "no"}
        )

    def _gm3_model_name(self) -> str:
        return str(getattr(getattr(self, "llm_model", None), "model_name", "") or "").lower()

    def _gm3_is_gpt4omini_model(self) -> bool:
        return "gpt-4o-mini" in self._gm3_model_name()

    def init_task_context(self, task_main: str, task_description: str = None) -> Any:
        message = super().init_task_context(task_main, task_description)
        self._gm3_last_prompt_signature = ""
        self._gm3_textgrad_seen_route_keys = set()
        self._gm3_textgrad_route_key_hits = {}
        self._gm3_textgrad_calls_this_episode = 0
        self._gm3_search_bias_queue = []
        self._gm3_runtime_blocked_actions = set()
        self._gm3_debug_append(
            "task_start",
            step_index=0,
            payload={
                "task": str(task_main or ""),
                "description": str(task_description or ""),
            },
        )
        return message

    def move_memory_state(self, action: str, observation: str, **kargs) -> None:
        super().move_memory_state(action, observation, **kargs)
        action_norm = self._normalize_action_text(str(action or ""))
        observation_norm = self._gm3_norm(str(observation or "")).strip(".!")
        if action_norm.startswith("examine ") and observation_norm == "ok":
            self._gm3_runtime_blocked_actions.add(self._gm3_norm(action_norm))
        self._gm3_debug_append(
            "env_feedback",
            step_index=int(getattr(self, "_gm2_debug_env_step", 0) or 0),
            payload={
                "action": str(action or ""),
                "reward": kargs.get("reward"),
                "observation": self._gm2_debug_text(str(observation or ""), limit=1200),
            },
        )

    def summarize(self, **kargs) -> str:
        ctx = getattr(self, "current_task_context", None)
        if ctx is None:
            return super().summarize(**kargs)
        task_main = str(getattr(ctx, "task_main", "") or "").lower()
        task_description = str(getattr(ctx, "task_description", "") or "")
        is_scienceworld = "scienceworld" in task_main or "scienceworld" in task_description.lower()
        if not is_scienceworld:
            return super().summarize(**kargs)

        base = task_description + self._gm3_compact_scienceworld_trajectory(
            str(getattr(ctx, "task_trajectory", "") or "")
        )
        if self.enable_overlay and self.episode_builder is not None:
            notes = self.episode_builder.planner_notes()
            if notes:
                return base + "\n\n" + "\n".join(notes)
        return base

    @staticmethod
    def _gm3_compact_scienceworld_trajectory(trajectory: str, *, keep_steps: int = 6, obs_limit: int = 520) -> str:
        segments = [seg.strip() for seg in str(trajectory or "").split("\n>") if seg.strip()]
        if not segments:
            return "\n\n>"
        omitted = max(0, len(segments) - keep_steps)
        kept = segments[-keep_steps:]
        lines: list[str] = ["\n\n>"]
        if omitted:
            lines.append(f"[... omitted {omitted} earlier ScienceWorld steps ...]")
        for segment in kept:
            action, sep, observation = segment.partition("\n")
            obs = re.sub(r"\s+", " ", observation.strip())
            if len(obs) > obs_limit:
                obs = obs[:obs_limit].rstrip() + "..."
            lines.append(f"{action.strip()}\n{obs}\n>")
        return "\n".join(lines)

    def _gm3_scienceworld_repair_action(
        self,
        *,
        raw_response: str,
        processed_action: str,
        env_ref: Any,
        task_config: dict | None,
        step_index: int,
        admissible: list[str],
    ) -> str:
        final_action = str(processed_action or "")
        try:
            query = self._build_external_query(env_ref=env_ref, task_config=task_config, step_index=step_index)
        except Exception:
            query = None
        profile = self._gm3_scienceworld_task_profile(query=query, env_ref=env_ref, task_config=task_config or {})
        processed_norm = self._normalize_action_text(final_action)
        admissible_by_norm = {self._normalize_action_text(cmd): cmd for cmd in admissible}
        processed_admissible = admissible_by_norm.get(processed_norm, "")
        best_score = self._gm3_scienceworld_float_attr(env_ref, "best_score", 0.0)
        last_score = self._gm3_scienceworld_float_attr(env_ref, "last_score", 0.0)
        blocked_actions = self._gm3_scienceworld_recent_invalid_actions(env_ref)
        reason = "scienceworld_advisory_only"

        if processed_admissible and not final_action:
            final_action = processed_admissible

        self._gm3_debug_append(
            "action_hook_observe",
            step_index=step_index,
            payload={
                "raw_response": self._gm2_debug_text(str(raw_response or ""), limit=1200),
                "processed_action": str(processed_action or ""),
                "final_action": str(final_action or ""),
                "changed": str(final_action or "") != str(processed_action or ""),
                "reason": reason,
                "scienceworld_profile": self._gm2_debug_jsonable(profile),
                "last_score": last_score,
                "best_score": best_score,
                "blocked_actions": sorted(blocked_actions)[:20],
                "admissible_sample": admissible[:20],
            },
        )
        return final_action

    def _gm3_scienceworld_task_profile(
        self,
        *,
        query: Any,
        env_ref: Any,
        task_config: dict,
    ) -> dict[str, Any]:
        belief = getattr(query, "belief", {}) if query is not None else {}
        dynamic = getattr(query, "dynamic_context", {}) if query is not None else {}
        goal = str(
            task_config.get("sw_task_desc", "")
            or getattr(query, "goal", "")
            or getattr(env_ref, "task_description", "")
            or task_config.get("game_task", "")
            or ""
        )
        task_name = str(
            task_config.get("sw_task_name", "")
            or belief.get("sw_task_name", "")
            or dynamic.get("sw_task_name", "")
            or task_config.get("sw_task", "")
            or ""
        )
        text = self._normalize_action_text(f"{task_name}. {goal}")
        targets: list[str] = []
        answer_targets: list[str] = []
        task_kind = "generic"

        state_patterns = (
            r"\b(?:boil|melt|freeze|combust)\s+(?:the\s+)?([a-z0-9][a-z0-9 \-]+?)(?:[.;,]|$)",
            r"\bchange (?:the )?state of matter of\s+(?:the\s+)?([a-z0-9][a-z0-9 \-]+?)(?:[.;,]|$)",
            r"\bchange\s+(?:the\s+)?([a-z0-9][a-z0-9 \-]+?)\s+state of matter(?:[.;,]|$)",
        )
        for pattern in state_patterns:
            match = re.search(pattern, text)
            if match:
                task_kind = "state_change"
                targets.append(match.group(1))
                break

        for match in re.finditer(r"\b(?:first|next|then),?\s*focus on\s+(?:the\s+)?([a-z0-9][a-z0-9 \-]+?)(?:[.;,]|$)", text):
            target = match.group(1)
            if target not in {"substance", "object", "thing"}:
                targets.append(target)
                task_kind = "ordered_focus"

        for match in re.finditer(r"\bif\b[^.;]*?\bfocus on\s+(?:the\s+)?([a-z0-9][a-z0-9 \-]+?)(?:[.;,]|$)", text):
            answer_targets.append(match.group(1))
            task_kind = "conditional_focus"

        if "dominant" in text or "recessive" in text:
            task_kind = "genetics"

        targets = self._gm3_scienceworld_dedupe_phrases(targets)
        answer_targets = self._gm3_scienceworld_dedupe_phrases(answer_targets)
        return {
            "task_kind": task_kind,
            "targets": targets,
            "answer_targets": answer_targets,
            "goal": goal[:300],
            "task_name": task_name[:120],
        }

    @staticmethod
    def _gm3_scienceworld_dedupe_phrases(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            clean = re.sub(r"\b(?:first|next|then|for compounds.*)$", "", str(value or "").strip())
            clean = re.sub(r"\s+", " ", clean).strip(" .,:;")
            norm = re.sub(r"^(?:the|a|an)\s+", "", clean.lower())
            if norm and norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out

    @staticmethod
    def _gm3_scienceworld_float_attr(env_ref: Any, name: str, default: float) -> float:
        try:
            return float(getattr(env_ref, name, default) or default)
        except Exception:
            return default

    def _gm3_scienceworld_focus_matches_profile(self, action: str, profile: dict[str, Any]) -> bool:
        target_text = self._gm3_scienceworld_action_target_text(action)
        if not target_text:
            return False
        if self._gm3_scienceworld_is_distractor_focus(target_text, profile):
            return False
        targets = list(profile.get("targets") or [])
        answer_targets = list(profile.get("answer_targets") or [])
        if answer_targets and self._gm3_scienceworld_phrase_matches_any(target_text, answer_targets):
            return True
        if targets:
            return self._gm3_scienceworld_phrase_matches_any(target_text, targets)
        return not self._gm3_scienceworld_is_generic_bad_target(target_text)

    def _gm3_scienceworld_action_matches_profile(self, action: str, profile: dict[str, Any]) -> bool:
        norm = self._normalize_action_text(action)
        targets = list(profile.get("targets") or []) + list(profile.get("answer_targets") or [])
        if not targets:
            return not self._gm3_scienceworld_is_generic_bad_target(self._gm3_scienceworld_action_target_text(norm))
        return self._gm3_scienceworld_phrase_matches_any(norm, targets)

    def _gm3_scienceworld_recent_invalid_actions(self, env_ref: Any) -> set[str]:
        blocked: set[str] = set()
        if env_ref is None:
            return blocked
        for record in list(getattr(env_ref, "current_history", []) or []):
            if not isinstance(record, dict):
                continue
            action = self._normalize_action_text(str(record.get("Action", "") or ""))
            observation = self._gm3_norm(str(record.get("Observation", "") or ""))
            if action and (
                "no known action matches" in observation
                or "not a known action" in observation
            ):
                blocked.add(self._gm3_norm(action))
        return blocked

    @staticmethod
    def _gm3_scienceworld_is_high_risk_action(action: str) -> bool:
        norm = re.sub(r"\s+", " ", str(action or "").strip().lower())
        return norm.startswith(("focus on ", "move ", "put ", "pour ", "mix ", "activate ", "connect "))

    @staticmethod
    def _gm3_scienceworld_action_target_text(action: str) -> str:
        norm = re.sub(r"\s+", " ", str(action or "").strip().rstrip(".。").lower())
        for prefix in ("focus on ", "move ", "put ", "pour ", "mix ", "activate ", "connect ", "open ", "go to ", "examine "):
            if norm.startswith(prefix):
                return norm[len(prefix):].strip()
        return norm

    def _gm3_scienceworld_is_distractor_focus(self, target_text: str, profile: dict[str, Any]) -> bool:
        target_norm = self._normalize_action_text(target_text)
        if target_norm in {"air", "inventory", "agent", "object", "thing"}:
            return True
        return False

    @staticmethod
    def _gm3_scienceworld_is_generic_bad_target(target_text: str) -> bool:
        norm = re.sub(r"\s+", " ", str(target_text or "").strip().lower())
        return norm in {"air", "inventory", "agent", "object", "thing"}

    def _gm3_scienceworld_phrase_matches_any(self, text: str, phrases: list[str]) -> bool:
        text_tokens = set(self._gm3_scienceworld_tokens(text))
        if not text_tokens:
            return False
        for phrase in phrases:
            phrase_tokens = self._gm3_scienceworld_tokens(phrase)
            if phrase_tokens and set(phrase_tokens).issubset(text_tokens):
                return True
        return False

    @staticmethod
    def _gm3_scienceworld_tokens(text: str) -> list[str]:
        stop = {
            "a", "an", "the", "of", "in", "on", "to", "with", "and", "or", "for",
            "substance", "object", "thing", "material", "unknown",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if token not in stop and len(token) > 1
        ]

    def repair_action(
        self,
        *,
        raw_response: str,
        processed_action: str,
        env_ref: Any,
        task_config: dict | None = None,
        step_index: int = 0,
    ) -> str:
        admissible = [
            str(cmd).strip()
            for cmd in (getattr(env_ref, "last_admissible_commands", []) or [])
            if str(cmd).strip()
        ]
        domain = self._infer_external_domain(task_config or {}, env_ref)
        if domain == "scienceworld":
            return self._gm3_scienceworld_repair_action(
                raw_response=raw_response,
                processed_action=processed_action,
                env_ref=env_ref,
                task_config=task_config,
                step_index=step_index,
                admissible=admissible,
            )
        if domain != "alfworld":
            self._gm3_debug_append(
                "action_hook_observe",
                step_index=step_index,
                payload={
                    "raw_response": self._gm2_debug_text(str(raw_response or ""), limit=1200),
                    "processed_action": str(processed_action or ""),
                    "final_action": str(processed_action or ""),
                    "changed": False,
                    "reason": f"gm3_prompt_only_for_{domain}",
                    "admissible_sample": admissible[:20],
                },
            )
            return str(processed_action or "")
        query = self._build_external_query(
            env_ref=env_ref,
            task_config=task_config,
            step_index=step_index,
        )
        final_action = str(processed_action or "")
        change_reason = ""
        blocked_actions = self._gm3_recent_examine_ok_actions(env_ref)

        def _debug_return(selected_action: str, reason: str) -> str:
            self._gm3_debug_append(
                "action_hook_observe",
                step_index=step_index,
                payload={
                    "raw_response": self._gm2_debug_text(str(raw_response or ""), limit=1200),
                    "processed_action": str(processed_action or ""),
                    "final_action": str(selected_action or ""),
                    "changed": str(selected_action or "") != str(processed_action or ""),
                    "reason": reason,
                    "search_bias_queue": self._gm2_debug_jsonable(self._gm3_search_bias_queue[:6]),
                    "blocked_actions": sorted(blocked_actions)[:10],
                    "admissible_sample": admissible[:20],
                },
            )
            return selected_action

        if admissible and query is not None:
            held_count = int(getattr(query, "held_relevant_count", 0) or 0)
            visible_match = bool(getattr(query, "goal_object_matches_visible", False))
            progress = str(getattr(query, "progress_state", "") or "")
            processed_norm = self._normalize_action_text(processed_action)
            admissible_by_norm = {self._normalize_action_text(cmd): cmd for cmd in admissible}
            processed_admissible = admissible_by_norm.get(processed_norm, "")
            is_concrete = self._is_concrete_alfworld_action(processed_action)

            embedded_action = self._gm3_embedded_phase_action(
                query=query,
                processed_action=processed_action,
                admissible=admissible,
                blocked_actions=blocked_actions,
            )
            if embedded_action:
                return _debug_return(embedded_action, "gm3_embedded_phase_action")

            object_guard = self._gm3_role_aware_object_guard_repair(
                query=query,
                processed_action=processed_action,
                admissible_actions=admissible,
                env_ref=env_ref,
                task_config=task_config or {},
                step_index=step_index,
            )
            if object_guard:
                if not self._gm3_action_is_blocked(object_guard, blocked_actions):
                    return _debug_return(object_guard, "gm3_object_guard")
                self._gm3_debug_append(
                    "action_hook_skip_blocked",
                    step_index=step_index,
                    payload={
                        "candidate": str(object_guard or ""),
                        "reason": "object_guard_examine_ok",
                    },
                )

            if held_count > 0:
                goal_roles = getattr(query, "goal_roles", {}) or {}
                target = self._gm3_base(str(goal_roles.get("object", "") or ""))
                tool = self._gm3_base(str(goal_roles.get("tool", "") or ""))
                destination = self._gm3_base(str(goal_roles.get("destination", "") or ""))
                preferred_actions: list[str] = []
                if tool and progress in {"carry_target", "process_target", "search_second"}:
                    preferred_actions.extend(
                        self._gm3_tool_priority_actions(target=target, tool=tool, admissible=admissible)
                    )
                if destination:
                    preferred_actions.extend(
                        self._gm3_destination_priority_actions(
                            target=target,
                            destination=destination,
                            admissible=admissible,
                        )
                    )
                preferred_actions = self._gm3_dedupe(preferred_actions, 3)
                if preferred_actions:
                    preferred = preferred_actions[0]
                    if not is_concrete or not processed_admissible:
                        return _debug_return(preferred, "gm3_held_phase_priority")
                    if self._normalize_action_text(processed_admissible).startswith(("go to ", "open ", "examine ")):
                        return _debug_return(preferred, "gm3_held_phase_override_search")

            if (
                admissible
                and query is not None
                and held_count <= 0
                and not visible_match
                and progress.startswith("search")
                and self._gm3_search_bias_repair_allowed(
                    query=query,
                    processed_action=processed_action,
                    processed_admissible=processed_admissible,
                )
            ):
                bias = [
                    item for item in self._gm3_search_bias_queue[:4]
                    if not self._gm3_action_is_blocked(str(item.get("action", "") or ""), blocked_actions)
                ][:1]
                if bias:
                    preferred = str(bias[0].get("action", "") or "").strip()
                    preferred_base = self._gm3_base(str(bias[0].get("base", "") or ""))
                    strong_bias = self._gm3_search_bias_is_strong(bias[0])
                    if preferred and strong_bias:
                        if not is_concrete or not processed_admissible:
                            return _debug_return(preferred, "gm3_search_bias_for_non_actionable_output")
                        if self._gm3_is_search_navigation_action(processed_admissible):
                            current_base = self._gm3_base(self._gm3_command_target_text(processed_admissible))
                            current_count = self._gm3_search_bias_exhausted_count(query, current_base)
                            preferred_count = self._gm3_search_bias_exhausted_count(query, preferred_base)
                            if (
                                self._normalize_action_text(processed_admissible) != self._normalize_action_text(preferred)
                                and current_base != preferred_base
                                and (current_count >= 2 or preferred_count < current_count)
                            ):
                                return _debug_return(preferred, "gm3_search_bias_override")
                elif self._gm3_search_bias_queue:
                    return _debug_return(final_action, "gm3_search_bias_all_blocked_examine_ok")

        self._gm3_debug_append(
            "action_hook_observe",
            step_index=step_index,
            payload={
                "raw_response": self._gm2_debug_text(str(raw_response or ""), limit=1200),
                "processed_action": str(processed_action or ""),
                "final_action": str(final_action or ""),
                "changed": False,
            },
        )
        return final_action

    def _retrieve_external_prompt_payload(self, **kargs) -> dict[str, list[str]]:
        if not self._external_enabled:
            return {"reference_cases": [], "execution_patterns": [], "insights": []}

        query = self._build_external_query(**kargs)
        step_index = int(kargs.get("step_index", 0) or 0)
        self._gm2_debug_last_step = step_index
        setting = str(self._graph_config_value("settings", "local_only") or "local_only")
        if query is None:
            note = self._external_error or "GraphMemory3 query unavailable for this task state."
            self._gm2_debug_append(
                "gm3_retrieve_unavailable",
                step_index=step_index,
                payload={"note": note, "setting": setting},
            )
            return {
                "reference_cases": [],
                "execution_patterns": [],
                "insights": [],
                "planner_notes": [f"[GM3] {note}"],
                "action_constraints": [],
                "repair_hints": [],
            }

        owner_scene = self._resolve_external_owner_scene(kargs.get("task_config"), kargs.get("env_ref"))
        local_memory = (
            self._external_empty_local(owner_scene)
            if setting in {"base", "global_only"}
            else self._external_local_memories.get(owner_scene, self._external_empty_local(owner_scene))
        )
        if setting not in {"base", "local_only"}:
            self._refresh_shared_global_memory()
        global_memory = (
            self._external_empty_global()
            if setting in {"base", "local_only"}
            else self._external_global_memory
        )
        bundle = self._external_retriever.retrieve(query, local_memory, global_memory)
        route = self._render_gm3_textloss_evidence(
            query=query,
            bundle=bundle,
            env_ref=kargs.get("env_ref"),
            local_memory=local_memory,
            global_memory=global_memory,
            setting=setting,
            owner_scene=owner_scene,
            step_index=step_index,
        )
        planner_notes = []
        if route.get("prompt"):
            planner_notes.append(
                "### GM3 MEMORY DECISION SUMMARY\n"
                + str(route["prompt"]).strip()
            )
        execution_patterns = self._gm3_scienceworld_execution_patterns(
            query=query,
            query_task=str(kargs.get("query_task") or ""),
        )

        self._gm2_debug_append(
            "gm3_retrieve",
            step_index=step_index,
            payload={
                "query": self._gm2_debug_query_snapshot(query),
                "setting": setting,
                "owner_scene": owner_scene,
                "local_memory_counts": self._gm2_debug_memory_counts(local_memory),
                "global_memory_counts": self._gm2_debug_memory_counts(global_memory),
                "bundle": self._gm2_debug_bundle_snapshot(bundle),
                "gm3_textloss": route.get("debug", {}),
                "rendered_prompt_sections": {
                    "planner_notes": [
                        self._gm2_debug_text(item, limit=3000)
                        for item in planner_notes[:3]
                    ],
                    "execution_patterns": [
                        self._gm2_debug_text(item, limit=3000)
                        for item in execution_patterns[:1]
                    ],
                    "counts": {
                        "planner_notes": len(planner_notes),
                        "execution_patterns": len(execution_patterns),
                    },
                },
            },
        )
        self._gm3_debug_append(
            "retrieve",
            step_index=step_index,
            payload={
                "query": self._gm2_debug_query_snapshot(query),
                "setting": setting,
                "owner_scene": owner_scene,
                "local_memory_counts": self._gm2_debug_memory_counts(local_memory),
                "global_memory_counts": self._gm2_debug_memory_counts(global_memory),
                "bundle": self._gm2_debug_bundle_snapshot(bundle),
                "gm3_textloss": route.get("debug", {}),
                "rendered_prompt": self._gm2_debug_text(
                    "\n\n".join(execution_patterns + planner_notes),
                    limit=5000,
                ),
            },
        )
        return {
            "reference_cases": [],
            "execution_patterns": execution_patterns,
            "insights": [],
            "planner_notes": planner_notes,
            "action_constraints": [],
            "repair_hints": [],
        }

    def _gm3_scienceworld_execution_patterns(self, *, query: Any, query_task: str) -> list[str]:
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
        if not task_family.startswith("scienceworld"):
            return []

        # ScienceWorld needs concrete procedural memory, but old valid-action
        # lists are too large and can pollute the current state's grounding.
        query_text = str(query_task or getattr(query, "goal", "") or "")
        successful, _, _ = super().retrieve_memory(
            query_task=query_text,
            successful_topk=8,
            failed_topk=0,
            insight_topk=0,
        )
        ranked_any_label = rank_messages_for_query(
            query_text,
            self.committed_messages,
            topk=16,
            label=None,
        )
        candidates: list[Any] = []
        for message in list(successful) + list(ranked_any_label):
            if message in candidates:
                continue
            if not self._gm3_message_is_scienceworld(message):
                continue
            if not self._gm3_scienceworld_message_has_progress(message):
                continue
            candidates.append(message)

        scored: list[tuple[float, Any]] = []
        for message in candidates:
            score = self._gm3_scienceworld_candidate_score(query=query, query_text=query_text, message=message)
            if score <= 0:
                continue
            scored.append((score, message))
        scored.sort(key=lambda item: item[0], reverse=True)

        patterns: list[str] = []
        for score, message in scored[:4]:
            pattern = self._gm3_scienceworld_progress_execution_pattern(message, match_score=score)
            if pattern:
                patterns.append(pattern)
            if len(patterns) >= 2:
                break
        return patterns

    def _gm3_scienceworld_progress_execution_pattern(self, message: Any, *, match_score: float = 0.0) -> str:
        description = self._gm3_scienceworld_clean_task_query(str(getattr(message, "task_description", "") or ""))
        trajectory = self._gm3_scienceworld_success_prefix(str(getattr(message, "task_trajectory", "") or ""))
        parts: list[str] = []
        if description:
            header = "### Similar task query"
            if match_score > 0:
                header += f" (match={match_score:.2f})"
            parts.append(header + ":\n" + description)
        if trajectory:
            parts.append("### Useful score-improving trajectory fragment:\n" + trajectory)
        return "\n\n".join(parts).strip()

    def _gm3_scienceworld_success_prefix(self, trajectory: str) -> str:
        segments = [seg.strip() for seg in str(trajectory or "").split("\n>") if seg.strip()]
        if not segments:
            return ""

        parsed: list[tuple[str, str, float | None, float | None, bool]] = []
        best_score = 0.0
        for segment in segments:
            action, _, observation = segment.partition("\n")
            action = re.sub(r"^\s*>\s*", "", action).strip()
            obs = observation.strip()
            score = self._gm3_scienceworld_score_from_text(obs)
            reward = self._gm3_scienceworld_reward_from_text(obs)
            is_failure = self._gm3_scienceworld_observation_is_failed_step(obs)
            if score is not None:
                best_score = max(best_score, score)
            parsed.append((action, obs, score, reward, is_failure))

        progress_steps: list[tuple[str, str, float | None, float | None]] = []
        setup_window: list[tuple[str, str, float | None, float | None]] = []
        emitted_actions: set[str] = set()
        previous_score = 0.0
        for action, obs, score, reward, is_failure in parsed:
            current_score = previous_score if score is None else score
            score_gain = current_score > previous_score
            reward_gain = reward is not None and reward > 0
            if is_failure:
                if score is not None:
                    previous_score = max(previous_score, score)
                continue
            if score_gain or reward_gain:
                for setup_action, setup_obs, setup_score, setup_reward in setup_window[-2:]:
                    setup_key = self._gm3_norm(setup_action)
                    if setup_key and setup_key not in emitted_actions:
                        progress_steps.append((setup_action, setup_obs, setup_score, setup_reward))
                        emitted_actions.add(setup_key)
                action_key = self._gm3_norm(action)
                if action_key and action_key not in emitted_actions:
                    progress_steps.append((action, self._gm3_scienceworld_clean_observation(obs), score, reward))
                    emitted_actions.add(action_key)
            elif action and not self._gm3_scienceworld_action_is_low_value(action):
                setup_window.append((action, self._gm3_scienceworld_clean_observation(obs), score, reward))
                setup_window = setup_window[-3:]
            if best_score > 0 and score is not None and score >= best_score:
                break
            if best_score <= 0 and reward is not None and reward > 0:
                break
            if score is not None:
                previous_score = max(previous_score, score)

        lines: list[str] = []
        for action, obs, score, reward in progress_steps[:8]:
            if not action:
                continue
            meta = []
            if score is not None:
                meta.append(f"score={score:g}")
            if reward is not None and reward > 0:
                meta.append(f"reward_delta={reward:g}")
            obs_line = f"Obs: {obs}" if obs else "Obs: progress observed."
            if meta:
                obs_line += " (" + ", ".join(meta) + ")"
            lines.append(f"{action}\n{obs_line}\n>")
        text = "\n".join(lines).strip()
        if len(text) > 1800:
            text = text[:1800].rstrip() + "\n..."
        return text

    def _gm3_scienceworld_candidate_score(self, *, query: Any, query_text: str, message: Any) -> float:
        current_text = self._gm3_scienceworld_clean_task_query(
            "\n".join(
                [
                    str(query_text or ""),
                    str(getattr(query, "goal", "") or ""),
                    str((getattr(query, "belief", {}) or {}).get("sw_task_name", "") or ""),
                    str((getattr(query, "dynamic_context", {}) or {}).get("sw_task_name", "") or ""),
                ]
            )
        )
        candidate_text = self._gm3_scienceworld_clean_task_query(
            "\n".join(
                [
                    str(getattr(message, "task_main", "") or ""),
                    str(getattr(message, "task_description", "") or ""),
                ]
            )
        )
        if not current_text or not candidate_text:
            return 0.0

        current_name = self._gm3_scienceworld_task_name(current_text)
        candidate_name = self._gm3_scienceworld_task_name(candidate_text)
        current_kind = self._gm3_scienceworld_task_kind(current_text)
        candidate_kind = self._gm3_scienceworld_task_kind(candidate_text)
        overlap = self._gm3_scienceworld_token_overlap(current_text, candidate_text)
        score = overlap * 20.0

        if current_name and candidate_name:
            if current_name == candidate_name:
                score += 80.0
            elif current_kind and candidate_kind and current_kind == candidate_kind:
                score += 30.0
            else:
                score -= 100.0
        elif current_kind and candidate_kind:
            score += 35.0 if current_kind == candidate_kind else -70.0

        current_group = str((getattr(query, "goal_roles", {}) or {}).get("scienceworld_family", "") or "").strip()
        if current_group and f"scienceworld_family={current_group}" in candidate_text:
            score += 12.0

        best = self._gm3_scienceworld_best_score_from_message(message)
        if best <= 0:
            score -= 25.0
        else:
            score += min(best, 100.0) / 10.0

        if "valid actions from the scienceworld engine" in candidate_text:
            score -= 35.0
        return score

    @staticmethod
    def _gm3_scienceworld_clean_task_query(text: str) -> str:
        cleaned = str(text or "").strip()
        markers = (
            "\nInitial observation:",
            "\nCurrent ScienceWorld score:",
            "\nValid actions from the ScienceWorld engine",
            "\n- Environment feedback",
            "\nEnvironment feedback",
            "\n### Detailed trajectory:",
        )
        for marker in markers:
            idx = cleaned.lower().find(marker.lower())
            if idx >= 0:
                cleaned = cleaned[:idx].strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        match = re.search(r"(Your task is to .+?)(?: Initial observation:| Current ScienceWorld score:| Valid actions from the ScienceWorld engine|$)", cleaned, flags=re.I)
        if match:
            cleaned = match.group(1).strip()
        if len(cleaned) > 420:
            cleaned = cleaned[:420].rstrip() + "..."
        return cleaned

    @staticmethod
    def _gm3_scienceworld_task_name(text: str) -> str:
        norm = re.sub(r"[_\-]+", " ", str(text or "").lower())
        patterns = (
            r"\bscienceworld\s+([a-z][a-z0-9 ]{1,40}?)(?:\s+v\d+|\s+train|\s+test|$)",
            r"\byour task is to\s+([a-z][a-z0-9 ]{1,40}?)(?:\s+the\b|\s+a\b|\s+an\b|\.|,|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, norm)
            if not match:
                continue
            candidate = re.sub(r"\b(?:determine whether|find|measure|test|use|the|a|an)\b", " ", match.group(1))
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate:
                first = candidate.split()[0]
                aliases = {
                    "boil": "boil",
                    "melt": "melt",
                    "freeze": "freeze",
                    "combust": "combust",
                    "conductivity": "conductivity",
                    "conduct": "conductivity",
                    "identify": "identify",
                    "classify": "classify",
                    "grow": "grow",
                }
                return aliases.get(first, first)
        for token in ("boil", "melt", "freeze", "combust", "conductivity", "conduct", "identify", "classify", "grow"):
            if re.search(rf"\b{re.escape(token)}\b", norm):
                return "conductivity" if token == "conduct" else token
        return ""

    @staticmethod
    def _gm3_scienceworld_task_kind(text: str) -> str:
        norm = str(text or "").lower()
        if any(token in norm for token in ("boil", "melt", "freeze", "combust", "state of matter")):
            return "state_change"
        if any(token in norm for token in ("conductivity", "conductive", "conduct electricity", "circuit", "electrical")):
            return "conductivity"
        if any(token in norm for token in ("unknown", "identify", "determine the substance", "which substance")):
            return "unknown_testing"
        if any(token in norm for token in ("living", "nonliving", "animal", "plant", "organism", "dominant", "recessive", "classification", "classify")):
            return "classification"
        if any(token in norm for token in ("inclined plane", "friction", "force", "mendelian", "genetics")):
            return "domain_specific"
        return ""

    @staticmethod
    def _gm3_scienceworld_token_overlap(a: str, b: str) -> float:
        stop = {
            "the", "a", "an", "to", "of", "and", "or", "is", "are", "in", "on", "for",
            "your", "task", "scienceworld", "current", "score", "valid", "actions",
            "from", "engine", "with", "then", "first", "also", "without",
        }
        ta = {tok for tok in re.findall(r"[a-z0-9]+", str(a or "").lower()) if tok not in stop and len(tok) > 2}
        tb = {tok for tok in re.findall(r"[a-z0-9]+", str(b or "").lower()) if tok not in stop and len(tok) > 2}
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))

    @staticmethod
    def _gm3_scienceworld_best_score_from_message(message: Any) -> float:
        text = "\n".join(
            [
                str(getattr(message, "task_description", "") or ""),
                str(getattr(message, "task_trajectory", "") or ""),
            ]
        )
        best = 0.0
        for match in re.finditer(r"\b(?:current scienceworld score|best_score|score)\s*[=:]\s*(-?\d+(?:\.\d+)?)", text, flags=re.I):
            try:
                best = max(best, float(match.group(1)))
            except Exception:
                continue
        return best

    @staticmethod
    def _gm3_scienceworld_clean_observation(text: str) -> str:
        obs = str(text or "").strip()
        obs = re.split(r"\bCurrent ScienceWorld score:", obs, maxsplit=1, flags=re.I)[0].strip()
        obs = re.split(r"\bValid actions from the ScienceWorld engine\b", obs, maxsplit=1, flags=re.I)[0].strip()
        obs = re.sub(r"\s+", " ", obs).strip()
        if len(obs) > 260:
            obs = obs[:260].rstrip() + "..."
        return obs

    @staticmethod
    def _gm3_scienceworld_action_is_low_value(action: str) -> bool:
        norm = re.sub(r"\s+", " ", str(action or "").strip().lower())
        return norm.startswith(("think", "look around")) or norm in {"inventory", "wait"}

    @staticmethod
    def _gm3_scienceworld_score_from_text(text: str) -> float | None:
        match = re.search(r"\bCurrent ScienceWorld score:\s*(-?\d+(?:\.\d+)?)", str(text or ""), flags=re.I)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _gm3_scienceworld_reward_from_text(text: str) -> float | None:
        match = re.search(r"\bLast reward delta:\s*(-?\d+(?:\.\d+)?)", str(text or ""), flags=re.I)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _gm3_scienceworld_observation_is_failed_step(text: str) -> bool:
        norm = re.sub(r"\s+", " ", str(text or "").strip().lower())
        return any(
            phrase in norm
            for phrase in (
                "no known action matches",
                "doesn't appear",
                "do not know how to",
                "can't do that",
                "cannot do that",
                "nothing happens",
            )
        )

    @staticmethod
    def _gm3_message_is_scienceworld(message: Any) -> bool:
        task_main = str(getattr(message, "task_main", "") or "").lower()
        task_description = str(getattr(message, "task_description", "") or "").lower()
        return "scienceworld" in task_main or "scienceworld" in task_description

    @staticmethod
    def _gm3_scienceworld_message_has_progress(message: Any) -> bool:
        if bool(getattr(message, "label", False)):
            return True
        text = "\n".join(
            [
                str(getattr(message, "task_description", "") or ""),
                str(getattr(message, "task_trajectory", "") or ""),
            ]
        )
        for match in re.finditer(r"\b(?:current scienceworld score|best_score|score)\s*[=:]\s*(-?\d+(?:\.\d+)?)", text, flags=re.I):
            try:
                if float(match.group(1)) > 0:
                    return True
            except Exception:
                continue
        return False

    def _render_gm3_textloss_evidence(
        self,
        *,
        query: Any,
        bundle: Any,
        env_ref: Any,
        local_memory: Any,
        global_memory: Any,
        setting: str,
        owner_scene: str,
        step_index: int,
    ) -> dict[str, Any]:
        goal_roles = getattr(query, "goal_roles", {}) or {}
        dynamic = getattr(query, "dynamic_context", {}) or {}
        admissible = [
            str(cmd).strip()
            for cmd in (getattr(env_ref, "last_admissible_commands", []) or [])
            if str(cmd).strip()
        ] if env_ref is not None else []
        visible = [str(x) for x in dynamic.get("visible_objects", []) or [] if str(x).strip()]
        held = [str(x) for x in dynamic.get("held_objects", []) or [] if str(x).strip()]
        exhausted = [str(x) for x in dynamic.get("exhausted_locations", []) or [] if str(x).strip()]
        blocked_actions = self._gm3_recent_examine_ok_actions(env_ref)

        candidates = self._gm3_candidate_sections(
            query=query,
            bundle=bundle,
            local_memory=local_memory,
            global_memory=global_memory,
            admissible=admissible,
            visible=visible,
            held=held,
            exhausted=exhausted,
            blocked_actions=blocked_actions,
            setting=setting,
            owner_scene=owner_scene,
            env_ref=env_ref,
        )
        routed = self._gm3_textloss_route(
            candidates=candidates,
            query=query,
            admissible=admissible,
            visible=visible,
            held=held,
            exhausted=exhausted,
        )
        selected = routed["selected"]
        if not selected:
            return {"prompt": "", "debug": routed}
        task_family_for_gate = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
        memory_slots = {"local_grounding", "source_roles", "global_workflow", "failure_avoidance"}
        if task_family_for_gate.startswith("bfcl"):
            memory_slots.add("phase_policy")
        if not any(str(section.get("slot", "") or "") in memory_slots for section in selected):
            routed["no_graph_memory_prompt"] = True
            return {"prompt": "", "debug": routed}

        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = self._gm3_base(str(goal_roles.get("object", "") or ""))
        tool = self._gm3_base(str(goal_roles.get("tool", "") or ""))
        destination = self._gm3_base(str(goal_roles.get("destination", "") or ""))
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
        source_evidence_table = self._gm3_source_evidence_table(
            local_memory=local_memory,
            global_memory=global_memory,
            target=target,
            tool=tool,
            destination=destination,
            task_family=task_family,
            admissible=admissible,
            exhausted=exhausted,
            blocked_actions=blocked_actions,
            setting=setting,
            owner_scene=owner_scene,
        )
        source_evidence_table = self._gm3_merge_source_evidence_table(
            source_evidence_table,
            self._gm3_previous_success_source_rows(
                query=query,
                env_ref=env_ref,
                admissible=admissible,
                blocked_actions=blocked_actions,
            ),
        )
        supported_source_scores = {
            str(row.get("source_base", "")): float(row.get("final_score", 0.0) or 0.0)
            for row in source_evidence_table
            if float(row.get("final_score", 0.0) or 0.0) > 0.05
        }
        routed["source_evidence_table"] = [
            {
                "source_base": row.get("source_base"),
                "source_role": row.get("source_role"),
                "evidence_level": row.get("evidence_level"),
                "positive_score": round(float(row.get("positive_score", 0.0) or 0.0), 4),
                "negative_score": round(float(row.get("negative_score", 0.0) or 0.0), 4),
                "exhausted_penalty": round(float(row.get("exhausted_penalty", 0.0) or 0.0), 4),
                "final_score": round(float(row.get("final_score", 0.0) or 0.0), 4),
                "admissible_actions": list(row.get("admissible_actions", []) or [])[:3],
                "evidence": list(row.get("evidence", []) or [])[:4],
            }
            for row in source_evidence_table[:20]
        ]

        signature = self._gm3_prompt_signature(query=query, selected=selected)
        repeated = bool(signature and signature == self._gm3_last_prompt_signature and step_index > 1)
        self._gm3_last_prompt_signature = signature

        priority_items: list[str] = []
        for slot in ("local_grounding", "source_roles"):
            priority_items.extend(
                self._gm3_concrete_priority_items(
                    self._gm3_items_for_slot(selected, slot, limit=4),
                    admissible=admissible,
                    blocked_actions=blocked_actions,
                    limit=2,
                )
            )
            if len(priority_items) >= 2:
                break
        if not priority_items and self._gm3_selected_slot_has_real_item(selected, "failure_avoidance"):
            priority_items.extend(
                self._gm3_failure_alternative_actions(
                    admissible=admissible,
                    exhausted=exhausted,
                    blocked_actions=blocked_actions,
                    supported_source_scores=supported_source_scores,
                    limit=2,
                )
            )
        if not priority_items:
            priority_items.extend(self._gm3_source_table_priority_actions(source_evidence_table, limit=2))
        self._gm3_search_bias_queue = self._gm3_search_bias_candidates(source_evidence_table, limit=4)
        routed["search_bias_queue"] = self._gm2_debug_jsonable(self._gm3_search_bias_queue[:6])

        should_emit, emit_reason = self._gm3_should_emit_summary(
            query=query,
            selected=selected,
            priority_items=priority_items,
            admissible=admissible,
            exhausted=exhausted,
            supported_source_scores=supported_source_scores,
        )
        routed["summary_emit_reason"] = emit_reason
        if not should_emit:
            routed["summary_suppressed"] = True
            return {"prompt": "", "debug": routed}

        summary = self._gm3_render_decision_summary(
            query=query,
            selected=selected,
            priority_items=priority_items,
            admissible=admissible,
            held=held,
            visible=visible,
            exhausted=exhausted,
        )
        optimized = self._gm3_textgrad_optimize_prompt(
            draft_prompt=summary,
            query=query,
            selected=selected,
            routed=routed,
            admissible=admissible,
            visible=visible,
            held=held,
            exhausted=exhausted,
            step_index=step_index,
            repeated_prompt=False,
        )
        if optimized.get("debug"):
            routed["textgrad_prompt_optimization"] = optimized["debug"]
        summary = str(optimized.get("prompt") or summary)
        routed["prompt_signature"] = signature
        routed["prompt_repeated"] = repeated
        return {"prompt": summary, "debug": routed}

    def _gm3_should_emit_summary(
        self,
        *,
        query: Any,
        selected: list[dict[str, Any]],
        priority_items: list[str],
        admissible: list[str],
        exhausted: list[str],
        supported_source_scores: dict[str, float] | None = None,
    ) -> tuple[bool, str]:
        """Avoid injecting weak memory as if it were decision support.

        GM3 should speak when it can add something actionable or clearly
        transferable. A source-type prior without a current admissible action is
        useful context, but by itself it is too weak to justify extra prompt.
        """
        progress = str(getattr(query, "progress_state", "") or "")
        held_count = int(getattr(query, "held_relevant_count", 0) or 0)
        visible_match = bool(getattr(query, "goal_object_matches_visible", False))
        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = self._gm3_base(str(goal_roles.get("object", "") or ""))
        tool = self._gm3_base(str(goal_roles.get("tool", "") or ""))
        destination = self._gm3_base(str(goal_roles.get("destination", "") or ""))
        slots = {str(section.get("slot", "") or "") for section in selected}
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))

        if task_family.startswith("bfcl") and "phase_policy" in slots:
            if progress in {"need_tool_call", "tool_result_observed", "invalid_action", "tool_error", "checker_failed"}:
                return True, "bfcl_phase_policy"

        if task_family.startswith("pddl"):
            if priority_items:
                for item in priority_items[:3]:
                    mapped = self._gm3_first_admissible_action_in_text(str(item or ""), admissible)
                    if mapped and self._gm3_pddl_action_advances_unsatisfied_goal(query, mapped):
                        return True, "pddl_memory_maps_to_goal_action"
                    if (
                        mapped
                        and progress == "search_preconditions"
                        and not self._gm3_pddl_is_meta_action(mapped)
                    ):
                        if self._gm3_is_gpt4omini_model():
                            continue
                        return True, "pddl_memory_maps_to_precondition_action"
            if self._gm3_pddl_current_action_hint(query, admissible):
                return True, "pddl_current_state_action_grounding"
            if "failure_avoidance" in slots and self._gm3_selected_slot_has_real_item(selected, "failure_avoidance"):
                return True, "pddl_failure_memory"
            if "global_workflow" in slots:
                global_line = self._gm3_summary_slot(selected, "global_workflow", default="")
                mapped = self._gm3_first_admissible_action_in_text(global_line, admissible)
                if mapped and self._gm3_pddl_action_advances_unsatisfied_goal(query, mapped):
                    return True, "pddl_global_workflow_grounded"
                if (
                    mapped
                    and progress == "search_preconditions"
                    and not self._gm3_pddl_is_meta_action(mapped)
                ):
                    if self._gm3_is_gpt4omini_model():
                        return False, "pddl_gpt4omini_skip_setup_only_global_workflow"
                    return True, "pddl_global_precondition_workflow_grounded"
                return False, "pddl_global_workflow_not_current_goal_grounded"

        if priority_items:
            return True, "current_memory_maps_to_admissible_action"
        if visible_match and self._gm3_take_priority_actions(target=target, admissible=admissible):
            return True, "visible_target_has_concrete_take_action"
        if held_count > 0:
            process_actions = self._gm3_tool_priority_actions(target=target, tool=tool, admissible=admissible) if tool else []
            delivery_actions = self._gm3_destination_priority_actions(target=target, destination=destination, admissible=admissible) if destination else []
            if process_actions or delivery_actions:
                return True, "held_target_has_concrete_process_or_delivery_action"

        exhausted_counts = self._gm3_exhausted_base_counts(exhausted)
        if "failure_avoidance" in slots and progress.startswith("search") and max(exhausted_counts.values(), default=0) >= 3:
            if self._gm3_failure_alternative_actions(
                admissible=admissible,
                exhausted=exhausted,
                supported_source_scores=supported_source_scores or {},
                limit=1,
            ):
                return True, "failure_memory_has_supported_alternative_action"
            return False, "failure_only_without_memory_supported_alternative"

        if "global_workflow" in slots and self._gm3_selected_slot_has_real_item(selected, "global_workflow"):
            return True, "global_workflow_matches_task_phase"

        return False, "only_weak_or_non_actionable_memory"

    def _gm3_candidate_sections(
        self,
        *,
        query: Any,
        bundle: Any,
        local_memory: Any,
        global_memory: Any,
        admissible: list[str],
        visible: list[str],
        held: list[str],
        exhausted: list[str],
        blocked_actions: set[str] | None = None,
        setting: str,
        owner_scene: str,
        env_ref: Any,
    ) -> list[dict[str, Any]]:
        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = self._gm3_base(str(goal_roles.get("object", "") or ""))
        tool = self._gm3_base(str(goal_roles.get("tool", "") or ""))
        destination = self._gm3_base(str(goal_roles.get("destination", "") or ""))
        progress = str(getattr(query, "progress_state", "") or "")
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))

        sections: list[dict[str, Any]] = []

        phase_items = self._gm3_phase_items(query=query, target=target, tool=tool, destination=destination)
        if phase_items:
            sections.append({"slot": "phase_policy", "title": "Phase policy from current graph state", "items": phase_items, "source": "state"})

        global_items = self._gm3_global_workflow_items(
            query=query,
            bundle=bundle,
            global_memory=global_memory,
            target=target,
            tool=tool,
            destination=destination,
            task_family=task_family,
            owner_scene=owner_scene,
            admissible=admissible,
            progress=progress,
        )
        if setting not in {"base", "local_only"} and global_items:
            sections.append({"slot": "global_workflow", "title": "Global transferable workflow", "items": global_items[:3], "source": "global"})

        local_items = self._gm3_local_grounding_items(
            query=query,
            bundle=bundle,
            local_memory=local_memory,
            target=target,
            destination=destination,
            task_family=task_family,
            admissible=admissible,
            visible=visible,
            exhausted=exhausted,
            blocked_actions=blocked_actions,
            progress=progress,
        )
        if setting not in {"base", "global_only"} and local_items:
            sections.append({"slot": "local_grounding", "title": "Local graph grounding", "items": local_items[:4], "source": "local"})

        source_table = self._gm3_source_evidence_table(
            local_memory=local_memory,
            global_memory=global_memory,
            admissible=admissible,
            exhausted=exhausted,
            target=target,
            tool=tool,
            destination=destination,
            task_family=task_family,
            setting=setting,
            owner_scene=owner_scene,
            blocked_actions=blocked_actions,
        )
        source_table = self._gm3_merge_source_evidence_table(
            source_table,
            self._gm3_previous_success_source_rows(
                query=query,
                env_ref=env_ref,
                admissible=admissible,
                blocked_actions=blocked_actions,
            ),
        )
        source_items = self._gm3_source_role_items(source_table=source_table, progress=progress)
        if source_items:
            sections.append({"slot": "source_roles", "title": "Current graph search priority", "items": source_items[:3], "source": "mixed"})

        failure_items = self._gm3_failure_items(
            bundle=bundle,
            target=target,
            progress=progress,
            exhausted=exhausted,
            query=query,
            local_memory=local_memory,
            global_memory=global_memory,
            task_family=task_family,
            owner_scene=owner_scene,
            admissible=admissible,
        )
        if failure_items:
            sections.append({"slot": "failure_avoidance", "title": "Failure patterns to avoid", "items": failure_items[:3], "source": "mixed"})

        return sections

    def _gm3_phase_items(self, *, query: Any, target: str, tool: str, destination: str) -> list[str]:
        return self._gm3_prompt_style(query=query).phase_items(
            self,
            query=query,
            target=target,
            tool=tool,
            destination=destination,
        )

    def _gm3_prompt_style(self, *, query: Any = None, task_family: str = ""):
        return prompt_style_for_query(query, task_family=task_family)

    def _gm3_global_workflow_items(
        self,
        *,
        query: Any,
        bundle: Any,
        global_memory: Any,
        target: str,
        tool: str,
        destination: str,
        task_family: str,
        owner_scene: str,
        admissible: list[str],
        progress: str = "",
    ) -> list[str]:
        raw_items = (
            list(getattr(bundle, "global_task_plan_items", []) or [])
            + list(getattr(bundle, "global_promoted_contribution", []) or [])
        )
        if task_family.startswith("fever"):
            raw_items += list(getattr(bundle, "global_promoted_items", []) or [])
        raw_items += list(getattr(bundle, "global_items", []) or [])
        rendered: list[str] = []
        for item in raw_items:
            if not self._gm3_keep_global_item_for_owner(item, global_memory=global_memory, owner_scene=owner_scene):
                continue
            text = self._gm3_clean(str(getattr(item, "summary", "") or ""))
            if not text or self._gm3_is_concrete_location_text(text):
                continue
            if task_family.startswith("fever"):
                line = self._gm3_render_fever_workflow_item(
                    item,
                    query=query,
                    text=text,
                    admissible=admissible,
                    progress=progress,
                    source_label="Global",
                )
                if line:
                    rendered.append(line)
                continue
            if self._gm3_is_failure_text(text):
                continue
            norm = self._gm3_norm(text)
            if not self._gm3_prompt_style(query=None, task_family=task_family).keep_global_text(
                self,
                text=text,
                norm_text=norm,
                task_family=task_family,
            ):
                continue
            if self._gm3_should_suppress_fever_workflow_for_phase(
                text=text,
                task_family=task_family,
                progress=progress,
            ):
                continue
            if not self._gm3_fever_workflow_fits_phase(
                text=text,
                task_family=task_family,
                progress=progress,
            ):
                continue
            if task_family.startswith("pddl"):
                if "check valid actions" in self._gm3_norm(text):
                    continue
                item_family = self._gm3_item_task_family(item)
                current_sub = task_family.split(":", 1)[-1] if ":" in task_family else ""
                item_sub = item_family.split(":", 1)[-1] if ":" in item_family else ""
                is_cross_domain = bool(current_sub and item_sub and item_sub != current_sub)
                mapped = self._gm3_first_admissible_action_in_text(text, admissible)
                # "check valid actions" is a universal PDDL meta-action that is
                # always admissible but never advances a goal literal.  Gripper
                # repair items that contain "check_valid_actions" in their text
                # would always map to it, flooding every domain with a useless
                # cross-domain grounding signal.  Strip it out here.
                if mapped and self._gm3_norm(mapped) in {"check_valid_actions", "check valid actions", "look", "look around"}:
                    mapped = ""
                if mapped and self._gm3_pddl_action_advances_unsatisfied_goal(query, mapped):
                    rendered.append(
                        f"Global PDDL memory maps to current valid operator `{mapped}`; use it only if it advances an unsatisfied goal literal."
                    )
                elif mapped and progress == "search_preconditions" and not self._gm3_pddl_is_meta_action(mapped):
                    if self._gm3_is_gpt4omini_model():
                        continue
                    rendered.append(
                        f"Global PDDL memory maps to current valid setup operator `{mapped}`; use it only as a precondition step toward the remaining goal."
                    )
                elif mapped:
                    continue
                elif not is_cross_domain:
                    rendered.append(
                        "Global PDDL transfer: reuse abstract planning discipline only "
                        "(satisfy operator preconditions, use current valid actions, and never copy arguments from another domain)."
                    )
                else:
                    # Cross-domain, un-groundable: still emit a domain-agnostic cue.
                    # This avoids triggering the all-none early-return gate in
                    # _gm3_render_decision_summary, which would suppress even the
                    # phase/state summary the LLM needs to navigate the task.
                    rendered.append(
                        "Global PDDL transfer: reuse abstract planning discipline only "
                        "(satisfy operator preconditions, use current valid actions, and never copy arguments from another domain)."
                    )
                continue
            if task_family.startswith("bfcl"):
                if "finishturn" in norm or "closure" in norm:
                    rendered.append(
                        "Global BFCL workflow: after a tool result answers the current user turn, use FinishTurn[]; do not copy old tool arguments."
                    )
                elif "failure" in norm or "blocked" in norm or "avoid" in norm:
                    rendered.append(
                        "Global BFCL recovery: avoid repeating failed or invalid tool calls; repair arguments from the current tool schema."
                    )
                else:
                    rendered.append(
                        "Global BFCL transfer: reuse only the tool-use discipline (minimal valid call, observe result, then FinishTurn[] when satisfied)."
                    )
                continue
            marker_ok = any(
                marker in norm
                for marker in (
                    "workflow",
                    "plan",
                    "->",
                    "take(",
                    "heat(",
                    "cool(",
                    "clean(",
                    "move(",
                    "repeat acquire",
                )
            )
            if not marker_ok:
                style = self._gm3_prompt_style(query=None, task_family=task_family)
                marker_ok = style.name == "fever" and any(
                    marker in norm for marker in ("fever evidence", "fever lookup", "fever recovery", "failure avoidance")
                )
            if not marker_ok:
                continue
            rendered.append(self._gm3_bind_slots(text, target=target, tool=tool, destination=destination))
        return self._gm3_dedupe(rendered, 4)

    def _gm3_local_grounding_items(
        self,
        *,
        query: Any,
        bundle: Any,
        local_memory: Any,
        target: str,
        destination: str,
        task_family: str,
        admissible: list[str],
        visible: list[str],
        exhausted: list[str],
        blocked_actions: set[str] | None = None,
        progress: str = "",
    ) -> list[str]:
        rendered: list[str] = []
        admissible_norm = {self._gm3_norm(cmd): cmd for cmd in admissible}
        visible_norm = " ".join(self._gm3_norm(x) for x in visible)
        if task_family.startswith("pddl"):
            hint, literal = self._gm3_pddl_current_action_hint_with_literal(query, admissible)
            if hint:
                detail = f" for unsatisfied goal `{self._gm3_shorten(literal, 70)}`" if literal else ""
                rendered.append(
                    f"current PDDL state/action grounding{detail}: admissible operator `{hint}` directly matches the remaining goal."
                )
        if target and target in visible_norm:
            for command_norm, command in admissible_norm.items():
                if command_norm.startswith("take ") and target in command_norm:
                    rendered.append(f"current observation grounds target_object={target}; admissible action `{command}` is phase-correct.")
                    break

        artifacts = getattr(local_memory, "artifacts_by_id", {}) or {}
        for artifact in (artifacts.values() if isinstance(artifacts, dict) else artifacts):
            payload = getattr(artifact, "payload", {}) or {}
            anchor = getattr(artifact, "anchor", {}) or {}
            if str(payload.get("pattern_kind", "") or "") != "scene_relation":
                continue
            if str(payload.get("relation_kind", "") or "") != "object_location_prior":
                continue
            if str(payload.get("object_role", "") or "") != "target_object":
                continue
            goal_sig = self._gm3_norm(str(anchor.get("goal_signature", "") or ""))
            if target and not goal_sig.startswith(f"{target}->"):
                continue
            source_instance = str(payload.get("source_instance", "") or "")
            source_base = str(payload.get("source_base", "") or "")
            source = source_instance or source_base
            if not source:
                continue
            source_instance_norm = self._gm3_norm(source_instance)
            source_base_norm = self._gm3_base(source_base or source_instance)
            if source_instance_norm and any(source_instance_norm == self._gm3_norm(x) for x in exhausted):
                continue
            mapped = self._gm3_admissible_source_action(source_instance, admissible) if source_instance else ""
            if self._gm3_action_is_blocked(mapped, blocked_actions):
                mapped = ""
            if mapped and source_instance_norm and source_instance_norm in self._gm3_norm(mapped):
                rendered.append(f"local graph links target_object={target} to source `{source_instance.replace('_', ' ')}`; currently admissible grounding is `{mapped}`.")
                continue
            if source_base_norm:
                actions = self._gm3_admissible_source_base_actions(
                    source_base_norm,
                    admissible,
                    exhausted,
                    blocked_actions=blocked_actions,
                    limit=2,
                )
                if actions:
                    queue = " -> ".join(f"`{action}`" for action in actions)
                    rendered.append(f"local graph links target_object={target} to source type `{source_base_norm}`; current admissible queue: {queue}.")
        for artifact in (artifacts.values() if isinstance(artifacts, dict) else artifacts):
            payload = getattr(artifact, "payload", {}) or {}
            anchor = getattr(artifact, "anchor", {}) or {}
            domain = self._gm3_norm(str(anchor.get("domain", "") or payload.get("domain", "") or ""))
            if domain not in {"pddl", "fever", "bfcl_mt", "bfcl"} and not task_family.startswith(("pddl_", "fever_", "bfcl_")):
                continue
            artifact_family = self._gm3_norm(str(anchor.get("task_family", "") or ""))
            family_matches = not task_family or not artifact_family or artifact_family == task_family
            if (
                not family_matches
                and domain == "fever"
                and task_family.startswith("fever")
                and (artifact_family in {"fever", "fever_claim_verification"} or artifact_family.startswith("fever_"))
            ):
                family_matches = True
            if not family_matches:
                continue
            pattern_kind = self._gm3_norm(str(payload.get("pattern_kind", "") or anchor.get("pattern_kind", "") or ""))
            if pattern_kind in {"scene_relation", "source_type_prior"}:
                continue
            action_patterns = [
                str(item).strip()
                for item in (
                    list(payload.get("action_patterns", []) or [])
                    + list(payload.get("repair_patterns", []) or [])
                    + list(payload.get("avoid_patterns", []) or [])
                )
                if str(item).strip()
            ]
            mapped = ""
            for pattern in action_patterns:
                mapped = self._gm3_first_admissible_action_in_text(pattern, admissible)
                if mapped:
                    break
                pattern_norm = self._gm3_norm(pattern)
                mapped = next((cmd for cmd in admissible if self._gm3_norm(cmd) == pattern_norm), "")
                if mapped:
                    break
            text = self._gm3_clean(str(getattr(artifact, "summary", "") or ""))
            if not text:
                continue
            if domain == "fever" or task_family.startswith("fever"):
                line = self._gm3_render_fever_workflow_item(
                    artifact,
                    query=query,
                    text=text,
                    admissible=admissible,
                    progress=progress,
                    source_label="Local",
                )
                if line:
                    rendered.append(line)
                continue
            if domain.startswith("bfcl") or task_family.startswith("bfcl"):
                if "finishturn" in self._gm3_norm(text) or "closure" in pattern_kind:
                    rendered.append(
                        "Local BFCL graph: consider FinishTurn[] after the current tool output satisfies this turn; never repeat the same completed call."
                    )
                elif mapped:
                    rendered.append(
                        f"Local BFCL graph saw workflow shape `{mapped}` before; reuse the shape only if it matches this user turn and current tool schema."
                    )
                elif pattern_kind in {"workflow", "closure", "rule", "precondition", "failure", "repair"}:
                    rendered.append(
                        "Local BFCL graph: use prior tool-call workflow only as a phase cue; choose arguments from the current user request."
                    )
                continue
            if self._gm3_should_suppress_fever_workflow_for_phase(
                text=text,
                task_family=task_family,
                progress=progress,
            ):
                continue
            if not self._gm3_fever_workflow_fits_phase(
                text=text,
                task_family=task_family,
                progress=progress,
            ):
                continue
            if not self._gm3_prompt_style(query=None, task_family=task_family or domain).keep_local_artifact_text(
                self,
                domain=domain,
                task_family=task_family,
                text=text,
                norm_text=self._gm3_norm(text),
            ):
                continue
            if mapped:
                if domain.startswith("pddl"):
                    if self._gm3_pddl_is_meta_action(mapped):
                        continue
                    if not self._gm3_pddl_action_advances_unsatisfied_goal(query, mapped):
                        if progress == "search_preconditions":
                            if self._gm3_is_gpt4omini_model():
                                continue
                            rendered.append(
                                f"Local PDDL graph maps prior precondition workflow to current valid setup operator `{mapped}`; use it only if it prepares an unsatisfied goal literal."
                            )
                            continue
                        if hint := self._gm3_pddl_current_action_hint(query, admissible):
                            rendered.append(
                                f"Local PDDL graph cannot reuse old operator `{mapped}` for the current unsatisfied goal; current grounded operator is `{hint}`."
                            )
                        continue
                    rendered.append(
                        f"Local PDDL graph maps prior workflow to current valid operator `{mapped}`; use it only if it advances an unsatisfied goal literal."
                    )
                    continue
                rendered.append(f"{text} Current admissible grounding: `{mapped}`.")
            elif domain.startswith("pddl"):
                if hint := self._gm3_pddl_current_action_hint(query, admissible):
                    rendered.append(
                        f"Local PDDL graph cannot copy old arguments; current grounded operator candidate is `{hint}` because it overlaps unsatisfied goal literals."
                    )
                continue
            elif pattern_kind in {"workflow", "closure", "rule", "precondition"}:
                rendered.append(text)
        if task_family.startswith("fever"):
            for item in (
                list(getattr(bundle, "local_items", []) or [])
                + list(getattr(bundle, "workflow_items", []) or [])
                + list(getattr(bundle, "repair_items", []) or [])
            ):
                if not str(getattr(item, "source", "") or "").startswith("local"):
                    continue
                if not self._gm3_fever_item_has_positive_transfer_signal(item):
                    continue
                text = self._gm3_clean(str(getattr(item, "summary", "") or ""))
                if not text:
                    continue
                line = self._gm3_render_fever_bundle_hint(
                    text=text,
                    query=query,
                    admissible=admissible,
                    progress=progress,
                )
                if line:
                    rendered.append(line)
        return self._gm3_dedupe(rendered, 5)

    def _gm3_render_fever_workflow_item(
        self,
        item: Any,
        *,
        query: Any,
        text: str,
        admissible: list[str],
        progress: str,
        source_label: str,
        allow_claim_type_fallback: bool = False,
    ) -> str:
        payload = getattr(item, "payload", {}) or {}
        anchor = getattr(item, "anchor", {}) or {}
        dynamic = getattr(item, "dynamic", {}) or {}
        item_id = str(
            getattr(item, "candidate_id", "")
            or getattr(item, "artifact_id", "")
            or getattr(item, "rule_id", "")
            or ""
        )
        norm = self._gm3_norm(text)
        item_id_norm = self._gm3_norm(item_id)
        pattern_blob = " ".join(
            part
            for part in (
                norm,
                item_id_norm,
                self._gm3_norm(str(dynamic.get("fever_pattern", "") or "")) if isinstance(dynamic, dict) else "",
                self._gm3_norm(str(payload.get("fever_pattern", "") or "")) if isinstance(payload, dict) else "",
            )
            if part
        )
        fever_pattern = self._gm3_norm(str(payload.get("fever_pattern", "") or dynamic.get("fever_pattern", "") or ""))
        if not fever_pattern:
            if "content_search_route" in item_id_norm or "content search route" in norm:
                fever_pattern = "content_search_route"
            elif "evidence_sufficiency_stop" in item_id_norm or "stop rule" in norm:
                fever_pattern = "evidence_sufficiency_stop"
            elif "no_results_recovery" in item_id_norm or "recovery" in norm:
                fever_pattern = "no_results_recovery"
            elif "premature_finish_failure" in item_id_norm or "failure" in norm:
                fever_pattern = "premature_finish_failure"
        if "fever" not in pattern_blob and not fever_pattern:
            return ""
        ctx = self._gm3_fever_claim_context(query)
        item_claim_type = self._gm3_norm(
            str(
                payload.get("claim_type")
                or dynamic.get("claim_type")
                or anchor.get("claim_type")
                or self._gm3_fever_claim_type_from_text(text)
                or ""
            )
        )
        current_claim_type = self._gm3_norm(ctx["claim_type"])
        if item_claim_type and item_claim_type not in {"claim_verification"} and item_claim_type != current_claim_type:
            if not allow_claim_type_fallback:
                return ""
            if "premature_finish_failure" in fever_pattern or "failure" in fever_pattern:
                return ""
        if progress == "ready_finish":
            return ""
        is_recovery = "no_results_recovery" in fever_pattern or "recovery" in pattern_blob
        is_premature_finish = "premature_finish_failure" in fever_pattern or "failure" in fever_pattern
        is_stop_rule = "evidence_sufficiency_stop" in fever_pattern or "stop rule" in norm
        is_content_route = "content_search_route" in fever_pattern or "content search route" in norm
        if not (is_recovery or is_premature_finish or is_stop_rule or is_content_route):
            return ""

        if is_content_route and source_label.lower() != "global":
            return ""

        if is_content_route and progress == "need_search":
            action = self._gm3_fever_grounded_action("search", ctx["entity"], admissible)
            if not action:
                return ""
            search_role = str(
                payload.get("search_role", "")
                or dynamic.get("search_role", "")
                or self._gm3_fever_value_from_item_id(item_id, "search_role")
                or "claim subject"
            )
            lookup_hint = self._gm3_fever_payload_keyword(payload) or ctx["lookup_keyword"]
            tail = f"; if the page is broad, lookup `{lookup_hint}` next" if lookup_hint else ""
            return (
                f"{source_label} FEVER content search route ({ctx['claim_type']}): successful routes search the "
                f"current claim's {search_role}; use `{action}`{tail}, then finish only from evidence."
            )

        if is_content_route:
            return ""

        if is_recovery and progress in {"search_failed", "invalid_action"}:
            action = self._gm3_fever_grounded_action("search", ctx["entity"], admissible)
            if not action:
                return ""
            return (
                f"{source_label} FEVER workflow ({ctx['claim_type']}): recover with a shorter current-claim query "
                f"using `{action}`; finish NOT ENOUGH INFO only after evidence search is exhausted."
            )

        if is_premature_finish and progress in {"search_failed", "invalid_action"}:
            return (
                f"{source_label} FEVER correction ({ctx['claim_type']}): avoid Finish before evidence settles the "
                "claim; use the current page/search result, not memory, to decide the label."
            )

        if is_stop_rule and progress == "need_lookup_or_finish":
            return (
                f"{source_label} FEVER stop rule ({ctx['claim_type']}): if the current evidence directly supports, "
                "contradicts, or fails to contain the relation, Finish from that evidence instead of forcing another Lookup."
            )
        return ""

    def _gm3_render_fever_bundle_hint(
        self,
        *,
        text: str,
        query: Any,
        admissible: list[str],
        progress: str,
    ) -> str:
        """Sanitize local FEVER bundle snippets by rebinding them to this claim.

        Local retrieval can surface old action examples such as
        ``try search(query=old_entity)``.  Those examples are useful only as a
        workflow cue, so this method strips old arguments and emits the current
        Search/Lookup action instead.
        """
        norm = self._gm3_norm(text)
        is_recovery = any(marker in norm for marker in ("no results", "not found", "reformulat", "recover", "invalid"))
        is_stop_rule = any(marker in norm for marker in ("stop rule", "already settles", "settles the claim", "sufficient", "forcing an extra lookup"))
        is_premature_finish = any(marker in norm for marker in ("premature finish", "avoid finish", "before evidence"))
        is_search_workflow = (
            "search" in norm
            and (
                "need_search" in norm
                or "need search" in norm
                or "try_search" in norm
                or "try search" in norm
            )
        )
        is_lookup_workflow = (
            "lookup" in norm
            and (
                "need_lookup_or_finish" in norm
                or "need lookup or finish" in norm
                or "try_lookup" in norm
                or "try lookup" in norm
            )
        )
        if not (is_recovery or is_stop_rule or is_premature_finish or is_search_workflow or is_lookup_workflow):
            return ""
        ctx = self._gm3_fever_claim_context(query)
        if is_search_workflow and progress == "need_search":
            # Search[current entity] is already part of the FEVER action space.
            # Rendering old local search transitions as advice adds no evidence
            # and can turn failed entity-specific examples into false positive
            # transfer signals.
            return ""
        if is_lookup_workflow and progress == "need_lookup_or_finish":
            return ""
        if is_recovery and progress in {"search_failed", "invalid_action"}:
            action = self._gm3_fever_grounded_action("search", ctx["entity"], admissible)
            if action:
                return (
                    f"Local FEVER recovery ({ctx['claim_type']}): previous failures improved by reformulating the "
                    f"current claim query; retry `{action}` only if the last search/page was not useful."
                )
        if is_stop_rule and progress == "need_lookup_or_finish":
            return (
                f"Local FEVER stop rule ({ctx['claim_type']}): if the current evidence already settles the claim, "
                "finish from evidence instead of adding a generic Lookup."
            )
        if is_premature_finish and progress in {"search_failed", "invalid_action"}:
            return (
                f"Local FEVER correction ({ctx['claim_type']}): avoid label guesses before evidence; memory is a "
                "failure warning, not a label prior."
            )
        return ""

    def _gm3_fever_claim_context(self, query: Any) -> dict[str, str]:
        belief = getattr(query, "belief", {}) or {}
        roles = getattr(query, "goal_roles", {}) or {}
        claim_type = str(belief.get("claim_type") or roles.get("claim_type") or "general_fact").strip() or "general_fact"
        entity = str(belief.get("primary_entity") or roles.get("object") or "").strip()
        raw_keywords = belief.get("relation_keywords") or roles.get("relation") or []
        if isinstance(raw_keywords, str):
            raw_keywords = [raw_keywords]
        keywords = [
            self._gm3_fever_display_arg(str(keyword or ""))
            for keyword in raw_keywords
            if str(keyword or "").strip()
        ]
        keyword = next((kw for kw in keywords if self._gm3_norm(kw) not in {"claim_relation", "claim_relation_keyword", "claim_keyword"}), "")
        return {
            "claim_type": claim_type,
            "entity": self._gm3_fever_display_arg(entity),
            "lookup_keyword": keyword,
        }

    def _gm3_fever_grounded_action(self, action_type: str, value: str, admissible: list[str]) -> str:
        cleaned = self._gm3_fever_display_arg(value)
        if not cleaned:
            return ""
        prefix = f"{action_type.lower()}["
        value_norm = self._gm3_norm(cleaned)
        for command in admissible:
            command_text = str(command or "").strip()
            if command_text.lower().startswith(prefix) and value_norm and value_norm in self._gm3_norm(command_text):
                return command_text
        if action_type.lower() == "search":
            return f"Search[{cleaned}]"
        if action_type.lower() == "lookup":
            return f"Lookup[{cleaned}]"
        return ""

    def _gm3_fever_display_arg(self, value: str) -> str:
        text = re.sub(r"[_\s]+", " ", str(value or "")).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            return ""
        small = {"a", "an", "and", "for", "in", "of", "on", "the", "to"}
        parts: list[str] = []
        for idx, part in enumerate(text.split()):
            low = part.lower()
            if len(low) == 1:
                parts.append(low.upper())
            elif idx > 0 and low in small:
                parts.append(low)
            else:
                parts.append(low[:1].upper() + low[1:])
        return " ".join(parts)

    def _gm3_fever_claim_type_from_text(self, text: str) -> str:
        match = re.search(r"\(([^)]+)\)", str(text or ""))
        return match.group(1).strip() if match else ""

    def _gm3_fever_payload_keyword(self, payload: dict[str, Any]) -> str:
        keywords = payload.get("lookup_keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        for keyword in keywords:
            if self._gm3_norm(str(keyword or "")) not in {"claim_relation", "claim_relation_keyword", "claim_keyword"}:
                return str(keyword or "")
        return ""

    def _gm3_fever_value_from_item_id(self, item_id: str, key: str) -> str:
        match = re.search(r"(?:^|\|)" + re.escape(key) + r"=([^|]+)", str(item_id or ""))
        if not match:
            return ""
        value = match.group(1).strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].split(",", 1)[0].strip()
        return value

    def _gm3_pddl_current_action_hint(self, query: Any, admissible: list[str]) -> str:
        hint, _literal = self._gm3_pddl_current_action_hint_with_literal(query, admissible)
        return hint

    def _gm3_pddl_is_meta_action(self, action: str) -> bool:
        return self._gm3_norm(action) in {
            "check_valid_actions",
            "check valid actions",
            "check valid action",
            "look",
            "look around",
        }

    def _gm3_pddl_current_action_hint_with_literal(self, query: Any, admissible: list[str]) -> tuple[str, str]:
        belief = getattr(query, "belief", {}) or {}
        goal_literals = [str(item or "") for item in (belief.get("goal_literals") or []) if str(item or "").strip()]
        current_literals = [str(item or "") for item in (belief.get("current_literals") or []) if str(item or "").strip()]
        current_norm = {self._gm3_norm(item) for item in current_literals}
        unsatisfied = [literal for literal in goal_literals if self._gm3_norm(literal) not in current_norm]
        if not admissible or not unsatisfied:
            return "", ""

        best: tuple[float, str, str] = (0.0, "", "")
        for command in admissible:
            score, literal = self._gm3_pddl_action_goal_alignment(query, command, unsatisfied_literals=unsatisfied)
            if score > best[0]:
                best = (score, str(command or "").strip(), literal)
        return best[1], best[2]

    def _gm3_pddl_action_advances_unsatisfied_goal(self, query: Any, action: str) -> bool:
        score, _literal = self._gm3_pddl_action_goal_alignment(query, action)
        return score >= 1.0

    def _gm3_pddl_action_goal_alignment(
        self,
        query: Any,
        action: str,
        *,
        unsatisfied_literals: list[str] | None = None,
    ) -> tuple[float, str]:
        action_text = str(action or "").strip()
        action_norm = self._gm3_norm(action_text)
        if not action_norm or self._gm3_pddl_is_meta_action(action_norm):
            return 0.0, ""
        belief = getattr(query, "belief", {}) or {}
        if unsatisfied_literals is None:
            goal_literals = [str(item or "") for item in (belief.get("goal_literals") or []) if str(item or "").strip()]
            current_literals = [str(item or "") for item in (belief.get("current_literals") or []) if str(item or "").strip()]
            current_norm = {self._gm3_norm(item) for item in current_literals}
            unsatisfied_literals = [literal for literal in goal_literals if self._gm3_norm(literal) not in current_norm]
        if not unsatisfied_literals:
            return 0.0, ""

        action_tokens = self._gm3_pddl_tokens(action_text)
        if not action_tokens:
            return 0.0, ""
        action_token_set = set(action_tokens)
        action_verb = action_tokens[0]
        best: tuple[float, str] = (0.0, "")
        for literal in unsatisfied_literals[:8]:
            object_tokens, predicate_tokens = self._gm3_pddl_literal_features(literal)
            if not object_tokens:
                continue
            hits = [tok for tok in object_tokens if tok in action_token_set]
            coverage = len(hits) / max(len(object_tokens), 1)
            ordered = self._gm3_pddl_ordered_token_match(object_tokens, action_tokens)
            if len(object_tokens) == 1:
                object_ok = coverage >= 1.0
            elif len(object_tokens) == 2:
                object_ok = coverage >= 1.0 and ordered
            else:
                object_ok = len(hits) >= 2 and coverage >= 0.66 and ordered
            if not object_ok:
                continue
            predicate_score = self._gm3_pddl_predicate_action_score(
                predicate_tokens=predicate_tokens,
                action_norm=action_norm,
                action_verb=action_verb,
                action_tokens=action_tokens,
            )
            if predicate_tokens and predicate_score <= 0.0 and len(object_tokens) <= 1:
                continue
            score = 0.65 + coverage + 0.12 * min(len(hits), 3) + predicate_score
            if ordered:
                score += 0.18
            if score > best[0]:
                best = (score, literal)
        return best

    @staticmethod
    def _gm3_pddl_tokens(text: str) -> list[str]:
        return [tok for tok in re.findall(r"[a-z0-9]+", str(text or "").lower()) if tok]

    def _gm3_pddl_literal_features(self, literal: str) -> tuple[list[str], set[str]]:
        tokens = self._gm3_pddl_tokens(literal)
        stop = {
            "and", "or", "not", "goal", "true", "is", "are", "the", "a", "an",
            "some", "of", "to", "from", "with", "your", "you", "have", "has",
            "be", "been", "being", "condition", "conditions",
        }
        predicates = {
            "on", "in", "at", "clear", "empty", "handempty", "holding", "inflated",
            "intact", "loose", "tight", "tightened", "fastened", "open", "closed",
            "locked", "unlocked", "ground", "table",
        }
        generic_nouns = {
            "wheel", "wheels", "hub", "hubs", "nut", "nuts", "block", "blocks",
            "ball", "balls", "room", "rooms", "arm", "arms",
        }
        predicate_tokens = {tok for tok in tokens if tok in predicates}
        object_tokens = [
            tok for tok in tokens
            if tok not in stop and tok not in predicates and tok not in generic_nouns
        ]
        if not object_tokens:
            object_tokens = [tok for tok in tokens if tok not in stop and tok not in predicates]
        return list(dict.fromkeys(object_tokens)), predicate_tokens

    @staticmethod
    def _gm3_pddl_ordered_token_match(needles: list[str], haystack: list[str]) -> bool:
        pos = 0
        for token in needles:
            try:
                idx = haystack.index(token, pos)
            except ValueError:
                return False
            pos = idx + 1
        return True

    @staticmethod
    def _gm3_pddl_predicate_action_score(
        *,
        predicate_tokens: set[str],
        action_norm: str,
        action_verb: str,
        action_tokens: list[str],
    ) -> float:
        score = 0.0
        token_set = set(action_tokens)
        if "inflated" in predicate_tokens and ("inflate" in token_set or action_verb == "inflate"):
            score += 0.95
        if "in" in predicate_tokens and (
            action_norm.startswith("put away ")
            or action_norm.startswith("putaway ")
            or action_verb in {"put", "insert", "load"}
        ):
            score += 0.75
        if "on" in predicate_tokens:
            if action_verb in {"stack", "putdown", "put", "place"}:
                score += 0.75
            if "table" in predicate_tokens and action_verb in {"putdown", "put", "place"}:
                score += 0.35
            if "ground" in predicate_tokens and action_norm.startswith("jack down"):
                score += 0.7
        if "at" in predicate_tokens and action_verb in {"drop", "move", "go", "board", "depart"}:
            score += 0.55
        if {"open", "closed", "locked", "unlocked"} & predicate_tokens:
            for pred in ("open", "close", "lock", "unlock"):
                if action_verb == pred or action_norm.startswith(pred + " "):
                    score += 0.55
        if {"loose", "tight", "tightened", "fastened"} & predicate_tokens:
            if action_verb in {"loosen", "tighten", "fasten", "undo"}:
                score += 0.65
        return score

    def _gm3_item_task_family(self, item: Any) -> str:
        """Extract the task_family from a memory item, falling back to candidate_id parsing."""
        family = str(getattr(item, "task_family", "") or "").strip()
        if family:
            return family
        anchor = getattr(item, "anchor", {}) or {}
        family = str(anchor.get("task_family", "") or "").strip()
        if family:
            return family
        cid = str(
            getattr(item, "candidate_id", "")
            or getattr(item, "artifact_id", "")
            or getattr(item, "rule_id", "")
            or ""
        )
        m = re.search(r"task_family=([^|]+)", cid)
        if m:
            return m.group(1).strip()
        return ""

    def _gm3_fever_item_has_positive_transfer_signal(self, item: Any) -> bool:
        """Only let successful local FEVER transitions become positive hints."""
        branch = self._gm3_norm(str(getattr(item, "branch_tag", "") or ""))
        if branch in {"failure_branch", "repair_branch"}:
            return False
        try:
            positive = int(float(getattr(item, "positive", 0) or 0))
            negative = int(float(getattr(item, "negative", 0) or 0))
            stalled = int(float(getattr(item, "stalled", 0) or 0))
        except (TypeError, ValueError):
            positive = negative = stalled = 0
        if negative > 0 and positive <= negative:
            return False
        if stalled > 0 and positive <= stalled:
            return False
        return True

    def _gm3_should_suppress_fever_workflow_for_phase(self, *, text: str, task_family: str, progress: str) -> bool:
        """Avoid carrying evidence-acquisition hints into FEVER final-label decisions."""
        if progress != "ready_finish":
            return False
        if not str(task_family or "").startswith("fever"):
            return False
        norm = self._gm3_norm(text)
        return any(
            marker in norm
            for marker in (
                "fever evidence workflow",
                "fever lookup workflow",
                "fever recovery workflow",
                "fever stop rule",
                "fever failure avoidance",
            )
        )

    def _gm3_fever_workflow_fits_phase(self, *, text: str, task_family: str, progress: str) -> bool:
        """Keep FEVER procedural memory phase-specific so it stays helpful."""
        if not str(task_family or "").startswith("fever"):
            return True
        norm = self._gm3_norm(text)
        if "fever" not in norm:
            return True
        if "fever evidence workflow" in norm:
            return progress in {"need_search", "search_failed", "invalid_action"}
        if "fever lookup workflow" in norm:
            return progress == "need_lookup_or_finish"
        if "fever recovery workflow" in norm:
            return progress in {"search_failed", "invalid_action"}
        if "fever stop rule" in norm:
            return progress == "need_lookup_or_finish"
        if "fever failure avoidance" in norm:
            return progress in {"need_search", "need_lookup_or_finish"}
        return True

    def _gm3_source_role_items(
        self,
        *,
        source_table: list[dict[str, Any]],
        progress: str,
    ) -> list[str]:
        if not progress.startswith("search"):
            return []
        rendered: list[str] = []
        seen_bases: set[str] = set()
        for row in source_table:
            score = float(row.get("final_score", 0.0) or 0.0)
            level = str(row.get("evidence_level", "") or "")
            base = self._gm3_base(str(row.get("source_base", "") or ""))
            if score <= 0 or not base or base in seen_bases:
                continue
            seen_bases.add(base)
            actions = list(row.get("admissible_actions", []) or [])[:3]
            evidence = f"{level} source evidence final_score={score:.2f}"
            if actions:
                queue = " -> ".join(f"`{action}`" for action in actions)
                rendered.append(f"try {base} before broad search; current admissible queue: {queue}. Evidence: {evidence}.")
            elif level in {"exact", "same_family"}:
                rendered.append(f"prefer source type {base} when it becomes actionable. Evidence: {evidence}.")
            if len(rendered) >= 3:
                break
        return self._gm3_dedupe(rendered, 3)

    def _gm3_source_base_evidence_rows(
        self,
        *,
        local_memory: Any,
        global_memory: Any,
        target: str,
        tool: str,
        destination: str,
        task_family: str,
        setting: str,
        owner_scene: str,
    ) -> list[tuple[float, str, str, str, int, float]]:
        """Score source bases from graph artifacts.

        Positive successful source priors support search alternatives; failure
        dominated priors contribute negative evidence so they can be excluded.
        The row shape intentionally matches `_gm3_source_role_items`.
        """
        rows: list[tuple[float, str, str, str, int, float]] = []

        def confidence_of(stats: Any) -> float:
            value = getattr(stats, "confidence", 0.0)
            try:
                return float(value() if callable(value) else value or 0.0)
            except Exception:
                return 0.0

        def add_row(
            *,
            source_label: str,
            transfer: str,
            base: str,
            stats: Any,
            weight: float,
            exact_bonus: float = 0.0,
        ) -> None:
            base_norm = self._gm3_base(base)
            if not base_norm or base_norm in {tool, destination}:
                return
            support = int(getattr(stats, "support", 0) or 0)
            success = int(getattr(stats, "success", 0) or 0)
            failure = int(getattr(stats, "failure", 0) or 0)
            confidence = confidence_of(stats)

            # Failure-dominated evidence is still useful: it should block the
            # base from becoming a fallback action later.
            if failure > success and support >= 2:
                penalty = weight * (0.18 + 0.08 * min(failure - success, 5) + 0.05 * min(support / 5.0, 1.0))
                rows.append((-penalty, source_label, transfer, base_norm, support, confidence))
                return

            if transfer == "role" and (support < 3 or confidence < 0.55):
                return
            if transfer == "exact" and support <= 0 and confidence < 0.45:
                return
            if transfer == "exact" and confidence < 0.30 and max(0, success - failure) < 3:
                return

            k = getattr(self, "_gm3_support_logistic_k", 0.35)
            support_factor = 1.0 / (1.0 + math.exp(-k * (support - 2)))
            transfer_factor = {"exact": 1.0, "same_family": 0.48, "role": 0.25}.get(transfer, 0.25)
            score = weight * (
                0.7 * confidence
                + 0.3 * support_factor
                + 0.04 * max(0, success - failure)
                + exact_bonus
            ) * transfer_factor
            rows.append((score, source_label, transfer, base_norm, support, confidence))

        def scan_source_type(memory: Any, source_label: str, weight: float) -> None:
            artifacts = getattr(memory, "artifacts_by_id", {}) or {}
            for artifact in (artifacts.values() if isinstance(artifacts, dict) else artifacts):
                if source_label == "global" and not self._gm3_keep_global_item_for_owner(
                    artifact,
                    global_memory=global_memory,
                    owner_scene=owner_scene,
                ):
                    continue
                payload = getattr(artifact, "payload", {}) or {}
                kind = str(payload.get("pattern_kind", "") or "")
                if kind == "source_type_prior":
                    base = self._gm3_base(str(payload.get("source_base", "") or payload.get("source_instance", "") or ""))
                    anchor = getattr(artifact, "anchor", {}) or {}
                    goal_object = self._gm3_base(str(payload.get("goal_object", "") or ""))
                    transfer = self._gm3_source_transfer_level(
                        goal_object=goal_object,
                        artifact_task_family=str(anchor.get("task_family", "") or ""),
                        target=target,
                        task_family=task_family,
                    )
                    add_row(
                        source_label=source_label,
                        transfer=transfer,
                        base=base,
                        stats=getattr(artifact, "stats", None),
                        weight=weight,
                        exact_bonus=0.05 if transfer == "exact" else 0.0,
                    )
                    continue

                if kind != "scene_relation":
                    continue
                relation_kind = str(payload.get("relation_kind", "") or "")
                if str(payload.get("object_role", "") or "") != "target_object":
                    continue
                anchor = getattr(artifact, "anchor", {}) or {}
                goal_sig = self._gm3_norm(str(anchor.get("goal_signature", "") or ""))
                goal_object = goal_sig.split("->", 1)[0].strip() if goal_sig else ""
                transfer = self._gm3_source_transfer_level(
                    goal_object=goal_object,
                    artifact_task_family=str(anchor.get("task_family", "") or ""),
                    target=target,
                    task_family=task_family,
                )
                if transfer == "role":
                    continue
                base = self._gm3_base(str(payload.get("source_base", "") or payload.get("source_instance", "") or ""))
                if not base:
                    continue
                if relation_kind == "object_location_prior":
                    add_row(
                        source_label=source_label,
                        transfer=transfer,
                        base=base,
                        stats=getattr(artifact, "stats", None),
                        weight=weight,
                        exact_bonus=0.12 if transfer == "exact" else 0.02,
                    )
                elif relation_kind == "searched_empty":
                    stats = getattr(artifact, "stats", None)
                    support = int(getattr(stats, "support", 0) or 0)
                    confidence = confidence_of(stats)
                    transfer_factor = {"exact": 1.0, "same_family": 0.45, "role": 0.20}.get(transfer, 0.20)
                    penalty = weight * (0.16 + 0.05 * min(max(support, 1) / 3.0, 1.0)) * transfer_factor
                    rows.append((-penalty, source_label, transfer, base, support, confidence))

        if setting not in {"base", "global_only"}:
            scan_source_type(local_memory, "local", getattr(self, "_gm3_local_weight", 1.0))
        if setting not in {"base", "local_only"}:
            scan_source_type(global_memory, "global", getattr(self, "_gm3_global_weight", 0.65))
        return rows

    def _gm3_source_transfer_level(
        self,
        *,
        goal_object: str,
        artifact_task_family: str,
        target: str,
        task_family: str,
    ) -> str:
        goal = self._gm3_base(goal_object)
        task_a = self._gm3_norm(artifact_task_family)
        task_b = self._gm3_norm(task_family)
        if target and goal == target:
            return "exact"
        if task_a and task_b and task_a == task_b:
            return "same_family"
        if not goal and task_a and task_b and task_a == task_b:
            return "same_family"
        return "role"

    def _gm3_source_evidence_table(
        self,
        *,
        local_memory: Any,
        global_memory: Any,
        target: str,
        tool: str,
        destination: str,
        task_family: str,
        admissible: list[str],
        exhausted: list[str],
        setting: str,
        owner_scene: str,
        blocked_actions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        strength = {"exact": 3, "same_family": 2, "role": 1}
        exhausted_counts = self._gm3_exhausted_base_counts(exhausted)

        for score, source_label, transfer, base, support, confidence in self._gm3_source_base_evidence_rows(
            local_memory=local_memory,
            global_memory=global_memory,
            target=target,
            tool=tool,
            destination=destination,
            task_family=task_family,
            setting=setting,
            owner_scene=owner_scene,
        ):
            base_norm = self._gm3_base(base)
            if not base_norm:
                continue
            row = grouped.setdefault(
                base_norm,
                {
                    "source_base": base_norm,
                    "source_role": self._gm3_location_role(base_norm) or "unknown",
                    "positive_score": 0.0,
                    "negative_score": 0.0,
                    "exhausted_penalty": 0.0,
                    "final_score": 0.0,
                    "evidence_level": "role",
                    "admissible_actions": [],
                    "evidence": [],
                },
            )
            score_value = float(score or 0.0)
            if score_value >= 0:
                row["positive_score"] = float(row.get("positive_score", 0.0) or 0.0) + score_value
            else:
                row["negative_score"] = float(row.get("negative_score", 0.0) or 0.0) + abs(score_value)
            current_level = str(row.get("evidence_level", "role") or "role")
            if strength.get(transfer, 0) > strength.get(current_level, 0):
                row["evidence_level"] = transfer
            row["evidence"].append(
                f"{source_label}:{transfer}:support={support}:confidence={float(confidence or 0.0):.2f}:score={score_value:.2f}"
            )

        for base, row in grouped.items():
            count = int(exhausted_counts.get(base, 0) or 0)
            if count:
                row["exhausted_penalty"] = 0.20 * min(count, 6)
            row["admissible_actions"] = self._gm3_admissible_source_base_actions(
                base,
                admissible,
                exhausted,
                blocked_actions=blocked_actions,
                limit=4,
            )
            row["final_score"] = (
                float(row.get("positive_score", 0.0) or 0.0)
                - float(row.get("negative_score", 0.0) or 0.0)
                - float(row.get("exhausted_penalty", 0.0) or 0.0)
            )
            if not row["admissible_actions"] and float(row["final_score"]) > 0:
                fallback = self._gm3_admissible_source_role_actions(
                    str(row.get("source_role", "") or ""),
                    admissible,
                    exhausted,
                    blocked_actions=blocked_actions,
                    excluded_bases=set(exhausted_counts) | {base},
                    limit=3,
                )
                if fallback:
                    row["admissible_actions"] = fallback
                    row["evidence"].append("role_fallback:source_base_not_actionable")

        rows = list(grouped.values())
        rows.sort(
            key=lambda row: (
                -float(row.get("final_score", 0.0) or 0.0),
                -strength.get(str(row.get("evidence_level", "role") or "role"), 0),
                str(row.get("source_base", "")),
            )
        )
        return rows

    def _gm3_source_table_priority_actions(self, source_table: list[dict[str, Any]], *, limit: int = 2) -> list[str]:
        return [item["action"] for item in self._gm3_search_bias_candidates(source_table, limit=limit)]

    def _gm3_search_bias_candidates(
        self,
        source_table: list[dict[str, Any]],
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_actions: set[str] = set()
        for row in source_table:
            level = str(row.get("evidence_level", "") or "")
            score = float(row.get("final_score", 0.0) or 0.0)
            source_scope = str(row.get("source_scope", "") or "")
            if level not in {"exact", "same_family"} and source_scope != "previous_success_source":
                continue
            threshold = 0.35 if level == "exact" else 0.32
            if source_scope == "previous_success_source":
                threshold = 0.20
            if score < threshold:
                continue
            for action in row.get("admissible_actions", []) or []:
                action_text = str(action or "").strip()
                key = self._gm3_norm(action_text)
                if not key or key in seen_actions:
                    continue
                seen_actions.add(key)
                candidates.append(
                    {
                        "action": action_text,
                        "base": self._gm3_base(self._gm3_command_target_text(action_text)),
                        "score": score,
                        "evidence_level": level,
                        "source_scope": source_scope or "memory",
                    }
                )
                break
            if len(candidates) >= limit:
                break
        return candidates[:limit]

    @staticmethod
    def _gm3_merge_source_evidence_table(
        base_rows: list[dict[str, Any]],
        extra_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in list(base_rows) + list(extra_rows):
            base = str(row.get("source_base", "") or "")
            if not base:
                continue
            previous = merged.get(base)
            if previous is None or float(row.get("final_score", 0.0) or 0.0) > float(previous.get("final_score", 0.0) or 0.0):
                merged[base] = dict(row)
        rows = list(merged.values())
        rows.sort(key=lambda row: (-float(row.get("final_score", 0.0) or 0.0), str(row.get("source_base", ""))))
        return rows

    def _gm3_previous_success_source_rows(
        self,
        *,
        query: Any,
        env_ref: Any,
        admissible: list[str],
        blocked_actions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not self._gm2_is_two_object_second_search(query):
            return []
        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = self._gm3_base(str(goal_roles.get("object", "") or ""))
        if not target or env_ref is None:
            return []
        history = list(getattr(env_ref, "current_history", []) or [])
        dynamic = getattr(query, "dynamic_context", {}) or {}
        checked_exact = {
            self._normalize_action_text(str(item)).replace(" ", "_")
            for item in (
                list(dynamic.get("exhausted_locations", []) or [])
                + list(dynamic.get("inspected_locations", []) or [])
                + list(dynamic.get("searched_locations", []) or [])
            )
            if str(item).strip()
        }
        seen_sources: set[tuple[str, str]] = set()
        rows: list[dict[str, Any]] = []
        pattern = re.escape(target.replace("_", " "))
        for row in reversed(history):
            action_text = self._normalize_action_text(str(row.get("Action", "") or "")).replace("_", " ")
            match = re.search(
                rf"\btake\s+{pattern}(?:\s+\d+)?\s+from\s+([a-z][a-z0-9]*(?:\s+\d+)?)\b",
                action_text,
            )
            if not match:
                continue
            source_instance = self._normalize_action_text(match.group(1)).replace(" ", "_")
            source_base = self._gm3_base(source_instance)
            if not source_base:
                continue
            key = (source_base, source_instance)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            actions = []
            exact = self._gm3_admissible_source_action(source_instance, admissible)
            if exact and source_instance not in checked_exact and not self._gm3_action_is_blocked(exact, blocked_actions):
                actions.append(exact)
            for action in self._gm3_admissible_source_base_actions(
                source_base,
                admissible,
                exhausted=[],
                blocked_actions=blocked_actions,
                limit=6,
            ):
                action_target = self._normalize_action_text(self._gm3_command_target_text(action)).replace(" ", "_")
                if action_target == source_instance and source_instance in checked_exact:
                    continue
                if action not in actions:
                    actions.append(action)
                if len(actions) >= 3:
                    break
            if not actions:
                continue
            rows.append(
                {
                    "source_base": source_base,
                    "source_role": self._gm3_location_role(source_base) or "unknown",
                    "positive_score": 1.35,
                    "negative_score": 0.0,
                    "exhausted_penalty": 0.0,
                    "final_score": 1.35,
                    "evidence_level": "exact",
                    "admissible_actions": actions[:3],
                    "evidence": [
                        f"episode_local:previous_success_source={source_instance}:prefer_same_base_next_unchecked",
                    ],
                    "source_scope": "previous_success_source",
                }
            )
            if len(rows) >= 2:
                break
        return rows

    def _gm3_supported_source_base_scores(
        self,
        *,
        local_memory: Any,
        global_memory: Any,
        target: str,
        tool: str,
        destination: str,
        task_family: str,
        setting: str,
        owner_scene: str,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for score, _source_label, _transfer, base, _support, _confidence in self._gm3_source_base_evidence_rows(
            local_memory=local_memory,
            global_memory=global_memory,
            target=target,
            tool=tool,
            destination=destination,
            task_family=task_family,
            setting=setting,
            owner_scene=owner_scene,
        ):
            if not base:
                continue
            scores[base] = scores.get(base, 0.0) + float(score or 0.0)
        return {base: score for base, score in scores.items() if score > 0.05}

    def _gm3_failure_items(
        self,
        *,
        bundle: Any,
        target: str,
        progress: str,
        exhausted: list[str],
        query: Any = None,
        local_memory: Any = None,
        global_memory: Any = None,
        task_family: str = "",
        owner_scene: str = "",
        admissible: list[str] | None = None,
    ) -> list[str]:
        raw_items = (
            list(getattr(bundle, "repair_items", []) or [])
            + list(getattr(bundle, "reflection_items", []) or [])
            + list(getattr(bundle, "blocked_actions", []) or [])
            + list(getattr(bundle, "warnings", []) or [])
        )
        rendered: list[str] = []
        for item in raw_items:
            text = self._gm3_clean(str(getattr(item, "summary", item) or ""))
            if not text:
                continue
            norm = self._gm3_norm(text)
            if not self._gm3_is_failure_text(text) and not any(marker in norm for marker in ("wrong object", "repeat", "stall", "avoid")):
                continue
            rendered.append(self._gm3_bind_slots(text, target=target, tool="", destination=""))
        if progress.startswith("search"):
            exhausted_bases: dict[str, int] = {}
            for location in exhausted:
                base = self._gm3_base(str(location or ""))
                if not base:
                    continue
                exhausted_bases[base] = exhausted_bases.get(base, 0) + 1
            overused = [base for base, count in sorted(exhausted_bases.items(), key=lambda kv: (-kv[1], kv[0])) if count >= 3]
            if overused:
                rendered.append(
                    "already searched many "
                    + "/".join(overused[:2])
                    + " instances without target_object; lower those source types and try another plausible source role."
                )
        return self._gm3_dedupe(rendered, 3)

    def _gm3_textloss_route(
        self,
        *,
        candidates: list[dict[str, Any]],
        query: Any,
        admissible: list[str],
        visible: list[str],
        held: list[str],
        exhausted: list[str],
    ) -> dict[str, Any]:
        scored: list[dict[str, Any]] = []
        for section in candidates:
            loss, reasons, dimensions = self._gm3_section_textloss(
                section=section,
                query=query,
                admissible=admissible,
                visible=visible,
                held=held,
                exhausted=exhausted,
            )
            enriched = dict(section)
            enriched["loss"] = loss
            enriched["loss_reasons"] = reasons
            enriched["loss_dimensions"] = dimensions
            scored.append(enriched)
        selected = self._gm3_select_composed_sections(scored=scored, query=query, visible=visible, held=held)
        return {
            "selected": selected,
            "scored_sections": scored,
            "routing_matrix": [
                {
                    "slot": item.get("slot"),
                    "source": item.get("source"),
                    "loss": item.get("loss"),
                    "dimensions": item.get("loss_dimensions"),
                    "selected": item in selected,
                    "reasons": item.get("loss_reasons"),
                }
                for item in scored
            ],
            "all_candidates": [
                {
                    "slot": item.get("slot"),
                    "source": item.get("source"),
                    "loss": item.get("loss"),
                    "loss_dimensions": item.get("loss_dimensions"),
                    "loss_reasons": item.get("loss_reasons"),
                    "items": item.get("items", [])[:4],
                }
                for item in scored
            ],
        }

    def _gm3_prompt_signature(self, *, query: Any, selected: list[dict[str, Any]]) -> str:
        goal_roles = getattr(query, "goal_roles", {}) or {}
        parts = [
            str(getattr(query, "progress_state", "") or ""),
            str(getattr(query, "current_stage", "") or ""),
            self._gm3_base(str(goal_roles.get("object", "") or "")),
            self._gm3_base(str(goal_roles.get("tool", "") or "")),
            self._gm3_base(str(goal_roles.get("destination", "") or "")),
        ]
        for section in selected:
            slot = str(section.get("slot", "") or "")
            items = [self._gm3_norm(str(item or "")) for item in section.get("items", [])[:2]]
            parts.append(slot + ":" + "|".join(items))
        return " || ".join(parts)

    @staticmethod
    def _gm3_items_for_slot(selected: list[dict[str, Any]], slot: str, *, limit: int = 2) -> list[str]:
        items: list[str] = []
        for section in selected:
            if str(section.get("slot", "") or "") != slot:
                continue
            for item in section.get("items", []) or []:
                text = str(item or "").strip()
                if text:
                    items.append(text)
                if len(items) >= limit:
                    return items
        return items

    def _gm3_concrete_priority_items(
        self,
        items: list[str],
        *,
        admissible: list[str],
        blocked_actions: set[str] | None = None,
        limit: int = 2,
    ) -> list[str]:
        """Return only current admissible actions mentioned by memory items."""
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            action = self._gm3_first_admissible_action_in_text(str(item or ""), admissible)
            if not action:
                continue
            if self._gm3_action_is_blocked(action, blocked_actions):
                continue
            key = self._gm3_norm(action)
            if key in seen:
                continue
            seen.add(key)
            out.append(action)
            if len(out) >= limit:
                break
        return out

    def _gm3_failure_alternative_actions(
        self,
        *,
        admissible: list[str],
        exhausted: list[str],
        blocked_actions: set[str] | None = None,
        supported_source_scores: dict[str, float] | None = None,
        limit: int = 2,
    ) -> list[str]:
        """Map failure search advice to memory-supported concrete actions.

        A failure warning can exclude an overused source base, but it should not
        invent the next source. The alternative must be supported by successful
        local/global source evidence and must map to a current admissible action.
        """
        supported_source_scores = supported_source_scores or {}
        exhausted_counts = self._gm3_exhausted_base_counts(exhausted)
        overused_bases = {base for base, count in exhausted_counts.items() if count >= 3}
        exhausted_locations = {self._gm3_norm(x) for x in exhausted if str(x).strip()}
        if not overused_bases or not supported_source_scores:
            return []

        candidates: list[tuple[float, str]] = []
        seen: set[str] = set()
        for command in admissible:
            cmd = self._gm3_norm(command)
            if not cmd.startswith(("go to ", "open ", "examine ")):
                continue
            target = self._gm3_command_target_text(cmd)
            base = self._gm3_base(target)
            if not base or base in overused_bases:
                continue
            score = float(supported_source_scores.get(base, 0.0) or 0.0)
            if score <= 0.05:
                continue
            if any(location and location in cmd for location in exhausted_locations):
                continue
            if self._gm3_action_is_blocked(command, blocked_actions):
                continue
            key = self._gm3_norm(command)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((score, str(command).strip()))
        candidates.sort(key=lambda item: (-item[0], self._gm3_norm(item[1])))
        return [action for _score, action in candidates[:limit]]

    @staticmethod
    def _gm3_is_search_navigation_action(action: str) -> bool:
        norm = " ".join(str(action or "").lower().split())
        return norm.startswith(("go to ", "open ", "examine "))

    def _gm3_recent_examine_ok_actions(self, env_ref: Any) -> set[str]:
        """Treat `examine X -> OK.` as negative search feedback for this episode."""
        blocked: set[str] = set(getattr(self, "_gm3_runtime_blocked_actions", set()) or set())
        if env_ref is None:
            return blocked
        for record in list(getattr(env_ref, "current_history", []) or []):
            if not isinstance(record, dict):
                continue
            action = self._normalize_action_text(str(record.get("Action", "") or ""))
            observation = self._gm3_norm(str(record.get("Observation", "") or "")).strip(".!")
            if action.startswith("examine ") and observation == "ok":
                blocked.add(self._gm3_norm(action))
        return blocked

    def _gm3_action_is_blocked(self, action: str, blocked_actions: set[str] | None) -> bool:
        if not blocked_actions:
            return False
        return self._gm3_norm(action) in blocked_actions

    def _gm3_is_two_object_task(self, query: Any) -> bool:
        required = int(getattr(query, "required_count", 0) or 0)
        if required >= 2:
            return True
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
        goal = self._gm3_norm(str(getattr(query, "goal", "") or ""))
        return "pick_two_obj" in task_family or "pick two" in goal or "two " in goal

    def _gm3_role_aware_object_guard_repair(
        self,
        *,
        query: Any,
        processed_action: str,
        admissible_actions: list[str],
        env_ref: Any,
        task_config: dict,
        step_index: int,
    ) -> str | None:
        obj_base = self._gm3_manipulated_object_base(processed_action)
        if obj_base and obj_base in self._gm3_allowed_role_bases(query):
            return None
        return self._deterministic_object_guard_repair(
            processed_action=processed_action,
            admissible_actions=admissible_actions,
            env_ref=env_ref,
            task_config=task_config,
            step_index=step_index,
        )

    def _gm3_allowed_role_bases(self, query: Any) -> set[str]:
        goal_roles = getattr(query, "goal_roles", {}) or {}
        roles = {
            self._gm3_base(str(goal_roles.get("object", "") or "")),
            self._gm3_base(str(goal_roles.get("tool", "") or "")),
            self._gm3_base(str(goal_roles.get("destination", "") or "")),
        }
        return {role for role in roles if role}

    def _gm3_manipulated_object_base(self, action: str) -> str:
        norm = self._normalize_action_text(str(action or "")).replace("_", " ")
        match = re.match(
            r"^(take|heat|cool|clean|move|put|use)\s+(.+?)(?:\s+from\s+.+|\s+with\s+.+|\s+to\s+.+|\s+in/on\s+.+|\s+in\s+.+|\s+on\s+.+|$)",
            norm,
        )
        if not match:
            return ""
        return self._gm3_base(str(match.group(2) or ""))

    def _gm3_embedded_phase_action(
        self,
        *,
        query: Any,
        processed_action: str,
        admissible: list[str],
        blocked_actions: set[str] | None,
    ) -> str:
        text = self._gm3_norm(processed_action)
        if not text.startswith("think"):
            return ""
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
        allowed_roles = self._gm3_allowed_role_bases(query)
        phase_prefixes = ("take ", "put ", "move ", "heat ", "clean ", "cool ", "examine ")
        for action in sorted((str(a).strip() for a in admissible if str(a).strip()), key=len, reverse=True):
            action_norm = self._gm3_norm(action)
            if not action_norm or action_norm not in text:
                continue
            if self._gm3_action_is_blocked(action, blocked_actions):
                continue
            if not action_norm.startswith(phase_prefixes):
                continue
            if action_norm.startswith("examine ") and not (
                task_family.startswith("look_at") or "examine" in text or "look at" in text
            ):
                continue
            obj_base = self._gm3_manipulated_object_base(action)
            if obj_base and allowed_roles and obj_base not in allowed_roles and not action_norm.startswith("examine "):
                continue
            return action
        return ""

    def _gm3_search_bias_repair_allowed(
        self,
        *,
        query: Any,
        processed_action: str,
        processed_admissible: str,
    ) -> bool:
        text = self._gm3_norm(processed_action)
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
        if task_family.startswith("look_at"):
            return False
        if self._gm3_mentions_completion(text):
            return False
        if self._gm3_mentions_phase_critical_action(query=query, text=text):
            return False
        if processed_admissible and not self._gm3_is_search_navigation_action(processed_admissible):
            return False
        return True

    def _gm3_search_bias_is_strong(self, item: dict[str, Any]) -> bool:
        source_scope = str(item.get("source_scope", "") or "")
        if source_scope == "previous_success_source":
            return True
        level = str(item.get("evidence_level", "") or "")
        score = float(item.get("score", 0.0) or 0.0)
        return level == "exact" and score >= 0.75

    @staticmethod
    def _gm3_mentions_completion(text: str) -> bool:
        low = str(text or "").lower()
        return any(
            phrase in low
            for phrase in (
                "task is complete",
                "task complete",
                "already completed",
                "successfully completed",
                "no further action",
                "no further actions",
            )
        )

    def _gm3_mentions_phase_critical_action(self, *, query: Any, text: str) -> bool:
        low = str(text or "").lower()
        if any(word in low for word in ("examine ", "look at ", "heat ", "clean ", "cool ", "put ", "move ")):
            return True
        goal_roles = getattr(query, "goal_roles", {}) or {}
        tool = self._gm3_base(str(goal_roles.get("tool", "") or ""))
        destination = self._gm3_base(str(goal_roles.get("destination", "") or ""))
        if tool and tool in self._gm3_norm(low):
            return True
        if destination and any(word in low for word in ("deliver", "place", "put")) and destination in self._gm3_norm(low):
            return True
        return False

    def _gm3_search_bias_exhausted_count(self, query: Any, base: str) -> int:
        dynamic = getattr(query, "dynamic_context", {}) or {}
        exhausted = dynamic.get("exhausted_locations", []) or []
        counts = self._gm3_exhausted_base_counts([str(x) for x in exhausted if str(x).strip()])
        return int(counts.get(self._gm3_base(base), 0) or 0)

    @staticmethod
    def _gm3_command_target_text(command_norm: str) -> str:
        text = str(command_norm or "").strip()
        for prefix in ("go to ", "open ", "examine "):
            if text.startswith(prefix):
                return text[len(prefix):].strip()
        return text

    def _gm3_first_admissible_action_in_text(self, text: str, admissible: list[str]) -> str:
        norm_text = self._gm3_norm(text)
        for action in sorted((str(a).strip() for a in admissible if str(a).strip()), key=len, reverse=True):
            action_norm = self._gm3_norm(action)
            if action_norm and action_norm in norm_text:
                return action
        return ""

    @staticmethod
    def _gm3_selected_slot_has_real_item(selected: list[dict[str, Any]], slot: str) -> bool:
        for section in selected:
            if str(section.get("slot", "") or "") != slot:
                continue
            return any(str(item or "").strip() for item in section.get("items", []) or [])
        return False

    def _gm3_render_decision_summary(
        self,
        *,
        query: Any,
        selected: list[dict[str, Any]],
        priority_items: list[str],
        admissible: list[str],
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        """Render the only GM3 prompt body injected into nvdamas.

        Keep this block short and slot based. It is intentionally not a
        trajectory dump: local memory grounds current actions, global memory
        gives transferable context, and failure memory gives one avoid cue.
        """
        goal_roles = getattr(query, "goal_roles", {}) or {}
        target = self._gm3_base(str(goal_roles.get("object", "") or ""))
        tool = self._gm3_base(str(goal_roles.get("tool", "") or ""))
        destination = self._gm3_base(str(goal_roles.get("destination", "") or ""))
        task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
        style = self._gm3_prompt_style(query=query, task_family=task_family)
        local_line = self._gm3_summary_slot(selected, "local_grounding", default="none.")
        global_line = self._gm3_summary_slot(selected, "global_workflow", default="none.")
        failure_line = self._gm3_summary_slot(selected, "failure_avoidance", default="none.")
        source_line = self._gm3_summary_slot(selected, "source_roles", default="")
        if source_line and local_line == "none.":
            local_line = source_line

        state_line = self._gm3_state_summary(
            query=query,
            target=target,
            tool=tool,
            destination=destination,
            held=held,
            visible=visible,
            exhausted=exhausted,
        )
        if local_line != "none." and not style.keep_local_artifact_text(
            self,
            domain=style.name,
            task_family=task_family,
            text=local_line,
            norm_text=self._gm3_norm(local_line),
        ):
            local_line = "none."
        if global_line != "none." and not style.keep_global_text(
            self,
            text=global_line,
            norm_text=self._gm3_norm(global_line),
            task_family=task_family,
        ):
            global_line = "none."
        local_line = style.format_memory_line(self, query=query, slot="local_grounding", text=local_line)
        global_line = style.format_memory_line(self, query=query, slot="global_workflow", text=global_line)
        failure_line = style.format_memory_line(self, query=query, slot="failure_avoidance", text=failure_line)
        if style.name in {"fever", "pddl"} and local_line == "none." and global_line == "none." and failure_line == "none.":
            return ""
        next_line = self._gm3_next_priority_line(
            query=query,
            priority_items=priority_items,
            admissible=admissible,
        )
        if style.name == "fever" and not priority_items:
            for correction in (failure_line, local_line, global_line):
                if correction != "none.":
                    next_line = correction
                    break
        caveat = self._gm3_confidence_caveat(
            selected=selected,
            next_line=next_line,
            admissible=admissible,
            task_family=task_family,
        )
        lines = [
            f"Current phase: {self._gm3_phase_label(query)}.",
            f"Current state: {state_line}",
            f"Local memory: {self._gm3_shorten(local_line, 150)}",
            f"Global memory: {self._gm3_shorten(global_line, 150)}",
            f"Failure memory: {self._gm3_shorten(failure_line, 140)}",
            f"Next priority: {self._gm3_shorten(next_line, 170)}",
            f"Confidence / caveat: {caveat}",
        ]
        return "\n".join(lines)

    def _gm3_summary_slot(self, selected: list[dict[str, Any]], slot: str, *, default: str) -> str:
        for section in selected:
            if str(section.get("slot", "") or "") != slot:
                continue
            for item in section.get("items", []) or []:
                text = self._gm3_shorten(str(item or ""), 180)
                if text:
                    return text
        return default

    def _gm3_phase_label(self, query: Any) -> str:
        macro = self._gm3_textgrad_macro_phase(query=query)
        return self._gm3_prompt_style(query=query).phase_label(self, query=query, macro=macro)

    def _gm3_state_summary(
        self,
        *,
        query: Any,
        target: str,
        tool: str,
        destination: str,
        held: list[str],
        visible: list[str],
        exhausted: list[str],
    ) -> str:
        return self._gm3_prompt_style(query=query).state_summary(
            self,
            query=query,
            target=target,
            tool=tool,
            destination=destination,
            held=held,
            visible=visible,
            exhausted=exhausted,
        )

    def _gm3_next_priority_line(
        self,
        *,
        query: Any,
        priority_items: list[str],
        admissible: list[str],
    ) -> str:
        return self._gm3_prompt_style(query=query).next_priority_line(
            self,
            query=query,
            priority_items=priority_items,
            admissible=admissible,
        )

    def _gm3_confidence_caveat(
        self,
        *,
        selected: list[dict[str, Any]],
        next_line: str,
        admissible: list[str],
        task_family: str = "",
    ) -> str:
        slots = {str(item.get("slot", "") or "") for item in selected}
        concrete = self._gm3_prompt_has_concrete_priority(next_line, admissible)
        if str(task_family or "").startswith("fever"):
            if concrete and ("local_grounding" in slots or "source_roles" in slots):
                return "medium; FEVER memory suggests evidence actions, but the label must follow current evidence."
            if "global_workflow" in slots:
                return "medium-low; FEVER global memory is procedural only and must not provide a label prior."
            return "low; decide from current evidence, not memory priors."
        if str(task_family or "").startswith("bfcl"):
            if "local_grounding" in slots:
                return "medium; BFCL memory is a workflow cue only, tool arguments must come from the current user turn."
            if "global_workflow" in slots:
                return "medium-low; BFCL global memory transfers tool-use discipline, not old argument values."
            return "low; current BFCL tool output and checker semantics have priority."
        if "local_grounding" in slots and concrete:
            return "high; local grounding maps to a current admissible action."
        if "source_roles" in slots and concrete:
            return "medium; source prior is grounded in current admissible actions."
        if "global_workflow" in slots:
            return "medium-low; global memory is advisory and must not override observations."
        return "low; memory has no concrete grounding, current observation has priority."

    @staticmethod
    def _gm3_shorten(text: str, limit: int) -> str:
        compact = " ".join(str(text or "").replace("\n", " ").split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

    def _gm3_tool_priority_actions(self, *, target: str, tool: str, admissible: list[str]) -> list[str]:
        tool_norm = self._gm3_base(tool)
        target_norm = self._gm3_base(target)
        go_actions: list[str] = []
        process_actions: list[str] = []
        for command in admissible:
            cmd = self._gm3_norm(command)
            if tool_norm and cmd.startswith("go to ") and tool_norm == self._gm3_base(cmd):
                go_actions.append(command)
            if tool_norm and tool_norm in cmd and any(cmd.startswith(prefix) for prefix in ("clean ", "cool ", "heat ")):
                if not target_norm or target_norm in cmd:
                    process_actions.append(command)
        out: list[str] = []
        if go_actions:
            out.append(go_actions[0])
        if process_actions:
            out.append(process_actions[0])
        return out

    def _gm3_destination_priority_actions(self, *, target: str, destination: str, admissible: list[str]) -> list[str]:
        dest_norm = self._gm3_base(destination)
        target_norm = self._gm3_base(target)
        go_actions: list[str] = []
        put_actions: list[str] = []
        for command in admissible:
            cmd = self._gm3_norm(command)
            if dest_norm and cmd.startswith("go to ") and dest_norm == self._gm3_base(cmd):
                go_actions.append(command)
            if dest_norm and cmd.startswith("put ") and dest_norm in cmd:
                if not target_norm or target_norm in cmd:
                    put_actions.append(command)
        out: list[str] = []
        if go_actions:
            out.append(go_actions[0])
        if put_actions:
            out.append(put_actions[0])
        return out

    def _gm3_take_priority_actions(self, *, target: str, admissible: list[str]) -> list[str]:
        target_norm = self._gm3_base(target)
        out: list[str] = []
        for command in admissible:
            cmd = self._gm3_norm(command)
            if cmd.startswith("take ") and (not target_norm or target_norm in cmd):
                out.append(command)
                break
        return out

    def _gm3_select_composed_sections(
        self,
        *,
        scored: list[dict[str, Any]],
        query: Any,
        visible: list[str],
        held: list[str],
    ) -> list[dict[str, Any]]:
        """Compose local/global memory by role instead of sorting one score list."""
        by_slot: dict[str, dict[str, Any]] = {}
        for item in sorted(scored, key=lambda x: float(x.get("loss", 99.0))):
            slot = str(item.get("slot", "") or "")
            if slot and slot not in by_slot:
                by_slot[slot] = item

        progress = str(getattr(query, "progress_state", "") or "")
        selected: list[dict[str, Any]] = []

        def add(slot: str, max_loss: float) -> None:
            item = by_slot.get(slot)
            if not item or item in selected:
                return
            if float(item.get("loss", 99.0) or 99.0) <= max_loss:
                selected.append(item)

        # Phase is the anchor. Without it, source priors become noisy.
        add("phase_policy", 1.15)

        # Local grounding is the current-state/action bridge.
        add("local_grounding", 1.05 if progress.startswith("search") else 1.20)

        if progress.startswith("search") and not held:
            # Source roles become useful only when grounded into the current
            # admissible frontier. This is the main GM3 search control signal.
            add("source_roles", 0.90)

        # Global is useful as macro workflow, but it should not crowd out the
        # current-state priority during search.
        add("global_workflow", 1.00 if progress.startswith("search") else 1.10)

        add("failure_avoidance", 1.20)
        return selected[:4]

    def _gm3_debug_append(self, event: str, *, step_index: int = 0, payload: dict[str, Any] | None = None) -> None:
        path = self._gm3_debug_trace_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "event": event,
                "task": str(getattr(self, "_gm2_debug_current_task", "") or ""),
                "step": int(step_index or 0),
                "payload": self._gm2_debug_jsonable(payload or {}),
            }
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            return

    def _gm3_textgrad_optimize_prompt(
        self,
        *,
        draft_prompt: str,
        query: Any,
        selected: list[dict[str, Any]],
        routed: dict[str, Any],
        admissible: list[str],
        visible: list[str],
        held: list[str],
        exhausted: list[str],
        step_index: int,
        repeated_prompt: bool,
    ) -> dict[str, Any]:
        """Optimize the GM3 memory block as a TextGrad variable.

        The optimized variable is only the GM3 memory prompt body. The original
        nvdamas solver prompt, action parser, and environment workflow are left
        untouched. If TextGrad is unavailable or the judge/rewrite loop fails,
        the deterministic draft prompt is returned unchanged.
        """
        draft = str(draft_prompt or "").strip()
        debug: dict[str, Any] = {
            "enabled": bool(getattr(self, "_gm3_use_textgrad", False)),
            "engine": str(getattr(self, "_gm3_textgrad_engine", "") or ""),
            "max_iters": int(getattr(self, "_gm3_textgrad_max_iters", 0) or 0),
            "max_calls_per_episode": int(getattr(self, "_gm3_textgrad_max_calls_per_episode", 0) or 0),
            "pass_threshold": float(getattr(self, "_gm3_textgrad_pass_threshold", 0.82) or 0.82),
            "draft_prompt": self._gm2_debug_text(draft, limit=2400),
            "iterations": [],
        }
        def finish(prompt: str, reason: str | None = None) -> dict[str, Any]:
            if reason:
                debug["skipped"] = reason
            self._gm3_debug_append(
                "textgrad_prompt_optimization_gate",
                step_index=step_index,
                payload=debug,
            )
            return {"prompt": prompt, "debug": debug}

        if not draft:
            return finish(draft_prompt, "empty_draft_prompt")
        if not getattr(self, "_gm3_use_textgrad", False):
            return finish(draft, "textgrad_disabled")
        if not self._gm3_textgrad_engine:
            return finish(draft, "missing_textgrad_engine")
        if getattr(self, "_gm3_textgrad_disabled_reason", ""):
            return finish(draft, str(self._gm3_textgrad_disabled_reason))

        route_key = self._gm3_textgrad_route_key(
            query=query,
            selected=selected,
            visible=visible,
            held=held,
        )
        debug["route_key"] = route_key
        route_hits = int(self._gm3_textgrad_route_key_hits.get(route_key, 0) or 0) + 1
        self._gm3_textgrad_route_key_hits[route_key] = route_hits
        debug["route_key_hits"] = route_hits
        should_optimize, gate_reason = self._gm3_should_run_textgrad(
            query=query,
            draft_prompt=draft,
            selected=selected,
            admissible=admissible,
            exhausted=exhausted,
            route_hits=route_hits,
            repeated_prompt=repeated_prompt,
        )
        debug["gate_reason"] = gate_reason
        if not should_optimize:
            return finish(draft, gate_reason)
        cached_prompt = str(self._gm3_textgrad_prompt_cache.get(route_key, "") or "").strip()
        if cached_prompt:
            debug["cached_prompt"] = self._gm2_debug_text(cached_prompt, limit=2200)
            return finish(cached_prompt, "cached_route_key_prompt")
        max_calls = int(getattr(self, "_gm3_textgrad_max_calls_per_episode", 0) or 0)
        if max_calls > 0 and self._gm3_textgrad_calls_this_episode >= max_calls:
            debug["calls_this_episode"] = self._gm3_textgrad_calls_this_episode
            return finish(draft, "max_textgrad_calls_per_episode_reached")
        self._gm3_textgrad_calls_this_episode += 1

        try:
            import textgrad as tg
        except Exception as exc:
            return finish(draft, f"textgrad_unavailable:{type(exc).__name__}:{exc}")

        context = self._gm3_textgrad_context(
            query=query,
            selected=selected,
            routed=routed,
            admissible=admissible,
            visible=visible,
            held=held,
            exhausted=exhausted,
        )
        debug["context"] = self._gm2_debug_text(context, limit=3600)
        self._gm3_debug_append(
            "textgrad_prompt_optimization_enter",
            step_index=step_index,
            payload={
                "route_key": route_key,
                "route_key_hits": route_hits,
                "draft_prompt": self._gm2_debug_text(draft, limit=1800),
            },
        )
        try:
            engine = tg.get_engine(self._gm3_textgrad_engine)
            set_backward = getattr(tg, "set_backward_engine", None)
            if callable(set_backward):
                set_backward(engine, override=True)
            variable = tg.Variable(
                draft,
                requires_grad=True,
                role_description=(
                    "GM3 memory prompt body injected into an ALFWorld agent. "
                    "It must combine local graph grounding and global transferable "
                    "memory into concise phase-specific decision support."
                ),
            )
            loss = tg.TextLoss(self._gm3_textgrad_loss_prompt(context), engine=engine)
            optimizer = tg.TGD(
                [variable],
                engine=engine,
                constraints=[
                    "Return only the optimized GM3 memory decision summary body.",
                    "Use exactly these field labels: Current phase, Current state, Local memory, Global memory, Failure memory, Next priority, Confidence / caveat.",
                    "Do not include markdown fences or the header '### GM3 MEMORY DECISION SUMMARY'.",
                    "Keep the prompt at seven lines or fewer.",
                    "Never recommend searching for sources when the target object is already held.",
                    "Global memory may provide workflow/source-role guidance but not concrete unseen locations.",
                    "Local memory may ground a recommendation only when it maps to current state/admissible actions.",
                ],
            )
            judge = tg.BlackboxLLM(
                engine=engine,
                system_prompt=self._gm3_textgrad_judge_system_prompt(),
            )

            best_prompt = draft
            best_score = -1.0
            final_reason = "max_iters_reached"
            max_iters = int(getattr(self, "_gm3_textgrad_max_iters", 0) or 0)
            for iteration in range(max_iters + 1):
                current = self._gm3_sanitize_optimized_prompt(variable.get_value(), fallback=draft)
                judge_payload = self._gm3_textgrad_judge_payload(context=context, prompt=current)
                judge_output = judge(
                    tg.Variable(
                        judge_payload,
                        requires_grad=False,
                        role_description="GM3 prompt judge input",
                    )
                )
                if hasattr(judge_output, "get_value"):
                    judge_value = judge_output.get_value()
                else:
                    judge_value = getattr(judge_output, "value", None) or judge_output
                judge_text = str(judge_value or "")
                parsed = self._gm3_parse_textgrad_judge(judge_text)
                score = float(parsed.get("score", 0.0) or 0.0)
                if score > best_score:
                    best_score = score
                    best_prompt = current

                debug["iterations"].append(
                    {
                        "iteration": iteration,
                        "prompt": self._gm2_debug_text(current, limit=2200),
                        "judge_raw": self._gm2_debug_text(judge_text, limit=1800),
                        "judge": parsed,
                    }
                )
                if bool(parsed.get("pass", False)) and score >= float(self._gm3_textgrad_pass_threshold):
                    final_reason = "judge_pass"
                    best_prompt = current
                    break
                if iteration >= max_iters:
                    break

                optimizer.zero_grad()
                objective = loss(variable)
                objective.backward()
                optimizer.step()

            final_prompt = self._gm3_sanitize_optimized_prompt(best_prompt, fallback=draft)
            quality_issue = self._gm3_optimized_prompt_quality_issue(
                final_prompt,
                draft_prompt=draft,
                query=query,
            )
            if quality_issue:
                debug["final_quality_issue"] = quality_issue
                final_reason = f"fallback_draft_after_quality_check:{quality_issue}"
                final_prompt = draft
            self._gm3_textgrad_seen_route_keys.add(route_key)
            self._gm3_textgrad_prompt_cache[route_key] = final_prompt
            debug["final_reason"] = final_reason
            debug["best_score"] = best_score
            debug["final_prompt"] = self._gm2_debug_text(final_prompt, limit=2400)
            self._gm3_debug_append(
                "textgrad_prompt_optimization",
                step_index=step_index,
                payload=debug,
            )
            return {"prompt": final_prompt, "debug": debug}
        except Exception as exc:
            debug["error"] = f"{type(exc).__name__}: {exc}"
            if "No backward engine provided" in str(exc):
                self._gm3_textgrad_disabled_reason = "textgrad_disabled_after_backward_engine_error"
            self._gm3_debug_append(
                "textgrad_prompt_optimization_error",
                step_index=step_index,
                payload=debug,
            )
            return {"prompt": draft, "debug": debug}

    def _gm3_should_run_textgrad(
        self,
        *,
        query: Any,
        draft_prompt: str,
        selected: list[dict[str, Any]],
        admissible: list[str],
        exhausted: list[str],
        route_hits: int,
        repeated_prompt: bool,
    ) -> tuple[bool, str]:
        progress = str(getattr(query, "progress_state", "") or "")
        held_count = int(getattr(query, "held_relevant_count", 0) or 0)
        visible_match = bool(getattr(query, "goal_object_matches_visible", False))
        slots = {str(item.get("slot", "") or "") for item in selected}
        concrete_priority = self._gm3_prompt_has_concrete_priority(draft_prompt, admissible)
        generic_queue_only = (
            "execute the next queued action" in draft_prompt.lower()
            and not concrete_priority
        )
        exhausted_bases = self._gm3_exhausted_base_counts(exhausted)
        max_exhausted = max(exhausted_bases.values()) if exhausted_bases else 0
        search_grounded = bool(slots & {"local_grounding", "source_roles"}) and concrete_priority
        draft_low = str(draft_prompt or "").lower()
        source_search_noise = bool(
            "source type" in draft_low
            or "broad search" in draft_low
            or "search target" in draft_low
            or "when it becomes actionable" in draft_low
        )

        if visible_match:
            if concrete_priority:
                return False, "local_textloss_accept_visible_concrete_priority"
            return True, "visible_phase_missing_concrete_take_priority"
        if held_count > 0:
            if concrete_priority and not source_search_noise:
                return False, "local_textloss_accept_held_concrete_priority"
            return True, "held_phase_needs_process_or_delivery_rewrite"

        stalled_search = bool(progress.startswith("search") and max_exhausted >= 3)
        has_transfer_signal = bool(slots & {"local_grounding", "source_roles", "global_workflow"})
        has_transfer_or_failure_signal = bool(slots & {"source_roles", "global_workflow", "failure_avoidance"})
        has_current_local_grounding = bool(slots & {"local_grounding"} and concrete_priority)

        if not progress.startswith("search"):
            return False, "textgrad_only_for_search_uncertainty"
        if has_current_local_grounding:
            return False, "local_textloss_accept_actionable_grounding"
        if "failure_avoidance" in slots and not has_transfer_signal:
            return False, "failure_only_summary_not_worth_textgrad"
        if not has_transfer_or_failure_signal:
            return False, "no_memory_signal_to_optimize"
        if not stalled_search and route_hits <= 1 and concrete_priority:
            return False, "local_textloss_accept_first_pass_concrete_priority"
        if generic_queue_only:
            return True, "generic_queue_without_concrete_priority"
        if progress.startswith("search") and search_grounded and not repeated_prompt:
            return False, "local_textloss_accept_search_phase_grounded"
        if stalled_search:
            return True, "search_space_stall_after_repeated_source_base"
        if not concrete_priority and has_transfer_or_failure_signal and route_hits > 1:
            return True, "memory_signal_without_concrete_priority"
        return False, "deterministic_template_sufficient"

    def _gm3_prompt_has_concrete_priority(self, prompt: str, admissible: list[str]) -> bool:
        text = str(prompt or "")
        norm = self._gm3_norm(text)
        for action in admissible[:35]:
            action_norm = self._gm3_norm(action)
            if action_norm and action_norm in norm:
                return True
        return bool(
            re.search(r"`(?:go to|open|take|put|heat|clean|cool|examine) [^`]+`", text.lower())
            or re.search(r"(?:^|\n)\s*[-*]?\s*(?:next priority:\s*)?(?:go to|open|take|put|heat|clean|cool|examine) [^\n`]+", text.lower())
        )

    def _gm3_optimized_prompt_quality_issue(self, prompt: str, *, draft_prompt: str, query: Any) -> str:
        text = str(prompt or "").strip()
        if not text:
            return "empty"
        low = text.lower()
        if "```" in text or "<think>" in low or "</think>" in low:
            return "format_noise"
        if "[" in text and "]" in text:
            return "unresolved_placeholder"
        if "✅" in text or "❌" in text or "**" in text:
            return "decorative_or_markdown_noise"
        nonempty_lines = [line for line in text.splitlines() if line.strip()]
        if len(nonempty_lines) > 7:
            return "too_many_lines"
        required_labels = (
            "current phase:",
            "current state:",
            "local memory:",
            "global memory:",
            "failure memory:",
            "next priority:",
            "confidence / caveat:",
        )
        missing = [label for label in required_labels if label not in low]
        if missing:
            return "missing_summary_fields"
        if len(text.split()) > max(75, len(str(draft_prompt or "").split()) + 25):
            return "too_verbose"
        goal_roles = getattr(query, "goal_roles", {}) or {}
        destination = self._gm3_base(str(goal_roles.get("destination", "") or ""))
        target = self._gm3_base(str(goal_roles.get("object", "") or ""))
        if "goal_destination" in low or "target_object" in low:
            return "unbound_slot"
        if target and f"target: {target}" not in low and target not in self._gm3_norm(text):
            return "target_missing"
        if destination and "destination" in low and destination not in self._gm3_norm(text):
            return "destination_placeholder_or_missing"
        return ""

    def _gm3_exhausted_base_counts(self, exhausted: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for location in exhausted:
            base = self._gm3_base(str(location or ""))
            if not base:
                continue
            counts[base] = counts.get(base, 0) + 1
        return counts

    def _gm3_textgrad_route_key(
        self,
        *,
        query: Any,
        selected: list[dict[str, Any]],
        visible: list[str],
        held: list[str],
    ) -> str:
        """Coarse key for expensive TextGrad optimization.

        TextGrad optimizes the *routing prompt template* for the current task
        phase. Fine-grained source queues still come from the deterministic
        graph router each step, so they do not belong in this key; otherwise we
        would call a judge LLM for every location in a search sweep.
        """
        goal_roles = getattr(query, "goal_roles", {}) or {}
        macro_phase = self._gm3_textgrad_macro_phase(query=query)
        slots = ",".join(sorted({str(item.get("slot", "") or "") for item in selected}))
        return " | ".join(
            [
                macro_phase,
                self._gm3_base(str(goal_roles.get("object", "") or "")),
                self._gm3_base(str(goal_roles.get("tool", "") or "")),
                self._gm3_base(str(goal_roles.get("destination", "") or "")),
                f"held={int(getattr(query, 'held_relevant_count', 0) or 0)}",
                f"visible={bool(getattr(query, 'goal_object_matches_visible', False))}",
                slots,
            ]
        )

    def _gm3_textgrad_macro_phase(self, *, query: Any) -> str:
        progress = str(getattr(query, "progress_state", "") or "")
        stage = str(getattr(query, "current_stage", "") or "")
        held = int(getattr(query, "held_relevant_count", 0) or 0)
        visible = bool(getattr(query, "goal_object_matches_visible", False))
        goal_roles = getattr(query, "goal_roles", {}) or {}
        tool = self._gm3_base(str(goal_roles.get("tool", "") or ""))
        if held > 0 and tool and progress in {"carry_target", "process_target"}:
            return "process_held_target"
        if held > 0:
            return "deliver_held_target"
        if visible:
            return "acquire_visible_target"
        if progress.startswith("search") or "search" in stage:
            if "second" in progress or "second" in stage:
                return "search_additional_target"
            return "search_target"
        return "continue_task"

    def _gm3_textgrad_context(
        self,
        *,
        query: Any,
        selected: list[dict[str, Any]],
        routed: dict[str, Any],
        admissible: list[str],
        visible: list[str],
        held: list[str],
        exhausted: list[str],
    ) -> str:
        goal_roles = getattr(query, "goal_roles", {}) or {}
        selected_text = "\n".join(
            f"- [{item.get('slot')}/{item.get('source')}] "
            + " | ".join(str(x) for x in item.get("items", [])[:4])
            for item in selected
        ) or "- (none)"
        losses = routed.get("routing_matrix", []) or []
        loss_text = "\n".join(
            f"- {item.get('slot')} source={item.get('source')} loss={item.get('loss')} "
            f"selected={item.get('selected')} reasons={item.get('reasons')}"
            for item in losses[:8]
        ) or "- (none)"
        return "\n".join(
            [
                "Task/query state:",
                f"- progress_state: {getattr(query, 'progress_state', '') or 'unknown'}",
                f"- current_stage: {getattr(query, 'current_stage', '') or 'unknown'}",
                f"- target_object: {goal_roles.get('object', '') or 'unknown'}",
                f"- processing_tool: {goal_roles.get('tool', '') or 'none'}",
                f"- goal_destination: {goal_roles.get('destination', '') or 'unknown'}",
                f"- held_relevant_count: {getattr(query, 'held_relevant_count', 0) or 0}",
                f"- goal_object_matches_visible: {bool(getattr(query, 'goal_object_matches_visible', False))}",
                "",
                "Current observation/action grounding:",
                f"- visible_objects: {visible[:12]}",
                f"- held_objects: {held[:8]}",
                f"- exhausted_locations: {exhausted[:12]}",
                f"- admissible_actions: {admissible[:35]}",
                "",
                "Selected local/global graph memory evidence:",
                selected_text,
                "",
                "Deterministic routing/loss diagnostics:",
                loss_text,
            ]
        )

    @staticmethod
    def _gm3_textgrad_loss_prompt(context: str) -> str:
        return (
            "You are optimizing one GM3 memory decision summary for an ALFWorld agent. "
            "The variable is the memory block that will be injected before the agent chooses the next action.\n\n"
            "Strict objective:\n"
            "1. It must be phase-correct: search only when target is not held; process/deliver when held.\n"
            "2. It must use local graph evidence for current-state grounding and admissible action priorities.\n"
            "3. It must use global memory only as transferable workflow/source-role guidance, not concrete unseen locations.\n"
            "4. It must suppress memory that is weak, wrong-phase, repetitive, or likely to distract the solver.\n"
            "5. It must improve both seen and unseen robustness: helpful when memory is relevant, harmless when it is not.\n"
            "6. It must be concise and action-oriented, with a clear Next priority when possible.\n"
            "7. It must use at most seven lines with these fields: Current phase, Current state, Local memory, Global memory, Failure memory, Next priority, Confidence / caveat.\n"
            "8. For two-object ALFWorld tasks, preserve the policy to finish one target's full required workflow first, including any required processing, delivery, and put action; only then start another target.\n\n"
            "Context for this decision:\n"
            f"{context}\n\n"
            "Give gradients/feedback only for rewriting the GM3 memory prompt body."
        )

    @staticmethod
    def _gm3_textgrad_judge_system_prompt() -> str:
        return (
            "You are a strict judge for a GM3 memory decision summary injected into an ALFWorld agent. "
            "Evaluate whether the prompt will likely help the next action without harming generalization. "
            "Return ONLY compact JSON with keys: pass, score, issues, rewrite_instruction. "
            "Do not include chain-of-thought, markdown, prose, or <think> tags. "
            "score is 0..1. pass should be true only if the prompt is phase-correct, concise, "
            "uses local/global memory appropriately, and avoids noisy or concrete global location transfer. "
            "A prompt with unresolved placeholders, decorative symbols, markdown emphasis, more than seven lines, "
            "or a missing concrete next priority must not pass."
        )

    @staticmethod
    def _gm3_textgrad_judge_payload(*, context: str, prompt: str) -> str:
        return (
            "Decision context:\n"
            f"{context}\n\n"
            "GM3 memory decision summary to judge:\n"
            f"{prompt}\n\n"
            "Judge requirements:\n"
            "- It must use these fields: Current phase, Current state, Local memory, Global memory, Failure memory, Next priority, Confidence / caveat.\n"
            "- It must be seven lines or fewer.\n"
            "- If target is held, the prompt must prioritize process/delivery and ignore source search.\n"
            "- If target is visible, it must prioritize taking the matching target.\n"
            "- For two-object ALFWorld tasks, it must preserve the one-target full-workflow policy, including any required processing, delivery, and put action before starting another target.\n"
            "- If in search phase, it may use source/action priorities only when they are task-relevant.\n"
            "- Local graph evidence should ground current actions; global evidence should stay abstract and transferable.\n"
            "- Penalize prompts that only say to execute a queued action without naming a concrete current priority.\n"
            "- Penalize broad generic advice, wrong object/tool/destination, concrete global scene positions, and long noisy text.\n"
            "- Penalize unresolved placeholders such as [goal_destination], decorative symbols, markdown emphasis, or verbose rewrites.\n"
            "Return JSON only."
        )

    def _gm3_parse_textgrad_judge(self, text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        parsed: dict[str, Any] = {"pass": False, "score": 0.0, "issues": [], "rewrite_instruction": raw[:500]}
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    parsed.update(data)
            except Exception:
                pass
        try:
            parsed["score"] = max(0.0, min(1.0, float(parsed.get("score", 0.0) or 0.0)))
        except Exception:
            parsed["score"] = 0.0
        parsed["pass"] = bool(parsed.get("pass", False))
        if not isinstance(parsed.get("issues"), list):
            parsed["issues"] = [str(parsed.get("issues", ""))]
        parsed["rewrite_instruction"] = str(parsed.get("rewrite_instruction", "") or raw[:500])[:800]
        return parsed

    def _gm3_sanitize_optimized_prompt(self, value: Any, *, fallback: str) -> str:
        text = str(value or "").strip()
        if not text:
            return str(fallback or "").strip()
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"^#+\s*GM3 (?:CURRENT MEMORY PRIORITY|MEMORY DECISION SUMMARY)\s*", "", text, flags=re.IGNORECASE)
        lines = [line.rstrip() for line in text.splitlines()]
        # Keep the injected memory block small; long optimized prompts were one
        # of the instability sources in earlier GM3 runs.
        compact = "\n".join(lines[:7]).strip()
        if len(compact) > 1000:
            compact = compact[:1000].rsplit("\n", 1)[0].strip()
        return compact or str(fallback or "").strip()

    def _gm3_keep_global_item_for_owner(self, item: Any, *, global_memory: Any, owner_scene: str) -> bool:
        """Keep global evidence only when it is transferable for this owner scene.

        In the scene-agent setup, an agent's local memory is its private
        in-domain experience. Global memory should represent other agents or
        multi-scene abstractions, not the same scene's local evidence promoted
        back as "global". If source scene metadata is missing, keep the item so
        older memory files remain usable.
        """
        if not getattr(self, "_gm3_global_exclude_owner", True):
            return True
        owner = str(owner_scene or "").strip()
        if not owner:
            return True

        scenes = self._gm3_source_scenes_for_item(item, global_memory=global_memory)
        if not scenes:
            return True
        return any(scene and scene != owner for scene in scenes)

    def _gm3_source_scenes_for_item(self, item: Any, *, global_memory: Any) -> set[str]:
        scenes = set(str(scene) for scene in (getattr(item, "source_scenes", set()) or set()) if str(scene))
        dynamic = getattr(item, "dynamic", None)
        if isinstance(dynamic, dict):
            scenes |= set(str(scene) for scene in (dynamic.get("source_scenes", []) or []) if str(scene))
        candidate_id = str(getattr(item, "candidate_id", "") or getattr(item, "artifact_id", "") or getattr(item, "rule_id", "") or "")
        if candidate_id and global_memory is not None:
            for container_name in ("artifacts_by_id", "rules_by_id", "candidates"):
                container = getattr(global_memory, container_name, {}) or {}
                obj = container.get(candidate_id) if isinstance(container, dict) else None
                if obj is not None:
                    scenes |= set(str(scene) for scene in (getattr(obj, "source_scenes", set()) or set()) if str(scene))
        return scenes

    def _gm3_section_textloss(
        self,
        *,
        section: dict[str, Any],
        query: Any,
        admissible: list[str],
        visible: list[str],
        held: list[str],
        exhausted: list[str],
    ) -> tuple[float, list[str], dict[str, float]]:
        slot = str(section.get("slot", "") or "")
        text = self._gm3_norm(" ".join(str(x) for x in section.get("items", []) or []))
        progress = str(getattr(query, "progress_state", "") or "")
        reasons: list[str] = []
        dimensions = {
            "phase_fit": 0.0,
            "transfer_value": 0.0,
            "grounding_value": 0.0,
            "actionability": 0.0,
            "noise": 0.0,
            "wrong_phase": 0.0,
        }
        loss = 1.0

        if slot == "phase_policy":
            loss -= 0.55
            dimensions["phase_fit"] += 0.55
            reasons.append("phase-aligned")
        if slot == "global_workflow":
            loss -= 0.25
            dimensions["transfer_value"] += 0.25
            reasons.append("global-workflow-transfer")
            task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
            if task_family.startswith("pddl"):
                mapped = self._gm3_first_admissible_action_in_text(
                    " ".join(str(x) for x in section.get("items", []) or []),
                    admissible,
                )
                if mapped and self._gm3_pddl_action_advances_unsatisfied_goal(query, mapped):
                    loss -= 0.22
                    dimensions["actionability"] += 0.22
                    reasons.append("pddl-global-goal-grounded")
                    if progress in {"search_preconditions", "advance_goal_literals"}:
                        loss -= 0.10
                        dimensions["transfer_value"] += 0.10
                        reasons.append("pddl-global-goal-grounded-phase-boost")
                elif (
                    mapped
                    and progress == "search_preconditions"
                    and not self._gm3_pddl_is_meta_action(mapped)
                ):
                    if self._gm3_is_gpt4omini_model():
                        loss += 0.22
                        dimensions["noise"] += 0.22
                        reasons.append("pddl-gpt4omini-setup-only-global-penalty")
                    else:
                        loss -= 0.12
                        dimensions["actionability"] += 0.12
                        dimensions["transfer_value"] += 0.08
                        loss -= 0.08
                        reasons.append("pddl-global-precondition-grounded")
                else:
                    loss += 0.42
                    dimensions["noise"] += 0.42
                    reasons.append("pddl-global-not-goal-grounded")
        if slot == "local_grounding":
            loss -= 0.35
            dimensions["grounding_value"] += 0.35
            reasons.append("local-current-grounding")
            if any(self._gm3_norm(cmd) in text for cmd in admissible[:30]):
                loss -= 0.25
                dimensions["actionability"] += 0.25
                reasons.append("admissible-grounded")
            task_family = self._gm3_norm(str(getattr(query, "task_family", "") or ""))
            if task_family.startswith("pddl"):
                mapped = self._gm3_first_admissible_action_in_text(
                    " ".join(str(x) for x in section.get("items", []) or []),
                    admissible,
                )
                if mapped and self._gm3_pddl_action_advances_unsatisfied_goal(query, mapped):
                    loss -= 0.28
                    dimensions["actionability"] += 0.28
                    reasons.append("pddl-local-goal-grounded")
                elif (
                    mapped
                    and progress == "search_preconditions"
                    and not self._gm3_pddl_is_meta_action(mapped)
                ):
                    if self._gm3_is_gpt4omini_model():
                        loss += 0.16
                        dimensions["noise"] += 0.16
                        reasons.append("pddl-gpt4omini-setup-only-local-penalty")
                    else:
                        loss -= 0.18
                        dimensions["actionability"] += 0.18
                        reasons.append("pddl-local-precondition-grounded")
                else:
                    loss += 0.18
                    dimensions["noise"] += 0.18
                    reasons.append("pddl-local-without-goal-grounding")
        if slot == "source_roles":
            loss += 0.1
            dimensions["noise"] += 0.1
            if progress.startswith("search"):
                loss -= 0.25
                dimensions["transfer_value"] += 0.25
                reasons.append("search-only-source-context")
                if "`go to " in str(section.get("items", [])) or "`open " in str(section.get("items", [])):
                    loss -= 0.20
                    dimensions["actionability"] += 0.20
                    reasons.append("source-prior-admissible-queue")
            else:
                loss += 0.55
                dimensions["wrong_phase"] += 0.55
                reasons.append("source-prior-wrong-phase")
        if slot == "failure_avoidance":
            loss -= 0.15
            dimensions["transfer_value"] += 0.15
            reasons.append("failure-avoidance")
        if held and slot == "source_roles":
            loss += 0.6
            dimensions["wrong_phase"] += 0.6
            reasons.append("held-target-no-search")
        if any(self._gm3_norm(x) and self._gm3_norm(x) in text for x in exhausted) and slot != "failure_avoidance":
            loss += 0.3
            dimensions["noise"] += 0.3
            reasons.append("mentions-exhausted-source")
        if self._gm3_is_concrete_location_text(text) and slot == "global_workflow":
            loss += 0.6
            dimensions["noise"] += 0.6
            reasons.append("global-concrete-location-noise")
        if len(text.split()) > 110:
            loss += 0.25
            dimensions["noise"] += 0.25
            reasons.append("too-long")

        # dynamic adjustments based on progress state.  
        # When searching, source roles and global workflow are more valuable; penalize them less.
        # When already holding target (process/delivery phases), de-emphasize new source search,
        # but emphasize local grounding.
        if progress.startswith("search"):
            if slot == "source_roles":
                loss -= 0.15
                dimensions["transfer_value"] += 0.15
                reasons.append("dynamic-search-source-boost")
            if slot == "global_workflow":
                loss -= 0.05
                dimensions["transfer_value"] += 0.05
                reasons.append("dynamic-search-global-boost")
        elif held:
            # held means the target object is already picked up; focus on process/delivery
            if slot == "source_roles":
                loss += 0.15
                dimensions["wrong_phase"] += 0.15
                reasons.append("dynamic-held-source-penalty")
            if slot == "global_workflow":
                loss += 0.05
                dimensions["noise"] += 0.05
                reasons.append("dynamic-held-global-penalty")
            if slot == "local_grounding":
                loss -= 0.05
                dimensions["grounding_value"] += 0.05
                reasons.append("dynamic-held-grounding")
        return max(0.0, loss), reasons, dimensions

    def _gm3_admissible_source_action(self, source: str, admissible: list[str]) -> str:
        source_norm = self._gm3_norm(source)
        source_base = self._gm3_base(source)
        for command in admissible:
            cmd = self._gm3_norm(command)
            if not cmd.startswith(("go to ", "open ", "examine ")):
                continue
            target = cmd.split(" ", 2)[-1] if cmd.startswith("go ") else cmd.split(" ", 1)[-1]
            if source_norm and source_norm == self._gm3_norm(target):
                return command
        for command in admissible:
            cmd = self._gm3_norm(command)
            if not cmd.startswith(("go to ", "open ", "examine ")):
                continue
            if source_base and source_base == self._gm3_base(self._gm3_command_target_text(cmd)):
                return command
        return ""

    def _gm3_admissible_source_base_actions(
        self,
        base: str,
        admissible: list[str],
        exhausted: list[str],
        *,
        blocked_actions: set[str] | None = None,
        limit: int = 3,
    ) -> list[str]:
        base_norm = self._gm3_base(base)
        exhausted_norm = {self._gm3_norm(x) for x in exhausted}
        actions: list[str] = []
        for command in admissible:
            cmd = self._gm3_norm(command)
            if not cmd.startswith(("go to ", "open ", "examine ")):
                continue
            if base_norm and base_norm != self._gm3_base(self._gm3_command_target_text(cmd)):
                continue
            if any(ex and ex in cmd for ex in exhausted_norm):
                continue
            if self._gm3_action_is_blocked(command, blocked_actions):
                continue
            actions.append(command)
            if len(actions) >= limit:
                break
        return actions

    def _gm3_admissible_source_role_actions(
        self,
        role: str,
        admissible: list[str],
        exhausted: list[str],
        *,
        blocked_actions: set[str] | None = None,
        excluded_bases: set[str] | None = None,
        limit: int = 3,
    ) -> list[str]:
        role_norm = str(role or "").strip()
        if not role_norm or role_norm == "unknown":
            return []
        excluded_bases = {self._gm3_base(x) for x in (excluded_bases or set()) if str(x).strip()}
        exhausted_norm = {self._gm3_norm(x) for x in exhausted}
        actions: list[str] = []
        seen_bases: set[str] = set()
        for command in admissible:
            cmd = self._gm3_norm(command)
            if not cmd.startswith(("go to ", "open ", "examine ")):
                continue
            target = self._gm3_command_target_text(cmd)
            base = self._gm3_base(target)
            if not base or base in seen_bases or base in excluded_bases:
                continue
            if self._gm3_location_role(base) != role_norm:
                continue
            if any(ex and ex in cmd for ex in exhausted_norm):
                continue
            if self._gm3_action_is_blocked(command, blocked_actions):
                continue
            seen_bases.add(base)
            actions.append(str(command).strip())
            if len(actions) >= limit:
                break
        return actions

    @staticmethod
    def _gm3_norm(value: str) -> str:
        return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())

    @classmethod
    def _gm3_base(cls, value: str) -> str:
        text = cls._gm3_norm(value)
        text = re.sub(r"\b(?:a|an|the|some)\s+", "", text)
        text = re.sub(r"\s+\d+\b", "", text)
        return text.strip()

    @staticmethod
    def _gm3_clean(value: str) -> str:
        text = str(value or "").strip()
        for prefix in ("Scene relation: ", "Plan: ", "Workflow pattern: ", "Workflow: ", "Reflection: ", "Closure: "):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text

    @classmethod
    def _gm3_is_concrete_location_text(cls, value: str) -> bool:
        text = cls._gm3_norm(value)
        raw = str(value or "").lower()
        return bool(re.search(r"\b[a-z]+_\d+\b", raw) or re.search(r"\b[a-z]+\s+\d+\b", text))

    @classmethod
    def _gm3_is_failure_text(cls, value: str) -> bool:
        text = cls._gm3_norm(value)
        return any(marker in text for marker in ("failed", "failure", "blocked", "wrong", "avoid", "not found", "empty"))

    @staticmethod
    def _gm3_dedupe(items: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            text = str(item or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _gm3_location_role(base: str) -> str:
        if base in {"countertop", "shelf", "diningtable", "sidetable", "dresser", "sofa", "armchair", "bed", "tvstand", "coffeetable", "desk"}:
            return "support_surface"
        if base in {"cabinet", "drawer", "fridge", "safe", "garbagecan", "microwave"}:
            return "container"
        if base in {"sinkbasin", "bathtub"}:
            return "receptacle"
        return ""

    @staticmethod
    def _gm3_bind_slots(text: str, *, target: str, tool: str, destination: str) -> str:
        rendered = str(text or "")
        if target:
            rendered = re.sub(r"\btarget_object\b", f"target_object={target}", rendered)
        if tool:
            rendered = re.sub(r"\bprocessing_tool\b|\btool\b", f"processing_tool={tool}", rendered)
        if destination:
            rendered = re.sub(r"\bgoal_destination\b|\bdestination\b", f"goal_destination={destination}", rendered)
        return rendered
