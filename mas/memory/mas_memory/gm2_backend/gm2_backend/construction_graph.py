from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
import re

from .alfworld_adapter import ALFWorldAdapter
from .graph_types import (
    ArtifactKind,
    CandidateType,
    EdgeType,
    EpisodeGraph,
    EpisodeRecord,
    GraphEdge,
    GraphNode,
    MemoryArtifact,
    MemoryRule,
    LocalGraphMemory,
    GlobalGraphMemory,
    NodeType,
    PromotionCandidate,
    RuleType,
    Stats,
)


@dataclass(slots=True)
class _ReplayBelief:
    progress: dict[str, object]
    next_progress: dict[str, object]
    search_ref: str | None
    revisit_count: int
    searched_locations: set[str]
    search_attempt_counts: dict[str, int]
    placed_relevant_count_est: int


def _record_graph_stats(stats: Stats, *, positive: bool, stalled: bool = False) -> None:
    stats.record(positive=positive, stalled=stalled)


def _episode_success(episode: EpisodeRecord) -> bool:
    status = str(episode.metadata.get("status", "")).lower()
    if status in {"success", "succeeded"}:
        return True
    if status in {"fail", "failed"}:
        return False
    return float(episode.metadata.get("final_score", 0.0)) > 0


def _task_family(episode: EpisodeRecord) -> str:
    task_id = episode.task_id or ""
    if "-" in task_id:
        return task_id.split("-", 1)[0]
    goal = episode.goal.lower()
    if "under the" in goal or "with the" in goal:
        return "look_at_obj_in_light"
    return re.sub(r"[^a-z0-9_]+", "_", goal).strip("_") or "unknown_task"


def _task_roles(episode: EpisodeRecord) -> dict[str, str]:
    adapter = ALFWorldAdapter(scene_label=episode.scene_id or "alfworld_scene")
    slots = adapter.goal_slots(episode.goal)
    roles: dict[str, str] = {}
    target = slots.get("object", "")
    destination = slots.get("destination", "")
    tool = slots.get("tool", "")
    if target:
        roles[target] = "target_object"
    if destination:
        roles[destination] = "goal_destination"
    if tool:
        roles[tool] = "tool_object"
    return roles


def _goal_arity(episode: EpisodeRecord) -> int:
    goal = episode.goal.lower()
    if goal.startswith("put two "):
        return 2
    return 1


def _normalize_entity(value: str, roles: dict[str, str]) -> str:
    lowered = value.lower()
    base = re.sub(r"_\d+$", "", lowered)
    for key, role in roles.items():
        if key == base or key in base:
            return role
    if base in {"drawer", "cabinet", "safe", "fridge", "microwave", "garbagecan", "sinkbasin", "bathtubbasin", "toilet"}:
        return "container"
    if base in {
        "desk",
        "dresser",
        "shelf",
        "sidetable",
        "coffeetable",
        "diningtable",
        "bed",
        "sofa",
        "armchair",
        "countertop",
        "tvstand",
        "stoveburner",
    }:
        return "support_surface"
    return base


def _abstract_action_text(action, roles: dict[str, str]) -> str:
    if not action.slots:
        return action.verb
    normalized = {key: _normalize_entity(value, roles) for key, value in action.slots.items()}
    parts = ",".join(f"{key}={normalized[key]}" for key in sorted(normalized))
    return f"{action.verb}({parts})"


def _location_signature(location: str | None, roles: dict[str, str]) -> str:
    if not location:
        return ""
    return _normalize_entity(location, roles)


def _goal_target_base(goal_roles: dict[str, str]) -> str:
    return re.sub(r"_\d+$", "", str(goal_roles.get("object", "")).lower())


def _is_target_object(action, target_base: str) -> bool:
    if not target_base:
        return False
    raw_object = str(action.slots.get("object", "")).lower()
    raw_base = re.sub(r"_\d+$", "", raw_object)
    return bool(raw_base) and raw_base == target_base


def _replay_episode_beliefs(episode: EpisodeRecord) -> list[_ReplayBelief]:
    adapter = ALFWorldAdapter(scene_label=episode.scene_id or "alfworld_scene")
    goal_roles = adapter.goal_slots(episode.goal)
    goal_destination = str(goal_roles.get("destination", ""))
    target_base = _goal_target_base(goal_roles)
    required_count = adapter.required_goal_count(episode.goal)

    placed_relevant_count_est = 0
    searched_locations: set[str] = set()
    search_attempt_counts: dict[str, int] = defaultdict(int)
    beliefs: list[_ReplayBelief] = []

    for step in episode.steps:
        progress = adapter.summarize_goal_progress(
            step.state,
            episode.goal,
            belief={
                "placed_relevant_count_est": placed_relevant_count_est,
            },
        )
        search_ref = adapter.search_reference_for_action(step.action, step.state)
        revisit_count = search_attempt_counts.get(search_ref, 0) if search_ref else 0
        beliefs.append(
            _ReplayBelief(
                progress=progress,
                next_progress={},
                search_ref=search_ref,
                revisit_count=revisit_count,
                searched_locations=set(searched_locations),
                search_attempt_counts=dict(search_attempt_counts),
                placed_relevant_count_est=placed_relevant_count_est,
            )
        )

        if step.feedback.success and search_ref:
            searched_locations.add(search_ref)
            search_attempt_counts[search_ref] += 1

        if step.feedback.success and step.action.verb == "take" and _is_target_object(step.action, target_base):
            source = str(step.action.slots.get("source", ""))
            if goal_destination and source == goal_destination:
                placed_relevant_count_est = max(placed_relevant_count_est - 1, 0)
        if step.feedback.success and step.action.verb == "move" and _is_target_object(step.action, target_base):
            destination = str(step.action.slots.get("destination", ""))
            if goal_destination and destination == goal_destination:
                placed_relevant_count_est = min(required_count, placed_relevant_count_est + 1)

        beliefs[-1].next_progress = adapter.summarize_goal_progress(
            step.next_state,
            episode.goal,
            belief={
                "placed_relevant_count_est": placed_relevant_count_est,
            },
        )

    return beliefs


def _step_advances_goal(
    step,
    progress: dict[str, object],
    next_progress: dict[str, object],
    *,
    revisit_count: int,
) -> bool:
    if step.feedback.failure_label:
        return False
    if int(next_progress.get("held_relevant_count", 0)) > int(progress.get("held_relevant_count", 0)):
        return True
    if int(next_progress.get("placed_relevant_count", 0)) > int(progress.get("placed_relevant_count", 0)):
        return True
    if bool(next_progress.get("goal_object_matches_visible")) and not bool(progress.get("goal_object_matches_visible")):
        return True
    if str(next_progress.get("progress_state", "")) != str(progress.get("progress_state", "")):
        return True
    if step.action.verb == "look" and any(delta.startswith("visible+=") for delta in step.feedback.state_delta):
        return True
    if step.action.verb in {"open", "examine"} and any(
        delta.startswith("opened=") or delta.startswith("visible+=") for delta in step.feedback.state_delta
    ):
        return True
    if step.action.verb == "go" and any(delta.startswith("location=") for delta in step.feedback.state_delta):
        return revisit_count == 0
    return False


def _goal_aligned_action(
    action,
    *,
    progress_state: str,
    roles: dict[str, str],
) -> bool:
    verb = action.verb
    normalized_slots = {key: _normalize_entity(value, roles) for key, value in action.slots.items()}
    if verb == "take":
        return normalized_slots.get("object") == "target_object"
    if verb == "move":
        if normalized_slots.get("object") != "target_object":
            return False
        destination = normalized_slots.get("destination", "")
        return destination == "goal_destination"
    if verb == "go":
        target = normalized_slots.get("target", "")
        if progress_state in {"carry_target", "finalize"}:
            return target == "goal_destination"
        return target in {"container", "support_surface", "tool_object"}
    if verb == "open":
        return normalized_slots.get("container") == "container"
    if verb == "examine":
        target = normalized_slots.get("target", "")
        if progress_state in {"carry_target", "finalize"}:
            return target in {"goal_destination", "target_object"}
        return target in {"container", "support_surface", "target_object", "tool_object"}
    if verb == "look":
        return True
    return False


def _induce_plan_template(
    episode: EpisodeRecord,
    *,
    beliefs: list[_ReplayBelief],
    roles: dict[str, str],
    goal_roles: dict[str, str],
    goal_arity: int,
) -> tuple[str, list[str], list[str]] | None:
    if not episode.steps:
        return None

    destination = str(goal_roles.get("destination", "")).lower()
    destination_base = re.sub(r"_\d+$", "", destination)
    target = str(goal_roles.get("object", "")).lower()

    first_take = None
    first_move = None
    source_role = ""
    for idx, step in enumerate(episode.steps):
        abstract_action = _abstract_action_text(step.action, roles)
        if (
            first_take is None
            and step.feedback.success
            and step.action.verb == "take"
            and _is_target_object(step.action, target)
        ):
            source_role = _normalize_entity(str(step.action.slots.get("source", "")), roles)
            first_take = abstract_action
        if (
            first_move is None
            and step.feedback.success
            and step.action.verb == "move"
            and _is_target_object(step.action, target)
            and destination
        ):
            move_destination = str(step.action.slots.get("destination", "")).lower()
            move_destination_base = re.sub(r"_\d+$", "", move_destination)
            move_destination_role = _normalize_entity(move_destination, roles)
            if (
                move_destination == destination
                or move_destination_base == destination_base
                or move_destination_role == "goal_destination"
            ):
                first_move = abstract_action
        if first_take and first_move:
            break

    if not first_take or not first_move:
        return None

    if source_role not in {"support_surface", "container"}:
        source_role = "support_surface"

    search_step = f"go(target={source_role})"
    delivery_step = "go(target=goal_destination)"
    plan_steps = [search_step, first_take, delivery_step, first_move]
    action_patterns = [search_step, first_take, delivery_step, first_move]

    if goal_arity > 1:
        plan_steps.append("repeat acquire and deliver for remaining target_object")

    summary = "Plan: " + " -> ".join(plan_steps)
    return summary, plan_steps, action_patterns


def _relation_action_patterns(source_instance: str, source_base: str, source_role: str) -> list[str]:
    patterns: list[str] = []
    if source_instance:
        patterns.append(f"go(target={source_instance})")
        if source_role == "container":
            patterns.append(f"open(container={source_instance})")
            patterns.append(f"examine(target={source_instance})")
        else:
            patterns.append(f"examine(target={source_instance})")
    if source_base and source_base != source_instance:
        patterns.append(f"go(target={source_base})")
        if source_role == "container":
            patterns.append(f"open(container={source_base})")
            patterns.append(f"examine(target={source_base})")
    if source_role:
        patterns.append(f"go(target={source_role})")
        if source_role == "container":
            patterns.append("open(container=container)")
        else:
            patterns.append(f"examine(target={source_role})")
    return list(dict.fromkeys(patterns))


def _condition_signature(payload: dict) -> str:
    pieces: list[str] = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, dict):
            inner = ",".join(f"{inner_key}={value[inner_key]}" for inner_key in sorted(value))
            pieces.append(f"{key}={{" + inner + "}}")
        elif isinstance(value, (list, tuple)):
            pieces.append(f"{key}=[{','.join(str(item) for item in value)}]")
        else:
            pieces.append(f"{key}={value}")
    return "|".join(pieces)


def _rule_specificity(rule: MemoryRule) -> float:
    score = 0.0
    condition_blob = " ".join(str(value) for value in rule.condition.values())
    specific_tokens = re.findall(r"\b[a-z]+_[0-9]+\b", condition_blob.lower())
    if specific_tokens:
        score += min(0.25, 0.08 * len(specific_tokens))
    if rule.goal_roles.get("object") and rule.goal_roles.get("destination"):
        score += 0.04
    return min(score, 0.35)


def _artifact_specificity(artifact: MemoryArtifact) -> float:
    score = 0.0
    anchor_blob = " ".join(str(value) for value in artifact.anchor.values())
    payload_blob = " ".join(str(value) for value in artifact.payload.values())
    specific_tokens = re.findall(r"\b[a-z]+_[0-9]+\b", f"{anchor_blob} {payload_blob}".lower())
    if specific_tokens:
        score += min(0.25, 0.08 * len(specific_tokens))
    if artifact.anchor.get("goal_signature") and artifact.anchor.get("progress_state"):
        score += 0.03
    return min(score, 0.35)


def _candidate_action_patterns(candidate: PromotionCandidate) -> tuple[str, ...]:
    patterns: list[str] = []
    for key in ("action", "repair_action"):
        value = str(candidate.structure.get(key, "")).strip()
        if value:
            patterns.append(value)
    patterns.extend(re.findall(r"[a-z_]+\([a-z0-9_,= ]+\)", candidate.summary.lower()))
    return tuple(dict.fromkeys(patterns))


def _candidate_support(candidate: PromotionCandidate) -> int:
    return candidate.positive + candidate.negative


def _keep_candidate(candidate: PromotionCandidate) -> bool:
    pattern_kind = str(candidate.structure.get("pattern_kind", ""))
    support = _candidate_support(candidate)
    confidence = candidate.confidence

    if candidate.candidate_type == CandidateType.PRECONDITION:
        # Keep only clearly helpful preconditions; generic negative rules such
        # as the old holding_before_move candidate are more harmful than useful.
        return support >= 2 and candidate.positive >= 2 and confidence >= 0.6

    if candidate.candidate_type == CandidateType.WORKFLOW:
        if pattern_kind == "closure":
            return candidate.positive >= 1
        # Generic search workflows need noticeably more positive than negative
        # evidence to stay in memory.
        return candidate.positive >= 2 and candidate.positive >= candidate.negative

    if candidate.candidate_type == CandidateType.REPAIR:
        return candidate.positive >= 1

    if candidate.candidate_type == CandidateType.FAILURE:
        if pattern_kind == "anti_pattern":
            # Keep only stable, repeatable anti-patterns as reusable failure
            # memory. One-off stalls add noise without giving a reliable blocker.
            return candidate.stalled >= 2
        # Raw failure candidates are useful as evidence during rule/artifact
        # induction, but they are too noisy to stay in the top-level candidate
        # pool. We rely on blocked/repair rules to turn failures into positive
        # guidance instead.
        return False

    return True


def _rule_action_patterns(rule: MemoryRule) -> tuple[str, ...]:
    patterns: list[str] = []
    for key in ("prefer_action", "block_action", "action"):
        value = str(rule.effect.get(key, "")).strip()
        if value:
            patterns.append(value)
    return tuple(dict.fromkeys(patterns))


class EpisodeGraphBuilder:
    def build(self, episode: EpisodeRecord) -> EpisodeGraph:
        episode_id = f"{episode.agent_id}:{episode.task_id}:{len(episode.steps)}"
        graph = EpisodeGraph(
            episode_id=episode_id,
            agent_id=episode.agent_id,
            scene_id=episode.scene_id,
            task_id=episode.task_id,
            goal=episode.goal,
            metadata=dict(episode.metadata),
        )

        for step in episode.steps:
            state_sig = f"state:{step.state.signature}"
            next_state_sig = f"state:{step.next_state.signature}"
            action_sig = f"action:{step.action.canonical_str}"

            graph.nodes.setdefault(
                state_sig,
                GraphNode(state_sig, NodeType.STATE, step.state.signature, asdict(step.state)),
            )
            graph.nodes.setdefault(
                action_sig,
                GraphNode(action_sig, NodeType.ACTION, step.action.canonical_str, asdict(step.action)),
            )
            graph.nodes.setdefault(
                next_state_sig,
                GraphNode(next_state_sig, NodeType.STATE, step.next_state.signature, asdict(step.next_state)),
            )

            graph.edges.append(
                GraphEdge(
                    src=state_sig,
                    dst=action_sig,
                    edge_type=EdgeType.TEMPORAL,
                    signature=f"{state_sig}|temporal|{action_sig}",
                    payload={"step_idx": step.step_idx},
                    stats=Stats(support=1, positive=1),
                )
            )
            graph.edges.append(
                GraphEdge(
                    src=action_sig,
                    dst=next_state_sig,
                    edge_type=EdgeType.TEMPORAL,
                    signature=f"{action_sig}|temporal|{next_state_sig}",
                    payload={"step_idx": step.step_idx},
                    stats=Stats(support=1, positive=1),
                )
            )

            if step.feedback.state_delta:
                delta_sig = f"delta:{step.step_idx}:{'|'.join(step.feedback.state_delta)}"
                graph.nodes.setdefault(
                    delta_sig,
                    GraphNode(
                        delta_sig,
                        NodeType.STATE,
                        delta_sig,
                        {"delta": list(step.feedback.state_delta)},
                    ),
                )
                graph.edges.append(
                    GraphEdge(
                        src=action_sig,
                        dst=delta_sig,
                        edge_type=EdgeType.CAUSES,
                        signature=f"{action_sig}|causes|{delta_sig}",
                        payload={"delta": list(step.feedback.state_delta)},
                        stats=Stats(support=1, positive=int(step.feedback.success), negative=int(not step.feedback.success)),
                    )
                )

            if step.subgoal:
                subgoal_sig = f"subgoal:{step.subgoal}"
                graph.nodes.setdefault(
                    subgoal_sig,
                    GraphNode(subgoal_sig, NodeType.SUBGOAL, step.subgoal, {"name": step.subgoal}),
                )
                graph.edges.append(
                    GraphEdge(
                        src=action_sig,
                        dst=subgoal_sig,
                        edge_type=EdgeType.ADVANCES_TO,
                        signature=f"{action_sig}|advances_to|{subgoal_sig}",
                        payload={"step_idx": step.step_idx},
                        stats=Stats(support=1, positive=int(step.feedback.success), negative=int(not step.feedback.success)),
                    )
                )

            if step.feedback.failure_label:
                failure_sig = f"failure:{step.feedback.failure_label}"
                graph.nodes.setdefault(
                    failure_sig,
                    GraphNode(
                        failure_sig,
                        NodeType.FAILURE,
                        step.feedback.failure_label,
                        {"label": step.feedback.failure_label},
                    ),
                )
                graph.edges.append(
                    GraphEdge(
                        src=action_sig,
                        dst=failure_sig,
                        edge_type=EdgeType.FAILS_UNDER,
                        signature=f"{action_sig}|fails_under|{failure_sig}",
                        payload={"step_idx": step.step_idx},
                        stats=Stats(support=1, negative=1),
                    )
                )

        return graph


class LocalGraphMaintainer:
    def update(self, memory: LocalGraphMemory, episode_graph: EpisodeGraph, episode: EpisodeRecord) -> LocalGraphMemory:
        if episode_graph.episode_id not in memory.episode_ids:
            memory.episode_ids.append(episode_graph.episode_id)
        episode_success = _episode_success(episode)

        for node in episode_graph.nodes.values():
            local_node = memory.nodes_by_signature.get(node.signature)
            if local_node is None:
                memory.nodes_by_signature[node.signature] = GraphNode(
                    node_id=node.node_id,
                    node_type=node.node_type,
                    signature=node.signature,
                    payload=dict(node.payload),
                )
                local_node = memory.nodes_by_signature[node.signature]
            _record_graph_stats(local_node.stats, positive=episode_success)

        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }
        for edge in episode_graph.edges:
            local_edge = memory.edges_by_signature.get(edge.signature)
            if local_edge is None:
                memory.edges_by_signature[edge.signature] = GraphEdge(
                    src=edge.src,
                    dst=edge.dst,
                    edge_type=edge.edge_type,
                    signature=edge.signature,
                    payload=dict(edge.payload),
                )
                local_edge = memory.edges_by_signature[edge.signature]
            step_idx = int(edge.payload.get("step_idx", -1))
            stalled = step_idx in stalled_steps
            positive = episode_success and edge.edge_type != EdgeType.FAILS_UNDER
            _record_graph_stats(local_edge.stats, positive=positive, stalled=stalled)

        candidates = self._induce_candidates(episode_graph, episode, episode_success=episode_success)
        for candidate in candidates:
            existing = memory.candidates.get(candidate.candidate_id)
            if existing is None:
                memory.candidates[candidate.candidate_id] = candidate
            else:
                existing.source_episode_ids |= candidate.source_episode_ids
                existing.source_scenes |= candidate.source_scenes
                existing.positive += candidate.positive
                existing.negative += candidate.negative
                existing.stalled += candidate.stalled
                existing.utility += candidate.utility

        rules = self._induce_rules(episode_graph, episode, episode_success=episode_success)
        for rule in rules:
            existing_rule = memory.rules_by_id.get(rule.rule_id)
            if existing_rule is None:
                rule.specificity = _rule_specificity(rule)
                memory.rules_by_id[rule.rule_id] = rule
            else:
                existing_rule.source_episode_ids |= rule.source_episode_ids
                existing_rule.source_scenes |= rule.source_scenes
                existing_rule.stats.support += rule.stats.support
                existing_rule.stats.success += rule.stats.success
                existing_rule.stats.failure += rule.stats.failure
                existing_rule.stats.stalled += rule.stats.stalled
                existing_rule.stats.utility += rule.stats.utility
                existing_rule.stats.transfer_success += rule.stats.transfer_success
                existing_rule.stats.transfer_trials += rule.stats.transfer_trials
                existing_rule.specificity = max(existing_rule.specificity, _rule_specificity(existing_rule))
        artifacts = self._induce_artifacts(
            episode_graph,
            episode,
            candidates=candidates,
            rules=rules,
            episode_success=episode_success,
        )
        for artifact in artifacts:
            existing_artifact = memory.artifacts_by_id.get(artifact.artifact_id)
            if existing_artifact is None:
                artifact.specificity = _artifact_specificity(artifact)
                memory.artifacts_by_id[artifact.artifact_id] = artifact
                continue
            existing_artifact.source_episode_ids |= artifact.source_episode_ids
            existing_artifact.source_scenes |= artifact.source_scenes
            existing_artifact.stats.support += artifact.stats.support
            existing_artifact.stats.success += artifact.stats.success
            existing_artifact.stats.failure += artifact.stats.failure
            existing_artifact.stats.stalled += artifact.stats.stalled
            existing_artifact.stats.utility += artifact.stats.utility
            existing_artifact.stats.transfer_success += artifact.stats.transfer_success
            existing_artifact.stats.transfer_trials += artifact.stats.transfer_trials
            existing_artifact.specificity = max(existing_artifact.specificity, _artifact_specificity(existing_artifact))
        return memory

    def refine_memory(self, memory: LocalGraphMemory) -> LocalGraphMemory:
        refined_candidates: dict[str, PromotionCandidate] = {}
        for candidate_id, candidate in memory.candidates.items():
            if _keep_candidate(candidate):
                refined_candidates[candidate_id] = candidate
        memory.candidates = refined_candidates
        return memory

    def _induce_candidates(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[PromotionCandidate]:
        candidates: list[PromotionCandidate] = []
        family = _task_family(episode)
        roles = _task_roles(episode)
        subgoals = [step.subgoal for step in episode.steps if step.subgoal]
        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }
        if len(subgoals) >= 2:
            workflow = tuple(dict.fromkeys(subgoals))
            workflow_id = f"workflow:{family}:" + "->".join(workflow)
            candidate = PromotionCandidate(
                candidate_id=workflow_id,
                candidate_type=CandidateType.WORKFLOW,
                summary=f"Workflow pattern: {' -> '.join(workflow)}",
                structure={"workflow": workflow, "goal": episode.goal, "task_family": family, "pattern_kind": "workflow"},
            )
            candidate.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                positive=episode_success,
                utility_delta=0.1 if episode_success else 0.02,
            )
            candidates.append(candidate)

        for step in episode.steps:
            if step.action.verb == "open" and step.state.location == step.action.slots.get("container"):
                cid = f"precondition:{family}:at_target_before_{step.action.verb}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.PRECONDITION,
                    summary="Being at container enables open(container).",
                    structure={
                        "precondition": {"location_matches_container": True},
                        "action": "open(container)",
                        "task_family": family,
                        "pattern_kind": "precondition",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=step.feedback.success,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.05 if step.feedback.success else 0.0,
                )
                candidates.append(candidate)

            goal_roles = ALFWorldAdapter(scene_label=episode.scene_id or "alfworld_scene").goal_slots(episode.goal)
            target_object = str(goal_roles.get("object", ""))
            goal_destination = str(goal_roles.get("destination", ""))
            moved_object = str(step.action.slots.get("object", ""))
            move_destination = str(step.action.slots.get("destination", ""))
            target_match = bool(target_object and moved_object and re.sub(r"_\\d+$", "", moved_object) == target_object)
            if (
                step.action.verb == "move"
                and step.action.slots.get("object") in step.state.held_objects
                and step.feedback.success
                and target_match
                and goal_destination
                and move_destination == goal_destination
            ):
                cid = f"precondition:holding_target_before_goal_move:{family}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.PRECONDITION,
                    summary="Holding the target object enables move(target_object, goal_destination).",
                    structure={
                        "precondition": {"holding_target_object": True},
                        "action": "move(target_object,goal_destination)",
                        "task_family": family,
                        "pattern_kind": "precondition",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.08,
                )
                candidates.append(candidate)

            if step.feedback.failure_label:
                abstract_action = _abstract_action_text(step.action, roles)
                cid = f"failure:{family}:{step.action.verb}:{step.feedback.failure_label}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.FAILURE,
                    summary=f"{abstract_action} fails under {step.feedback.failure_label}.",
                    structure={
                        "action": abstract_action,
                        "failure_label": step.feedback.failure_label,
                        "task_family": family,
                        "pattern_kind": "failure",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    utility_delta=0.03,
                )
                candidates.append(candidate)

        for prev_step, next_step in zip(episode.steps, episode.steps[1:]):
            if prev_step.feedback.failure_label and next_step.feedback.success:
                abstract_repair = _abstract_action_text(next_step.action, roles)
                cid = f"repair:{family}:{prev_step.feedback.failure_label}:{next_step.action.verb}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.REPAIR,
                    summary=f"After {prev_step.feedback.failure_label}, try {abstract_repair}.",
                    structure={
                        "failure_label": prev_step.feedback.failure_label,
                        "repair_action": abstract_repair,
                        "task_family": family,
                        "pattern_kind": "repair",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=True,
                    utility_delta=0.06,
                )
                candidates.append(candidate)

        if episode_success and len(episode.steps) >= 2:
            suffix = episode.steps[-min(3, len(episode.steps)) :]
            closure_actions = tuple(_abstract_action_text(step.action, roles) for step in suffix)
            cid = f"closure:{family}:{'->'.join(closure_actions)}"
            candidate = PromotionCandidate(
                candidate_id=cid,
                candidate_type=CandidateType.WORKFLOW,
                summary=f"Closure pattern: {' -> '.join(closure_actions)}",
                structure={
                    "workflow": closure_actions,
                    "task_family": family,
                    "pattern_kind": "closure",
                },
            )
            candidate.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                positive=True,
                utility_delta=0.18,
            )
            candidates.append(candidate)

        if stalled_steps:
            stalled_actions = tuple(
                _abstract_action_text(step.action, roles)
                for step in episode.steps
                if step.step_idx in stalled_steps
            )
            if stalled_actions:
                cid = f"anti:{family}:{'->'.join(stalled_actions[:3])}"
                candidate = PromotionCandidate(
                    candidate_id=cid,
                    candidate_type=CandidateType.FAILURE,
                    summary=f"Anti-pattern: repeated {' -> '.join(stalled_actions[:3])} leads to stall.",
                    structure={
                        "anti_pattern": stalled_actions[:3],
                        "task_family": family,
                        "pattern_kind": "anti_pattern",
                    },
                )
                candidate.observe(
                    scene_id=episode.scene_id,
                    episode_id=episode_graph.episode_id,
                    positive=False,
                    stalled=True,
                    utility_delta=0.0,
                )
                candidates.append(candidate)

        return candidates

    def _induce_rules(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        episode_success: bool,
    ) -> list[MemoryRule]:
        adapter = ALFWorldAdapter(scene_label=episode.scene_id)
        family = adapter.infer_task_family(episode.goal)
        goal_roles = adapter.goal_slots(episode.goal)
        goal_arity = adapter.required_goal_count(episode.goal)
        roles = _task_roles(episode)
        beliefs = _replay_episode_beliefs(episode)
        rules: dict[str, MemoryRule] = {}

        def upsert_rule(
            *,
            rule_type: RuleType,
            summary: str,
            progress_state: str,
            condition: dict,
            effect: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            rid = f"{rule_type.value}:{family}:{progress_state}:{_condition_signature(condition)}:{_condition_signature(effect)}"
            rule = rules.get(rid)
            if rule is None:
                rule = MemoryRule(
                    rule_id=rid,
                    rule_type=rule_type,
                    summary=summary,
                    task_family=family,
                    goal_arity=goal_arity,
                    progress_state=progress_state,
                    goal_roles=dict(goal_roles),
                    condition=dict(condition),
                    effect=dict(effect),
                )
                rules[rid] = rule
            rule.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }

        for idx, step in enumerate(episode.steps):
            belief = beliefs[idx]
            progress = belief.progress
            next_progress = belief.next_progress
            progress_state = str(progress["progress_state"])
            next_progress_state = str(next_progress["progress_state"])
            abstract_action = _abstract_action_text(step.action, roles)
            action_role = _normalize_entity(step.action.slots.get("object", "") or step.action.slots.get("target", "") or step.action.slots.get("destination", "") or step.action.slots.get("container", ""), roles)
            target_object = str(goal_roles.get("object", ""))
            target_base = re.sub(r"_\d+$", "", target_object)
            raw_object_slot = str(step.action.slots.get("object", ""))
            raw_object_base = re.sub(r"_\d+$", "", raw_object_slot.lower())
            action_condition = {
                "action_family": step.action.family.value,
                "location_role": _location_signature(step.state.location, roles),
                "holding_target": progress["held_relevant_count"] > 0,
                "goal_visible": progress["goal_object_matches_visible"],
                "action_role": action_role,
            }
            action_effect = {"prefer_action": abstract_action, "action_role": action_role}
            aligned_action = _goal_aligned_action(step.action, progress_state=progress_state, roles=roles)
            goal_advancing_step = _step_advances_goal(
                step,
                progress,
                next_progress,
                revisit_count=belief.revisit_count,
            )

            if (
                step.action.verb == "take"
                and target_base
                and raw_object_base
                and raw_object_base != target_base
            ):
                upsert_rule(
                    rule_type=RuleType.BLOCKED,
                    summary=f"Blocked: during {progress_state}, avoid {abstract_action} when searching for {target_base}.",
                    progress_state=progress_state,
                    condition={
                        "holding_target": progress["held_relevant_count"] > 0,
                        "target_object": target_base,
                    },
                    effect={"block_action": abstract_action, "action_role": "non_target_object"},
                    success=False,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.0,
                )

            if step.feedback.success and aligned_action:
                allow_precondition = True
                if step.action.verb == "go" and progress_state not in {"carry_target", "finalize"}:
                    allow_precondition = False
                if step.action.verb in {"go", "look"} and not goal_advancing_step:
                    allow_precondition = False
                if step.action.verb in {"open", "examine"} and belief.revisit_count > 1 and not goal_advancing_step:
                    allow_precondition = False

                if allow_precondition:
                    upsert_rule(
                        rule_type=RuleType.PRECONDITION,
                        summary=f"Precondition: when {progress_state}, prefer {abstract_action}.",
                        progress_state=progress_state,
                        condition=action_condition,
                        effect=action_effect,
                        success=True,
                        stalled=step.step_idx in stalled_steps,
                        utility_delta=0.08,
                    )

                if (next_progress_state != progress_state or goal_advancing_step) and not (
                    step.action.verb == "go" and progress_state not in {"carry_target", "finalize"}
                ):
                    upsert_rule(
                        rule_type=RuleType.WORKFLOW,
                        summary=(
                            f"Workflow: {progress_state} -> {next_progress_state} via {step.action.verb}."
                            if next_progress_state != progress_state
                            else f"Workflow: within {progress_state}, {step.action.verb} advances the search."
                        ),
                        progress_state=progress_state,
                        condition={
                            "from": progress_state,
                            "holding_target": progress["held_relevant_count"] > 0,
                            "remaining": progress["remaining_relevant_count"],
                        },
                        effect={
                            "to": next_progress_state,
                            "via": step.action.verb,
                            "prefer_action": abstract_action,
                        },
                        success=True,
                        utility_delta=0.12 if next_progress_state != progress_state else 0.06,
                    )

                if (
                    goal_arity == 1
                    and progress_state in {"carry_target", "finalize"}
                    and step.action.verb == "move"
                    and goal_roles.get("destination")
                    and step.action.slots.get("destination") == goal_roles.get("destination")
                ):
                    upsert_rule(
                        rule_type=RuleType.CLOSURE,
                        summary=f"Closure: when carrying the target, prefer {abstract_action}.",
                        progress_state=progress_state,
                        condition={
                            "holding_target": True,
                            "remaining": progress["remaining_relevant_count"],
                        },
                        effect={"prefer_action": abstract_action, "action_role": "goal_destination"},
                        success=True,
                        utility_delta=0.18,
                    )
            else:
                upsert_rule(
                    rule_type=RuleType.BLOCKED,
                    summary=f"Blocked: avoid {abstract_action} under {step.feedback.failure_label or 'current context'}.",
                    progress_state=progress_state,
                    condition={
                        "failure_label": step.feedback.failure_label or "",
                        "holding_target": progress["held_relevant_count"] > 0,
                    },
                    effect={"block_action": abstract_action, "action_role": action_role},
                    success=False,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.0,
                )

            if (
                goal_arity > 1
                and progress_state == "search_second"
                and progress["held_relevant_count"] > 0
                and step.action.verb == "move"
                and goal_roles.get("destination")
                and step.action.slots.get("destination") != goal_roles.get("destination")
            ):
                upsert_rule(
                    rule_type=RuleType.BLOCKED,
                    summary="Blocked: while searching for the remaining target, do not place the held target on a non-goal destination.",
                    progress_state=progress_state,
                    condition={
                        "holding_target": True,
                        "remaining": progress["remaining_relevant_count"],
                    },
                    effect={"block_action": abstract_action, "action_role": "non_goal_destination"},
                    success=False,
                    utility_delta=0.0,
                )

            if idx > 0:
                prev_step = episode.steps[idx - 1]
                prev_progress = beliefs[idx - 1].progress
                if prev_step.feedback.failure_label and step.feedback.success and goal_advancing_step:
                    upsert_rule(
                        rule_type=RuleType.REPAIR,
                        summary=f"Repair: after {prev_step.feedback.failure_label}, try {abstract_action}.",
                        progress_state=str(prev_progress["progress_state"]),
                        condition={
                            "failure_label": prev_step.feedback.failure_label,
                            "holding_target": prev_progress["held_relevant_count"] > 0,
                        },
                        effect={"prefer_action": abstract_action, "action_role": action_role},
                        success=True,
                        utility_delta=0.1,
                    )

        return list(rules.values())

    def _induce_artifacts(
        self,
        episode_graph: EpisodeGraph,
        episode: EpisodeRecord,
        *,
        candidates: list[PromotionCandidate],
        rules: list[MemoryRule],
        episode_success: bool,
    ) -> list[MemoryArtifact]:
        adapter = ALFWorldAdapter(scene_label=episode.scene_id)
        family = adapter.infer_task_family(episode.goal)
        goal_roles = adapter.goal_slots(episode.goal)
        goal_arity = adapter.required_goal_count(episode.goal)
        roles = _task_roles(episode)
        beliefs = _replay_episode_beliefs(episode)
        artifacts: dict[str, MemoryArtifact] = {}

        node_ids_by_signature = {node.signature: node.node_id for node in episode_graph.nodes.values()}
        edge_refs = {edge.signature for edge in episode_graph.edges}
        stalled_steps = {
            step.step_idx
            for step in episode.steps
            if not step.feedback.failure_label and not step.feedback.state_delta and not step.feedback.done
        }

        def upsert_artifact(
            *,
            kind: ArtifactKind,
            summary: str,
            anchor: dict,
            payload: dict,
            success: bool,
            stalled: bool = False,
            utility_delta: float = 0.0,
        ) -> None:
            aid = f"{kind.value}:{_condition_signature(anchor)}:{_condition_signature(payload)}"
            artifact = artifacts.get(aid)
            if artifact is None:
                artifact = MemoryArtifact(
                    artifact_id=aid,
                    kind=kind,
                    summary=summary,
                    anchor=dict(anchor),
                    payload=dict(payload),
                )
                artifacts[aid] = artifact
            artifact.observe(
                scene_id=episode.scene_id,
                episode_id=episode_graph.episode_id,
                success=success,
                stalled=stalled,
                utility_delta=utility_delta,
            )

        for candidate in candidates:
            if candidate.positive < max(1, candidate.negative):
                continue
            if candidate.candidate_type == CandidateType.PRECONDITION:
                continue
            pattern_kind = str(candidate.structure.get("pattern_kind", candidate.candidate_type.value))
            if candidate.candidate_type == CandidateType.FAILURE and pattern_kind != "anti_pattern":
                continue
            action_patterns = list(_candidate_action_patterns(candidate))
            upsert_artifact(
                kind=ArtifactKind.PROTOTYPE,
                summary=f"Prototype: {candidate.summary}",
                anchor={
                    "task_family": family,
                    "goal_arity": goal_arity,
                    "pattern_kind": pattern_kind,
                    "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                },
                payload={
                    "source": "candidate",
                    "pattern_kind": pattern_kind,
                    "structure": dict(candidate.structure),
                    "action_patterns": action_patterns,
                },
                success=True,
                utility_delta=max(candidate.utility, 0.02),
            )

        for rule in rules:
            upsert_artifact(
                kind=ArtifactKind.RULE,
                summary=rule.summary,
                anchor={
                    "task_family": rule.task_family,
                    "goal_arity": rule.goal_arity,
                    "progress_state": rule.progress_state,
                    "goal_signature": f"{rule.goal_roles.get('object', '')}->{rule.goal_roles.get('destination', '')}",
                    "rule_type": rule.rule_type.value,
                },
                payload={
                    "source": "rule",
                    "rule_type": rule.rule_type.value,
                    "condition": dict(rule.condition),
                    "effect": dict(rule.effect),
                    "action_patterns": list(_rule_action_patterns(rule)),
                },
                success=rule.stats.success >= rule.stats.failure,
                stalled=rule.stats.stalled > 0,
                utility_delta=max(rule.stats.utility_avg, 0.0),
            )

        for idx, step in enumerate(episode.steps):
            belief = beliefs[idx]
            progress = belief.progress
            next_progress = belief.next_progress
            progress_state = str(progress["progress_state"])
            next_progress_state = str(next_progress["progress_state"])
            abstract_action = _abstract_action_text(step.action, roles)
            action_role = _normalize_entity(
                step.action.slots.get("object", "")
                or step.action.slots.get("target", "")
                or step.action.slots.get("destination", "")
                or step.action.slots.get("container", ""),
                roles,
            )
            graph_refs = [
                signature
                for signature in (
                    f"state:{step.state.signature}|temporal|action:{step.action.canonical_str}",
                    f"action:{step.action.canonical_str}|temporal|state:{step.next_state.signature}",
                )
                if signature in edge_refs
            ]
            aligned_action = _goal_aligned_action(step.action, progress_state=progress_state, roles=roles)
            goal_advancing_step = _step_advances_goal(
                step,
                progress,
                next_progress,
                revisit_count=belief.revisit_count,
            )

            if (
                step.feedback.success
                and aligned_action
                and (next_progress_state != progress_state or goal_advancing_step)
                and not (step.action.verb == "go" and progress_state not in {"carry_target", "finalize"})
            ):
                upsert_artifact(
                    kind=ArtifactKind.PROTOTYPE,
                    summary=(
                        f"Prototype: {progress_state} -> {next_progress_state} via {step.action.verb}."
                        if next_progress_state != progress_state
                        else f"Prototype: within {progress_state}, {step.action.verb} supports progress."
                    ),
                    anchor={
                        "task_family": family,
                        "goal_arity": goal_arity,
                        "progress_state": progress_state,
                        "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                        "action_family": step.action.family.value,
                        "role_focus": action_role,
                    },
                    payload={
                        "source": "episode_graph",
                        "transition": {"from": progress_state, "to": next_progress_state, "via": step.action.verb},
                        "action_patterns": [abstract_action],
                        "graph_refs": graph_refs,
                        "state_ref": node_ids_by_signature.get(step.state.signature, ""),
                        "next_state_ref": node_ids_by_signature.get(step.next_state.signature, ""),
                    },
                    success=True,
                    stalled=step.step_idx in stalled_steps,
                    utility_delta=0.08 if next_progress_state == progress_state else 0.14,
                )

            diagnosis = ""
            if step.feedback.failure_label:
                diagnosis = step.feedback.failure_label
            elif step.step_idx in stalled_steps and (belief.revisit_count > 0 or step.action.verb in {"look", "examine", "open"}):
                diagnosis = "search_stall"
            else:
                target_object = str(goal_roles.get("object", ""))
                target_base = re.sub(r"_\d+$", "", target_object)
                raw_object = str(step.action.slots.get("object", ""))
                raw_object_base = re.sub(r"_\d+$", "", raw_object.lower())
                if step.action.verb == "take" and target_base and raw_object_base and raw_object_base != target_base:
                    diagnosis = "wrong_target_object"
                elif (
                    step.action.verb == "go"
                    and step.action.slots.get("target") == goal_roles.get("destination")
                    and progress_state not in {"carry_target", "finalize"}
                    and progress["held_relevant_count"] <= 0
                ):
                    diagnosis = "premature_destination_focus"
                elif (
                    goal_arity > 1
                    and progress_state == "search_second"
                    and progress["held_relevant_count"] > 0
                    and step.action.verb == "move"
                    and goal_roles.get("destination")
                    and step.action.slots.get("destination") != goal_roles.get("destination")
                ):
                    diagnosis = "premature_non_goal_place"

            if not diagnosis:
                continue

            repair_patterns: list[str] = []
            for future_offset, future_step in enumerate(episode.steps[idx + 1 : idx + 5], start=idx + 1):
                future_belief = beliefs[future_offset]
                if future_step.feedback.success and _step_advances_goal(
                    future_step,
                    future_belief.progress,
                    future_belief.next_progress,
                    revisit_count=future_belief.revisit_count,
                ):
                    repair_patterns.append(_abstract_action_text(future_step.action, roles))
                    break
            if not repair_patterns:
                continue

            upsert_artifact(
                kind=ArtifactKind.REFLECTION,
                summary=f"Reflection: during {progress_state}, {diagnosis.replace('_', ' ')} tends to derail progress.",
                anchor={
                    "task_family": family,
                    "goal_arity": goal_arity,
                    "progress_state": progress_state,
                    "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                    "failure_signature": diagnosis,
                },
                payload={
                    "source": "episode_graph",
                    "diagnosis": diagnosis,
                    "trigger_action": abstract_action,
                    "avoid_patterns": [abstract_action],
                    "repair_patterns": repair_patterns,
                    "graph_refs": graph_refs,
                    "holding_target": bool(progress["held_relevant_count"] > 0),
                },
                success=False,
                stalled=step.step_idx in stalled_steps,
                utility_delta=0.12 if repair_patterns else 0.04,
            )

        if episode_success:
            plan_template = _induce_plan_template(
                episode,
                beliefs=beliefs,
                roles=roles,
                goal_roles=goal_roles,
                goal_arity=goal_arity,
            )
            if plan_template is not None:
                summary, plan_steps, action_patterns = plan_template
                upsert_artifact(
                    kind=ArtifactKind.PROTOTYPE,
                    summary=summary,
                    anchor={
                        "task_family": family,
                        "goal_arity": goal_arity,
                        "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                        "artifact_role": "plan",
                    },
                    payload={
                        "source": "plan_template",
                        "pattern_kind": "plan",
                        "plan_steps": plan_steps,
                        "action_patterns": action_patterns,
                    },
                    success=True,
                    utility_delta=0.16,
                )

            layout_id = str(episode.metadata.get("layout_id", ""))
            target_base = re.sub(r"_\d+$", "", str(goal_roles.get("object", "")).lower())
            for idx, step in enumerate(episode.steps):
                belief = beliefs[idx]
                next_progress = belief.next_progress
                progress = belief.progress
                if (
                    step.feedback.success
                    and step.action.verb == "take"
                    and target_base
                    and _is_target_object(step.action, target_base)
                ):
                    source_raw = str(step.action.slots.get("source", "")).lower()
                    source_instance = source_raw
                    source_base = re.sub(r"_\d+$", "", source_instance)
                    source_role = _normalize_entity(source_raw, roles)
                    upsert_artifact(
                        kind=ArtifactKind.PROTOTYPE,
                        summary=f"Scene relation: in this layout, target_object was found at {source_instance} ({source_role}).",
                        anchor={
                            "task_family": family,
                            "goal_arity": goal_arity,
                            "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                            "artifact_role": "scene_relation",
                            "layout_id": layout_id,
                        },
                        payload={
                            "source": "scene_relation",
                            "pattern_kind": "scene_relation",
                            "relation_kind": "object_location_prior",
                            "object_role": "target_object",
                            "source_role": source_role,
                            "source_base": source_base,
                            "source_instance": source_instance,
                            "evidence_type": "take_success",
                            "action_patterns": _relation_action_patterns(source_instance, source_base, source_role),
                        },
                        success=True,
                        utility_delta=0.14,
                    )

                search_ref = str(belief.search_ref or "").lower()
                if (
                    step.feedback.success
                    and search_ref
                    and step.action.verb in {"open", "examine", "look"}
                    and not bool(next_progress.get("goal_object_matches_visible"))
                    and int(progress.get("held_relevant_count", 0)) <= 0
                ):
                    source_instance = search_ref
                    source_base = re.sub(r"_\d+$", "", source_instance)
                    source_role = _normalize_entity(search_ref, roles)
                    upsert_artifact(
                        kind=ArtifactKind.PROTOTYPE,
                        summary=f"Scene relation: in this layout, target_object was not found after searching {source_instance}.",
                        anchor={
                            "task_family": family,
                            "goal_arity": goal_arity,
                            "goal_signature": f"{goal_roles.get('object', '')}->{goal_roles.get('destination', '')}",
                            "artifact_role": "scene_relation",
                            "layout_id": layout_id,
                        },
                        payload={
                            "source": "scene_relation",
                            "pattern_kind": "scene_relation",
                            "relation_kind": "searched_empty",
                            "object_role": "target_object",
                            "source_role": source_role,
                            "source_base": source_base,
                            "source_instance": source_instance,
                            "evidence_type": "searched_empty",
                            "action_patterns": _relation_action_patterns(source_instance, source_base, source_role),
                        },
                        success=True,
                        utility_delta=0.08,
                    )

        return list(artifacts.values())


class GlobalPromoter:
    def __init__(self, score_threshold: float = 0.35) -> None:
        self.score_threshold = score_threshold

    def _specificity_penalty(self, candidate: PromotionCandidate) -> float:
        lowered = candidate.summary.lower()
        specific_tokens = re.findall(r"\b[a-z]+_\d+\b", lowered)
        if not specific_tokens:
            return 0.0
        return min(0.24, 0.08 * len(specific_tokens))

    def _transferability_score(self, candidate: PromotionCandidate) -> float:
        support = candidate.positive + candidate.negative
        if support <= 0:
            return -1.0
        confidence = candidate.positive / support
        negative_rate = candidate.negative / support
        stall_rate = candidate.stalled / support
        scene_term = min(candidate.coverage / 3.0, 1.0)
        utility_term = candidate.utility / support
        family_term = 0.15 if candidate.structure.get("task_family") else 0.0
        abstraction_term = 0.1 if self._specificity_penalty(candidate) == 0.0 else -self._specificity_penalty(candidate)
        return (
            0.65 * confidence
            + 0.45 * scene_term
            + 0.25 * utility_term
            + family_term
            + abstraction_term
            - 0.45 * negative_rate
            - 0.55 * stall_rate
        )

    def _min_scene_coverage(self, candidate: PromotionCandidate) -> int:
        pattern_kind = str(candidate.structure.get("pattern_kind", ""))
        support = candidate.positive + candidate.negative
        abstract_enough = self._specificity_penalty(candidate) == 0.0
        if candidate.candidate_type == CandidateType.WORKFLOW and pattern_kind in {"closure", "workflow"}:
            if abstract_enough and candidate.positive >= 4 and support >= 4 and candidate.negative == 0:
                return 1
        return 2

    def promote(self, global_memory: GlobalGraphMemory, local_memories: list[LocalGraphMemory], batch_name: str) -> GlobalGraphMemory:
        aggregated: dict[str, PromotionCandidate] = {}
        for local in local_memories:
            for candidate in local.candidates.values():
                merged = aggregated.get(candidate.candidate_id)
                if merged is None:
                    merged = PromotionCandidate(
                        candidate_id=candidate.candidate_id,
                        candidate_type=candidate.candidate_type,
                        summary=candidate.summary,
                        structure=dict(candidate.structure),
                    )
                    aggregated[candidate.candidate_id] = merged
                merged.source_episode_ids |= candidate.source_episode_ids
                merged.source_scenes |= candidate.source_scenes
                merged.positive += candidate.positive
                merged.negative += candidate.negative
                merged.stalled += candidate.stalled
                merged.utility += candidate.utility

        for candidate in aggregated.values():
            pattern_kind = str(candidate.structure.get("pattern_kind", ""))
            transferability = self._transferability_score(candidate)
            adjusted_score = 0.55 * candidate.prior_score + 0.9 * transferability
            min_scene_coverage = self._min_scene_coverage(candidate)
            support = candidate.positive + candidate.negative
            confidence = candidate.confidence

            if candidate.candidate_type == CandidateType.PRECONDITION:
                if support < 2 or candidate.positive < 2 or confidence < 0.6:
                    continue
            elif candidate.candidate_type == CandidateType.WORKFLOW and pattern_kind == "workflow":
                if candidate.positive < 2 or candidate.positive < candidate.negative:
                    continue
            elif candidate.candidate_type == CandidateType.WORKFLOW and pattern_kind == "closure":
                if candidate.positive < 2:
                    continue
            elif candidate.candidate_type == CandidateType.FAILURE and pattern_kind == "anti_pattern":
                if candidate.stalled < 2:
                    continue
            if adjusted_score >= self.score_threshold and candidate.coverage >= min_scene_coverage:
                global_memory.candidates[candidate.candidate_id] = candidate

        aggregated_rules: dict[str, MemoryRule] = {}
        for local in local_memories:
            for rule in local.rules_by_id.values():
                merged = aggregated_rules.get(rule.rule_id)
                if merged is None:
                    merged = MemoryRule(
                        rule_id=rule.rule_id,
                        rule_type=rule.rule_type,
                        summary=rule.summary,
                        task_family=rule.task_family,
                        goal_arity=rule.goal_arity,
                        progress_state=rule.progress_state,
                        goal_roles=dict(rule.goal_roles),
                        condition=dict(rule.condition),
                        effect=dict(rule.effect),
                    )
                    aggregated_rules[rule.rule_id] = merged
                merged.source_episode_ids |= rule.source_episode_ids
                merged.source_scenes |= rule.source_scenes
                merged.stats.support += rule.stats.support
                merged.stats.success += rule.stats.success
                merged.stats.failure += rule.stats.failure
                merged.stats.stalled += rule.stats.stalled
                merged.stats.utility += rule.stats.utility
                merged.stats.transfer_success += rule.stats.transfer_success
                merged.stats.transfer_trials += rule.stats.transfer_trials
                merged.specificity = max(merged.specificity, rule.specificity)
                merged.conflict = max(merged.conflict, rule.conflict)

        for rule in aggregated_rules.values():
            min_scene_coverage = 2
            if rule.support >= 2 and rule.coverage >= min_scene_coverage and rule.promotion_score >= self.score_threshold:
                global_memory.rules_by_id[rule.rule_id] = rule

        aggregated_artifacts: dict[str, MemoryArtifact] = {}
        for local in local_memories:
            for artifact in local.artifacts_by_id.values():
                merged = aggregated_artifacts.get(artifact.artifact_id)
                if merged is None:
                    merged = MemoryArtifact(
                        artifact_id=artifact.artifact_id,
                        kind=artifact.kind,
                        summary=artifact.summary,
                        anchor=dict(artifact.anchor),
                        payload=dict(artifact.payload),
                        level="global",
                    )
                    aggregated_artifacts[artifact.artifact_id] = merged
                merged.source_episode_ids |= artifact.source_episode_ids
                merged.source_scenes |= artifact.source_scenes
                merged.stats.support += artifact.stats.support
                merged.stats.success += artifact.stats.success
                merged.stats.failure += artifact.stats.failure
                merged.stats.stalled += artifact.stats.stalled
                merged.stats.utility += artifact.stats.utility
                merged.stats.transfer_success += artifact.stats.transfer_success
                merged.stats.transfer_trials += artifact.stats.transfer_trials
                merged.specificity = max(merged.specificity, artifact.specificity)
                merged.conflict = max(merged.conflict, artifact.conflict)

        for artifact in aggregated_artifacts.values():
            min_scene_coverage = 2
            min_support = 2 if artifact.kind != ArtifactKind.REFLECTION else 1
            if artifact.support >= min_support and artifact.coverage >= min_scene_coverage and artifact.promotion_score >= self.score_threshold:
                global_memory.artifacts_by_id[artifact.artifact_id] = artifact

        global_memory.promoted_batches.append(batch_name)
        return global_memory
