from dataclasses import dataclass
from typing import Any
import hashlib
import json
import math
import os
import re
import time
import networkx as nx
from networkx.readwrite import json_graph

from .memory_base import MASMemoryBase
from .local_instance_graph import LocalInstanceGraph
from .global_experience_graph import GlobalExperienceGraph
from ..common import MASMessage, StateChain
from mas.llm import Message
from mas.utils import write_json, load_json
from .prompt import GMemoryPrompts


_LOCAL_GRAPH_SYSTEM_PROMPT = (
    "You are an information extraction assistant for an embodied agent.\n"
    "Given an observation from the environment, you must extract:\n"
    "1) concrete object instances with their attributes\n"
    "2) relations between instances/events (subject, relation, object).\n"
    "\n"
    "IMPORTANT:\n"
    "- Do NOT create any instance that represents the agent/player (e.g., category 'Agent', labels containing 'agent', 'player', 'you').\n"
    "- Do NOT use 'Agent'/'agent' as instance_id, label, or category.\n"
    "- Do NOT restrict relation types. Relations may be spatial (On/Inside/At), "
    "stateful (IsOpen/IsDirty), causal/temporal (After/Before/LeadsTo), or verb/action-like.\n"
    "\n"
    "ACTION RELATIONS (CRITICAL):\n"
    "- You MUST actively extract action/verb relations whenever the observation implies an interaction.\n"
    "- Pay special attention to these verbs and their variants/synonyms: take, open, put, clean, heat, cool, use.\n"
    "- Prefer normalized relation labels such as: Take/TakeFrom, Put/PutIn/PutOn, Open/Close, Clean, Heat, Cool, Use.\n"
    "- Create verb edges between the most relevant object instances mentioned. If the agent is the doer, OMIT the agent node "
    "and connect the affected objects directly (e.g., Apple-[TakeFrom]->Fridge; Soap-[Clean]->Sink; Knife-[Use]->Bread).\n"
    "- If the observation describes an action/event without a clear object, you may create an Event_* instance and connect it.\n"
    "- Always return a single valid JSON object only. No extra text."
)

_LOCAL_GRAPH_USER_PROMPT = (
    "Extract entities and relations from the following observation.\n"
    "Return a JSON object with the following schema.\n"
    "\n"
    "Guidelines:\n"
    "- Do NOT add any 'Agent' / 'agent' / 'player' / 'you' instance. The graph should represent the scene/world only.\n"
    "- Do NOT restrict relation types: any verb/action word is allowed as a relation label.\n"
    "- ACTION RELATIONS: in addition to spatial relations, you MUST infer verb edges for interactions, especially for: take/open/put/clean/heat/cool/use.\n"
    "  Examples (use similar normalized labels):\n"
    "  - \"You take the apple from the fridge.\" => Apple_1 -[TakeFrom]-> Fridge_1\n"
    "  - \"You put the apple in the microwave.\" => Apple_1 -[PutIn]-> Microwave_1\n"
    "  - \"You open the cabinet.\" => Cabinet_1 -[Open]-> Event_1 (or Cabinet_1 -[IsOpen]-> Open if you prefer state edge)\n"
    "  - \"You clean the sink with soap.\" => Soap_1 -[Clean]-> Sink_1 (and/or Sink_1 -[IsClean]-> Clean)\n"
    "  - \"You heat the soup using the microwave.\" => Microwave_1 -[Heat]-> Soup_1\n"
    "- The subject/object fields should refer to instance_id strings. If you must reference an "
    "implicit event, create an instance_id for it (e.g., Event_1) and connect it with edges.\n"
    "- Prefer stable instance ids within the same observation (e.g., 'cabinet 2' -> Cabinet_2).\n"
    "\n"
    "Schema:\n"
    "{{\n"
    '  "instances": [\n'
    "    {{\n"
    '      "instance_id": "string, unique id for this instance (e.g., Apple_1, Sink_A)",\n'
    '      "label": "string, surface name in the text",\n'
    '      "category": "string, abstract type (e.g., Apple, Sink, Fork)",\n'
    '      "state": "string, optional state (e.g., Clean, Dirty, Sliced, Open, Closed)",\n'
    '      "room": "string, optional room or region name (e.g., Kitchen, LivingRoom)"\n'
    "    }}, ...\n"
    "  ],\n"
    '  "relations": [\n'
    "    {{\n"
    '      "subject": "string, instance_id of subject",\n'
    '      "relation": "string, relation label (ANY string allowed, including verbs/actions like Open/TakeFrom/GoTo/Wash)",\n'
    '      "object": "string, instance_id of object"\n'
    "    }}, ...\n"
    "  ]\n"
    "}}\n"
    "If some fields are unknown, you may omit them.\n"
    "Observation:\n"
    "{observation}\n"
    "\n"
    "Optional room hint (may be empty): {room_hint}\n"
)

_INSIGHT_SYSTEM_PROMPT = (
    "You are an experience abstraction assistant for embodied agents.\n"
    "Given a successful task and trajectory, produce HIGH-LEVEL, reusable experience rules.\n"
    "Rules must be general and transferable across tasks, not tied to specific instance ids.\n"
    "Desired abstraction level examples:\n"
    "- To change the state of an object (e.g., clean/checked), bring it to a functional receptacle and execute the relevant action.\n"
    "- Small movable objects are more likely to appear in enclosed receptacles than on open floor regions.\n"
    "Return ONLY a JSON array of strings, each being one concise insight."
)

_INSIGHT_USER_PROMPT = (
    "Task:\n{task}\n\n"
    "Trajectory:\n{trajectory}\n\n"
    "Output 2-4 high-level experience insights as a JSON array.\n"
    "Constraints:\n"
    "- No task-specific instance numbers (e.g., Apple_1, Drawer_3).\n"
    "- Emphasize action preconditions, search priors, and state-change strategies.\n"
    "- Keep each insight one sentence."
)

_QUERY_ENTITY_SYSTEM_PROMPT = "You extract key entities/concepts from a task query for retrieval."

_QUERY_ENTITY_USER_PROMPT = (
    "Extract key object/place/action concepts from the query text below.\n"
    "Return ONLY a JSON array of lowercase strings.\n\n"
    "Query:\n{query}"
)

_MERGED_PATH_SUMMARY_SYSTEM = (
    "You synthesize common embodied-agent strategies from several past task queries and short trajectory excerpts.\n"
    "Output ONE concise paragraph (3-6 sentences) of transferable guidance. No bullet lists. No JSON."
)

_MERGED_PATH_SUMMARY_USER = (
    "Current task query:\n{current_query}\n\n"
    "Inferred relation path between matched concepts in the merged scene graph:\n{path_text}\n\n"
    "Similar past cases (for reference):\n{cases_block}\n\n"
    "Write the common experience paragraph now."
)


@dataclass
class SelectiveMemMASMemory(MASMemoryBase):
    """
    SelectiveMem: decoupled Local Instance Graph (GL) and Global Experience Graph (GG).

    This class reuses the standard MASMemoryBase interface so it can be plugged in
    via `mas_memory` argument just like `g-memory` or `memgraph`.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.local_graph = LocalInstanceGraph()
        self._load_local_instance_graph_from_disk_best_effort()
        self.global_graph = GlobalExperienceGraph(persist_dir=self.persist_dir)
        self.insight_bank_path = os.path.join(self.persist_dir, "insight_bank.json")
        self.insight_bank: list[dict[str, Any]] = self._load_insight_bank()
        self._query_insight_archive_path = os.path.join(
            self.persist_dir, "query_insight_archive.json"
        )
        self._merged_lg_path = os.path.join(self.persist_dir, "merged_instance_graph.json")
        self._fingerprint_path = os.path.join(self.persist_dir, "trajectory_fingerprints.jsonl")
        self.query_insight_archive: list[dict[str, Any]] = self._load_query_insight_archive()
        self.merged_lg: nx.MultiDiGraph = nx.MultiDiGraph()
        self._load_merged_lg_from_disk()
        self._fingerprint_records: list[dict[str, Any]] = self._load_fingerprint_records()
        # Success-case counters for periodic visualization/recording.
        self._success_case_counter: int = len(self._fingerprint_records)
        self._viz_every_n_success: int = int(self.global_config.get("viz_every_n_success", 1))

        # Retrieval & summarization audit logs.
        self._retrieval_trace_path = os.path.join(self.persist_dir, "retrieval_trace.jsonl")
        self._gg_common_summaries_path = os.path.join(self.persist_dir, "gg_common_summaries.jsonl")
        self.last_saved_message: MASMessage | None = None
        self._global_retriever: "SelectiveMemMASMemory | None" = None
        self._insights_only_mode: bool = bool(self.global_config.get("insights_only", False))

    def _load_local_instance_graph_from_disk_best_effort(self) -> None:
        """
        Ensure local LG is available for later test runs.
        The training process writes `local_instance_graph.json` into `persist_dir`;
        this loader reconstructs the in-memory MultiDiGraph so LocalPath retrieval works
        without re-running the whole training subset.
        """
        try:
            path = os.path.join(self.persist_dir, "local_instance_graph.json")
            raw = load_json(path)
            if not isinstance(raw, dict):
                return
            nodes = raw.get("nodes") or []
            edges = raw.get("edges") or []
            if not isinstance(nodes, list) or not isinstance(edges, list):
                return
            g = nx.MultiDiGraph()
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                nid = n.get("id")
                if nid is None:
                    continue
                attrs = {k: v for k, v in n.items() if k != "id"}
                g.add_node(str(nid), **attrs)
            for e in edges:
                if not isinstance(e, dict):
                    continue
                u = e.get("from")
                v = e.get("to")
                if u is None or v is None:
                    continue
                key = e.get("key", 0)
                attrs = {k: v for k, v in e.items() if k not in {"from", "to", "key"}}
                try:
                    key = int(key)
                except Exception:
                    key = 0
                g.add_edge(str(u), str(v), key=key, **attrs)
            self.local_graph.graph = g
        except Exception:
            # Best-effort: never crash evaluation if loading fails.
            pass

    # ---- write side ----

    def add_memory(self, mas_message: MASMessage) -> None:
        """Entry point when a task is completed in this agent."""
        if self.global_config.get("freeze_memory", False):
            return

        prof = self._timing_profile_on()
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        super().add_memory(mas_message)
        super_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._abstract_experience_from_message(mas_message)
        abstract_s = time.perf_counter() - t0
        viz_s = 0.0
        # Periodic visualization for LG/GG snapshots.
        # - Always on success (existing behavior).
        # - Optionally on failure (for debugging / dynamics comparison).
        if mas_message.label is True:
            self._success_case_counter += 1
            if self._viz_every_n_success <= 1 or (
                self._success_case_counter % self._viz_every_n_success == 0
            ):
                t0 = time.perf_counter()
                self._visualize_local_graph_once(mas_message)
                self._visualize_merged_lg_once(mas_message)
                viz_s = time.perf_counter() - t0
        elif mas_message.label is False and bool(self.global_config.get("viz_on_failure", False)):
            t0 = time.perf_counter()
            self._visualize_local_graph_once(mas_message)
            self._visualize_merged_lg_once(mas_message)
            viz_s = time.perf_counter() - t0
        self.last_saved_message = mas_message
        if prof:
            self._timing_profile_emit(
                {
                    "kind": "selectivemem_add_memory",
                    "episode_index_1based": self.global_config.get("episode_index_1based"),
                    "label": mas_message.label,
                    "super_add_memory_s": super_s,
                    "abstract_experience_s": abstract_s,
                    "visualize_lg_gg_s": viz_s,
                    "total_s": time.perf_counter() - t_all,
                }
            )

    def add_memory_from_peer(
        self,
        mas_message: MASMessage,
        source_id: str | None = None,
    ) -> None:
        """
        Add memory contributed by another agent.

        For now we just record source and feed into the same abstraction pipeline.
        """
        if source_id:
            mas_message.add_extra_field("source_id", source_id)
            mas_message.add_extra_field("source_scope", "global")
        if self.global_config.get("freeze_memory", False):
            return
        prof = self._timing_profile_on()
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        super().add_memory(mas_message)
        super_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self._abstract_experience_from_message(mas_message)
        abstract_s = time.perf_counter() - t0
        viz_s = 0.0
        # Keep visualization/recording behavior consistent with add_memory().
        if mas_message.label is True:
            self._success_case_counter += 1
            if self._viz_every_n_success <= 1 or (
                self._success_case_counter % self._viz_every_n_success == 0
            ):
                t0 = time.perf_counter()
                self._visualize_local_graph_once(mas_message)
                self._visualize_merged_lg_once(mas_message)
                viz_s = time.perf_counter() - t0
        elif mas_message.label is False and bool(self.global_config.get("viz_on_failure", False)):
            t0 = time.perf_counter()
            self._visualize_local_graph_once(mas_message)
            self._visualize_merged_lg_once(mas_message)
            viz_s = time.perf_counter() - t0
        self.last_saved_message = mas_message
        if prof:
            self._timing_profile_emit(
                {
                    "kind": "selectivemem_add_memory_from_peer",
                    "episode_index_1based": self.global_config.get("episode_index_1based"),
                    "label": mas_message.label,
                    "super_add_memory_s": super_s,
                    "abstract_experience_s": abstract_s,
                    "visualize_lg_gg_s": viz_s,
                    "total_s": time.perf_counter() - t_all,
                }
            )

    def set_global_retriever(self, global_retriever: "SelectiveMemMASMemory") -> None:
        """
        Attach the global SelectiveMem instance. When set, retrieve_memory() delegates to it
        (same pattern as GMemory) so GG retrieval uses merged graph + fingerprints under global_dir.
        """
        self._global_retriever = global_retriever

    def _timing_profile_on(self) -> bool:
        from mas.timing_profile import enabled

        return enabled(global_config=self.global_config)

    def _timing_profile_emit(self, row: dict[str, Any]) -> None:
        from mas.timing_profile import append

        logd = self.global_config.get("viz_output_dir") or self.global_config.get("working_dir") or self.persist_dir
        append(str(logd), "timing_selectivemem.jsonl", row)

    def _abstract_experience_from_message(self, mas_message: MASMessage) -> None:
        """
        Local→Global 抽象：
        1) 对每个状态的 observation 文本调用 LLM 自动抽取实体与关系，更新本地图 GL；
        2) 利用状态中的 (action, room/location) 信息，把经验写入全局 GG（可选保留）；
        3) 写入 query 绑定 insights 库、轨迹指纹、合并语义 LG。
        """
        # Build local/merged graphs for both success and failure so we can snapshot dynamics on failures.
        # Only successful episodes contribute to global fingerprints / trajectory archives.
        if mas_message.label is not True and mas_message.label is not False:
            return
        is_success = mas_message.label is True

        state_chain: StateChain | None = mas_message.chain_of_states
        if not state_chain:
            return

        task_name = mas_message.get_extra_field("task_name") or self.global_config.get(
            "task_name", ""
        )
        is_mtmind2web = str(task_name).startswith("mtmind2web")
        source_id = mas_message.get_extra_field("source_id")
        traj_id = mas_message.get_extra_field("trajectory_id")
        if not traj_id:
            traj_id = hashlib.sha256(
                f"{mas_message.task_main}|{mas_message.task_description}|{time.time()}".encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
            mas_message.add_extra_field("trajectory_id", traj_id)

        query_text = self._build_query_text_for_archive(mas_message)
        # Note: the current GG mechanism stores trajectories during successful episodes,
        # and defers "common insights" summarization until retrieve_memory() time.
        # (We keep legacy insight utilities in code for backward compatibility.)

        task_categories: set[str] = set()
        trajectory_edges: list[tuple[str, str, str]] = []
        step_index = 0

        for state in state_chain:
            graph = getattr(state, "graph", {}) or {}
            action = graph.get("action") or ""
            obs = graph.get("observation") or ""
            room = graph.get("room") or graph.get("location") or ""

            # MT-Mind2Web is evaluated as one-shot action sequence prediction.
            # Observations may be too sparse for graph extraction, so we synthesize
            # a readable scene/action text from query + candidate actions + chosen indices.
            obs_for_extract = str(obs or "")
            if is_mtmind2web:
                obs_for_extract = self._build_mtmind2web_extraction_observation(
                    mas_message=mas_message,
                    action=str(action or ""),
                    raw_observation=obs_for_extract,
                    step_index=step_index,
                )

            # 1) 使用 LLM 从 observation 自动构建 / 更新本地图 GL
            if obs_for_extract:
                parsed = self.update_local_from_observation_text(
                    obs_for_extract,
                    room_hint=room,
                    trajectory_id=traj_id,
                    step_index=step_index,
                )
                for inst in parsed.get("instances", []):
                    cat = str(inst.get("category") or "").strip().lower()
                    if cat:
                        task_categories.add(cat)
                self._project_parsed_to_merged_lg(
                    parsed, trajectory_id=traj_id, step_index=step_index, room_hint=room or ""
                )
                trajectory_edges.extend(self._canonical_edges_from_parsed(parsed))
                step_index += 1

        if is_success and trajectory_edges:
            self._append_fingerprint_record(
                trajectory_id=traj_id,
                query_text=query_text,
                edge_sequence=trajectory_edges,
                trajectory_excerpt=self._trajectory_excerpt(mas_message),
            )

    def _build_mtmind2web_extraction_observation(
        self,
        *,
        mas_message: MASMessage,
        action: str,
        raw_observation: str,
        step_index: int,
    ) -> str:
        """
        Build a natural-language extraction text for MT-Mind2Web.
        Includes query and a readable action sequence decoded from Finish[[...]].
        """
        raw_obs = (raw_observation or "").strip()
        task_main = (mas_message.task_main or "").strip()
        task_desc = (mas_message.task_description or "").strip()
        candidates = self._extract_candidate_actions(task_main + "\n" + task_desc)
        rendered_action = self._render_mtmind2web_action_nl(action, candidates)

        parts: list[str] = []
        if task_desc:
            parts.append(f"Task query: {task_desc}")
        elif task_main:
            parts.append(f"Task query: {task_main}")
        if rendered_action:
            parts.append(f"Predicted action sequence: {rendered_action}")
        if raw_obs:
            parts.append(f"Environment feedback: {raw_obs}")
        parts.append(f"Step index: {step_index}")
        return "\n".join(p for p in parts if p.strip())

    def _extract_candidate_actions(self, text: str) -> dict[int, str]:
        cand: dict[int, str] = {}
        for line in (text or "").splitlines():
            m = re.match(r"^\s*(\d+)\.\s*(.+?)\s*$", line)
            if not m:
                continue
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            cand[idx] = m.group(2).strip()
        return cand

    def _render_mtmind2web_action_nl(self, action: str, candidates: dict[int, str]) -> str:
        act = (action or "").strip()
        m = re.search(r"Finish\[\[(.*?)\]\]", act)
        if not m:
            return act
        idx_blob = m.group(1).strip()
        if not idx_blob:
            return "no action selected"
        picks: list[int] = []
        for tok in idx_blob.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                picks.append(int(tok))
            except Exception:
                continue
        if not picks:
            return act
        steps: list[str] = []
        for i in picks:
            if i in candidates:
                steps.append(f"[{i}] {candidates[i]}")
            else:
                steps.append(f"[{i}] <unknown_candidate>")
        return " -> ".join(steps)

    # ---- read side ----

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        **kwargs: Any,
    ) -> tuple[list, list, list]:
        """
        GG (new global graph mechanism):
        1) Use query embedding similarity to find the top-5 most similar *successful queries*.
        2) For each matched query, pick a representative successful trajectory fingerprint.
        3) Run a single LLM summarizer (temperature=0) to produce common experience across these 5 trajectories.
        4) Save both the summarizer output and the selected trajectories for audit/debug.

        When ``set_global_retriever`` has been called (e.g. pass2 with merged global memory), all GG
        retrieval runs on that global instance so prompts use global merged_lg + global fingerprints.
        """
        if self._global_retriever is not None and self._global_retriever is not self:
            return self._global_retriever.retrieve_memory(
                query_task=query_task,
                successful_topk=successful_topk,
                failed_topk=failed_topk,
                **kwargs,
            )

        prof = self._timing_profile_on()
        t_all = time.perf_counter()
        acc: dict[str, Any] = {
            "kind": "selectivemem_retrieve_memory",
            "episode_index_1based": self.global_config.get("episode_index_1based"),
            "query_preview": (query_task or "")[:200],
        }

        t0 = time.perf_counter()
        true_msgs, false_msgs, _ = super().retrieve_memory(
            query_task=query_task,
            successful_topk=successful_topk,
            failed_topk=failed_topk,
            **kwargs,
        )
        acc["super_retrieve_s"] = time.perf_counter() - t0

        acc["top_summarize_s"] = 0.0
        # Summarize the most similar retrieved successful trajectory into key steps (temp=0),
        # and inject the summary instead of the raw trajectory text.
        if true_msgs and bool(self.global_config.get("summarize_top_success_trajectory", True)):
            top: MASMessage = true_msgs[0]
            raw_task = str(getattr(top, "task_description", "") or "")
            raw_traj = str(getattr(top, "task_trajectory", "") or "")
            if raw_traj.strip():
                t_sum = time.perf_counter()
                user_prompt = GMemoryPrompts.extract_true_traj_user_prompt.format(
                    task=raw_task,
                    trajectory=raw_task + "\n" + raw_traj,
                )
                key_steps = self.llm_model(
                    messages=[
                        Message("system", GMemoryPrompts.extract_true_traj_system_prompt),
                        Message("user", user_prompt),
                    ],
                    temperature=0,
                    max_tokens=512,
                ).strip()
                acc["top_summarize_s"] = time.perf_counter() - t_sum
                if not key_steps:
                    raise RuntimeError(
                        "summarize_top_success_trajectory enabled but summarizer returned empty output"
                    )
                top.add_extra_field("raw_trajectory_full", raw_traj)
                top.add_extra_field("key_steps", key_steps)
                top.task_trajectory = ""

        insights: list[str] = []
        if not self._fingerprint_records:
            acc["branch"] = "no_fingerprints"
            acc["total_s"] = time.perf_counter() - t_all
            if prof:
                self._timing_profile_emit(acc)
            return true_msgs, false_msgs, insights

        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        t0 = time.perf_counter()
        query_entities = self._extract_query_entities(query_task)
        acc["extract_query_entities_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        merged_best_path_nodes: list[str] = []
        current_edge_seq: list[tuple[str, str, str]] = []
        matched_nodes = self._match_entity_nodes_merged(query_entities)
        path_list = self._collect_merged_shortest_paths(
            matched_nodes, max_paths=3, max_hops=4
        )
        if path_list:
            best_path = max(path_list, key=lambda p: len(p))
            merged_best_path_nodes = [str(n) for n in best_path]
            current_edge_seq = self._edge_seq_from_node_path(best_path)
        acc["merged_path_infer_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        topn_queries = int(self.global_config.get("gg_topn_queries", 5))
        current_q_emb = self.embedding_func.embed_query(query_task)

        query_emb_cache: dict[str, list[float]] = {}
        best_score_by_query: dict[str, float] = {}

        for rec in self._fingerprint_records:
            qt = str(rec.get("query_text") or "").strip()
            if not qt or not current_q_emb:
                continue
            if qt not in query_emb_cache:
                query_emb_cache[qt] = self.embedding_func.embed_query(qt)
            qemb = query_emb_cache.get(qt) or []
            if not qemb:
                continue
            s = self._cosine_sim(current_q_emb, qemb)
            if s <= 0:
                continue
            if qt not in best_score_by_query or s > best_score_by_query[qt]:
                best_score_by_query[qt] = float(s)

        top_queries_sorted = sorted(
            best_score_by_query.items(), key=lambda x: x[1], reverse=True
        )[: max(1, topn_queries)]
        acc["embedding_rank_s"] = time.perf_counter() - t0
        acc["n_distinct_query_texts_embedded"] = len(query_emb_cache)

        t0 = time.perf_counter()
        selected_trajectories: list[dict[str, Any]] = []

        for qt, qscore in top_queries_sorted:
            candidates = [
                r for r in self._fingerprint_records if str(r.get("query_text") or "").strip() == qt
            ]
            best_rec = None
            best_overlap = -1.0
            for rec in candidates:
                edge_seq_raw = rec.get("edge_sequence") or []
                edge_seq: list[tuple[str, str, str]] = []
                for row in edge_seq_raw:
                    if isinstance(row, (list, tuple)) and len(row) == 3:
                        edge_seq.append((str(row[0]), str(row[1]), str(row[2])))
                overlap = (
                    self._edge_sequence_jaccard(current_edge_seq, edge_seq)
                    if current_edge_seq and edge_seq
                    else 0.0
                )
                if overlap > best_overlap:
                    best_overlap = float(overlap)
                    best_rec = rec
            if best_rec is None:
                continue
            selected_trajectories.append(
                {
                    "query_text": qt,
                    "query_score": qscore,
                    "trajectory_id": best_rec.get("trajectory_id"),
                    "trajectory_excerpt": best_rec.get("trajectory_excerpt"),
                    "overlap": best_overlap,
                    "edge_sequence_len": len(best_rec.get("edge_sequence") or []),
                }
            )
        acc["fingerprint_select_s"] = time.perf_counter() - t0

        if len(selected_trajectories) < 2:
            acc["branch"] = "insufficient_selected_trajectories"
            acc["n_selected"] = len(selected_trajectories)
            acc["total_s"] = time.perf_counter() - t_all
            if prof:
                self._timing_profile_emit(acc)
            return true_msgs, false_msgs, insights

        def _edge_seq_to_text(es: list[tuple[str, str, str]], max_edges: int = 16) -> str:
            if not es:
                return ""
            parts: list[str] = []
            for i, (u, rel, v) in enumerate(es):
                if i >= max_edges:
                    break
                parts.append(f"{u}-[{rel}]->{v}")
            return " ; ".join(parts)

        path_text = _edge_seq_to_text(current_edge_seq, max_edges=24)

        cases_block_lines: list[str] = []
        for idx, case in enumerate(selected_trajectories, start=1):
            traj_id = case.get("trajectory_id") or ""
            qt = (case.get("query_text") or "")[:200]
            overlap = float(case.get("overlap") or 0.0)
            ex = (case.get("trajectory_excerpt") or "")[:500]
            traj_rec = next(
                (
                    r
                    for r in self._fingerprint_records
                    if r.get("trajectory_id") == traj_id
                    and str(r.get("query_text") or "").strip() == str(case.get("query_text") or "").strip()
                ),
                None,
            )
            edge_snippet = ""
            if traj_rec:
                es_raw = traj_rec.get("edge_sequence") or []
                es_snip: list[tuple[str, str, str]] = []
                for row in es_raw:
                    if isinstance(row, (list, tuple)) and len(row) == 3:
                        es_snip.append((str(row[0]), str(row[1]), str(row[2])))
                edge_snippet = _edge_seq_to_text(es_snip, max_edges=10)
            cases_block_lines.append(
                f"Case {idx}: query={qt!r}, traj_id={traj_id!r}, overlap={overlap:.3f}\n"
                f"Edges: {edge_snippet}\n"
                f"Excerpt: {ex}"
            )
        cases_block = "\n\n".join(cases_block_lines)

        user_prompt = _MERGED_PATH_SUMMARY_USER.format(
            current_query=query_task,
            path_text=path_text or "",
            cases_block=cases_block,
        )

        t_llm = time.perf_counter()
        llm_output = (
            self.llm_model(
                [
                    Message("system", _MERGED_PATH_SUMMARY_SYSTEM),
                    Message("user", user_prompt),
                ],
                temperature=0,
                max_tokens=512,
            )
            .strip()
        )
        acc["gg_common_llm_s"] = time.perf_counter() - t_llm

        if not llm_output:
            acc["branch"] = "empty_gg_llm_out"
            acc["total_s"] = time.perf_counter() - t_all
            if prof:
                self._timing_profile_emit(acc)
            return true_msgs, false_msgs, insights

        common_insight = llm_output
        insights.append(f"[GGCommon] {common_insight}")

        t_io = time.perf_counter()
        summary_rec = {
            "timestamp": ts,
            "query_task": query_task,
            "query_entities": query_entities,
            "current_path_nodes": merged_best_path_nodes,
            "current_edge_seq_preview": path_text,
            "top_queries": [{"query_text": qt, "score": sc} for qt, sc in top_queries_sorted],
            "selected_trajectories": selected_trajectories,
            "llm_output": common_insight,
        }
        with open(self._gg_common_summaries_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_rec, ensure_ascii=False) + "\n")
        self._mirror_artifacts_to_viz_output_dir(
            [os.path.basename(self._gg_common_summaries_path)]
        )

        trace_rec = {
            **summary_rec,
            "retrieval_trace_version": 2,
        }
        with open(self._retrieval_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_rec, ensure_ascii=False) + "\n")
        self._mirror_artifacts_to_viz_output_dir(
            [os.path.basename(self._retrieval_trace_path)]
        )
        acc["audit_write_mirror_s"] = time.perf_counter() - t_io

        acc["branch"] = "full_gg_insight"
        acc["total_s"] = time.perf_counter() - t_all
        if prof:
            self._timing_profile_emit(acc)

        return true_msgs, false_msgs, insights

    # ---- local graph utilities ----

    def update_local_from_observation_text(
        self,
        observation: str,
        room_hint: str | None = None,
        trajectory_id: str | None = None,
        step_index: int | None = None,
    ) -> dict[str, Any]:
        prof = self._timing_profile_on()
        t_all = time.perf_counter()

        prompt = _LOCAL_GRAPH_USER_PROMPT.format(
            observation=observation,
            room_hint=room_hint or "",
        )
        messages = [
            Message("system", _LOCAL_GRAPH_SYSTEM_PROMPT),
            Message("user", prompt),
        ]
        t0 = time.perf_counter()
        raw = self.llm_model(messages, temperature=0, max_tokens=512)
        llm_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        data = self._parse_local_graph_response(raw)
        data = self._filter_scene_only_entities(data)
        parse_filter_s = time.perf_counter() - t0

        visit_entry: dict[str, Any] | None = None
        if trajectory_id is not None and step_index is not None:
            visit_entry = {
                "traj": trajectory_id,
                "step": int(step_index),
                "room": room_hint or "",
            }

        t0 = time.perf_counter()
        # 写入节点
        for inst in data.get("instances", []):
            instance_id = inst.get("instance_id")
            if not instance_id:
                continue
            self.local_graph.upsert_instance(
                instance_id=instance_id,
                label=inst.get("label"),
                category=inst.get("category"),
                state=inst.get("state"),
                room=inst.get("room") or room_hint,
                visit_trace_entry=visit_entry,
            )

        # 写入关系
        for rel in data.get("relations", []):
            subj = rel.get("subject")
            obj = rel.get("object")
            relation = rel.get("relation")
            if not subj or not obj or not relation:
                continue
            self.local_graph.add_relation(subj, obj, relation)
        graph_upsert_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Real-time persistence of LG snapshots (+ optional mirror to viz_output_dir).
        self._persist_local_graph_snapshots_best_effort()
        persist_s = time.perf_counter() - t0
        if prof:
            self._timing_profile_emit(
                {
                    "kind": "selectivemem_update_local_observation",
                    "episode_index_1based": self.global_config.get("episode_index_1based"),
                    "trajectory_id": trajectory_id,
                    "step_index": step_index,
                    "llm_s": llm_s,
                    "parse_filter_s": parse_filter_s,
                    "graph_upsert_s": graph_upsert_s,
                    "persist_s": persist_s,
                    "obs_len": len(observation or ""),
                    "total_s": time.perf_counter() - t_all,
                }
            )
        return data

    def _filter_scene_only_entities(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Enforce scene-only graph: remove any 'agent/player/you' related instances and incident edges.
        This is a safety net in case the LLM violates the prompt.
        """
        instances = data.get("instances") or []
        relations = data.get("relations") or []
        if not isinstance(instances, list):
            instances = []
        if not isinstance(relations, list):
            relations = []

        banned_ids: set[str] = set()
        filtered_instances: list[dict[str, Any]] = []
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            iid = str(inst.get("instance_id") or "").strip()
            label = str(inst.get("label") or "").strip().lower()
            cat = str(inst.get("category") or "").strip().lower()
            # Any explicit agent-like mention is banned.
            if cat in {"agent", "player"}:
                if iid:
                    banned_ids.add(iid)
                continue
            if "agent" in label or "player" in label or label == "you":
                if iid:
                    banned_ids.add(iid)
                continue
            if iid.lower().startswith("agent"):
                banned_ids.add(iid)
                continue
            filtered_instances.append(inst)

        filtered_relations: list[dict[str, Any]] = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            subj = str(rel.get("subject") or "").strip()
            obj = str(rel.get("object") or "").strip()
            if subj in banned_ids or obj in banned_ids:
                continue
            # Also drop edges that explicitly mention agent in relation label (rare).
            r = str(rel.get("relation") or "").strip().lower()
            if "agent" in r or "player" in r:
                continue
            filtered_relations.append(rel)

        return {"instances": filtered_instances, "relations": filtered_relations}

    def _persist_local_graph_snapshots_best_effort(self) -> None:
        """
        Persist LG and derived graphs on each update (real-time).
        Best-effort: never crash training due to IO.
        """
        try:
            local_path = os.path.join(self.persist_dir, "local_instance_graph.json")
            write_json(self.local_graph.snapshot(), local_path)
            merged_path = os.path.join(self.persist_dir, "local_category_graph.json")
            write_json(self._category_graph_snapshot(self._build_category_graph()), merged_path)
            self._mirror_artifacts_to_viz_output_dir(
                ["local_instance_graph.json", "local_category_graph.json"]
            )
        except Exception:
            pass

    def _viz_output_dir(self) -> str | None:
        raw = self.global_config.get("viz_output_dir")
        if raw is None or str(raw).strip() == "":
            return None
        return os.path.abspath(str(raw).strip())

    def _mirror_artifacts_to_viz_output_dir(self, filenames: list[str]) -> None:
        vdir = self._viz_output_dir()
        if not vdir:
            return
        import shutil

        os.makedirs(vdir, exist_ok=True)
        for name in filenames:
            src = os.path.join(self.persist_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(vdir, name))

    def _graph_viz_meta(self, mas_message: MASMessage, graph_kind: str) -> dict[str, Any]:
        """Metadata for archived LG/GG JSON (collab A/B, episode, trajectory, etc.)."""
        gc = self.global_config
        ex = mas_message.extra_fields or {}
        return {
            "graph_kind": graph_kind,
            "collab_side": gc.get("collab_side"),
            "collab_scene": gc.get("collab_scene"),
            "train_phase": gc.get("train_phase"),
            "episode_index_1based": gc.get("episode_index_1based"),
            "episode_index_0based": gc.get("episode_index_0based"),
            "task_name": ex.get("task_name") or gc.get("task_name") or "",
            "trajectory_id": ex.get("trajectory_id"),
            "source_id": ex.get("source_id"),
            "success_case_export_index": self._success_case_counter,
            "exported_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "memory_persist_dir": self.persist_dir,
        }

    def _viz_json_filename(self, graph_kind: str, ts: str) -> str:
        gc = self.global_config
        side = str(gc.get("collab_side") or "X")
        scene = str(gc.get("collab_scene") or "unknown")
        ep_raw = gc.get("episode_index_1based")
        ep_s = f"ep{int(ep_raw):04d}" if ep_raw is not None else "epNA"
        safe_side = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in side)
        safe_scene = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in scene)
        prefix = "LG" if graph_kind == "LG" else "GG"
        return f"{prefix}_viz_{safe_side}_{safe_scene}_{ep_s}_{ts}.json"

    def _write_graph_viz_bundle(
        self,
        mas_message: MASMessage,
        graph_kind: str,
        graph_payload: dict[str, Any],
    ) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_dir = self._viz_output_dir() or self.persist_dir
        os.makedirs(base_dir, exist_ok=True)
        fname = self._viz_json_filename(graph_kind, ts)
        out_path = os.path.join(base_dir, fname)
        bundle = {
            "format": "selectivemem_graph_viz_v1",
            "meta": self._graph_viz_meta(mas_message, graph_kind),
            "graph": graph_payload,
        }
        write_json(bundle, out_path)

    def _visualize_local_graph_once(self, mas_message: MASMessage) -> None:
        """
        Export local LG as JSON once per successful trajectory (see ``viz_every_n_success``).
        Includes meta (collab A/B, episode, trajectory) for later plotting.
        """
        self._write_graph_viz_bundle(
            mas_message,
            "LG",
            self.local_graph.snapshot(),
        )

    def _visualize_merged_lg_once(self, mas_message: MASMessage) -> None:
        """
        Export merged canonical graph (GG) as node-link JSON + meta once per successful trajectory
        (subject to ``viz_every_n_success``).
        """
        # Use NetworkX default node-link schema. Do not pass compatibility args here:
        # some environments ship older NetworkX versions where `link=` is unsupported.
        payload = json_graph.node_link_data(self.merged_lg)
        self._write_graph_viz_bundle(mas_message, "GG", payload)

    def _parse_local_graph_response(self, response: str) -> dict[str, Any]:
        """
        解析 LLM 返回的 JSON，对不规范情况做鲁棒处理。
        """
        if not response:
            return {"instances": [], "relations": []}
        # 尝试截取第一个 JSON 对象
        match = re.search(r"\{[\s\S]*\}", response)
        if not match:
            return {"instances": [], "relations": []}
        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict):
                instances = payload.get("instances") or []
                relations = payload.get("relations") or []
                if not isinstance(instances, list):
                    instances = []
                if not isinstance(relations, list):
                    relations = []
                return {
                    "instances": instances,
                    "relations": relations,
                }
        except json.JSONDecodeError:
            return {"instances": [], "relations": []}
        return {"instances": [], "relations": []}

    # ---- query archive / merged LG / fingerprints ----

    def _build_query_text_for_archive(self, mas_message: MASMessage) -> str:
        main = (mas_message.task_main or "").strip()
        desc = (mas_message.task_description or "").strip()
        q = f"{main}\n{desc}".strip()
        max_len = int(self.global_config.get("query_archive_max_chars", 4000))
        return q[:max_len] if max_len > 0 else q

    def _load_query_insight_archive(self) -> list[dict[str, Any]]:
        payload = load_json(self._query_insight_archive_path)
        if isinstance(payload, list):
            return payload
        return []

    def _append_query_insight_record(
        self,
        query_text: str,
        insights: list[str],
        trajectory_id: str,
        source_id: str | None,
    ) -> None:
        if not query_text or not insights:
            return
        try:
            emb = self.embedding_func.embed_text(query_text)
        except Exception:
            emb = []
        rec = {
            "query_text": query_text,
            "embedding": emb,
            "insights": [str(x).strip() for x in insights if str(x).strip()],
            "meta": {
                "trajectory_id": trajectory_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source_id": source_id,
            },
        }
        self.query_insight_archive.append(rec)
        try:
            write_json(self.query_insight_archive, self._query_insight_archive_path)
            self._mirror_artifacts_to_viz_output_dir(["query_insight_archive.json"])
        except Exception:
            pass

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb)

    def _retrieve_query_similar_insights(self, query_task: str, topk: int = 3) -> list[str]:
        if not query_task.strip() or not self.query_insight_archive:
            return []
        try:
            qv = self.embedding_func.embed_query(query_task)
        except Exception:
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in self.query_insight_archive:
            emb = rec.get("embedding")
            if not isinstance(emb, list) or not emb:
                continue
            s = self._cosine_sim(qv, emb)
            if s > 0:
                scored.append((s, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[str] = []
        for s, rec in scored[: max(1, topk)]:
            ins = rec.get("insights") or []
            qt = (rec.get("query_text") or "")[:200]
            blob = "; ".join(str(x) for x in ins[:6])
            out.append(f"[QuerySim score={s:.3f} ref={qt!r}] {blob}")
        return out

    def _canonical_from_inst_dict(self, inst: dict[str, Any]) -> str:
        cat = str(inst.get("category") or "").strip().lower()
        lab = str(inst.get("label") or "").strip().lower()
        if cat and lab:
            return f"{cat}:{lab}"
        return cat or lab or ""

    def _canonical_for_instance_id(self, parsed: dict[str, Any], iid: str) -> str:
        for inst in parsed.get("instances") or []:
            if not isinstance(inst, dict):
                continue
            if str(inst.get("instance_id") or "").strip() == str(iid).strip():
                return self._canonical_from_inst_dict(inst)
        return ""

    def _canonical_edges_from_parsed(
        self, parsed: dict[str, Any]
    ) -> list[tuple[str, str, str]]:
        out: list[tuple[str, str, str]] = []
        for rel in parsed.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            s = str(rel.get("subject") or "").strip()
            o = str(rel.get("object") or "").strip()
            r = str(rel.get("relation") or "").strip()
            if not s or not o or not r:
                continue
            cs = self._canonical_for_instance_id(parsed, s)
            co = self._canonical_for_instance_id(parsed, o)
            if cs and co:
                out.append((cs, r, co))
        return out

    def _append_merged_node_provenance(
        self, canon: str, entry: dict[str, Any]
    ) -> None:
        data = self.merged_lg.nodes[canon]
        prov: list[dict[str, Any]] = data.setdefault("provenance", [])
        for x in prov:
            if not isinstance(x, dict):
                continue
            if (
                x.get("traj_id") == entry.get("traj_id")
                and x.get("step") == entry.get("step")
                and x.get("orig_instance_id") == entry.get("orig_instance_id")
            ):
                return
        prov.append(entry)

    def _append_merged_edge_provenance(
        self, u: str, v: str, key: int, entry: dict[str, Any]
    ) -> None:
        d = self.merged_lg[u][v][key]
        prov: list[dict[str, Any]] = d.setdefault("provenance", [])
        for x in prov:
            if not isinstance(x, dict):
                continue
            if (
                x.get("traj_id") == entry.get("traj_id")
                and x.get("step") == entry.get("step")
                and x.get("orig_subject") == entry.get("orig_subject")
                and x.get("orig_object") == entry.get("orig_object")
            ):
                return
        prov.append(entry)

    def _merge_or_add_merged_edge(
        self, u: str, v: str, relation: str, eprov: dict[str, Any]
    ) -> None:
        if u not in self.merged_lg:
            self.merged_lg.add_node(u, provenance=[])
        if v not in self.merged_lg:
            self.merged_lg.add_node(v, provenance=[])
        if self.merged_lg.has_edge(u, v):
            for k in self.merged_lg[u][v]:
                d = self.merged_lg[u][v][k]
                if str(d.get("relation") or "") == relation:
                    self._append_merged_edge_provenance(u, v, k, eprov)
                    return
        k = 0
        while self.merged_lg.has_edge(u, v, key=k):
            k += 1
        self.merged_lg.add_edge(u, v, key=k, relation=relation, provenance=[eprov])

    def _project_parsed_to_merged_lg(
        self,
        parsed: dict[str, Any],
        trajectory_id: str,
        step_index: int,
        room_hint: str,
    ) -> None:
        for inst in parsed.get("instances") or []:
            if not isinstance(inst, dict):
                continue
            iid = str(inst.get("instance_id") or "").strip()
            canon = self._canonical_from_inst_dict(inst)
            if not canon:
                continue
            if canon not in self.merged_lg:
                self.merged_lg.add_node(
                    canon,
                    label=inst.get("label"),
                    category=inst.get("category"),
                    provenance=[],
                )
            self._append_merged_node_provenance(
                canon,
                {
                    "traj_id": trajectory_id,
                    "orig_instance_id": iid,
                    "step": step_index,
                    "room": room_hint,
                },
            )
        for rel in parsed.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            s = str(rel.get("subject") or "").strip()
            o = str(rel.get("object") or "").strip()
            r = str(rel.get("relation") or "").strip()
            if not s or not o or not r:
                continue
            cs = self._canonical_for_instance_id(parsed, s)
            co = self._canonical_for_instance_id(parsed, o)
            if not cs or not co:
                continue
            self._merge_or_add_merged_edge(
                cs,
                co,
                r,
                {
                    "traj_id": trajectory_id,
                    "step": step_index,
                    "orig_subject": s,
                    "orig_object": o,
                    "room": room_hint,
                },
            )
        self._persist_merged_lg_best_effort()

    def _persist_merged_lg_best_effort(self) -> None:
        # Strict mode: use networkx default node-link format for this environment.
        # If networkx API/format changes, we want a hard error (no silent compatibility shims).
        data = json_graph.node_link_data(self.merged_lg)
        write_json(data, self._merged_lg_path)
        self._mirror_artifacts_to_viz_output_dir(["merged_instance_graph.json"])

    def _load_merged_lg_from_disk(self) -> None:
        raw = load_json(self._merged_lg_path)
        if raw is None:
            return
        # Strict mode: expect networkx default node-link format (incl. `edges` key).
        g = json_graph.node_link_graph(raw, multigraph=True, directed=True)
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        self.merged_lg = g

    def _load_fingerprint_records(self) -> list[dict[str, Any]]:
        if not os.path.isfile(self._fingerprint_path):
            return []
        out: list[dict[str, Any]] = []
        try:
            with open(self._fingerprint_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return out

    def _append_fingerprint_record(
        self,
        trajectory_id: str,
        query_text: str,
        edge_sequence: list[tuple[str, str, str]],
        trajectory_excerpt: str,
    ) -> None:
        rec = {
            "trajectory_id": trajectory_id,
            "query_text": query_text,
            "edge_sequence": [list(t) for t in edge_sequence],
            "trajectory_excerpt": trajectory_excerpt[:2000],
        }
        self._fingerprint_records.append(rec)
        try:
            with open(self._fingerprint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _trajectory_excerpt(self, mas_message: MASMessage) -> str:
        t = mas_message.task_trajectory or ""
        return str(t)[:2500]

    @staticmethod
    def _edge_sequence_jaccard(
        a: list[tuple[str, str, str]], b: list[tuple[str, str, str]]
    ) -> float:
        sa = {f"{x[0]}||{x[1]}||{x[2]}" for x in a}
        sb = {f"{x[0]}||{x[1]}||{x[2]}" for x in b}
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def _match_entity_nodes_merged(self, query_entities: list[str]) -> list[str]:
        if not self.merged_lg.number_of_nodes():
            return []
        nodes = list(self.merged_lg.nodes())
        matched: list[str] = []
        for ent in query_entities:
            e = ent.strip().lower()
            if not e:
                continue
            if e in self.merged_lg:
                matched.append(e)
                continue
            for n in nodes:
                ns = str(n).lower()
                nd = self.merged_lg.nodes[n] or {}
                lab = str(nd.get("label") or "").lower()
                cat = str(nd.get("category") or "").lower()
                if e == ns or e in ns or ns in e or (lab and e in lab) or (cat and e in cat):
                    matched.append(str(n))
                    break
        return list(dict.fromkeys(matched))

    def _edge_seq_from_node_path(self, path: list[str]) -> list[tuple[str, str, str]]:
        seq: list[tuple[str, str, str]] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not self.merged_lg.has_edge(u, v):
                continue
            k = next(iter(self.merged_lg[u][v]))
            rel = str(self.merged_lg[u][v][k].get("relation") or "")
            seq.append((u, rel, v))
        return seq

    def _collect_merged_shortest_paths(
        self,
        matched_nodes: list[str],
        max_paths: int = 3,
        max_hops: int = 4,
    ) -> list[list[str]]:
        if len(matched_nodes) < 2:
            return []
        undirected = self.merged_lg.to_undirected()
        paths: list[list[str]] = []
        seen: set[tuple[str, str]] = set()
        for i in range(len(matched_nodes)):
            for j in range(i + 1, len(matched_nodes)):
                s, t = matched_nodes[i], matched_nodes[j]
                pair = tuple(sorted([s, t]))
                if pair in seen:
                    continue
                seen.add(pair)
                try:
                    path = nx.shortest_path(undirected, source=s, target=t)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if len(path) - 1 > max_hops:
                    continue
                paths.append(path)
                if len(paths) >= max_paths:
                    return paths
        return paths

    def _retrieve_merged_path_experience(
        self,
        query_task: str,
        query_entities: list[str],
        topk_paths: int = 3,
        topk_traj: int = 3,
    ) -> str:
        if self.merged_lg.number_of_nodes() < 2:
            return ""
        matched = self._match_entity_nodes_merged(query_entities)
        if len(matched) < 2:
            return ""
        path_list = self._collect_merged_shortest_paths(
            matched, max_paths=topk_paths, max_hops=4
        )
        if not path_list:
            return ""
        best_path = max(path_list, key=lambda p: len(p))
        edge_seq = self._edge_seq_from_node_path(best_path)
        if not edge_seq:
            return ""
        path_chunks: list[str] = []
        for i, nid in enumerate(best_path):
            path_chunks.append(nid)
            if i < len(best_path) - 1:
                u, v = best_path[i], best_path[i + 1]
                rels: list[str] = []
                if self.merged_lg.has_edge(u, v):
                    for _k, d in self.merged_lg[u][v].items():
                        rels.append(str((d or {}).get("relation") or ""))
                path_chunks.append(
                    f"-[{ '|'.join(sorted(set(rels))[:2]) or 'linked' }]-"
                )
        path_text = " ".join(path_chunks)

        scored: list[tuple[float, dict[str, Any]]] = []
        for rec in self._fingerprint_records:
            raw_es = rec.get("edge_sequence") or []
            tup_es: list[tuple[str, str, str]] = []
            for row in raw_es:
                if isinstance(row, (list, tuple)) and len(row) == 3:
                    tup_es.append((str(row[0]), str(row[1]), str(row[2])))
            sc = self._edge_sequence_jaccard(edge_seq, tup_es)
            if sc > 0:
                scored.append((sc, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for _, r in scored[: max(1, topk_traj)]]
        if not top:
            return ""

        case_lines: list[str] = []
        for idx, rec in enumerate(top, start=1):
            q = (rec.get("query_text") or "")[:400]
            ex = (rec.get("trajectory_excerpt") or "")[:500]
            es = rec.get("edge_sequence") or []
            sub = " ".join(
                f"{e[0]} -{e[1]}-> {e[2]}"
                for e in es[:8]
                if isinstance(e, (list, tuple)) and len(e) == 3
            )
            case_lines.append(
                f"--- Case {idx} ---\nQuery: {q}\nSubpath (canonical): {sub}\nTrajectory excerpt: {ex}"
            )
        # Expose MergedPath as compact 1-hop triples around matched entities (category-level),
        # rather than a long paragraph. This is meant for prompt injection.
        def _cat(node_id: str) -> str:
            s = (node_id or "").strip()
            # node ids often look like "dishsponge:dishsponge 2" → keep only category "dishsponge"
            if ":" in s:
                s = s.split(":", 1)[0].strip()
            # fallback: drop trailing instance index tokens
            if " " in s:
                s = s.split(" ", 1)[0].strip()
            return s

        matched_set = set(matched)
        triples: list[str] = []
        seen_tri: set[tuple[str, str, str]] = set()

        # Collect 1-hop outgoing + incoming edges around matched nodes.
        for nid in matched:
            if not self.merged_lg.has_node(nid):
                continue
            # outgoing
            for _u, v, _k, d in self.merged_lg.out_edges(nid, keys=True, data=True):
                rel = str((d or {}).get("relation") or "").strip()
                h, t = _cat(str(nid)), _cat(str(v))
                tri = (h, rel, t)
                if rel and tri not in seen_tri:
                    seen_tri.add(tri)
                    triples.append(f"{h} - {rel} - {t}")
            # incoming
            for u, _v, _k, d in self.merged_lg.in_edges(nid, keys=True, data=True):
                rel = str((d or {}).get("relation") or "").strip()
                h, t = _cat(str(u)), _cat(str(nid))
                tri = (h, rel, t)
                if rel and tri not in seen_tri:
                    seen_tri.add(tri)
                    triples.append(f"{h} - {rel} - {t}")

        if not triples:
            return ""
        # Keep it short to avoid flooding the prompt.
        triples = triples[:12]
        return "[MergedPath]\n" + "\n".join(triples)

    def find_local_instances(self, concept: str) -> list[str]:
        return self.local_graph.find_instances_by_concept(concept)

    # ---- compatibility helpers (for existing training scripts) ----

    def persist_entity_graph(self) -> None:
        """
        For compatibility with MemoryBankGraphMASMemory.
        Persist both the global experience graph (GG) and the last-seen
        local instance graph (GL) snapshot to disk so that collaborative
        training/eval scripts can inspect saved graphs.
        """
        self.global_graph.persist()
        try:
            local_path = os.path.join(self.persist_dir, "local_instance_graph.json")
            write_json(self.local_graph.snapshot(), local_path)
            merged_path = os.path.join(self.persist_dir, "local_category_graph.json")
            write_json(self._category_graph_snapshot(self._build_category_graph()), merged_path)
            self._persist_merged_lg_best_effort()
            self._mirror_artifacts_to_viz_output_dir(
                [
                    "local_instance_graph.json",
                    "local_category_graph.json",
                    "merged_instance_graph.json",
                ]
            )
        except Exception:
            # Persistence of GL is best-effort; avoid crashing training.
            pass

    # ---- helpers: insights / entity retrieval ----

    def _load_insight_bank(self) -> list[dict[str, Any]]:
        payload = load_json(self.insight_bank_path)
        if isinstance(payload, list):
            return payload
        return []

    def _persist_insight_bank(self) -> None:
        write_json(self.insight_bank, self.insight_bank_path)

    def _summarize_task_insights(self, mas_message: MASMessage) -> list[str]:
        if mas_message.label is not True:
            return []
        prompt = _INSIGHT_USER_PROMPT.format(
            task=mas_message.task_description or "",
            trajectory=mas_message.task_trajectory or "",
        )
        raw = self.llm_model(
            [Message("system", _INSIGHT_SYSTEM_PROMPT), Message("user", prompt)],
            temperature=0,
            max_tokens=320,
        )
        return self._parse_json_string_list(raw)

    def _upsert_insights(self, insights: list[str], related_entities: list[str]) -> None:
        rel_set = set([e.strip().lower() for e in related_entities if e and e.strip()])
        for insight in insights:
            text = (insight or "").strip()
            if not text:
                continue
            hit = None
            for item in self.insight_bank:
                if (item.get("insight") or "").strip().lower() == text.lower():
                    hit = item
                    break
            if hit is None:
                self.insight_bank.append(
                    {"insight": text, "related_entities": sorted(rel_set), "count": 1}
                )
            else:
                old = set([x.strip().lower() for x in hit.get("related_entities", []) if x])
                hit["related_entities"] = sorted(old | rel_set)
                hit["count"] = int(hit.get("count", 0)) + 1
        self._persist_insight_bank()

    def _extract_query_entities(self, query: str) -> list[str]:
        if not query:
            return []
        raw = self.llm_model(
            [
                Message("system", _QUERY_ENTITY_SYSTEM_PROMPT),
                Message("user", _QUERY_ENTITY_USER_PROMPT.format(query=query)),
            ],
            temperature=0,
            max_tokens=128,
        )
        entities = [e.strip().lower() for e in self._parse_json_string_list(raw) if e.strip()]
        if entities:
            return entities
        tokens = re.findall(r"[a-zA-Z_]+", query.lower())
        stop = {"the", "a", "an", "and", "or", "to", "in", "on", "of", "with", "put", "some", "your", "task", "is"}
        return [t for t in tokens if t not in stop][:6]

    def _parse_json_string_list(self, raw: str) -> list[str]:
        if not raw:
            return []
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []
        try:
            arr = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(arr, list):
            return []
        return [str(x).strip() for x in arr if str(x).strip()]

    def _infer_category(self, instance_id: str, data: dict[str, Any]) -> str:
        category = str(data.get("category") or "").strip().lower()
        if category:
            return category
        label = str(data.get("label") or "").strip().lower()
        if label:
            return label
        m = re.match(r"([a-zA-Z_]+)", instance_id or "")
        return m.group(1).lower() if m else str(instance_id).strip().lower()

    def _build_category_graph(self) -> nx.DiGraph:
        merged = nx.DiGraph()
        inst2cat: dict[str, str] = {}
        for inst_id, data in self.local_graph.graph.nodes(data=True):
            cat = self._infer_category(str(inst_id), data or {})
            if not cat:
                continue
            inst2cat[str(inst_id)] = cat
            if cat not in merged:
                merged.add_node(cat)

        for u, v, data in self.local_graph.graph.edges(data=True):
            cu = inst2cat.get(str(u))
            cv = inst2cat.get(str(v))
            if not cu or not cv:
                continue
            relation = str((data or {}).get("relation") or "").strip() or "related_to"
            if merged.has_edge(cu, cv):
                rels = set(merged[cu][cv].get("relations", []))
                rels.add(relation)
                merged[cu][cv]["relations"] = sorted(rels)
                merged[cu][cv]["count"] = int(merged[cu][cv].get("count", 1)) + 1
            else:
                merged.add_edge(cu, cv, relations=[relation], count=1)
        return merged

    def _retrieve_local_paths(
        self,
        graph: nx.DiGraph,
        query_entities: list[str],
        max_paths: int = 3,
        max_hops: int = 4,
    ) -> list[str]:
        if not query_entities or len(graph.nodes) == 0:
            return []
        all_nodes = list(graph.nodes)
        matched_nodes: list[str] = []
        for ent in query_entities:
            e = ent.strip().lower()
            if not e:
                continue
            if e in graph:
                matched_nodes.append(e)
                continue
            for n in all_nodes:
                if e in n or n in e:
                    matched_nodes.append(n)
                    break
        matched_nodes = list(dict.fromkeys(matched_nodes))
        if len(matched_nodes) < 2:
            return []

        undirected = graph.to_undirected()
        paths: list[str] = []
        seen: set[tuple[str, str]] = set()

        def _collect_one_hop_triples(center_nodes: set[str], cap: int = 5) -> list[str]:
            """
            Collect up to `cap` (u, relation, v) triples within 1 hop around `center_nodes`.
            Category graph is a DiGraph, so we collect both outgoing and incoming relations.
            """
            triples: list[str] = []
            seen_triple: set[tuple[str, str, str]] = set()

            # Limit search scope to keep prompts short.
            for n in list(center_nodes):
                if len(triples) >= cap:
                    break

                # Outgoing edges: n -> nbr
                try:
                    nbrs_out = set(graph.successors(n))
                except Exception:
                    nbrs_out = set()
                for nbr in nbrs_out:
                    if len(triples) >= cap:
                        break
                    if graph.has_edge(n, nbr):
                        rels = graph[n][nbr].get("relations", []) or []
                        for rel in rels:
                            key = (n, str(rel), str(nbr))
                            if key in seen_triple:
                                continue
                            seen_triple.add(key)
                            triples.append(f"{n}-[{rel}]->{nbr}")
                            if len(triples) >= cap:
                                break

                if len(triples) >= cap:
                    break

                # Incoming edges: nbr -> n
                try:
                    nbrs_in = set(graph.predecessors(n))
                except Exception:
                    nbrs_in = set()
                for nbr in nbrs_in:
                    if len(triples) >= cap:
                        break
                    if graph.has_edge(nbr, n):
                        rels = graph[nbr][n].get("relations", []) or []
                        for rel in rels:
                            key = (nbr, str(rel), n)
                            if key in seen_triple:
                                continue
                            seen_triple.add(key)
                            triples.append(f"{nbr}-[{rel}]->{n}")
                            if len(triples) >= cap:
                                break

                if len(triples) >= cap:
                    break
            return triples[:cap]

        for i in range(len(matched_nodes)):
            for j in range(i + 1, len(matched_nodes)):
                s, t = matched_nodes[i], matched_nodes[j]
                pair = tuple(sorted([s, t]))
                if pair in seen:
                    continue
                seen.add(pair)
                try:
                    path = nx.shortest_path(undirected, source=s, target=t)
                except nx.NetworkXNoPath:
                    continue
                if len(path) - 1 > max_hops:
                    continue
                chunks: list[str] = []
                path_nodes_set: set[str] = set(path)
                for idx, node in enumerate(path):
                    chunks.append(node)
                    if idx < len(path) - 1:
                        a = path[idx]
                        b = path[idx + 1]
                        rels = []
                        if graph.has_edge(a, b):
                            rels.extend(graph[a][b].get("relations", []))
                        if graph.has_edge(b, a):
                            rels.extend(graph[b][a].get("relations", []))
                        rel = "|".join(sorted(set(rels))[:2]) if rels else "linked"
                        chunks.append(f"-[{rel}]-")
                # Add one-hop neighbor triples for extra local context.
                one_hop_triples = _collect_one_hop_triples(path_nodes_set, cap=5)
                triples_text = "; ".join(one_hop_triples) if one_hop_triples else "[]"
                paths.append(
                    "[LocalPath] (仅供参考，不必严格按此结构执行) "
                    + " ".join(chunks)
                    + f" | OneHopTriples(max5): {triples_text}"
                )
                if len(paths) >= max_paths:
                    return paths
        return paths

    def _retrieve_top_insights(self, query_entities: list[str], topk: int = 3) -> list[str]:
        if not self.insight_bank:
            return []
        qset = set([q.strip().lower() for q in query_entities if q and q.strip()])
        scored: list[tuple[int, int, str]] = []
        for item in self.insight_bank:
            insight = (item.get("insight") or "").strip()
            rel = set([x.strip().lower() for x in item.get("related_entities", []) if x])
            overlap = len(qset & rel)
            if overlap <= 0:
                continue
            count = int(item.get("count", 0))
            scored.append((overlap, count, insight))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [f"[InsightHit overlap={ov}, seen={cnt}] {text}" for ov, cnt, text in scored[:max(1, topk)]]

    def _category_graph_snapshot(self, graph: nx.DiGraph) -> dict[str, Any]:
        return {
            "nodes": [{"id": node} for node in graph.nodes],
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "relations": data.get("relations", []),
                    "count": data.get("count", 1),
                }
                for u, v, data in graph.edges(data=True)
            ],
        }

