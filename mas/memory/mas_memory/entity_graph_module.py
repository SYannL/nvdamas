import os
import json
import re
from typing import Any

from mas.llm import Message
from mas.utils import load_json, write_json
from ..common import MASMessage
from ..utils import cosine_similarity
from .insight_prompts import (
    META_SYSTEM,
    META_USER_TASK,
    META_USER_CHAIN,
    SUMMARY_SYSTEM,
    SUMMARY_USER_TASK,
    SUMMARY_USER_CHAIN,
)


_ENTITY_SYSTEM_PROMPT = "You extract concise domain entities."
_ENTITY_USER_PROMPT = (
    "Extract the key entities or concepts from the text below. "
    "Return a JSON array of strings. Only return the JSON array.\n\n"
    "Text:\n{text}"
)


class EntityGraphModule:
    def __init__(
        self,
        *,
        persist_dir: str,
        embedding_func,
        llm_model,
        merge_threshold: float = 0.9,
        persist_every: int = 10,
    ) -> None:
        self.graph_path: str = os.path.join(persist_dir, "entity_graph.json")
        self.embedding_func = embedding_func
        self.llm_model = llm_model
        self._entity_merge_threshold = merge_threshold
        self._persist_every = max(1, persist_every)
        self._tasks_since_persist = 0
        self.graph: dict[str, Any] = self._load_graph()

    def add_task_from_message(self, mas_message: MASMessage) -> None:
        steps = self._extract_trajectory_steps(mas_message.task_trajectory)
        if not steps:
            return

        chain_id = f"chain-{self.graph['meta']['next_chain_id']}"
        self.graph["meta"]["next_chain_id"] += 1
        self.graph["chains"][chain_id] = {
            "task_main": mas_message.task_main,
            "knowledge_summary": mas_message.get_extra_field("knowledge_summary"),
            "steps": steps,
        }

        prev_node_id = None
        for step_idx, step in enumerate(steps):
            thought_text = self._extract_thought_content(step)
            entities = self._extract_entities(thought_text)
            entity_text = ", ".join(entities) if entities else thought_text
            if not entity_text:
                entity_text = step
            embedding = self.embedding_func.embed_text(entity_text)

            node_id = self._get_or_create_node(
                chain_id=chain_id,
                step_idx=step_idx,
                content=step,
                entities=entities,
                entity_text=entity_text,
                embedding=embedding,
            )

            if prev_node_id is not None:
                self.graph["edges"].append({
                    "from": prev_node_id,
                    "to": node_id,
                    "chain_id": chain_id,
                    "step_idx": step_idx
                })
            prev_node_id = node_id

        self._tasks_since_persist += 1
        if self._tasks_since_persist >= self._persist_every:
            self.persist()

    def persist(self) -> None:
        """Force persist the entity graph to disk (merge nodes and write file)."""
        if self._tasks_since_persist > 0 or self.graph.get("chains"):
            self._merge_nodes_by_entity()
            self._persist_graph()
        self._tasks_since_persist = 0

    def retrieve_insights(
        self,
        query_task: str,
        top_nodes: int = 10,
        top_chains: int = 3,
        entity_sim_threshold: float = 0.9,
    ) -> list[str]:
        """
        Retrieve insights by matching query entities against each node's entities.
        Node is "hit" when at least one of its entities has similarity > threshold with any query entity.
        Chains are ranked by total hit count of their nodes.
        """
        if not self.graph["nodes"]:
            return []

        query_entities = self._extract_entities(query_task)
        if not query_entities:
            return []

        all_node_entities: set[str] = set()
        for node in self.graph["nodes"]:
            all_node_entities.update(node.get("entities") or [])

        all_entities = set(query_entities) | all_node_entities
        entity_embeddings = self._embed_entities_cached(list(all_entities))
        query_vecs = [entity_embeddings[e] for e in query_entities if e in entity_embeddings]
        if not query_vecs:
            return []

        scored_nodes: list[tuple[dict, int]] = []
        for node in self.graph["nodes"]:
            node_entities = node.get("entities") or []
            if not node_entities:
                continue
            hit_count = self._count_entity_hits(
                node_entities, query_vecs, entity_embeddings, entity_sim_threshold
            )
            if hit_count > 0:
                scored_nodes.append((node, hit_count))

        if not scored_nodes:
            return []

        scored_nodes.sort(key=lambda item: item[1], reverse=True)
        selected_nodes = scored_nodes[:top_nodes]

        chain_scores: dict[str, int] = {}
        for node, hit_count in selected_nodes:
            chain_ids = node.get("chain_ids") or []
            for chain_id in chain_ids:
                chain_scores[chain_id] = chain_scores.get(chain_id, 0) + hit_count

        if not chain_scores:
            return []

        sorted_chains = sorted(
            chain_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_chains]

        insights: list[str] = []
        for idx, (chain_id, _) in enumerate(sorted_chains, 1):
            chain_steps = self.graph.get("chains", {}).get(chain_id, {}).get("steps") or []
            if not chain_steps:
                continue
            steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(chain_steps)])
            summary = self._summarize_chain(steps, chain_id=chain_id)
            if summary:
                insights.append(f"[Graph Chain {idx}] {summary}")
        return insights

    def summarize_task_knowledge(self, mas_message: MASMessage) -> str:
        if mas_message.label is not True:
            return ""
        task = mas_message.task_description or ""
        trajectory = mas_message.task_trajectory or ""
        if not task and not trajectory:
            return ""

        # Stage 1: Meta-LLM decides what should be summarized
        meta_prompt = META_USER_TASK.format(task=task, trajectory=trajectory)
        meta_messages = [
            Message("system", META_SYSTEM),
            Message("user", meta_prompt),
        ]
        guidance = self.llm_model(meta_messages, temperature=0, max_tokens=256).strip()
        if not guidance:
            return ""

        # Stage 2: Generate summary following the guidance
        summary_prompt = SUMMARY_USER_TASK.format(
            guidance=guidance, task=task, trajectory=trajectory
        )
        summary_messages = [
            Message("system", SUMMARY_SYSTEM),
            Message("user", summary_prompt),
        ]
        return self.llm_model(summary_messages, temperature=0, max_tokens=256).strip()

    def _extract_trajectory_steps(self, task_trajectory: str) -> list[str]:
        if not task_trajectory:
            return []
        segments = [seg.strip() for seg in task_trajectory.split("\n>") if seg.strip()]
        steps = []
        for seg in segments:
            cleaned = seg.lstrip(">")
            cleaned = cleaned.strip()
            if cleaned:
                steps.append(cleaned)
        return steps

    def _extract_thought_content(self, step: str) -> str:
        lines = [line.strip() for line in step.splitlines() if line.strip()]
        for line in lines:
            if line.lower().startswith("thought:"):
                return line.split(":", 1)[1].strip()
        return step

    def _extract_entities(self, text: str) -> list[str]:
        if not text:
            return []
        prompt = _ENTITY_USER_PROMPT.format(text=text)
        messages = [
            Message("system", _ENTITY_SYSTEM_PROMPT),
            Message("user", prompt),
        ]
        response = self.llm_model(messages, temperature=0, max_tokens=128)
        entities = self._parse_json_list(response)
        cleaned = []
        for ent in entities:
            ent = ent.strip()
            ent = re.sub(r"\s+", " ", ent)
            if ent:
                cleaned.append(ent)
        return cleaned

    def _embed_entities_cached(self, entities: list[str]) -> dict[str, list[float]]:
        """Embed entities with cache to avoid duplicate embedding calls."""
        result: dict[str, list[float]] = {}
        for ent in entities:
            if not ent or ent in result:
                continue
            result[ent] = self.embedding_func.embed_text(ent)
        return result

    def _count_entity_hits(
        self,
        node_entities: list[str],
        query_vecs: list[list[float]],
        node_entity_embeddings: dict[str, list[float]],
        threshold: float,
    ) -> int:
        """Count how many node entities match (sim > threshold) any query entity."""
        if not node_entities or not query_vecs:
            return 0
        hit_count = 0
        for node_ent in node_entities:
            node_emb = node_entity_embeddings.get(node_ent)
            if not node_emb:
                continue
            max_sim = max(
                cosine_similarity(node_emb, q_vec) for q_vec in query_vecs
            )
            if max_sim >= threshold:
                hit_count += 1
        return hit_count

    def _summarize_chain(self, steps: str, chain_id: str) -> str:
        chain_meta = self.graph.get("chains", {}).get(chain_id, {})
        knowledge_hint = chain_meta.get("knowledge_summary")
        if knowledge_hint:
            steps = f"{steps}\n\nKnown knowledge summary:\n{knowledge_hint}"

        # Stage 1: Meta-LLM decides what should be summarized
        meta_prompt = META_USER_CHAIN.format(steps=steps)
        meta_messages = [
            Message("system", META_SYSTEM),
            Message("user", meta_prompt),
        ]
        guidance = self.llm_model(meta_messages, temperature=0, max_tokens=256).strip()
        if not guidance:
            return ""

        # Stage 2: Generate summary following the guidance
        summary_prompt = SUMMARY_USER_CHAIN.format(guidance=guidance, steps=steps)
        summary_messages = [
            Message("system", SUMMARY_SYSTEM),
            Message("user", summary_prompt),
        ]
        return self.llm_model(summary_messages, temperature=0, max_tokens=256).strip()

    def _load_graph(self) -> dict[str, Any]:
        graph = load_json(self.graph_path)
        if graph is None:
            graph = {
                "nodes": [],
                "edges": [],
                "chains": {},
                "meta": {
                    "next_node_id": 0,
                    "next_chain_id": 0
                }
            }
        else:
            graph.setdefault("nodes", [])
            graph.setdefault("edges", [])
            graph.setdefault("chains", {})
            graph.setdefault("meta", {})
            graph["meta"].setdefault("next_node_id", len(graph["nodes"]))
            graph["meta"].setdefault("next_chain_id", len(graph["chains"]))
            for node in graph["nodes"]:
                if "chain_ids" not in node and "chain_id" in node:
                    node["chain_ids"] = [node.get("chain_id")]
                node.pop("chain_id", None)
        return graph

    def _persist_graph(self) -> None:
        write_json(self.graph, self.graph_path)

    @staticmethod
    def _parse_json_list(response: str) -> list[str]:
        if not response:
            return []
        match = re.search(r"\[[\s\S]*\]", response)
        if match:
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, list):
                    return [str(item) for item in payload]
            except json.JSONDecodeError:
                pass
        items = re.split(r"[,\n;]", response)
        return [item.strip().strip('"').strip("'") for item in items if item.strip()]

    def _get_or_create_node(
        self,
        chain_id: str,
        step_idx: int,
        content: str,
        entities: list[str],
        entity_text: str,
        embedding: list[float]
    ) -> int:
        best_id = None
        best_sim = 0.0
        for node in self.graph["nodes"]:
            node_embedding = node.get("embedding")
            if not node_embedding:
                continue
            sim = cosine_similarity(embedding, node_embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = node.get("id")

        if best_id is not None and best_sim >= self._entity_merge_threshold:
            target = next(n for n in self.graph["nodes"] if n.get("id") == best_id)
            chain_ids = target.setdefault("chain_ids", [])
            if chain_id not in chain_ids:
                chain_ids.append(chain_id)
            if entities:
                target_entities = set(target.get("entities") or [])
                target_entities.update(entities)
                target["entities"] = sorted(target_entities)
            target.setdefault("entity_text", entity_text)
            return best_id

        node_id = self.graph["meta"]["next_node_id"]
        self.graph["meta"]["next_node_id"] += 1
        node = {
            "id": node_id,
            "chain_ids": [chain_id],
            "step_idx": step_idx,
            "content": content,
            "entities": entities,
            "entity_text": entity_text,
            "embedding": embedding,
        }
        self.graph["nodes"].append(node)
        return node_id

    def _merge_nodes_by_entity(self) -> None:
        nodes = self.graph.get("nodes", [])
        if len(nodes) < 2:
            return
        merged_ids: dict[int, int] = {}
        for i in range(len(nodes)):
            a = nodes[i]
            if a["id"] in merged_ids:
                continue
            emb_a = a.get("embedding")
            if not emb_a:
                continue
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if b["id"] in merged_ids:
                    continue
                emb_b = b.get("embedding")
                if not emb_b:
                    continue
                sim = cosine_similarity(emb_a, emb_b)
                if sim >= self._entity_merge_threshold:
                    merged_ids[b["id"]] = a["id"]
                    a_chain_ids = set(a.get("chain_ids") or [])
                    a_chain_ids.update(b.get("chain_ids") or [])
                    a["chain_ids"] = sorted(a_chain_ids)
                    a_entities = set(a.get("entities") or [])
                    a_entities.update(b.get("entities") or [])
                    a["entities"] = sorted(a_entities)
                    if not a.get("entity_text"):
                        a["entity_text"] = b.get("entity_text")

        if not merged_ids:
            return

        for edge in self.graph.get("edges", []):
            if edge["from"] in merged_ids:
                edge["from"] = merged_ids[edge["from"]]
            if edge["to"] in merged_ids:
                edge["to"] = merged_ids[edge["to"]]

        kept_ids = set(merged_ids.values())
        self.graph["nodes"] = [n for n in nodes if n["id"] not in merged_ids or n["id"] in kept_ids]
