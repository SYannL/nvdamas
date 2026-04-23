import os
from dataclasses import dataclass, field
from typing import Any

from mas.utils import load_json, write_json


@dataclass
class GlobalExperienceGraph:
    """
    Global experience graph (GG) storing abstract concept-level knowledge.

    Nodes represent concepts such as objects, actions, or rooms.
    Edges represent relations such as requires / can_be_cleaned_at / found_in with weights.
    """

    persist_dir: str
    graph_filename: str = "global_experience_graph.json"
    _graph: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = os.path.join(self.persist_dir, self.graph_filename)
        self._graph = self._load_graph()

    # ---- public API ----

    @property
    def graph(self) -> dict[str, Any]:
        return self._graph

    def persist(self) -> None:
        """Public helper to force persist the graph to disk."""
        self._persist_graph()

    def add_experience(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        success: bool = True,
        weight_delta: float = 1.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Add or reinforce an abstract triplet: subject -[relation]-> object.
        Success increases edge weight, failure can be represented with negative or smaller delta.
        """
        subj_id = self._get_or_create_concept(subject)
        obj_id = self._get_or_create_concept(obj)

        edge = self._find_edge(subj_id, relation, obj_id)
        if edge is None:
            edge = {
                "from": subj_id,
                "to": obj_id,
                "relation": relation,
                "weight": 0.0,
                "meta": {},
            }
            self._graph["edges"].append(edge)

        delta = weight_delta if success else -abs(weight_delta)
        edge["weight"] = float(edge.get("weight", 0.0) + delta)
        if meta:
            edge_meta = edge.setdefault("meta", {})
            for k, v in meta.items():
                edge_meta[k] = v

        self._persist_graph()

    def query_plans(self, query_concepts: list[str]) -> dict[str, Any]:
        """
        Lightweight query over GG to suggest required concepts / rooms.
        For now, a simple heuristic over edges; can be extended later.
        """
        required_concepts: set[str] = set()
        suggested_rooms: dict[str, float] = {}
        support_paths: list[dict[str, Any]] = []

        concept_ids = {
            cid: node
            for cid, node in self._graph.get("nodes", {}).items()
            if node.get("name") in query_concepts
        }
        if not concept_ids:
            return {
                "required_concepts": [],
                "suggested_rooms": [],
                "support_paths": [],
            }

        id_to_name = {cid: data.get("name") for cid, data in self._graph["nodes"].items()}
        for edge in self._graph.get("edges", []):
            src_name = id_to_name.get(edge.get("from"))
            dst_name = id_to_name.get(edge.get("to"))
            if not src_name or not dst_name:
                continue
            if src_name not in query_concepts:
                continue

            relation = edge.get("relation")
            weight = float(edge.get("weight", 0.0))
            if relation == "requires" and weight > 0:
                required_concepts.add(dst_name)
                support_paths.append(
                    {
                        "from": src_name,
                        "relation": relation,
                        "to": dst_name,
                        "weight": weight,
                    }
                )
            if relation == "found_in" and weight > 0:
                suggested_rooms[dst_name] = suggested_rooms.get(dst_name, 0.0) + weight
                support_paths.append(
                    {
                        "from": src_name,
                        "relation": relation,
                        "to": dst_name,
                        "weight": weight,
                    }
                )

        sorted_rooms = sorted(
            suggested_rooms.items(), key=lambda x: x[1], reverse=True
        )

        return {
            "required_concepts": sorted(required_concepts),
            "suggested_rooms": sorted_rooms,
            "support_paths": support_paths,
        }

    # ---- internal helpers ----

    def _load_graph(self) -> dict[str, Any]:
        graph = load_json(self._path)
        if graph is None:
            graph = {
                "nodes": {},
                "edges": [],
                "meta": {
                    "next_node_id": 0,
                },
            }
        else:
            graph.setdefault("nodes", {})
            graph.setdefault("edges", [])
            graph.setdefault("meta", {})
            graph["meta"].setdefault("next_node_id", len(graph["nodes"]))
        return graph

    def _persist_graph(self) -> None:
        write_json(self._graph, self._path)

    def _get_or_create_concept(self, name: str, concept_type: str | None = None) -> str:
        for cid, node in self._graph["nodes"].items():
            if node.get("name") == name:
                return cid
        cid = f"c{self._graph['meta']['next_node_id']}"
        self._graph["meta"]["next_node_id"] += 1
        self._graph["nodes"][cid] = {
            "name": name,
        }
        if concept_type is not None:
            self._graph["nodes"][cid]["type"] = concept_type
        return cid

    def _find_edge(self, from_id: str, relation: str, to_id: str) -> dict[str, Any] | None:
        for edge in self._graph.get("edges", []):
            if (
                edge.get("from") == from_id
                and edge.get("to") == to_id
                and edge.get("relation") == relation
            ):
                return edge
        return None

