import networkx as nx
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocalInstanceGraph:
    """Per-agent local instance graph (GL) capturing concrete objects and spatial relations."""

    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    def clear(self) -> None:
        self.graph.clear()

    def upsert_instance(
        self,
        instance_id: str,
        *,
        label: str | None = None,
        category: str | None = None,
        state: str | None = None,
        room: str | None = None,
        **attrs: Any,
    ) -> None:
        """Create or update a concrete instance node."""
        if instance_id not in self.graph:
            self.graph.add_node(instance_id)
        data = self.graph.nodes[instance_id]
        if label is not None:
            data["label"] = label
        if category is not None:
            data["category"] = category
        if state is not None:
            data["state"] = state
        if room is not None:
            data["room"] = room
        for k, v in attrs.items():
            data[k] = v

    def add_relation(
        self,
        subject_id: str,
        object_id: str,
        relation: str,
        **attrs: Any,
    ) -> None:
        """Add a directed relation edge such as Inside/On/At."""
        if subject_id not in self.graph:
            self.graph.add_node(subject_id)
        if object_id not in self.graph:
            self.graph.add_node(object_id)
        edge_attrs = {"relation": relation}
        edge_attrs.update(attrs)
        self.graph.add_edge(subject_id, object_id, **edge_attrs)

    def find_instances_by_concept(self, concept: str) -> list[str]:
        """Return instance ids whose category or label matches a given concept."""
        matches: list[str] = []
        for node_id, data in self.graph.nodes(data=True):
            label = str(data.get("label") or "")
            category = str(data.get("category") or "")
            if concept == label or concept == category:
                matches.append(node_id)
        return matches

    def snapshot(self) -> dict[str, Any]:
        """Return a lightweight serializable snapshot for debugging or persistence."""
        return {
            "nodes": [
                {"id": node_id, **(data or {})}
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "key": k,
                    **(data or {}),
                }
                for u, v, k, data in self.graph.edges(keys=True, data=True)
            ],
        }

