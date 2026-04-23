#!/usr/bin/env python3
"""
Visualize SelectiveMem's Local Instance Graph (GL) snapshot JSON.

Usage:
  python scripts/alfworld/visualize_local_instance_graph.py \
    --input path/to/local_instance_graph.json \
    --output path/to/local_instance_graph.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx


def build_graph(snapshot: dict) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for node in snapshot.get("nodes", []):
        node_id = node.get("id")
        if not node_id:
            continue
        attrs = dict(node)
        attrs.pop("id", None)
        g.add_node(node_id, **attrs)
    for edge in snapshot.get("edges", []):
        u = edge.get("from")
        v = edge.get("to")
        if not u or not v:
            continue
        attrs = dict(edge)
        attrs.pop("from", None)
        attrs.pop("to", None)
        key = attrs.pop("key", None)
        if key is None:
            g.add_edge(u, v, **attrs)
        else:
            g.add_edge(u, v, key=key, **attrs)
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--max_nodes", type=int, default=80)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    snapshot = json.loads(inp.read_text(encoding="utf-8"))
    g = build_graph(snapshot)

    # Optional downsample for readability: keep top-degree nodes
    if args.max_nodes and g.number_of_nodes() > args.max_nodes:
        degrees = sorted(g.degree, key=lambda x: x[1], reverse=True)
        keep = {n for n, _d in degrees[: args.max_nodes]}
        g = g.subgraph(keep).copy()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for visualization.") from exc

    plt.figure(figsize=(16, 12))
    pos = nx.spring_layout(g, seed=42, k=0.8 / max(1, g.number_of_nodes()) ** 0.5)

    node_labels = {}
    for n, data in g.nodes(data=True):
        label = data.get("label") or data.get("category") or n
        cat = data.get("category")
        st = data.get("state")
        parts = [str(label)]
        if cat:
            parts.append(f"[{cat}]")
        if st:
            parts.append(f"({st})")
        node_labels[n] = " ".join(parts)

    nx.draw_networkx_nodes(g, pos, node_size=900, node_color="#E8F0FE", edgecolors="#5F6368", linewidths=1.0)
    nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.0, edge_color="#5F6368", alpha=0.7)

    edge_labels = {}
    for u, v, k, data in g.edges(keys=True, data=True):
        rel = data.get("relation") or ""
        if not rel and data.get("relations"):
            rel = "|".join([str(x) for x in data.get("relations", [])[:3]])
        if not rel:
            continue
        edge_labels[(u, v, k)] = str(rel)
    if edge_labels:
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7, rotate=False, label_pos=0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

