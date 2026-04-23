#!/usr/bin/env python3
"""
Merge two categorized LG graphs (A/B) into one, merging same-id nodes and color-annotating sources.

Inputs are SelectiveMem categorized LG snapshots:
  local_category_graph.json (schema: {"nodes":[{"id":...}], "edges":[{"from":...,"to":...,"relations":[...]}]})

Behavior
- Nodes are merged by node id (string). Node attr `sources` is {"A"}, {"B"}, or {"A","B"}.
- Edges are merged by (u, rel, v). Edge attr `sources` is {"A"}, {"B"}, or {"A","B"}.
- Renders a PNG with:
  - Node colors: A-only blue, B-only red, shared green
  - Edge colors: A-only blue, B-only red, shared green
  - Labels show node id; edge labels show relation (for a subset to keep readable)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_cat_multidigraph(snapshot: dict[str, Any]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for n in snapshot.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if nid is None:
            continue
        attrs = dict(n)
        attrs.pop("id", None)
        g.add_node(str(nid), **attrs)

    for e in snapshot.get("edges", []) or []:
        if not isinstance(e, dict):
            continue
        u = e.get("from")
        v = e.get("to")
        if u is None or v is None:
            continue
        u = str(u)
        v = str(v)
        rels: list[str] = []
        if e.get("relations"):
            rels = [str(x).strip() for x in (e.get("relations") or []) if str(x).strip()]
        elif e.get("relation"):
            rels = [str(e.get("relation")).strip()]
        if not rels:
            rels = ["related_to"]
        for rel in rels:
            g.add_edge(u, v, relation=rel)
    return g


def _merge_ab(a: nx.MultiDiGraph, b: nx.MultiDiGraph) -> nx.MultiDiGraph:
    out = nx.MultiDiGraph()

    # nodes
    for n, data in a.nodes(data=True):
        out.add_node(str(n), **(data or {}), sources={"A"})
    for n, data in b.nodes(data=True):
        nid = str(n)
        if nid in out:
            src = out.nodes[nid].get("sources")
            if isinstance(src, set):
                src.add("B")
            else:
                out.nodes[nid]["sources"] = {"A", "B"}
            # keep existing attrs; best-effort merge missing keys
            for k, v in (data or {}).items():
                out.nodes[nid].setdefault(k, v)
        else:
            out.add_node(nid, **(data or {}), sources={"B"})

    # edges merge by (u, rel, v)
    edge_sources: dict[tuple[str, str, str], set[str]] = {}
    for u, v, _k, d in a.edges(keys=True, data=True):
        rel = str((d or {}).get("relation") or "").strip() or "related_to"
        edge_sources.setdefault((str(u), rel, str(v)), set()).add("A")
    for u, v, _k, d in b.edges(keys=True, data=True):
        rel = str((d or {}).get("relation") or "").strip() or "related_to"
        edge_sources.setdefault((str(u), rel, str(v)), set()).add("B")

    for (u, rel, v), srcs in edge_sources.items():
        out.add_edge(u, v, relation=rel, sources=set(srcs))

    return out


def _downsample_keep_shared(g: nx.MultiDiGraph, max_nodes: int) -> nx.MultiDiGraph:
    if max_nodes <= 0 or g.number_of_nodes() <= max_nodes:
        return g
    shared = {n for n, d in g.nodes(data=True) if isinstance(d.get("sources"), set) and d["sources"] == {"A", "B"}}
    deg = sorted(g.degree, key=lambda x: x[1], reverse=True)
    picked: list[str] = []
    for n, _d in deg:
        nid = str(n)
        if nid in shared or len(picked) < max_nodes:
            picked.append(nid)
        if len(set(picked)) >= max_nodes and shared.issubset(set(picked)):
            break
    keep = set(picked) | shared
    return g.subgraph([n for n in g.nodes if str(n) in keep]).copy()


def _render(g: nx.MultiDiGraph, out_png: Path, *, max_nodes: int, dpi: int, figsize: tuple[float, float]) -> None:
    g2 = _downsample_keep_shared(g, max_nodes=max_nodes)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib required") from exc

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(g2, seed=42, k=0.9 / max(1, g2.number_of_nodes()) ** 0.5)

    def node_color(n: str) -> str:
        src = g2.nodes[n].get("sources")
        if src == {"A"}:
            return "#D2E3FC"  # blue
        if src == {"B"}:
            return "#FCE8E6"  # red
        if src == {"A", "B"}:
            return "#E6F4EA"  # green
        return "#E8EAED"

    nodelist = [str(n) for n in g2.nodes()]
    node_colors = [node_color(n) for n in nodelist]
    node_sizes = [5200 if g2.nodes[n].get("sources") == {"A", "B"} else 3600 for n in nodelist]

    nx.draw_networkx_nodes(
        g2,
        pos,
        nodelist=nodelist,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#5F6368",
        linewidths=1.2,
    )
    nx.draw_networkx_labels(g2, pos, labels={n: n for n in nodelist}, font_size=10)

    # edge colors by source
    edges_blue = []
    edges_red = []
    edges_green = []
    for u, v, _k, d in g2.edges(keys=True, data=True):
        src = (d or {}).get("sources")
        e = (str(u), str(v))
        if src == {"A"}:
            edges_blue.append(e)
        elif src == {"B"}:
            edges_red.append(e)
        elif src == {"A", "B"}:
            edges_green.append(e)

    if edges_blue:
        nx.draw_networkx_edges(g2, pos, edgelist=edges_blue, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.6, edge_color="#1A73E8", alpha=0.55)
    if edges_red:
        nx.draw_networkx_edges(g2, pos, edgelist=edges_red, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.6, edge_color="#D93025", alpha=0.55)
    if edges_green:
        nx.draw_networkx_edges(g2, pos, edgelist=edges_green, arrows=True, arrowstyle="-|>", arrowsize=14, width=2.6, edge_color="#188038", alpha=0.85)

    # edge labels (cap)
    edge_labels: dict[tuple[str, str], str] = {}
    for u, v, _k, d in g2.edges(keys=True, data=True):
        if len(edge_labels) >= 60:
            break
        rel = str((d or {}).get("relation") or "").strip()
        if rel:
            edge_labels[(str(u), str(v))] = rel
    if edge_labels:
        nx.draw_networkx_edge_labels(g2, pos, edge_labels=edge_labels, font_size=9, rotate=False, label_pos=0.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_graph", required=True, help="A local_category_graph.json (or compatible)")
    ap.add_argument("--b_graph", required=True, help="B local_category_graph.json (or compatible)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_nodes", type=int, default=220)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--figsize", default="20,14")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = tuple(float(x.strip()) for x in str(args.figsize).split(",", 1))
    if len(fs) != 2:
        raise SystemExit("--figsize must be like '20,14'")

    a = _build_cat_multidigraph(_read_json(Path(args.a_graph)))
    b = _build_cat_multidigraph(_read_json(Path(args.b_graph)))
    merged = _merge_ab(a, b)

    # Save merged graph as nodes/edges JSON for reuse.
    nodes = [{"id": n, **(d or {}), "sources": sorted(list((d or {}).get("sources") or []))} for n, d in merged.nodes(data=True)]
    edges = []
    for u, v, _k, d in merged.edges(keys=True, data=True):
        dd = dict(d or {})
        src = dd.get("sources")
        if isinstance(src, set):
            dd["sources"] = sorted(list(src))
        edges.append({"from": str(u), "to": str(v), **dd})
    (out_dir / "AB_merged_category_graph.json").write_text(
        json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _render(merged, out_dir / "AB_merged_category_graph.png", max_nodes=int(args.max_nodes), dpi=int(args.dpi), figsize=fs)
    print(f"[merge_ab_category_graph_viz] wrote outputs to {out_dir}")
    print(json.dumps({"a_nodes": a.number_of_nodes(), "b_nodes": b.number_of_nodes(), "merged_nodes": merged.number_of_nodes(),
                      "a_edges": a.number_of_edges(), "b_edges": b.number_of_edges(), "merged_edges": merged.number_of_edges()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

