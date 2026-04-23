#!/usr/bin/env python3
"""
Build and visualize a merged A/B line-graph from the *latest* categorized LG snapshots.

What it does
- Discover the latest `local_category_graph.json` for side A and side B under a collab run dir
  (prefers `train_local_gg/**/local_category_graph.json`, falls back to `train_local/**/...`).
- Convert each categorized LG (nodes/edges schema) into a directed "line graph":
  - Each original edge (u --rel--> v) becomes a node identified by a canonical triple id.
  - A directed edge exists between two triple-nodes if the tail of the first equals the head of the second
    (i.e., they can be chained).
- Merge A/B line graphs by *node id* (same triple id across A and B becomes the same node).
- Render PNGs:
  - `A_linegraph.png`, `B_linegraph.png`, `AB_linegraph_merged.png`

This is meant to highlight reusable relation-chains (edge patterns) learned from LGs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_lg_from_nodes_edges(snapshot: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Input schema (SelectiveMem categorized LG):
      {"nodes":[{"id":...,"label":...,"category":...}], "edges":[{"from":...,"to":...,"relation":...}]}
    """
    g = nx.MultiDiGraph()
    for node in snapshot.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if nid is None:
            continue
        attrs = dict(node)
        attrs.pop("id", None)
        g.add_node(str(nid), **attrs)
    for edge in snapshot.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        u = edge.get("from")
        v = edge.get("to")
        if u is None or v is None:
            continue
        attrs = dict(edge)
        attrs.pop("from", None)
        attrs.pop("to", None)
        key = attrs.pop("key", None)
        if key is None:
            g.add_edge(str(u), str(v), **attrs)
        else:
            try:
                k = int(key)
            except Exception:
                k = 0
            g.add_edge(str(u), str(v), key=k, **attrs)
    return g


def _triple_id(u: str, rel: str, v: str) -> str:
    # Canonical triple id used to merge A/B.
    return f"{u}||{rel}||{v}"


def _lg_to_linegraph(lg: nx.MultiDiGraph, *, source: str) -> nx.DiGraph:
    """
    Build a directed line-graph over relation-triples.
    Node = one triple (u, rel, v). Edge = chainability (v1 == u2).
    """
    l = nx.DiGraph()

    # Build triple-nodes.
    triples: list[tuple[str, str, str]] = []
    for u, v, _k, data in lg.edges(keys=True, data=True):
        d = data or {}
        rel = str(d.get("relation") or "").strip() or "related_to"
        uu, vv = str(u), str(v)
        tid = _triple_id(uu, rel, vv)
        if tid not in l:
            l.add_node(
                tid,
                label=f"{uu} - {rel} - {vv}",
                u=uu,
                rel=rel,
                v=vv,
                sources={source},
            )
        else:
            # Multi-edge duplicates collapse; keep source set.
            s = l.nodes[tid].get("sources") or set()
            if isinstance(s, set):
                s.add(source)
                l.nodes[tid]["sources"] = s
        triples.append((uu, rel, vv))

    # Chainability edges: (u1,rel1,v1) -> (u2,rel2,v2) if v1 == u2.
    # Use adjacency on original LG for efficiency.
    out_by_u: dict[str, list[str]] = {}
    in_by_v: dict[str, list[str]] = {}
    for uu, rel, vv in triples:
        tid = _triple_id(uu, rel, vv)
        out_by_u.setdefault(uu, []).append(tid)
        in_by_v.setdefault(vv, []).append(tid)

    # If a triple ends at x and another starts at x, connect them.
    for x, incoming in in_by_v.items():
        outgoing = out_by_u.get(x) or []
        if not outgoing:
            continue
        for t1 in incoming:
            for t2 in outgoing:
                if t1 == t2:
                    continue
                l.add_edge(t1, t2)

    return l


def _merge_linegraphs(a: nx.DiGraph, b: nx.DiGraph) -> nx.DiGraph:
    """
    Merge by node id (canonical triple id). Union edges.
    Track sources as a set attribute.
    """
    out = nx.DiGraph()
    out = nx.compose(out, a)
    out = nx.compose(out, b)

    # Merge `sources` sets for overlapping nodes.
    for n in out.nodes:
        sa = a.nodes[n].get("sources") if n in a else None
        sb = b.nodes[n].get("sources") if n in b else None
        merged: set[str] = set()
        for s in (sa, sb, out.nodes[n].get("sources")):
            if isinstance(s, set):
                merged |= set(s)
        out.nodes[n]["sources"] = merged
    return out


def _downsample_by_degree(g: nx.DiGraph, max_nodes: int) -> nx.DiGraph:
    if max_nodes <= 0 or g.number_of_nodes() <= max_nodes:
        return g
    deg = sorted(g.degree, key=lambda x: x[1], reverse=True)
    keep = {n for n, _d in deg[:max_nodes]}
    return g.subgraph(keep).copy()


def _render_linegraph_to_png(
    g: nx.DiGraph,
    out_png: Path,
    *,
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    g2 = _downsample_by_degree(g, max_nodes=max_nodes)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to render graphs to PNG.") from exc

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(g2, seed=42, k=0.9 / max(1, g2.number_of_nodes()) ** 0.5)

    labels = {n: (g2.nodes[n].get("label") or n) for n in g2.nodes}
    # Bigger nodes (line-graph nodes carry more text).
    node_size = 5200
    label_font_size = 10

    # Color nodes by source coverage.
    colors = []
    for n in g2.nodes:
        src = g2.nodes[n].get("sources")
        if isinstance(src, set) and src == {"A"}:
            colors.append("#D2E3FC")  # blue-ish
        elif isinstance(src, set) and src == {"B"}:
            colors.append("#FCE8E6")  # red-ish
        elif isinstance(src, set) and ("A" in src and "B" in src):
            colors.append("#E6F4EA")  # green-ish (shared)
        else:
            colors.append("#E8EAED")  # gray

    nx.draw_networkx_nodes(
        g2,
        pos,
        node_size=node_size,
        node_color=colors,
        edgecolors="#5F6368",
        linewidths=1.0,
    )
    nx.draw_networkx_edges(
        g2,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=1.0,
        edge_color="#5F6368",
        alpha=0.5,
    )
    nx.draw_networkx_labels(g2, pos, labels=labels, font_size=label_font_size)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def _pick_latest_category_json(collab_run_dir: Path, *, prefer_stage: str) -> Path:
    """
    Prefer:
      <collab_run_dir>/<prefer_stage>/**/local_category_graph.json
    Else fallback to:
      <collab_run_dir>/train_local/**/local_category_graph.json
    Pick the newest by mtime across matches.
    """
    collab_run_dir = collab_run_dir.resolve()
    candidates: list[Path] = []
    p1 = collab_run_dir / prefer_stage
    if p1.is_dir():
        candidates.extend(list(p1.rglob("local_category_graph.json")))
    if not candidates:
        p2 = collab_run_dir / "train_local"
        if p2.is_dir():
            candidates.extend(list(p2.rglob("local_category_graph.json")))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise SystemExit(f"No local_category_graph.json found under {collab_run_dir} (prefer {prefer_stage}).")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--collab_run_dir",
        required=True,
        help="e.g. logs/alfworld_collab_eval/<run_id>/autogen/memory/selectivemem/gpt-4o-mini",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--prefer_stage",
        default="train_local_gg",
        choices=["train_local_gg", "train_local"],
        help="Where to prefer picking the latest categorized LG snapshot from.",
    )
    ap.add_argument("--max_nodes", type=int, default=120)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--figsize", type=str, default="18,12", help="e.g. '18,12'")
    args = ap.parse_args()

    collab_run_dir = Path(args.collab_run_dir)
    out_dir = Path(args.out_dir)
    fs = tuple(float(x.strip()) for x in str(args.figsize).split(",", 1))
    if len(fs) != 2:
        raise SystemExit("--figsize must be like '18,12'")

    # Latest A/B categorized LG: choose latest file under each stage dir that includes a scene subdir.
    # Convention: under collab_run_dir/<stage>/<scene>/local_category_graph.json
    a_path = _pick_latest_category_json(collab_run_dir, prefer_stage=args.prefer_stage)
    b_path = _pick_latest_category_json(collab_run_dir, prefer_stage=args.prefer_stage)
    # If both point to the same newest file (possible when only one side exists), try to pick distinct scenes.
    if a_path == b_path:
        stage_dir = (collab_run_dir / args.prefer_stage) if (collab_run_dir / args.prefer_stage).is_dir() else (collab_run_dir / "train_local")
        all_cat = sorted([p for p in stage_dir.rglob("local_category_graph.json") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
        if len(all_cat) >= 2:
            a_path, b_path = all_cat[0], all_cat[1]

    a_snap = _read_json(a_path)
    b_snap = _read_json(b_path)
    a_lg = _build_lg_from_nodes_edges(a_snap)
    b_lg = _build_lg_from_nodes_edges(b_snap)

    a_line = _lg_to_linegraph(a_lg, source="A")
    b_line = _lg_to_linegraph(b_lg, source="B")
    ab = _merge_linegraphs(a_line, b_line)

    out_dir.mkdir(parents=True, exist_ok=True)
    _render_linegraph_to_png(a_line, out_dir / "A_linegraph.png", max_nodes=args.max_nodes, dpi=args.dpi, figsize=fs)
    _render_linegraph_to_png(b_line, out_dir / "B_linegraph.png", max_nodes=args.max_nodes, dpi=args.dpi, figsize=fs)
    _render_linegraph_to_png(ab, out_dir / "AB_linegraph_merged.png", max_nodes=args.max_nodes, dpi=args.dpi, figsize=fs)

    meta = {
        "a_category_json": str(a_path),
        "b_category_json": str(b_path),
        "a_line_nodes": int(a_line.number_of_nodes()),
        "a_line_edges": int(a_line.number_of_edges()),
        "b_line_nodes": int(b_line.number_of_nodes()),
        "b_line_edges": int(b_line.number_of_edges()),
        "ab_line_nodes": int(ab.number_of_nodes()),
        "ab_line_edges": int(ab.number_of_edges()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge_latest_ab_lg_linegraph_viz] wrote PNGs under {out_dir}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

