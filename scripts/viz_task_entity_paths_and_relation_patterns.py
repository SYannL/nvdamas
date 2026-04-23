#!/usr/bin/env python3
"""
Visualize:
1) Graph-1: entity-matched nodes + union of shortest paths between every pair (subgraph),
   with matched nodes and path edges highlighted in thick red.
2) Graph-2: search for paths that match the *relation-sequence patterns* extracted from Graph-1 paths
   (ignore node identity), and highlight matched pattern paths in thick red.

Input graphs are expected to be SelectiveMem categorized LG snapshots:
  local_category_graph.json  (schema: {"nodes":[{"id":...}], "edges":[{"from":...,"to":...,"relations":[...]}]})

Typical usage:
  python scripts/viz_task_entity_paths_and_relation_patterns.py \
    --graph1 logs/.../train_local_gg/kitchen/local_category_graph.json \
    --graph2 logs/.../train_local_gg/bathroom/local_category_graph.json \
    --task "put a candle in cart" \
    --out_dir logs/.../viz_task_pattern
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_cat_lg(snapshot: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Build MultiDiGraph from categorized LG JSON.
    - node id: category string
    - edge attrs: relation(s)
    """
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
        rels = [r for r in rels if r]
        if not rels:
            rels = ["related_to"]
        for rel in rels:
            g.add_edge(u, v, relation=rel)
    return g


STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "in",
    "on",
    "of",
    "with",
    "put",
    "pick",
    "place",
    "take",
    "find",
    "your",
    "task",
    "is",
    "then",
}


def _extract_entities_heuristic(task: str) -> list[str]:
    if not task:
        return []
    tokens = re.findall(r"[a-zA-Z_]+", task.lower())
    out: list[str] = []
    for t in tokens:
        if t in STOP:
            continue
        if t not in out:
            out.append(t)
    return out[:8]


def _match_nodes(g: nx.MultiDiGraph, entities: list[str]) -> list[str]:
    nodes = list(g.nodes())
    matched: list[str] = []
    for ent in entities:
        e = (ent or "").strip().lower()
        if not e:
            continue
        if e in g:
            matched.append(e)
            continue
        for n in nodes:
            ns = str(n).lower()
            nd = g.nodes[n] or {}
            lab = str(nd.get("label") or "").lower()
            cat = str(nd.get("category") or "").lower()
            if e == ns or e in ns or ns in e or (lab and e in lab) or (cat and e in cat):
                matched.append(str(n))
                break
    # unique, keep order
    return list(dict.fromkeys(matched))


def _shortest_path_nodes(g: nx.MultiDiGraph, s: str, t: str, *, max_hops: int) -> list[str] | None:
    und = g.to_undirected()
    try:
        p = nx.shortest_path(und, source=s, target=t)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if len(p) - 1 > max_hops:
        return None
    return [str(x) for x in p]


def _path_edges_and_rel_sequences(g: nx.MultiDiGraph, path_nodes: list[str]) -> tuple[set[tuple[str, str]], list[list[str]]]:
    """
    Return:
    - edges_on_path as (u,v) pairs (direction ignored for highlighting; we store both directions if exists)
    - relation sequences: list of possible relation-seq variants (handles multiedges by branching)
    """
    if len(path_nodes) < 2:
        return set(), []

    edges_und: set[tuple[str, str]] = set()
    seqs: list[list[str]] = [[]]
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        edges_und.add((u, v))
        edges_und.add((v, u))

        rels: list[str] = []
        if g.has_edge(u, v):
            for _k, d in g[u][v].items():
                rel = str((d or {}).get("relation") or "").strip()
                if rel:
                    rels.append(rel)
        if not rels and g.has_edge(v, u):
            # if only reverse exists, still treat as path segment relation
            for _k, d in g[v][u].items():
                rel = str((d or {}).get("relation") or "").strip()
                if rel:
                    rels.append(rel)
        rels = list(dict.fromkeys(rels)) or ["related_to"]

        # branch sequences on multiple relations
        new_seqs: list[list[str]] = []
        for base in seqs:
            for r in rels:
                new_seqs.append(base + [r])
        seqs = new_seqs
        # cap explosion
        if len(seqs) > 64:
            seqs = seqs[:64]

    # unique sequences
    uniq: list[list[str]] = []
    seen = set()
    for s in seqs:
        key = "||".join(s)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return edges_und, uniq


def _edge_rel_matches(d: dict[str, Any], target_rel: str) -> bool:
    rel = str((d or {}).get("relation") or "").strip()
    if rel and rel == target_rel:
        return True
    rels = d.get("relations") or []
    for r in rels:
        if str(r).strip() == target_rel:
            return True
    return False


def _find_paths_by_relation_sequence(
    g: nx.MultiDiGraph,
    rel_seq: list[str],
    *,
    max_matches: int,
) -> list[list[str]]:
    """
    Find node-paths in g whose consecutive edges can realize rel_seq (directed).
    Node identity is not constrained.
    """
    if not rel_seq:
        return []
    matches: list[list[str]] = []

    # Pre-index outgoing edges by relation for faster starts.
    out_by_rel: dict[str, list[tuple[str, str]]] = {}
    for u, v, _k, d in g.edges(keys=True, data=True):
        rel = str((d or {}).get("relation") or "").strip() or "related_to"
        out_by_rel.setdefault(rel, []).append((str(u), str(v)))

    starts = out_by_rel.get(rel_seq[0], [])
    for u0, v0 in starts:
        if len(matches) >= max_matches:
            break
        # path nodes length = len(rel_seq)+1
        base = [u0, v0]

        def dfs(idx: int, curr: str, path: list[str]) -> None:
            if len(matches) >= max_matches:
                return
            if idx >= len(rel_seq):
                matches.append(list(path))
                return
            r = rel_seq[idx]
            # explore outgoing edges with relation r
            if curr not in g:
                return
            for _u, v, _k, d in g.out_edges(curr, keys=True, data=True):
                rel = str((d or {}).get("relation") or "").strip() or "related_to"
                if rel != r:
                    continue
                path.append(str(v))
                dfs(idx + 1, str(v), path)
                path.pop()

        dfs(1, v0, base)
    return matches


def _contiguous_subsequences(seq: list[str], k: int) -> list[list[str]]:
    if k <= 0 or k > len(seq):
        return []
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]


def _choose_one_relation_path_match(
    g: nx.MultiDiGraph,
    rel_sequences: list[list[str]],
    *,
    max_matches_per_seq: int,
    min_len: int = 1,
) -> tuple[list[str], list[str], int]:
    """
    Choose ONE match in graph2 by relation-edge pattern.

    Priority:
    - Try longer patterns first.
    - For a given full pattern, if no full-length match exists, try contiguous subpaths
      of length L-1, L-2, ... down to min_len.

    Returns:
      (matched_relation_sequence_used, matched_path_nodes, num_patterns_with_any_hit)
    """
    # Deduplicate while keeping preference for longer sequences.
    uniq: list[list[str]] = []
    seen = set()
    for s in sorted(rel_sequences, key=lambda x: (-len(x), "||".join(x))):
        key = "||".join(s)
        if key not in seen:
            seen.add(key)
            uniq.append(list(s))

    patterns_with_any_hit = 0
    for full in uniq:
        L = len(full)
        if L < min_len:
            continue
        # Try full length then shorter contiguous windows.
        for k in range(L, min_len - 1, -1):
            for sub in _contiguous_subsequences(full, k):
                hits = _find_paths_by_relation_sequence(
                    g, sub, max_matches=int(max_matches_per_seq)
                )
                if hits:
                    patterns_with_any_hit += 1
                    return list(sub), list(hits[0]), patterns_with_any_hit
        # Even if no hit, we still considered this pattern.
    return [], [], patterns_with_any_hit


def _render_highlighted_subgraph(
    g: nx.MultiDiGraph,
    out_png: Path,
    *,
    highlight_nodes: set[str],
    highlight_edges_und: set[tuple[str, str]],
    title: str,
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    # Build a view subgraph containing all nodes involved in highlighted edges, plus highlight_nodes.
    keep_nodes: set[str] = set(highlight_nodes)
    for u, v in highlight_edges_und:
        keep_nodes.add(str(u))
        keep_nodes.add(str(v))
    g2 = g.subgraph([n for n in g.nodes if str(n) in keep_nodes]).copy()

    # downsample if too big
    if max_nodes > 0 and g2.number_of_nodes() > max_nodes:
        deg = sorted(g2.degree, key=lambda x: x[1], reverse=True)
        keep = {n for n, _d in deg[:max_nodes]}
        g2 = g2.subgraph(keep).copy()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib required") from exc

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(g2, seed=42, k=0.9 / max(1, g2.number_of_nodes()) ** 0.5)

    # nodes
    node_colors = []
    node_edge_colors = []
    node_sizes = []
    labels = {}
    for n, data in g2.nodes(data=True):
        nid = str(n)
        labels[nid] = nid
        if nid in highlight_nodes:
            node_colors.append("#FCE8E6")
            node_edge_colors.append("#D93025")
            node_sizes.append(5200)
        else:
            node_colors.append("#E8F0FE")
            node_edge_colors.append("#5F6368")
            node_sizes.append(3600)

    nx.draw_networkx_nodes(
        g2,
        pos,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        node_size=node_sizes,
        linewidths=2.2,
    )
    nx.draw_networkx_labels(g2, pos, labels=labels, font_size=11)

    # edges
    red_edgelist = []
    gray_edgelist = []
    for u, v, _k, d in g2.edges(keys=True, data=True):
        uu, vv = str(u), str(v)
        if (uu, vv) in highlight_edges_und or (vv, uu) in highlight_edges_und:
            red_edgelist.append((uu, vv))
        else:
            gray_edgelist.append((uu, vv))

    if gray_edgelist:
        nx.draw_networkx_edges(
            g2,
            pos,
            edgelist=gray_edgelist,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.0,
            edge_color="#9AA0A6",
            alpha=0.35,
        )
    if red_edgelist:
        nx.draw_networkx_edges(
            g2,
            pos,
            edgelist=red_edgelist,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            width=3.4,
            edge_color="#D93025",
            alpha=0.95,
        )

    # edge labels (only for highlighted edges, to keep it readable)
    edge_labels: dict[tuple[str, str], str] = {}
    for u, v, _k, d in g2.edges(keys=True, data=True):
        uu, vv = str(u), str(v)
        if (uu, vv) not in highlight_edges_und and (vv, uu) not in highlight_edges_und:
            continue
        rel = str((d or {}).get("relation") or "").strip() or "related_to"
        edge_labels[(uu, vv)] = rel
    if edge_labels:
        nx.draw_networkx_edge_labels(g2, pos, edge_labels=edge_labels, font_size=10, rotate=False, label_pos=0.5)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def _render_full_graph_with_one_path_highlight(
    g: nx.MultiDiGraph,
    out_png: Path,
    *,
    path_nodes: list[str],
    title: str,
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    """
    Render (approximately) the full graph in gray, and highlight ONE directed node-path in thick red.
    If the graph is too large, downsample by degree but always keep the highlighted path nodes.
    """
    path_nodes = [str(x) for x in (path_nodes or [])]
    keep_path = set(path_nodes)

    g2 = g
    if max_nodes > 0 and g.number_of_nodes() > max_nodes:
        deg = sorted(g.degree, key=lambda x: x[1], reverse=True)
        keep = [n for n, _d in deg]
        picked: list[str] = []
        for n in keep:
            if str(n) in keep_path or len(picked) < max_nodes:
                picked.append(str(n))
            if len(set(picked)) >= max_nodes and keep_path.issubset(set(picked)):
                break
        picked_set = set(picked) | keep_path
        g2 = g.subgraph([n for n in g.nodes if str(n) in picked_set]).copy()
    else:
        g2 = g.copy()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib required") from exc

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(g2, seed=42, k=0.9 / max(1, g2.number_of_nodes()) ** 0.5)

    # Base nodes/edges (gray)
    base_node_colors = []
    base_node_edge_colors = []
    base_node_sizes = []
    labels = {}
    for n, _data in g2.nodes(data=True):
        nid = str(n)
        labels[nid] = nid
        if nid in keep_path:
            base_node_colors.append("#FCE8E6")
            base_node_edge_colors.append("#D93025")
            base_node_sizes.append(5200)
        else:
            base_node_colors.append("#E8EAED")
            base_node_edge_colors.append("#9AA0A6")
            base_node_sizes.append(3000)

    nx.draw_networkx_nodes(
        g2,
        pos,
        node_color=base_node_colors,
        edgecolors=base_node_edge_colors,
        node_size=base_node_sizes,
        linewidths=1.8,
    )
    nx.draw_networkx_labels(g2, pos, labels=labels, font_size=10)

    # Draw all edges in gray (directed with arrows)
    all_edges = [(str(u), str(v)) for u, v in g2.edges()]
    if all_edges:
        nx.draw_networkx_edges(
            g2,
            pos,
            edgelist=all_edges,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.0,
            edge_color="#9AA0A6",
            alpha=0.25,
        )

    # Highlight only ONE path (directed)
    red_edges = []
    edge_labels: dict[tuple[str, str], str] = {}
    for i in range(len(path_nodes) - 1):
        u = str(path_nodes[i])
        v = str(path_nodes[i + 1])
        if u not in g2 or v not in g2:
            continue
        red_edges.append((u, v))
        # label relation if edge exists in forward direction; otherwise try reverse (still label)
        rel = ""
        if g2.has_edge(u, v):
            k = next(iter(g2[u][v]))
            rel = str(g2[u][v][k].get("relation") or "").strip()
        if not rel and g2.has_edge(v, u):
            k = next(iter(g2[v][u]))
            rel = str(g2[v][u][k].get("relation") or "").strip()
        if rel:
            edge_labels[(u, v)] = rel

    if red_edges:
        nx.draw_networkx_edges(
            g2,
            pos,
            edgelist=red_edges,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            width=4.0,
            edge_color="#D93025",
            alpha=0.95,
        )
    if edge_labels:
        nx.draw_networkx_edge_labels(
            g2,
            pos,
            edge_labels=edge_labels,
            font_size=10,
            rotate=False,
            label_pos=0.5,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph1", required=True, help="categorized LG json (graph-1)")
    ap.add_argument("--graph2", required=True, help="categorized LG json (graph-2)")
    ap.add_argument("--task", default="", help="task text used for heuristic entity extraction")
    ap.add_argument("--entities", default="", help="comma-separated entities (override task parsing)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_hops", type=int, default=4)
    ap.add_argument("--max_nodes", type=int, default=120)
    ap.add_argument("--max_pattern_matches_per_seq", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--figsize", default="18,12")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g1 = _build_cat_lg(_read_json(Path(args.graph1)))
    g2 = _build_cat_lg(_read_json(Path(args.graph2)))

    if args.entities.strip():
        entities = [x.strip().lower() for x in args.entities.split(",") if x.strip()]
    else:
        entities = _extract_entities_heuristic(args.task)

    matched = _match_nodes(g1, entities)
    if len(matched) < 2:
        raise SystemExit(f"Need >=2 matched nodes in graph1. entities={entities}, matched={matched}")

    # Graph-1: union of shortest paths between every pair.
    highlight_nodes = set(matched)
    highlight_edges_und: set[tuple[str, str]] = set()
    rel_seqs: list[list[str]] = []
    paths_meta: list[dict[str, Any]] = []
    for s, t in combinations(matched, 2):
        p = _shortest_path_nodes(g1, s, t, max_hops=int(args.max_hops))
        if not p:
            continue
        edges_und, seqs = _path_edges_and_rel_sequences(g1, p)
        highlight_edges_und |= edges_und
        rel_seqs.extend(seqs)
        paths_meta.append({"pair": [s, t], "path_nodes": p, "relation_sequences": seqs})

    # unique relation sequences
    uniq_rel_seqs: list[list[str]] = []
    seen = set()
    for s in rel_seqs:
        key = "||".join(s)
        if key not in seen:
            seen.add(key)
            uniq_rel_seqs.append(s)

    _render_highlighted_subgraph(
        g1,
        out_dir / "graph1_entity_shortestpath_highlight.png",
        highlight_nodes=highlight_nodes,
        highlight_edges_und=highlight_edges_und,
        title=f"Graph-1: matched entities {matched} + shortest-path subgraph",
        max_nodes=int(args.max_nodes),
        dpi=int(args.dpi),
        figsize=tuple(float(x.strip()) for x in str(args.figsize).split(",", 1)),
    )

    # Graph-2: find pattern paths matching relation sequences.
    chosen_seq, chosen_path, patterns_with_any_hit = _choose_one_relation_path_match(
        g2,
        uniq_rel_seqs,
        max_matches_per_seq=int(args.max_pattern_matches_per_seq),
        min_len=1,
    )
    if not chosen_path:
        raise SystemExit(
            f"No relation-edge pattern (nor any contiguous subpath) match found in graph2. "
            f"patterns_tried={len(uniq_rel_seqs)}"
        )
    pattern_hits = [{"note": "see chosen_pattern_* fields; this run highlights exactly one matched (sub)path"}]

    _render_full_graph_with_one_path_highlight(
        g2,
        out_dir / "graph2_relationpattern_highlight.png",
        path_nodes=chosen_path,
        title=f"Graph-2: one relation-edge (sub)path highlighted (len={len(chosen_seq)}), patterns_with_any_hit~{patterns_with_any_hit}",
        max_nodes=int(args.max_nodes),
        dpi=int(args.dpi),
        figsize=tuple(float(x.strip()) for x in str(args.figsize).split(",", 1)),
    )

    audit = {
        "graph1": str(Path(args.graph1)),
        "graph2": str(Path(args.graph2)),
        "task": args.task,
        "entities": entities,
        "matched_nodes_graph1": matched,
        "max_hops": int(args.max_hops),
        "shortest_paths": paths_meta,
        "relation_sequences_unique": uniq_rel_seqs,
        "pattern_hits_summary": pattern_hits,
        "chosen_pattern_relation_sequence": chosen_seq,
        "chosen_pattern_path_nodes": chosen_path,
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[viz_task_entity_paths_and_relation_patterns] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

