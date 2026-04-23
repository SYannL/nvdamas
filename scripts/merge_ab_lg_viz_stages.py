#!/usr/bin/env python3
"""
Batch-merge A/B LG snapshots (LG_viz_*.json bundles) across stages/timepoints and visualize.

For a collab run, each side (A/B) writes LG snapshots as JSON bundles:
  LG_viz_<side>_<scene>_epXXXX_<timestamp>.json
format: selectivemem_graph_viz_v1

This script:
- Discovers LG_viz bundles under a stage dir (train_local or train_local_gg).
- Aligns A/B by episode_index_1based.
- Picks several timepoints (default 5): start, 1/4, middle, 3/4, end.
- For each timepoint:
  - Converts instance-level LG snapshot into a *category-level* MultiDiGraph.
  - Merges A/B category graphs (merge nodes by category id; edges by (u,rel,v)).
  - Color-annotates node/edge sources: A-only (blue), B-only (red), shared (green).
  - Writes PNG + merged JSON + audit.

Typical:
  python scripts/merge_ab_lg_viz_stages.py \
    --collab_run_dir logs/alfworld_collab_eval/<run_id>/autogen/memory/selectivemem/gpt-4o-mini \
    --stage train_local \
    --out_dir logs/alfworld_collab_eval/<run_id>/viz_ab_merged_stages_train_local

  python scripts/merge_ab_lg_viz_stages.py \
    --collab_run_dir logs/alfworld_collab_eval/<run_id>/autogen/memory/selectivemem/gpt-4o-mini \
    --stage train_local_gg \
    --out_dir logs/alfworld_collab_eval/<run_id>/viz_ab_merged_stages_train_local_gg
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_lg_viz(stage_dir: Path) -> dict[str, dict[int, Path]]:
    """
    Returns:
      {"A": {ep: path}, "B": {ep: path}}
    If multiple files exist for same side+ep, keep the newest by mtime.
    """
    out: dict[str, dict[int, Path]] = {"A": {}, "B": {}}
    if not stage_dir.is_dir():
        return out
    for p in stage_dir.rglob("LG_viz_*.json"):
        if not p.is_file():
            continue
        try:
            obj = _read_json(p)
            meta = obj.get("meta") or {}
            side = str(meta.get("collab_side") or "").strip().upper()
            ep = meta.get("episode_index_1based")
            if side not in {"A", "B"} or not isinstance(ep, int):
                continue
        except Exception:
            continue
        prev = out[side].get(int(ep))
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            out[side][int(ep)] = p
    return out


def _to_category_multidigraph_from_lg_bundle(bundle: dict[str, Any]) -> nx.MultiDiGraph:
    """
    Input: selectivemem_graph_viz_v1 bundle with graph.nodes/graph.edges in instance-level ids.
    Output: category-level MultiDiGraph with nodes=category strings, edges with relation.
    """
    g = nx.MultiDiGraph()
    graph = bundle.get("graph") or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    inst2cat: dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "").strip()
        if not nid:
            continue
        cat = str(n.get("category") or n.get("label") or nid).strip()
        if not cat:
            continue
        inst2cat[nid] = cat
        if cat not in g:
            g.add_node(cat, category=cat, label=cat)

    for e in edges:
        if not isinstance(e, dict):
            continue
        u = str(e.get("from") or "").strip()
        v = str(e.get("to") or "").strip()
        if not u or not v:
            continue
        cu = inst2cat.get(u)
        cv = inst2cat.get(v)
        if not cu or not cv:
            continue
        rel = str(e.get("relation") or "").strip() or "related_to"
        g.add_edge(cu, cv, relation=rel)
    return g


def _merge_ab(a: nx.MultiDiGraph, b: nx.MultiDiGraph) -> nx.MultiDiGraph:
    out = nx.MultiDiGraph()
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
            for k, v in (data or {}).items():
                out.nodes[nid].setdefault(k, v)
        else:
            out.add_node(nid, **(data or {}), sources={"B"})

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
    shared = {
        n
        for n, d in g.nodes(data=True)
        if isinstance(d.get("sources"), set) and d["sources"] == {"A", "B"}
    }
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
            return "#D2E3FC"
        if src == {"B"}:
            return "#FCE8E6"
        if src == {"A", "B"}:
            return "#E6F4EA"
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

    edges_blue: list[tuple[str, str]] = []
    edges_red: list[tuple[str, str]] = []
    edges_green: list[tuple[str, str]] = []
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

    # edge labels cap
    edge_labels: dict[tuple[str, str], str] = {}
    for u, v, _k, d in g2.edges(keys=True, data=True):
        if len(edge_labels) >= 70:
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


def _pick_timepoints(eps: list[int], k: int) -> list[int]:
    eps = sorted(set(int(x) for x in eps))
    if not eps:
        return []
    if k <= 1:
        return [eps[-1]]
    if len(eps) <= k:
        return eps
    # start, 1/4, 1/2, 3/4, end (for k==5); generalize by evenly spaced quantiles
    idxs = []
    for i in range(k):
        pos = round(i * (len(eps) - 1) / (k - 1))
        idxs.append(int(pos))
    picked = [eps[i] for i in idxs]
    # unique while preserving order
    out: list[int] = []
    seen = set()
    for x in picked:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collab_run_dir", required=True)
    ap.add_argument("--stage", required=True, choices=["train_local", "train_local_gg"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_points", type=int, default=5)
    ap.add_argument("--max_nodes", type=int, default=260)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--figsize", default="20,14")
    args = ap.parse_args()

    collab_run_dir = Path(args.collab_run_dir).resolve()
    stage_dir = collab_run_dir / args.stage
    discovered = _discover_lg_viz(stage_dir)
    a_map = discovered.get("A") or {}
    b_map = discovered.get("B") or {}
    common_eps = sorted(set(a_map.keys()) & set(b_map.keys()))
    if not common_eps:
        raise SystemExit(f"No aligned A/B episodes found under {stage_dir}")

    picked_eps = _pick_timepoints(common_eps, int(args.num_points))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = tuple(float(x.strip()) for x in str(args.figsize).split(",", 1))
    if len(fs) != 2:
        raise SystemExit("--figsize must be like '20,14'")

    summary: dict[str, Any] = {
        "collab_run_dir": str(collab_run_dir),
        "stage": args.stage,
        "aligned_episode_count": len(common_eps),
        "picked_episodes": picked_eps,
        "pairs": [],
    }

    for ep in picked_eps:
        a_path = a_map[int(ep)]
        b_path = b_map[int(ep)]
        a_bundle = _read_json(a_path)
        b_bundle = _read_json(b_path)
        a_cat = _to_category_multidigraph_from_lg_bundle(a_bundle)
        b_cat = _to_category_multidigraph_from_lg_bundle(b_bundle)
        merged = _merge_ab(a_cat, b_cat)

        subdir = out_dir / f"ep{int(ep):04d}"
        subdir.mkdir(parents=True, exist_ok=True)

        # Save merged JSON
        nodes = [
            {
                "id": n,
                **(d or {}),
                "sources": sorted(list((d or {}).get("sources") or [])),
            }
            for n, d in merged.nodes(data=True)
        ]
        edges = []
        for u, v, _k, d in merged.edges(keys=True, data=True):
            dd = dict(d or {})
            src = dd.get("sources")
            if isinstance(src, set):
                dd["sources"] = sorted(list(src))
            edges.append({"from": str(u), "to": str(v), **dd})
        merged_json_path = subdir / "AB_merged_category_graph.json"
        merged_json_path.write_text(
            json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Render
        out_png = subdir / "AB_merged_category_graph.png"
        _render(
            merged,
            out_png,
            max_nodes=int(args.max_nodes),
            dpi=int(args.dpi),
            figsize=fs,
        )

        audit = {
            "episode_index_1based": int(ep),
            "a_lg_viz_path": str(a_path),
            "b_lg_viz_path": str(b_path),
            "a_meta": a_bundle.get("meta") or {},
            "b_meta": b_bundle.get("meta") or {},
            "a_cat_nodes": int(a_cat.number_of_nodes()),
            "b_cat_nodes": int(b_cat.number_of_nodes()),
            "merged_nodes": int(merged.number_of_nodes()),
            "a_cat_edges": int(a_cat.number_of_edges()),
            "b_cat_edges": int(b_cat.number_of_edges()),
            "merged_edges": int(merged.number_of_edges()),
        }
        (subdir / "audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["pairs"].append(audit)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[merge_ab_lg_viz_stages] wrote {len(picked_eps)} merged stage figures under {out_dir}")


if __name__ == "__main__":
    main()

