#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化 `traj.extracted_entities_relations.jsonl` 的三元组 (subject, relation, object)。

输出一个自包含 HTML（使用 vis-network CDN）到同目录，支持交互拖拽、搜索、按 relation 过滤。

典型用法：
  python scripts/alfworld/visualize_traj_triples.py \\
    --input_jsonl logs/alfworld/autogen/memory/selectivemem/gpt-4o-mini/traj.extracted_entities_relations.jsonl \\
    --task_idx 1 --max_edges 400
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


_TRAILING_ID_RE = re.compile(r"^(?P<base>.*?)(?:[_\s]+(?P<num>\d+))$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _strip_trailing_numeric_id(text: str) -> str:
    """
    Normalize node ids/labels by removing trailing numeric identifiers.
    Examples:
      - "Fridge_1" -> "Fridge"
      - "countertop 2" -> "countertop"
    """
    s = str(text or "").strip()
    if not s:
        return s
    m = _TRAILING_ID_RE.match(s)
    if not m:
        return s
    base = (m.group("base") or "").strip()
    return base if base else s


def _canonical_key(text: str) -> str:
    """
    Canonicalize ids/labels/categories for merging & coloring.
    - strip trailing numeric id
    - lowercase
    - normalize separators / remove non-alnum
    Examples:
      - "SinkBasin_1" / "sinkbasin 2" -> "sinkbasin"
      - "PaperTowelRoll_1" / "papertowelroll 1" -> "papertowelroll"
    """
    s = _strip_trailing_numeric_id(text)
    s = s.strip().lower()
    s = _NON_ALNUM_RE.sub("", s)
    return s


def _pretty_label(text: str) -> str:
    """Human-friendly label from canonical key."""
    ck = _canonical_key(text)
    return ck if not ck else ck  # keep compact; UI tooltip carries full meta


def _node_title(meta: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in ("instance_id", "label", "category", "state", "room"):
        v = meta.get(k)
        if v is None or str(v).strip() == "":
            continue
        parts.append(f"{k}: {v}")
    return "\n".join(parts) if parts else ""


def _color_for_category(cat: str) -> str:
    # Deterministic palette by hash (no extra deps).
    palette = [
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ab",
    ]
    if not cat:
        return "#999999"
    h = 0
    for ch in cat:
        h = (h * 131 + ord(ch)) % 10_000
    return palette[h % len(palette)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize extracted SelectiveMem triples (HTML).")
    ap.add_argument(
        "--input_jsonl",
        type=str,
        default="logs/alfworld/autogen/memory/selectivemem/gpt-4o-mini/traj.extracted_entities_relations.jsonl",
        help="Path to traj.extracted_entities_relations.jsonl",
    )
    ap.add_argument("--output_html", type=str, default=None, help="Output HTML path (default: alongside input).")
    ap.add_argument("--task_idx", type=int, default=None, help="Filter by task_idx (as in JSONL).")
    ap.add_argument("--max_edges", type=int, default=800, help="Max number of edges to include (default: 800).")
    ap.add_argument(
        "--include_relations",
        type=str,
        default="",
        help="Comma-separated relation whitelist, e.g. 'TakeFrom,PutIn,Open,Cool,Heat,Use,On,Inside,Contains'.",
    )
    ap.add_argument("--hide_event_nodes", action="store_true", help="Hide nodes whose id starts with 'Event_'.")
    ap.add_argument(
        "--include_isolated_nodes",
        action="store_true",
        default=True,
        help="Include nodes that appear in instances even if no edges (default: on).",
    )
    ap.add_argument(
        "--no_include_isolated_nodes",
        action="store_false",
        dest="include_isolated_nodes",
        help="Only include nodes that appear in kept edges.",
    )
    ap.add_argument(
        "--strip_numeric_ids",
        action="store_true",
        default=True,
        help="Strip trailing numeric ids from node ids/labels (default: on).",
    )
    ap.add_argument(
        "--no_strip_numeric_ids",
        action="store_false",
        dest="strip_numeric_ids",
        help="Disable stripping trailing numeric ids.",
    )
    args = ap.parse_args()

    in_path = Path(args.input_jsonl).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"input_jsonl not found: {in_path}")

    out_path = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html
        else in_path.with_suffix("").with_suffix(".triples.html")
    )
    os.makedirs(out_path.parent, exist_ok=True)

    rows = _read_jsonl(in_path)
    if args.task_idx is not None:
        rows = [r for r in rows if int(r.get("task_idx", -1)) == int(args.task_idx)]

    rel_allow: set[str] = set()
    if args.include_relations.strip():
        rel_allow = {x.strip() for x in args.include_relations.split(",") if x.strip()}

    # Build nodes from instances + relation endpoints (normalize ids/categories for merging & consistent colors).
    node_meta: dict[str, dict[str, Any]] = {}  # key: canonical node id
    edges_raw: list[dict[str, Any]] = []
    rel_counter: Counter[str] = Counter()

    for r in rows:
        parsed = r.get("parsed") or {}
        instances = parsed.get("instances") or []
        relations = parsed.get("relations") or []
        if isinstance(instances, list):
            for inst in instances:
                if not isinstance(inst, dict):
                    continue
                nid0 = str(inst.get("instance_id") or "").strip()
                nid = _canonical_key(nid0) if args.strip_numeric_ids else str(nid0).strip()
                if not str(nid).strip():
                    continue
                if args.hide_event_nodes and str(nid0).startswith("Event_"):
                    continue
                if nid not in node_meta:
                    node_meta[nid] = dict(inst)
                    node_meta[nid]["instance_id"] = nid
                    # Keep original id as hint if we normalized.
                    if nid0:
                        node_meta[nid]["original_ids"] = [nid0]
                else:
                    # Merge provenance of original ids
                    if nid0:
                        lst = node_meta[nid].get("original_ids")
                        if isinstance(lst, list) and nid0 not in lst:
                            lst.append(nid0)
                # Normalize category for coloring/grouping
                cat0 = str(inst.get("category") or "").strip()
                if cat0:
                    node_meta[nid].setdefault("original_categories", [])
                    oc = node_meta[nid].get("original_categories")
                    if isinstance(oc, list) and cat0 not in oc:
                        oc.append(cat0)
                    node_meta[nid]["category_canon"] = _canonical_key(cat0) if args.strip_numeric_ids else cat0
        if isinstance(relations, list):
            for rel in relations:
                if not isinstance(rel, dict):
                    continue
                s0 = str(rel.get("subject") or "").strip()
                o0 = str(rel.get("object") or "").strip()
                s = _canonical_key(s0) if args.strip_numeric_ids else s0
                o = _canonical_key(o0) if args.strip_numeric_ids else o0
                lab = str(rel.get("relation") or "").strip()
                if not s or not o or not lab:
                    continue
                if rel_allow and lab not in rel_allow:
                    continue
                if args.hide_event_nodes and (s0.startswith("Event_") or o0.startswith("Event_")):
                    continue
                rel_counter[lab] += 1
                edges_raw.append(
                    {
                        "from": s,
                        "to": o,
                        "label": lab,
                        "title": f"relation: {lab}\\nfrom: {s}\\nto: {o}\\n(task_idx={r.get('task_idx')}, step={r.get('step')})",
                        "task_idx": r.get("task_idx"),
                        "step": r.get("step"),
                    }
                )
                if s not in node_meta:
                    node_meta[s] = {"instance_id": s, "original_ids": [s0] if s0 else []}
                if o not in node_meta:
                    node_meta[o] = {"instance_id": o, "original_ids": [o0] if o0 else []}

    # De-duplicate edges after normalization: same (from,to,label) may appear many times.
    # Keep a count + a few example occurrences for tooltip.
    edge_bucket: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in edges_raw:
        k = (str(e.get("from")), str(e.get("to")), str(e.get("label")))
        if k not in edge_bucket:
            edge_bucket[k] = {
                "from": k[0],
                "to": k[1],
                "label": k[2],
                "count": 1,
                "examples": [(e.get("task_idx"), e.get("step"))],
            }
        else:
            edge_bucket[k]["count"] += 1
            ex = edge_bucket[k].get("examples")
            if isinstance(ex, list) and len(ex) < 8:
                ex.append((e.get("task_idx"), e.get("step")))

    edges: list[dict[str, Any]] = []
    for (_f, _t, _lab), agg in edge_bucket.items():
        ex = agg.get("examples") or []
        ex_txt = ", ".join([f"(task={a}, step={b})" for a, b in ex if a is not None and b is not None])
        title = f"relation: {agg['label']}\\nfrom: {agg['from']}\\nto: {agg['to']}\\ncount: {agg['count']}"
        if ex_txt:
            title += f"\\nexamples: {ex_txt}"
        edges.append(
            {
                "from": agg["from"],
                "to": agg["to"],
                "label": f"{agg['label']} ({agg['count']})" if int(agg["count"]) > 1 else agg["label"],
                "title": title,
                "value": int(agg["count"]),
            }
        )

    # Limit edges (keep most frequent relations first to preserve structure).
    if args.max_edges is not None and len(edges) > int(args.max_edges):
        edges.sort(
            key=lambda e: (
                -int(e.get("value") or 1),
                str(e.get("label") or ""),
                str(e.get("from") or ""),
                str(e.get("to") or ""),
            )
        )
        edges = edges[: int(args.max_edges)]

    # Nodes that appear in edges.
    used_nodes: set[str] = set()
    for e in edges:
        used_nodes.add(str(e["from"]))
        used_nodes.add(str(e["to"]))

    # Optionally keep isolated nodes (instances without relations).
    if args.include_isolated_nodes:
        for nid in node_meta.keys():
            used_nodes.add(str(nid))

    nodes: list[dict[str, Any]] = []
    for nid in sorted(used_nodes):
        meta = node_meta.get(nid) or {"instance_id": nid}
        cat_canon = str(meta.get("category_canon") or meta.get("category") or "")
        label0 = str(meta.get("label") or nid)
        label = _pretty_label(label0) if args.strip_numeric_ids else (label0 or nid)
        if not label.strip():
            label = str(nid)
        nodes.append(
            {
                "id": nid,
                "label": label,
                "title": _node_title(meta),
                "group": cat_canon or "unknown",
                "color": _color_for_category(cat_canon),
            }
        )

    rel_list = [k for k, _v in rel_counter.most_common()]

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>NV_DAMAS Triples Graph</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
    #topbar {{ padding: 10px 12px; border-bottom: 1px solid #ddd; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    #network {{ width: 100%; height: calc(100vh - 58px); }}
    .label {{ font-size: 12px; color: #333; }}
    select, input {{ padding: 6px 8px; }}
    .stat {{ font-size: 12px; color: #666; }}
  </style>
</head>
<body>
  <div id="topbar">
    <span class="label"><b>关系过滤</b></span>
    <select id="relFilter">
      <option value="">(全部)</option>
    </select>
    <span class="label"><b>节点搜索</b></span>
    <input id="nodeSearch" placeholder="输入 instance_id 或 label..." size="28"/>
    <button id="btnFocus">定位</button>
    <span class="stat" id="stats"></span>
  </div>
  <div id="network"></div>

  <script>
    const NODES = {json.dumps(nodes, ensure_ascii=False)};
    const EDGES_ALL = {json.dumps(edges, ensure_ascii=False)};
    const RELS = {json.dumps(rel_list, ensure_ascii=False)};

    const nodes = new vis.DataSet(NODES);
    let edges = new vis.DataSet(EDGES_ALL);
    const container = document.getElementById('network');

    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      nodes: {{
        shape: 'dot',
        size: 10,
        font: {{ size: 12 }},
        borderWidth: 1
      }},
      edges: {{
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
        font: {{ align: 'middle' }},
        smooth: {{ type: 'dynamic' }},
        color: {{ color: '#888' }}
      }},
      physics: {{
        stabilization: false,
        barnesHut: {{ gravitationalConstant: -25000, springLength: 120, springConstant: 0.02 }}
      }},
      interaction: {{
        hover: true,
        navigationButtons: true,
        keyboard: true
      }}
    }};
    const network = new vis.Network(container, data, options);

    const relFilter = document.getElementById('relFilter');
    for (const r of RELS) {{
      const opt = document.createElement('option');
      opt.value = r;
      opt.textContent = r;
      relFilter.appendChild(opt);
    }}

    function applyRelationFilter() {{
      const rel = relFilter.value;
      const filtered = rel ? EDGES_ALL.filter(e => e.label === rel) : EDGES_ALL;
      edges.clear();
      edges.add(filtered);
      document.getElementById('stats').textContent =
        `nodes=${{nodes.length}}  edges=${{filtered.length}}` + (rel ? `  (relation=${{rel}})` : '');
    }}
    relFilter.addEventListener('change', applyRelationFilter);
    applyRelationFilter();

    document.getElementById('btnFocus').addEventListener('click', () => {{
      const q = (document.getElementById('nodeSearch').value || '').trim().toLowerCase();
      if (!q) return;
      const all = nodes.get();
      let hit = null;
      for (const n of all) {{
        const id = (n.id || '').toLowerCase();
        const label = (n.label || '').toLowerCase();
        if (id.includes(q) || label.includes(q)) {{
          hit = n;
          break;
        }}
      }}
      if (!hit) return;
      network.selectNodes([hit.id]);
      network.focus(hit.id, {{ scale: 1.4, animation: true }});
    }});
  </script>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote: {out_path}")
    print(f"Included nodes={len(nodes)}, edges={len(edges)} (input rows={len(rows)})")
    if rel_counter:
        top = ", ".join([f"{k}:{v}" for k, v in rel_counter.most_common(12)])
        print(f"Top relations: {top}")


if __name__ == "__main__":
    main()

