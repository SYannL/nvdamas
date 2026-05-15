#!/usr/bin/env python3
"""Render real GraphMemory3 local/global snapshots for a PDDL run.

The output is intentionally compact: the persisted local graphs are large, so
the local domain figures show true counts plus the highest-support procedural
rules instead of all nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


DOMAINS = ("gripper", "blockworld", "barman", "tyreworld")


PALETTE = {
    "blue": "#1f5aa6",
    "blue_fill": "#eef6ff",
    "green": "#2f7d32",
    "green_fill": "#eff9ef",
    "purple": "#6a2ca0",
    "purple_fill": "#f6efff",
    "orange": "#d95f02",
    "orange_fill": "#fff4e8",
    "red": "#c62828",
    "red_fill": "#fff0f0",
    "gray": "#5f6368",
    "gray_fill": "#f7f7f7",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def dot_id(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def wrap_text(text: str, width: int = 54) -> str:
    words = str(text).replace("\n", " ").split()
    lines: list[str] = []
    line: list[str] = []
    size = 0
    for word in words:
        extra = len(word) + (1 if line else 0)
        if line and size + extra > width:
            lines.append(" ".join(line))
            line = [word]
            size = len(word)
        else:
            line.append(word)
            size += extra
    if line:
        lines.append(" ".join(line))
    return "<BR/>".join(esc(x) for x in lines)


def wrap_plain(text: str, width: int = 54) -> str:
    words = str(text).replace("\n", " ").split()
    lines: list[str] = []
    line: list[str] = []
    size = 0
    for word in words:
        extra = len(word) + (1 if line else 0)
        if line and size + extra > width:
            lines.append(" ".join(line))
            line = [word]
            size = len(word)
        else:
            line.append(word)
            size += extra
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def quote_label(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def html_table(title: str, rows: list[tuple[str, Any]], color: str) -> str:
    border = PALETTE[color]
    fill = PALETTE[f"{color}_fill"]
    body = [
        f'<TR><TD BGCOLOR="{fill}" COLSPAN="2"><B>{esc(title)}</B></TD></TR>',
    ]
    for key, value in rows:
        body.append(
            f'<TR><TD ALIGN="LEFT"><B>{esc(key)}</B></TD>'
            f'<TD ALIGN="LEFT">{esc(value)}</TD></TR>'
        )
    return (
        f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="6" '
        f'COLOR="{border}">{"".join(body)}</TABLE>>'
    )


def rule_score(rule: dict[str, Any]) -> tuple[float, float, float]:
    stats = rule.get("stats", {})
    return (
        float(stats.get("support", 0) or 0),
        float(stats.get("utility", 0) or 0),
        float(stats.get("confidence", 0) or 0),
    )


def top_rules(rules: list[dict[str, Any]], n: int = 8) -> list[dict[str, Any]]:
    return sorted(rules, key=rule_score, reverse=True)[:n]


def edge_label(rule: dict[str, Any]) -> str:
    effect = rule.get("effect", {})
    stats = rule.get("stats", {})
    action = effect.get("surface_action") or effect.get("prefer_action") or effect.get("via") or "action"
    support = stats.get("support", 0)
    confidence = stats.get("confidence", 0)
    return f"{action}\\nsupport={support}, conf={confidence:.2f}"


def workflow_endpoints(rule: dict[str, Any]) -> tuple[str, str]:
    condition = rule.get("condition", {})
    effect = rule.get("effect", {})
    start = condition.get("from") or rule.get("progress_state") or "stage"
    end = effect.get("to") or rule.get("progress_state") or "stage"
    return str(start), str(end)


def load_domain(base: Path, domain: str) -> dict[str, Any]:
    p = base / "local" / domain / "graph_memory3" / f"local_{domain}.json"
    return load_json(p)


def episode_counts(base: Path, domain: str) -> tuple[int, int, int]:
    path = base / "local" / domain / "graph_memory3" / "episodes.jsonl"
    total = success = failure = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            label = json.loads(line).get("label")
            if label is True:
                success += 1
            elif label is False:
                failure += 1
    return total, success, failure


def domain_summary(base: Path, domain: str) -> dict[str, Any]:
    data = load_domain(base, domain)
    total, success, failure = episode_counts(base, domain)
    return {
        "data": data,
        "episodes": total,
        "success": success,
        "failure": failure,
        "nodes": Counter(node.get("node_type") for node in data.get("nodes", [])),
        "edges": Counter(edge.get("edge_type") for edge in data.get("edges", [])),
        "candidates": Counter(c.get("candidate_type") for c in data.get("candidates", [])),
        "rules": Counter(r.get("rule_type") for r in data.get("rules", [])),
        "artifacts": Counter(a.get("kind") for a in data.get("artifacts", [])),
    }


def render_with_dot(dot_path: Path) -> None:
    for fmt in ("svg", "png"):
        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(dot_path.with_suffix(f".{fmt}"))],
            check=True,
        )


def render_graphviz(dot_path: Path, *, engine: str = "dot", formats: tuple[str, ...] = ("svg", "png")) -> None:
    for fmt in formats:
        subprocess.run(
            [engine, f"-T{fmt}", str(dot_path), "-o", str(dot_path.with_suffix(f".{fmt}"))],
            check=True,
        )


def node_ref(raw_id: str) -> str:
    return "n_" + hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]


def compact_node_label(node: dict[str, Any]) -> str:
    node_type = node.get("node_type", "node")
    payload = node.get("payload", {}) or {}
    stats = node.get("stats", {}) or {}
    if node_type == "action":
        text = payload.get("surface_form") or node.get("signature") or node.get("node_id")
    elif node_type == "state":
        stage = payload.get("workflow_stage") or "state"
        support = stats.get("support", 0)
        confidence = stats.get("confidence", 0)
        text = f"{stage}\\ns={support}, c={confidence:.2f}"
    elif node_type == "subgoal":
        text = payload.get("summary") or payload.get("goal") or node.get("signature") or "subgoal"
    elif node_type == "failure":
        text = payload.get("failure_label") or payload.get("summary") or node.get("signature") or "failure"
    else:
        text = node.get("signature") or node.get("node_id") or node_type
    return wrap_plain(str(text), 24)


def full_tooltip(obj: dict[str, Any], max_len: int = 1200) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return quote_label(text)


def svg_title(obj: dict[str, Any], max_len: int = 1800) -> str:
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return esc(text)


def write_full_domain_network_svg(base: Path, out_dir: Path, domain: str) -> None:
    data = load_domain(base, domain)
    summary = domain_summary(base, domain)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_by_id = {str(node.get("node_id")): node for node in nodes}
    all_ids: set[str] = set(node_by_id)
    for edge in edges:
        all_ids.add(str(edge.get("src")))
        all_ids.add(str(edge.get("dst")))

    stage_order = {
        "initial_planning": 0,
        "search_preconditions": 2,
        "advance_goal_literals": 4,
        "goal_satisfied": 6,
    }

    state_col: dict[str, int] = {}
    for raw_id, node in node_by_id.items():
        if node.get("node_type") == "state":
            stage = ((node.get("payload") or {}).get("workflow_stage") or "unknown")
            state_col[raw_id] = stage_order.get(str(stage), 1)

    neighbor_cols: dict[str, list[int]] = {raw_id: [] for raw_id in all_ids}
    for edge in edges:
        src = str(edge.get("src"))
        dst = str(edge.get("dst"))
        if src in state_col:
            neighbor_cols.setdefault(dst, []).append(state_col[src])
        if dst in state_col:
            neighbor_cols.setdefault(src, []).append(state_col[dst])

    buckets: dict[int, list[str]] = {}
    for raw_id in sorted(all_ids):
        node = node_by_id.get(raw_id, {"node_id": raw_id, "node_type": "missing_ref"})
        node_type = node.get("node_type")
        if node_type == "state":
            col = state_col.get(raw_id, 1)
        elif node_type == "action":
            cols = neighbor_cols.get(raw_id) or [1]
            col = min(7, max(1, int(round(sum(cols) / len(cols))) + 1))
        elif node_type == "subgoal":
            col = 7
        elif node_type == "failure":
            col = 8
        else:
            col = 9
        buckets.setdefault(col, []).append(raw_id)

    col_gap = 340
    row_gap = 24
    pad_x = 120
    pad_y = 150
    max_col = max(buckets) if buckets else 1
    max_bucket = max((len(v) for v in buckets.values()), default=1)
    width = pad_x * 2 + (max_col + 1) * col_gap
    height = max(1200, pad_y * 2 + max_bucket * row_gap)
    pos: dict[str, tuple[float, float]] = {}
    for col, raw_ids in buckets.items():
        raw_ids.sort(
            key=lambda raw_id: (
                str(node_by_id.get(raw_id, {}).get("node_type", "z")),
                compact_node_label(node_by_id.get(raw_id, {"node_id": raw_id})),
                raw_id,
            )
        )
        for idx, raw_id in enumerate(raw_ids):
            x = pad_x + col * col_gap
            y = pad_y + idx * row_gap
            pos[raw_id] = (x, y)

    def xy(raw_id: str) -> tuple[float, float]:
        return pos.get(raw_id, (pad_x, pad_y))

    node_color = {
        "state": ("#1f5aa6", "#eef6ff"),
        "action": ("#b36b00", "#fff7df"),
        "subgoal": ("#2f7d32", "#eff9ef"),
        "failure": ("#c62828", "#fff0f0"),
    }
    edge_color = {
        "temporal": "#333333",
        "causes": "#111111",
        "advances_to": "#2f7d32",
        "fails_under": "#c62828",
    }
    edge_dash = {"fails_under": " stroke-dasharray='6 4'"}

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".legend { font-size: 14px; }",
        ".node-label { font-size: 8px; pointer-events: none; }",
        ".edge { opacity: 0.22; }",
        ".node { opacity: 0.94; }",
        "</style>",
        "<defs>",
    ]
    for name, color in edge_color.items():
        out.append(
            f'<marker id="arrow-{name}" markerWidth="6" markerHeight="6" refX="5" refY="3" '
            f'orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L0,6 L6,3 z" fill="{color}" /></marker>'
        )
    out.extend(
        [
            "</defs>",
            '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
            f'<text class="title" x="24" y="38">FULL Local Graph Memory: {esc(domain)}</text>',
            f'<text class="legend" x="24" y="64">{data.get("node_count")} nodes, '
            f'{data.get("edge_count")} edges; {summary["episodes"]} episodes '
            f'({summary["success"]} success / {summary["failure"]} fail). '
            f'Hover nodes/edges for original JSON.</text>',
            '<g class="legend" transform="translate(24,86)">',
            '<circle cx="8" cy="8" r="7" fill="#eef6ff" stroke="#1f5aa6"/><text x="22" y="12">state</text>',
            '<rect x="92" y="1" width="18" height="14" rx="3" fill="#fff7df" stroke="#b36b00"/><text x="118" y="12">action</text>',
            '<circle cx="196" cy="8" r="7" fill="#eff9ef" stroke="#2f7d32"/><text x="210" y="12">subgoal</text>',
            '<polygon points="306,1 314,1 321,8 314,15 306,15 299,8" fill="#fff0f0" stroke="#c62828"/><text x="330" y="12">failure</text>',
            '<line x1="420" y1="8" x2="470" y2="8" stroke="#333333" marker-end="url(#arrow-temporal)"/><text x="478" y="12">temporal</text>',
            '<line x1="570" y1="8" x2="620" y2="8" stroke="#2f7d32" marker-end="url(#arrow-advances_to)"/><text x="628" y="12">advances_to</text>',
            '<line x1="750" y1="8" x2="800" y2="8" stroke="#c62828" stroke-dasharray="6 4" marker-end="url(#arrow-fails_under)"/><text x="808" y="12">fails_under</text>',
            "</g>",
            '<g id="edges">',
        ]
    )

    for edge in edges:
        src = str(edge.get("src"))
        dst = str(edge.get("dst"))
        x1, y1 = xy(src)
        x2, y2 = xy(dst)
        edge_type = str(edge.get("edge_type"))
        color = edge_color.get(edge_type, "#777777")
        marker = f"arrow-{edge_type}" if edge_type in edge_color else "arrow-temporal"
        dash = edge_dash.get(edge_type, "")
        out.append(
            f'<line class="edge" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="0.8"{dash} marker-end="url(#{marker})">'
            f"<title>{svg_title(edge)}</title></line>"
        )
    out.append("</g>")
    out.append('<g id="nodes">')

    for raw_id in sorted(all_ids):
        node = node_by_id.get(raw_id, {"node_id": raw_id, "node_type": "missing_ref"})
        node_type = str(node.get("node_type"))
        stroke, fill = node_color.get(node_type, ("#777777", "#f7f7f7"))
        x, y = xy(raw_id)
        tooltip = svg_title(node)
        label = compact_node_label(node).split("\n")[0]
        label = label[:34] + ("..." if len(label) > 34 else "")
        out.append(f'<g class="node"><title>{tooltip}</title>')
        if node_type == "action":
            out.append(
                f'<rect x="{x - 11:.2f}" y="{y - 7:.2f}" width="22" height="14" rx="3" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            )
        elif node_type == "failure":
            pts = [
                (x - 7, y - 10),
                (x + 7, y - 10),
                (x + 12, y),
                (x + 7, y + 10),
                (x - 7, y + 10),
                (x - 12, y),
            ]
            out.append(
                '<polygon points="'
                + " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
                + f'" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            )
        else:
            radius = 5.5 if node_type == "state" else 7.5
            out.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            )
        # Label only action/subgoal/failure nodes; state labels make the full graph unreadable.
        if node_type != "state":
            out.append(
                f'<text class="node-label" x="{x + 8:.2f}" y="{y - 8:.2f}" fill="#222">{esc(label)}</text>'
            )
        out.append("</g>")

    out.extend(["</g>", "</svg>"])
    svg_path = out_dir / f"pddl_gm3_full_local_{domain}.svg"
    svg_path.write_text("\n".join(out), encoding="utf-8")


def write_full_domain(base: Path, out_dir: Path, domain: str, *, full_png: bool = False) -> None:
    data = load_domain(base, domain)
    summary = domain_summary(base, domain)
    node_style = {
        "state": ("ellipse", PALETTE["blue"], PALETTE["blue_fill"]),
        "action": ("box", "#b36b00", "#fff7df"),
        "subgoal": ("ellipse", PALETTE["green"], PALETTE["green_fill"]),
        "failure": ("octagon", PALETTE["red"], PALETTE["red_fill"]),
    }
    edge_style = {
        "temporal": ("#333333", "solid", "T"),
        "causes": ("#111111", "solid", "C"),
        "advances_to": (PALETTE["green"], "solid", "A"),
        "fails_under": (PALETTE["red"], "dashed", "F"),
    }

    lines = [
        "digraph G {",
        '  graph [layout=sfdp, overlap=false, splines=true, bgcolor="white", pad="0.3", outputorder=edgesfirst];',
        '  node [fontname="Helvetica", fontsize=8, margin="0.05,0.03"];',
        '  edge [fontname="Helvetica", fontsize=7, arrowsize=0.45, penwidth=0.7];',
        f'  label="FULL Local Graph Memory: {esc(domain)} | '
        f'{data.get("node_count")} nodes, {data.get("edge_count")} edges | '
        f'{summary["episodes"]} episodes ({summary["success"]} success / {summary["failure"]} fail)";',
        '  labelloc="t";',
        '  legend [shape=plain, label=<',
        '    <TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5">',
        '      <TR><TD BGCOLOR="#f7f7f7"><B>Legend</B></TD></TR>',
        '      <TR><TD ALIGN="LEFT">blue ellipse = state</TD></TR>',
        '      <TR><TD ALIGN="LEFT">orange box = action</TD></TR>',
        '      <TR><TD ALIGN="LEFT">green ellipse = subgoal</TD></TR>',
        '      <TR><TD ALIGN="LEFT">red octagon = failure</TD></TR>',
        '      <TR><TD ALIGN="LEFT">T temporal, C causes, A advances_to, F fails_under</TD></TR>',
        '    </TABLE>',
        '  >];',
    ]

    known_node_ids = {node.get("node_id") for node in data.get("nodes", [])}
    for node in data.get("nodes", []):
        raw_id = str(node.get("node_id"))
        shape, color, fill = node_style.get(node.get("node_type"), ("ellipse", PALETTE["gray"], PALETTE["gray_fill"]))
        lines.append(
            f'  {node_ref(raw_id)} [shape={shape}, style="filled", '
            f'color="{color}", fillcolor="{fill}", '
            f'label="{quote_label(compact_node_label(node))}", tooltip="{full_tooltip(node)}"];'
        )

    missing_refs: set[str] = set()
    for idx, edge in enumerate(data.get("edges", [])):
        src = str(edge.get("src"))
        dst = str(edge.get("dst"))
        if src not in known_node_ids:
            missing_refs.add(src)
        if dst not in known_node_ids:
            missing_refs.add(dst)
        color, style, label = edge_style.get(edge.get("edge_type"), (PALETTE["gray"], "solid", str(edge.get("edge_type"))))
        stats = edge.get("stats", {}) or {}
        edge_label_text = f"{label}\\ns={stats.get('support', 0)}"
        lines.append(
            f'  {node_ref(src)} -> {node_ref(dst)} [color="{color}", style="{style}", '
            f'label="{quote_label(edge_label_text)}", tooltip="{full_tooltip(edge)}"];'
        )

    for raw_id in sorted(missing_refs):
        lines.append(
            f'  {node_ref(raw_id)} [shape=diamond, style="filled", color="{PALETTE["gray"]}", '
            f'fillcolor="{PALETTE["gray_fill"]}", label="{quote_label(wrap_plain(raw_id, 24))}", '
            f'tooltip="{quote_label(raw_id)}"];'
        )

    lines.append("}")
    dot_path = out_dir / f"pddl_gm3_full_local_{domain}.dot"
    dot_path.write_text("\n".join(lines), encoding="utf-8")
    write_full_domain_network_svg(base, out_dir, domain)
    if full_png:
        render_graphviz(dot_path, engine="sfdp", formats=("png",))


def _goal_satisfied_count(record: dict[str, Any], goal_literals: list[str]) -> int:
    current = {_normalize_for_sample(x) for x in record.get("Current Literals", []) or []}
    return sum(1 for item in goal_literals if _normalize_for_sample(item) in current)


def _normalize_for_sample(text: Any) -> str:
    return " ".join(str(text).lower().replace(".", " ").replace("_", " ").split())


def _sample_progress(record: dict[str, Any], goal_literals: list[str]) -> str:
    done = bool(record.get("Done", False))
    satisfied = _goal_satisfied_count(record, goal_literals)
    remaining = max(len(goal_literals) - satisfied, 0)
    step = int(record.get("Step", 0) or 0)
    if done or remaining <= 0:
        return "goal_satisfied"
    if satisfied <= 0 and step <= 0:
        return "initial_planning"
    if satisfied <= 0:
        return "search_preconditions"
    return "advance_goal_literals"


def _sample_delta(prev: dict[str, Any], cur: dict[str, Any]) -> tuple[list[str], list[str]]:
    prev_lits = {_normalize_for_sample(x): str(x) for x in prev.get("Current Literals", []) or []}
    cur_lits = {_normalize_for_sample(x): str(x) for x in cur.get("Current Literals", []) or []}
    added = [cur_lits[k] for k in sorted(cur_lits.keys() - prev_lits.keys())]
    removed = [prev_lits[k] for k in sorted(prev_lits.keys() - cur_lits.keys())]
    return added[:4], removed[:4]


def _find_rule(data: dict[str, Any], *, rule_type: str, action: str, stage: str) -> dict[str, Any] | None:
    action_l = action.lower()
    for rule in data.get("rules", []):
        if rule.get("rule_type") != rule_type:
            continue
        if str(rule.get("progress_state")) != stage:
            continue
        if action_l in str(rule.get("summary", "")).lower():
            return rule
    return None


def _svg_text_lines(lines: list[str], x: int, y: int, *, size: int = 13, fill: str = "#222") -> list[str]:
    out = []
    for i, line in enumerate(lines):
        out.append(f'<text x="{x}" y="{y + i * (size + 5)}" font-size="{size}" fill="{fill}">{esc(line)}</text>')
    return out


def write_sample_construction_retrieval(base: Path, out_dir: Path) -> None:
    """A small real-data teaching diagram from gripper-0."""
    history_path = base / "local" / "gripper" / "graph_memory3" / "dynamic_histories" / "history_gripper_0_success.json"
    local_path = base / "local" / "gripper" / "graph_memory3" / "local_gripper.json"
    global_path = base / "global" / "graph_memory3" / "global_memory.json"
    payload = load_json(history_path)
    local_data = load_json(local_path)
    global_data = load_json(global_path)
    history = payload.get("history", [])
    goal_literals = payload.get("goal_literals", []) or []
    shown_steps = history[:6]  # initial + first five real actions

    width, height = 2200, 1250
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; }",
        ".title { font-size: 25px; font-weight: 700; }",
        ".h { font-size: 18px; font-weight: 700; }",
        ".small { font-size: 12px; }",
        ".box { fill: white; stroke-width: 1.5; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        '<text class="title" x="30" y="42">Real GM3 Example: how one gripper episode becomes graph memory and retrieval hints</text>',
        '<text x="30" y="70" font-size="14">Source: dynamic_histories/history_gripper_0_success.json + local_gripper.json + global_memory.json</text>',
    ]

    # Panel frames.
    panels = [
        (25, 95, 970, 520, "#1f5aa6", "1. Episode graph from one true trajectory"),
        (1025, 95, 540, 520, "#2f7d32", "2. Merged into local graph memory"),
        (1595, 95, 575, 520, "#6a2ca0", "3. Retrieve on a later query"),
        (25, 650, 2145, 540, "#d95f02", "4. What the agent receives"),
    ]
    for x, y, w, h, color, title in panels:
        out.append(f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="10" stroke="{color}"/>')
        out.append(f'<text class="h" x="{x + 16}" y="{y + 30}" fill="{color}">{esc(title)}</text>')

    # Episode graph: states/actions/subgoals/deltas for the first five actions.
    sx, sy = 75, 180
    state_positions: list[tuple[int, int]] = []
    for idx, rec in enumerate(shown_steps):
        x = sx + idx * 170
        y = sy + (idx % 2) * 110
        state_positions.append((x, y))
        stage = _sample_progress(rec, goal_literals)
        sat = _goal_satisfied_count(rec, goal_literals)
        fill = "#eef6ff" if stage != "advance_goal_literals" else "#eff9ef"
        out.append(f'<circle cx="{x}" cy="{y}" r="47" fill="{fill}" stroke="#1f5aa6" stroke-width="1.5"/>')
        out.extend(
            _svg_text_lines(
                [f"x{idx}", stage, f"{sat}/{len(goal_literals)} goals"],
                x - 38,
                y - 12,
                size=11,
            )
        )
        out.append(f"<title>{svg_title(rec)}</title>")
    for idx in range(1, len(shown_steps)):
        prev_x, prev_y = state_positions[idx - 1]
        cur_x, cur_y = state_positions[idx]
        action = str(shown_steps[idx].get("Action"))
        ax = (prev_x + cur_x) // 2
        ay = (prev_y + cur_y) // 2 - 42
        out.append(f'<line x1="{prev_x + 48}" y1="{prev_y}" x2="{ax - 50}" y2="{ay}" stroke="#333" marker-end="url(#arrow-black)"/>')
        out.append(f'<rect x="{ax - 67}" y="{ay - 25}" width="134" height="50" rx="6" fill="#fff7df" stroke="#b36b00"/>')
        out.extend(_svg_text_lines([f"a{idx}", action], ax - 58, ay - 5, size=11))
        out.append(f'<line x1="{ax + 67}" y1="{ay}" x2="{cur_x - 48}" y2="{cur_y}" stroke="#333" marker-end="url(#arrow-black)"/>')
        added, removed = _sample_delta(shown_steps[idx - 1], shown_steps[idx])
        delta = " + ".join(added[:2]) if added else "state changed"
        sg = _sample_progress(shown_steps[idx], goal_literals)
        # Causes and AdvancesTo edges.
        out.append(f'<line x1="{ax}" y1="{ay + 25}" x2="{ax}" y2="{ay + 72}" stroke="#111" marker-end="url(#arrow-black)"/>')
        out.append(f'<rect x="{ax - 72}" y="{ay + 76}" width="144" height="42" rx="6" fill="#f7f7f7" stroke="#666"/>')
        out.extend(_svg_text_lines(["delta", delta[:34]], ax - 62, ay + 94, size=10))
        out.append(f'<line x1="{ax + 67}" y1="{ay + 10}" x2="{ax + 115}" y2="{ay + 58}" stroke="#2f7d32" marker-end="url(#arrow-green)"/>')
        out.append(f'<ellipse cx="{ax + 158}" cy="{ay + 70}" rx="55" ry="28" fill="#eff9ef" stroke="#2f7d32"/>')
        out.extend(_svg_text_lines(["subgoal", sg], ax + 118, ay + 66, size=10))

    out.append('<defs>')
    out.append('<marker id="arrow-black" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#333"/></marker>')
    out.append('<marker id="arrow-green" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#2f7d32"/></marker>')
    out.append("</defs>")

    # Local memory rules panel.
    local_rules = [
        ("pick ball1 rooma right", "initial_planning"),
        ("pick ball2 rooma left", "search_preconditions"),
        ("move rooma roomb", "search_preconditions"),
        ("drop ball1 roomb right", "search_preconditions"),
        ("drop ball2 roomb left", "advance_goal_literals"),
    ]
    out.extend(_svg_text_lines(["The maintainer merges repeated signatures:", "same state/action/rule -> support/confidence stats"], 1045, 155, size=13))
    y = 220
    for action, stage in local_rules:
        pre = _find_rule(local_data, rule_type="precondition", action=action, stage=stage)
        wf = _find_rule(local_data, rule_type="workflow", action=action, stage=stage)
        for rule in [pre, wf]:
            if not rule:
                continue
            stats = rule.get("stats", {})
            text = f"{rule.get('rule_type')}: {rule.get('summary')} | support={stats.get('support')}, conf={stats.get('confidence'):.2f}"
            out.append(f'<rect x="1045" y="{y}" width="490" height="43" rx="6" fill="#eff9ef" stroke="#2f7d32"/>')
            out.extend(_svg_text_lines([wrap_plain(text, 70).split("\n")[0], *wrap_plain(text, 70).split("\n")[1:2]], 1056, y + 18, size=11))
            y += 52
            if y > 585:
                break
        if y > 585:
            break

    # Retrieval panel.
    query_lines = [
        "Query state: after move rooma -> roomb",
        "progress_state = search_preconditions",
        "unsatisfied: ball1/2/3/4 at roomb",
        "admissible: drop ball1 roomb right,",
        "            drop ball2 roomb left, ...",
    ]
    out.extend(_svg_text_lines(query_lines, 1620, 155, size=13))
    local_drop = _find_rule(local_data, rule_type="workflow", action="drop ball1 roomb right", stage="search_preconditions")
    global_drop = None
    for rule in global_data.get("rules", []):
        if rule.get("rule_type") == "workflow" and "drop ball1 roomb right" in str(rule.get("summary", "")).lower():
            global_drop = rule
            break
    retrieve_rows = [
        ("local rule", local_drop),
        ("global rule", global_drop),
    ]
    y = 330
    for label, rule in retrieve_rows:
        if not rule:
            continue
        stats = rule.get("stats", {})
        lines = [
            label,
            str(rule.get("summary")),
            f"support={stats.get('support')}, confidence={stats.get('confidence'):.2f}",
        ]
        out.append(f'<rect x="1620" y="{y}" width="515" height="86" rx="8" fill="#f6efff" stroke="#6a2ca0"/>')
        out.extend(_svg_text_lines([wrap_plain(line, 60) for line in lines], 1634, y + 22, size=12))
        y += 105

    # Injected prompt explanation.
    prompt_lines = [
        "GM3 does not inject the whole graph. It retrieves ranked SupportItems from the graph memory.",
        "For this query, the useful evidence says:",
        "",
        "Local memory: current PDDL state/action grounding: valid operator `drop ball1 roomb right`",
        "matches an unsatisfied goal literal and historically moves search_preconditions -> advance_goal_literals.",
        "",
        "Global memory: the same abstract workflow was promoted from gripper local memory, so it can also",
        "support the decision when evaluating local+global.",
        "",
        "Next priority: execute a current admissible operator that advances an unsatisfied goal literal;",
        "here, `drop ball1 roomb right` or `drop ball2 roomb left`, not an unavailable copied action.",
    ]
    out.extend(_svg_text_lines(prompt_lines, 55, 705, size=15))
    out.append("</svg>")

    (out_dir / "pddl_gm3_real_sample_construction_retrieval.svg").write_text("\n".join(out), encoding="utf-8")


def write_overview(base: Path, out_dir: Path) -> None:
    global_data = load_json(base / "global" / "graph_memory3" / "global_memory.json")
    summaries = {domain: domain_summary(base, domain) for domain in DOMAINS}
    global_rule_sources = Counter(
        scene for rule in global_data.get("rules", []) for scene in rule.get("source_scenes", [])
    )
    global_artifact_sources = Counter(
        scene for artifact in global_data.get("artifacts", []) for scene in artifact.get("source_scenes", [])
    )

    lines = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="white", pad="0.18", nodesep="0.35", ranksep="0.55"];',
        '  node [shape=plain, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", color="#333333", arrowsize=0.8];',
    ]

    total_eps = sum(s["episodes"] for s in summaries.values())
    total_success = sum(s["success"] for s in summaries.values())
    lines.append(
        "  run "
        + "[label="
        + html_table(
            "1. Run / episode sources",
            [
                ("run id", "pddl_gm3_claude_haiku45_20260514_144924"),
                ("stored episodes", f"{total_eps} ({total_success} labeled success)"),
                ("domains", ", ".join(DOMAINS)),
                ("memory type", "graph_memory3"),
            ],
            "blue",
        )
        + "];"
    )

    local_rows = []
    for domain, summary in summaries.items():
        data = summary["data"]
        local_rows.append(
            (
                domain,
                f"{summary['episodes']} eps, {data.get('node_count')} nodes, "
                f"{data.get('edge_count')} edges, {len(data.get('rules', []))} rules",
            )
        )
    lines.append("  local [label=" + html_table("2. Local graph memories", local_rows, "green") + "];")

    induced_rows = []
    for domain, summary in summaries.items():
        data = summary["data"]
        induced_rows.append(
            (
                domain,
                f"{len(data.get('candidates', []))} candidates, "
                f"{len(data.get('rules', []))} rules, {len(data.get('artifacts', []))} artifacts",
            )
        )
    lines.append(
        "  induced [label=" + html_table("3. Induced units from local graphs", induced_rows, "purple") + "];"
    )

    rules_by_type = Counter(r.get("rule_type") for r in global_data.get("rules", []))
    artifacts_by_kind = Counter(a.get("kind") for a in global_data.get("artifacts", []))
    global_rows = [
        ("promoted batch", ", ".join(global_data.get("promoted_batches", []))),
        ("global rules", f"{global_data.get('rule_count')} {dict(rules_by_type)}"),
        ("global artifacts", f"{global_data.get('artifact_count')} {dict(artifacts_by_kind)}"),
        (
            "rule sources",
            ", ".join(f"{k.replace('pddl:', '')}:{v}" for k, v in sorted(global_rule_sources.items())),
        ),
        (
            "artifact sources",
            ", ".join(f"{k.replace('pddl:', '')}:{v}" for k, v in sorted(global_artifact_sources.items())),
        ),
    ]
    lines.append("  global [label=" + html_table("4. Promoted global memory", global_rows, "orange") + "];")

    lines.extend(
        [
            '  run -> local [label="merge trajectories"];',
            '  local -> induced [label="scan typed paths"];',
            '  induced -> global [label="promote + abstract"];',
            "}",
        ]
    )
    dot_path = out_dir / "pddl_gm3_real_overview.dot"
    dot_path.write_text("\n".join(lines), encoding="utf-8")
    render_with_dot(dot_path)


def write_global(base: Path, out_dir: Path) -> None:
    global_data = load_json(base / "global" / "graph_memory3" / "global_memory.json")
    lines = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="white", pad="0.2", nodesep="0.35", ranksep="0.65"];',
        '  node [shape=plain, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", color="#444444", arrowsize=0.75];',
    ]

    lines.append(
        "  global [label="
        + html_table(
            "Final global memory",
            [
                ("rules", global_data.get("rule_count")),
                ("artifacts", global_data.get("artifact_count")),
                ("candidates", global_data.get("candidate_count")),
                ("insights", "0"),
            ],
            "orange",
        )
        + "];"
    )

    for domain in DOMAINS:
        scene = f"pddl:{domain}"
        rules = [r for r in global_data.get("rules", []) if scene in r.get("source_scenes", [])]
        artifacts = [a for a in global_data.get("artifacts", []) if scene in a.get("source_scenes", [])]
        node = dot_id(domain)
        rows = [
            ("global rules", len(rules)),
            ("global artifacts", len(artifacts)),
            ("rule types", dict(Counter(r.get("rule_type") for r in rules))),
            ("artifact kinds", dict(Counter(a.get("kind") for a in artifacts))),
        ]
        lines.append(f"  {node} [label={html_table(domain, rows, 'green')}];")
        lines.append(f'  {node} -> global [label="promoted"];')

        top = top_rules(rules, 4)
        if top:
            rule_rows = []
            for idx, rule in enumerate(top, 1):
                stats = rule.get("stats", {})
                rule_rows.append(
                    (
                        f"#{idx}",
                        f"{rule.get('rule_type')}: {rule.get('summary')} "
                        f"(support={stats.get('support')}, conf={stats.get('confidence'):.2f})",
                    )
                )
            rnode = f"{node}_rules"
            lines.append(f"  {rnode} [label={html_table('top promoted rules', rule_rows, 'purple')}];")
            lines.append(f"  {node} -> {rnode};")
            lines.append(f"  {rnode} -> global;")

    lines.append("}")
    dot_path = out_dir / "pddl_gm3_real_global.dot"
    dot_path.write_text("\n".join(lines), encoding="utf-8")
    render_with_dot(dot_path)


def write_domain(base: Path, out_dir: Path, domain: str) -> None:
    summary = domain_summary(base, domain)
    data = summary["data"]
    rules = top_rules(data.get("rules", []), 8)
    lines = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="white", pad="0.2", nodesep="0.35", ranksep="0.55"];',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, arrowsize=0.7];',
        f'  label="Local Graph Memory: {esc(domain)} (real snapshot, top-support rules)";',
        '  labelloc="t";',
    ]
    stats_rows = [
        ("episodes", f"{summary['episodes']} total, {summary['success']} success, {summary['failure']} fail"),
        ("nodes", f"{data.get('node_count')} {dict(summary['nodes'])}"),
        ("edges", f"{data.get('edge_count')} {dict(summary['edges'])}"),
        ("candidates", f"{len(data.get('candidates', []))} {dict(summary['candidates'])}"),
        ("rules", f"{len(data.get('rules', []))} {dict(summary['rules'])}"),
        ("artifacts", f"{len(data.get('artifacts', []))} {dict(summary['artifacts'])}"),
    ]
    lines.append("  stats [shape=plain, label=" + html_table("snapshot counts", stats_rows, "green") + "];")

    stage_nodes = set()
    for rule in rules:
        start, end = workflow_endpoints(rule)
        stage_nodes.add(start)
        stage_nodes.add(end)
    for stage in sorted(stage_nodes):
        fill = PALETTE["blue_fill"]
        color = PALETTE["blue"]
        if "goal" in stage:
            fill = PALETTE["green_fill"]
            color = PALETTE["green"]
        lines.append(
            f'  stage_{dot_id(stage)} [shape=ellipse, style="filled", '
            f'fillcolor="{fill}", color="{color}", label="{esc(stage)}"];'
        )

    for idx, rule in enumerate(rules, 1):
        start, end = workflow_endpoints(rule)
        positive = rule.get("rule_type") in {"workflow", "precondition"}
        color = PALETTE["green"] if positive else PALETTE["red"]
        fill = PALETTE["green_fill"] if positive else PALETTE["red_fill"]
        style = "solid" if positive else "dashed"
        stats = rule.get("stats", {})
        label = (
            f"#{idx} {rule.get('rule_type')}\n"
            f"{rule.get('summary')}\n"
            f"support={stats.get('support')}, conf={stats.get('confidence'):.2f}, "
            f"stalled={stats.get('stalled', 0)}"
        )
        rule_node = f"rule_{idx}"
        lines.append(
            f'  {rule_node} [shape=box, style="rounded,filled", fillcolor="{fill}", '
            f'color="{color}", label="{quote_label(wrap_plain(label, 54))}"];'
        )
        lines.append(f'  stage_{dot_id(start)} -> {rule_node} [color="{color}", style="{style}"];')
        lines.append(f'  {rule_node} -> stage_{dot_id(end)} [color="{color}", style="{style}"];')

    candidate_rows = []
    for idx, cand in enumerate(
        sorted(data.get("candidates", []), key=lambda c: (c.get("positive", 0), c.get("negative", 0)), reverse=True)[:6],
        1,
    ):
        candidate_rows.append(
            (
                f"#{idx}",
                f"{cand.get('candidate_type')}: {cand.get('summary')} "
                f"(pos={cand.get('positive')}, neg={cand.get('negative')})",
            )
        )
    if candidate_rows:
        lines.append("  candidates [shape=plain, label=" + html_table("top local candidates", candidate_rows, "purple") + "];")
        lines.append("  stats -> candidates [style=dotted, color=\"#777777\"];")

    artifact_rows = []
    for idx, artifact in enumerate(data.get("artifacts", [])[:6], 1):
        artifact_rows.append((f"#{idx}", artifact.get("summary", "")))
    if artifact_rows:
        lines.append("  artifacts [shape=plain, label=" + html_table("sample local artifacts", artifact_rows, "orange") + "];")
        lines.append("  candidates -> artifacts [style=dotted, color=\"#777777\"];")

    lines.append("}")
    dot_path = out_dir / f"pddl_gm3_real_local_{domain}.dot"
    dot_path.write_text("\n".join(lines), encoding="utf-8")
    render_with_dot(dot_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--full-png",
        action="store_true",
        help="Also rasterize full local graphs to PNG. This can be slow for large local memories.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    write_overview(args.base, args.out_dir)
    write_global(args.base, args.out_dir)
    write_sample_construction_retrieval(args.base, args.out_dir)
    for domain in DOMAINS:
        write_domain(args.base, args.out_dir, domain)
        write_full_domain(args.base, args.out_dir, domain, full_png=args.full_png)

    print(f"Wrote GraphMemory3 visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
