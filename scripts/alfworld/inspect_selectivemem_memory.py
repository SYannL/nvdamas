#!/usr/bin/env python3
"""
Print a summary of SelectiveMem persisted graphs (nodes + relations) after an alfworld run.

Default persist directory matches tasks/run.py:
  <repo>/.db/<model_type>/alfworld/<mas_type>/memory/<mas_memory>/<mas_memory>/

Examples:
  python scripts/alfworld/inspect_selectivemem_memory.py
  python scripts/alfworld/inspect_selectivemem_memory.py \\
    --persist_dir .db/gpt-4o-mini/alfworld/autogen/memory/selectivemem/selectivemem
  python scripts/alfworld/inspect_selectivemem_memory.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_get_model_type():
    sys.path.insert(0, str(_REPO_ROOT / "tasks"))
    from utils import get_model_type  # noqa: WPS433

    return get_model_type


def default_persist_dir(model: str, mas_type: str, mas_memory: str, task: str) -> Path:
    get_model_type = _import_get_model_type()
    mt = get_model_type(model)
    return (
        _REPO_ROOT
        / ".db"
        / mt
        / task
        / mas_type
        / "memory"
        / mas_memory
        / mas_memory
    )


def _load(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _print_local_instance(data: dict, full: bool) -> None:
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    print(f"  nodes: {len(nodes)}  edges: {len(edges)}")
    if not full:
        return
    for n in nodes:
        nid = n.get("id", "?")
        rest = {k: v for k, v in n.items() if k != "id"}
        print(f"    [node] {nid}  {rest}")
    for e in edges:
        rel = e.get("relation", e.get("relations", "?"))
        print(f"    [edge] {e.get('from')} --{rel}--> {e.get('to')}")


def _print_category(data: dict, full: bool) -> None:
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    print(f"  nodes: {len(nodes)}  edges: {len(edges)}")
    if not full:
        return
    for n in nodes:
        print(f"    [node] {n}")
    for e in edges:
        print(f"    [edge] {e.get('from')} -> {e.get('to')}  relations={e.get('relations')} count={e.get('count')}")


def _print_global_gg(data: dict, full: bool) -> None:
    nodes_map = data.get("nodes") or {}
    edges = data.get("edges") or []
    print(f"  concept nodes: {len(nodes_map)}  edges: {len(edges)}")
    if not full:
        return
    id_to_name = {cid: nd.get("name", cid) for cid, nd in nodes_map.items()}
    for cid, nd in sorted(nodes_map.items(), key=lambda x: x[0]):
        print(f"    [concept] {cid}  name={nd.get('name')}  type={nd.get('type')}")
    for e in edges:
        fr = id_to_name.get(e.get("from"), e.get("from"))
        to = id_to_name.get(e.get("to"), e.get("to"))
        print(
            f"    [edge] {fr} --{e.get('relation')}--> {to}  weight={e.get('weight')}"
        )


def _print_merged_node_link(data: dict, full: bool) -> None:
    # networkx node_link_data: "nodes", "links" (or legacy "edges")
    nodes = data.get("nodes") or []
    links = data.get("links") or data.get("edges") or []
    print(f"  nodes: {len(nodes)}  links: {len(links)}")
    if not full:
        return
    for n in nodes:
        print(f"    [node] {n}")
    for L in links:
        rel = L.get("relation") or L.get("label")
        print(
            f"    [link] {L.get('source')} --{rel}--> {L.get('target')}  key={L.get('key')}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect SelectiveMem graph JSONs on disk.")
    ap.add_argument(
        "--persist_dir",
        type=str,
        default=None,
        help="Explicit directory containing local_instance_graph.json etc.",
    )
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="With default path only.")
    ap.add_argument("--mas_type", type=str, default="autogen", help="With default path only.")
    ap.add_argument("--mas_memory", type=str, default="selectivemem", help="With default path only.")
    ap.add_argument("--task", type=str, default="alfworld", help="With default path only.")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Print every node and every relation edge (can be long).",
    )
    args = ap.parse_args()

    if args.persist_dir:
        pdir = Path(args.persist_dir).expanduser().resolve()
    else:
        pdir = default_persist_dir(args.model, args.mas_type, args.mas_memory, args.task)

    print(f"Persist directory: {pdir}")
    print(f"Exists: {pdir.is_dir()}\n")

    artifacts = [
        ("Local instance graph (GL)", "local_instance_graph.json", _print_local_instance),
        ("Local category graph", "local_category_graph.json", _print_category),
        ("Merged instance graph (semantic LG)", "merged_instance_graph.json", _print_merged_node_link),
        ("Global experience graph (GG)", "global_experience_graph.json", _print_global_gg),
    ]

    any_ok = False
    for title, fname, printer in artifacts:
        path = pdir / fname
        print(f"=== {title} ({fname}) ===")
        if not path.is_file():
            print(f"  (missing) {path}")
            print()
            continue
        any_ok = True
        st = path.stat()
        print(f"  size_bytes={st.st_size}")
        raw = _load(path)
        if raw is None:
            print("  (invalid json)")
            print()
            continue
        if not isinstance(raw, dict):
            print(f"  (unexpected root type {type(raw).__name__})")
            print()
            continue
        printer(raw, args.full)
        print()

    aux = [
        "insight_bank.json",
        "query_insight_archive.json",
        "trajectory_fingerprints.jsonl",
        "retrieval_trace.jsonl",
        "gg_common_summaries.jsonl",
    ]
    print("=== Auxiliary artifacts ===")
    for a in aux:
        path = pdir / a
        if path.is_file():
            print(f"  ok  {a}  ({path.stat().st_size} bytes)")
        else:
            print(f"  --  {a}  (missing)")

    if not any_ok:
        sys.exit(
            "No graph JSON found. Check --persist_dir or run from repo root after "
            "`tasks/run.py --task alfworld --mas_memory selectivemem ...`"
        )


if __name__ == "__main__":
    main()
