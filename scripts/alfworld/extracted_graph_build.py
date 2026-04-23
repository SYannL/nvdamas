# -*- coding: utf-8 -*-
"""
从 traj.extracted_entities_relations.jsonl 聚合图，节点/边均按与 visualize_traj_triples 相同的规则做 categorization：
instance_id 经 _canonical_key 合并（去掉尾部数字、小写、去分隔符），边端点同步规范化。

用于 query↔图 embedding 匹配与可视化，保证二者在同一套「分类图」上。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_TRAILING_ID_RE = re.compile(r"^(?P<base>.*?)(?:[_\s]+(?P<num>\d+))$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _strip_trailing_numeric_id(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return s
    m = _TRAILING_ID_RE.match(s)
    if not m:
        return s
    base = (m.group("base") or "").strip()
    return base if base else s


def _canonical_key(text: str) -> str:
    s = _strip_trailing_numeric_id(text)
    s = s.strip().lower()
    s = _NON_ALNUM_RE.sub("", s)
    return s


def _norm_edge_key(subj: str, rel: str, obj: str) -> tuple[str, str, str]:
    return (subj.strip(), rel.strip(), obj.strip())


def node_match_text(canonical_node_id: str) -> str:
    """匹配池中的节点项：仅为 canonical id（如 soapbar）。"""
    return canonical_node_id.strip()


def edge_match_text(relation_canon: str) -> str:
    """匹配池中的边项：仅规范化 relation（如 clean），不含端点 triple。"""
    return relation_canon.strip()


def build_graph_candidates(jsonl_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    分类（categorized）后的唯一节点与边。
    - 节点候选项 text = canonical instance_id（如 soapbar）。
    - 边候选项 text = 仅规范化 relation（多条边 relation 相同则 text 相同）。
    record 仍含完整 subject/relation/object 便于命中回溯与可视化。
    """
    node_meta: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed = row.get("parsed")
            if not isinstance(parsed, dict):
                continue
            insts = parsed.get("instances") or []
            rels = parsed.get("relations") or []

            for inst in insts:
                if not isinstance(inst, dict):
                    continue
                iid0 = str(inst.get("instance_id") or "").strip()
                if not iid0:
                    continue
                nid = _canonical_key(iid0)
                if not nid:
                    continue
                if nid not in node_meta:
                    merged = dict(inst)
                    merged["instance_id"] = nid
                    merged["original_ids"] = [iid0]
                    node_meta[nid] = merged
                else:
                    cur = node_meta[nid]
                    oids = list(cur.get("original_ids") or [])
                    if iid0 and iid0 not in oids:
                        oids.append(iid0)
                    cur_compact = {k: v for k, v in cur.items() if k != "original_ids"}
                    if len(json.dumps(inst, sort_keys=True)) >= len(json.dumps(cur_compact, sort_keys=True)):
                        new = dict(inst)
                        new["instance_id"] = nid
                        new["original_ids"] = oids
                        node_meta[nid] = new
                    else:
                        cur["original_ids"] = oids

            for rel in rels:
                if not isinstance(rel, dict):
                    continue
                s0 = str(rel.get("subject") or "").strip()
                o0 = str(rel.get("object") or "").strip()
                r0 = str(rel.get("relation") or "").strip()
                if not s0 or not o0 or not r0:
                    continue
                s = _canonical_key(s0)
                o = _canonical_key(o0)
                r = _canonical_key(r0) or r0.lower()
                if not s or not o or not r:
                    continue
                k = _norm_edge_key(s, r, o)
                edges[k] = {"subject": s, "relation": r, "object": o}

    for s, r, o in list(edges.keys()):
        for end in (s, o):
            if end not in node_meta:
                node_meta[end] = {
                    "instance_id": end,
                    "label": end,
                    "category": "",
                    "original_ids": [],
                }

    candidates: list[dict[str, Any]] = []
    for nid in sorted(node_meta.keys()):
        meta = node_meta[nid]
        text = node_match_text(nid)
        candidates.append(
            {
                "kind": "node",
                "key": f"node:{nid}",
                "text": text,
                "record": dict(meta),
            }
        )

    for (s, r, o), _e in sorted(edges.items()):
        ts = edge_match_text(r)
        candidates.append(
            {
                "kind": "edge",
                "key": f"edge:{s}|{r}|{o}",
                "text": ts,
                "record": {"subject": s, "relation": r, "object": o},
            }
        )

    meta = {
        "node_count": len(node_meta),
        "edge_count": len(edges),
        "graph_mode": "categorized",
    }
    return candidates, meta
