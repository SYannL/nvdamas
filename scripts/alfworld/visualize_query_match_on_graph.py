#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 query_concept_graph_match_results.jsonl 中某 trial 的匹配结果，
在 bathroom / kitchen 各自的 traj.extracted_entities_relations.jsonl 的 **categorized 聚合图**上可视化
（与 match_query_concepts_to_extracted_graphs / visualize_traj_triples 的 canonical 规则一致）。
匹配到的节点与边均为红色，其余为灰色。

高亮键：节点为 canonical instance_id；边为 (canonical subject, relation, canonical object)。
同一 trial 下 **每一行** JSONL 都会参与：先并入高亮集合（相同节点/边会合并），再把每条记录的 `concept`
挂到对应节点/边的 tooltip（故像「Clean」这类概念仍会显示在边上，即使多条记录叠在同一条边）。

若结果文件来自旧版「未分类」匹配，会对 matched_record 中的原始 id 做 canonical 映射。

**可视化阈值**：`--match_threshold` 仅影响图中红/灰高亮与 tooltip 聚合；JSONL 仍应含完整 `match_score`。
低于阈值的行仍会出现在表格中，但「图高亮」为否。

节点标签会去掉末尾数字（与分类图规则一致）；含 `nothing … seen` 的观测噪声节点及其边在图中不展示。

过阈值的高亮若因噪声过滤或路径不一致未出现在当前图候选中，会 **补回** 对应节点/边以便在图中标红（端点落在 junk 的边仍不补）。
"""

from __future__ import annotations

import argparse
import html as html_lib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

_egb_path = Path(__file__).resolve().parent / "extracted_graph_build.py"
_egb_spec = importlib.util.spec_from_file_location("extracted_graph_build", _egb_path)
if _egb_spec is None or _egb_spec.loader is None:
    raise RuntimeError(f"cannot load {_egb_path}")
_egb = importlib.util.module_from_spec(_egb_spec)
_egb_spec.loader.exec_module(_egb)
build_graph_candidates = _egb.build_graph_candidates
_canonical_key = _egb._canonical_key
_strip_trailing_numeric_id = _egb._strip_trailing_numeric_id
node_match_text = _egb.node_match_text
edge_match_text = _egb.edge_match_text

# ALFWorld 观测句如 "nothing seen in cabinet 7" 不应作为实体节点展示
_JUNK_OBSERVATION_NODE_RE = re.compile(r"(?i)nothing.*seen")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    rows.append(o)
            except json.JSONDecodeError:
                continue
    return rows


def _graph_keys_from_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[set[str], set[tuple[str, str, str]]]:
    node_ids: set[str] = set()
    edge_keys: set[tuple[str, str, str]] = set()
    for c in candidates:
        if c.get("kind") == "node":
            rec = c.get("record") or {}
            iid = str(rec.get("instance_id") or "").strip()
            if iid:
                node_ids.add(iid)
        elif c.get("kind") == "edge":
            rec = c.get("record") or {}
            s = str(rec.get("subject") or "").strip()
            o = str(rec.get("object") or "").strip()
            rel = str(rec.get("relation") or "").strip()
            if s and o and rel:
                edge_keys.add((s, rel, o))
    return node_ids, edge_keys


def _vis_score_passes_threshold(sc: Any, thr: float) -> bool:
    """JSONL 中分数多为 float，偶见字符串；无法解析时保持旧行为（视为过阈，以免静默丢高亮）。"""
    if isinstance(sc, (int, float)):
        return float(sc) >= thr
    if isinstance(sc, str):
        t = sc.strip()
        if not t:
            return True
        try:
            return float(t) >= thr
        except ValueError:
            return True
    return True


def _collect_highlights_and_concepts(
    rows: list[dict[str, Any]],
    graph_node_ids: set[str],
    graph_edge_keys: set[tuple[str, str, str]],
    score_threshold: float,
) -> tuple[
    set[str],
    set[tuple[str, str, str]],
    dict[str, list[str]],
    dict[tuple[str, str, str], list[str]],
    list[dict[str, Any]],
]:
    """
    遍历该 domain 下该 trial 的 **全部** 匹配行，无省略。
    仅当 match_score >= score_threshold 时并入红/高亮与节点/边 tooltip；
    表格仍列出全部行，并标注是否满足可视化阈值。
    """
    hit_nodes: set[str] = set()
    hit_edges: set[tuple[str, str, str]] = set()
    node_concepts: dict[str, list[str]] = {}
    edge_concepts: dict[tuple[str, str, str], list[str]] = {}
    details: list[dict[str, Any]] = []
    thr = float(score_threshold)

    for r in rows:
        sc = r.get("match_score")
        passes = _vis_score_passes_threshold(sc, thr)

        concept = str(r.get("concept") or "").strip()
        mk = r.get("match_kind")
        kind = mk if isinstance(mk, str) else ""
        detail: dict[str, Any] = {
            "concept": concept,
            "match_kind": kind,
            "match_score": sc,
            "matched_text": str(r.get("matched_text") or ""),
            "vis_threshold": thr,
            "vis_highlight": passes,
        }

        mr = r.get("matched_record") or {}
        if not isinstance(mr, dict):
            detail["in_graph"] = None
            details.append(detail)
            continue

        if kind == "node":
            iid = str(mr.get("instance_id") or "").strip()
            cid = _canonical_key(iid) if iid else ""
            if cid and passes:
                hit_nodes.add(cid)
                lst = node_concepts.setdefault(cid, [])
                if concept and concept not in lst:
                    lst.append(concept)
                detail["canonical_id"] = cid
                detail["in_graph"] = cid in graph_node_ids
            elif cid:
                detail["canonical_id"] = cid
                detail["in_graph"] = cid in graph_node_ids
            else:
                detail["in_graph"] = False
            details.append(detail)

        elif kind == "edge":
            s0 = str(mr.get("subject") or "").strip()
            o0 = str(mr.get("object") or "").strip()
            rel0 = str(mr.get("relation") or "").strip()
            s = _canonical_key(s0) if s0 else ""
            o = _canonical_key(o0) if o0 else ""
            rel = (_canonical_key(rel0) or rel0.lower()) if rel0 else ""
            if s and o and rel:
                ek = (s, rel, o)
                if passes:
                    hit_edges.add(ek)
                    lst = edge_concepts.setdefault(ek, [])
                    if concept and concept not in lst:
                        lst.append(concept)
                detail["canonical_edge"] = {"subject": s, "relation": rel, "object": o}
                detail["in_graph"] = ek in graph_edge_keys
            else:
                detail["in_graph"] = False
            details.append(detail)
        else:
            detail["in_graph"] = None
            details.append(detail)

    return hit_nodes, hit_edges, node_concepts, edge_concepts, details


def _node_label(rec: dict[str, Any]) -> str:
    cat = _strip_trailing_numeric_id(str(rec.get("category") or "").strip())
    lab = _strip_trailing_numeric_id(str(rec.get("label") or "").strip())
    iid = _strip_trailing_numeric_id(str(rec.get("instance_id") or "").strip())
    if lab:
        return lab if len(lab) < 28 else lab[:25] + "…"
    if cat:
        return cat
    return iid or "?"


def _observation_noise_node(rec: dict[str, Any]) -> bool:
    parts = [
        str(rec.get("label") or ""),
        str(rec.get("instance_id") or ""),
        *(str(x) for x in (rec.get("original_ids") or []) if x),
    ]
    blob = " ".join(parts)
    return bool(_JUNK_OBSERVATION_NODE_RE.search(blob))


def _filter_viz_noise_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """去掉观测噪声节点（如 nothing seen in cabinet 7）及其关联边，仅影响可视化。返回 (过滤后候选, junk canonical id 集合)。"""
    junk_ids: set[str] = set()
    for c in candidates:
        if c.get("kind") != "node":
            continue
        rec = c.get("record") or {}
        iid = str(rec.get("instance_id") or "").strip()
        if not iid:
            continue
        if _observation_noise_node(rec if isinstance(rec, dict) else {}):
            junk_ids.add(iid)
    out: list[dict[str, Any]] = []
    for c in candidates:
        kind = c.get("kind")
        if kind == "node":
            rec = c.get("record") or {}
            iid = str(rec.get("instance_id") or "").strip()
            if iid in junk_ids:
                continue
        elif kind == "edge":
            rec = c.get("record") or {}
            s = str(rec.get("subject") or "").strip()
            o = str(rec.get("object") or "").strip()
            if s in junk_ids or o in junk_ids:
                continue
        out.append(c)
    return out, junk_ids


def _index_candidate_nodes_edges(
    candidates: list[dict[str, Any]],
) -> tuple[set[str], set[tuple[str, str, str]]]:
    nids: set[str] = set()
    ekeys: set[tuple[str, str, str]] = set()
    for c in candidates:
        if c.get("kind") == "node":
            rec = c.get("record") or {}
            iid = str(rec.get("instance_id") or "").strip()
            if iid:
                nids.add(iid)
        elif c.get("kind") == "edge":
            rec = c.get("record") or {}
            s = str(rec.get("subject") or "").strip()
            o = str(rec.get("object") or "").strip()
            rel = str(rec.get("relation") or "").strip()
            if s and o and rel:
                ekeys.add((s, rel, o))
    return nids, ekeys


def _placeholder_node_candidate(nid: str) -> dict[str, Any]:
    return {
        "kind": "node",
        "key": f"node:{nid}",
        "text": node_match_text(nid),
        "record": {
            "instance_id": nid,
            "label": nid,
            "category": "",
            "original_ids": [],
        },
    }


def _backfill_candidates_for_highlights(
    candidates: list[dict[str, Any]],
    hit_nodes: set[str],
    hit_edges: set[tuple[str, str, str]],
    junk_ids: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """
    保证「过阈值」的 hit_nodes / hit_edges 在当前候选里都有对应条目，否则 vis 无法画红。
    典型原因：噪声过滤删掉了边/节点，但 JSONL 匹配仍指向该三元组；或匹配与可视化用了不同 jsonl。
    """
    nids, ekeys = _index_candidate_nodes_edges(candidates)
    extra: list[dict[str, Any]] = []

    for nid in hit_nodes:
        if not nid or nid in junk_ids or nid in nids:
            continue
        nids.add(nid)
        extra.append(_placeholder_node_candidate(nid))

    for s, r, o in hit_edges:
        if not s or not o or not r:
            continue
        if s in junk_ids or o in junk_ids:
            continue
        ek = (s, r, o)
        if ek in ekeys:
            continue
        ekeys.add(ek)
        if s not in nids:
            nids.add(s)
            extra.append(_placeholder_node_candidate(s))
        if o not in nids:
            nids.add(o)
            extra.append(_placeholder_node_candidate(o))
        extra.append(
            {
                "kind": "edge",
                "key": f"edge:{s}|{r}|{o}",
                "text": edge_match_text(r),
                "record": {"subject": s, "relation": r, "object": o},
            }
        )

    return candidates + extra, len(extra)


def _viz_meta_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    n = sum(1 for c in candidates if c.get("kind") == "node")
    e = sum(1 for c in candidates if c.get("kind") == "edge")
    return {"node_count": n, "edge_count": e}


def _ensure_placeholder_nodes(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """若边端点未出现在实例节点中，补占位节点以便 vis 可渲染。"""
    node_ids: set[str] = set()
    for c in candidates:
        if c.get("kind") != "node":
            continue
        rec = c.get("record") or {}
        iid = str(rec.get("instance_id") or "").strip()
        if iid:
            node_ids.add(iid)
    extra: list[dict[str, Any]] = []
    for c in candidates:
        if c.get("kind") != "edge":
            continue
        rec = c.get("record") or {}
        for key in ("subject", "object"):
            iid = str(rec.get(key) or "").strip()
            if iid and iid not in node_ids:
                node_ids.add(iid)
                extra.append(
                    {
                        "kind": "node",
                        "key": f"node:{iid}",
                        "text": f"id:{iid}",
                        "record": {"instance_id": iid, "label": iid, "category": ""},
                    }
                )
    return candidates + extra


def _to_vis_payload(
    candidates: list[dict[str, Any]],
    hit_nodes: set[str],
    hit_edges: set[tuple[str, str, str]],
    node_concepts: dict[str, list[str]],
    edge_concepts: dict[tuple[str, str, str], list[str]],
    *,
    color_hit: str,
    color_gray_node: str,
    color_gray_edge: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = _ensure_placeholder_nodes(candidates)
    nodes_out: list[dict[str, Any]] = []
    edges_out: list[dict[str, Any]] = []

    for c in candidates:
        if c.get("kind") != "node":
            continue
        rec = c.get("record") or {}
        iid = str(rec.get("instance_id") or "").strip()
        if not iid:
            continue
        hit = iid in hit_nodes
        base_title = json.dumps(rec, ensure_ascii=False)[:800]
        nc = node_concepts.get(iid) or []
        if nc:
            base_title += "\n---\n匹配概念（本 trial 全部行）: " + ", ".join(nc)
        nodes_out.append(
            {
                "id": iid,
                "label": _node_label(rec if isinstance(rec, dict) else {}),
                "title": base_title,
                "color": {
                    "background": color_hit if hit else color_gray_node,
                    "border": "#742a2a" if hit else "#b0b0b0",
                    "highlight": {"background": "#fc8181" if hit else "#e2e8f0", "border": "#742a2a"},
                },
                "font": {"color": "#ffffff" if hit else "#4a5568", "size": 12},
                "borderWidth": 2 if hit else 1,
                "size": 14 if hit else 10,
            }
        )

    seen_edge: set[str] = set()
    for c in candidates:
        if c.get("kind") != "edge":
            continue
        rec = c.get("record") or {}
        s = str(rec.get("subject") or "").strip()
        o = str(rec.get("object") or "").strip()
        rel = str(rec.get("relation") or "").strip()
        if not s or not o or not rel:
            continue
        eid = f"{s}|{rel}|{o}"
        if eid in seen_edge:
            continue
        seen_edge.add(eid)
        hit = (s, rel, o) in hit_edges
        ek = (s, rel, o)
        etitle = f"{s} -[{rel}]-> {o}"
        ec = edge_concepts.get(ek) or []
        if ec:
            etitle += "\n匹配概念: " + ", ".join(ec)
        edges_out.append(
            {
                "id": eid,
                "from": s,
                "to": o,
                "label": rel,
                "title": etitle,
                "color": {"color": color_hit if hit else color_gray_edge},
                "width": 3 if hit else 1,
            }
        )

    return nodes_out, edges_out


def _escape_html_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _render_match_table(details: list[dict[str, Any]]) -> str:
    if not details:
        return "<p class='muted'>（无记录）</p>"
    parts: list[str] = [
        "<table class='mtable'><thead><tr>"
        "<th>#</th><th>概念</th><th>类型</th><th>分数</th><th>可视化阈值</th><th>图高亮</th>"
        "<th>命中文本（截断）</th><th>在图中</th>"
        "</tr></thead><tbody>"
    ]
    for i, d in enumerate(details, 1):
        c = html_lib.escape(str(d.get("concept") or ""))
        k = html_lib.escape(str(d.get("match_kind") or ""))
        sc = d.get("match_score")
        if isinstance(sc, (int, float)):
            scs = html_lib.escape(f"{float(sc):.4f}")
        else:
            scs = html_lib.escape(str(sc))
        th = d.get("vis_threshold")
        if isinstance(th, (int, float)):
            ths = html_lib.escape(f"{float(th):.4f}")
        else:
            ths = html_lib.escape("—" if th is None else str(th))
        vh = d.get("vis_highlight")
        if vh is False:
            mok_s = "✗"
        elif vh is True:
            mok_s = "✓"
        else:
            mok_s = "—"
        mt_esc = html_lib.escape(str(d.get("matched_text") or "")[:140])
        ig = d.get("in_graph")
        if ig is True:
            ig_s = "✓"
        elif ig is False:
            ig_s = "✗"
        else:
            ig_s = "—"
        parts.append(
            f"<tr><td>{i}</td><td>{c}</td><td>{k}</td><td>{scs}</td><td>{ths}</td><td>{mok_s}</td>"
            f"<td>{mt_esc}</td><td>{ig_s}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize query↔graph match highlights (HTML, vis-network).")
    ap.add_argument("--trial_id", type=str, required=True, help="例如 trial_T20190908_214946_567644")
    ap.add_argument(
        "--results_jsonl",
        type=str,
        default="logs/alfworld/query_concept_graph_match_results.jsonl",
    )
    ap.add_argument(
        "--bathroom_jsonl",
        type=str,
        default="logs/alfworld/autogen/memory/g-memory/gpt-4o-mini/0406bathroom/traj.extracted_entities_relations.jsonl",
    )
    ap.add_argument(
        "--kitchen_jsonl",
        type=str,
        default="logs/alfworld/autogen/memory/selectivemem/gpt-4o-mini/traj.extracted_entities_relations.jsonl",
    )
    ap.add_argument("--out_html", type=str, default=None, help="默认写到 logs/alfworld/ 下按 trial 命名")
    ap.add_argument(
        "--match_threshold",
        type=float,
        default=0.0,
        help="仅可视化：match_score ≥ 此 cosine 阈值才在图中标红并写入边/节点 tooltip；默认 0 表示全部高亮。",
    )
    args = ap.parse_args()

    trial = args.trial_id.strip()
    results_path = _REPO_ROOT / args.results_jsonl
    if not results_path.is_file():
        raise FileNotFoundError(results_path)

    all_rows = _read_jsonl(results_path)
    bath_rows = [r for r in all_rows if r.get("trial_id") == trial and r.get("domain") == "bathroom"]
    kit_rows = [r for r in all_rows if r.get("trial_id") == trial and r.get("domain") == "kitchen"]

    goal = ""
    if bath_rows:
        goal = str(bath_rows[0].get("goal_instruction") or "")
    elif kit_rows:
        goal = str(kit_rows[0].get("goal_instruction") or "")

    b_jsonl = _REPO_ROOT / args.bathroom_jsonl
    k_jsonl = _REPO_ROOT / args.kitchen_jsonl
    if not b_jsonl.is_file():
        raise FileNotFoundError(b_jsonl)
    if not k_jsonl.is_file():
        raise FileNotFoundError(k_jsonl)

    cand_b, meta_b = build_graph_candidates(b_jsonl)
    cand_k, meta_k = build_graph_candidates(k_jsonl)
    cand_b, junk_b = _filter_viz_noise_candidates(cand_b)
    cand_k, junk_k = _filter_viz_noise_candidates(cand_k)
    meta_b.update(_viz_meta_counts(cand_b))
    meta_k.update(_viz_meta_counts(cand_k))

    gn_b, ge_b = _graph_keys_from_candidates(cand_b)
    gn_k, ge_k = _graph_keys_from_candidates(cand_k)

    vthr = float(args.match_threshold)
    hn_b, he_b, nc_b, ec_b, det_b = _collect_highlights_and_concepts(bath_rows, gn_b, ge_b, vthr)
    hn_k, he_k, nc_k, ec_k, det_k = _collect_highlights_and_concepts(kit_rows, gn_k, ge_k, vthr)

    cand_b, bf_n_b = _backfill_candidates_for_highlights(cand_b, hn_b, he_b, junk_b)
    cand_k, bf_n_k = _backfill_candidates_for_highlights(cand_k, hn_k, he_k, junk_k)
    meta_b.update(_viz_meta_counts(cand_b))
    meta_k.update(_viz_meta_counts(cand_k))
    if bf_n_b:
        print(
            f"  [viz] bathroom: 为高亮补回 {bf_n_b} 条图候选（节点/边），使过阈匹配在图中可见",
            file=sys.stderr,
        )
    if bf_n_k:
        print(
            f"  [viz] kitchen:  为高亮补回 {bf_n_k} 条图候选（节点/边），使过阈匹配在图中可见",
            file=sys.stderr,
        )

    COLOR_HIT = "#c53030"
    COLOR_GRAY_NODE = "#cbd5e0"
    COLOR_GRAY_EDGE = "#cbd5e0"

    nodes_b, edges_b = _to_vis_payload(
        cand_b, hn_b, he_b, nc_b, ec_b,
        color_hit=COLOR_HIT,
        color_gray_node=COLOR_GRAY_NODE,
        color_gray_edge=COLOR_GRAY_EDGE,
    )
    nodes_k, edges_k = _to_vis_payload(
        cand_k, hn_k, he_k, nc_k, ec_k,
        color_hit=COLOR_HIT,
        color_gray_node=COLOR_GRAY_NODE,
        color_gray_edge=COLOR_GRAY_EDGE,
    )

    out_html = (
        Path(args.out_html).expanduser().resolve()
        if args.out_html
        else _REPO_ROOT / "logs" / "alfworld" / f"query_match_vis__{trial}.html"
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)

    title = f"Query match graph — {trial}"
    nb_h = len(hn_b) + len(he_b)
    nk_h = len(hn_k) + len(he_k)
    nrows_b = len(bath_rows)
    nrows_k = len(kit_rows)

    table_b = _render_match_table(det_b)
    table_k = _render_match_table(det_k)

    warn_html = ""
    if bath_rows and not kit_rows:
        warn_html = """
  <div class="warnbox">
    <b>注意：</b> 本 trial 在 JSONL 里只有 <code>domain=bathroom</code> 的记录，没有 <code>domain=kitchen</code>。
    右侧 Kitchen 表为空、图全灰，<b>不是</b> kitchen 图里没有 Clean 等边，而是<strong>结果文件里根本没有本 trial 的 kitchen 匹配行</strong>。
    请用当前仓库里的脚本重新跑匹配（每个任务会对 bathroom + kitchen 两图各写一行）：
    <code>python scripts/alfworld/match_query_concepts_to_extracted_graphs.py</code>
    生成新 JSONL 后再可视化。全文件约应有 ~280 行（10+10 任务 × 概念数 × 2 域）；若仍约 121 行则为旧版输出。
  </div>"""
        print(
            "[warn] 本 trial 无 kitchen 域记录；Kitchen 侧无法高亮。请重新运行 match_query_concepts_to_extracted_graphs.py",
            file=sys.stderr,
        )
    elif kit_rows and not bath_rows:
        warn_html = """
  <div class="warnbox">
    <b>注意：</b> 仅有 <code>domain=kitchen</code> 记录，无 bathroom 行；左侧无法高亮。请重新跑匹配脚本生成双域结果。
  </div>"""
        print("[warn] 本 trial 无 bathroom 域记录。", file=sys.stderr)

    payload_b = _escape_html_json({"nodes": nodes_b, "edges": edges_b, "meta": meta_b, "highlights": {"nodes": len(hn_b), "edges": len(he_b)}})
    payload_k = _escape_html_json({"nodes": nodes_k, "edges": edges_k, "meta": meta_k, "highlights": {"nodes": len(hn_k), "edges": len(he_k)}})

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 0; }}
    #bar {{ padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-size: 13px; color: #2d3748; }}
    #bar code {{ background: #edf2f7; padding: 2px 6px; border-radius: 4px; }}
    .wrap {{ display: flex; height: calc(100vh - 52px - min(240px, 32vh)); min-height: 200px; }}
    .pane {{ flex: 1; display: flex; flex-direction: column; min-width: 0; }}
    .pane:first-child {{ border-right: 1px solid #e2e8f0; }}
    .hdr {{ padding: 8px 12px; font-weight: 600; font-size: 14px; background: #f7fafc; }}
    .net {{ flex: 1; min-height: 0; }}
    #tables {{ display: flex; max-height: 240px; overflow-y: auto; border-bottom: 1px solid #e2e8f0; font-size: 12px; }}
    #tables .col {{ flex: 1; padding: 8px 10px; min-width: 0; }}
    #tables .col:first-child {{ border-right: 1px solid #e2e8f0; }}
    .mtable {{ width: 100%; border-collapse: collapse; }}
    .mtable th, .mtable td {{ border: 1px solid #e2e8f0; padding: 4px 6px; text-align: left; vertical-align: top; }}
    .mtable th {{ background: #edf2f7; }}
    .muted {{ color: #718096; margin: 4px 0; }}
    .warnbox {{ margin: 0; padding: 10px 14px; background: #fff5f5; border-bottom: 1px solid #feb2b2; font-size: 13px; color: #742a2a; line-height: 1.45; }}
    .warnbox code {{ background: #fed7d7; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
  </style>
</head>
<body>
{warn_html}
  <div id="bar">
    <b>Trial</b> <code>{trial}</code>
    &nbsp;|&nbsp; <b>Goal</b> {html_lib.escape(goal[:220])}
    &nbsp;|&nbsp; 可视化阈值 <b>{vthr:.4f}</b>（仅影响图中红/灰；JSONL 不变）
    &nbsp;|&nbsp; bathroom <b>{nrows_b}</b> 条 → 过阈高亮 <b>{nb_h}</b> 单元 &nbsp;
    kitchen <b>{nrows_k}</b> 条 → <b>{nk_h}</b> 单元
  </div>
  <div id="tables">
    <div class="col">
      <div><b>Bathroom</b> 全部匹配行（均参与聚合；「在图中」指 categorized 图内是否存在该节点/边）</div>
      {table_b}
    </div>
    <div class="col">
      <div><b>Kitchen</b> 全部匹配行</div>
      {table_k}
    </div>
  </div>
  <div class="wrap">
    <div class="pane">
      <div class="hdr">Bathroom 图 (nodes={meta_b.get("node_count")}, edges={meta_b.get("edge_count")})</div>
      <div class="net" id="netB"></div>
    </div>
    <div class="pane">
      <div class="hdr">Kitchen 图 (nodes={meta_k.get("node_count")}, edges={meta_k.get("edge_count")})</div>
      <div class="net" id="netK"></div>
    </div>
  </div>
  <script>
    const DATA_B = {payload_b};
    const DATA_K = {payload_k};

    function makeNetwork(containerId, data) {{
      const el = document.getElementById(containerId);
      const nodes = new vis.DataSet(data.nodes);
      const edges = new vis.DataSet(data.edges);
      const net = new vis.Network(el, {{ nodes, edges }}, {{
        nodes: {{ font: {{ size: 12 }} }},
        edges: {{
          arrows: {{ to: {{ enabled: true, scaleFactor: 0.55 }} }},
          font: {{ align: 'middle', size: 10 }},
          smooth: {{ type: 'dynamic' }}
        }},
        physics: {{
          stabilization: {{ iterations: 120 }},
          barnesHut: {{ gravitationalConstant: -28000, springLength: 110, springConstant: 0.035 }}
        }},
        interaction: {{ hover: true, navigationButtons: true, keyboard: true }}
      }});
      net.once('stabilizationIterationsDone', () => {{ net.setOptions({{ physics: false }}); }});
      return net;
    }}
    makeNetwork('netB', {{ nodes: DATA_B.nodes, edges: DATA_B.edges }});
    makeNetwork('netK', {{ nodes: DATA_K.nodes, edges: DATA_K.edges }});
  </script>
</body>
</html>
"""

    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_html}")
    print(
        f"  bathroom: {nrows_b} 条匹配记录 → 去重高亮 node={len(hn_b)} edge={len(he_b)} "
        f"(vis: nodes={len(nodes_b)} edges={len(edges_b)})"
    )
    print(
        f"  kitchen:  {nrows_k} 条匹配记录 → 去重高亮 node={len(hn_k)} edge={len(he_k)} "
        f"(vis: nodes={len(nodes_k)} edges={len(edges_k)})"
    )


if __name__ == "__main__":
    main()
