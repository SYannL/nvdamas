#!/usr/bin/env python3
"""
Visualize LG / GG graphs for a given run stage and export PNGs.

Supports:
- SelectiveMem LG snapshot JSON: local_instance_graph.json / local_category_graph.json
  (schema: {"nodes":[{"id":...}], "edges":[{"from":...,"to":...}]})
- SelectiveMem merged GG JSON: merged_instance_graph.json (node-link format)
- SelectiveMem viz bundles: LG_viz_*.json / GG_viz_*.json (`format: selectivemem_graph_viz_v1`, with `meta`
  e.g. collab_side A/B, collab_scene, episode_index_1based, and `graph` payload for plotting).
  **All** such files in a stage directory are rendered (not only the latest), so you can compare LG/GG over time;
  output PNG basenames include the same stem as the source JSON (timestamp in filename), so snapshots stay aligned.
- Legacy Graphviz DOT (*.dot): optional `dot -Tpng` / matplotlib fallback when JSON is absent

Typical usage (stage directory):
  python scripts/visualize_stage_lg_gg.py \
    --stage_dir logs/alfworld_collab_eval/<run_id>/autogen/memory/selectivemem/gpt-4o-mini/train_local_gg/kitchen \
    --out_dir   logs/alfworld_collab_eval/<run_id>/viz

Collaboration run (A/B in one folder with A_<scene>_ / B_<scene>_ prefixes; LG raw+categorized; GG DOT + JSON):
  python scripts/visualize_stage_lg_gg.py \
    --collab_run_dir logs/alfworld_collab_eval/<run_id>/autogen/memory/selectivemem/gpt-4o-mini \
    --out_dir logs/alfworld_collab_eval/<run_id>/viz_collab

Exactly three figures (recommended for eval_collab + SelectiveMem):
  - A/B: LG from local_category_graph.json (native category-level LG, same as training snapshots)
  - Global: categorized projection of merged_instance_graph.json under .db/.../global/<mas_memory>/
    (the GG rebuilt by rebuild_selectivemem_global_from_locals — what pass2 retrieval searches on)
  python scripts/visualize_stage_lg_gg.py \
    --collab_run_dir logs/alfworld_collab_eval/<run_id>/autogen/memory/selectivemem/gpt-4o-mini \
    --out_dir logs/alfworld_collab_eval/<run_id>/viz_three \
    --collab_three_graphs

If merged_instance_graph.json is missing under .db (e.g. older runs), the viz script can rebuild it
from persisted locals by dynamically loading rebuild_selectivemem_global_from_locals (no edits to training code).
Use --no_rebuild_global to disable.

Or specify files directly:
  python scripts/visualize_stage_lg_gg.py \
    --lg_json path/to/local_instance_graph.json \
    --gg_json path/to/merged_instance_graph.json \
    --out_dir path/to/out
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fname(prefix: str, basename: str) -> str:
    """Prefix output filenames (e.g. A_kitchen_LG.json.png)."""
    return f"{prefix}{basename}" if prefix else basename


def _viz_json_stem_safe(path: Path) -> str:
    """
    Safe basename stem for PNGs, 1:1 with the source LG_viz_*.json / GG_viz_*.json file.
    Filenames from training usually embed time (e.g. ..._20260330_143013), so sorting PNGs ≈ timeline.
    """
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def _viz_bundle_paths(discovered: dict[str, Any], key: str) -> list[Path]:
    """Normalize discovered[key] as a list of existing paths (supports legacy single Path)."""
    raw = discovered.get(key)
    if raw is None:
        return []
    if isinstance(raw, Path):
        return [raw] if raw.exists() else []
    if isinstance(raw, list):
        return [p for p in raw if isinstance(p, Path) and p.exists()]
    return []


def _build_graph_from_nodes_edges(snapshot: dict[str, Any]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for node in snapshot.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id is None:
            continue
        attrs = dict(node)
        attrs.pop("id", None)
        g.add_node(str(node_id), **attrs)
    for edge in snapshot.get("edges", []):
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


def _build_graph_from_node_link(snapshot: dict[str, Any]) -> nx.MultiDiGraph:
    # NetworkX node-link schema key may be "links" (common default) or "edges" (some exports).
    # Choose explicitly; if neither exists, let it fail loudly.
    if isinstance(snapshot, dict):
        if "links" in snapshot:
            edge_key = "links"
        elif "edges" in snapshot:
            edge_key = "edges"
        else:
            raise KeyError("node-link snapshot missing both 'links' and 'edges' keys")
    else:
        edge_key = "links"

    try:
        # NetworkX uses `link=` for the dict key holding edge lists ("links" or "edges").
        g = json_graph.node_link_graph(
            snapshot, link=edge_key, multigraph=True, directed=True
        )
    except TypeError:
        # Older NetworkX: no `link=`; normalize "edges" -> "links" if needed.
        snap = dict(snapshot) if isinstance(snapshot, dict) else snapshot
        if isinstance(snap, dict) and "links" not in snap and "edges" in snap:
            snap = {**snap, "links": snap["edges"]}
        g = json_graph.node_link_graph(snap, multigraph=True, directed=True)
    if not isinstance(g, nx.MultiDiGraph):
        g = nx.MultiDiGraph(g)
    return g


def _infer_category(node_id: str, data: dict[str, Any]) -> str:
    cat = str((data or {}).get("category") or "").strip().lower()
    if cat:
        return cat
    lab = str((data or {}).get("label") or "").strip().lower()
    if lab:
        return lab
    s = str(node_id or "").strip().lower()
    for sep in (":", "_"):
        if sep in s:
            return s.split(sep, 1)[0].strip() or s
    return s


def _to_categorized_multigraph(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Merge nodes by category, but keep ALL edges as MultiDiGraph (do not aggregate).
    """
    out = nx.MultiDiGraph()
    node_cat: dict[str, str] = {}
    for n, data in g.nodes(data=True):
        c = _infer_category(str(n), data or {})
        if not c:
            continue
        node_cat[str(n)] = c
        if c not in out:
            out.add_node(c, category=c, label=c)

    for u, v, _k, data in g.edges(keys=True, data=True):
        cu = node_cat.get(str(u))
        cv = node_cat.get(str(v))
        if not cu or not cv:
            continue
        d = data or {}
        rel = str(d.get("relation") or "").strip()
        if not rel and d.get("relations"):
            for r in (d.get("relations") or [])[:5]:
                rr = str(r).strip()
                if rr:
                    out.add_edge(cu, cv, relation=rr)
            continue
        out.add_edge(cu, cv, relation=rel or "related_to")
    return out


def _downsample_by_degree(g: nx.MultiDiGraph, max_nodes: int) -> nx.MultiDiGraph:
    if max_nodes <= 0 or g.number_of_nodes() <= max_nodes:
        return g
    degrees = sorted(g.degree, key=lambda x: x[1], reverse=True)
    keep = {n for n, _d in degrees[:max_nodes]}
    return g.subgraph(keep).copy()


def _render_nx_to_png(
    g: nx.MultiDiGraph,
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
        raise RuntimeError("matplotlib is required to render JSON graphs to PNG.") from exc

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(g2, seed=42, k=0.9 / max(1, g2.number_of_nodes()) ** 0.5)

    node_labels: dict[str, str] = {}
    for n, data in g2.nodes(data=True):
        d = data or {}
        label = d.get("label") or d.get("category") or n
        cat = d.get("category")
        st = d.get("state")
        parts = [str(label)]
        if cat and str(cat).strip().lower() not in str(label).strip().lower():
            parts.append(f"[{cat}]")
        if st:
            parts.append(f"({st})")
        node_labels[str(n)] = " ".join(parts)

    # Bigger nodes for readability in dense graphs.
    node_size = 4400
    label_font_size = 12
    edge_label_font_size = 10

    nx.draw_networkx_nodes(
        g2,
        pos,
        node_size=node_size,
        node_color="#E8F0FE",
        edgecolors="#5F6368",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(g2, pos, labels=node_labels, font_size=label_font_size)
    nx.draw_networkx_edges(
        g2,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=1.0,
        edge_color="#5F6368",
        alpha=0.7,
    )

    edge_labels: dict[tuple[str, str, int], str] = {}
    for u, v, k, data in g2.edges(keys=True, data=True):
        d = data or {}
        rel = d.get("relation") or ""
        if not rel and d.get("relations"):
            rel = "|".join([str(x) for x in (d.get("relations") or [])[:3]])
        if rel:
            edge_labels[(str(u), str(v), int(k) if isinstance(k, int) else 0)] = str(rel)
    if edge_labels:
        nx.draw_networkx_edge_labels(
            g2,
            pos,
            edge_labels=edge_labels,
            font_size=edge_label_font_size,
            rotate=False,
            label_pos=0.5,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt  # noqa: F401
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def _render_dot_to_png(dot_path: Path, out_png: Path, *, dpi: int) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["dot", "-Tpng", f"-Gdpi={dpi}", str(dot_path), "-o", str(out_png)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Graphviz `dot` not found. Please install graphviz to render .dot files.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="replace")[:800]
        raise RuntimeError(f"Graphviz failed to render dot: {stderr}") from exc


def _read_dot_simple_subset(dot_path: Path) -> nx.MultiDiGraph | None:
    """
    Parse our exported `digraph` DOT (quoted ids, -> edges, optional [label="..."]).
    No pydot required; sufficient for SelectiveMem LG/GG exports.
    """
    try:
        raw = dot_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    g = nx.MultiDiGraph()
    edge_re = re.compile(
        r'^\s*"(?P<u>[^"]+)"\s*->\s*"(?P<v>[^"]+)"\s*(?:\[(?P<attrs>[^\]]*)\]\s*)?;?\s*$'
    )
    node_re = re.compile(r'^\s*"(?P<nid>[^"]+)"\s*\[(?P<attrs>[^\]]*)\]\s*;?\s*$')
    label_re = re.compile(r'label\s*=\s*"((?:\\.|[^"\\])*)"')

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("//"):
            continue
        if s.startswith("digraph") or s.startswith("graph ") or s == "}":
            continue
        if "->" in s:
            m = edge_re.match(s)
            if not m:
                continue
            u, v = m.group("u"), m.group("v")
            attrs_s = m.group("attrs") or ""
            lm = label_re.search(attrs_s)
            rel = ""
            if lm:
                rel = lm.group(1).replace("\\n", " ").strip()
            g.add_node(u)
            g.add_node(v)
            g.add_edge(u, v, relation=rel)
            continue
        m = node_re.match(s)
        if m:
            nid = m.group("nid")
            attrs_s = m.group("attrs") or ""
            lm = label_re.search(attrs_s)
            label = nid
            if lm:
                label = lm.group(1).replace("\\n", " ").strip()
            cat = ""
            if ":" in nid:
                cat = nid.split(":", 1)[0].strip()
            g.add_node(nid, label=label, category=cat or None)

    return g if g.number_of_nodes() else None


def _read_dot_as_multidigraph(dot_path: Path) -> nx.MultiDiGraph | None:
    """Parse a DOT file into MultiDiGraph: try pydot, else simple subset parser."""
    try:
        from networkx.drawing.nx_pydot import read_dot

        g = read_dot(str(dot_path))
    except Exception:
        try:
            import pydot  # noqa: F401
            from networkx.drawing.nx_pydot import from_pydot

            graphs = pydot.graph_from_dot_file(str(dot_path))
            if not graphs:
                g = None
            else:
                g = from_pydot(graphs[0])
        except Exception:
            g = None
    if g is not None and g.number_of_nodes() > 0:
        if not isinstance(g, nx.MultiDiGraph):
            g = nx.MultiDiGraph(g)
        return g
    return _read_dot_simple_subset(dot_path)


def _render_gg_dot_to_matplotlib(
    gg_dot: Path,
    out_dir: Path,
    *,
    filename_prefix: str,
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
    emit_categorized: bool,
) -> bool:
    """
    When merged_instance_graph.json is missing, still draw GG with spring layout (like LG).
    Writes GG_graph.nx.png and GG_graph_categorized.nx.png (optional).
    """
    g = _read_dot_as_multidigraph(gg_dot)
    if g is None:
        print(f"[visualize_stage_lg_gg] GG matplotlib from DOT skipped (could not parse): {gg_dot}")
        return False
    fp = filename_prefix
    _render_nx_to_png(
        g,
        out_dir / _fname(fp, "GG_graph.nx.png"),
        max_nodes=max_nodes,
        dpi=dpi,
        figsize=figsize,
    )
    if emit_categorized:
        g_cat = _to_categorized_multigraph(g)
        _render_nx_to_png(
            g_cat,
            out_dir / _fname(fp, "GG_graph_categorized.nx.png"),
            max_nodes=max_nodes,
            dpi=dpi,
            figsize=figsize,
        )
    return True


def _pick_stage_files(stage_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not stage_dir.exists():
        return out

    lg_dot = sorted(stage_dir.glob("LG_*.dot"))
    gg_dot = sorted(stage_dir.glob("GG_*.dot"))
    if lg_dot:
        out["lg_dot"] = lg_dot[-1]
    if gg_dot:
        out["gg_dot"] = gg_dot[-1]

    lg_viz = sorted(stage_dir.glob("LG_viz*.json"))
    if lg_viz:
        out["lg_viz_json"] = lg_viz
    gg_viz = sorted(stage_dir.glob("GG_viz*.json"))
    if gg_viz:
        out["gg_viz_json"] = gg_viz

    lg_json = stage_dir / "local_instance_graph.json"
    if lg_json.exists():
        out["lg_json"] = lg_json
    cat_json = stage_dir / "local_category_graph.json"
    if cat_json.exists():
        out["lg_category_json"] = cat_json

    merged = stage_dir / "merged_instance_graph.json"
    if merged.exists():
        out["gg_json"] = merged
    return out


def _experiment_run_id_dir_from_memory_run_dir(memory_run_dir: Path) -> Path:
    """
    .../<run_id>/autogen/memory/<mas_memory>/<model>/ -> .../<run_id>/
    """
    p = memory_run_dir.resolve()
    # parents[0]=model, [1]=mas_memory, [2]=memory, [3]=autogen, [4]=run_id
    return p.parents[4]


def _resolve_global_merged_json_from_logs_collab_dir(
    collab_run_dir: Path,
    *,
    mas_memory: str,
) -> Path | None:
    """
    Map logs/.../<eval_ns>/<run_id>/<mas_type>/memory/<mas_memory>/<model_type>/
    to ./.db/<model_type>/<eval_ns>/<run_id>/<mas_type>/memory/<mas_memory>/global/<mas_memory>/merged_instance_graph.json
    (same layout as scripts/eval_collab_domain_adaptation.py base_dir + global + namespace).
    """
    p = collab_run_dir.resolve()
    parts = p.parts
    if len(parts) < 7 or parts[-3] != "memory":
        return None
    model_type = parts[-1]
    mem = parts[-2]
    if mem != mas_memory:
        print(
            f"[visualize_stage_lg_gg] warn: path mas_memory={mem!r} != --mas_memory={mas_memory!r}; "
            "using path segment for DB resolution."
        )
    mas_type = parts[-4]
    run_id = parts[-5]
    eval_ns = parts[-6]
    merged = (
        Path(".db")
        / model_type
        / eval_ns
        / run_id
        / mas_type
        / "memory"
        / mas_memory
        / "global"
        / mas_memory
        / "merged_instance_graph.json"
    )
    return merged.resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_syspath() -> None:
    """Running `python scripts/foo.py` puts `scripts/` on sys.path; `import mas` needs repo root."""
    r = _repo_root()
    if str(r) not in sys.path:
        sys.path.insert(0, str(r))


def _infer_model_name_from_collab_run_dir(collab_run_dir: Path) -> str:
    return collab_run_dir.resolve().name


def _memory_root_from_merged_path(merged_path: Path) -> Path:
    """
    .../autogen/memory/<mas_memory>/global/<mas_memory>/merged_instance_graph.json
    -> .../autogen/memory/<mas_memory>
    """
    # merged: .../global/<namespace>/merged_instance_graph.json
    return merged_path.parent.parent.parent


def _list_scene_names_train_local_gg(collab_run_dir: Path) -> list[str]:
    tlg = collab_run_dir / "train_local_gg"
    if not tlg.is_dir():
        return []
    return sorted([p.name for p in tlg.iterdir() if p.is_dir()], key=lambda x: x.lower())


def _load_rebuild_selectivemem_global_fn():  # type: ignore[no-untyped-def]
    """Load rebuild_selectivemem_global_from_locals without requiring scripts/ to be a package."""
    _ensure_repo_on_syspath()
    repo = _repo_root()
    path = repo / "scripts" / "eval_collab_domain_adaptation.py"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot load rebuild helper: {path}")
    spec = importlib.util.spec_from_file_location("eval_collab_domain_adaptation_viz", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "rebuild_selectivemem_global_from_locals", None)
    if not callable(fn):
        raise AttributeError("rebuild_selectivemem_global_from_locals missing in eval_collab_domain_adaptation")
    return fn


def _ensure_global_merged_json(
    collab_run_dir: Path,
    *,
    mas_memory: str,
    model_name: str,
    allow_rebuild: bool,
) -> Path:
    """
    Return path to merged_instance_graph.json under .db/.../global/<mas_memory>/.
    If missing and allow_rebuild, call the same rebuild as training (merge local persisted merged_lg).
    """
    cand = _resolve_global_merged_json_from_logs_collab_dir(collab_run_dir, mas_memory=mas_memory)
    if cand is None:
        raise SystemExit("Could not map collab_run_dir to a .db global merged path (unexpected layout).")
    if cand.is_file():
        return cand
    if not allow_rebuild:
        raise SystemExit(
            f"Global merged GG missing: {cand}\n"
            "Pass --gg_json, or omit --no_rebuild_global to rebuild from .db/local/<scene>/."
        )
    _ensure_repo_on_syspath()
    scenes = _list_scene_names_train_local_gg(collab_run_dir)
    if not scenes:
        raise SystemExit(f"No scenes under {collab_run_dir / 'train_local_gg'}; cannot rebuild global GG.")
    mem_root = _memory_root_from_merged_path(cand)
    local_dirs = [str(mem_root / "local" / s) for s in scenes]
    global_dir = str(mem_root / "global")
    missing_locals = [d for d in local_dirs if not Path(d).is_dir()]
    if missing_locals:
        raise SystemExit(f"Missing local memory dirs: {missing_locals}; cannot rebuild.")
    print(f"[visualize_stage_lg_gg] rebuilding global merged GG -> {cand}")
    try:
        rebuild = _load_rebuild_selectivemem_global_fn()
        rebuild(
            local_dirs=local_dirs,
            global_dir=global_dir,
            model_name=model_name,
            snapshot_tag="viz_rebuild",
        )
    except Exception as exc:
        raise SystemExit(f"rebuild_selectivemem_global_from_locals failed: {exc}") from exc
    if not cand.is_file():
        raise SystemExit(f"Rebuild finished but merged file still missing: {cand}")
    return cand


def _render_collab_three_graphs(
    collab_run_dir: Path,
    out_dir: Path,
    *,
    mas_memory: str,
    gg_json_override: Path | None,
    model_name: str,
    allow_rebuild_global: bool,
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    """
    Emit exactly three PNGs:
      A_<scene>_LG_categorized.png  — from train_local_gg/<scene>/local_category_graph.json
      B_<scene>_LG_categorized.png
      global_GG_categorized.png     — categorized merged GG used at retrieval (global persist_dir)
    """
    collab_run_dir = collab_run_dir.resolve()
    out_dir = out_dir.resolve()
    tlg = collab_run_dir / "train_local_gg"
    if not tlg.is_dir():
        raise SystemExit(f"train_local_gg not found under {collab_run_dir}")

    if gg_json_override is not None:
        gg_path = gg_json_override
        if not gg_path.is_file():
            raise SystemExit(f"--gg_json not found: {gg_path}")
    else:
        gg_path = _ensure_global_merged_json(
            collab_run_dir,
            mas_memory=mas_memory,
            model_name=model_name,
            allow_rebuild=allow_rebuild_global,
        )

    scene_dirs = sorted(
        [p for p in tlg.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )
    if len(scene_dirs) < 1:
        raise SystemExit(f"No scene subdirs under {tlg}")

    for idx, scene_dir in enumerate(scene_dirs):
        label = chr(ord("A") + idx)
        prefix = f"{label}_{scene_dir.name}_"
        cat_json = scene_dir / "local_category_graph.json"
        if not cat_json.is_file():
            raise SystemExit(f"Missing {cat_json} (native categorized LG snapshot).")
        snap = _read_json(cat_json)
        g = _build_graph_from_nodes_edges(snap)
        _render_nx_to_png(
            g,
            out_dir / _fname(prefix, "LG_categorized.png"),
            max_nodes=max_nodes,
            dpi=dpi,
            figsize=figsize,
        )

    snap_gg = _read_json(gg_path)
    g_raw = _build_graph_from_node_link(snap_gg)
    g_cat = _to_categorized_multigraph(g_raw)
    _render_nx_to_png(
        g_cat,
        out_dir / "global_GG_categorized.png",
        max_nodes=max_nodes,
        dpi=dpi,
        figsize=figsize,
    )
    print(f"[visualize_stage_lg_gg] wrote 3-panel collab figures under {out_dir} (global GG: {gg_path})")


def _render_gg_json_pngs(
    gg_json: Path,
    out_dir: Path,
    *,
    filename_prefix: str = "",
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
    both_raw_and_categorized: bool,
) -> None:
    """Render merged GG from node-link JSON: matplotlib PNG(s)."""
    if not gg_json.exists():
        return
    snap = _read_json(gg_json)
    g_raw = _build_graph_from_node_link(snap)
    if both_raw_and_categorized:
        _render_nx_to_png(
            g_raw,
            out_dir / _fname(filename_prefix, "GG_merged.json.png"),
            max_nodes=max_nodes,
            dpi=dpi,
            figsize=figsize,
        )
        g_cat = _to_categorized_multigraph(g_raw)
        _render_nx_to_png(
            g_cat,
            out_dir / _fname(filename_prefix, "GG_categorized.json.png"),
            max_nodes=max_nodes,
            dpi=dpi,
            figsize=figsize,
        )
    else:
        _render_nx_to_png(
            g_raw,
            out_dir / _fname(filename_prefix, "GG_merged.json.png"),
            max_nodes=max_nodes,
            dpi=dpi,
            figsize=figsize,
        )


def _render_one_discovered_bundle(
    out_dir: Path,
    discovered: dict[str, Any],
    *,
    filename_prefix: str = "",
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
    categorized: bool,
    both: bool,
    lg_json_both: bool = False,
    gg_json_graph_both: bool,
    gg_dot_nx: bool = True,
) -> bool:
    """
    Render everything found under a single stage/global directory into out_dir.
    filename_prefix: e.g. A_kitchen_ so outputs become A_kitchen_LG.json.png.
    If lg_json_both: always emit LG.json + LG_categorized (and LG_category + multiedge when present).
    If gg_json_graph_both: for merged_instance_graph.json, emit raw + categorized matplotlib PNGs
    (in addition to any DOT rendering).
    Every LG_viz*.json / GG_viz*.json snapshot in the directory is rendered; PNG names use the same stem
    as the JSON file so you can compare training dynamics at different times.
    If gg_dot_nx and there is GG.dot but no merged JSON, parse DOT with pydot and emit GG_graph.nx.png
    (+ categorized) so GG is still drawn like LG.
    Returns True if anything was rendered.
    """
    fp = filename_prefix
    lg_both = bool(both) or bool(lg_json_both)
    any_out = False
    lg_dot = discovered.get("lg_dot")
    gg_dot = discovered.get("gg_dot")
    lg_json = discovered.get("lg_json")
    lg_cat_json = discovered.get("lg_category_json")
    gg_json = discovered.get("gg_json")
    lg_viz_paths = _viz_bundle_paths(discovered, "lg_viz_json")
    gg_viz_paths = _viz_bundle_paths(discovered, "gg_viz_json")

    if lg_dot:
        _render_dot_to_png(lg_dot, out_dir / _fname(fp, "LG.dot.png"), dpi=dpi)
        any_out = True
    if gg_dot:
        _render_dot_to_png(gg_dot, out_dir / _fname(fp, "GG.dot.png"), dpi=dpi)
        any_out = True
        if gg_dot_nx and not (gg_json and gg_json.exists()):
            _render_gg_dot_to_matplotlib(
                gg_dot,
                out_dir,
                filename_prefix=fp,
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
                emit_categorized=bool(gg_json_graph_both) or bool(both),
            )

    if lg_json and lg_json.exists():
        snap = _read_json(lg_json)
        g = _build_graph_from_nodes_edges(snap)
        if lg_both:
            _render_nx_to_png(g, out_dir / _fname(fp, "LG.json.png"), max_nodes=max_nodes, dpi=dpi, figsize=figsize)
            g2 = _to_categorized_multigraph(g)
            _render_nx_to_png(
                g2,
                out_dir / _fname(fp, "LG_categorized.json.png"),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
        else:
            if categorized:
                g = _to_categorized_multigraph(g)
            out_name = "LG_categorized.json.png" if categorized else "LG.json.png"
            _render_nx_to_png(g, out_dir / _fname(fp, out_name), max_nodes=max_nodes, dpi=dpi, figsize=figsize)
        any_out = True

    for lg_viz_json in lg_viz_paths:
        snap_bundle = _read_json(lg_viz_json)
        if snap_bundle.get("format") != "selectivemem_graph_viz_v1" or not isinstance(
            snap_bundle.get("graph"), dict
        ):
            print(f"[visualize_stage_lg_gg] skip LG viz (bad format): {lg_viz_json}")
            continue
        stem = _viz_json_stem_safe(lg_viz_json)
        g_v = _build_graph_from_nodes_edges(snap_bundle["graph"])
        if lg_both:
            _render_nx_to_png(
                g_v,
                out_dir / _fname(fp, f"{stem}.LG.png"),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
            g2v = _to_categorized_multigraph(g_v)
            _render_nx_to_png(
                g2v,
                out_dir / _fname(fp, f"{stem}.LG.categorized.png"),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
        else:
            if categorized:
                g_v = _to_categorized_multigraph(g_v)
            out_nm = f"{stem}.LG.categorized.png" if categorized else f"{stem}.LG.png"
            _render_nx_to_png(
                g_v,
                out_dir / _fname(fp, out_nm),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
        any_out = True

    if lg_cat_json and lg_cat_json.exists():
        snap = _read_json(lg_cat_json)
        g = _build_graph_from_nodes_edges(snap)
        if lg_both:
            _render_nx_to_png(
                g,
                out_dir / _fname(fp, "LG_category.json.png"),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
            g2 = _to_categorized_multigraph(g)
            _render_nx_to_png(
                g2,
                out_dir / _fname(fp, "LG_category.multiedge.png"),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
        else:
            if categorized:
                g = _to_categorized_multigraph(g)
            out_name = "LG_category.multiedge.png" if categorized else "LG_category.json.png"
            _render_nx_to_png(g, out_dir / _fname(fp, out_name), max_nodes=max_nodes, dpi=dpi, figsize=figsize)
        any_out = True

    if gg_json and gg_json.exists():
        if gg_json_graph_both or both:
            _render_gg_json_pngs(
                gg_json,
                out_dir,
                filename_prefix=fp,
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
                both_raw_and_categorized=True,
            )
        else:
            snap = _read_json(gg_json)
            g_raw = _build_graph_from_node_link(snap)
            if categorized:
                g_use = _to_categorized_multigraph(g_raw)
                out_name = "GG_categorized.json.png"
            else:
                g_use = g_raw
                out_name = "GG_merged.json.png"
            _render_nx_to_png(g_use, out_dir / _fname(fp, out_name), max_nodes=max_nodes, dpi=dpi, figsize=figsize)
        any_out = True

    for gg_viz_json in gg_viz_paths:
        g_bundle = _read_json(gg_viz_json)
        if g_bundle.get("format") != "selectivemem_graph_viz_v1" or not isinstance(
            g_bundle.get("graph"), dict
        ):
            print(f"[visualize_stage_lg_gg] skip GG viz (bad format): {gg_viz_json}")
            continue
        stem = _viz_json_stem_safe(gg_viz_json)
        g_raw_v = _build_graph_from_node_link(g_bundle["graph"])
        if gg_json_graph_both or both:
            _render_nx_to_png(
                g_raw_v,
                out_dir / _fname(fp, f"{stem}.GG.merged.png"),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
            g_cat_v = _to_categorized_multigraph(g_raw_v)
            _render_nx_to_png(
                g_cat_v,
                out_dir / _fname(fp, f"{stem}.GG.categorized.png"),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
        else:
            if categorized:
                g_use_v = _to_categorized_multigraph(g_raw_v)
                out_nm = f"{stem}.GG.categorized.png"
            else:
                g_use_v = g_raw_v
                out_nm = f"{stem}.GG.merged.png"
            _render_nx_to_png(
                g_use_v,
                out_dir / _fname(fp, out_nm),
                max_nodes=max_nodes,
                dpi=dpi,
                figsize=figsize,
            )
        any_out = True

    return any_out


def _render_collab_bundle(
    collab_run_dir: Path,
    out_dir: Path,
    *,
    mas_memory: str,
    model_name: str,
    allow_rebuild_global: bool,
    max_nodes: int,
    dpi: int,
    figsize: tuple[float, float],
    categorized: bool,
    both: bool,
    gg_dot_nx: bool,
) -> None:
    """
    For eval_collab layout: <collab_run_dir>/train_local_gg/<scene>/ (A,B,...)
    Global GG JSON is resolved from ./.db/.../global/<mas_memory>/merged_instance_graph.json
    (same tree as eval_collab_domain_adaptation), not from under logs/.
    """
    collab_run_dir = collab_run_dir.resolve()
    out_dir = out_dir.resolve()
    tlg = collab_run_dir / "train_local_gg"
    if not tlg.is_dir():
        raise SystemExit(f"train_local_gg not found under {collab_run_dir}")

    global_merged_db = _resolve_global_merged_json_from_logs_collab_dir(collab_run_dir, mas_memory=mas_memory)
    exp_run = _experiment_run_id_dir_from_memory_run_dir(collab_run_dir)
    global_mem_dir_legacy = exp_run / "global" / mas_memory

    rendered_any = False
    scene_dirs = sorted(
        [p for p in tlg.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )
    for idx, scene_dir in enumerate(scene_dirs):
        label = chr(ord("A") + idx)
        prefix = f"{label}_{scene_dir.name}_"
        disc = _pick_stage_files(scene_dir)
        ok = _render_one_discovered_bundle(
            out_dir,
            disc,
            filename_prefix=prefix,
            max_nodes=max_nodes,
            dpi=dpi,
            figsize=figsize,
            categorized=categorized,
            both=both,
            lg_json_both=True,
            gg_json_graph_both=True,
            gg_dot_nx=gg_dot_nx,
        )
        if ok:
            rendered_any = True

    global_disc: dict[str, Path] = {}
    gg_resolved: Path | None = None
    if global_merged_db:
        if global_merged_db.is_file():
            gg_resolved = global_merged_db
        elif allow_rebuild_global:
            gg_resolved = _ensure_global_merged_json(
                collab_run_dir,
                mas_memory=mas_memory,
                model_name=model_name,
                allow_rebuild=True,
            )
        else:
            print(
                f"[visualize_stage_lg_gg] warn: global merged missing at {global_merged_db}; "
                f"trying legacy {global_mem_dir_legacy} (use default rebuild or remove --no_rebuild_global)"
            )
    if gg_resolved is not None:
        global_disc["gg_json"] = gg_resolved
    else:
        global_disc = _pick_stage_files(global_mem_dir_legacy)

    if global_disc:
        ok_g = _render_one_discovered_bundle(
            out_dir,
            global_disc,
            filename_prefix="global_",
            max_nodes=max_nodes,
            dpi=dpi,
            figsize=figsize,
            categorized=categorized,
            both=both,
            lg_json_both=False,
            gg_json_graph_both=True,
            gg_dot_nx=gg_dot_nx,
        )
        if ok_g:
            rendered_any = True
    else:
        print(
            f"[visualize_stage_lg_gg] skip global: no merged_instance_graph.json at {global_merged_db} "
            f"and no legacy dir {global_mem_dir_legacy}"
        )

    if not rendered_any:
        raise SystemExit(
            f"No graphs rendered under {tlg}. "
            f"Ensure merged GG exists at {global_merged_db} (run eval_collab merge), "
            f"or use --collab_three_graphs / --gg_json."
        )


def main() -> None:
    _ensure_repo_on_syspath()
    p = argparse.ArgumentParser()
    p.add_argument("--stage_dir", type=str, default="", help="Directory to auto-discover LG/GG files.")
    p.add_argument(
        "--collab_run_dir",
        type=str,
        default="",
        help="Collaboration memory root: .../autogen/memory/<mas>/<model>/. Renders train_local_gg/<A,B,...> and global/<mas_memory>/GG.",
    )
    p.add_argument(
        "--mas_memory",
        type=str,
        default="selectivemem",
        help="Namespace folder under global/ (merged_instance_graph.json lives at .../global/<mas_memory>/).",
    )
    p.add_argument(
        "--collab_three_graphs",
        action="store_true",
        help="With --collab_run_dir: emit only A/B LG_categorized.png from local_category_graph.json "
        "and global_GG_categorized.png from .db/.../merged_instance_graph.json (retrieval GG).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="LLM id for rebuilding global GG if missing (default: infer from collab dir name, e.g. gpt-4o-mini).",
    )
    p.add_argument(
        "--no_rebuild_global",
        action="store_true",
        help="Do not call rebuild_selectivemem_global_from_locals when merged_instance_graph.json is missing.",
    )
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for PNGs.")
    p.add_argument("--lg_json", type=str, default="", help="Explicit LG JSON (nodes/edges schema).")
    p.add_argument("--gg_json", type=str, default="", help="Explicit GG JSON (node-link schema).")
    p.add_argument("--lg_dot", type=str, default="", help="Explicit LG DOT file.")
    p.add_argument("--gg_dot", type=str, default="", help="Explicit GG DOT file.")
    p.add_argument("--max_nodes", type=int, default=120, help="Downsample JSON graphs for readability.")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI (higher => clearer, larger files).")
    p.add_argument("--figsize", type=str, default="22,16", help="Matplotlib figure size, e.g. '22,16'.")
    p.add_argument("--categorized", action="store_true", help="Render categorized graphs (merge nodes by category).")
    p.add_argument("--both", action="store_true", help="Render both raw and categorized versions when possible.")
    p.add_argument(
        "--gg_json_graph_both",
        action="store_true",
        help="For merged_instance_graph.json: also emit GG_categorized.json.png (default on in --collab_run_dir).",
    )
    p.add_argument(
        "--lg_json_both",
        action="store_true",
        help="Emit LG.json + LG_categorized (and LG_category + multiedge when present). On by default for --collab_run_dir.",
    )
    p.add_argument(
        "--no_gg_dot_nx",
        action="store_true",
        help="Do not build matplotlib GG from GG_*.dot when merged_instance_graph.json is missing (requires pydot).",
    )
    args = p.parse_args()

    try:
        w, h = args.figsize.split(",", 1)
        figsize = (float(w.strip()), float(h.strip()))
    except Exception:
        figsize = (22.0, 16.0)

    out_dir = Path(args.out_dir).resolve()

    if args.collab_run_dir:
        cr = Path(args.collab_run_dir)
        model_name = (args.model or "").strip() or _infer_model_name_from_collab_run_dir(cr)
        allow_rebuild = not bool(args.no_rebuild_global)
        if args.collab_three_graphs:
            gg_ov = Path(args.gg_json).resolve() if args.gg_json else None
            _render_collab_three_graphs(
                cr,
                out_dir,
                mas_memory=args.mas_memory,
                gg_json_override=gg_ov,
                model_name=model_name,
                allow_rebuild_global=allow_rebuild,
                max_nodes=int(args.max_nodes),
                dpi=int(args.dpi),
                figsize=figsize,
            )
            return
        _render_collab_bundle(
            cr,
            out_dir,
            mas_memory=args.mas_memory,
            model_name=model_name,
            allow_rebuild_global=allow_rebuild,
            max_nodes=int(args.max_nodes),
            dpi=int(args.dpi),
            figsize=figsize,
            categorized=bool(args.categorized),
            both=bool(args.both),
            gg_dot_nx=not bool(args.no_gg_dot_nx),
        )
        return

    stage_dir = Path(args.stage_dir).resolve() if args.stage_dir else None
    discovered: dict[str, Any] = dict(_pick_stage_files(stage_dir)) if stage_dir else {}

    lg_dot = Path(args.lg_dot).resolve() if args.lg_dot else discovered.get("lg_dot")
    gg_dot = Path(args.gg_dot).resolve() if args.gg_dot else discovered.get("gg_dot")
    lg_json = Path(args.lg_json).resolve() if args.lg_json else discovered.get("lg_json")
    lg_cat_json = discovered.get("lg_category_json")
    gg_json = Path(args.gg_json).resolve() if args.gg_json else discovered.get("gg_json")

    if lg_dot:
        discovered["lg_dot"] = lg_dot
    if gg_dot:
        discovered["gg_dot"] = gg_dot
    if lg_json:
        discovered["lg_json"] = lg_json
    if gg_json:
        discovered["gg_json"] = gg_json
    if lg_cat_json:
        discovered["lg_category_json"] = lg_cat_json

    gg_both = bool(args.both) or bool(args.gg_json_graph_both)
    ok = _render_one_discovered_bundle(
        out_dir,
        discovered,
        filename_prefix="",
        max_nodes=int(args.max_nodes),
        dpi=int(args.dpi),
        figsize=figsize,
        categorized=bool(args.categorized),
        both=bool(args.both),
        lg_json_both=bool(args.lg_json_both),
        gg_json_graph_both=gg_both,
        gg_dot_nx=not bool(args.no_gg_dot_nx),
    )

    if not ok:
        stage = str(stage_dir) if stage_dir else "(none)"
        raise SystemExit(
            "No LG/GG files found.\n"
            f"- stage_dir={stage}\n"
            "Provide --stage_dir, --collab_run_dir, or explicit --lg_json/--gg_json/--lg_dot/--gg_dot."
        )


if __name__ == "__main__":
    main()

