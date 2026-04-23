#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 `traj.json`（由 scripts/extract_selectivemem_trajectory.py 生成）中读取每一步的 observation，
使用 SelectiveMem 的本地图抽取 prompt（temp=0）提取 instances + relations，并将结果写回同目录。

默认输出：
- traj.extracted_entities_relations.jsonl  (逐 step 记录)
- traj.extracted_entities_relations.meta.json (统计信息)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mas.llm import GPTChat, Message
from mas.memory.mas_memory.selectivemem import _LOCAL_GRAPH_SYSTEM_PROMPT, _LOCAL_GRAPH_USER_PROMPT


def _parse_local_graph_response(response: str) -> dict[str, Any]:
    if not response:
        return {"instances": [], "relations": []}
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        return {"instances": [], "relations": []}
    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            instances = payload.get("instances") or []
            relations = payload.get("relations") or []
            if not isinstance(instances, list):
                instances = []
            if not isinstance(relations, list):
                relations = []
            return {"instances": instances, "relations": relations}
    except json.JSONDecodeError:
        return {"instances": [], "relations": []}
    return {"instances": [], "relations": []}


def _filter_scene_only_entities(data: dict[str, Any]) -> dict[str, Any]:
    instances = data.get("instances") or []
    relations = data.get("relations") or []
    if not isinstance(instances, list):
        instances = []
    if not isinstance(relations, list):
        relations = []

    banned_ids: set[str] = set()
    filtered_instances: list[dict[str, Any]] = []
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        iid = str(inst.get("instance_id") or "").strip()
        label = str(inst.get("label") or "").strip().lower()
        cat = str(inst.get("category") or "").strip().lower()
        if cat in {"agent", "player"}:
            if iid:
                banned_ids.add(iid)
            continue
        if "agent" in label or "player" in label or label == "you":
            if iid:
                banned_ids.add(iid)
            continue
        if iid.lower().startswith("agent"):
            banned_ids.add(iid)
            continue
        filtered_instances.append(inst)

    filtered_relations: list[dict[str, Any]] = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        subj = str(rel.get("subject") or "").strip()
        obj = str(rel.get("object") or "").strip()
        if subj in banned_ids or obj in banned_ids:
            continue
        r = str(rel.get("relation") or "").strip().lower()
        if "agent" in r or "player" in r:
            continue
        filtered_relations.append(rel)

    return {"instances": filtered_instances, "relations": filtered_relations}


def _iter_traj_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = payload.get("tasks") or []
    if not isinstance(tasks, list):
        return []
    out: list[dict[str, Any]] = []
    for task_idx, t in enumerate(tasks):
        if not isinstance(t, dict):
            continue
        steps = t.get("steps") or []
        if not isinstance(steps, list):
            continue
        task_id = t.get("task_id")
        for st in steps:
            if not isinstance(st, dict):
                continue
            out.append(
                {
                    "task_idx": task_idx,
                    "task_id": task_id,
                    "step": st.get("step"),
                    "act": st.get("act"),
                    "obs": st.get("obs"),
                }
            )
    return out


_TLS = threading.local()


def _get_llm(model_name: str) -> GPTChat:
    llm = getattr(_TLS, "llm", None)
    if llm is None or getattr(llm, "model_name", None) != model_name:
        llm = GPTChat(model_name=model_name)
        _TLS.llm = llm
    return llm


def _extract_one(
    item: dict[str, Any],
    *,
    model_name: str,
) -> dict[str, Any]:
    obs = str(item.get("obs") or "").strip()
    prompt = _LOCAL_GRAPH_USER_PROMPT.format(
        observation=obs,
        room_hint="",
    )
    messages = [
        Message("system", _LOCAL_GRAPH_SYSTEM_PROMPT),
        Message("user", prompt),
    ]
    t0 = time.perf_counter()
    raw: str | None = None
    err: str | None = None
    try:
        llm = _get_llm(model_name)
        raw = llm(messages, temperature=0, max_tokens=512)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    llm_s = time.perf_counter() - t0

    parsed: dict[str, Any] | None = None
    if raw is not None:
        parsed = _filter_scene_only_entities(_parse_local_graph_response(raw))

    return {
        "task_idx": item.get("task_idx"),
        "task_id": item.get("task_id"),
        "step": item.get("step"),
        "act": item.get("act"),
        "obs": obs,
        "llm_s": llm_s,
        "raw": raw,
        "parsed": parsed,
        "error": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract instances/relations from traj.json observations (SelectiveMem prompt, temp=0).")
    ap.add_argument(
        "--traj_json",
        type=str,
        default="logs/alfworld/autogen/memory/selectivemem/gpt-4o-mini/traj.json",
        help="Path to traj.json",
    )
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model name.")
    ap.add_argument("--max_steps", type=int, default=None, help="Optional cap for quick smoke test.")
    ap.add_argument(
        "--skip_ok_obs",
        action="store_true",
        help="Skip observations equal to 'OK.' (often from think steps).",
    )
    ap.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Print progress every N processed steps (default: 20).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads for LLM extraction (default: 1).",
    )
    args = ap.parse_args()

    traj_path = Path(args.traj_json).expanduser().resolve()
    if not traj_path.is_file():
        raise FileNotFoundError(f"traj_json not found: {traj_path}")

    out_dir = traj_path.parent
    out_jsonl = out_dir / "traj.extracted_entities_relations.jsonl"
    out_meta = out_dir / "traj.extracted_entities_relations.meta.json"

    with open(traj_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    steps = _iter_traj_steps(payload if isinstance(payload, dict) else {})

    total = 0
    extracted = 0
    skipped = 0
    parse_empty = 0
    failed_llm = 0
    t0_all = time.perf_counter()

    # Pre-filter steps (skip logic happens before threading).
    cap = min(len(steps), int(args.max_steps)) if args.max_steps is not None else len(steps)
    to_run: list[dict[str, Any]] = []
    for idx, item in enumerate(steps[:cap]):
        total += 1
        obs = str(item.get("obs") or "").strip()
        if not obs:
            skipped += 1
            continue
        if args.skip_ok_obs and obs == "OK.":
            skipped += 1
            continue
        to_run.append(item)

    processed = skipped
    with open(out_jsonl, "w", encoding="utf-8") as wf:
        if int(args.workers) <= 1:
            for item in to_run:
                row = _extract_one(item, model_name=args.model)
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed += 1
                extracted += 1
                if row.get("error"):
                    failed_llm += 1
                parsed = row.get("parsed") or {}
                if not (parsed.get("instances") or parsed.get("relations")):
                    parse_empty += 1
                if args.log_every > 0 and (processed % int(args.log_every) == 0):
                    elapsed = time.perf_counter() - t0_all
                    rate = elapsed / max(1, processed)
                    remaining = max(0, cap - processed)
                    eta_s = remaining * rate
                    print(
                        f"[progress] {processed}/{cap}  extracted={extracted}  skipped={skipped}  failed_llm={failed_llm}  "
                        f"avg_step={rate:.2f}s  ETA={eta_s/60:.1f}m",
                        flush=True,
                    )
        else:
            workers = max(1, int(args.workers))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_extract_one, item, model_name=args.model) for item in to_run]
                for fut in as_completed(futs):
                    row = fut.result()
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    processed += 1
                    extracted += 1
                    if row.get("error"):
                        failed_llm += 1
                    parsed = row.get("parsed") or {}
                    if not (parsed.get("instances") or parsed.get("relations")):
                        parse_empty += 1
                    if args.log_every > 0 and (processed % int(args.log_every) == 0):
                        elapsed = time.perf_counter() - t0_all
                        rate = elapsed / max(1, processed)
                        remaining = max(0, cap - processed)
                        eta_s = remaining * rate
                        print(
                            f"[progress] {processed}/{cap}  extracted={extracted}  skipped={skipped}  failed_llm={failed_llm}  "
                            f"avg_step={rate:.2f}s  ETA={eta_s/60:.1f}m  workers={workers}",
                            flush=True,
                        )

    meta = {
        "traj_json": str(traj_path),
        "output_jsonl": str(out_jsonl),
        "model": args.model,
        "temperature": 0,
        "max_tokens": 512,
        "skip_ok_obs": bool(args.skip_ok_obs),
        "workers": int(args.workers),
        "total_steps_in_file": len(steps),
        "processed_steps": cap,
        "extracted_steps": extracted,
        "skipped_steps": skipped,
        "parsed_empty_steps": parse_empty,
        "wall_s": time.perf_counter() - t0_all,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()

