#!/usr/bin/env python3
"""Materialize fixed ScienceWorld collaborative-eval subsets (v2, per-room domains).

We define each *initial room* (parsed from the reset observation) as its own domain:
  - hallway__train.json / hallway__test.json
  - bathroom__train.json / bathroom__test.json
  - art_studio__train.json / art_studio__test.json
  - ...

This matches the user's request: "都是各是各的，分出几个房间来" (not a/b).

The output JSON schema is compatible with the existing multidomain collab pipeline
loader (domain__split.json). Rooms are slugified for filenames (spaces -> '_').
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scienceworld import ScienceWorldEnv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "scienceworld" / "collab_subsets" / "v2_room"

_ROOM_RE = re.compile(r"^This room is called the ([^.]+)\.", flags=re.IGNORECASE | re.MULTILINE)


def _task_id_key(task_id: str) -> tuple[int, int, str]:
    parts = str(task_id).split("-")
    try:
        a = int(parts[0])
    except Exception:
        a = 10**9
    try:
        b = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        b = 10**9
    return (a, b, str(task_id))


def extract_initial_room(observation: str) -> str:
    text = str(observation or "").strip()
    if not text:
        return ""
    m = _ROOM_RE.search(text)
    if not m:
        return ""
    return str(m.group(1)).strip().lower()


def slugify_room(room: str) -> str:
    r = str(room or "").strip().lower()
    if not r:
        return "unknown"
    # Keep simple + stable: letters/numbers/underscore only.
    r = r.replace(" ", "_")
    r = re.sub(r"[^a-z0-9_]+", "_", r)
    r = re.sub(r"_+", "_", r).strip("_")
    return r or "unknown"


@dataclass(frozen=True)
class Candidate:
    task_id: str
    task_name: str
    variation_idx: int
    initial_room: str
    task_desc: str

    @property
    def selection_key(self) -> str:
        return f"{self.task_id}|v{self.variation_idx:04d}|{self.initial_room}"


def iter_candidates(
    env: ScienceWorldEnv,
    *,
    max_variations_per_task: int,
    simplification_str: str,
) -> Iterable[Candidate]:
    tasks = sorted(env.tasks.items(), key=lambda x: _task_id_key(x[0]))
    for task_id, task_name in tasks:
        try:
            max_var_total = int(env.getMaxVariations(task_name))
        except Exception:
            continue
        limit = min(max_var_total, int(max_variations_per_task))
        for var in range(limit):
            env.load(
                taskName=str(task_id),
                variationIdx=int(var),
                simplificationStr=str(simplification_str or ""),
                generateGoldPath=False,
            )
            obs, info = env.reset()
            room = extract_initial_room(obs)
            # Use env.get_task_description() for stable text; also keep info['taskDesc'] if present.
            try:
                desc = env.get_task_description()
            except Exception:
                desc = str(info.get("taskDesc", "") or "")
            yield Candidate(
                task_id=str(task_id),
                task_name=str(info.get("taskName", "") or task_name),
                variation_idx=int(info.get("variationIdx", var)),
                initial_room=room,
                task_desc=str(desc or "").strip(),
            )


def pick_balanced(
    candidates: list[Candidate],
    *,
    domain: str,
    train_n: int,
    test_n: int,
) -> tuple[list[Candidate], list[Candidate]]:
    # Stable ordering: task_id major/minor then variation.
    work = sorted(candidates, key=lambda c: (_task_id_key(c.task_id), int(c.variation_idx)))
    picked_train: list[Candidate] = []
    picked_test: list[Candidate] = []
    seen: set[tuple[str, int]] = set()

    def take(dst: list[Candidate], limit: int, pool: list[Candidate]) -> None:
        for c in pool:
            key = (c.task_id, c.variation_idx)
            if key in seen:
                continue
            if domain_from_room(c.initial_room) != domain:
                continue
            dst.append(c)
            seen.add(key)
            if len(dst) >= limit:
                return

    # Train from early variations, test from later variations (disjoint).
    # Use a deterministic split point based on variation_idx.
    early = [c for c in work if c.variation_idx < 1000000]  # placeholder (keeps order)
    late = list(reversed(work))  # take test from the tail for more disjointness

    take(picked_train, train_n, early)
    take(picked_test, test_n, late)
    if len(picked_train) != train_n or len(picked_test) != test_n:
        raise ValueError(
            f"Not enough candidates for {domain}: train {len(picked_train)}/{train_n}, "
            f"test {len(picked_test)}/{test_n}."
        )
    return picked_train, picked_test


def to_row(c: Candidate, *, subset_group: str, split_name: str, selection_rank: int) -> dict[str, Any]:
    return {
        "subset_group": subset_group,
        "source_split": "scienceworld_runtime_v1",
        "selection_policy": "initial_room_per_room_domain_v2",
        "selection_key": c.selection_key,
        "selection_rank": int(selection_rank),
        "sw_task": c.task_id,
        "variation_idx": int(c.variation_idx),
        "simplification_str": "",
        "sw_task_name": c.task_name,
        "sw_scene_room": c.initial_room,
        "sw_task_desc": c.task_desc,
        "env_name": "scienceworld",
        "memco_domain": "scienceworld",
        # Keep split annotation for audit (pipeline uses file name).
        "collab_split": split_name,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Materialize ScienceWorld collab subsets v2 (per-room domains).")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--train_n", type=int, default=100, help="Per-room train size.")
    p.add_argument("--test_n", type=int, default=20, help="Per-room test size.")
    p.add_argument(
        "--rooms",
        type=str,
        default="",
        help=(
            "Optional comma-separated room list (raw names as in observation, e.g. 'hallway,art studio'). "
            "If empty, automatically choose top-K rooms by candidate count."
        ),
    )
    p.add_argument(
        "--top_k_rooms",
        type=int,
        default=6,
        help="When --rooms is empty, choose this many most frequent rooms.",
    )
    p.add_argument(
        "--max_variations_per_task",
        type=int,
        default=80,
        help="Cap per-task variations to keep runtime reasonable (deterministic first K).",
    )
    p.add_argument("--simplification", type=str, default="", help="ScienceWorld simplification string.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ScienceWorldEnv(envStepLimit=2)
    try:
        all_cands = list(
            iter_candidates(
                env,
                max_variations_per_task=int(args.max_variations_per_task),
                simplification_str=str(args.simplification or ""),
            )
        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Audit: show discovered rooms (optional, in manifest).
    room_counts: dict[str, int] = {}
    for c in all_cands:
        room_counts[c.initial_room] = room_counts.get(c.initial_room, 0) + 1

    requested_rooms: list[str] = []
    if str(args.rooms or "").strip():
        requested_rooms = [x.strip().lower() for x in str(args.rooms).split(",") if x.strip()]
    else:
        top_k = max(1, int(args.top_k_rooms))
        ordered = sorted(room_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        requested_rooms = [name for name, _n in ordered if name][:top_k]

    domains = [slugify_room(r) for r in requested_rooms]
    # Keep order but drop duplicates.
    domains = list(dict.fromkeys(domains))

    manifest: dict[str, Any] = {
        "subset_version": "v2_room",
        "selection_policy": "initial_room_per_room_domain_v2",
        "domains": domains,
        "rooms_raw": requested_rooms,
        "max_variations_per_task": int(args.max_variations_per_task),
        "train_n_target_per_domain": int(args.train_n),
        "test_n_target_per_domain": int(args.test_n),
        "discovered_rooms_top": sorted(room_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30],
        "files": {},
    }

    # Build per-room pools once.
    by_domain: dict[str, list[Candidate]] = {}
    for c in all_cands:
        dom = slugify_room(c.initial_room)
        if dom not in domains:
            continue
        by_domain.setdefault(dom, []).append(c)

    for dom in domains:
        pool = by_domain.get(dom, [])
        if not pool:
            raise ValueError(f"No candidates for domain={dom}. Check --rooms / --top_k_rooms.")
        # Deterministic split:
        # - pick test from the tail (up to test_n, or fewer if insufficient)
        # - pick train from the head excluding the chosen test (up to train_n, or fewer)
        work = sorted(pool, key=lambda c: (_task_id_key(c.task_id), int(c.variation_idx)))
        total = len(work)
        test_target = int(args.test_n)
        train_target = int(args.train_n)
        test_n = min(test_target, total)
        # Reserve test first so "use remaining" doesn't wipe out test entirely.
        test_set: set[tuple[str, int]] = set()
        picked_test: list[Candidate] = []
        for c2 in reversed(work):
            key = (c2.task_id, c2.variation_idx)
            if key in test_set:
                continue
            picked_test.append(c2)
            test_set.add(key)
            if len(picked_test) >= test_n:
                break
        picked_test.reverse()

        picked_train: list[Candidate] = []
        for c2 in work:
            key = (c2.task_id, c2.variation_idx)
            if key in test_set:
                continue
            picked_train.append(c2)
            if len(picked_train) >= train_target:
                break

        tr_rows = [
            to_row(c, subset_group=dom, split_name="train", selection_rank=i + 1)
            for i, c in enumerate(picked_train)
        ]
        te_rows = [
            to_row(c, subset_group=dom, split_name="test", selection_rank=i + 1)
            for i, c in enumerate(picked_test)
        ]
        tr_path = out_dir / f"{dom}__train.json"
        te_path = out_dir / f"{dom}__test.json"
        tr_path.write_text(json.dumps(tr_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        te_path.write_text(json.dumps(te_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["files"][tr_path.name] = {
            "domain": dom,
            "split": "train",
            "num_tasks": len(tr_rows),
            "num_candidates_total": total,
            "target": train_target,
        }
        manifest["files"][te_path.name] = {
            "domain": dom,
            "split": "test",
            "num_tasks": len(te_rows),
            "num_candidates_total": total,
            "target": test_target,
        }
        print(f"[ok] wrote {tr_path} ({len(tr_rows)} tasks)", flush=True)
        print(f"[ok] wrote {te_path} ({len(te_rows)} tasks)", flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

