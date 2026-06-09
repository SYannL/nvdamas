#!/usr/bin/env python3
"""Materialize ScienceWorld v4 subsets grouped by task id major number.

Domain definition:
  - Domain "1" contains all tasks whose id starts with "1-" (1-1, 1-2, ...).
  - Domain "2" contains all tasks whose id starts with "2-", and so on.

Sampling definition:
  - For every task template in a domain, take up to --train_per_task official
    train variations.
  - For every task template in a domain, take up to --test_per_task official
    test variations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENCEWORLD_ROOT = REPO_ROOT / "data" / "ScienceWorld"
DEFAULT_OUT_DIR = SCIENCEWORLD_ROOT / "collab_subsets" / "v4_id_grouped"

if str(SCIENCEWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCIENCEWORLD_ROOT))

from scienceworld import ScienceWorldEnv  # noqa: E402


_ROOM_RE = re.compile(r"^This room is called the ([^.]+)\.", flags=re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class RuntimeRow:
    sw_task: str
    sw_task_name: str
    variation_idx: int
    sw_scene_room: str
    sw_task_desc: str


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


def _extract_initial_room(observation: str) -> str:
    match = _ROOM_RE.search(str(observation or "").strip())
    return str(match.group(1)).strip().lower() if match else ""


def _task_id_key(task_id: str) -> tuple[int, int, str]:
    parts = str(task_id).split("-")
    try:
        major = int(parts[0])
    except Exception:
        major = 10**9
    try:
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        minor = 10**9
    return major, minor, str(task_id)


def _task_major(task_id: str) -> str:
    return str(task_id).split("-", 1)[0].strip()


def _get_variations(env: ScienceWorldEnv, split: str, *, max_variations: int) -> list[int]:
    split = str(split or "").strip().lower()
    if split == "train":
        values = list(env.get_variations_train())
    elif split == "dev":
        values = list(env.get_variations_dev())
    elif split == "test":
        values = list(env.get_variations_test())
    elif split == "all":
        values = list(range(int(max_variations)))
    else:
        raise ValueError(f"Unsupported ScienceWorld split: {split!r}")
    return [int(v) for v in values]


def _max_variations(env: ScienceWorldEnv, task_id: str) -> int:
    try:
        return int(env.get_max_variations(task_id))
    except Exception:
        return int(env.getMaxVariations(task_id))


def _load_variation(
    env: ScienceWorldEnv,
    *,
    task_id: str,
    task_name: str,
    variation_idx: int,
    simplification_str: str,
) -> RuntimeRow:
    env.load(
        taskName=str(task_id),
        variationIdx=int(variation_idx),
        simplificationStr=str(simplification_str or ""),
        generateGoldPath=False,
    )
    observation, info = env.reset()
    try:
        desc = env.get_task_description()
    except Exception:
        desc = str(info.get("taskDesc", "") or "")
    return RuntimeRow(
        sw_task=str(task_id),
        sw_task_name=str(info.get("taskName", "") or task_name),
        variation_idx=int(info.get("variationIdx", variation_idx)),
        sw_scene_room=_extract_initial_room(observation),
        sw_task_desc=str(desc or "").strip(),
    )


def _iter_rows_for_task(
    env: ScienceWorldEnv,
    *,
    task_id: str,
    task_name: str,
    split: str,
    per_task_limit: int,
    simplification_str: str,
) -> list[RuntimeRow]:
    max_vars = _max_variations(env, task_id)
    env.load(
        taskName=str(task_id),
        variationIdx=0,
        simplificationStr=str(simplification_str or ""),
        generateGoldPath=False,
    )
    variations = [v for v in _get_variations(env, split, max_variations=max_vars) if 0 <= int(v) < max_vars]
    variations = sorted(dict.fromkeys(int(v) for v in variations))
    if per_task_limit > 0:
        variations = variations[: int(per_task_limit)]
    return [
        _load_variation(
            env,
            task_id=task_id,
            task_name=task_name,
            variation_idx=variation_idx,
            simplification_str=simplification_str,
        )
        for variation_idx in variations
    ]


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sw_task") or ""),
        int(row.get("variation_idx", 0) or 0),
        _slug(str(row.get("sw_scene_room", "") or "")),
    )


def _annotate(
    row: RuntimeRow,
    *,
    domain: str,
    task_name: str,
    role: str,
    rank: int,
    split: str,
) -> dict[str, Any]:
    return {
        "subset_group": domain,
        "source_split": f"scienceworld_runtime_{split}",
        "selection_policy": "id_major_grouped_v4_runtime",
        "selection_rank": int(rank),
        "selection_key": f"{row.sw_task}|v{row.variation_idx:04d}|domain_{domain}|{role}",
        "sw_task": row.sw_task,
        "variation_idx": int(row.variation_idx),
        "simplification_str": "",
        "sw_task_name": row.sw_task_name,
        "sw_scene_room": row.sw_scene_room,
        "sw_task_desc": row.sw_task_desc,
        "env_name": "scienceworld",
        "task_type": "scienceworld",
        "dataset_family": "scienceworld",
        "memco_domain": "scienceworld",
        "scienceworld_domain": domain,
        "scienceworld_family": domain,
        "scienceworld_role": role,
        "scienceworld_task_template": str(task_name),
        "scienceworld_task_group": domain,
        "collab_split": "train" if role == "train" else "test",
    }


def _materialize_domain(
    env: ScienceWorldEnv,
    *,
    domain: str,
    task_items: list[tuple[str, str]],
    train_split: str,
    test_split: str,
    train_per_task: int,
    test_per_task: int,
    simplification_str: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    task_audit: list[dict[str, Any]] = []

    for task_id, task_name in task_items:
        raw_train = _iter_rows_for_task(
            env,
            task_id=task_id,
            task_name=task_name,
            split=train_split,
            per_task_limit=train_per_task,
            simplification_str=simplification_str,
        )
        raw_test = _iter_rows_for_task(
            env,
            task_id=task_id,
            task_name=task_name,
            split=test_split,
            per_task_limit=test_per_task,
            simplification_str=simplification_str,
        )
        train_rows.extend(
            _annotate(row, domain=domain, task_name=task_name, role="train", rank=len(train_rows) + i, split=train_split)
            for i, row in enumerate(raw_train)
        )
        test_rows.extend(
            _annotate(row, domain=domain, task_name=task_name, role="test", rank=len(test_rows) + i, split=test_split)
            for i, row in enumerate(raw_test)
        )
        task_audit.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "num_train": len(raw_train),
                "num_test": len(raw_test),
            }
        )

    audit = {
        "task_templates": [{"task_id": tid, "task_name": name} for tid, name in task_items],
        "per_template_counts": task_audit,
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "train_rooms": sorted({str(row.get("sw_scene_room", "")) for row in train_rows}),
        "test_rooms": sorted({str(row.get("sw_scene_room", "")) for row in test_rows}),
    }
    return train_rows, test_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize ScienceWorld v4 subsets grouped by task id major number.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--domains", type=str, default="", help="Comma-separated major ids. Empty means all.")
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "dev", "test", "all"])
    parser.add_argument("--test_split", type=str, default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--train_per_task", type=int, default=10, help="Train variations per task template; <=0 keeps all.")
    parser.add_argument("--test_per_task", type=int, default=2, help="Test variations per task template; <=0 keeps all.")
    parser.add_argument("--simplification", type=str, default="", help="ScienceWorld simplification string.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = ScienceWorldEnv(envStepLimit=2)
    try:
        groups: dict[str, list[tuple[str, str]]] = {}
        for task_id, task_name in sorted(env.tasks.items(), key=lambda kv: _task_id_key(str(kv[0]))):
            major = _task_major(str(task_id))
            groups.setdefault(major, []).append((str(task_id), str(task_name)))

        requested = [_slug(item) for item in str(args.domains or "").split(",") if item.strip()]
        domains = requested or sorted(groups.keys(), key=lambda item: int(item) if item.isdigit() else 10**9)
        unknown = [domain for domain in domains if domain not in groups]
        if unknown:
            raise ValueError(f"Unknown ScienceWorld major-id domains: {unknown}. Known: {sorted(groups)}")

        manifest: dict[str, Any] = {
            "subset_version": "v4_id_grouped",
            "selection_policy": "id_major_grouped_v4_runtime",
            "source": "original_scienceworld_runtime",
            "scienceworld_root": str(SCIENCEWORLD_ROOT),
            "domains": domains,
            "train_split": str(args.train_split),
            "test_split": str(args.test_split),
            "train_per_task": int(args.train_per_task),
            "test_per_task": int(args.test_per_task),
            "simplification_str": str(args.simplification or ""),
            "files": {},
            "domain_specs": {
                domain: [{"task_id": tid, "task_name": name} for tid, name in groups[domain]]
                for domain in domains
            },
        }

        merged_test: list[dict[str, Any]] = []
        for domain in domains:
            train_rows, test_rows, audit = _materialize_domain(
                env,
                domain=domain,
                task_items=groups[domain],
                train_split=str(args.train_split),
                test_split=str(args.test_split),
                train_per_task=int(args.train_per_task),
                test_per_task=int(args.test_per_task),
                simplification_str=str(args.simplification or ""),
            )
            train_path = out_dir / f"{domain}__train.json"
            test_path = out_dir / f"{domain}__test.json"
            with train_path.open("w", encoding="utf-8") as writer:
                json.dump(train_rows, writer, ensure_ascii=False, indent=2)
            with test_path.open("w", encoding="utf-8") as writer:
                json.dump(test_rows, writer, ensure_ascii=False, indent=2)
            merged_test.extend(test_rows)
            manifest["files"][train_path.name] = {"domain": domain, "split": "train", **audit}
            manifest["files"][test_path.name] = {"domain": domain, "split": "test", **audit}

        merged_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
        for row in merged_test:
            merged_by_key.setdefault(_row_key(row), row)
        merged_rows = list(merged_by_key.values())
        merged_rows.sort(key=lambda row: (int(str(row.get("scienceworld_domain", "0"))), _row_key(row)))
        merged_path = out_dir / "merged__test.json"
        with merged_path.open("w", encoding="utf-8") as writer:
            json.dump(merged_rows, writer, ensure_ascii=False, indent=2)
        manifest["files"][merged_path.name] = {
            "split": "test",
            "num_tasks": len(merged_rows),
            "source_domain_files": [f"{domain}__test.json" for domain in domains],
            "mixed_held_out": True,
        }

        manifest_path = out_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as writer:
            json.dump(manifest, writer, ensure_ascii=False, indent=2)
        print(json.dumps({"out_dir": str(out_dir), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
