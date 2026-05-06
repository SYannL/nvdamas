#!/usr/bin/env python3
"""Materialize ScienceWorld_2 family-domain mixed held-out subsets.

This keeps the multidomain eval shape used by GM3:
  - each domain has its own local train split;
  - all local memories promote into one shared global memory;
  - each domain's local+global memory is evaluated on the same mixed held-out test.

The script reuses an existing ScienceWorld subset pool (by default v2_room) and
does not depend on launching the ScienceWorld simulator.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "ScienceWorld" / "collab_subsets" / "v2_room"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "ScienceWorld" / "collab_subsets" / "v3_family_mixed"


FAMILY_SPECS: dict[str, dict[str, list[str]]] = {
    "conductivity": {
        "source": ["test-conductivity"],
        "target": ["test-conductivity-of-unknown-substances"],
    },
    "melting_point": {
        "source": ["measure-melting-point-known-substance"],
        "target": ["measure-melting-point-unknown-substance"],
    },
    "friction": {
        "source": ["inclined-plane-friction-named-surfaces"],
        "target": ["inclined-plane-friction-unnamed-surfaces"],
    },
    "genetics": {
        "source": ["mendelian-genetics-known-plant"],
        "target": ["mendelian-genetics-unknown-plant"],
    },
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sw_task") or row.get("task_id") or ""),
        int(row.get("variation_idx", row.get("variationIdx", 0)) or 0),
        _slug(str(row.get("sw_scene_room", "") or "")),
    )


def _load_pool(source_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*__train.json")) + sorted(source_dir.glob("*__test.json")):
        with path.open("r", encoding="utf-8") as reader:
            payload = json.load(reader)
        if not isinstance(payload, list):
            raise ValueError(f"ScienceWorld subset file must be a JSON list: {path}")
        for item in payload:
            if not isinstance(item, dict):
                continue
            row = copy.deepcopy(item)
            row.setdefault("source_subset_file", str(path))
            rows.append(row)
    deduped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(_row_key(row), row)
    return list(deduped.values())


def _select_rows(
    pool: list[dict[str, Any]],
    *,
    task_names: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    wanted = {_slug(name) for name in task_names}
    selected = [
        copy.deepcopy(row)
        for row in pool
        if _slug(str(row.get("sw_task_name", "") or "")) in wanted
    ]
    selected.sort(key=lambda row: (_slug(str(row.get("sw_task_name", "") or "")), _row_key(row)))
    if limit > 0:
        selected = selected[:limit]
    return selected


def _annotate(row: dict[str, Any], *, domain: str, role: str, rank: int) -> dict[str, Any]:
    item = copy.deepcopy(row)
    item["subset_group"] = domain
    item["source_split"] = "scienceworld_2_family_mixed"
    item["selection_policy"] = "family_source_to_mixed_target_v3"
    item["selection_rank"] = int(rank)
    item["env_name"] = "scienceworld"
    item["task_type"] = "scienceworld_2"
    item["dataset_family"] = "scienceworld_2"
    item["gm3_domain"] = "scienceworld"
    item["scienceworld2_domain"] = domain
    item["scienceworld2_family"] = domain
    item["scienceworld2_role"] = role
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize ScienceWorld_2 family-domain mixed subsets.")
    parser.add_argument("--source_dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join(FAMILY_SPECS.keys()),
        help="Comma-separated domain names from the built-in family specs.",
    )
    parser.add_argument("--train_n", type=int, default=0, help="Per-domain train cap; <=0 keeps all source rows.")
    parser.add_argument("--test_n", type=int, default=0, help="Per-domain target cap; <=0 keeps all target rows.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = [_slug(item) for item in str(args.domains or "").split(",") if item.strip()]
    unknown = [domain for domain in domains if domain not in FAMILY_SPECS]
    if unknown:
        raise ValueError(f"Unknown ScienceWorld_2 domains: {unknown}. Known: {sorted(FAMILY_SPECS)}")

    pool = _load_pool(source_dir)
    manifest: dict[str, Any] = {
        "subset_version": "v3_family_mixed",
        "selection_policy": "family_source_to_mixed_target_v3",
        "source_dir": str(source_dir),
        "domains": domains,
        "train_n_cap": int(args.train_n),
        "test_n_cap": int(args.test_n),
        "files": {},
        "family_specs": FAMILY_SPECS,
    }

    merged_test: list[dict[str, Any]] = []
    for domain in domains:
        spec = FAMILY_SPECS[domain]
        train_rows = [
            _annotate(row, domain=domain, role="source_train", rank=i)
            for i, row in enumerate(
                _select_rows(pool, task_names=spec["source"], limit=int(args.train_n))
            )
        ]
        test_rows = [
            _annotate(row, domain=domain, role="target_test", rank=i)
            for i, row in enumerate(
                _select_rows(pool, task_names=spec["target"], limit=int(args.test_n))
            )
        ]
        if not train_rows:
            raise ValueError(f"No source train rows selected for domain={domain}: {spec['source']}")
        if not test_rows:
            raise ValueError(f"No target test rows selected for domain={domain}: {spec['target']}")

        train_path = out_dir / f"{domain}__train.json"
        test_path = out_dir / f"{domain}__test.json"
        with train_path.open("w", encoding="utf-8") as writer:
            json.dump(train_rows, writer, ensure_ascii=False, indent=2)
        with test_path.open("w", encoding="utf-8") as writer:
            json.dump(test_rows, writer, ensure_ascii=False, indent=2)
        merged_test.extend(test_rows)
        manifest["files"][train_path.name] = {
            "domain": domain,
            "split": "train",
            "num_tasks": len(train_rows),
            "source_task_names": spec["source"],
        }
        manifest["files"][test_path.name] = {
            "domain": domain,
            "split": "test",
            "num_tasks": len(test_rows),
            "target_task_names": spec["target"],
        }

    merged_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in merged_test:
        merged_by_key.setdefault(_row_key(row), row)
    merged_rows = list(merged_by_key.values())
    merged_rows.sort(key=lambda row: (str(row.get("scienceworld2_domain", "")), _row_key(row)))
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


if __name__ == "__main__":
    main()
