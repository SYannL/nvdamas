#!/usr/bin/env python3
"""Materialize ScienceWorld family-domain subsets from the original runtime.

This script intentionally does not reuse v2_room.  It launches the bundled
ScienceWorld simulator, enumerates the original task variations, and writes the
multidomain files used by the MemCo collab pipeline:

  - {domain}__train.json: source task family variations from the original train fold
  - {domain}__test.json: target task family variations from the original test fold
  - merged__test.json: mixed held-out target set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENCEWORLD_ROOT = REPO_ROOT / "data" / "ScienceWorld"
DEFAULT_OUT_DIR = SCIENCEWORLD_ROOT / "collab_subsets" / "v3_family_mixed"

if str(SCIENCEWORLD_ROOT) not in sys.path:
    sys.path.insert(0, str(SCIENCEWORLD_ROOT))

from scienceworld import ScienceWorldEnv  # noqa: E402


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


def _task_name_to_id(env: ScienceWorldEnv) -> dict[str, str]:
    out: dict[str, str] = {}
    for task_id, task_name in sorted(env.tasks.items(), key=lambda kv: _task_id_key(str(kv[0]))):
        out[str(task_name)] = str(task_id)
    return out


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


def _max_variations(env: ScienceWorldEnv, task_name: str) -> int:
    try:
        return int(env.get_max_variations(task_name))
    except Exception:
        return int(env.getMaxVariations(task_name))


def _load_variation(
    env: ScienceWorldEnv,
    *,
    task_id: str,
    task_name: str,
    variation_idx: int,
    simplification_str: str,
) -> RuntimeRow:
    env.load(
        taskName=task_id,
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
    limit: int,
    simplification_str: str,
) -> Iterable[RuntimeRow]:
    max_vars = _max_variations(env, task_name)
    # Load once before asking the Java side for split-specific variation ids.
    env.load(
        taskName=task_id,
        variationIdx=0,
        simplificationStr=str(simplification_str or ""),
        generateGoldPath=False,
    )
    variations = [v for v in _get_variations(env, split, max_variations=max_vars) if 0 <= int(v) < max_vars]
    variations = sorted(dict.fromkeys(int(v) for v in variations))
    if limit > 0:
        variations = variations[: int(limit)]
    for variation_idx in variations:
        yield _load_variation(
            env,
            task_id=task_id,
            task_name=task_name,
            variation_idx=variation_idx,
            simplification_str=simplification_str,
        )


def _row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("sw_task") or ""),
        int(row.get("variation_idx", 0) or 0),
        _slug(str(row.get("sw_scene_room", "") or "")),
    )


def _annotate(row: RuntimeRow, *, domain: str, role: str, rank: int, split: str) -> dict[str, Any]:
    return {
        "subset_group": domain,
        "source_split": f"scienceworld_runtime_{split}",
        "selection_policy": "family_source_to_mixed_target_v3_runtime",
        "selection_rank": int(rank),
        "selection_key": f"{row.sw_task}|v{row.variation_idx:04d}|{domain}|{role}",
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
        "collab_split": "train" if role == "source_train" else "test",
    }


def _materialize_domain(
    env: ScienceWorldEnv,
    *,
    task_name_to_id: dict[str, str],
    domain: str,
    train_split: str,
    test_split: str,
    train_n: int,
    test_n: int,
    simplification_str: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec = FAMILY_SPECS[domain]
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"source_task_names": spec["source"], "target_task_names": spec["target"]}

    for task_name in spec["source"]:
        task_id = task_name_to_id.get(task_name)
        if not task_id:
            raise KeyError(f"ScienceWorld source task not found: {task_name}")
        per_task_limit = train_n - len(train_rows) if train_n > 0 else 0
        rows = list(
            _iter_rows_for_task(
                env,
                task_id=task_id,
                task_name=task_name,
                split=train_split,
                limit=per_task_limit,
                simplification_str=simplification_str,
            )
        )
        train_rows.extend(
            _annotate(row, domain=domain, role="source_train", rank=len(train_rows) + i, split=train_split)
            for i, row in enumerate(rows)
        )
        if train_n > 0 and len(train_rows) >= train_n:
            break

    for task_name in spec["target"]:
        task_id = task_name_to_id.get(task_name)
        if not task_id:
            raise KeyError(f"ScienceWorld target task not found: {task_name}")
        per_task_limit = test_n - len(test_rows) if test_n > 0 else 0
        rows = list(
            _iter_rows_for_task(
                env,
                task_id=task_id,
                task_name=task_name,
                split=test_split,
                limit=per_task_limit,
                simplification_str=simplification_str,
            )
        )
        test_rows.extend(
            _annotate(row, domain=domain, role="target_test", rank=len(test_rows) + i, split=test_split)
            for i, row in enumerate(rows)
        )
        if test_n > 0 and len(test_rows) >= test_n:
            break

    audit["num_train"] = len(train_rows)
    audit["num_test"] = len(test_rows)
    audit["train_rooms"] = sorted({str(row.get("sw_scene_room", "")) for row in train_rows})
    audit["test_rooms"] = sorted({str(row.get("sw_scene_room", "")) for row in test_rows})
    return train_rows, test_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize ScienceWorld runtime family-domain subsets.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join(FAMILY_SPECS.keys()),
        help="Comma-separated domain names from the built-in family specs.",
    )
    parser.add_argument("--train_split", type=str, default="train", choices=["train", "dev", "test", "all"])
    parser.add_argument("--test_split", type=str, default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--train_n", type=int, default=60, help="Per-domain source train cap; <=0 keeps all split rows.")
    parser.add_argument("--test_n", type=int, default=40, help="Per-domain target test cap; <=0 keeps all split rows.")
    parser.add_argument("--simplification", type=str, default="", help="ScienceWorld simplification string.")
    args = parser.parse_args()

    domains = [_slug(item) for item in str(args.domains or "").split(",") if item.strip()]
    unknown = [domain for domain in domains if domain not in FAMILY_SPECS]
    if unknown:
        raise ValueError(f"Unknown ScienceWorld domains: {unknown}. Known: {sorted(FAMILY_SPECS)}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "subset_version": "v3_family_mixed",
        "selection_policy": "family_source_to_mixed_target_v3_runtime",
        "source": "original_scienceworld_runtime",
        "scienceworld_root": str(SCIENCEWORLD_ROOT),
        "domains": domains,
        "train_split": str(args.train_split),
        "test_split": str(args.test_split),
        "train_n_cap": int(args.train_n),
        "test_n_cap": int(args.test_n),
        "simplification_str": str(args.simplification or ""),
        "files": {},
        "family_specs": FAMILY_SPECS,
    }

    env = ScienceWorldEnv(envStepLimit=2)
    try:
        task_name_to_id = _task_name_to_id(env)
        merged_test: list[dict[str, Any]] = []
        for domain in domains:
            train_rows, test_rows, audit = _materialize_domain(
                env,
                task_name_to_id=task_name_to_id,
                domain=domain,
                train_split=str(args.train_split),
                test_split=str(args.test_split),
                train_n=int(args.train_n),
                test_n=int(args.test_n),
                simplification_str=str(args.simplification or ""),
            )
            if not train_rows:
                raise ValueError(f"No runtime source train rows selected for domain={domain}: {FAMILY_SPECS[domain]['source']}")
            if not test_rows:
                raise ValueError(f"No runtime target test rows selected for domain={domain}: {FAMILY_SPECS[domain]['target']}")

            train_path = out_dir / f"{domain}__train.json"
            test_path = out_dir / f"{domain}__test.json"
            with train_path.open("w", encoding="utf-8") as writer:
                json.dump(train_rows, writer, ensure_ascii=False, indent=2)
            with test_path.open("w", encoding="utf-8") as writer:
                json.dump(test_rows, writer, ensure_ascii=False, indent=2)
            merged_test.extend(test_rows)
            manifest["files"][train_path.name] = {"domain": domain, "split": "train", **audit}
            manifest["files"][test_path.name] = {
                "domain": domain,
                "split": "test",
                "num_tasks": len(test_rows),
                "target_task_names": FAMILY_SPECS[domain]["target"],
            }

        merged_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
        for row in merged_test:
            merged_by_key.setdefault(_row_key(row), row)
        merged_rows = list(merged_by_key.values())
        merged_rows.sort(key=lambda row: (str(row.get("scienceworld_domain", "")), _row_key(row)))
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
