#!/usr/bin/env python3
"""Materialize ALFWorld collaborative-eval subsets v3: split by scene domain.

v2 groups were limited to two coarse categories (kitchen_statechange vs home_search).
In v3 we split by ALFRED scene domains:
  - kitchen
  - living
  - bedroom
  - bathroom

For each (scene_domain, split), we put *all* game.tw-pddl under json_2.1.1 for that
domain into a single JSON file.

Default splits are chosen to keep file count close to v2:
  - train
  - valid_unseen
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "alfworld" / "json_2.1.1"
OUTPUT_DIR = REPO_ROOT / "data" / "alfworld" / "collab_subsets" / "v3"

DEFAULT_SPLITS = ("train", "valid_unseen")
DOMAIN_ORDER = ("kitchen", "living", "bedroom", "bathroom", "other")


def _load_materialize_module():
    path = Path(__file__).resolve().parent / "materialize_collab_subsets.py"
    spec = importlib.util.spec_from_file_location("materialize_collab_subsets", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_splits(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def materialize_v3(*, splits: list[str], validate: bool) -> None:
    mc = _load_materialize_module()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "subset_version": "v3",
        "selection_policy": "all_games_per_scene_domain_per_split",
        "description": (
            "One JSON per (scene_domain, official_split). "
            "Each file contains all game.tw-pddl under json_2.1.1 for that scene domain."
        ),
        "splits": splits,
        "validate_gamefiles": validate,
        "files": {},
    }

    for split_name in splits:
        split_root = DATA_ROOT / split_name
        if not split_root.exists():
            continue

        by_domain: dict[str, list[Path]] = {}
        for gamefile_path in split_root.glob("**/game.tw-pddl"):
            task_dir = gamefile_path.parts[-3]
            scene_id = int(task_dir.rsplit("-", 1)[-1])
            domain = mc.scene_domain(scene_id)
            by_domain.setdefault(domain, []).append(gamefile_path)

        for domain in DOMAIN_ORDER:
            if domain not in by_domain:
                continue

            paths = sorted(by_domain[domain])
            rows: list[dict[str, Any]] = []
            for gamefile_path in paths:
                if validate and not mc.is_valid_gamefile(gamefile_path):
                    continue
                row = mc.build_candidate(gamefile_path, split_name, domain)
                # Override metadata to reflect v3 selection policy.
                row["selection_policy"] = "all_games_per_scene_domain_v3"
                rows.append(row)

            if not rows:
                continue

            rows.sort(key=lambda row: row["selection_key"])
            for rank, item in enumerate(rows, start=1):
                item["selection_rank"] = rank

            output_name = f"{domain}__{split_name}.json"
            output_path = OUTPUT_DIR / output_name
            with output_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)

            manifest["files"][output_name] = {
                "scene_domain": domain,
                "split_name": split_name,
                "num_tasks": len(rows),
            }
            print(f"[ok] wrote {output_path} ({len(rows)} tasks)", flush=True)

    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {manifest_path} ({len(manifest['files'])} files)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize ALFWorld collab subsets v3 (per scene domain).")
    parser.add_argument(
        "--splits",
        type=str,
        default=",".join(DEFAULT_SPLITS),
        help='Comma-separated splits, e.g. "train,valid_unseen" (default) or "train,valid_seen,valid_train,valid_unseen".',
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run TextWorld load check per gamefile (very slow).",
    )
    args = parser.parse_args()
    splits = _parse_splits(args.splits)
    materialize_v3(splits=splits, validate=args.validate)


if __name__ == "__main__":
    main()
