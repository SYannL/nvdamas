#!/usr/bin/env python3
"""Create a small quick-test subset from existing v3 collab subsets.

Output folder: data/alfworld/collab_subsets/v3_s

Policy (per scene_domain file in v3):
  - train: keep first 100 tasks
  - valid_unseen (test): keep first 20 tasks

We keep the original deterministic ordering from v3 (and additionally sort by
selection_key for safety) so repeated runs produce identical subsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_DIR = REPO_ROOT / "data" / "alfworld" / "collab_subsets" / "v3"
OUT_DIR = REPO_ROOT / "data" / "alfworld" / "collab_subsets" / "v3_s"

TRAIN_KEEP = 100
TEST_KEEP = 20


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as r:
        return json.load(r)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        json.dump(rows, w, ensure_ascii=False, indent=2)

def _write_json_any(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        json.dump(payload, w, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize v3_s from v3 (truncate).")
    parser.add_argument("--clean", action="store_true", help="Remove existing v3_s files first.")
    args = parser.parse_args()

    if args.clean and OUT_DIR.exists():
        for p in OUT_DIR.glob("*.json"):
            p.unlink(missing_ok=True)

    # scene_domain files from current v3 script
    candidates = [
        ("kitchen", "train"),
        ("living", "train"),
        ("bedroom", "train"),
        ("bathroom", "train"),
        ("kitchen", "valid_unseen"),
        ("living", "valid_unseen"),
        ("bedroom", "valid_unseen"),
        ("bathroom", "valid_unseen"),
    ]

    manifest: dict[str, Any] = {
        "subset_version": "v3_s",
        "source_subset_dir": str(V3_DIR.relative_to(REPO_ROOT)),
        "selection_policy": "truncate_firstN_per_scene_domain_file",
        "files": {},
    }

    for scene_domain, split_name in candidates:
        in_name = f"{scene_domain}__{split_name}.json"
        in_path = V3_DIR / in_name
        if not in_path.exists():
            # If v3 changes, skip silently so the script still runs.
            continue

        rows = _load_json(in_path)
        # Ensure deterministic ordering.
        rows.sort(key=lambda r: r.get("selection_key") or "")

        keep_n = TRAIN_KEEP if split_name == "train" else TEST_KEEP
        rows_s = rows[:keep_n]

        out_name = in_name  # keep the same file naming convention
        out_path = OUT_DIR / out_name
        _write_json(out_path, rows_s)

        manifest["files"][out_name] = {
            "scene_domain": scene_domain,
            "split_name": split_name,
            "num_tasks": len(rows_s),
            "keep_n": keep_n,
            "source_num_tasks": len(rows),
        }
        print(f"[ok] wrote {out_path} ({len(rows_s)}/{len(rows)} tasks)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_any(OUT_DIR / "manifest.json", manifest)
    print(f"[ok] wrote {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

