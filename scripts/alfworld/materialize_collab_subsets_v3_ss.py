#!/usr/bin/env python3
"""Create an extra-small quick-test subset from existing v3 collab subsets.

Output folder: data/alfworld/collab_subsets/v3_ss

Policy (per scene_domain file in v3, train only):
  - train: keep first 20 tasks per domain
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_DIR = REPO_ROOT / "data" / "alfworld" / "collab_subsets" / "v3"
OUT_DIR = REPO_ROOT / "data" / "alfworld" / "collab_subsets" / "v3_ss"

TRAIN_KEEP = 20
DOMAINS = ("kitchen", "living", "bedroom", "bathroom")


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as r:
        return json.load(r)


def _write_json_any(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        json.dump(payload, w, ensure_ascii=False, indent=2)


def main() -> None:
    if OUT_DIR.exists():
        for p in OUT_DIR.glob("*.json"):
            p.unlink(missing_ok=True)

    manifest: dict[str, Any] = {
        "subset_version": "v3_ss",
        "source_subset_dir": str(V3_DIR.relative_to(REPO_ROOT)),
        "selection_policy": "truncate_first20_train_per_scene_domain_file",
        "files": {},
    }

    for domain in DOMAINS:
        in_name = f"{domain}__train.json"
        in_path = V3_DIR / in_name
        if not in_path.exists():
            continue

        rows = _load_json(in_path)
        rows.sort(key=lambda r: r.get("selection_key") or "")
        out_rows = rows[:TRAIN_KEEP]

        out_path = OUT_DIR / in_name
        _write_json_any(out_path, out_rows)

        manifest["files"][in_name] = {
            "scene_domain": domain,
            "split_name": "train",
            "num_tasks": len(out_rows),
            "keep_n": TRAIN_KEEP,
            "source_num_tasks": len(rows),
        }
        print(f"[ok] wrote {out_path} ({len(out_rows)}/{len(rows)} tasks)", flush=True)

    _write_json_any(OUT_DIR / "manifest.json", manifest)
    print(f"[ok] wrote {OUT_DIR / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

