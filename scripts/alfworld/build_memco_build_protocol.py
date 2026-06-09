#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as reader:
        return json.load(reader)


def _normalize_gamefile(value: str) -> str:
    normalized = str(value or "").replace("\\", "/").strip()
    if normalized.startswith("ALFWORLD_DATA/"):
        normalized = "data/alfworld/" + normalized[len("ALFWORLD_DATA/") :]
    return normalized


def _infer_scene_name(subset_path: Path, rows: list[dict[str, Any]]) -> str:
    scene_values = {str(row.get("scene_domain", "")).strip() for row in rows if row.get("scene_domain")}
    if len(scene_values) == 1:
        return next(iter(scene_values))
    stem = subset_path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    raise ValueError(f"Could not infer scene name from {subset_path}")


def _collect_train_gamefiles(subset_paths: list[Path]) -> dict[str, set[str]]:
    per_scene: dict[str, set[str]] = {}
    for subset_path in subset_paths:
        rows = _load_json(subset_path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {subset_path}")
        scene_name = _infer_scene_name(subset_path, rows)
        bucket = per_scene.setdefault(scene_name, set())
        for row in rows:
            env_kwargs = row.get("env_kwargs") or {}
            gamefile = env_kwargs.get("gamefile")
            if gamefile:
                bucket.add(_normalize_gamefile(str(gamefile)))
    return per_scene


def _index_histories(history_dir: Path) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = {}
    for path in sorted(history_dir.rglob("history_*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        gamefile = _normalize_gamefile(str(payload.get("game_file", "")))
        history = payload.get("history")
        if not gamefile or not isinstance(history, list) or len(history) < 2:
            continue
        indexed.setdefault(gamefile, []).append(str(path.resolve()))
    return indexed


def build_protocol(train_subset_jsons: list[Path], history_dir: Path) -> dict[str, Any]:
    per_scene_gamefiles = _collect_train_gamefiles(train_subset_jsons)
    history_index = _index_histories(history_dir)
    scenes = sorted(per_scene_gamefiles.keys())
    agents: dict[str, dict[str, list[str]]] = {}

    for scene_name, gamefiles in per_scene_gamefiles.items():
        matched_histories: list[str] = []
        missing: list[str] = []
        for gamefile in sorted(gamefiles):
            hits = history_index.get(gamefile, [])
            if not hits:
                missing.append(gamefile)
                continue
            matched_histories.extend(sorted(hits))
        if missing:
            preview = "\n".join(f"- {item}" for item in missing[:10])
            raise FileNotFoundError(
                f"Missing exported histories for {len(missing)} train gamefiles in scene '{scene_name}'.\n{preview}"
            )
        agents[scene_name] = {
            "train_histories": matched_histories,
            "promotion_histories": list(matched_histories),
        }

    return {
        "scenes": scenes,
        "agents": agents,
        "metadata": {
            "history_dir": str(history_dir.resolve()),
            "train_subset_jsons": [str(path.resolve()) for path in train_subset_jsons],
            "protocol_kind": "memco_build_only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a memco memory-build protocol from current-project train subsets and exported history_*.json files.")
    parser.add_argument(
        "--train-subset-json",
        nargs="+",
        required=True,
        help="One or more ALFWorld train subset JSON files, e.g. data/alfworld/collab_subsets/v3_s/kitchen__train.json",
    )
    parser.add_argument(
        "--history-dir",
        required=True,
        help="Directory containing exported memco-compatible history_*.json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output protocol JSON path.",
    )
    args = parser.parse_args()

    subset_paths = [Path(path).expanduser().resolve() for path in args.train_subset_json]
    history_dir = Path(args.history_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    payload = build_protocol(subset_paths, history_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
