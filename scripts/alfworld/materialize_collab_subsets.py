#!/usr/bin/env python3
"""Materialize fixed ALFWorld collaborative-eval subsets.

Selection policy:
1. Filter by predefined scene domains and task families.
2. Within each task family, sort candidates by a stable key:
   (scene_id, object_target, receptacle_target, parent_target, trial_id).
3. Select examples with a scene-balanced round-robin over ascending scene_id.

This produces fixed JSON files that can be committed and reused directly by
evaluation scripts, instead of sampling subsets at runtime.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "alfworld" / "json_2.1.1"
OUTPUT_DIR = REPO_ROOT / "data" / "alfworld" / "collab_subsets" / "v2"

TASK_TYPE_TO_FEWSHOT = {
    "pick_and_place_simple": "put",
    "pick_and_place_with_movable_recep": "put",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "look_at_obj_in_light": "examine",
    "pick_two_obj_and_place": "puttwo",
}

TASK_TYPE_TO_ENV_NAME = {
    "pick_and_place_simple": "pick_and_place",
    "pick_and_place_with_movable_recep": "pick_and_place",
    "pick_clean_then_place_in_recep": "pick_clean_then_place",
    "pick_heat_then_place_in_recep": "pick_heat_then_place",
    "pick_cool_then_place_in_recep": "pick_cool_then_place",
    "look_at_obj_in_light": "look_at_obj",
    "pick_two_obj_and_place": "pick_two_obj",
}

GROUP_SPECS = {
    "kitchen_statechange": {
        "scene_domains": {"kitchen"},
        "task_types": (
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ),
        "counts": {
            # 使用更大的子集规模，保证每种任务类型和场景都有足够样本
            "train": 40,
            "valid_unseen": 12,
        },
    },
    "home_search": {
        "scene_domains": {"living", "bedroom", "bathroom"},
        "task_types": (
            "pick_and_place_simple",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
        ),
        "counts": {
            "train": 40,
            "valid_unseen": 12,
        },
    },
}


def scene_domain(scene_id: int) -> str:
    if 1 <= scene_id <= 30:
        return "kitchen"
    if 201 <= scene_id <= 230:
        return "living"
    if 301 <= scene_id <= 330:
        return "bedroom"
    if 401 <= scene_id <= 430:
        return "bathroom"
    return "other"


def extract_goal_instruction(gamefile_path: Path) -> str:
    with gamefile_path.open("r", encoding="utf-8") as reader:
        data = json.load(reader)

    grammar = data.get("grammar", "")
    match = re.search(r'"task"\s*:\s*\[\s*\{\s*"rhs"\s*:\s*"([^"]+)"', grammar)
    if not match:
        raise ValueError(f"Could not extract goal text from {gamefile_path}")
    return match.group(1)


def parse_task_dir(task_dir_name: str) -> dict:
    task_type, object_target, parent_target, receptacle_target, scene_id_text = task_dir_name.rsplit("-", 4)
    scene_id = int(scene_id_text)
    trial_parts = task_dir_name.split("/")
    return {
        "raw_task_type": task_type,
        "object_target": object_target,
        "parent_target": parent_target,
        "receptacle_target": receptacle_target,
        "scene_id": scene_id,
        "scene_domain": scene_domain(scene_id),
    }


def build_candidate(gamefile_path: Path, split_name: str, group_name: str) -> dict:
    relpath = str(gamefile_path.relative_to(REPO_ROOT))
    task_dir = gamefile_path.parts[-3]
    trial_id = gamefile_path.parts[-2]
    parsed = parse_task_dir(task_dir)

    raw_task_type = parsed["raw_task_type"]
    selection_key = (
        f"{parsed['scene_id']:03d}|{parsed['object_target']}|"
        f"{parsed['receptacle_target']}|{parsed['parent_target']}|{trial_id}"
    )

    return {
        "subset_group": group_name,
        "source_split": split_name,
        "selection_policy": "scene_balanced_round_robin_v1",
        "selection_key": selection_key,
        "trial_id": trial_id,
        "scene_id": parsed["scene_id"],
        "scene_domain": parsed["scene_domain"],
        "alfworld_task_type": raw_task_type,
        "object_target": parsed["object_target"],
        "parent_target": parsed["parent_target"],
        "receptacle_target": parsed["receptacle_target"],
        "goal_instruction": extract_goal_instruction(gamefile_path),
        "env_kwargs": {
            "config": "alfworld",
            "gamefile": relpath,
        },
        "task_type": TASK_TYPE_TO_FEWSHOT[raw_task_type],
        "env_name": TASK_TYPE_TO_ENV_NAME[raw_task_type],
    }


def is_valid_gamefile(gamefile_path: Path) -> bool:
    """Check whether a single ALFWorld gamefile can be loaded without runtime errors.

    Validation runs in a subprocess so that crashes (e.g. duplicate object, KeyError 'val1')
    do not exit the parent; we detect non-zero returncode and exclude such games.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; p=sys.argv[1]; "
            "from textworld.helpers import start; e=start(p, request_infos=None); e.close()",
            str(gamefile_path),
        ],
        capture_output=True,
        timeout=15,
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace") or proc.stdout.decode(errors="replace") or f"exit {proc.returncode}"
        print(f"[warn] skip invalid ALFWorld gamefile {gamefile_path}: {msg.strip()}")
        return False
    return True


def select_scene_balanced(candidates: list[dict], count: int) -> list[dict]:
    buckets: dict[int, deque[dict]] = defaultdict(deque)
    for candidate in sorted(candidates, key=lambda row: row["selection_key"]):
        buckets[candidate["scene_id"]].append(candidate)

    ordered_scene_ids = sorted(buckets)
    selected: list[dict] = []
    while len(selected) < count:
        progressed = False
        for scene_id in ordered_scene_ids:
            if buckets[scene_id]:
                selected.append(buckets[scene_id].popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break

    if len(selected) != count:
        raise ValueError(f"Failed to select {count} candidates, only got {len(selected)}")
    return selected


def materialize_group_split(group_name: str, split_name: str, per_type_count: int) -> list[dict]:
    spec = GROUP_SPECS[group_name]
    split_root = DATA_ROOT / split_name
    selected_all: list[dict] = []

    for raw_task_type in spec["task_types"]:
        candidates: list[dict] = []
        for gamefile_path in split_root.glob("**/game.tw-pddl"):
            task_dir = gamefile_path.parts[-3]
            parsed = parse_task_dir(task_dir)
            if parsed["raw_task_type"] != raw_task_type:
                continue
            if parsed["scene_domain"] not in spec["scene_domains"]:
                continue
            # 只保留能够正常加载的 gamefile，避免后续训练 / 评估阶段出现 KeyError: 'val1' 等错误
            if not is_valid_gamefile(gamefile_path):
                continue
            candidates.append(build_candidate(gamefile_path, split_name, group_name))

        picked = select_scene_balanced(candidates, per_type_count)
        for rank, item in enumerate(picked, start=1):
            item["task_type_rank"] = rank
        selected_all.extend(picked)

    selected_all.sort(key=lambda row: (row["alfworld_task_type"], row["task_type_rank"], row["selection_key"]))
    for rank, item in enumerate(selected_all, start=1):
        item["selection_rank"] = rank
    return selected_all


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "subset_version": "v2",
        "selection_policy": "scene_balanced_round_robin_v1",
        "description": {
            "kitchen_statechange": "Kitchen-only state-changing tasks: heat/cool/clean.",
            "home_search": "Living/bedroom/bathroom search-style tasks: put/puttwo/examine.",
        },
        "files": {},
    }

    for group_name, spec in GROUP_SPECS.items():
        for split_name, per_type_count in spec["counts"].items():
            rows = materialize_group_split(group_name, split_name, per_type_count)
            output_name = f"{group_name}__{split_name}.json"
            output_path = OUTPUT_DIR / output_name
            with output_path.open("w", encoding="utf-8") as writer:
                json.dump(rows, writer, ensure_ascii=False, indent=2)
            manifest["files"][output_name] = {
                "group_name": group_name,
                "split_name": split_name,
                "num_tasks": len(rows),
                "task_types": list(spec["task_types"]),
                "per_type_count": per_type_count,
            }
            print(f"[ok] wrote {output_path} ({len(rows)} tasks)")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {manifest_path}")


if __name__ == "__main__":
    main()
