#!/usr/bin/env python3
"""
Split PDDL JSONL dataset into A/B train sets by difficulty（旧流程）。

按游戏划分 domain 请改用同目录下的 split_pddl_by_gamename.py。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split PDDL dataset into A/B by difficulty.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/pddl/test.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--a-difficulty",
        type=str,
        default="easy",
        help="Difficulty value for A split.",
    )
    parser.add_argument(
        "--b-difficulty",
        type=str,
        default="hard",
        help="Difficulty value for B split.",
    )
    parser.add_argument(
        "--output-a",
        type=str,
        default="data/pddl/pddl_ab_train_A.jsonl",
        help="Output JSONL for A split.",
    )
    parser.add_argument(
        "--output-b",
        type=str,
        default="data/pddl/pddl_ab_train_B.jsonl",
        help="Output JSONL for B split.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="data/pddl/pddl_ab_train_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows in each split before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when --shuffle is enabled.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_difficulty = Counter()
    by_subtask = Counter()
    by_difficulty_subtask: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        diff = str(row.get("difficulty", ""))
        subtask = str((row.get("additional_info") or {}).get("subtask", "unknown"))
        by_difficulty[diff] += 1
        by_subtask[subtask] += 1
        by_difficulty_subtask[diff][subtask] += 1
    return {
        "count": len(rows),
        "difficulty_distribution": dict(by_difficulty),
        "subtask_distribution": dict(by_subtask),
        "difficulty_x_subtask": {k: dict(v) for k, v in by_difficulty_subtask.items()},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    rows = load_rows(in_path)

    a_diff = args.a_difficulty.strip().lower()
    b_diff = args.b_difficulty.strip().lower()
    if not a_diff or not b_diff:
        raise SystemExit("Both --a-difficulty and --b-difficulty must be non-empty.")
    if a_diff == b_diff:
        raise SystemExit("--a-difficulty and --b-difficulty must be different values.")

    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    # Reconstruct per-subtask problem_index expected by PDDLEnv:
    # index increments within each subtask in source order.
    subtask_seen: Counter[str] = Counter()
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        subtask = str((row.get("additional_info") or {}).get("subtask", "")).strip().lower()
        enriched = dict(row)
        if subtask:
            enriched["game_name"] = subtask
            enriched["problem_index"] = int(subtask_seen[subtask])
            subtask_seen[subtask] += 1
        enriched_rows.append(enriched)

    for row in enriched_rows:
        diff = str(row.get("difficulty", "")).strip().lower()
        if diff == a_diff:
            a_rows.append(row)
        elif diff == b_diff:
            b_rows.append(row)
        else:
            skipped_rows.append(row)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(a_rows)
        rng.shuffle(b_rows)

    out_a = Path(args.output_a)
    out_b = Path(args.output_b)
    write_jsonl(out_a, a_rows)
    write_jsonl(out_b, b_rows)

    summary = {
        "input": str(in_path.resolve()),
        "a_difficulty": a_diff,
        "b_difficulty": b_diff,
        "total_input": len(rows),
        "a_split": summarize(a_rows),
        "b_split": summarize(b_rows),
        "skipped_count": len(skipped_rows),
        "skipped_difficulty_distribution": dict(Counter(str(r.get("difficulty", "")).strip().lower() for r in skipped_rows)),
        "output_a": str(out_a.resolve()),
        "output_b": str(out_b.resolve()),
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"A split rows: {len(a_rows)} -> {out_a}")
    print(f"B split rows: {len(b_rows)} -> {out_b}")
    print(f"Skipped rows: {len(skipped_rows)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

