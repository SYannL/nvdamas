#!/usr/bin/env python3
"""
将 PDDL 标注 JSONL 按游戏名（domain）拆成多份，每游戏一个 domain，一份 JSONL。

game_name 来源：行内已有 `game_name` 则用之；否则用 `additional_info.subtask`。
每域内按源文件顺序分配 `problem_index`（0..n-1），供 PDDLEnv.fix_problem_index 使用。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_filename(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
    return s.strip("_") or "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按 game_name / subtask 拆分 PDDL JSONL。")
    p.add_argument("--input", type=str, default="data/pddl/test.jsonl", help="输入 JSONL。")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/pddl",
        help="输出目录；每域写入 pddl_domain_<game>.jsonl。",
    )
    p.add_argument(
        "--summary",
        type=str,
        default="data/pddl/pddl_domain_split_summary.json",
        help="统计摘要 JSON。",
    )
    p.add_argument("--shuffle-within-game", action="store_true", help="各域内打乱后再写盘。")
    p.add_argument("--seed", type=int, default=42, help="域内 shuffle 随机种子。")
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    rows = load_rows(in_path)

    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []

    for row in rows:
        gn = str(row.get("game_name") or "").strip().lower()
        if not gn:
            sub = (row.get("additional_info") or {}).get("subtask")
            gn = str(sub or "").strip().lower()
        if not gn:
            skipped.append(row)
            continue
        enriched = dict(row)
        enriched["game_name"] = gn
        by_game[gn].append(enriched)

    for game, g_rows in by_game.items():
        for i, r in enumerate(g_rows):
            r["problem_index"] = int(i)

    if args.shuffle_within_game:
        rng = random.Random(int(args.seed))
        for game in list(by_game.keys()):
            chunk = by_game[game]
            rng.shuffle(chunk)
            for i, r in enumerate(chunk):
                r["problem_index"] = int(i)

    written: dict[str, str] = {}
    for game, g_rows in sorted(by_game.items(), key=lambda x: x[0]):
        fn = _safe_filename(game)
        out_path = out_dir / f"pddl_domain_{fn}.jsonl"
        write_jsonl(out_path, g_rows)
        written[game] = str(out_path.resolve())

    summary = {
        "input": str(in_path.resolve()),
        "total_input": len(rows),
        "games": list(sorted(by_game.keys())),
        "per_game_counts": {g: len(v) for g, v in sorted(by_game.items(), key=lambda x: x[0])},
        "output_files": written,
        "skipped_count": len(skipped),
        "skipped_sample_ids": [r.get("id") for r in skipped[:20]],
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Games: {summary['games']}")
    for g, c in summary["per_game_counts"].items():
        print(f"  {g}: {c} -> {written[g]}")
    if skipped:
        print(f"Skipped (no game_name/subtask): {len(skipped)}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
