#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对任意 `total_task.log`（包括 g-memory / selectivemem）做同样的“轨迹 -> 三元组 -> 图可视化”流水线：

1) 从 log 提取 step（traj.json，复用 scripts/extract_selectivemem_trajectory.py 的格式）
2) 对 traj.json 的 observation 用 SelectiveMem 的抽取 prompt (temp=0) 提取 instances/relations
3) 对抽取到的三元组构图并输出交互 HTML

所有产物默认写回到 input_log 同目录（或你指定的 output_dir）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: str | None = None) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline: total_task.log -> traj.json -> triples JSONL -> HTML graph.")
    ap.add_argument("--input_log", required=True, help="Path to total_task.log (g-memory/selectivemem both ok).")
    ap.add_argument("--output_dir", default=None, help="Where to write outputs (default: alongside input_log).")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model for extraction (temp=0).")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers for extraction.")
    ap.add_argument("--log_every", type=int, default=50, help="Progress print frequency.")
    ap.add_argument("--max_steps", type=int, default=None, help="Optional cap for quick debug.")
    ap.add_argument("--task_idx", type=int, default=None, help="Optional filter for visualization.")
    ap.add_argument("--max_edges", type=int, default=800, help="Max edges in visualization HTML.")
    ap.add_argument("--force", action="store_true", help="Re-generate traj.json even if it exists.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_log = Path(args.input_log).expanduser().resolve()
    if not input_log.is_file():
        raise FileNotFoundError(f"input_log not found: {input_log}")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_log.parent
    os.makedirs(out_dir, exist_ok=True)

    traj_json = out_dir / "traj.json"
    triples_jsonl = out_dir / "traj.extracted_entities_relations.jsonl"

    # 1) Extract traj.json
    if args.force or (not traj_json.is_file()):
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "extract_selectivemem_trajectory.py"),
                "--input_log",
                str(input_log),
                "--output_json",
                str(traj_json),
            ],
            cwd=str(repo_root),
        )
    else:
        print(f"[skip] traj.json exists: {traj_json}", flush=True)

    # 2) Extract triples JSONL
    cmd2 = [
        sys.executable,
        str(repo_root / "scripts" / "alfworld" / "extract_traj_entities_relations.py"),
        "--traj_json",
        str(traj_json),
        "--model",
        str(args.model),
        "--skip_ok_obs",
        "--workers",
        str(int(args.workers)),
        "--log_every",
        str(int(args.log_every)),
    ]
    if args.max_steps is not None:
        cmd2 += ["--max_steps", str(int(args.max_steps))]
    _run(cmd2, cwd=str(repo_root))

    # 3) Visualize HTML
    cmd3 = [
        sys.executable,
        str(repo_root / "scripts" / "alfworld" / "visualize_traj_triples.py"),
        "--input_jsonl",
        str(triples_jsonl),
        "--max_edges",
        str(int(args.max_edges)),
    ]
    if args.task_idx is not None:
        cmd3 += ["--task_idx", str(int(args.task_idx))]
    _run(cmd3, cwd=str(repo_root))

    print(f"[done] outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

