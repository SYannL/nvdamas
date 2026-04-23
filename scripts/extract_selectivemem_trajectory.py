#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Any


TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ")
TASK_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - -+ Task: (\d+) -+")
STEP_RAW_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - \[detail\]\[step (\d+)\] solver\.output\.raw\.try_\d+: (.*)$"
)
STEP_PROCESSED_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - \[detail\]\[step (\d+)\] solver\.output\.processed\.try_\d+: (.*)$"
)
GT_STEP_RAW_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - \[detail\]\[step (\d+)\] ground_truth\.output\.raw\.try_\d+: (.*)$"
)
ACT_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - Act (\d+): (.*)$")
OBS_RE = re.compile(r"^Obs (\d+): (.*)$")


def _extract_think_from_raw(raw_text: str) -> str:
    """
    Try to extract think blocks from model raw output.
    We keep it simple: collect lines that contain 'think:' (case-insensitive) and return them joined.
    """
    if not raw_text:
        return ""
    out: list[str] = []
    for ln in str(raw_text).splitlines():
        if "think:" in ln.lower():
            out.append(ln.strip())
    return "\n".join(out).strip()


def _ensure_step(steps_by_k: dict[int, dict[str, Any]], k: int) -> dict[str, Any]:
    if k not in steps_by_k:
        steps_by_k[k] = {
            "step": k,
            "solver_output_raw": "",
            "solver_output_processed": "",
            "solver_think": "",
            "ground_truth_output_raw": "",
            "ground_truth_think": "",
            "act": "",
            "obs": "",
        }
    return steps_by_k[k]


def extract(input_log: str) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []

    current_task_id: int | None = None
    steps_by_k: dict[int, dict[str, Any]] = {}
    obs_collecting_step: int | None = None
    obs_lines: list[str] = []

    def flush_task() -> None:
        nonlocal current_task_id, steps_by_k, obs_collecting_step, obs_lines
        if current_task_id is None:
            return
        if obs_collecting_step is not None:
            st = _ensure_step(steps_by_k, obs_collecting_step)
            st["obs"] = "\n".join(obs_lines).strip()
            obs_collecting_step = None
            obs_lines = []

        steps_sorted = [steps_by_k[k] for k in sorted(steps_by_k.keys())]
        tasks.append(
            {
                "task_id": current_task_id,
                "steps": steps_sorted,
            }
        )
        current_task_id = None
        steps_by_k = {}

    with open(input_log, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m_task = TASK_RE.match(line)
            if m_task:
                flush_task()
                current_task_id = int(m_task.group(1))
                continue

            if current_task_id is None:
                continue

            # If we are collecting obs and encounter a timestamped new record, close obs.
            if obs_collecting_step is not None and TS_PREFIX_RE.match(line):
                st = _ensure_step(steps_by_k, obs_collecting_step)
                st["obs"] = "\n".join(obs_lines).strip()
                obs_collecting_step = None
                obs_lines = []

            m_raw = STEP_RAW_RE.match(line)
            if m_raw:
                k = int(m_raw.group(1))
                raw_out = (m_raw.group(2) or "").rstrip()
                st = _ensure_step(steps_by_k, k)
                # Keep latest raw for this step (tries may overwrite).
                st["solver_output_raw"] = raw_out
                st["solver_think"] = _extract_think_from_raw(raw_out)
                continue

            m_processed = STEP_PROCESSED_RE.match(line)
            if m_processed:
                k = int(m_processed.group(1))
                processed = (m_processed.group(2) or "").strip()
                st = _ensure_step(steps_by_k, k)
                # Keep the latest processed action for this step (tries may overwrite).
                st["solver_output_processed"] = processed
                continue

            m_gt_raw = GT_STEP_RAW_RE.match(line)
            if m_gt_raw:
                k = int(m_gt_raw.group(1))
                raw_out = (m_gt_raw.group(2) or "").rstrip()
                st = _ensure_step(steps_by_k, k)
                st["ground_truth_output_raw"] = raw_out
                st["ground_truth_think"] = _extract_think_from_raw(raw_out)
                continue

            m_act = ACT_RE.match(line)
            if m_act:
                k = int(m_act.group(1))
                act = (m_act.group(2) or "").strip()
                st = _ensure_step(steps_by_k, k)
                st["act"] = act
                continue

            m_obs = OBS_RE.match(line)
            if m_obs:
                k = int(m_obs.group(1))
                first = m_obs.group(2)
                _ensure_step(steps_by_k, k)
                obs_collecting_step = k
                obs_lines = [first]
                continue

            if obs_collecting_step is not None and not TS_PREFIX_RE.match(line):
                obs_lines.append(line)

    flush_task()

    return {
        "input_log": os.path.abspath(input_log),
        "tasks": tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract SelectiveMem total_task.log trajectory into JSON (by task -> steps)."
    )
    ap.add_argument("--input_log", required=True, help="Path to total_task.log")
    ap.add_argument(
        "--output_json",
        required=True,
        help="Output JSON path (single file).",
    )
    args = ap.parse_args()

    data = extract(args.input_log)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()

