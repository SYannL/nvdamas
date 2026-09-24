#!/usr/bin/env python3
"""Aggregate ALFWorld system-component accuracy, steps, and prompt-use metrics.

Injection metrics are measured per MemCo retrieval call over evaluation tasks:
zeros are retained when the router emits no memory prompt.  Fixed-concat entry
counts come from its explicit selected Local/Global lists.  Adaptive entry
counts are the non-empty Local/Global/Failure memory lines actually rendered.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_NAME = "alfworld_multidomain_global_eval.json"
TRACE_NAME = "memco_debug_trace.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model/Qwen3-32B"))
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="LABEL=RUN_ID",
        help="Repeat for every condition to include in the output table.",
    )
    return parser.parse_args()


def parse_conditions(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        label, separator, run_id = value.partition("=")
        if not separator or not label.strip() or not run_id.strip():
            raise ValueError(f"invalid --condition {value!r}; expected LABEL=RUN_ID")
        parsed.append((label.strip(), run_id.strip()))
    return parsed


def find_report(repo_root: Path, run_id: str) -> Path:
    matches: list[Path] = []
    report_root = repo_root / "Report" / "alfworld" / "memco"
    for path in report_root.glob(f"*/{REPORT_NAME}"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("run_id", "")) == run_id:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"no ALFWorld report found for run_id={run_id}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def trace_path(repo_root: Path, memory_dir: str) -> Path:
    path = Path(memory_dir)
    if not path.is_absolute():
        path = repo_root / path
    return path / "memco" / TRACE_NAME


def read_episodes(path: Path) -> list[list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing MemCo trace: {path}")
    episodes: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    with path.open(encoding="utf-8", errors="replace") as reader:
        for line in reader:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event") == "task_start":
                current = []
                episodes.append(current)
            elif record.get("event") == "retrieve" and current is not None:
                current.append(record)
    return episodes


def rendered_prompt(record: dict[str, Any]) -> str:
    return str(record.get("payload", {}).get("rendered_prompt", "") or "").strip()


def adaptive_rendered_entry_count(prompt: str) -> int:
    count = 0
    for line in prompt.splitlines():
        match = re.match(r"^(Local|Global|Failure) memory:\s*(.*)$", line.strip(), flags=re.I)
        if not match:
            continue
        value = match.group(2).strip().lower()
        if value and value not in {"none", "none."}:
            count += 1
    return count


def injected_entry_count(record: dict[str, Any], prompt: str) -> int:
    if not prompt:
        return 0
    debug = record.get("payload", {}).get("memco_textloss", {}) or {}
    if str(debug.get("mode", "")) == "fixed_topk_use_all":
        return len(debug.get("selected_local", []) or []) + len(debug.get("selected_global", []) or [])
    return adaptive_rendered_entry_count(prompt)


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
    )


def segment_metrics(episodes: list[list[dict[str, Any]]], tokenizer: Any) -> dict[str, Any]:
    calls = 0
    emitted = 0
    entries = 0
    tokens = 0
    truncated = 0
    for episode in episodes:
        for record in episode:
            calls += 1
            prompt = rendered_prompt(record)
            if not prompt:
                continue
            emitted += 1
            entries += injected_entry_count(record, prompt)
            tokens += len(tokenizer.encode(prompt, add_special_tokens=False))
            if "...[truncated" in prompt:
                truncated += 1
    return {
        "retrieval_calls": calls,
        "emitted_calls": emitted,
        "injected_entries": entries,
        "memory_prompt_tokens": tokens,
        "truncated_prompts": truncated,
    }


def domain_from_memory_dir(memory_dir: str) -> str:
    parts = Path(memory_dir).parts
    try:
        index = parts.index("local")
    except ValueError:
        return "global"
    return parts[index + 1] if index + 1 < len(parts) else "global"


def summarize_condition(
    *,
    repo_root: Path,
    label: str,
    run_id: str,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_path = find_report(repo_root, run_id)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    train_counts = {
        str(row.get("domain", "")): int(row.get("num_tasks", 0) or 0)
        for row in report.get("train_results", [])
    }
    episodes_by_trace: dict[Path, list[list[dict[str, Any]]]] = {}
    offsets: dict[Path, int] = {}
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in report.get("eval_results", []):
        split = str(row.get("split", ""))
        path = trace_path(repo_root, str(row.get("memory_dir", "")))
        if path not in episodes_by_trace:
            episodes_by_trace[path] = read_episodes(path)
            offsets[path] = train_counts.get(domain_from_memory_dir(str(row.get("memory_dir", ""))), 0)

        start = offsets[path]
        task_count = int(row.get("num_tasks", 0) or 0)
        stop = start + task_count
        episodes = episodes_by_trace[path]
        if stop > len(episodes):
            raise ValueError(
                f"trace has too few episodes for {label}/{split}: "
                f"need through {stop}, found {len(episodes)} in {path}"
            )
        offsets[path] = stop
        prompt_metrics = segment_metrics(episodes[start:stop], tokenizer)

        target = totals[split]
        target["tasks"] += task_count
        completed = int(row.get("num_completed", 0) or 0)
        target["completed"] += completed
        target["skipped"] += int(row.get("num_skipped", 0) or 0)
        target["success"] += int(row.get("num_success", 0) or 0)
        target["step_sum"] += float(row.get("avg_trajectory_steps", 0.0) or 0.0) * completed
        for key, value in prompt_metrics.items():
            target[key] += value

    rows: list[dict[str, Any]] = []
    for split in ("valid_seen", "valid_unseen"):
        value = totals.get(split, {})
        tasks = int(value.get("tasks", 0))
        completed = int(value.get("completed", 0))
        calls = int(value.get("retrieval_calls", 0))
        emitted = int(value.get("emitted_calls", 0))
        if tasks <= 0 or completed <= 0 or calls <= 0:
            raise ValueError(f"no complete evaluation metrics for {label}/{split}")
        rows.append(
            {
                "condition": label,
                "run_id": run_id,
                "split": split,
                "accuracy": float(value.get("success", 0.0)) / completed,
                "avg_steps": float(value.get("step_sum", 0.0)) / completed,
                "avg_injected_entries_per_retrieval": float(value.get("injected_entries", 0.0)) / calls,
                "avg_memory_prompt_tokens_per_retrieval": float(value.get("memory_prompt_tokens", 0.0)) / calls,
                "memory_emit_rate": emitted / calls,
                "retrieval_calls": calls,
                "tasks": tasks,
                "completed": completed,
                "skipped": int(value.get("skipped", 0)),
                "truncated_prompts": int(value.get("truncated_prompts", 0)),
            }
        )
    metadata = {
        "condition": label,
        "run_id": run_id,
        "report": str(report_path),
        "trace_files": [str(path) for path in episodes_by_trace],
    }
    return rows, metadata


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_definition": {
            "denominator": "all MemCo retrieval calls in evaluation episodes; non-emission counts as zero",
            "adaptive_entries": "non-empty Local/Global/Failure memory lines actually rendered",
            "fixed_concat_entries": "selected Local plus selected Global entries actually rendered",
            "memory_prompt_tokens": "Qwen3-32B tokenizer tokens in the emitted MemCo-only prompt block",
            "accuracy_and_steps": "project report convention over completed tasks; skipped tasks are reported separately",
        },
        "conditions": metadata,
        "rows": rows,
    }
    (output_dir / "system_component_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = list(rows[0])
    with (output_dir / "system_component_metrics.csv").open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    lines = [
        "# ALFWorld system-component metrics",
        "",
        "Averages use all evaluation retrieval calls as the denominator; a step with no emitted memory prompt contributes zero.",
        "",
        "| Condition | Split | Accuracy | Avg steps | Avg injected entries/retrieval | Avg memory tokens/retrieval | Emit rate | Completed/Tasks | Skipped |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['split']} | {100.0 * row['accuracy']:.2f}% | "
            f"{row['avg_steps']:.2f} | {row['avg_injected_entries_per_retrieval']:.3f} | "
            f"{row['avg_memory_prompt_tokens_per_retrieval']:.2f} | {100.0 * row['memory_emit_rate']:.2f}% | "
            f"{row['completed']}/{row['tasks']} | {row['skipped']} |"
        )
    (output_dir / "system_component_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    tokenizer_path = args.tokenizer_path
    if not tokenizer_path.is_absolute():
        tokenizer_path = repo_root / tokenizer_path
    tokenizer = load_tokenizer(tokenizer_path)

    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for label, run_id in parse_conditions(args.condition):
        condition_rows, condition_metadata = summarize_condition(
            repo_root=repo_root,
            label=label,
            run_id=run_id,
            tokenizer=tokenizer,
        )
        rows.extend(condition_rows)
        metadata.append(condition_metadata)
    write_outputs(args.output_dir.resolve(), rows, metadata)
    print(args.output_dir.resolve() / "system_component_metrics.md")


if __name__ == "__main__":
    main()
