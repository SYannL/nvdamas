#!/usr/bin/env python3
"""Collect the controlled ALFWorld budget sweep into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_BUDGETS = (10, 30, 50)
EXPECTED_METHODS = ("ICL", "MemCo")
SPLIT_NAMES = {"valid_seen": "Seen", "valid_unseen": "Unseen"}


def read_status(path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    selected: dict[tuple[str, int], tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            method = str(row.get("method", ""))
            budget_raw = str(row.get("budget", ""))
            state = str(row.get("state", ""))
            if method not in EXPECTED_METHODS or not budget_raw.isdigit():
                continue
            key = (method, int(budget_raw))
            if state in {"completed", "reference"}:
                selected[key] = (state, str(row.get("run_id", "")))
    return selected


def read_reference(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["method"], int(row["budget"]), row["split"]): dict(row)
        for row in rows
    }


def find_report(method: str, run_id: str) -> Path:
    memory = "empty" if method == "ICL" else "memco"
    candidates = sorted((ROOT / "Report" / "alfworld" / memory).glob("*/*.json"))
    matches: list[Path] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("run_id") == run_id and isinstance(payload.get("eval_results"), list):
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"no report found for {method=} {run_id=}")
    return matches[-1]


def weighted_metric(rows: list[dict[str, Any]], key: str) -> float:
    weighted = 0.0
    count = 0
    for row in rows:
        weight = int(row.get("num_tasks", 0))
        weighted += float(row.get(key, 0.0)) * weight
        count += weight
    if count <= 0:
        raise ValueError(f"cannot aggregate {key}: no evaluated tasks")
    return weighted / count


def collect(status_path: Path, reference_path: Path) -> list[dict[str, Any]]:
    selected_runs = read_status(status_path)
    references = read_reference(reference_path)
    missing = [
        (method, budget)
        for method in EXPECTED_METHODS
        for budget in EXPECTED_BUDGETS
        if (method, budget) not in selected_runs
    ]
    if missing:
        raise ValueError(f"budget sweep is incomplete: {missing}")

    output: list[dict[str, Any]] = []
    for method in EXPECTED_METHODS:
        for budget in EXPECTED_BUDGETS:
            state, run_id = selected_runs[(method, budget)]
            if state == "reference":
                for split_label in SPLIT_NAMES.values():
                    key = (method, budget, split_label)
                    if key not in references:
                        raise ValueError(f"missing final-result reference row: {key}")
                    row = dict(references[key])
                    row["budget"] = int(row["budget"])
                    row["accuracy"] = float(row["accuracy"])
                    row["success_rate_pct"] = float(row["success_rate_pct"])
                    row["avg_steps"] = float(row["avg_steps"])
                    row["num_scopes"] = int(row["num_scopes"])
                    row["tasks_per_scope"] = int(row["tasks_per_scope"])
                    output.append(row)
                continue
            report = find_report(method, run_id)
            payload = json.loads(report.read_text(encoding="utf-8"))
            for split_raw, split_label in SPLIT_NAMES.items():
                rows = [row for row in payload["eval_results"] if row.get("split") == split_raw]
                if not rows:
                    raise ValueError(f"missing {split_raw} in {report}")
                output.append(
                    {
                        "method": method,
                        "budget": budget,
                        "split": split_label,
                        "accuracy": weighted_metric(rows, "accuracy"),
                        "success_rate_pct": 100.0 * weighted_metric(rows, "accuracy"),
                        "avg_steps": weighted_metric(rows, "avg_trajectory_steps"),
                        "num_scopes": len(rows),
                        "tasks_per_scope": int(rows[0].get("num_tasks", 0)),
                        "run_id": run_id,
                        "report": str(report.resolve()),
                    }
                )
    return output


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    indexed = {(row["method"], row["budget"], row["split"]): row for row in rows}
    md_path = output_dir / "summary.md"
    lines = [
        "# Qwen3-32B ALFWorld maximum-interaction-budget sensitivity",
        "",
        "| Method | Budget | Seen Acc. (%) | Unseen Acc. (%) | Seen Steps | Unseen Steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in EXPECTED_METHODS:
        for budget in EXPECTED_BUDGETS:
            seen = indexed[(method, budget, "Seen")]
            unseen = indexed[(method, budget, "Unseen")]
            lines.append(
                f"| {method} | {budget} | {seen['success_rate_pct']:.2f} | "
                f"{unseen['success_rate_pct']:.2f} | {seen['avg_steps']:.2f} | "
                f"{unseen['avg_steps']:.2f} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().with_name("final_budget30_reference.csv"),
    )
    args = parser.parse_args()
    write_outputs(
        collect(args.status.resolve(), args.reference.resolve()),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
