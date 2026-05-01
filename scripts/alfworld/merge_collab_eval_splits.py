from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_domains(value: str) -> list[str]:
    domains = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not domains:
        raise ValueError("`--domains` 不能为空。")
    seen: set[str] = set()
    out: list[str] = []
    for domain in domains:
        if domain in seen:
            continue
        seen.add(domain)
        out.append(domain)
    return out


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not splits:
        splits = ["valid_seen", "valid_unseen"]
    allowed = {"valid_seen", "valid_unseen"}
    invalid = [s for s in splits if s not in allowed]
    if invalid:
        raise ValueError(f"不支持的 split: {invalid}，仅支持 {sorted(allowed)}")
    return list(dict.fromkeys(splits))


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as reader:
        data = json.load(reader)
    if not isinstance(data, list):
        raise ValueError(f"文件不是任务列表: {path}")
    return data


def unique_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in tasks:
        gamefile = ((row.get("env_kwargs") or {}).get("gamefile") or "").strip()
        if not gamefile:
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        else:
            key = gamefile
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def merge_split(
    subset_dir: Path,
    out_dir: Path,
    domains: list[str],
    split_name: str,
) -> dict[str, Any]:
    merged: list[dict[str, Any]] = []
    source_files: list[str] = []
    for domain in domains:
        src = subset_dir / f"{domain}__{split_name}.json"
        if not src.exists():
            raise FileNotFoundError(f"缺少 domain 子集文件: {src}")
        rows = load_tasks(src)
        merged.extend(rows)
        source_files.append(str(src))

    deduped = unique_tasks(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"merged__{split_name}.json"
    with out_path.open("w", encoding="utf-8") as writer:
        json.dump(deduped, writer, ensure_ascii=False, indent=2)

    return {
        "split": split_name,
        "output_file": str(out_path),
        "num_tasks_raw": len(merged),
        "num_tasks_dedup": len(deduped),
        "source_files": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并 ALFWorld 多 domain 的 valid_seen/valid_unseen 评测集。"
    )
    parser.add_argument(
        "--subset_dir",
        type=str,
        default="data/alfworld/collab_subsets/v3",
        help="输入 domain 子集目录，例如 data/alfworld/collab_subsets/v3",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="输出目录；默认写回 subset_dir",
    )
    parser.add_argument(
        "--domains",
        type=str,
        required=True,
        help="逗号分隔 domain 列表，例如 bathroom,bedroom,kitchen,living",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="valid_seen,valid_unseen",
        help="逗号分隔 split 列表，仅支持 valid_seen,valid_unseen",
    )
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else subset_dir
    domains = parse_domains(args.domains)
    splits = parse_splits(args.splits)

    if not subset_dir.exists():
        raise FileNotFoundError(f"subset_dir 不存在: {subset_dir}")

    rows = [merge_split(subset_dir, out_dir, domains, split_name) for split_name in splits]
    manifest = {
        "subset_dir": str(subset_dir),
        "output_dir": str(out_dir),
        "domains": domains,
        "splits": splits,
        "results": rows,
    }
    manifest_path = out_dir / "merged_eval_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
