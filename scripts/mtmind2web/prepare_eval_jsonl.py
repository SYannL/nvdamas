import argparse
import json
from pathlib import Path


def iter_turn_records(raw_split_dir: Path):
    for json_file in sorted(raw_split_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            conversations = json.load(f)
        for conv in conversations:
            task_id = conv.get("task_id", "")
            website = conv.get("website", "")
            domain = conv.get("domain", "")
            subdomain = conv.get("subdomain", "")
            for turn in conv.get("turns", []):
                rec = {
                    "task_id": task_id,
                    "annotation_id": turn.get("annotation_id", ""),
                    "website": website,
                    "domain": domain,
                    "subdomain": subdomain,
                    "task": turn.get("confirmed_task", ""),
                    "gold_action_reprs": turn.get("action_reprs", []),
                    "env_name": "mtmind2web",
                }
                yield rec


def main():
    parser = argparse.ArgumentParser(
        description="Build lightweight MT-Mind2Web eval jsonl from raw HF files."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="data/MT-Mind2Web",
        help="Root of raw MT-Mind2Web folder.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        required=True,
        choices=["test_task", "test_website", "test_subdomain", "train"],
        help="Which raw split directory to read under input_root.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output jsonl file.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Optional head limit for debugging.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    raw_split_dir = input_root / args.split_dir
    if not raw_split_dir.exists():
        raise FileNotFoundError(f"Raw split dir not found: {raw_split_dir}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for rec in iter_turn_records(raw_split_dir):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            if args.max_turns is not None and count >= args.max_turns:
                break

    print(f"Saved {count} turn records to {output_path}")


if __name__ == "__main__":
    main()
