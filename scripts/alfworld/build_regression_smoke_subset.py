import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_subset_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_gamefile_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, start=1):
        copied = dict(row)
        copied.setdefault("_source_index_1based", idx)
        gf = (copied.get("env_kwargs") or {}).get("gamefile")
        if gf:
            out[str(gf)] = copied
    return out


def pick_regressions(
    good_records: list[dict[str, Any]],
    bad_records: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    source_map = build_gamefile_map(source_rows)
    good_by_gamefile = {str(r.get("gamefile")): r for r in good_records if r.get("gamefile")}
    bad_by_gamefile = {str(r.get("gamefile")): r for r in bad_records if r.get("gamefile")}

    picked: list[dict[str, Any]] = []
    for gf, good in good_by_gamefile.items():
        bad = bad_by_gamefile.get(gf)
        if not bad:
            continue
        if bool(good.get("success")) and (not bool(bad.get("success"))):
            row = source_map.get(gf)
            if not row:
                continue
            copied = dict(row)
            copied["_source_index_1based"] = row.get("_source_index_1based")
            copied["_regression_label"] = "won_in_good_run_lost_in_bad_run"
            copied["_good_run_success"] = bool(good.get("success"))
            copied["_bad_run_success"] = bool(bad.get("success"))
            picked.append(copied)
    picked.sort(key=lambda r: int(r.get("_source_index_1based", 0)))
    return picked[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--good-kitchen-records", required=True)
    ap.add_argument("--good-living-records", required=True)
    ap.add_argument("--bad-kitchen-records", required=True)
    ap.add_argument("--bad-living-records", required=True)
    ap.add_argument("--source-subset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit-per-domain", type=int, default=5)
    args = ap.parse_args()

    source_dir = Path(args.source_subset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_kitchen = load_subset_rows(source_dir / "kitchen__train.json")
    source_living = load_subset_rows(source_dir / "living__train.json")
    picked_kitchen = pick_regressions(
        load_jsonl(Path(args.good_kitchen_records)),
        load_jsonl(Path(args.bad_kitchen_records)),
        source_kitchen,
        args.limit_per_domain,
    )
    picked_living = pick_regressions(
        load_jsonl(Path(args.good_living_records)),
        load_jsonl(Path(args.bad_living_records)),
        source_living,
        args.limit_per_domain,
    )

    (out_dir / "kitchen__train.json").write_text(
        json.dumps(picked_kitchen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "living__train.json").write_text(
        json.dumps(picked_living, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for fname in ("kitchen__valid_unseen.json", "living__valid_unseen.json", "manifest.json"):
        src = source_dir / fname
        if src.exists():
            (out_dir / fname).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "subset_version": out_dir.name,
        "source_subset_dir": str(source_dir),
        "selection_policy": "gamefile_joined_structured_task_records_good_success_bad_failure",
        "limit_per_domain": int(args.limit_per_domain),
        "files": {
            "kitchen__train.json": {
                "num_tasks": len(picked_kitchen),
                "selected_indices_1based": [int(r["_source_index_1based"]) for r in picked_kitchen],
                "selected_gamefiles": [(r.get("env_kwargs") or {}).get("gamefile") for r in picked_kitchen],
            },
            "living__train.json": {
                "num_tasks": len(picked_living),
                "selected_indices_1based": [int(r["_source_index_1based"]) for r in picked_living],
                "selected_gamefiles": [(r.get("env_kwargs") or {}).get("gamefile") for r in picked_living],
            },
        },
    }
    (out_dir / "manifest.generated.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(out_dir)
    print("kitchen", len(picked_kitchen))
    print("living", len(picked_living))


if __name__ == "__main__":
    main()
