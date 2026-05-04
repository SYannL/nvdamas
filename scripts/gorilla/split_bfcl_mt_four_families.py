#!/usr/bin/env python3
"""
将 BFCL_v4_multi_turn_base 按四族 domain 拆分（与文本/GMM 对齐的签名规则）：
  FS / Travel / Vehicle / Trading

每族内按 id 字典序、train_ratio 确定性切分；训练写入 4 个文件，测试合并为一个文件。
同时写出对应的 possible_answer 子集。

接入 collab 评测：eval_collab_multidomain_global.py 中 --bfcl_use_family_collab_split 时，
默认的 --bfcl_family_train_questions / --bfcl_family_test_* 即指向本脚本产出路径（可全部覆盖）。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def involved_classes_signature(row: dict) -> str:
    ic = row.get("involved_classes")
    if not isinstance(ic, list) or not ic:
        return "unknown"
    parts = sorted({str(x).strip() for x in ic if str(x).strip()})
    return "+".join(parts) if parts else "unknown"


def family_from_signature(sig: str) -> str:
    """与此前分析一致：FS / Travel / Vehicle / Trading。"""
    if sig.startswith("GorillaFileSystem") or "GorillaFileSystem+" in sig or "+GorillaFileSystem" in sig:
        return "fs"
    if "TravelAPI" in sig:
        return "travel"
    if "VehicleControlAPI" in sig:
        return "vehicle"
    if "TradingBot" in sig:
        return "trading"
    return "other"


def family_from_row(row: dict) -> str:
    return family_from_signature(involved_classes_signature(row))


def split_group_sorted(group: list[dict], train_ratio: float) -> tuple[list[dict], list[dict]]:
    """同 eval_collab_multidomain_global.bfcl_split_train_test 的单组逻辑。"""
    tr = float(train_ratio)
    if tr <= 0 or tr >= 1:
        raise ValueError("train_ratio 必须在 (0, 1) 内")
    group = sorted(group, key=lambda x: str(x.get("id", "")))
    n = len(group)
    if n <= 0:
        return [], []
    if n == 1:
        return list(group), []
    n_train = int(tr * n)
    if n_train <= 0:
        n_train = 1
    if n_train >= n:
        n_train = n - 1
    return group[:n_train], group[n_train:]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "data/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_base.json"
        ),
    )
    ap.add_argument(
        "--answers",
        type=Path,
        default=Path(
            "data/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(
            "data/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/collab_family_split"
        ),
    )
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument(
        "--require_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅保留在 possible_answer 中出现 id 的题目（与评测一致）",
    )
    args = ap.parse_args()

    q_rows = load_jsonl(args.questions)
    a_rows = load_jsonl(args.answers)
    ans_by_id = {str(r.get("id", "")).strip(): r for r in a_rows if str(r.get("id", "")).strip()}

    if args.require_answer:
        work = [r for r in q_rows if str(r.get("id", "")).strip() in ans_by_id]
    else:
        work = list(q_rows)

    by_fam: dict[str, list[dict]] = defaultdict(list)
    other: list[dict] = []
    for r in work:
        fam = family_from_row(r)
        if fam == "other":
            other.append(r)
        else:
            by_fam[fam].append(r)

    if other:
        raise SystemExit(f"存在无法归入四族的样本 {len(other)} 条，请检查 involved_classes。示例 id: {other[0].get('id')!r}")

    out_q = args.out_dir
    out_a = args.out_dir / "possible_answer"
    stem = args.questions.stem

    all_test_q: list[dict] = []
    all_test_a: list[dict] = []

    summary: dict[str, object] = {"train_ratio": args.train_ratio, "families": {}}

    fam_order = [("trading", "trading"), ("travel", "travel"), ("vehicle", "vehicle"), ("fs", "fs")]
    for fam_key, slug in fam_order:
        group = by_fam[fam_key]
        tr_rows, te_rows = split_group_sorted(group, args.train_ratio)
        tr_ids = {str(r.get("id", "")).strip() for r in tr_rows}
        te_ids = {str(r.get("id", "")).strip() for r in te_rows}
        tr_ans = [ans_by_id[i] for i in sorted(tr_ids) if i in ans_by_id]
        te_ans = [ans_by_id[i] for i in sorted(te_ids) if i in ans_by_id]

        q_train_path = out_q / f"{stem}__family_{slug}__train.json"
        a_train_path = out_a / f"{stem}__family_{slug}__train.json"

        write_jsonl(q_train_path, tr_rows)
        write_jsonl(a_train_path, tr_ans)

        all_test_q.extend(te_rows)
        all_test_a.extend(te_ans)

        summary["families"][slug] = {
            "n_total": len(group),
            "n_train": len(tr_rows),
            "n_test": len(te_rows),
            "train_questions": str(q_train_path),
            "train_answers": str(a_train_path),
        }

    # 合并测试集（题目 + 答案），按 id 排序写出
    all_test_q = sorted(all_test_q, key=lambda x: str(x.get("id", "")))
    all_test_a = sorted(all_test_a, key=lambda x: str(x.get("id", "")))
    merged_q = out_q / f"{stem}__family__test_all.json"
    merged_a = out_a / f"{stem}__family__test_all.json"
    write_jsonl(merged_q, all_test_q)
    write_jsonl(merged_a, all_test_a)

    summary["merged_test"] = {
        "n_questions": len(all_test_q),
        "questions": str(merged_q),
        "answers": str(merged_a),
    }
    manifest = args.out_dir / "manifest.json"
    with manifest.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
