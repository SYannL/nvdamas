#!/usr/bin/env python3
"""
Annotate FEVER samples with LLM-discovered domain labels (no fixed taxonomy).

Reads JSONL rows (id, claim, label, evidence), calls gpt-4o-mini (or override),
writes per-row annotations + an aggregate distribution JSON.

Loads API credentials from repo-root .env (OPENAI_API_KEY, optional OPENAI_API_BASE).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """You are a domain discovery annotator for FEVER-style fact verification samples.

Your job is to assign a semantic domain label to each sample WITHOUT relying on any fixed predefined domain list.

Input fields:
- claim: the statement to verify
- label: one of SUPPORTS, REFUTES, NOT ENOUGH INFO
- evidence_titles: may be empty

Requirements:
1) Do not use a fixed domain taxonomy. Create domain labels adaptively.
2) Labels must be reusable and stable across similar samples.
3) Use snake_case English labels, 1-3 words (e.g., film_tv, popular_music, political_history).
4) Focus on knowledge/topic domain, NOT on verdict label (SUPPORTS/REFUTES/NEI).
5) Always output one main domain.
6) Optionally output up to 2 secondary domains if clearly relevant.
7) If evidence_titles is empty, infer from claim semantics only.
8) Output JSON only, with no extra text.

Output JSON schema:
{
  "main_domain": "snake_case_label",
  "secondary_domains": ["snake_case_label_1", "snake_case_label_2"],
  "confidence": 0.0
}

Confidence guideline:
- 0.80-1.00: domain is very clear
- 0.50-0.79: mostly clear with some ambiguity
- 0.00-0.49: weak signal or strongly mixed topics
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def evidence_titles_from_row(row: dict) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for grp in row.get("evidence") or []:
        for it in grp:
            if isinstance(it, list) and len(it) >= 3 and it[2]:
                t = str(it[2])
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
    return titles


def parse_args() -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(description="LLM domain labels for FEVER JSONL samples.")
    p.add_argument(
        "--input",
        type=str,
        default=str(root / "data/fever/fever_dev.jsonl"),
        help="Input JSONL (FEVER rows).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(root / "data/fever/fever_dev_balanced_300_domains_gpt4omini.jsonl"),
        help="Output JSONL (one annotation per input row).",
    )
    p.add_argument(
        "--distribution",
        type=str,
        default=str(root / "data/fever/fever_dev_balanced_300_domains_distribution_gpt4omini.json"),
        help="Aggregate distribution JSON.",
    )
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="Chat model name.")
    p.add_argument(
        "--env-file",
        type=str,
        default=str(root / ".env"),
        help="Dotenv file with OPENAI_API_KEY (and optional OPENAI_API_BASE).",
    )
    p.add_argument("--sleep", type=float, default=0.15, help="Seconds between API calls.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process first N rows.")
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Shuffle input rows before annotation (helps quota mode).",
    )
    p.add_argument(
        "--quota-ab-per-label",
        type=int,
        default=0,
        help=(
            "If >0, keep annotating until each label set "
            "(SUPPORTS/REFUTES/NOT ENOUGH INFO) has at least this many "
            "A_film_tv and B_music samples, or input is exhausted."
        ),
    )
    p.add_argument(
        "--ab-output",
        type=str,
        default=str(root / "data/fever/fever_dev_ab_quota_candidates.jsonl"),
        help="Output JSONL containing only AB-matched rows in quota mode.",
    )
    p.add_argument(
        "--ab-summary",
        type=str,
        default=str(root / "data/fever/fever_dev_ab_quota_summary.json"),
        help="Summary JSON for AB quota mode.",
    )
    p.add_argument(
        "--train-per-label-domain",
        type=int,
        default=0,
        help="If >0, reserve this many samples per (label, A/B) for train split.",
    )
    p.add_argument(
        "--test-per-label-domain",
        type=int,
        default=0,
        help="If >0, reserve this many samples per (label, A/B) for test split.",
    )
    p.add_argument(
        "--ab-train-a-output",
        type=str,
        default=str(root / "data/fever/fever_ab_train_A.jsonl"),
        help="Output JSONL for A-domain train split.",
    )
    p.add_argument(
        "--ab-train-b-output",
        type=str,
        default=str(root / "data/fever/fever_ab_train_B.jsonl"),
        help="Output JSONL for B-domain train split.",
    )
    p.add_argument(
        "--ab-test-a-output",
        type=str,
        default=str(root / "data/fever/fever_ab_test_A.jsonl"),
        help="Output JSONL for A-domain test split.",
    )
    p.add_argument(
        "--ab-test-b-output",
        type=str,
        default=str(root / "data/fever/fever_ab_test_B.jsonl"),
        help="Output JSONL for B-domain test split.",
    )
    return p.parse_args()


def map_domain_to_ab(domain: str) -> str | None:
    d = str(domain or "").strip().lower()
    if not d:
        return None
    film_pat = re.compile(r"(?:^|_)(film|tv|television|series|cinema)(?:_|$)")
    music_pat = re.compile(r"(?:^|_)(music|song|album|band|singer|rapper|artist|record|production|collaboration)(?:_|$)")
    if film_pat.search(d):
        return "A_film_tv"
    if music_pat.search(d):
        return "B_music"
    return None


def quota_satisfied(by_label_ab: dict[str, Counter[str]], quota: int) -> bool:
    if quota <= 0:
        return False
    labels = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
    for lab in labels:
        c = by_label_ab.get(lab, Counter())
        if c.get("A_film_tv", 0) < quota or c.get("B_music", 0) < quota:
            return False
    return True


def label_quota_satisfied(by_label_ab: dict[str, Counter[str]], label: str, quota: int) -> bool:
    if quota <= 0:
        return False
    c = by_label_ab.get(label, Counter())
    return c.get("A_film_tv", 0) >= quota and c.get("B_music", 0) >= quota


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            f"OPENAI_API_KEY not set. Add it to {args.env_file} or export it in the shell."
        )

    base_url = os.environ.get("OPENAI_API_BASE", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)

    in_path = Path(args.input)
    rows: list[dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(args.shuffle_seed).shuffle(rows)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_mode = args.train_per_label_domain > 0 and args.test_per_label_domain > 0
    split_target = args.train_per_label_domain + args.test_per_label_domain

    main_domain_counter: Counter[str] = Counter()
    by_label: dict[str, Counter[str]] = {}
    by_label_ab: dict[str, Counter[str]] = {}
    by_label_ab_rows: dict[str, dict[str, list[dict]]] = {
        "SUPPORTS": {"A_film_tv": [], "B_music": []},
        "REFUTES": {"A_film_tv": [], "B_music": []},
        "NOT ENOUGH INFO": {"A_film_tv": [], "B_music": []},
    }
    n_ok, n_fail = 0, 0
    ab_rows: list[dict] = []

    with out_path.open("w", encoding="utf-8") as wf:
        for i, row in enumerate(rows, 1):
            row_label = str(row.get("label") or "")
            if args.quota_ab_per_label > 0 and label_quota_satisfied(by_label_ab, row_label, args.quota_ab_per_label):
                # This label already has enough A/B samples; skip API call to save cost/time.
                if quota_satisfied(by_label_ab, args.quota_ab_per_label):
                    print(
                        f"Quota reached at {i-1} processed rows: each label has "
                        f"A_film_tv>= {args.quota_ab_per_label} and B_music>= {args.quota_ab_per_label}.",
                        flush=True,
                    )
                    break
                continue
            if split_mode and row_label in by_label_ab_rows:
                a_done = len(by_label_ab_rows[row_label]["A_film_tv"]) >= split_target
                b_done = len(by_label_ab_rows[row_label]["B_music"]) >= split_target
                if a_done and b_done:
                    all_done = True
                    for lab in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
                        la_done = len(by_label_ab_rows[lab]["A_film_tv"]) >= split_target
                        lb_done = len(by_label_ab_rows[lab]["B_music"]) >= split_target
                        if not (la_done and lb_done):
                            all_done = False
                            break
                    if all_done:
                        print(
                            f"Split targets reached at {i-1} processed rows: "
                            f"each label has A/B >= {split_target}.",
                            flush=True,
                        )
                        break
                    continue

            payload = {
                "id": row.get("id"),
                "claim": row.get("claim"),
                "label": row_label,
                "evidence_titles": evidence_titles_from_row(row),
            }
            user_prompt = (
                "Assign semantic domain labels for this FEVER sample:\n\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            )

            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                txt = (resp.choices[0].message.content or "").strip()
                ann = json.loads(txt)

                main_domain = ann.get("main_domain", "")
                if not isinstance(main_domain, str) or not main_domain.strip():
                    main_domain = "unparsed"
                main_domain = main_domain.strip()

                secondary = ann.get("secondary_domains", [])
                if not isinstance(secondary, list):
                    secondary = []
                secondary = [str(x).strip() for x in secondary if str(x).strip()][:2]

                conf = ann.get("confidence", 0.0)
                try:
                    conf_f = float(conf)
                except (TypeError, ValueError):
                    conf_f = 0.0

                out = {
                    "id": row.get("id"),
                    "label": row.get("label"),
                    "claim": row.get("claim"),
                    "main_domain": main_domain,
                    "secondary_domains": secondary,
                    "confidence": conf_f,
                }
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")

                main_domain_counter[main_domain] += 1
                lab = row_label
                by_label.setdefault(lab, Counter())[main_domain] += 1
                ab_tag = map_domain_to_ab(main_domain)
                if ab_tag:
                    current = by_label_ab.setdefault(lab, Counter()).get(ab_tag, 0)
                    if args.quota_ab_per_label <= 0 or current < args.quota_ab_per_label:
                        by_label_ab[lab][ab_tag] += 1
                        ab_rows.append({**out, "ab_domain": ab_tag})
                    if split_mode and lab in by_label_ab_rows:
                        if len(by_label_ab_rows[lab][ab_tag]) < split_target:
                            by_label_ab_rows[lab][ab_tag].append({**out, "ab_domain": ab_tag})
                n_ok += 1
            except Exception as e:
                n_fail += 1
                out = {
                    "id": row.get("id"),
                    "label": row.get("label"),
                    "claim": row.get("claim"),
                    "error": f"{type(e).__name__}: {e}",
                }
                wf.write(json.dumps(out, ensure_ascii=False) + "\n")

            if i % 20 == 0:
                print(f"Processed {i}/{len(rows)} ... ok={n_ok}, fail={n_fail}", flush=True)
                if args.quota_ab_per_label > 0:
                    print(
                        "[quota-progress] "
                        f"SUPPORTS A/B={by_label_ab.get('SUPPORTS', Counter()).get('A_film_tv', 0)}/"
                        f"{by_label_ab.get('SUPPORTS', Counter()).get('B_music', 0)} | "
                        f"REFUTES A/B={by_label_ab.get('REFUTES', Counter()).get('A_film_tv', 0)}/"
                        f"{by_label_ab.get('REFUTES', Counter()).get('B_music', 0)} | "
                        f"NEI A/B={by_label_ab.get('NOT ENOUGH INFO', Counter()).get('A_film_tv', 0)}/"
                        f"{by_label_ab.get('NOT ENOUGH INFO', Counter()).get('B_music', 0)}",
                        flush=True,
                    )
            if args.quota_ab_per_label > 0 and quota_satisfied(by_label_ab, args.quota_ab_per_label):
                print(
                    f"Quota reached at {i} samples: each label has "
                    f"A_film_tv>= {args.quota_ab_per_label} and B_music>= {args.quota_ab_per_label}.",
                    flush=True,
                )
                break
            if args.sleep > 0:
                time.sleep(args.sleep)

    dist = {
        "model": args.model,
        "input_file": str(in_path.resolve()),
        "output_file": str(out_path.resolve()),
        "total": len(rows),
        "ok": n_ok,
        "fail": n_fail,
        "main_domain_distribution": dict(main_domain_counter.most_common()),
        "by_label_main_domain_distribution": {k: dict(v.most_common()) for k, v in by_label.items()},
    }
    dist_path = Path(args.distribution)
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    with dist_path.open("w", encoding="utf-8") as f:
        json.dump(dist, f, ensure_ascii=False, indent=2)

    if args.quota_ab_per_label > 0:
        ab_out_path = Path(args.ab_output)
        ab_out_path.parent.mkdir(parents=True, exist_ok=True)
        with ab_out_path.open("w", encoding="utf-8") as f:
            for r in ab_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ab_summary = {
            "quota_ab_per_label": args.quota_ab_per_label,
            "reached": quota_satisfied(by_label_ab, args.quota_ab_per_label),
            "ab_total": len(ab_rows),
            "ab_by_label": {
                lab: {
                    "A_film_tv": by_label_ab.get(lab, Counter()).get("A_film_tv", 0),
                    "B_music": by_label_ab.get(lab, Counter()).get("B_music", 0),
                }
                for lab in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
            },
        }
        ab_summary_path = Path(args.ab_summary)
        ab_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with ab_summary_path.open("w", encoding="utf-8") as f:
            json.dump(ab_summary, f, ensure_ascii=False, indent=2)
        print(f"ab candidates: {ab_out_path}", flush=True)
        print(f"ab summary: {ab_summary_path}", flush=True)
    if split_mode:
        train_a: list[dict] = []
        train_b: list[dict] = []
        test_a: list[dict] = []
        test_b: list[dict] = []
        split_summary: dict[str, dict[str, dict[str, int]]] = {}
        for lab in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
            a_rows = by_label_ab_rows[lab]["A_film_tv"]
            b_rows = by_label_ab_rows[lab]["B_music"]
            train_a.extend(a_rows[: args.train_per_label_domain])
            test_a.extend(a_rows[args.train_per_label_domain : args.train_per_label_domain + args.test_per_label_domain])
            train_b.extend(b_rows[: args.train_per_label_domain])
            test_b.extend(b_rows[args.train_per_label_domain : args.train_per_label_domain + args.test_per_label_domain])
            split_summary[lab] = {
                "A_film_tv": {
                    "collected": len(a_rows),
                    "train": min(len(a_rows), args.train_per_label_domain),
                    "test": max(
                        0,
                        min(
                            len(a_rows) - args.train_per_label_domain,
                            args.test_per_label_domain,
                        ),
                    ),
                },
                "B_music": {
                    "collected": len(b_rows),
                    "train": min(len(b_rows), args.train_per_label_domain),
                    "test": max(
                        0,
                        min(
                            len(b_rows) - args.train_per_label_domain,
                            args.test_per_label_domain,
                        ),
                    ),
                },
            }

        outputs = [
            (Path(args.ab_train_a_output), train_a),
            (Path(args.ab_train_b_output), train_b),
            (Path(args.ab_test_a_output), test_a),
            (Path(args.ab_test_b_output), test_b),
        ]
        for pth, data in outputs:
            pth.parent.mkdir(parents=True, exist_ok=True)
            with pth.open("w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        split_summary_path = Path(args.ab_summary)
        split_payload = {
            "split_mode": True,
            "train_per_label_domain": args.train_per_label_domain,
            "test_per_label_domain": args.test_per_label_domain,
            "target_per_label_domain": split_target,
            "summary_by_label": split_summary,
            "files": {
                "train_A": str(Path(args.ab_train_a_output).resolve()),
                "train_B": str(Path(args.ab_train_b_output).resolve()),
                "test_A": str(Path(args.ab_test_a_output).resolve()),
                "test_B": str(Path(args.ab_test_b_output).resolve()),
            },
            "counts": {
                "train_A": len(train_a),
                "train_B": len(train_b),
                "test_A": len(test_a),
                "test_B": len(test_b),
            },
        }
        with split_summary_path.open("w", encoding="utf-8") as f:
            json.dump(split_payload, f, ensure_ascii=False, indent=2)
        print(f"split train_A: {Path(args.ab_train_a_output)}", flush=True)
        print(f"split train_B: {Path(args.ab_train_b_output)}", flush=True)
        print(f"split test_A: {Path(args.ab_test_a_output)}", flush=True)
        print(f"split test_B: {Path(args.ab_test_b_output)}", flush=True)
        print(f"split summary: {split_summary_path}", flush=True)

    print("Done.", flush=True)
    print(f"annotations: {out_path}", flush=True)
    print(f"distribution: {dist_path}", flush=True)
    print(f"top main_domain: {main_domain_counter.most_common(15)}", flush=True)


if __name__ == "__main__":
    main()
