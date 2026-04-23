import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


def load_subject_rows(path: Path, subject: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("subject_name") != subject:
                continue
            rows.append(row)
    return rows


def batch_iter(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 128) -> np.ndarray:
    vectors = []
    for batch in batch_iter(texts, batch_size):
        vec = model.encode(batch, batch_size=batch_size, normalize_embeddings=True)
        vectors.append(np.asarray(vec, dtype=np.float32))
    return np.vstack(vectors)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


def select_pairs_in_range(
    phys_emb: np.ndarray,
    pharma_emb: np.ndarray,
    phys_questions: list[str],
    pharma_questions: list[str],
    top_k: int = 20,
    min_sim: float = 0.90,
    max_sim: float = 0.95,
    per_phys_cap: int = 50
) -> list[tuple[int, int, float]]:
    best_pairs: list[tuple[int, int, float]] = []
    used_phys: set[int] = set()
    used_pharma: set[int] = set()

    candidates: list[tuple[int, int, float]] = []
    pharma_t = pharma_emb.T
    for i in range(phys_emb.shape[0]):
        sims = phys_emb[i] @ pharma_t
        idx = np.where((sims >= min_sim) & (sims <= max_sim))[0]
        if idx.size == 0:
            continue
        if idx.size > per_phys_cap:
            idx = idx[np.argsort(sims[idx])[::-1][:per_phys_cap]]
        for j in idx:
            if _normalize(phys_questions[i]) == _normalize(pharma_questions[j]):
                # Skip exact duplicate questions across datasets
                continue
            candidates.append((i, int(j), float(sims[j])))

    candidates.sort(key=lambda x: x[2], reverse=True)

    for i, j, score in candidates:
        if i in used_phys or j in used_pharma:
            continue
        if _normalize(phys_questions[i]) == _normalize(pharma_questions[j]):
            continue
        best_pairs.append((i, j, score))
        used_phys.add(i)
        used_pharma.add(j)
        if len(best_pairs) >= top_k:
            break

    return best_pairs


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            obj = {
                "question": row.get("question"),
                "cop": row.get("cop"),
                "opa": row.get("opa"),
                "opb": row.get("opb"),
                "opc": row.get("opc"),
                "opd": row.get("opd"),
                "subject_name": row.get("subject_name"),
                "id": row.get("id"),
                "choice_type": row.get("choice_type"),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _select_without_similarity(
    phys_rows: list[dict],
    pharma_rows: list[dict],
    total_size: int,
    seed: int
) -> tuple[list[dict], list[dict]]:
    import random

    random.seed(seed)
    phys_pool = phys_rows[:]
    pharma_pool = pharma_rows[:]
    random.shuffle(phys_pool)
    random.shuffle(pharma_pool)

    phys_selected = phys_pool[:total_size]
    phys_norm = {_normalize(row.get("question", "")) for row in phys_selected}

    pharma_selected = []
    for row in pharma_pool:
        if _normalize(row.get("question", "")) in phys_norm:
            continue
        pharma_selected.append(row)
        if len(pharma_selected) >= total_size:
            break

    if len(phys_selected) < total_size or len(pharma_selected) < total_size:
        raise ValueError(
            f"Not enough samples without similarity (phys={len(phys_selected)}, pharma={len(pharma_selected)})."
        )
    return phys_selected, pharma_selected


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build MedMCQA dual datasets.")
    parser.add_argument("--mode", choices=["nosim", "similarity"], default="nosim")
    parser.add_argument("--train_size", type=int, default=150)
    parser.add_argument("--test_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path("data/medmcqa/train.json")
    phys_rows = load_subject_rows(src, "Physiology")
    pharma_rows = load_subject_rows(src, "Pharmacology")

    train_size = args.train_size
    test_size = args.test_size
    total_size = train_size + test_size

    if args.mode == "similarity":
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        phys_questions = [row.get("question", "") for row in phys_rows]
        pharma_questions = [row.get("question", "") for row in pharma_rows]

        phys_emb = embed_texts(model, phys_questions)
        pharma_emb = embed_texts(model, pharma_questions)

        pairs = select_pairs_in_range(
            phys_emb,
            pharma_emb,
            phys_questions,
            pharma_questions,
            top_k=total_size,
            min_sim=0.90,
            max_sim=0.95,
            per_phys_cap=50
        )
        if len(pairs) < total_size:
            raise ValueError(
                f"Not enough pairs in similarity range 0.90-0.95: {len(pairs)} found."
            )
        phys_selected = [phys_rows[i] for i, _, _ in pairs]
        pharma_selected = [pharma_rows[j] for _, j, _ in pairs]
    else:
        phys_selected, pharma_selected = _select_without_similarity(
            phys_rows, pharma_rows, total_size, args.seed
        )

    phys_train = phys_selected[:train_size]
    phys_test = phys_selected[train_size:train_size + test_size]
    pharma_train = pharma_selected[:train_size]
    pharma_test = pharma_selected[train_size:train_size + test_size]

    if args.mode == "nosim":
        write_jsonl(phys_train, Path(f"data/medmcqa/medmcqa_physio_{train_size}_build.jsonl"))
        write_jsonl(phys_test, Path(f"data/medmcqa/medmcqa_physio_{test_size}_test.jsonl"))
        write_jsonl(pharma_train, Path(f"data/medmcqa/medmcqa_pharma_{train_size}_build.jsonl"))
        write_jsonl(pharma_test, Path(f"data/medmcqa/medmcqa_pharma_{test_size}_test.jsonl"))
    else:
        write_jsonl(phys_train, Path("data/medmcqa/medmcqa_physio_150.jsonl"))
        write_jsonl(phys_test, Path("data/medmcqa/medmcqa_physio_20_test.jsonl"))
        write_jsonl(pharma_train, Path("data/medmcqa/medmcqa_pharma_150.jsonl"))
        write_jsonl(pharma_test, Path("data/medmcqa/medmcqa_pharma_20_test.jsonl"))

    print(
        f"[{args.mode}] Selected {len(phys_selected)} per-domain "
        f"({len(phys_train)} train + {len(phys_test)} test)."
    )


if __name__ == "__main__":
    main()
