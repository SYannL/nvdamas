#!/usr/bin/env python3
"""Frozen-memory retrieval ablations for MemCo.

This wrapper deliberately avoids changes to the production MemCo modules.  It
patches the retrieval surface in-process, reuses a staged frozen Local/Global
memory snapshot, and then delegates evaluation to
``eval_collab_multidomain_global.py``.

Supported experiments:

* ``eq11``: set selected Eq. (11) relevance weights to zero while leaving all
  evidence, action-grounding, filtering, routing, and rendering logic intact.
* ``dense``: replace structured candidate ranking with deterministic top-k
  cosine retrieval backed by an OpenAI-compatible embedding endpoint.
* ``bm25``: replace structured candidate ranking with deterministic BM25
  retrieval over the same serialized query/memory text.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterable
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for entry in (str(SCRIPTS_DIR), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

import eval_collab_multidomain_global as eval_runner  # noqa: E402
from mas.memory.mas_memory.memco_backend import retrieval_graph as rg  # noqa: E402
from mas.memory.mas_memory.memco_backend.graph_types import (  # noqa: E402
    ArtifactKind,
    CandidateType,
    MemoryQuery,
    RuleType,
    SupportBundle,
    SupportItem,
)


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _reuse_staged_global(*, global_dir: str, memory_namespace: str = "memco", **_: Any) -> None:
    path = Path(global_dir) / memory_namespace / "global_memory.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"staged frozen global memory is missing: {path}")


def _parse_mask(raw: str) -> tuple[float, float, float]:
    pieces = [piece.strip() for piece in raw.split(",")]
    if len(pieces) != 3:
        raise ValueError("--eq11-weights must contain exactly three comma-separated values")
    values = tuple(float(piece) for piece in pieces)
    if any(value not in {0.0, 1.0} for value in values):
        raise ValueError("Eq. (11) ablations support only binary weights 0 or 1")
    return values  # type: ignore[return-value]


def _install_eq11_mask(weights: tuple[float, float, float]) -> None:
    active = {
        facet
        for facet, weight in zip(("state", "task", "goal"), weights)
        if weight > 0.0
    }

    def masked_weights(_query: MemoryQuery) -> tuple[float, float, float]:
        return weights

    original_priority = rg._facet_priority

    def masked_priority(query: MemoryQuery) -> list[str]:
        # A disabled term must not leak back through the facet-aware top-k
        # selector.  All other action/evidence/safety paths remain unchanged.
        return [facet for facet in original_priority(query) if facet in active]

    rg._relevance_weights = masked_weights
    rg._facet_priority = masked_priority


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _query_text(query: MemoryQuery) -> str:
    actions = [action.canonical_str for action in query.admissible_actions]
    dynamic = query.dynamic_context or {}
    lines = [
        "Retrieve memory relevant to the agent's next decision.",
        f"goal: {query.goal}",
        f"task_family: {query.task_family}",
        f"current_stage: {query.current_stage or ''}",
        f"progress_state: {query.progress_state or ''}",
        f"location: {query.location or ''}",
        "goal_roles: " + json.dumps(query.goal_roles, ensure_ascii=False, sort_keys=True),
        "visible_objects: " + " ".join(str(x) for x in dynamic.get("visible_objects", []) or []),
        "held_objects: " + " ".join(str(x) for x in dynamic.get("held_objects", []) or []),
        "admissible_actions: " + " | ".join(actions),
        f"failure_label: {query.failure_label or ''}",
    ]
    return "\n".join(lines)


def _item_text(item: SupportItem) -> str:
    dynamic = item.dynamic if isinstance(item.dynamic, dict) else {}
    stable_dynamic = {
        key: dynamic[key]
        for key in sorted(dynamic)
        if key in {
            "artifact_kind",
            "anchor",
            "payload",
            "plan_steps",
            "relation_kind",
            "source_role",
            "source_base",
            "layout_id",
        }
    }
    text = "\n".join(
        [
            f"summary: {item.summary}",
            f"task_family: {item.task_family}",
            f"pattern_kind: {item.pattern_kind}",
            f"candidate_type: {item.candidate_type.value}",
            "action_patterns: " + " | ".join(item.action_patterns),
            "metadata: " + json.dumps(stable_dynamic, ensure_ascii=False, sort_keys=True),
        ]
    )
    # Keep every standalone embedding input comfortably below the embedding
    # server's 8k-token limit even when a payload contains long graph refs.
    return text[:12000]


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        batch_size: int = 64,
        cache_path: str = "",
    ) -> None:
        self.endpoint = base_url.rstrip("/") + "/embeddings"
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.cache: dict[str, list[float]] = {}
        self.cache_path = Path(cache_path).expanduser() if str(cache_path).strip() else None
        self._reported_first_request = False

    def _persistent_key(self, text: str) -> str:
        return f"{self.model}:{self._key(text)}"

    def _open_cache(self) -> sqlite3.Connection | None:
        if self.cache_path is None:
            return None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.cache_path), timeout=120)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "cache_key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        return connection

    def _load_persistent(self, texts: list[str]) -> None:
        connection = self._open_cache()
        if connection is None:
            return
        by_persistent_key = {self._persistent_key(text): text for text in texts}
        keys = list(by_persistent_key)
        try:
            for start in range(0, len(keys), 400):
                batch = keys[start : start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT cache_key, dim, vector FROM embeddings WHERE cache_key IN ({placeholders})",
                    batch,
                )
                for cache_key, dim, vector_blob in rows:
                    values = array("f")
                    values.frombytes(vector_blob)
                    if len(values) != int(dim):
                        continue
                    text = by_persistent_key[str(cache_key)]
                    self.cache[self._key(text)] = [float(value) for value in values]
        finally:
            connection.close()

    def _store_persistent(self, rows: list[tuple[str, list[float]]]) -> None:
        connection = self._open_cache()
        if connection is None or not rows:
            if connection is not None:
                connection.close()
            return
        payload = []
        for text, vector in rows:
            packed = array("f", vector).tobytes()
            payload.append((self._persistent_key(text), len(vector), sqlite3.Binary(packed)))
        try:
            connection.executemany(
                "INSERT OR IGNORE INTO embeddings(cache_key, dim, vector) VALUES (?, ?, ?)",
                payload,
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        uncached_texts = [text for text in texts if self._key(text) not in self.cache]
        if uncached_texts:
            self._load_persistent(uncached_texts)

        missing: list[str] = []
        seen_missing: set[str] = set()
        for text in texts:
            key = self._key(text)
            if key not in self.cache and key not in seen_missing:
                missing.append(text)
                seen_missing.add(key)

        if missing and not self._reported_first_request:
            print(
                "[retrieval-ablation] dense runtime verified: "
                f"requesting {len(missing)} uncached embeddings from {self.endpoint}",
                flush=True,
            )
            self._reported_first_request = True

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
            request = Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                method="POST",
            )
            with urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
            rows = sorted(result.get("data", []), key=lambda row: int(row.get("index", 0)))
            if len(rows) != len(batch):
                raise RuntimeError(
                    f"embedding endpoint returned {len(rows)} rows for a batch of {len(batch)}"
                )
            persistent_rows: list[tuple[str, list[float]]] = []
            for text, row in zip(batch, rows):
                vector = [float(value) for value in row.get("embedding", [])]
                if not vector:
                    raise RuntimeError("embedding endpoint returned an empty vector")
                self.cache[self._key(text)] = vector
                persistent_rows.append((text, vector))
            self._store_persistent(persistent_rows)

        return [self.cache[self._key(text)] for text in texts]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _bm25_scores(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not documents:
        return []
    query_terms = _tokens(query)
    doc_terms = [_tokens(document) for document in documents]
    lengths = [len(terms) for terms in doc_terms]
    avg_len = sum(lengths) / max(len(lengths), 1)
    doc_freq: Counter[str] = Counter()
    for terms in doc_terms:
        doc_freq.update(set(terms))
    total = len(doc_terms)
    scores: list[float] = []
    for terms, length in zip(doc_terms, lengths):
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            norm = frequency + k1 * (1.0 - b + b * length / max(avg_len, 1e-9))
            score += idf * frequency * (k1 + 1.0) / max(norm, 1e-9)
        scores.append(score)
    return scores


def _dedupe(items: Iterable[SupportItem]) -> list[SupportItem]:
    kept: dict[tuple[str, str], SupportItem] = {}
    for item in items:
        key = (str(item.candidate_id), str(item.summary))
        if key not in kept:
            kept[key] = item
    return list(kept.values())


class GenericTextRetriever:
    """Dense/BM25 deterministic top-k over the frozen MemCo record pool."""

    mode = "bm25"
    embedding_client: EmbeddingClient | None = None

    def __init__(self, top_k: int = 5, router: Any = None) -> None:
        del router
        self.top_k = max(1, int(top_k))
        self.helper = rg.LightweightRetriever(local_top_k=self.top_k, global_top_k=self.top_k)
        self._reported_runtime = False

    def _collect(
        self,
        query: MemoryQuery,
        local_memory: Any,
        global_memory: Any,
    ) -> tuple[list[SupportItem], list[SupportItem]]:
        graph_context = self.helper._anchor_subgraph_context(query, local_memory)
        facts = self.helper._ground_facts(query, local_memory)
        graph_items = list(graph_context.get("items", []) or [])

        local_items: list[SupportItem] = list(facts) + graph_items
        global_items: list[SupportItem] = []

        for artifact in local_memory.artifacts_by_id.values():
            local_items.append(rg._artifact_support_item(query, artifact, "local_artifact", graph_context))
        for artifact in global_memory.artifacts_by_id.values():
            global_items.append(rg._artifact_support_item(query, artifact, "global_artifact", graph_context))
        for rule in local_memory.rules_by_id.values():
            local_items.append(rg._rule_support_item(query, rule, "local_rule", graph_context))
        for rule in global_memory.rules_by_id.values():
            global_items.append(rg._rule_support_item(query, rule, "global_rule", graph_context))

        # Older snapshots can contain candidates but no typed artifacts/rules.
        if not local_memory.artifacts_by_id and not local_memory.rules_by_id:
            local_items.extend(rg._candidate_rank(query, local_memory.candidates.values(), "local_pattern"))
        if not global_memory.artifacts_by_id and not global_memory.rules_by_id:
            global_items.extend(rg._candidate_rank(query, global_memory.candidates.values(), "global_motif"))

        def eligible(item: SupportItem) -> bool:
            if item.pattern_kind in {"closure", RuleType.CLOSURE.value} and query.remaining_relevant_count > 0:
                return False
            return bool(str(item.summary or "").strip())

        return (
            [item for item in _dedupe(local_items) if eligible(item)],
            [item for item in _dedupe(global_items) if eligible(item)],
        )

    def _score(self, query: MemoryQuery, items: list[SupportItem]) -> list[SupportItem]:
        if not items:
            return []
        query_text = _query_text(query)
        documents = [_item_text(item) for item in items]
        if self.mode == "dense":
            if self.embedding_client is None:
                raise RuntimeError("dense retriever has no embedding client")
            vectors = self.embedding_client.embed([query_text, *documents])
            query_vector, document_vectors = vectors[0], vectors[1:]
            scores = [_cosine(query_vector, vector) for vector in document_vectors]
        elif self.mode == "bm25":
            scores = _bm25_scores(query_text, documents)
        else:
            raise ValueError(f"unsupported generic retrieval mode: {self.mode}")

        for item, score in zip(items, scores):
            item.score = float(score)
        return sorted(items, key=lambda item: (-float(item.score), str(item.candidate_id)))[: self.top_k]

    @staticmethod
    def _bundle(query: MemoryQuery, local: list[SupportItem], global_: list[SupportItem]) -> SupportBundle:
        bundle = SupportBundle(
            query=f"query:{query.goal}",
            goal_object=str(query.goal_roles.get("object", "")),
            goal_destination=str(query.goal_roles.get("destination", "")),
            goal_tool=str(query.goal_roles.get("tool", "")),
            progress_state=str(query.progress_state or ""),
        )
        bundle.local_items = list(local)
        bundle.global_items = list(global_)
        bundle.local_graph_items = [item for item in local if item.source == "local_graph"]
        bundle.local_promoted_items = [item for item in local if item.source != "local_graph"]
        bundle.global_promoted_items = list(global_)
        combined = list(local) + list(global_)
        bundle.fact_items = [item for item in local if item.pattern_kind == "fact"][:3]
        bundle.relation_items = [item for item in combined if item.pattern_kind == "scene_relation"][:2]
        bundle.plan_items = [item for item in combined if item.pattern_kind == "plan"][:2]
        bundle.global_task_plan_items = [
            item
            for item in global_
            if item.pattern_kind in {"plan", "workflow", RuleType.WORKFLOW.value, "closure", RuleType.CLOSURE.value}
        ][:2]
        bundle.workflow_items = [
            item
            for item in combined
            if item.pattern_kind in {"plan", "workflow", RuleType.WORKFLOW.value, "transition", "graph_transition"}
            or item.candidate_type == CandidateType.WORKFLOW
        ][:3]
        bundle.precondition_items = [
            item for item in combined if item.candidate_type == CandidateType.PRECONDITION
        ][:3]
        bundle.repair_items = [
            item
            for item in combined
            if item.candidate_type == CandidateType.REPAIR or item.pattern_kind == RuleType.REPAIR.value
        ][:3]
        bundle.closure_items = [
            item for item in combined if item.pattern_kind in {"closure", RuleType.CLOSURE.value}
        ][:2]
        bundle.reflection_items = [
            item
            for item in combined
            if item.dynamic.get("artifact_kind") == ArtifactKind.REFLECTION.value
        ][:3]
        bundle.local_graph_contribution = bundle.local_graph_items[:2]
        bundle.local_promoted_contribution = bundle.local_promoted_items[:2]
        bundle.global_promoted_contribution = bundle.global_promoted_items[:2]
        bundle.fused_support_items = _dedupe(
            bundle.local_graph_contribution
            + bundle.local_promoted_contribution
            + bundle.global_promoted_contribution
        )[:6]
        bundle.workflow_hints = [item.summary for item in bundle.workflow_items[:3]]
        bundle.task_need_analysis = [
            "Need memory selected by a generic text-retrieval baseline under the current query."
        ]
        return bundle

    def retrieve(self, query: MemoryQuery, local_memory: Any, global_memory: Any) -> SupportBundle:
        local_pool, global_pool = self._collect(query, local_memory, global_memory)
        if not self._reported_runtime:
            print(
                "[retrieval-ablation] generic runtime verified: "
                f"mode={self.mode} local_pool={len(local_pool)} global_pool={len(global_pool)} "
                f"top_k={self.top_k}",
                flush=True,
            )
            self._reported_runtime = True
        return self._bundle(query, self._score(query, local_pool), self._score(query, global_pool))


def _install_generic_retriever(args: argparse.Namespace) -> None:
    GenericTextRetriever.mode = args.retrieval_ablation
    if args.retrieval_ablation == "dense":
        GenericTextRetriever.embedding_client = EmbeddingClient(
            args.embedding_api_base,
            args.embedding_model,
            batch_size=args.embedding_batch_size,
            cache_path=args.embedding_cache_path,
        )
    rg.QueryBasedRetriever = GenericTextRetriever


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--retrieval-ablation",
        required=True,
        choices=("eq11", "dense", "bm25"),
    )
    parser.add_argument("--eq11-weights", default="1,1,1")
    parser.add_argument(
        "--embedding-api-base",
        default=os.getenv("OPENAI_EMBEDDING_API_BASE", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("MEMCO_RETRIEVAL_EMBEDDING_MODEL", "qwen3-embedding-api"),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-cache-path",
        default=os.getenv("NV_MEMCO_RETRIEVAL_EMBEDDING_CACHE", ""),
    )
    args, remaining = parser.parse_known_args()

    eval_runner.rebuild_memco_global_from_locals = _reuse_staged_global

    # ALFWorld evaluates every episode in a fresh Python interpreter.  Export
    # the ablation configuration and make the adjacent sitecustomize module
    # discoverable so each isolated worker installs the same runtime patch
    # before importing the production evaluator.  This preserves the normal
    # subprocess evaluation protocol and leaves production MemCo code intact.
    os.environ["NV_MEMCO_RETRIEVAL_ABLATION"] = args.retrieval_ablation
    os.environ["NV_MEMCO_EQ11_WEIGHTS"] = args.eq11_weights
    os.environ["NV_MEMCO_RETRIEVAL_EMBEDDING_API_BASE"] = args.embedding_api_base
    os.environ["NV_MEMCO_RETRIEVAL_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["NV_MEMCO_RETRIEVAL_EMBEDDING_BATCH_SIZE"] = str(args.embedding_batch_size)
    os.environ["NV_MEMCO_RETRIEVAL_EMBEDDING_CACHE"] = args.embedding_cache_path
    ablation_dir = str(Path(__file__).resolve().parent)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [part for part in inherited_pythonpath.split(os.pathsep) if part]
    if ablation_dir not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([ablation_dir, *pythonpath_parts])
    print(
        "[retrieval-ablation] isolated-worker propagation enabled via sitecustomize: "
        f"mode={args.retrieval_ablation}",
        flush=True,
    )

    if args.retrieval_ablation == "eq11":
        weights = _parse_mask(args.eq11_weights)
        _install_eq11_mask(weights)
        print(f"[retrieval-ablation] Eq.11 weights={weights}; disabled facets are skipped", flush=True)
    else:
        _install_generic_retriever(args)
        print(
            f"[retrieval-ablation] generic retriever={args.retrieval_ablation} "
            f"embedding_model={args.embedding_model if args.retrieval_ablation == 'dense' else 'n/a'}",
            flush=True,
        )

    sys.argv = [sys.argv[0], *remaining]
    eval_runner.main()


if __name__ == "__main__":
    main()
