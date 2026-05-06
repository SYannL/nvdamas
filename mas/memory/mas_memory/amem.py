from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

import json
import os

import numpy as np

from .memory_base import MASMemoryBase
from ..common import MASMessage


@dataclass
class _AMemNote:
    """
    Lightweight Agentic-Memory-style note.

    This is a simplified variant of the baseline A-Mem note:
    - We keep content/context/keywords/tags/links/timestamp
    - We intentionally avoid extra LLM evolution logic for now to keep the
      integration cheap and robust; evolution hooks can be added later.
    """

    content: str
    context: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    links: list[int] = field(default_factory=list)
    timestamp: str | None = None


class _AMemRetriever:
    """
    Minimal embedding-based retriever compatible with MAS EmbeddingFunc.

    It stores a list of note indices and their embeddings, and exposes a
    cosine-similarity top-k retrieval API.
    """

    def __init__(self, embedding_func) -> None:
        self._embedding_func = embedding_func
        self._embeddings: np.ndarray | None = None
        self._indices: list[int] = []

    def _encode(self, text: str) -> np.ndarray:
        text = str(text or "").strip()
        if not text:
            return np.zeros((1, getattr(self._embedding_func, "dim", 768)), dtype=np.float32)
        if hasattr(self._embedding_func, "embed_text"):
            v = self._embedding_func.embed_text(text)
        else:
            v = self._embedding_func.embed_query(text)
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def add(self, note_index: int, text: str) -> None:
        vec = self._encode(text)
        if self._embeddings is None:
            self._embeddings = vec
            self._indices = [note_index]
            return
        self._embeddings = np.vstack([self._embeddings, vec])
        self._indices.append(note_index)

    def rebuild(self, items: list[tuple[int, str]]) -> None:
        if not items:
            self._embeddings = None
            self._indices = []
            return
        vecs = [self._encode(text) for _, text in items]
        self._embeddings = np.vstack(vecs)
        self._indices = [idx for idx, _ in items]

    def search(self, query: str, k: int = 5) -> list[int]:
        if self._embeddings is None or not len(self._indices):
            return []
        q = self._encode(query)
        # cosine similarity
        denom = (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8) * (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True).T + 1e-8
        )
        scores = (q @ self._embeddings.T) / denom
        scores = scores[0]
        order = np.argsort(-scores)
        top = order[: max(k, 0)]
        return [self._indices[int(i)] for i in top]

    def to_disk(self, path: str) -> None:
        base = os.path.splitext(path)[0]
        idx_path = base + "_idx.json"
        emb_path = base + "_emb.npy"
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(self._indices, f)
        if self._embeddings is not None:
            np.save(emb_path, self._embeddings)

    def from_disk(self, path: str) -> None:
        base = os.path.splitext(path)[0]
        idx_path = base + "_idx.json"
        emb_path = base + "_emb.npy"
        if not os.path.exists(idx_path) or not os.path.exists(emb_path):
            return
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                self._indices = list(json.load(f) or [])
            self._embeddings = np.load(emb_path)
        except Exception:
            self._indices = []
            self._embeddings = None


class AMemMASMemory(MASMemoryBase):
    """
    Simplified Agentic-Memory-style backend integrated into MASMemoryBase.

    设计目标：
    - 完全遵守 MASMemoryBase 生命周期（init_task_context / move_memory_state /
      save_task_context / retrieve_prompt_payload）。
    - 复用项目已有的 EmbeddingFunc，而不是重新加载 sentence-transformers。
    - 保持实现轻量级、可跨任务泛化（alfworld / pddl / fever 等）。

    配置（通过 global_config）：
        amem_k_success: successful_topk（默认 2）
        amem_k_failed: failed_topk（当前实现不区分正负例，默认 0）
        amem_k_insight: insight_topk（默认 4）
        amem_enable_insights: bool，是否把 context/tags 暴露为 insights（默认 True）
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._notes: list[_AMemNote] = []
        self._retriever = _AMemRetriever(self.embedding_func)
        self._notes_path = os.path.join(self.persist_dir, "amem_notes.jsonl")
        self._retriever_path = os.path.join(self.persist_dir, "amem_retriever.pkl")
        self.refresh_each_step = False
        self._load_from_disk_best_effort()

    # ------------------------ persistence helpers ------------------------ #
    def _load_from_disk_best_effort(self) -> None:
        if os.path.exists(self._notes_path):
            try:
                with open(self._notes_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        raw: dict[str, Any] = json.loads(line)
                        self._notes.append(
                            _AMemNote(
                                content=str(raw.get("content") or ""),
                                context=str(raw.get("context") or ""),
                                keywords=list(raw.get("keywords") or []),
                                tags=list(raw.get("tags") or []),
                                links=list(raw.get("links") or []),
                                timestamp=raw.get("timestamp"),
                            )
                        )
            except Exception:
                self._notes = []
        self._retriever.from_disk(self._retriever_path)

    def _flush_to_disk(self) -> None:
        os.makedirs(os.path.dirname(self._notes_path), exist_ok=True)
        try:
            with open(self._notes_path, "w", encoding="utf-8") as f:
                for n in self._notes:
                    f.write(
                        json.dumps(
                            {
                                "content": n.content,
                                "context": n.context,
                                "keywords": n.keywords,
                                "tags": n.tags,
                                "links": n.links,
                                "timestamp": n.timestamp,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except Exception:
            # Persistence failure should never break the main loop.
            pass
        try:
            self._retriever.to_disk(self._retriever_path)
        except Exception:
            pass

    # --------------------------- MAS lifecycle --------------------------- #
    def add_memory(self, mas_message: MASMessage) -> None:
        """
        把一次完整任务压缩成一条 note：
        - content: 任务描述 + 轨迹（execution pattern）
        - context: 任务主描述
        - keywords/tags: 通过简单启发式从文本中提取 token
        """
        if self.global_config.get("freeze_memory", False):
            return

        text = self._message_to_execution_pattern(mas_message)
        context = str(mas_message.task_description or "").strip()
        keywords, tags = self._extract_keywords_and_tags(text)

        note = _AMemNote(
            content=text,
            context=context,
            keywords=keywords,
            tags=tags,
            links=[],
            timestamp=None,
        )
        note_index = len(self._notes)
        self._notes.append(note)

        # 索引中同时使用 content + metadata（context/keywords/tags）
        retr_text = f"{note.content}\n{note.context}\n{' '.join(note.keywords)}\n{' '.join(note.tags)}"
        self._retriever.add(note_index, retr_text)
        self._flush_to_disk()

    def retrieve_memory(
        self,
        **kargs: Any,
    ) -> Tuple[list[MASMessage], list[MASMessage], list[str]]:
        query_task = str(kargs.get("query_task") or "")
        if not query_task.strip() or not self._notes:
            return [], [], []

        successful_topk = int(kargs.get("successful_topk", self.global_config.get("amem_k_success", 2)) or 0)
        failed_topk = int(kargs.get("failed_topk", self.global_config.get("amem_k_failed", 0)) or 0)
        insight_topk = int(kargs.get("insight_topk", self.global_config.get("amem_k_insight", 4)) or 0)

        top_indices = self._retriever.search(query_task, k=successful_topk + failed_topk)
        if not top_indices:
            return [], [], []

        successful: list[MASMessage] = []
        failed: list[MASMessage] = []
        for idx in top_indices[:successful_topk]:
            note = self._notes[idx]
            msg = MASMessage(
                task_main=query_task,
                task_description=note.context,
            )
            # Use the stored content as pseudo-trajectory so that downstream
            # formatting reuses the same execution pattern convention.
            msg.task_trajectory = note.content
            successful.append(msg)

        # 当前实现不区分 failed，保持接口兼容
        if failed_topk > 0:
            for idx in top_indices[successful_topk : successful_topk + failed_topk]:
                note = self._notes[idx]
                msg = MASMessage(
                    task_main=query_task,
                    task_description=note.context,
                )
                msg.task_trajectory = note.content
                failed.append(msg)

        insights: list[str] = []
        if bool(self.global_config.get("amem_enable_insights", True)):
            for idx in top_indices[: max(insight_topk, 0)]:
                note = self._notes[idx]
                parts: list[str] = []
                if note.context:
                    parts.append(f"[Context] {note.context}")
                if note.keywords:
                    parts.append(f"[Keywords] {', '.join(note.keywords)}")
                if note.tags:
                    parts.append(f"[Tags] {', '.join(note.tags)}")
                text = "\n".join(parts).strip()
                if text:
                    insights.append(text)

        return successful, failed, insights

    # ----------------------------- utilities ----------------------------- #
    def _extract_keywords_and_tags(self, text: str) -> tuple[list[str], list[str]]:
        """
        极简 keyword/tag 提取：不额外调用 LLM，仅做基于 token 的过滤。
        对比其它 memory backend，这里更像“弱化版 A-mem”，但保证代价小且稳定。
        """
        import re

        raw_tokens = re.split(r"[^a-zA-Z0-9_]+", text.lower())
        tokens = [t for t in raw_tokens if len(t) >= 3]
        # 去掉非常高频的停用词（只做一个小黑名单）
        stop = {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "will",
            "your",
            "about",
            "into",
        }
        tokens = [t for t in tokens if t not in stop]
        # 前若干个 token 作为 keywords/tags（保持 deterministic）
        uniq: list[str] = []
        for t in tokens:
            if t not in uniq:
                uniq.append(t)
        keywords = uniq[:8]
        tags = uniq[:4]
        return keywords, tags

