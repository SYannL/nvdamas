"""
MemRL episodic memory (Memp + MemOS) integrated as a MASMemoryBase implementation.

Requires optional dependencies: see ``requirements-memrl.txt`` at repo root
(``memoryos`` / MemOS stack). MemRL 实现代码位于同目录下的 ``memrl/`` 子包。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from mas.llm import LLMCallable

from ..common import MASMessage
from .memory_base import MASMemoryBase


def _meta_success(mem: dict[str, Any]) -> Optional[bool]:
    md = mem.get("metadata")
    if md is None:
        return None
    try:
        if hasattr(md, "model_extra") and md.model_extra:
            raw = md.model_extra.get("success")
        else:
            d = md.model_dump() if hasattr(md, "model_dump") else {}
            raw = d.get("success")
    except Exception:
        raw = None
    if isinstance(raw, bool):
        return raw
    if raw in (0, 1):
        return bool(raw)
    return None


def _mem_text(mem: dict[str, Any]) -> str:
    c = mem.get("content")
    if isinstance(c, str) and c.strip():
        return c.strip()
    return ""


def _resolve_chat_base_url() -> Optional[str]:
    u = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
    s = u.strip()
    return s if s else None


def _resolve_embedding_base_url(gc: dict[str, Any]) -> Optional[str]:
    """
    Embedding 网关可与对话分离（本地 Qwen + 官方或其它 embedding 服务）。

    优先级：global_config ``memrl_embedding_api_base`` >
    ``OPENAI_EMBEDDING_API_BASE`` / ``OPENAI_EMBEDDING_BASE_URL`` >
    与对话相同（``OPENAI_API_BASE`` / ``OPENAI_BASE_URL``）。
    若环境变量存在且为空字符串，则视为使用 SDK 默认官方 endpoint。
    """
    raw = gc.get("memrl_embedding_api_base")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    env_emb = os.environ.get("OPENAI_EMBEDDING_API_BASE") or os.environ.get(
        "OPENAI_EMBEDDING_BASE_URL"
    )
    if env_emb is not None:
        s = env_emb.strip()
        return s if s else None
    return _resolve_chat_base_url()


def _resolve_embedding_api_key(gc: dict[str, Any]) -> str:
    raw = gc.get("memrl_embedding_api_key")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    k = os.environ.get("OPENAI_EMBEDDING_API_KEY", "")
    if (k or "").strip():
        return k.strip()
    return os.environ.get("OPENAI_API_KEY", "")


def _resolve_embedding_model(gc: dict[str, Any]) -> str:
    """
    与主模型（Chat）完全独立：仅用于 MemOS / MemRL 的向量写入与检索。

    优先级：``memrl_embedding_model``（global_config）>
    ``MEMRL_EMBEDDING_MODEL`` / ``OPENAI_EMBEDDING_MODEL`` > 默认 ``text-embedding-3-large``。
    本地部署时通常配合 ``OPENAI_EMBEDDING_API_BASE`` 指向本机 embedding 服务。
    """
    raw = gc.get("memrl_embedding_model")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    env_m = os.environ.get("MEMRL_EMBEDDING_MODEL") or os.environ.get("OPENAI_EMBEDDING_MODEL")
    if env_m and str(env_m).strip():
        return str(env_m).strip()
    return "text-embedding-3-large"


@dataclass
class MemRLMASMemory(MASMemoryBase):
    """
    Bridges the project MAS lifecycle (init_task_context → retrieve → add_memory)
    to MemRL's ``MemoryService`` (retrieve_query / update_values / add_memories).

    Configuration (via ``global_config``):
        memrl_user_id: stable MemOS user id (default: hash of persist_dir)
        memrl_k_retrieve: top-k for retrieve_query (default: 5)
        memrl_sim_threshold: similarity threshold (default: 0.5)
        memrl_build_strategy: trajectory|script|proceduralization
        memrl_retrieve_strategy: random|query|avefact
        memrl_update_strategy: vanilla|validation|adjustment
        memrl_enable_value_driven: bool (default True)
        memrl_embedding_model: 向量模型名，与主模型无关；也可用环境变量 MEMRL_EMBEDDING_MODEL（默认 text-embedding-3-large）
        memrl_embedding_api_base: embedding-only base URL（或使用环境变量 OPENAI_EMBEDDING_API_BASE）
        memrl_embedding_api_key: embedding-only API key（或使用 OPENAI_EMBEDDING_API_KEY）
        memrl_epsilon, memrl_tau, memrl_alpha, memrl_gamma, memrl_topk, weight_sim, weight_q: RL knobs
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._memrl_task_query: str = ""
        self._memrl_retrieved_ids: list[str] = []
        self._memrl_retrieved_queries: list[tuple[Any, Any]] = []
        self._memory_service: Any = None
        self._mos_temp_dir: str = os.path.join(self.persist_dir, "_memrl_mos_runtime")
        self._global_retriever: Optional[MemRLMASMemory] = None
        self.last_saved_message: MASMessage | None = None

    def set_global_retriever(self, retriever: MemRLMASMemory) -> None:
        self._global_retriever = retriever

    # ---- MemOS / MemoryService lifecycle ----

    def _mos_config_path(self) -> str:
        os.makedirs(self._mos_temp_dir, exist_ok=True)
        path = os.path.join(self._mos_temp_dir, "mos_config.json")
        gc = self.global_config
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
        embed_key = _resolve_embedding_api_key(gc)
        embed_base = _resolve_embedding_base_url(gc)
        model = getattr(self.llm_model, "model_name", "") or "gpt-4o-mini"
        embed_model = _resolve_embedding_model(gc)
        cfg = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": model,
                    "api_key": api_key,
                    "api_base": base_url or None,
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": model,
                            "api_key": api_key,
                            "api_base": base_url or None,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": embed_model,
                            "api_key": embed_key,
                            "base_url": embed_base,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"chunk_size": 500}},
                },
            },
            "user_manager": {
                "backend": "sqlite",
                "config": {"db_path": os.path.join(self._mos_temp_dir, "users.db")},
            },
            "top_k": int(self.global_config.get("memrl_mos_top_k", 5)),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return path

    def _ensure_service(self) -> Any:
        if self._memory_service is not None:
            return self._memory_service
        try:
            from .memrl.providers.embedding import OpenAIEmbedder
            from .memrl.providers.llm import OpenAILLM
            from .memrl.service.memory_service import MemoryService
            from .memrl.service.strategies import (
                BuildStrategy,
                RetrieveStrategy,
                StrategyConfiguration,
                UpdateStrategy,
            )
            from .memrl.service.value_driven import RLConfig
        except ImportError as e:
            raise ImportError(
                "MemRL memory 需要可选依赖：请在仓库根目录执行 "
                "`pip install -r requirements-memrl.txt`"
            ) from e

        gc = self.global_config
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not (api_key or "").strip():
            raise ValueError("OPENAI_API_KEY 未设置，无法初始化 MemRL / MemOS。")
        base_url = _resolve_chat_base_url()
        embed_base = _resolve_embedding_base_url(gc)
        embed_key = _resolve_embedding_api_key(gc)
        model = getattr(self.llm_model, "model_name", "") or "gpt-4o-mini"
        embed_model = _resolve_embedding_model(gc)

        llm_provider = OpenAILLM(
            api_key=api_key,
            base_url=base_url,
            model=model,
            default_temperature=float(gc.get("memrl_temperature", 0.7)),
            token_log_dir=self._mos_temp_dir,
        )
        embedding_provider = OpenAIEmbedder(
            api_key=embed_key,
            base_url=embed_base,
            model=embed_model,
            max_text_len=int(gc.get("memrl_max_text_len", 8196)),
            token_log_dir=self._mos_temp_dir,
        )

        build_s = BuildStrategy(str(gc.get("memrl_build_strategy", "proceduralization")))
        retr_s = RetrieveStrategy(str(gc.get("memrl_retrieve_strategy", "query")))
        upd_s = UpdateStrategy(str(gc.get("memrl_update_strategy", "adjustment")))
        strategies = StrategyConfiguration(build_s, retr_s, upd_s)

        uid = str(gc.get("memrl_user_id") or "").strip()
        if not uid:
            uid = "nv_" + hashlib.md5(self.persist_dir.encode("utf-8")).hexdigest()[:16]

        rl = RLConfig(
            epsilon=float(gc.get("memrl_epsilon", 0.1)),
            tau=float(gc.get("memrl_tau", 0.35)),
            alpha=float(gc.get("memrl_alpha", 0.1)),
            gamma=float(gc.get("memrl_gamma", 0.0)),
            topk=int(gc.get("memrl_topk", 5)),
            weight_sim=float(gc.get("memrl_weight_sim", 0.5)),
            weight_q=float(gc.get("memrl_weight_q", 0.5)),
            sim_threshold=float(gc.get("memrl_sim_threshold", 0.5)),
        )

        cube_root = os.path.join(self.persist_dir, "mem_cubes")
        os.makedirs(cube_root, exist_ok=True)

        self._memory_service = MemoryService(
            mos_config_path=self._mos_config_path(),
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            strategy_config=strategies,
            user_id=uid,
            num_workers=int(gc.get("memrl_num_workers", 4)),
            max_keywords=int(gc.get("memrl_max_keywords", 8)),
            add_similarity_threshold=float(gc.get("memrl_add_similarity_threshold", 0.9)),
            enable_value_driven=bool(gc.get("memrl_enable_value_driven", True)),
            rl_config=rl,
            db_max_concurrency=int(gc.get("memrl_db_max_concurrency", 4)),
            mem_cube_root=cube_root,
            use_z_score_normalization=bool(gc.get("memrl_use_z_score_norm", True)),
        )
        return self._memory_service

    def init_task_context(
        self,
        task_main: str,
        task_description: str | None = None,
    ) -> MASMessage:
        self._memrl_task_query = str(task_main or "").strip()
        self._memrl_retrieved_ids = []
        self._memrl_retrieved_queries = []
        return super().init_task_context(task_main, task_description)

    def _patterns_from_service(
        self,
        service: Any,
        task: str,
    ) -> tuple[list[str], list[str], list[str], list[tuple[Any, Any]]]:
        gc = self.global_config
        k = int(gc.get("memrl_k_retrieve", 5))
        thr = float(gc.get("memrl_sim_threshold", 0.5))
        try:
            raw = service.retrieve_query(task_description=task, k=k, threshold=thr)
        except Exception:
            return [], [], [], []
        if isinstance(raw, tuple) and raw:
            payload = raw[0]
            topq = raw[1] if len(raw) > 1 else []
        else:
            payload, topq = {}, []
        selected = (payload or {}).get("selected") or []
        succ_texts: list[str] = []
        fail_texts: list[str] = []
        ids: list[str] = []
        for mem in selected:
            if not isinstance(mem, dict):
                continue
            mid = mem.get("memory_id")
            if mid is not None:
                ids.append(str(mid))
            txt = _mem_text(mem)
            if not txt:
                continue
            st = _meta_success(mem)
            if st is True:
                succ_texts.append(txt)
            elif st is False:
                fail_texts.append(txt)
            else:
                succ_texts.append(txt)
        qlist: list[tuple[Any, Any]] = []
        if isinstance(topq, list):
            for item in topq:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    qlist.append((item[0], item[1]))
        return succ_texts, fail_texts, ids, qlist

    def retrieve_memory(self, **kargs: Any) -> tuple[list, list, list]:
        task = self._memrl_task_query or str(kargs.get("query_task") or "")
        if not task:
            return [], [], []
        svc = self._ensure_service()
        succ, fail, ids, topq = self._patterns_from_service(svc, task)
        self._memrl_retrieved_ids = ids
        self._memrl_retrieved_queries = topq

        if self._global_retriever is not None:
            try:
                gsvc = self._global_retriever._ensure_service()
                gs, gf, _, _ = self._global_retriever._patterns_from_service(gsvc, task)
                succ = list(gs) + list(succ)
                fail = list(gf) + list(fail)
            except Exception:
                pass

        return succ, fail, []

    def retrieve_prompt_payload(self, **kargs: Any) -> dict[str, list[str]]:
        successful, failed, insights = self.retrieve_memory(**kargs)
        execution_patterns: list[str] = []
        for item in successful:
            if str(item).strip():
                execution_patterns.append(str(item))
        repair_hints = (
            ["[Memory] Similar failed cases exist; consider changing the current action pattern."]
            if failed
            else []
        )
        for item in failed:
            if str(item).strip():
                repair_hints.append(str(item))
        return {
            "reference_cases": [],
            "execution_patterns": execution_patterns,
            "insights": [str(item) for item in insights if str(item).strip()],
            "planner_notes": [],
            "action_constraints": [],
            "repair_hints": repair_hints,
        }

    def add_memory(self, mas_message: MASMessage) -> None:
        if self.global_config.get("freeze_memory", False):
            self.last_saved_message = mas_message
            return
        task_q = self._memrl_task_query or str(mas_message.task_main or "").strip()
        traj = str(mas_message.task_trajectory or "").strip()
        if not task_q or not traj:
            self.last_saved_message = mas_message
            return
        svc = self._ensure_service()
        success = bool(mas_message.label)
        retrieved_ids_list = [list(self._memrl_retrieved_ids)]
        retrieved_queries = [self._memrl_retrieved_queries if self._memrl_retrieved_queries else None]

        try:
            if getattr(svc, "enable_value_driven", False) and getattr(svc, "_q_updater", None):
                successes = [1.0 if success else 0.0]
                svc.update_values(successes, retrieved_ids_list)
        except Exception:
            pass

        meta = {
            "source_benchmark": str(self.global_config.get("task_name") or "nvdamasgm"),
            "success": success,
            "q_value": None,
            "q_visits": 0,
            "q_updated_at": datetime.now().isoformat(),
            "last_used_at": datetime.now().isoformat(),
            "reward_ma": 0.0,
        }
        try:
            svc.add_memories(
                task_descriptions=[task_q],
                trajectories=[traj],
                successes=[success],
                retrieved_memory_queries=retrieved_queries,
                retrieved_memory_ids_list=[self._memrl_retrieved_ids if self._memrl_retrieved_ids else None],
                metadatas=[meta],
            )
        except Exception:
            pass
        self.last_saved_message = mas_message

    def add_memory_from_peer(self, mas_message: MASMessage, source_id: str | None = None) -> None:
        if self.global_config.get("freeze_memory", False):
            return
        task_q = str(mas_message.task_main or "").strip()
        traj = str(mas_message.task_trajectory or "").strip()
        if not task_q or not traj:
            return
        svc = self._ensure_service()
        success = bool(mas_message.label)
        meta = {
            "source_benchmark": str(source_id or "peer"),
            "success": success,
        }
        try:
            svc.add_memories(
                task_descriptions=[task_q],
                trajectories=[traj],
                successes=[success],
                retrieved_memory_queries=[None],
                retrieved_memory_ids_list=[None],
                metadatas=[meta],
            )
        except Exception:
            pass

    def backward(self, reward: Any, **kwargs: Any) -> None:
        """MemRL Q-updates are applied in ``add_memory`` (aligned with AlfworldRunner)."""
        return

    def persist_entity_graph(self) -> None:
        """Hook for runners that flush memory after training; MemOS persists incrementally."""
        return

