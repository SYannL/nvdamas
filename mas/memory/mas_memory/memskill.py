from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import os
import re
import sys
import time

import numpy as np

from .memory_base import MASMemoryBase
from ..common import MASMessage
from mas.llm import Message
from mas.utils import load_json, write_json


_INSERT_DESCRIPTION = (
    "Memory management skill for capturing new, durable facts from the current text chunk "
    "that are not already in memory."
)
_INSERT_TEMPLATE = """Skill: Insert New Memory
Purpose: Capture new, durable facts from the current text chunk that are missing in memory.
When to use:
- The text chunk introduces new facts, events, plans, or context worth storing.
- The information is stable and likely useful later.
How to apply:
- Compare against retrieved memories to avoid duplicates.
- Split distinct facts into separate items.
- Keep each item concise and specific.
Constraints:
- Skip trivial, fleeting, or speculative content.
- Do not update or delete existing memories.
Action type: INSERT only.
"""

_UPDATE_DESCRIPTION = (
    "Memory management skill for revising an existing memory item when the text chunk "
    "provides corrections or new details."
)
_UPDATE_TEMPLATE = """Skill: Update Existing Memory
Purpose: Revise a retrieved memory with new or corrected information from the text chunk.
When to use:
- The text chunk clarifies, corrects, or extends a retrieved memory.
How to apply:
- Select the best matching memory item.
- Merge new details into a single updated item.
- Preserve accurate details that still hold.
Constraints:
- Do not create new memories.
- Do not delete items.
Action type: UPDATE only.
"""

_DELETE_DESCRIPTION = (
    "Memory management skill for removing memory items that are incorrect, outdated, or superseded."
)
_DELETE_TEMPLATE = """Skill: Delete Invalid Memory
Purpose: Remove a retrieved memory that is wrong, outdated, or superseded by the text chunk.
When to use:
- The text chunk clearly contradicts a memory.
- A plan or fact is explicitly canceled or replaced.
How to apply:
- Only delete when evidence is explicit.
Constraints:
- If uncertain, prefer no action over deletion.
Action type: DELETE only.
"""

_NOOP_DESCRIPTION = "Memory management skill for confirming that no memory changes are required."
_NOOP_TEMPLATE = """Skill: No Operation
Purpose: Confirm no memory changes are needed for the text chunk.
When to use:
- The text chunk contains no new, corrective, or actionable information.
Constraints:
- Emit NOOP only if none of the selected skills produce actions.
Action type: NOOP only.
"""


def _initial_operations(include_noop: bool = True) -> dict[str, dict[str, Any]]:
    ops = {
        "insert": {
            "name": "insert",
            "description": _INSERT_DESCRIPTION,
            "instruction_template": _INSERT_TEMPLATE,
            "update_type": "insert",
            "meta_info": _new_meta("initial"),
        },
        "update": {
            "name": "update",
            "description": _UPDATE_DESCRIPTION,
            "instruction_template": _UPDATE_TEMPLATE,
            "update_type": "update",
            "meta_info": _new_meta("initial"),
        },
        "delete": {
            "name": "delete",
            "description": _DELETE_DESCRIPTION,
            "instruction_template": _DELETE_TEMPLATE,
            "update_type": "delete",
            "meta_info": _new_meta("initial"),
        },
        "noop": {
            "name": "noop",
            "description": _NOOP_DESCRIPTION,
            "instruction_template": _NOOP_TEMPLATE,
            "update_type": "noop",
            "meta_info": _new_meta("initial"),
        },
    }
    if not include_noop:
        ops.pop("noop", None)
    return ops


def _new_meta(origin: str) -> dict[str, Any]:
    return {
        "usage_count": 0,
        "avg_reward": 0.0,
        "recent_rewards": [],
        "recent_usage_ema": 0.0,
        "created_at": origin,
        "last_modified": origin,
    }


@dataclass
class _Operation:
    name: str
    description: str
    instruction_template: str
    update_type: str
    meta_info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_Operation":
        return cls(
            name=str(data.get("name", "")).strip(),
            description=str(data.get("description", "")).strip(),
            instruction_template=str(data.get("instruction_template", "")).strip(),
            update_type=str(data.get("update_type", "")).strip().lower(),
            meta_info=dict(data.get("meta_info") or _new_meta("loaded")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "instruction_template": self.instruction_template,
            "update_type": self.update_type,
            "meta_info": dict(self.meta_info or {}),
        }

    def update_stats(self, reward: float) -> None:
        meta = self.meta_info
        meta.setdefault("usage_count", 0)
        meta.setdefault("avg_reward", 0.0)
        meta.setdefault("recent_rewards", [])
        meta["usage_count"] += 1
        n = max(int(meta["usage_count"]), 1)
        meta["avg_reward"] = float(meta["avg_reward"]) + (float(reward) - float(meta["avg_reward"])) / n
        meta["recent_rewards"].append(float(reward))
        meta["recent_rewards"] = meta["recent_rewards"][-20:]
        meta["last_modified"] = "runtime"


class _OperationBank:
    def __init__(self, path: str, *, max_ops: int = 30, skip_noop: bool = False) -> None:
        self.path = path
        self.max_ops = max(1, int(max_ops))
        self.operations: dict[str, _Operation] = {}
        self._load(skip_noop=skip_noop)

    def _load(self, *, skip_noop: bool) -> None:
        payload = load_json(self.path)
        if isinstance(payload, dict) and isinstance(payload.get("operation_bank"), dict):
            payload = payload["operation_bank"]
        if isinstance(payload, dict) and isinstance(payload.get("operations"), dict):
            self.max_ops = max(1, int(payload.get("max_ops", self.max_ops) or self.max_ops))
            source = payload["operations"]
        else:
            source = payload if isinstance(payload, dict) else _initial_operations(include_noop=not skip_noop)
        self._load_from_mapping(source, skip_noop=skip_noop)
        if not self.operations:
            self._load_from_mapping(_initial_operations(include_noop=not skip_noop), skip_noop=skip_noop)
        self.save()

    def _load_from_mapping(self, source: Any, *, skip_noop: bool) -> None:
        if not isinstance(source, dict):
            return
        for name, data in source.items():
            if isinstance(data, dict):
                op = _Operation.from_dict(data)
                if op.name and not (skip_noop and op.update_type == "noop"):
                    self.operations[op.name] = op

    def load_external_payload(self, payload: Any, *, skip_noop: bool = False) -> bool:
        if isinstance(payload, dict) and isinstance(payload.get("operation_bank"), dict):
            payload = payload["operation_bank"]
        if isinstance(payload, dict) and isinstance(payload.get("operations"), dict):
            self.max_ops = max(1, int(payload.get("max_ops", self.max_ops) or self.max_ops))
            source = payload["operations"]
        else:
            source = payload
        parsed: dict[str, _Operation] = {}
        if isinstance(source, dict):
            for _, data in source.items():
                if not isinstance(data, dict):
                    continue
                op = _Operation.from_dict(data)
                if op.name and not (skip_noop and op.update_type == "noop"):
                    parsed[op.name] = op
        if not parsed:
            return False
        self.operations = parsed
        self.save()
        return True

    def save(self) -> None:
        write_json({name: op.to_dict() for name, op in sorted(self.operations.items())}, self.path)

    def candidates(self) -> list[_Operation]:
        return [self.operations[name] for name in sorted(self.operations.keys())]

    def by_names(self, names: list[str]) -> list[_Operation]:
        out: list[_Operation] = []
        seen: set[str] = set()
        for name in names:
            key = str(name).strip().lower()
            if key in self.operations and key not in seen:
                out.append(self.operations[key])
                seen.add(key)
        return out

    def add_or_replace(self, op: _Operation) -> None:
        if not op.name:
            return
        if op.name not in self.operations and len(self.operations) >= self.max_ops:
            removable = [
                item for item in self.operations.values()
                if item.name not in {"insert", "update", "delete", "noop"}
            ] or list(self.operations.values())
            worst = min(
                removable,
                key=lambda item: (
                    float(item.meta_info.get("avg_reward", 0.0)),
                    int(item.meta_info.get("usage_count", 0)),
                ),
            )
            self.operations.pop(worst.name, None)
        self.operations[op.name] = op
        self.save()


def _build_internal_ppo_classes(torch_module: Any):
    nn = torch_module.nn

    class _InternalPPOController(nn.Module):
        def __init__(
            self,
            *,
            state_dim: int,
            op_dim: int,
            hidden_dim: int = 256,
            device: str = "cpu",
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            vf_clip: float = 0.0,
            new_action_p_min: float = 0.0,
            new_action_delta_max: float = 0.0,
            action_top_k: int = 1,
        ) -> None:
            super().__init__()
            self.device = device
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            self.vf_clip = vf_clip
            self.new_action_p_min = new_action_p_min
            self.new_action_delta_max = new_action_delta_max
            self.new_action_bias_scale = 0.0
            self.action_top_k = action_top_k
            self.state_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.op_net = nn.Sequential(
                nn.Linear(op_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.actor_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.critic_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.to(device)

        def forward(self, state_embedding: Any, op_embeddings: Any, deterministic: bool = False, new_op_mask: Any = None):
            state_emb = state_embedding.unsqueeze(0)
            op_embs = op_embeddings.unsqueeze(0)
            state_h = self.state_net(state_emb)
            op_h = self.op_net(op_embs)
            state_expanded = state_h.unsqueeze(1).expand(-1, op_h.shape[1], -1)
            logits = self.actor_head(torch_module.cat([state_expanded, op_h], dim=-1)).squeeze(0).squeeze(-1)
            value = self.critic_head(state_h).squeeze(-1)[0]
            k = min(max(int(self.action_top_k), 1), int(logits.shape[0]))
            if k == 1:
                if deterministic:
                    action = torch_module.argmax(logits).item()
                else:
                    action = torch_module.distributions.Categorical(logits=logits).sample().item()
                log_prob = torch_module.distributions.Categorical(logits=logits).log_prob(
                    torch_module.tensor(action, device=self.device)
                ).item()
                return action, log_prob, value.item()
            if deterministic:
                _, top_k_indices = torch_module.topk(logits, k)
            else:
                gumbel = torch_module.distributions.Gumbel(0, 1).sample(logits.shape).to(logits.device)
                _, top_k_indices = torch_module.topk(logits + gumbel, k)
            actions = top_k_indices.tolist()
            probs = torch_module.softmax(logits, dim=-1)
            selected = probs[top_k_indices].clamp(min=1e-8)
            log_prob = torch_module.log(selected).sum().item()
            return actions, log_prob, value.item()

    class _InternalBaseTextEncoder:
        def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            device: str = "cpu",
            encode_batch_size: int = 64,
            use_flash_attn: bool = True,
        ) -> None:
            self.model_name = model_name
            self.device = device
            self.encode_batch_size = encode_batch_size
            model_name_lower = model_name.lower()
            self._use_sentence_transformer = (
                model_name.startswith("sentence-transformers/")
                or "qwen3-embedding" in model_name_lower
            )
            if self._use_sentence_transformer:
                from sentence_transformers import SentenceTransformer

                kwargs: dict[str, Any] = {"device": device}
                if "qwen3-embedding" in model_name_lower:
                    model_kwargs: dict[str, Any] = {}
                    if str(device).startswith("cuda"):
                        model_kwargs["torch_dtype"] = (
                            torch_module.bfloat16
                            if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported()
                            else torch_module.float16
                        )
                        if use_flash_attn:
                            model_kwargs["attn_implementation"] = "flash_attention_2"
                    kwargs["model_kwargs"] = model_kwargs
                    kwargs["tokenizer_kwargs"] = {"padding_side": "left"}
                self.model = SentenceTransformer(model_name, **kwargs)
                self.tokenizer = None
                if hasattr(self.model, "get_embedding_dimension"):
                    self._embedding_dim = int(self.model.get_embedding_dimension())
                else:
                    self._embedding_dim = int(self.model.get_sentence_embedding_dimension())
            else:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(device)
                self.model.eval()
                self._embedding_dim = int(self.model.config.hidden_size)

        @property
        def embedding_dim(self) -> int:
            return self._embedding_dim

        def encode(self, texts: str | list[str], batch_size: int | None = None) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                return np.zeros((0, self._embedding_dim), dtype=np.float32)
            if self._use_sentence_transformer:
                return np.asarray(
                    self.model.encode(texts, batch_size=batch_size or self.encode_batch_size),
                    dtype=np.float32,
                )
            encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch_module.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                emb = torch_module.sum(token_embeddings * mask, dim=1) / torch_module.clamp(mask.sum(dim=1), min=1e-9)
            return emb.detach().cpu().numpy().astype(np.float32)

    class _InternalStateEncoder:
        def __init__(
            self,
            model_name: str,
            device: str,
            fusion_mode: str = "mean",
            fusion_tau: float = 1.0,
            encode_batch_size: int = 64,
            use_flash_attn: bool = True,
        ) -> None:
            self.fusion_mode = fusion_mode
            self.fusion_tau = fusion_tau
            self._base_encoder = _InternalBaseTextEncoder(model_name, device, encode_batch_size, use_flash_attn)

        @property
        def embedding_dim(self) -> int:
            return self._base_encoder.embedding_dim

        def _encode_texts(self, texts: str | list[str]) -> np.ndarray:
            return self._base_encoder.encode(texts)

        def encode(
            self,
            session_text: str,
            retrieved_memories: list[str],
            session_embedding: np.ndarray | None = None,
            memory_embeddings: np.ndarray | list[np.ndarray] | None = None,
            fusion_mode: str | None = None,
            fusion_tau: float | None = None,
        ) -> np.ndarray:
            session_emb = session_embedding if session_embedding is not None else self._encode_texts(session_text)
            if session_emb.ndim == 2:
                session_emb = session_emb[0]
            if memory_embeddings is None or len(retrieved_memories) == 0:
                memory_emb = np.zeros(session_emb.shape[0], dtype=np.float32)
            else:
                mem_arr = np.asarray(memory_embeddings, dtype=np.float32)
                if mem_arr.ndim == 1:
                    mem_arr = mem_arr.reshape(1, -1)
                mode = fusion_mode or self.fusion_mode
                if mode == "sim_weighted":
                    sess_norm = session_emb / (np.linalg.norm(session_emb) + 1e-8)
                    mem_norm = mem_arr / (np.linalg.norm(mem_arr, axis=1, keepdims=True) + 1e-8)
                    sims = np.dot(mem_norm, sess_norm) / max(float(fusion_tau or self.fusion_tau), 1e-8)
                    sims = sims - np.max(sims)
                    weights = np.exp(sims).astype(np.float32)
                    weights = weights / (weights.sum() + 1e-8)
                    memory_emb = np.sum(mem_arr * weights[:, None], axis=0)
                else:
                    memory_emb = np.mean(mem_arr, axis=0)
            return np.concatenate([session_emb, memory_emb], axis=0)

    class _InternalOpEncoder:
        def __init__(
            self,
            model_name: str,
            device: str,
            encode_batch_size: int = 64,
            use_flash_attn: bool = True,
        ) -> None:
            self._base_encoder = _InternalBaseTextEncoder(model_name, device, encode_batch_size, use_flash_attn)

        @property
        def embedding_dim(self) -> int:
            return self._base_encoder.embedding_dim

        def encode(self, op_descriptions: list[str]) -> np.ndarray:
            return self._base_encoder.encode(op_descriptions)

    return _InternalPPOController, _InternalStateEncoder, _InternalOpEncoder


class _MemSkillPPOAdapter:
    """
    Thin inference adapter for the original MemSkill PPO controller.

    The original project trains the controller in its own trainer loop.  Inside
    nvdamas we keep that trainer out of the normal evaluation workflow, but we
    can load a trained checkpoint and use the policy to select memory skills.
    """

    def __init__(self, owner: "MemSkillMASMemory") -> None:
        self.owner = owner
        self.status: dict[str, Any] = {"loaded": False}
        checkpoint_path = str(
            owner._cfg("memskill_checkpoint_path", "checkpoint_path", default="") or ""
        ).strip()
        if not checkpoint_path:
            raise ValueError("memskill_controller=ppo requires memskill_checkpoint_path/checkpoint_path")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"MemSkill PPO checkpoint not found: {checkpoint_path}")

        repo_path = str(
            owner._cfg("memskill_ppo_repo_path", "memskill_repo_path", default="/workspace/MemSkill-main")
            or "/workspace/MemSkill-main"
        ).strip()
        if repo_path and not os.path.isdir(repo_path) and os.path.isdir("/bigdata/xenial/MemSkill-main"):
            repo_path = "/bigdata/xenial/MemSkill-main"
        if repo_path and os.path.isdir(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        import torch

        controller_source = "original_repo"
        try:
            from src.controller import PPOController, StateEncoder, OpEncoder  # type: ignore
        except Exception:
            PPOController, StateEncoder, OpEncoder = _build_internal_ppo_classes(torch)
            controller_source = "internal_fallback"

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "controller_state_dict" not in checkpoint:
            raise ValueError(
                "MemSkill PPO checkpoint must contain controller_state_dict. "
                "For operation-bank-only files use memskill_operation_bank_path instead."
            )
        config = dict(checkpoint.get("config") or {})
        cfg = owner._cfg

        device = str(cfg("memskill_ppo_device", "device", default=config.get("device", "cpu")) or "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        state_encoder_name = str(
            cfg("memskill_state_encoder", "state_encoder", default=config.get("state_encoder", "Qwen/Qwen3-Embedding-0.6B"))
        )
        op_encoder_name = str(
            cfg("memskill_op_encoder", "op_encoder", default=config.get("op_encoder", "Qwen/Qwen3-Embedding-0.6B"))
        )
        use_flash_attn = bool(cfg("memskill_use_flash_attn", "use_flash_attn", default=config.get("use_flash_attn", True)))
        encode_batch_size = int(cfg("memskill_encode_batch_size", "encode_batch_size", default=config.get("encode_batch_size", 8)))
        state_fusion = str(cfg("memskill_state_fusion", "state_fusion", default=config.get("state_fusion", "mean")))
        state_fusion_tau = float(cfg("memskill_state_fusion_tau", "state_fusion_tau", default=config.get("state_fusion_tau", 1.0)))

        self.state_encoder = StateEncoder(
            model_name=state_encoder_name,
            device=device,
            fusion_mode=state_fusion,
            fusion_tau=state_fusion_tau,
            encode_batch_size=encode_batch_size,
            use_flash_attn=use_flash_attn,
        )
        self.op_encoder = OpEncoder(
            model_name=op_encoder_name,
            device=device,
            encode_batch_size=encode_batch_size,
            use_flash_attn=use_flash_attn,
        )
        state_dim = 2 * int(self.state_encoder.embedding_dim)
        op_dim = int(self.op_encoder.embedding_dim)
        self.controller = PPOController(
            state_dim=state_dim,
            op_dim=op_dim,
            hidden_dim=int(cfg("memskill_controller_hidden_dim", "controller_hidden_dim", default=config.get("controller_hidden_dim", 256))),
            device=device,
            gamma=float(cfg("memskill_gamma", "gamma", default=config.get("gamma", 0.99))),
            gae_lambda=float(cfg("memskill_gae_lambda", "gae_lambda", default=config.get("gae_lambda", 0.95))),
            clip_epsilon=float(cfg("memskill_clip_epsilon", "clip_epsilon", default=config.get("clip_epsilon", 0.2))),
            entropy_coef=float(cfg("memskill_entropy_coef", "entropy_coef", default=config.get("entropy_coef", 0.01))),
            value_coef=float(cfg("memskill_value_coef", "value_coef", default=config.get("value_coef", 0.5))),
            vf_clip=float(cfg("memskill_vf_clip", "vf_clip", default=config.get("vf_clip", 0.0))),
            new_action_p_min=float(cfg("memskill_new_action_p_min", "new_action_p_min", default=config.get("new_action_p_min", 0.0))),
            new_action_delta_max=float(cfg("memskill_new_action_delta_max", "new_action_delta_max", default=config.get("new_action_delta_max", 0.0))),
            action_top_k=int(cfg("memskill_action_top_k", "action_top_k", default=config.get("action_top_k", 1))),
        )
        self.controller.load_state_dict(checkpoint["controller_state_dict"])
        self.controller.eval()
        self.torch = torch
        self.status = {
            "loaded": True,
            "checkpoint_path": checkpoint_path,
            "repo_path": repo_path,
            "controller_source": controller_source,
            "device": device,
            "state_encoder": state_encoder_name,
            "op_encoder": op_encoder_name,
            "state_dim": state_dim,
            "op_dim": op_dim,
            "action_top_k": int(getattr(self.controller, "action_top_k", 1)),
        }

    def select(self, session_text: str, retrieved_memories: list[str], candidates: list[_Operation]) -> list[str]:
        if not candidates:
            return []
        with self.torch.no_grad():
            session_embedding = self.state_encoder._encode_texts(session_text or " ")
            if getattr(session_embedding, "ndim", 1) == 2:
                session_embedding = session_embedding[0]
            memory_embeddings = None
            if retrieved_memories:
                memory_embeddings = self.state_encoder._encode_texts(retrieved_memories)
            state_embedding = self.state_encoder.encode(
                session_text or " ",
                retrieved_memories,
                session_embedding=session_embedding,
                memory_embeddings=memory_embeddings,
            )
            op_embeddings = self.op_encoder.encode([op.description or op.name for op in candidates])
            state_tensor = self.torch.tensor(state_embedding, dtype=self.torch.float32, device=self.device)
            op_tensor = self.torch.tensor(op_embeddings, dtype=self.torch.float32, device=self.device)
            action_idx, log_prob, value = self.controller(state_tensor, op_tensor, deterministic=True)
        indices = action_idx if isinstance(action_idx, list) else [action_idx]
        names = [candidates[int(idx)].name for idx in indices if 0 <= int(idx) < len(candidates)]
        self.status.update({"last_log_prob": float(log_prob), "last_value": float(value), "last_selected": names})
        return names


@dataclass
class _MemoryItem:
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: int = 0
    created_at: int = 0
    content_history: list[str] = field(default_factory=list)
    operation_history: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_MemoryItem":
        return cls(
            content=str(data.get("content", "")),
            embedding=list(data.get("embedding") or []),
            metadata=dict(data.get("metadata") or {}),
            access_count=int(data.get("access_count", 0) or 0),
            last_accessed=int(data.get("last_accessed", 0) or 0),
            created_at=int(data.get("created_at", 0) or 0),
            content_history=list(data.get("content_history") or []),
            operation_history=[str(x) for x in data.get("operation_history") or []],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "embedding": list(self.embedding or []),
            "metadata": dict(self.metadata or {}),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "content_history": list(self.content_history),
            "operation_history": list(self.operation_history),
        }

    def clone(self) -> "_MemoryItem":
        return _MemoryItem.from_dict(self.to_dict())


class _MemoryBank:
    def __init__(self, path: str, embedding_func: Any, *, top_k: int = 5) -> None:
        self.path = path
        self.embedding_func = embedding_func
        self.top_k = max(1, int(top_k))
        self.timestep = 0
        self.memories: list[_MemoryItem] = []
        self._load()

    def _load(self) -> None:
        payload = load_json(self.path)
        if isinstance(payload, dict):
            self.timestep = int(payload.get("timestep", 0) or 0)
            rows = payload.get("memories") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        self.memories = [_MemoryItem.from_dict(row) for row in rows if isinstance(row, dict)]

    def save(self) -> None:
        write_json(
            {
                "timestep": self.timestep,
                "memories": [item.to_dict() for item in self.memories],
            },
            self.path,
        )

    def clear(self) -> None:
        self.timestep = 0
        self.memories = []

    def export_items(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.memories]

    def import_items(self, rows: list[Any], *, source_id: str | None = None) -> int:
        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = _MemoryItem.from_dict(row)
            if source_id:
                item.metadata = dict(item.metadata or {})
                item.metadata.setdefault("source_id", source_id)
                item.metadata.setdefault("source_scope", "global")
            if item.content:
                self.memories.append(item)
                count += 1
        return count

    def embed(self, text: str) -> list[float]:
        if hasattr(self.embedding_func, "embed_text"):
            return list(self.embedding_func.embed_text(text))
        return list(self.embedding_func.embed_query(text))

    def retrieve(self, query: str, *, top_k: int | None = None) -> tuple[list[str], list[int], list[float]]:
        if not self.memories:
            return [], [], []
        q = np.asarray(self.embed(query or " "), dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        scores: list[float] = []
        for item in self.memories:
            emb = np.asarray(item.embedding or [], dtype=np.float32)
            if emb.size == 0 or emb.shape != q.shape:
                scores.append(float("-inf"))
                continue
            scores.append(float(np.dot(q_norm, emb / (np.linalg.norm(emb) + 1e-8))))
        k = min(int(top_k or self.top_k), len(self.memories))
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
        for idx in ranked:
            self.memories[idx].access_count += 1
            self.memories[idx].last_accessed = self.timestep
        return [self.memories[idx].content for idx in ranked], ranked, [scores[idx] for idx in ranked]

    def add(self, content: str, *, metadata: dict[str, Any] | None = None, operations: list[str] | None = None) -> None:
        text = str(content or "").strip()
        if not text:
            return
        item = _MemoryItem(
            content=text,
            embedding=self.embed(text),
            metadata=metadata or {},
            created_at=self.timestep,
            last_accessed=self.timestep,
            operation_history=[str(op) for op in operations or [] if str(op).strip()],
        )
        self.memories.append(item)

    def update(self, index: int, content: str, *, operations: list[str] | None = None) -> None:
        if index < 0 or index >= len(self.memories):
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")
        item = self.memories[index]
        item.content_history.append(item.content)
        item.content = str(content or "").strip()
        item.embedding = self.embed(item.content)
        item.last_accessed = self.timestep
        for op in operations or []:
            if str(op).strip():
                item.operation_history.append(str(op))

    def delete(self, index: int) -> None:
        if index < 0 or index >= len(self.memories):
            raise IndexError(f"Memory index {index} out of range [0, {len(self.memories)})")
        del self.memories[index]

    def step(self) -> None:
        self.timestep += 1


@dataclass
class _ExecutionResult:
    action_type: str
    success: bool
    memory_index: int = -1
    memory_content: str = ""
    reasoning: str = ""


@dataclass
class MemSkillMASMemory(MASMemoryBase):
    """
    MemSkill-style memory backend for nvdamas.

    It keeps MemSkill's skill bank + memory bank + executor loop while exposing
    the standard MASMemoryBase runtime contract used by AutoGen/Strategy.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.skill_bank_path = os.path.join(self.persist_dir, "skill_bank.json")
        self.memory_bank_path = os.path.join(self.persist_dir, "memory_bank.json")
        self.failure_pool_path = os.path.join(self.persist_dir, "failure_cases.json")
        self.trace_path = os.path.join(self.persist_dir, "memskill_trace.jsonl")
        self.replay_path = os.path.join(self.persist_dir, "replay_messages.json")
        self.finalize_report_path = os.path.join(self.persist_dir, "memskill_finalize_report.json")
        skip_noop = bool(self._cfg("memskill_skip_noop", "skip_noop", default=False))
        self.operation_bank = _OperationBank(
            self.skill_bank_path,
            max_ops=int(self._cfg("memskill_max_ops", "max_ops", default=30)),
            skip_noop=skip_noop,
        )
        self._load_external_operation_bank(skip_noop=skip_noop)
        self.memory_bank = _MemoryBank(
            self.memory_bank_path,
            self.embedding_func,
            top_k=int(self._cfg("memskill_top_k", "mem_top_k", default=5)),
        )
        failure_payload = load_json(self.failure_pool_path)
        self.failure_pool: list[dict[str, Any]] = failure_payload if isinstance(failure_payload, list) else []
        replay_payload = load_json(self.replay_path)
        self.replay_pool: list[dict[str, Any]] = replay_payload if isinstance(replay_payload, list) else []
        self._global_retriever: "MemSkillMASMemory | None" = None
        self._last_selected_operation_names: list[str] = []
        self._last_saved_message: MASMessage | None = None
        self.last_saved_message: MASMessage | None = None
        self._designer_counter = 0
        self._finalizing = False
        self._ppo_adapter: _MemSkillPPOAdapter | None = None
        self._ppo_adapter_error: str | None = None

    def _cfg(self, *names: str, default: Any = None) -> Any:
        for name in names:
            if name in self.global_config and self.global_config.get(name) is not None:
                return self.global_config.get(name)
        return default

    def _load_external_operation_bank(self, *, skip_noop: bool) -> None:
        bank_path = str(
            self._cfg(
                "memskill_operation_bank_path",
                "operation_bank_path",
                "memskill_checkpoint_path",
                "checkpoint_path",
                default="",
            )
            or ""
        ).strip()
        if not bank_path or not os.path.exists(bank_path):
            return
        payload: Any = None
        try:
            if bank_path.endswith(".json"):
                payload = load_json(bank_path)
            else:
                import torch

                payload = torch.load(bank_path, map_location="cpu", weights_only=False)
        except TypeError:
            try:
                import torch

                payload = torch.load(bank_path, map_location="cpu")
            except Exception as exc:
                self._trace({"kind": "operation_bank_load_error", "path": bank_path, "error": str(exc)})
                return
        except Exception as exc:
            self._trace({"kind": "operation_bank_load_error", "path": bank_path, "error": str(exc)})
            return
        if self.operation_bank.load_external_payload(payload, skip_noop=skip_noop):
            self._trace(
                {
                    "kind": "operation_bank_loaded",
                    "path": bank_path,
                    "num_operations": len(self.operation_bank.operations),
                }
            )

    def set_global_retriever(self, global_retriever: "MemSkillMASMemory") -> None:
        self._global_retriever = global_retriever

    def add_memory_from_peer(self, mas_message: MASMessage, source_id: str | None = None) -> None:
        if source_id:
            mas_message.add_extra_field("source_id", source_id)
            mas_message.add_extra_field("source_scope", "global")
        self.add_memory(mas_message)

    def export_memory_items(self) -> list[dict[str, Any]]:
        return self.memory_bank.export_items()

    def export_failure_cases(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.failure_pool]

    def add_memory_items_from_peer(self, items: list[Any], source_id: str | None = None) -> int:
        count = self.memory_bank.import_items(items or [], source_id=source_id)
        if count:
            self._persist()
            self._trace({"kind": "merge_memory_items", "source_id": source_id, "count": count})
        return count

    def add_failure_cases_from_peer(self, failures: list[Any], source_id: str | None = None) -> int:
        count = 0
        for row in failures or []:
            if not isinstance(row, dict):
                continue
            copied = dict(row)
            if source_id:
                copied.setdefault("source_id", source_id)
                copied.setdefault("source_scope", "global")
            self.failure_pool.append(copied)
            count += 1
        if count:
            size = int(self._cfg("memskill_failure_pool_size", default=200))
            self.failure_pool = self.failure_pool[-max(size, 1):]
            self._persist()
            self._trace({"kind": "merge_failure_cases", "source_id": source_id, "count": count})
        return count

    def add_memory(self, mas_message: MASMessage) -> None:
        self._last_saved_message = mas_message
        self.last_saved_message = mas_message
        if self._cfg("freeze_memory", "memskill_freeze_memory", default=False):
            return
        self._record_replay_message(mas_message)
        self._construct_from_message(mas_message)

    def _construct_from_message(self, mas_message: MASMessage) -> None:
        if mas_message.label is not True:
            self._record_failure_case(mas_message)
            return

        session_text = self._message_to_session_text(mas_message)
        for chunk_idx, chunk in enumerate(self._chunk_text(session_text)):
            retrieved, retrieved_indices, scores = self.memory_bank.retrieve(
                chunk,
                top_k=int(self._cfg("memskill_construction_top_k", "mem_top_k", default=5)),
            )
            selected_ops = self._select_operations(chunk, retrieved)
            op_names = [op.name for op in selected_ops]
            results = self._execute_operations(selected_ops, chunk, retrieved)
            self._apply_results(results, retrieved_indices, op_names, mas_message, chunk_idx)
            self.memory_bank.step()
            self._last_selected_operation_names = op_names
            self._trace(
                {
                    "kind": "construct",
                    "chunk_idx": chunk_idx,
                    "selected_operations": op_names,
                    "retrieved_indices": retrieved_indices,
                    "scores": scores,
                    "num_results": len(results),
                    "task_preview": (mas_message.task_main or "")[:160],
                }
            )
        self._persist()

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        insight_topk: int = 3,
        **kwargs: Any,
    ) -> tuple[list, list, list]:
        if self._global_retriever is not None and self._global_retriever is not self:
            return self._global_retriever.retrieve_memory(
                query_task=query_task,
                successful_topk=successful_topk,
                failed_topk=failed_topk,
                insight_topk=insight_topk,
                **kwargs,
            )
        requested_topk = max(0, int(successful_topk or 0))
        configured_topk = self._cfg("memskill_retrieve_top_k", "mem_top_k_eval", default=None)
        memory_topk = int(configured_topk) if configured_topk is not None else requested_topk
        if memory_topk > 0:
            memories, _, _ = self.memory_bank.retrieve(query_task or " ", top_k=memory_topk)
        else:
            memories = []
        failed = self._retrieve_failures(query_task or " ", top_k=failed_topk)
        insights = (
            self._skill_insights(query_task or " ", memories, top_k=insight_topk)
            if self._expose_skill_notes_to_solver()
            else []
        )
        success_limit = memory_topk if configured_topk is not None else requested_topk
        return memories[:max(0, success_limit)], failed, insights

    def retrieve_prompt_payload(self, **kwargs: Any) -> dict[str, list[str]]:
        if self._global_retriever is not None and self._global_retriever is not self:
            return self._global_retriever.retrieve_prompt_payload(**kwargs)
        query_task = str(kwargs.get("query_task") or "")
        raw_successful_topk = kwargs.get("successful_topk", 1)
        successful_topk = int(raw_successful_topk) if raw_successful_topk is not None else 1
        failed_topk = int(kwargs.get("failed_topk", 1) or 0)
        insight_topk = int(kwargs.get("insight_topk", 3) or 3)
        configured_topk = self._cfg("memskill_retrieve_top_k", "mem_top_k_eval", default=None)
        memory_topk = int(configured_topk) if configured_topk is not None else successful_topk
        if memory_topk > 0:
            memories, _, _ = self.memory_bank.retrieve(query_task or " ", top_k=memory_topk)
        else:
            memories = []
        failures = self._retrieve_failures(query_task or " ", top_k=failed_topk)
        if self._expose_skill_notes_to_solver():
            selected_ops = self._select_operations(query_task, [str(item) for item in memories])
            insights = [
                f"[MemSkill:{op.name}] {op.description}"
                for op in selected_ops[:max(0, insight_topk)]
                if op.name != "noop"
            ]
            planner_notes = self._render_skill_notes(selected_ops)
            self._last_selected_operation_names = [op.name for op in selected_ops]
        else:
            insights = []
            planner_notes = []
        repair_hints = []
        if failures:
            repair_hints.append(
                "[MemSkill] Similar failed episodes were observed; avoid repeating their action pattern."
            )
        return {
            "reference_cases": [],
            "execution_patterns": [str(item) for item in memories if str(item).strip()],
            "insights": [str(item) for item in insights if str(item).strip()],
            "planner_notes": planner_notes,
            "action_constraints": [],
            "repair_hints": repair_hints,
        }

    def _expose_skill_notes_to_solver(self) -> bool:
        # MemSkill skills are memory-construction actions.  They should not be
        # injected into the task-solving prompt unless explicitly requested.
        return bool(self._cfg("memskill_expose_skill_notes", "expose_skill_notes", default=False))

    def update_memory(self, query: str, **kwargs: Any) -> None:
        if self._cfg("freeze_memory", "memskill_freeze_memory", default=False):
            return
        self._evolve_skills(trigger_query=query, force=True)

    def backward(self, reward, **kwargs: Any) -> None:
        if self._cfg("freeze_memory", "memskill_freeze_memory", default=False):
            return
        numeric_reward = 1.0 if bool(reward) else 0.0
        for name in self._last_selected_operation_names:
            op = self.operation_bank.operations.get(name)
            if op is not None:
                op.update_stats(numeric_reward)
        self.operation_bank.save()
        if bool(self._cfg("memskill_enable_designer", "enable_designer", default=False)):
            self._designer_counter += 1
            freq = max(1, int(self.global_config.get("memskill_designer_freq", 20)))
            if self._designer_counter % freq == 0:
                self._evolve_skills(trigger_query="", force=False)

    def finalize_training(
        self,
        saved_messages: list[Any] | None = None,
        *,
        source_id: str | None = None,
        train_controller: bool | None = None,
        rebuild_memory: bool | None = None,
    ) -> dict[str, Any]:
        """
        Optional nvdamas-compatible MemSkill-A finalize stage.

        Defaults are conservative: the method is a no-op unless the caller enables
        finalize/rebuild explicitly. This keeps existing memory baselines unchanged.
        """
        messages = self._coerce_messages(saved_messages)
        if not messages:
            messages = self._coerce_messages(self.replay_pool)

        train_controller = bool(
            self._cfg("memskill_train_controller", default=False)
            if train_controller is None
            else train_controller
        )
        rebuild_memory = bool(
            self._cfg("memskill_finalize_rebuild", default=True)
            if rebuild_memory is None
            else rebuild_memory
        )
        report: dict[str, Any] = {
            "kind": "memskill_finalize",
            "source_id": source_id,
            "num_replay_messages": len(messages),
            "train_controller_requested": train_controller,
            "rebuild_memory": rebuild_memory,
            "controller_training": "not_requested",
            "num_memories_before": len(self.memory_bank.memories),
            "num_failures_before": len(self.failure_pool),
        }

        if train_controller:
            report["controller_training"] = self._train_controller_from_replay(messages)

        if rebuild_memory:
            self._finalizing = True
            try:
                self.memory_bank.clear()
                self.failure_pool = []
                for msg in messages:
                    if source_id and not msg.get_extra_field("source_id"):
                        msg.add_extra_field("source_id", source_id)
                    self._construct_from_message(msg)
            finally:
                self._finalizing = False
            self._persist()

        report.update(
            {
                "num_memories_after": len(self.memory_bank.memories),
                "num_failures_after": len(self.failure_pool),
                "operation_bank_size": len(self.operation_bank.operations),
            }
        )
        write_json(report, self.finalize_report_path)
        self._trace(report)
        return report

    def _train_controller_from_replay(self, messages: list[MASMessage]) -> str:
        # The heavy original PPO optimizer is intentionally not run inline during
        # nvdamas evaluation.  If a trained checkpoint is configured, we load it
        # and use it for controller inference during the rebuild/eval phases.
        if str(self._cfg("memskill_controller", default="llm")).lower() == "ppo":
            adapter = self._get_ppo_adapter()
            if adapter is not None:
                return "ppo_checkpoint_loaded_for_inference"
            return f"ppo_adapter_not_configured: {self._ppo_adapter_error or 'missing checkpoint'}"
        return "not_applicable_for_non_ppo_controller"

    def _get_ppo_adapter(self) -> _MemSkillPPOAdapter | None:
        if self._ppo_adapter is not None:
            return self._ppo_adapter
        if self._ppo_adapter_error is not None:
            return None
        try:
            self._ppo_adapter = _MemSkillPPOAdapter(self)
            self._trace({"kind": "ppo_adapter_loaded", **self._ppo_adapter.status})
        except Exception as exc:
            self._ppo_adapter_error = str(exc)
            self._trace({"kind": "ppo_adapter_error", "error": self._ppo_adapter_error})
            if bool(self._cfg("memskill_require_ppo", "require_ppo", default=False)):
                raise
        return self._ppo_adapter

    def _persist(self) -> None:
        self.memory_bank.save()
        self.operation_bank.save()
        write_json(self.failure_pool[-int(self._cfg("memskill_failure_pool_size", default=200)):], self.failure_pool_path)
        if self.replay_pool:
            size = int(self._cfg("memskill_replay_pool_size", default=5000))
            write_json(self.replay_pool[-max(size, 1):], self.replay_path)

    def _record_replay_message(self, mas_message: MASMessage) -> None:
        if self._finalizing:
            return
        if not bool(self._cfg("memskill_collect_replay", "memskill_finalize_local", default=False)):
            return
        try:
            self.replay_pool.append(MASMessage.to_dict(mas_message))
            size = int(self._cfg("memskill_replay_pool_size", default=5000))
            self.replay_pool = self.replay_pool[-max(size, 1):]
            write_json(self.replay_pool, self.replay_path)
        except Exception as exc:
            self._trace({"kind": "replay_record_error", "error": str(exc)})

    def _coerce_messages(self, rows: list[Any] | None) -> list[MASMessage]:
        messages: list[MASMessage] = []
        for row in rows or []:
            if isinstance(row, MASMessage):
                messages.append(row)
            elif isinstance(row, dict):
                try:
                    messages.append(MASMessage.from_dict(row))
                except Exception:
                    continue
        return messages

    def _message_to_session_text(self, mas_message: MASMessage) -> str:
        parts = [
            f"Task: {mas_message.task_main or ''}",
            f"Task description:\n{mas_message.task_description or ''}",
            f"Trajectory:\n{mas_message.task_trajectory or ''}",
        ]
        feedback = mas_message.get_extra_field("feedback") if hasattr(mas_message, "get_extra_field") else None
        if feedback:
            parts.append(f"Feedback:\n{feedback}")
        source_id = mas_message.get_extra_field("source_id")
        if source_id:
            parts.append(f"Source domain: {source_id}")
        return "\n\n".join(part.strip() for part in parts if str(part).strip())

    def _chunk_text(self, text: str) -> list[str]:
        chunk_chars = int(self._cfg("memskill_chunk_chars", default=0))
        text = str(text or "").strip()
        if not text:
            return []
        if chunk_chars <= 0 or len(text) <= chunk_chars:
            return [text]
        lines = text.splitlines()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            extra = len(line) + 1
            if current and current_len + extra > chunk_chars:
                chunks.append("\n".join(current).strip())
                current = [line]
                current_len = extra
            else:
                current.append(line)
                current_len += extra
        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _select_operations(self, session_text: str, retrieved_memories: list[str]) -> list[_Operation]:
        controller = str(self.global_config.get("memskill_controller", "llm")).lower()
        if controller == "heuristic":
            names = ["insert"] if not retrieved_memories else ["insert", "update", "delete"]
            return self.operation_bank.by_names(names)
        if controller == "ppo":
            names = self._select_operations_with_ppo(session_text, retrieved_memories)
            selected = self.operation_bank.by_names(names)
            if selected:
                return selected
            if bool(self._cfg("memskill_require_ppo", "require_ppo", default=False)):
                raise RuntimeError(
                    "MemSkill PPO controller was requested but no valid policy selection was produced: "
                    f"{self._ppo_adapter_error or 'unknown error'}"
                )
            self._trace(
                {
                    "kind": "ppo_adapter_fallback",
                    "error": self._ppo_adapter_error,
                    "fallback": "llm",
                }
            )
        names = self._select_operations_with_llm(session_text, retrieved_memories)
        selected = self.operation_bank.by_names(names)
        if selected:
            return selected
        fallback = ["insert"] if not retrieved_memories else ["insert", "update", "delete"]
        return self.operation_bank.by_names(fallback)

    def _select_operations_with_ppo(self, session_text: str, retrieved_memories: list[str]) -> list[str]:
        adapter = self._get_ppo_adapter()
        if adapter is None:
            return []
        candidates = self.operation_bank.candidates()
        names = adapter.select(session_text, retrieved_memories, candidates)
        self._trace(
            {
                "kind": "ppo_select",
                "selected_operations": names,
                "num_candidates": len(candidates),
                "status": adapter.status,
            }
        )
        return names

    def _select_operations_with_llm(self, session_text: str, retrieved_memories: list[str]) -> list[str]:
        skills = []
        for op in self.operation_bank.candidates():
            skills.append(
                f"- {op.name}: {op.description} (action type: {op.update_type.upper()})"
            )
        memories = "\n".join(f"{i}. {m}" for i, m in enumerate(retrieved_memories)) or "(none)"
        prompt = (
            "Select the MemSkill memory-management skills that should be applied to the current text chunk.\n"
            "Return JSON only: {\"skills\": [\"insert\", \"update\", ...]}.\n\n"
            f"Available skills:\n{chr(10).join(skills)}\n\n"
            f"Retrieved memories:\n{memories}\n\n"
            f"Current text chunk:\n{session_text[:6000]}"
        )
        try:
            response = self.llm_model(
                [Message("system", "You are a deterministic memory-skill controller."), Message("user", prompt)],
                temperature=0,
                max_tokens=256,
            )
        except Exception:
            return []
        data = self._parse_json_object(response)
        raw_names = data.get("skills") if isinstance(data, dict) else None
        if not isinstance(raw_names, list):
            raw_names = re.findall(r"\b(insert|update|delete|noop)\b", response or "", flags=re.IGNORECASE)
        return [str(name).strip().lower() for name in raw_names if str(name).strip()]

    def _execute_operations(
        self,
        operations: list[_Operation],
        session_text: str,
        retrieved_memories: list[str],
    ) -> list[_ExecutionResult]:
        non_noop = [op for op in operations if op.update_type.lower() != "noop" and op.name.lower() != "noop"]
        if not non_noop:
            return [_ExecutionResult("NOOP", True, reasoning="Selected NOOP only")]
        prompt = self._build_executor_prompt(non_noop, session_text, retrieved_memories)
        try:
            response = self.llm_model(
                [Message("system", "You are a memory management executor."), Message("user", prompt)],
                temperature=float(self._cfg("memskill_executor_temperature", "temperature", default=0.0)),
                max_tokens=int(self._cfg("memskill_executor_max_tokens", "max_new_tokens", default=1024)),
            )
        except Exception as exc:
            return [_ExecutionResult("NOOP", False, reasoning=f"Executor LLM call failed: {exc}")]
        return self._parse_executor_response(response, len(retrieved_memories))

    def _build_executor_prompt(
        self,
        operations: list[_Operation],
        session_text: str,
        retrieved_memories: list[str],
    ) -> str:
        mem_text = "\n".join(f"{i}. {mem}" for i, mem in enumerate(retrieved_memories)) or "(No existing memories retrieved)"
        skill_blocks = []
        for idx, op in enumerate(operations, start=1):
            skill_blocks.append(
                f"[Skill {idx}] {op.name}\n"
                f"Description: {op.description}\n"
                f"Allowed action: {op.update_type.upper()}\n"
                f"Instructions:\n{op.instruction_template}"
            )
        return (
            "You are a memory management executor. Apply the selected skills to the input text "
            "chunk and retrieved memories, then output memory actions.\n\n"
            f"Input Text Chunk:\n{session_text}\n\n"
            f"Retrieved Memories (0-based index):\n{mem_text}\n\n"
            f"Selected Skills:\n{chr(10).join(skill_blocks)}\n\n"
            "Guidelines:\n"
            "- Apply any skill as needed; a skill may be used multiple times.\n"
            "- Read the input text chunk carefully line by line and apply any skill as needed.\n"
            "- Only use action types supported by the selected skills.\n"
            "- MEMORY_INDEX is 0-based and must reference the retrieved memories list.\n"
            "- Output only action blocks in the format below.\n"
            "- Do not include explanations or REASONING lines.\n\n"
            "INSERT block:\nACTION: INSERT\nMEMORY_ITEM: <concise but complete summary with essential details>\n\n"
            "UPDATE block:\nACTION: UPDATE\nMEMORY_INDEX: <0-based index>\nUPDATED_MEMORY: <concise but complete merged summary>\n\n"
            "DELETE block:\nACTION: DELETE\nMEMORY_INDEX: <0-based index>\n"
        )

    def _parse_executor_response(self, response: str, num_retrieved: int) -> list[_ExecutionResult]:
        text = self._strip_fence(response or "")
        action_matches = list(re.finditer(r"(?<!\w)ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b", text, re.I))
        if not action_matches:
            action_matches = list(
                re.finditer(r"(?im)^(?:[-*]\s*)?(INSERT|UPDATE|DELETE|NOOP)\s*(?::|=|-)?\s*$", text)
            )
        if not action_matches:
            data = self._parse_json_object(text)
            if isinstance(data, list):
                return self._parse_json_actions(data, num_retrieved)
            actions = data.get("actions") if isinstance(data, dict) else None
            if isinstance(actions, list):
                return self._parse_json_actions(actions, num_retrieved)
            return [_ExecutionResult("NOOP", False, reasoning="Failed to parse executor response")]
        results: list[_ExecutionResult] = []
        for idx, match in enumerate(action_matches):
            end = action_matches[idx + 1].start() if idx + 1 < len(action_matches) else len(text)
            results.extend(self._parse_action_block(text[match.start():end], num_retrieved))
        return results or [_ExecutionResult("NOOP", False, reasoning="No executor actions parsed")]

    def _parse_action_block(self, block: str, num_retrieved: int) -> list[_ExecutionResult]:
        action_match = re.search(r"ACTION\s*(?::|=|-)?\s*(INSERT|UPDATE|DELETE|NOOP)\b", block, re.I)
        if not action_match:
            return []
        action = action_match.group(1).upper()
        if action == "NOOP":
            return [_ExecutionResult("NOOP", True)]
        if action == "INSERT":
            matches = re.findall(
                r"(?:MEMORY[_ ]ITEM|NEW[_ ]MEMORY|CONTENT|MEMORY)\s*(?::|=|-)?\s*(.+?)(?=\n\s*(?:ACTION|REASONING|MEMORY[_ ]ITEM|UPDATED[_ ]MEMORY|MEMORY[_ ]INDEX)\b|$)",
                block,
                flags=re.I | re.S,
            )
            return [_ExecutionResult("INSERT", True, memory_content=m.strip()) for m in matches if m.strip()]
        indices = [int(x) for x in re.findall(r"MEMORY[_ ]INDEX\s*(?::|=|-)?\s*(\d+)", block, flags=re.I)]
        if action == "DELETE":
            return [
                _ExecutionResult("DELETE", True, memory_index=i)
                for i in indices
                if 0 <= i < num_retrieved
            ]
        updates = re.findall(
            r"UPDATED[_ ]MEMORY\s*(?::|=|-)?\s*(.+?)(?=\n\s*(?:ACTION|REASONING|UPDATED[_ ]MEMORY|MEMORY[_ ]INDEX)\b|$)",
            block,
            flags=re.I | re.S,
        )
        return [
            _ExecutionResult("UPDATE", True, memory_index=mem_idx, memory_content=content.strip())
            for mem_idx, content in zip(indices, updates)
            if 0 <= mem_idx < num_retrieved and content.strip()
        ]

    def _parse_json_actions(self, rows: list[Any], num_retrieved: int) -> list[_ExecutionResult]:
        out: list[_ExecutionResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action", row.get("ACTION", ""))).strip().upper()
            if action == "INSERT":
                content = str(row.get("memory_item", row.get("MEMORY_ITEM", ""))).strip()
                if content:
                    out.append(_ExecutionResult("INSERT", True, memory_content=content))
            elif action == "UPDATE":
                content = str(row.get("updated_memory", row.get("UPDATED_MEMORY", ""))).strip()
                try:
                    idx = int(row.get("memory_index", row.get("MEMORY_INDEX", -1)))
                except Exception:
                    idx = -1
                if 0 <= idx < num_retrieved and content:
                    out.append(_ExecutionResult("UPDATE", True, memory_index=idx, memory_content=content))
            elif action == "DELETE":
                try:
                    idx = int(row.get("memory_index", row.get("MEMORY_INDEX", -1)))
                except Exception:
                    idx = -1
                if 0 <= idx < num_retrieved:
                    out.append(_ExecutionResult("DELETE", True, memory_index=idx))
            elif action == "NOOP":
                out.append(_ExecutionResult("NOOP", True))
        return out

    def _apply_results(
        self,
        results: list[_ExecutionResult],
        retrieved_indices: list[int],
        op_names: list[str],
        mas_message: MASMessage,
        chunk_idx: int,
    ) -> None:
        updates = [r for r in results if r.success and r.action_type == "UPDATE"]
        deletes = [r for r in results if r.success and r.action_type == "DELETE"]
        inserts = [r for r in results if r.success and r.action_type == "INSERT"]
        for result in updates:
            try:
                self.memory_bank.update(retrieved_indices[result.memory_index], result.memory_content, operations=op_names)
            except Exception as exc:
                self._trace({"kind": "apply_error", "action": "UPDATE", "error": str(exc)})
        delete_targets: list[int] = []
        for result in deletes:
            try:
                delete_targets.append(retrieved_indices[result.memory_index])
            except Exception as exc:
                self._trace({"kind": "apply_error", "action": "DELETE", "error": str(exc)})
        for actual_idx in sorted(set(delete_targets), reverse=True):
            try:
                self.memory_bank.delete(actual_idx)
            except Exception as exc:
                self._trace({"kind": "apply_error", "action": "DELETE", "error": str(exc)})
        for result in inserts:
            self.memory_bank.add(
                result.memory_content,
                operations=op_names,
                metadata={
                    "task_main": mas_message.task_main,
                    "label": mas_message.label,
                    "chunk_idx": chunk_idx,
                    "source_id": mas_message.get_extra_field("source_id"),
                },
            )

    def _retrieve_failures(self, query: str, *, top_k: int) -> list[str]:
        if top_k <= 0 or not self.failure_pool:
            return []
        q = np.asarray(self.memory_bank.embed(query or " "), dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self.failure_pool:
            emb = np.asarray(row.get("embedding") or [], dtype=np.float32)
            if emb.size == q.size:
                score = float(np.dot(q_norm, emb / (np.linalg.norm(emb) + 1e-8)))
            else:
                score = 0.0
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            f"Failed case: {row.get('task_main', '')}\nFeedback: {row.get('feedback', '')}\nTrajectory: {row.get('trajectory', '')[:1200]}"
            for _, row in scored[:top_k]
        ]

    def _skill_insights(self, query: str, memories: list[str], *, top_k: int) -> list[str]:
        ops = self._select_operations(query, memories)
        return [
            f"[MemSkill:{op.name}] {op.description}"
            for op in ops[:max(0, top_k)]
            if op.name != "noop"
        ]

    def _render_skill_notes(self, ops: list[_Operation]) -> list[str]:
        useful = [op for op in ops if op.name != "noop"]
        if not useful:
            return []
        lines = ["### MEMSKILL SELECTED MEMORY SKILLS"]
        for op in useful:
            lines.append(f"- {op.name}: {op.description}")
        return ["\n".join(lines)]

    def _record_failure_case(self, mas_message: MASMessage) -> None:
        query = mas_message.task_main or mas_message.task_description or ""
        feedback = ""
        desc = mas_message.task_description or ""
        marker = "- Environment feedback"
        if marker in desc:
            feedback = desc.split(marker, 1)[-1].strip()
        row = {
            "task_main": mas_message.task_main,
            "task_description": mas_message.task_description,
            "trajectory": mas_message.task_trajectory,
            "feedback": feedback,
            "source_id": mas_message.get_extra_field("source_id"),
            "created_at": time.time(),
            "embedding": self.memory_bank.embed(query or " "),
        }
        self.failure_pool.append(row)
        size = int(self.global_config.get("memskill_failure_pool_size", 200))
        self.failure_pool = self.failure_pool[-max(size, 1):]
        self._persist()

    def _evolve_skills(self, *, trigger_query: str, force: bool) -> None:
        if not self.failure_pool:
            return
        if not force and len(self.failure_pool) < int(self.global_config.get("memskill_designer_min_failures", 3)):
            return
        cases = self.failure_pool[-int(self.global_config.get("memskill_designer_cases", 5)):]
        existing = "\n".join(
            f"- {op.name}: {op.description}\n{op.instruction_template}"
            for op in self.operation_bank.candidates()
        )
        case_text = "\n\n".join(
            f"Task: {row.get('task_main')}\nFeedback: {row.get('feedback')}\nTrajectory: {str(row.get('trajectory', ''))[:1000]}"
            for row in cases
        )
        prompt = (
            "You are the MemSkill designer. Improve the memory-management skill bank using hard failure cases.\n"
            "Return JSON only with optional key add_or_update, a list of skills. Each skill must have "
            "name, description, instruction_template, update_type.\n\n"
            f"Trigger query: {trigger_query}\n\nExisting skills:\n{existing}\n\nFailure cases:\n{case_text}"
        )
        try:
            response = self.llm_model(
                [Message("system", "You evolve reusable memory skills."), Message("user", prompt)],
                temperature=0,
                max_tokens=int(self._cfg("memskill_designer_max_tokens", "max_new_tokens", default=1024)),
            )
        except Exception as exc:
            self._trace({"kind": "designer_error", "error": str(exc)})
            return
        data = self._parse_json_object(response)
        rows = data.get("add_or_update") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = re.sub(r"[^a-z0-9_]+", "_", str(row.get("name", "")).strip().lower()).strip("_")
            if not name:
                continue
            op = _Operation(
                name=name,
                description=str(row.get("description", "")).strip(),
                instruction_template=str(row.get("instruction_template", "")).strip(),
                update_type=str(row.get("update_type", "insert")).strip().lower(),
                meta_info=_new_meta("designer"),
            )
            if op.description and op.instruction_template and op.update_type in {"insert", "update", "delete", "noop"}:
                self.operation_bank.add_or_replace(op)
        self.operation_bank.save()

    def _parse_json_object(self, text: str) -> Any:
        raw = self._strip_fence(text or "")
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            list_start = raw.find("[")
            list_end = raw.rfind("]") + 1
            if list_start >= 0 and list_end > list_start:
                try:
                    parsed = json.loads(raw[list_start:list_end])
                    return parsed if isinstance(parsed, list) else {}
                except Exception:
                    return {}
            return {}
        try:
            return json.loads(raw[start:end])
        except Exception:
            return {}

    @staticmethod
    def _strip_fence(text: str) -> str:
        stripped = str(text or "").strip()
        if stripped.startswith("```"):
            parts = stripped.split("```")
            if len(parts) >= 3:
                stripped = parts[1]
                if "\n" in stripped:
                    first, rest = stripped.split("\n", 1)
                    if first.strip().lower() in {"json", "text", "python"}:
                        stripped = rest
        return stripped.strip()

    def _trace(self, row: dict[str, Any]) -> None:
        try:
            row = {"ts": time.time(), **row}
            with open(self.trace_path, "a", encoding="utf-8") as writer:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass
