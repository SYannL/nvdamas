from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base_env import BaseEnv, BaseRecorder

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BFCL_PKG = _REPO_ROOT / "data" / "gorilla" / "berkeley-function-call-leaderboard"
_FUNC_DOC_DIR = _BFCL_PKG / "bfcl_eval" / "data" / "multi_turn_func_doc"

_CLASS_TO_DOC = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def _ensure_bfcl_on_path() -> Path:
    p = _BFCL_PKG.resolve()
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return p


def _load_tool_functions(involved_classes: list[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for cls in involved_classes or []:
        fname = _CLASS_TO_DOC.get(str(cls))
        if not fname:
            continue
        path = _FUNC_DOC_DIR / fname
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tools.append(json.loads(line))
    return tools


def _turn_user_text(turn_block: Any) -> str:
    if isinstance(turn_block, list):
        parts: list[str] = []
        for msg in turn_block:
            if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                parts.append(str(msg.get("content", "")).strip())
        return "\n".join(parts).strip()
    return str(turn_block or "").strip()


def _try_official_python_fc_calls(raw: str) -> list[str] | None:
    """
    与 BFCL leaderboard 一致：AST 解码（见 tasks/envs/bfcl_mt_official_decode.py），
    支持 [call1, call2]、单 call、以及自动补外层 []。
    """
    s = raw.strip()
    if not s or s.startswith("Exec") or s.lower().startswith("finishturn"):
        return None
    try:
        from tasks.envs.bfcl_mt_official_decode import decode_python_fc_calls

        out = decode_python_fc_calls(s)
        if not out:
            return None
        return list(out)
    except Exception:
        return None


class BfclMtEnv(BaseEnv):
    """BFCL V4 multi_turn_base: 官方 Python 调用列表 + 可选 legacy Exec/ExecMany + FinishTurn。"""

    def __init__(self, env_config: dict[str, Any], max_trials: int = 120) -> None:
        self.env_config = env_config
        self.max_trials = int(max_trials)
        self.memco_domain = "bfcl_mt"
        self.config: dict[str, Any] = {}
        self.reward: float = 0.0
        self.steps: int = 0
        self.done: bool = False
        self.won: bool = False
        self.current_history: list[dict[str, Any]] = []
        self.last_admissible_commands: list[str] = []
        self.goal_instruction: str = ""
        self.game_name: str = ""
        self.goal_literals_text: list[str] = []
        self.current_literals_text: list[str] = []
        self.infos: dict[str, Any] = {}
        self.states: list[str] = []
        self._test_entry: dict[str, Any] = {}
        self._ground_truth: list[Any] = []
        self._turn_texts: list[str] = []
        self._turn_idx: int = 0
        self._episode_model: list[list[list[str]]] = []
        self._current_turn_steps: list[list[str]] = []
        self._tool_summary: str = ""
        self._functions_text: str = ""

    def set_env(self, configs: dict) -> tuple[str, str]:
        _ensure_bfcl_on_path()
        self.config = dict(configs or {})
        entry = self.config.get("bfcl_entry") or {}
        if not entry.get("id"):
            raise ValueError("bfcl_mt task needs bfcl_entry.id")
        gt = self.config.get("bfcl_ground_truth")
        if not isinstance(gt, list):
            raise ValueError("bfcl_mt task needs bfcl_ground_truth list")
        self._test_entry = {
            "id": entry["id"],
            "question": entry.get("question", []),
            "initial_config": entry.get("initial_config", {}),
            "involved_classes": list(entry.get("involved_classes") or []),
        }
        self._ground_truth = gt
        q = entry.get("question") or []
        self._turn_texts = [_turn_user_text(t) for t in q]
        if not self._turn_texts:
            raise ValueError("bfcl_entry.question is empty")
        self.game_name = str(self.config.get("bfcl_shard_domain") or "bfcl_shard").strip() or "bfcl_shard"
        tools = _load_tool_functions(self._test_entry["involved_classes"])
        self._functions_text = json.dumps(tools, ensure_ascii=False, indent=2)[:120_000]
        ic = ", ".join(self._test_entry["involved_classes"])
        self.goal_instruction = f"BFCL task {entry['id']} | APIs: {ic}"
        self._turn_idx = 0
        self._episode_model = []
        self._current_turn_steps = []
        self.reward = 0.0
        self.done = False
        self.won = False
        self.steps = 0
        self._tool_summary = ""
        self.last_admissible_commands = self._default_admissible()
        self.states = [self._observation_text()]
        self.current_literals_text = list(self._test_entry["involved_classes"])
        self.goal_literals_text = list(self._test_entry["involved_classes"])
        self.current_history = [
            {
                "Step": 0,
                "Observation": self.states[0],
                "Goal": self.goal_instruction,
                "Goal Literals": list(self.goal_literals_text),
                "Current Literals": list(self.current_literals_text),
                "Admissible Commands": list(self.last_admissible_commands),
                "Score": 0.0,
                "Reward": 0.0,
                "Done": False,
                "Valid": True,
            }
        ]
        task_main = self.goal_instruction
        tool_blob = self._functions_text[:24_000]
        if len(self._functions_text) > 24_000:
            tool_blob += "\n... (tools JSON truncated) ..."
        task_description = self.states[0] + f"\n\n### Tool definitions (JSON array)\n{tool_blob}"
        return task_main, task_description

    def _observation_text(self) -> str:
        turn_hdr = f"=== BFCL turn {self._turn_idx + 1}/{len(self._turn_texts)} ===\n"
        user_part = self._turn_texts[self._turn_idx] if self._turn_idx < len(self._turn_texts) else ""
        api = ", ".join(self._test_entry.get("involved_classes") or [])
        tail = ""
        if self._tool_summary:
            tail = f"\nLast tool batch stdout:\n{self._tool_summary[:8000]}"
        return (
            f"{turn_hdr}"
            f"User:\n{user_part}\n\n"
            f"Involved API classes: {api}\n"
            f"{tail}"
        )

    @staticmethod
    def _default_admissible() -> list[str]:
        return [
            "BFCL Python: [call1(...), call2(...)] or single call(...)",
            "Exec[method_call]  (legacy)",
            "ExecMany[c1###c2]  (legacy)",
            "FinishTurn[]",
        ]

    @classmethod
    def process_action(cls, action: str) -> str:
        text = str(action or "").strip()
        while text.startswith(">"):
            text = text[1:].lstrip()
        if "```" in text:
            parts = text.split("```")
            for block in parts:
                b = block.strip()
                if (
                    b.startswith("Exec")
                    or b.startswith("FinishTurn")
                    or b.startswith("finishturn")
                    or b.startswith("[")
                    or re.match(r"^[\w.]+\s*\(", b)
                ):
                    text = b
                    break
        if ":" in text and not text.startswith("Exec") and not text.strip().startswith("["):
            low = text.lower()
            if "execturn" not in low:
                text = text.split(":", 1)[-1].strip()
        text = text.strip()
        # Some models occasionally emit malformed FinishTurn tokens (e.g. `nishTurn[]`).
        # Normalize them so the runtime protocol doesn't hard-crash.
        compact = re.sub(r"\s+", "", text.lower())
        if compact in ("finishturn[]", "nishturn[]", "ishturn[]"):
            return "FinishTurn[]"
        return text

    def step(self, action: str) -> tuple[str, float, bool]:
        self.steps += 1
        raw = self.process_action(action)
        low = raw.lower()
        if (
            "thought" in low
            and not raw.startswith("Exec")
            and not raw.startswith("FinishTurn")
            and not raw.strip().startswith("[")
        ):
            obs = "OK."
            self._append_history(raw, obs, 0.0, False)
            return obs, 0.0, False

        if raw.startswith("FinishTurn"):
            self._episode_model.append(list(self._current_turn_steps))
            self._current_turn_steps = []
            self._turn_idx += 1
            if self._turn_idx >= len(self._turn_texts):
                ok = self._finalize_episode()
                self.done = True
                self.won = ok
                self.reward = 1.0 if ok else 0.0
                obs = "Episode complete. " + ("PASS (BFCL checker)" if ok else "FAIL (BFCL checker)")
                self._append_history(raw, obs, self.reward, True)
                return obs, self.reward, True
            obs = self._observation_text()
            self.states.append(obs)
            self._append_history(raw, obs, 0.0, False)
            return obs, 0.0, False

        calls: list[str] = []
        if raw.startswith("ExecMany["):
            inner = _extract_brackets(raw, "ExecMany")
            calls = [c.strip() for c in inner.split("###") if c.strip()]
        elif raw.startswith("Exec["):
            inner = _extract_brackets(raw, "Exec")
            if inner.strip():
                calls = [inner.strip()]
        else:
            official = _try_official_python_fc_calls(raw)
            if official:
                calls = official
            else:
                obs = (
                    "Invalid action. Use BFCL Python format [call1(...), call2(...)] or a single call(...); "
                    "legacy: Exec[...], ExecMany[a###b], or FinishTurn[]."
                )
                self._append_history(raw, obs, -0.1, False)
                return obs, -0.1, False

        if not calls:
            obs = "No calls parsed."
            self._append_history(raw, obs, 0.0, False)
            return obs, 0.0, False

        _ensure_bfcl_on_path()
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

        outs: list[str] = []
        try:
            exec_out, _instances = execute_multi_turn_func_call(
                func_call_list=calls,
                initial_config=self._test_entry["initial_config"],
                involved_classes=self._test_entry["involved_classes"],
                model_name="nvdamasgm_solver",
                test_entry_id=str(self._test_entry["id"]),
                long_context=False,
                is_evaL_run=False,
            )
            outs = [str(x) for x in exec_out]
        except Exception as e:
            outs = [f"Error: {type(e).__name__}: {e}"]
        self._current_turn_steps.append(calls)
        self._tool_summary = "\n".join(outs)[:12_000]
        obs = self._tool_summary[:8000] if self._tool_summary else "(no output)"
        self.current_literals_text = [self._tool_summary[:400]]
        self.states.append(obs)
        self._append_history(raw, obs, 0.0, False)
        return obs, 0.0, False

    def _finalize_episode(self) -> bool:
        _ensure_bfcl_on_path()
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

        model_list = list(self._episode_model)
        if len(model_list) != len(self._ground_truth):
            return False
        chk = multi_turn_checker(
            model_list,
            self._ground_truth,
            self._test_entry,
            "multi_turn_base",
            "nvdamasgm",
        )
        return bool(chk.get("valid"))

    def feedback(self) -> tuple[float, bool, str]:
        ok = bool(self.reward >= 1.0)
        return float(self.reward), ok, "BFCL episode finished." if ok else "BFCL episode failed."

    def _append_history(self, action: str, observation: str, reward: float, done: bool) -> None:
        self.last_admissible_commands = self._default_admissible()
        self.current_history.append(
            {
                "Step": int(self.steps),
                "Action": action,
                "Observation": observation,
                "Goal": self.goal_instruction,
                "Goal Literals": list(self.goal_literals_text),
                "Current Literals": list(self.current_literals_text),
                "Admissible Commands": list(self.last_admissible_commands),
                "Score": float(self.reward),
                "Reward": float(reward),
                "Done": bool(done),
                "Valid": True,
            }
        )

    def has_exportable_history(self) -> bool:
        return bool(self.current_history)

    def export_memco_history(
        self,
        output_dir: str,
        *,
        model_id: str = "",
        status_override: str | None = None,
    ) -> str | None:
        if not self.has_exportable_history():
            return None
        bid = str(self._test_entry.get("id") or self.config.get("bfcl_id") or "unknown")
        final_done = bool(self.current_history[-1].get("Done", False)) if self.current_history else False
        final_score = float(self.reward or 0.0)
        status = status_override or ("success" if final_done and final_score > 0 else "fail")
        domain = str(self.config.get("bfcl_shard_domain") or self.game_name or "default")
        payload = {
            "last_updated": __import__("time").strftime("%Y%m%d_%H%M%S"),
            "game_file": "",
            "game_name": self.game_name,
            "game_index": bid,
            "game_task": self.goal_instruction,
            "goal_literals": list(self.goal_literals_text),
            "status": status,
            "step_count": max(len(self.current_history) - 1, 0),
            "final_score": final_score,
            "history": list(self.current_history),
            "model_id": model_id,
            "memco_domain": self.memco_domain,
            "task_family": f"bfcl_mt:{domain}",
            "task_config": dict(self.config),
        }
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = abs(hash(bid)) % 10_000_000
        safe_domain = domain.replace("/", "_")
        out_path = out_dir / f"history_bfcl_mt_{safe_domain}_{digest}_{status}.json"
        with open(out_path, "w", encoding="utf-8") as writer:
            json.dump(payload, writer, ensure_ascii=False, indent=2)
        return str(out_path)


def _extract_brackets(raw: str, prefix: str) -> str:
    """
    取 Exec[...] / ExecMany[...] 中第一层方括号内的内容。
    须用深度计数：参数里常有 list 字面量 door=[...]，且模型可能把多个 Exec 连成
    Exec[a]###Exec[b]（若用 rfind(']') 会截到最外层错误，导致 BFCL 执行报 unmatched ']'）。
    """
    s = raw.strip()
    head = f"{prefix}["
    if not s.startswith(head):
        return ""
    i = len(prefix)
    assert s[i] == "["
    depth = 1
    for j in range(i + 1, len(s)):
        ch = s[j]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j]
    return ""


@dataclass
class BfclMtRecorder(BaseRecorder):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.task = "bfcl_mt"
        self.counts = 0
        self.dones = 0
        self.rewards = 0.0

    def task_begin(self, task_id: int, task_config: dict) -> None:
        super().task_begin(task_id, task_config)
        self.log(f"---------- BFCL MT Task: {task_id} ----------")

    def task_end(self, reward: float, done: bool) -> None:
        self.rewards += float(reward)
        self.dones += 1 if done else 0
        self.counts += 1
        ave_r = self.rewards / self.counts if self.counts else 0.0
        ave_d = self.dones / self.counts if self.counts else 0.0
        self.log(f"reward: {reward}, ave reward: {ave_r}. done: {done}, ave done: {ave_d}")
