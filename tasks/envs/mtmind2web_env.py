import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from .base_env import BaseEnv, BaseRecorder


_ACTION_RE = re.compile(r"^\[(?P<elem_type>[^\]]*)\]\s*(?P<elem_text>.*?)\s*->\s*(?P<op>[A-Z]+)(?::\s*(?P<value>.*))?$")


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _token_f1(pred: str, gold: str) -> float:
    p_toks = _normalize_text(pred).split()
    g_toks = _normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    p_counts = {}
    g_counts = {}
    for t in p_toks:
        p_counts[t] = p_counts.get(t, 0) + 1
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, g_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _parse_action_repr(line: str) -> dict[str, str]:
    m = _ACTION_RE.match((line or "").strip())
    if not m:
        return {
            "elem_type": "",
            "elem_text": _normalize_text(line),
            "op": "",
            "value": "",
            "raw": _normalize_text(line),
        }
    return {
        "elem_type": _normalize_text(m.group("elem_type")),
        "elem_text": _normalize_text(m.group("elem_text")),
        "op": _normalize_text(m.group("op")).upper(),
        "value": _normalize_text(m.group("value") or ""),
        "raw": _normalize_text(line),
    }


def _parse_finish_payload(payload: str) -> list[str]:
    text = (payload or "").strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict):
            actions = obj.get("actions")
            if isinstance(actions, list):
                return [str(x).strip() for x in actions if str(x).strip()]
    except Exception:
        pass
    if "\n" in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return [ln for ln in lines if "->" in ln]
    parts = [p.strip() for p in text.split("||") if p.strip()]
    return parts


class MTMind2WebEnv(BaseEnv):
    def __init__(self, env_config: dict[str, Any], max_trials: int = 4):
        self.env_config = env_config
        self.max_trials = max_trials
        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        required = ["task", "gold_action_reprs", "website", "domain", "subdomain"]
        for key in required:
            if configs.get(key) is None:
                raise ValueError(f"Missing required field `{key}` in task config.")
        self.config = configs
        self.gold_action_reprs: list[str] = list(configs.get("gold_action_reprs") or [])
        # Candidate action pool for constrained generation (no invented actions).
        # Current version uses all labeled actions in this turn as the candidate set.
        self.action_candidates: list[str] = list(dict.fromkeys(self.gold_action_reprs))
        self.latest_metrics = {}
        cand_lines = [
            f"{idx + 1}. {act}" for idx, act in enumerate(self.action_candidates)
        ]
        cand_block = "\n".join(cand_lines) if cand_lines else "(empty)"
        task_text = (
            f"Website: {configs.get('website')}\n"
            f"Domain: {configs.get('domain')}\n"
            f"Subdomain: {configs.get('subdomain')}\n"
            f"Task: {configs.get('task')}\n\n"
            "Candidate Actions (must choose from this list only):\n"
            f"{cand_block}\n\n"
            "Output one final action sequence using candidate indices only:\n"
            "Action: Finish[<json_array_of_candidate_indices>]\n"
            "Example: Finish[[1,2,3]]\n"
            "Rules:\n"
            "- Do not invent new actions.\n"
            "- Every index must be from the candidate list.\n"
            "- Keep predicted execution order in your index sequence.\n"
            "- Do not include explanation outside Thought/Action format."
        )
        return task_text, task_text

    def reset(self) -> None:
        self.reward = 0.0
        self.done = False
        self.latest_metrics = {}

    @classmethod
    def process_action(cls, action: str) -> str:
        text = (action or "").strip()
        if not text:
            return ""
        # Most robust path: if the model includes Thought + Action together,
        # directly extract the first Finish[...] span from the full text.
        m = re.search(r"Finish\[(.*)\]", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return f"Finish[{m.group(1)}]"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if ln.lower().startswith("action:"):
                return ln[len("action:") :].strip()
        return lines[-1] if lines else text

    @staticmethod
    def _parse_action(action: str) -> tuple[Optional[str], Optional[str]]:
        m = re.match(r"^(\w+)\[(.*)\]$", action.strip(), flags=re.DOTALL)
        if not m:
            return None, None
        return m.group(1), m.group(2)

    def _evaluate(self, pred_actions: list[str]) -> dict[str, float]:
        gold = self.gold_action_reprs
        n = max(len(gold), len(pred_actions))
        if n == 0:
            return {"ele_acc": 1.0, "op_f1": 1.0, "ssr": 1.0, "tsr": 1.0}

        ele_hits = 0
        op_f1_sum = 0.0
        step_successes = 0
        for i in range(n):
            g = gold[i] if i < len(gold) else ""
            p = pred_actions[i] if i < len(pred_actions) else ""
            g_parsed = _parse_action_repr(g)
            p_parsed = _parse_action_repr(p)

            elem_match = (
                g_parsed["elem_type"] == p_parsed["elem_type"]
                and g_parsed["elem_text"] == p_parsed["elem_text"]
            )
            if elem_match:
                ele_hits += 1

            g_op_token = f"{g_parsed['op']}:{g_parsed['value']}".strip(":")
            p_op_token = f"{p_parsed['op']}:{p_parsed['value']}".strip(":")
            op_f1_sum += _token_f1(p_op_token, g_op_token)

            need_value = g_parsed["op"] in {"TYPE", "SELECT"}
            op_match = g_parsed["op"] == p_parsed["op"]
            value_match = (not need_value) or (g_parsed["value"] == p_parsed["value"])
            if elem_match and op_match and value_match:
                step_successes += 1

        ele_acc = ele_hits / n
        op_f1 = op_f1_sum / n
        ssr = step_successes / n
        tsr = 1.0 if step_successes == n else 0.0
        return {"ele_acc": ele_acc, "op_f1": op_f1, "ssr": ssr, "tsr": tsr}

    def step(self, action: str) -> tuple[str, float, bool]:
        parsed_action = self.process_action(action)
        action_type, payload = self._parse_action(parsed_action)
        if action_type is None:
            return (
                'Invalid action. Use: Action: Finish[["[button] Search -> CLICK", "..."]]',
                -1.0,
                False,
            )
        if action_type.lower() != "finish":
            return "Only Finish[...] is supported in this eval env.", -1.0, False

        pred_actions_raw = _parse_finish_payload(payload or "")
        # If payload is index-based (e.g., [1,2,3]), project to candidate actions.
        pred_actions: list[str] = []
        for item in pred_actions_raw:
            s = str(item).strip()
            if s.isdigit():
                idx = int(s) - 1
                if 0 <= idx < len(self.action_candidates):
                    pred_actions.append(self.action_candidates[idx])
                continue
            pred_actions.append(s)
        metrics = self._evaluate(pred_actions)
        self.latest_metrics = metrics
        self.reward = metrics["tsr"]
        self.done = True
        obs = (
            f"Ele.Acc={metrics['ele_acc']:.4f}, "
            f"Op.F1={metrics['op_f1']:.4f}, "
            f"SSR={metrics['ssr']:.4f}, "
            f"TSR={metrics['tsr']:.4f}"
        )
        return obs, self.reward, True

    def feedback(self) -> tuple[float, bool, str]:
        if not self.done:
            return 0.0, False, "Task not finished."
        m = self.latest_metrics or {}
        msg = (
            f"Turn evaluation finished. "
            f"Ele.Acc={m.get('ele_acc', 0.0):.4f}, "
            f"Op.F1={m.get('op_f1', 0.0):.4f}, "
            f"SSR={m.get('ssr', 0.0):.4f}, "
            f"TSR={m.get('tsr', 0.0):.4f}"
        )
        return self.reward, bool(self.reward == 1.0), msg


@dataclass
class MTMind2WebRecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = "mtmind2web"
        self.counts = 0
        self.sum_ele_acc = 0.0
        self.sum_op_f1 = 0.0
        self.sum_ssr = 0.0
        self.sum_tsr = 0.0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        self.log(f"---------- Task: {task_id} ----------")

    def task_end(self, reward: float, done: bool):
        env_metrics = {}
        try:
            env_metrics = getattr(self.current_task_config.get("env_ref"), "latest_metrics", {})
        except Exception:
            env_metrics = {}
        ele = float(env_metrics.get("ele_acc", 0.0))
        opf = float(env_metrics.get("op_f1", 0.0))
        ssr = float(env_metrics.get("ssr", 0.0))
        tsr = float(env_metrics.get("tsr", 0.0))
        self.counts += 1
        self.sum_ele_acc += ele
        self.sum_op_f1 += opf
        self.sum_ssr += ssr
        self.sum_tsr += tsr
        n = self.counts
        self.log(
            f"task_metrics: Ele.Acc={ele:.4f}, Op.F1={opf:.4f}, SSR={ssr:.4f}, TSR={tsr:.4f}\n"
            f"running_avg: Ele.Acc={self.sum_ele_acc/n:.4f}, Op.F1={self.sum_op_f1/n:.4f}, "
            f"SSR={self.sum_ssr/n:.4f}, TSR={self.sum_tsr/n:.4f}"
        )

    def dataset_end(self) -> None:
        super().dataset_end()
        n = self.counts
        if n <= 0:
            return
        msg = (
            f"FINAL_DATASET_METRICS: tasks={n} "
            f"Ele.Acc={self.sum_ele_acc/n:.4f} "
            f"Op.F1={self.sum_op_f1/n:.4f} "
            f"SSR={self.sum_ssr/n:.4f} "
            f"TSR={self.sum_tsr/n:.4f}"
        )
        self.log(msg)
        print(msg, flush=True)
