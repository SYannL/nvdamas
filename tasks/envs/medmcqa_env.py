import re
import os
import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Optional

from openai import AsyncOpenAI

from .base_env import BaseEnv, BaseRecorder
from .utils import LangChainWiki, match_exactly, normalize_answer


OPENAI_SEARCH_MODEL = "gpt-4o-mini"

# ANSI colors for correct/incorrect feedback in observation
_ANSI_GREEN = "\033[92m"
_ANSI_RED = "\033[91m"
_ANSI_RESET = "\033[0m"


async def _async_openai_search_summary(query: str) -> str:
    """
    Use OpenAI Responses API + web_search tool to retrieve web evidence
    for the given query and summarize it into a few bullet points.
    This function only performs search + evidence summarization and
    does NOT choose an answer option.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable OPENAI_API_KEY is not set, cannot use OpenAI web_search.")

    client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key=api_key,
    )

    system_prompt = (
        "You are a medical research assistant. "
        "Use the web_search tool to gather up-to-date factual information relevant to the query. "
        "Then summarize the key evidence in 3-6 short bullet points. "
        "Do NOT choose an option; only provide evidence."
    )

    response = await client.responses.create(
        model=OPENAI_SEARCH_MODEL,
        tools=[{"type": "web_search"}],
        input=f"{system_prompt}\n\nQuery: {query}",
    )

    text = getattr(response, "output_text", None)
    if not text:
        text = str(response)
    return text


def openai_search_summary(query: str) -> str:
    """
    Synchronous wrapper around the async web_search helper for CLI usage.
    If the call fails, return a short fallback message instead of raising,
    so that the MAS pipeline can continue running.
    """
    try:
        return asyncio.run(_async_openai_search_summary(query))
    except Exception as e:
        print(f"[MedMCQAEnv] openai web_search failed: {e}")
        return "Web search failed or unavailable."


class MedMCQAEnv(BaseEnv):
    def __init__(
        self,
        env_config: dict[str, Any],
        max_trials: int = 7,
    ) -> None:
        self.env_config = env_config
        self.explorer = LangChainWiki()
        self.max_trials: int = max_trials
        self.reset()

    def set_env(self, configs: dict) -> tuple[str, str]:
        if configs.get("answer") is None:
            raise ValueError("Please provide the answer for the question.")
        if configs.get("task") is None:
            raise ValueError("The configs dict should have the `task` attribute.")
        if configs.get("options") is None:
            raise ValueError("The configs dict should have the `options` attribute.")
        self.config = configs

        task_value = self.config.get("task")
        options = self.config.get("options", {})
        options_block = "\n".join([f"{key}. {value}" for key, value in options.items()])
        subject_name = self.config.get("subject_name")
        subject_line = f"Subject: {subject_name}\n" if subject_name else ""
        task: str = f"{subject_line}Question: {task_value}\nOptions:\n{options_block}"
        # Cache the full question text so that external tools (e.g., web_search)
        # can use the entire question (not just a single entity) as the query.
        self.current_task = task
        return task, task

    def reset(self) -> None:
        # Keep current_task (set by set_env); it is needed for first Search.
        self.reward: float = 0
        self.selected: Optional[str] = None
        self.summary: Optional[str] = None
        self._search_count: int = 0

    def _try_extract_finish_from_thought(self, text: str) -> Optional[str]:
        """
        If a Thought contains a clear answer commitment (e.g. 'aligns with option A/B/C/D'),
        extract the option letter and treat it as implicit Finish[option].
        Returns the choice letter (A, B, C, or D) or None if no clear commitment found.

        Improvement: only allow inference when at least one Search/Lookup has been done
        (config allow_thought_finish_without_search=False).
        """
        if not text or "thought" not in text.lower():
            return None
        # Only allow Thought->Finish after at least one Search (avoid "I will search" being treated as Finish)
        if not self.config.get("allow_thought_finish_without_search", False):
            if self._search_count < 1:
                return None
        # Remove "Thought:" prefix for matching
        body = text.split(":", 1)[-1].strip() if ":" in text else text
        body_lower = body.lower()
        options = self.config.get("options", {})
        # Patterns that indicate answer commitment; prefer last match (final decision)
        patterns = [
            r"\boption\s*([A-Da-d])\b",
            r"\b(?:the\s+)?(?:correct\s+)?answer\s+is\s+([A-Da-d])\b",
            r"\baligns?\s+with\s+(?:option\s+)?([A-Da-d])\b",
            r"\bchoose\s+([A-Da-d])\b",
            r"\b(?:I\s+)?(?:will\s+)?(?:choose|select)\s+([A-Da-d])\b",
            r"\b([A-Da-d])\s+is\s+(?:the\s+)?(?:correct\s+)?(?:answer|option)\b",
            r"\b(?:so\s+)?(?:the\s+)?(?:correct\s+)?(?:option\s+)?(?:is\s+)?([A-Da-d])\b",
        ]
        candidates = []
        for pat in patterns:
            for m in re.finditer(pat, body_lower, re.IGNORECASE):
                letter = (m.group(1) or "").upper()
                if letter in options:
                    candidates.append((m.end(), letter))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]
        # Fallback: infer from option text in body when thought shows conclusion intent
        conclusion_markers = [
            "therefore", "thus", "so", "hence", "in conclusion", "conclude", "conclusion",
            "best option", "correct answer", "the answer", "should be", "is correct",
            "因此", "所以", "综上", "结论", "答案", "正确选项",
        ]
        if any(m in body_lower for m in conclusion_markers):
            inferred = self._infer_option_from_text(body, options)
            if inferred:
                return inferred
        return None

    def step(self, action: str) -> tuple[str, float, bool]:
        raw_action = action
        action = self.process_action(action)

        action_type: Optional[str] = None
        argument: Optional[str] = None

        if self._parse_action_type(action) == "thought":
            extracted = self._try_extract_finish_from_thought(raw_action)
            if extracted is not None:
                action_type, argument = "Finish", extracted
            else:
                return "OK.", 0, False

        if action_type is None:
            action_type, argument = self._parse_action(action)
        if action_type is None:
            action_type, argument = self._fuzzy_parse_action(raw_action)
        if action_type is None:
            action_type, argument = self._fuzzy_parse_action(action)
        if action_type is None:
            inferred = self._infer_option_from_text(raw_action, self.config.get("options", {}))
            if inferred:
                action_type, argument = "Finish", inferred

        if action_type == "Search":
            if self.config.get("disable_search_tools"):
                return "Search/Lookup tools are disabled for this run.", -1, False
            if not argument:
                return "Invalid Action. Search expects a topic.", -1, False

            # First Search: always use the full question (force search on the task).
            # Subsequent Search: use argument (agent's query).
            if self._search_count == 0:
                query_text = self.current_task if self.current_task else argument
            else:
                query_text = argument
            try:
                observation = openai_search_summary(query_text).strip("\n").strip()
            except Exception as e:
                print(f"[MedMCQAEnv] Search with OpenAI web_search failed: {e}")
                observation = "Web search failed or unavailable."

            self._search_count += 1
            self.summary = observation
            return observation, 0, False

        if action_type == "Lookup":
            if self.config.get("disable_search_tools"):
                return "Search/Lookup tools are disabled for this run.", -1, False
            if not argument:
                return "Invalid Action. Lookup expects a keyword.", -1, False
            # In OpenAI web_search mode we do not use Wikipedia Lookup. Return the
            # last Search (web_search) result so the agent uses that evidence directly.
            if self.summary:
                observation = self.summary
            else:
                observation = (
                    "No prior search result. Use the evidence from your last Search or answer with Finish."
                )
            return observation, 0, False

        if action_type == "Finish":
            if self.success_fn(argument):
                observation = f"{_ANSI_GREEN}Answer is CORRECT{_ANSI_RESET}"
                self.reward = 1
                return observation, 1, True
            observation = f"{_ANSI_RED}Answer is INCORRECT{_ANSI_RESET}"
            return observation, 0, True

        # When prompt restricts to Search+Finish only (e.g. force_search), keep message consistent
        if self.config.get("force_search_mode", False):
            observation = (
                "Invalid Action. Valid Actions are Search[<topic>] and Finish[<answer>]."
            )
        else:
            observation = (
                "Invalid Action. Valid Actions are Search[<topic>], Lookup[<keyword>], "
                "and Finish[<answer>]."
            )
        return observation, -1, False

    @staticmethod
    def _parse_action_type(action: str) -> Literal["action", "thought"]:
        return "thought" if "thought" in action.lower() else "action"

    @staticmethod
    def process_action(action: str) -> str:
        """Extract the single action to execute. Prefer first line that is Search/Lookup/Finish
        so that 'Thought: ...\\nSearch[query]' executes Search, not Thought."""
        raw = action.strip().replace("<", "").replace(">", "").replace("OK.", "").replace("OK", "")
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        # Prefer first line that parses as a full Action (Search/Lookup/Finish)
        for line in lines:
            # Strip "Action N:" prefix only; do not split on ":" (argument may contain colons)
            normalized = re.sub(r"^Action\s+\d+:\s*", "", line, flags=re.IGNORECASE).strip()
            atype, _ = MedMCQAEnv._parse_action(normalized)
            if atype in ("Search", "Lookup", "Finish"):
                return normalized
        # Try to extract Action spanning multiple lines (e.g. "Search[query with\nnewline]")
        for prefix in ("Search[", "Lookup[", "Finish["):
            idx = raw.find(prefix)
            if idx == -1:
                continue
            start = idx + len(prefix)
            depth = 1
            i = start
            while i < len(raw) and depth > 0:
                if raw[i] == "[":
                    depth += 1
                elif raw[i] == "]":
                    depth -= 1
                i += 1
            if depth == 0:
                candidate = raw[idx : i]
                atype, _ = MedMCQAEnv._parse_action(candidate)
                if atype:
                    return candidate
        # Fallback: first line (may be Thought)
        first = lines[0] if lines else ""
        if MedMCQAEnv._parse_action_type(first) == "thought":
            return first
        if ":" in first:
            first = first.split(":", 1)[1].strip()
        return first

    @staticmethod
    def _parse_action(string: str) -> tuple[Optional[str], Optional[str]]:
        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, string)
        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument
        return None, None

    @staticmethod
    def _extract_choice_letter(text: str) -> Optional[str]:
        cleaned = text.strip()
        if re.fullmatch(r"[A-Da-d]", cleaned):
            return cleaned.upper()
        if re.fullmatch(r"[1-4]", cleaned):
            return "ABCD"[int(cleaned) - 1]
        match = re.search(r"\b([A-Da-d])\b", cleaned)
        if match:
            return match.group(1).upper()
        return None

    def _infer_option_from_text(self, text: str, options: dict[str, str]) -> Optional[str]:
        normalized_text = normalize_answer(text)
        for key, option_text in options.items():
            if not option_text:
                continue
            if normalize_answer(option_text) in normalized_text:
                return key
        return None

    def _fuzzy_parse_action(self, text: str) -> tuple[Optional[str], Optional[str]]:
        lower = text.lower()
        if "finish" in lower or "answer" in lower:
            option = self._extract_choice_letter(text)
            if not option:
                option = self._infer_option_from_text(text, self.config.get("options", {}))
            if option:
                return "Finish", option
        return None, None

    def _append_source(self, observation: str) -> str:
        document = getattr(self.explorer, "document", None)
        if document is None:
            return observation
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("page")
        if source:
            return f"{observation}\nSource: {source}"
        return observation

    def success_fn(self, agent_ans: str) -> bool:
        if agent_ans is None:
            return False
        answer_key = self.config.get("answer_key")
        answer_text = self.config.get("answer")
        options = self.config.get("options", {})

        choice_letter = self._extract_choice_letter(str(agent_ans))
        if answer_key and choice_letter == answer_key:
            return True

        if answer_text and match_exactly(str(agent_ans), str(answer_text)):
            return True

        for key, text in options.items():
            if match_exactly(str(agent_ans), str(text)):
                return key == answer_key

        return False

    def feedback(self) -> tuple[float, bool, str]:
        feedback: str = (
            "You successfully finished this task."
            if self.reward == 1
            else "You failed the task."
        )
        done = self.reward == 1
        return self.reward, done, feedback


@dataclass
class MedMCQARecorder(BaseRecorder):
    def __post_init__(self):
        super().__post_init__()
        self.task = "medmcqa"
        self.counts = 0
        self.dones = 0
        self.rewards = 0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        message: str = f"---------- Task: {task_id} ----------"
        self.log(message)

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.dones += done
        self.counts += 1

        accuracy = self.rewards / self.counts
        done_rate = self.dones / self.counts
        message = (
            f"reward: {reward}, ave reward: {accuracy}.\n"
            f"done: {done}, ave done: {done_rate}"
        )
        self.log(message)
