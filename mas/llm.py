import os

from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .utils import load_config


# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

URL = os.environ["OPENAI_API_BASE"]
KEY = os.environ["OPENAI_API_KEY"]
print('# api url: ', URL)


completion_tokens, prompt_tokens = 0, 0

@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY
        )
        self._is_qwen = "qwen" in (model_name or "").lower()
        self._is_gpt4omini = "gpt-4o-mini" in (model_name or "").lower()

    @staticmethod
    def _is_action_only_env_prompt(content: str) -> bool:
        text = str(content or "")
        lowered = text.lower()
        markers = (
            "admissible commands",
            "check valid actions",
            "the goal is to satisfy the following conditions",
            "what action would you like to do next",
            "valid commands",
            "valid actions from the scienceworld engine",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _looks_like_thought_answer(answer: str) -> bool:
        lowered = str(answer or "").strip().lower()
        return lowered.startswith("think:") or lowered.startswith("thought:") or lowered.startswith("i need to ")

    @staticmethod
    def _looks_like_prompt_echo(answer: str) -> bool:
        text = str(answer or "").strip()
        lowered = text.lower()
        markers = (
            "## your turn: take action!",
            "here is your initial observation:",
            "**here is your task:",
            "do not copy object names",
            "if an action fails or stalls",
            "use the above examples and insights as a foundation",
        )
        if any(marker in lowered for marker in markers):
            return True
        if "\n" in text and text.count("\n") >= 3:
            return True
        if text.startswith(">") and len(text) < 8:
            return True
        return False

    @staticmethod
    def _sanitize_action_only_answer(answer: str) -> str:
        text = str(answer or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text
        first = lines[0].lstrip(">").strip()
        if len(lines) == 1:
            return first
        lowered = first.lower()
        if lowered.startswith(("think:", "thought:")):
            return first
        if any(
            marker in lowered
            for marker in (
                "## your turn",
                "here is your initial observation",
                "here is your task",
                "do not copy object names",
                "use the above examples",
            )
        ):
            return text
        return first

    @staticmethod
    def _safe_action_only_fallback(prompt: str) -> str:
        lowered = str(prompt or "").lower()
        if "check valid actions" in lowered:
            return "check valid actions"
        return ""

    def _inject_action_only_guard(self, content: str, *, strict_retry: bool = False) -> str:
        content = str(content or "")
        if self._is_qwen:
            if "/no_think" not in content:
                return "/no_think\nOutput exactly one valid command and nothing else.\n" + content
            return content

        if self._is_gpt4omini:
            prefix = (
                "Return exactly one admissible command from the current command list.\n"
                "Do not output think:, reasoning, explanation, or multiple lines.\n"
            )
            if strict_retry:
                prefix = (
                    "Return exactly one admissible command from the current admissible command list.\n"
                    "Do not output think:, Thought:, analysis, explanation, Markdown, or any extra text.\n"
                    "Your entire answer must be the command only.\n"
                )
            if prefix not in content:
                return prefix + content
        return content

    def _prepare_messages(self, messages: List[Message], *, strict_retry: bool = False) -> tuple[list[dict], bool, str]:
        prepared = [{"role": msg.role, "content": msg.content} for msg in messages]
        for msg in reversed(prepared):
            if msg["role"] == "user":
                content = str(msg.get("content") or "")
                is_action_only = self._is_action_only_env_prompt(content)
                if is_action_only:
                    msg["content"] = self._inject_action_only_guard(content, strict_retry=strict_retry)
                action_prompt = content
                break
        else:
            is_action_only = False
            action_prompt = ""
        return prepared, is_action_only, action_prompt

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS
    ) -> str:
        import time
        global prompt_tokens, completion_tokens

        request_kwargs = {}
        if self._is_qwen:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        max_retries = 5  
        wait_time = 1 

        for attempt in range(max_retries):
            try:
                prepared_messages, is_action_only, action_prompt = self._prepare_messages(
                    messages,
                    strict_retry=(attempt > 0),
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=prepared_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=num_comps,
                    stop=stop_strs,
                    **request_kwargs,
                )

                answer = response.choices[0].message.content
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                
                if answer is None:
                    # Treat None as a hard error; upstream loops can otherwise hang.
                    print("Error: LLM returned None")
                    continue
                answer = answer.strip()
                if not answer:
                    # Avoid silent empty outputs causing infinite retries in downstream parsers.
                    raise RuntimeError("Empty LLM response")
                if is_action_only and self._is_gpt4omini:
                    answer = self._sanitize_action_only_answer(answer)
                if is_action_only and self._is_gpt4omini and self._looks_like_thought_answer(answer):
                    fallback = self._safe_action_only_fallback(action_prompt)
                    return fallback or answer
                if is_action_only and self._is_gpt4omini and self._looks_like_prompt_echo(answer):
                    fallback = self._safe_action_only_fallback(action_prompt)
                    return fallback or answer
                return answer  

            except Exception as e:
                # Convert error to string as safely as possible (avoid nested Unicode errors)
                try:
                    error_message = str(e)
                except Exception:
                    error_message = ""

                # For rate limit or 429 errors, back off and retry
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                    continue

                if request_kwargs and any(
                    marker in error_message
                    for marker in ("chat_template_kwargs", "enable_thinking", "extra_body")
                ):
                    request_kwargs = {}
                    time.sleep(wait_time)
                    continue

                # For all other errors, print a short preview and retry a few times,
                # then raise to prevent downstream infinite loops.
                preview = (error_message or repr(e))[:500]
                print(f"[GPTChat] error: {type(e).__name__}: {preview}")
                time.sleep(wait_time)
                continue

        raise RuntimeError("GPTChat failed after retries")


def get_price():
    global completion_tokens, prompt_tokens
    return completion_tokens, prompt_tokens, completion_tokens*60/1000000+prompt_tokens*30/1000000
