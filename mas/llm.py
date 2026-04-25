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

    def _prepare_messages(self, messages: List[Message]) -> list[dict]:
        prepared = [{"role": msg.role, "content": msg.content} for msg in messages]
        if not self._is_qwen:
            return prepared

        # Qwen3 defaults to a thinking preamble on many OpenAI-compatible servers.
        # For ALFWorld we need a short executable command, not a long hidden trace.
        for msg in reversed(prepared):
            if msg["role"] == "user":
                content = str(msg.get("content") or "")
                if "/no_think" not in content:
                    msg["content"] = (
                        "/no_think\n"
                        "Output exactly one valid command and nothing else.\n"
                        f"{content}"
                    )
                break
        return prepared

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
        
        messages = self._prepare_messages(messages)
        request_kwargs = {}
        if self._is_qwen:
            request_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        max_retries = 5  
        wait_time = 1 

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
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
