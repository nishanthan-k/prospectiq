import time
import uuid
from typing import Optional, Union
from groq import AsyncGroq
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logger import log
from app.llm.prompts import SALES_ASSISTANT_SYSTEM_PROMPT

settings = get_settings()


class LLMRequest(BaseModel):
    """Unified LLM request payload with defaults."""

    prompt: str
    system_prompt: Optional[str] = SALES_ASSISTANT_SYSTEM_PROMPT
    model: str = Field(default_factory=lambda: settings.default_model)
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    request_id: Optional[str] = None


class LLMResponse(BaseModel):
    """Structured LLM response output with correlation ID and latency."""

    request_id: str
    content: str
    finish_reason: str
    is_truncated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


def get_groq_client() -> AsyncGroq:
    """Return an AsyncGroq client instance."""
    return AsyncGroq(api_key=settings.groq_api_key or "gsk_placeholder")


async def call_llm(
    request: Union[str, LLMRequest, dict],
    client: Optional[AsyncGroq] = None,
) -> LLMResponse:
    """
    Execute an LLM call using Groq with request correlation and latency tracking.

    Accepts:
    - Raw prompt string: call_llm("Analyze this company")
    - Dictionary: call_llm({"prompt": "...", "temperature": 0.2})
    - Pydantic model: call_llm(LLMRequest(prompt="..."))
    """
    if isinstance(request, str):
        req = LLMRequest(prompt=request)
    elif isinstance(request, dict):
        req = LLMRequest(**request)
    else:
        req = request

    request_id = req.request_id or f"req_{uuid.uuid4().hex[:8]}"
    groq_client = client or get_groq_client()

    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    log.info(
        f"[{request_id}] LLM call started: model={req.model}, "
        f"temperature={req.temperature}, max_tokens={req.max_tokens}"
    )

    start_time = time.perf_counter()

    try:
        response = await groq_client.chat.completions.create(
            model=req.model,
            messages=messages,  # type: ignore
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log.error(f"[{request_id}] LLM call failed after {latency_ms:.0f}ms: {exc}")
        raise

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    choice = response.choices[0]
    finish_reason = choice.finish_reason or "stop"
    is_truncated = finish_reason == "length"
    usage = response.usage

    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    total_tokens = usage.total_tokens if usage else 0

    log.info(
        f"[{request_id}] LLM call completed in {latency_ms:.0f}ms: "
        f"tokens={total_tokens} (prompt={prompt_tokens}, completion={completion_tokens}), "
        f"finish_reason={finish_reason}, is_truncated={is_truncated}"
    )

    return LLMResponse(
        request_id=request_id,
        content=choice.message.content or "",
        finish_reason=finish_reason,
        is_truncated=is_truncated,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=round(latency_ms, 2),
    )
