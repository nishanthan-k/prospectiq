import asyncio
import time
import uuid
from typing import Callable, Optional, Union
from groq import (
    APIConnectionError,
    APITimeoutError,
    AsyncGroq,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.logger import log
from app.llm.prompts import SALES_ASSISTANT_SYSTEM_PROMPT

settings = get_settings()


class LLMRequest(BaseModel):
    """Unified LLM request payload with defaults and field validations."""

    prompt: str = Field(..., min_length=1, description="Prompt text for the model")
    system_prompt: Optional[str] = Field(
        default=SALES_ASSISTANT_SYSTEM_PROMPT,
        description="System prompt guiding assistant behavior",
    )
    model: str = Field(
        default_factory=lambda: settings.default_model,
        min_length=1,
        description="Model identifier",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0.0 to 1.0)",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum generation tokens (1 to 8192)",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability mass (0.0 to 1.0)",
    )
    request_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional correlation request ID",
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional per-request timeout in seconds (defaults to global setting)",
    )
    max_retries: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional per-request retry count after initial failure (defaults to global)",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt cannot be empty or only whitespace")
        return v.strip()


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


class LLMTimeoutError(Exception):
    """Raised when an LLM request exceeds its configured timeout."""

    def __init__(self, request_id: str, timeout_seconds: float):
        self.request_id = request_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"LLM request [{request_id}] timed out after {timeout_seconds}s")


RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    TimeoutError,
    RateLimitError,
    InternalServerError,
)


def _make_before_sleep_logger(
    request_id: str, max_retries: int
) -> Callable[[RetryCallState], None]:
    def _log_retry(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        wait_s = retry_state.next_action.sleep if retry_state.next_action else 0.0
        log.warning(
            f"[{request_id}] LLM attempt {retry_state.attempt_number}/{max_retries + 1} "
            f"failed ({exc}); retrying in {wait_s:.2f}s"
        )

    return _log_retry


def get_groq_client() -> AsyncGroq:
    """Return an AsyncGroq client instance."""
    return AsyncGroq(
        api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


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
    timeout_seconds = (
        req.timeout_seconds if req.timeout_seconds is not None else settings.llm_timeout_seconds
    )
    max_retries = req.max_retries if req.max_retries is not None else settings.llm_max_retries
    groq_client = client or get_groq_client()

    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    log.info(
        f"[{request_id}] LLM call started: model={req.model}, "
        f"temperature={req.temperature}, max_tokens={req.max_tokens}, "
        f"timeout={timeout_seconds}s, max_retries={max_retries}"
    )

    start_time = time.perf_counter()

    retryer = AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        before_sleep=_make_before_sleep_logger(request_id, max_retries),
        reraise=True,
    )

    try:
        async for attempt in retryer:
            with attempt:
                response = await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model=req.model,
                        messages=messages,  # type: ignore
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        top_p=req.top_p,
                        timeout=timeout_seconds,
                    ),
                    timeout=timeout_seconds,
                )
    except (APITimeoutError, TimeoutError) as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log.error(
            f"[{request_id}] LLM call timed out after {latency_ms:.0f}ms "
            f"(limit={timeout_seconds}s)"
        )
        raise LLMTimeoutError(request_id, timeout_seconds) from exc
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
