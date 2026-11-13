import asyncio
import inspect
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional, Union
from fastapi.responses import StreamingResponse
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
    stream: bool = Field(
        default=False,
        description="Whether to stream the response as Server-Sent Events",
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


def _prepare_request(
    params: Union[LLMRequest, str, dict],
    client: Optional[AsyncGroq],
) -> tuple[LLMRequest, str, float, int, AsyncGroq, list[dict[str, str]]]:
    if isinstance(params, str):
        req = LLMRequest(prompt=params)
    elif isinstance(params, dict):
        req = LLMRequest(**params)
    else:
        req = params

    request_id = req.request_id or f"req_{uuid.uuid4().hex[:8]}"
    timeout_seconds = (
        req.timeout_seconds if req.timeout_seconds is not None else settings.llm_timeout_seconds
    )
    max_retries = req.max_retries if req.max_retries is not None else settings.llm_max_retries
    groq_client = client or get_groq_client()

    messages: list[dict[str, str]] = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    return req, request_id, timeout_seconds, max_retries, groq_client, messages


async def _execute_create(
    groq_client: AsyncGroq,
    req: LLMRequest,
    messages: list[dict[str, str]],
    timeout_seconds: float,
    max_retries: int,
    request_id: str,
    start_time: float,
    stream: bool = False,
) -> Any:
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
                return await asyncio.wait_for(
                    groq_client.chat.completions.create(
                        model=req.model,
                        messages=messages,  # type: ignore
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        top_p=req.top_p,
                        stream=stream,
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


async def call_llm(
    params: Union[LLMRequest, str, dict],
    client: Optional[AsyncGroq] = None,
) -> Union[LLMResponse, StreamingResponse]:
    """
    Execute an LLM call using Groq with request correlation and latency tracking.

    When params.stream is True, returns a StreamingResponse with Server-Sent Events.
    When params.stream is False (default), returns an LLMResponse.
    """
    req, request_id, timeout_seconds, max_retries, groq_client, messages = _prepare_request(
        params, client
    )

    log.info(
        f"[{request_id}] LLM call started: model={req.model}, "
        f"temperature={req.temperature}, max_tokens={req.max_tokens}, "
        f"timeout={timeout_seconds}s, max_retries={max_retries}, stream={req.stream}"
    )

    start_time = time.perf_counter()
    response = await _execute_create(
        groq_client,
        req,
        messages,
        timeout_seconds,
        max_retries,
        request_id,
        start_time,
        stream=req.stream,
    )

    if req.stream:
        async def _event_stream() -> AsyncIterator[str]:
            finish_reason = "stop"
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            try:
                async with asyncio.timeout(timeout_seconds):
                    async for chunk in response:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            if choice.finish_reason:
                                finish_reason = choice.finish_reason
                            if choice.delta and choice.delta.content:
                                yield f"data: {json.dumps({'request_id': request_id, 'delta': choice.delta.content})}\n\n"

                        x_groq = getattr(chunk, "x_groq", None)
                        chunk_usage = getattr(chunk, "usage", None) or (
                            x_groq.usage if x_groq is not None else None
                        )
                        if chunk_usage:
                            prompt_tokens = chunk_usage.prompt_tokens
                            completion_tokens = chunk_usage.completion_tokens
                            total_tokens = chunk_usage.total_tokens

                latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                is_truncated = finish_reason == "length"
                log.info(
                    f"[{request_id}] LLM stream completed in {latency_ms:.0f}ms: "
                    f"tokens={total_tokens} (prompt={prompt_tokens}, completion={completion_tokens}), "
                    f"finish_reason={finish_reason}, is_truncated={is_truncated}"
                )
                done_payload = {
                    "request_id": request_id,
                    "finish_reason": finish_reason,
                    "is_truncated": is_truncated,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": latency_ms,
                }
                yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
            except (APITimeoutError, TimeoutError):
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                log.error(f"[{request_id}] LLM stream timed out after {latency_ms:.0f}ms")
                err_payload = {
                    "request_id": request_id,
                    "detail": f"LLM request [{request_id}] timed out after {timeout_seconds}s",
                }
                yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                log.error(f"[{request_id}] LLM stream failed after {latency_ms:.0f}ms: {exc}")
                yield f"event: error\ndata: {json.dumps({'request_id': request_id, 'detail': str(exc)})}\n\n"
            finally:
                close_fn = getattr(response, "close", None)
                if callable(close_fn):
                    res = close_fn()
                    if inspect.isawaitable(res):
                        await res

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

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
