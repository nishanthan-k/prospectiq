import asyncio
import inspect
import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional, Union

from fastapi.responses import StreamingResponse
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, Field, field_validator, model_validator
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.json_utils import _safe_parse_json
from app.core.logger import log
from app.llm.prompts import SALES_ASSISTANT_SYSTEM_PROMPT

settings = get_settings()


class ToolCallPayload(BaseModel):
    """Structured tool invocation requested by the LLM."""

    id: str = Field(..., description="Unique tool call invocation ID")
    name: str = Field(..., description="Target tool name to execute")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed JSON arguments for the tool",
    )
    raw_arguments: Optional[str] = Field(
        default=None,
        description="Raw arguments payload string as received from model",
    )
    parse_error: Optional[str] = Field(
        default=None,
        description="Parsing error message if arguments were malformed JSON",
    )
    extra_content: Optional[dict[str, Any]] = Field(
        default=None,
        description="Provider-specific extra content (e.g. Google Gemini thought_signature)",
    )


class LLMRequest(BaseModel):
    """Unified LLM request payload with defaults and field validations."""

    prompt: Optional[str] = Field(default=None, description="Prompt text for the model")
    messages: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional multi-turn conversation messages list",
    )
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
    response_format: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional response format specification, e.g. {'type': 'json_object'}",
    )
    tools: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional list of tool definitions in OpenAI/Gemini function-calling format",
    )
    tool_choice: Optional[Union[str, dict[str, Any]]] = Field(
        default=None,
        description="Optional tool choice directive ('auto', 'none', 'required', or specific tool object)",
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not v.strip():
                raise ValueError("Prompt cannot be empty or only whitespace")
            return v.strip()
        return v

    @model_validator(mode="after")
    def validate_prompt_or_messages(self) -> "LLMRequest":
        if self.messages is None and self.prompt is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        if self.messages is not None and len(self.messages) == 0:
            raise ValueError("'messages' cannot be empty when provided")
        return self


class LLMResponse(BaseModel):
    """Structured LLM response output with correlation ID and latency."""

    request_id: str
    content: Optional[str] = ""
    finish_reason: str = "stop"
    is_truncated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls: list[ToolCallPayload] = Field(
        default_factory=list,
        description="Structured tool calls requested by the model",
    )
    raw_message: Optional[dict[str, Any]] = Field(
        default=None,
        description="Raw assistant message dict for exact multi-turn replay",
    )


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


def _extract_rate_limit_wait(exc: BaseException) -> Optional[float]:
    """
    Extract rate limit wait time in seconds from a RateLimitError.
    First tries reading the 'retry-after' header from exc.response.
    Falls back to regex parsing 'try again in <seconds>s' from str(exc).
    Applies a +0.5s buffer and caps at 60.0s.
    """
    if not isinstance(exc, RateLimitError):
        return None

    wait_s: Optional[float] = None

    # 1. Try reading 'retry-after' header from response if available
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait_s = float(retry_after)
                except (ValueError, TypeError):
                    pass

    # 2. Fallback: Parse 'try again in <seconds>s' from exception message
    if wait_s is None:
        match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
        if match:
            try:
                wait_s = float(match.group(1))
            except (ValueError, TypeError):
                pass

    if wait_s is not None and wait_s >= 0:
        return min(wait_s + 0.5, 60.0)

    return None


def _smart_wait(retry_state: RetryCallState) -> float:
    """
    Rate-limit-aware Tenacity wait strategy.
    Returns the extracted rate-limit wait time on RateLimitError,
    otherwise falls back to exponential backoff.
    """
    if retry_state.outcome and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        if exc is not None:
            rl_wait = _extract_rate_limit_wait(exc)
            if rl_wait is not None:
                return rl_wait

    fallback = wait_exponential(
        min=settings.llm_retry_min_wait,
        max=settings.llm_retry_max_wait,
    )
    return fallback(retry_state)


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


_llm_client: Optional[AsyncOpenAI] = None


def get_gemini_client() -> AsyncOpenAI:
    """Return the singleton AsyncOpenAI client instance configured for Google Gemini."""
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            api_key=settings.gemini_api_key or "missing_gemini_api_key",
            base_url=settings.gemini_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
    return _llm_client


async def close_gemini_client() -> None:
    """Close the singleton AsyncOpenAI client and reset it."""
    global _llm_client
    if _llm_client is not None:
        await _llm_client.close()
        _llm_client = None


# Generic aliases
get_llm_client = get_gemini_client
close_llm_client = close_gemini_client


def _prepare_request(
    params: Union[LLMRequest, str, dict],
    client: Optional[AsyncOpenAI],
) -> tuple[LLMRequest, str, float, int, AsyncOpenAI, list[dict[str, Any]]]:
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
    llm_client = client or get_gemini_client()

    if req.messages is not None:
        messages: list[dict[str, Any]] = list(req.messages)
    else:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        if req.prompt:
            messages.append({"role": "user", "content": req.prompt})

    return req, request_id, timeout_seconds, max_retries, llm_client, messages


async def _execute_create(
    llm_client: AsyncOpenAI,
    req: LLMRequest,
    messages: list[dict[str, Any]],
    timeout_seconds: float,
    max_retries: int,
    request_id: str,
    start_time: float,
    stream: bool = False,
) -> Any:
    retryer = AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=_smart_wait,
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        before_sleep=_make_before_sleep_logger(request_id, max_retries),
        reraise=True,
    )

    try:
        async for attempt in retryer:
            with attempt:
                kwargs: dict[str, Any] = {
                    "model": req.model,
                    "messages": messages,  # type: ignore
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "top_p": req.top_p,
                    "stream": stream,
                    "timeout": timeout_seconds,
                }
                if req.response_format:
                    kwargs["response_format"] = req.response_format
                if req.tools:
                    kwargs["tools"] = req.tools
                if req.tool_choice:
                    kwargs["tool_choice"] = req.tool_choice

                return await asyncio.wait_for(
                    llm_client.chat.completions.create(**kwargs),
                    timeout=timeout_seconds,
                )
    except (APITimeoutError, TimeoutError) as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log.error(
            f"[{request_id}] LLM call timed out after {latency_ms:.0f}ms (limit={timeout_seconds}s)"
        )
        raise LLMTimeoutError(request_id, timeout_seconds) from exc
    except Exception as exc:
        err_str = str(exc).lower()
        if (
            ("413" in err_str or "rate_limit" in err_str or "too large" in err_str)
            and req.model == settings.default_model
            and settings.fast_model != settings.default_model
        ):
            log.warning(
                f"[{request_id}] Model {req.model} exceeded token limit ({exc}). "
                f"Automatically falling back to fast model '{settings.fast_model}'"
            )
            req.model = settings.fast_model
            return await _execute_create(
                llm_client=llm_client,
                req=req,
                messages=messages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                request_id=request_id,
                start_time=start_time,
                stream=stream,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log.error(f"[{request_id}] LLM call failed after {latency_ms:.0f}ms: {exc}")
        raise


async def call_llm(
    params: Union[LLMRequest, str, dict],
    client: Optional[AsyncOpenAI] = None,
) -> Union[LLMResponse, StreamingResponse]:
    """
    Execute an LLM call using Google Gemini with request correlation and latency tracking.

    When params.stream is True, returns a StreamingResponse with Server-Sent Events.
    When params.stream is False (default), returns an LLMResponse.
    """
    req, request_id, timeout_seconds, max_retries, llm_client, messages = _prepare_request(
        params, client
    )

    log.info(
        f"[{request_id}] LLM call started: model={req.model}, "
        f"temperature={req.temperature}, max_tokens={req.max_tokens}, "
        f"timeout={timeout_seconds}s, max_retries={max_retries}, stream={req.stream}"
    )

    start_time = time.perf_counter()
    response = await _execute_create(
        llm_client,
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

                        chunk_usage = getattr(chunk, "usage", None)
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

    # Extract and parse tool calls safely
    parsed_tool_calls: list[ToolCallPayload] = []
    raw_tool_calls = getattr(choice.message, "tool_calls", None)
    if raw_tool_calls:
        for rtc in raw_tool_calls:
            tc_id = getattr(rtc, "id", "") or str(uuid.uuid4())
            fn = getattr(rtc, "function", None)
            fn_name = getattr(fn, "name", "") if fn else ""
            raw_args = getattr(fn, "arguments", "{}") if fn else "{}"
            extra_content = getattr(rtc, "extra_content", None)
            if not isinstance(extra_content, dict):
                extra_content = None

            raw_args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
            parse_error: Optional[str] = None
            if isinstance(raw_args, str):
                trimmed = raw_args.strip()
                if not trimmed:
                    parse_error = "Malformed JSON arguments: empty or whitespace-only string"
                    parsed_args = {}
                else:
                    parsed_args = _safe_parse_json(trimmed)
                    if parsed_args is None or not isinstance(parsed_args, dict):
                        parse_error = f"Malformed JSON arguments: {trimmed[:100]}"
                        parsed_args = {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parse_error = f"Invalid arguments format of type: {type(raw_args).__name__}"
                parsed_args = {}

            parsed_tool_calls.append(
                ToolCallPayload(
                    id=tc_id,
                    name=fn_name,
                    arguments=parsed_args,
                    raw_arguments=raw_args_str,
                    parse_error=parse_error,
                    extra_content=extra_content,
                )
            )

    raw_message: Optional[dict[str, Any]] = None
    if hasattr(choice, "message"):
        model_dump_fn = getattr(choice.message, "model_dump", None)
        if callable(model_dump_fn):
            try:
                dumped = model_dump_fn(exclude_unset=True)
                if isinstance(dumped, dict):
                    raw_message = dumped
            except Exception:
                raw_message = None

    log.info(
        f"[{request_id}] LLM call completed in {latency_ms:.0f}ms: "
        f"tokens={total_tokens} (prompt={prompt_tokens}, completion={completion_tokens}), "
        f"finish_reason={finish_reason}, is_truncated={is_truncated}, "
        f"tool_calls={len(parsed_tool_calls)}"
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
        tool_calls=parsed_tool_calls,
        raw_message=raw_message,
    )
