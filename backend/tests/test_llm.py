import asyncio
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest
from groq import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError

from app.core.config import get_settings
from app.llm.client import LLMRequest, LLMTimeoutError, call_llm


def create_mock_completion(
    content: str = "Test response",
    finish_reason: str = "stop",
    prompt_tokens: int = 15,
    completion_tokens: int = 25,
):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.finish_reason = finish_reason

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


@pytest.mark.asyncio
async def test_call_llm_success():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = create_mock_completion(
        content="Snowflake is a data cloud platform."
    )

    req = LLMRequest(prompt="What is Snowflake?", temperature=0.3)
    response = await call_llm(req, client=mock_client)

    assert response.content == "Snowflake is a data cloud platform."
    assert response.finish_reason == "stop"
    assert response.is_truncated is False
    assert response.prompt_tokens == 15
    assert response.completion_tokens == 25
    assert response.total_tokens == 40
    assert response.latency_ms >= 0
    assert response.request_id.startswith("req_")


@pytest.mark.asyncio
async def test_call_llm_truncation_detected():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = create_mock_completion(
        content="Incomplete sentence...",
        finish_reason="length",
    )

    req = LLMRequest(prompt="Tell me everything", max_tokens=10)
    response = await call_llm(req, client=mock_client)

    assert response.finish_reason == "length"
    assert response.is_truncated is True


@pytest.mark.asyncio
async def test_call_llm_custom_request_id():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = create_mock_completion()

    req = LLMRequest(prompt="Hello", request_id="custom_req_007")
    response = await call_llm(req, client=mock_client)

    assert response.request_id == "custom_req_007"


@pytest.mark.asyncio
async def test_call_llm_exception_reraised():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("Groq network error")

    with pytest.raises(RuntimeError, match="Groq network error"):
        await call_llm(LLMRequest(prompt="Will fail"), client=mock_client)


@pytest.mark.asyncio
async def test_call_llm_uses_global_timeout_by_default():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = create_mock_completion()

    await call_llm(LLMRequest(prompt="Hello"), client=mock_client)

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["timeout"] == get_settings().llm_timeout_seconds


@pytest.mark.asyncio
async def test_call_llm_uses_per_request_timeout_override():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = create_mock_completion()

    await call_llm(LLMRequest(prompt="Hello", timeout_seconds=5.5), client=mock_client)

    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["timeout"] == 5.5


@pytest.mark.asyncio
async def test_call_llm_api_timeout_error_raises_llm_timeout_error():
    mock_client = AsyncMock()
    mock_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=mock_request)

    req = LLMRequest(
        prompt="Slow query",
        request_id="req_timeout_1",
        timeout_seconds=2.0,
        max_retries=0,
    )
    with pytest.raises(LLMTimeoutError, match=r"req_timeout_1.*2\.0s") as exc_info:
        await call_llm(req, client=mock_client)

    assert exc_info.value.request_id == "req_timeout_1"
    assert exc_info.value.timeout_seconds == 2.0


@pytest.mark.asyncio
async def test_call_llm_slow_call_enforces_timeout():
    async def slow_create(**kwargs):
        await asyncio.sleep(0.2)
        return create_mock_completion()

    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = slow_create

    req = LLMRequest(prompt="Hang", request_id="req_slow", timeout_seconds=0.01, max_retries=0)
    with pytest.raises(LLMTimeoutError, match=r"req_slow.*0\.01s"):
        await call_llm(req, client=mock_client)


@pytest.mark.asyncio
async def test_call_llm_retries_with_global_default_and_succeeds(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "llm_retry_max_wait", 0.0)

    mock_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_response = httpx.Response(429, request=mock_request)
    rate_limit_err = RateLimitError("Rate limited", response=mock_response, body=None)

    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = [
        rate_limit_err,
        APIConnectionError(request=mock_request),
        APITimeoutError(request=mock_request),
        create_mock_completion(content="Recovered on 4th attempt"),
    ]

    response = await call_llm(LLMRequest(prompt="Hello"), client=mock_client)

    assert response.content == "Recovered on 4th attempt"
    assert mock_client.chat.completions.create.call_count == 4


@pytest.mark.asyncio
async def test_call_llm_per_request_max_retries_override(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "llm_retry_max_wait", 0.0)

    mock_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=mock_request)

    with pytest.raises(APIConnectionError):
        await call_llm(LLMRequest(prompt="Hello", max_retries=1), client=mock_client)

    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_call_llm_retries_exhausted_on_timeout_raises_llm_timeout_error(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_retry_min_wait", 0.0)
    monkeypatch.setattr(settings, "llm_retry_max_wait", 0.0)

    mock_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(request=mock_request)

    req = LLMRequest(
        prompt="Timeout all",
        request_id="req_t_retry",
        timeout_seconds=1.5,
        max_retries=2,
    )
    with pytest.raises(LLMTimeoutError, match=r"req_t_retry.*1\.5s"):
        await call_llm(req, client=mock_client)

    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_call_llm_non_retryable_error_fails_immediately():
    mock_request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_response = httpx.Response(400, request=mock_request)
    bad_req_err = BadRequestError("Invalid model", response=mock_response, body=None)

    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = bad_req_err

    with pytest.raises(BadRequestError):
        await call_llm(LLMRequest(prompt="Hello", max_retries=3), client=mock_client)

    assert mock_client.chat.completions.create.call_count == 1
