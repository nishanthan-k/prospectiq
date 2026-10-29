from unittest.mock import AsyncMock, MagicMock
import pytest

from app.llm.client import LLMRequest, call_llm


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
