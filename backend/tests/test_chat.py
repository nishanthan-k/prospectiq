import json
import json
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.responses import StreamingResponse
from httpx import AsyncClient

from app.llm import LLMResponse, LLMTimeoutError


@pytest.mark.asyncio
async def test_chat_success(client: AsyncClient):
    mock_response = LLMResponse(
        request_id="req_test123",
        content="Stripe is a payments company.",
        finish_reason="stop",
        is_truncated=False,
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        latency_ms=120.5,
    )

    with patch("app.main.call_llm", new=AsyncMock(return_value=mock_response)):
        response = await client.post(
            "/api/chat",
            json={"prompt": "What does Stripe do?", "temperature": 0.5},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "req_test123"
    assert data["content"] == "Stripe is a payments company."
    assert data["finish_reason"] == "stop"
    assert data["is_truncated"] is False
    assert data["total_tokens"] == 70
    assert data["latency_ms"] == 120.5


@pytest.mark.asyncio
async def test_chat_blank_prompt_validation_error(client: AsyncClient):
    response = await client.post("/api/chat", json={"prompt": "   "})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("Prompt cannot be empty or only whitespace" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_chat_temperature_out_of_bounds(client: AsyncClient):
    response = await client.post("/api/chat", json={"prompt": "Hello", "temperature": 1.5})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("less than or equal to 1" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_chat_max_tokens_out_of_bounds(client: AsyncClient):
    response = await client.post("/api/chat", json={"prompt": "Hello", "max_tokens": 0})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("greater than or equal to 1" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_chat_timeout_seconds_out_of_bounds(client: AsyncClient):
    response = await client.post("/api/chat", json={"prompt": "Hello", "timeout_seconds": 0})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("greater than 0" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_chat_max_retries_out_of_bounds(client: AsyncClient):
    response = await client.post("/api/chat", json={"prompt": "Hello", "max_retries": -1})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("greater than or equal to 0" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_chat_timeout_returns_504(client: AsyncClient):
    with patch(
        "app.main.call_llm",
        new=AsyncMock(side_effect=LLMTimeoutError("req_timeout_504", 5.0)),
    ):
        response = await client.post(
            "/api/chat",
            json={"prompt": "Hello", "timeout_seconds": 5.0},
        )

    assert response.status_code == 504
    assert "req_timeout_504" in response.json()["detail"]
    assert "5.0s" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_success_streams_sse(client: AsyncClient):
    async def mock_stream():
        yield f"data: {json.dumps({'request_id': 'req_test123', 'delta': 'Stripe is '})}\n\n"
        yield f"data: {json.dumps({'request_id': 'req_test123', 'delta': 'a payments company.'})}\n\n"
        yield f"event: done\ndata: {json.dumps({'request_id': 'req_test123', 'finish_reason': 'stop'})}\n\n"

    mock_streaming_response = StreamingResponse(
        mock_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

    with patch("app.main.call_llm", new=AsyncMock(return_value=mock_streaming_response)):
        response = await client.post(
            "/api/chat",
            json={"prompt": "What does Stripe do?", "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert '"delta": "Stripe is "' in response.text
    assert '"delta": "a payments company."' in response.text
    assert "event: done" in response.text


@pytest.mark.asyncio
async def test_chat_streaming_timeout_returns_504(client: AsyncClient):
    with patch(
        "app.main.call_llm",
        new=AsyncMock(side_effect=LLMTimeoutError("req_stream_timeout", 5.0)),
    ):
        response = await client.post(
            "/api/chat",
            json={"prompt": "Hello", "stream": True, "timeout_seconds": 5.0},
        )

    assert response.status_code == 504
    assert "req_stream_timeout" in response.json()["detail"]
