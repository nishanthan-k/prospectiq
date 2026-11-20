import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.llm import LLMResponse, LLMTimeoutError, ToolCallPayload
from app.research.models import ExtractedPage, SearchResult


@pytest.mark.asyncio
async def test_agent_api_blank_goal_validation_error(client: AsyncClient):
    """Empty or whitespace-only goal returns 422 Unprocessable Entity."""
    response = await client.post("/api/agent/run", json={"goal": "     "})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("Goal cannot be empty" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_agent_api_goal_too_short_validation_error(client: AsyncClient):
    """Goal shorter than 5 characters returns 422 validation error."""
    response = await client.post("/api/agent/run", json={"goal": "help"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_agent_api_max_steps_bounds_validation_error(client: AsyncClient):
    """max_steps outside [1, 10] returns 422 validation error."""
    resp_low = await client.post(
        "/api/agent/run",
        json={"goal": "Investigate target company", "max_steps": 0},
    )
    assert resp_low.status_code == 422

    resp_high = await client.post(
        "/api/agent/run",
        json={"goal": "Investigate target company", "max_steps": 11},
    )
    assert resp_high.status_code == 422


@pytest.mark.asyncio
async def test_agent_api_success_end_to_end_mocked_llm(client: AsyncClient):
    """End-to-end multi-step agent execution via API with mocked LLM and tools."""
    # Step 1: LLM issues a search tool call
    step1_resp = LLMResponse(
        request_id="res_1",
        content="",
        tool_calls=[
            ToolCallPayload(
                id="call_search_1",
                name="search_company",
                arguments={"query": "Retool internal tools"},
            )
        ],
        finish_reason="tool_calls",
        prompt_tokens=60,
        completion_tokens=25,
        total_tokens=85,
    )

    # Step 2: LLM synthesizes final answer
    step2_resp = LLMResponse(
        request_id="res_2",
        content="Retool is a low-code development platform used by developers to build internal tools.",
        finish_reason="stop",
        prompt_tokens=140,
        completion_tokens=30,
        total_tokens=170,
    )

    mock_search_results = [
        SearchResult(
            title="Retool | Build internal tools remarkably fast",
            url="https://retool.com",
            snippet="Build internal applications, dashboards, and workflows fast.",
            source="bing",
        )
    ]

    with (
        patch("app.agent.runner.call_llm", side_effect=[step1_resp, step2_resp]),
        patch(
            "app.agent.tools.definitions.search_company",
            new=AsyncMock(return_value=mock_search_results),
        ),
    ):
        response = await client.post(
            "/api/agent/run",
            json={
                "goal": "Explain what Retool does and who uses it",
                "company_name": "Retool",
                "domain": "retool.com",
                "max_steps": 4,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"].startswith("agt_")
    assert data["goal"] == "Explain what Retool does and who uses it"
    assert "Retool is a low-code development platform" in data["final_answer"]
    assert data["steps_taken"] == 2
    assert data["is_completed"] is True
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool_name"] == "search_company"
    assert data["tool_calls"][0]["arguments"] == {"query": "Retool internal tools"}
    assert data["tool_calls"][0]["is_error"] is False
    assert data["total_tokens"] == 255
    assert data["latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_agent_api_citations_extraction(client: AsyncClient):
    """Agent extracts and verifies citations when tool fetched content matches cited quote."""
    # Step 1: LLM calls fetch_webpage
    step1_resp = LLMResponse(
        request_id="res_1",
        content="",
        tool_calls=[
            ToolCallPayload(
                id="call_fetch_1",
                name="fetch_webpage",
                arguments={"url": "https://stripe.com"},
            )
        ],
        finish_reason="tool_calls",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
    )

    # Step 2: LLM responds with structured JSON citations in final answer
    cited_json = json.dumps(
        {
            "summary": "Stripe is a financial infrastructure platform for the internet.",
            "citations": [
                {
                    "claim": "Stripe provides financial infrastructure.",
                    "quote": "Millions of companies of all sizes use Stripe online to accept payments and manage their businesses.",
                    "source_url": "https://stripe.com",
                    "relevance_score": 0.95,
                }
            ],
        }
    )

    step2_resp = LLMResponse(
        request_id="res_2",
        content=cited_json,
        finish_reason="stop",
        prompt_tokens=200,
        completion_tokens=60,
        total_tokens=260,
    )

    mock_page = ExtractedPage(
        url="https://stripe.com",
        title="Stripe | Financial Infrastructure",
        content="Millions of companies of all sizes use Stripe online to accept payments and manage their businesses.",
        chunks=[
            "Millions of companies of all sizes use Stripe online to accept payments and manage their businesses."
        ],
    )

    with (
        patch("app.agent.runner.call_llm", side_effect=[step1_resp, step2_resp]),
        patch("app.agent.tools.definitions.extract_page", new=AsyncMock(return_value=mock_page)),
    ):
        response = await client.post(
            "/api/agent/run",
            json={
                "goal": "Verify Stripe's payment infrastructure",
                "domain": "stripe.com",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source_url"] == "https://stripe.com"
    assert "use Stripe online to accept payments" in data["citations"][0]["quote"]


@pytest.mark.asyncio
async def test_agent_api_llm_timeout_returns_504(client: AsyncClient):
    """Agent API forwards LLM timeout to 504 Gateway Timeout response."""
    with patch(
        "app.agent.runner.call_llm",
        side_effect=LLMTimeoutError("Groq API timed out after 30.0s", timeout_seconds=30.0),
    ):
        response = await client.post(
            "/api/agent/run",
            json={"goal": "Investigate target company"},
        )

    assert response.status_code == 504
    data = response.json()
    assert "timed out" in data["detail"]


@pytest.mark.asyncio
async def test_agent_api_allowed_tools_filtering(client: AsyncClient):
    """Agent filters available tools when allowed_tools list is specified in request."""
    mock_resp = LLMResponse(
        request_id="res_direct",
        content="Direct synthesized answer without tools.",
        finish_reason="stop",
        prompt_tokens=40,
        completion_tokens=10,
        total_tokens=50,
    )

    captured_requests = []

    async def capture_call_llm(req, **kwargs):
        captured_requests.append(req)
        return mock_resp

    with patch("app.agent.runner.call_llm", side_effect=capture_call_llm):
        response = await client.post(
            "/api/agent/run",
            json={
                "goal": "Assess target fit with single tool",
                "allowed_tools": ["search_company"],
            },
        )

    assert response.status_code == 200
    assert len(captured_requests) == 1
    req_sent = captured_requests[0]
    # Tools passed to LLM must only contain search_company
    tool_names = [t["function"]["name"] for t in req_sent.tools]
    assert tool_names == ["search_company"]
