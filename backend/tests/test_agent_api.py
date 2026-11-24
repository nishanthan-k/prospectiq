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
        side_effect=LLMTimeoutError("Gemini API timed out after 30.0s", timeout_seconds=30.0),
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


# --- Investigate API Tests ---


@pytest.mark.asyncio
async def test_agent_investigate_api_validation(client: AsyncClient):
    """Empty company_name returns 422 Unprocessable Entity."""
    response = await client.post("/api/agent/investigate", json={"company_name": "   "})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("Company name cannot be empty" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_agent_investigate_api_success(client: AsyncClient):
    """Investigate endpoint conducts search and returns structured company profile."""
    step1_resp = LLMResponse(
        request_id="res_inv1",
        content="",
        tool_calls=[
            ToolCallPayload(
                id="call_search_inv",
                name="search_company",
                arguments={"query": "Retool low code developer internal tools"},
            )
        ],
        finish_reason="tool_calls",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
    )

    investigate_json = json.dumps(
        {
            "summary": "Retool is a low-code enterprise development platform for building internal software.",
            "primary_offerings": ["Retool Apps", "Retool Workflows", "Retool Database"],
            "business_model": "B2B SaaS Subscription",
            "target_audience": "Software engineering teams and operations leaders",
        }
    )

    step2_resp = LLMResponse(
        request_id="res_inv2",
        content=investigate_json,
        finish_reason="stop",
        prompt_tokens=150,
        completion_tokens=50,
        total_tokens=200,
    )

    mock_search = [
        SearchResult(
            title="Retool | Low Code Internal Tools",
            url="https://retool.com",
            snippet="Build apps and workflows connected to any database or API.",
            source="bing",
        )
    ]

    with (
        patch("app.agent.runner.call_llm", side_effect=[step1_resp, step2_resp]),
        patch(
            "app.agent.tools.definitions.search_company", new=AsyncMock(return_value=mock_search)
        ),
    ):
        response = await client.post(
            "/api/agent/investigate",
            json={
                "company_name": "Retool",
                "domain": "retool.com",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"].startswith("inv_")
    assert data["company_name"] == "Retool"
    assert data["domain"] == "retool.com"
    assert "low-code enterprise development" in data["summary"]
    assert "Retool Apps" in data["primary_offerings"]
    assert "B2B SaaS" in data["business_model"]
    assert "engineering" in data["target_audience"].lower()
    assert data["steps_taken"] == 2
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool_name"] == "search_company"


# --- Qualify API Tests ---


@pytest.mark.asyncio
async def test_agent_qualify_api_validation(client: AsyncClient):
    """Empty company_name returns 422 Unprocessable Entity."""
    response = await client.post("/api/agent/qualify", json={"company_name": "  "})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("Company name cannot be empty" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_agent_qualify_api_success(client: AsyncClient):
    """Qualify endpoint evaluates company against ICP criteria and returns business case."""
    # Step 1: LLM executes evaluate_lead_fit with company summary
    step1_resp = LLMResponse(
        request_id="res_qlf1",
        content="",
        tool_calls=[
            ToolCallPayload(
                id="call_lead_fit",
                name="evaluate_lead_fit",
                arguments={
                    "company_name": "Datadog",
                    "offerings_summary": "Cloud monitoring, observability platform for enterprise B2B SaaS architecture, developer APIs, and security compliance.",
                },
            )
        ],
        finish_reason="tool_calls",
        prompt_tokens=80,
        completion_tokens=30,
        total_tokens=110,
    )

    # Step 2: Final response
    step2_resp = LLMResponse(
        request_id="res_qlf2",
        content="Datadog strongly qualifies for our platform based on its B2B SaaS cloud architecture and enterprise compliance profile.",
        finish_reason="stop",
        prompt_tokens=150,
        completion_tokens=40,
        total_tokens=190,
    )

    with patch("app.agent.runner.call_llm", side_effect=[step1_resp, step2_resp]):
        response = await client.post(
            "/api/agent/qualify",
            json={
                "company_name": "Datadog",
                "domain": "datadoghq.com",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"].startswith("qlf_")
    assert data["company_name"] == "Datadog"
    assert data["domain"] == "datadoghq.com"

    qual = data["qualification"]
    assert qual["is_fit"] is True
    assert qual["fit_score"] >= 0.5
    assert qual["status"] in ["QUALIFIED", "PARTIAL_FIT"]
    assert len(qual["why_fit"]) > 0
    assert len(qual["matched_criteria"]) > 0
    assert len(qual["proposal_hooks"]) > 0


# --- Unified Agent Run API Tests ---


@pytest.mark.asyncio
async def test_agent_run_with_company_name_only(client: AsyncClient):
    """Calling /api/agent/run with only company_name auto-generates default qualification goal."""
    mock_resp = LLMResponse(
        request_id="res_auto_goal",
        content="Supabase is an open-source Firebase alternative providing Postgres backend services.",
        finish_reason="stop",
        prompt_tokens=40,
        completion_tokens=20,
        total_tokens=60,
    )

    with patch("app.agent.runner.call_llm", return_value=mock_resp):
        response = await client.post(
            "/api/agent/run",
            json={"company_name": "Supabase"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["goal"] == "Investigate Supabase and qualify fit for platform outreach"
    assert "Supabase is an open-source" in data["final_answer"]


@pytest.mark.asyncio
async def test_agent_run_includes_qualification_verdict(client: AsyncClient):
    """Full /api/agent/run execution returns structured qualification when evaluate_lead_fit was executed."""
    step1_resp = LLMResponse(
        request_id="res_run_qual1",
        content="",
        tool_calls=[
            ToolCallPayload(
                id="call_lead_fit_run",
                name="evaluate_lead_fit",
                arguments={
                    "company_name": "Confluent",
                    "offerings_summary": "Confluent provides enterprise cloud data streaming, Apache Kafka B2B SaaS platform for developer APIs.",
                },
            )
        ],
        finish_reason="tool_calls",
        prompt_tokens=70,
        completion_tokens=25,
        total_tokens=95,
    )

    step2_resp = LLMResponse(
        request_id="res_run_qual2",
        content="Confluent is evaluated and qualified as an enterprise data streaming platform.",
        finish_reason="stop",
        prompt_tokens=130,
        completion_tokens=30,
        total_tokens=160,
    )

    with patch("app.agent.runner.call_llm", side_effect=[step1_resp, step2_resp]):
        response = await client.post(
            "/api/agent/run",
            json={
                "company_name": "Confluent",
                "goal": "Investigate and qualify Confluent for data infrastructure partnership",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["qualification"] is not None
    assert data["qualification"]["is_fit"] is True
    assert data["qualification"]["fit_score"] >= 0.5
    assert len(data["qualification"]["why_fit"]) > 0
    assert len(data["qualification"]["proposal_hooks"]) > 0
