import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from app.agent.tools.base import BaseTool, ToolOutput, ToolPermissionLevel
from app.agent.tools.definitions import (
    EvaluateLeadFitTool,
    FetchWebpageTool,
    SearchCompanyTool,
)
from app.agent.tools.registry import ToolRegistry, get_default_registry
from app.research.models import ExtractedPage, SearchResult


# Dummy tool for base tool testing
class DummyArgs(BaseModel):
    query: str = Field(..., min_length=2)
    limit: int = Field(default=5, ge=1, le=10)


class DummyTool(BaseTool):
    name = "dummy_tool"
    description = "A dummy testing tool."
    args_schema = DummyArgs
    permission_level = ToolPermissionLevel.READ_ONLY

    async def execute(self, query: str, limit: int = 5) -> dict[str, Any]:
        if query == "fail":
            raise RuntimeError("Database connection lost")
        if query == "sleep":
            await asyncio.sleep(2.0)
        return {"query": query, "count": limit}


def test_base_tool_openai_schema_generation():
    """to_openai_schema produces valid OpenAI/Gemini function-calling specification."""
    tool = DummyTool()
    schema = tool.to_openai_schema()

    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "dummy_tool"
    assert fn["description"] == "A dummy testing tool."
    assert "parameters" in fn
    props = fn["parameters"]["properties"]
    assert "query" in props
    assert "limit" in props
    assert "title" not in fn["parameters"]


@pytest.mark.asyncio
async def test_base_tool_argument_validation_and_execution():
    """run_with_guard validates arguments and returns ToolOutput."""
    tool = DummyTool()

    # Valid execution
    out = await tool.run_with_guard({"query": "hello", "limit": 3})
    assert isinstance(out, ToolOutput)
    assert out.success is True
    assert out.data == {"query": "hello", "count": 3}
    assert out.error is None

    # Invalid argument type (limit as invalid string)
    bad_out = await tool.run_with_guard({"query": "hello", "limit": "not_an_int"})
    assert bad_out.success is False
    assert "Invalid arguments for tool 'dummy_tool'" in (bad_out.error or "")

    # Missing required argument
    missing_out = await tool.run_with_guard({})
    assert missing_out.success is False
    assert "Invalid arguments" in (missing_out.error or "")


@pytest.mark.asyncio
async def test_base_tool_timeout_enforcement():
    """run_with_guard enforces timeout and catches asyncio.TimeoutError gracefully."""
    tool = DummyTool()
    out = await tool.run_with_guard({"query": "sleep"}, timeout_seconds=0.05)
    assert out.success is False
    assert "timed out after 0.05s" in (out.error or "")


@pytest.mark.asyncio
async def test_base_tool_exception_containment():
    """run_with_guard catches unhandled tool exceptions and returns clean error string."""
    tool = DummyTool()
    out = await tool.run_with_guard({"query": "fail"})
    assert out.success is False
    assert "Database connection lost" in (out.error or "")


@pytest.mark.asyncio
async def test_search_company_tool_execution():
    """SearchCompanyTool correctly wraps search_company and returns formatted results."""
    mock_results = [
        SearchResult(
            title="Stripe: Financial Infrastructure",
            url="https://stripe.com",
            snippet="Online payment processing for internet businesses.",
            source="bing",
        )
    ]
    with patch(
        "app.agent.tools.definitions.search_company", new=AsyncMock(return_value=mock_results)
    ):
        tool = SearchCompanyTool()
        out = await tool.run_with_guard({"query": "Stripe", "max_results": 2})

    assert out.success is True
    assert len(out.data) == 1
    assert out.data[0]["title"] == "Stripe: Financial Infrastructure"
    assert out.data[0]["url"] == "https://stripe.com"


@pytest.mark.asyncio
async def test_fetch_webpage_tool_with_budget_guard():
    """FetchWebpageTool enforces TPM output budget (<= 1,400 chars) on large pages."""
    long_chunk = "A" * 800
    mock_page = ExtractedPage(
        url="https://acme.com",
        title="Acme Corporation",
        content="Acme landing page content.",
        chunks=[long_chunk, long_chunk, long_chunk],  # 2,400 chars total
        internal_links=[],
    )

    with patch("app.agent.tools.definitions.extract_page", new=AsyncMock(return_value=mock_page)):
        tool = FetchWebpageTool()
        out = await tool.run_with_guard({"url": "https://acme.com"})

    assert out.success is True
    data = out.data
    assert data["title"] == "Acme Corporation"
    assert data["status"] == "success"
    # Content must be bounded within the budget guard (< 1,500 chars)
    assert len(data["content"]) <= 1450


@pytest.mark.asyncio
async def test_fetch_webpage_tool_handles_failed_page():
    """FetchWebpageTool handles None return from extract_page (SSRF or HTTP block) safely."""
    with patch("app.agent.tools.definitions.extract_page", new=AsyncMock(return_value=None)):
        tool = FetchWebpageTool()
        out = await tool.run_with_guard({"url": "http://169.254.169.254/latest"})

    assert out.success is True
    assert out.data["status"] == "failed"
    assert "Unable to access webpage" in out.data["content"]


@pytest.mark.asyncio
async def test_evaluate_lead_fit_tool():
    """EvaluateLeadFitTool evaluates target criteria match ratio deterministically."""
    tool = EvaluateLeadFitTool()

    # Good fit scenario
    summary_good = (
        "Snowflake delivers enterprise cloud data warehousing, automated data sharing, "
        "and multi-cloud compliance governance for high-scale enterprise engineering organizations."
    )
    criteria = ["enterprise cloud data", "compliance governance", "consumer mobile gaming"]

    out = await tool.run_with_guard(
        {
            "company_name": "Snowflake",
            "offerings_summary": summary_good,
            "target_criteria": criteria,
        }
    )
    assert out.success is True
    data = out.data
    assert data["company_name"] == "Snowflake"
    assert len(data["matched_criteria"]) == 2
    assert "consumer mobile gaming" in data["unmatched_criteria"]
    assert data["fit_score"] == 0.67
    assert data["is_qualified"] is True


def test_tool_registry_management():
    """ToolRegistry registers, retrieves, and generates schemas with filters."""
    reg = ToolRegistry()
    dummy = DummyTool()
    reg.register(dummy)

    assert reg.has("dummy_tool") is True
    assert reg.get("dummy_tool") == dummy
    assert len(reg.list_tools()) == 1

    schemas = reg.get_openai_tools()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "dummy_tool"

    # Filter with non-matching name
    empty_schemas = reg.get_openai_tools(allowed_names=["non_existent"])
    assert len(empty_schemas) == 0


@pytest.mark.asyncio
async def test_tool_registry_execute_tool():
    """ToolRegistry dispatches execution by name and handles unknown tool requests."""
    reg = ToolRegistry()
    reg.register(DummyTool())

    # Registered tool
    out = await reg.execute_tool("dummy_tool", {"query": "test"})
    assert out.success is True
    assert out.data["query"] == "test"

    # Unregistered tool
    unknown_out = await reg.execute_tool("unknown_tool", {})
    assert unknown_out.success is False
    assert "is not registered" in (unknown_out.error or "")


def test_get_default_registry():
    """get_default_registry includes core sales intelligence tools."""
    reg = get_default_registry()
    assert reg.has("search_company") is True
    assert reg.has("fetch_webpage") is True
    assert reg.has("evaluate_lead_fit") is True
    assert len(reg.list_tools()) >= 3
