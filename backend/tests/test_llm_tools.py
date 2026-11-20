import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.llm import LLMRequest, LLMResponse, ToolCallPayload, call_llm


def test_llm_request_prompt_backward_compatibility():
    """LLMRequest with prompt continues to work seamlessly without messages."""
    req = LLMRequest(prompt="What is Stripe?", temperature=0.5)
    assert req.prompt == "What is Stripe?"
    assert req.messages is None
    assert req.tools is None
    assert req.tool_choice is None


def test_llm_request_messages_support():
    """LLMRequest supports messages directly without requiring prompt."""
    msgs = [
        {"role": "system", "content": "You are a sales assistant."},
        {"role": "user", "content": "Evaluate Snowflake."},
    ]
    req = LLMRequest(messages=msgs, temperature=0.2)
    assert req.messages == msgs
    assert req.prompt is None


def test_llm_request_validates_prompt_or_messages():
    """Either prompt or messages must be provided, and prompt cannot be blank."""
    with pytest.raises(ValidationError, match="Either 'prompt' or 'messages' must be provided"):
        LLMRequest()

    with pytest.raises(ValidationError, match="Prompt cannot be empty or only whitespace"):
        LLMRequest(prompt="   ")

    with pytest.raises(ValidationError, match="'messages' cannot be empty when provided"):
        LLMRequest(messages=[])


def test_llm_response_defaults_tool_calls_empty():
    """Standard LLMResponse defaults tool_calls to an empty list."""
    resp = LLMResponse(
        request_id="req1",
        content="Final answer.",
        finish_reason="stop",
    )
    assert resp.tool_calls == []


@pytest.mark.asyncio
async def test_call_llm_parses_single_tool_call():
    """call_llm parses a single tool call from the Groq message into ToolCallPayload."""
    raw_tc = MagicMock()
    raw_tc.id = "call_search_1"
    raw_tc.function = MagicMock()
    raw_tc.function.name = "search_company"
    raw_tc.function.arguments = json.dumps({"query": "Snowflake Inc", "max_results": 3})

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [raw_tc]

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = MagicMock(prompt_tokens=40, completion_tokens=25, total_tokens=65)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    req = LLMRequest(
        prompt="Search for Snowflake",
        tools=[{"type": "function", "function": {"name": "search_company"}}],
    )
    resp = await call_llm(req, client=mock_client)

    assert isinstance(resp, LLMResponse)
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert isinstance(tc, ToolCallPayload)
    assert tc.id == "call_search_1"
    assert tc.name == "search_company"
    assert tc.arguments == {"query": "Snowflake Inc", "max_results": 3}


@pytest.mark.asyncio
async def test_call_llm_parses_parallel_tool_calls():
    """call_llm parses multiple parallel tool calls emitted in a single turn."""
    raw_tc1 = MagicMock()
    raw_tc1.id = "call_search_1"
    raw_tc1.function = MagicMock()
    raw_tc1.function.name = "search_company"
    raw_tc1.function.arguments = json.dumps({"query": "Snowflake official"})

    raw_tc2 = MagicMock()
    raw_tc2.id = "call_fetch_2"
    raw_tc2.function = MagicMock()
    raw_tc2.function.name = "fetch_webpage"
    raw_tc2.function.arguments = json.dumps({"url": "https://snowflake.com"})

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message = MagicMock()
    mock_choice.message.content = ""
    mock_choice.message.tool_calls = [raw_tc1, raw_tc2]

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = MagicMock(prompt_tokens=60, completion_tokens=40, total_tokens=100)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    req = LLMRequest(
        prompt="Search and fetch Snowflake",
        tools=[
            {"type": "function", "function": {"name": "search_company"}},
            {"type": "function", "function": {"name": "fetch_webpage"}},
        ],
    )
    resp = await call_llm(req, client=mock_client)

    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].name == "search_company"
    assert resp.tool_calls[0].arguments == {"query": "Snowflake official"}
    assert resp.tool_calls[1].name == "fetch_webpage"
    assert resp.tool_calls[1].arguments == {"url": "https://snowflake.com"}


@pytest.mark.asyncio
async def test_call_llm_resilient_tool_arguments_parsing():
    """call_llm uses _safe_parse_json to handle trailing commas or minor JSON defects in tool arguments."""
    raw_tc = MagicMock()
    raw_tc.id = "call_defect_1"
    raw_tc.function = MagicMock()
    raw_tc.function.name = "search_company"
    # JSON with trailing comma (invalid under strict json.loads, handled by _safe_parse_json)
    raw_tc.function.arguments = '{"query": "Databricks", "max_results": 5, }'

    mock_choice = MagicMock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message = MagicMock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [raw_tc]

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = MagicMock(prompt_tokens=30, completion_tokens=15, total_tokens=45)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    req = LLMRequest(
        prompt="Find Databricks",
        tools=[{"type": "function", "function": {"name": "search_company"}}],
    )
    resp = await call_llm(req, client=mock_client)

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].arguments == {"query": "Databricks", "max_results": 5}


@pytest.mark.asyncio
async def test_call_llm_forwards_tools_and_tool_choice():
    """call_llm forwards tools and tool_choice parameters to groq completions create."""
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message = MagicMock(content="Done.", tool_calls=None)

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = MagicMock(prompt_tokens=20, completion_tokens=5, total_tokens=25)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    tools_spec = [{"type": "function", "function": {"name": "search_company"}}]
    req = LLMRequest(
        prompt="Evaluate company",
        tools=tools_spec,
        tool_choice="auto",
    )
    await call_llm(req, client=mock_client)

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == tools_spec
    assert call_kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_call_llm_preserves_multi_turn_messages():
    """call_llm passes multi-turn messages array directly to Groq client."""
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message = MagicMock(content="Here is the report.", tool_calls=None)

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = MagicMock(prompt_tokens=80, completion_tokens=20, total_tokens=100)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

    messages = [
        {"role": "system", "content": "You are a sales agent."},
        {"role": "user", "content": "Analyze Stripe."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call1"}]},
        {
            "role": "tool",
            "tool_call_id": "call1",
            "name": "search_company",
            "content": '{"results": []}',
        },
    ]
    req = LLMRequest(messages=messages)
    await call_llm(req, client=mock_client)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["messages"] == messages
