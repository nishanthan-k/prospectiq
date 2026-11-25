from unittest.mock import AsyncMock, patch

import pytest

from app.agent.models import AgentRunResult
from app.agent.runner import AgentRunner
from app.llm import LLMResponse, ToolCallPayload


@pytest.mark.asyncio
async def test_agent_runner_immediate_completion():
    """Agent completes in a single turn if LLM provides a direct text response without tool calls."""
    runner = AgentRunner()

    mock_resp = LLMResponse(
        request_id="res_1",
        content="Snowflake is a cloud data platform founded in 2012.",
        finish_reason="stop",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
    )

    with patch("app.agent.runner.call_llm", new=AsyncMock(return_value=mock_resp)):
        result = await runner.run(goal="Explain what Snowflake does")

    assert isinstance(result, AgentRunResult)
    assert result.steps_taken == 1
    assert result.is_completed is True
    assert "Snowflake is a cloud data platform" in result.final_answer
    assert len(result.tool_calls) == 0
    assert result.total_tokens == 70


@pytest.mark.asyncio
async def test_agent_runner_single_tool_execution():
    """Agent calls a search tool in step 1, receives the result, and synthesizes the answer in step 2."""
    runner = AgentRunner()

    # Turn 1: Model requests search_company tool
    turn1_resp = LLMResponse(
        request_id="res_t1",
        content=None,
        finish_reason="tool_calls",
        prompt_tokens=50,
        completion_tokens=25,
        total_tokens=75,
        tool_calls=[
            ToolCallPayload(
                id="call_1",
                name="search_company",
                arguments={"query": "Snowflake Inc", "max_results": 2},
            )
        ],
    )

    # Turn 2: Model finishes with synthesis text
    turn2_resp = LLMResponse(
        request_id="res_t2",
        content="Based on the search results, Snowflake is a cloud data warehouse provider.",
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=30,
        total_tokens=130,
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        with patch("app.agent.tools.definitions.search_company", new=AsyncMock(return_value=[])):
            result = await runner.run(goal="Research Snowflake Inc")

    assert result.steps_taken == 2
    assert result.is_completed is True
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.tool_name == "search_company"
    assert tc.tool_call_id == "call_1"
    assert tc.is_error is False
    assert "Snowflake is a cloud data warehouse provider" in result.final_answer
    assert result.total_tokens == 205


@pytest.mark.asyncio
async def test_agent_runner_parallel_tool_calls():
    """Agent emits multiple parallel tool calls in one turn; runner batches execution and appends responses."""
    runner = AgentRunner()

    turn1_resp = LLMResponse(
        request_id="res_p1",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=60,
        completion_tokens=40,
        total_tokens=100,
        tool_calls=[
            ToolCallPayload(
                id="call_s1",
                name="search_company",
                arguments={"query": "Databricks official"},
            ),
            ToolCallPayload(
                id="call_f2",
                name="fetch_webpage",
                arguments={"url": "https://databricks.com"},
            ),
        ],
    )

    turn2_resp = LLMResponse(
        request_id="res_p2",
        content="Databricks is a data intelligence platform built on Apache Spark.",
        finish_reason="stop",
        prompt_tokens=150,
        completion_tokens=30,
        total_tokens=180,
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        with patch("app.agent.tools.definitions.search_company", new=AsyncMock(return_value=[])):
            with patch(
                "app.agent.tools.definitions.extract_page", new=AsyncMock(return_value=None)
            ):
                result = await runner.run(goal="Research Databricks")

    assert result.steps_taken == 2
    assert len(result.tool_calls) == 2
    tool_names = [tc.tool_name for tc in result.tool_calls]
    assert "search_company" in tool_names
    assert "fetch_webpage" in tool_names


@pytest.mark.asyncio
async def test_agent_runner_cycle_detection():
    """If the LLM emits identical tool calls consecutively, runner detects cycle and injects warning."""
    runner = AgentRunner(max_steps=3)

    repeated_tc = ToolCallPayload(
        id="call_loop",
        name="search_company",
        arguments={"query": "Stripe payments"},
    )

    # Step 1: Tool call
    resp1 = LLMResponse(
        request_id="r1",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        tool_calls=[repeated_tc],
    )
    # Step 2: Exact same tool call (cycle)
    resp2 = LLMResponse(
        request_id="r2",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=80,
        completion_tokens=20,
        total_tokens=100,
        tool_calls=[repeated_tc],
    )
    # Step 3: Model breaks out and provides answer
    resp3 = LLMResponse(
        request_id="r3",
        content="Stripe is a payments platform.",
        finish_reason="stop",
        prompt_tokens=110,
        completion_tokens=15,
        total_tokens=125,
    )

    with patch("app.agent.runner.call_llm", side_effect=[resp1, resp2, resp3]):
        with patch("app.agent.tools.definitions.search_company", new=AsyncMock(return_value=[])):
            result = await runner.run(goal="Investigate Stripe")

    assert result.steps_taken == 3
    assert len(result.tool_calls) == 2
    # Verify that the second call received the cycle warning note
    second_call_output = result.tool_calls[1].output
    assert isinstance(second_call_output, dict)
    assert "_system_note" in second_call_output
    assert (
        "Warning: This exact tool call was already executed" in second_call_output["_system_note"]
    )


@pytest.mark.asyncio
async def test_agent_runner_forced_synthesis_on_max_steps():
    """When step limit is reached without final text, runner invokes forced synthesis turn."""
    runner = AgentRunner()

    tool_call_item = ToolCallPayload(
        id="call_step",
        name="search_company",
        arguments={"query": "AWS"},
    )

    # Steps 0 and 1 keep returning tool calls
    resp_loop1 = LLMResponse(
        request_id="r_l1",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=50,
        completion_tokens=20,
        total_tokens=70,
        tool_calls=[tool_call_item],
    )
    resp_loop2 = LLMResponse(
        request_id="r_l2",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=80,
        completion_tokens=20,
        total_tokens=100,
        tool_calls=[tool_call_item],
    )
    # Forced synthesis turn returns final text with tool_choice=None
    synth_resp = LLMResponse(
        request_id="r_synth",
        content="Synthesized partial report on AWS.",
        finish_reason="stop",
        prompt_tokens=120,
        completion_tokens=25,
        total_tokens=145,
    )

    with patch("app.agent.runner.call_llm", side_effect=[resp_loop1, resp_loop2, synth_resp]):
        with patch("app.agent.tools.definitions.search_company", new=AsyncMock(return_value=[])):
            result = await runner.run(goal="Investigate AWS", max_steps=2)

    # 2 tool turns + 1 forced synthesis turn = 3 steps taken
    assert result.steps_taken == 3
    assert result.is_completed is False  # Flagged that max_steps was hit
    assert "Synthesized partial report on AWS" in result.final_answer
    assert len(result.tool_calls) == 2


@pytest.mark.asyncio
async def test_agent_runner_tool_error_resilience():
    """If a tool fails with an error or bad arguments, the error is recorded and agent continues."""
    runner = AgentRunner()

    # Step 1: Model requests unknown tool
    turn1_resp = LLMResponse(
        request_id="res_err",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=40,
        completion_tokens=20,
        total_tokens=60,
        tool_calls=[
            ToolCallPayload(
                id="call_bad",
                name="non_existent_tool",
                arguments={},
            )
        ],
    )
    # Step 2: Model recognizes failure and outputs fallback response
    turn2_resp = LLMResponse(
        request_id="res_recover",
        content="I could not execute the requested tool, but here are the known facts.",
        finish_reason="stop",
        prompt_tokens=80,
        completion_tokens=25,
        total_tokens=105,
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        result = await runner.run(goal="Test error recovery")

    assert result.steps_taken == 2
    assert result.is_completed is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].is_error is True
    assert "is not registered" in str(result.tool_calls[0].output)


@pytest.mark.asyncio
async def test_agent_runner_malformed_arguments_rejected():
    """Agent runner rejects tool calls with malformed arguments without executing the tool."""
    runner = AgentRunner()

    malformed_tc = ToolCallPayload(
        id="call_bad_args",
        name="search_company",
        arguments={},
        parse_error="Malformed JSON arguments: {bad_json",
    )
    turn1_resp = LLMResponse(
        request_id="res_malformed",
        content="",
        finish_reason="tool_calls",
        tool_calls=[malformed_tc],
    )
    turn2_resp = LLMResponse(
        request_id="res_synth",
        content="Final answer after argument error.",
        finish_reason="stop",
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        with patch("app.agent.tools.definitions.search_company") as mock_search:
            result = await runner.run(goal="Test malformed args")

    mock_search.assert_not_called()
    assert result.steps_taken == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].is_error is True
    assert "Malformed tool call arguments" in str(result.tool_calls[0].output)


@pytest.mark.asyncio
async def test_agent_runner_valid_empty_json_arguments_succeeds():
    """Valid empty JSON arguments ({}) are distinguishable from malformed JSON and execute normally."""
    runner = AgentRunner()

    valid_empty_tc = ToolCallPayload(
        id="call_valid_empty",
        name="search_company",
        arguments={"query": "fallback"},
        parse_error=None,
    )
    turn1_resp = LLMResponse(
        request_id="res_empty_valid",
        content="",
        finish_reason="tool_calls",
        tool_calls=[valid_empty_tc],
    )
    turn2_resp = LLMResponse(
        request_id="res_synth",
        content="Success with valid args.",
        finish_reason="stop",
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        with patch("app.agent.tools.definitions.search_company", new=AsyncMock(return_value=[])):
            result = await runner.run(goal="Test valid args")

    assert result.steps_taken == 2
    assert result.tool_calls[0].is_error is False


@pytest.mark.asyncio
async def test_agent_runner_max_tool_calls_per_turn_enforcement():
    """Agent runner bounds tool execution: excess tool calls beyond max_tool_calls_per_turn are rejected."""
    runner = AgentRunner(max_tool_calls_per_turn=2)

    tool_calls = [
        ToolCallPayload(id=f"c_{i}", name="search_company", arguments={"query": f"q_{i}"})
        for i in range(4)
    ]
    turn1_resp = LLMResponse(
        request_id="res_many_tools",
        content="",
        finish_reason="tool_calls",
        tool_calls=tool_calls,
    )
    turn2_resp = LLMResponse(
        request_id="res_done",
        content="Completed after bounding tools.",
        finish_reason="stop",
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        with patch("app.agent.tools.definitions.search_company", new=AsyncMock(return_value=[])):
            result = await runner.run(goal="Test tool limits")

    assert len(result.tool_calls) == 4
    # First 2 succeeded
    assert result.tool_calls[0].is_error is False
    assert result.tool_calls[1].is_error is False
    # Next 2 were rejected by bounds check
    assert result.tool_calls[2].is_error is True
    assert "exceeded maximum allowable tool calls" in str(result.tool_calls[2].output)
    assert result.tool_calls[3].is_error is True
    assert "exceeded maximum allowable tool calls" in str(result.tool_calls[3].output)


@pytest.mark.asyncio
async def test_agent_runner_failed_webpage_extraction_recorded_as_error():
    """Failed webpage extraction records is_error=True in tool call trace."""
    runner = AgentRunner()

    turn1_resp = LLMResponse(
        request_id="res_fetch",
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCallPayload(
                id="c_fetch",
                name="fetch_webpage",
                arguments={"url": "https://blocked.example.com"},
            )
        ],
    )
    turn2_resp = LLMResponse(
        request_id="res_done",
        content="Finished despite page fetch failure.",
        finish_reason="stop",
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        with patch("app.agent.tools.definitions.extract_page", new=AsyncMock(return_value=None)):
            result = await runner.run(goal="Test failed fetch")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].is_error is True
    assert "Unable to access webpage" in result.tool_calls[0].output["error"]


@pytest.mark.asyncio
async def test_agent_runner_validates_inputs():
    """AgentRunner validates goal, max_steps, and allowed_tools internally."""
    runner = AgentRunner()

    # Blank goal
    with pytest.raises(ValueError, match="goal cannot be empty"):
        await runner.run(goal="   ")

    # Invalid max_steps
    with pytest.raises(ValueError, match="max_steps must be an integer between 1 and 20"):
        await runner.run(goal="Valid goal", max_steps=0)

    with pytest.raises(ValueError, match="max_steps must be an integer between 1 and 20"):
        await runner.run(goal="Valid goal", max_steps=25)

    # Unknown tool in allowed_tools
    with pytest.raises(ValueError, match="Unknown tool in allowed_tools: 'fake_tool'"):
        await runner.run(goal="Valid goal", allowed_tools=["fake_tool"])

    # Constructor validation
    with pytest.raises(ValueError, match="max_steps must be an integer between 1 and 20"):
        AgentRunner(max_steps=0)

    with pytest.raises(ValueError, match="Unknown tool in allowed_tools"):
        AgentRunner(allowed_tools=["unknown_tool"])


@pytest.mark.asyncio
async def test_agent_runner_empty_final_response_handled():
    """Empty final response from model is flagged as incomplete when retries exhaust."""
    runner = AgentRunner(max_steps=1)

    empty_resp = LLMResponse(
        request_id="res_empty",
        content="   ",
        finish_reason="stop",
        tool_calls=[],
    )

    with patch("app.agent.runner.call_llm", return_value=empty_resp):
        result = await runner.run(goal="Evaluate something", max_steps=1)

    assert result.is_completed is False
    assert "failed to produce a final answer" in result.final_answer
