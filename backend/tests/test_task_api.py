from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.llm import LLMResponse, ToolCallPayload
from app.research.models import SearchResult


@pytest.mark.asyncio
async def test_task_observability_endpoints(client: AsyncClient):
    """GET /api/tasks/{id}, /steps, and /tools expose persisted audit logs."""
    turn1 = LLMResponse(
        request_id="r_t1",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=60,
        completion_tokens=20,
        total_tokens=80,
        tool_calls=[
            ToolCallPayload(
                id="call_search_1",
                name="search_company",
                arguments={"query": "Snowflake", "max_results": 2},
            )
        ],
    )
    turn2 = LLMResponse(
        request_id="r_t2",
        content="Snowflake is a cloud data platform.",
        finish_reason="stop",
        prompt_tokens=110,
        completion_tokens=25,
        total_tokens=135,
    )
    mock_results = [
        SearchResult(
            title="Snowflake",
            url="https://snowflake.com",
            snippet="Data cloud platform",
            source="bing",
        )
    ]

    with patch("app.agent.runner.call_llm", side_effect=[turn1, turn2]):
        with patch(
            "app.agent.tools.definitions.search_company",
            new=AsyncMock(return_value=mock_results),
        ):
            run_resp = await client.post(
                "/api/agent/run",
                json={"goal": "Investigate Snowflake", "company_name": "Snowflake"},
            )

    assert run_resp.status_code == 200
    run_data = run_resp.json()
    task_id = run_data["task_id"]
    assert task_id.startswith("task_")

    # 1. GET /api/tasks/{task_id}
    task_resp = await client.get(f"/api/tasks/{task_id}")
    assert task_resp.status_code == 200
    task_data = task_resp.json()
    assert task_data["status"] == "COMPLETED"
    assert task_data["steps_taken"] == 2
    assert task_data["total_tokens"] == 215
    assert task_data["final_answer"] == "Snowflake is a cloud data platform."

    # 2. GET /api/tasks/{task_id}/steps
    steps_resp = await client.get(f"/api/tasks/{task_id}/steps")
    assert steps_resp.status_code == 200
    steps = steps_resp.json()
    assert len(steps) == 2
    assert steps[0]["assistant_message"]["role"] == "assistant"
    assert steps[0]["tool_messages"][0]["tool_call_id"] == "call_search_1"

    # 3. GET /api/tasks/{task_id}/tools
    tools_resp = await client.get(f"/api/tasks/{task_id}/tools")
    assert tools_resp.status_code == 200
    tools = tools_resp.json()
    assert len(tools) == 1
    assert tools[0]["tool_name"] == "search_company"
    assert tools[0]["status"] == "COMPLETED"
    assert len(tools[0]["idempotency_key"]) == 64

    # 4. 404 on unknown task_id
    assert (await client.get("/api/tasks/task_missing")).status_code == 404
    assert (await client.get("/api/tasks/task_missing/steps")).status_code == 404
    assert (await client.get("/api/tasks/task_missing/tools")).status_code == 404
