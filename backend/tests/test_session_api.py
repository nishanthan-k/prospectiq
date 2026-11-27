from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.llm import LLMResponse, ToolCallPayload
from app.research.models import ExtractedPage


@pytest.mark.asyncio
async def test_session_crud_and_soft_delete(client: AsyncClient):
    """POST/GET/DELETE /api/sessions manage persistent sessions cleanly."""
    # 1. Validation error on blank title
    bad_resp = await client.post("/api/sessions", json={"title": "   "})
    assert bad_resp.status_code == 422

    # 2. Create session
    create_resp = await client.post(
        "/api/sessions",
        json={
            "title": "Stripe Evaluation",
            "company_name": "Stripe",
            "domain": "stripe.com",
            "context": {"owner": "AE-1"},
            "metadata": {"priority": "high"},
        },
    )
    assert create_resp.status_code == 201
    sess = create_resp.json()
    sess_id = sess["id"]
    assert sess_id.startswith("sess_")
    assert sess["title"] == "Stripe Evaluation"
    assert sess["metadata"] == {"priority": "high"}

    # 3. List sessions
    list_resp = await client.get("/api/sessions?limit=10&offset=0")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    # 4. Get session detail
    detail_resp = await client.get(f"/api/sessions/{sess_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["memories"] == []
    assert detail["tasks"] == []

    # 5. Soft delete
    del_resp = await client.delete(f"/api/sessions/{sess_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Soft-deleted session returns 404
    assert (await client.get(f"/api/sessions/{sess_id}")).status_code == 404
    assert (await client.get(f"/api/sessions/{sess_id}/memories")).status_code == 404
    assert (await client.delete(f"/api/sessions/{sess_id}")).status_code == 404


@pytest.mark.asyncio
async def test_run_agent_in_session_persists_memories(client: AsyncClient):
    """POST /api/sessions/{id}/agent/run executes agent and populates session memories."""
    create_resp = await client.post(
        "/api/sessions",
        json={"title": "Retool", "company_name": "Retool", "domain": "retool.com"},
    )
    sess_id = create_resp.json()["id"]

    turn1 = LLMResponse(
        request_id="r_1",
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCallPayload(
                id="c_retool",
                name="fetch_webpage",
                arguments={"url": "https://retool.com"},
            )
        ],
    )
    turn2 = LLMResponse(
        request_id="r_2",
        content="Retool builds internal software platforms.",
        finish_reason="stop",
    )
    mock_page = ExtractedPage(
        url="https://retool.com",
        title="Retool",
        content="Build internal applications, dashboards, and workflows fast.",
        chunks=["Build internal applications, dashboards, and workflows fast."],
        internal_links=[],
    )

    with patch("app.agent.runner.call_llm", side_effect=[turn1, turn2]):
        with patch(
            "app.agent.tools.definitions.extract_page", new=AsyncMock(return_value=mock_page)
        ):
            run_resp = await client.post(
                f"/api/sessions/{sess_id}/agent/run",
                json={"goal": "Investigate Retool offerings"},
            )

    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["session_id"] == sess_id
    assert run_data["task_id"].startswith("task_")

    mem_resp = await client.get(f"/api/sessions/{sess_id}/memories")
    assert mem_resp.status_code == 200
    memories = mem_resp.json()
    assert len(memories) == 1
    assert memories[0]["source_url"] == "https://retool.com"
    assert memories[0]["is_verified"] is True

    # Missing session_id returns 404 on both endpoints
    assert (
        await client.post("/api/sessions/sess_missing/agent/run", json={"goal": "Investigate"})
    ).status_code == 404
    assert (
        await client.post(
            "/api/agent/run", json={"goal": "Investigate", "session_id": "sess_missing"}
        )
    ).status_code == 404
