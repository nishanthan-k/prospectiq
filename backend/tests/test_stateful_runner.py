import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.runner import AgentRunner
from app.db import Base, SessionRepository, TaskRepository, create_db_engine
from app.llm import LLMResponse, ToolCallPayload
from app.research.models import ExtractedPage


@pytest_asyncio.fixture
async def db_session():
    """Provide an isolated in-memory async SQLite session for stateful runner tests."""
    engine = create_db_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_stateful_runner_checkpoints_steps_tools_and_memories(db_session: AsyncSession):
    """Stateful run creates implicit session/task, checkpoints steps & tools, and persists memories."""
    runner = AgentRunner()

    turn1_resp = LLMResponse(
        request_id="r_s0",
        content="",
        finish_reason="tool_calls",
        prompt_tokens=100,
        completion_tokens=30,
        total_tokens=130,
        tool_calls=[
            ToolCallPayload(
                id="call_fetch_ok",
                name="fetch_webpage",
                arguments={"url": "https://stripe.com"},
            ),
            ToolCallPayload(
                id="call_fetch_fail",
                name="fetch_webpage",
                arguments={"url": "https://blocked.example.com"},
            ),
        ],
    )
    turn2_resp = LLMResponse(
        request_id="r_s1",
        content="Stripe provides global payment infrastructure.",
        finish_reason="stop",
        prompt_tokens=200,
        completion_tokens=40,
        total_tokens=240,
    )

    mock_page = ExtractedPage(
        url="https://stripe.com",
        title="Stripe",
        content="Financial infrastructure for the internet.",
        chunks=["Financial infrastructure for the internet."],
        internal_links=[],
    )

    async def fake_extract(url: str):
        return mock_page if "stripe.com" in url else None

    with patch("app.agent.runner.call_llm", side_effect=[turn1_resp, turn2_resp]):
        with patch("app.agent.tools.definitions.extract_page", side_effect=fake_extract):
            result = await runner.run(
                goal="Investigate Stripe",
                company_name="Stripe",
                domain="stripe.com",
                db=db_session,
            )

    assert result.is_completed is True
    assert result.session_id is not None and result.session_id.startswith("sess_")
    assert result.task_id is not None and result.task_id.startswith("task_")

    t_repo = TaskRepository(db_session)
    s_repo = SessionRepository(db_session)

    # Verify Task status and metrics
    task = await t_repo.get_task(result.task_id)
    assert task is not None
    assert task.status == "COMPLETED"
    assert task.steps_taken == 2
    assert task.total_tokens == 370
    assert task.final_answer == "Stripe provides global payment infrastructure."

    # Verify TaskSteps recorded full protocol messages
    steps = await t_repo.get_steps(result.task_id)
    assert len(steps) == 2
    assert steps[0].assistant_message["role"] == "assistant"
    assert len(steps[0].assistant_message["tool_calls"]) == 2
    assert len(steps[0].tool_messages) == 2
    assert steps[0].tool_messages[0]["tool_call_id"] == "call_fetch_ok"
    assert steps[1].assistant_message["content"] == "Stripe provides global payment infrastructure."

    # Verify ToolExecution records (1 COMPLETED, 1 FAILED)
    execs = await t_repo.get_tool_executions(result.task_id)
    assert len(execs) == 2
    statuses = {e.tool_call_id: e.status for e in execs}
    assert statuses["call_fetch_ok"] == "COMPLETED"
    assert statuses["call_fetch_fail"] == "FAILED"

    # Verify SessionMemory extracted grounded fact from stripe.com
    memories = await s_repo.get_memories(result.session_id)
    assert len(memories) == 1
    assert memories[0].source_url == "https://stripe.com"
    assert memories[0].confidence == 0.90
    assert memories[0].is_verified is True


@pytest.mark.asyncio
async def test_stateful_runner_multi_turn_session_injects_prior_memories(db_session: AsyncSession):
    """Follow-up task in the same session injects persisted SessionMemory into prompt."""
    s_repo = SessionRepository(db_session)
    sess = await s_repo.create_session(
        title="Retool Research", company_name="Retool", domain="retool.com"
    )
    runner = AgentRunner()

    # Run 1: fetches webpage and persists SessionMemory
    turn1 = LLMResponse(
        request_id="r1_0",
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCallPayload(
                id="c_retool", name="fetch_webpage", arguments={"url": "https://retool.com"}
            )
        ],
    )
    turn2 = LLMResponse(
        request_id="r1_1",
        content="Retool builds internal software fast.",
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
            res1 = await runner.run(
                goal="What does Retool build?",
                session_id=sess.id,
                db=db_session,
            )
    assert res1.session_id == sess.id

    # Run 2: follow-up question on same session_id receives prior memory in prompt
    followup_mock = AsyncMock(
        return_value=LLMResponse(
            request_id="r2_0",
            content="Based on prior observations, Retool builds internal tools.",
            finish_reason="stop",
        )
    )
    with patch("app.agent.runner.call_llm", new=followup_mock):
        res2 = await runner.run(
            goal="Summarize their core offering again.",
            session_id=sess.id,
            db=db_session,
        )

    assert res2.is_completed is True
    sent_req = followup_mock.call_args.args[0]
    user_prompt = sent_req.messages[1]["content"]
    assert "<untrusted_session_memory>" in user_prompt
    assert "https://retool.com" in user_prompt
    assert "Build internal applications, dashboards, and workflows fast." in user_prompt


@pytest.mark.asyncio
async def test_stateful_runner_unknown_tool_crash_and_missing_session(db_session: AsyncSession):
    """Unhandled tool exception marks ToolExecution UNKNOWN and Task FAILED; missing session raises."""
    runner = AgentRunner()

    with pytest.raises(ValueError, match="Session 'sess_missing' not found"):
        await runner.run(goal="Hello", session_id="sess_missing", db=db_session)

    # Simulate BaseException (e.g. CancelledError/KeyboardInterrupt-like or patched execute_tool crash)
    turn1 = LLMResponse(
        request_id="r_crash",
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCallPayload(id="c_crash", name="search_company", arguments={"query": "Stripe"})
        ],
    )

    with patch("app.agent.runner.call_llm", return_value=turn1):
        with patch.object(
            runner.registry,
            "execute_tool",
            side_effect=RuntimeError("Unexpected registry crash"),
        ):
            with pytest.raises(RuntimeError, match="Unexpected registry crash"):
                await runner.run(goal="Crash test", db=db_session)

    t_repo = TaskRepository(db_session)
    s_repo = SessionRepository(db_session)
    sessions = await s_repo.list_sessions()
    assert len(sessions) == 1
    tasks = await t_repo.list_tasks_for_session(sessions[0].id)
    assert len(tasks) == 1
    assert tasks[0].status == "FAILED"
    assert "Unexpected registry crash" in (tasks[0].error_message or "")

    execs = await t_repo.get_tool_executions(tasks[0].id)
    assert len(execs) == 1
    assert execs[0].status == "UNKNOWN"
    assert json.loads(json.dumps(execs[0].input_arguments)) == {"query": "Stripe"}


@pytest.mark.asyncio
async def test_stateful_runner_parallel_partial_crash_persists_sibling_result(
    db_session: AsyncSession,
):
    """If one parallel tool crashes and another succeeds, both ToolExecutions record exact statuses."""
    runner = AgentRunner()

    turn1 = LLMResponse(
        request_id="r_partial",
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCallPayload(id="c_crash", name="search_company", arguments={"query": "crash"}),
            ToolCallPayload(id="c_ok", name="search_company", arguments={"query": "ok"}),
        ],
    )

    async def partial_execute(name: str, args: dict, **kwargs):
        if args.get("query") == "crash":
            raise RuntimeError("Boom on c_crash")
        from app.agent.tools.base import ToolOutput

        return ToolOutput(success=True, data=[{"title": "OK"}], error=None)

    with patch("app.agent.runner.call_llm", return_value=turn1):
        with patch.object(runner.registry, "execute_tool", side_effect=partial_execute):
            with pytest.raises(RuntimeError, match="Boom on c_crash"):
                await runner.run(goal="Partial crash test", db=db_session)

    s_repo = SessionRepository(db_session)
    t_repo = TaskRepository(db_session)
    sessions = await s_repo.list_sessions()
    tasks = await t_repo.list_tasks_for_session(sessions[0].id)
    assert tasks[0].status == "FAILED"

    execs = await t_repo.get_tool_executions(tasks[0].id)
    statuses = {e.tool_call_id: e.status for e in execs}
    assert statuses == {"c_crash": "UNKNOWN", "c_ok": "COMPLETED"}


@pytest.mark.asyncio
async def test_stateful_runner_reuses_idempotent_tool_output_and_session_context(
    db_session: AsyncSession,
):
    """Duplicate idempotent tool+args in the same task reuses cached output; follow-ups inherit session context."""
    s_repo = SessionRepository(db_session)
    sess = await s_repo.create_session(
        title="Context & Idempotency",
        company_name="Stripe",
        domain="stripe.com",
        context={"notes": "Enterprise fintech priority"},
    )
    runner = AgentRunner()

    turn1 = LLMResponse(
        request_id="r_idem_1",
        content="",
        finish_reason="tool_calls",
        tool_calls=[ToolCallPayload(id="c1", name="search_company", arguments={"query": "Stripe"})],
    )
    turn2 = LLMResponse(
        request_id="r_idem_2",
        content="",
        finish_reason="tool_calls",
        tool_calls=[ToolCallPayload(id="c2", name="search_company", arguments={"query": "Stripe"})],
    )
    turn3 = LLMResponse(
        request_id="r_idem_3",
        content="Stripe enterprise payments summary.",
        finish_reason="stop",
    )

    search_mock = AsyncMock(return_value=[])
    with patch("app.agent.runner.call_llm", side_effect=[turn1, turn2, turn3]):
        with patch("app.agent.tools.definitions.search_company", new=search_mock):
            res1 = await runner.run(goal="First search", session_id=sess.id, db=db_session)

    # Underlying search_company called only ONCE despite two identical tool invocations
    assert search_mock.await_count == 1
    assert res1.is_completed is True

    # Follow-up run inherits sess.context and prior completed Task summary
    followup_mock = AsyncMock(
        return_value=LLMResponse(
            request_id="r_follow", content="Follow-up done.", finish_reason="stop"
        )
    )
    with patch("app.agent.runner.call_llm", new=followup_mock):
        await runner.run(goal="Follow up", session_id=sess.id, db=db_session)

    prompt = followup_mock.call_args.args[0].messages[1]["content"]
    assert "Enterprise fintech priority" in prompt
    assert "PRIOR CONVERSATIONAL TASKS" in prompt
    assert "Stripe enterprise payments summary." in prompt
