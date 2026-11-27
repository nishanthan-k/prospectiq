import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import (
    Base,
    Session,
    SessionMemory,
    SessionRepository,
    Task,
    TaskRepository,
    TaskStep,
    ToolExecution,
    create_db_engine,
)


@pytest_asyncio.fixture
async def db_session():
    """Provide an isolated in-memory async SQLite session with all Level 4 tables."""
    engine = create_db_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_session_repository_crud_and_soft_delete(db_session: AsyncSession):
    """SessionRepository creates, lists, retrieves, and soft-deletes sessions."""
    repo = SessionRepository(db_session)

    s1 = await repo.create_session(
        title="Research Retool",
        company_name="Retool",
        domain="retool.com",
        context={"icp": ["B2B SaaS"]},
        metadata={"source": "inbound"},
    )
    assert s1.id.startswith("sess_")
    assert s1.company_name == "Retool"
    assert s1.metadata_ == {"source": "inbound"}

    fetched = await repo.get_session(s1.id)
    assert fetched is not None
    assert fetched.title == "Research Retool"

    listed = await repo.list_sessions()
    assert len(listed) == 1

    # Soft delete hides session from default queries
    deleted = await repo.soft_delete_session(s1.id)
    assert deleted is True
    assert await repo.get_session(s1.id) is None
    assert await repo.list_sessions() == []
    assert await repo.get_session(s1.id, include_deleted=True) is not None
    assert await repo.soft_delete_session("sess_missing") is False


@pytest.mark.asyncio
async def test_session_memory_upsert_provenance_and_confidence(db_session: AsyncSession):
    """SessionMemory upserts respect confidence/verification upgrades and provenance."""
    repo = SessionRepository(db_session)
    sess = await repo.create_session(title="Stripe Research", company_name="Stripe")

    mem1 = await repo.upsert_memory(
        session_id=sess.id,
        memory_type="fact",
        key="company.offerings",
        value={"summary": "Online payments API"},
        source_url="https://stripe.com",
        source_quote="Financial infrastructure for the internet.",
        confidence=0.80,
        is_verified=True,
    )
    assert mem1.id.startswith("mem_")
    assert mem1.confidence == 0.80

    # Lower confidence update does not overwrite higher-confidence verified fact
    mem_lower = await repo.upsert_memory(
        session_id=sess.id,
        memory_type="fact",
        key="company.offerings",
        value={"summary": "Unverified blog guess"},
        source_url="https://blog.example.com",
        confidence=0.40,
        is_verified=False,
    )
    assert mem_lower.value == {"summary": "Online payments API"}
    assert mem_lower.confidence == 0.80

    # Unverified observation with equal/higher confidence does NOT overwrite verified fact
    mem_unverified = await repo.upsert_memory(
        session_id=sess.id,
        memory_type="fact",
        key="company.offerings",
        value={"summary": "Unverified equal-confidence guess"},
        source_url="https://blog.example.com",
        confidence=0.80,
        is_verified=False,
    )
    assert mem_unverified.value == {"summary": "Online payments API"}
    assert mem_unverified.is_verified is True

    # Higher confidence verified update overwrites cleanly
    mem_higher = await repo.upsert_memory(
        session_id=sess.id,
        memory_type="fact",
        key="company.offerings",
        value={"summary": "Payments, Billing, Connect, and Treasury"},
        source_url="https://stripe.com/payments",
        source_quote="Accept payments and manage revenue.",
        confidence=0.95,
        is_verified=True,
    )
    assert mem_higher.id == mem1.id
    assert mem_higher.confidence == 0.95
    assert mem_higher.value["summary"] == "Payments, Billing, Connect, and Treasury"

    memories = await repo.get_memories(sess.id, memory_type="fact")
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_task_repository_lifecycle_steps_and_leases(db_session: AsyncSession):
    """TaskRepository manages task leases, heartbeats, steps, and completion."""
    s_repo = SessionRepository(db_session)
    t_repo = TaskRepository(db_session)

    sess = await s_repo.create_session(title="Snowflake", company_name="Snowflake")
    task = await t_repo.create_task(session_id=sess.id, goal="Evaluate ICP fit", max_steps=4)
    assert task.id.startswith("task_")
    assert task.status == "PENDING"
    assert task.lease_expires_at is None

    # Mark running with worker lease
    running = await t_repo.mark_running(task.id, locked_by="worker-1", lease_seconds=60)
    assert running is not None
    assert running.status == "RUNNING"
    assert running.locked_by == "worker-1"
    assert running.lease_expires_at is not None
    assert running.heartbeat_at is not None

    refreshed = await t_repo.refresh_heartbeat(task.id, locked_by="worker-1", lease_seconds=120)
    assert refreshed is not None
    assert refreshed.lease_expires_at is not None

    # Record TaskStep with exact OpenAI/Gemini protocol messages
    step = await t_repo.record_step(
        task_id=task.id,
        step_index=0,
        assistant_message={
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        },
        prompt_tokens=120,
        completion_tokens=30,
        latency_ms=450.5,
    )
    assert step.id.startswith("step_")

    await t_repo.update_step_tool_messages(
        step.id,
        [{"role": "tool", "tool_call_id": "call_1", "name": "search_company", "content": "[]"}],
    )
    steps = await t_repo.get_steps(task.id)
    assert len(steps) == 1
    assert steps[0].tool_messages[0]["tool_call_id"] == "call_1"

    completed = await t_repo.mark_completed(
        task.id,
        final_answer="Strong ICP fit.",
        steps_taken=2,
        total_tokens=400,
        latency_ms=1200.0,
    )
    assert completed is not None
    assert completed.status == "COMPLETED"
    assert completed.lease_expires_at is None
    assert completed.completed_at is not None


@pytest.mark.asyncio
async def test_tool_execution_idempotency_and_unknown_state(db_session: AsyncSession):
    """ToolExecution tracks deterministic idempotency keys and COMPLETED/FAILED/UNKNOWN states."""
    s_repo = SessionRepository(db_session)
    t_repo = TaskRepository(db_session)

    sess = await s_repo.create_session(title="Databricks")
    task = await t_repo.create_task(session_id=sess.id, goal="Fetch website")
    step = await t_repo.record_step(
        task_id=task.id, step_index=0, assistant_message={"role": "assistant"}
    )

    # Idempotency key is deterministic regardless of dict key order
    k1 = t_repo.compute_idempotency_key(task.id, "fetch_webpage", {"a": 1, "b": 2})
    k2 = t_repo.compute_idempotency_key(task.id, "fetch_webpage", {"b": 2, "a": 1})
    assert k1 == k2

    exec_ok = await t_repo.start_tool_execution(
        task_id=task.id,
        step_id=step.id,
        tool_name="fetch_webpage",
        tool_call_id="call_ok",
        input_arguments={"url": "https://databricks.com"},
    )
    assert exec_ok.status == "RUNNING"
    await t_repo.complete_tool_execution(
        exec_ok.id, output_payload={"status": "success"}, latency_ms=210.0
    )

    cached = await t_repo.find_completed_by_idempotency_key(exec_ok.idempotency_key)
    assert cached is not None
    assert cached.id == exec_ok.id

    # Unhandled crash transitions to UNKNOWN
    exec_unk = await t_repo.start_tool_execution(
        task_id=task.id,
        step_id=step.id,
        tool_name="write_crm",
        tool_call_id="call_unk",
        input_arguments={"id": 42},
        is_idempotent=False,
        permission_level="write_action",
    )
    await t_repo.fail_tool_execution(
        exec_unk.id, error_message="Worker crashed mid-flight", status="UNKNOWN"
    )

    with pytest.raises(ValueError, match="Invalid failure status"):
        await t_repo.fail_tool_execution(exec_unk.id, error_message="bad", status="BOGUS")

    all_execs = await t_repo.get_tool_executions(task.id)
    assert len(all_execs) == 2
    assert all_execs[0].status == "COMPLETED"
    assert all_execs[1].status == "UNKNOWN"


@pytest.mark.asyncio
async def test_session_cascade_delete_removes_children(db_session: AsyncSession):
    """Deleting a Session cascades to Tasks, TaskSteps, ToolExecutions, and SessionMemories."""
    s_repo = SessionRepository(db_session)
    t_repo = TaskRepository(db_session)

    sess = await s_repo.create_session(title="Cascade Test")
    await s_repo.upsert_memory(sess.id, "fact", "k", {"v": 1}, confidence=0.9)
    task = await t_repo.create_task(sess.id, "Run")
    step = await t_repo.record_step(task.id, 0, {"role": "assistant"})
    await t_repo.start_tool_execution(task.id, step.id, "search_company", "c1", {"query": "x"})
    await db_session.flush()

    await db_session.delete(sess)
    await db_session.flush()

    assert (await db_session.execute(select(Session))).scalars().all() == []
    assert (await db_session.execute(select(SessionMemory))).scalars().all() == []
    assert (await db_session.execute(select(Task))).scalars().all() == []
    assert (await db_session.execute(select(TaskStep))).scalars().all() == []
    assert (await db_session.execute(select(ToolExecution))).scalars().all() == []


@pytest.mark.asyncio
async def test_session_normalizes_blank_fields_and_task_lease_guards(db_session: AsyncSession):
    """Whitespace-only company_name/domain normalize to None; mark_running enforces lease/terminal guards."""
    s_repo = SessionRepository(db_session)
    t_repo = TaskRepository(db_session)

    sess = await s_repo.create_session(title="  Clean Test  ", company_name="   ", domain="  ")
    assert sess.title == "Clean Test"
    assert sess.company_name is None
    assert sess.domain is None

    task = await t_repo.create_task(sess.id, "Lease guard test")
    await t_repo.mark_running(task.id, locked_by="worker-1", lease_seconds=60)

    # Another worker cannot steal an active unexpired lease
    with pytest.raises(ValueError, match="holds an active lease"):
        await t_repo.mark_running(task.id, locked_by="worker-2", lease_seconds=60)

    # Completed task cannot be restarted via mark_running
    await t_repo.mark_completed(
        task.id, final_answer="Done", steps_taken=1, total_tokens=10, latency_ms=5.0
    )
    with pytest.raises(ValueError, match="already in terminal state"):
        await t_repo.mark_running(task.id, locked_by="worker-1", lease_seconds=60)
