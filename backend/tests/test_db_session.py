from typing import Any

import pytest
from sqlalchemy import JSON, String, select
from sqlalchemy.orm import Mapped, mapped_column

from app.db import (
    Base,
    check_db_connection,
    close_db,
    create_db_engine,
    get_db,
    get_engine,
    get_session_factory,
)
from app.db import session as db_session_module


class _DummyRecord(Base):
    __tablename__ = "_test_dummy_records"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


@pytest.mark.asyncio
async def test_create_db_engine_sqlite_and_postgres_dialects():
    """create_db_engine configures both SQLite aiosqlite and PostgreSQL asyncpg engines."""
    sqlite_engine = create_db_engine("sqlite+aiosqlite:///:memory:")
    try:
        assert sqlite_engine.dialect.name == "sqlite"
        assert await check_db_connection(sqlite_engine) is True
    finally:
        await sqlite_engine.dispose()

    pg_engine = create_db_engine("postgresql+asyncpg://user:pass@localhost:5432/prospectiq")
    try:
        assert pg_engine.dialect.name == "postgresql"
    finally:
        await pg_engine.dispose()


@pytest.mark.asyncio
async def test_get_db_commits_and_serializes_json(monkeypatch):
    """get_db commits valid transactions and round-trips portable JSON payloads."""
    await close_db()
    monkeypatch.setattr(db_session_module.settings, "database_url", "sqlite+aiosqlite:///:memory:")

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sample_payload = {"company": "Stripe", "tags": ["fintech", "payments"], "score": 0.95}

    gen = get_db()
    session = await anext(gen)
    session.add(_DummyRecord(id="rec_1", payload=sample_payload))
    with pytest.raises(StopAsyncIteration):
        await anext(gen)

    # Verify committed in a fresh session
    factory = get_session_factory()
    async with factory() as verify_session:
        row = (
            await verify_session.execute(select(_DummyRecord).where(_DummyRecord.id == "rec_1"))
        ).scalar_one()
        assert row.payload == sample_payload

    await close_db()


@pytest.mark.asyncio
async def test_get_db_rolls_back_on_exception(monkeypatch):
    """get_db rolls back uncommitted changes when an exception is raised."""
    await close_db()
    monkeypatch.setattr(db_session_module.settings, "database_url", "sqlite+aiosqlite:///:memory:")

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    gen = get_db()
    session = await anext(gen)
    session.add(_DummyRecord(id="rec_rollback", payload={"status": "uncommitted"}))
    with pytest.raises(RuntimeError, match="Simulated failure"):
        await gen.athrow(RuntimeError("Simulated failure"))

    # Verify row was rolled back
    factory = get_session_factory()
    async with factory() as verify_session:
        row = (
            await verify_session.execute(
                select(_DummyRecord).where(_DummyRecord.id == "rec_rollback")
            )
        ).scalar_one_or_none()
        assert row is None

    await close_db()
