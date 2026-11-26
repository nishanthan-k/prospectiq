from collections.abc import AsyncIterator
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.core.logger import log

settings = get_settings()


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 async ORM models."""


_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def create_db_engine(
    database_url: Optional[str] = None,
    echo: Optional[bool] = None,
) -> AsyncEngine:
    """
    Create an async SQLAlchemy engine supporting both PostgreSQL (asyncpg)
    and SQLite (aiosqlite).
    """
    url = database_url or settings.database_url
    sql_echo = echo if echo is not None else settings.db_echo

    kwargs: dict[str, Any] = {"echo": sql_echo}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10

    return create_async_engine(url, **kwargs)


def get_engine() -> AsyncEngine:
    """Return the singleton AsyncEngine instance, creating it if needed."""
    global _engine
    if _engine is None:
        _engine = create_db_engine()
        log.info("Initialized async database engine")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the singleton async_sessionmaker bound to the active engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a scoped AsyncSession with commit/rollback handling."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_db_connection(engine: Optional[AsyncEngine] = None) -> bool:
    """Verify database connectivity by executing a lightweight SELECT 1 query."""
    target_engine = engine or get_engine()
    async with target_engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True


async def close_db() -> None:
    """Dispose the singleton AsyncEngine and reset session factory."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        log.info("Disposed async database engine")
