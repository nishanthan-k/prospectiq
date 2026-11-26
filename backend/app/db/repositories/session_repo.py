from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session, SessionMemory


class SessionRepository:
    """Repository for managing persistent research sessions and grounded memories."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_session(
        self,
        title: str,
        company_name: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Session:
        """Create and persist a new research session."""
        sess = Session(
            title=title.strip(),
            company_name=company_name.strip() if company_name else None,
            domain=domain.strip() if domain else None,
            context=context or {},
            metadata_=metadata or {},
        )
        self.db.add(sess)
        await self.db.flush()
        return sess

    async def get_session(
        self, session_id: str, include_deleted: bool = False
    ) -> Optional[Session]:
        """Retrieve a session by ID, excluding soft-deleted sessions by default."""
        stmt = select(Session).where(Session.id == session_id)
        if not include_deleted:
            stmt = stmt.where(Session.is_deleted.is_(False))
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_sessions(
        self, limit: int = 20, offset: int = 0, include_deleted: bool = False
    ) -> list[Session]:
        """List sessions ordered by most recently updated."""
        stmt = select(Session)
        if not include_deleted:
            stmt = stmt.where(Session.is_deleted.is_(False))
        stmt = stmt.order_by(Session.updated_at.desc()).limit(limit).offset(offset)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def soft_delete_session(self, session_id: str) -> bool:
        """Soft-delete a session. Returns True if found and marked deleted."""
        sess = await self.get_session(session_id, include_deleted=False)
        if sess is None:
            return False
        now = datetime.now(timezone.utc)
        sess.is_deleted = True
        sess.deleted_at = now
        sess.updated_at = now
        await self.db.flush()
        return True

    async def upsert_memory(
        self,
        session_id: str,
        memory_type: str,
        key: str,
        value: Any,
        source_url: Optional[str] = None,
        source_quote: Optional[str] = None,
        confidence: float = 0.0,
        is_verified: bool = False,
        extracted_at: Optional[datetime] = None,
    ) -> SessionMemory:
        """
        Insert or update a grounded SessionMemory fact keyed by (session_id, memory_type, key).
        Updates existing memory if new confidence >= existing confidence or if it is verified.
        """
        stmt = select(SessionMemory).where(
            SessionMemory.session_id == session_id,
            SessionMemory.memory_type == memory_type,
            SessionMemory.key == key,
        )
        existing = (await self.db.execute(stmt)).scalar_one_or_none()
        now = datetime.now(timezone.utc)
        obs_time = extracted_at or now

        if existing is not None:
            if confidence >= existing.confidence or (is_verified and not existing.is_verified):
                existing.value = value
                existing.source_url = source_url
                existing.source_quote = source_quote
                existing.confidence = confidence
                existing.is_verified = is_verified
                existing.extracted_at = obs_time
                existing.updated_at = now
                await self.db.flush()
            return existing

        mem = SessionMemory(
            session_id=session_id,
            memory_type=memory_type,
            key=key,
            value=value,
            source_url=source_url,
            source_quote=source_quote,
            confidence=confidence,
            is_verified=is_verified,
            extracted_at=obs_time,
        )
        self.db.add(mem)
        await self.db.flush()
        return mem

    async def get_memories(
        self, session_id: str, memory_type: Optional[str] = None
    ) -> list[SessionMemory]:
        """Retrieve all working memories for a session, ordered by confidence and recency."""
        stmt = select(SessionMemory).where(SessionMemory.session_id == session_id)
        if memory_type is not None:
            stmt = stmt.where(SessionMemory.memory_type == memory_type)
        stmt = stmt.order_by(SessionMemory.confidence.desc(), SessionMemory.extracted_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
