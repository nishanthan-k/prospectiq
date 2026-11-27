from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.models import AgentRunRequest, AgentRunResponse
from app.agent.service import run_sales_agent
from app.db import SessionRepository, TaskRepository, get_db

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionCreateRequest(BaseModel):
    """Payload to create a persistent prospect research session."""

    title: str = Field(..., min_length=1, max_length=255, description="Session title")
    company_name: Optional[str] = Field(default=None, max_length=255)
    domain: Optional[str] = Field(default=None, max_length=255)
    context: Optional[dict[str, Any]] = Field(
        default=None, description="CRM notes or custom ICP criteria"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Arbitrary session tags and configuration"
    )

    @field_validator("title")
    @classmethod
    def validate_title_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Title cannot be empty or only whitespace")
        return v.strip()

    @field_validator("company_name", "domain")
    @classmethod
    def normalize_optional_strings(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip()
        return cleaned if cleaned else None


class SessionMemoryResponse(BaseModel):
    """Grounded working memory record with source provenance."""

    id: str
    session_id: str
    memory_type: str
    key: str
    value: Any
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    confidence: float
    is_verified: bool
    extracted_at: datetime
    updated_at: datetime


class TaskSummaryResponse(BaseModel):
    """Compact summary of a task executed within a session."""

    id: str
    session_id: str
    goal: str
    status: str
    steps_taken: int
    total_tokens: int
    latency_ms: float
    created_at: datetime
    completed_at: Optional[datetime] = None


class SessionResponse(BaseModel):
    """Persistent research session metadata."""

    id: str
    title: str
    company_name: Optional[str] = None
    domain: Optional[str] = None
    context: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SessionDetailResponse(SessionResponse):
    """Session details including active memories and task history."""

    memories: list[SessionMemoryResponse] = Field(default_factory=list)
    tasks: list[TaskSummaryResponse] = Field(default_factory=list)


def _to_session_response(sess: Any) -> SessionResponse:
    return SessionResponse(
        id=sess.id,
        title=sess.title,
        company_name=sess.company_name,
        domain=sess.domain,
        context=sess.context,
        metadata=sess.metadata_,
        created_at=sess.created_at,
        updated_at=sess.updated_at,
    )


def _to_memory_response(mem: Any) -> SessionMemoryResponse:
    return SessionMemoryResponse(
        id=mem.id,
        session_id=mem.session_id,
        memory_type=mem.memory_type,
        key=mem.key,
        value=mem.value,
        source_url=mem.source_url,
        source_quote=mem.source_quote,
        confidence=mem.confidence,
        is_verified=mem.is_verified,
        extracted_at=mem.extracted_at,
        updated_at=mem.updated_at,
    )


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    request: SessionCreateRequest, db: AsyncSession = Depends(get_db)
) -> SessionResponse:
    """Create a new persistent prospect research session."""
    repo = SessionRepository(db)
    sess = await repo.create_session(
        title=request.title,
        company_name=request.company_name,
        domain=request.domain,
        context=request.context,
        metadata=request.metadata,
    )
    return _to_session_response(sess)


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[SessionResponse]:
    """List active research sessions ordered by most recently updated."""
    repo = SessionRepository(db)
    sessions = await repo.list_sessions(limit=limit, offset=offset)
    return [_to_session_response(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> SessionDetailResponse:
    """Fetch session details along with grounded memories and task executions."""
    s_repo = SessionRepository(db)
    sess = await s_repo.get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    t_repo = TaskRepository(db)
    memories = await s_repo.get_memories(session_id)
    tasks = await t_repo.list_tasks_for_session(session_id)

    base = _to_session_response(sess)
    return SessionDetailResponse(
        **base.model_dump(),
        memories=[_to_memory_response(m) for m in memories],
        tasks=[
            TaskSummaryResponse(
                id=t.id,
                session_id=t.session_id,
                goal=t.goal,
                status=t.status,
                steps_taken=t.steps_taken,
                total_tokens=t.total_tokens,
                latency_ms=t.latency_ms,
                created_at=t.created_at,
                completed_at=t.completed_at,
            )
            for t in tasks
        ],
    )


@router.get("/{session_id}/memories", response_model=list[SessionMemoryResponse])
async def get_session_memories(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> list[SessionMemoryResponse]:
    """Retrieve all extracted working memories with provenance for a session."""
    s_repo = SessionRepository(db)
    sess = await s_repo.get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    memories = await s_repo.get_memories(session_id)
    return [_to_memory_response(m) for m in memories]


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    """Soft-delete a research session."""
    s_repo = SessionRepository(db)
    deleted = await s_repo.soft_delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/{session_id}/agent/run", response_model=AgentRunResponse)
async def run_agent_in_session(
    session_id: str,
    request: AgentRunRequest,
    db: AsyncSession = Depends(get_db),
) -> AgentRunResponse:
    """Execute an autonomous sales agent goal scoped to an existing session."""
    s_repo = SessionRepository(db)
    sess = await s_repo.get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    scoped_request = request.model_copy(update={"session_id": session_id})
    try:
        return await run_sales_agent(scoped_request, db=db)
    except ValueError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=400, detail=str(exc)) from exc
