from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import SessionRepository, TaskRepository, get_db

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskResponse(BaseModel):
    """Full persistent task status, worker lease, and execution metrics."""

    id: str
    session_id: str
    goal: str
    status: str
    locked_by: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    max_steps: int
    steps_taken: int
    final_answer: Optional[str] = None
    error_message: Optional[str] = None
    total_tokens: int
    latency_ms: float
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


class TaskStepResponse(BaseModel):
    """Per-turn reasoning checkpoint with exact OpenAI/Gemini protocol payloads."""

    id: str
    task_id: str
    step_index: int
    assistant_message: dict[str, Any]
    tool_messages: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    created_at: datetime


class ToolExecutionResponse(BaseModel):
    """Granular tool execution audit record with idempotency key and status."""

    id: str
    task_id: str
    step_id: Optional[str] = None
    idempotency_key: str
    tool_name: str
    tool_call_id: str
    status: str
    is_idempotent: bool
    input_arguments: dict[str, Any]
    output_payload: Optional[Any] = None
    error_message: Optional[str] = None
    latency_ms: float
    permission_level: str
    approval_status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


async def _get_active_task_or_404(task_id: str, db: AsyncSession):
    t_repo = TaskRepository(db)
    task = await t_repo.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    s_repo = SessionRepository(db)
    if await s_repo.get_session(task.session_id, include_deleted=False) is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task, t_repo


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)) -> TaskResponse:
    """Fetch task status, worker lease info, token consumption, and final answer."""
    task, _ = await _get_active_task_or_404(task_id, db)

    return TaskResponse(
        id=task.id,
        session_id=task.session_id,
        goal=task.goal,
        status=task.status,
        locked_by=task.locked_by,
        lease_expires_at=task.lease_expires_at,
        heartbeat_at=task.heartbeat_at,
        max_steps=task.max_steps,
        steps_taken=task.steps_taken,
        final_answer=task.final_answer,
        error_message=task.error_message,
        total_tokens=task.total_tokens,
        latency_ms=task.latency_ms,
        created_at=task.created_at,
        updated_at=task.updated_at,
        completed_at=task.completed_at,
    )


@router.get("/{task_id}/steps", response_model=list[TaskStepResponse])
async def get_task_steps(
    task_id: str, db: AsyncSession = Depends(get_db)
) -> list[TaskStepResponse]:
    """Fetch full reasoning steps and raw message protocol payloads for a task."""
    _, repo = await _get_active_task_or_404(task_id, db)

    steps = await repo.get_steps(task_id)
    return [
        TaskStepResponse(
            id=s.id,
            task_id=s.task_id,
            step_index=s.step_index,
            assistant_message=s.assistant_message,
            tool_messages=s.tool_messages,
            prompt_tokens=s.prompt_tokens,
            completion_tokens=s.completion_tokens,
            latency_ms=s.latency_ms,
            created_at=s.created_at,
        )
        for s in steps
    ]


@router.get("/{task_id}/tools", response_model=list[ToolExecutionResponse])
async def get_task_tools(
    task_id: str, db: AsyncSession = Depends(get_db)
) -> list[ToolExecutionResponse]:
    """Fetch granular tool execution audit logs with idempotency keys and latencies."""
    _, repo = await _get_active_task_or_404(task_id, db)

    execs = await repo.get_tool_executions(task_id)
    return [
        ToolExecutionResponse(
            id=e.id,
            task_id=e.task_id,
            step_id=e.step_id,
            idempotency_key=e.idempotency_key,
            tool_name=e.tool_name,
            tool_call_id=e.tool_call_id,
            status=e.status,
            is_idempotent=e.is_idempotent,
            input_arguments=e.input_arguments,
            output_payload=e.output_payload,
            error_message=e.error_message,
            latency_ms=e.latency_ms,
            permission_level=e.permission_level,
            approval_status=e.approval_status,
            created_at=e.created_at,
            completed_at=e.completed_at,
        )
        for e in execs
    ]
