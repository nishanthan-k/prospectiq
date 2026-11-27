import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Task, TaskStep, ToolExecution


class TaskRepository:
    """Repository for task lifecycle, step checkpointing, and idempotent tool execution."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    @staticmethod
    def compute_idempotency_key(task_id: str, tool_name: str, args: dict[str, Any]) -> str:
        """Compute deterministic SHA-256 idempotency key for a tool invocation."""
        canonical_args = json.dumps(args, sort_keys=True, separators=(",", ":"))
        raw = f"{task_id}:{tool_name}:{canonical_args}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    async def create_task(
        self,
        session_id: str,
        goal: str,
        max_steps: int = 6,
        locked_by: Optional[str] = None,
        lease_seconds: int = 60,
    ) -> Task:
        """Create a new task in PENDING state with optional initial worker lease."""
        now = datetime.now(timezone.utc)
        lease_expires = now + timedelta(seconds=lease_seconds) if locked_by else None
        task = Task(
            session_id=session_id,
            goal=goal,
            status="PENDING",
            max_steps=max_steps,
            locked_by=locked_by,
            lease_expires_at=lease_expires,
            heartbeat_at=now if locked_by else None,
        )
        self.db.add(task)
        await self.db.flush()
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Fetch a task by ID."""
        stmt = select(Task).where(Task.id == task_id)
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def list_tasks_for_session(self, session_id: str) -> list[Task]:
        """List all tasks belonging to a session in chronological order."""
        stmt = select(Task).where(Task.session_id == session_id).order_by(Task.created_at.asc())
        return list((await self.db.execute(stmt)).scalars().all())

    async def mark_running(
        self, task_id: str, locked_by: str, lease_seconds: int = 60
    ) -> Optional[Task]:
        """Transition a task to RUNNING and acquire/refresh its worker lease."""
        task = await self.get_task(task_id)
        if task is None:
            return None
        if task.status in ("COMPLETED", "FAILED"):
            raise ValueError(f"Task '{task_id}' is already in terminal state '{task.status}'")

        now = datetime.now(timezone.utc)
        if (
            task.status == "RUNNING"
            and task.locked_by
            and task.locked_by != locked_by
            and task.lease_expires_at is not None
        ):
            lease_exp = (
                task.lease_expires_at.replace(tzinfo=timezone.utc)
                if task.lease_expires_at.tzinfo is None
                else task.lease_expires_at
            )
            if lease_exp > now:
                raise ValueError(
                    f"Task '{task_id}' holds an active lease owned by '{task.locked_by}'"
                )

        task.status = "RUNNING"
        task.locked_by = locked_by
        task.heartbeat_at = now
        task.lease_expires_at = now + timedelta(seconds=lease_seconds)
        task.updated_at = now
        await self.db.flush()
        return task

    async def refresh_heartbeat(
        self, task_id: str, locked_by: str, lease_seconds: int = 60
    ) -> Optional[Task]:
        """Extend the worker lease and update heartbeat timestamp."""
        task = await self.get_task(task_id)
        if task is None:
            return None
        now = datetime.now(timezone.utc)
        task.locked_by = locked_by
        task.heartbeat_at = now
        task.lease_expires_at = now + timedelta(seconds=lease_seconds)
        task.updated_at = now
        await self.db.flush()
        return task

    async def mark_completed(
        self,
        task_id: str,
        final_answer: str,
        steps_taken: int,
        total_tokens: int,
        latency_ms: float,
    ) -> Optional[Task]:
        """Transition a task to COMPLETED with final synthesis metrics."""
        task = await self.get_task(task_id)
        if task is None:
            return None
        now = datetime.now(timezone.utc)
        task.status = "COMPLETED"
        task.final_answer = final_answer
        task.steps_taken = steps_taken
        task.total_tokens = total_tokens
        task.latency_ms = latency_ms
        task.completed_at = now
        task.updated_at = now
        task.lease_expires_at = None
        await self.db.flush()
        return task

    async def mark_failed(
        self,
        task_id: str,
        error_message: str,
        steps_taken: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> Optional[Task]:
        """Transition a task to FAILED with diagnostic error message."""
        task = await self.get_task(task_id)
        if task is None:
            return None
        now = datetime.now(timezone.utc)
        task.status = "FAILED"
        task.error_message = error_message
        task.steps_taken = steps_taken
        task.total_tokens = total_tokens
        task.latency_ms = latency_ms
        task.completed_at = now
        task.updated_at = now
        task.lease_expires_at = None
        await self.db.flush()
        return task

    async def record_step(
        self,
        task_id: str,
        step_index: int,
        assistant_message: dict[str, Any],
        tool_messages: Optional[list[dict[str, Any]]] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> TaskStep:
        """Persist a TaskStep checkpoint with exact LLM/tool protocol messages."""
        step = TaskStep(
            task_id=task_id,
            step_index=step_index,
            assistant_message=assistant_message,
            tool_messages=tool_messages or [],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        self.db.add(step)
        await self.db.flush()
        return step

    async def update_step_tool_messages(
        self, step_id: str, tool_messages: list[dict[str, Any]]
    ) -> Optional[TaskStep]:
        """Update the 1:1 tool response messages on an existing TaskStep."""
        stmt = select(TaskStep).where(TaskStep.id == step_id)
        step = (await self.db.execute(stmt)).scalar_one_or_none()
        if step is None:
            return None
        step.tool_messages = tool_messages
        await self.db.flush()
        return step

    async def get_steps(self, task_id: str) -> list[TaskStep]:
        """Fetch all reasoning steps for a task ordered by step_index."""
        stmt = (
            select(TaskStep).where(TaskStep.task_id == task_id).order_by(TaskStep.step_index.asc())
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def start_tool_execution(
        self,
        task_id: str,
        step_id: Optional[str],
        tool_name: str,
        tool_call_id: str,
        input_arguments: dict[str, Any],
        is_idempotent: bool = True,
        permission_level: str = "read_only",
        approval_status: str = "NONE",
    ) -> ToolExecution:
        """Insert a ToolExecution record in RUNNING state before tool dispatch."""
        idem_key = self.compute_idempotency_key(task_id, tool_name, input_arguments)
        execution = ToolExecution(
            task_id=task_id,
            step_id=step_id,
            idempotency_key=idem_key,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status="RUNNING",
            is_idempotent=is_idempotent,
            input_arguments=input_arguments,
            permission_level=permission_level,
            approval_status=approval_status,
        )
        self.db.add(execution)
        await self.db.flush()
        return execution

    async def complete_tool_execution(
        self, execution_id: str, output_payload: Any, latency_ms: float
    ) -> Optional[ToolExecution]:
        """Mark a ToolExecution as COMPLETED with its structured output."""
        stmt = select(ToolExecution).where(ToolExecution.id == execution_id)
        execution = (await self.db.execute(stmt)).scalar_one_or_none()
        if execution is None:
            return None
        execution.status = "COMPLETED"
        execution.output_payload = output_payload
        execution.latency_ms = latency_ms
        execution.completed_at = datetime.now(timezone.utc)
        await self.db.flush()
        return execution

    async def fail_tool_execution(
        self,
        execution_id: str,
        error_message: str,
        latency_ms: float = 0.0,
        status: str = "FAILED",
    ) -> Optional[ToolExecution]:
        """Mark a ToolExecution as FAILED or UNKNOWN on error/crash."""
        if status not in ("FAILED", "UNKNOWN"):
            raise ValueError(f"Invalid failure status '{status}'; expected FAILED or UNKNOWN")
        stmt = select(ToolExecution).where(ToolExecution.id == execution_id)
        execution = (await self.db.execute(stmt)).scalar_one_or_none()
        if execution is None:
            return None
        execution.status = status
        execution.error_message = error_message
        execution.latency_ms = latency_ms
        execution.completed_at = datetime.now(timezone.utc)
        await self.db.flush()
        return execution

    async def find_completed_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[ToolExecution]:
        """Look up a previously COMPLETED ToolExecution matching an idempotency key."""
        stmt = (
            select(ToolExecution)
            .where(
                ToolExecution.idempotency_key == idempotency_key,
                ToolExecution.status == "COMPLETED",
            )
            .order_by(ToolExecution.created_at.desc())
            .limit(1)
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def find_unknown_non_idempotent_by_key(
        self, idempotency_key: str
    ) -> Optional[ToolExecution]:
        """Look up a prior UNKNOWN non-idempotent ToolExecution matching an idempotency key."""
        stmt = (
            select(ToolExecution)
            .where(
                ToolExecution.idempotency_key == idempotency_key,
                ToolExecution.status == "UNKNOWN",
                ToolExecution.is_idempotent.is_(False),
            )
            .order_by(ToolExecution.created_at.desc())
            .limit(1)
        )
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def get_tool_executions(self, task_id: str) -> list[ToolExecution]:
        """Fetch all tool execution audit records for a task."""
        stmt = (
            select(ToolExecution)
            .where(ToolExecution.task_id == task_id)
            .order_by(ToolExecution.created_at.asc())
        )
        return list((await self.db.execute(stmt)).scalars().all())
