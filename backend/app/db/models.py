import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class Session(Base):
    """Persistent conversational research session for a target prospect."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: _new_id("sess"))
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    domain: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, default=dict, nullable=False
    )
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False
    )

    tasks: Mapped[list["Task"]] = relationship(
        "Task", back_populates="session", cascade="all, delete-orphan"
    )
    memories: Mapped[list["SessionMemory"]] = relationship(
        "SessionMemory", back_populates="session", cascade="all, delete-orphan"
    )


class Task(Base):
    """Persistent agent execution task with worker lease and lifecycle tracking."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: _new_id("task"))
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False, index=True)
    locked_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    max_steps: Mapped[int] = mapped_column(Integer, default=6, nullable=False)
    steps_taken: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    final_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped["Session"] = relationship("Session", back_populates="tasks")
    steps: Mapped[list["TaskStep"]] = relationship(
        "TaskStep",
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="TaskStep.step_index",
    )
    tool_executions: Mapped[list["ToolExecution"]] = relationship(
        "ToolExecution", back_populates="task", cascade="all, delete-orphan"
    )


class TaskStep(Base):
    """Per-turn reasoning checkpoint storing exact OpenAI/Gemini protocol messages."""

    __tablename__ = "task_steps"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: _new_id("step"))
    task_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    assistant_message: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    tool_messages: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )

    task: Mapped["Task"] = relationship("Task", back_populates="steps")
    tool_executions: Mapped[list["ToolExecution"]] = relationship(
        "ToolExecution", back_populates="step", cascade="all, delete-orphan"
    )


class ToolExecution(Base):
    """Granular tool execution audit log with idempotency key and state machine."""

    __tablename__ = "tool_executions"
    __table_args__ = (Index("ix_tool_executions_idempotency_key", "idempotency_key"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: _new_id("tcall"))
    task_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    step_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("task_steps.id", ondelete="CASCADE"), nullable=True, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", nullable=False)
    is_idempotent: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    input_arguments: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    output_payload: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    permission_level: Mapped[str] = mapped_column(String(32), default="read_only", nullable=False)
    approval_status: Mapped[str] = mapped_column(String(32), default="NONE", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    task: Mapped["Task"] = relationship("Task", back_populates="tool_executions")
    step: Mapped[Optional["TaskStep"]] = relationship("TaskStep", back_populates="tool_executions")


class SessionMemory(Base):
    """Grounded working memory fact with source provenance and confidence."""

    __tablename__ = "session_memories"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "memory_type",
            "key",
            name="uq_session_memories_session_type_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: _new_id("mem"))
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    memory_type: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[Any] = mapped_column(JSON, default=dict, nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    source_quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False
    )

    session: Mapped["Session"] = relationship("Session", back_populates="memories")
