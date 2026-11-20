from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from app.research.models import EvidenceCitation


class ToolCallRecord(BaseModel):
    """Audit trace of a single tool execution within the agent loop."""

    step: int = Field(..., description="Step index (0-based) when this tool was invoked")
    tool_name: str = Field(..., description="Name of the executed tool")
    tool_call_id: str = Field(..., description="Correlated tool_call_id from LLM response")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters passed to the tool"
    )
    output: Any = Field(..., description="Structured output payload or error message")
    is_error: bool = Field(default=False, description="True if tool failed or raised an error")
    latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds")


class AgentRunResult(BaseModel):
    """Internal result produced by AgentRunner execution."""

    final_answer: str = Field(..., description="Final synthesized response addressing the goal")
    steps_taken: int = Field(..., ge=1, description="Number of LLM reasoning steps executed")
    is_completed: bool = Field(
        default=True, description="Whether goal was completed or hit max_steps ceiling"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Chronological trace of all tool invocations"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Cumulative tokens consumed across all turns"
    )
    latency_ms: float = Field(default=0.0, description="Total execution duration in milliseconds")


class AgentRunRequest(BaseModel):
    """API payload to execute an autonomous sales research goal."""

    goal: str = Field(
        ..., min_length=5, description="High-level investigative or qualification objective"
    )
    company_name: Optional[str] = Field(
        default=None, description="Target company name if applicable"
    )
    domain: Optional[str] = Field(
        default=None, description="Target domain if known (e.g. 'stripe.com')"
    )
    context: Optional[str] = Field(
        default=None, description="Optional CRM context or background notes"
    )
    max_steps: int = Field(
        default=6, ge=1, le=10, description="Maximum agent execution turns (1 to 10)"
    )
    allowed_tools: Optional[list[str]] = Field(
        default=None, description="Optional allowlist of tool names"
    )

    @field_validator("goal")
    @classmethod
    def validate_goal_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Goal cannot be empty or only whitespace")
        return v.strip()


class AgentRunResponse(BaseModel):
    """API response summarizing agent investigation, tool audit trace, and metrics."""

    request_id: str
    goal: str
    final_answer: str
    steps_taken: int
    is_completed: bool
    tool_calls: list[ToolCallRecord]
    citations: list[EvidenceCitation] = Field(
        default_factory=list, description="Verifiable evidence citations gathered during execution"
    )
    total_tokens: int
    latency_ms: float
