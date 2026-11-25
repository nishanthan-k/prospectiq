from typing import Any, Optional

from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator

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


class QualificationVerdict(BaseModel):
    """Structured business qualification case for outreach and proposal generation."""

    is_fit: bool = Field(..., description="Whether company qualifies as a fit for our platform ICP")
    fit_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Qualification confidence score between 0.0 and 1.0",
    )
    status: str = Field(..., description="'QUALIFIED' | 'DISQUALIFIED' | 'PARTIAL_FIT'")
    why_fit: list[str] = Field(
        default_factory=list, description="Specific evidence and reasons why the company fits"
    )
    why_not_fit: list[str] = Field(
        default_factory=list, description="Gaps, missing prerequisites, or disqualifiers"
    )
    matched_criteria: list[str] = Field(
        default_factory=list, description="ICP criteria successfully matched"
    )
    unmatched_criteria: list[str] = Field(
        default_factory=list, description="ICP criteria not satisfied"
    )
    proposal_hooks: list[str] = Field(
        default_factory=list,
        description="Actionable value propositions or angles for outreach and proposal planning",
    )


class AgentInvestigateRequest(BaseModel):
    """API payload to investigate a company's offerings and business profile."""

    company_name: str = Field(..., min_length=1, description="Target company name (e.g. 'Retool')")
    domain: Optional[str] = Field(
        default=None, description="Target domain if known (e.g. 'retool.com')"
    )
    context: Optional[str] = Field(default=None, description="Optional CRM context or notes")
    max_steps: StrictInt = Field(
        default=4, ge=1, le=10, description="Maximum agent execution turns (1 to 10)"
    )

    @field_validator("company_name")
    @classmethod
    def validate_company_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Company name cannot be empty or only whitespace")
        return v.strip()


class AgentInvestigateResponse(BaseModel):
    """API response summarizing company investigation findings and audit trail."""

    request_id: str
    company_name: str
    domain: Optional[str] = None
    summary: str = Field(..., description="Synthesized executive overview of the company")
    primary_offerings: list[str] = Field(
        default_factory=list, description="Core products, platforms, or services"
    )
    business_model: str = Field(
        default="Unknown", description="Identified business/revenue model (e.g. B2B SaaS)"
    )
    target_audience: str = Field(
        default="Unknown", description="Primary buyer persona or ideal customer profile"
    )
    citations: list[EvidenceCitation] = Field(
        default_factory=list, description="Verifiable evidence citations"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Audit trace of tools used"
    )
    steps_taken: int
    total_tokens: int
    latency_ms: float


class AgentQualifyRequest(BaseModel):
    """API payload to evaluate company fit against the platform's ICP."""

    company_name: str = Field(..., min_length=1, description="Target company name (e.g. 'Retool')")
    domain: Optional[str] = Field(
        default=None, description="Target domain if known (e.g. 'retool.com')"
    )
    context: Optional[str] = Field(default=None, description="Optional CRM context or notes")
    max_steps: StrictInt = Field(
        default=4, ge=1, le=10, description="Maximum agent execution turns (1 to 10)"
    )

    @field_validator("company_name")
    @classmethod
    def validate_company_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Company name cannot be empty or only whitespace")
        return v.strip()


class AgentQualifyResponse(BaseModel):
    """API response containing the qualification verdict, business case, and audit trail."""

    request_id: str
    company_name: str
    domain: Optional[str] = None
    qualification: QualificationVerdict
    citations: list[EvidenceCitation] = Field(
        default_factory=list, description="Verifiable evidence citations"
    )
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="Audit trace of tools used"
    )
    steps_taken: int
    total_tokens: int
    latency_ms: float


class AgentRunRequest(BaseModel):
    """API payload to execute an autonomous sales research goal."""

    goal: Optional[str] = Field(
        default=None, description="High-level investigative or qualification objective"
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
    max_steps: StrictInt = Field(
        default=6, ge=1, le=10, description="Maximum agent execution turns (1 to 10)"
    )
    allowed_tools: Optional[list[str]] = Field(
        default=None, description="Optional allowlist of tool names"
    )

    @model_validator(mode="after")
    def validate_goal_or_company(self) -> "AgentRunRequest":
        if self.goal is not None:
            cleaned = self.goal.strip()
            if not cleaned:
                raise ValueError("Goal cannot be empty or only whitespace")
            if len(cleaned) < 5:
                raise ValueError("Goal must be at least 5 characters long")
            self.goal = cleaned
        elif self.company_name and self.company_name.strip():
            self.goal = (
                f"Investigate {self.company_name.strip()} and qualify fit for platform outreach"
            )
        else:
            raise ValueError("Either 'goal' or 'company_name' must be provided")
        return self


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
    qualification: Optional[QualificationVerdict] = Field(
        default=None, description="Structured qualification verdict if evaluated during run"
    )
    total_tokens: int
    latency_ms: float
