from typing import Any, Optional

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator


class ResearchRequest(BaseModel):
    """Payload for company research & qualification retrieval."""

    company_name: str = Field(..., min_length=1, description="Target company name")
    domain: Optional[str] = Field(
        default=None,
        description="Optional company domain (e.g. 'stripe.com')",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional existing company summary or lead context from CRM",
    )

    @field_validator("company_name")
    @classmethod
    def validate_company_name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Company name cannot be empty or only whitespace")
        return v.strip()


class SearchResult(BaseModel):
    """Individual search engine result item."""

    title: str
    url: str
    snippet: str = ""
    source: str = "bing"  # "bing" | "duckduckgo" | "google" | "mock"


class ExtractedPage(BaseModel):
    """Scraped and parsed webpage content."""

    url: str
    title: str
    content: str
    chunks: list[str] = Field(default_factory=list)
    internal_links: list[str] = Field(default_factory=list)


class EvidenceCitation(BaseModel):
    """Verifiable piece of evidence citing source URL and verbatim quote."""

    claim: str = Field(description="Factual claim about the company or product")
    quote: str = Field(description="Verbatim text quote from the source supporting the claim")
    source_url: str = Field(description="Exact URL where the quote was extracted from")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    criterion: Optional[str] = Field(
        default=None,
        description="Target qualification criterion this evidence supports",
    )


class LLMCitationPayload(BaseModel):
    """Structured citation item as produced by the synthesis LLM."""

    claim: str = Field(..., description="Factual claim about the company")
    quote: str = Field(..., description="Verbatim text quote from the source supporting the claim")
    source_url: str = Field(..., description="Exact URL where the quote was extracted from")
    relevance_score: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    criterion: Optional[str] = Field(
        default=None,
        description="Target qualification criterion this evidence supports",
    )

    @field_validator("claim", "quote", "source_url")
    @classmethod
    def validate_strings_not_blank(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Citation claim, quote, and source_url must be non-empty strings")
        return v.strip()


class LLMResearchOutput(BaseModel):
    """Strict schema for validating structured JSON produced by synthesis LLM."""

    summary: str = Field(..., description="Executive summary of the company")
    citations: list[LLMCitationPayload] = Field(
        ..., description="List of structured citations supporting the summary"
    )
    is_insufficient_evidence: StrictBool = Field(
        ..., description="Strict boolean indicating if evidence was insufficient"
    )
    insufficient_reasons: list[str] = Field(
        default_factory=list,
        description="List of reasons why evidence was insufficient if applicable",
    )

    @field_validator("summary")
    @classmethod
    def validate_summary_type(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("Summary must be a string")
        return v.strip()

    @model_validator(mode="after")
    def validate_summary_when_sufficient(self) -> "LLMResearchOutput":
        # If evidence is claimed to be sufficient, a substantive non-empty summary is required
        if not self.is_insufficient_evidence and not self.summary:
            raise ValueError("Summary must be a non-empty string when evidence is sufficient")
        return self


class ResearchResponse(BaseModel):
    """Structured research report with verifiable citations and sufficiency check."""

    request_id: str
    company_name: str
    domain: Optional[str] = None
    summary: str
    citations: list[EvidenceCitation] = Field(default_factory=list)
    is_insufficient_evidence: bool = False
    insufficient_reasons: list[str] = Field(default_factory=list)
    sources_scraped: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
