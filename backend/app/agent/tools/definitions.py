import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from app.agent.tools.base import BaseTool, ToolOutput, ToolPermissionLevel
from app.core.logger import log
from app.research.extractor import extract_page
from app.research.search import search_company

# --- 1. SearchCompanyTool ---


class SearchCompanyInput(BaseModel):
    """Input parameters for search_company tool."""

    query: str = Field(
        ...,
        min_length=1,
        description="Search query to find official websites, news, or profiles for the target company",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of search results to return (1 to 5)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()


class SearchCompanyTool(BaseTool):
    """Tool for discovering company websites and information across public search engines."""

    name = "search_company"
    description = (
        "Search the public web for target company domains, product pages, leadership profiles, "
        "and corporate announcements. Returns titles, URLs, and snippets."
    )
    args_schema = SearchCompanyInput
    permission_level = ToolPermissionLevel.READ_ONLY

    async def execute(self, query: str, max_results: int = 3) -> list[dict[str, str]]:
        log.info(f"Executing search_company: query='{query}', max_results={max_results}")
        results = await search_company(query=query, max_results=max_results)
        return [
            {
                "title": sr.title,
                "url": sr.url,
                "snippet": sr.snippet,
            }
            for sr in results
        ]


# --- 2. FetchWebpageTool ---


class FetchWebpageInput(BaseModel):
    """Input parameters for fetch_webpage tool."""

    url: str = Field(
        ...,
        min_length=1,
        description="The full HTTP or HTTPS URL of the webpage to scrape and extract text from",
    )

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, v: str) -> str:
        url_clean = v.strip()
        if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
            url_clean = f"https://{url_clean}"
        return url_clean


class FetchWebpageTool(BaseTool):
    """Tool for securely fetching and extracting clean text from a public webpage."""

    name = "fetch_webpage"
    description = (
        "Safely fetch and read a public webpage. Enforces SSRF protection (blocking internal/private IPs) "
        "and extracts structured body text, headers, and bullet points. Output is capped to 1,400 characters."
    )
    args_schema = FetchWebpageInput
    permission_level = ToolPermissionLevel.READ_ONLY

    async def execute(self, url: str) -> Any:
        log.info(f"Executing fetch_webpage: url='{url}'")
        page = await extract_page(url=url)
        if not page:
            log.warning(f"Failed to extract content from webpage: url='{url}'")
            return ToolOutput(
                success=False,
                data={"url": url, "status": "failed"},
                error=(
                    f"Unable to access webpage '{url}'. The target was either blocked by SSRF checks, "
                    f"returned a non-200 status, or yielded no extractable text."
                ),
            )

        # Context Budget Guard: Limit tool response to 1,200–1,500 chars max
        # Combine title and top high-density chunks
        combined_chunks: list[str] = []
        accumulated_len = 0
        max_budget = 1400

        for chunk in page.chunks:
            if accumulated_len + len(chunk) > max_budget:
                remaining = max_budget - accumulated_len
                if remaining > 100:
                    combined_chunks.append(chunk[:remaining] + "...")
                break
            combined_chunks.append(chunk)
            accumulated_len += len(chunk)

        body_snippet = (
            "\n\n".join(combined_chunks) if combined_chunks else page.content[:max_budget].strip()
        )

        return {
            "url": page.url,
            "title": page.title,
            "status": "success",
            "content": body_snippet,
        }


# --- 3. EvaluateLeadFitTool ---

DEFAULT_PLATFORM_ICP_CRITERIA: list[str] = [
    "B2B SaaS, Cloud, or Enterprise Technology business model",
    "Digital products, developer APIs, or cloud infrastructure offerings",
    "Enterprise or mid-market commercial customer audience",
    "Multi-user, security, or compliance operational requirements",
]


class EvaluateLeadFitInput(BaseModel):
    """Input parameters for evaluate_lead_fit tool."""

    company_name: str = Field(..., description="Target company name being evaluated")
    offerings_summary: str = Field(
        ...,
        description="Factual summary of the company's offerings, products, services, or business model",
    )
    target_criteria: Optional[list[str]] = Field(
        default=None,
        description="Optional qualification criteria to test against. Defaults to platform ICP criteria if omitted.",
    )

    @field_validator("company_name", "offerings_summary")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty or only whitespace")
        return v.strip()

    @field_validator("target_criteria")
    @classmethod
    def validate_target_criteria(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is not None:
            cleaned = [c.strip() for c in v if c and c.strip()]
            if not cleaned:
                raise ValueError("target_criteria cannot be empty or contain only blank items")
            return cleaned
        return v


class EvaluateLeadFitTool(BaseTool):
    """Tool for evaluating target company qualification against sales qualification criteria."""

    name = "evaluate_lead_fit"
    description = (
        "Deterministically evaluate whether a company's offerings and business model match "
        "target sales qualification criteria. Computes fit score, matching criteria, and gaps."
    )
    args_schema = EvaluateLeadFitInput
    permission_level = ToolPermissionLevel.READ_ONLY

    async def execute(
        self,
        company_name: str,
        offerings_summary: str,
        target_criteria: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        raw_criteria = target_criteria or DEFAULT_PLATFORM_ICP_CRITERIA
        criteria = [c.strip() for c in raw_criteria if c and c.strip()]
        if not criteria:
            criteria = DEFAULT_PLATFORM_ICP_CRITERIA

        log.info(
            f"Executing evaluate_lead_fit for '{company_name}' against {len(criteria)} criteria"
        )
        summary_lower = offerings_summary.lower()

        matched_criteria: list[str] = []
        unmatched_criteria: list[str] = []

        for criterion in criteria:
            crit_lower = criterion.lower()
            subclauses = [c.strip() for c in re.split(r",|\bor\b", crit_lower) if c.strip()]
            criterion_matched = False

            if len(subclauses) > 1:
                for sub in subclauses:
                    sub_clean = re.sub(r"[^\w\s]", " ", sub)
                    tokens = [t for t in sub_clean.split() if len(t) >= 2]
                    if not tokens:
                        continue
                    matches = sum(
                        1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", summary_lower)
                    )
                    threshold = 1.0 if len(tokens) <= 2 else 0.5
                    if matches / len(tokens) >= threshold:
                        criterion_matched = True
                        break

            if not criterion_matched:
                crit_clean = re.sub(r"[^\w\s]", " ", crit_lower)
                tokens = [t for t in crit_clean.split() if len(t) >= 2]
                if not tokens:
                    # Tokenless criteria must never auto-match
                    criterion_matched = False
                else:
                    matches = sum(
                        1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", summary_lower)
                    )
                    if matches / len(tokens) >= 0.4:
                        criterion_matched = True

            if criterion_matched:
                matched_criteria.append(criterion)
            else:
                unmatched_criteria.append(criterion)

        total_criteria = len(criteria)
        fit_score = round(len(matched_criteria) / total_criteria, 2) if total_criteria > 0 else 0.0
        is_qualified = fit_score >= 0.5
        status = (
            "QUALIFIED"
            if fit_score >= 0.75
            else ("PARTIAL_FIT" if fit_score >= 0.5 else "DISQUALIFIED")
        )

        why_fit = [f"Matches: {crit}" for crit in matched_criteria]
        why_not_fit = [f"No evidence found for: {crit}" for crit in unmatched_criteria]
        proposal_hooks = [
            f"Tailor outreach around {matched_criteria[0]}"
            if matched_criteria
            else "Focus on core infrastructure efficiency"
        ]
        if len(matched_criteria) > 1:
            proposal_hooks.append(f"Highlight synergies with {matched_criteria[1]}")

        return {
            "company_name": company_name,
            "fit_score": fit_score,
            "is_qualified": is_qualified,
            "status": status,
            "why_fit": why_fit,
            "why_not_fit": why_not_fit,
            "matched_criteria": matched_criteria,
            "unmatched_criteria": unmatched_criteria,
            "proposal_hooks": proposal_hooks,
            "evaluation_note": (
                f"Lexical heuristic evaluation: {company_name} matches {len(matched_criteria)}/{total_criteria} "
                f"target criteria (fit score: {fit_score * 100:.0f}%, status: {status})."
            ),
        }
