import re
from typing import Optional

from pydantic import BaseModel, Field

BOILERPLATE_TERMS = {
    "cookie",
    "privacy policy",
    "terms of service",
    "terms of use",
    "all rights reserved",
    "copyright",
    "login",
    "sign in",
    "forgot password",
}

RECRUITMENT_TERMS = {
    "job opening",
    "we're hiring",
    "we are hiring",
    "apply now",
    "job description",
    "salary",
    "looking for talents",
    "responsibilities:",
    "requirements:",
    "experience level",
    "employment type",
    "equal opportunity employer",
}

# Universal commercial offering indicators across all company types (product, media, software, services, retail, etc.)
BUSINESS_OFFERING_TERMS = {
    "about",
    "company",
    "overview",
    "services",
    "products",
    "solutions",
    "offerings",
    "clients",
    "customers",
    "mission",
}


class RankedChunk(BaseModel):
    url: str
    chunk: str
    score: float
    matched_criteria: list[str] = Field(default_factory=list)


def score_chunk(
    chunk: str,
    company_name: str,
    context_summary: Optional[str] = None,
    qualification_criteria: Optional[list[str]] = None,
) -> tuple[float, list[str]]:
    """
    Score a text chunk based on relevance to company name, existing summary context,
    and business/sales intelligence indicators.
    """
    chunk_lower = chunk.lower()
    score = 0.0
    matched_criteria: list[str] = []

    # Penalize short chunks or boilerplate
    if len(chunk.strip()) < 40:
        return 0.0, []

    for bp in BOILERPLATE_TERMS:
        if bp in chunk_lower:
            score -= 1.0

    # Penalize job postings and recruitment boilerplate
    for recruit in RECRUITMENT_TERMS:
        if recruit in chunk_lower:
            score -= 3.0

    # Company name match
    company_tokens = re.findall(r"\w+", company_name.lower())
    for token in company_tokens:
        if len(token) > 2 and token in chunk_lower:
            score += 1.0

    # Universal commercial offering indicators
    for term in BUSINESS_OFFERING_TERMS:
        if term in chunk_lower:
            score += 0.5

    # Existing summary context match
    if context_summary:
        summary_tokens = [w for w in re.findall(r"\w+", context_summary.lower()) if len(w) > 3]
        if summary_tokens:
            matches = sum(1 for w in summary_tokens if w in chunk_lower)
            score += (matches / len(summary_tokens)) * 1.5

    # Optional qualification criteria matches
    for criterion in (qualification_criteria or []):
        words = [w for w in re.findall(r"\w+", criterion.lower()) if len(w) > 3]
        if not words:
            continue
        matches = sum(1 for w in words if w in chunk_lower)
        if matches > 0:
            ratio = matches / len(words)
            score += ratio * 2.0
            if ratio >= 0.5:
                matched_criteria.append(criterion)

    normalized_score = max(0.0, round(score, 2))
    return normalized_score, matched_criteria


def rank_chunks(
    chunk_tuples: list[tuple[str, str]],
    company_name: str,
    context_summary: Optional[str] = None,
    qualification_criteria: Optional[list[str]] = None,
    top_k: int = 6,
) -> list[RankedChunk]:
    """
    Rank chunks from extracted pages by relevance to company and context summary.
    `chunk_tuples`: list of (url, text_chunk)
    """
    scored_chunks: list[RankedChunk] = []

    for url, text in chunk_tuples:
        score, matched = score_chunk(
            chunk=text,
            company_name=company_name,
            context_summary=context_summary,
            qualification_criteria=qualification_criteria,
        )
        if score > 0.0:
            scored_chunks.append(
                RankedChunk(
                    url=url,
                    chunk=text,
                    score=score,
                    matched_criteria=matched,
                )
            )

    # Sort descending by score
    scored_chunks.sort(key=lambda x: x.score, reverse=True)
    return scored_chunks[:top_k]
