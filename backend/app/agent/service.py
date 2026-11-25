import re
import time
import uuid
from typing import Any, Optional

from app.agent.models import (
    AgentInvestigateRequest,
    AgentInvestigateResponse,
    AgentQualifyRequest,
    AgentQualifyResponse,
    AgentRunRequest,
    AgentRunResponse,
    QualificationVerdict,
    ToolCallRecord,
)
from app.agent.runner import AgentRunner
from app.agent.tools.definitions import DEFAULT_PLATFORM_ICP_CRITERIA
from app.core.json_utils import _safe_parse_json
from app.core.logger import log
from app.research.evidence import _verify_and_clean_citations
from app.research.models import EvidenceCitation
from app.research.ranker import RankedChunk


def _extract_citations(
    final_answer: str, tool_calls: list[ToolCallRecord]
) -> list[EvidenceCitation]:
    """
    Attempt to extract and verify citations from the agent's final answer.
    If the agent emitted a structured JSON with citations, verifies them against
    content scraped during tool execution via _verify_and_clean_citations.
    """
    parsed = _safe_parse_json(final_answer)
    if not isinstance(parsed, dict) or "citations" not in parsed:
        return []

    raw_citations = parsed.get("citations")
    if not isinstance(raw_citations, list):
        return []

    # Build available chunks from successful fetch_webpage tool calls
    available_chunks: list[RankedChunk] = []
    for tc in tool_calls:
        if tc.tool_name == "fetch_webpage" and not tc.is_error and isinstance(tc.output, dict):
            url = tc.output.get("url", "")
            content = tc.output.get("content", "")
            if url and content:
                available_chunks.append(
                    RankedChunk(
                        chunk=content,
                        score=1.0,
                        url=url,
                        title=tc.output.get("title", ""),
                    )
                )

    if not available_chunks or not raw_citations:
        return []

    return _verify_and_clean_citations(raw_citations, available_chunks)


def _extract_qualification_verdict(
    final_answer: str,
    tool_calls: list[ToolCallRecord],
    company_name: Optional[str] = None,
) -> Optional[QualificationVerdict]:
    """
    Extract structured qualification verdict from evaluate_lead_fit tool calls
    or JSON emitted in the final synthesis.
    """
    # 1. Check for successful evaluate_lead_fit tool execution
    for tc in reversed(tool_calls):
        if tc.tool_name == "evaluate_lead_fit" and not tc.is_error and isinstance(tc.output, dict):
            out = tc.output
            is_fit = bool(out.get("is_qualified", False))
            fit_score = float(out.get("fit_score", 0.0))
            status = str(
                out.get("status")
                or (
                    "QUALIFIED"
                    if fit_score >= 0.75
                    else ("PARTIAL_FIT" if fit_score >= 0.5 else "DISQUALIFIED")
                )
            )
            matched = [str(c) for c in (out.get("matched_criteria") or [])]
            unmatched = [str(c) for c in (out.get("unmatched_criteria") or [])]
            why_fit = [str(r) for r in (out.get("why_fit") or [f"Matches: {c}" for c in matched])]
            why_not_fit = [
                str(r)
                for r in (
                    out.get("why_not_fit") or [f"No evidence found for: {c}" for c in unmatched]
                )
            ]
            hooks = [str(h) for h in (out.get("proposal_hooks") or [])]
            if not hooks and matched:
                hooks = [f"Position solution around enterprise requirements in {matched[0]}"]

            return QualificationVerdict(
                is_fit=is_fit,
                fit_score=fit_score,
                status=status,
                why_fit=why_fit,
                why_not_fit=why_not_fit,
                matched_criteria=matched,
                unmatched_criteria=unmatched,
                proposal_hooks=hooks,
            )

    # 2. Check if final_answer contains structured qualification JSON
    parsed = _safe_parse_json(final_answer)
    if isinstance(parsed, dict):
        qual_data = (
            parsed.get("qualification") if isinstance(parsed.get("qualification"), dict) else parsed
        )
        if "is_fit" in qual_data or "fit_score" in qual_data or "is_qualified" in qual_data:
            try:
                is_fit = bool(qual_data.get("is_fit", qual_data.get("is_qualified", False)))
                fit_score = float(qual_data.get("fit_score", 1.0 if is_fit else 0.0))
                status = str(
                    qual_data.get("status")
                    or (
                        "QUALIFIED"
                        if fit_score >= 0.75
                        else ("PARTIAL_FIT" if fit_score >= 0.5 else "DISQUALIFIED")
                    )
                )
                matched = [str(c) for c in (qual_data.get("matched_criteria") or [])]
                unmatched = [str(c) for c in (qual_data.get("unmatched_criteria") or [])]
                why_fit = [
                    str(r) for r in (qual_data.get("why_fit") or [f"Matches: {c}" for c in matched])
                ]
                why_not_fit = [
                    str(r)
                    for r in (
                        qual_data.get("why_not_fit")
                        or [f"Missing evidence for: {c}" for c in unmatched]
                    )
                ]
                hooks = [str(h) for h in (qual_data.get("proposal_hooks") or [])]
                return QualificationVerdict(
                    is_fit=is_fit,
                    fit_score=fit_score,
                    status=status,
                    why_fit=why_fit,
                    why_not_fit=why_not_fit,
                    matched_criteria=matched,
                    unmatched_criteria=unmatched,
                    proposal_hooks=hooks,
                )
            except Exception as e:
                log.warning(f"Failed to parse qualification from JSON in final_answer: {e}")

    return None


def _evaluate_text_against_icp(
    text: str, criteria: Optional[list[str]] = None
) -> QualificationVerdict:
    """
    Deterministic fallback evaluator when no evaluate_lead_fit tool was triggered.
    Scores text against default platform ICP criteria.
    """
    raw_criteria = criteria or DEFAULT_PLATFORM_ICP_CRITERIA
    target_criteria = [c.strip() for c in raw_criteria if c and c.strip()]
    if not target_criteria:
        target_criteria = DEFAULT_PLATFORM_ICP_CRITERIA
    summary_lower = text.lower()
    matched: list[str] = []
    unmatched: list[str] = []

    for criterion in target_criteria:
        crit_lower = criterion.lower()
        subclauses = [c.strip() for c in re.split(r",|\bor\b", crit_lower) if c.strip()]
        criterion_matched = False

        if len(subclauses) > 1:
            for sub in subclauses:
                sub_clean = re.sub(r"[^\w\s]", " ", sub)
                tokens = [t for t in sub_clean.split() if len(t) >= 2]
                if not tokens:
                    continue
                matches = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", summary_lower))
                threshold = 1.0 if len(tokens) <= 2 else 0.5
                if matches / len(tokens) >= threshold:
                    criterion_matched = True
                    break

        if not criterion_matched:
            crit_clean = re.sub(r"[^\w\s]", " ", crit_lower)
            tokens = [t for t in crit_clean.split() if len(t) >= 2]
            if not tokens:
                criterion_matched = False
            else:
                matches = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", summary_lower))
                if matches / len(tokens) >= 0.4:
                    criterion_matched = True

        if criterion_matched:
            matched.append(criterion)
        else:
            unmatched.append(criterion)

    total = len(target_criteria)
    score = round(len(matched) / total, 2) if total > 0 else 0.0
    is_fit = score >= 0.5
    status = "QUALIFIED" if score >= 0.75 else ("PARTIAL_FIT" if score >= 0.5 else "DISQUALIFIED")
    why_fit = (
        [f"Matches: {c}" for c in matched]
        if matched
        else ["No verified ICP criteria matched in available sources."]
    )
    why_not_fit = [f"Missing evidence for: {c}" for c in unmatched] if unmatched else []
    hooks = (
        [f"Position value proposition around {matched[0]}"]
        if matched
        else ["Explore core workflow modernization"]
    )
    if len(matched) > 1:
        hooks.append(f"Highlight operational scalability with {matched[1]}")

    return QualificationVerdict(
        is_fit=is_fit,
        fit_score=score,
        status=status,
        why_fit=why_fit,
        why_not_fit=why_not_fit,
        matched_criteria=matched,
        unmatched_criteria=unmatched,
        proposal_hooks=hooks,
    )


def _extract_investigate_profile(final_answer: str, company_name: str) -> dict[str, Any]:
    """
    Extract structured executive company profile from agent answer.
    """
    parsed = _safe_parse_json(final_answer)
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary", "") or parsed.get("overview", "")).strip()
        offerings = (
            parsed.get("primary_offerings") or parsed.get("offerings") or parsed.get("products")
        )
        biz_model = str(parsed.get("business_model", "")).strip()
        audience = str(parsed.get("target_audience", "") or parsed.get("audience", "")).strip()

        if summary:
            return {
                "summary": summary,
                "primary_offerings": (
                    [str(o) for o in offerings]
                    if isinstance(offerings, list)
                    else [str(offerings)]
                    if offerings
                    else [f"{company_name} core platform and services"]
                ),
                "business_model": biz_model or "B2B SaaS / Enterprise Technology",
                "target_audience": audience or "Enterprise & Technical Teams",
            }

    # Plain-text heuristics
    lines = [line.strip() for line in final_answer.splitlines() if line.strip()]
    bullet_items = [
        re.sub(r"^[-*•\d.]+\s*", "", line) for line in lines if re.match(r"^[-*•\d.]+\s+", line)
    ]

    summary = lines[0] if lines else final_answer.strip()
    primary_offerings = (
        bullet_items[:5] if bullet_items else [f"{company_name} core platform and services"]
    )

    text_lower = final_answer.lower()
    biz_model = "B2B SaaS"
    if "marketplace" in text_lower:
        biz_model = "B2B Marketplace"
    elif "api" in text_lower or "developer platform" in text_lower:
        biz_model = "Developer Platform / B2B SaaS"
    elif "infrastructure" in text_lower:
        biz_model = "Cloud & Enterprise Infrastructure"

    audience = "Enterprise Organizations & Engineering Teams"
    if "developer" in text_lower or "engineers" in text_lower:
        audience = "Software Engineers, Developers & Tech Leaders"
    elif "finance" in text_lower or "payment" in text_lower:
        audience = "Commercial Businesses, Merchants & Finance Teams"

    return {
        "summary": summary,
        "primary_offerings": primary_offerings,
        "business_model": biz_model,
        "target_audience": audience,
    }


async def investigate_company(request: AgentInvestigateRequest) -> AgentInvestigateResponse:
    """
    Dedicated endpoint service for investigating target company offerings, business model,
    and target audience using search and page retrieval tools.
    """
    request_id = f"inv_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    goal = (
        f"Investigate company '{request.company_name}'"
        + (f" (domain: {request.domain})" if request.domain else "")
        + ". Discover core product offerings, target audience, and business model."
    )

    runner = AgentRunner(
        max_steps=request.max_steps,
        allowed_tools=["search_company", "fetch_webpage"],
    )

    result = await runner.run(
        goal=goal,
        company_name=request.company_name,
        domain=request.domain,
        context=request.context,
        request_id=request_id,
    )

    citations = _extract_citations(result.final_answer, result.tool_calls)
    profile = _extract_investigate_profile(result.final_answer, request.company_name)
    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return AgentInvestigateResponse(
        request_id=request_id,
        company_name=request.company_name,
        domain=request.domain,
        summary=profile["summary"],
        primary_offerings=profile["primary_offerings"],
        business_model=profile["business_model"],
        target_audience=profile["target_audience"],
        citations=citations,
        tool_calls=result.tool_calls,
        steps_taken=result.steps_taken,
        total_tokens=result.total_tokens,
        latency_ms=total_latency_ms,
    )


async def qualify_company(request: AgentQualifyRequest) -> AgentQualifyResponse:
    """
    Dedicated endpoint service for qualifying a target company against our platform ICP.
    Investigates offerings and executes qualification evaluation.
    """
    request_id = f"qlf_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    goal = (
        f"Investigate and qualify '{request.company_name}'"
        + (f" (domain: {request.domain})" if request.domain else "")
        + " against platform ICP criteria: "
        + "; ".join(DEFAULT_PLATFORM_ICP_CRITERIA)
        + ". Gather evidence with search and fetch, then use evaluate_lead_fit to produce qualification verdict."
    )

    runner = AgentRunner(
        max_steps=request.max_steps,
        allowed_tools=["search_company", "fetch_webpage", "evaluate_lead_fit"],
    )

    result = await runner.run(
        goal=goal,
        company_name=request.company_name,
        domain=request.domain,
        context=request.context,
        request_id=request_id,
    )

    citations = _extract_citations(result.final_answer, result.tool_calls)
    verdict = _extract_qualification_verdict(
        result.final_answer, result.tool_calls, company_name=request.company_name
    )
    if not verdict:
        verdict = _evaluate_text_against_icp(result.final_answer)

    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return AgentQualifyResponse(
        request_id=request_id,
        company_name=request.company_name,
        domain=request.domain,
        qualification=verdict,
        citations=citations,
        tool_calls=result.tool_calls,
        steps_taken=result.steps_taken,
        total_tokens=result.total_tokens,
        latency_ms=total_latency_ms,
    )


async def run_sales_agent(request: AgentRunRequest) -> AgentRunResponse:
    """
    Unified autonomous sales agent execution (Investigate + Qualify = Run).
    Supports open-ended goals as well as automated qualification.
    """
    request_id = f"agt_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    effective_goal = request.goal or (
        f"Investigate {request.company_name} and qualify fit for platform outreach"
        if request.company_name
        else "Sales intelligence investigation"
    )

    log.bind(
        request_id=request_id,
        goal=effective_goal,
        company_name=request.company_name,
        domain=request.domain,
        max_steps=request.max_steps,
    ).info(f"[{request_id}] Starting sales agent run: '{effective_goal}'")

    runner = AgentRunner(
        max_steps=request.max_steps,
        allowed_tools=request.allowed_tools,
    )

    result = await runner.run(
        goal=effective_goal,
        company_name=request.company_name,
        domain=request.domain,
        context=request.context,
        request_id=request_id,
    )

    citations = _extract_citations(result.final_answer, result.tool_calls)
    qualification = _extract_qualification_verdict(
        result.final_answer, result.tool_calls, company_name=request.company_name
    )
    total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    log.bind(
        request_id=request_id,
        steps_taken=result.steps_taken,
        is_completed=result.is_completed,
        total_tokens=result.total_tokens,
        latency_ms=total_latency_ms,
    ).info(
        f"[{request_id}] Agent run finished in {result.steps_taken} steps ({total_latency_ms}ms)"
    )

    return AgentRunResponse(
        request_id=request_id,
        goal=effective_goal,
        final_answer=result.final_answer,
        steps_taken=result.steps_taken,
        is_completed=result.is_completed,
        tool_calls=result.tool_calls,
        citations=citations,
        qualification=qualification,
        total_tokens=result.total_tokens,
        latency_ms=total_latency_ms,
    )
