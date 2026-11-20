import time
import uuid

from app.agent.models import AgentRunRequest, AgentRunResponse, ToolCallRecord
from app.agent.runner import AgentRunner
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


async def run_sales_agent(request: AgentRunRequest) -> AgentRunResponse:
    """
    Execute an autonomous sales agent run for a given user goal:
    1. Generates correlation request_id.
    2. Initializes AgentRunner with configured limits and allowed tools.
    3. Runs multi-turn ReAct reasoning loop.
    4. Extracts and verifies citations if present.
    5. Returns AgentRunResponse with execution trace, metrics, and answer.
    """
    request_id = f"agt_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    log.bind(
        request_id=request_id,
        goal=request.goal,
        company_name=request.company_name,
        domain=request.domain,
        max_steps=request.max_steps,
    ).info(f"[{request_id}] Starting sales agent run: '{request.goal}'")

    runner = AgentRunner(
        max_steps=request.max_steps,
        allowed_tools=request.allowed_tools,
    )

    result = await runner.run(
        goal=request.goal,
        company_name=request.company_name,
        domain=request.domain,
        context=request.context,
    )

    citations = _extract_citations(result.final_answer, result.tool_calls)
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
        goal=request.goal,
        final_answer=result.final_answer,
        steps_taken=result.steps_taken,
        is_completed=result.is_completed,
        tool_calls=result.tool_calls,
        citations=citations,
        total_tokens=result.total_tokens,
        latency_ms=total_latency_ms,
    )
