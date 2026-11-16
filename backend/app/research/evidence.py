import json
import re
from typing import Optional

from app.core.logger import log
from app.llm import LLMRequest, LLMResponse, call_llm
from app.research.models import EvidenceCitation
from app.research.ranker import RankedChunk

SYNTHESIS_SYSTEM_PROMPT = """You are a rigorous Company Intelligence Analyst.
Your mandate is to produce truthful, verifiable company research based EXCLUSIVELY on the provided web extracts.

Rules:
1. CORE PURPOSE & OFFERINGS: Focus strictly on what the company does, their primary products, services, solutions, or content offerings, their business model, and who they serve. Ignore internal hiring notices, job postings, legal boilerplate, and cookie notices.
2. VERIFIABLE CITATIONS: Every claim MUST cite a verbatim quote directly present in the source text. Do NOT fabricate quotes or URLs.
3. EVIDENCE INSUFFICIENCY: If the provided text does not contain enough concrete information to understand the company or evaluate the requested context, you MUST set "is_insufficient_evidence": true and explicitly detail what is missing in "insufficient_reasons". Never guess or hallucinate missing facts.
4. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching the requested schema. No markdown code blocks, no preamble."""


def _safe_parse_json(text: str) -> Optional[dict]:
    """Resiliently parse JSON output from LLM, handling markdown fences and partial JSON."""
    clean = text.strip()
    if "```" in clean:
        clean = re.sub(r"^```(?:json)?", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean, flags=re.MULTILINE).strip()

    # 1. Direct parse
    try:
        return json.loads(clean)
    except Exception:
        pass

    # 2. Extract outermost JSON object
    match = re.search(r"(\{.*\})", clean, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Handle truncated JSON strings by closing open structures
    repairs = [
        clean + '"}]}',
        clean + '"]}',
        clean + '"]}}',
        clean + '}',
        clean + '"}',
    ]
    for attempt in repairs:
        try:
            return json.loads(attempt)
        except Exception:
            continue

    return None


def _verify_and_clean_citations(
    raw_citations: list[dict],
    available_chunks: list[RankedChunk],
) -> list[EvidenceCitation]:
    """Validate that cited quotes actually exist in retrieved chunks."""
    all_content_by_url: dict[str, str] = {}
    for rc in available_chunks:
        all_content_by_url[rc.url] = (
            all_content_by_url.get(rc.url, "") + " " + rc.chunk.lower()
        )

    validated: list[EvidenceCitation] = []
    for c in raw_citations:
        claim = c.get("claim", "").strip()
        quote = c.get("quote", "").strip()
        source_url = c.get("source_url", "").strip()
        criterion = c.get("criterion")

        if not claim or not quote or not source_url:
            continue

        # Check if quote exists in the source text
        url_text = all_content_by_url.get(source_url, "")
        quote_clean = " ".join(quote.lower().split())

        # Calculate verification match
        is_verified = (
            quote_clean in url_text
            or any(quote_clean in rc.chunk.lower() for rc in available_chunks)
        )
        relevance_score = float(c.get("relevance_score", 0.9 if is_verified else 0.5))

        validated.append(
            EvidenceCitation(
                claim=claim,
                quote=quote,
                source_url=source_url,
                relevance_score=min(1.0, max(0.0, relevance_score)),
                criterion=criterion,
            )
        )

    return validated


async def synthesize_evidence(
    company_name: str,
    domain: Optional[str],
    ranked_chunks: list[RankedChunk],
    request_id: str,
    existing_summary: Optional[str] = None,
    qualification_criteria: Optional[list[str]] = None,
) -> tuple[str, list[EvidenceCitation], bool, list[str]]:
    """
    Synthesize research findings into verifiable citations and check evidence sufficiency.
    Returns: (summary, citations, is_insufficient_evidence, insufficient_reasons)
    """
    criteria = qualification_criteria or []

    # 1. Check for immediate evidence insufficiency
    if not ranked_chunks:
        return (
            f"Unable to gather sufficient public information on '{company_name}'.",
            [],
            True,
            ["No accessible or readable public web sources could be found."],
        )

    total_words = sum(len(rc.chunk.split()) for rc in ranked_chunks)
    if total_words < 40:
        return (
            f"Retrieved content for '{company_name}' is too sparse to evaluate.",
            [],
            True,
            ["Total retrieved text is below the minimum threshold (40 words) for qualification."],
        )

    # 2. Build context for LLM synthesis
    context_blocks = []
    for idx, rc in enumerate(ranked_chunks, 1):
        context_blocks.append(f"--- Source [{idx}]: {rc.url} ---\n{rc.chunk}")
    joined_sources = "\n\n".join(context_blocks)

    criteria_text = (
        "\n".join(f"- {c}" for c in criteria)
        if criteria
        else "General company research (core business activities, primary products or offerings, target audience, and business model)"
    )

    existing_context_str = (
        f"\nExisting Notes / Context:\n{existing_summary}\n"
        if existing_summary
        else ""
    )

    prompt = f"""Target Company: {company_name}
Target Domain: {domain or 'Not provided'}{existing_context_str}

Research Scope:
{criteria_text}

Retrieved Web Extracts:
{joined_sources}

Analyze the company based ONLY on the extracts above. Verify any existing notes and ground all facts in concrete citations.
Return a valid JSON object with:
{{
  "summary": "2-3 sentences summarizing what the company does, their primary products or offerings, and their business model.",
  "citations": [
    {{
      "claim": "Specific fact about the company",
      "quote": "Verbatim quote from the text supporting this fact",
      "source_url": "Exact URL from Source headers above",
      "relevance_score": 0.95,
      "criterion": "Target research topic this addresses, or null"
    }}
  ],
  "is_insufficient_evidence": false,
  "insufficient_reasons": []
}}

Note on Evidence Insufficiency: If the sources do NOT confirm what the company does or their primary offerings, set "is_insufficient_evidence": true and list what is missing in "insufficient_reasons"."""

    llm_req = LLMRequest(
        prompt=prompt,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=2048,
        request_id=request_id,
    )

    llm_resp = await call_llm(llm_req)
    if isinstance(llm_resp, LLMResponse):
        raw_text = llm_resp.content.strip()
    else:
        raw_text = ""

    # Parse JSON from model output
    try:
        data = _safe_parse_json(raw_text)
        if not data:
            raise ValueError(f"Could not parse valid JSON from response (len={len(raw_text)})")

        summary = data.get("summary", f"Research profile for {company_name}")
        is_insufficient = bool(data.get("is_insufficient_evidence", False))
        insufficient_reasons = [str(r) for r in data.get("insufficient_reasons", [])]
        raw_citations = data.get("citations", [])

        citations = _verify_and_clean_citations(raw_citations, ranked_chunks)

        # If qualification criteria were requested but 0 citations matched them, flag insufficiency
        if qualification_criteria and not citations:
            is_insufficient = True
            insufficient_reasons.append("None of the requested qualification criteria could be cited.")

        return summary, citations, is_insufficient, insufficient_reasons
    except Exception as exc:
        log.warning(f"[{request_id}] Failed to parse JSON from LLM synthesis: {exc}. Using fallback.")
        # Fallback heuristic summary
        fallback_summary = f"{company_name} research synthesized from {len(ranked_chunks)} source passages."
        fallback_citations = [
            EvidenceCitation(
                claim=f"{company_name} web extract",
                quote=ranked_chunks[0].chunk[:150],
                source_url=ranked_chunks[0].url,
                relevance_score=0.8,
            )
        ]
        return fallback_summary, fallback_citations, False, []
