import json
from unittest.mock import AsyncMock, patch

import pytest

from app.llm import LLMResponse
from app.research.evidence import synthesize_evidence
from app.research.ranker import RankedChunk


@pytest.mark.asyncio
async def test_synthesize_evidence_flags_insufficiency_when_empty():
    summary, citations, is_insufficient, reasons = await synthesize_evidence(
        company_name="Ghost Company",
        domain="ghost.example",
        ranked_chunks=[],
        request_id="test_req_insufficient",
    )

    assert is_insufficient is True
    assert len(reasons) > 0
    assert "No accessible or readable public web sources" in reasons[0]
    assert len(citations) == 0


@pytest.mark.asyncio
async def test_synthesize_evidence_flags_insufficiency_when_sparse():
    # Fewer than 40 words triggers evidence insufficiency
    sparse_chunks = [
        RankedChunk(
            url="https://acme.com",
            chunk="Acme is a software company based in California.",
            score=1.0,
            matched_criteria=[],
        )
    ]
    summary, citations, is_insufficient, reasons = await synthesize_evidence(
        company_name="Acme",
        domain="acme.com",
        ranked_chunks=sparse_chunks,
        request_id="test_req_sparse",
    )

    assert is_insufficient is True
    assert any("below the minimum threshold" in r for r in reasons)


@pytest.mark.asyncio
async def test_synthesize_evidence_with_llm_citations():
    mock_chunks = [
        RankedChunk(
            url="https://stripe.com",
            chunk=(
                "Stripe is a financial infrastructure platform for businesses. "
                "Millions of companies from the world's largest enterprises to the most ambitious startups "
                "use Stripe to accept payments, grow their revenue, and accelerate new business opportunities. "
                "Headquartered in San Francisco and Dublin, Stripe powers online and in-person payments "
                "with deep developer APIs and multi-currency billing engines."
            ),
            score=3.5,
            matched_criteria=["B2B SaaS"],
        )
    ]

    llm_payload = {
        "summary": "Stripe provides global payments and financial infrastructure.",
        "citations": [
            {
                "claim": "Stripe is a financial infrastructure platform for businesses.",
                "quote": "Stripe is a financial infrastructure platform for businesses.",
                "source_url": "https://stripe.com",
                "relevance_score": 0.95,
                "criterion": "B2B SaaS",
            }
        ],
        "is_insufficient_evidence": False,
        "insufficient_reasons": [],
    }

    mock_llm_response = LLMResponse(
        request_id="req_syn_1",
        content=json.dumps(llm_payload),
        finish_reason="stop",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=250.0,
    )

    with patch("app.research.evidence.call_llm", new=AsyncMock(return_value=mock_llm_response)):
        summary, citations, is_insufficient, reasons = await synthesize_evidence(
            company_name="Stripe",
            domain="stripe.com",
            ranked_chunks=mock_chunks,
            request_id="req_syn_1",
            existing_summary="Payments platform for internet businesses",
        )

    assert is_insufficient is False
    assert len(citations) == 1
    assert citations[0].claim == "Stripe is a financial infrastructure platform for businesses."
    assert citations[0].source_url == "https://stripe.com"
    assert citations[0].relevance_score == 0.95
