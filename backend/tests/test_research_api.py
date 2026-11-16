from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.research.models import EvidenceCitation, ResearchResponse


@pytest.mark.asyncio
async def test_research_api_blank_company_validation_error(client: AsyncClient):
    response = await client.post("/api/research", json={"company_name": "   "})
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("Company name cannot be empty" in err["msg"] for err in errors)


@pytest.mark.asyncio
async def test_research_api_success(client: AsyncClient):
    mock_research_response = ResearchResponse(
        request_id="res_test123",
        company_name="Snowflake",
        domain="snowflake.com",
        summary="Snowflake provides a cloud data platform for data warehousing and analytics.",
        citations=[
            EvidenceCitation(
                claim="Snowflake provides a cloud data platform.",
                quote="The Data Cloud enables organizations to unite siloed data.",
                source_url="https://snowflake.com",
                relevance_score=0.95,
                criterion="Cloud Data Platform",
            )
        ],
        is_insufficient_evidence=False,
        insufficient_reasons=[],
        sources_scraped=["https://snowflake.com"],
        latency_ms=450.0,
    )

    with patch(
        "app.main.conduct_research",
        new=AsyncMock(return_value=mock_research_response),
    ):
        response = await client.post(
            "/api/research",
            json={
                "company_name": "Snowflake",
                "domain": "snowflake.com",
                "summary": "Cloud data warehousing platform",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "res_test123"
    assert data["company_name"] == "Snowflake"
    assert data["is_insufficient_evidence"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["source_url"] == "https://snowflake.com"
    assert data["sources_scraped"] == ["https://snowflake.com"]
