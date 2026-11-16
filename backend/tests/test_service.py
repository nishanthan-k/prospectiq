from unittest.mock import patch

import pytest

from app.research.models import ExtractedPage, ResearchRequest, SearchResult
from app.research.service import _extract_clean_host, conduct_research, is_search_result_relevant


def test_is_search_result_relevant():
    # 1. Rejects autocorrected unrelated company (Ebitaus vs Ebenus)
    sr_unrelated = SearchResult(
        title="Ebenus - Engineering Consulting & Solutions",
        url="https://ebenus.co.in/",
        snippet="Ebenus Tech offers mechanical and digital transformation services.",
        source="duckduckgo",
    )
    assert is_search_result_relevant(sr_unrelated, "Ebitaus Private Limited") is False

    # 2. Accepts matching brand name in title/snippet
    sr_matching = SearchResult(
        title="Ebitaus Technologies - Enterprise Cloud",
        url="https://ebitaus.com/",
        snippet="Ebitaus provides cloud data migration.",
        source="duckduckgo",
    )
    assert is_search_result_relevant(sr_matching, "Ebitaus Private Limited") is True

    # 3. Accepts acronyms and shorter domains (e.g. IBM for International Business Machines)
    sr_ibm = SearchResult(
        title="IBM - United States",
        url="https://ibm.com/",
        snippet="International Business Machines Corporation is a multinational technology corporation.",
        source="duckduckgo",
    )
    assert is_search_result_relevant(sr_ibm, "International Business Machines") is True

    # 4. Accepts TCS for Tata Consultancy Services
    sr_tcs = SearchResult(
        title="TCS Careers and Solutions",
        url="https://tcs.com/",
        snippet="Global leader in IT consulting and digital solutions.",
        source="duckduckgo",
    )
    assert is_search_result_relevant(sr_tcs, "Tata Consultancy Services") is True


def test_extract_clean_host():
    assert _extract_clean_host("https://www.ncspartners.com/in") == "ncspartners.com"
    assert _extract_clean_host("http://stripe.com/docs") == "stripe.com"
    assert _extract_clean_host("example.com") == "example.com"
    assert _extract_clean_host("www.retool.com") == "retool.com"


@pytest.mark.asyncio
async def test_conduct_research_with_domain_does_not_scrape_third_party():
    req = ResearchRequest(
        company_name="NCS Partners",
        domain="https://www.ncspartners.com/in",
        summary="Supply chain consulting and software solutions",
    )

    primary_page = ExtractedPage(
        url="https://www.ncspartners.com/in",
        title="NCS Partners India",
        content="NCS Partners provides comprehensive supply chain consulting and enterprise solutions.",
        chunks=[
            "NCS Partners provides comprehensive supply chain consulting and enterprise solutions for businesses globally."
        ],
        internal_links=["https://www.ncspartners.com/in/about", "https://www.ncspartners.com/in/services"],
    )

    about_page = ExtractedPage(
        url="https://www.ncspartners.com/in/about",
        title="About NCS Partners",
        content="Our global team of supply chain experts offers services spanning multiple industries.",
        chunks=[
            "Our global team of supply chain experts offers services spanning multiple industries and time zones."
        ],
        internal_links=[],
    )

    async def mock_extract(url, client=None):
        if "about" in url:
            return about_page
        return primary_page

    with patch("app.research.service.validate_url", return_value=(True, None)):
        with patch("app.research.service.extract_page", side_effect=mock_extract):
            with patch("app.research.service.synthesize_evidence") as mock_synth:
                mock_synth.return_value = (
                    "NCS Partners is a supply chain consulting firm.",
                    [],
                    False,
                    [],
                )
                resp = await conduct_research(req)

    assert resp.company_name == "NCS Partners"
    assert "https://www.ncspartners.com/in" in resp.sources_scraped
    assert all("ncspartners.com" in s for s in resp.sources_scraped)
    # Ensure third-party domains (like fleetx) were NOT scraped
    assert not any("fleetx" in s for s in resp.sources_scraped)


@pytest.mark.asyncio
async def test_conduct_research_rejects_ssrf_domain():
    """If user passes a domain that targets a private/internal IP, it should not be scraped."""
    req = ResearchRequest(
        company_name="Malicious Target",
        domain="http://169.254.169.254/latest/meta-data/",
    )

    with patch("app.research.service.extract_page") as mock_extract:
        with patch("app.research.service.synthesize_evidence") as mock_synth:
            mock_synth.return_value = (
                "Unable to gather sufficient public information.",
                [],
                True,
                ["No accessible public web sources"],
            )
            resp = await conduct_research(req)

    assert mock_extract.call_count == 0
    assert len(resp.sources_scraped) == 0
