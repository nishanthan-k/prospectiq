import asyncio
import re
import time
import urllib.parse
import uuid

from app.core.logger import log
from app.research.evidence import synthesize_evidence
from app.research.extractor import extract_page
from app.research.models import ExtractedPage, ResearchRequest, ResearchResponse
from app.research.ranker import rank_chunks
from app.research.search import search_company

MAX_SOURCES = 3


def _extract_clean_host(domain_or_url: str) -> str:
    """Extract clean hostname from URL or domain string."""
    d = domain_or_url.strip()
    if not d.startswith("http://") and not d.startswith("https://"):
        d = f"https://{d}"
    parsed = urllib.parse.urlparse(d)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


async def conduct_research(request: ResearchRequest) -> ResearchResponse:
    """
    Execute autonomous company research and retrieval:
    1. Search public web engines without API key (DuckDuckGo first).
    2. Prioritize primary company domain and discover internal company subpages.
    3. Filter out irrelevant third-party domains.
    4. Extract clean page contents using BeautifulSoup.
    5. Rank text chunks penalizing recruitment/career boilerplate.
    6. Synthesize verifiable citations with resilient parsing.
    """
    request_id = f"res_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    log.bind(
        request_id=request_id,
        company_name=request.company_name,
        domain=request.domain,
        has_summary=bool(request.summary),
    ).info(f"[{request_id}] Starting research for '{request.company_name}'")

    clean_host = _extract_clean_host(request.domain) if request.domain else ""

    # 1. Search query construction (concise query without raw URLs)
    search_query = f"{request.company_name} {clean_host}".strip()
    search_results = await search_company(search_query, max_results=5)

    # 2. URL selection: prioritize primary company domain
    urls_to_scrape: list[str] = []
    if request.domain:
        primary_url = (
            request.domain
            if request.domain.startswith("http://") or request.domain.startswith("https://")
            else f"https://{request.domain}"
        )
        urls_to_scrape.append(primary_url)

    # Add relevant search results (filter out unrelated third-party domains)
    company_tokens = [w for w in re.findall(r"\w+", request.company_name.lower()) if len(w) > 2]
    for sr in search_results:
        if sr.url in urls_to_scrape:
            continue
        sr_url_lower = sr.url.lower()
        sr_title_lower = sr.title.lower()
        sr_snip_lower = sr.snippet.lower()

        is_same_domain = bool(clean_host and clean_host in sr_url_lower)
        is_relevant_company = any(
            tok in sr_title_lower or tok in sr_snip_lower for tok in company_tokens
        )
        if is_same_domain or is_relevant_company:
            urls_to_scrape.append(sr.url)
        if len(urls_to_scrape) >= MAX_SOURCES:
            break

    # 3. Extract content from URLs concurrently
    extraction_tasks = [extract_page(url) for url in urls_to_scrape]
    extracted_pages: list[ExtractedPage] = []
    if extraction_tasks:
        results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, ExtractedPage):
                extracted_pages.append(res)

    # If primary page was extracted and we have capacity, scrape internal pages from target site
    if request.domain and extracted_pages and len(extracted_pages) < MAX_SOURCES:
        primary_page = extracted_pages[0]
        extra_internal = [
            link for link in primary_page.internal_links if link not in [p.url for p in extracted_pages]
        ]
        needed = MAX_SOURCES - len(extracted_pages)
        if extra_internal and needed > 0:
            extra_tasks = [extract_page(link) for link in extra_internal[:needed]]
            extra_results = await asyncio.gather(*extra_tasks, return_exceptions=True)
            for res in extra_results:
                if isinstance(res, ExtractedPage):
                    extracted_pages.append(res)

    sources_scraped = [p.url for p in extracted_pages]

    # 4. Collect and rank chunks
    chunk_tuples: list[tuple[str, str]] = []
    for page in extracted_pages:
        for chunk in page.chunks:
            chunk_tuples.append((page.url, chunk))

    ranked_chunks = rank_chunks(
        chunk_tuples,
        company_name=request.company_name,
        context_summary=request.summary,
        top_k=6,
    )

    # 5. Synthesize verifiable citations & check evidence sufficiency
    summary, citations, is_insufficient, insufficient_reasons = await synthesize_evidence(
        company_name=request.company_name,
        domain=request.domain,
        ranked_chunks=ranked_chunks,
        request_id=request_id,
        existing_summary=request.summary,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    log.bind(
        request_id=request_id,
        latency_ms=latency_ms,
        sources_scraped=len(sources_scraped),
        citations_count=len(citations),
        is_insufficient_evidence=is_insufficient,
    ).info(
        f"[{request_id}] Research completed in {latency_ms:.0f}ms: "
        f"citations={len(citations)}, insufficient={is_insufficient}"
    )

    return ResearchResponse(
        request_id=request_id,
        company_name=request.company_name,
        domain=request.domain,
        summary=summary,
        citations=citations,
        is_insufficient_evidence=is_insufficient,
        insufficient_reasons=insufficient_reasons,
        sources_scraped=sources_scraped,
        latency_ms=latency_ms,
    )

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    log.bind(
        request_id=request_id,
        latency_ms=latency_ms,
        sources_scraped=len(sources_scraped),
        citations_count=len(citations),
        is_insufficient_evidence=is_insufficient,
    ).info(
        f"[{request_id}] Research completed in {latency_ms:.0f}ms: "
        f"citations={len(citations)}, insufficient={is_insufficient}"
    )

    return ResearchResponse(
        request_id=request_id,
        company_name=request.company_name,
        domain=request.domain,
        summary=summary,
        citations=citations,
        is_insufficient_evidence=is_insufficient,
        insufficient_reasons=insufficient_reasons,
        sources_scraped=sources_scraped,
        latency_ms=latency_ms,
    )

