import asyncio
import re
import time
import urllib.parse
import uuid

from app.core.logger import log
from app.research.evidence import synthesize_evidence
from app.research.extractor import extract_page
from app.research.models import ExtractedPage, ResearchRequest, ResearchResponse, SearchResult
from app.research.ranker import rank_chunks
from app.research.search import search_company
from app.research.security import is_same_domain_or_subdomain, validate_url

MAX_SOURCES = 3

GENERIC_COMPANY_SUFFIXES = {
    "private",
    "limited",
    "ltd",
    "pvt",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "co",
    "company",
    "llc",
    "group",
    "holdings",
    "services",
    "solutions",
}


def is_search_result_relevant(sr: SearchResult, company_name: str) -> bool:
    """
    Check if a search result's title, snippet, or URL mentions the company's brand name or acronym.
    Prevents scraping unrelated companies returned by search engine fuzzy autocorrect
    without breaking companies with shorter or acronym domains (e.g. IBM, 3M, TCS).
    """
    # 1. Distinctive brand tokens
    brand_tokens = [
        t for t in re.findall(r"\w+", company_name.lower())
        if len(t) >= 3 and t not in GENERIC_COMPANY_SUFFIXES
    ]
    if not brand_tokens:
        brand_tokens = [t for t in re.findall(r"\w+", company_name.lower()) if len(t) >= 3]

    content_to_check = f"{sr.title} {sr.snippet} {sr.url}".lower()

    # Check if any distinctive brand token appears in title, snippet, or URL
    if any(tok in content_to_check for tok in brand_tokens):
        return True

    # 2. Acronyms (e.g. TCS from Tata Consultancy Services, IBM from International Business Machines)
    all_words = [w for w in re.findall(r"\w+", company_name) if len(w) > 0]
    if len(all_words) >= 2:
        full_acronym = "".join(w[0] for w in all_words).lower()
        if len(full_acronym) >= 2 and re.search(rf"\b{re.escape(full_acronym)}\b", content_to_check):
            return True
        non_legal_words = [
            w for w in all_words
            if w.lower() not in {"private", "limited", "ltd", "pvt", "corp", "inc", "co"}
        ]
        if len(non_legal_words) >= 2:
            short_acronym = "".join(w[0] for w in non_legal_words).lower()
            if len(short_acronym) >= 2 and re.search(rf"\b{re.escape(short_acronym)}\b", content_to_check):
                return True

    return False


def _extract_clean_host(domain_or_url: str) -> str:
    """Extract clean hostname from URL or domain string."""
    d = domain_or_url.strip()
    if not d.startswith("http://") and not d.startswith("https://"):
        d = f"https://{d}"
    try:
        parsed = urllib.parse.urlsplit(d)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return d


async def conduct_research(request: ResearchRequest) -> ResearchResponse:
    """
    Execute autonomous company research and retrieval:
    1. If domain provided: prioritize target domain and discover internal subpages (/about, /services).
       Revalidate internal links and enforce SSRF protections.
    2. If domain omitted: search public engine to identify the official company domain.
    3. Extract clean page contents using BeautifulSoup.
    4. Rank text chunks using generic commercial offering indicators and penalizing recruitment.
    5. Synthesize verifiable citations with resilient parsing and token management.
    """
    request_id = f"res_{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    log.bind(
        request_id=request_id,
        company_name=request.company_name,
        domain=request.domain,
        has_summary=bool(request.summary),
    ).info(f"[{request_id}] Starting research for '{request.company_name}'")

    extracted_pages: list[ExtractedPage] = []

    # Case A: User provided target domain -> Domain-First Extraction
    if request.domain:
        clean_host = _extract_clean_host(request.domain)
        primary_url = (
            request.domain
            if request.domain.startswith("http://") or request.domain.startswith("https://")
            else f"https://{request.domain}"
        )

        is_safe, reason = validate_url(primary_url, allow_dns_resolution=True)
        if is_safe:
            primary_page = await extract_page(primary_url)
            if primary_page:
                extracted_pages.append(primary_page)

                # Crawl up to 2 high-value internal company subpages on the same domain
                if primary_page.internal_links:
                    needed = MAX_SOURCES - len(extracted_pages)
                    internal_candidates = [
                        link for link in primary_page.internal_links
                        if is_same_domain_or_subdomain(link, clean_host) and link != primary_url
                    ][:needed]

                    if internal_candidates:
                        sub_results = await asyncio.gather(
                            *[extract_page(link) for link in internal_candidates],
                            return_exceptions=True,
                        )
                        for res in sub_results:
                            if isinstance(res, ExtractedPage):
                                extracted_pages.append(res)
        else:
            log.warning(f"[{request_id}] Provided domain '{primary_url}' failed SSRF validation: {reason}")
    else:
        # Case B: Domain was omitted -> Search for company domain
        search_query = request.company_name
        search_results = await search_company(search_query, max_results=5)
        if not search_results:
            search_results = await search_company(f"{request.company_name} official website", max_results=5)

        urls_to_scrape: list[str] = []
        for sr in search_results:
            if not is_search_result_relevant(sr, request.company_name):
                log.info(
                    f"[{request_id}] Skipping unrelated search result '{sr.title}' ({sr.url}) "
                    f"for company '{request.company_name}'"
                )
                continue

            is_safe, _ = validate_url(sr.url, allow_dns_resolution=True)
            if is_safe and sr.url not in urls_to_scrape:
                urls_to_scrape.append(sr.url)
            if len(urls_to_scrape) >= MAX_SOURCES:
                break

        if urls_to_scrape:
            results = await asyncio.gather(
                *[extract_page(url) for url in urls_to_scrape],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, ExtractedPage):
                    extracted_pages.append(res)

    sources_scraped = [p.url for p in extracted_pages]

    # Collect and rank chunks from extracted pages
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

    # Synthesize verifiable citations & check evidence sufficiency
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

