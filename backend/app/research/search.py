import base64
import urllib.parse
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from app.core.logger import log
from app.research.models import SearchResult

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _decode_bing_url(href: str) -> str:
    """Decode Bing redirect URL (/ck/a?...) to the original destination URL."""
    if "bing.com/ck/a?" not in href:
        return href
    try:
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        u_param = qs.get("u", [""])[0]
        if u_param.startswith("a1"):
            raw_b64 = u_param[2:]
            raw_b64 += "=" * ((4 - len(raw_b64) % 4) % 4)
            return base64.urlsafe_b64decode(raw_b64).decode("utf-8", errors="ignore")
    except Exception:
        pass
    return href


def _decode_ddg_url(href: str) -> str:
    """Decode DuckDuckGo redirect URL (/l/?uddg=...) to the original destination URL."""
    if "uddg=" not in href:
        return href
    try:
        parsed = urllib.parse.urlparse(href)
        qs = urllib.parse.parse_qs(parsed.query)
        return qs.get("uddg", [href])[0]
    except Exception:
        return href


async def _search_bing(
    client: httpx.AsyncClient, query: str, max_results: int
) -> list[SearchResult]:
    """Search Bing HTML without an API key."""
    url = f"https://www.bing.com/search?q={urllib.parse.quote_plus(query)}"
    response = await client.get(url, headers=DEFAULT_HEADERS, timeout=10.0, follow_redirects=True)
    if response.status_code != 200:
        log.warning(f"Bing search returned status {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[SearchResult] = []
    for li in soup.select("li.b_algo"):
        h2 = li.find("h2")
        a = h2.find("a") if h2 else None
        if not a or not a.get("href"):
            continue

        raw_href = a["href"]
        clean_url = _decode_bing_url(raw_href)
        if not clean_url.startswith("http"):
            continue

        title = a.get_text(strip=True)
        snippet_elem = li.find("p") or li.find("div", class_="b_caption")
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

        results.append(
            SearchResult(
                title=title,
                url=clean_url,
                snippet=snippet,
                source="bing",
            )
        )
        if len(results) >= max_results:
            break

    return results


async def _search_ddg(
    client: httpx.AsyncClient, query: str, max_results: int
) -> list[SearchResult]:
    """Search DuckDuckGo HTML without an API key."""
    url = "https://html.duckduckgo.com/html/"
    response = await client.post(
        url,
        data={"q": query},
        headers=DEFAULT_HEADERS,
        timeout=10.0,
        follow_redirects=True,
    )
    if response.status_code != 200:
        log.warning(f"DuckDuckGo search returned status {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[SearchResult] = []
    for res in soup.select(".result"):
        a = res.select_one(".result__title a")
        if not a or not a.get("href"):
            continue

        clean_url = _decode_ddg_url(a["href"])
        if not clean_url.startswith("http"):
            continue

        title = a.get_text(strip=True)
        snippet_elem = res.select_one(".result__snippet")
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

        results.append(
            SearchResult(
                title=title,
                url=clean_url,
                snippet=snippet,
                source="duckduckgo",
            )
        )
        if len(results) >= max_results:
            break

    return results


async def search_company(
    query: str,
    max_results: int = 5,
    client: Optional[httpx.AsyncClient] = None,
) -> list[SearchResult]:
    """
    Search for target company information across public search engines without API keys.
    Tries DuckDuckGo first; if empty or fails, falls back to Bing.
    """
    should_close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=12.0)
        should_close_client = True

    try:
        # 1. Try DuckDuckGo first (high accuracy, no-API-key HTML endpoint)
        try:
            results = await _search_ddg(client, query, max_results)
            if results:
                log.info(f"DuckDuckGo search returned {len(results)} results for '{query}'")
                return results
        except Exception as exc:
            log.warning(f"DuckDuckGo search failed for '{query}': {exc}; falling back to Bing")

        # 2. Fallback to Bing
        try:
            results = await _search_bing(client, query, max_results)
            if results:
                log.info(f"Bing search returned {len(results)} results for '{query}'")
                return results
        except Exception as exc:
            log.error(f"Bing search failed for '{query}': {exc}")

        return []
    finally:
        if should_close_client:
            await client.aclose()
