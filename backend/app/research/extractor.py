import re
import urllib.parse
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from app.core.logger import log
from app.research.models import ExtractedPage

UNWANTED_TAGS = [
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
    "svg",
    "form",
    "iframe",
]

DEFAULT_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip redundant newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_html_to_page(url: str, html_content: str) -> ExtractedPage:
    """Parse raw HTML into an ExtractedPage with cleaned text and chunks using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif soup.find("h1"):
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        title = url

    # Discover high-value internal links (services, about, products, solutions)
    parsed_base = urllib.parse.urlparse(url)
    base_netloc = parsed_base.netloc.lower()
    internal_links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        joined = urllib.parse.urljoin(url, href)
        parsed_joined = urllib.parse.urlparse(joined)
        if parsed_joined.netloc.lower() == base_netloc:
            path_lower = parsed_joined.path.lower()
            if any(
                k in path_lower
                for k in ["about", "service", "product", "solution", "overview", "company"]
            ):
                clean_link = f"{parsed_joined.scheme}://{parsed_joined.netloc}{parsed_joined.path}"
                if clean_link != url and clean_link not in internal_links:
                    internal_links.append(clean_link)

    # Remove unwanted tags
    for tag_name in UNWANTED_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Extract text blocks from paragraphs, headings, list items
    raw_blocks: list[str] = []
    content_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "article", "section"])

    if content_elements:
        for elem in content_elements:
            t = _clean_text(elem.get_text())
            if len(t) >= 30:  # Ignore trivial navigation fragments
                raw_blocks.append(t)
    else:
        # Fallback to stripped strings if no structured tags found
        body = soup.find("body") or soup
        fallback_text = _clean_text(" ".join(body.stripped_strings))
        if fallback_text:
            raw_blocks = [fallback_text]

    # Combine blocks and build chunks (chunks of ~300 to 800 chars)
    full_content = "\n\n".join(raw_blocks)
    chunks: list[str] = []

    current_chunk: list[str] = []
    current_length = 0

    for block in raw_blocks:
        if current_length + len(block) > 600 and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [block]
            current_length = len(block)
        else:
            current_chunk.append(block)
            current_length += len(block)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return ExtractedPage(
        url=url,
        title=title,
        content=full_content[:6000],  # Bound maximum page content
        chunks=chunks,
        internal_links=internal_links[:5],
    )


async def extract_page(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
    timeout_seconds: float = 8.0,
) -> Optional[ExtractedPage]:
    """Fetch webpage HTML and extract clean text using BeautifulSoup."""
    should_close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_FETCH_HEADERS, timeout=timeout_seconds)
        should_close_client = True

    try:
        response = await client.get(url, follow_redirects=True)
        if response.status_code != 200:
            log.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return None

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            log.info(f"Skipping non-HTML resource {url} ({content_type})")
            return None

        return parse_html_to_page(url, response.text)
    except Exception as exc:
        log.warning(f"Error extracting page from {url}: {exc}")
        return None
    finally:
        if should_close_client:
            await client.aclose()
