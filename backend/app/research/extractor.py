import re
import urllib.parse
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from app.core.logger import log
from app.research.models import ExtractedPage
from app.research.security import is_same_domain_or_subdomain, validate_url, validate_url_async

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
    "button",
    "input",
    "select",
    "textarea",
    "dialog",
]

DEFAULT_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

MAX_PAGE_CONTENT_BYTES = 2 * 1024 * 1024  # 2MB response limit


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip redundant newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_html_to_page(url: str, html_content: str) -> ExtractedPage:
    """
    Parse raw HTML into an ExtractedPage with cleaned text and chunks using BeautifulSoup.
    Prevents duplicate extraction from nested containers and strips boilerplate/banners.
    """
    soup = BeautifulSoup(html_content or "", "html.parser")

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
    parsed_base = urllib.parse.urlsplit(url)
    base_netloc = parsed_base.netloc.lower()
    internal_links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        joined = urllib.parse.urljoin(url, href)
        parsed_joined = urllib.parse.urlsplit(joined)

        # Validate that the link belongs strictly to the same domain or valid subdomain
        if is_same_domain_or_subdomain(parsed_joined.netloc, base_netloc):
            path_lower = parsed_joined.path.lower()
            if any(
                k in path_lower
                for k in [
                    "about",
                    "service",
                    "product",
                    "solution",
                    "overview",
                    "company",
                    "platform",
                ]
            ):
                clean_link = f"{parsed_joined.scheme}://{parsed_joined.netloc}{parsed_joined.path}"
                is_safe, _ = validate_url(clean_link, allow_dns_resolution=False)
                if is_safe and clean_link != url and clean_link not in internal_links:
                    internal_links.append(clean_link)

    # Extract high-density meta descriptions (standard and Open Graph) before decomposing tags
    meta_descriptions: list[str] = []
    meta_desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta_desc and meta_desc.get("content"):
        c = _clean_text(str(meta_desc["content"]))
        if len(c) >= 15:
            meta_descriptions.append(c)

    og_desc = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
    if og_desc and og_desc.get("content"):
        c = _clean_text(str(og_desc["content"]))
        if len(c) >= 15 and c not in meta_descriptions:
            meta_descriptions.append(c)

    # Remove unwanted tags
    for tag_name in UNWANTED_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Remove cookie, consent, and privacy banners
    for el in soup.find_all(True):
        attrs = getattr(el, "attrs", None)
        if not isinstance(attrs, dict):
            continue
        classes = (
            " ".join(attrs.get("class", []))
            if isinstance(attrs.get("class"), list)
            else str(attrs.get("class", ""))
        )
        element_id = str(attrs.get("id", ""))
        classes_and_id = f"{classes} {element_id}".lower()
        if any(
            b in classes_and_id
            for b in [
                "cookie-banner",
                "cookie-consent",
                "gdpr-banner",
                "privacy-banner",
                "cookie-notice",
                "cookie",
            ]
        ):
            el.decompose()

    # Extract text blocks from leaf-level content elements only
    # (Avoids duplicating text from outer <section>, <article>, or <div> wrappers)
    raw_blocks: list[str] = []
    seen_blocks: set[str] = set()

    # Add meta descriptions first as high-signal summary context
    for md in meta_descriptions:
        seen_blocks.add(md.lower())
        raw_blocks.append(md)

    leaf_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"}
    content_elements = soup.find_all(list(leaf_tags))

    if content_elements:
        for elem in content_elements:
            # Skip if this element contains child content elements to avoid duplication
            if elem.find(list(leaf_tags)):
                continue

            t = _clean_text(elem.get_text())
            t_lower = t.lower()
            if len(t) >= 15 and t_lower not in seen_blocks:
                seen_blocks.add(t_lower)
                raw_blocks.append(t)
    else:
        # Fallback to stripped strings if no structured tags found
        body = soup.find("body") or soup
        fallback_text = _clean_text(" ".join(body.stripped_strings))
        if fallback_text and len(fallback_text) >= 15:
            raw_blocks = [fallback_text]

    # Combine blocks and build chunks (chunks of ~300 to 600 chars)
    full_content = "\n\n".join(raw_blocks)
    chunks: list[str] = []

    current_chunk: list[str] = []
    current_length = 0

    for block in raw_blocks:
        if current_length + len(block) > 600 and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [block]
            current_length = len(block)
        else:
            current_chunk.append(block)
            current_length += len(block)

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

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
    max_redirects: int = 3,
    max_bytes: int = MAX_PAGE_CONTENT_BYTES,
) -> Optional[ExtractedPage]:
    """
    Fetch webpage HTML and extract clean text using BeautifulSoup.
    - Enforces SSRF validation on the initial URL and on all redirect hops.
    - Enforces HTTP streaming to read response bytes incrementally.
    - Stops reading as soon as max_bytes is reached (even with chunked or missing Content-Length).
    - Decodes text safely and parses with BeautifulSoup.
    """
    should_close_client = False
    if client is None:
        client = httpx.AsyncClient(headers=DEFAULT_FETCH_HEADERS, timeout=timeout_seconds)
        should_close_client = True

    try:
        current_url = url

        # Manual redirect loop to validate each hop against SSRF
        for _ in range(max_redirects + 1):
            is_valid_hop, hop_reason = await validate_url_async(
                current_url, allow_dns_resolution=True
            )
            if not is_valid_hop:
                log.warning(f"SSRF check rejected URL '{current_url}': {hop_reason}")
                return None

            async with client.stream("GET", current_url, follow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        log.warning(f"Redirect from {current_url} missing Location header.")
                        return None
                    current_url = urllib.parse.urljoin(current_url, location)
                    continue

                if response.status_code != 200:
                    log.warning(f"Failed to fetch {current_url}: HTTP {response.status_code}")
                    return None

                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    log.info(f"Skipping non-HTML resource {current_url} ({content_type})")
                    return None

                # Check Content-Length if present and valid
                content_length_hdr = response.headers.get("content-length")
                if content_length_hdr:
                    try:
                        cl = int(content_length_hdr)
                        if cl > max_bytes:
                            log.warning(
                                f"Response Content-Length {cl} exceeds limit {max_bytes} for {current_url}"
                            )
                            return None
                    except ValueError:
                        pass

                # Stream response bytes incrementally up to max_bytes
                chunks: list[bytes] = []
                total_bytes = 0
                exceeded = False

                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        log.warning(
                            f"Response size exceeded limit ({max_bytes} bytes) during streaming for {current_url}"
                        )
                        exceeded = True
                        break
                    chunks.append(chunk)

                if exceeded:
                    return None

                encoding = response.encoding or "utf-8"
                raw_bytes = b"".join(chunks)
                text_content = raw_bytes.decode(encoding, errors="replace")
                return parse_html_to_page(current_url, text_content)

        log.warning(f"Exceeded max redirects ({max_redirects}) for {url}")
        return None
    except Exception as exc:
        log.warning(f"Error extracting page from {url}: {exc}")
        return None
    finally:
        if should_close_client:
            await client.aclose()
