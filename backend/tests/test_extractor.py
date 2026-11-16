import httpx
from unittest.mock import AsyncMock, patch

import pytest

from app.research.extractor import extract_page, parse_html_to_page


def test_parse_html_to_page_strips_unwanted_tags():
    html = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Acme Corp | Enterprise Cloud</title>
            <script>console.log("analytics tracking");</script>
            <style>body { color: red; }</style>
        </head>
        <body>
            <nav><a href="/home">Home</a><a href="/login">Login</a></nav>
            <h1>Welcome to Acme Cloud Platform</h1>
            <p>Acme Corp provides automated cloud infrastructure for mid-market engineering teams.</p>
            <p>Our solutions include automated CI/CD pipelines, multi-region deployments, and SOC2 compliance monitoring.</p>
            <footer><p>© 2026 Acme Corp. All rights reserved.</p></footer>
        </body>
    </html>
    """
    page = parse_html_to_page("https://acme.com", html)

    assert page.title == "Acme Corp | Enterprise Cloud"
    assert "analytics tracking" not in page.content
    assert "body { color: red; }" not in page.content
    assert "© 2026 Acme Corp" not in page.content  # footer decomposed
    assert "Acme Corp provides automated cloud infrastructure" in page.content
    assert len(page.chunks) >= 1


def test_parse_html_to_page_prevents_nested_element_duplication():
    """Ensure text inside nested containers (<section><article><p>...</p></article></section>) is extracted once."""
    nested_html = """
    <html>
        <body>
            <section class="main-content">
                <article class="feature-card">
                    <h2>Continuous Integration Platform</h2>
                    <p>Automate build, test, and container deployments across hybrid clouds.</p>
                </article>
            </section>
        </body>
    </html>
    """
    page = parse_html_to_page("https://example.com/platform", nested_html)

    # Count occurrences: "Continuous Integration Platform" and the paragraph should appear exactly once
    assert page.content.count("Continuous Integration Platform") == 1
    assert page.content.count("Automate build, test, and container deployments across hybrid clouds.") == 1


def test_parse_html_to_page_strips_cookie_and_consent_banners():
    html_with_banners = """
    <html>
        <body>
            <div class="cookie-banner gdpr-notice">
                <p>We use cookies to improve your user experience and track analytics. Accept all cookies.</p>
            </div>
            <div id="privacy-banner">
                <p>Please review our updated privacy policy before browsing this site.</p>
            </div>
            <main>
                <h1>Autonomous Sales Intelligence</h1>
                <p>ProspectIQ discovers verified prospective accounts and qualifies market fit.</p>
            </main>
        </body>
    </html>
    """
    page = parse_html_to_page("https://example.com", html_with_banners)

    assert "Accept all cookies" not in page.content
    assert "updated privacy policy" not in page.content
    assert "ProspectIQ discovers verified prospective accounts" in page.content


def test_parse_html_to_page_deduplicates_repeated_content():
    """Repeated identical blocks across the page should not bloat extraction chunks."""
    repeated_html = """
    <html>
        <body>
            <p>Enterprise grade security with automated SOC2 compliance monitoring.</p>
            <p>Enterprise grade security with automated SOC2 compliance monitoring.</p>
            <p>Enterprise grade security with automated SOC2 compliance monitoring.</p>
            <p>Multi-tenant infrastructure supporting over 50,000 active customer tenants.</p>
        </body>
    </html>
    """
    page = parse_html_to_page("https://example.com", repeated_html)

    assert page.content.count("Enterprise grade security with automated SOC2 compliance monitoring.") == 1
    assert "Multi-tenant infrastructure" in page.content


def test_parse_html_to_page_handles_malformed_html():
    """Parser should never crash on malformed tags, unclosed brackets, or unexpected encoding."""
    malformed_cases = [
        "<html><body><div><h1>Unclosed heading<p>Unclosed paragraph<div><span>Deep</span>",
        "<<<<invalid>>>><<<<>>>",
        "",
        "   \n\t  ",
        "<html><head><title>Broken Title",
    ]
    for case in malformed_cases:
        page = parse_html_to_page("https://example.com", case)
        assert page.url == "https://example.com"
        assert isinstance(page.chunks, list)


def test_parse_html_to_page_handles_low_text_page():
    html_low = """
    <html>
        <body>
            <a href="/login">Log in</a>
            <span>Ok</span>
        </body>
    </html>
    """
    page = parse_html_to_page("https://example.com", html_low)
    assert len(page.chunks) == 0


@pytest.mark.asyncio
async def test_extract_page_handles_http_errors():
    transport = httpx.MockTransport(lambda req: httpx.Response(404))
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com/notfound", client=client)
    assert page is None


@pytest.mark.asyncio
async def test_extract_page_skips_non_html():
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4")
    )
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com/file.pdf", client=client)
    assert page is None


@pytest.mark.asyncio
async def test_extract_page_rejects_ssrf_on_initial_url():
    """extract_page must reject SSRF targets before issuing any HTTP request."""
    mock_client = AsyncMock()
    page = await extract_page("http://169.254.169.254/latest/meta-data", client=mock_client)
    assert page is None
    mock_client.stream.assert_not_called()


@pytest.mark.asyncio
async def test_extract_page_rejects_redirect_to_internal_address():
    """If a public URL redirects to a private/loopback address, extract_page must abort."""
    def handle_request(req):
        if "redirect" in str(req.url):
            return httpx.Response(302, headers={"location": "http://127.0.0.1:8000/internal-metrics"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<p>Internal</p>")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))

    with patch("app.research.extractor.validate_url") as mock_val:
        mock_val.side_effect = [(True, None), (False, "Disallowed loopback IP")]
        page = await extract_page("https://public-site.com/redirect", client=client)

    assert page is None


@pytest.mark.asyncio
async def test_extract_page_response_below_size_limit():
    """Responses smaller than max_bytes must download and parse successfully."""
    html_bytes = b"<html><body><p>Acme Corp provides automated cloud infrastructure.</p></body></html>"
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/html"}, content=html_bytes)
    )
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com", client=client, max_bytes=500)

    assert page is not None
    assert "Acme Corp provides automated cloud infrastructure." in page.content


@pytest.mark.asyncio
async def test_extract_page_response_exactly_at_size_limit():
    """Responses exactly equal to max_bytes must download and parse successfully."""
    html_bytes = b"<html><body><p>Acme Corp exactly at the size limit test.</p></body></html>"
    limit = len(html_bytes)
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/html"}, content=html_bytes)
    )
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com", client=client, max_bytes=limit)

    assert page is not None
    assert "Acme Corp exactly at the size limit test." in page.content


@pytest.mark.asyncio
async def test_extract_page_response_exceeding_size_limit():
    """Responses exceeding max_bytes must abort streaming and return None."""
    html_bytes = b"<html><body><p>Acme Corp payload that is definitely way too long for limit.</p></body></html>"
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/html"}, content=html_bytes)
    )
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com", client=client, max_bytes=30)

    assert page is None


@pytest.mark.asyncio
async def test_extract_page_missing_content_length_exceeding_limit():
    """Chunked response without Content-Length must abort when streaming exceeds limit."""
    content_chunks = [b"Chunk 1: 15B--", b"Chunk 2: 15B--", b"Chunk 3: 15B--"]
    # Without Content-Length header
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=iter(content_chunks),
        )
    )
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        # Limit is 25 bytes; chunk 1 + chunk 2 = 30 bytes > 25
        page = await extract_page("https://example.com", client=client, max_bytes=25)

    assert page is None


@pytest.mark.asyncio
async def test_extract_page_incorrect_content_length_exceeding_limit():
    """If Content-Length header is smaller than max_bytes but stream exceeds it, must abort."""
    long_content = b"A" * 500
    # Header claims 40 bytes, but body sends 500 bytes with limit 100
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200,
            headers={"content-type": "text/html", "content-length": "40"},
            content=long_content,
        )
    )
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com", client=client, max_bytes=100)

    assert page is None


@pytest.mark.asyncio
async def test_extract_page_empty_response():
    """Empty 0-byte response should be handled safely without error."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/html"}, content=b"")
    )
    client = httpx.AsyncClient(transport=transport)

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com", client=client)

    assert page is not None
    assert len(page.chunks) == 0


@pytest.mark.asyncio
async def test_extract_page_valid_redirect_followed():
    """Valid redirects to public URLs must be followed and parsed."""
    def handle_request(req):
        if "redirect" in str(req.url):
            return httpx.Response(302, headers={"location": "https://public-site.com/destination"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><h1>Destination Page</h1><p>Acme landing content.</p></body></html>",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://public-site.com/redirect", client=client)

    assert page is not None
    assert page.title == "Destination Page"
