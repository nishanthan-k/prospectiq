from unittest.mock import AsyncMock, MagicMock, patch

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
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.is_redirect = False
    mock_client.get.return_value = mock_resp

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com/notfound", client=mock_client)
    assert page is None


@pytest.mark.asyncio
async def test_extract_page_skips_non_html():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.is_redirect = False
    mock_resp.headers = {"content-type": "application/pdf"}
    mock_client.get.return_value = mock_resp

    with patch("app.research.extractor.validate_url", return_value=(True, None)):
        page = await extract_page("https://example.com/file.pdf", client=mock_client)
    assert page is None


@pytest.mark.asyncio
async def test_extract_page_rejects_ssrf_on_initial_url():
    """extract_page must reject SSRF targets before issuing any HTTP request."""
    mock_client = AsyncMock()
    page = await extract_page("http://169.254.169.254/latest/meta-data", client=mock_client)
    assert page is None
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_extract_page_rejects_redirect_to_internal_address():
    """If a public URL redirects to a private/loopback address, extract_page must abort."""
    mock_client = AsyncMock()

    redirect_resp = MagicMock()
    redirect_resp.status_code = 302
    redirect_resp.is_redirect = True
    redirect_resp.headers = {"location": "http://127.0.0.1:8000/internal-metrics"}

    mock_client.get.return_value = redirect_resp

    # Initial URL passes validation, but redirect destination fails
    with patch("app.research.extractor.validate_url") as mock_val:
        # First call (initial URL) is valid; second call (redirect) is invalid
        mock_val.side_effect = [(True, None), (False, "Disallowed loopback IP")]
        page = await extract_page("https://public-site.com/redirect", client=mock_client)

    assert page is None
    # Only one request made; redirect was aborted before fetching 127.0.0.1
    assert mock_client.get.call_count == 1
