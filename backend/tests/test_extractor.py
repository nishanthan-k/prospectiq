from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_extract_page_handles_http_errors():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_client.get.return_value = mock_resp

    page = await extract_page("https://example.com/notfound", client=mock_client)
    assert page is None


@pytest.mark.asyncio
async def test_extract_page_skips_non_html():
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/pdf"}
    mock_client.get.return_value = mock_resp

    page = await extract_page("https://example.com/file.pdf", client=mock_client)
    assert page is None
