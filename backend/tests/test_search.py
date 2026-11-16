from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.research.search import _decode_bing_url, _decode_ddg_url, search_company


def test_decode_bing_url():
    # Regular URL unchanged
    assert _decode_bing_url("https://stripe.com/payments") == "https://stripe.com/payments"

    # Base64 encoded redirect URL: https://stripe.com/
    # base64.urlsafe_b64encode(b"https://stripe.com/") -> b'aHR0cHM6Ly9zdHJpcGUuY29tLw=='
    encoded_href = "https://www.bing.com/ck/a?!&&p=123&u=a1aHR0cHM6Ly9zdHJpcGUuY29tLw&ntb=1"
    decoded = _decode_bing_url(encoded_href)
    assert decoded == "https://stripe.com/"


def test_decode_ddg_url():
    # Regular URL unchanged
    assert _decode_ddg_url("https://stripe.com") == "https://stripe.com"

    # DDG redirect format
    encoded_href = "/l/?uddg=https%3A%2F%2Fstripe.com%2Fpayments&rut=..."
    assert _decode_ddg_url(encoded_href) == "https://stripe.com/payments"


@pytest.mark.asyncio
async def test_search_company_ddg_success():
    mock_ddg_html = """
    <div class="result">
        <h2 class="result__title">
            <a href="https://stripe.com">Stripe Home</a>
        </h2>
        <div class="result__snippet">Financial infrastructure for the internet.</div>
    </div>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = mock_ddg_html

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_resp

    results = await search_company("Stripe", max_results=3, client=mock_client)

    assert len(results) == 1
    assert results[0].title == "Stripe Home"
    assert results[0].url == "https://stripe.com"
    assert "Financial infrastructure" in results[0].snippet
    assert results[0].source == "duckduckgo"


@pytest.mark.asyncio
async def test_search_company_fallback_to_bing_when_ddg_fails():
    mock_client = AsyncMock()
    # DDG fails with connection error
    mock_client.post.side_effect = httpx.ConnectError("DDG unreachable")

    # Bing succeeds
    mock_bing_html = """
    <html>
        <body>
            <ol id="b_results">
                <li class="b_algo">
                    <h2><a href="https://stripe.com/payments">Stripe Payments</a></h2>
                    <p>Global payment processing platform for online businesses.</p>
                </li>
            </ol>
        </body>
    </html>
    """
    mock_bing_resp = MagicMock()
    mock_bing_resp.status_code = 200
    mock_bing_resp.text = mock_bing_html
    mock_client.get.return_value = mock_bing_resp

    results = await search_company("Stripe", max_results=2, client=mock_client)

    assert len(results) == 1
    assert results[0].title == "Stripe Payments"
    assert results[0].url == "https://stripe.com/payments"
    assert results[0].source == "bing"
