import socket
from unittest.mock import patch

from app.research.security import is_same_domain_or_subdomain, validate_url


def test_validate_url_valid_public_urls():
    # DNS resolution disabled for offline unit test stability
    valid_urls = [
        "https://stripe.com",
        "https://www.ncspartners.com/in",
        "http://example.com/about-us",
        "https://sub.domain.company.org/solutions/platform",
        "https://github.com/torvalds/linux",
    ]
    for url in valid_urls:
        is_valid, reason = validate_url(url, allow_dns_resolution=False)
        assert is_valid is True, f"Expected {url} to be valid, got reason: {reason}"
        assert reason is None


def test_validate_url_invalid_schemes():
    invalid_scheme_urls = [
        "ftp://example.com/file.txt",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "gopher://gopher.floodgap.com",
        "data:text/html,<h1>Pwned</h1>",
        "ws://example.com/feed",
        "ldap://example.com/dc=test",
    ]
    for url in invalid_scheme_urls:
        is_valid, reason = validate_url(url, allow_dns_resolution=False)
        assert is_valid is False
        assert "Only http and https are permitted" in reason


def test_validate_url_rejects_credentials_in_url():
    is_valid, reason = validate_url("http://admin:secret@example.com", allow_dns_resolution=False)
    assert is_valid is False
    assert "credentials are not allowed" in reason


def test_validate_url_rejects_localhost_and_internal_domains():
    disallowed_domains = [
        "http://localhost",
        "http://localhost:8000/api",
        "https://mycompany.local/dashboard",
        "http://service.internal/secret",
        "http://broadcasthost",
        "http://dev.localhost/test",
    ]
    for url in disallowed_domains:
        is_valid, reason = validate_url(url, allow_dns_resolution=False)
        assert is_valid is False
        assert "Disallowed local or internal hostname" in reason


def test_validate_url_rejects_private_and_loopback_ip_literals():
    disallowed_ips = [
        "http://127.0.0.1:8000",
        "http://127.0.0.2",
        "https://10.0.0.1/admin",
        "http://192.168.1.254/router",
        "http://172.16.0.5",
        "http://172.31.255.255",
        "http://169.254.169.254/latest/meta-data/",  # Cloud metadata
        "http://[::1]:8080",  # IPv6 loopback
        "http://[fc00::1]/internal",  # IPv6 private
        "http://0.0.0.0",
    ]
    for url in disallowed_ips:
        is_valid, reason = validate_url(url, allow_dns_resolution=False)
        assert is_valid is False
        assert "private, loopback, or reserved" in reason


def test_validate_url_dns_resolution_detects_private_ip():
    """If a public-looking domain name resolves to a private IP, it must be rejected."""
    # Mock socket.getaddrinfo to simulate malicious DNS rebinding
    mock_addrinfo = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 0)),
    ]
    with patch("socket.getaddrinfo", return_value=mock_addrinfo):
        is_valid, reason = validate_url("https://malicious-rebinding.example.com", allow_dns_resolution=True)
        assert is_valid is False
        assert "resolved to disallowed IP" in reason


def test_validate_url_dns_resolution_passes_public_ip():
    mock_addrinfo = [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0)),  # example.com
    ]
    with patch("socket.getaddrinfo", return_value=mock_addrinfo):
        is_valid, reason = validate_url("https://example.com", allow_dns_resolution=True)
        assert is_valid is True
        assert reason is None


def test_is_same_domain_or_subdomain_exact_matches():
    assert is_same_domain_or_subdomain("https://stripe.com/about", "stripe.com") is True
    assert is_same_domain_or_subdomain("http://www.stripe.com/pricing", "stripe.com") is True
    assert is_same_domain_or_subdomain("ncspartners.com", "https://www.ncspartners.com/in") is True


def test_is_same_domain_or_subdomain_valid_subdomains():
    assert is_same_domain_or_subdomain("https://blog.stripe.com/post", "stripe.com") is True
    assert is_same_domain_or_subdomain("docs.stripe.com", "stripe.com") is True
    assert is_same_domain_or_subdomain("https://in.ncspartners.com", "ncspartners.com") is True


def test_is_same_domain_or_subdomain_rejects_hostname_substring_attacks():
    """Attackers registering example.com.attacker.com must NOT match example.com."""
    assert is_same_domain_or_subdomain("https://stripe.com.attacker.com", "stripe.com") is False
    assert is_same_domain_or_subdomain("http://fakestripe.com", "stripe.com") is False
    assert is_same_domain_or_subdomain("https://ncspartners.com.evil.co", "ncspartners.com") is False
    assert is_same_domain_or_subdomain("https://not-ncspartners.com", "ncspartners.com") is False
