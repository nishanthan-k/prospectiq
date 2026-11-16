"""
Security Module: URL Validation & SSRF Protection

Security Architecture & DNS Rebinding (TOCTOU) Assessment:
---------------------------------------------------------
1. Application-Level Pre-Flight Validation:
   The `validate_url()` function validates untrusted URLs against SSRF:
   - Enforces strictly allowed HTTP and HTTPS schemes.
   - Rejects credential injection (user:pass@host).
   - Rejects local, internal, and multicast hostnames (e.g. localhost, *.local, *.internal).
   - Rejects IP literals resolving to private (RFC 1918), loopback (127.0.0.0/8),
     link-local/cloud metadata (169.254.0.0/16), IPv6 unique-local (fc00::/7), and reserved ranges.
   - Pre-resolves hostnames using `socket.getaddrinfo()` to verify all resolved A/AAAA records
     point to globally routable public IPs.

2. DNS Rebinding / Time-of-Check to Time-of-Use (TOCTOU) Analysis:
   In asynchronous Python networking using `httpx` (built on `httpcore` and `anyio`), establishing
   an HTTP/HTTPS connection triggers an internal DNS resolution via the event loop's getaddrinfo.
   Between `validate_url()` and the actual socket connection, an attacker controlling an authoritative
   DNS server with zero TTL could theoretically return a public IP during pre-flight validation and
   subsequently return a private IP (e.g. 127.0.0.1 or 169.254.169.254) when httpx connects.

3. Connection-Level Pinning vs. Networking Fragility:
   Forcing `httpx` to connect directly to pre-resolved IPs while preserving original Host headers,
   HTTPS SNI (Server Name Indication), TLS certificate verification, connection pooling, and
   HTTP/2 negotiation requires implementing an intrusive custom `httpcore.AsyncHTTPTransport`
   or patching internal socket streams. Across Python/httpcore versions, custom socket stream
   interceptors are notoriously fragile and prone to breaking connection pool lifecycle,
   IPv4/IPv6 dual-stack fallback, and TLS hostname verification.

4. Production Security Boundary & Defense-in-Depth Recommendations:
   Before exposing company research and scraping endpoints to untrusted external users in production,
   the remaining DNS rebinding window must be mitigated at the infrastructure boundary:
   - Egress Firewall / Security Groups: Block egress traffic from scraper worker nodes to private
     ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.169.254) at the VPC/iptables level.
   - Dedicated Forward Proxy: Route outbound scraping requests through an egress proxy (e.g.,
     Envoy, Squid, or Smokescreen) configured to reject private IP egress after single DNS resolution.
   - Hardened Internal DNS: Employ resolving resolvers (e.g., CoreDNS, Unbound) with DNS rebinding
     filtering (`private-address` directive) enabled to strip RFC 1918 addresses from public DNS responses.
"""

import ipaddress
import socket
import urllib.parse
from typing import Optional, Union

DISALLOWED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "local",
    "internal",
    "broadcasthost",
}

DISALLOWED_SCHEMES = {"file", "ftp", "gopher", "data", "javascript", "ws", "wss", "ldap"}


def _is_disallowed_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    """Check if an IP address is private, loopback, link-local, reserved, or non-public."""
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_url(url: str, allow_dns_resolution: bool = True) -> tuple[bool, Optional[str]]:
    """
    Validate a URL against SSRF vulnerabilities:
    - Enforces HTTP/HTTPS scheme.
    - Rejects localhost and internal domains.
    - Rejects private, loopback, link-local, and reserved IP addresses.
    - Resolves hostnames via DNS and validates resolved IPs.
    Returns (is_valid, error_reason).
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string."

    url_clean = url.strip()
    try:
        parsed = urllib.parse.urlsplit(url_clean)
    except Exception as exc:
        return False, f"Failed to parse URL: {exc}"

    # 1. Scheme validation
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"Disallowed URL scheme '{scheme}'. Only http and https are permitted."

    # 2. Hostname validation
    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL."

    hostname = hostname.lower().strip(".")

    # Reject credential injection in URL (e.g., http://user:pass@host)
    if parsed.username or parsed.password:
        return False, "User credentials are not allowed in URLs."

    # Check known local/internal hostnames
    if (
        hostname in DISALLOWED_HOSTNAMES
        or hostname.endswith(".localhost")
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
        or hostname.endswith(".arpa")
    ):
        return False, f"Disallowed local or internal hostname '{hostname}'."

    # 3. Check IP address literal
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_disallowed_ip(ip):
            return False, f"Target IP '{ip}' is in a private, loopback, or reserved range."
        return True, None
    except ValueError:
        # Not an IP literal; it's a domain name
        pass

    # 4. Resolve DNS and check all resolved IP addresses
    if allow_dns_resolution:
        try:
            addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
            if not addr_info:
                return False, f"No DNS records found for hostname '{hostname}'."

            for family, socktype, proto, canonname, sockaddr in addr_info:
                ip_str = sockaddr[0]
                try:
                    ip = ipaddress.ip_address(ip_str)
                    if _is_disallowed_ip(ip):
                        return False, (
                            f"Hostname '{hostname}' resolved to disallowed IP '{ip}' "
                            f"(private, loopback, or reserved)."
                        )
                except ValueError:
                    continue
        except socket.gaierror as exc:
            return False, f"DNS resolution failed for hostname '{hostname}': {exc}"
        except Exception as exc:
            return False, f"Error validating hostname '{hostname}': {exc}"

    return True, None


def is_same_domain_or_subdomain(candidate_url_or_host: str, base_host: str) -> bool:
    """
    Safely check if candidate belongs to the exact base domain or a valid subdomain.
    Prevents hostname substring attacks like 'example.com.attacker.com'.
    """
    if not candidate_url_or_host or not base_host:
        return False

    cand = candidate_url_or_host.strip().lower()
    if "://" in cand or cand.startswith("//"):
        parsed = urllib.parse.urlsplit(cand)
        cand_host = parsed.hostname or ""
    else:
        cand_host = cand.split("/")[0].split(":")[0]

    base = base_host.strip().lower()
    if "://" in base or base.startswith("//"):
        parsed_base = urllib.parse.urlsplit(base)
        base_clean = parsed_base.hostname or ""
    else:
        base_clean = base.split("/")[0].split(":")[0]

    if base_clean.startswith("www."):
        base_clean = base_clean[4:]
    if cand_host.startswith("www."):
        cand_host = cand_host[4:]

    if not cand_host or not base_clean:
        return False

    # Exact match or strictly suffixed with '.<base>' (e.g. blog.stripe.com -> stripe.com)
    return cand_host == base_clean or cand_host.endswith(f".{base_clean}")
