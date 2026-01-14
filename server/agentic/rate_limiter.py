"""
Unified Rate Limiter for Web Scraping Operations

Provides centralized rate limiting across all scrapers using aiometer.
Supports per-domain limits, global concurrency control, and adaptive throttling.

Based on scraping audit recommendations (2026-01-13).

Usage:
    from agentic.rate_limiter import get_rate_limiter, RateLimitedClient

    limiter = get_rate_limiter()
    async with RateLimitedClient(limiter) as client:
        results = await client.fetch_all(urls)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

import aiometer
import httpx

from .proxy_manager import get_proxy_manager, ProxyManager
from .retry_strategy import get_retry_strategy, UnifiedRetryStrategy

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class DomainConfig:
    """Rate limit configuration for a specific domain."""
    max_concurrent: int = 5
    max_per_second: float = 2.0
    timeout: float = 30.0
    max_retries: int = 3


# Domain-specific rate limits based on known restrictions
DEFAULT_DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    # Search engines - conservative limits
    "google.com": DomainConfig(max_concurrent=2, max_per_second=0.5),
    "duckduckgo.com": DomainConfig(max_concurrent=3, max_per_second=1.0),

    # Code/tech sites - moderate limits
    "github.com": DomainConfig(max_concurrent=5, max_per_second=2.0),
    "stackoverflow.com": DomainConfig(max_concurrent=5, max_per_second=2.0),
    "reddit.com": DomainConfig(max_concurrent=3, max_per_second=1.0),

    # Documentation sites - higher limits
    "docs.python.org": DomainConfig(max_concurrent=10, max_per_second=5.0),
    "developer.mozilla.org": DomainConfig(max_concurrent=10, max_per_second=5.0),

    # Industrial/technical sites
    "fanucamerica.com": DomainConfig(max_concurrent=3, max_per_second=1.0),
    "rockwellautomation.com": DomainConfig(max_concurrent=3, max_per_second=1.0),
    "plctalk.net": DomainConfig(max_concurrent=5, max_per_second=2.0),

    # Local services - higher limits
    "localhost": DomainConfig(max_concurrent=20, max_per_second=50.0),
    "127.0.0.1": DomainConfig(max_concurrent=20, max_per_second=50.0),
}

# Default for unknown domains
DEFAULT_CONFIG = DomainConfig(max_concurrent=5, max_per_second=2.0)


@dataclass
class FetchResult:
    """Result of a rate-limited fetch operation."""
    url: str
    success: bool
    status_code: Optional[int] = None
    content: Optional[bytes] = None
    content_type: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0


@dataclass
class RateLimiterStats:
    """Statistics for rate limiter operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_duration_ms: float = 0.0
    rate_limited_count: int = 0
    domains_accessed: Dict[str, int] = field(default_factory=dict)


class UnifiedRateLimiter:
    """
    Centralized rate limiter for all web scraping operations.

    Features:
    - Per-domain rate limiting using aiometer
    - Global concurrency control
    - Automatic retry with exponential backoff
    - Statistics tracking
    - Adaptive throttling on 429 responses
    """

    def __init__(
        self,
        domain_configs: Optional[Dict[str, DomainConfig]] = None,
        default_config: Optional[DomainConfig] = None,
        global_max_concurrent: int = 50,
        user_agent: str = "RecoveryBot/1.0 (Scraper; +https://recoverybot.app)"
    ):
        """
        Initialize the rate limiter.

        Args:
            domain_configs: Custom per-domain configurations
            default_config: Default config for unknown domains
            global_max_concurrent: Maximum concurrent requests across all domains
            user_agent: User-Agent string for all requests
        """
        self.domain_configs = {
            **DEFAULT_DOMAIN_CONFIGS,
            **(domain_configs or {})
        }
        self.default_config = default_config or DEFAULT_CONFIG
        self.global_max_concurrent = global_max_concurrent
        self.user_agent = user_agent

        # Adaptive rate tracking per domain
        self._domain_rates: Dict[str, float] = {}
        self._rate_lock = asyncio.Lock()

        # Statistics
        self._stats = RateLimiterStats()

        logger.info(
            f"UnifiedRateLimiter initialized: "
            f"global_max={global_max_concurrent}, "
            f"configured_domains={len(self.domain_configs)}"
        )

    def get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        # Handle ports for localhost
        if parsed.hostname in ("localhost", "127.0.0.1"):
            return parsed.hostname
        # Return full domain without www prefix
        domain = parsed.hostname or ""
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def get_config(self, url: str) -> DomainConfig:
        """Get rate limit configuration for a URL."""
        domain = self.get_domain(url)

        # Check exact match
        if domain in self.domain_configs:
            return self.domain_configs[domain]

        # Check parent domain match (e.g., docs.github.com -> github.com)
        for configured_domain, config in self.domain_configs.items():
            if domain.endswith(f".{configured_domain}") or domain == configured_domain:
                return config

        return self.default_config

    async def get_current_rate(self, domain: str) -> float:
        """Get current rate limit for domain (may be reduced due to 429s)."""
        async with self._rate_lock:
            if domain in self._domain_rates:
                return self._domain_rates[domain]
            config = self.domain_configs.get(domain, self.default_config)
            return config.max_per_second

    async def reduce_rate(self, domain: str, factor: float = 0.5):
        """Reduce rate limit for a domain (called on 429 response)."""
        async with self._rate_lock:
            config = self.domain_configs.get(domain, self.default_config)
            current = self._domain_rates.get(domain, config.max_per_second)
            new_rate = max(0.1, current * factor)  # Minimum 0.1 req/s
            self._domain_rates[domain] = new_rate
            self._stats.rate_limited_count += 1
            logger.warning(f"Rate limited on {domain}, reducing to {new_rate:.2f} req/s")

    async def restore_rate(self, domain: str, consecutive_success: int = 10):
        """Gradually restore rate limit after successful requests."""
        async with self._rate_lock:
            if domain not in self._domain_rates:
                return
            config = self.domain_configs.get(domain, self.default_config)
            current = self._domain_rates[domain]
            if current < config.max_per_second:
                new_rate = min(config.max_per_second, current * 1.1)
                self._domain_rates[domain] = new_rate

    async def fetch_url(
        self,
        client: httpx.AsyncClient,
        url: str,
        method: str = "GET",
        **kwargs
    ) -> FetchResult:
        """
        Fetch a single URL with rate limiting and retry logic.

        Args:
            client: httpx AsyncClient instance
            url: URL to fetch
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for httpx request

        Returns:
            FetchResult with response data or error
        """
        domain = self.get_domain(url)
        config = self.get_config(url)

        self._stats.total_requests += 1
        self._stats.domains_accessed[domain] = self._stats.domains_accessed.get(domain, 0) + 1

        start_time = time.time()
        retries = 0
        last_error = None

        while retries <= config.max_retries:
            try:
                # Set headers
                headers = kwargs.pop("headers", {})
                headers.setdefault("User-Agent", self.user_agent)

                # Make request
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, **kwargs)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, **kwargs)
                else:
                    response = await client.request(method, url, headers=headers, **kwargs)

                duration_ms = (time.time() - start_time) * 1000

                # Handle rate limiting
                if response.status_code == 429:
                    await self.reduce_rate(domain)
                    retries += 1
                    self._stats.retried_requests += 1
                    # Exponential backoff
                    await asyncio.sleep(2 ** retries)
                    continue

                # Success - gradually restore rate
                if retries == 0:
                    await self.restore_rate(domain)

                self._stats.successful_requests += 1
                self._stats.total_duration_ms += duration_ms

                return FetchResult(
                    url=url,
                    success=True,
                    status_code=response.status_code,
                    content=response.content,
                    content_type=response.headers.get("content-type"),
                    duration_ms=duration_ms,
                    retries=retries
                )

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                retries += 1
                self._stats.retried_requests += 1
                await asyncio.sleep(2 ** retries)

            except httpx.HTTPError as e:
                last_error = f"HTTP Error: {e}"
                retries += 1
                self._stats.retried_requests += 1
                await asyncio.sleep(2 ** retries)

            except Exception as e:
                last_error = f"Error: {e}"
                break

        # All retries exhausted
        duration_ms = (time.time() - start_time) * 1000
        self._stats.failed_requests += 1
        self._stats.total_duration_ms += duration_ms

        return FetchResult(
            url=url,
            success=False,
            error=last_error,
            duration_ms=duration_ms,
            retries=retries
        )

    async def fetch_all(
        self,
        urls: List[str],
        client: Optional[httpx.AsyncClient] = None,
        method: str = "GET",
        **kwargs
    ) -> List[FetchResult]:
        """
        Fetch multiple URLs with per-domain rate limiting.

        URLs are grouped by domain and each domain group is processed
        with its own rate limits, while all domains run concurrently.

        Args:
            urls: List of URLs to fetch
            client: Optional httpx client (created if not provided)
            method: HTTP method
            **kwargs: Additional request arguments

        Returns:
            List of FetchResult in same order as input URLs
        """
        if not urls:
            return []

        # Group URLs by domain
        from collections import defaultdict
        grouped: Dict[str, List[tuple]] = defaultdict(list)
        for i, url in enumerate(urls):
            domain = self.get_domain(url)
            grouped[domain].append((i, url))

        logger.info(f"Fetching {len(urls)} URLs across {len(grouped)} domains")

        # Prepare result array
        results: List[Optional[FetchResult]] = [None] * len(urls)

        async def fetch_domain_batch(
            http_client: httpx.AsyncClient,
            domain: str,
            url_tuples: List[tuple]
        ):
            """Fetch all URLs for a single domain with rate limiting."""
            config = self.domain_configs.get(domain, self.default_config)
            current_rate = await self.get_current_rate(domain)

            # Create fetch jobs
            jobs = [
                partial(self.fetch_url, http_client, url, method, **kwargs)
                for _, url in url_tuples
            ]

            # Run with aiometer rate limiting
            domain_results = await aiometer.run_all(
                jobs,
                max_at_once=config.max_concurrent,
                max_per_second=current_rate
            )

            # Store results at original indices
            for (original_idx, _), result in zip(url_tuples, domain_results):
                results[original_idx] = result

        # Create or use provided client
        if client is None:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as new_client:
                # Run all domain batches concurrently
                domain_tasks = [
                    fetch_domain_batch(new_client, domain, url_tuples)
                    for domain, url_tuples in grouped.items()
                ]
                await asyncio.gather(*domain_tasks)
        else:
            domain_tasks = [
                fetch_domain_batch(client, domain, url_tuples)
                for domain, url_tuples in grouped.items()
            ]
            await asyncio.gather(*domain_tasks)

        # Filter out any None results (shouldn't happen)
        return [r for r in results if r is not None]

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        total = self._stats.total_requests
        success_rate = (
            self._stats.successful_requests / total if total > 0 else 0
        )
        avg_duration = (
            self._stats.total_duration_ms / total if total > 0 else 0
        )

        return {
            "total_requests": self._stats.total_requests,
            "successful_requests": self._stats.successful_requests,
            "failed_requests": self._stats.failed_requests,
            "retried_requests": self._stats.retried_requests,
            "success_rate": round(success_rate, 3),
            "avg_duration_ms": round(avg_duration, 1),
            "rate_limited_count": self._stats.rate_limited_count,
            "domains_accessed": dict(self._stats.domains_accessed),
            "active_rate_adjustments": dict(self._domain_rates)
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._stats = RateLimiterStats()


class RateLimitedClient:
    """
    Convenience wrapper for httpx.AsyncClient with rate limiting and proxy support.

    Usage:
        limiter = get_rate_limiter()
        async with RateLimitedClient(limiter) as client:
            result = await client.get("https://example.com")
            results = await client.fetch_all([url1, url2, url3])
    """

    def __init__(
        self,
        limiter: UnifiedRateLimiter,
        timeout: float = 30.0,
        use_proxy: bool = True,
        **client_kwargs
    ):
        self.limiter = limiter
        self.timeout = timeout
        self.use_proxy = use_proxy
        self.client_kwargs = client_kwargs
        self._client: Optional[httpx.AsyncClient] = None
        self._proxy_url: Optional[str] = None
        self._proxy_manager: Optional[ProxyManager] = None

    async def __aenter__(self):
        # Get proxy if enabled and proxies are configured
        proxy_config = None
        if self.use_proxy:
            self._proxy_manager = get_proxy_manager()
            if self._proxy_manager.has_proxies():
                self._proxy_url = await self._proxy_manager.get_proxy()
                proxy_config = self._proxy_manager.get_proxy_config(self._proxy_url)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            proxy=proxy_config,
            **self.client_kwargs
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def get(self, url: str, **kwargs) -> FetchResult:
        """GET request with rate limiting."""
        result = await self.limiter.fetch_url(self._client, url, "GET", **kwargs)
        # Report proxy result for health tracking
        if self._proxy_manager and self._proxy_url:
            await self._proxy_manager.report_result(
                self._proxy_url,
                success=result.success,
                latency_ms=result.duration_ms if hasattr(result, 'duration_ms') else 0
            )
        return result

    async def post(self, url: str, **kwargs) -> FetchResult:
        """POST request with rate limiting."""
        result = await self.limiter.fetch_url(self._client, url, "POST", **kwargs)
        if self._proxy_manager and self._proxy_url:
            await self._proxy_manager.report_result(
                self._proxy_url,
                success=result.success,
                latency_ms=result.duration_ms if hasattr(result, 'duration_ms') else 0
            )
        return result

    async def fetch_all(self, urls: List[str], **kwargs) -> List[FetchResult]:
        """Fetch multiple URLs with rate limiting."""
        return await self.limiter.fetch_all(urls, self._client, **kwargs)


# Singleton instance
_rate_limiter: Optional[UnifiedRateLimiter] = None


def get_rate_limiter(
    custom_configs: Optional[Dict[str, DomainConfig]] = None,
    **kwargs
) -> UnifiedRateLimiter:
    """
    Get or create the global rate limiter instance.

    Args:
        custom_configs: Optional domain-specific configurations
        **kwargs: Additional arguments for UnifiedRateLimiter

    Returns:
        UnifiedRateLimiter instance
    """
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = UnifiedRateLimiter(
            domain_configs=custom_configs,
            **kwargs
        )

    return _rate_limiter


def reset_rate_limiter():
    """Reset the global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = None
