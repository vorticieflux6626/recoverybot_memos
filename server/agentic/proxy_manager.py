"""
Proxy Manager - Intelligent Proxy Pool Rotation

Provides proxy rotation for web scraping with:
- Round-robin and weighted rotation strategies
- Health tracking with automatic removal of failing proxies
- Background health checks every 15 minutes
- Support for residential/datacenter proxy mix
- Graceful fallback to direct connection

Phase 4 of scraping consolidation (2026-01).

Usage:
    from agentic.proxy_manager import get_proxy_manager

    # Get proxy for request
    proxy = await get_proxy_manager().get_proxy()

    # Report result for health tracking
    await get_proxy_manager().report_result(proxy, success=True, latency_ms=150)

    # Use with httpx
    async with httpx.AsyncClient(proxies=proxy) as client:
        response = await client.get(url)
"""

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("agentic.proxy_manager")


# ============================================
# CONFIGURATION
# ============================================

class RotationStrategy(Enum):
    """Proxy rotation strategies."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED = "weighted"  # Prefer proxies with better success rates
    LEAST_USED = "least_used"


@dataclass
class ProxyConfig:
    """Configuration for proxy manager."""
    # Proxy list (comma-separated URLs)
    proxy_list: str = ""

    # Rotation strategy
    rotation_strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN

    # Health check interval (seconds)
    health_check_interval: int = 900  # 15 minutes

    # Health check timeout (seconds)
    health_check_timeout: float = 10.0

    # Minimum success rate to keep proxy active (0.0-1.0)
    min_success_rate: float = 0.5

    # Minimum requests before evaluating success rate
    min_requests_for_eval: int = 10

    # Enable fallback to direct connection
    fallback_to_direct: bool = True

    # Test URL for health checks
    health_check_url: str = "https://httpbin.org/ip"

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        """Load configuration from environment variables."""
        return cls(
            proxy_list=os.getenv("PROXY_LIST", ""),
            rotation_strategy=RotationStrategy(
                os.getenv("PROXY_ROTATION_STRATEGY", "round_robin")
            ),
            health_check_interval=int(os.getenv("PROXY_HEALTH_CHECK_INTERVAL", "900")),
            health_check_timeout=float(os.getenv("PROXY_HEALTH_CHECK_TIMEOUT", "10.0")),
            min_success_rate=float(os.getenv("PROXY_MIN_SUCCESS_RATE", "0.5")),
            fallback_to_direct=os.getenv("PROXY_FALLBACK_TO_DIRECT", "true").lower() == "true",
        )


# ============================================
# PROXY HEALTH TRACKING
# ============================================

@dataclass
class ProxyHealth:
    """Health statistics for a single proxy."""
    url: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_used: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_passed: bool = True
    consecutive_failures: int = 0
    is_active: bool = True

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0  # Assume healthy until proven otherwise
        return self.successful_requests / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.last_used = datetime.now(timezone.utc)
        self.last_success = datetime.now(timezone.utc)
        self.consecutive_failures = 0

    def record_failure(self, error: Optional[str] = None):
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_used = datetime.now(timezone.utc)
        self.last_failure = datetime.now(timezone.utc)
        self.consecutive_failures += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "url": self._mask_url(self.url),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "consecutive_failures": self.consecutive_failures,
            "is_active": self.is_active,
            "health_check_passed": self.health_check_passed,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }

    def _mask_url(self, url: str) -> str:
        """Mask proxy URL for logging (hide credentials)."""
        try:
            parsed = urlparse(url)
            if parsed.password:
                return f"{parsed.scheme}://{parsed.username}:****@{parsed.hostname}:{parsed.port}"
            return url
        except Exception:
            return "****"


# ============================================
# PROXY POOL MANAGER
# ============================================

class ProxyManager:
    """
    Manages a pool of proxies with rotation and health tracking.

    Features:
    - Multiple rotation strategies (round-robin, random, weighted)
    - Automatic health checking every 15 minutes
    - Removes unhealthy proxies from rotation
    - Tracks success/failure rates per proxy
    - Graceful fallback to direct connection
    """

    def __init__(self, config: Optional[ProxyConfig] = None):
        """
        Initialize proxy manager.

        Args:
            config: Proxy configuration (loads from env if not provided)
        """
        self.config = config or ProxyConfig.from_env()

        # Parse proxy list
        self._proxies: Dict[str, ProxyHealth] = {}
        self._parse_proxy_list()

        # Rotation state
        self._rotation_index = 0
        self._lock = asyncio.Lock()

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._total_requests = 0
        self._direct_requests = 0
        self._proxy_requests = 0

        logger.info(
            f"ProxyManager initialized: {len(self._proxies)} proxies, "
            f"strategy={self.config.rotation_strategy.value}, "
            f"fallback_to_direct={self.config.fallback_to_direct}"
        )

    def _parse_proxy_list(self):
        """Parse proxy list from config."""
        if not self.config.proxy_list:
            logger.warning("No proxies configured - will use direct connections")
            return

        for proxy_url in self.config.proxy_list.split(","):
            proxy_url = proxy_url.strip()
            if proxy_url:
                self._proxies[proxy_url] = ProxyHealth(url=proxy_url)
                logger.debug(f"Added proxy: {self._proxies[proxy_url]._mask_url(proxy_url)}")

    # ============================================
    # PROXY SELECTION
    # ============================================

    async def get_proxy(self) -> Optional[str]:
        """
        Get the next proxy URL based on rotation strategy.

        Returns:
            Proxy URL string for httpx, or None for direct connection
        """
        async with self._lock:
            self._total_requests += 1

            # Get active proxies
            active_proxies = self._get_active_proxies()

            if not active_proxies:
                if self.config.fallback_to_direct:
                    self._direct_requests += 1
                    logger.debug("No active proxies, using direct connection")
                    return None
                else:
                    raise RuntimeError("No active proxies available and fallback disabled")

            # Select proxy based on strategy
            proxy_url = self._select_proxy(active_proxies)
            self._proxy_requests += 1

            return proxy_url

    def _get_active_proxies(self) -> List[ProxyHealth]:
        """Get list of active proxies."""
        active = []
        for proxy in self._proxies.values():
            if not proxy.is_active:
                continue
            if not proxy.health_check_passed:
                continue
            # Check if success rate is too low (only after min requests)
            if (proxy.total_requests >= self.config.min_requests_for_eval and
                proxy.success_rate < self.config.min_success_rate):
                proxy.is_active = False
                logger.warning(
                    f"Deactivating proxy {proxy._mask_url(proxy.url)}: "
                    f"success_rate={proxy.success_rate:.2%} < {self.config.min_success_rate:.2%}"
                )
                continue
            # Check for too many consecutive failures
            if proxy.consecutive_failures >= 5:
                proxy.is_active = False
                logger.warning(
                    f"Deactivating proxy {proxy._mask_url(proxy.url)}: "
                    f"{proxy.consecutive_failures} consecutive failures"
                )
                continue
            active.append(proxy)
        return active

    def _select_proxy(self, active_proxies: List[ProxyHealth]) -> str:
        """Select proxy based on rotation strategy."""
        if self.config.rotation_strategy == RotationStrategy.ROUND_ROBIN:
            proxy = active_proxies[self._rotation_index % len(active_proxies)]
            self._rotation_index += 1
            return proxy.url

        elif self.config.rotation_strategy == RotationStrategy.RANDOM:
            proxy = random.choice(active_proxies)
            return proxy.url

        elif self.config.rotation_strategy == RotationStrategy.WEIGHTED:
            # Weight by success rate
            weights = [p.success_rate + 0.1 for p in active_proxies]  # Add 0.1 to avoid zero weight
            proxy = random.choices(active_proxies, weights=weights, k=1)[0]
            return proxy.url

        elif self.config.rotation_strategy == RotationStrategy.LEAST_USED:
            # Prefer proxies with fewer requests
            proxy = min(active_proxies, key=lambda p: p.total_requests)
            return proxy.url

        else:
            # Default to round-robin
            proxy = active_proxies[self._rotation_index % len(active_proxies)]
            self._rotation_index += 1
            return proxy.url

    # ============================================
    # RESULT REPORTING
    # ============================================

    async def report_result(
        self,
        proxy_url: Optional[str],
        success: bool,
        latency_ms: float = 0.0,
        error: Optional[str] = None
    ):
        """
        Report the result of using a proxy.

        Args:
            proxy_url: The proxy URL used (None for direct)
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
            error: Error message if failed
        """
        if proxy_url is None:
            return  # Direct connection, nothing to track

        async with self._lock:
            if proxy_url not in self._proxies:
                return

            proxy = self._proxies[proxy_url]
            if success:
                proxy.record_success(latency_ms)
            else:
                proxy.record_failure(error)

    # ============================================
    # HEALTH CHECKING
    # ============================================

    async def start_health_checks(self):
        """Start background health check task."""
        if self._running:
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started proxy health check background task")

    async def stop_health_checks(self):
        """Stop background health check task."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        logger.info("Stopped proxy health check background task")

    async def _health_check_loop(self):
        """Background loop for health checks."""
        while self._running:
            try:
                await self.run_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def run_health_checks(self):
        """Run health checks on all proxies."""
        if not self._proxies:
            return

        logger.info(f"Running health checks on {len(self._proxies)} proxies...")

        async def check_proxy(proxy: ProxyHealth):
            try:
                async with httpx.AsyncClient(
                    proxy=proxy.url,
                    timeout=self.config.health_check_timeout
                ) as client:
                    start = time.time()
                    response = await client.get(self.config.health_check_url)
                    latency = (time.time() - start) * 1000

                    if response.status_code == 200:
                        proxy.health_check_passed = True
                        proxy.last_health_check = datetime.now(timezone.utc)
                        logger.debug(
                            f"Health check passed: {proxy._mask_url(proxy.url)} "
                            f"({latency:.0f}ms)"
                        )
                        return True
                    else:
                        proxy.health_check_passed = False
                        logger.warning(
                            f"Health check failed: {proxy._mask_url(proxy.url)} "
                            f"(status={response.status_code})"
                        )
                        return False
            except Exception as e:
                proxy.health_check_passed = False
                logger.warning(
                    f"Health check error: {proxy._mask_url(proxy.url)} ({e})"
                )
                return False

        # Run health checks concurrently
        tasks = [check_proxy(proxy) for proxy in self._proxies.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        passed = sum(1 for r in results if r is True)
        logger.info(f"Health checks complete: {passed}/{len(self._proxies)} passed")

    # ============================================
    # STATISTICS
    # ============================================

    def get_stats(self) -> Dict[str, Any]:
        """Get proxy manager statistics."""
        active_count = len(self._get_active_proxies())
        return {
            "total_proxies": len(self._proxies),
            "active_proxies": active_count,
            "inactive_proxies": len(self._proxies) - active_count,
            "rotation_strategy": self.config.rotation_strategy.value,
            "fallback_to_direct": self.config.fallback_to_direct,
            "total_requests": self._total_requests,
            "proxy_requests": self._proxy_requests,
            "direct_requests": self._direct_requests,
            "proxy_usage_rate": self._proxy_requests / self._total_requests if self._total_requests > 0 else 0,
            "proxies": [p.to_dict() for p in self._proxies.values()],
        }

    def get_active_proxy_count(self) -> int:
        """Get count of active proxies."""
        return len(self._get_active_proxies())

    def has_proxies(self) -> bool:
        """Check if any proxies are configured."""
        return len(self._proxies) > 0

    # ============================================
    # HTTPX INTEGRATION HELPERS
    # ============================================

    def get_proxy_config(self, proxy_url: Optional[str]) -> Optional[str]:
        """
        Get httpx proxy configuration.

        Args:
            proxy_url: Proxy URL or None for direct

        Returns:
            Proxy URL string for httpx 'proxy' parameter, or None

        Note:
            httpx 0.28+ uses 'proxy' parameter (string) instead of 'proxies' (dict).
            Use with: httpx.AsyncClient(proxy=get_proxy_config(url))
        """
        return proxy_url


# ============================================
# SINGLETON INSTANCE
# ============================================

_proxy_manager_instance: Optional[ProxyManager] = None


def get_proxy_manager() -> ProxyManager:
    """Get singleton proxy manager instance."""
    global _proxy_manager_instance
    if _proxy_manager_instance is None:
        _proxy_manager_instance = ProxyManager()
    return _proxy_manager_instance


def reset_proxy_manager():
    """Reset singleton instance (for testing)."""
    global _proxy_manager_instance
    _proxy_manager_instance = None


# ============================================
# CONTEXT MANAGER FOR EASY USE
# ============================================

class ProxiedClient:
    """
    Async context manager for httpx client with proxy rotation.

    Usage:
        async with ProxiedClient() as (client, proxy_url):
            response = await client.get(url)
            # Success/failure automatically reported
    """

    def __init__(
        self,
        timeout: float = 30.0,
        follow_redirects: bool = True,
        headers: Optional[Dict[str, str]] = None,
        **client_kwargs
    ):
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.headers = headers or {}
        self.client_kwargs = client_kwargs
        self._client: Optional[httpx.AsyncClient] = None
        self._proxy_url: Optional[str] = None
        self._start_time: float = 0

    async def __aenter__(self) -> tuple[httpx.AsyncClient, Optional[str]]:
        """Enter context and get proxied client."""
        manager = get_proxy_manager()
        self._proxy_url = await manager.get_proxy()
        self._start_time = time.time()

        proxy_config = manager.get_proxy_config(self._proxy_url)

        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            headers=self.headers,
            proxy=proxy_config,
            **self.client_kwargs
        )

        return self._client, self._proxy_url

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and report result."""
        latency_ms = (time.time() - self._start_time) * 1000

        manager = get_proxy_manager()
        if exc_type is None:
            await manager.report_result(self._proxy_url, success=True, latency_ms=latency_ms)
        else:
            await manager.report_result(
                self._proxy_url,
                success=False,
                latency_ms=latency_ms,
                error=str(exc_val)
            )

        if self._client:
            await self._client.aclose()


# ============================================
# CLI INTERFACE
# ============================================

async def main():
    """CLI interface for testing proxy manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Proxy Manager CLI")
    parser.add_argument("--stats", action="store_true", help="Show proxy stats")
    parser.add_argument("--health-check", action="store_true", help="Run health checks")
    parser.add_argument("--test-url", help="Test URL to fetch through proxies")

    args = parser.parse_args()

    manager = get_proxy_manager()

    if args.health_check:
        await manager.run_health_checks()

    if args.test_url:
        print(f"Testing URL: {args.test_url}")
        async with ProxiedClient() as (client, proxy_url):
            print(f"Using proxy: {proxy_url or 'direct'}")
            response = await client.get(args.test_url)
            print(f"Status: {response.status_code}")
            print(f"Content length: {len(response.content)}")

    if args.stats or not (args.health_check or args.test_url):
        import json
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
