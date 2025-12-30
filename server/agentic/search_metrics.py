"""
Search and Scrape Metrics Tracking

Provides detailed logging and metrics for:
1. Search provider performance (SearXNG, DuckDuckGo, Brave)
2. Rate limit tracking with exponential backoff
3. Scrape success/failure by domain
4. Engine cycling intelligence

Usage:
    from agentic.search_metrics import SearchMetrics, get_search_metrics

    metrics = get_search_metrics()
    metrics.record_search("searxng", query, results_count, duration_ms)
    metrics.record_rate_limit("duckduckgo")
    metrics.record_scrape("stackoverflow.com", success=True, content_length=5000)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import threading

logger = logging.getLogger(__name__)


@dataclass
class ProviderStats:
    """Statistics for a single search provider"""
    name: str
    total_searches: int = 0
    successful_searches: int = 0
    failed_searches: int = 0
    rate_limit_hits: int = 0
    total_results: int = 0
    total_duration_ms: float = 0.0
    last_rate_limit: Optional[datetime] = None
    rate_limit_backoff_until: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_searches == 0:
            return 0.0
        return self.successful_searches / self.total_searches

    @property
    def avg_duration_ms(self) -> float:
        if self.successful_searches == 0:
            return 0.0
        return self.total_duration_ms / self.successful_searches

    @property
    def avg_results_per_search(self) -> float:
        if self.successful_searches == 0:
            return 0.0
        return self.total_results / self.successful_searches

    def is_rate_limited(self) -> bool:
        """Check if provider is in rate limit backoff period"""
        if self.rate_limit_backoff_until is None:
            return False
        return datetime.now(timezone.utc) < self.rate_limit_backoff_until

    def get_backoff_seconds(self) -> int:
        """Calculate exponential backoff duration based on recent rate limits"""
        if self.rate_limit_hits == 0:
            return 0

        # Base: 5s, doubles each hit, max 5 minutes
        base_backoff = 5
        max_backoff = 300
        backoff = min(base_backoff * (2 ** (self.rate_limit_hits - 1)), max_backoff)
        return int(backoff)


@dataclass
class DomainStats:
    """Statistics for scraping a single domain"""
    domain: str
    total_attempts: int = 0
    successful_scrapes: int = 0
    failed_scrapes: int = 0
    total_content_chars: int = 0
    total_duration_ms: float = 0.0
    failure_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_scrapes / self.total_attempts

    @property
    def avg_content_chars(self) -> int:
        if self.successful_scrapes == 0:
            return 0
        return int(self.total_content_chars / self.successful_scrapes)


@dataclass
class EngineStats:
    """Statistics for a SearXNG internal engine (google, bing, etc.)"""
    name: str
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_results: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    backoff_until: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 1.0  # Assume available if never used
        return self.successful_queries / self.total_queries

    def is_in_backoff(self) -> bool:
        if self.backoff_until is None:
            return False
        return datetime.now(timezone.utc) < self.backoff_until

    def get_backoff_duration(self) -> int:
        """Calculate backoff based on recent failures"""
        if self.failed_queries == 0:
            return 0
        # Base: 30s for CAPTCHA, 60s for timeout, doubles each consecutive failure
        # Max 10 minutes
        base = 30
        return min(base * (2 ** min(self.failed_queries - 1, 4)), 600)


class SearchMetrics:
    """
    Thread-safe metrics tracking for search and scrape operations.

    Features:
    - Per-provider search statistics
    - Rate limit tracking with exponential backoff
    - Per-engine tracking for SearXNG internal engines
    - Per-domain scrape success/failure rates
    - Query logging for debugging
    - Provider selection recommendations
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._providers: Dict[str, ProviderStats] = {}
        self._domains: Dict[str, DomainStats] = {}
        self._engines: Dict[str, EngineStats] = {}  # SearXNG internal engines
        self._recent_queries: List[Dict] = []  # Rolling log of recent queries
        self._max_recent_queries = 100
        self._start_time = datetime.now(timezone.utc)

        # Initialize known providers
        for name in ["searxng", "duckduckgo", "brave"]:
            self._providers[name] = ProviderStats(name=name)

        # Initialize known SearXNG engines
        for name in ["google", "bing", "duckduckgo", "brave", "wikipedia",
                     "arxiv", "semantic_scholar", "google_scholar", "pubmed",
                     "github", "stackoverflow", "pypi", "npm", "dockerhub"]:
            self._engines[name] = EngineStats(name=name)

    def record_search(
        self,
        provider: str,
        query: str,
        results_count: int,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Record a search operation"""
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderStats(name=provider)

            stats = self._providers[provider]
            stats.total_searches += 1

            if success:
                stats.successful_searches += 1
                stats.total_results += results_count
                stats.total_duration_ms += duration_ms
            else:
                stats.failed_searches += 1

            # Log to recent queries
            self._recent_queries.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "query": query[:100],
                "results": results_count,
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "error": error
            })

            # Trim to max size
            if len(self._recent_queries) > self._max_recent_queries:
                self._recent_queries = self._recent_queries[-self._max_recent_queries:]

        # Log for visibility
        status = "✓" if success else "✗"
        logger.info(
            f"[SEARCH] {status} {provider}: '{query[:50]}...' → {results_count} results "
            f"({duration_ms:.0f}ms)"
        )

    def record_rate_limit(self, provider: str) -> int:
        """
        Record a rate limit hit and calculate backoff.

        Returns:
            Backoff duration in seconds
        """
        with self._lock:
            if provider not in self._providers:
                self._providers[provider] = ProviderStats(name=provider)

            stats = self._providers[provider]
            stats.rate_limit_hits += 1
            stats.last_rate_limit = datetime.now(timezone.utc)

            backoff_seconds = stats.get_backoff_seconds()
            stats.rate_limit_backoff_until = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)

        logger.warning(
            f"[RATE_LIMIT] {provider} rate limited (hit #{stats.rate_limit_hits}), "
            f"backing off for {backoff_seconds}s"
        )

        return backoff_seconds

    def reset_rate_limit(self, provider: str) -> None:
        """Reset rate limit counter after successful search"""
        with self._lock:
            if provider in self._providers:
                stats = self._providers[provider]
                # Decay the hit counter gradually
                if stats.rate_limit_hits > 0:
                    stats.rate_limit_hits = max(0, stats.rate_limit_hits - 1)
                stats.rate_limit_backoff_until = None

    def record_scrape(
        self,
        domain: str,
        success: bool,
        content_length: int = 0,
        duration_ms: float = 0.0,
        failure_reason: Optional[str] = None
    ) -> None:
        """Record a scrape operation"""
        with self._lock:
            if domain not in self._domains:
                self._domains[domain] = DomainStats(domain=domain)

            stats = self._domains[domain]
            stats.total_attempts += 1
            stats.total_duration_ms += duration_ms

            if success:
                stats.successful_scrapes += 1
                stats.total_content_chars += content_length
                stats.last_success = datetime.now(timezone.utc)
            else:
                stats.failed_scrapes += 1
                stats.last_failure = datetime.now(timezone.utc)
                if failure_reason:
                    stats.failure_reasons[failure_reason] += 1

        # Log for visibility
        status = "✓" if success else "✗"
        detail = f"{content_length:,} chars" if success else failure_reason or "unknown"
        logger.info(f"[SCRAPE] {status} {domain}: {detail} ({duration_ms:.0f}ms)")

    def is_provider_available(self, provider: str) -> Tuple[bool, str]:
        """
        Check if a provider should be used.

        Returns:
            (is_available, reason)
        """
        with self._lock:
            if provider not in self._providers:
                return True, "no stats yet"

            stats = self._providers[provider]

            # Check rate limit backoff
            if stats.is_rate_limited():
                remaining = (stats.rate_limit_backoff_until - datetime.now(timezone.utc)).seconds
                return False, f"rate limited, {remaining}s remaining"

            # Check if provider has very low success rate
            if stats.total_searches >= 10 and stats.success_rate < 0.2:
                return False, f"low success rate ({stats.success_rate:.0%})"

            return True, "available"

    def record_engine_result(
        self,
        engine: str,
        success: bool,
        results_count: int = 0,
        error: Optional[str] = None
    ) -> None:
        """Record a SearXNG internal engine result"""
        with self._lock:
            if engine not in self._engines:
                self._engines[engine] = EngineStats(name=engine)

            stats = self._engines[engine]
            stats.total_queries += 1

            if success:
                stats.successful_queries += 1
                stats.total_results += results_count
                # Clear backoff on success
                stats.backoff_until = None
                # Decay failure counter
                if stats.failed_queries > 0:
                    stats.failed_queries = max(0, stats.failed_queries - 1)
            else:
                stats.failed_queries += 1
                stats.last_error = error
                stats.last_error_time = datetime.now(timezone.utc)
                # Set backoff
                backoff = stats.get_backoff_duration()
                stats.backoff_until = datetime.now(timezone.utc) + timedelta(seconds=backoff)

                logger.warning(
                    f"[ENGINE] {engine} failed: {error}, backing off {backoff}s"
                )

    def record_unresponsive_engines(
        self,
        unresponsive: List[List[str]]
    ) -> None:
        """
        Process SearXNG's unresponsive_engines list.

        Format: [['engine_name', 'error_reason'], ...]
        Example: [['duckduckgo', 'CAPTCHA'], ['google', 'timeout']]
        """
        for item in unresponsive:
            if len(item) >= 2:
                engine_name, error_reason = item[0], item[1]
                self.record_engine_result(engine_name, success=False, error=error_reason)

    def get_available_engines(
        self,
        requested_engines: List[str] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Get list of engines that are not in backoff.

        Args:
            requested_engines: List of engines to filter

        Returns:
            (available_engines, skipped_reasons)
        """
        if requested_engines is None:
            requested_engines = list(self._engines.keys())

        available = []
        skipped = {}

        with self._lock:
            for engine in requested_engines:
                if engine not in self._engines:
                    available.append(engine)  # Unknown engine, assume available
                    continue

                stats = self._engines[engine]
                if stats.is_in_backoff():
                    remaining = (stats.backoff_until - datetime.now(timezone.utc)).seconds
                    skipped[engine] = f"in backoff ({remaining}s remaining, last error: {stats.last_error})"
                elif stats.success_rate < 0.3 and stats.total_queries >= 5:
                    skipped[engine] = f"low success rate ({stats.success_rate:.0%})"
                else:
                    available.append(engine)

        return available, skipped

    def get_engine_stats(self, engine: str) -> Optional[Dict]:
        """Get stats for a specific SearXNG engine"""
        with self._lock:
            if engine not in self._engines:
                return None

            stats = self._engines[engine]
            return {
                "name": stats.name,
                "total_queries": stats.total_queries,
                "successful": stats.successful_queries,
                "failed": stats.failed_queries,
                "success_rate": f"{stats.success_rate:.1%}",
                "total_results": stats.total_results,
                "last_error": stats.last_error,
                "is_in_backoff": stats.is_in_backoff(),
                "backoff_seconds": (stats.backoff_until - datetime.now(timezone.utc)).seconds if stats.is_in_backoff() else 0
            }

    def get_best_provider(
        self,
        preferred_order: List[str] = None
    ) -> Tuple[str, str]:
        """
        Get the best available provider.

        Args:
            preferred_order: List of providers in preference order

        Returns:
            (provider_name, selection_reason)
        """
        if preferred_order is None:
            preferred_order = ["searxng", "duckduckgo", "brave"]

        for provider in preferred_order:
            available, reason = self.is_provider_available(provider)
            if available:
                return provider, reason

        # All providers rate-limited - return first with shortest backoff
        with self._lock:
            shortest_wait = float("inf")
            best_provider = preferred_order[0]

            for provider in preferred_order:
                if provider in self._providers:
                    stats = self._providers[provider]
                    if stats.rate_limit_backoff_until:
                        wait = (stats.rate_limit_backoff_until - datetime.now(timezone.utc)).total_seconds()
                        if wait < shortest_wait:
                            shortest_wait = wait
                            best_provider = provider

            return best_provider, f"all rate-limited, {best_provider} has shortest wait"

    def should_skip_domain(self, domain: str) -> Tuple[bool, str]:
        """
        Check if a domain should be skipped based on failure history.

        Returns:
            (should_skip, reason)
        """
        with self._lock:
            if domain not in self._domains:
                return False, "no history"

            stats = self._domains[domain]

            # Skip domains with consistent failures (>5 attempts, <20% success)
            if stats.total_attempts >= 5 and stats.success_rate < 0.2:
                return True, f"low success rate ({stats.success_rate:.0%})"

            # Skip domains with recent repeated failures
            if stats.failed_scrapes >= 3 and stats.last_failure:
                time_since_failure = datetime.now(timezone.utc) - stats.last_failure
                if time_since_failure < timedelta(hours=1):
                    return True, "recent repeated failures"

            return False, "ok"

    def get_summary(self) -> Dict:
        """Get a summary of all metrics"""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

            provider_stats = {}
            for name, stats in self._providers.items():
                provider_stats[name] = {
                    "total_searches": stats.total_searches,
                    "success_rate": f"{stats.success_rate:.1%}",
                    "avg_results": round(stats.avg_results_per_search, 1),
                    "avg_duration_ms": round(stats.avg_duration_ms, 0),
                    "rate_limit_hits": stats.rate_limit_hits,
                    "is_rate_limited": stats.is_rate_limited()
                }

            domain_stats = {}
            for domain, stats in self._domains.items():
                domain_stats[domain] = {
                    "attempts": stats.total_attempts,
                    "success_rate": f"{stats.success_rate:.1%}",
                    "avg_content": stats.avg_content_chars,
                    "failures": dict(stats.failure_reasons)
                }

            # Sort domains by attempts (most used first)
            domain_stats = dict(
                sorted(domain_stats.items(), key=lambda x: x[1]["attempts"], reverse=True)[:20]
            )

            # Engine stats (SearXNG internal engines)
            engine_stats = {}
            for name, stats in self._engines.items():
                if stats.total_queries > 0:  # Only include used engines
                    engine_stats[name] = {
                        "total_queries": stats.total_queries,
                        "success_rate": f"{stats.success_rate:.1%}",
                        "total_results": stats.total_results,
                        "failed": stats.failed_queries,
                        "last_error": stats.last_error,
                        "is_in_backoff": stats.is_in_backoff()
                    }

            # Sort engines by usage
            engine_stats = dict(
                sorted(engine_stats.items(), key=lambda x: x[1]["total_queries"], reverse=True)
            )

            return {
                "uptime_seconds": round(uptime),
                "providers": provider_stats,
                "engines": engine_stats,  # SearXNG internal engines
                "top_domains": domain_stats,
                "recent_queries": self._recent_queries[-10:]
            }

    def get_provider_stats(self, provider: str) -> Optional[Dict]:
        """Get stats for a specific provider"""
        with self._lock:
            if provider not in self._providers:
                return None

            stats = self._providers[provider]
            return {
                "name": stats.name,
                "total_searches": stats.total_searches,
                "successful": stats.successful_searches,
                "failed": stats.failed_searches,
                "success_rate": f"{stats.success_rate:.1%}",
                "total_results": stats.total_results,
                "avg_results_per_search": round(stats.avg_results_per_search, 1),
                "avg_duration_ms": round(stats.avg_duration_ms, 0),
                "rate_limit_hits": stats.rate_limit_hits,
                "is_rate_limited": stats.is_rate_limited(),
                "backoff_seconds": stats.get_backoff_seconds() if stats.is_rate_limited() else 0
            }


# Singleton instance
_metrics_instance: Optional[SearchMetrics] = None
_metrics_lock = threading.Lock()


def get_search_metrics() -> SearchMetrics:
    """Get the global SearchMetrics instance"""
    global _metrics_instance
    with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = SearchMetrics()
    return _metrics_instance
