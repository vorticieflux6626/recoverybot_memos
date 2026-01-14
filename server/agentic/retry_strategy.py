"""
Unified Retry Strategy - Centralized Backoff and Circuit Breakers

Provides unified retry logic for all HTTP operations:
- Exponential backoff with jitter (prevents thundering herd)
- Per-domain circuit breakers (isolates failing domains)
- Centralized failure tracking across all scrapers
- Configurable retry policies per operation type

Phase 4 of scraping consolidation (2026-01).

Usage:
    from agentic.retry_strategy import get_retry_strategy, RetryContext

    strategy = get_retry_strategy()

    # Check if domain is available
    if not strategy.is_domain_available("example.com"):
        return  # Circuit is open

    # Execute with retry
    async with RetryContext(strategy, "example.com") as ctx:
        for attempt in ctx.attempts(max_retries=3):
            try:
                result = await fetch(url)
                ctx.record_success()
                return result
            except Exception as e:
                ctx.record_failure(e)
                if attempt.is_last:
                    raise
                await attempt.wait()
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar
from urllib.parse import urlparse

logger = logging.getLogger("agentic.retry_strategy")

T = TypeVar("T")


# ============================================
# CONFIGURATION
# ============================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    # Base delay for exponential backoff (seconds)
    base_delay: float = 1.0

    # Maximum delay cap (seconds)
    max_delay: float = 60.0

    # Jitter factor (0.0-1.0, adds randomness to prevent thundering herd)
    jitter_factor: float = 0.25

    # Default max retries
    default_max_retries: int = 3

    # Exponential backoff multiplier
    backoff_multiplier: float = 2.0


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker."""
    # Failures before opening circuit
    failure_threshold: int = 5

    # Time to wait before testing recovery (seconds)
    recovery_timeout: float = 30.0

    # Successes needed to close circuit from half-open
    success_threshold: int = 2

    # Time window for counting failures (seconds)
    failure_window: float = 60.0


# ============================================
# CIRCUIT BREAKER
# ============================================

@dataclass
class CircuitBreaker:
    """
    Circuit breaker for a single domain.

    States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered
    """
    domain: str
    config: CircuitConfig = field(default_factory=CircuitConfig)

    # State tracking
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Failure history for windowed counting
    failure_times: List[datetime] = field(default_factory=list)

    def is_available(self) -> bool:
        """Check if requests should be allowed."""
        self._update_state()

        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.HALF_OPEN:
            return True  # Allow test request
        else:  # OPEN
            return False

    def record_success(self):
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                logger.info(f"Circuit CLOSED for {self.domain} (recovered)")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed request."""
        now = datetime.now(timezone.utc)
        self.failure_times.append(now)
        self.last_failure_time = now

        # Clean old failures outside window
        cutoff = now - timedelta(seconds=self.config.failure_window)
        self.failure_times = [t for t in self.failure_times if t > cutoff]

        if self.state == CircuitState.HALF_OPEN:
            # Immediately open on failure during recovery test
            self._transition_to(CircuitState.OPEN)
            logger.warning(f"Circuit OPEN for {self.domain} (failed recovery test)")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = len(self.failure_times)
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    f"Circuit OPEN for {self.domain} "
                    f"({self.failure_count} failures in {self.config.failure_window}s)"
                )

    def _update_state(self):
        """Update state based on time elapsed."""
        if self.state == CircuitState.OPEN:
            elapsed = (datetime.now(timezone.utc) - self.last_state_change).total_seconds()
            if elapsed >= self.config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info(f"Circuit HALF_OPEN for {self.domain} (testing recovery)")

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        self.state = new_state
        self.last_state_change = datetime.now(timezone.utc)
        if new_state == CircuitState.CLOSED:
            self.failure_count = 0
            self.success_count = 0
            self.failure_times = []
        elif new_state == CircuitState.HALF_OPEN:
            self.success_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker stats."""
        return {
            "domain": self.domain,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
        }


# ============================================
# RETRY ATTEMPT
# ============================================

@dataclass
class RetryAttempt:
    """Represents a single retry attempt."""
    attempt_number: int
    max_retries: int
    config: RetryConfig
    domain: str

    @property
    def is_last(self) -> bool:
        """Check if this is the last attempt."""
        return self.attempt_number >= self.max_retries

    @property
    def is_first(self) -> bool:
        """Check if this is the first attempt."""
        return self.attempt_number == 1

    def get_delay(self) -> float:
        """Calculate delay before retry with jitter."""
        if self.is_first:
            return 0.0  # No delay on first attempt

        # Exponential backoff
        delay = self.config.base_delay * (
            self.config.backoff_multiplier ** (self.attempt_number - 2)
        )

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        jitter = delay * self.config.jitter_factor * random.random()
        delay += jitter

        return delay

    async def wait(self):
        """Wait before the next retry."""
        delay = self.get_delay()
        if delay > 0:
            logger.debug(
                f"Retry {self.attempt_number}/{self.max_retries} for {self.domain}, "
                f"waiting {delay:.2f}s"
            )
            await asyncio.sleep(delay)


# ============================================
# RETRY CONTEXT
# ============================================

class RetryContext:
    """
    Context manager for retry operations.

    Usage:
        async with RetryContext(strategy, "example.com") as ctx:
            for attempt in ctx.attempts(max_retries=3):
                try:
                    result = await fetch(url)
                    ctx.record_success()
                    return result
                except Exception as e:
                    ctx.record_failure(e)
                    if attempt.is_last:
                        raise
                    await attempt.wait()
    """

    def __init__(self, strategy: "UnifiedRetryStrategy", domain: str):
        self.strategy = strategy
        self.domain = domain
        self._start_time: float = 0
        self._success: bool = False
        self._last_error: Optional[Exception] = None

    async def __aenter__(self) -> "RetryContext":
        self._start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self._start_time) * 1000
        self.strategy.record_latency(self.domain, latency_ms)
        return False  # Don't suppress exceptions

    def attempts(self, max_retries: Optional[int] = None) -> Iterator[RetryAttempt]:
        """
        Generate retry attempts.

        Args:
            max_retries: Override default max retries
        """
        max_retries = max_retries or self.strategy.retry_config.default_max_retries

        for i in range(1, max_retries + 1):
            yield RetryAttempt(
                attempt_number=i,
                max_retries=max_retries,
                config=self.strategy.retry_config,
                domain=self.domain
            )

    def record_success(self):
        """Record successful operation."""
        self._success = True
        self.strategy.record_success(self.domain)

    def record_failure(self, error: Optional[Exception] = None):
        """Record failed operation."""
        self._last_error = error
        self.strategy.record_failure(self.domain, error)


# ============================================
# UNIFIED RETRY STRATEGY
# ============================================

class UnifiedRetryStrategy:
    """
    Centralized retry strategy with circuit breakers.

    Features:
    - Per-domain circuit breakers
    - Exponential backoff with jitter
    - Latency tracking
    - Configurable retry policies
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitConfig] = None
    ):
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitConfig()

        # Per-domain circuit breakers
        self._circuits: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

        # Latency tracking
        self._latencies: Dict[str, List[float]] = {}
        self._max_latency_history = 100

        logger.info(
            f"UnifiedRetryStrategy initialized: "
            f"base_delay={self.retry_config.base_delay}s, "
            f"max_delay={self.retry_config.max_delay}s, "
            f"jitter={self.retry_config.jitter_factor}"
        )

    def _get_domain(self, url_or_domain: str) -> str:
        """Extract domain from URL or return as-is."""
        if url_or_domain.startswith(("http://", "https://")):
            return urlparse(url_or_domain).netloc
        return url_or_domain

    def _get_circuit(self, domain: str) -> CircuitBreaker:
        """Get or create circuit breaker for domain."""
        domain = self._get_domain(domain)
        if domain not in self._circuits:
            self._circuits[domain] = CircuitBreaker(
                domain=domain,
                config=self.circuit_config
            )
        return self._circuits[domain]

    def is_domain_available(self, domain: str) -> bool:
        """Check if domain is available (circuit not open)."""
        circuit = self._get_circuit(domain)
        return circuit.is_available()

    def record_success(self, domain: str):
        """Record successful request to domain."""
        circuit = self._get_circuit(domain)
        circuit.record_success()

    def record_failure(self, domain: str, error: Optional[Exception] = None):
        """Record failed request to domain."""
        circuit = self._get_circuit(domain)
        circuit.record_failure(error)

    def record_latency(self, domain: str, latency_ms: float):
        """Record request latency."""
        domain = self._get_domain(domain)
        if domain not in self._latencies:
            self._latencies[domain] = []

        self._latencies[domain].append(latency_ms)

        # Keep only recent history
        if len(self._latencies[domain]) > self._max_latency_history:
            self._latencies[domain] = self._latencies[domain][-self._max_latency_history:]

    def get_avg_latency(self, domain: str) -> float:
        """Get average latency for domain."""
        domain = self._get_domain(domain)
        latencies = self._latencies.get(domain, [])
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def context(self, domain: str) -> RetryContext:
        """Create retry context for domain."""
        return RetryContext(self, domain)

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            "retry_config": {
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay,
                "jitter_factor": self.retry_config.jitter_factor,
                "default_max_retries": self.retry_config.default_max_retries,
            },
            "circuit_config": {
                "failure_threshold": self.circuit_config.failure_threshold,
                "recovery_timeout": self.circuit_config.recovery_timeout,
                "success_threshold": self.circuit_config.success_threshold,
            },
            "circuits": {
                domain: circuit.get_stats()
                for domain, circuit in self._circuits.items()
            },
            "latencies": {
                domain: {
                    "avg_ms": round(sum(lats) / len(lats), 1) if lats else 0,
                    "min_ms": round(min(lats), 1) if lats else 0,
                    "max_ms": round(max(lats), 1) if lats else 0,
                    "count": len(lats),
                }
                for domain, lats in self._latencies.items()
            },
        }

    def reset_circuit(self, domain: str):
        """Manually reset circuit breaker for domain."""
        domain = self._get_domain(domain)
        if domain in self._circuits:
            self._circuits[domain]._transition_to(CircuitState.CLOSED)
            logger.info(f"Manually reset circuit for {domain}")


# ============================================
# SINGLETON INSTANCE
# ============================================

_retry_strategy_instance: Optional[UnifiedRetryStrategy] = None


def get_retry_strategy() -> UnifiedRetryStrategy:
    """Get singleton retry strategy instance."""
    global _retry_strategy_instance
    if _retry_strategy_instance is None:
        _retry_strategy_instance = UnifiedRetryStrategy()
    return _retry_strategy_instance


def reset_retry_strategy():
    """Reset singleton instance (for testing)."""
    global _retry_strategy_instance
    _retry_strategy_instance = None


# ============================================
# DECORATOR FOR EASY USE
# ============================================

def with_retry(
    domain: str,
    max_retries: int = 3,
    strategy: Optional[UnifiedRetryStrategy] = None
):
    """
    Decorator for automatic retry handling.

    Usage:
        @with_retry("api.example.com", max_retries=3)
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                return await client.get("https://api.example.com/data")
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args, **kwargs) -> T:
            strat = strategy or get_retry_strategy()

            if not strat.is_domain_available(domain):
                raise RuntimeError(f"Circuit open for {domain}")

            async with strat.context(domain) as ctx:
                last_error = None
                for attempt in ctx.attempts(max_retries):
                    try:
                        result = await func(*args, **kwargs)
                        ctx.record_success()
                        return result
                    except Exception as e:
                        last_error = e
                        ctx.record_failure(e)
                        if attempt.is_last:
                            raise
                        await attempt.wait()

                raise last_error  # Should not reach here

        return wrapper
    return decorator


# ============================================
# CLI INTERFACE
# ============================================

async def main():
    """CLI interface for testing retry strategy."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Retry Strategy CLI")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("--test-domain", help="Test circuit breaker for domain")
    parser.add_argument("--simulate-failures", type=int, default=0, help="Simulate N failures")

    args = parser.parse_args()

    strategy = get_retry_strategy()

    if args.test_domain:
        print(f"Testing domain: {args.test_domain}")

        # Simulate failures if requested
        for i in range(args.simulate_failures):
            strategy.record_failure(args.test_domain, Exception(f"Simulated failure {i+1}"))
            print(f"  Recorded failure {i+1}")

        available = strategy.is_domain_available(args.test_domain)
        print(f"  Domain available: {available}")

        circuit = strategy._get_circuit(args.test_domain)
        print(f"  Circuit state: {circuit.state.value}")

    if args.stats or not args.test_domain:
        stats = strategy.get_stats()
        print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
