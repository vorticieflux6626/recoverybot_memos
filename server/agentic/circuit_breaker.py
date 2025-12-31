"""
Circuit Breaker Pattern for Production Reliability.

Part of G.4.1: Production Hardening - Circuit breakers for external service calls.

Implements the circuit breaker pattern to prevent cascading failures when
external services (LLMs, web APIs, vector databases) fail or become slow.

Three States:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit tripped, requests rejected immediately (fail fast)
- HALF_OPEN: Testing recovery with limited requests

Key Features:
- Per-provider circuit breakers (LLM, Search, Embedding, etc.)
- Configurable failure thresholds and timeouts
- Exponential backoff for recovery attempts
- Fallback chain support (Provider A → B → C)
- Metrics collection for monitoring
- Thread-safe async implementation

Research Basis:
- 2025 Multi-Agent RAG Breakthrough Report
- Microsoft Azure resilience patterns
- Netflix Hystrix design principles

Usage:
    from agentic.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerRegistry,
        with_circuit_breaker
    )

    # Option 1: Direct usage
    cb = CircuitBreaker(name="ollama_llm", config=CircuitBreakerConfig(
        failure_threshold=5,
        reset_timeout=30.0,
        half_open_requests=3
    ))

    result = await cb.call(llm_provider.generate, prompt)

    # Option 2: Decorator
    @with_circuit_breaker("ollama_llm")
    async def generate_response(prompt: str) -> str:
        return await llm_provider.generate(prompt)

    # Option 3: Registry with fallback chain
    registry = CircuitBreakerRegistry()
    result = await registry.call_with_fallback(
        providers=[ollama_provider, openai_provider, cached_provider],
        method_name="generate",
        args=(prompt,)
    )
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Union
import traceback

logger = logging.getLogger("agentic.circuit_breaker")

T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""
    pass


class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and request is rejected."""
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is OPEN. Retry after {retry_after:.1f}s")


class AllCircuitsOpenError(CircuitBreakerError):
    """Raised when all circuits in a fallback chain are open."""
    def __init__(self, provider_names: List[str]):
        self.provider_names = provider_names
        super().__init__(f"All circuits are OPEN: {provider_names}")


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    # Failure detection
    failure_threshold: int = 5       # Consecutive failures to trip circuit
    failure_window: float = 60.0     # Window in seconds for counting failures

    # Recovery
    reset_timeout: float = 30.0      # Seconds before attempting recovery
    half_open_requests: int = 3      # Successful requests needed to close
    recovery_backoff: float = 1.5    # Exponential backoff multiplier

    # Timeouts
    call_timeout: float = 30.0       # Timeout for individual calls
    max_timeout: float = 120.0       # Maximum timeout after backoff

    # Behavior
    exclude_exceptions: List[type] = field(default_factory=list)  # Don't count these as failures
    fallback_value: Any = None       # Default value when circuit is open


@dataclass
class CircuitMetrics:
    """Metrics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0          # Calls rejected due to open circuit
    timeout_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    open_count: int = 0              # Number of times circuit opened
    recovery_count: int = 0          # Number of successful recoveries


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.

    Prevents cascading failures by failing fast when a service is unhealthy,
    then gradually testing recovery before resuming normal operation.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique identifier for this circuit
            config: Configuration options
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._last_failure_time: float = 0
        self._half_open_successes: int = 0
        self._current_backoff: float = self.config.reset_timeout
        self._lock = asyncio.Lock()

        logger.info(f"CircuitBreaker '{name}' initialized: threshold={self.config.failure_threshold}")

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        """Circuit metrics."""
        return self._metrics

    @property
    def is_closed(self) -> bool:
        """Whether circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Whether circuit is open (rejecting requests)."""
        return self._state == CircuitState.OPEN

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1
        self._metrics.last_state_change_time = time.time()

        if new_state == CircuitState.OPEN:
            self._metrics.open_count += 1
        elif new_state == CircuitState.CLOSED and old_state == CircuitState.HALF_OPEN:
            self._metrics.recovery_count += 1
            self._current_backoff = self.config.reset_timeout  # Reset backoff on recovery

        logger.warning(
            f"CircuitBreaker '{self.name}': {old_state.value} → {new_state.value} "
            f"(failures={self._metrics.consecutive_failures}, "
            f"backoff={self._current_backoff:.1f}s)"
        )

    async def _should_allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if enough time has passed to try recovery
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._current_backoff:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_successes = 0
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests for testing
                return True

            return False

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.last_success_time = time.time()
            self._metrics.consecutive_failures = 0
            self._metrics.consecutive_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.half_open_requests:
                    await self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._last_failure_time = time.time()
            self._metrics.last_failure_time = self._last_failure_time
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0

            # Check if this exception type should be excluded
            if any(isinstance(error, exc_type) for exc_type in self.config.exclude_exceptions):
                logger.debug(f"CircuitBreaker '{self.name}': Excluded exception: {type(error).__name__}")
                return

            if self._state == CircuitState.CLOSED:
                if self._metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Failed during recovery attempt - increase backoff
                self._current_backoff = min(
                    self._current_backoff * self.config.recovery_backoff,
                    self.config.max_timeout
                )
                await self._transition_to(CircuitState.OPEN)

    async def call(
        self,
        func: Callable[..., Any],
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to call (async or sync)
            *args: Positional arguments for func
            timeout: Override default timeout
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitOpenError: If circuit is open
            TimeoutError: If call times out
            Exception: Any exception from func
        """
        # Check if request should be allowed
        if not await self._should_allow_request():
            self._metrics.rejected_calls += 1
            retry_after = self._current_backoff - (time.time() - self._last_failure_time)
            raise CircuitOpenError(self.name, max(0, retry_after))

        # Execute the call with timeout
        call_timeout = timeout or self.config.call_timeout
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=call_timeout)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                    timeout=call_timeout
                )

            await self._record_success()
            return result

        except asyncio.TimeoutError:
            self._metrics.timeout_calls += 1
            await self._record_failure(TimeoutError(f"Call timed out after {call_timeout}s"))
            raise

        except Exception as e:
            await self._record_failure(e)
            raise

    def get_status(self) -> Dict[str, Any]:
        """Get current status as dict."""
        return {
            "name": self.name,
            "state": self._state.value,
            "metrics": {
                "total_calls": self._metrics.total_calls,
                "successful_calls": self._metrics.successful_calls,
                "failed_calls": self._metrics.failed_calls,
                "rejected_calls": self._metrics.rejected_calls,
                "timeout_calls": self._metrics.timeout_calls,
                "consecutive_failures": self._metrics.consecutive_failures,
                "open_count": self._metrics.open_count,
                "recovery_count": self._metrics.recovery_count,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "reset_timeout": self.config.reset_timeout,
                "call_timeout": self.config.call_timeout,
            },
            "current_backoff": self._current_backoff,
            "last_failure": datetime.fromtimestamp(self._last_failure_time, UTC).isoformat() if self._last_failure_time else None,
        }

    async def reset(self) -> None:
        """Manually reset circuit to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._metrics.consecutive_failures = 0
            self._current_backoff = self.config.reset_timeout
            logger.info(f"CircuitBreaker '{self.name}': Manually reset to CLOSED")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers with fallback support.

    Provides:
    - Centralized circuit breaker management
    - Fallback chains (Provider A → B → C)
    - Aggregate health monitoring
    """

    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Register a new circuit breaker."""
        if name in self._breakers:
            return self._breakers[name]

        breaker = CircuitBreaker(name, config)
        self._breakers[name] = breaker
        return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            return self.register(name, config)
        return self._breakers[name]

    async def call_with_fallback(
        self,
        providers: List[tuple[str, Callable]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute call with fallback chain.

        Tries each provider in order until one succeeds.

        Args:
            providers: List of (name, callable) tuples
            *args: Arguments for callable
            **kwargs: Keyword arguments for callable

        Returns:
            Result from first successful provider

        Raises:
            AllCircuitsOpenError: If all providers fail or are open
        """
        errors = []

        for name, provider in providers:
            breaker = self.get_or_create(name)

            try:
                result = await breaker.call(provider, *args, **kwargs)
                logger.debug(f"Fallback chain: '{name}' succeeded")
                return result

            except CircuitOpenError as e:
                logger.debug(f"Fallback chain: '{name}' circuit is open")
                errors.append((name, e))
                continue

            except Exception as e:
                logger.warning(f"Fallback chain: '{name}' failed: {e}")
                errors.append((name, e))
                continue

        # All providers failed
        failed_names = [name for name, _ in errors]
        raise AllCircuitsOpenError(failed_names)

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """Get aggregate health summary."""
        total = len(self._breakers)
        closed = sum(1 for b in self._breakers.values() if b.state == CircuitState.CLOSED)
        open_circuits = sum(1 for b in self._breakers.values() if b.state == CircuitState.OPEN)
        half_open = sum(1 for b in self._breakers.values() if b.state == CircuitState.HALF_OPEN)

        return {
            "total_circuits": total,
            "closed": closed,
            "open": open_circuits,
            "half_open": half_open,
            "health_percentage": (closed / total * 100) if total > 0 else 100,
            "unhealthy_circuits": [
                name for name, b in self._breakers.items()
                if b.state != CircuitState.CLOSED
            ]
        }


# Global registry instance
_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get or create global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """
    Decorator to wrap a function with circuit breaker protection.

    Usage:
        @with_circuit_breaker("ollama_llm")
        async def generate_response(prompt: str) -> str:
            return await llm.generate(prompt)
    """
    def decorator(func: Callable) -> Callable:
        registry = get_circuit_breaker_registry()
        breaker = registry.get_or_create(name, config)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        # Attach breaker reference for testing
        wrapper._circuit_breaker = breaker
        return wrapper

    return decorator


# Pre-configured circuit breakers for common services
LLM_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    reset_timeout=30.0,
    half_open_requests=2,
    call_timeout=60.0,     # LLM calls can be slow
    max_timeout=180.0,
)

SEARCH_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    reset_timeout=15.0,
    half_open_requests=3,
    call_timeout=10.0,
    max_timeout=60.0,
)

EMBEDDING_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    reset_timeout=10.0,
    half_open_requests=2,
    call_timeout=30.0,
    max_timeout=90.0,
)

SCRAPING_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    reset_timeout=60.0,    # Longer cooldown for web scraping
    half_open_requests=2,
    call_timeout=30.0,
    max_timeout=300.0,     # Some pages are slow
)


def create_llm_circuit_breaker(name: str = "llm") -> CircuitBreaker:
    """Create circuit breaker configured for LLM calls."""
    return CircuitBreaker(name, LLM_CIRCUIT_CONFIG)


def create_search_circuit_breaker(name: str = "search") -> CircuitBreaker:
    """Create circuit breaker configured for search calls."""
    return CircuitBreaker(name, SEARCH_CIRCUIT_CONFIG)


def create_embedding_circuit_breaker(name: str = "embedding") -> CircuitBreaker:
    """Create circuit breaker configured for embedding calls."""
    return CircuitBreaker(name, EMBEDDING_CIRCUIT_CONFIG)


def create_scraping_circuit_breaker(name: str = "scraping") -> CircuitBreaker:
    """Create circuit breaker configured for web scraping."""
    return CircuitBreaker(name, SCRAPING_CIRCUIT_CONFIG)


# Convenience type for fallback providers
FallbackProvider = tuple[str, Callable[..., Any]]


async def execute_with_fallback(
    primary: tuple[str, Callable],
    fallbacks: List[tuple[str, Callable]],
    *args,
    cached_fallback: Optional[Any] = None,
    **kwargs
) -> Any:
    """
    Execute with primary provider and fallback chain.

    Args:
        primary: (name, callable) for primary provider
        fallbacks: List of (name, callable) for fallback providers
        *args: Arguments for callable
        cached_fallback: Optional cached value to use if all providers fail
        **kwargs: Keyword arguments for callable

    Returns:
        Result from successful provider or cached_fallback

    Raises:
        AllCircuitsOpenError: If all providers fail and no cached_fallback
    """
    registry = get_circuit_breaker_registry()

    try:
        return await registry.call_with_fallback(
            [primary] + fallbacks,
            *args,
            **kwargs
        )
    except AllCircuitsOpenError:
        if cached_fallback is not None:
            logger.warning("All circuits open, using cached fallback")
            return cached_fallback
        raise


def get_circuit_health() -> Dict[str, Any]:
    """Get health status of all registered circuits."""
    registry = get_circuit_breaker_registry()
    return registry.get_health_summary()


def get_circuit_status(name: str) -> Optional[Dict[str, Any]]:
    """Get status of a specific circuit."""
    registry = get_circuit_breaker_registry()
    breaker = registry.get(name)
    return breaker.get_status() if breaker else None


async def reset_circuit(name: str) -> bool:
    """Manually reset a circuit to closed state."""
    registry = get_circuit_breaker_registry()
    breaker = registry.get(name)
    if breaker:
        await breaker.reset()
        return True
    return False
