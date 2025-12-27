"""
TTL-Based KV Cache Manager (Continuum-inspired)

Implements per-tool latency tracking and TTL-based cache pinning to prevent
premature KV cache eviction during long-running tool operations.

Research basis: Continuum (2025) - "Efficient and Robust Multi-Turn LLM Agent
Scheduling with KV Cache TTL", arXiv 2511.02230

Key concepts:
- Track historical latency per tool type
- Pin KV cache entries for expected tool duration + buffer
- Prevent Ollama from evicting cache during web scraping (3-8s)
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("agentic.ttl_cache")


class ToolType(str, Enum):
    """Tool types with distinct latency profiles"""
    WEB_SEARCH = "web_search"
    WEB_SCRAPE = "web_scrape"
    VL_SCRAPE = "vl_screenshot_scrape"
    OLLAMA_GENERATE = "ollama_generate"
    OLLAMA_EMBED = "ollama_embed"
    RAG_SEARCH = "rag_search"
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"


# Default latency estimates (ms) for cold start
DEFAULT_LATENCIES = {
    ToolType.WEB_SEARCH: 1500,      # DuckDuckGo search
    ToolType.WEB_SCRAPE: 4000,      # Page fetch + extract
    ToolType.VL_SCRAPE: 8000,       # Screenshot + VL model
    ToolType.OLLAMA_GENERATE: 3000, # LLM generation
    ToolType.OLLAMA_EMBED: 500,     # Embedding generation
    ToolType.RAG_SEARCH: 800,       # Vector search
    ToolType.MEMORY_STORE: 200,     # PostgreSQL write
    ToolType.MEMORY_RETRIEVE: 300,  # PostgreSQL read
}


@dataclass
class LatencyStats:
    """Track latency statistics for a tool type with sliding window"""
    samples: List[float] = field(default_factory=list)
    max_samples: int = 100
    tool_type: Optional[ToolType] = None

    def add_sample(self, latency_ms: float):
        """Add a latency sample, maintaining sliding window"""
        self.samples.append(latency_ms)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    @property
    def p50_latency(self) -> float:
        """Median latency"""
        if not self.samples:
            return DEFAULT_LATENCIES.get(self.tool_type, 2000)
        sorted_samples = sorted(self.samples)
        return sorted_samples[len(sorted_samples) // 2]

    @property
    def p95_latency(self) -> float:
        """95th percentile latency - use for TTL calculation"""
        if not self.samples:
            return DEFAULT_LATENCIES.get(self.tool_type, 2000) * 1.5
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p99_latency(self) -> float:
        """99th percentile for worst-case estimation"""
        if not self.samples:
            return DEFAULT_LATENCIES.get(self.tool_type, 2000) * 2
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def avg_latency(self) -> float:
        """Average latency"""
        if not self.samples:
            return DEFAULT_LATENCIES.get(self.tool_type, 2000)
        return sum(self.samples) / len(self.samples)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> Dict:
        return {
            "avg_ms": round(self.avg_latency, 1),
            "p50_ms": round(self.p50_latency, 1),
            "p95_ms": round(self.p95_latency, 1),
            "p99_ms": round(self.p99_latency, 1),
            "sample_count": self.sample_count
        }


@dataclass
class CachePin:
    """Represents a pinned cache entry"""
    cache_id: str
    program_id: str
    tool_type: ToolType
    pin_start: float
    pin_end: float  # Calculated TTL expiry
    actual_start: float  # When tool actually started

    @property
    def remaining_ttl_ms(self) -> float:
        """Remaining TTL in milliseconds"""
        return max(0, (self.pin_end - time.time()) * 1000)

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.pin_end


class TTLCacheManager:
    """
    Continuum-inspired TTL-based KV cache pinning manager.

    Tracks per-tool latencies and pins KV cache entries for the expected
    duration of tool execution, preventing premature eviction by Ollama.

    Usage:
        manager = TTLCacheManager()

        # Before tool call
        pin = manager.pin_for_tool(program_id, ToolType.WEB_SCRAPE, cache_id)

        # Execute tool...
        result = await web_scraper.scrape(url)

        # After tool call
        manager.record_completion(pin, success=True)
    """

    def __init__(
        self,
        ttl_buffer_factor: float = 1.3,
        min_ttl_ms: float = 1000,
        max_ttl_ms: float = 60000,
        cleanup_interval_s: float = 5.0
    ):
        """
        Initialize TTL cache manager.

        Args:
            ttl_buffer_factor: Multiplier on p95 latency for TTL (1.3 = 30% buffer)
            min_ttl_ms: Minimum TTL in milliseconds
            max_ttl_ms: Maximum TTL in milliseconds (prevent runaway pins)
            cleanup_interval_s: How often to clean up expired pins
        """
        self.ttl_buffer_factor = ttl_buffer_factor
        self.min_ttl_ms = min_ttl_ms
        self.max_ttl_ms = max_ttl_ms
        self.cleanup_interval_s = cleanup_interval_s

        # Per-tool latency tracking
        self.tool_stats: Dict[ToolType, LatencyStats] = {}
        for tool_type in ToolType:
            stats = LatencyStats(tool_type=tool_type)
            self.tool_stats[tool_type] = stats

        # Active pins: cache_id -> CachePin
        self.active_pins: Dict[str, CachePin] = {}

        # Program tracking for grouped eviction
        self.program_pins: Dict[str, List[str]] = defaultdict(list)

        # Statistics
        self.stats = {
            "pins_created": 0,
            "pins_expired": 0,
            "pins_completed": 0,
            "pins_extended": 0,
            "evictions_prevented": 0
        }

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(f"TTLCacheManager initialized (buffer={ttl_buffer_factor}x, "
                   f"min={min_ttl_ms}ms, max={max_ttl_ms}ms)")

    def calculate_ttl(self, tool_type: ToolType) -> float:
        """
        Calculate TTL for a tool type based on historical latency.

        Returns TTL in seconds.
        """
        stats = self.tool_stats.get(tool_type)
        if not stats:
            base_latency = DEFAULT_LATENCIES.get(tool_type, 2000)
        else:
            base_latency = stats.p95_latency

        # Apply buffer and clamp to bounds
        ttl_ms = base_latency * self.ttl_buffer_factor
        ttl_ms = max(self.min_ttl_ms, min(self.max_ttl_ms, ttl_ms))

        return ttl_ms / 1000  # Convert to seconds

    def pin_for_tool(
        self,
        program_id: str,
        tool_type: ToolType,
        cache_id: Optional[str] = None
    ) -> CachePin:
        """
        Pin KV cache for the duration of a tool call.

        Args:
            program_id: Unique ID for the current agentic workflow
            tool_type: Type of tool being invoked
            cache_id: Optional specific cache ID (defaults to program_id)

        Returns:
            CachePin object for tracking
        """
        now = time.time()
        ttl_seconds = self.calculate_ttl(tool_type)

        if cache_id is None:
            cache_id = f"{program_id}_{tool_type.value}_{int(now * 1000)}"

        pin = CachePin(
            cache_id=cache_id,
            program_id=program_id,
            tool_type=tool_type,
            pin_start=now,
            pin_end=now + ttl_seconds,
            actual_start=now
        )

        self.active_pins[cache_id] = pin
        self.program_pins[program_id].append(cache_id)
        self.stats["pins_created"] += 1

        logger.debug(f"Pinned cache {cache_id} for {tool_type.value} "
                    f"(TTL: {ttl_seconds:.1f}s)")

        return pin

    def extend_pin(self, cache_id: str, additional_seconds: float) -> bool:
        """
        Extend TTL for an active pin (for unexpectedly long operations).

        Returns True if extended, False if pin not found.
        """
        pin = self.active_pins.get(cache_id)
        if not pin:
            return False

        pin.pin_end += additional_seconds
        # Clamp to max TTL from original start
        max_end = pin.pin_start + (self.max_ttl_ms / 1000)
        pin.pin_end = min(pin.pin_end, max_end)

        self.stats["pins_extended"] += 1
        logger.debug(f"Extended pin {cache_id} by {additional_seconds:.1f}s")

        return True

    def record_completion(
        self,
        pin: CachePin,
        success: bool = True,
        actual_latency_ms: Optional[float] = None
    ):
        """
        Record tool completion and update latency statistics.

        Args:
            pin: The CachePin from pin_for_tool()
            success: Whether the tool completed successfully
            actual_latency_ms: Actual latency (calculated from pin if not provided)
        """
        now = time.time()

        # Calculate actual latency
        if actual_latency_ms is None:
            actual_latency_ms = (now - pin.actual_start) * 1000

        # Update stats for successful completions
        if success:
            self.tool_stats[pin.tool_type].add_sample(actual_latency_ms)

        # Remove from active pins
        if pin.cache_id in self.active_pins:
            del self.active_pins[pin.cache_id]
            self.stats["pins_completed"] += 1

        # Remove from program tracking
        if pin.cache_id in self.program_pins.get(pin.program_id, []):
            self.program_pins[pin.program_id].remove(pin.cache_id)

        logger.debug(f"Completed {pin.tool_type.value} in {actual_latency_ms:.0f}ms "
                    f"(success={success})")

    def is_pinned(self, cache_id: str) -> bool:
        """Check if a cache entry is currently pinned (and not expired)"""
        pin = self.active_pins.get(cache_id)
        if not pin:
            return False
        if pin.is_expired:
            return False
        return True

    def get_pin_ttl(self, cache_id: str) -> float:
        """Get remaining TTL for a pinned cache entry (in seconds)"""
        pin = self.active_pins.get(cache_id)
        if not pin:
            return 0
        return pin.remaining_ttl_ms / 1000

    def get_program_pins(self, program_id: str) -> List[CachePin]:
        """Get all active pins for a program/workflow"""
        cache_ids = self.program_pins.get(program_id, [])
        return [self.active_pins[cid] for cid in cache_ids if cid in self.active_pins]

    def should_evict(self, cache_id: str) -> bool:
        """
        Check if a cache entry can be safely evicted.

        Returns False if pinned and not expired.
        This method can be called by cache eviction logic.
        """
        if not self.is_pinned(cache_id):
            return True

        self.stats["evictions_prevented"] += 1
        return False

    def cleanup_expired(self) -> int:
        """
        Remove expired pins from tracking.

        Returns number of pins cleaned up.
        """
        now = time.time()
        expired = []

        for cache_id, pin in self.active_pins.items():
            if pin.is_expired:
                expired.append(cache_id)

        for cache_id in expired:
            pin = self.active_pins.pop(cache_id)
            if cache_id in self.program_pins.get(pin.program_id, []):
                self.program_pins[pin.program_id].remove(cache_id)
            self.stats["pins_expired"] += 1

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired pins")

        return len(expired)

    async def start_cleanup_loop(self):
        """Start background cleanup task"""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval_s)
                    self.cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started TTL cache cleanup loop")

    async def stop_cleanup_loop(self):
        """Stop background cleanup task"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped TTL cache cleanup loop")

    def get_tool_latency_stats(self, tool_type: ToolType) -> Dict:
        """Get latency statistics for a tool type"""
        stats = self.tool_stats.get(tool_type)
        if not stats:
            return {"error": "Unknown tool type"}
        return stats.to_dict()

    def get_all_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "manager_stats": self.stats.copy(),
            "active_pins": len(self.active_pins),
            "active_programs": len([p for p in self.program_pins.values() if p]),
            "tool_latencies": {
                tool_type.value: self.tool_stats[tool_type].to_dict()
                for tool_type in ToolType
            },
            "config": {
                "ttl_buffer_factor": self.ttl_buffer_factor,
                "min_ttl_ms": self.min_ttl_ms,
                "max_ttl_ms": self.max_ttl_ms
            }
        }


# Global instance
_ttl_manager: Optional[TTLCacheManager] = None


def get_ttl_cache_manager() -> TTLCacheManager:
    """Get the global TTL cache manager instance"""
    global _ttl_manager
    if _ttl_manager is None:
        _ttl_manager = TTLCacheManager()
    return _ttl_manager


# Context manager for convenient usage
class ToolCallContext:
    """
    Context manager for tool call TTL pinning.

    Usage:
        async with ToolCallContext(program_id, ToolType.WEB_SCRAPE) as ctx:
            result = await scraper.scrape(url)
        # Pin automatically released and latency recorded
    """

    def __init__(
        self,
        program_id: str,
        tool_type: ToolType,
        cache_id: Optional[str] = None,
        manager: Optional[TTLCacheManager] = None
    ):
        self.program_id = program_id
        self.tool_type = tool_type
        self.cache_id = cache_id
        self.manager = manager or get_ttl_cache_manager()
        self.pin: Optional[CachePin] = None
        self.success = True

    async def __aenter__(self) -> "ToolCallContext":
        self.pin = self.manager.pin_for_tool(
            self.program_id,
            self.tool_type,
            self.cache_id
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pin:
            self.success = exc_type is None
            self.manager.record_completion(self.pin, success=self.success)
        return False  # Don't suppress exceptions

    def mark_failed(self):
        """Manually mark the operation as failed"""
        self.success = False
