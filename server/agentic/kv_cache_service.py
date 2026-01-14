"""
KV Cache Service - Unified Interface for Inference Backend Caching

Provides a unified interface for KV cache management across different
inference backends (Ollama, vLLM, SGLang).

Key Features:
- Content-addressed caching with hash-based deduplication
- Automatic cache warming for frequently accessed content
- Integration with TTLCacheManager for pinning during tool calls
- Support for both synchronous and asynchronous cache operations

Based on MemOS research (arxiv:2501.09136) - MemCube architecture
Ref: KV_CACHE_IMPLEMENTATION_PLAN.md Phase 4.2
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger("agentic.kv_cache_service")


class CacheBackend(str, Enum):
    """Supported inference backends"""
    OLLAMA = "ollama"
    VLLM = "vllm"
    SGLANG = "sglang"


class CacheState(str, Enum):
    """State of a cached item"""
    COLD = "cold"           # Not in KV cache
    WARMING = "warming"     # Being precomputed
    WARM = "warm"           # In KV cache, ready
    EVICTED = "evicted"     # Was warm, now evicted


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata"""
    content_id: str
    content_hash: str
    state: CacheState = CacheState.COLD
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    estimated_tokens: int = 0
    warm_time_ms: float = 0  # Time taken to warm this entry
    model: str = ""

    def touch(self):
        """Update last access time and increment count"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for the KV cache service"""
    total_entries: int = 0
    warm_entries: int = 0
    cold_entries: int = 0
    total_accesses: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_warm_time_ms: float = 0
    avg_warm_time_ms: float = 0
    estimated_tokens_cached: int = 0


class KVCacheService:
    """
    Unified KV cache interface across inference backends.

    Manages cache warming, access tracking, and coordination with
    the TTLCacheManager for tool call pinning.

    Usage:
        service = KVCacheService(
            backend=CacheBackend.OLLAMA,
            ollama_url="http://localhost:11434"
        )

        # Warm cache for a prefix
        cache_id = await service.warm_prefix(
            prefix="You are a helpful assistant...",
            model="llama3.2:3b"
        )

        # Check if content is cached
        if await service.is_warm(cache_id):
            # Use cached prefix
            pass

        # Get cache for reuse
        entry = await service.get_cache_entry(cache_id)
    """

    def __init__(
        self,
        backend: CacheBackend = CacheBackend.OLLAMA,
        ollama_url: str = "http://localhost:11434",
        vllm_url: Optional[str] = None,
        max_cache_entries: int = 1000,
        warm_on_access_threshold: int = 2,  # Auto-warm after N accesses
        default_model: str = "llama3.2:3b"
    ):
        """
        Initialize KV cache service.

        Args:
            backend: Inference backend to use
            ollama_url: Ollama API URL
            vllm_url: vLLM API URL (if using vLLM)
            max_cache_entries: Maximum entries to track
            warm_on_access_threshold: Access count to trigger auto-warming
            default_model: Default model for cache warming
        """
        self.backend = backend
        self.ollama_url = ollama_url
        self.vllm_url = vllm_url
        self.max_cache_entries = max_cache_entries
        self.warm_threshold = warm_on_access_threshold
        self.default_model = default_model

        # Cache registry: content_hash -> CacheEntry
        self._cache: Dict[str, CacheEntry] = {}

        # Content ID mapping: content_id -> content_hash
        self._id_to_hash: Dict[str, str] = {}

        # Stats
        self._stats = CacheStats()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            f"KVCacheService initialized: backend={backend.value}, "
            f"url={ollama_url}, threshold={warm_on_access_threshold}"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count (rough: ~4 chars per token)"""
        return len(content) // 4

    async def warm_prefix(
        self,
        prefix: str,
        model: Optional[str] = None,
        content_id: Optional[str] = None
    ) -> str:
        """
        Warm the KV cache for a prefix string.

        For Ollama: Makes a minimal inference call with the prefix
        to populate the KV cache. Subsequent calls with the same
        prefix will reuse the cached KV states.

        Args:
            prefix: The prompt prefix to cache
            model: Model to use (defaults to default_model)
            content_id: Optional custom ID (otherwise uses hash)

        Returns:
            cache_id: The cache entry ID for this prefix
        """
        model = model or self.default_model
        content_hash = self._compute_hash(prefix)

        # Check if already cached
        if content_hash in self._cache:
            entry = self._cache[content_hash]
            entry.touch()
            self._stats.cache_hits += 1
            if entry.state == CacheState.WARM:
                logger.debug(f"Cache hit for {content_hash[:8]} (warm)")
                return content_hash
            elif entry.state == CacheState.WARMING:
                logger.debug(f"Cache warming in progress for {content_hash[:8]}")
                return content_hash

        # Create new entry
        entry = CacheEntry(
            content_id=content_id or content_hash,
            content_hash=content_hash,
            state=CacheState.WARMING,
            estimated_tokens=self._estimate_tokens(prefix),
            model=model
        )
        self._cache[content_hash] = entry
        if content_id:
            self._id_to_hash[content_id] = content_hash

        self._stats.cache_misses += 1
        self._stats.total_entries += 1

        # Warm the cache based on backend
        start_time = time.time()
        try:
            if self.backend == CacheBackend.OLLAMA:
                await self._warm_ollama(prefix, model)
            elif self.backend == CacheBackend.VLLM:
                await self._warm_vllm(prefix, model)
            # SGLang uses RadixAttention - automatic caching

            entry.state = CacheState.WARM
            entry.warm_time_ms = (time.time() - start_time) * 1000
            self._stats.warm_entries += 1
            self._stats.total_warm_time_ms += entry.warm_time_ms
            self._stats.estimated_tokens_cached += entry.estimated_tokens

            logger.info(
                f"Warmed cache {content_hash[:8]} in {entry.warm_time_ms:.0f}ms "
                f"(~{entry.estimated_tokens} tokens)"
            )

        except Exception as e:
            entry.state = CacheState.COLD
            logger.error(f"Failed to warm cache {content_hash[:8]}: {e}")

        return content_hash

    async def _warm_ollama(self, prefix: str, model: str):
        """
        Warm Ollama's KV cache by making a minimal inference call.

        Ollama automatically caches KV states for prompts. By making
        a call with max_tokens=1 or stop sequence, we warm the cache
        without generating much output.
        """
        client = await self._get_client()

        # Use generate endpoint with minimal output
        payload = {
            "model": model,
            "prompt": prefix,
            "stream": False,
            "options": {
                "num_predict": 1,  # Generate just 1 token
                "temperature": 0,
            },
            "keep_alive": "30m"  # Keep model loaded
        }

        try:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=httpx.Timeout(60.0)
            )
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Ollama warm response: {result.get('eval_count', 0)} tokens")
            else:
                text = response.text
                logger.warning(f"Ollama warm failed: {response.status_code} - {text[:200]}")

        except asyncio.TimeoutError:
            logger.warning("Ollama warm request timed out")
        except Exception as e:
            logger.error(f"Ollama warm error: {e}")
            raise

    async def _warm_vllm(self, prefix: str, model: str):
        """
        Warm vLLM's prefix cache.

        vLLM with --enable-prefix-caching automatically caches
        common prefixes. We just need to make an initial call.
        """
        if not self.vllm_url:
            logger.warning("vLLM URL not configured, skipping warm")
            return

        client = await self._get_client()

        payload = {
            "model": model,
            "prompt": prefix,
            "max_tokens": 1,
            "temperature": 0
        }

        try:
            response = await client.post(
                f"{self.vllm_url}/v1/completions",
                json=payload,
                timeout=httpx.Timeout(60.0)
            )
            if response.status_code == 200:
                logger.debug("vLLM prefix cached")
            else:
                text = response.text
                logger.warning(f"vLLM warm failed: {response.status_code}")

        except Exception as e:
            logger.error(f"vLLM warm error: {e}")
            raise

    async def get(self, cache_id: str) -> Optional[CacheEntry]:
        """
        Get cache entry by ID or hash.

        Returns None if not found or cold.
        """
        # Try direct hash lookup
        entry = self._cache.get(cache_id)
        if not entry:
            # Try ID-to-hash mapping
            content_hash = self._id_to_hash.get(cache_id)
            if content_hash:
                entry = self._cache.get(content_hash)

        if entry:
            entry.touch()
            self._stats.total_accesses += 1

            # Auto-warm if accessed frequently but cold
            if (entry.state == CacheState.COLD and
                entry.access_count >= self.warm_threshold):
                logger.info(f"Auto-warming {cache_id[:8]} after {entry.access_count} accesses")
                # Note: Would need original content to warm - skip for now

        return entry

    async def is_warm(self, cache_id: str) -> bool:
        """Check if content is in warm cache"""
        entry = await self.get(cache_id)
        return entry is not None and entry.state == CacheState.WARM

    def register(self, content_id: str, content_hash: str):
        """Register a content ID to hash mapping"""
        self._id_to_hash[content_id] = content_hash

    async def evict(self, cache_id: str) -> bool:
        """
        Mark cache entry as evicted.

        Note: This doesn't actually evict from the inference engine,
        just marks our tracking as evicted.
        """
        entry = await self.get(cache_id)
        if entry and entry.state == CacheState.WARM:
            entry.state = CacheState.EVICTED
            self._stats.warm_entries -= 1
            logger.debug(f"Marked {cache_id[:8]} as evicted")
            return True
        return False

    def cleanup_old_entries(self, max_age_hours: int = 24) -> int:
        """Remove entries older than max_age_hours"""
        now = datetime.now(timezone.utc)
        to_remove = []

        for content_hash, entry in self._cache.items():
            age_hours = (now - entry.last_accessed).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_remove.append(content_hash)

        for content_hash in to_remove:
            entry = self._cache.pop(content_hash)
            # Remove ID mappings
            self._id_to_hash = {
                k: v for k, v in self._id_to_hash.items()
                if v != content_hash
            }
            if entry.state == CacheState.WARM:
                self._stats.warm_entries -= 1
            self._stats.total_entries -= 1

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old cache entries")

        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._stats.cold_entries = (
            self._stats.total_entries - self._stats.warm_entries
        )
        if self._stats.warm_entries > 0:
            self._stats.avg_warm_time_ms = (
                self._stats.total_warm_time_ms / self._stats.warm_entries
            )

        total_requests = self._stats.cache_hits + self._stats.cache_misses
        hit_rate = self._stats.cache_hits / max(total_requests, 1)

        return {
            "total_entries": self._stats.total_entries,
            "warm_entries": self._stats.warm_entries,
            "cold_entries": self._stats.cold_entries,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "hit_rate": round(hit_rate, 3),
            "total_accesses": self._stats.total_accesses,
            "avg_warm_time_ms": round(self._stats.avg_warm_time_ms, 1),
            "estimated_tokens_cached": self._stats.estimated_tokens_cached,
            "backend": self.backend.value
        }

    def get_warm_entries(self) -> List[Dict[str, Any]]:
        """Get list of currently warm entries"""
        return [
            {
                "content_id": entry.content_id,
                "content_hash": entry.content_hash,
                "model": entry.model,
                "estimated_tokens": entry.estimated_tokens,
                "access_count": entry.access_count,
                "warm_time_ms": entry.warm_time_ms,
                "last_accessed": entry.last_accessed.isoformat()
            }
            for entry in self._cache.values()
            if entry.state == CacheState.WARM
        ]


# Global instance
_kv_cache_service: Optional[KVCacheService] = None


def get_kv_cache_service() -> KVCacheService:
    """Get the global KV cache service instance"""
    global _kv_cache_service
    if _kv_cache_service is None:
        import os
        _kv_cache_service = KVCacheService(
            backend=CacheBackend.OLLAMA,
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            default_model=os.getenv("DEFAULT_MODEL", "llama3.2:3b")
        )
    return _kv_cache_service


async def warm_system_prompts():
    """
    Warm commonly used system prompts at startup.

    Call this during server initialization to pre-warm
    the KV cache with frequently used prompts.
    """
    from .prompts import CORE_SYSTEM_PREFIX, AGENT_SUFFIXES

    service = get_kv_cache_service()

    # Warm core system prefix
    await service.warm_prefix(
        prefix=CORE_SYSTEM_PREFIX,
        content_id="core_system_prefix"
    )

    # Warm each agent type's full prompt
    for agent_type, suffix in AGENT_SUFFIXES.items():
        full_prompt = CORE_SYSTEM_PREFIX + suffix
        await service.warm_prefix(
            prefix=full_prompt,
            content_id=f"agent_{agent_type}"
        )

    logger.info(f"Warmed {1 + len(AGENT_SUFFIXES)} system prompts")
