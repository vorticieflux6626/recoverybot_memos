"""
Centralized Redis Cache Service for Web Scraping

Unified caching layer that consolidates multiple cache implementations:
- Content cache (replaces SQLite-based content_cache.py)
- Query result cache with semantic search support
- Rate limit tracking per domain

Key Features:
- Async Redis with connection pooling
- Circuit breaker for graceful degradation to in-memory fallback
- MessagePack serialization for efficient storage
- Content-addressed caching with SHA-256 hash keys
- TTL with jitter to prevent cache stampede
- Thread-safe operations

References:
- Phase 2 of scraping infrastructure improvements (2026-01)
- Redis async patterns from redis_embeddings_cache.py
"""

import asyncio
import hashlib
import json
import logging
import random
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict

try:
    import redis.asyncio as redis
    from redis.asyncio.connection import ConnectionPool
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    ConnectionPool = None  # type: ignore
    REDIS_AVAILABLE = False

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    msgpack = None  # type: ignore
    MSGPACK_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

logger = logging.getLogger("agentic.redis_cache")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, using fallback
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 30.0      # Seconds before trying half-open
    success_threshold: int = 2          # Successes in half-open to close


@dataclass
class CircuitBreakerState:
    """Runtime state for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_state_change: float = field(default_factory=time.time)


@dataclass
class CacheEntry:
    """A cached content entry."""
    url: str
    url_hash: str
    title: str
    content: str
    content_type: str
    content_hash: str
    success: bool
    error: Optional[str]
    created_at: float
    expires_at: float
    hit_count: int = 0


@dataclass
class CacheStats:
    """Statistics for the cache service."""
    content_entries: int = 0
    query_entries: int = 0
    content_hits: int = 0
    content_misses: int = 0
    query_hits: int = 0
    query_misses: int = 0
    semantic_hits: int = 0
    redis_errors: int = 0
    fallback_operations: int = 0
    circuit_opens: int = 0


class RedisCacheService:
    """
    Centralized Redis cache service for web scraping operations.

    Architecture:
    ```
    Request
        |
        v
    [Circuit Breaker] --> Check Redis health
        |
        v (if healthy)
    [Redis Cache] --> Fast, distributed
        |
        v (if Redis fails)
    [In-Memory Fallback] --> Local LRU cache
    ```

    Usage:
    ```python
    cache = RedisCacheService()
    await cache.connect()

    # Cache content
    await cache.set_content(url, title, content, content_type, success=True)
    result = await cache.get_content(url)

    # Cache query results
    await cache.set_query_result(query, result, embedding=embedding_vector)
    result = await cache.get_query_result(query)

    # Semantic search
    similar = await cache.find_similar_query(query_embedding)
    ```
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/3",
        key_prefix: str = "scrape_cache",
        content_ttl_seconds: int = 3600,      # 1 hour for content
        query_ttl_seconds: int = 900,          # 15 min for queries
        max_local_entries: int = 1000,         # Fallback cache size
        circuit_config: Optional[CircuitBreakerConfig] = None,
        enable_msgpack: bool = True
    ):
        """
        Initialize Redis cache service.

        Args:
            redis_url: Redis connection URL (using DB 3 for scraping cache)
            key_prefix: Prefix for all Redis keys
            content_ttl_seconds: Default TTL for content cache
            query_ttl_seconds: Default TTL for query cache
            max_local_entries: Max entries in fallback cache
            circuit_config: Circuit breaker configuration
            enable_msgpack: Use MessagePack for serialization (faster)
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.content_ttl_seconds = content_ttl_seconds
        self.query_ttl_seconds = query_ttl_seconds
        self.max_local_entries = max_local_entries
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.enable_msgpack = enable_msgpack and MSGPACK_AVAILABLE

        # Redis client (lazy initialization)
        self._redis: Optional[redis.Redis] = None
        self._pool: Optional[ConnectionPool] = None
        self._connected = False

        # Circuit breaker state
        self._circuit = CircuitBreakerState()
        self._circuit_lock = asyncio.Lock()

        # In-memory fallback cache (LRU)
        self._local_content: OrderedDict[str, CacheEntry] = OrderedDict()
        self._local_queries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._local_embeddings: OrderedDict[str, bytes] = OrderedDict()

        # Statistics
        self._stats = CacheStats()

        logger.info(
            f"RedisCacheService initialized: url={redis_url}, "
            f"prefix={key_prefix}, msgpack={self.enable_msgpack}"
        )

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Connect to Redis with connection pooling.

        Returns:
            True if connected, False if using fallback
        """
        if self._connected:
            return True

        if not REDIS_AVAILABLE:
            logger.warning("Redis client not available, using in-memory fallback")
            return False

        try:
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                decode_responses=False  # We handle encoding ourselves
            )

            self._redis = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._redis.ping()
            self._connected = True
            self._circuit.state = CircuitState.CLOSED

            logger.info(f"Connected to Redis at {self.redis_url}")
            return True

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
            self._connected = False
            await self._record_failure()
            return False

    async def disconnect(self):
        """Disconnect from Redis and cleanup."""
        if self._redis:
            await self._redis.aclose()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False
        logger.info("Disconnected from Redis")

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health and return status."""
        try:
            if self._connected and self._redis:
                start = time.time()
                await self._redis.ping()
                latency_ms = (time.time() - start) * 1000

                info = await self._redis.info("memory")
                used_memory = info.get("used_memory_human", "unknown")

                return {
                    "status": "healthy",
                    "connected": True,
                    "latency_ms": round(latency_ms, 2),
                    "memory_used": used_memory,
                    "circuit_state": self._circuit.state.value
                }
        except Exception as e:
            pass

        return {
            "status": "fallback",
            "connected": False,
            "circuit_state": self._circuit.state.value,
            "local_content_entries": len(self._local_content),
            "local_query_entries": len(self._local_queries)
        }

    # =========================================================================
    # Circuit Breaker
    # =========================================================================

    async def _check_circuit(self) -> bool:
        """
        Check if circuit allows Redis operations.

        Returns:
            True if Redis operations should be attempted
        """
        async with self._circuit_lock:
            if self._circuit.state == CircuitState.CLOSED:
                return True

            if self._circuit.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                elapsed = time.time() - self._circuit.last_failure_time
                if elapsed >= self.circuit_config.recovery_timeout:
                    self._circuit.state = CircuitState.HALF_OPEN
                    self._circuit.success_count = 0
                    self._circuit.last_state_change = time.time()
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    return True
                return False

            # HALF_OPEN: allow testing
            return True

    async def _record_success(self):
        """Record successful Redis operation."""
        async with self._circuit_lock:
            if self._circuit.state == CircuitState.HALF_OPEN:
                self._circuit.success_count += 1
                if self._circuit.success_count >= self.circuit_config.success_threshold:
                    self._circuit.state = CircuitState.CLOSED
                    self._circuit.failure_count = 0
                    self._circuit.last_state_change = time.time()
                    logger.info("Circuit breaker CLOSED - Redis recovered")

    async def _record_failure(self):
        """Record failed Redis operation."""
        async with self._circuit_lock:
            self._circuit.failure_count += 1
            self._circuit.last_failure_time = time.time()
            self._stats.redis_errors += 1

            if self._circuit.state == CircuitState.HALF_OPEN:
                # Failed during testing, reopen
                self._circuit.state = CircuitState.OPEN
                self._circuit.last_state_change = time.time()
                self._stats.circuit_opens += 1
                logger.warning("Circuit breaker OPEN - Redis still failing")

            elif (self._circuit.state == CircuitState.CLOSED and
                  self._circuit.failure_count >= self.circuit_config.failure_threshold):
                self._circuit.state = CircuitState.OPEN
                self._circuit.last_state_change = time.time()
                self._stats.circuit_opens += 1
                logger.warning(
                    f"Circuit breaker OPEN after {self._circuit.failure_count} failures"
                )

    # =========================================================================
    # Key Generation
    # =========================================================================

    def _hash_url(self, url: str) -> str:
        """Generate SHA-256 hash for URL (content-addressed)."""
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _hash_content(self, content: str) -> str:
        """Generate SHA-256 hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _hash_query(self, query: str) -> str:
        """Generate hash for query (normalized)."""
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _make_content_key(self, url_hash: str) -> str:
        """Generate Redis key for content cache."""
        return f"{self.key_prefix}:content:{url_hash}"

    def _make_query_key(self, query_hash: str) -> str:
        """Generate Redis key for query cache."""
        return f"{self.key_prefix}:query:{query_hash}"

    def _make_embedding_key(self, query_hash: str) -> str:
        """Generate Redis key for query embedding."""
        return f"{self.key_prefix}:emb:{query_hash}"

    # =========================================================================
    # Serialization
    # =========================================================================

    def _serialize(self, data: Dict[str, Any]) -> bytes:
        """Serialize data to bytes."""
        if self.enable_msgpack:
            return msgpack.packb(data, use_bin_type=True)
        return json.dumps(data).encode('utf-8')

    def _deserialize(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes to data."""
        if self.enable_msgpack:
            return msgpack.unpackb(data, raw=False)
        return json.loads(data.decode('utf-8'))

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding vector to bytes."""
        return struct.pack(f'{len(embedding)}f', *embedding)

    def _deserialize_embedding(self, blob: bytes) -> List[float]:
        """Deserialize embedding vector from bytes."""
        num_floats = len(blob) // 4
        return list(struct.unpack(f'{num_floats}f', blob))

    # =========================================================================
    # TTL Management
    # =========================================================================

    def _jitter_ttl(self, base_ttl: int, jitter_pct: float = 0.1) -> int:
        """
        Add random jitter to TTL to prevent cache stampede.

        When multiple entries expire simultaneously, they cause a "stampede"
        of refresh requests. Jitter spreads out expiration times.
        """
        jitter = int(base_ttl * jitter_pct)
        return base_ttl + random.randint(-jitter, jitter)

    # =========================================================================
    # Content Cache Operations
    # =========================================================================

    async def get_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached content for a URL.

        Args:
            url: The URL to look up

        Returns:
            Cached content dict or None if not found/expired
        """
        url_hash = self._hash_url(url)

        # Try Redis if circuit allows
        if await self._check_circuit() and self._connected:
            try:
                key = self._make_content_key(url_hash)
                data = await self._redis.get(key)

                if data:
                    await self._record_success()
                    result = self._deserialize(data)

                    # Update hit count (fire and forget)
                    asyncio.create_task(self._increment_hit_count(key))

                    self._stats.content_hits += 1
                    logger.debug(f"Redis cache HIT for {url[:60]}...")

                    result["from_cache"] = True
                    return result

            except Exception as e:
                logger.debug(f"Redis get error: {e}")
                await self._record_failure()

        # Fallback to local cache
        if url_hash in self._local_content:
            entry = self._local_content[url_hash]
            if entry.expires_at > time.time():
                # Move to end (LRU)
                self._local_content.move_to_end(url_hash)
                entry.hit_count += 1

                self._stats.content_hits += 1
                self._stats.fallback_operations += 1
                logger.debug(f"Local cache HIT for {url[:60]}...")

                return {
                    "url": entry.url,
                    "title": entry.title,
                    "content": entry.content,
                    "content_type": entry.content_type,
                    "success": entry.success,
                    "error": entry.error,
                    "from_cache": True
                }
            else:
                # Expired, remove
                del self._local_content[url_hash]

        self._stats.content_misses += 1
        logger.debug(f"Cache MISS for {url[:60]}...")
        return None

    async def set_content(
        self,
        url: str,
        title: str,
        content: str,
        content_type: str,
        success: bool,
        error: Optional[str] = None,
        ttl_override: Optional[int] = None
    ) -> bool:
        """
        Cache scraped content for a URL.

        Args:
            url: The URL that was scraped
            title: Page title
            content: Extracted content
            content_type: html, pdf, error
            success: Whether scraping succeeded
            error: Error message if failed
            ttl_override: Custom TTL for this entry

        Returns:
            True if cached successfully
        """
        url_hash = self._hash_url(url)
        content_hash = self._hash_content(content) if content else ""
        now = time.time()
        base_ttl = ttl_override if ttl_override is not None else self.content_ttl_seconds
        ttl = self._jitter_ttl(base_ttl)
        expires_at = now + ttl

        data = {
            "url": url,
            "url_hash": url_hash,
            "title": title,
            "content": content,
            "content_type": content_type,
            "content_hash": content_hash,
            "success": success,
            "error": error,
            "created_at": now,
            "hit_count": 0
        }

        # Try Redis if circuit allows
        if await self._check_circuit() and self._connected:
            try:
                key = self._make_content_key(url_hash)
                await self._redis.setex(key, ttl, self._serialize(data))
                await self._record_success()
                logger.debug(f"Cached content in Redis for {url[:60]}... (TTL: {ttl}s)")
                return True

            except Exception as e:
                logger.debug(f"Redis set error: {e}")
                await self._record_failure()

        # Fallback to local cache
        self._local_content[url_hash] = CacheEntry(
            url=url,
            url_hash=url_hash,
            title=title,
            content=content,
            content_type=content_type,
            content_hash=content_hash,
            success=success,
            error=error,
            created_at=now,
            expires_at=expires_at
        )

        # Enforce max size (LRU eviction)
        while len(self._local_content) > self.max_local_entries:
            self._local_content.popitem(last=False)

        self._stats.fallback_operations += 1
        logger.debug(f"Cached content locally for {url[:60]}... (TTL: {ttl}s)")
        return True

    async def _increment_hit_count(self, key: str):
        """Increment hit count for a cache entry (async, non-blocking)."""
        try:
            if self._connected and self._redis:
                data = await self._redis.get(key)
                if data:
                    entry = self._deserialize(data)
                    entry["hit_count"] = entry.get("hit_count", 0) + 1
                    ttl = await self._redis.ttl(key)
                    if ttl > 0:
                        await self._redis.setex(key, ttl, self._serialize(entry))
        except Exception:
            pass  # Non-critical operation

    # =========================================================================
    # Query Cache Operations
    # =========================================================================

    async def get_query_result(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.

        Args:
            query: The search query

        Returns:
            Cached result dict or None if not found/expired
        """
        query_hash = self._hash_query(query)

        # Try Redis if circuit allows
        if await self._check_circuit() and self._connected:
            try:
                key = self._make_query_key(query_hash)
                data = await self._redis.get(key)

                if data:
                    await self._record_success()
                    result = self._deserialize(data)

                    asyncio.create_task(self._increment_hit_count(key))

                    self._stats.query_hits += 1
                    logger.info(f"Query cache HIT for '{query[:40]}...'")

                    result["from_cache"] = True
                    return result

            except Exception as e:
                logger.debug(f"Redis get error: {e}")
                await self._record_failure()

        # Fallback to local cache
        if query_hash in self._local_queries:
            entry = self._local_queries[query_hash]
            if entry.get("expires_at", 0) > time.time():
                self._local_queries.move_to_end(query_hash)
                entry["hit_count"] = entry.get("hit_count", 0) + 1

                self._stats.query_hits += 1
                self._stats.fallback_operations += 1
                logger.info(f"Local query cache HIT for '{query[:40]}...'")

                result = dict(entry)
                result["from_cache"] = True
                return result
            else:
                del self._local_queries[query_hash]
                if query_hash in self._local_embeddings:
                    del self._local_embeddings[query_hash]

        self._stats.query_misses += 1
        return None

    async def set_query_result(
        self,
        query: str,
        result: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        ttl_override: Optional[int] = None
    ) -> bool:
        """
        Cache result for a query with optional embedding.

        Args:
            query: The search query
            result: The search result to cache
            embedding: Optional embedding for semantic search
            ttl_override: Custom TTL

        Returns:
            True if cached successfully
        """
        query_hash = self._hash_query(query)
        now = time.time()
        base_ttl = ttl_override if ttl_override is not None else self.query_ttl_seconds
        ttl = self._jitter_ttl(base_ttl)
        expires_at = now + ttl

        # Remove from_cache flag before storing
        data = {k: v for k, v in result.items() if k != "from_cache"}
        data["query"] = query
        data["query_hash"] = query_hash
        data["created_at"] = now
        data["hit_count"] = 0

        # Try Redis if circuit allows
        if await self._check_circuit() and self._connected:
            try:
                query_key = self._make_query_key(query_hash)
                await self._redis.setex(query_key, ttl, self._serialize(data))

                # Store embedding separately if provided
                if embedding:
                    emb_key = self._make_embedding_key(query_hash)
                    await self._redis.setex(
                        emb_key, ttl, self._serialize_embedding(embedding)
                    )

                await self._record_success()
                logger.info(f"Cached query result for '{query[:40]}...' (TTL: {ttl}s)")
                return True

            except Exception as e:
                logger.debug(f"Redis set error: {e}")
                await self._record_failure()

        # Fallback to local cache
        data["expires_at"] = expires_at
        self._local_queries[query_hash] = data

        if embedding:
            self._local_embeddings[query_hash] = self._serialize_embedding(embedding)

        # Enforce max size
        while len(self._local_queries) > self.max_local_entries:
            old_hash = next(iter(self._local_queries))
            del self._local_queries[old_hash]
            if old_hash in self._local_embeddings:
                del self._local_embeddings[old_hash]

        self._stats.fallback_operations += 1
        logger.info(f"Cached query result locally for '{query[:40]}...' (TTL: {ttl}s)")
        return True

    async def find_similar_query(
        self,
        query_embedding: List[float],
        similarity_threshold: float = 0.85,
        max_candidates: int = 100
    ) -> Optional[Dict[str, Any]]:
        """
        Find semantically similar cached query using embedding similarity.

        Args:
            query_embedding: Embedding vector of the query
            similarity_threshold: Minimum cosine similarity
            max_candidates: Max entries to scan for similarity

        Returns:
            Cached result if similar query found, None otherwise
        """
        if not NUMPY_AVAILABLE:
            logger.warning("NumPy not available for semantic similarity")
            return None

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None

        best_match = None
        best_similarity = 0.0

        # Try Redis if circuit allows
        if await self._check_circuit() and self._connected:
            try:
                # Scan for embedding keys
                pattern = f"{self.key_prefix}:emb:*"
                candidates = []

                async for key in self._redis.scan_iter(pattern, count=max_candidates):
                    candidates.append(key)
                    if len(candidates) >= max_candidates:
                        break

                for emb_key in candidates:
                    try:
                        emb_data = await self._redis.get(emb_key)
                        if not emb_data:
                            continue

                        cached_embedding = self._deserialize_embedding(emb_data)
                        cached_vec = np.array(cached_embedding)
                        cached_norm = np.linalg.norm(cached_vec)

                        if cached_norm == 0:
                            continue

                        similarity = float(
                            np.dot(query_vec, cached_vec) / (query_norm * cached_norm)
                        )

                        if similarity >= similarity_threshold and similarity > best_similarity:
                            # Get the query result
                            query_hash = emb_key.decode().split(":")[-1]
                            query_key = self._make_query_key(query_hash)
                            result_data = await self._redis.get(query_key)

                            if result_data:
                                best_similarity = similarity
                                best_match = self._deserialize(result_data)
                                best_match["similarity_score"] = similarity

                    except Exception:
                        continue

                await self._record_success()

            except Exception as e:
                logger.debug(f"Redis semantic search error: {e}")
                await self._record_failure()

        # Also check local cache
        for query_hash, emb_blob in self._local_embeddings.items():
            if query_hash not in self._local_queries:
                continue

            entry = self._local_queries[query_hash]
            if entry.get("expires_at", 0) <= time.time():
                continue

            cached_embedding = self._deserialize_embedding(emb_blob)
            cached_vec = np.array(cached_embedding)
            cached_norm = np.linalg.norm(cached_vec)

            if cached_norm == 0:
                continue

            similarity = float(
                np.dot(query_vec, cached_vec) / (query_norm * cached_norm)
            )

            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = dict(entry)
                best_match["similarity_score"] = similarity

        if best_match:
            self._stats.semantic_hits += 1
            best_match["from_cache"] = True
            best_match["semantic_match"] = True
            logger.info(
                f"Semantic cache HIT for query "
                f"(similarity: {best_similarity:.3f})"
            )

        return best_match

    # =========================================================================
    # Cache Management
    # =========================================================================

    async def invalidate_content(self, url: str) -> bool:
        """Invalidate cached content for a URL."""
        url_hash = self._hash_url(url)
        deleted = False

        if await self._check_circuit() and self._connected:
            try:
                key = self._make_content_key(url_hash)
                deleted = await self._redis.delete(key) > 0
                await self._record_success()
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
                await self._record_failure()

        if url_hash in self._local_content:
            del self._local_content[url_hash]
            deleted = True

        return deleted

    async def invalidate_query(self, query: str) -> bool:
        """Invalidate cached result for a query."""
        query_hash = self._hash_query(query)
        deleted = False

        if await self._check_circuit() and self._connected:
            try:
                query_key = self._make_query_key(query_hash)
                emb_key = self._make_embedding_key(query_hash)
                deleted = await self._redis.delete(query_key, emb_key) > 0
                await self._record_success()
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
                await self._record_failure()

        if query_hash in self._local_queries:
            del self._local_queries[query_hash]
            deleted = True
        if query_hash in self._local_embeddings:
            del self._local_embeddings[query_hash]

        return deleted

    async def clear_all(self) -> int:
        """Clear all cache entries. Returns count of entries cleared."""
        count = 0

        if await self._check_circuit() and self._connected:
            try:
                # Clear content cache
                pattern = f"{self.key_prefix}:content:*"
                async for key in self._redis.scan_iter(pattern):
                    await self._redis.delete(key)
                    count += 1

                # Clear query cache
                pattern = f"{self.key_prefix}:query:*"
                async for key in self._redis.scan_iter(pattern):
                    await self._redis.delete(key)
                    count += 1

                # Clear embedding cache
                pattern = f"{self.key_prefix}:emb:*"
                async for key in self._redis.scan_iter(pattern):
                    await self._redis.delete(key)
                    count += 1

                await self._record_success()

            except Exception as e:
                logger.debug(f"Redis clear error: {e}")
                await self._record_failure()

        # Clear local caches
        count += len(self._local_content) + len(self._local_queries)
        self._local_content.clear()
        self._local_queries.clear()
        self._local_embeddings.clear()

        logger.info(f"Cleared {count} cache entries")
        return count

    async def cleanup_expired(self) -> Tuple[int, int]:
        """
        Remove expired entries from local cache.
        Redis handles TTL automatically.

        Returns:
            Tuple of (content_removed, query_removed)
        """
        now = time.time()
        content_removed = 0
        query_removed = 0

        # Clean content cache
        expired = [k for k, v in self._local_content.items() if v.expires_at <= now]
        for k in expired:
            del self._local_content[k]
            content_removed += 1

        # Clean query cache
        expired = [k for k, v in self._local_queries.items() if v.get("expires_at", 0) <= now]
        for k in expired:
            del self._local_queries[k]
            if k in self._local_embeddings:
                del self._local_embeddings[k]
            query_removed += 1

        if content_removed > 0 or query_removed > 0:
            logger.info(
                f"Local cache cleanup: {content_removed} content, "
                f"{query_removed} query entries removed"
            )

        return content_removed, query_removed

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # Count Redis entries if connected
        redis_content_count = 0
        redis_query_count = 0

        if await self._check_circuit() and self._connected:
            try:
                async for _ in self._redis.scan_iter(f"{self.key_prefix}:content:*"):
                    redis_content_count += 1
                async for _ in self._redis.scan_iter(f"{self.key_prefix}:query:*"):
                    redis_query_count += 1
                await self._record_success()
            except Exception:
                await self._record_failure()

        total_content = redis_content_count + len(self._local_content)
        total_queries = redis_query_count + len(self._local_queries)

        content_requests = self._stats.content_hits + self._stats.content_misses
        query_requests = self._stats.query_hits + self._stats.query_misses

        return {
            "connected": self._connected,
            "circuit_state": self._circuit.state.value,
            "redis_url": self.redis_url if self._connected else "fallback",
            "content_cache": {
                "redis_entries": redis_content_count,
                "local_entries": len(self._local_content),
                "total_entries": total_content,
                "hits": self._stats.content_hits,
                "misses": self._stats.content_misses,
                "hit_rate": (
                    self._stats.content_hits / content_requests
                    if content_requests > 0 else 0.0
                )
            },
            "query_cache": {
                "redis_entries": redis_query_count,
                "local_entries": len(self._local_queries),
                "total_entries": total_queries,
                "hits": self._stats.query_hits,
                "misses": self._stats.query_misses,
                "semantic_hits": self._stats.semantic_hits,
                "hit_rate": (
                    self._stats.query_hits / query_requests
                    if query_requests > 0 else 0.0
                )
            },
            "ttl_config": {
                "content_ttl_seconds": self.content_ttl_seconds,
                "query_ttl_seconds": self.query_ttl_seconds
            },
            "errors": {
                "redis_errors": self._stats.redis_errors,
                "circuit_opens": self._stats.circuit_opens,
                "fallback_operations": self._stats.fallback_operations
            }
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_redis_cache: Optional[RedisCacheService] = None


def get_redis_cache_service(
    redis_url: str = "redis://localhost:6379/3"
) -> RedisCacheService:
    """Get or create singleton RedisCacheService."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCacheService(redis_url=redis_url)
    return _redis_cache


async def get_redis_cache_service_async(
    redis_url: str = "redis://localhost:6379/3"
) -> RedisCacheService:
    """Get or create and connect singleton RedisCacheService."""
    cache = get_redis_cache_service(redis_url)
    await cache.connect()
    return cache
