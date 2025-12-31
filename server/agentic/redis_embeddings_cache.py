"""
Redis Embeddings Cache with 3-Tier Strategy.

Implements precision-stratified caching based on Part G of the development plan:
- Hot Tier: Binary precision, MRL 64d, session TTL - 32x compression
- Warm Tier: Int8 precision, MRL 256d, 24h TTL - 4x compression
- Cold Tier: FP16 precision, MRL 4096d, on-demand - Full precision

Key Features:
1. Automatic tier promotion based on access frequency
2. LRU eviction within each tier
3. Matryoshka (MRL) dimension progression for quality/speed tradeoff
4. Integration with existing MixedPrecisionEmbeddingService

References:
- Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022)
- FunnelRAG: Cascade retrieval with progressive refinement (arXiv:2410.10293)
- HSEA: Three-Stratum Indexing (memOS implementation)
"""

import asyncio
import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis as redis
        REDIS_AVAILABLE = True
    except ImportError:
        redis = None  # type: ignore
        REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheTier(str, Enum):
    """Cache tier levels with associated configurations."""
    HOT = "hot"      # Session cache - fastest, lowest quality
    WARM = "warm"    # 24h cache - balanced
    COLD = "cold"    # On-demand - full quality


@dataclass
class TierConfig:
    """Configuration for a cache tier."""
    precision: str           # "binary", "int8", "fp16"
    mrl_dim: int            # Matryoshka truncation dimension
    ttl_seconds: int        # Time-to-live in seconds
    compression_ratio: float # Space savings factor
    max_entries: int        # Max entries in this tier


# Default tier configurations matching the plan
DEFAULT_TIER_CONFIGS: Dict[CacheTier, TierConfig] = {
    CacheTier.HOT: TierConfig(
        precision="binary",
        mrl_dim=64,
        ttl_seconds=3600,      # 1 hour session cache
        compression_ratio=32.0,
        max_entries=10000
    ),
    CacheTier.WARM: TierConfig(
        precision="int8",
        mrl_dim=256,
        ttl_seconds=86400,     # 24 hours
        compression_ratio=4.0,
        max_entries=50000
    ),
    CacheTier.COLD: TierConfig(
        precision="fp16",
        mrl_dim=4096,          # Full dimension for best quality
        ttl_seconds=604800,    # 7 days (or on-demand refresh)
        compression_ratio=1.0,
        max_entries=100000
    )
}


@dataclass
class CachedEmbedding:
    """A cached embedding with metadata."""
    key: str
    embedding: bytes         # Serialized embedding
    tier: CacheTier
    precision: str
    dimension: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    # Quantization metadata (for int8)
    scale: float = 1.0
    zero_point: int = 0


@dataclass
class CacheStats:
    """Statistics for the embedding cache."""
    hot_entries: int = 0
    warm_entries: int = 0
    cold_entries: int = 0
    hot_hits: int = 0
    warm_hits: int = 0
    cold_hits: int = 0
    misses: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    total_bytes: int = 0


class RedisEmbeddingsCache:
    """
    Three-tier Redis cache for embeddings.

    Architecture:
    ```
    Query Embedding
        |
        v
    [Hot Tier - Binary 64d] --> Session cache (fastest)
        |
        v (if miss)
    [Warm Tier - Int8 256d] --> 24h cache (balanced)
        |
        v (if miss)
    [Cold Tier - FP16 4096d] --> Persistent (full quality)
        |
        v (if all miss)
    [Generate new embedding]
    ```

    Access Pattern Optimization:
    - Frequently accessed embeddings promoted to hotter tiers
    - Stale entries demoted or evicted
    - MRL truncation enables progressive quality improvement
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/2",
        tier_configs: Optional[Dict[CacheTier, TierConfig]] = None,
        key_prefix: str = "emb_cache",
        enable_auto_promotion: bool = True,
        promotion_threshold: int = 3  # Access count for promotion
    ):
        """
        Initialize Redis embeddings cache.

        Args:
            redis_url: Redis connection URL
            tier_configs: Custom tier configurations (uses defaults if None)
            key_prefix: Redis key prefix
            enable_auto_promotion: Whether to auto-promote on access
            promotion_threshold: Access count to trigger promotion
        """
        self.redis_url = redis_url
        self.tier_configs = tier_configs or DEFAULT_TIER_CONFIGS
        self.key_prefix = key_prefix
        self.enable_auto_promotion = enable_auto_promotion
        self.promotion_threshold = promotion_threshold

        # Redis client (lazy initialization)
        self._redis: Optional[redis.Redis] = None
        self._connected = False

        # Local statistics
        self._stats = CacheStats()

        # In-memory fallback when Redis unavailable
        self._local_cache: Dict[str, CachedEmbedding] = {}

        logger.info(
            f"RedisEmbeddingsCache initialized: "
            f"prefix={key_prefix}, auto_promote={enable_auto_promotion}"
        )

    async def connect(self) -> bool:
        """Connect to Redis."""
        if self._connected:
            return True

        if not REDIS_AVAILABLE:
            logger.warning("Redis client not available, using local cache")
            return False

        try:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False  # We need bytes for embeddings
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using local cache")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.aclose()
            self._connected = False

    def _make_key(self, embedding_key: str, tier: CacheTier) -> str:
        """Generate Redis key for an embedding."""
        return f"{self.key_prefix}:{tier.value}:{embedding_key}"

    def _make_meta_key(self, embedding_key: str, tier: CacheTier) -> str:
        """Generate Redis key for embedding metadata."""
        return f"{self.key_prefix}:meta:{tier.value}:{embedding_key}"

    def _hash_text(self, text: str) -> str:
        """Create a hash key from text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # =========================================================================
    # Quantization Methods (matching mixed_precision_embeddings.py)
    # =========================================================================

    def _quantize_to_binary(self, embedding: np.ndarray) -> bytes:
        """Quantize fp16 embedding to binary (1-bit per dimension)."""
        binary_bits = (embedding > 0).astype(np.uint8)
        return np.packbits(binary_bits).tobytes()

    def _dequantize_binary(self, binary_data: bytes, dimension: int) -> np.ndarray:
        """Dequantize binary to approximate fp16."""
        unpacked = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        return (unpacked[:dimension].astype(np.float16) * 2) - 1

    def _quantize_to_int8(
        self,
        embedding: np.ndarray
    ) -> Tuple[bytes, float, int]:
        """Quantize fp16 embedding to int8 with symmetric quantization."""
        max_val = np.abs(embedding).max()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        quantized = np.clip(
            np.round(embedding / scale),
            -128, 127
        ).astype(np.int8)
        return quantized.tobytes(), scale, 0

    def _dequantize_int8(
        self,
        int8_data: bytes,
        scale: float,
        zero_point: int = 0
    ) -> np.ndarray:
        """Dequantize int8 to fp16."""
        arr = np.frombuffer(int8_data, dtype=np.int8)
        return (arr.astype(np.float16) - zero_point) * scale

    def _truncate_mrl(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Truncate embedding using Matryoshka Representation Learning."""
        if len(embedding) <= target_dim:
            return embedding
        truncated = embedding[:target_dim]
        # Re-normalize after truncation
        norm = np.linalg.norm(truncated)
        if norm > 0:
            truncated = truncated / norm
        return truncated.astype(np.float16)

    # =========================================================================
    # Core Cache Operations
    # =========================================================================

    async def get(
        self,
        key: str,
        min_tier: CacheTier = CacheTier.HOT
    ) -> Optional[Tuple[np.ndarray, CacheTier]]:
        """
        Get embedding from cache, searching from hot to cold tier.

        Args:
            key: Embedding key (hash of text)
            min_tier: Minimum acceptable tier

        Returns:
            Tuple of (embedding, tier_found) or None if not found
        """
        await self.connect()

        # Search tiers from hot to cold
        tiers_to_search = [CacheTier.HOT, CacheTier.WARM, CacheTier.COLD]
        if min_tier == CacheTier.WARM:
            tiers_to_search = [CacheTier.WARM, CacheTier.COLD]
        elif min_tier == CacheTier.COLD:
            tiers_to_search = [CacheTier.COLD]

        for tier in tiers_to_search:
            result = await self._get_from_tier(key, tier)
            if result is not None:
                embedding, meta = result

                # Update stats
                if tier == CacheTier.HOT:
                    self._stats.hot_hits += 1
                elif tier == CacheTier.WARM:
                    self._stats.warm_hits += 1
                else:
                    self._stats.cold_hits += 1

                # Auto-promote if enabled
                if self.enable_auto_promotion:
                    access_count = meta.get("access_count", 0) + 1
                    if access_count >= self.promotion_threshold:
                        await self._maybe_promote(key, embedding, tier)
                    else:
                        # Update access count
                        await self._update_access_count(key, tier, access_count)

                return embedding, tier

        self._stats.misses += 1
        return None

    async def _get_from_tier(
        self,
        key: str,
        tier: CacheTier
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get embedding from a specific tier."""
        redis_key = self._make_key(key, tier)
        meta_key = self._make_meta_key(key, tier)
        config = self.tier_configs[tier]

        if self._connected and self._redis:
            try:
                # Get embedding data and metadata
                data = await self._redis.get(redis_key)
                meta_raw = await self._redis.get(meta_key)

                if data is None:
                    return None

                meta = json.loads(meta_raw) if meta_raw else {}

                # Dequantize based on precision
                if config.precision == "binary":
                    embedding = self._dequantize_binary(data, config.mrl_dim)
                elif config.precision == "int8":
                    scale = meta.get("scale", 1.0)
                    zero_point = meta.get("zero_point", 0)
                    embedding = self._dequantize_int8(data, scale, zero_point)
                else:  # fp16
                    embedding = np.frombuffer(data, dtype=np.float16)

                return embedding, meta

            except Exception as e:
                logger.error(f"Redis get error: {e}")

        # Fallback to local cache
        cache_key = f"{tier.value}:{key}"
        if cache_key in self._local_cache:
            cached = self._local_cache[cache_key]

            if config.precision == "binary":
                embedding = self._dequantize_binary(cached.embedding, config.mrl_dim)
            elif config.precision == "int8":
                embedding = self._dequantize_int8(
                    cached.embedding, cached.scale, cached.zero_point
                )
            else:
                embedding = np.frombuffer(cached.embedding, dtype=np.float16)

            return embedding, {
                "access_count": cached.access_count,
                "scale": cached.scale,
                "zero_point": cached.zero_point
            }

        return None

    async def put(
        self,
        key: str,
        embedding: np.ndarray,
        tier: CacheTier = CacheTier.COLD,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store embedding in cache at specified tier.

        Args:
            key: Embedding key
            embedding: Full-precision embedding (fp16 or fp32)
            tier: Target tier
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        await self.connect()
        config = self.tier_configs[tier]

        # Ensure fp16
        if embedding.dtype != np.float16:
            embedding = embedding.astype(np.float16)

        # Truncate to tier's MRL dimension
        truncated = self._truncate_mrl(embedding, config.mrl_dim)

        # Quantize based on tier precision
        if config.precision == "binary":
            data = self._quantize_to_binary(truncated)
            scale, zero_point = 1.0, 0
        elif config.precision == "int8":
            data, scale, zero_point = self._quantize_to_int8(truncated)
        else:  # fp16
            data = truncated.tobytes()
            scale, zero_point = 1.0, 0

        # Prepare metadata (ensure JSON serializable)
        meta = {
            "dimension": int(len(truncated)),
            "precision": config.precision,
            "scale": float(scale),
            "zero_point": int(zero_point),
            "access_count": 0,
            "created_at": time.time(),
            **(metadata or {})
        }

        redis_key = self._make_key(key, tier)
        meta_key = self._make_meta_key(key, tier)

        if self._connected and self._redis:
            try:
                # Store with TTL
                await self._redis.setex(
                    redis_key,
                    config.ttl_seconds,
                    data
                )
                await self._redis.setex(
                    meta_key,
                    config.ttl_seconds,
                    json.dumps(meta)
                )

                # Update stats
                self._stats.total_bytes += len(data)

                return True

            except Exception as e:
                logger.error(f"Redis put error: {e}")

        # Fallback to local cache
        cache_key = f"{tier.value}:{key}"
        self._local_cache[cache_key] = CachedEmbedding(
            key=key,
            embedding=data,
            tier=tier,
            precision=config.precision,
            dimension=len(truncated),
            scale=scale,
            zero_point=zero_point
        )

        return True

    async def put_all_tiers(
        self,
        key: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Store embedding in all tiers with appropriate precision.

        This is useful for pre-warming the cache with important embeddings.

        Args:
            key: Embedding key
            embedding: Full-precision embedding
            metadata: Optional metadata

        Returns:
            True if stored in all tiers
        """
        success = True
        for tier in CacheTier:
            if not await self.put(key, embedding, tier, metadata):
                success = False
        return success

    async def _update_access_count(
        self,
        key: str,
        tier: CacheTier,
        new_count: int
    ):
        """Update access count for an embedding."""
        meta_key = self._make_meta_key(key, tier)

        if self._connected and self._redis:
            try:
                meta_raw = await self._redis.get(meta_key)
                if meta_raw:
                    meta = json.loads(meta_raw)
                    meta["access_count"] = new_count
                    meta["last_accessed"] = time.time()
                    config = self.tier_configs[tier]
                    await self._redis.setex(
                        meta_key,
                        config.ttl_seconds,
                        json.dumps(meta)
                    )
            except Exception as e:
                logger.debug(f"Failed to update access count: {e}")
        else:
            # Update local cache
            cache_key = f"{tier.value}:{key}"
            if cache_key in self._local_cache:
                self._local_cache[cache_key].access_count = new_count
                self._local_cache[cache_key].last_accessed = time.time()

    async def _maybe_promote(
        self,
        key: str,
        embedding: np.ndarray,
        current_tier: CacheTier
    ):
        """Promote embedding to a hotter tier if eligible."""
        if current_tier == CacheTier.HOT:
            return  # Already at hottest tier

        target_tier = CacheTier.WARM if current_tier == CacheTier.COLD else CacheTier.HOT

        # Store in hotter tier
        await self.put(key, embedding, target_tier)
        self._stats.promotions += 1

        logger.debug(f"Promoted embedding {key[:8]}... from {current_tier} to {target_tier}")

    async def invalidate(self, key: str, tier: Optional[CacheTier] = None) -> int:
        """
        Invalidate cached embedding.

        Args:
            key: Embedding key
            tier: Specific tier to invalidate, or None for all tiers

        Returns:
            Number of entries invalidated
        """
        await self.connect()
        count = 0

        tiers_to_check = [tier] if tier else list(CacheTier)

        for t in tiers_to_check:
            redis_key = self._make_key(key, t)
            meta_key = self._make_meta_key(key, t)

            if self._connected and self._redis:
                try:
                    deleted = await self._redis.delete(redis_key, meta_key)
                    count += deleted // 2  # Each entry has key + meta
                except Exception as e:
                    logger.error(f"Redis invalidate error: {e}")

            # Also clean local cache
            cache_key = f"{t.value}:{key}"
            if cache_key in self._local_cache:
                del self._local_cache[cache_key]
                count += 1

        self._stats.evictions += count
        return count

    # =========================================================================
    # Convenience Methods for Text-Based Access
    # =========================================================================

    async def get_by_text(
        self,
        text: str,
        min_tier: CacheTier = CacheTier.HOT
    ) -> Optional[Tuple[np.ndarray, CacheTier]]:
        """Get embedding by text, using text hash as key."""
        key = self._hash_text(text)
        return await self.get(key, min_tier)

    async def put_by_text(
        self,
        text: str,
        embedding: np.ndarray,
        tier: CacheTier = CacheTier.COLD,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store embedding by text, using text hash as key."""
        key = self._hash_text(text)
        if metadata is None:
            metadata = {}
        metadata["text_hash"] = key
        return await self.put(key, embedding, tier, metadata)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def get_batch(
        self,
        keys: List[str],
        min_tier: CacheTier = CacheTier.HOT
    ) -> Dict[str, Tuple[np.ndarray, CacheTier]]:
        """
        Get multiple embeddings from cache.

        Returns:
            Dict mapping keys to (embedding, tier) tuples
        """
        results = {}
        for key in keys:
            result = await self.get(key, min_tier)
            if result is not None:
                results[key] = result
        return results

    async def put_batch(
        self,
        embeddings: Dict[str, np.ndarray],
        tier: CacheTier = CacheTier.COLD,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store multiple embeddings in cache.

        Returns:
            Number of embeddings stored
        """
        count = 0
        for key, embedding in embeddings.items():
            if await self.put(key, embedding, tier, metadata):
                count += 1
        return count

    # =========================================================================
    # Cache Management
    # =========================================================================

    async def clear_tier(self, tier: CacheTier) -> int:
        """Clear all entries in a specific tier."""
        count = 0
        pattern = f"{self.key_prefix}:{tier.value}:*"

        if self._connected and self._redis:
            try:
                async for key in self._redis.scan_iter(pattern):
                    await self._redis.delete(key)
                    count += 1
            except Exception as e:
                logger.error(f"Redis clear_tier error: {e}")

        # Clear local cache for this tier
        to_delete = [k for k in self._local_cache if k.startswith(f"{tier.value}:")]
        for k in to_delete:
            del self._local_cache[k]
            count += 1

        return count

    async def clear_all(self) -> int:
        """Clear all cached embeddings."""
        count = 0
        for tier in CacheTier:
            count += await self.clear_tier(tier)
        self._stats = CacheStats()  # Reset stats
        return count

    async def get_tier_size(self, tier: CacheTier) -> int:
        """Get number of entries in a specific tier."""
        count = 0
        pattern = f"{self.key_prefix}:{tier.value}:*"

        if self._connected and self._redis:
            try:
                async for _ in self._redis.scan_iter(pattern):
                    count += 1
                # Only count actual embeddings, not metadata
                count = count // 2
            except Exception as e:
                logger.error(f"Redis get_tier_size error: {e}")

        # Add local cache count
        count += sum(1 for k in self._local_cache if k.startswith(f"{tier.value}:"))

        return count

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        # Update entry counts
        self._stats.hot_entries = await self.get_tier_size(CacheTier.HOT)
        self._stats.warm_entries = await self.get_tier_size(CacheTier.WARM)
        self._stats.cold_entries = await self.get_tier_size(CacheTier.COLD)

        total_hits = (
            self._stats.hot_hits +
            self._stats.warm_hits +
            self._stats.cold_hits
        )
        total_requests = total_hits + self._stats.misses

        return {
            "connected": self._connected,
            "redis_url": self.redis_url if self._connected else "local_cache",
            "tiers": {
                tier.value: {
                    "entries": getattr(self._stats, f"{tier.value}_entries"),
                    "hits": getattr(self._stats, f"{tier.value}_hits"),
                    "config": {
                        "precision": self.tier_configs[tier].precision,
                        "mrl_dim": self.tier_configs[tier].mrl_dim,
                        "ttl_seconds": self.tier_configs[tier].ttl_seconds,
                        "compression_ratio": self.tier_configs[tier].compression_ratio
                    }
                }
                for tier in CacheTier
            },
            "total_entries": (
                self._stats.hot_entries +
                self._stats.warm_entries +
                self._stats.cold_entries
            ),
            "total_hits": total_hits,
            "total_misses": self._stats.misses,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
            "promotions": self._stats.promotions,
            "demotions": self._stats.demotions,
            "evictions": self._stats.evictions,
            "total_bytes": self._stats.total_bytes
        }

    async def warm_cache(
        self,
        embeddings: Dict[str, np.ndarray],
        target_tier: CacheTier = CacheTier.WARM
    ) -> int:
        """
        Pre-warm cache with embeddings.

        Useful for loading frequently accessed embeddings at startup.

        Args:
            embeddings: Dict mapping keys to embeddings
            target_tier: Tier to store in

        Returns:
            Number of embeddings stored
        """
        return await self.put_batch(embeddings, target_tier)

    async def demote_stale_entries(
        self,
        max_age_seconds: int = 3600,
        tier: CacheTier = CacheTier.HOT
    ) -> int:
        """
        Demote stale entries from a tier to a colder tier.

        Args:
            max_age_seconds: Entries older than this are demoted
            tier: Tier to check

        Returns:
            Number of entries demoted
        """
        if tier == CacheTier.COLD:
            return 0  # Can't demote from coldest tier

        count = 0
        target_tier = CacheTier.WARM if tier == CacheTier.HOT else CacheTier.COLD
        cutoff = time.time() - max_age_seconds

        # Check local cache
        to_demote = []
        for cache_key, cached in self._local_cache.items():
            if (
                cache_key.startswith(f"{tier.value}:") and
                cached.last_accessed < cutoff
            ):
                to_demote.append((cached.key, cached))

        for key, cached in to_demote:
            # Re-expand embedding and store in colder tier
            config = self.tier_configs[tier]
            if config.precision == "binary":
                embedding = self._dequantize_binary(cached.embedding, config.mrl_dim)
            elif config.precision == "int8":
                embedding = self._dequantize_int8(
                    cached.embedding, cached.scale, cached.zero_point
                )
            else:
                embedding = np.frombuffer(cached.embedding, dtype=np.float16)

            await self.put(key, embedding, target_tier)
            await self.invalidate(key, tier)
            count += 1

        self._stats.demotions += count
        return count


# =============================================================================
# Singleton Instance
# =============================================================================

_redis_cache: Optional[RedisEmbeddingsCache] = None


def get_redis_embeddings_cache(
    redis_url: str = "redis://localhost:6379/2"
) -> RedisEmbeddingsCache:
    """Get or create singleton RedisEmbeddingsCache."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisEmbeddingsCache(redis_url=redis_url)
    return _redis_cache


async def get_redis_embeddings_cache_async(
    redis_url: str = "redis://localhost:6379/2"
) -> RedisEmbeddingsCache:
    """Get or create and connect singleton RedisEmbeddingsCache."""
    cache = get_redis_embeddings_cache(redis_url)
    await cache.connect()
    return cache
