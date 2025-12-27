"""
Three-Tier Memory Architecture for Agentic Search

Implements MemOS MemCube-inspired memory hierarchy for 80-94% TTFT reduction:

Tier 1: PLAINTEXT MEMORY (Cold)
  - Documents, knowledge graphs, prompt templates
  - Storage: PostgreSQL + local cache
  - Latency: 10-50ms

Tier 2: ACTIVATION MEMORY (Warm)
  - KV cache states, precomputed attention
  - Storage: Inference engine VRAM via KVCacheService
  - Latency: 1-5ms (cache hit)

Tier 3: PARAMETRIC MEMORY (Hot)
  - LoRA weights, fine-tuned parameters
  - Storage: Model weights in VRAM
  - Latency: 0ms (always loaded)
  - Note: Not implemented yet - requires model customization

The key insight from MemOS research (arxiv:2501.09136):
- Frequently accessed plaintext can be "promoted" to activation memory
- This precomputes the KV cache, eliminating TTFT on subsequent access
- Infrequently accessed content is "demoted" back to plaintext

Ref: KV_CACHE_IMPLEMENTATION_PLAN.md Phase 4.1
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import json

from .kv_cache_service import KVCacheService, get_kv_cache_service, CacheState

logger = logging.getLogger("agentic.memory_tiers")


class MemoryTier(str, Enum):
    """Memory tier levels"""
    COLD = "cold"       # Plaintext storage (PostgreSQL)
    WARM = "warm"       # KV cache (activation memory)
    HOT = "hot"         # Parametric (LoRA weights) - future


class ContentType(str, Enum):
    """Types of content stored in memory tiers"""
    SYSTEM_PROMPT = "system_prompt"
    CONTEXT_CHUNK = "context_chunk"
    SCRAPED_CONTENT = "scraped_content"
    USER_MEMORY = "user_memory"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SEARCH_RESULT = "search_result"


@dataclass
class MemoryEntry:
    """An entry in the memory tier system"""
    content_id: str
    content_hash: str
    content: str
    content_type: ContentType
    tier: MemoryTier = MemoryTier.COLD
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Access tracking
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Promotion tracking
    promoted_at: Optional[datetime] = None
    demoted_at: Optional[datetime] = None

    # Size estimation
    estimated_tokens: int = 0

    def touch(self):
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "content_hash": self.content_hash,
            "content_type": self.content_type.value,
            "tier": self.tier.value,
            "access_count": self.access_count,
            "estimated_tokens": self.estimated_tokens,
            "last_accessed": self.last_accessed.isoformat(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


class PlaintextStorage:
    """
    Cold tier storage using in-memory cache + optional PostgreSQL.

    For now, uses in-memory storage. Can be extended to use
    the existing memOS PostgreSQL database for persistence.
    """

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._storage: Dict[str, MemoryEntry] = {}

    async def store(
        self,
        content_id: str,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict] = None
    ) -> MemoryEntry:
        """Store content in cold tier"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]

        entry = MemoryEntry(
            content_id=content_id,
            content_hash=content_hash,
            content=content,
            content_type=content_type,
            tier=MemoryTier.COLD,
            metadata=metadata or {},
            estimated_tokens=len(content) // 4
        )

        self._storage[content_id] = entry

        # Evict oldest entries if over limit
        if len(self._storage) > self.max_entries:
            self._evict_oldest()

        return entry

    async def get(self, content_id: str) -> Optional[MemoryEntry]:
        """Retrieve content from cold tier"""
        entry = self._storage.get(content_id)
        if entry:
            entry.touch()
        return entry

    async def delete(self, content_id: str) -> bool:
        """Delete content from cold tier"""
        if content_id in self._storage:
            del self._storage[content_id]
            return True
        return False

    async def search(
        self,
        content_type: Optional[ContentType] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Search cold tier by content type"""
        results = []
        for entry in self._storage.values():
            if content_type is None or entry.content_type == content_type:
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    def _evict_oldest(self):
        """Evict oldest entries when over limit"""
        if len(self._storage) <= self.max_entries:
            return

        # Sort by last_accessed, remove oldest
        sorted_entries = sorted(
            self._storage.items(),
            key=lambda x: x[1].last_accessed
        )

        to_evict = len(self._storage) - self.max_entries
        for content_id, _ in sorted_entries[:to_evict]:
            del self._storage[content_id]

        logger.debug(f"Evicted {to_evict} entries from cold storage")

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        by_type = {}
        total_tokens = 0
        for entry in self._storage.values():
            type_key = entry.content_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            total_tokens += entry.estimated_tokens

        return {
            "total_entries": len(self._storage),
            "by_type": by_type,
            "estimated_tokens": total_tokens,
            "max_entries": self.max_entries
        }


class MemoryTierManager:
    """
    Manages memory promotion/demotion between tiers.

    Key behaviors:
    - Content starts in cold tier (plaintext storage)
    - After N accesses, content is promoted to warm tier (KV cache)
    - Warm content has near-zero TTFT on subsequent access
    - Content can be demoted back to cold on eviction or inactivity

    Usage:
        manager = MemoryTierManager()

        # Store content (starts cold)
        entry = await manager.store(
            content_id="ctx_123",
            content="User is interested in recovery centers...",
            content_type=ContentType.CONTEXT_CHUNK
        )

        # Access multiple times
        for _ in range(3):
            ctx = await manager.get_context("ctx_123")

        # Content is now warm (promoted to KV cache)
        assert entry.tier == MemoryTier.WARM
    """

    def __init__(
        self,
        kv_cache_service: Optional[KVCacheService] = None,
        plaintext_storage: Optional[PlaintextStorage] = None,
        promotion_threshold: int = 3,
        demotion_hours: int = 24,
        auto_promote_types: Optional[Set[ContentType]] = None
    ):
        """
        Initialize memory tier manager.

        Args:
            kv_cache_service: Service for warm tier (KV cache)
            plaintext_storage: Service for cold tier (plaintext)
            promotion_threshold: Access count to trigger auto-promotion
            demotion_hours: Hours of inactivity before demotion
            auto_promote_types: Content types to promote immediately
        """
        self.kv_cache = kv_cache_service or get_kv_cache_service()
        self.plaintext = plaintext_storage or PlaintextStorage()
        self.promotion_threshold = promotion_threshold
        self.demotion_hours = demotion_hours
        self.auto_promote_types = auto_promote_types or {
            ContentType.SYSTEM_PROMPT  # Always promote system prompts
        }

        # Track what's in each tier
        self._warm_entries: Set[str] = set()

        # Stats
        self._stats = {
            "promotions": 0,
            "demotions": 0,
            "cold_hits": 0,
            "warm_hits": 0,
        }

        logger.info(
            f"MemoryTierManager initialized: "
            f"promotion_threshold={promotion_threshold}, "
            f"demotion_hours={demotion_hours}"
        )

    async def store(
        self,
        content_id: str,
        content: str,
        content_type: ContentType,
        metadata: Optional[Dict] = None,
        initial_tier: MemoryTier = MemoryTier.COLD
    ) -> MemoryEntry:
        """
        Store content in the memory tier system.

        Args:
            content_id: Unique identifier for the content
            content: The text content to store
            content_type: Type of content (affects promotion behavior)
            metadata: Optional metadata
            initial_tier: Starting tier (default: COLD)

        Returns:
            MemoryEntry with storage details
        """
        # Store in plaintext first
        entry = await self.plaintext.store(
            content_id=content_id,
            content=content,
            content_type=content_type,
            metadata=metadata
        )

        # Auto-promote certain content types
        if content_type in self.auto_promote_types:
            await self.promote_to_warm(content_id)
        elif initial_tier == MemoryTier.WARM:
            await self.promote_to_warm(content_id)

        return entry

    async def get_context(
        self,
        content_id: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve context, checking warm tier first.

        Automatically tracks access and promotes frequently
        accessed content to warm tier.

        Args:
            content_id: Content identifier
            user_id: Optional user ID for personalization

        Returns:
            Dict with source tier and content, or None if not found
        """
        # Check warm tier first
        if content_id in self._warm_entries:
            cache_entry = await self.kv_cache.get(content_id)
            if cache_entry and cache_entry.state == CacheState.WARM:
                self._stats["warm_hits"] += 1
                # Also touch the plaintext entry
                plaintext_entry = await self.plaintext.get(content_id)
                return {
                    "source": "warm",
                    "tier": MemoryTier.WARM.value,
                    "content": plaintext_entry.content if plaintext_entry else None,
                    "entry": plaintext_entry.to_dict() if plaintext_entry else None
                }

        # Fall back to cold tier
        entry = await self.plaintext.get(content_id)
        if not entry:
            return None

        self._stats["cold_hits"] += 1

        # Check if should promote
        if entry.access_count >= self.promotion_threshold:
            await self.promote_to_warm(content_id)

        return {
            "source": "cold",
            "tier": MemoryTier.COLD.value,
            "content": entry.content,
            "entry": entry.to_dict()
        }

    async def promote_to_warm(self, content_id: str) -> bool:
        """
        Promote content from cold to warm tier.

        This precomputes the KV cache for the content,
        significantly reducing TTFT on subsequent access.
        """
        entry = await self.plaintext.get(content_id)
        if not entry:
            logger.warning(f"Cannot promote {content_id}: not found in cold tier")
            return False

        if content_id in self._warm_entries:
            logger.debug(f"Content {content_id[:8]} already warm")
            return True

        try:
            # Warm the KV cache
            cache_id = await self.kv_cache.warm_prefix(
                prefix=entry.content,
                content_id=content_id
            )

            # Update tracking
            entry.tier = MemoryTier.WARM
            entry.promoted_at = datetime.now(timezone.utc)
            self._warm_entries.add(content_id)
            self._stats["promotions"] += 1

            logger.info(
                f"Promoted {content_id[:8]} to warm tier "
                f"(~{entry.estimated_tokens} tokens)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to promote {content_id[:8]}: {e}")
            return False

    async def demote_to_cold(self, content_id: str) -> bool:
        """
        Demote content from warm back to cold tier.

        This marks the KV cache as evicted but keeps the
        plaintext content available.
        """
        if content_id not in self._warm_entries:
            return False

        entry = await self.plaintext.get(content_id)
        if entry:
            entry.tier = MemoryTier.COLD
            entry.demoted_at = datetime.now(timezone.utc)

        # Mark as evicted in KV cache service
        await self.kv_cache.evict(content_id)

        self._warm_entries.discard(content_id)
        self._stats["demotions"] += 1

        logger.info(f"Demoted {content_id[:8]} to cold tier")
        return True

    async def demote_inactive(self) -> int:
        """
        Demote warm entries that haven't been accessed recently.

        Returns count of entries demoted.
        """
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(hours=self.demotion_hours)
        to_demote = []

        for content_id in self._warm_entries:
            entry = await self.plaintext.get(content_id)
            if entry and entry.last_accessed < threshold:
                to_demote.append(content_id)

        for content_id in to_demote:
            await self.demote_to_cold(content_id)

        if to_demote:
            logger.info(f"Demoted {len(to_demote)} inactive entries")

        return len(to_demote)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive tier statistics"""
        cold_stats = self.plaintext.get_stats()
        kv_stats = self.kv_cache.get_stats()

        total_hits = self._stats["cold_hits"] + self._stats["warm_hits"]
        warm_hit_rate = self._stats["warm_hits"] / max(total_hits, 1)

        return {
            "tier_stats": {
                "warm_entries": len(self._warm_entries),
                "cold_entries": cold_stats["total_entries"],
                "promotions": self._stats["promotions"],
                "demotions": self._stats["demotions"],
            },
            "hit_stats": {
                "warm_hits": self._stats["warm_hits"],
                "cold_hits": self._stats["cold_hits"],
                "warm_hit_rate": round(warm_hit_rate, 3),
            },
            "cold_storage": cold_stats,
            "kv_cache": kv_stats,
            "config": {
                "promotion_threshold": self.promotion_threshold,
                "demotion_hours": self.demotion_hours,
                "auto_promote_types": [t.value for t in self.auto_promote_types]
            }
        }

    def get_warm_content_ids(self) -> List[str]:
        """Get list of content IDs currently in warm tier"""
        return list(self._warm_entries)


# Global instance
_tier_manager: Optional[MemoryTierManager] = None


def get_memory_tier_manager() -> MemoryTierManager:
    """Get the global memory tier manager instance"""
    global _tier_manager
    if _tier_manager is None:
        _tier_manager = MemoryTierManager()
    return _tier_manager


async def initialize_memory_tiers():
    """
    Initialize memory tier system with pre-warmed content.

    Call this during server startup to:
    1. Initialize the tier manager
    2. Pre-warm system prompts
    3. Load frequently accessed content
    """
    from .kv_cache_service import warm_system_prompts

    manager = get_memory_tier_manager()

    # Warm system prompts
    await warm_system_prompts()

    logger.info("Memory tier system initialized")
    return manager
