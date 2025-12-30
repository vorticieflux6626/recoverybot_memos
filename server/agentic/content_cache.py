"""
Content Cache for Agentic Search

Implements Phase 2 optimizations:
- Content hash cache for scraped pages (avoid re-scraping)
- Query embedding cache for semantic deduplication
- Result cache with TTL for identical queries

Performance Impact:
- Eliminates redundant scraping (saves ~10s per cached URL)
- Reduces network I/O and external API calls
- Enables faster repeated/similar queries
"""

import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agentic.cache")


class ContentCache:
    """
    SQLite-backed content cache for scraped web pages.

    Features:
    - URL-based caching with content hashing
    - Configurable TTL (from settings or default 1 hour)
    - Automatic cleanup of expired entries
    - Thread-safe operations
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        max_entries: Optional[int] = None
    ):
        """
        Initialize the content cache.

        Args:
            cache_dir: Directory to store cache database
            ttl_seconds: Time-to-live for cache entries (uses settings if None)
            max_entries: Max cache entries before cleanup (uses settings if None)
        """
        # Get settings for defaults
        try:
            from config.settings import get_settings
            settings = get_settings()
            default_ttl = settings.content_cache_ttl
            default_max_entries = settings.max_cache_entries
        except ImportError:
            default_ttl = 3600
            default_max_entries = 10000

        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent / "cache")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / "content_cache.db"
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else default_ttl
        self.max_entries = max_entries if max_entries is not None else default_max_entries
        self._lock = threading.Lock()

        self._init_db()
        logger.info(f"Content cache initialized: {self.db_path} (TTL: {self.ttl_seconds}s, max: {self.max_entries})")

    def _init_db(self):
        """Initialize the SQLite database schema"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Content cache table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS content_cache (
                        url_hash TEXT PRIMARY KEY,
                        url TEXT NOT NULL,
                        title TEXT,
                        content TEXT,
                        content_type TEXT,
                        content_hash TEXT,
                        success INTEGER,
                        error TEXT,
                        created_at REAL,
                        expires_at REAL,
                        hit_count INTEGER DEFAULT 0
                    )
                """)

                # Query result cache table with embedding support
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_cache (
                        query_hash TEXT PRIMARY KEY,
                        query TEXT NOT NULL,
                        result_json TEXT,
                        embedding BLOB,
                        created_at REAL,
                        expires_at REAL,
                        hit_count INTEGER DEFAULT 0
                    )
                """)

                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_expires
                    ON content_cache(expires_at)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_query_expires
                    ON query_cache(expires_at)
                """)

                # Add embedding column if it doesn't exist (migration for existing DBs)
                try:
                    cursor.execute("ALTER TABLE query_cache ADD COLUMN embedding BLOB")
                except sqlite3.OperationalError:
                    pass  # Column already exists

                conn.commit()
            finally:
                conn.close()

    def _hash_url(self, url: str) -> str:
        """Generate hash for URL"""
        return hashlib.sha256(url.encode()).hexdigest()[:32]

    def _hash_content(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        # Normalize query (lowercase, remove extra whitespace)
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _jitter_ttl(self, base_ttl: int, jitter_pct: float = 0.1) -> int:
        """
        Add random jitter to TTL to prevent cache stampede.

        When multiple workers have cached the same content with identical TTLs,
        they all expire at the same time, causing a "stampede" of simultaneous
        refresh requests. Adding jitter spreads out the expiration times.

        Args:
            base_ttl: Base TTL in seconds
            jitter_pct: Percentage of jitter (0.1 = ±10%)

        Returns:
            TTL with random jitter applied
        """
        jitter = int(base_ttl * jitter_pct)
        return base_ttl + random.randint(-jitter, jitter)

    def get_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached content for a URL.

        Returns None if not cached or expired.
        """
        url_hash = self._hash_url(url)
        now = time.time()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT url, title, content, content_type, success, error, hit_count
                    FROM content_cache
                    WHERE url_hash = ? AND expires_at > ?
                """, (url_hash, now))

                row = cursor.fetchone()
                if row:
                    # Update hit count
                    cursor.execute("""
                        UPDATE content_cache SET hit_count = hit_count + 1
                        WHERE url_hash = ?
                    """, (url_hash,))
                    conn.commit()

                    logger.debug(f"Cache HIT for {url[:60]}... (hits: {row[6] + 1})")
                    return {
                        "url": row[0],
                        "title": row[1],
                        "content": row[2],
                        "content_type": row[3],
                        "success": bool(row[4]),
                        "error": row[5],
                        "from_cache": True
                    }

                logger.debug(f"Cache MISS for {url[:60]}...")
                return None
            finally:
                conn.close()

    def set_content(
        self,
        url: str,
        title: str,
        content: str,
        content_type: str,
        success: bool,
        error: Optional[str] = None,
        ttl_override: Optional[int] = None
    ):
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
        """
        url_hash = self._hash_url(url)
        content_hash = self._hash_content(content) if content else ""
        now = time.time()
        base_ttl = ttl_override if ttl_override is not None else self.ttl_seconds
        ttl = self._jitter_ttl(base_ttl)  # Add jitter to prevent cache stampede
        expires_at = now + ttl

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO content_cache
                    (url_hash, url, title, content, content_type, content_hash,
                     success, error, created_at, expires_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    url_hash, url, title, content, content_type, content_hash,
                    int(success), error, now, expires_at
                ))
                conn.commit()
                logger.debug(f"Cached content for {url[:60]}... (TTL: {ttl}s)")
            finally:
                conn.close()

    def get_query_result(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.

        Returns None if not cached or expired.
        """
        query_hash = self._hash_query(query)
        now = time.time()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT result_json, hit_count
                    FROM query_cache
                    WHERE query_hash = ? AND expires_at > ?
                """, (query_hash, now))

                row = cursor.fetchone()
                if row:
                    # Update hit count
                    cursor.execute("""
                        UPDATE query_cache SET hit_count = hit_count + 1
                        WHERE query_hash = ?
                    """, (query_hash,))
                    conn.commit()

                    logger.info(f"Query cache HIT for '{query[:40]}...' (hits: {row[1] + 1})")
                    result = json.loads(row[0])
                    result["from_cache"] = True
                    return result

                return None
            finally:
                conn.close()

    def set_query_result(
        self,
        query: str,
        result: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        ttl_override: Optional[int] = None
    ):
        """
        Cache result for a query with optional embedding for semantic search.

        Args:
            query: The search query
            result: The search result to cache
            embedding: Optional embedding vector for semantic deduplication
            ttl_override: Custom TTL (default: 15 minutes for query results)
        """
        query_hash = self._hash_query(query)
        now = time.time()
        # Query results have shorter TTL (15 min) as they may need fresher data
        base_ttl = ttl_override if ttl_override is not None else 900
        ttl = self._jitter_ttl(base_ttl)  # Add jitter to prevent cache stampede
        expires_at = now + ttl

        # Remove from_cache flag before storing
        result_to_store = {k: v for k, v in result.items() if k != "from_cache"}

        # Serialize embedding if provided
        embedding_blob = None
        if embedding:
            embedding_blob = self._serialize_embedding(embedding)

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO query_cache
                    (query_hash, query, result_json, embedding, created_at, expires_at, hit_count)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                """, (
                    query_hash, query, json.dumps(result_to_store), embedding_blob, now, expires_at
                ))
                conn.commit()
                logger.info(f"Cached query result for '{query[:40]}...' (TTL: {ttl}s, embedding: {embedding is not None})")
            finally:
                conn.close()

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding vector to bytes for SQLite storage"""
        import struct
        return struct.pack(f'{len(embedding)}f', *embedding)

    def _deserialize_embedding(self, blob: bytes) -> List[float]:
        """Deserialize embedding vector from bytes"""
        import struct
        num_floats = len(blob) // 4  # 4 bytes per float
        return list(struct.unpack(f'{num_floats}f', blob))

    def find_similar_query(
        self,
        query_embedding: List[float],
        similarity_threshold: float = 0.85,
        top_k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Find semantically similar cached query using embedding similarity.

        This enables cache hits for queries that are worded differently
        but have the same semantic meaning.

        Args:
            query_embedding: Embedding vector of the query
            similarity_threshold: Minimum cosine similarity (0.85 = very similar)
            top_k: Number of top matches to consider (returns best one)

        Returns:
            Cached result if similar query found, None otherwise
        """
        import numpy as np

        now = time.time()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT query_hash, query, result_json, embedding, hit_count
                    FROM query_cache
                    WHERE expires_at > ? AND embedding IS NOT NULL
                """, (now,))

                rows = cursor.fetchall()
                if not rows:
                    return None

                query_vec = np.array(query_embedding)
                query_norm = np.linalg.norm(query_vec)
                if query_norm == 0:
                    return None

                best_match = None
                best_similarity = 0.0

                for row in rows:
                    query_hash, cached_query, result_json, embedding_blob, hit_count = row

                    if not embedding_blob:
                        continue

                    cached_embedding = self._deserialize_embedding(embedding_blob)
                    cached_vec = np.array(cached_embedding)
                    cached_norm = np.linalg.norm(cached_vec)

                    if cached_norm == 0:
                        continue

                    # Cosine similarity
                    similarity = float(np.dot(query_vec, cached_vec) / (query_norm * cached_norm))

                    if similarity >= similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            "query_hash": query_hash,
                            "cached_query": cached_query,
                            "result_json": result_json,
                            "hit_count": hit_count,
                            "similarity": similarity
                        }

                if best_match:
                    # Update hit count
                    cursor.execute("""
                        UPDATE query_cache SET hit_count = hit_count + 1
                        WHERE query_hash = ?
                    """, (best_match["query_hash"],))
                    conn.commit()

                    logger.info(f"Semantic cache HIT: '{best_match['cached_query'][:30]}...' "
                               f"(similarity: {best_match['similarity']:.3f})")

                    result = json.loads(best_match["result_json"])
                    result["from_cache"] = True
                    result["semantic_match"] = True
                    result["matched_query"] = best_match["cached_query"]
                    result["similarity_score"] = best_match["similarity"]
                    return result

                return None
            finally:
                conn.close()

    def cleanup_expired(self) -> Tuple[int, int]:
        """
        Remove expired entries from both caches.

        Returns:
            Tuple of (content_removed, query_removed)
        """
        now = time.time()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Clean content cache
                cursor.execute("DELETE FROM content_cache WHERE expires_at < ?", (now,))
                content_removed = cursor.rowcount

                # Clean query cache
                cursor.execute("DELETE FROM query_cache WHERE expires_at < ?", (now,))
                query_removed = cursor.rowcount

                conn.commit()

                if content_removed > 0 or query_removed > 0:
                    logger.info(f"Cache cleanup: {content_removed} content, {query_removed} query entries removed")

                return content_removed, query_removed
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Content cache stats
                cursor.execute("""
                    SELECT COUNT(*), SUM(hit_count),
                           SUM(CASE WHEN expires_at > ? THEN 1 ELSE 0 END)
                    FROM content_cache
                """, (time.time(),))
                content_row = cursor.fetchone()

                # Query cache stats
                cursor.execute("""
                    SELECT COUNT(*), SUM(hit_count),
                           SUM(CASE WHEN expires_at > ? THEN 1 ELSE 0 END)
                    FROM query_cache
                """, (time.time(),))
                query_row = cursor.fetchone()

                return {
                    "content_cache": {
                        "total_entries": content_row[0] or 0,
                        "total_hits": content_row[1] or 0,
                        "valid_entries": content_row[2] or 0
                    },
                    "query_cache": {
                        "total_entries": query_row[0] or 0,
                        "total_hits": query_row[1] or 0,
                        "valid_entries": query_row[2] or 0
                    },
                    "ttl_seconds": self.ttl_seconds,
                    "db_path": str(self.db_path)
                }
            finally:
                conn.close()

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM content_cache")
                cursor.execute("DELETE FROM query_cache")
                conn.commit()
                logger.info("Cache cleared")
            finally:
                conn.close()


# Global cache instance
_cache_instance: Optional[ContentCache] = None


def get_content_cache() -> ContentCache:
    """Get the global content cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ContentCache()
    return _cache_instance
