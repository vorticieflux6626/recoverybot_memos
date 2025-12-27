"""
Scratchpad-Integrated Cache - ROG + LbMAS inspired caching for agentic search.

Based on:
- ROG (2025): Chain-style reasoning with intermediate answer caching
- LbMAS (2025): Blackboard architecture with shared memory optimization

Key features:
- Finding cache: Deduplicate scraped content via semantic hashing
- Sub-query cache: Cache intermediate reasoning results (ROG-style)
- Mission decomposition cache: Reuse query decompositions
- Semantic similarity matching for cache hits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import json
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedFinding:
    """Cached search finding with source attribution"""
    content: str
    source_url: str
    confidence: float
    embedding: Optional[List[float]] = None
    cached_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication"""
        normalized = ' '.join(self.content.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class CachedSubQuery:
    """Cached sub-query result (ROG-style intermediate answer)"""
    query: str
    answer: str
    sources: List[str]
    confidence: float = 0.0
    cached_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    query_hash: str = ""

    def __post_init__(self):
        if not self.query_hash:
            self.query_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute normalized query hash"""
        normalized = ' '.join(self.query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class CachedMissionDecomposition:
    """Cached mission decomposition result"""
    mission: str
    sub_questions: List[Dict[str, Any]]  # [{question, criteria, priority}]
    cached_at: datetime = field(default_factory=datetime.now)
    success_rate: float = 0.0  # How often this decomposition led to good results


class ScratchpadCache:
    """
    Integrated cache layer for the blackboard/scratchpad architecture.

    Combines:
    - LbMAS shared memory patterns for agent coordination
    - ROG intermediate answer caching for reasoning chains
    - Semantic deduplication for finding management
    """

    def __init__(self, db_path: str = "data/scratchpad_cache.db",
                 max_findings: int = 1000, max_subqueries: int = 500):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_findings = max_findings
        self.max_subqueries = max_subqueries

        # In-memory caches for hot data
        self.finding_cache: Dict[str, CachedFinding] = {}
        self.subquery_cache: Dict[str, CachedSubQuery] = {}
        self.mission_cache: Dict[str, CachedMissionDecomposition] = {}

        # Embedding cache for semantic matching
        self.embedding_cache: Dict[str, List[float]] = {}

        # Statistics
        self.stats = {
            'finding_hits': 0,
            'finding_misses': 0,
            'finding_semantic_hits': 0,
            'subquery_hits': 0,
            'subquery_misses': 0,
            'mission_hits': 0,
            'mission_misses': 0,
            'dedup_saved': 0  # Content deduplicated
        }

        # Initialize SQLite for persistence
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for cache persistence"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS findings (
                content_hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_url TEXT,
                confidence REAL,
                embedding BLOB,
                cached_at TEXT,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS subqueries (
                query_hash TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT,  -- JSON array
                confidence REAL,
                cached_at TEXT,
                access_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS missions (
                mission_hash TEXT PRIMARY KEY,
                mission TEXT NOT NULL,
                sub_questions TEXT,  -- JSON array
                cached_at TEXT,
                success_rate REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_findings_cached_at ON findings(cached_at);
            CREATE INDEX IF NOT EXISTS idx_subqueries_cached_at ON subqueries(cached_at);
        """)

        conn.commit()
        conn.close()

        # Load recent entries into memory
        self._load_hot_cache()

    def _load_hot_cache(self, limit: int = 200):
        """Load most recently accessed entries into memory"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Load recent findings
        cursor.execute("""
            SELECT content_hash, content, source_url, confidence, embedding, cached_at, access_count
            FROM findings ORDER BY access_count DESC, cached_at DESC LIMIT ?
        """, (limit,))

        for row in cursor.fetchall():
            embedding = np.frombuffer(row[4], dtype=np.float32).tolist() if row[4] else None
            self.finding_cache[row[0]] = CachedFinding(
                content=row[1],
                source_url=row[2],
                confidence=row[3],
                embedding=embedding,
                cached_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                access_count=row[6],
                content_hash=row[0]
            )

        # Load recent subqueries
        cursor.execute("""
            SELECT query_hash, query, answer, sources, confidence, cached_at, access_count
            FROM subqueries ORDER BY access_count DESC, cached_at DESC LIMIT ?
        """, (limit,))

        for row in cursor.fetchall():
            self.subquery_cache[row[0]] = CachedSubQuery(
                query=row[1],
                answer=row[2],
                sources=json.loads(row[3]) if row[3] else [],
                confidence=row[4],
                cached_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                access_count=row[6],
                query_hash=row[0]
            )

        conn.close()
        logger.info(f"Loaded {len(self.finding_cache)} findings, "
                   f"{len(self.subquery_cache)} subqueries into hot cache")

    # --- Finding Cache (for scraped content) ---

    def cache_finding(self, content: str, source_url: str,
                      confidence: float, embedding: Optional[List[float]] = None) -> str:
        """
        Cache a search finding with source attribution.

        Returns the content hash for deduplication tracking.
        """
        finding = CachedFinding(
            content=content,
            source_url=source_url,
            confidence=confidence,
            embedding=embedding
        )

        # Check for duplicate
        if finding.content_hash in self.finding_cache:
            existing = self.finding_cache[finding.content_hash]
            existing.access_count += 1
            self.stats['dedup_saved'] += 1
            logger.debug(f"Deduplicated finding from {source_url}")
            return finding.content_hash

        # Evict if needed
        if len(self.finding_cache) >= self.max_findings:
            self._evict_findings()

        # Add to memory cache
        self.finding_cache[finding.content_hash] = finding

        # Persist to SQLite
        self._persist_finding(finding)

        return finding.content_hash

    def _persist_finding(self, finding: CachedFinding):
        """Persist finding to SQLite"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        embedding_bytes = None
        if finding.embedding:
            embedding_bytes = np.array(finding.embedding, dtype=np.float32).tobytes()

        cursor.execute("""
            INSERT OR REPLACE INTO findings
            (content_hash, content, source_url, confidence, embedding, cached_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            finding.content_hash,
            finding.content,
            finding.source_url,
            finding.confidence,
            embedding_bytes,
            finding.cached_at.isoformat(),
            finding.access_count
        ))

        conn.commit()
        conn.close()

    def get_finding(self, content: str) -> Optional[CachedFinding]:
        """Check if finding exists in cache by exact content match"""
        normalized = ' '.join(content.lower().split())
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        if content_hash in self.finding_cache:
            self.stats['finding_hits'] += 1
            finding = self.finding_cache[content_hash]
            finding.access_count += 1
            return finding

        self.stats['finding_misses'] += 1
        return None

    def find_similar_findings(self, embedding: List[float],
                              threshold: float = 0.85,
                              max_results: int = 5) -> List[Tuple[CachedFinding, float]]:
        """
        Find semantically similar cached findings.

        Returns list of (finding, similarity_score) tuples.
        """
        similar = []
        query_vec = np.array(embedding)

        for finding in self.finding_cache.values():
            if finding.embedding:
                finding_vec = np.array(finding.embedding)
                similarity = self._cosine_similarity(query_vec, finding_vec)
                if similarity >= threshold:
                    similar.append((finding, similarity))
                    finding.access_count += 1

        if similar:
            self.stats['finding_semantic_hits'] += len(similar)

        # Sort by similarity descending
        similar.sort(key=lambda x: -x[1])
        return similar[:max_results]

    def get_findings_by_source(self, source_url: str) -> List[CachedFinding]:
        """Get all cached findings from a specific source URL"""
        return [f for f in self.finding_cache.values() if f.source_url == source_url]

    # --- Sub-Query Cache (ROG-style intermediate answers) ---

    def cache_subquery(self, query: str, answer: str,
                       sources: List[str], confidence: float = 0.0) -> str:
        """
        Cache intermediate reasoning result (ROG-style).

        Returns the query hash.
        """
        subquery = CachedSubQuery(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence
        )

        # Check for existing
        if subquery.query_hash in self.subquery_cache:
            existing = self.subquery_cache[subquery.query_hash]
            # Update if new answer has higher confidence
            if confidence > existing.confidence:
                existing.answer = answer
                existing.sources = sources
                existing.confidence = confidence
            existing.access_count += 1
            return subquery.query_hash

        # Evict if needed
        if len(self.subquery_cache) >= self.max_subqueries:
            self._evict_subqueries()

        # Add to cache
        self.subquery_cache[subquery.query_hash] = subquery

        # Persist
        self._persist_subquery(subquery)

        return subquery.query_hash

    def _persist_subquery(self, subquery: CachedSubQuery):
        """Persist subquery to SQLite"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO subqueries
            (query_hash, query, answer, sources, confidence, cached_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            subquery.query_hash,
            subquery.query,
            subquery.answer,
            json.dumps(subquery.sources),
            subquery.confidence,
            subquery.cached_at.isoformat(),
            subquery.access_count
        ))

        conn.commit()
        conn.close()

    def get_subquery(self, query: str) -> Optional[CachedSubQuery]:
        """Check if sub-query result is cached"""
        normalized = ' '.join(query.lower().split())
        query_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        if query_hash in self.subquery_cache:
            self.stats['subquery_hits'] += 1
            result = self.subquery_cache[query_hash]
            result.access_count += 1
            return result

        self.stats['subquery_misses'] += 1
        return None

    def get_related_subqueries(self, query: str,
                               embedding: Optional[List[float]] = None,
                               threshold: float = 0.8) -> List[CachedSubQuery]:
        """
        Find related cached sub-queries that might inform the current query.

        Uses keyword overlap if no embedding provided.
        """
        query_words = set(query.lower().split())
        related = []

        for cached in self.subquery_cache.values():
            cached_words = set(cached.query.lower().split())
            overlap = len(query_words & cached_words) / max(len(query_words), 1)
            if overlap >= 0.5:  # 50% word overlap
                related.append(cached)

        # Sort by confidence
        related.sort(key=lambda x: -x.confidence)
        return related[:5]

    # --- Mission Decomposition Cache ---

    def cache_mission_decomposition(self, mission: str,
                                     sub_questions: List[Dict[str, Any]],
                                     success_rate: float = 0.0) -> str:
        """
        Cache query decomposition for similar future queries.

        Returns mission hash.
        """
        normalized = ' '.join(mission.lower().split())
        mission_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        decomposition = CachedMissionDecomposition(
            mission=mission,
            sub_questions=sub_questions,
            success_rate=success_rate
        )

        self.mission_cache[mission_hash] = decomposition

        # Persist
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO missions
            (mission_hash, mission, sub_questions, cached_at, success_rate)
            VALUES (?, ?, ?, ?, ?)
        """, (
            mission_hash,
            mission,
            json.dumps(sub_questions),
            decomposition.cached_at.isoformat(),
            success_rate
        ))
        conn.commit()
        conn.close()

        return mission_hash

    def get_mission_decomposition(self, mission: str) -> Optional[CachedMissionDecomposition]:
        """Retrieve cached mission decomposition"""
        normalized = ' '.join(mission.lower().split())
        mission_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        if mission_hash in self.mission_cache:
            self.stats['mission_hits'] += 1
            return self.mission_cache[mission_hash]

        self.stats['mission_misses'] += 1
        return None

    def update_mission_success_rate(self, mission: str, success: bool):
        """Update success rate for a cached mission decomposition"""
        normalized = ' '.join(mission.lower().split())
        mission_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        if mission_hash in self.mission_cache:
            decomposition = self.mission_cache[mission_hash]
            # Exponential moving average
            alpha = 0.3
            new_value = 1.0 if success else 0.0
            decomposition.success_rate = alpha * new_value + (1 - alpha) * decomposition.success_rate

    # --- Eviction Strategies ---

    def _evict_findings(self, evict_ratio: float = 0.2):
        """LRU eviction for findings cache"""
        # Sort by access count (ascending) then by cached_at (oldest first)
        sorted_findings = sorted(
            self.finding_cache.items(),
            key=lambda x: (x[1].access_count, x[1].cached_at)
        )
        # Remove bottom percentage
        to_remove = max(1, int(len(sorted_findings) * evict_ratio))
        for key, _ in sorted_findings[:to_remove]:
            del self.finding_cache[key]
        logger.debug(f"Evicted {to_remove} findings from cache")

    def _evict_subqueries(self, evict_ratio: float = 0.2):
        """Time-based eviction for sub-query cache"""
        sorted_queries = sorted(
            self.subquery_cache.items(),
            key=lambda x: x[1].cached_at
        )
        to_remove = max(1, int(len(sorted_queries) * evict_ratio))
        for key, _ in sorted_queries[:to_remove]:
            del self.subquery_cache[key]
        logger.debug(f"Evicted {to_remove} subqueries from cache")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # --- Statistics and Maintenance ---

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive cache statistics"""
        total_finding_requests = self.stats['finding_hits'] + self.stats['finding_misses']
        total_subquery_requests = self.stats['subquery_hits'] + self.stats['subquery_misses']
        total_mission_requests = self.stats['mission_hits'] + self.stats['mission_misses']

        return {
            'finding_cache': {
                'size': len(self.finding_cache),
                'hit_rate': self.stats['finding_hits'] / max(1, total_finding_requests),
                'semantic_hits': self.stats['finding_semantic_hits'],
                'dedup_saved': self.stats['dedup_saved']
            },
            'subquery_cache': {
                'size': len(self.subquery_cache),
                'hit_rate': self.stats['subquery_hits'] / max(1, total_subquery_requests)
            },
            'mission_cache': {
                'size': len(self.mission_cache),
                'hit_rate': self.stats['mission_hits'] / max(1, total_mission_requests)
            },
            'raw_stats': self.stats
        }

    def clear_all(self):
        """Clear all caches (for testing or reset)"""
        self.finding_cache.clear()
        self.subquery_cache.clear()
        self.mission_cache.clear()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM findings")
        cursor.execute("DELETE FROM subqueries")
        cursor.execute("DELETE FROM missions")
        conn.commit()
        conn.close()

        self.stats = {k: 0 for k in self.stats}
        logger.info("All scratchpad caches cleared")

    def compact_db(self):
        """Compact SQLite database to reclaim space"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("VACUUM")
        conn.close()
        logger.info("Scratchpad cache database compacted")


# Singleton instance
_scratchpad_cache: Optional[ScratchpadCache] = None


def get_scratchpad_cache(db_path: str = "data/scratchpad_cache.db") -> ScratchpadCache:
    """Get or create the singleton ScratchpadCache instance."""
    global _scratchpad_cache
    if _scratchpad_cache is None:
        _scratchpad_cache = ScratchpadCache(db_path)
    return _scratchpad_cache
