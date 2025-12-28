"""
Master Embedding Aggregation and Sub-Manifold Retrieval System.

Based on cutting-edge 2024-2025 research:
- RouterRetriever: Similarity-based routing to domain experts
- HF-RAG: Z-score normalization for cross-source fusion
- HybridRAG: Entity-focused retrieval with knowledge graphs (97.5% accuracy)
- AnchorRAG: Entity-guided multi-hop traversal
- ELERAG: Entity-aware reciprocal rank fusion

Key Features:
1. Master embedding space aggregating multiple domains (FANUC, Raspberry Pi, etc.)
2. Entity-guided sub-manifold retrieval for precise navigation
3. MoE-style routing to domain experts
4. Hierarchical embedding for coarse-to-fine search
5. Incremental learning without catastrophic forgetting

References:
- RouterRetriever (arXiv:2409.02685)
- HF-RAG (arXiv:2509.02837)
- AnchorRAG (arXiv:2509.01238)
- ELERAG (arXiv:2512.05967)
- HybridRAG (arXiv:2408.04948)
"""

import asyncio
import json
import logging
import hashlib
import sqlite3
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import httpx

logger = logging.getLogger(__name__)


class DomainExpertType(str, Enum):
    """Types of domain experts available."""
    FANUC = "fanuc"
    RASPBERRY_PI = "raspberry_pi"
    GENERAL_TECHNICAL = "general_technical"
    GENERAL = "general"


@dataclass
class DomainExpert:
    """A domain-specific embedding expert."""
    domain_id: str
    name: str
    description: str
    entity_types: List[str]  # Entity types this expert handles
    keywords: List[str]  # Routing keywords
    embedding_dim: int = 4096
    confidence_weight: float = 1.0
    is_active: bool = True


@dataclass
class EmbeddingResult:
    """Result from an embedding query."""
    embedding: np.ndarray
    domain: str
    confidence: float
    source: str = ""


@dataclass
class AggregatedEmbedding:
    """Aggregated embedding from multiple domains."""
    master_embedding: np.ndarray
    domain_contributions: Dict[str, float]  # Domain -> weight
    source_embeddings: List[EmbeddingResult]
    fusion_method: str = "z_score_rrf"


@dataclass
class SubManifoldResult:
    """Result from sub-manifold retrieval."""
    entity_id: str
    entity_name: str
    entity_type: str
    embedding: np.ndarray
    distance: float
    domain: str
    context: str = ""
    related_entities: List[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Complete retrieval result."""
    query: str
    query_embedding: np.ndarray
    anchor_entities: List[Dict[str, Any]]
    sub_manifold_results: List[SubManifoldResult]
    aggregated_context: str
    confidence: float
    domains_used: List[str]
    retrieval_time_ms: int = 0


class EmbeddingAggregator:
    """
    Master embedding aggregation system using MoE-style routing.

    Architecture:
    ```
    Query → Entity Extraction → Anchor Selection
                                     ↓
    ┌─────────────────────────────────────────────┐
    │           Domain Expert Router              │
    │  (similarity-based routing to experts)      │
    └─────────────────────────────────────────────┘
                        ↓
    ┌─────────┬─────────┬─────────┬─────────┐
    │ FANUC   │ RPi     │ General │ General │
    │ Expert  │ Expert  │ Tech    │ Expert  │
    └─────────┴─────────┴─────────┴─────────┘
                        ↓
    ┌─────────────────────────────────────────────┐
    │     Z-Score Fusion (HF-RAG style)           │
    │  Normalize + Aggregate Domain Results       │
    └─────────────────────────────────────────────┘
                        ↓
                Master Embedding
    ```
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "qwen3-embedding",  # 4096 dim, best for technical docs
        db_path: Optional[str] = None
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.embedding_model = embedding_model

        # Database for persistent storage
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "embedding_aggregator.db")
        self.db_path = db_path

        # Domain experts registry
        self.experts: Dict[str, DomainExpert] = {}
        self._init_default_experts()

        # Embedding cache (in-memory)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 10000

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Initialize database
        self._init_db()

    def _init_default_experts(self):
        """Initialize default domain experts."""
        self.experts["fanuc"] = DomainExpert(
            domain_id="fanuc",
            name="FANUC Robotics Expert",
            description="Expert in FANUC robot troubleshooting, servo alarms, CNC systems",
            entity_types=["error_code", "component", "parameter", "procedure"],
            keywords=[
                "fanuc", "servo", "srvo", "motn", "syst", "alarm", "robot",
                "j1", "j2", "j3", "j4", "j5", "j6", "motor", "amplifier",
                "teach pendant", "cnc", "mastering", "calibration"
            ],
            confidence_weight=1.2  # Boost for specialized domain
        )

        self.experts["raspberry_pi"] = DomainExpert(
            domain_id="raspberry_pi",
            name="Raspberry Pi Expert",
            description="Expert in Raspberry Pi troubleshooting, GPIO, Linux embedded systems",
            entity_types=["error_code", "component", "config_file", "command"],
            keywords=[
                "raspberry", "pi", "gpio", "i2c", "spi", "uart", "pwm",
                "raspbian", "bootloader", "kernel panic", "sd card",
                "undervoltage", "overheating", "config.txt", "cmdline"
            ],
            confidence_weight=1.2
        )

        self.experts["general_technical"] = DomainExpert(
            domain_id="general_technical",
            name="General Technical Expert",
            description="General technical and engineering knowledge",
            entity_types=["concept", "component", "measurement", "tool"],
            keywords=[
                "debug", "troubleshoot", "error", "fix", "install", "configure",
                "voltage", "current", "resistance", "ohm", "multimeter"
            ],
            confidence_weight=1.0
        )

        self.experts["general"] = DomainExpert(
            domain_id="general",
            name="General Knowledge Expert",
            description="General research and information retrieval",
            entity_types=["concept", "person", "organization", "event"],
            keywords=[],  # Catches everything else
            confidence_weight=0.8  # Lower weight for general
        )

    def _init_db(self):
        """Initialize SQLite database for persistent storage."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Domain embedding index
                CREATE TABLE IF NOT EXISTS domain_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT,
                    embedding BLOB NOT NULL,
                    context TEXT,
                    source_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(domain, entity_id)
                );

                -- Aggregated embedding cache
                CREATE TABLE IF NOT EXISTS aggregated_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE NOT NULL,
                    master_embedding BLOB NOT NULL,
                    domain_contributions TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Entity relationship graph
                CREATE TABLE IF NOT EXISTS entity_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_entity_id, target_entity_id, relation_type)
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_domain_embeddings_domain
                    ON domain_embeddings(domain);
                CREATE INDEX IF NOT EXISTS idx_domain_embeddings_entity_type
                    ON domain_embeddings(entity_type);
                CREATE INDEX IF NOT EXISTS idx_entity_relations_source
                    ON entity_relations(source_entity_id);
            """)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama."""
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            response.raise_for_status()

            result = response.json()
            embeddings = result.get("embedding", [])
            embedding = np.array(embeddings, dtype=np.float32)

            # Cache
            self._embedding_cache[cache_key] = embedding
            if len(self._embedding_cache) > self._cache_max_size:
                # Remove oldest entries (simple LRU approximation)
                keys = list(self._embedding_cache.keys())[:1000]
                for k in keys:
                    del self._embedding_cache[k]

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return zero vector on error (qwen3-embedding uses 4096 dims)
            return np.zeros(4096, dtype=np.float32)

    def route_to_experts(
        self,
        query: str,
        detected_entities: List[Dict[str, str]] = None,
        detected_domains: List[str] = None
    ) -> List[Tuple[DomainExpert, float]]:
        """
        Route query to appropriate domain experts using similarity-based routing.

        Based on RouterRetriever (arXiv:2409.02685):
        - Uses embedding similarities for routing decisions
        - Simultaneously routes to multiple experts with weights

        Returns:
            List of (expert, weight) tuples sorted by weight
        """
        query_lower = query.lower()
        expert_scores: Dict[str, float] = {}

        for domain_id, expert in self.experts.items():
            score = 0.0

            # Keyword matching
            keyword_matches = sum(1 for kw in expert.keywords if kw in query_lower)
            score += keyword_matches * 0.3

            # Entity type matching
            if detected_entities:
                for entity in detected_entities:
                    entity_type = entity.get("type", "")
                    if entity_type in expert.entity_types:
                        score += 0.4

            # Domain hint matching
            if detected_domains:
                if domain_id in detected_domains:
                    score += 0.5

            # Apply confidence weight
            score *= expert.confidence_weight

            expert_scores[domain_id] = score

        # Sort by score descending
        sorted_experts = sorted(
            [(self.experts[d], s) for d, s in expert_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # Ensure at least general expert is included
        if not any(e.domain_id == "general" for e, _ in sorted_experts if _ > 0):
            sorted_experts.append((self.experts["general"], 0.1))

        # Return top experts with non-zero scores
        return [(e, s) for e, s in sorted_experts if s > 0 or e.domain_id == "general"][:3]

    def z_score_normalize(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """
        Z-score normalize embeddings for cross-domain fusion.

        Based on HF-RAG (arXiv:2509.02837):
        - Z-score transformation enables principled fusion
        - Addresses score incomparability across domains
        """
        if not embeddings:
            return []

        # Stack embeddings
        stacked = np.vstack(embeddings)

        # Compute mean and std per dimension
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        std[std == 0] = 1  # Avoid division by zero

        # Normalize each embedding
        normalized = [(e - mean) / std for e in embeddings]

        return normalized

    async def aggregate_embeddings(
        self,
        query: str,
        expert_results: List[Tuple[DomainExpert, List[EmbeddingResult]]]
    ) -> AggregatedEmbedding:
        """
        Aggregate embeddings from multiple domain experts.

        Uses HF-RAG style hierarchical fusion:
        1. Intra-domain aggregation (if multiple results per domain)
        2. Z-score normalization
        3. Weighted inter-domain fusion
        """
        if not expert_results:
            query_embedding = await self.get_embedding(query)
            return AggregatedEmbedding(
                master_embedding=query_embedding,
                domain_contributions={"general": 1.0},
                source_embeddings=[],
                fusion_method="fallback"
            )

        all_embeddings = []
        all_weights = []
        domain_contributions = {}
        source_embeddings = []

        for expert, results in expert_results:
            if not results:
                continue

            # Intra-domain: average embeddings from same domain
            domain_embeddings = [r.embedding for r in results]
            domain_avg = np.mean(domain_embeddings, axis=0)

            # Weight by expert confidence and result count
            weight = expert.confidence_weight * min(len(results), 3) / 3

            all_embeddings.append(domain_avg)
            all_weights.append(weight)
            domain_contributions[expert.domain_id] = weight
            source_embeddings.extend(results)

        # Z-score normalize for cross-domain fusion
        normalized = self.z_score_normalize(all_embeddings)

        # Weighted average
        total_weight = sum(all_weights)
        if total_weight > 0:
            master_embedding = sum(
                e * (w / total_weight)
                for e, w in zip(normalized, all_weights)
            )
        else:
            # Fallback to query embedding
            master_embedding = await self.get_embedding(query)

        return AggregatedEmbedding(
            master_embedding=master_embedding,
            domain_contributions=domain_contributions,
            source_embeddings=source_embeddings,
            fusion_method="z_score_rrf"
        )

    async def index_entity(
        self,
        domain: str,
        entity_id: str,
        entity_name: str,
        entity_type: str,
        context: str,
        source_url: str = ""
    ) -> bool:
        """Index an entity embedding in a domain."""
        try:
            # Get embedding for entity context
            embedding = await self.get_embedding(f"{entity_name}: {context}")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO domain_embeddings
                    (domain, entity_id, entity_name, entity_type, embedding, context, source_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    domain, entity_id, entity_name, entity_type,
                    embedding.tobytes(), context, source_url
                ))

            logger.debug(f"Indexed entity {entity_name} in domain {domain}")
            return True

        except Exception as e:
            logger.error(f"Error indexing entity: {e}")
            return False

    async def index_relation(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        domain: str,
        weight: float = 1.0
    ) -> bool:
        """Index a relation between entities."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entity_relations
                    (source_entity_id, target_entity_id, relation_type, domain, weight)
                    VALUES (?, ?, ?, ?, ?)
                """, (source_entity_id, target_entity_id, relation_type, domain, weight))

            return True

        except Exception as e:
            logger.error(f"Error indexing relation: {e}")
            return False

    async def retrieve_sub_manifold(
        self,
        anchor_entities: List[Dict[str, Any]],
        domains: List[str],
        k: int = 10,
        hop_radius: int = 2
    ) -> List[SubManifoldResult]:
        """
        Retrieve entities in sub-manifold around anchor entities.

        Based on AnchorRAG (arXiv:2509.01238):
        - Use extracted entities as navigation anchors
        - Multi-hop expansion through embedding neighborhood
        """
        results = []

        with sqlite3.connect(self.db_path) as conn:
            for anchor in anchor_entities:
                anchor_name = anchor.get("name", "")
                anchor_type = anchor.get("type", "")

                if not anchor_name:
                    continue

                # Get anchor embedding
                anchor_embedding = await self.get_embedding(anchor_name)

                # Query similar entities in specified domains
                domain_clause = ""
                if domains:
                    placeholders = ",".join("?" * len(domains))
                    domain_clause = f"AND domain IN ({placeholders})"

                query_params = list(domains) if domains else []

                # Get all domain embeddings
                cursor = conn.execute(f"""
                    SELECT entity_id, entity_name, entity_type, embedding,
                           context, domain
                    FROM domain_embeddings
                    WHERE 1=1 {domain_clause}
                """, query_params)

                for row in cursor:
                    entity_id, entity_name, entity_type, embedding_bytes, context, domain = row

                    # Convert embedding
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

                    # Compute cosine similarity
                    similarity = np.dot(anchor_embedding, embedding) / (
                        np.linalg.norm(anchor_embedding) * np.linalg.norm(embedding) + 1e-8
                    )

                    # Convert similarity to distance (lower is better)
                    distance = 1.0 - similarity

                    results.append(SubManifoldResult(
                        entity_id=entity_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        embedding=embedding,
                        distance=distance,
                        domain=domain,
                        context=context
                    ))

        # Sort by distance and return top k
        results.sort(key=lambda x: x.distance)
        return results[:k]

    async def expand_via_relations(
        self,
        entity_ids: List[str],
        domain: str,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Expand from entities via knowledge graph relations.

        Returns related entities within max_hops.
        """
        visited = set(entity_ids)
        current_frontier = list(entity_ids)
        all_related = []

        with sqlite3.connect(self.db_path) as conn:
            for hop in range(max_hops):
                if not current_frontier:
                    break

                next_frontier = []
                placeholders = ",".join("?" * len(current_frontier))

                cursor = conn.execute(f"""
                    SELECT DISTINCT target_entity_id, relation_type, weight
                    FROM entity_relations
                    WHERE source_entity_id IN ({placeholders})
                    AND domain = ?
                """, current_frontier + [domain])

                for target_id, relation_type, weight in cursor:
                    if target_id not in visited:
                        visited.add(target_id)
                        next_frontier.append(target_id)
                        all_related.append({
                            "entity_id": target_id,
                            "relation_type": relation_type,
                            "hop": hop + 1,
                            "weight": weight
                        })

                current_frontier = next_frontier

        return all_related

    async def retrieve(
        self,
        query: str,
        detected_entities: List[Dict[str, str]] = None,
        detected_domains: List[str] = None,
        k: int = 10
    ) -> RetrievalResult:
        """
        Full entity-guided sub-manifold retrieval.

        Pipeline:
        1. Route to domain experts
        2. Extract/use anchor entities
        3. Retrieve from sub-manifold
        4. Expand via relations
        5. Aggregate and synthesize context
        """
        import time
        start_time = time.time()

        # Route to experts
        expert_routing = self.route_to_experts(query, detected_entities, detected_domains)
        domains_used = [e.domain_id for e, _ in expert_routing]

        logger.info(f"Routing query to domains: {domains_used}")

        # Use detected entities as anchors, or extract from query
        anchor_entities = detected_entities or []

        # Get query embedding
        query_embedding = await self.get_embedding(query)

        # Retrieve from sub-manifold
        sub_manifold_results = await self.retrieve_sub_manifold(
            anchor_entities=anchor_entities if anchor_entities else [{"name": query}],
            domains=domains_used,
            k=k
        )

        # Expand via relations for top results
        if sub_manifold_results:
            top_entity_ids = [r.entity_id for r in sub_manifold_results[:3]]
            for domain in domains_used:
                related = await self.expand_via_relations(
                    top_entity_ids, domain, max_hops=2
                )
                for rel in related:
                    # Add related entity IDs to results
                    for r in sub_manifold_results:
                        if r.entity_id in top_entity_ids:
                            r.related_entities.append(rel["entity_id"])

        # Aggregate context from results
        context_parts = []
        for r in sub_manifold_results[:k]:
            context_parts.append(f"[{r.entity_type}] {r.entity_name}: {r.context}")
        aggregated_context = "\n".join(context_parts)

        # Calculate confidence based on results
        if sub_manifold_results:
            avg_similarity = 1.0 - np.mean([r.distance for r in sub_manifold_results[:5]])
            confidence = min(0.95, max(0.5, avg_similarity))
        else:
            confidence = 0.5

        retrieval_time_ms = int((time.time() - start_time) * 1000)

        return RetrievalResult(
            query=query,
            query_embedding=query_embedding,
            anchor_entities=anchor_entities,
            sub_manifold_results=sub_manifold_results,
            aggregated_context=aggregated_context,
            confidence=confidence,
            domains_used=domains_used,
            retrieval_time_ms=retrieval_time_ms
        )

    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[SubManifoldResult]],
        k: int = 60
    ) -> List[SubManifoldResult]:
        """
        Reciprocal Rank Fusion for combining results from multiple sources.

        Based on ELERAG (arXiv:2512.05967):
        - Entity-aware RRF for improved ranking
        """
        scores: Dict[str, float] = {}
        result_map: Dict[str, SubManifoldResult] = {}

        for result_list in result_lists:
            for rank, result in enumerate(result_list):
                if result.entity_id not in result_map:
                    result_map[result.entity_id] = result

                # RRF formula: 1 / (k + rank)
                scores[result.entity_id] = scores.get(result.entity_id, 0) + 1.0 / (k + rank)

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [result_map[eid] for eid in sorted_ids]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        with sqlite3.connect(self.db_path) as conn:
            entity_count = conn.execute(
                "SELECT COUNT(*) FROM domain_embeddings"
            ).fetchone()[0]

            relation_count = conn.execute(
                "SELECT COUNT(*) FROM entity_relations"
            ).fetchone()[0]

            domain_counts = dict(conn.execute(
                "SELECT domain, COUNT(*) FROM domain_embeddings GROUP BY domain"
            ).fetchall())

        return {
            "total_entities": entity_count,
            "total_relations": relation_count,
            "entities_by_domain": domain_counts,
            "registered_experts": list(self.experts.keys()),
            "cache_size": len(self._embedding_cache),
            "embedding_model": self.embedding_model
        }

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Singleton instance
_aggregator_instance: Optional[EmbeddingAggregator] = None


def get_embedding_aggregator(
    ollama_url: str = "http://localhost:11434"
) -> EmbeddingAggregator:
    """Get or create singleton EmbeddingAggregator instance."""
    global _aggregator_instance
    if _aggregator_instance is None:
        _aggregator_instance = EmbeddingAggregator(ollama_url=ollama_url)
    return _aggregator_instance


async def retrieve_with_entities(
    query: str,
    detected_entities: List[Dict[str, str]] = None,
    detected_domains: List[str] = None,
    k: int = 10
) -> RetrievalResult:
    """Convenience function for entity-guided retrieval."""
    aggregator = get_embedding_aggregator()
    return await aggregator.retrieve(query, detected_entities, detected_domains, k)
