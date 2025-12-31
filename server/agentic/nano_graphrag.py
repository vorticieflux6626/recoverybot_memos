"""
Nano-GraphRAG Integration Module.

Part of G.3: Graph Enhancement for memOS agentic search.

Integrates nano-graphrag/LightRAG dual-level retrieval and HippoRAG's Personalized
PageRank with the existing memOS infrastructure:
- SemanticMemoryNetwork (A-MEM): Zettelkasten-style entity connections
- DomainCorpus: Domain-specific entity types and relationships
- EntityTracker: GSW-style entity extraction

Key Features:
- Dual-level retrieval: Entity-level (local) + Community-level (global)
- Personalized PageRank for multi-hop reasoning
- Leiden community detection with automatic summarization
- Seamless integration with existing memOS components

Research Basis:
- nano-graphrag (MIT): ~1100 lines, easy-to-hack GraphRAG
- LightRAG (EMNLP 2025): Dual-level retrieval with graph structures
- HippoRAG (NeurIPS 2024): PPR for 10-30x cheaper multi-hop retrieval
- Leiden Algorithm: Fast hierarchical community detection

Usage:
    from agentic.nano_graphrag import NanoGraphRAG, get_nano_graphrag

    graphrag = get_nano_graphrag()

    # Index documents
    await graphrag.insert("FANUC SRVO-063 indicates encoder issues...")

    # Query with different modes
    local_result = await graphrag.query("SRVO-063 error", mode="local")
    global_result = await graphrag.query("servo troubleshooting", mode="global")
    hybrid_result = await graphrag.query("encoder replacement", mode="hybrid")
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import json
import re
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import igraph
    from leidenalg import find_partition, ModularityVertexPartition
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    igraph = None

import httpx

logger = logging.getLogger("agentic.nano_graphrag")


class QueryMode(str, Enum):
    """Query modes for graph-based retrieval."""
    LOCAL = "local"      # Entity neighborhood traversal
    GLOBAL = "global"    # Community-level aggregation
    HYBRID = "hybrid"    # Combined local + global
    NAIVE = "naive"      # Bypass graph, standard RAG


class EntityType(str, Enum):
    """Entity types extracted from text."""
    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    LOCATION = "location"
    ERROR_CODE = "error_code"
    COMPONENT = "component"
    PROCEDURE = "procedure"
    PARAMETER = "parameter"
    EVENT = "event"
    OTHER = "other"


@dataclass
class Entity:
    """An entity extracted from text."""
    id: str
    name: str
    entity_type: EntityType
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.id == other.id


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash((self.source_id, self.target_id, self.relation_type))


@dataclass
class Community:
    """A community of related entities."""
    id: str
    entity_ids: List[str]
    level: int = 0  # Hierarchy level (0 = leaf, higher = more abstract)
    summary: str = ""
    title: str = ""
    rank: float = 0.0  # Importance score
    embedding: Optional[List[float]] = None


@dataclass
class GraphRAGConfig:
    """Configuration for NanoGraphRAG."""
    # LLM settings
    ollama_url: str = "http://localhost:11434"
    extraction_model: str = "qwen3:8b"
    summary_model: str = "gemma3:4b"
    embedding_model: str = "mxbai-embed-large"

    # Graph settings
    max_entities_per_chunk: int = 20
    max_relationships_per_chunk: int = 30
    entity_similarity_threshold: float = 0.85

    # Community detection
    enable_communities: bool = True
    min_community_size: int = 3
    max_community_levels: int = 3

    # PPR settings
    ppr_damping: float = 0.85  # Alpha for Personalized PageRank
    ppr_max_iterations: int = 100
    ppr_tolerance: float = 1e-6

    # Retrieval settings
    local_top_k: int = 10
    global_top_k: int = 5
    community_weight: float = 0.3  # Weight for community vs entity scores

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class QueryResult:
    """Result from a graph-based query."""
    answer: str
    mode: QueryMode
    entities: List[Entity]
    communities: List[Community]
    context_chunks: List[str]
    ppr_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class NanoGraphRAG:
    """
    Lightweight GraphRAG implementation with dual-level retrieval.

    Combines nano-graphrag's simplicity with LightRAG's dual-level approach
    and HippoRAG's Personalized PageRank for multi-hop reasoning.

    Architecture:
    - Entity extraction via LLM
    - NetworkX graph storage (with optional Neo4j)
    - Leiden community detection
    - PPR-based retrieval with community aggregation
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx required. Install with: pip install networkx")

        self.config = config or GraphRAGConfig()

        # Core graph structure
        self.graph: nx.Graph = nx.Graph()

        # Entity and chunk storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.chunks: Dict[str, str] = {}  # chunk_id -> content
        self.chunk_entities: Dict[str, Set[str]] = defaultdict(set)  # chunk_id -> entity_ids

        # Community structure
        self.communities: Dict[str, Community] = {}
        self.entity_to_community: Dict[str, str] = {}  # entity_id -> community_id

        # Caches
        self._embedding_cache: Dict[str, List[float]] = {}
        self._query_cache: Dict[str, QueryResult] = {}

        # Statistics
        self.stats = {
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "communities_detected": 0,
            "queries_processed": 0,
            "ppr_runs": 0,
            "cache_hits": 0
        }

        logger.info(
            f"NanoGraphRAG initialized: "
            f"communities={'enabled' if self.config.enable_communities else 'disabled'}, "
            f"ppr_damping={self.config.ppr_damping}"
        )

    # =========================================================================
    # ENTITY EXTRACTION
    # =========================================================================

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        cache_key = hashlib.sha256(text[:500].encode()).hexdigest()[:16]

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.config.ollama_url}/api/embeddings",
                    json={
                        "model": self.config.embedding_model,
                        "prompt": text[:1000]
                    }
                )
                response.raise_for_status()
                embedding = response.json().get("embedding", [])

                if self.config.enable_cache:
                    self._embedding_cache[cache_key] = embedding

                return embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return []

    async def _extract_entities_llm(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships using LLM."""
        prompt = f"""Extract entities and relationships from the following text.

TEXT:
{text[:3000]}

OUTPUT FORMAT (JSON):
{{
  "entities": [
    {{"name": "entity name", "type": "one of: person, organization, product, technology, concept, location, error_code, component, procedure, parameter, event, other", "description": "brief description"}}
  ],
  "relationships": [
    {{"source": "source entity name", "target": "target entity name", "relation": "relationship type", "description": "brief description"}}
  ]
}}

Focus on:
- Technical terms and components
- Error codes and their meanings
- Procedures and their relationships
- Cause-effect relationships

Return ONLY valid JSON, no other text."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.extraction_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_ctx": 8192
                        }
                    }
                )
                response.raise_for_status()
                result_text = response.json().get("response", "")

                # Parse JSON from response
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if not json_match:
                    logger.warning("No JSON found in extraction response")
                    return [], []

                data = json.loads(json_match.group())

                entities = []
                for e in data.get("entities", [])[:self.config.max_entities_per_chunk]:
                    entity_type = e.get("type", "other").lower()
                    try:
                        etype = EntityType(entity_type)
                    except ValueError:
                        etype = EntityType.OTHER

                    entity_id = self._generate_entity_id(e["name"])
                    entities.append(Entity(
                        id=entity_id,
                        name=e["name"],
                        entity_type=etype,
                        description=e.get("description", "")
                    ))

                relationships = []
                for r in data.get("relationships", [])[:self.config.max_relationships_per_chunk]:
                    source_id = self._generate_entity_id(r["source"])
                    target_id = self._generate_entity_id(r["target"])
                    relationships.append(Relationship(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=r.get("relation", "related_to"),
                        description=r.get("description", "")
                    ))

                return entities, relationships

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [], []

    def _generate_entity_id(self, name: str) -> str:
        """Generate consistent entity ID from name."""
        normalized = name.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    def _generate_chunk_id(self, content: str) -> str:
        """Generate chunk ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================

    async def insert(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Insert text into the graph-based index.

        Args:
            text: Text to index
            metadata: Optional metadata for the text

        Returns:
            Statistics about the insertion
        """
        start_time = time.time()

        # Generate chunk ID
        chunk_id = self._generate_chunk_id(text)

        # Skip if already indexed
        if chunk_id in self.chunks:
            return {"status": "skipped", "reason": "already_indexed"}

        self.chunks[chunk_id] = text

        # Extract entities and relationships
        entities, relationships = await self._extract_entities_llm(text)

        # Add entities to graph
        new_entities = 0
        for entity in entities:
            if entity.id not in self.entities:
                # Get embedding for entity
                entity.embedding = await self._get_embedding(
                    f"{entity.name}: {entity.description}"
                )
                entity.source_chunks.append(chunk_id)
                self.entities[entity.id] = entity
                self.graph.add_node(
                    entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    description=entity.description
                )
                new_entities += 1
            else:
                # Update existing entity
                self.entities[entity.id].source_chunks.append(chunk_id)

            self.chunk_entities[chunk_id].add(entity.id)

        # Add relationships to graph
        new_relationships = 0
        for rel in relationships:
            if rel.source_id in self.entities and rel.target_id in self.entities:
                rel_key = f"{rel.source_id}:{rel.target_id}:{rel.relation_type}"
                if rel_key not in self.relationships:
                    rel.source_chunks.append(chunk_id)
                    self.relationships[rel_key] = rel
                    self.graph.add_edge(
                        rel.source_id,
                        rel.target_id,
                        relation=rel.relation_type,
                        weight=rel.weight
                    )
                    new_relationships += 1

        # Update statistics
        self.stats["entities_extracted"] += new_entities
        self.stats["relationships_extracted"] += new_relationships

        # Rebuild communities if enabled
        if self.config.enable_communities and new_entities > 0:
            await self._detect_communities()

        duration_ms = (time.time() - start_time) * 1000

        return {
            "status": "indexed",
            "chunk_id": chunk_id,
            "entities_added": new_entities,
            "relationships_added": new_relationships,
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "duration_ms": duration_ms
        }

    async def insert_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Insert multiple texts into the index."""
        results = []
        for text in texts:
            result = await self.insert(text)
            results.append(result)

        return {
            "total": len(texts),
            "indexed": sum(1 for r in results if r["status"] == "indexed"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "results": results
        }

    # =========================================================================
    # COMMUNITY DETECTION (Leiden Algorithm)
    # =========================================================================

    async def _detect_communities(self) -> int:
        """
        Detect communities using Leiden algorithm.

        Falls back to connected components if leidenalg not available.
        """
        if len(self.graph.nodes) < self.config.min_community_size:
            return 0

        if LEIDEN_AVAILABLE:
            return await self._leiden_communities()
        else:
            return self._connected_component_communities()

    async def _leiden_communities(self) -> int:
        """Detect communities using Leiden algorithm with igraph."""
        # Convert NetworkX to igraph
        ig_graph = igraph.Graph()
        node_mapping = {n: i for i, n in enumerate(self.graph.nodes())}
        reverse_mapping = {i: n for n, i in node_mapping.items()}

        ig_graph.add_vertices(len(node_mapping))
        edges = [(node_mapping[u], node_mapping[v]) for u, v in self.graph.edges()]
        ig_graph.add_edges(edges)

        # Run Leiden algorithm
        partition = find_partition(ig_graph, ModularityVertexPartition)

        # Build communities
        self.communities.clear()
        self.entity_to_community.clear()

        for comm_idx, members in enumerate(partition):
            if len(members) >= self.config.min_community_size:
                comm_id = f"comm_{comm_idx}"
                entity_ids = [reverse_mapping[m] for m in members]

                # Generate community summary
                summary = await self._summarize_community(entity_ids)

                self.communities[comm_id] = Community(
                    id=comm_id,
                    entity_ids=entity_ids,
                    level=0,
                    summary=summary,
                    title=f"Community {comm_idx}",
                    rank=len(members) / len(self.graph.nodes)
                )

                for eid in entity_ids:
                    self.entity_to_community[eid] = comm_id

        self.stats["communities_detected"] = len(self.communities)
        logger.info(f"Detected {len(self.communities)} communities via Leiden")

        return len(self.communities)

    def _connected_component_communities(self) -> int:
        """Fallback: Use connected components as communities."""
        self.communities.clear()
        self.entity_to_community.clear()

        components = list(nx.connected_components(self.graph))

        for comp_idx, members in enumerate(components):
            if len(members) >= self.config.min_community_size:
                comm_id = f"comp_{comp_idx}"
                entity_ids = list(members)

                self.communities[comm_id] = Community(
                    id=comm_id,
                    entity_ids=entity_ids,
                    level=0,
                    title=f"Component {comp_idx}",
                    rank=len(members) / len(self.graph.nodes)
                )

                for eid in entity_ids:
                    self.entity_to_community[eid] = comm_id

        self.stats["communities_detected"] = len(self.communities)
        logger.info(f"Detected {len(self.communities)} communities via components")

        return len(self.communities)

    async def _summarize_community(self, entity_ids: List[str]) -> str:
        """Generate summary for a community of entities."""
        entities = [self.entities[eid] for eid in entity_ids if eid in self.entities]

        if not entities:
            return "Empty community"

        # Build entity descriptions
        entity_descriptions = []
        for e in entities[:10]:  # Limit to 10 for context
            entity_descriptions.append(f"- {e.name} ({e.entity_type.value}): {e.description}")

        prompt = f"""Summarize this group of related entities in 2-3 sentences.

ENTITIES:
{chr(10).join(entity_descriptions)}

Write a concise summary of what these entities represent collectively.
Focus on their common theme or relationship."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.summary_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.5}
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")[:500]
        except Exception as e:
            logger.warning(f"Community summarization failed: {e}")
            return f"Group of {len(entities)} related entities"

    # =========================================================================
    # PERSONALIZED PAGERANK (HippoRAG-style)
    # =========================================================================

    def _personalized_pagerank(
        self,
        seed_entities: List[str],
        alpha: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Run Personalized PageRank from seed entities.

        Based on HippoRAG (NeurIPS 2024) for multi-hop reasoning.

        Args:
            seed_entities: Entity IDs to use as seeds (personalization vector)
            alpha: Damping factor (default from config)

        Returns:
            Dictionary mapping entity_id -> PPR score
        """
        if not seed_entities:
            return {}

        alpha = alpha or self.config.ppr_damping

        # Build personalization vector
        personalization = {}
        valid_seeds = [e for e in seed_entities if e in self.graph]

        if not valid_seeds:
            return {}

        for entity_id in valid_seeds:
            personalization[entity_id] = 1.0 / len(valid_seeds)

        try:
            # Run NetworkX PPR
            ppr_scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=self.config.ppr_max_iterations,
                tol=self.config.ppr_tolerance
            )

            self.stats["ppr_runs"] += 1
            return ppr_scores

        except Exception as e:
            logger.error(f"PPR computation failed: {e}")
            return {}

    async def _find_seed_entities(self, query: str) -> List[str]:
        """Find seed entities for PPR based on query."""
        # Get query embedding
        query_embedding = await self._get_embedding(query)

        if not query_embedding:
            return []

        # Find similar entities by embedding
        similarities = []
        for entity_id, entity in self.entities.items():
            if entity.embedding:
                sim = self._cosine_similarity(query_embedding, entity.embedding)
                similarities.append((entity_id, sim))

        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)

        seed_entities = [
            eid for eid, sim in similarities[:self.config.local_top_k]
            if sim >= 0.5  # Minimum similarity threshold
        ]

        return seed_entities

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b:
            return 0.0

        a_np = np.array(a)
        b_np = np.array(b)

        dot = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    # =========================================================================
    # QUERY / RETRIEVAL
    # =========================================================================

    async def query(
        self,
        question: str,
        mode: Union[str, QueryMode] = QueryMode.HYBRID,
        top_k: Optional[int] = None
    ) -> QueryResult:
        """
        Query the graph-based index.

        Args:
            question: Query string
            mode: Query mode (local, global, hybrid, naive)
            top_k: Number of results to return

        Returns:
            QueryResult with answer, entities, communities, and context
        """
        start_time = time.time()

        if isinstance(mode, str):
            mode = QueryMode(mode.lower())

        top_k = top_k or self.config.local_top_k

        # Check cache
        cache_key = f"{question}:{mode.value}:{top_k}"
        if self.config.enable_cache and cache_key in self._query_cache:
            self.stats["cache_hits"] += 1
            return self._query_cache[cache_key]

        self.stats["queries_processed"] += 1

        # Find seed entities for PPR
        seed_entities = await self._find_seed_entities(question)

        if mode == QueryMode.NAIVE or not seed_entities:
            # Fallback to naive RAG
            result = await self._naive_query(question, top_k)
        elif mode == QueryMode.LOCAL:
            result = await self._local_query(question, seed_entities, top_k)
        elif mode == QueryMode.GLOBAL:
            result = await self._global_query(question, seed_entities, top_k)
        else:  # HYBRID
            result = await self._hybrid_query(question, seed_entities, top_k)

        result.mode = mode
        result.latency_ms = (time.time() - start_time) * 1000

        # Cache result
        if self.config.enable_cache:
            self._query_cache[cache_key] = result

        return result

    async def _local_query(
        self,
        question: str,
        seed_entities: List[str],
        top_k: int
    ) -> QueryResult:
        """Local query: Entity neighborhood traversal with PPR."""
        # Run PPR from seeds
        ppr_scores = self._personalized_pagerank(seed_entities)

        if not ppr_scores:
            return QueryResult(
                answer="No relevant entities found",
                mode=QueryMode.LOCAL,
                entities=[],
                communities=[],
                context_chunks=[],
                confidence=0.0
            )

        # Get top entities by PPR score
        sorted_entities = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        top_entity_ids = [eid for eid, _ in sorted_entities[:top_k]]

        # Gather entities and their chunks
        entities = [self.entities[eid] for eid in top_entity_ids if eid in self.entities]

        # Collect context chunks
        context_chunks = set()
        for entity in entities:
            for chunk_id in entity.source_chunks[:3]:
                if chunk_id in self.chunks:
                    context_chunks.add(self.chunks[chunk_id])

        # Generate answer
        answer = await self._synthesize_answer(question, list(context_chunks), entities)

        return QueryResult(
            answer=answer,
            mode=QueryMode.LOCAL,
            entities=entities,
            communities=[],
            context_chunks=list(context_chunks),
            ppr_scores={eid: ppr_scores.get(eid, 0) for eid in top_entity_ids},
            confidence=self._calculate_confidence(ppr_scores, top_entity_ids)
        )

    async def _global_query(
        self,
        question: str,
        seed_entities: List[str],
        top_k: int
    ) -> QueryResult:
        """Global query: Community-level aggregation."""
        if not self.communities:
            # No communities, fall back to local
            return await self._local_query(question, seed_entities, top_k)

        # Find communities of seed entities
        relevant_communities = set()
        for eid in seed_entities:
            if eid in self.entity_to_community:
                relevant_communities.add(self.entity_to_community[eid])

        # If no direct matches, find closest communities by embedding
        if not relevant_communities:
            query_embedding = await self._get_embedding(question)
            community_scores = []

            for comm_id, community in self.communities.items():
                if community.embedding:
                    sim = self._cosine_similarity(query_embedding, community.embedding)
                    community_scores.append((comm_id, sim))

            community_scores.sort(key=lambda x: x[1], reverse=True)
            relevant_communities = {cid for cid, _ in community_scores[:top_k]}

        # Gather community data
        communities = [
            self.communities[cid]
            for cid in relevant_communities
            if cid in self.communities
        ]

        # Collect entities from communities
        entity_ids = set()
        for comm in communities:
            entity_ids.update(comm.entity_ids)

        entities = [self.entities[eid] for eid in entity_ids if eid in self.entities]

        # Context from community summaries
        context_chunks = [comm.summary for comm in communities if comm.summary]

        # Generate answer using community summaries
        answer = await self._synthesize_answer(question, context_chunks, entities[:top_k])

        return QueryResult(
            answer=answer,
            mode=QueryMode.GLOBAL,
            entities=entities[:top_k],
            communities=communities,
            context_chunks=context_chunks,
            confidence=sum(c.rank for c in communities) / len(communities) if communities else 0.0
        )

    async def _hybrid_query(
        self,
        question: str,
        seed_entities: List[str],
        top_k: int
    ) -> QueryResult:
        """Hybrid query: Combines local + global approaches."""
        # Run both queries
        local_result = await self._local_query(question, seed_entities, top_k // 2)
        global_result = await self._global_query(question, seed_entities, top_k // 2)

        # Combine entities (deduplicated)
        seen_ids = set()
        combined_entities = []

        for entity in local_result.entities + global_result.entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                combined_entities.append(entity)

        # Combine context
        combined_chunks = list(set(local_result.context_chunks + global_result.context_chunks))

        # Generate unified answer
        answer = await self._synthesize_answer(
            question,
            combined_chunks[:10],
            combined_entities[:top_k]
        )

        # Blend confidence scores
        local_weight = 1 - self.config.community_weight
        global_weight = self.config.community_weight
        confidence = (
            local_weight * local_result.confidence +
            global_weight * global_result.confidence
        )

        return QueryResult(
            answer=answer,
            mode=QueryMode.HYBRID,
            entities=combined_entities[:top_k],
            communities=global_result.communities,
            context_chunks=combined_chunks,
            ppr_scores=local_result.ppr_scores,
            confidence=confidence
        )

    async def _naive_query(self, question: str, top_k: int) -> QueryResult:
        """Naive query: Standard embedding-based RAG without graph."""
        query_embedding = await self._get_embedding(question)

        if not query_embedding:
            return QueryResult(
                answer="Unable to process query",
                mode=QueryMode.NAIVE,
                entities=[],
                communities=[],
                context_chunks=[],
                confidence=0.0
            )

        # Find similar entities
        similarities = []
        for entity_id, entity in self.entities.items():
            if entity.embedding:
                sim = self._cosine_similarity(query_embedding, entity.embedding)
                similarities.append((entity_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_entity_ids = [eid for eid, _ in similarities[:top_k]]

        entities = [self.entities[eid] for eid in top_entity_ids if eid in self.entities]

        # Collect chunks
        context_chunks = set()
        for entity in entities:
            for chunk_id in entity.source_chunks[:2]:
                if chunk_id in self.chunks:
                    context_chunks.add(self.chunks[chunk_id])

        answer = await self._synthesize_answer(question, list(context_chunks), entities)

        return QueryResult(
            answer=answer,
            mode=QueryMode.NAIVE,
            entities=entities,
            communities=[],
            context_chunks=list(context_chunks),
            confidence=similarities[0][1] if similarities else 0.0
        )

    async def _synthesize_answer(
        self,
        question: str,
        context_chunks: List[str],
        entities: List[Entity]
    ) -> str:
        """Synthesize answer from context and entities."""
        if not context_chunks and not entities:
            return "No relevant information found."

        # Build context
        context_parts = []

        if entities:
            entity_info = []
            for e in entities[:10]:
                entity_info.append(f"- {e.name} ({e.entity_type.value}): {e.description}")
            context_parts.append("KEY ENTITIES:\n" + "\n".join(entity_info))

        if context_chunks:
            context_parts.append("CONTEXT:\n" + "\n---\n".join(context_chunks[:5]))

        context = "\n\n".join(context_parts)

        prompt = f"""Answer the following question based on the provided context.

QUESTION: {question}

{context}

Provide a comprehensive answer. If the context doesn't contain enough information, say so."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.extraction_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.5, "num_ctx": 8192}
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Error generating answer: {e}"

    def _calculate_confidence(
        self,
        ppr_scores: Dict[str, float],
        top_entities: List[str]
    ) -> float:
        """Calculate confidence score based on PPR scores."""
        if not ppr_scores or not top_entities:
            return 0.0

        top_scores = [ppr_scores.get(eid, 0) for eid in top_entities]

        # Confidence based on score concentration
        max_score = max(top_scores) if top_scores else 0
        sum_scores = sum(top_scores)

        if sum_scores == 0:
            return 0.0

        # Higher confidence if scores are concentrated
        concentration = max_score / sum_scores

        return min(1.0, concentration * len(top_entities) * max_score * 10)

    # =========================================================================
    # GRAPH ANALYTICS
    # =========================================================================

    def get_entity_neighbors(self, entity_id: str, depth: int = 1) -> List[Entity]:
        """Get neighboring entities up to specified depth."""
        if entity_id not in self.graph:
            return []

        neighbors = set()
        current_level = {entity_id}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in neighbors:
                        next_level.add(neighbor)
                        neighbors.add(neighbor)
            current_level = next_level

        return [self.entities[eid] for eid in neighbors if eid in self.entities]

    def get_path(self, source_id: str, target_id: str) -> List[Entity]:
        """Get shortest path between two entities."""
        if source_id not in self.graph or target_id not in self.graph:
            return []

        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            return [self.entities[eid] for eid in path if eid in self.entities]
        except nx.NetworkXNoPath:
            return []

    def get_important_entities(self, top_k: int = 10) -> List[Tuple[Entity, float]]:
        """Get most important entities by PageRank."""
        if not self.graph.nodes:
            return []

        pagerank = nx.pagerank(self.graph)
        sorted_entities = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

        return [
            (self.entities[eid], score)
            for eid, score in sorted_entities[:top_k]
            if eid in self.entities
        ]

    # =========================================================================
    # STATISTICS AND EXPORT
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            **self.stats,
            "graph": {
                "nodes": len(self.graph.nodes),
                "edges": len(self.graph.edges),
                "density": nx.density(self.graph) if self.graph.nodes else 0,
                "connected_components": nx.number_connected_components(self.graph) if self.graph.nodes else 0
            },
            "storage": {
                "entities": len(self.entities),
                "relationships": len(self.relationships),
                "chunks": len(self.chunks),
                "communities": len(self.communities)
            },
            "cache": {
                "embeddings": len(self._embedding_cache),
                "queries": len(self._query_cache)
            },
            "config": {
                "ppr_damping": self.config.ppr_damping,
                "communities_enabled": self.config.enable_communities,
                "local_top_k": self.config.local_top_k
            }
        }

    def export_graph(self) -> Dict[str, Any]:
        """Export graph structure for visualization."""
        return {
            "nodes": [
                {
                    "id": eid,
                    "name": e.name,
                    "type": e.entity_type.value,
                    "description": e.description
                }
                for eid, e in self.entities.items()
            ],
            "edges": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "relation": r.relation_type,
                    "weight": r.weight
                }
                for r in self.relationships.values()
            ],
            "communities": [
                {
                    "id": c.id,
                    "entities": c.entity_ids,
                    "summary": c.summary,
                    "rank": c.rank
                }
                for c in self.communities.values()
            ]
        }

    def clear(self):
        """Clear all stored data."""
        self.graph.clear()
        self.entities.clear()
        self.relationships.clear()
        self.chunks.clear()
        self.chunk_entities.clear()
        self.communities.clear()
        self.entity_to_community.clear()
        self._embedding_cache.clear()
        self._query_cache.clear()

        for key in self.stats:
            self.stats[key] = 0


# =============================================================================
# INTEGRATION WITH EXISTING MEMOS COMPONENTS
# =============================================================================

class GraphRAGIntegration:
    """
    Integration layer connecting NanoGraphRAG with existing memOS components.

    Bridges:
    - SemanticMemoryNetwork: Uses A-MEM connections for entity relationships
    - DomainCorpus: Uses domain-specific entity extraction patterns
    - HSEAController: Connects to three-stratum retrieval
    """

    def __init__(
        self,
        graphrag: Optional[NanoGraphRAG] = None,
        config: Optional[GraphRAGConfig] = None
    ):
        self.graphrag = graphrag or NanoGraphRAG(config)
        self.config = config or GraphRAGConfig()

        # Lazy imports for optional memOS components
        self._semantic_memory = None
        self._domain_corpus = None
        self._hsea_controller = None

    def _get_semantic_memory(self):
        """Lazy-load SemanticMemoryNetwork."""
        if self._semantic_memory is None:
            try:
                from .semantic_memory import SemanticMemoryNetwork, get_semantic_memory_network
                self._semantic_memory = get_semantic_memory_network()
            except ImportError:
                logger.debug("SemanticMemoryNetwork not available")
        return self._semantic_memory

    def _get_domain_corpus(self):
        """Lazy-load DomainCorpusManager."""
        if self._domain_corpus is None:
            try:
                from .domain_corpus import get_domain_corpus_manager
                self._domain_corpus = get_domain_corpus_manager()
            except ImportError:
                logger.debug("DomainCorpusManager not available")
        return self._domain_corpus

    async def sync_from_semantic_memory(self):
        """
        Sync entities and connections from SemanticMemoryNetwork.

        A-MEM memories with connections become graph relationships.
        """
        memory = self._get_semantic_memory()
        if not memory:
            return {"synced": 0}

        synced = 0
        for memory_id, mem in memory.memories.items():
            # Add as entity
            entity = Entity(
                id=memory_id,
                name=mem.content[:50],
                entity_type=EntityType(mem.memory_type.value) if mem.memory_type.value in [e.value for e in EntityType] else EntityType.OTHER,
                description=mem.content,
                embedding=mem.embedding
            )

            if entity.id not in self.graphrag.entities:
                self.graphrag.entities[entity.id] = entity
                self.graphrag.graph.add_node(
                    entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type.value
                )

            # Add connections as relationships
            for conn in mem.connections:
                rel = Relationship(
                    source_id=memory_id,
                    target_id=conn.target_id,
                    relation_type=conn.connection_type.value,
                    weight=conn.strength
                )
                rel_key = f"{rel.source_id}:{rel.target_id}:{rel.relation_type}"

                if rel_key not in self.graphrag.relationships:
                    self.graphrag.relationships[rel_key] = rel
                    if conn.target_id in self.graphrag.graph:
                        self.graphrag.graph.add_edge(
                            memory_id,
                            conn.target_id,
                            relation=conn.connection_type.value,
                            weight=conn.strength
                        )

            synced += 1

        return {"synced": synced, "total_entities": len(self.graphrag.entities)}

    async def sync_from_domain_corpus(self, domain_id: str):
        """
        Sync entities from a domain corpus.

        Domain entities become graph nodes with domain-specific types.
        """
        corpus_manager = self._get_domain_corpus()
        if not corpus_manager:
            return {"synced": 0}

        corpus = corpus_manager.get_corpus(domain_id)
        if not corpus:
            return {"synced": 0, "error": f"Domain {domain_id} not found"}

        synced = 0
        # Implementation would iterate over corpus entities
        # and add them to the graph

        return {"synced": synced, "domain": domain_id}

    async def enhanced_query(
        self,
        question: str,
        mode: str = "hybrid",
        use_semantic_memory: bool = True,
        use_domain_corpus: Optional[str] = None
    ) -> QueryResult:
        """
        Enhanced query that combines GraphRAG with other memOS components.
        """
        # First, do standard GraphRAG query
        result = await self.graphrag.query(question, mode=mode)

        # Optionally enhance with semantic memory context
        if use_semantic_memory:
            memory = self._get_semantic_memory()
            if memory:
                # Search semantic memory for related memories
                query_embedding = await self.graphrag._get_embedding(question)
                if query_embedding:
                    # This would be implemented to search A-MEM
                    pass

        # Optionally enhance with domain corpus
        if use_domain_corpus:
            corpus_manager = self._get_domain_corpus()
            if corpus_manager:
                # This would query the specific domain corpus
                pass

        return result


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_nano_graphrag: Optional[NanoGraphRAG] = None
_graphrag_integration: Optional[GraphRAGIntegration] = None


def get_nano_graphrag(config: Optional[GraphRAGConfig] = None) -> NanoGraphRAG:
    """Get or create singleton NanoGraphRAG instance."""
    global _nano_graphrag
    if _nano_graphrag is None:
        _nano_graphrag = NanoGraphRAG(config)
    return _nano_graphrag


def get_graphrag_integration(config: Optional[GraphRAGConfig] = None) -> GraphRAGIntegration:
    """Get or create singleton GraphRAGIntegration instance."""
    global _graphrag_integration
    if _graphrag_integration is None:
        _graphrag_integration = GraphRAGIntegration(config=config)
    return _graphrag_integration


async def initialize_graphrag() -> NanoGraphRAG:
    """Initialize NanoGraphRAG system."""
    graphrag = get_nano_graphrag()
    logger.info("NanoGraphRAG initialized")
    return graphrag
