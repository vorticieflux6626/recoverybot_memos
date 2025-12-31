"""
HopRAG: Multi-Hop Passage Graphs for Complex Question Answering

Implements passage-level knowledge graphs for multi-hop reasoning retrieval.
Based on "HopRAG: Multi-Hop Reasoning via Knowledge Graph Retrieval" (February 2025)

Key innovations:
- Passage graphs connect semantically related text chunks
- Multi-hop traversal follows reasoning chains
- Hop-aware scoring considers path coherence
- 76% higher answer metric, 65% retrieval F1 improvement

Architecture:
    Query → Seed Passages → Graph Expansion → Path Scoring → Answer Synthesis

Author: Claude Code
Date: December 2025
"""

import asyncio
import hashlib
import heapq
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class EdgeType(str, Enum):
    """Types of edges in the passage graph."""
    SEMANTIC = "semantic"  # High embedding similarity
    LEXICAL = "lexical"  # Shared key terms
    ENTITY = "entity"  # Shared named entities
    CAUSAL = "causal"  # Cause-effect relationship
    TEMPORAL = "temporal"  # Time sequence
    STRUCTURAL = "structural"  # Same document/section
    # P0 Fix: Pseudo-query edges (core HopRAG innovation)
    PSEUDO_QUERY = "pseudo_query"  # LLM-generated question connections


class HopStrategy(str, Enum):
    """Strategies for multi-hop expansion."""
    BREADTH_FIRST = "breadth_first"  # Explore all neighbors at each level
    BEST_FIRST = "best_first"  # Priority queue by score
    BEAM_SEARCH = "beam_search"  # Keep top-k at each level
    PERSONALIZED_PAGERANK = "personalized_pagerank"  # PPR from seed nodes


@dataclass
class HopRAGConfig:
    """Configuration for HopRAG."""
    # Model settings
    embedding_model: str = "mxbai-embed-large"
    llm_model: str = "qwen2.5:7b"  # P0 Fix: LLM for pseudo-query generation
    ollama_url: str = "http://localhost:11434"

    # Graph construction
    similarity_threshold: float = 0.65  # Minimum similarity for edge
    max_edges_per_node: int = 10  # Limit fanout
    edge_types: List[EdgeType] = field(default_factory=lambda: [
        EdgeType.SEMANTIC,
        EdgeType.ENTITY,
        EdgeType.LEXICAL,
        EdgeType.PSEUDO_QUERY,  # P0 Fix: Enable pseudo-query edges
    ])

    # Multi-hop settings
    max_hops: int = 4  # P0 Fix: Increased from 3 to 4 per paper recommendation
    hop_strategy: HopStrategy = HopStrategy.BEAM_SEARCH
    beam_width: int = 5  # For beam search
    min_path_score: float = 0.4  # Minimum score to keep path

    # P0 Fix: Pseudo-query settings
    pseudo_queries_per_passage: int = 3  # Number of pseudo-queries to generate
    pseudo_query_similarity_threshold: float = 0.6  # Lower than semantic threshold
    enable_llm_reasoning: bool = True  # Enable LLM-based neighbor selection

    # Scoring
    hop_decay: float = 0.85  # Score decay per hop (prevents infinite expansion)
    edge_type_weights: Dict[EdgeType, float] = field(default_factory=lambda: {
        EdgeType.SEMANTIC: 1.0,
        EdgeType.ENTITY: 0.9,
        EdgeType.CAUSAL: 0.95,
        EdgeType.LEXICAL: 0.7,
        EdgeType.TEMPORAL: 0.8,
        EdgeType.STRUCTURAL: 0.6,
        EdgeType.PSEUDO_QUERY: 0.95,  # P0 Fix: High weight for pseudo-query edges
    })

    # P0 Fix: Helpfulness metric parameters (H(v,q) = α×SIM + (1-α)×IMP)
    helpfulness_alpha: float = 0.7  # Weight for similarity vs importance

    # Performance
    batch_size: int = 10
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class Passage:
    """A passage node in the graph."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    source_doc: str = ""
    section: str = ""
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    # P0 Fix: Pseudo-query storage (core HopRAG innovation)
    outgoing_queries: List[str] = field(default_factory=list)  # "What questions does this lead to?"
    incoming_queries: List[str] = field(default_factory=list)  # "What questions does this answer?"
    outgoing_embeddings: List[np.ndarray] = field(default_factory=list)
    incoming_embeddings: List[np.ndarray] = field(default_factory=list)
    # P0 Fix: Importance score for Helpfulness metric
    importance_score: float = 0.0  # IMP(v) based on arrival counts
    arrival_count: int = 0  # Number of times reached during traversal
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Passage):
            return self.id == other.id
        return False


@dataclass
class Edge:
    """An edge connecting two passages."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float  # Similarity/relevance score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningPath:
    """A multi-hop reasoning path through the graph."""
    passages: List[str]  # Ordered passage IDs
    edges: List[Edge]
    score: float
    hop_scores: List[float]  # Score at each hop
    reasoning_chain: str = ""  # Natural language summary of the path

    def __len__(self):
        return len(self.passages)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passages": self.passages,
            "num_hops": len(self.edges),
            "score": self.score,
            "hop_scores": self.hop_scores,
            "reasoning": self.reasoning_chain[:200] if self.reasoning_chain else "",
        }


@dataclass
class PassageGraph:
    """The complete passage graph."""
    passages: Dict[str, Passage] = field(default_factory=dict)
    edges: Dict[str, List[Edge]] = field(default_factory=dict)  # source_id -> edges
    reverse_edges: Dict[str, List[Edge]] = field(default_factory=dict)  # target_id -> edges

    def add_passage(self, passage: Passage):
        """Add a passage to the graph."""
        self.passages[passage.id] = passage
        if passage.id not in self.edges:
            self.edges[passage.id] = []
        if passage.id not in self.reverse_edges:
            self.reverse_edges[passage.id] = []

    def add_edge(self, edge: Edge):
        """Add an edge to the graph."""
        if edge.source_id not in self.edges:
            self.edges[edge.source_id] = []
        if edge.target_id not in self.reverse_edges:
            self.reverse_edges[edge.target_id] = []

        self.edges[edge.source_id].append(edge)
        self.reverse_edges[edge.target_id].append(edge)

    def get_neighbors(self, passage_id: str) -> List[Tuple[str, Edge]]:
        """Get all neighbors of a passage."""
        neighbors = []
        for edge in self.edges.get(passage_id, []):
            neighbors.append((edge.target_id, edge))
        return neighbors

    def get_passage(self, passage_id: str) -> Optional[Passage]:
        """Get a passage by ID."""
        return self.passages.get(passage_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "num_passages": len(self.passages),
            "num_edges": sum(len(e) for e in self.edges.values()),
            "avg_edges_per_passage": sum(len(e) for e in self.edges.values()) / max(1, len(self.passages)),
        }


@dataclass
class HopRAGResult:
    """Result from HopRAG retrieval."""
    query: str
    seed_passages: List[Passage]
    reasoning_paths: List[ReasoningPath]
    all_relevant_passages: List[Passage]
    retrieval_time_ms: float
    num_hops_explored: int
    # P1 Fix: Include helpfulness scores for analysis
    helpfulness_scores: Dict[str, float] = field(default_factory=dict)
    arrival_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "num_seeds": len(self.seed_passages),
            "num_paths": len(self.reasoning_paths),
            "num_passages": len(self.all_relevant_passages),
            "retrieval_time_ms": self.retrieval_time_ms,
            "max_hops": max((len(p) for p in self.reasoning_paths), default=0),
            "top_paths": [p.to_dict() for p in self.reasoning_paths[:3]],
            # P1 Fix: Include helpfulness info
            "helpfulness_alpha": 0.7,  # Paper default
            "top_helpfulness": {
                pid: self.helpfulness_scores.get(pid, 0.0)
                for pid in list(self.helpfulness_scores.keys())[:5]
            } if self.helpfulness_scores else {},
        }


class HopRAGBuilder:
    """Builds HopRAG passage graphs."""

    def __init__(self, config: Optional[HopRAGConfig] = None):
        self.config = config or HopRAGConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.ollama_url,
                timeout=60.0,
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _embed_text(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "prompt": text[:8000],
                },
            )
            response.raise_for_status()
            embedding = np.array(response.json()["embedding"])

            if self.config.enable_cache:
                self._embedding_cache[text_hash] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return np.zeros(1024)

    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts."""
        embeddings = await asyncio.gather(
            *[self._embed_text(text) for text in texts]
        )
        return list(embeddings)

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        ))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Simple keyword extraction based on word frequency
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

        # Filter common words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
            'from', 'that', 'this', 'with', 'they', 'will', 'would', 'there',
        }
        words = [w for w in words if w not in stop_words]

        # Return top frequent words
        from collections import Counter
        return [w for w, _ in Counter(words).most_common(10)]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text (simple implementation)."""
        import re
        # Extract capitalized phrases, error codes, etc.
        entities = []

        # Error codes (e.g., SRVO-063, MOTN-023)
        codes = re.findall(r'\b[A-Z]{2,5}-\d{2,4}\b', text)
        entities.extend(codes)

        # Capitalized proper nouns (simple heuristic)
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(caps[:5])

        # Technical terms (all caps)
        tech = re.findall(r'\b[A-Z]{2,10}\b', text)
        entities.extend([t for t in tech if t not in {'THE', 'AND', 'FOR', 'WITH'}][:5])

        return list(set(entities))

    def _find_lexical_overlap(
        self,
        kw1: List[str],
        kw2: List[str],
    ) -> float:
        """Calculate Jaccard similarity of keywords."""
        if not kw1 or not kw2:
            return 0.0
        set1, set2 = set(kw1), set(kw2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _find_entity_overlap(
        self,
        ent1: List[str],
        ent2: List[str],
    ) -> float:
        """Calculate entity overlap score."""
        if not ent1 or not ent2:
            return 0.0
        set1, set2 = set(ent1), set(ent2)
        intersection = len(set1 & set2)
        min_size = min(len(set1), len(set2))
        return intersection / min_size if min_size > 0 else 0.0

    async def _generate_pseudo_queries(self, passage: Passage) -> Tuple[List[str], List[str]]:
        """Generate outgoing and incoming pseudo-queries for a passage.

        P0 Fix: Core HopRAG innovation - LLM-generated edges based on questions.

        Outgoing queries: "What questions would a reader naturally ask after reading this?"
        Incoming queries: "What questions does this passage directly answer?"

        Returns:
            Tuple of (outgoing_queries, incoming_queries)
        """
        try:
            client = await self._get_client()

            # Truncate content for LLM
            content_preview = passage.content[:1500]

            # Generate outgoing queries (what this passage leads to)
            outgoing_prompt = f"""Given this passage, generate exactly {self.config.pseudo_queries_per_passage} follow-up questions that a reader would naturally ask after reading it. These questions should be answerable by OTHER related passages (not this one).

Passage:
{content_preview}

Output ONLY the questions, one per line, without numbering:"""

            outgoing_response = await client.post(
                "/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": outgoing_prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 1024}  # Increased for thinking models
                },
                timeout=60.0,
            )
            outgoing_response.raise_for_status()
            resp_data = outgoing_response.json()
            # Handle thinking models - check response first, then thinking
            outgoing_text = resp_data.get("response", "") or ""
            if not outgoing_text.strip() and "thinking" in resp_data:
                # Extract questions from thinking if response is empty
                thinking_text = resp_data.get("thinking", "")
                # Look for quoted questions or question patterns
                questions = re.findall(r'"([^"]+\?)"', thinking_text)
                if questions:
                    outgoing_text = "\n".join(questions[:self.config.pseudo_queries_per_passage])

            # Generate incoming queries (what questions this passage answers)
            incoming_prompt = f"""Given this passage, generate exactly {self.config.pseudo_queries_per_passage} questions that this passage directly answers. These are questions someone might search for that would lead them to this information.

Passage:
{content_preview}

Output ONLY the questions, one per line, without numbering:"""

            incoming_response = await client.post(
                "/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": incoming_prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 1024}  # Increased for thinking models
                },
                timeout=60.0,
            )
            incoming_response.raise_for_status()
            resp_data = incoming_response.json()
            # Handle thinking models - check response first, then thinking
            incoming_text = resp_data.get("response", "") or ""
            if not incoming_text.strip() and "thinking" in resp_data:
                # Extract questions from thinking if response is empty
                thinking_text = resp_data.get("thinking", "")
                # Look for quoted questions or question patterns
                questions = re.findall(r'"([^"]+\?)"', thinking_text)
                if questions:
                    incoming_text = "\n".join(questions[:self.config.pseudo_queries_per_passage])

            # Parse responses into list of queries
            outgoing_queries = [q.strip() for q in outgoing_text.strip().split('\n')
                               if q.strip() and len(q.strip()) > 10][:self.config.pseudo_queries_per_passage]
            incoming_queries = [q.strip() for q in incoming_text.strip().split('\n')
                               if q.strip() and len(q.strip()) > 10][:self.config.pseudo_queries_per_passage]

            logger.debug(f"Generated {len(outgoing_queries)} outgoing, {len(incoming_queries)} incoming queries for {passage.id}")
            return outgoing_queries, incoming_queries

        except Exception as e:
            logger.warning(f"Pseudo-query generation failed for {passage.id}: {e}")
            return [], []

    async def _build_pseudo_query_edges(
        self,
        passages: List[Passage],
        graph: 'PassageGraph',
    ) -> int:
        """Build edges based on pseudo-query similarity.

        P0 Fix: Connect passages via their pseudo-queries.
        Edge created when passage A's outgoing query matches passage B's incoming query.

        Returns:
            Number of edges created
        """
        edges_created = 0

        # First, generate and embed pseudo-queries for all passages
        logger.info(f"Generating pseudo-queries for {len(passages)} passages...")
        for passage in passages:
            if not passage.outgoing_queries:
                outgoing, incoming = await self._generate_pseudo_queries(passage)
                passage.outgoing_queries = outgoing
                passage.incoming_queries = incoming

                # Embed the queries
                if outgoing:
                    passage.outgoing_embeddings = await self._embed_batch(outgoing)
                if incoming:
                    passage.incoming_embeddings = await self._embed_batch(incoming)

        # Now build edges based on query similarity
        logger.info("Building pseudo-query edges...")
        for p1 in passages:
            if not p1.outgoing_queries or not p1.outgoing_embeddings:
                continue

            for p2 in passages:
                if p1.id == p2.id:
                    continue
                if not p2.incoming_queries or not p2.incoming_embeddings:
                    continue

                # Check similarity between p1's outgoing and p2's incoming queries
                best_similarity = 0.0
                best_query_pair = ("", "")

                for out_idx, out_emb in enumerate(p1.outgoing_embeddings):
                    for in_idx, in_emb in enumerate(p2.incoming_embeddings):
                        sim = self._compute_similarity(out_emb, in_emb)
                        if sim > best_similarity:
                            best_similarity = sim
                            best_query_pair = (
                                p1.outgoing_queries[out_idx] if out_idx < len(p1.outgoing_queries) else "",
                                p2.incoming_queries[in_idx] if in_idx < len(p2.incoming_queries) else ""
                            )

                # Create edge if similarity exceeds threshold
                if best_similarity >= self.config.pseudo_query_similarity_threshold:
                    edge = Edge(
                        source_id=p1.id,
                        target_id=p2.id,
                        edge_type=EdgeType.PSEUDO_QUERY,
                        weight=best_similarity * self.config.edge_type_weights.get(EdgeType.PSEUDO_QUERY, 0.95),
                        metadata={
                            "outgoing_query": best_query_pair[0],
                            "incoming_query": best_query_pair[1],
                            "raw_similarity": best_similarity,
                        }
                    )
                    graph.add_edge(edge)
                    edges_created += 1

        logger.info(f"Created {edges_created} pseudo-query edges")
        return edges_created

    async def build_graph(
        self,
        passages: List[Tuple[str, str]],  # (passage_id, content)
        source_doc: str = "",
    ) -> PassageGraph:
        """Build a passage graph from documents."""
        graph = PassageGraph()

        if not passages:
            return graph

        # Create passage nodes
        passage_objects = []
        for pid, content in passages:
            passage = Passage(
                id=pid,
                content=content,
                source_doc=source_doc,
                keywords=self._extract_keywords(content),
                entities=self._extract_entities(content),
            )
            passage_objects.append(passage)
            graph.add_passage(passage)

        # Embed all passages
        logger.info(f"Embedding {len(passage_objects)} passages...")
        contents = [p.content for p in passage_objects]
        embeddings = await self._embed_batch(contents)
        for passage, emb in zip(passage_objects, embeddings):
            passage.embedding = emb

        # Build edges between passages
        logger.info("Building edges...")
        for i, p1 in enumerate(passage_objects):
            edges_for_p1 = []

            for j, p2 in enumerate(passage_objects):
                if i == j:
                    continue

                # Check different edge types
                for edge_type in self.config.edge_types:
                    weight = 0.0

                    if edge_type == EdgeType.SEMANTIC:
                        weight = self._compute_similarity(p1.embedding, p2.embedding)

                    elif edge_type == EdgeType.LEXICAL:
                        weight = self._find_lexical_overlap(p1.keywords, p2.keywords)

                    elif edge_type == EdgeType.ENTITY:
                        weight = self._find_entity_overlap(p1.entities, p2.entities)

                    elif edge_type == EdgeType.STRUCTURAL:
                        # Same document = structural connection
                        if p1.source_doc and p1.source_doc == p2.source_doc:
                            weight = 0.5

                    # Apply edge type weight
                    weight *= self.config.edge_type_weights.get(edge_type, 1.0)

                    if weight >= self.config.similarity_threshold:
                        edge = Edge(
                            source_id=p1.id,
                            target_id=p2.id,
                            edge_type=edge_type,
                            weight=weight,
                        )
                        edges_for_p1.append(edge)

            # Sort edges by weight and keep top-k
            edges_for_p1.sort(key=lambda e: e.weight, reverse=True)
            for edge in edges_for_p1[:self.config.max_edges_per_node]:
                graph.add_edge(edge)

        # P0 Fix: Build pseudo-query edges (core HopRAG innovation)
        if EdgeType.PSEUDO_QUERY in self.config.edge_types:
            pseudo_edges = await self._build_pseudo_query_edges(passage_objects, graph)
            logger.info(f"Added {pseudo_edges} pseudo-query edges")

        logger.info(f"Built graph: {graph.to_dict()}")
        return graph


class HopRAGRetriever:
    """Retrieves multi-hop reasoning paths from passage graphs.

    P0 Fix: Implements Retrieve-Reason-Prune pipeline with LLM-based neighbor selection.
    """

    def __init__(
        self,
        graph: PassageGraph,
        config: Optional[HopRAGConfig] = None,
    ):
        self.graph = graph
        self.config = config or HopRAGConfig()
        self._client: Optional[httpx.AsyncClient] = None
        # P0 Fix: Track arrival counts for importance scoring
        self._arrival_counts: Dict[str, int] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.ollama_url,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _calculate_helpfulness(
        self,
        passage: Passage,
        query_emb: np.ndarray,
    ) -> float:
        """Calculate Helpfulness metric H(v,q) = α×SIM + (1-α)×IMP.

        P0 Fix: Paper's dual-signal ranking combining similarity and importance.

        Args:
            passage: Passage to score
            query_emb: Query embedding

        Returns:
            Helpfulness score (0-1)
        """
        # SIM: Similarity to query
        sim = self._compute_similarity(query_emb, passage.embedding)

        # IMP: Importance based on arrival counts (normalized)
        arrival = self._arrival_counts.get(passage.id, 0)
        max_arrivals = max(self._arrival_counts.values()) if self._arrival_counts else 1
        imp = arrival / max(max_arrivals, 1)

        # Combine with alpha weighting
        alpha = self.config.helpfulness_alpha
        helpfulness = alpha * sim + (1 - alpha) * imp

        return helpfulness

    async def _llm_select_neighbors(
        self,
        query: str,
        current_passages: List[Passage],
        available_neighbors: List[Tuple[Passage, Edge]],
    ) -> List[Tuple[Passage, Edge, float]]:
        """Use LLM to select which neighbors to explore.

        P0 Fix: Core Retrieve-Reason-Prune innovation - LLM decides path direction.

        Args:
            query: Original user query
            current_passages: Passages we're expanding from
            available_neighbors: Candidate neighbors with edges

        Returns:
            Selected neighbors with reasoning scores (passage, edge, score)
        """
        if not available_neighbors:
            return []

        if not self.config.enable_llm_reasoning:
            # Fallback to embedding-based selection
            return [(p, e, e.weight) for p, e in available_neighbors]

        try:
            client = await self._get_client()

            # Format current context
            current_context = "\n".join([
                f"- {p.content[:200]}..." for p in current_passages[:3]
            ])

            # Format neighbors
            neighbor_descriptions = []
            for i, (passage, edge) in enumerate(available_neighbors[:10], 1):
                edge_info = f"[{edge.edge_type.value}]" if edge.edge_type else ""
                neighbor_descriptions.append(
                    f"{i}. {edge_info} {passage.content[:150]}..."
                )
            neighbors_text = "\n".join(neighbor_descriptions)

            prompt = f"""You are helping navigate a knowledge graph to answer a question.

Question: {query}

Current passages you've collected:
{current_context}

Available neighbor passages to explore:
{neighbors_text}

Which of these neighbor passages (1-{len(available_neighbors)}) should we explore to better answer the question?

Consider:
- Which passages add NEW information not already covered?
- Which passages are directly relevant to the question?
- Which passages might lead to the answer?

Respond with ONLY a JSON array of selected passage numbers with relevance scores (0-1):
{{"selections": [{{"id": 1, "score": 0.9, "reason": "..."}}]}}"""

            response = await client.post(
                "/api/generate",
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 256}
                },
                timeout=30.0,
            )
            response.raise_for_status()
            response_text = response.json().get("response", "")

            # Parse LLM response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    selections = parsed.get("selections", [])

                    selected = []
                    for sel in selections:
                        idx = sel.get("id", 1) - 1
                        score = sel.get("score", 0.5)
                        if 0 <= idx < len(available_neighbors):
                            passage, edge = available_neighbors[idx]
                            selected.append((passage, edge, score))

                    if selected:
                        logger.debug(f"LLM selected {len(selected)} neighbors")
                        return selected

            except json.JSONDecodeError:
                pass

            # Fallback: return all with edge weights
            return [(p, e, e.weight) for p, e in available_neighbors]

        except Exception as e:
            logger.warning(f"LLM neighbor selection failed: {e}")
            return [(p, e, e.weight) for p, e in available_neighbors]

    async def _embed_query(self, query: str) -> np.ndarray:
        """Get embedding for query."""
        try:
            client = await self._get_client()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "prompt": query,
                },
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"])
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return np.zeros(1024)

    def _compute_similarity(
        self,
        query_emb: np.ndarray,
        passage_emb: np.ndarray,
    ) -> float:
        """Compute cosine similarity."""
        if passage_emb is None or query_emb is None:
            return 0.0
        return float(np.dot(query_emb, passage_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(passage_emb) + 1e-8
        ))

    def _get_seed_passages(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[Passage, float]]:
        """Get initial seed passages by similarity."""
        scores = []
        for passage in self.graph.passages.values():
            score = self._compute_similarity(query_emb, passage.embedding)
            scores.append((passage, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    async def _expand_beam_search_with_reasoning(
        self,
        query: str,
        query_emb: np.ndarray,
        seeds: List[Tuple[Passage, float]],
    ) -> List[ReasoningPath]:
        """Expand paths using beam search with LLM reasoning.

        P0 Fix: Implements Retrieve-Reason-Prune pipeline from HopRAG paper.
        Uses LLM to select which neighbors to explore and Helpfulness metric for scoring.
        """
        # Initialize beams with seed passages
        current_beams: List[ReasoningPath] = []
        for passage, score in seeds:
            path = ReasoningPath(
                passages=[passage.id],
                edges=[],
                score=score,
                hop_scores=[score],
            )
            current_beams.append(path)
            # P0 Fix: Track arrivals for importance scoring
            self._arrival_counts[passage.id] = self._arrival_counts.get(passage.id, 0) + 1

        all_paths = list(current_beams)
        visited_states: Set[Tuple[str, ...]] = {(p.passages[-1],) for p in current_beams}

        # Expand for each hop
        for hop in range(self.config.max_hops):
            next_beams = []

            for path in current_beams:
                current_id = path.passages[-1]
                current_passage = self.graph.get_passage(current_id)

                # RETRIEVE: Get all neighbors
                raw_neighbors = self.graph.get_neighbors(current_id)

                # Filter out already visited
                available_neighbors = []
                for neighbor_id, edge in raw_neighbors:
                    if neighbor_id in path.passages:
                        continue
                    state = tuple(sorted(path.passages + [neighbor_id]))
                    if state in visited_states:
                        continue
                    neighbor_passage = self.graph.get_passage(neighbor_id)
                    if neighbor_passage:
                        available_neighbors.append((neighbor_passage, edge))

                if not available_neighbors:
                    continue

                # REASON: Use LLM to select which neighbors to explore
                current_passages = [self.graph.get_passage(pid) for pid in path.passages
                                   if self.graph.get_passage(pid)]
                current_passages = [p for p in current_passages if p]

                selected = await self._llm_select_neighbors(
                    query, current_passages, available_neighbors
                )

                # PRUNE: Only keep high-scoring selections
                for neighbor_passage, edge, llm_score in selected:
                    state = tuple(sorted(path.passages + [neighbor_passage.id]))
                    if state in visited_states:
                        continue
                    visited_states.add(state)

                    # P0 Fix: Track arrivals for Helpfulness metric
                    self._arrival_counts[neighbor_passage.id] = (
                        self._arrival_counts.get(neighbor_passage.id, 0) + 1
                    )

                    # P0 Fix: Use Helpfulness metric instead of raw similarity
                    helpfulness = self._calculate_helpfulness(neighbor_passage, query_emb)

                    # Combine path score with hop decay and LLM reasoning score
                    new_score = (
                        path.score * self.config.hop_decay +
                        helpfulness * edge.weight * llm_score
                    ) / (1 + self.config.hop_decay)

                    if new_score < self.config.min_path_score:
                        continue

                    # Create new path
                    new_path = ReasoningPath(
                        passages=path.passages + [neighbor_passage.id],
                        edges=path.edges + [edge],
                        score=new_score,
                        hop_scores=path.hop_scores + [helpfulness],
                    )
                    next_beams.append(new_path)

            if not next_beams:
                break

            # Keep top-k beams
            next_beams.sort(key=lambda p: p.score, reverse=True)
            current_beams = next_beams[:self.config.beam_width]
            all_paths.extend(current_beams)

        # Deduplicate and sort all paths
        unique_paths = {}
        for path in all_paths:
            key = tuple(path.passages)
            if key not in unique_paths or path.score > unique_paths[key].score:
                unique_paths[key] = path

        result_paths = list(unique_paths.values())
        result_paths.sort(key=lambda p: p.score, reverse=True)
        return result_paths

    def _expand_beam_search(
        self,
        query_emb: np.ndarray,
        seeds: List[Tuple[Passage, float]],
    ) -> List[ReasoningPath]:
        """Expand paths using beam search with Helpfulness metric.

        P1 Fix: Now uses Helpfulness metric H(v,q) = α×SIM + (1-α)×IMP
        for scoring instead of raw similarity. Tracks arrival counts
        for importance scoring.
        """
        # Initialize beams with seed passages
        current_beams: List[ReasoningPath] = []
        for passage, score in seeds:
            path = ReasoningPath(
                passages=[passage.id],
                edges=[],
                score=score,
                hop_scores=[score],
            )
            current_beams.append(path)
            # P1 Fix: Track arrivals for importance scoring
            self._arrival_counts[passage.id] = self._arrival_counts.get(passage.id, 0) + 1

        all_paths = list(current_beams)
        visited_states: Set[Tuple[str, ...]] = {(p.passages[-1],) for p in current_beams}

        # Expand for each hop
        for hop in range(self.config.max_hops):
            next_beams = []

            for path in current_beams:
                current_id = path.passages[-1]

                # Get neighbors
                neighbors = self.graph.get_neighbors(current_id)

                for neighbor_id, edge in neighbors:
                    # Skip if already in path (avoid cycles)
                    if neighbor_id in path.passages:
                        continue

                    # Create state signature for pruning
                    state = tuple(sorted(path.passages + [neighbor_id]))
                    if state in visited_states:
                        continue
                    visited_states.add(state)

                    # Calculate new score with hop decay
                    neighbor_passage = self.graph.get_passage(neighbor_id)
                    if not neighbor_passage:
                        continue

                    # P1 Fix: Track arrivals for importance scoring
                    self._arrival_counts[neighbor_id] = (
                        self._arrival_counts.get(neighbor_id, 0) + 1
                    )

                    # P1 Fix: Use Helpfulness metric instead of raw similarity
                    helpfulness = self._calculate_helpfulness(neighbor_passage, query_emb)

                    # Combine path score with hop decay using Helpfulness
                    new_score = (
                        path.score * self.config.hop_decay +
                        helpfulness * edge.weight
                    ) / (1 + self.config.hop_decay)

                    if new_score < self.config.min_path_score:
                        continue

                    # Create new path
                    new_path = ReasoningPath(
                        passages=path.passages + [neighbor_id],
                        edges=path.edges + [edge],
                        score=new_score,
                        hop_scores=path.hop_scores + [helpfulness],
                    )
                    next_beams.append(new_path)

            if not next_beams:
                break

            # Keep top-k beams
            next_beams.sort(key=lambda p: p.score, reverse=True)
            current_beams = next_beams[:self.config.beam_width]
            all_paths.extend(current_beams)

        # Deduplicate and sort all paths
        unique_paths = {}
        for path in all_paths:
            key = tuple(path.passages)
            if key not in unique_paths or path.score > unique_paths[key].score:
                unique_paths[key] = path

        result_paths = list(unique_paths.values())
        result_paths.sort(key=lambda p: p.score, reverse=True)
        return result_paths

    def _expand_ppr(
        self,
        query_emb: np.ndarray,
        seeds: List[Tuple[Passage, float]],
        damping: float = 0.85,
        iterations: int = 10,
    ) -> List[ReasoningPath]:
        """Expand using Personalized PageRank."""
        # Initialize PPR scores
        n_passages = len(self.graph.passages)
        if n_passages == 0:
            return []

        passage_ids = list(self.graph.passages.keys())
        id_to_idx = {pid: i for i, pid in enumerate(passage_ids)}

        # Personalization vector based on seeds
        personalization = np.zeros(n_passages)
        for passage, score in seeds:
            idx = id_to_idx.get(passage.id)
            if idx is not None:
                personalization[idx] = score

        # Normalize
        total = personalization.sum()
        if total > 0:
            personalization /= total

        # Run PPR iterations
        scores = personalization.copy()
        for _ in range(iterations):
            new_scores = np.zeros(n_passages)

            for i, pid in enumerate(passage_ids):
                # Propagate score to neighbors
                neighbors = self.graph.get_neighbors(pid)
                if neighbors:
                    neighbor_share = scores[i] * damping / len(neighbors)
                    for neighbor_id, edge in neighbors:
                        j = id_to_idx.get(neighbor_id)
                        if j is not None:
                            new_scores[j] += neighbor_share * edge.weight

            # Add teleportation
            new_scores += (1 - damping) * personalization
            scores = new_scores

        # Build paths from high-scoring passages
        sorted_indices = np.argsort(scores)[::-1]

        paths = []
        for idx in sorted_indices[:self.config.beam_width]:
            pid = passage_ids[idx]
            path = ReasoningPath(
                passages=[pid],
                edges=[],
                score=float(scores[idx]),
                hop_scores=[float(scores[idx])],
            )
            paths.append(path)

        return paths

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> HopRAGResult:
        """Retrieve multi-hop reasoning paths."""
        start_time = time.time()

        query_emb = await self._embed_query(query)

        # Get seed passages
        seeds = self._get_seed_passages(query_emb, top_k=self.config.beam_width)

        if not seeds:
            return HopRAGResult(
                query=query,
                seed_passages=[],
                reasoning_paths=[],
                all_relevant_passages=[],
                retrieval_time_ms=(time.time() - start_time) * 1000,
                num_hops_explored=0,
            )

        # P0 Fix: Reset arrival counts for new query (for Helpfulness metric)
        self._arrival_counts = {}

        # Expand paths based on strategy
        if self.config.hop_strategy == HopStrategy.PERSONALIZED_PAGERANK:
            paths = self._expand_ppr(query_emb, seeds)
        elif self.config.enable_llm_reasoning:
            # P0 Fix: Use LLM-based Retrieve-Reason-Prune pipeline
            paths = await self._expand_beam_search_with_reasoning(query, query_emb, seeds)
        else:
            paths = self._expand_beam_search(query_emb, seeds)

        # Collect all unique passages from paths
        all_passage_ids = set()
        for path in paths:
            all_passage_ids.update(path.passages)

        all_passages = [
            self.graph.get_passage(pid)
            for pid in all_passage_ids
            if self.graph.get_passage(pid)
        ]

        # P1 Fix: Sort by Helpfulness metric H(v,q) = α×SIM + (1-α)×IMP
        # instead of raw similarity for final ranking
        all_passages.sort(
            key=lambda p: self._calculate_helpfulness(p, query_emb),
            reverse=True,
        )

        # P1 Fix: Calculate and store helpfulness scores for all passages
        helpfulness_scores = {}
        for passage in all_passages:
            passage.importance_score = self._arrival_counts.get(passage.id, 0)
            helpfulness_scores[passage.id] = self._calculate_helpfulness(passage, query_emb)

        return HopRAGResult(
            query=query,
            seed_passages=[p for p, _ in seeds],
            reasoning_paths=paths[:top_k],
            all_relevant_passages=all_passages[:top_k],
            retrieval_time_ms=(time.time() - start_time) * 1000,
            num_hops_explored=max((len(p) - 1 for p in paths), default=0),
            # P1 Fix: Include helpfulness data in result
            helpfulness_scores=helpfulness_scores,
            arrival_counts=dict(self._arrival_counts),
        )


# Convenience functions
_graphs: Dict[str, PassageGraph] = {}
_builder_instance: Optional[HopRAGBuilder] = None


async def get_hoprag_builder(
    config: Optional[HopRAGConfig] = None,
) -> HopRAGBuilder:
    """Get or create HopRAG builder instance."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = HopRAGBuilder(config)
    return _builder_instance


async def build_hoprag_graph(
    passages: List[Tuple[str, str]],
    graph_id: str = "default",
    source_doc: str = "",
    config: Optional[HopRAGConfig] = None,
) -> PassageGraph:
    """Build and store a HopRAG passage graph."""
    builder = await get_hoprag_builder(config)
    graph = await builder.build_graph(passages, source_doc)
    _graphs[graph_id] = graph
    return graph


async def hoprag_retrieve(
    query: str,
    graph_id: str = "default",
    top_k: int = 10,
    config: Optional[HopRAGConfig] = None,
) -> HopRAGResult:
    """Retrieve from a stored HopRAG graph."""
    graph = _graphs.get(graph_id)
    if graph is None:
        raise ValueError(f"Graph '{graph_id}' not found. Build it first with build_hoprag_graph().")

    retriever = HopRAGRetriever(graph, config)
    result = await retriever.retrieve(query, top_k)
    await retriever.close()
    return result


def get_hoprag_graph(graph_id: str = "default") -> Optional[PassageGraph]:
    """Get a stored HopRAG graph."""
    return _graphs.get(graph_id)


def list_hoprag_graphs() -> Dict[str, Dict[str, Any]]:
    """List all stored HopRAG graphs."""
    return {
        graph_id: graph.to_dict()
        for graph_id, graph in _graphs.items()
    }
