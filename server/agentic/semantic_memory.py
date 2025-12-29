"""
A-MEM Semantic Memory Network Module

Based on A-MEM research (arXiv 2502.12110) achieving 35% F1 improvement.

Key insight: Zettelkasten-inspired interconnected memory where each memory
carries structured text attributes, embedding vectors, and dynamic connections
based on semantic similarity.

Features:
- Dynamic connection establishment based on embedding similarity
- Memory traversal: finding → related findings → sources
- Bidirectional links for exploration
- Connection strength based on similarity + co-occurrence
- Automatic connection updates on new memory addition

References:
- A-MEM: Agent Memory via Zettelkasten (arXiv 2502.12110)
- Zettelkasten Method: Luhmann's slip-box system
"""

import asyncio
import logging
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import httpx
import numpy as np

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memories in the network."""
    FINDING = "finding"            # Discovered fact or claim
    SOURCE = "source"              # Information source (URL, document)
    ENTITY = "entity"              # Named entity (person, org, concept)
    REASONING = "reasoning"        # Intermediate reasoning step
    OBSERVATION = "observation"    # Tool output or observation
    EXAMPLE = "example"            # Successful pattern
    QUESTION = "question"          # Decomposed question
    ANSWER = "answer"              # Answer to a question


class ConnectionType(str, Enum):
    """Types of connections between memories."""
    SEMANTIC = "semantic"          # Based on embedding similarity
    REFERENCE = "reference"        # Explicit citation/reference
    SUPPORTS = "supports"          # Evidence supporting a claim
    CONTRADICTS = "contradicts"    # Conflicting information
    DERIVED_FROM = "derived_from"  # Conclusion from evidence
    ANSWERS = "answers"            # Answer to question
    RELATED = "related"            # General relatedness


@dataclass
class MemoryConnection:
    """A connection between two memories."""
    target_id: str
    connection_type: ConnectionType
    strength: float = 0.5          # 0-1, how strong the connection is
    bidirectional: bool = True     # Whether connection goes both ways
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "connection_type": self.connection_type.value,
            "strength": self.strength,
            "bidirectional": self.bidirectional,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConnection":
        return cls(
            target_id=data["target_id"],
            connection_type=ConnectionType(data["connection_type"]),
            strength=data.get("strength", 0.5),
            bidirectional=data.get("bidirectional", True),
            metadata=data.get("metadata", {})
        )


@dataclass
class Memory:
    """A single memory node in the network."""
    id: str
    memory_type: MemoryType
    content: str                   # Main text content
    attributes: Dict[str, Any]     # Structured attributes
    embedding: Optional[List[float]] = None
    connections: List[MemoryConnection] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "attributes": self.attributes,
            "embedding": self.embedding,
            "connections": [c.to_dict() for c in self.connections],
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            id=data["id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            attributes=data.get("attributes", {}),
            embedding=data.get("embedding"),
            connections=[MemoryConnection.from_dict(c) for c in data.get("connections", [])],
            created_at=data.get("created_at", time.time()),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed", time.time())
        )


@dataclass
class TraversalResult:
    """Result of traversing the memory network."""
    path: List[str]                # Memory IDs in traversal order
    memories: List[Memory]         # Actual memories
    total_strength: float          # Product of connection strengths
    connection_types: List[ConnectionType]  # Types along the path


class SemanticMemoryNetwork:
    """
    Zettelkasten-inspired interconnected memory network.

    Each memory carries:
    - Structured text attributes
    - Embedding vector
    - Dynamic connections based on similarity

    Key features:
    - Automatic connection establishment on memory addition
    - Bidirectional traversal
    - Connection strength based on semantic similarity
    - Access-based recency weighting
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "mxbai-embed-large",
        similarity_threshold: float = 0.7,
        max_connections: int = 10,
        auto_connect: bool = True
    ):
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_connections = max_connections
        self.auto_connect = auto_connect

        # Memory storage
        self.memories: Dict[str, Memory] = {}

        # Index for fast lookup by type
        self._type_index: Dict[MemoryType, Set[str]] = {t: set() for t in MemoryType}

        # Embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}

    def _generate_id(self, content: str, memory_type: str) -> str:
        """Generate unique memory ID."""
        hash_input = f"{content}:{memory_type}:{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        cache_key = hashlib.sha256(text[:500].encode()).hexdigest()[:16]

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        text_truncated = text[:1000] if len(text) > 1000 else text

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text_truncated
                    }
                )
                response.raise_for_status()
                embedding = response.json().get("embedding", [])

            self._embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return []

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        attributes: Dict[str, Any] = None,
        explicit_connections: List[Tuple[str, ConnectionType]] = None
    ) -> Memory:
        """
        Add a new memory to the network.

        Automatically:
        1. Generates embedding
        2. Finds similar memories
        3. Establishes connections based on similarity threshold

        Args:
            content: Main text content
            memory_type: Type of memory
            attributes: Optional structured attributes
            explicit_connections: List of (memory_id, connection_type) tuples

        Returns:
            The created Memory object
        """
        attributes = attributes or {}
        explicit_connections = explicit_connections or []

        memory_id = self._generate_id(content, memory_type.value)

        # Get embedding
        embedding = await self._get_embedding(content)

        # Create memory
        memory = Memory(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            attributes=attributes,
            embedding=embedding
        )

        # Add explicit connections
        for target_id, conn_type in explicit_connections:
            if target_id in self.memories:
                memory.connections.append(MemoryConnection(
                    target_id=target_id,
                    connection_type=conn_type,
                    strength=0.9  # Explicit connections are strong
                ))

                # Add reverse connection if bidirectional
                target_memory = self.memories[target_id]
                reverse_conn = MemoryConnection(
                    target_id=memory_id,
                    connection_type=conn_type,
                    strength=0.9
                )
                if reverse_conn.target_id not in [c.target_id for c in target_memory.connections]:
                    target_memory.connections.append(reverse_conn)

        # Auto-connect based on similarity
        if self.auto_connect and embedding:
            await self._establish_connections(memory)

        # Store memory
        self.memories[memory_id] = memory
        self._type_index[memory_type].add(memory_id)

        logger.debug(
            f"Added memory {memory_id} ({memory_type.value}) "
            f"with {len(memory.connections)} connections"
        )

        return memory

    async def _establish_connections(self, new_memory: Memory) -> None:
        """
        Establish semantic connections between new memory and existing ones.

        Uses embedding similarity to find related memories.
        """
        if not new_memory.embedding:
            return

        similarities: List[Tuple[str, float]] = []

        for memory_id, memory in self.memories.items():
            if memory_id == new_memory.id:
                continue

            if not memory.embedding:
                continue

            similarity = self._cosine_similarity(
                new_memory.embedding,
                memory.embedding
            )

            if similarity >= self.similarity_threshold:
                similarities.append((memory_id, similarity))

        # Sort by similarity and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:self.max_connections]

        # Establish connections
        for target_id, similarity in top_similar:
            # Skip if already connected
            existing_targets = {c.target_id for c in new_memory.connections}
            if target_id in existing_targets:
                continue

            # Add connection to new memory
            new_memory.connections.append(MemoryConnection(
                target_id=target_id,
                connection_type=ConnectionType.SEMANTIC,
                strength=similarity
            ))

            # Add reverse connection
            target_memory = self.memories[target_id]
            reverse_existing = {c.target_id for c in target_memory.connections}
            if new_memory.id not in reverse_existing:
                target_memory.connections.append(MemoryConnection(
                    target_id=new_memory.id,
                    connection_type=ConnectionType.SEMANTIC,
                    strength=similarity
                ))

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID, updating access statistics."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access_count += 1
            memory.last_accessed = time.time()
        return memory

    def get_connected(
        self,
        memory_id: str,
        connection_type: Optional[ConnectionType] = None,
        min_strength: float = 0.0
    ) -> List[Memory]:
        """
        Get all memories connected to the given memory.

        Args:
            memory_id: Source memory ID
            connection_type: Filter by connection type (optional)
            min_strength: Minimum connection strength

        Returns:
            List of connected memories
        """
        memory = self.memories.get(memory_id)
        if not memory:
            return []

        connected = []
        for conn in memory.connections:
            if connection_type and conn.connection_type != connection_type:
                continue
            if conn.strength < min_strength:
                continue

            target = self.memories.get(conn.target_id)
            if target:
                connected.append(target)

        return connected

    async def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        min_strength: float = 0.3,
        target_type: Optional[MemoryType] = None
    ) -> List[TraversalResult]:
        """
        Traverse the memory network from a starting point.

        Uses BFS to find paths through the network.

        Args:
            start_id: Starting memory ID
            max_depth: Maximum traversal depth
            min_strength: Minimum connection strength to follow
            target_type: Stop at memories of this type (optional)

        Returns:
            List of traversal results (paths through the network)
        """
        if start_id not in self.memories:
            return []

        results = []
        visited = {start_id}

        # BFS queue: (current_id, path, total_strength, conn_types)
        queue = [(start_id, [start_id], 1.0, [])]

        while queue:
            current_id, path, strength, conn_types = queue.pop(0)

            if len(path) > max_depth + 1:
                continue

            current = self.memories.get(current_id)
            if not current:
                continue

            # Check if we found a target
            if target_type and current.memory_type == target_type and len(path) > 1:
                memories = [self.memories[mid] for mid in path if mid in self.memories]
                results.append(TraversalResult(
                    path=path,
                    memories=memories,
                    total_strength=strength,
                    connection_types=conn_types
                ))
                continue

            # Explore connections
            for conn in current.connections:
                if conn.target_id in visited:
                    continue
                if conn.strength < min_strength:
                    continue

                visited.add(conn.target_id)
                new_path = path + [conn.target_id]
                new_strength = strength * conn.strength
                new_types = conn_types + [conn.connection_type]

                queue.append((conn.target_id, new_path, new_strength, new_types))

        # Sort by total strength
        results.sort(key=lambda r: r.total_strength, reverse=True)

        return results

    def get_by_type(self, memory_type: MemoryType) -> List[Memory]:
        """Get all memories of a specific type."""
        return [
            self.memories[mid]
            for mid in self._type_index[memory_type]
            if mid in self.memories
        ]

    async def find_similar(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[MemoryType] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Find memories similar to a query.

        Args:
            query: Search query
            top_k: Number of results
            memory_type: Filter by type (optional)

        Returns:
            List of (memory, similarity) tuples
        """
        query_embedding = await self._get_embedding(query)
        if not query_embedding:
            return []

        similarities = []

        for memory_id, memory in self.memories.items():
            if memory_type and memory.memory_type != memory_type:
                continue
            if not memory.embedding:
                continue

            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            similarities.append((memory, similarity))

        # Sort and return top K
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_context_for_query(
        self,
        start_ids: List[str],
        max_context_items: int = 10
    ) -> str:
        """
        Generate context string from memory network.

        Traverses from starting points to gather related context.
        """
        context_items = []
        seen = set()

        for start_id in start_ids:
            memory = self.memories.get(start_id)
            if not memory or start_id in seen:
                continue

            seen.add(start_id)
            context_items.append(f"[{memory.memory_type.value}] {memory.content}")

            # Add connected items
            for conn in memory.connections[:3]:  # Top 3 connections
                connected = self.memories.get(conn.target_id)
                if connected and conn.target_id not in seen:
                    seen.add(conn.target_id)
                    context_items.append(
                        f"  → [{connected.memory_type.value}] {connected.content[:200]}"
                    )

            if len(context_items) >= max_context_items:
                break

        return "\n".join(context_items[:max_context_items])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the memory network."""
        return {
            "memories": {mid: m.to_dict() for mid, m in self.memories.items()},
            "similarity_threshold": self.similarity_threshold,
            "max_connections": self.max_connections
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], ollama_url: str = "http://localhost:11434") -> "SemanticMemoryNetwork":
        """Deserialize a memory network."""
        network = cls(
            ollama_url=ollama_url,
            similarity_threshold=data.get("similarity_threshold", 0.7),
            max_connections=data.get("max_connections", 10)
        )

        for memory_id, memory_data in data.get("memories", {}).items():
            memory = Memory.from_dict(memory_data)
            network.memories[memory_id] = memory
            network._type_index[memory.memory_type].add(memory_id)

        return network

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        total_connections = sum(len(m.connections) for m in self.memories.values())

        return {
            "total_memories": len(self.memories),
            "total_connections": total_connections,
            "memories_by_type": {
                t.value: len(ids) for t, ids in self._type_index.items()
            },
            "avg_connections_per_memory": (
                total_connections / len(self.memories) if self.memories else 0
            ),
            "similarity_threshold": self.similarity_threshold
        }


# Singleton instance
_semantic_memory: Optional[SemanticMemoryNetwork] = None


def get_semantic_memory(
    ollama_url: str = "http://localhost:11434"
) -> SemanticMemoryNetwork:
    """Get or create the semantic memory network singleton."""
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = SemanticMemoryNetwork(ollama_url=ollama_url)
    return _semantic_memory
