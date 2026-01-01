"""
Hyperbolic Embeddings for Hierarchical Document Retrieval

Implements HyperbolicRAG-inspired retrieval using Poincaré ball geometry.
Based on:
- HyperbolicRAG (arXiv:2511.18808) - Hierarchy-aware retrieval with hyperbolic geometry
- Poincaré Embeddings (NeurIPS 2017) - Learning hierarchical representations
- geoopt library for Poincaré ball operations

Key Insight:
- Euclidean space is inefficient for hierarchical structures
- Hyperbolic space naturally encodes tree-like hierarchies
- General concepts cluster near origin, specific details near boundary

For FANUC documentation: manual → chapter → section → procedure → step

Author: Claude Code
Date: December 2025
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import sqlite3
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HierarchyLevel(Enum):
    """Hierarchy levels for document structure."""
    CORPUS = 0      # Entire corpus (most general)
    MANUAL = 1      # Manual/document level
    CHAPTER = 2     # Chapter/section
    SECTION = 3     # Subsection
    PROCEDURE = 4   # Procedure/topic
    STEP = 5        # Individual step (most specific)


@dataclass
class HyperbolicDocument:
    """Document with hyperbolic embedding."""
    doc_id: str
    content: str
    euclidean_embedding: np.ndarray  # Original dense embedding
    hyperbolic_embedding: np.ndarray  # Projected to Poincaré ball
    hierarchy_level: HierarchyLevel = HierarchyLevel.STEP
    parent_id: Optional[str] = None
    depth: float = 0.0  # Radial distance from origin (0=general, 1=specific)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperbolicSearchResult:
    """Result from hyperbolic search."""
    doc_id: str
    content: str
    euclidean_score: float
    hyperbolic_score: float
    fused_score: float
    hierarchy_level: HierarchyLevel
    depth: float
    metadata: Dict[str, Any]


class PoincareBall:
    """
    Poincaré Ball model for hyperbolic geometry.

    The Poincaré ball is the unit ball with the metric:
    ds² = 4 / (1 - ||x||²)² * ||dx||²

    Properties:
    - Points near origin: general/abstract concepts
    - Points near boundary: specific/detailed concepts
    - Geodesic distance grows exponentially near boundary
    """

    def __init__(self, dim: int = 768, curvature: float = -1.0, eps: float = 1e-5):
        """
        Initialize Poincaré ball.

        Args:
            dim: Embedding dimension
            curvature: Negative curvature (default -1.0)
            eps: Small value for numerical stability
        """
        self.dim = dim
        self.c = abs(curvature)  # Use absolute value
        self.eps = eps

    def _clamp_norm(self, x: np.ndarray, max_norm: float = 1.0 - 1e-5) -> np.ndarray:
        """Clamp vector to stay inside the ball."""
        norm = np.linalg.norm(x)
        if norm > max_norm:
            return x * (max_norm / norm)
        return x

    def _mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Möbius addition in the Poincaré ball.

        x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) /
                (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
        """
        x_norm_sq = np.sum(x ** 2)
        y_norm_sq = np.sum(y ** 2)
        xy = np.sum(x * y)

        c = self.c
        num = (1 + 2 * c * xy + c * y_norm_sq) * x + (1 - c * x_norm_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_norm_sq * y_norm_sq

        result = num / (denom + self.eps)
        return self._clamp_norm(result)

    def exp_map(self, v: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Exponential map: project tangent vector to manifold.

        exp_x(v) = x ⊕ (tanh(√c * λ_x * ||v|| / 2) * v / (√c * ||v||))

        where λ_x = 2 / (1 - c||x||²) is the conformal factor

        Args:
            v: Tangent vector (Euclidean embedding to project)
            x: Base point (default: origin)

        Returns:
            Point on Poincaré ball
        """
        if x is None:
            x = np.zeros(self.dim)

        v_norm = np.linalg.norm(v) + self.eps
        x_norm_sq = np.sum(x ** 2)

        # Conformal factor
        lambda_x = 2 / (1 - self.c * x_norm_sq + self.eps)

        # Compute exponential map
        sqrt_c = np.sqrt(self.c)
        t = np.tanh(sqrt_c * lambda_x * v_norm / 2)
        direction = v / v_norm

        # Result before Möbius addition
        y = t * direction / sqrt_c

        if np.allclose(x, 0):
            return self._clamp_norm(y)

        return self._mobius_add(x, y)

    def log_map(self, y: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Logarithmic map: project point to tangent space.

        log_x(y) = (2 / (√c * λ_x)) * arctanh(√c * ||-x ⊕ y||) * (-x ⊕ y) / ||-x ⊕ y||
        """
        if x is None:
            x = np.zeros(self.dim)

        x_norm_sq = np.sum(x ** 2)
        lambda_x = 2 / (1 - self.c * x_norm_sq + self.eps)

        # Möbius subtraction: -x ⊕ y
        neg_x = -x
        diff = self._mobius_add(neg_x, y)
        diff_norm = np.linalg.norm(diff) + self.eps

        sqrt_c = np.sqrt(self.c)
        coeff = (2 / (sqrt_c * lambda_x)) * np.arctanh(sqrt_c * diff_norm)

        return coeff * diff / diff_norm

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Geodesic distance in Poincaré ball.

        d(x, y) = (2/√c) * arctanh(√c * ||−x ⊕ y||)
        """
        neg_x = -x
        diff = self._mobius_add(neg_x, y)
        diff_norm = np.linalg.norm(diff)

        sqrt_c = np.sqrt(self.c)
        return (2 / sqrt_c) * np.arctanh(np.clip(sqrt_c * diff_norm, 0, 1 - self.eps))

    def get_depth(self, x: np.ndarray) -> float:
        """
        Get radial depth (distance from origin).

        Points near origin are general, near boundary are specific.
        Normalized to [0, 1] range.
        """
        norm = np.linalg.norm(x)
        # Use arctanh for smooth mapping
        return np.tanh(norm * 2)  # Normalize to roughly [0, 1]


class HyperbolicRetriever:
    """
    Retriever using hyperbolic embeddings for hierarchical documents.

    Features:
    - Projects Euclidean embeddings to Poincaré ball
    - Hierarchy-aware depth encoding
    - Mutual-ranking fusion (Euclidean + Hyperbolic)
    - SQLite persistence for embeddings
    """

    def __init__(
        self,
        dim: int = 768,
        curvature: float = -1.0,
        euclidean_weight: float = 0.5,
        db_path: str = "data/hyperbolic_embeddings.db"
    ):
        """
        Initialize hyperbolic retriever.

        Args:
            dim: Embedding dimension
            curvature: Poincaré ball curvature
            euclidean_weight: Weight for Euclidean scores in fusion (0-1)
            db_path: Path to SQLite database
        """
        self.dim = dim
        self.manifold = PoincareBall(dim=dim, curvature=curvature)
        self.euclidean_weight = euclidean_weight
        self.hyperbolic_weight = 1.0 - euclidean_weight
        self.db_path = db_path

        # In-memory cache
        self.documents: Dict[str, HyperbolicDocument] = {}

        # Initialize database
        self._init_db()

        logger.info(f"HyperbolicRetriever initialized: dim={dim}, c={curvature}")

    def _init_db(self):
        """Initialize SQLite database for persistence."""
        import os
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperbolic_documents (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                euclidean_embedding BLOB NOT NULL,
                hyperbolic_embedding BLOB NOT NULL,
                hierarchy_level INTEGER DEFAULT 5,
                parent_id TEXT,
                depth REAL DEFAULT 0.0,
                metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hierarchy_level
            ON hyperbolic_documents(hierarchy_level)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_parent_id
            ON hyperbolic_documents(parent_id)
        """)

        conn.commit()
        conn.close()

    def _embed_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Convert numpy array to bytes."""
        return embedding.astype(np.float32).tobytes()

    def _bytes_to_embed(self, data: bytes, dim: int) -> np.ndarray:
        """Convert bytes to numpy array."""
        return np.frombuffer(data, dtype=np.float32).reshape(-1)[:dim]

    def project_to_hyperbolic(
        self,
        euclidean_embedding: np.ndarray,
        hierarchy_level: HierarchyLevel = HierarchyLevel.STEP,
        scale_by_depth: bool = True
    ) -> np.ndarray:
        """
        Project Euclidean embedding to Poincaré ball.

        Args:
            euclidean_embedding: Dense vector from encoder
            hierarchy_level: Document hierarchy level
            scale_by_depth: Whether to scale embedding by hierarchy depth

        Returns:
            Hyperbolic embedding in Poincaré ball
        """
        # Normalize Euclidean embedding
        emb = euclidean_embedding.copy().astype(np.float64)
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm

        # Scale based on hierarchy level
        # Higher levels (more specific) should be closer to boundary
        if scale_by_depth:
            # Map hierarchy level to target radius
            # CORPUS (0) → 0.1, STEP (5) → 0.9
            target_radius = 0.1 + (hierarchy_level.value / 5) * 0.8
            emb = emb * target_radius

        # Project to Poincaré ball using exponential map
        hyperbolic = self.manifold.exp_map(emb)

        return hyperbolic.astype(np.float32)

    async def add_document(
        self,
        doc_id: str,
        content: str,
        euclidean_embedding: np.ndarray,
        hierarchy_level: HierarchyLevel = HierarchyLevel.STEP,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HyperbolicDocument:
        """
        Add a document with hyperbolic embedding.

        Args:
            doc_id: Unique document ID
            content: Document text
            euclidean_embedding: Pre-computed dense embedding
            hierarchy_level: Position in document hierarchy
            parent_id: Parent document ID (for tree structure)
            metadata: Additional metadata

        Returns:
            HyperbolicDocument with both embeddings
        """
        # Ensure correct dimension
        if len(euclidean_embedding) != self.dim:
            # Truncate or pad
            if len(euclidean_embedding) > self.dim:
                euclidean_embedding = euclidean_embedding[:self.dim]
            else:
                padded = np.zeros(self.dim)
                padded[:len(euclidean_embedding)] = euclidean_embedding
                euclidean_embedding = padded

        # Project to hyperbolic space
        hyperbolic_embedding = self.project_to_hyperbolic(
            euclidean_embedding,
            hierarchy_level
        )

        # Calculate depth
        depth = self.manifold.get_depth(hyperbolic_embedding)

        # Create document
        doc = HyperbolicDocument(
            doc_id=doc_id,
            content=content,
            euclidean_embedding=euclidean_embedding,
            hyperbolic_embedding=hyperbolic_embedding,
            hierarchy_level=hierarchy_level,
            parent_id=parent_id,
            depth=depth,
            metadata=metadata or {}
        )

        # Store in memory
        self.documents[doc_id] = doc

        # Persist to database
        self._save_document(doc)

        logger.debug(f"Added hyperbolic document: {doc_id}, level={hierarchy_level.name}, depth={depth:.3f}")

        return doc

    def _save_document(self, doc: HyperbolicDocument):
        """Save document to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now(timezone.utc).timestamp()

        cursor.execute("""
            INSERT OR REPLACE INTO hyperbolic_documents
            (doc_id, content, euclidean_embedding, hyperbolic_embedding,
             hierarchy_level, parent_id, depth, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.doc_id,
            doc.content,
            self._embed_to_bytes(doc.euclidean_embedding),
            self._embed_to_bytes(doc.hyperbolic_embedding),
            doc.hierarchy_level.value,
            doc.parent_id,
            doc.depth,
            json.dumps(doc.metadata),
            now,
            now
        ))

        conn.commit()
        conn.close()

    def load_all_documents(self):
        """Load all documents from database into memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM hyperbolic_documents")
        rows = cursor.fetchall()

        for row in rows:
            doc = HyperbolicDocument(
                doc_id=row[0],
                content=row[1],
                euclidean_embedding=self._bytes_to_embed(row[2], self.dim),
                hyperbolic_embedding=self._bytes_to_embed(row[3], self.dim),
                hierarchy_level=HierarchyLevel(row[4]),
                parent_id=row[5],
                depth=row[6],
                metadata=json.loads(row[7]) if row[7] else {}
            )
            self.documents[doc.doc_id] = doc

        conn.close()
        logger.info(f"Loaded {len(self.documents)} hyperbolic documents from database")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a) + 1e-8
        norm_b = np.linalg.norm(b) + 1e-8
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _hyperbolic_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity in hyperbolic space.

        Uses negative geodesic distance normalized to [0, 1].
        """
        distance = self.manifold.distance(a.astype(np.float64), b.astype(np.float64))
        # Convert distance to similarity (smaller distance = higher similarity)
        # Use exponential decay for smooth mapping
        return float(np.exp(-distance))

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        hierarchy_filter: Optional[List[HierarchyLevel]] = None,
        use_fusion: bool = True
    ) -> List[HyperbolicSearchResult]:
        """
        Search for documents using hyperbolic-aware retrieval.

        Args:
            query_embedding: Dense query embedding
            top_k: Number of results to return
            hierarchy_filter: Only return docs at these levels
            use_fusion: Whether to use mutual-ranking fusion

        Returns:
            List of search results with scores
        """
        if not self.documents:
            self.load_all_documents()

        if not self.documents:
            return []

        # Ensure correct dimension
        if len(query_embedding) != self.dim:
            if len(query_embedding) > self.dim:
                query_embedding = query_embedding[:self.dim]
            else:
                padded = np.zeros(self.dim)
                padded[:len(query_embedding)] = query_embedding
                query_embedding = padded

        # Project query to hyperbolic space
        query_hyperbolic = self.project_to_hyperbolic(
            query_embedding,
            HierarchyLevel.STEP,  # Assume query is specific
            scale_by_depth=False   # Don't scale query
        )

        results = []

        for doc_id, doc in self.documents.items():
            # Apply hierarchy filter
            if hierarchy_filter and doc.hierarchy_level not in hierarchy_filter:
                continue

            # Compute Euclidean similarity
            euclidean_score = self._cosine_similarity(
                query_embedding, doc.euclidean_embedding
            )

            # Compute hyperbolic similarity
            hyperbolic_score = self._hyperbolic_similarity(
                query_hyperbolic, doc.hyperbolic_embedding
            )

            # Fuse scores
            if use_fusion:
                fused_score = (
                    self.euclidean_weight * euclidean_score +
                    self.hyperbolic_weight * hyperbolic_score
                )
            else:
                fused_score = hyperbolic_score

            results.append(HyperbolicSearchResult(
                doc_id=doc_id,
                content=doc.content,
                euclidean_score=euclidean_score,
                hyperbolic_score=hyperbolic_score,
                fused_score=fused_score,
                hierarchy_level=doc.hierarchy_level,
                depth=doc.depth,
                metadata=doc.metadata
            ))

        # Sort by fused score
        results.sort(key=lambda x: x.fused_score, reverse=True)

        return results[:top_k]

    async def search_by_hierarchy(
        self,
        query_embedding: np.ndarray,
        target_level: HierarchyLevel,
        top_k: int = 5,
        include_parents: bool = True,
        include_children: bool = True
    ) -> Dict[str, List[HyperbolicSearchResult]]:
        """
        Search with hierarchy-aware expansion.

        Args:
            query_embedding: Dense query embedding
            target_level: Primary hierarchy level to search
            top_k: Number of results per level
            include_parents: Include parent levels in results
            include_children: Include child levels in results

        Returns:
            Dict mapping hierarchy level names to results
        """
        results = {}

        # Determine which levels to search
        levels_to_search = [target_level]

        if include_parents:
            for level in HierarchyLevel:
                if level.value < target_level.value:
                    levels_to_search.append(level)

        if include_children:
            for level in HierarchyLevel:
                if level.value > target_level.value:
                    levels_to_search.append(level)

        # Search each level
        for level in levels_to_search:
            level_results = await self.search(
                query_embedding,
                top_k=top_k,
                hierarchy_filter=[level],
                use_fusion=True
            )
            results[level.name] = level_results

        return results

    async def get_document_tree(
        self,
        doc_id: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get document with its hierarchy tree.

        Args:
            doc_id: Document ID
            max_depth: Maximum tree depth to traverse

        Returns:
            Tree structure with parent and children
        """
        if doc_id not in self.documents:
            return {}

        doc = self.documents[doc_id]

        # Build tree
        tree = {
            "doc_id": doc.doc_id,
            "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            "hierarchy_level": doc.hierarchy_level.name,
            "depth": doc.depth,
            "parent": None,
            "children": []
        }

        # Get parent
        if doc.parent_id and doc.parent_id in self.documents:
            parent = self.documents[doc.parent_id]
            tree["parent"] = {
                "doc_id": parent.doc_id,
                "content": parent.content[:100] + "...",
                "hierarchy_level": parent.hierarchy_level.name
            }

        # Get children (documents with this as parent)
        children = [
            d for d in self.documents.values()
            if d.parent_id == doc_id
        ]

        for child in children[:10]:  # Limit children
            tree["children"].append({
                "doc_id": child.doc_id,
                "content": child.content[:100] + "...",
                "hierarchy_level": child.hierarchy_level.name,
                "depth": child.depth
            })

        return tree

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        if not self.documents:
            self.load_all_documents()

        level_counts = {}
        depth_sum = 0.0

        for doc in self.documents.values():
            level = doc.hierarchy_level.name
            level_counts[level] = level_counts.get(level, 0) + 1
            depth_sum += doc.depth

        avg_depth = depth_sum / len(self.documents) if self.documents else 0

        return {
            "total_documents": len(self.documents),
            "by_hierarchy_level": level_counts,
            "average_depth": avg_depth,
            "embedding_dim": self.dim,
            "curvature": self.manifold.c,
            "euclidean_weight": self.euclidean_weight,
            "hyperbolic_weight": self.hyperbolic_weight
        }


# Singleton instance
_hyperbolic_retriever: Optional[HyperbolicRetriever] = None


def get_hyperbolic_retriever(
    dim: int = 768,
    euclidean_weight: float = 0.5
) -> HyperbolicRetriever:
    """Get or create hyperbolic retriever singleton."""
    global _hyperbolic_retriever

    if _hyperbolic_retriever is None:
        _hyperbolic_retriever = HyperbolicRetriever(
            dim=dim,
            euclidean_weight=euclidean_weight
        )

    return _hyperbolic_retriever


# Convenience function for hierarchy level detection
def detect_hierarchy_level(content: str, metadata: Dict[str, Any] = None) -> HierarchyLevel:
    """
    Detect hierarchy level from content and metadata.

    Uses heuristics to classify document depth:
    - Very short content with section headers → CHAPTER
    - Error codes with cause/remedy → STEP
    - Procedure steps → STEP
    - General concepts → SECTION
    """
    content_lower = content.lower()
    metadata = metadata or {}

    # Check metadata hints
    if "hierarchy_level" in metadata:
        return HierarchyLevel[metadata["hierarchy_level"].upper()]

    if "type" in metadata:
        type_map = {
            "manual": HierarchyLevel.MANUAL,
            "chapter": HierarchyLevel.CHAPTER,
            "section": HierarchyLevel.SECTION,
            "procedure": HierarchyLevel.PROCEDURE,
            "step": HierarchyLevel.STEP,
            "error_code": HierarchyLevel.STEP,
        }
        if metadata["type"] in type_map:
            return type_map[metadata["type"]]

    # Content-based heuristics
    word_count = len(content.split())

    # Error code pattern (very specific)
    import re
    if re.search(r'\b[A-Z]{3,5}-\d{3}\b', content):
        return HierarchyLevel.STEP

    # Chapter/section indicators
    chapter_patterns = ["chapter", "section", "overview", "introduction"]
    if any(p in content_lower for p in chapter_patterns) and word_count < 500:
        return HierarchyLevel.CHAPTER

    # Procedure indicators
    procedure_patterns = ["procedure", "steps", "how to", "instructions"]
    if any(p in content_lower for p in procedure_patterns):
        return HierarchyLevel.PROCEDURE

    # Default based on length
    if word_count < 100:
        return HierarchyLevel.STEP
    elif word_count < 500:
        return HierarchyLevel.SECTION
    else:
        return HierarchyLevel.CHAPTER
