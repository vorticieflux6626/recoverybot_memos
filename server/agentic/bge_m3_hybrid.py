"""
BGE-M3 Hybrid Retrieval for Agentic Search.

Implements multi-mode retrieval combining:
1. Dense embeddings (semantic similarity via BGE-M3)
2. Sparse embeddings (lexical matching via BM25)
3. Multi-vector (ColBERT-style late interaction - optional)

Based on research:
- BGE-M3: Multilingual embedding with 1024 dimensions
- BM25: Probabilistic lexical matching for exact terms
- ColBERT: Token-level interaction for fine-grained matching

Hybrid Scoring Formula:
    score = α * dense_score + β * sparse_score + γ * multivec_score

Default weights (BGE-M3 paper recommendations):
    α = 0.35 (dense)
    β = 0.35 (sparse)
    γ = 0.30 (multi-vector)

When multi-vector not available:
    α = 0.50 (dense)
    β = 0.50 (sparse)

References:
- Chen et al., "BGE M3-Embedding" (arXiv 2024)
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25" (2009)
- Khattab & Zaharia, "ColBERT" (SIGIR 2020)
"""

import asyncio
import hashlib
import json
import logging
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class RetrievalMode(Enum):
    """Retrieval modes for hybrid search."""
    DENSE_ONLY = "dense_only"          # Semantic only
    SPARSE_ONLY = "sparse_only"        # Lexical only (BM25)
    HYBRID = "hybrid"                  # Dense + Sparse
    FULL_HYBRID = "full_hybrid"        # Dense + Sparse + Multi-vector


@dataclass
class HybridDocument:
    """Document with hybrid embeddings."""
    doc_id: str
    content: str
    dense_embedding: Optional[np.ndarray] = None
    sparse_vector: Optional[Dict[str, float]] = None  # Term -> TF-IDF weight
    token_embeddings: Optional[np.ndarray] = None  # ColBERT-style token embeddings
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "has_dense": self.dense_embedding is not None,
            "has_sparse": self.sparse_vector is not None,
            "has_multivec": self.token_embeddings is not None,
            "metadata": self.metadata
        }


@dataclass
class HybridSearchResult:
    """Search result with hybrid scores."""
    doc_id: str
    content: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    multivec_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "scores": {
                "dense": round(self.dense_score, 4),
                "sparse": round(self.sparse_score, 4),
                "multivec": round(self.multivec_score, 4),
                "combined": round(self.combined_score, 4)
            },
            "metadata": self.metadata
        }


@dataclass
class HybridRetrievalStats:
    """Statistics for hybrid retrieval."""
    documents_indexed: int = 0
    vocabulary_size: int = 0
    avg_doc_length: float = 0.0
    dense_index_size_mb: float = 0.0
    sparse_index_size_mb: float = 0.0
    mode: str = "hybrid"


# =============================================================================
# BM25 Sparse Indexer
# =============================================================================

class BM25Index:
    """
    BM25 sparse index for lexical matching.

    BM25 Formula:
        score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))

    Where:
        - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
        - f(qi, D) = term frequency of qi in document D
        - |D| = length of document D
        - avgdl = average document length
        - k1 = term frequency saturation (default 1.5)
        - b = length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # Document storage
        self.documents: Dict[str, str] = {}  # doc_id -> content
        self.doc_lengths: Dict[str, int] = {}  # doc_id -> length
        self.avg_doc_length: float = 0.0

        # Inverted index: term -> {doc_id -> term_frequency}
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Document frequency: term -> number of documents containing term
        self.doc_freq: Dict[str, int] = defaultdict(int)

        # Vocabulary
        self.vocabulary: Set[str] = set()

        # Pre-computed IDF values
        self.idf_cache: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization with lowercasing and basic cleanup."""
        # Remove special characters, lowercase, split on whitespace
        text = re.sub(r'[^\w\s-]', ' ', text.lower())
        tokens = text.split()
        # Filter short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'here', 'there', 'when', 'where', 'why', 'how', 'all',
                     'each', 'few', 'more', 'most', 'other', 'some', 'such',
                     'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                     'too', 'very', 'and', 'but', 'if', 'or', 'because', 'until',
                     'while', 'this', 'that', 'these', 'those', 'it', 'its'}
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def add_document(self, doc_id: str, content: str):
        """Add a document to the index."""
        tokens = self.tokenize(content)
        self.documents[doc_id] = content
        self.doc_lengths[doc_id] = len(tokens)

        # Count term frequencies
        term_freqs = Counter(tokens)

        # Update inverted index
        for term, freq in term_freqs.items():
            if doc_id not in self.inverted_index[term]:
                self.doc_freq[term] += 1
            self.inverted_index[term][doc_id] = freq
            self.vocabulary.add(term)

        # Update average document length
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / len(self.documents)

        # Invalidate IDF cache
        self.idf_cache.clear()

    def compute_idf(self, term: str) -> float:
        """Compute IDF for a term."""
        if term in self.idf_cache:
            return self.idf_cache[term]

        N = len(self.documents)
        n = self.doc_freq.get(term, 0)

        # BM25 IDF formula with smoothing
        idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
        self.idf_cache[term] = idf
        return idf

    def get_sparse_vector(self, content: str) -> Dict[str, float]:
        """Get sparse TF-IDF vector for content."""
        tokens = self.tokenize(content)
        term_freqs = Counter(tokens)

        sparse_vec = {}
        for term, freq in term_freqs.items():
            if term in self.vocabulary:
                idf = self.compute_idf(term)
                # Use BM25-style TF-IDF
                tf = freq / len(tokens) if tokens else 0
                sparse_vec[term] = tf * idf

        return sparse_vec

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25 scoring.

        Returns list of (doc_id, score) tuples.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            idf = self.compute_idf(term)

            for doc_id, tf in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_id]

                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)

                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "documents": len(self.documents),
            "vocabulary_size": len(self.vocabulary),
            "avg_doc_length": round(self.avg_doc_length, 2),
            "k1": self.k1,
            "b": self.b
        }


# =============================================================================
# BGE-M3 Hybrid Retriever
# =============================================================================

class BGEM3HybridRetriever:
    """
    Hybrid retriever combining BGE-M3 dense embeddings with BM25 sparse matching.

    Architecture:
    1. Dense Stage: Semantic similarity using BGE-M3 embeddings (1024d)
    2. Sparse Stage: Lexical matching using BM25
    3. Fusion: RRF (Reciprocal Rank Fusion) or weighted linear combination

    Memory Efficiency:
    - SQLite-backed persistence for large corpora
    - Lazy loading of embeddings
    - Configurable caching
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        db_path: Optional[str] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        multivec_weight: float = 0.0,  # Optional ColBERT component
        use_rrf: bool = True  # Use Reciprocal Rank Fusion
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.multivec_weight = multivec_weight
        self.use_rrf = use_rrf

        # Normalize weights
        total = dense_weight + sparse_weight + multivec_weight
        if total > 0:
            self.dense_weight /= total
            self.sparse_weight /= total
            self.multivec_weight /= total

        # BM25 sparse index
        self.bm25_index = BM25Index()

        # Dense embedding storage
        self.dense_embeddings: Dict[str, np.ndarray] = {}

        # Document content
        self.documents: Dict[str, HybridDocument] = {}

        # Database path for persistence
        self.db_path = db_path or "/tmp/bge_m3_hybrid.db"
        self._init_db()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Dimension detection
        self._embedding_dim: Optional[int] = None

    def _init_db(self):
        """Initialize SQLite database for persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content TEXT,
                dense_embedding BLOB,
                sparse_vector TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_stats (
                stat_key TEXT PRIMARY KEY,
                stat_value TEXT
            )
        """)

        conn.commit()
        conn.close()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get dense embedding from BGE-M3 via Ollama."""
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["embedding"], dtype=np.float32)

            # Cache dimension
            if self._embedding_dim is None:
                self._embedding_dim = len(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            # Return zero vector as fallback
            dim = self._embedding_dim or 1024
            return np.zeros(dim, dtype=np.float32)

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HybridDocument:
        """
        Add a document to the hybrid index.

        Creates:
        1. Dense embedding via BGE-M3
        2. Sparse vector via BM25 tokenization
        """
        # Get dense embedding
        dense_emb = await self.get_embedding(content)

        # Add to BM25 index and get sparse vector
        self.bm25_index.add_document(doc_id, content)
        sparse_vec = self.bm25_index.get_sparse_vector(content)

        # Create document
        doc = HybridDocument(
            doc_id=doc_id,
            content=content,
            dense_embedding=dense_emb,
            sparse_vector=sparse_vec,
            metadata=metadata or {}
        )

        # Store in memory
        self.documents[doc_id] = doc
        self.dense_embeddings[doc_id] = dense_emb

        # Persist to SQLite
        self._persist_document(doc)

        logger.info(f"Indexed document {doc_id}: dense={len(dense_emb)}d, sparse={len(sparse_vec)} terms")
        return doc

    def _persist_document(self, doc: HybridDocument):
        """Persist document to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        dense_blob = doc.dense_embedding.tobytes() if doc.dense_embedding is not None else None
        sparse_json = json.dumps(doc.sparse_vector) if doc.sparse_vector else None
        metadata_json = json.dumps(doc.metadata)

        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (doc_id, content, dense_embedding, sparse_vector, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (doc.doc_id, doc.content, dense_blob, sparse_json, metadata_json))

        conn.commit()
        conn.close()

    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[HybridDocument]:
        """
        Batch add documents to the index.

        Args:
            documents: List of {"doc_id": str, "content": str, "metadata": dict}
        """
        results = []
        for doc_data in documents:
            doc = await self.add_document(
                doc_id=doc_data["doc_id"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
            results.append(doc)
        return results

    async def search(
        self,
        query: str,
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        dense_candidates: int = 100,
        sparse_candidates: int = 100
    ) -> List[HybridSearchResult]:
        """
        Hybrid search combining dense and sparse retrieval.

        Pipeline:
        1. Dense retrieval: Top-N by cosine similarity
        2. Sparse retrieval: Top-N by BM25 score
        3. Fusion: RRF or weighted combination

        Args:
            query: Search query
            top_k: Number of results to return
            mode: Retrieval mode (dense_only, sparse_only, hybrid)
            dense_candidates: Number of dense candidates
            sparse_candidates: Number of sparse candidates
        """
        if not self.documents:
            return []

        results: Dict[str, HybridSearchResult] = {}

        # Dense retrieval
        if mode in [RetrievalMode.DENSE_ONLY, RetrievalMode.HYBRID, RetrievalMode.FULL_HYBRID]:
            query_embedding = await self.get_embedding(query)
            dense_results = self._dense_search(query_embedding, dense_candidates)

            for doc_id, score in dense_results:
                if doc_id not in results:
                    doc = self.documents[doc_id]
                    results[doc_id] = HybridSearchResult(
                        doc_id=doc_id,
                        content=doc.content,
                        metadata=doc.metadata
                    )
                results[doc_id].dense_score = score

        # Sparse retrieval
        if mode in [RetrievalMode.SPARSE_ONLY, RetrievalMode.HYBRID, RetrievalMode.FULL_HYBRID]:
            sparse_results = self.bm25_index.search(query, sparse_candidates)

            # Normalize BM25 scores to 0-1 range
            if sparse_results:
                max_score = max(s for _, s in sparse_results)
                if max_score > 0:
                    sparse_results = [(d, s / max_score) for d, s in sparse_results]

            for doc_id, score in sparse_results:
                if doc_id not in results:
                    doc = self.documents.get(doc_id)
                    if doc:
                        results[doc_id] = HybridSearchResult(
                            doc_id=doc_id,
                            content=doc.content,
                            metadata=doc.metadata
                        )
                if doc_id in results:
                    results[doc_id].sparse_score = score

        # Fusion
        if self.use_rrf:
            # Reciprocal Rank Fusion
            self._apply_rrf(results)
        else:
            # Weighted linear combination
            for result in results.values():
                result.combined_score = (
                    self.dense_weight * result.dense_score +
                    self.sparse_weight * result.sparse_score +
                    self.multivec_weight * result.multivec_score
                )

        # Sort and return top_k
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.combined_score,
            reverse=True
        )

        return sorted_results[:top_k]

    def _dense_search(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Search using dense embeddings (cosine similarity)."""
        if not self.dense_embeddings:
            return []

        scores = []
        query_norm = np.linalg.norm(query_embedding)

        for doc_id, doc_embedding in self.dense_embeddings.items():
            # Cosine similarity
            doc_norm = np.linalg.norm(doc_embedding)
            if query_norm > 0 and doc_norm > 0:
                similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
            else:
                similarity = 0.0
            scores.append((doc_id, float(similarity)))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _apply_rrf(self, results: Dict[str, HybridSearchResult], k: int = 60):
        """
        Apply Reciprocal Rank Fusion.

        RRF Formula:
            RRF(d) = Σ 1 / (k + rank_i(d))

        Where k is a constant (typically 60) and rank_i is the rank in list i.
        """
        # Get rankings
        dense_ranked = sorted(
            [(r.doc_id, r.dense_score) for r in results.values()],
            key=lambda x: x[1], reverse=True
        )
        sparse_ranked = sorted(
            [(r.doc_id, r.sparse_score) for r in results.values()],
            key=lambda x: x[1], reverse=True
        )

        # Create rank maps
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_ranked)}
        sparse_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_ranked)}

        # Compute RRF scores
        for doc_id, result in results.items():
            dense_rank = dense_ranks.get(doc_id, len(results) + 1)
            sparse_rank = sparse_ranks.get(doc_id, len(results) + 1)

            rrf_score = 0.0
            if result.dense_score > 0:
                rrf_score += self.dense_weight / (k + dense_rank)
            if result.sparse_score > 0:
                rrf_score += self.sparse_weight / (k + sparse_rank)

            result.combined_score = rrf_score

    async def load_from_db(self):
        """Load documents from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT doc_id, content, dense_embedding, sparse_vector, metadata FROM documents")
        rows = cursor.fetchall()

        for row in rows:
            doc_id, content, dense_blob, sparse_json, metadata_json = row

            # Reconstruct dense embedding
            dense_emb = None
            if dense_blob:
                dim = self._embedding_dim or 1024
                dense_emb = np.frombuffer(dense_blob, dtype=np.float32)
                if len(dense_emb) != dim:
                    dense_emb = np.resize(dense_emb, dim)

            # Reconstruct sparse vector
            sparse_vec = json.loads(sparse_json) if sparse_json else None

            # Reconstruct metadata
            metadata = json.loads(metadata_json) if metadata_json else {}

            # Create document
            doc = HybridDocument(
                doc_id=doc_id,
                content=content,
                dense_embedding=dense_emb,
                sparse_vector=sparse_vec,
                metadata=metadata
            )

            self.documents[doc_id] = doc
            if dense_emb is not None:
                self.dense_embeddings[doc_id] = dense_emb

            # Rebuild BM25 index
            self.bm25_index.add_document(doc_id, content)

        conn.close()
        logger.info(f"Loaded {len(self.documents)} documents from database")

    def get_stats(self) -> HybridRetrievalStats:
        """Get retrieval statistics."""
        dense_size = sum(
            emb.nbytes for emb in self.dense_embeddings.values()
        ) / (1024 * 1024)  # MB

        sparse_size = sum(
            len(json.dumps(doc.sparse_vector))
            for doc in self.documents.values()
            if doc.sparse_vector
        ) / (1024 * 1024)  # MB

        return HybridRetrievalStats(
            documents_indexed=len(self.documents),
            vocabulary_size=len(self.bm25_index.vocabulary),
            avg_doc_length=self.bm25_index.avg_doc_length,
            dense_index_size_mb=round(dense_size, 2),
            sparse_index_size_mb=round(sparse_size, 4),
            mode="hybrid" if self.sparse_weight > 0 else "dense_only"
        )

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# ColBERT-Style Multi-Vector (Optional)
# =============================================================================

class ColBERTReranker:
    """
    ColBERT-style late interaction reranker.

    Computes MaxSim: For each query token, find max similarity to any doc token.
    Then sum these MaxSims for the final score.

    Note: This requires token-level embeddings which BGE-M3 supports in the
    original implementation but not through Ollama's embedding API.

    For now, this is a placeholder that can be enabled when using BGE-M3
    directly via the sentence-transformers or FlagEmbedding library.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    async def rerank(
        self,
        query: str,
        query_tokens: List[np.ndarray],
        candidates: List[Tuple[str, List[np.ndarray]]]  # (doc_id, token_embeddings)
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates using MaxSim scoring.

        Returns (doc_id, score) sorted by score descending.
        """
        if not self.enabled:
            return []

        scores = []

        for doc_id, doc_tokens in candidates:
            # MaxSim: for each query token, find max similarity to any doc token
            max_sims = []
            for q_token in query_tokens:
                max_sim = 0.0
                for d_token in doc_tokens:
                    sim = np.dot(q_token, d_token) / (
                        np.linalg.norm(q_token) * np.linalg.norm(d_token) + 1e-8
                    )
                    max_sim = max(max_sim, sim)
                max_sims.append(max_sim)

            # Sum of MaxSims
            score = sum(max_sims)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# =============================================================================
# Singleton and Factory
# =============================================================================

_hybrid_retriever: Optional[BGEM3HybridRetriever] = None


def get_hybrid_retriever(
    ollama_url: str = "http://localhost:11434",
    model: str = "bge-m3",
    db_path: Optional[str] = None
) -> BGEM3HybridRetriever:
    """Get or create the global hybrid retriever instance."""
    global _hybrid_retriever

    if _hybrid_retriever is None:
        _hybrid_retriever = BGEM3HybridRetriever(
            ollama_url=ollama_url,
            model=model,
            db_path=db_path
        )

    return _hybrid_retriever


async def create_hybrid_retriever(
    ollama_url: str = "http://localhost:11434",
    model: str = "bge-m3",
    db_path: Optional[str] = None,
    load_existing: bool = True
) -> BGEM3HybridRetriever:
    """Create a new hybrid retriever, optionally loading existing data."""
    retriever = BGEM3HybridRetriever(
        ollama_url=ollama_url,
        model=model,
        db_path=db_path
    )

    if load_existing:
        await retriever.load_from_db()

    return retriever
