"""
BGE-M3 Hybrid Retrieval for Agentic Search.

Implements multi-mode retrieval combining:
1. Dense embeddings (semantic similarity via BGE-M3)
2. Sparse embeddings (lexical matching via BM25)
3. Multi-vector (ColBERT-style late interaction)

Based on research:
- BGE-M3: Multilingual embedding with 1024 dimensions
- BM25: Probabilistic lexical matching for exact terms
- ColBERT: Token-level interaction for fine-grained matching

Hybrid Scoring Formula:
    score = α * dense_score + β * sparse_score + γ * multivec_score

Default weights (BGE-M3 paper recommendations):
    α = 0.40 (dense)
    β = 0.30 (sparse/lexical)
    γ = 0.30 (multi-vector/ColBERT)

When multi-vector not available:
    α = 0.50 (dense)
    β = 0.50 (sparse)

References:
- Chen et al., "BGE M3-Embedding" (arXiv 2024)
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25" (2009)
- Khattab & Zaharia, "ColBERT" (SIGIR 2020)

Updated: December 2025 - Added FlagEmbedding support for ColBERT vectors (G.1.1)
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

# Try to import FlagEmbedding for native BGE-M3 with ColBERT support
_FLAG_EMBEDDING_AVAILABLE = False
_BGE_M3_MODEL = None

try:
    from FlagEmbedding import BGEM3FlagModel
    _FLAG_EMBEDDING_AVAILABLE = True
    logger.info("FlagEmbedding available - ColBERT mode enabled")
except ImportError:
    logger.warning("FlagEmbedding not available - ColBERT mode disabled, using Ollama fallback")


def get_bge_m3_model(use_fp16: bool = True) -> Optional['BGEM3FlagModel']:
    """
    Get or create the global BGE-M3 model instance.

    Uses FP16 for memory efficiency (~1.2GB VRAM vs 2.4GB for FP32).
    Enables return_colbert_vecs by default.

    Args:
        use_fp16: Whether to use FP16 precision (default: True, saves 50% VRAM)

    Returns:
        BGEM3FlagModel instance or None if FlagEmbedding not available
    """
    global _BGE_M3_MODEL

    if not _FLAG_EMBEDDING_AVAILABLE:
        return None

    if _BGE_M3_MODEL is None:
        logger.info("Loading BGE-M3 model with ColBERT support (use_fp16=%s)...", use_fp16)
        try:
            _BGE_M3_MODEL = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=use_fp16
            )
            logger.info("BGE-M3 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            return None

    return _BGE_M3_MODEL


@dataclass
class BGEM3Embeddings:
    """
    Container for all BGE-M3 embedding types.

    BGE-M3 produces three types of embeddings:
    1. Dense: 1024-dimensional dense vector for semantic similarity
    2. Lexical (Sparse): Term weights for BM25-style lexical matching
    3. ColBERT: Token-level embeddings for MaxSim late interaction
    """
    dense: np.ndarray  # Shape: (1024,)
    lexical: Optional[Dict[str, float]] = None  # Term -> weight
    colbert: Optional[np.ndarray] = None  # Shape: (num_tokens, 1024)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dense_dim": len(self.dense) if self.dense is not None else 0,
            "lexical_terms": len(self.lexical) if self.lexical else 0,
            "colbert_tokens": self.colbert.shape[0] if self.colbert is not None else 0
        }


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
    """Statistics for hybrid retrieval (G.1.1 update: includes ColBERT)."""
    documents_indexed: int = 0
    vocabulary_size: int = 0
    avg_doc_length: float = 0.0
    dense_index_size_mb: float = 0.0
    sparse_index_size_mb: float = 0.0
    colbert_index_size_mb: float = 0.0  # G.1.1
    mode: str = "hybrid"
    colbert_enabled: bool = False  # G.1.1
    weights: Dict[str, float] = field(default_factory=dict)  # G.1.1


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

    def save_to_db(self, conn: sqlite3.Connection):
        """
        Persist BM25 index structures to SQLite for fast reload.

        Stores:
        - Inverted index as JSON
        - Document frequencies
        - Vocabulary
        - Index statistics
        """
        cursor = conn.cursor()

        # Create BM25 index table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bm25_index (
                term TEXT PRIMARY KEY,
                doc_postings TEXT,  -- JSON: {doc_id: term_freq}
                doc_freq INTEGER
            )
        """)

        # Store each term's posting list
        for term in self.vocabulary:
            postings = dict(self.inverted_index.get(term, {}))
            cursor.execute(
                "INSERT OR REPLACE INTO bm25_index (term, doc_postings, doc_freq) VALUES (?, ?, ?)",
                (term, json.dumps(postings), self.doc_freq.get(term, 0))
            )

        # Store index stats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bm25_stats (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        stats = {
            "avg_doc_length": str(self.avg_doc_length),
            "k1": str(self.k1),
            "b": str(self.b),
            "doc_count": str(len(self.documents))
        }

        for key, value in stats.items():
            cursor.execute(
                "INSERT OR REPLACE INTO bm25_stats (key, value) VALUES (?, ?)",
                (key, value)
            )

        conn.commit()
        logger.info(f"Saved BM25 index: {len(self.vocabulary)} terms, {len(self.documents)} docs")

    def load_from_db(self, conn: sqlite3.Connection) -> bool:
        """
        Load BM25 index from SQLite for fast startup.

        Returns True if successfully loaded, False if no saved index exists.
        """
        cursor = conn.cursor()

        # Check if BM25 index table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bm25_index'"
        )
        if not cursor.fetchone():
            return False

        # Load inverted index
        cursor.execute("SELECT term, doc_postings, doc_freq FROM bm25_index")
        rows = cursor.fetchall()

        if not rows:
            return False

        for term, postings_json, doc_freq in rows:
            postings = json.loads(postings_json)
            self.inverted_index[term] = {k: int(v) for k, v in postings.items()}
            self.doc_freq[term] = doc_freq
            self.vocabulary.add(term)

        # Load stats
        cursor.execute("SELECT key, value FROM bm25_stats")
        for key, value in cursor.fetchall():
            if key == "avg_doc_length":
                self.avg_doc_length = float(value)
            elif key == "k1":
                self.k1 = float(value)
            elif key == "b":
                self.b = float(value)

        # Note: documents and doc_lengths are loaded separately from main documents table
        logger.info(f"Loaded BM25 index: {len(self.vocabulary)} terms")
        return True


# =============================================================================
# BGE-M3 Hybrid Retriever
# =============================================================================

class BGEM3HybridRetriever:
    """
    Hybrid retriever combining BGE-M3 dense embeddings with BM25 sparse matching.

    Architecture:
    1. Dense Stage: Semantic similarity using BGE-M3 embeddings (1024d)
    2. Sparse Stage: Lexical matching using BM25 OR BGE-M3 lexical weights
    3. ColBERT Stage: Token-level MaxSim for fine-grained matching (optional)
    4. Fusion: RRF (Reciprocal Rank Fusion) or weighted linear combination

    Memory Efficiency:
    - SQLite-backed persistence for large corpora
    - Lazy loading of embeddings
    - FP16 mode for 50% VRAM reduction
    - Configurable caching

    Updated (G.1.1): FlagEmbedding support with ColBERT mode enabled.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "bge-m3",
        db_path: Optional[str] = None,
        dense_weight: float = 0.40,  # Updated per BGE-M3 paper
        sparse_weight: float = 0.30,
        multivec_weight: float = 0.30,  # ColBERT component enabled by default
        use_rrf: bool = True,  # Use Reciprocal Rank Fusion
        use_flag_embedding: bool = True,  # Use FlagEmbedding if available (G.1.1)
        use_fp16: bool = True  # Use FP16 for efficiency (G.1.1)
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.use_flag_embedding = use_flag_embedding and _FLAG_EMBEDDING_AVAILABLE
        self.use_fp16 = use_fp16
        self.use_rrf = use_rrf

        # Store original weights
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.multivec_weight = multivec_weight

        # Adjust weights if ColBERT not available
        if not self.use_flag_embedding and multivec_weight > 0:
            # Redistribute ColBERT weight to dense and sparse
            logger.info("ColBERT not available, redistributing weight to dense+sparse")
            self.dense_weight = 0.50
            self.sparse_weight = 0.50
            self.multivec_weight = 0.0

        # Normalize weights
        total = self.dense_weight + self.sparse_weight + self.multivec_weight
        if total > 0:
            self.dense_weight /= total
            self.sparse_weight /= total
            self.multivec_weight /= total

        # BM25 sparse index (used as fallback or complement)
        self.bm25_index = BM25Index()

        # Dense embedding storage
        self.dense_embeddings: Dict[str, np.ndarray] = {}

        # ColBERT token embeddings storage (G.1.1)
        self.colbert_embeddings: Dict[str, np.ndarray] = {}

        # Document content
        self.documents: Dict[str, HybridDocument] = {}

        # Database path for persistence
        self.db_path = db_path or "/tmp/bge_m3_hybrid.db"
        self._init_db()

        # HTTP client (for Ollama fallback)
        self._client: Optional[httpx.AsyncClient] = None

        # Dimension detection
        self._embedding_dim: Optional[int] = None

        # BGE-M3 model (lazy loaded)
        self._bge_m3_model = None

        # Stats
        self._colbert_enabled = self.use_flag_embedding
        logger.info(
            f"BGEM3HybridRetriever initialized: dense={self.dense_weight:.2f}, "
            f"sparse={self.sparse_weight:.2f}, colbert={self.multivec_weight:.2f}, "
            f"flag_embedding={self.use_flag_embedding}, fp16={self.use_fp16}"
        )

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

        # Try to load persisted BM25 index for fast startup
        if self.bm25_index.load_from_db(conn):
            logger.info("BM25 index loaded from persistence")
        else:
            logger.info("No persisted BM25 index found, will rebuild on document load")

        conn.close()

    def _get_bge_m3_model(self):
        """Get or create the BGE-M3 model instance (lazy loading)."""
        if self._bge_m3_model is None and self.use_flag_embedding:
            self._bge_m3_model = get_bge_m3_model(use_fp16=self.use_fp16)
        return self._bge_m3_model

    def encode_with_colbert(
        self,
        texts: List[str],
        batch_size: int = 12,
        max_length: int = 8192
    ) -> List[BGEM3Embeddings]:
        """
        Encode texts using FlagEmbedding BGE-M3 with ColBERT support.

        This is the core method for G.1.1 - enables all three embedding types:
        - Dense (1024d): Semantic similarity
        - Lexical: Token weights for BM25-style matching
        - ColBERT: Token-level embeddings for MaxSim

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            max_length: Maximum sequence length

        Returns:
            List of BGEM3Embeddings containing all three embedding types
        """
        model = self._get_bge_m3_model()

        if model is None:
            logger.warning("FlagEmbedding not available, falling back to Ollama-only embeddings")
            return []

        try:
            # Encode with all three modes (G.1.1 key change)
            output = model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True  # G.1.1: Enable ColBERT
            )

            results = []
            for i in range(len(texts)):
                # Dense embedding
                dense = np.array(output['dense_vecs'][i], dtype=np.float32)

                # Lexical weights (sparse)
                lexical = None
                if 'lexical_weights' in output and i < len(output['lexical_weights']):
                    lexical = output['lexical_weights'][i]

                # ColBERT token embeddings
                colbert = None
                if 'colbert_vecs' in output and i < len(output['colbert_vecs']):
                    colbert = np.array(output['colbert_vecs'][i], dtype=np.float32)

                results.append(BGEM3Embeddings(
                    dense=dense,
                    lexical=lexical,
                    colbert=colbert
                ))

            logger.debug(f"Encoded {len(texts)} texts with ColBERT: dense={results[0].dense.shape if results else 0}, colbert={results[0].colbert.shape if results and results[0].colbert is not None else 0}")
            return results

        except Exception as e:
            logger.error(f"FlagEmbedding encoding failed: {e}")
            return []

    def compute_colbert_score(
        self,
        query_colbert: np.ndarray,
        doc_colbert: np.ndarray
    ) -> float:
        """
        Compute ColBERT MaxSim score between query and document.

        MaxSim: For each query token, find max similarity to any doc token.
        Final score is the sum of these max similarities.

        Args:
            query_colbert: Query token embeddings, shape (num_query_tokens, 1024)
            doc_colbert: Document token embeddings, shape (num_doc_tokens, 1024)

        Returns:
            MaxSim score (higher = more relevant)
        """
        if query_colbert is None or doc_colbert is None:
            return 0.0

        if len(query_colbert) == 0 or len(doc_colbert) == 0:
            return 0.0

        # Normalize embeddings
        query_norm = query_colbert / (np.linalg.norm(query_colbert, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_colbert / (np.linalg.norm(doc_colbert, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
        sim_matrix = np.dot(query_norm, doc_norm.T)

        # MaxSim: max similarity for each query token
        max_sims = np.max(sim_matrix, axis=1)

        # Sum of max similarities
        score = float(np.sum(max_sims))

        # Normalize by query length for comparability
        score = score / len(query_colbert)

        return score

    # ===== PATTERNS FOR RETRIEVAL MODE SELECTION =====
    # Patterns that indicate exact-match queries (prefer sparse/BM25)
    EXACT_MATCH_PATTERNS = [
        # Error codes
        r"[A-Z]{2,5}-\d{3,4}",  # SRVO-063, MOTN-023
        r"[A-Z]{4}\d{4}",       # ABCD1234 style codes
        r"\bERR[-_]?\d+\b",     # ERR-123, ERR123

        # Part numbers
        r"[A-Z]\d{2}[A-Z]-\d{4,5}",  # A06B-0001
        r"\b[A-Z]{2,3}\d{4,6}\b",     # ABC12345

        # Version numbers
        r"\bv?\d+\.\d+\.\d+\b",       # v1.2.3, 1.2.3

        # Model numbers
        r"\bR-30i[AB]\b",            # FANUC controllers
        r"\bM-\d{2,4}i[ABC]\b",      # FANUC robots
        r"\bMC[3-6]\b",              # KraussMaffei controllers

        # Parameters
        r"\$[A-Z_]+\[\d+\]",         # $PARAM[1]
        r"\$[A-Z_]+\.[A-Z_]+",       # $MCR.ALARM
    ]

    # Patterns that indicate conceptual/semantic queries (prefer dense)
    SEMANTIC_PATTERNS = [
        r"\bhow\s+to\b",
        r"\bwhat\s+is\b",
        r"\bwhy\s+does\b",
        r"\bexplain\b",
        r"\bdifference\s+between\b",
        r"\bcompare\b",
        r"\bbest\s+practice",
        r"\brecommend",
        r"\balternative",
        r"\bsimilar\s+to\b",
    ]

    # ===== RRF K PARAMETERS BY DOMAIN TYPE =====
    # Lower k = more weight to top-ranked documents
    # Higher k = more uniform weighting across ranks
    # Domain-specific tuning based on result distribution characteristics
    RRF_K_BY_DOMAIN = {
        # Technical domains: Prefer strong top matches (lower k)
        "fanuc": 40,           # Error codes need exact matches
        "imm": 45,             # Machine codes need precision
        "error_code": 35,      # Highest precision for exact codes
        "part_number": 35,     # Part numbers need exact matches

        # Academic domains: More uniform weighting (higher k)
        "academic": 70,        # Research needs diverse sources
        "research": 70,        # Multiple viewpoints valuable

        # Technical docs: Balanced
        "technical": 55,       # Mix of exact and conceptual
        "qa": 50,              # Q&A benefits from diversity

        # Default
        "general": 60,
        "default": 60,
    }

    def select_retrieval_mode(
        self,
        query: str,
        query_type: Optional[str] = None
    ) -> RetrievalMode:
        """
        Select optimal retrieval mode based on query characteristics.

        Args:
            query: The search query
            query_type: Optional pre-classified query type

        Returns:
            RetrievalMode appropriate for the query

        Rules:
        1. Exact-match patterns (error codes, part numbers) → SPARSE_ONLY
        2. Semantic patterns (how to, explain, compare) → DENSE_ONLY
        3. Technical queries with specific terms → HYBRID
        4. Default → HYBRID
        """
        query_lower = query.lower()

        # Check for exact-match patterns first
        for pattern in self.EXACT_MATCH_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.info(f"Retrieval mode: SPARSE_ONLY (matched exact pattern)")
                return RetrievalMode.SPARSE_ONLY

        # Query type hints
        if query_type:
            query_type_lower = query_type.lower()
            if query_type_lower in ["error_code", "part_number", "exact_match", "fanuc", "imm"]:
                # Even with query type hint, check if there's a semantic pattern
                has_semantic = any(
                    re.search(p, query_lower) for p in self.SEMANTIC_PATTERNS
                )
                if has_semantic:
                    logger.info(f"Retrieval mode: HYBRID (exact type + semantic question)")
                    return RetrievalMode.HYBRID
                logger.info(f"Retrieval mode: SPARSE_ONLY (query_type={query_type})")
                return RetrievalMode.SPARSE_ONLY

        # Check for semantic patterns
        for pattern in self.SEMANTIC_PATTERNS:
            if re.search(pattern, query_lower):
                logger.info(f"Retrieval mode: DENSE_ONLY (matched semantic pattern)")
                return RetrievalMode.DENSE_ONLY

        # Default to hybrid
        logger.info(f"Retrieval mode: HYBRID (default)")
        return RetrievalMode.HYBRID

    def get_rrf_k(self, query_type: Optional[str] = None) -> int:
        """
        Get domain-appropriate RRF k parameter.

        Args:
            query_type: Query type/domain for k selection

        Returns:
            RRF k parameter (lower = more top-heavy ranking)
        """
        if not query_type:
            return self.RRF_K_BY_DOMAIN["default"]

        query_type_lower = query_type.lower()
        k = self.RRF_K_BY_DOMAIN.get(query_type_lower, self.RRF_K_BY_DOMAIN["default"])
        logger.debug(f"RRF k={k} for query_type={query_type}")
        return k

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

        Creates (G.1.1 update):
        1. Dense embedding via BGE-M3 (1024d)
        2. Sparse vector via BM25 tokenization OR BGE-M3 lexical weights
        3. ColBERT token embeddings for MaxSim (if FlagEmbedding available)
        """
        dense_emb = None
        sparse_vec = None
        colbert_emb = None

        # Try FlagEmbedding first (G.1.1)
        if self.use_flag_embedding:
            embeddings = self.encode_with_colbert([content])
            if embeddings:
                dense_emb = embeddings[0].dense
                colbert_emb = embeddings[0].colbert
                # Use BGE-M3 lexical weights if available
                if embeddings[0].lexical:
                    sparse_vec = embeddings[0].lexical
                    logger.debug(f"Using BGE-M3 lexical weights: {len(sparse_vec)} terms")

        # Fallback to Ollama for dense embedding
        if dense_emb is None:
            dense_emb = await self.get_embedding(content)

        # Always maintain BM25 index for compatibility
        self.bm25_index.add_document(doc_id, content)

        # Use BM25 sparse vector if BGE-M3 lexical not available
        if sparse_vec is None:
            sparse_vec = self.bm25_index.get_sparse_vector(content)

        # Create document
        doc = HybridDocument(
            doc_id=doc_id,
            content=content,
            dense_embedding=dense_emb,
            sparse_vector=sparse_vec,
            token_embeddings=colbert_emb,  # Store ColBERT embeddings
            metadata=metadata or {}
        )

        # Store in memory
        self.documents[doc_id] = doc
        self.dense_embeddings[doc_id] = dense_emb

        # Store ColBERT embeddings (G.1.1)
        if colbert_emb is not None:
            self.colbert_embeddings[doc_id] = colbert_emb

        # Persist to SQLite
        self._persist_document(doc)

        colbert_info = f", colbert={colbert_emb.shape}" if colbert_emb is not None else ""
        logger.info(f"Indexed document {doc_id}: dense={len(dense_emb)}d, sparse={len(sparse_vec)} terms{colbert_info}")
        return doc

    def _persist_document(self, doc: HybridDocument):
        """Persist document to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        dense_blob = doc.dense_embedding.tobytes() if doc.dense_embedding is not None else None
        # Convert numpy float16/32 to Python float for JSON serialization
        sparse_vec_native = None
        if doc.sparse_vector:
            sparse_vec_native = {k: float(v) for k, v in doc.sparse_vector.items()}
        sparse_json = json.dumps(sparse_vec_native) if sparse_vec_native else None
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

        # Persist BM25 index after batch addition
        self.save_bm25_index()

        return results

    def save_bm25_index(self):
        """Persist BM25 index to SQLite for fast reload on restart."""
        conn = sqlite3.connect(self.db_path)
        try:
            self.bm25_index.save_to_db(conn)
        finally:
            conn.close()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        dense_candidates: int = 100,
        sparse_candidates: int = 100,
        query_type: Optional[str] = None
    ) -> List[HybridSearchResult]:
        """
        Hybrid search combining dense, sparse, and ColBERT retrieval.

        Pipeline (G.1.1 update):
        1. Dense retrieval: Top-N by cosine similarity
        2. Sparse retrieval: Top-N by BM25 score
        3. ColBERT retrieval: MaxSim scoring (if FlagEmbedding available)
        4. Fusion: RRF or weighted combination

        Args:
            query: Search query
            top_k: Number of results to return
            mode: Retrieval mode (dense_only, sparse_only, hybrid, full_hybrid)
            dense_candidates: Number of dense candidates
            sparse_candidates: Number of sparse candidates
            query_type: Optional query type for domain-specific RRF tuning
        """
        if not self.documents:
            return []

        results: Dict[str, HybridSearchResult] = {}
        query_colbert = None  # For ColBERT scoring

        # Get query embeddings - use FlagEmbedding if available (G.1.1)
        if self.use_flag_embedding and mode == RetrievalMode.FULL_HYBRID:
            query_embeddings = self.encode_with_colbert([query])
            if query_embeddings:
                query_embedding = query_embeddings[0].dense
                query_colbert = query_embeddings[0].colbert
                logger.debug(f"Query ColBERT shape: {query_colbert.shape if query_colbert is not None else None}")
            else:
                query_embedding = await self.get_embedding(query)
        else:
            query_embedding = await self.get_embedding(query)

        # Dense retrieval
        if mode in [RetrievalMode.DENSE_ONLY, RetrievalMode.HYBRID, RetrievalMode.FULL_HYBRID]:
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

        # ColBERT retrieval (G.1.1)
        if mode == RetrievalMode.FULL_HYBRID and query_colbert is not None and self.colbert_embeddings:
            logger.info(f"Computing ColBERT MaxSim scores for {len(results)} candidates")
            colbert_scores = []

            for doc_id in results:
                doc_colbert = self.colbert_embeddings.get(doc_id)
                if doc_colbert is not None:
                    score = self.compute_colbert_score(query_colbert, doc_colbert)
                    results[doc_id].multivec_score = score
                    colbert_scores.append(score)

            # Normalize ColBERT scores to 0-1 range
            if colbert_scores:
                max_colbert = max(colbert_scores) if colbert_scores else 1.0
                if max_colbert > 0:
                    for doc_id in results:
                        if results[doc_id].multivec_score > 0:
                            results[doc_id].multivec_score /= max_colbert

        # Fusion
        if self.use_rrf:
            # Reciprocal Rank Fusion with domain-specific k
            rrf_k = self.get_rrf_k(query_type)
            colbert_info = f", colbert={len(self.colbert_embeddings)}" if mode == RetrievalMode.FULL_HYBRID else ""
            logger.info(f"Using RRF with k={rrf_k} for query_type={query_type or 'default'}{colbert_info}")
            self._apply_rrf(results, k=rrf_k, include_colbert=(mode == RetrievalMode.FULL_HYBRID))
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

        # Convert all numpy float types to Python floats for JSON serialization
        for result in sorted_results:
            result.dense_score = float(result.dense_score)
            result.sparse_score = float(result.sparse_score)
            result.multivec_score = float(result.multivec_score)
            result.combined_score = float(result.combined_score)

        return sorted_results[:top_k]

    async def search_with_auto_mode(
        self,
        query: str,
        top_k: int = 10,
        query_type: Optional[str] = None,
        dense_candidates: int = 100,
        sparse_candidates: int = 100
    ) -> List[HybridSearchResult]:
        """
        Search with automatic retrieval mode selection.

        Automatically selects SPARSE_ONLY, DENSE_ONLY, or HYBRID mode
        based on query characteristics.

        Args:
            query: Search query
            top_k: Number of results to return
            query_type: Optional pre-classified query type for hints
            dense_candidates: Number of dense candidates (for hybrid/dense)
            sparse_candidates: Number of sparse candidates (for hybrid/sparse)

        Returns:
            List of search results
        """
        # Auto-select retrieval mode
        mode = self.select_retrieval_mode(query, query_type)

        # Log mode selection
        logger.info(f"Auto-selected retrieval mode: {mode.value} for query: {query[:50]}...")

        # Execute search with selected mode and domain-specific RRF
        return await self.search(
            query=query,
            top_k=top_k,
            mode=mode,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            query_type=query_type
        )

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

    def _apply_rrf(
        self,
        results: Dict[str, HybridSearchResult],
        k: int = 60,
        include_colbert: bool = False
    ):
        """
        Apply Reciprocal Rank Fusion (G.1.1 update: includes ColBERT).

        RRF Formula:
            RRF(d) = Σ weight_i / (k + rank_i(d))

        Where k is a constant (typically 60) and rank_i is the rank in list i.
        Weights: dense=0.40, sparse=0.30, colbert=0.30 (when enabled)
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

        # ColBERT ranking (G.1.1)
        colbert_ranks = {}
        if include_colbert:
            colbert_ranked = sorted(
                [(r.doc_id, r.multivec_score) for r in results.values()],
                key=lambda x: x[1], reverse=True
            )
            colbert_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(colbert_ranked)}

        # Compute RRF scores
        for doc_id, result in results.items():
            dense_rank = dense_ranks.get(doc_id, len(results) + 1)
            sparse_rank = sparse_ranks.get(doc_id, len(results) + 1)

            rrf_score = 0.0
            if result.dense_score > 0:
                rrf_score += self.dense_weight / (k + dense_rank)
            if result.sparse_score > 0:
                rrf_score += self.sparse_weight / (k + sparse_rank)

            # Add ColBERT contribution (G.1.1)
            if include_colbert and result.multivec_score > 0:
                colbert_rank = colbert_ranks.get(doc_id, len(results) + 1)
                rrf_score += self.multivec_weight / (k + colbert_rank)

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
        """Get retrieval statistics (G.1.1 update: includes ColBERT)."""
        dense_size = sum(
            emb.nbytes for emb in self.dense_embeddings.values()
        ) / (1024 * 1024)  # MB

        sparse_size = sum(
            len(json.dumps({k: float(v) for k, v in doc.sparse_vector.items()}))
            for doc in self.documents.values()
            if doc.sparse_vector
        ) / (1024 * 1024)  # MB

        # ColBERT index size (G.1.1)
        colbert_size = sum(
            emb.nbytes for emb in self.colbert_embeddings.values()
        ) / (1024 * 1024) if self.colbert_embeddings else 0.0

        # Determine mode
        if self.multivec_weight > 0 and self.colbert_embeddings:
            mode = "full_hybrid"  # Dense + Sparse + ColBERT
        elif self.sparse_weight > 0:
            mode = "hybrid"  # Dense + Sparse
        else:
            mode = "dense_only"

        return HybridRetrievalStats(
            documents_indexed=len(self.documents),
            vocabulary_size=len(self.bm25_index.vocabulary),
            avg_doc_length=self.bm25_index.avg_doc_length,
            dense_index_size_mb=round(dense_size, 2),
            sparse_index_size_mb=round(sparse_size, 4),
            colbert_index_size_mb=round(colbert_size, 2),
            mode=mode,
            colbert_enabled=self._colbert_enabled,
            weights={
                "dense": round(self.dense_weight, 2),
                "sparse": round(self.sparse_weight, 2),
                "colbert": round(self.multivec_weight, 2)
            }
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
