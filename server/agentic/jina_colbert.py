"""
Jina-ColBERT-v2 Late Interaction Retrieval.

Part of G.5.3: Advanced RAG Techniques - 8K context ColBERT with MRL.

Based on Jina-ColBERT-v2 (August 2024):
- 8192 token context window (vs 512 in original ColBERT)
- 89 language support
- Matryoshka Representation Learning (MRL) dimensions
- +6.5% on English BEIR over ColBERTv2
- Late interaction for fine-grained matching

Key Benefits:
- Process long documents without chunking
- Better multilingual support
- MRL dimensions (64/96/128) for HSEA integration
- Late interaction captures token-level semantics

Research Basis:
- Jina-ColBERT-v2 (August 2024)
- ColBERTv2 (NAACL 2022)
- Matryoshka Representation Learning (NeurIPS 2022)

Usage:
    from agentic.jina_colbert import (
        JinaColBERT,
        ColBERTConfig,
        get_jina_colbert
    )

    colbert = get_jina_colbert()
    # Encode documents
    doc_embeddings = await colbert.encode_documents(["doc1", "doc2"])
    # Encode query
    query_embedding = await colbert.encode_query("search query")
    # Score
    scores = colbert.score(query_embedding, doc_embeddings)
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

logger = logging.getLogger("agentic.jina_colbert")

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info("sentence-transformers not available")

# Try to import ColBERT
try:
    from colbert.modeling.checkpoint import Checkpoint
    from colbert.infra import ColBERTConfig as ColBERTLibConfig
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False
    logger.info("ColBERT library not available, using fallback")


class MRLDimension(int, Enum):
    """Matryoshka Representation Learning dimensions."""
    DIM_64 = 64  # Coarse, fast
    DIM_96 = 96  # Balanced
    DIM_128 = 128  # Full precision for ColBERT


class ScoringMethod(str, Enum):
    """Scoring method for late interaction."""
    MAX_SIM = "max_sim"  # MaxSim (original ColBERT)
    AVG_MAX_SIM = "avg_max_sim"  # Average of max similarities
    SUM_MAX_SIM = "sum_max_sim"  # Sum of max similarities


@dataclass
class ColBERTConfig:
    """Configuration for Jina-ColBERT-v2."""
    # Model settings
    model_name: str = "jinaai/jina-colbert-v2"
    fallback_model: str = "BAAI/bge-m3"  # Fallback if ColBERT unavailable
    device: str = "cuda"  # cuda or cpu

    # Dimension settings
    mrl_dimension: MRLDimension = MRLDimension.DIM_128
    normalize_embeddings: bool = True

    # Context settings
    max_query_length: int = 512
    max_document_length: int = 8192  # Jina-ColBERT-v2 supports 8K

    # Scoring
    scoring_method: ScoringMethod = ScoringMethod.MAX_SIM

    # Performance
    batch_size: int = 32
    use_fp16: bool = True

    # Caching
    enable_cache: bool = True
    cache_size: int = 10000


@dataclass
class ColBERTEmbedding:
    """ColBERT token-level embedding."""
    text: str
    embeddings: np.ndarray  # Shape: (num_tokens, embedding_dim)
    token_count: int
    dimension: int
    is_query: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColBERTScore:
    """Score result from ColBERT."""
    query_id: str
    document_id: str
    score: float
    token_scores: Optional[np.ndarray] = None  # Per-token max scores
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColBERTSearchResult:
    """Search result with ColBERT scoring."""
    document_id: str
    score: float
    rank: int
    text: str
    token_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class JinaColBERT:
    """
    Jina-ColBERT-v2 late interaction retrieval.

    Provides token-level embeddings and MaxSim scoring for
    fine-grained semantic matching.
    """

    def __init__(
        self,
        config: Optional[ColBERTConfig] = None,
    ):
        """
        Initialize Jina-ColBERT.

        Args:
            config: ColBERT configuration
        """
        self.config = config or ColBERTConfig()

        self._model = None
        self._tokenizer = None
        self._is_colbert = False
        self._initialized = False

        # Embedding cache
        self._cache: Dict[str, ColBERTEmbedding] = {}

        # Statistics
        self._total_queries = 0
        self._total_documents = 0
        self._cache_hits = 0
        self._avg_encode_time_ms = 0.0

        logger.info(f"JinaColBERT initialized with dim={self.config.mrl_dimension.value}")

    async def initialize(self) -> bool:
        """Initialize the model (lazy loading)."""
        if self._initialized:
            return True

        try:
            if COLBERT_AVAILABLE:
                # Try to load actual ColBERT model
                try:
                    colbert_config = ColBERTLibConfig(
                        doc_maxlen=self.config.max_document_length,
                        query_maxlen=self.config.max_query_length,
                    )
                    self._model = Checkpoint(self.config.model_name, colbert_config)
                    self._is_colbert = True
                    logger.info(f"Loaded ColBERT model: {self.config.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load ColBERT: {e}, using fallback")

            if not self._is_colbert and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Fallback to sentence-transformers
                self._model = SentenceTransformer(
                    self.config.fallback_model,
                    device=self.config.device
                )
                logger.info(f"Using fallback model: {self.config.fallback_model}")

            if self._model is None:
                logger.warning("No embedding model available, using random embeddings")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def _compute_cache_key(self, text: str, is_query: bool) -> str:
        """Compute cache key for embeddings."""
        prefix = "q:" if is_query else "d:"
        content = f"{prefix}{text}:{self.config.mrl_dimension.value}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _encode_with_colbert(
        self,
        texts: List[str],
        is_query: bool
    ) -> List[np.ndarray]:
        """Encode using actual ColBERT model."""
        if is_query:
            embeddings = self._model.queryFromText(texts)
        else:
            embeddings = self._model.docFromText(texts)

        # Convert to numpy and truncate to MRL dimension
        result = []
        for emb in embeddings:
            emb_np = emb.cpu().numpy()
            if emb_np.shape[-1] > self.config.mrl_dimension.value:
                emb_np = emb_np[..., :self.config.mrl_dimension.value]
            result.append(emb_np)

        return result

    async def _encode_with_sentence_transformer(
        self,
        texts: List[str],
        is_query: bool
    ) -> List[np.ndarray]:
        """Encode using sentence-transformers (fallback)."""
        # Sentence transformers gives single embedding per text
        # We'll simulate token-level by splitting and encoding
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        )

        # For ColBERT compatibility, reshape to (num_tokens, dim)
        # Using a simple heuristic: split by words
        result = []
        dim = self.config.mrl_dimension.value

        for i, text in enumerate(texts):
            base_emb = embeddings[i]

            # Truncate to MRL dimension
            if len(base_emb) > dim:
                base_emb = base_emb[:dim]

            # Simulate token embeddings by repeating with small noise
            words = text.split()
            num_tokens = min(len(words), 512 if is_query else 8192)
            num_tokens = max(1, num_tokens)

            # Create token-level embeddings with position-based variation
            token_embs = np.zeros((num_tokens, dim), dtype=np.float32)
            for j in range(num_tokens):
                # Add position-based variation
                noise = np.random.randn(dim).astype(np.float32) * 0.01
                position_factor = 1 + (j / num_tokens) * 0.1
                token_embs[j] = base_emb * position_factor + noise

            # Normalize
            norms = np.linalg.norm(token_embs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            token_embs = token_embs / norms

            result.append(token_embs)

        return result

    async def _encode_fallback(
        self,
        texts: List[str],
        is_query: bool
    ) -> List[np.ndarray]:
        """Fallback encoding using random embeddings."""
        dim = self.config.mrl_dimension.value
        result = []

        for text in texts:
            words = text.split()
            num_tokens = min(len(words), 512 if is_query else 8192)
            num_tokens = max(1, num_tokens)

            # Deterministic random based on text hash
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)

            embeddings = rng.randn(num_tokens, dim).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms > 0, norms, 1)

            result.append(embeddings)

        return result

    async def encode_query(self, query: str) -> ColBERTEmbedding:
        """Encode a query into token-level embeddings."""
        await self.initialize()

        # Check cache
        cache_key = self._compute_cache_key(query, is_query=True)
        if self.config.enable_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        start_time = time.perf_counter()

        # Truncate query
        words = query.split()
        if len(words) > self.config.max_query_length:
            query = " ".join(words[:self.config.max_query_length])

        # Encode
        if self._is_colbert:
            embeddings = await self._encode_with_colbert([query], is_query=True)
        elif self._model is not None:
            embeddings = await self._encode_with_sentence_transformer([query], is_query=True)
        else:
            embeddings = await self._encode_fallback([query], is_query=True)

        emb_array = embeddings[0]
        encode_time = (time.perf_counter() - start_time) * 1000

        result = ColBERTEmbedding(
            text=query,
            embeddings=emb_array,
            token_count=emb_array.shape[0],
            dimension=emb_array.shape[1],
            is_query=True,
            metadata={"encode_time_ms": encode_time}
        )

        # Update statistics
        self._total_queries += 1
        self._avg_encode_time_ms = (
            (self._avg_encode_time_ms * (self._total_queries - 1) + encode_time)
            / self._total_queries
        )

        # Cache
        if self.config.enable_cache:
            if len(self._cache) >= self.config.cache_size:
                # Remove oldest entries
                keys_to_remove = list(self._cache.keys())[:len(self._cache) // 4]
                for k in keys_to_remove:
                    del self._cache[k]
            self._cache[cache_key] = result

        return result

    async def encode_document(self, document: str, doc_id: Optional[str] = None) -> ColBERTEmbedding:
        """Encode a document into token-level embeddings."""
        await self.initialize()

        # Check cache
        cache_key = self._compute_cache_key(document, is_query=False)
        if self.config.enable_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        start_time = time.perf_counter()

        # Truncate document
        words = document.split()
        if len(words) > self.config.max_document_length:
            document = " ".join(words[:self.config.max_document_length])

        # Encode
        if self._is_colbert:
            embeddings = await self._encode_with_colbert([document], is_query=False)
        elif self._model is not None:
            embeddings = await self._encode_with_sentence_transformer([document], is_query=False)
        else:
            embeddings = await self._encode_fallback([document], is_query=False)

        emb_array = embeddings[0]
        encode_time = (time.perf_counter() - start_time) * 1000

        result = ColBERTEmbedding(
            text=document,
            embeddings=emb_array,
            token_count=emb_array.shape[0],
            dimension=emb_array.shape[1],
            is_query=False,
            metadata={"encode_time_ms": encode_time, "doc_id": doc_id}
        )

        self._total_documents += 1

        # Cache
        if self.config.enable_cache:
            if len(self._cache) >= self.config.cache_size:
                keys_to_remove = list(self._cache.keys())[:len(self._cache) // 4]
                for k in keys_to_remove:
                    del self._cache[k]
            self._cache[cache_key] = result

        return result

    async def encode_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> List[ColBERTEmbedding]:
        """Encode multiple documents."""
        tasks = []
        for i, doc in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i}"
            tasks.append(self.encode_document(doc, doc_id))

        return await asyncio.gather(*tasks)

    def compute_max_sim(
        self,
        query_emb: np.ndarray,
        doc_emb: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute MaxSim score between query and document.

        For each query token, finds the maximum similarity to any
        document token, then averages across query tokens.

        Args:
            query_emb: Query embeddings (Q, D)
            doc_emb: Document embeddings (N, D)

        Returns:
            (score, per_token_scores)
        """
        # Compute similarity matrix: (Q, N)
        sim_matrix = np.dot(query_emb, doc_emb.T)

        # Max similarity for each query token
        max_sims = np.max(sim_matrix, axis=1)

        # Aggregate
        if self.config.scoring_method == ScoringMethod.MAX_SIM:
            score = float(np.mean(max_sims))
        elif self.config.scoring_method == ScoringMethod.SUM_MAX_SIM:
            score = float(np.sum(max_sims))
        else:  # AVG_MAX_SIM
            score = float(np.mean(max_sims))

        return score, max_sims

    def score(
        self,
        query: ColBERTEmbedding,
        documents: List[ColBERTEmbedding]
    ) -> List[ColBERTScore]:
        """Score documents against a query."""
        results = []

        for doc in documents:
            score, token_scores = self.compute_max_sim(
                query.embeddings,
                doc.embeddings
            )

            results.append(ColBERTScore(
                query_id=self._compute_cache_key(query.text, True)[:8],
                document_id=doc.metadata.get("doc_id", "unknown"),
                score=score,
                token_scores=token_scores,
            ))

        return results

    async def search(
        self,
        query: str,
        documents: List[Tuple[str, str]],  # (doc_id, doc_text)
        top_k: int = 10
    ) -> List[ColBERTSearchResult]:
        """
        Search documents using ColBERT.

        Args:
            query: Search query
            documents: List of (doc_id, doc_text) tuples
            top_k: Number of results to return

        Returns:
            Ranked search results
        """
        # Encode query
        query_emb = await self.encode_query(query)

        # Encode documents
        doc_embs = await self.encode_documents(
            [d[1] for d in documents],
            [d[0] for d in documents]
        )

        # Score
        scores = self.score(query_emb, doc_embs)

        # Sort and return top-k
        scored = list(zip(scores, documents, doc_embs))
        scored.sort(key=lambda x: x[0].score, reverse=True)

        results = []
        for rank, (score, (doc_id, doc_text), doc_emb) in enumerate(scored[:top_k], 1):
            # Convert numpy float16 to native Python float for JSON serialization
            token_scores_list = None
            if score.token_scores is not None:
                token_scores_list = [float(x) for x in score.token_scores.tolist()]

            results.append(ColBERTSearchResult(
                document_id=doc_id,
                score=float(score.score) if hasattr(score.score, 'item') else score.score,
                rank=rank,
                text=doc_text[:500],  # Truncate for result
                token_scores=token_scores_list,
                metadata={
                    "token_count": doc_emb.token_count,
                }
            ))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        return {
            "total_queries": self._total_queries,
            "total_documents": self._total_documents,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "avg_encode_time_ms": round(self._avg_encode_time_ms, 2),
            "is_colbert_model": self._is_colbert,
            "model_initialized": self._initialized,
            "config": {
                "model_name": self.config.model_name if self._is_colbert else self.config.fallback_model,
                "mrl_dimension": self.config.mrl_dimension.value,
                "max_document_length": self.config.max_document_length,
                "scoring_method": self.config.scoring_method.value,
            }
        }

    def clear_cache(self) -> int:
        """Clear embedding cache. Returns count cleared."""
        count = len(self._cache)
        self._cache.clear()
        return count


# Global instance
_jina_colbert: Optional[JinaColBERT] = None


def get_jina_colbert(
    config: Optional[ColBERTConfig] = None
) -> JinaColBERT:
    """Get or create global Jina-ColBERT instance."""
    global _jina_colbert
    if _jina_colbert is None:
        _jina_colbert = JinaColBERT(config)
    return _jina_colbert


async def colbert_encode(
    text: str,
    is_query: bool = True
) -> ColBERTEmbedding:
    """Convenience function for ColBERT encoding."""
    colbert = get_jina_colbert()
    if is_query:
        return await colbert.encode_query(text)
    else:
        return await colbert.encode_document(text)


async def colbert_search(
    query: str,
    documents: List[Tuple[str, str]],
    top_k: int = 10
) -> List[ColBERTSearchResult]:
    """Convenience function for ColBERT search."""
    return await get_jina_colbert().search(query, documents, top_k)
