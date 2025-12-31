"""
Cross-Encoder Reranker using BGE-Reranker-v2-M3.

Implements high-precision reranking for the final stage of hybrid retrieval:
- Takes top-50 candidates from initial retrieval
- Uses cross-encoder for pairwise query-document scoring
- Returns top-k results with refined scores

This is Part G.1.6 of the RAG Architecture Improvements plan.

Key Features:
1. Lazy model loading to avoid VRAM usage when disabled
2. Batch processing for efficient GPU utilization
3. Score normalization for consistent thresholds
4. Integration with BGEM3HybridRetriever

References:
- BGE-Reranker-v2-M3 (BAAI, 2024)
- ColBERTv2 (NAACL 2022) - late interaction
- MS MARCO reranking benchmarks
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid loading model at module import time
_FLAG_RERANKER_AVAILABLE = False
_RERANKER_MODEL = None

try:
    from FlagEmbedding import FlagReranker
    _FLAG_RERANKER_AVAILABLE = True
    logger.info("FlagReranker available - cross-encoder reranking enabled")
except ImportError:
    logger.warning("FlagReranker not available - reranking disabled")


@dataclass
class RerankedResult:
    """Result from cross-encoder reranking."""
    doc_id: str
    original_score: float
    rerank_score: float
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerStats:
    """Statistics from reranking operation."""
    input_count: int = 0
    output_count: int = 0
    rerank_time_ms: float = 0
    model_name: str = ""
    max_score: float = 0
    min_score: float = 0
    avg_score: float = 0


class CrossEncoderReranker:
    """
    Cross-encoder reranker using BGE-Reranker-v2-M3.

    Usage:
        reranker = CrossEncoderReranker()

        # Rerank search results
        reranked = await reranker.rerank(
            query="FANUC SRVO-063 alarm troubleshooting",
            documents=[
                {"doc_id": "1", "content": "SRVO-063 indicates..."},
                {"doc_id": "2", "content": "For servo alarms..."},
            ],
            top_k=10
        )

    Model Info:
        - bge-reranker-v2-m3: ~1GB, supports 100+ languages
        - Latency: ~100-300ms for 50 documents
        - VRAM: ~1.5GB when loaded

    Architecture:
        ```
        Query + Document
            |
            v
        [Cross-Encoder BERT]
            |
            v
        Relevance Score (0-1)
        ```
    """

    # Available reranker models
    MODELS = {
        "bge-reranker-v2-m3": {
            "name": "BAAI/bge-reranker-v2-m3",
            "size_gb": 1.0,
            "languages": 100,
            "description": "Multilingual, high quality"
        },
        "bge-reranker-v2-gemma": {
            "name": "BAAI/bge-reranker-v2-gemma",
            "size_gb": 2.0,
            "languages": 100,
            "description": "Higher quality, larger"
        },
        "bge-reranker-large": {
            "name": "BAAI/bge-reranker-large",
            "size_gb": 0.5,
            "languages": 1,  # English only
            "description": "Fast, English only"
        }
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        max_length: int = 512,
        batch_size: int = 32,
        normalize_scores: bool = True,
        device: Optional[str] = None  # None = auto-detect
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
            use_fp16: Use FP16 for faster inference
            max_length: Max tokens per query-document pair
            batch_size: Batch size for reranking
            normalize_scores: Normalize scores to 0-1 range
            device: Device to use (None for auto-detect)
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_scores = normalize_scores
        self.device = device

        # Lazy-loaded model
        self._model = None
        self._is_loaded = False

        # Statistics
        self._total_reranks = 0
        self._total_documents = 0
        self._total_time_ms = 0

        logger.info(
            f"CrossEncoderReranker initialized: model={model_name}, "
            f"fp16={use_fp16}, batch_size={batch_size}"
        )

    def _load_model(self):
        """Lazy load the reranker model."""
        global _RERANKER_MODEL

        if not _FLAG_RERANKER_AVAILABLE:
            logger.warning("FlagReranker not available, skipping model load")
            return False

        if self._is_loaded:
            return True

        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            start = time.time()

            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device
            )

            # Cache globally for reuse
            _RERANKER_MODEL = self._model

            load_time = time.time() - start
            self._is_loaded = True
            logger.info(f"Reranker model loaded in {load_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if reranker is available."""
        return _FLAG_RERANKER_AVAILABLE

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> Tuple[List[RerankedResult], RerankerStats]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of documents with 'doc_id' and 'content' keys
            top_k: Number of top results to return
            score_threshold: Minimum score to include in results

        Returns:
            Tuple of (reranked_results, stats)
        """
        stats = RerankerStats(
            input_count=len(documents),
            model_name=self.model_name
        )

        if not documents:
            return [], stats

        if not self._load_model():
            # Fallback: return documents as-is with original scores
            logger.warning("Reranker model not available, returning original order")
            results = [
                RerankedResult(
                    doc_id=doc.get("doc_id", str(i)),
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
                for i, doc in enumerate(documents)
            ]
            return results[:top_k], stats

        # Run reranking in thread pool to not block async
        loop = asyncio.get_event_loop()
        results, stats = await loop.run_in_executor(
            None,
            self._rerank_sync,
            query,
            documents,
            top_k,
            score_threshold,
            stats
        )

        return results, stats

    def _rerank_sync(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int,
        score_threshold: float,
        stats: RerankerStats
    ) -> Tuple[List[RerankedResult], RerankerStats]:
        """Synchronous reranking (runs in thread pool)."""
        start_time = time.time()

        try:
            # Prepare query-document pairs
            pairs = [
                [query, doc.get("content", "")]
                for doc in documents
            ]

            # Compute scores
            scores = self._model.compute_score(
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length
            )

            # Normalize scores if requested
            if self.normalize_scores and len(scores) > 0:
                # Apply sigmoid for 0-1 range
                scores = self._sigmoid(np.array(scores))
            else:
                scores = np.array(scores)

            # Build results with scores
            results = []
            for i, doc in enumerate(documents):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score >= score_threshold:
                    results.append(RerankedResult(
                        doc_id=doc.get("doc_id", str(i)),
                        original_score=doc.get("score", 0.0),
                        rerank_score=score,
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {})
                    ))

            # Sort by rerank score
            results.sort(key=lambda r: r.rerank_score, reverse=True)

            # Update stats
            stats.rerank_time_ms = (time.time() - start_time) * 1000
            stats.output_count = min(len(results), top_k)

            if results:
                result_scores = [r.rerank_score for r in results]
                stats.max_score = max(result_scores)
                stats.min_score = min(result_scores)
                stats.avg_score = sum(result_scores) / len(result_scores)

            # Update global stats
            self._total_reranks += 1
            self._total_documents += len(documents)
            self._total_time_ms += stats.rerank_time_ms

            logger.debug(
                f"Reranked {len(documents)} docs in {stats.rerank_time_ms:.1f}ms, "
                f"top score={stats.max_score:.3f}"
            )

            return results[:top_k], stats

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order on error
            results = [
                RerankedResult(
                    doc_id=doc.get("doc_id", str(i)),
                    original_score=doc.get("score", 0.0),
                    rerank_score=doc.get("score", 0.0),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
                for i, doc in enumerate(documents)
            ]
            stats.rerank_time_ms = (time.time() - start_time) * 1000
            return results[:top_k], stats

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function for score normalization."""
        return 1 / (1 + np.exp(-x))

    async def rerank_hybrid_results(
        self,
        query: str,
        results: List[Any],  # HybridSearchResult from bge_m3_hybrid
        top_k: int = 10
    ) -> List[Any]:
        """
        Rerank results from BGEM3HybridRetriever.

        This is a convenience method for integrating with the existing
        hybrid retrieval pipeline.

        Args:
            query: Search query
            results: List of HybridSearchResult objects
            top_k: Number of results to return

        Returns:
            Reranked HybridSearchResult list
        """
        if not results:
            return results

        # Convert to dict format for reranking
        documents = [
            {
                "doc_id": r.doc_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata
            }
            for r in results
        ]

        reranked, stats = await self.rerank(query, documents, top_k)

        # Map back to original result type
        result_map = {r.doc_id: r for r in results}
        reranked_results = []

        for rr in reranked:
            if rr.doc_id in result_map:
                original = result_map[rr.doc_id]
                # Update the score to be the reranked score
                # Create new instance with updated score
                from .bge_m3_hybrid import HybridSearchResult
                reranked_results.append(HybridSearchResult(
                    doc_id=original.doc_id,
                    score=rr.rerank_score,  # Use reranked score
                    dense_score=original.dense_score,
                    sparse_score=original.sparse_score,
                    colbert_score=getattr(original, 'colbert_score', None),
                    content=original.content,
                    metadata={
                        **original.metadata,
                        "original_score": rr.original_score,
                        "reranker_used": True,
                        "reranker_model": self.model_name
                    }
                ))

        return reranked_results

    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "available": _FLAG_RERANKER_AVAILABLE,
            "model_loaded": self._is_loaded,
            "model_name": self.model_name,
            "use_fp16": self.use_fp16,
            "batch_size": self.batch_size,
            "total_reranks": self._total_reranks,
            "total_documents": self._total_documents,
            "total_time_ms": self._total_time_ms,
            "avg_docs_per_rerank": (
                self._total_documents / self._total_reranks
                if self._total_reranks > 0 else 0
            ),
            "avg_time_per_rerank_ms": (
                self._total_time_ms / self._total_reranks
                if self._total_reranks > 0 else 0
            )
        }

    def unload_model(self):
        """Unload model to free VRAM."""
        global _RERANKER_MODEL

        if self._model is not None:
            del self._model
            self._model = None
            _RERANKER_MODEL = None
            self._is_loaded = False
            logger.info("Reranker model unloaded")


# =============================================================================
# Singleton Instance
# =============================================================================

_reranker: Optional[CrossEncoderReranker] = None


def get_cross_encoder_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3"
) -> CrossEncoderReranker:
    """Get or create singleton CrossEncoderReranker."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(model_name=model_name)
    return _reranker


async def get_cross_encoder_reranker_async(
    model_name: str = "BAAI/bge-reranker-v2-m3"
) -> CrossEncoderReranker:
    """Get singleton CrossEncoderReranker with model pre-loaded."""
    reranker = get_cross_encoder_reranker(model_name)
    # Pre-load model
    await asyncio.get_event_loop().run_in_executor(
        None, reranker._load_model
    )
    return reranker
