"""
Cascade Retriever for Hierarchical Retrieval Optimization.

Part of G.2: Hierarchical Retrieval Optimization (Phase 2)
- G.2.1: Binary quantization with 3x oversampling + rescoring
- G.2.2: Matryoshka cascade (64→256→1024) with early exit

Implements FunnelRAG-inspired cascade retrieval with precision-aware
early termination and progressive refinement.

Research Basis:
- FunnelRAG (arXiv:2410.10293): Cascade retrieval with progressive filtering
- Matryoshka Representation Learning (NeurIPS 2022): Nested embeddings
- Binary Embedding Retrieval: 32x compression with oversampling

Usage:
    from agentic.cascade_retriever import CascadeRetriever

    retriever = CascadeRetriever()
    await retriever.index_document("doc1", "FANUC SRVO-063 alarm...")

    results, stats = await retriever.search(
        query="robot alarm",
        top_k=10,
        oversampling_factor=3
    )
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("agentic.cascade_retriever")


class CascadeStage(str, Enum):
    """Stages in the cascade retrieval pipeline."""
    BINARY = "binary"           # 1-bit, Hamming distance
    MRL_64 = "mrl_64"          # 64-dim, coarse semantics
    MRL_256 = "mrl_256"        # 256-dim, balanced
    MRL_1024 = "mrl_1024"      # 1024-dim, fine-grained
    FULL = "full"              # Full precision (4096-dim)


@dataclass
class CascadeConfig:
    """Configuration for cascade retrieval."""
    # Oversampling factors at each stage
    binary_oversample: int = 3       # 3x oversampling at binary stage
    mrl_64_oversample: int = 3       # 3x at MRL-64
    mrl_256_oversample: int = 2      # 2x at MRL-256
    mrl_1024_oversample: int = 1     # 1x at MRL-1024 (final filtering)

    # Early exit thresholds
    early_exit_score: float = 0.92   # Exit if top score exceeds this
    entropy_threshold: float = 0.15  # Exit if score entropy below this
    stability_patience: int = 2      # Ranking stability check iterations

    # Stage enable/disable
    enable_binary: bool = True
    enable_mrl_cascade: bool = True
    enable_early_exit: bool = True

    # Performance tuning
    max_candidates_per_stage: int = 1000
    min_candidates: int = 10


@dataclass
class CascadeStats:
    """Statistics from cascade retrieval."""
    stages_executed: List[str] = field(default_factory=list)
    candidates_per_stage: Dict[str, int] = field(default_factory=dict)
    time_per_stage_ms: Dict[str, float] = field(default_factory=dict)
    early_exit_triggered: bool = False
    early_exit_stage: Optional[str] = None
    early_exit_reason: Optional[str] = None
    total_time_ms: float = 0.0
    final_candidates: int = 0


@dataclass
class CascadeResult:
    """Result from cascade retrieval."""
    doc_id: str
    score: float
    final_stage: CascadeStage
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Per-stage scores for analysis
    stage_scores: Dict[str, float] = field(default_factory=dict)


class CascadeRetriever:
    """
    Implements FunnelRAG-style cascade retrieval with MRL dimensions.

    Pipeline:
    ```
    Query
      |
      v
    [Binary Stage] ---> 3x oversampling ---> N candidates
          |
          v
    [MRL-64 Stage] ---> 3x oversampling ---> N/3 candidates
          |
          v (early exit check)
    [MRL-256 Stage] --> 2x oversampling ---> N/6 candidates
          |
          v (early exit check)
    [MRL-1024 Stage] -> Final scoring ----> top_k results
          |
          v (optional full precision for ties)
    [Full Stage] -----> Tie-breaking -----> Final results
    ```
    """

    # MRL dimension progression (Matryoshka nesting)
    MRL_DIMENSIONS = [64, 256, 1024, 4096]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "mxbai-embed-large",
        config: Optional[CascadeConfig] = None
    ):
        """
        Initialize cascade retriever.

        Args:
            ollama_url: Ollama API URL
            embedding_model: Model for generating embeddings
            config: Cascade configuration
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.embedding_model = embedding_model
        self.config = config or CascadeConfig()

        # Indices at different precision levels
        self._binary_index: Dict[str, bytes] = {}
        self._mrl_indices: Dict[int, Dict[str, np.ndarray]] = {
            dim: {} for dim in self.MRL_DIMENSIONS
        }
        self._full_embeddings: Dict[str, np.ndarray] = {}

        # Document metadata
        self._documents: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._total_indexed = 0
        self._total_queries = 0
        self._early_exits = 0

        logger.info(
            f"CascadeRetriever initialized: model={embedding_model}, "
            f"binary_oversample={self.config.binary_oversample}x"
        )

    async def _get_embedding(
        self,
        text: str,
        truncate_dim: Optional[int] = None
    ) -> np.ndarray:
        """Get embedding from Ollama."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = np.array(data.get("embedding", []), dtype=np.float32)

                    # MRL truncation if requested
                    if truncate_dim and truncate_dim < len(embedding):
                        embedding = embedding[:truncate_dim]
                        # Re-normalize
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                    return embedding
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return np.zeros(1024, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return np.zeros(1024, dtype=np.float32)

    def _to_binary(self, embedding: np.ndarray) -> bytes:
        """Convert embedding to binary (1-bit per dimension)."""
        binary_bits = (embedding > 0).astype(np.uint8)
        return np.packbits(binary_bits).tobytes()

    def _hamming_distance(self, a: bytes, b: bytes) -> int:
        """Compute Hamming distance between binary embeddings."""
        xor_result = bytes(x ^ y for x, y in zip(a, b))
        return sum(bin(byte).count('1') for byte in xor_result)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _compute_entropy(self, scores: List[float]) -> float:
        """Compute entropy of score distribution."""
        if not scores or len(scores) < 2:
            return 0.0

        # Normalize to probability distribution
        total = sum(scores)
        if total == 0:
            return 0.0

        probs = [s / total for s in scores]

        # Compute entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize by max entropy
        max_entropy = np.log2(len(scores))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _check_ranking_stability(
        self,
        current_ranking: List[str],
        history: List[List[str]],
        patience: int
    ) -> bool:
        """Check if ranking has stabilized."""
        if len(history) < patience:
            return False

        # Check if top-k ranking matches in recent iterations
        for prev_ranking in history[-patience:]:
            if prev_ranking[:len(current_ranking)] != current_ranking:
                return False

        return True

    async def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index a document at all cascade levels.

        Args:
            doc_id: Unique document identifier
            content: Document content to embed
            metadata: Optional metadata

        Returns:
            Indexing result with statistics
        """
        start_time = time.time()

        # Get full embedding
        full_emb = await self._get_embedding(content)
        self._full_embeddings[doc_id] = full_emb

        # Generate binary representation
        self._binary_index[doc_id] = self._to_binary(full_emb)

        # Generate MRL truncations
        for dim in self.MRL_DIMENSIONS:
            if dim <= len(full_emb):
                truncated = full_emb[:dim]
                norm = np.linalg.norm(truncated)
                if norm > 0:
                    truncated = truncated / norm
                self._mrl_indices[dim][doc_id] = truncated

        # Store metadata
        self._documents[doc_id] = {
            "content": content,
            "metadata": metadata or {},
            "indexed_at": time.time()
        }

        self._total_indexed += 1

        return {
            "doc_id": doc_id,
            "indexed": True,
            "dimensions": len(full_emb),
            "mrl_levels": [dim for dim in self.MRL_DIMENSIONS if dim <= len(full_emb)],
            "time_ms": (time.time() - start_time) * 1000
        }

    async def search(
        self,
        query: str,
        top_k: int = 10,
        oversampling_factor: Optional[int] = None
    ) -> Tuple[List[CascadeResult], CascadeStats]:
        """
        Execute cascade retrieval with oversampling and early exit.

        Args:
            query: Search query
            top_k: Number of final results
            oversampling_factor: Override default oversampling (3x)

        Returns:
            Tuple of (results, statistics)
        """
        start_time = time.time()
        stats = CascadeStats()
        self._total_queries += 1

        # Get query embedding at full precision
        query_full = await self._get_embedding(query)

        # Track candidates and their scores
        candidate_scores: Dict[str, Dict[str, float]] = {}
        current_candidates = list(self._documents.keys())
        ranking_history: List[List[str]] = []

        # Calculate stage candidate counts with oversampling
        base_oversample = oversampling_factor or self.config.binary_oversample
        stage_k = {
            CascadeStage.BINARY: min(top_k * base_oversample * 3, len(current_candidates)),
            CascadeStage.MRL_64: min(top_k * base_oversample * 2, len(current_candidates)),
            CascadeStage.MRL_256: min(top_k * base_oversample, len(current_candidates)),
            CascadeStage.MRL_1024: min(top_k * 2, len(current_candidates)),
            CascadeStage.FULL: top_k
        }

        # Stage 1: Binary search (if enabled)
        if self.config.enable_binary and self._binary_index:
            stage_start = time.time()
            stats.stages_executed.append("binary")

            query_binary = self._to_binary(query_full)

            binary_scores = []
            for doc_id in current_candidates:
                if doc_id in self._binary_index:
                    distance = self._hamming_distance(query_binary, self._binary_index[doc_id])
                    max_distance = len(query_binary) * 8
                    similarity = 1.0 - (distance / max_distance)
                    binary_scores.append((doc_id, similarity))

                    if doc_id not in candidate_scores:
                        candidate_scores[doc_id] = {}
                    candidate_scores[doc_id]["binary"] = similarity

            binary_scores.sort(key=lambda x: x[1], reverse=True)
            current_candidates = [doc_id for doc_id, _ in binary_scores[:stage_k[CascadeStage.BINARY]]]

            stats.candidates_per_stage["binary"] = len(current_candidates)
            stats.time_per_stage_ms["binary"] = (time.time() - stage_start) * 1000

        # Stage 2-4: MRL cascade (if enabled)
        if self.config.enable_mrl_cascade:
            mrl_stages = [
                (CascadeStage.MRL_64, 64),
                (CascadeStage.MRL_256, 256),
                (CascadeStage.MRL_1024, 1024),
            ]

            for stage, dim in mrl_stages:
                if dim not in self._mrl_indices or not self._mrl_indices[dim]:
                    continue

                stage_start = time.time()
                stage_name = f"mrl_{dim}"
                stats.stages_executed.append(stage_name)

                # Get query at this MRL dimension
                query_mrl = query_full[:dim] if dim <= len(query_full) else query_full
                norm = np.linalg.norm(query_mrl)
                if norm > 0:
                    query_mrl = query_mrl / norm

                mrl_scores = []
                for doc_id in current_candidates:
                    if doc_id in self._mrl_indices[dim]:
                        doc_mrl = self._mrl_indices[dim][doc_id]
                        similarity = self._cosine_similarity(query_mrl, doc_mrl)
                        mrl_scores.append((doc_id, similarity))

                        if doc_id not in candidate_scores:
                            candidate_scores[doc_id] = {}
                        candidate_scores[doc_id][stage_name] = similarity

                mrl_scores.sort(key=lambda x: x[1], reverse=True)
                current_candidates = [doc_id for doc_id, _ in mrl_scores[:stage_k[stage]]]

                stats.candidates_per_stage[stage_name] = len(current_candidates)
                stats.time_per_stage_ms[stage_name] = (time.time() - stage_start) * 1000

                # Early exit check
                if self.config.enable_early_exit and mrl_scores:
                    top_score = mrl_scores[0][1]
                    all_scores = [s for _, s in mrl_scores[:min(20, len(mrl_scores))]]
                    entropy = self._compute_entropy(all_scores)

                    current_ranking = [doc_id for doc_id, _ in mrl_scores[:top_k]]
                    ranking_history.append(current_ranking)

                    # Check exit conditions
                    if top_score >= self.config.early_exit_score:
                        stats.early_exit_triggered = True
                        stats.early_exit_stage = stage_name
                        stats.early_exit_reason = f"score_threshold ({top_score:.3f} >= {self.config.early_exit_score})"
                        self._early_exits += 1
                        logger.debug(f"Early exit at {stage_name}: {stats.early_exit_reason}")
                        break

                    if entropy < self.config.entropy_threshold:
                        stats.early_exit_triggered = True
                        stats.early_exit_stage = stage_name
                        stats.early_exit_reason = f"low_entropy ({entropy:.3f} < {self.config.entropy_threshold})"
                        self._early_exits += 1
                        logger.debug(f"Early exit at {stage_name}: {stats.early_exit_reason}")
                        break

                    if self._check_ranking_stability(
                        current_ranking,
                        ranking_history,
                        self.config.stability_patience
                    ):
                        stats.early_exit_triggered = True
                        stats.early_exit_stage = stage_name
                        stats.early_exit_reason = "ranking_stable"
                        self._early_exits += 1
                        logger.debug(f"Early exit at {stage_name}: {stats.early_exit_reason}")
                        break

        # Final stage: Full precision scoring (for top candidates only)
        if not stats.early_exit_triggered:
            stage_start = time.time()
            stats.stages_executed.append("full")

            full_scores = []
            for doc_id in current_candidates[:stage_k[CascadeStage.FULL] * 2]:
                if doc_id in self._full_embeddings:
                    similarity = self._cosine_similarity(query_full, self._full_embeddings[doc_id])
                    full_scores.append((doc_id, similarity))

                    if doc_id not in candidate_scores:
                        candidate_scores[doc_id] = {}
                    candidate_scores[doc_id]["full"] = similarity

            full_scores.sort(key=lambda x: x[1], reverse=True)
            current_candidates = [doc_id for doc_id, _ in full_scores[:top_k]]

            stats.candidates_per_stage["full"] = len(current_candidates)
            stats.time_per_stage_ms["full"] = (time.time() - stage_start) * 1000

        # Build final results
        results = []
        final_stage = CascadeStage.FULL
        if stats.early_exit_stage:
            final_stage = CascadeStage(stats.early_exit_stage.replace("mrl_", "MRL_").upper())

        for doc_id in current_candidates[:top_k]:
            doc_info = self._documents.get(doc_id, {})
            scores = candidate_scores.get(doc_id, {})

            # Use best available score
            final_score = scores.get("full") or scores.get("mrl_1024") or \
                         scores.get("mrl_256") or scores.get("mrl_64") or \
                         scores.get("binary", 0.0)

            results.append(CascadeResult(
                doc_id=doc_id,
                score=final_score,
                final_stage=final_stage,
                content=doc_info.get("content", ""),
                metadata=doc_info.get("metadata", {}),
                stage_scores=scores
            ))

        stats.final_candidates = len(results)
        stats.total_time_ms = (time.time() - start_time) * 1000

        return results, stats

    async def search_with_oversampling(
        self,
        query: str,
        top_k: int = 10,
        binary_oversample: int = 3,
        mrl_oversample: int = 2
    ) -> Tuple[List[CascadeResult], CascadeStats]:
        """
        Search with explicit oversampling configuration.

        G.2.1: Binary quantization with 3x oversampling + rescoring

        Args:
            query: Search query
            top_k: Final number of results
            binary_oversample: Oversampling at binary stage (default 3x)
            mrl_oversample: Oversampling at MRL stages (default 2x)

        Returns:
            Tuple of (results, statistics)
        """
        # Create custom config for this search
        custom_config = CascadeConfig(
            binary_oversample=binary_oversample,
            mrl_64_oversample=mrl_oversample,
            mrl_256_oversample=mrl_oversample,
            mrl_1024_oversample=1,
            enable_binary=True,
            enable_mrl_cascade=True,
            enable_early_exit=self.config.enable_early_exit
        )

        # Temporarily swap config
        original_config = self.config
        self.config = custom_config

        try:
            results, stats = await self.search(query, top_k)
            return results, stats
        finally:
            self.config = original_config

    async def mrl_cascade_search(
        self,
        query: str,
        top_k: int = 10,
        start_dim: int = 64,
        end_dim: int = 1024
    ) -> Tuple[List[CascadeResult], CascadeStats]:
        """
        Pure MRL cascade search without binary stage.

        G.2.2: Matryoshka cascade (64→256→1024) with early exit

        Args:
            query: Search query
            top_k: Final number of results
            start_dim: Starting MRL dimension
            end_dim: Ending MRL dimension

        Returns:
            Tuple of (results, statistics)
        """
        custom_config = CascadeConfig(
            enable_binary=False,  # Skip binary stage
            enable_mrl_cascade=True,
            enable_early_exit=True,
            early_exit_score=self.config.early_exit_score,
            entropy_threshold=self.config.entropy_threshold
        )

        original_config = self.config
        self.config = custom_config

        try:
            results, stats = await self.search(query, top_k)
            return results, stats
        finally:
            self.config = original_config

    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_indexed": self._total_indexed,
            "total_queries": self._total_queries,
            "early_exits": self._early_exits,
            "early_exit_rate": self._early_exits / max(self._total_queries, 1),
            "index_sizes": {
                "binary": len(self._binary_index),
                "mrl_64": len(self._mrl_indices.get(64, {})),
                "mrl_256": len(self._mrl_indices.get(256, {})),
                "mrl_1024": len(self._mrl_indices.get(1024, {})),
                "full": len(self._full_embeddings)
            },
            "config": {
                "binary_oversample": self.config.binary_oversample,
                "early_exit_score": self.config.early_exit_score,
                "entropy_threshold": self.config.entropy_threshold
            }
        }

    def clear_index(self):
        """Clear all indices."""
        self._binary_index.clear()
        for dim in self.MRL_DIMENSIONS:
            self._mrl_indices[dim].clear()
        self._full_embeddings.clear()
        self._documents.clear()
        self._total_indexed = 0
        logger.info("Cascade index cleared")


# Global instance
_cascade_retriever: Optional[CascadeRetriever] = None


def get_cascade_retriever(
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "mxbai-embed-large",
    config: Optional[CascadeConfig] = None
) -> CascadeRetriever:
    """Get or create global cascade retriever instance."""
    global _cascade_retriever
    if _cascade_retriever is None:
        _cascade_retriever = CascadeRetriever(
            ollama_url=ollama_url,
            embedding_model=embedding_model,
            config=config
        )
    return _cascade_retriever


async def get_cascade_retriever_async(
    ollama_url: str = "http://localhost:11434",
    embedding_model: str = "mxbai-embed-large",
    config: Optional[CascadeConfig] = None
) -> CascadeRetriever:
    """Async wrapper for get_cascade_retriever."""
    return get_cascade_retriever(ollama_url, embedding_model, config)
