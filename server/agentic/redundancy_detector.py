"""
Redundancy Detection & Clustering Module

Identifies semantically similar documents and selects the best representative
from each cluster. This reduces token usage while preserving information coverage.

Research basis:
- Context-Picker (arXiv 2512.14465): Leave-One-Out minimal sufficient set
- RAGAS deduplication: Semantic similarity filtering
- InfoGain-RAG: Quality-weighted representative selection

Thresholds from research:
- Semantic similarity > 0.85: Consider as duplicates
- Cluster representative: Highest quality/information gain
"""

import asyncio
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import numpy as np
import httpx

logger = logging.getLogger(__name__)


@dataclass
class DocumentCluster:
    """A cluster of semantically similar documents."""
    cluster_id: str
    documents: List[Dict[str, Any]]
    embeddings: List[List[float]]
    centroid: List[float]
    representative_idx: int
    similarity_scores: List[float]  # Similarity of each doc to centroid
    avg_similarity: float

    @property
    def representative(self) -> Dict[str, Any]:
        """Get the cluster representative document."""
        return self.documents[self.representative_idx]

    @property
    def size(self) -> int:
        return len(self.documents)


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    original_count: int
    deduplicated_count: int
    clusters: List[DocumentCluster]
    removed_documents: List[Dict[str, Any]]
    similarity_threshold: float
    processing_time_ms: float

    @property
    def reduction_ratio(self) -> float:
        if self.original_count == 0:
            return 0.0
        return 1.0 - (self.deduplicated_count / self.original_count)


class SelectionMethod(str, Enum):
    """Methods for selecting cluster representative."""
    LONGEST = "longest"              # Longest content
    SHORTEST = "shortest"            # Shortest (most concise)
    HIGHEST_QUALITY = "quality"      # Highest quality score
    MOST_CENTRAL = "central"         # Closest to centroid
    INFORMATION_GAIN = "dig"         # Highest DIG score
    SOURCE_AUTHORITY = "authority"   # Most authoritative source


class RedundancyDetector:
    """
    Detect and cluster semantically similar documents.

    Uses embedding-based similarity to identify redundant content,
    then selects the best representative from each cluster.

    This significantly reduces context size while preserving coverage.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "mxbai-embed-large",
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 1
    ):
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self._embedding_cache: Dict[str, List[float]] = {}

    def _hash_content(self, content: str) -> str:
        """Generate content hash for caching."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, with caching."""
        cache_key = self._hash_content(text[:2000])

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Truncate for embedding
        text_truncated = text[:2000] if len(text) > 2000 else text

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

    async def _get_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Get embeddings for multiple texts in parallel."""
        tasks = [self._get_embedding(t) for t in texts]
        return await asyncio.gather(*tasks)

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _calculate_centroid(
        self,
        embeddings: List[List[float]]
    ) -> List[float]:
        """Calculate centroid of embeddings."""
        if not embeddings:
            return []
        return list(np.mean(embeddings, axis=0))

    async def detect_clusters(
        self,
        documents: List[Dict[str, Any]],
        content_key: str = "content"
    ) -> List[DocumentCluster]:
        """
        Cluster documents by semantic similarity.

        Uses agglomerative clustering with single-linkage:
        - Start with each document as its own cluster
        - Merge clusters where any pair exceeds similarity threshold
        - Continue until no more merges possible

        Args:
            documents: List of documents with content
            content_key: Key to access document content

        Returns:
            List of DocumentClusters
        """
        import time
        start_time = time.time()

        if not documents:
            return []

        # Get embeddings for all documents
        contents = [doc.get(content_key, "") for doc in documents]
        embeddings = await self._get_embeddings_batch(contents)

        # Initialize: each document is its own cluster
        clusters: List[Set[int]] = [{i} for i in range(len(documents))]

        # Build similarity matrix (upper triangular)
        n = len(documents)
        similarity_matrix = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self.similarity_threshold:
                    similarity_matrix[(i, j)] = sim

        # Merge clusters iteratively
        changed = True
        while changed:
            changed = False
            new_clusters = []
            merged = set()

            for i, cluster_i in enumerate(clusters):
                if i in merged:
                    continue

                merged_cluster = set(cluster_i)

                for j, cluster_j in enumerate(clusters):
                    if j <= i or j in merged:
                        continue

                    # Check if any pair of documents should be merged
                    should_merge = False
                    for doc_i in cluster_i:
                        for doc_j in cluster_j:
                            key = (min(doc_i, doc_j), max(doc_i, doc_j))
                            if key in similarity_matrix:
                                should_merge = True
                                break
                        if should_merge:
                            break

                    if should_merge:
                        merged_cluster.update(cluster_j)
                        merged.add(j)
                        changed = True

                new_clusters.append(merged_cluster)
                merged.add(i)

            clusters = new_clusters

        # Build DocumentCluster objects
        result_clusters = []
        for idx, doc_indices in enumerate(clusters):
            if len(doc_indices) < self.min_cluster_size:
                continue

            indices = sorted(doc_indices)
            cluster_docs = [documents[i] for i in indices]
            cluster_embeddings = [embeddings[i] for i in indices]
            centroid = self._calculate_centroid(cluster_embeddings)

            # Calculate similarity to centroid for each doc
            similarities = [
                self._cosine_similarity(emb, centroid)
                for emb in cluster_embeddings
            ]

            # Find most central document (closest to centroid)
            representative_idx = similarities.index(max(similarities))

            cluster = DocumentCluster(
                cluster_id=f"cluster_{idx}",
                documents=cluster_docs,
                embeddings=cluster_embeddings,
                centroid=centroid,
                representative_idx=representative_idx,
                similarity_scores=similarities,
                avg_similarity=sum(similarities) / len(similarities) if similarities else 0.0
            )
            result_clusters.append(cluster)

        duration = (time.time() - start_time) * 1000
        logger.info(
            f"Clustering complete: {len(documents)} docs → {len(result_clusters)} clusters "
            f"(threshold={self.similarity_threshold}) in {duration:.0f}ms"
        )

        return result_clusters

    def select_representatives(
        self,
        clusters: List[DocumentCluster],
        method: SelectionMethod = SelectionMethod.MOST_CENTRAL,
        dig_scores: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Select best representative from each cluster.

        Args:
            clusters: Document clusters
            method: Selection method
            dig_scores: Optional DIG scores for documents (for DIG method)

        Returns:
            List of selected representative documents
        """
        representatives = []

        for cluster in clusters:
            if method == SelectionMethod.MOST_CENTRAL:
                # Already calculated in cluster
                rep = cluster.representative

            elif method == SelectionMethod.LONGEST:
                # Longest content
                idx = max(
                    range(len(cluster.documents)),
                    key=lambda i: len(cluster.documents[i].get("content", ""))
                )
                rep = cluster.documents[idx]

            elif method == SelectionMethod.SHORTEST:
                # Shortest content (most concise)
                idx = min(
                    range(len(cluster.documents)),
                    key=lambda i: len(cluster.documents[i].get("content", ""))
                )
                rep = cluster.documents[idx]

            elif method == SelectionMethod.HIGHEST_QUALITY:
                # Use quality_score if available
                idx = max(
                    range(len(cluster.documents)),
                    key=lambda i: cluster.documents[i].get("quality_score", 0.5)
                )
                rep = cluster.documents[idx]

            elif method == SelectionMethod.INFORMATION_GAIN:
                # Use DIG scores
                if dig_scores:
                    idx = max(
                        range(len(cluster.documents)),
                        key=lambda i: dig_scores.get(
                            cluster.documents[i].get("id") or
                            cluster.documents[i].get("url") or
                            f"doc_{i}",
                            0.0
                        )
                    )
                    rep = cluster.documents[idx]
                else:
                    rep = cluster.representative

            elif method == SelectionMethod.SOURCE_AUTHORITY:
                # Prioritize authoritative sources
                authority_domains = {
                    "arxiv.org": 1.0,
                    "github.com": 0.9,
                    "stackoverflow.com": 0.85,
                    ".edu": 0.8,
                    ".gov": 0.8,
                    "wikipedia.org": 0.7
                }

                def get_authority(doc):
                    url = doc.get("url", "")
                    for domain, score in authority_domains.items():
                        if domain in url:
                            return score
                    return 0.5

                idx = max(
                    range(len(cluster.documents)),
                    key=lambda i: get_authority(cluster.documents[i])
                )
                rep = cluster.documents[idx]

            else:
                rep = cluster.representative

            # Add cluster metadata
            rep = dict(rep)
            rep["cluster_id"] = cluster.cluster_id
            rep["cluster_size"] = cluster.size
            rep["avg_cluster_similarity"] = cluster.avg_similarity

            representatives.append(rep)

        return representatives

    async def deduplicate(
        self,
        documents: List[Dict[str, Any]],
        content_key: str = "content",
        selection_method: SelectionMethod = SelectionMethod.MOST_CENTRAL,
        dig_scores: Optional[Dict[str, float]] = None
    ) -> DeduplicationResult:
        """
        Full deduplication pipeline: cluster and select representatives.

        Args:
            documents: Documents to deduplicate
            content_key: Key for document content
            selection_method: How to select cluster representative
            dig_scores: Optional DIG scores for selection

        Returns:
            DeduplicationResult with deduplicated documents
        """
        import time
        start_time = time.time()

        if not documents:
            return DeduplicationResult(
                original_count=0,
                deduplicated_count=0,
                clusters=[],
                removed_documents=[],
                similarity_threshold=self.similarity_threshold,
                processing_time_ms=0.0
            )

        # Cluster documents
        clusters = await self.detect_clusters(documents, content_key)

        # Select representatives
        representatives = self.select_representatives(
            clusters,
            method=selection_method,
            dig_scores=dig_scores
        )

        # Track removed documents
        rep_ids = {
            doc.get("id") or doc.get("url") or doc.get("content", "")[:100]
            for doc in representatives
        }
        removed = [
            doc for doc in documents
            if (doc.get("id") or doc.get("url") or doc.get("content", "")[:100]) not in rep_ids
        ]

        duration = (time.time() - start_time) * 1000

        result = DeduplicationResult(
            original_count=len(documents),
            deduplicated_count=len(representatives),
            clusters=clusters,
            removed_documents=removed,
            similarity_threshold=self.similarity_threshold,
            processing_time_ms=duration
        )

        logger.info(
            f"Deduplication: {result.original_count} → {result.deduplicated_count} docs "
            f"({result.reduction_ratio:.1%} reduction) in {duration:.0f}ms"
        )

        return result

    async def find_near_duplicates(
        self,
        documents: List[Dict[str, Any]],
        content_key: str = "content",
        threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Find all near-duplicate pairs above threshold.

        Useful for debugging and analysis.

        Returns:
            List of (doc_idx1, doc_idx2, similarity) tuples
        """
        threshold = threshold or self.similarity_threshold

        contents = [doc.get(content_key, "") for doc in documents]
        embeddings = await self._get_embeddings_batch(contents)

        duplicates = []
        n = len(documents)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    duplicates.append((i, j, sim))

        # Sort by similarity descending
        duplicates.sort(key=lambda x: x[2], reverse=True)

        return duplicates

    def clear_cache(self) -> int:
        """Clear embedding cache."""
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        return count


# Singleton instance
_redundancy_detector: Optional[RedundancyDetector] = None


def get_redundancy_detector(
    ollama_url: str = "http://localhost:11434",
    similarity_threshold: float = 0.85
) -> RedundancyDetector:
    """Get or create the redundancy detector singleton."""
    global _redundancy_detector
    if _redundancy_detector is None:
        _redundancy_detector = RedundancyDetector(
            ollama_url=ollama_url,
            similarity_threshold=similarity_threshold
        )
    return _redundancy_detector
