"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

Implements hierarchical document summarization for improved retrieval.
Based on "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (ICLR 2024)

Key innovations:
- Recursive summarization builds tree of increasingly abstract summaries
- Clustering at each level groups semantically similar content
- Multi-level retrieval traverses tree from root to leaves
- 20% improvement on QuALITY benchmark over flat retrieval

Architecture:
    Level 0: Original documents (leaves)
    Level 1: Cluster summaries (first abstraction)
    Level 2: Meta-summaries (higher abstraction)
    ...
    Level N: Root summary (most abstract)

Author: Claude Code
Date: December 2025
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import httpx
import numpy as np

from .llm_config import get_llm_config

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClusteringMethod(str, Enum):
    """Clustering methods for RAPTOR."""
    AGGLOMERATIVE = "agglomerative"  # Hierarchical, hard assignments
    GMM = "gmm"  # Gaussian Mixture Model, soft assignments (paper default)
    SEMANTIC = "semantic"  # Pure embedding similarity
    HYBRID = "hybrid"  # Combine semantic + lexical


class RetrievalMode(str, Enum):
    """Retrieval modes for RAPTOR."""
    TREE_GUIDED = "tree_guided"  # Traverse tree from root to leaves (original)
    COLLAPSED = "collapsed"  # Flatten tree, rank all nodes (paper alternative)


class SummarizationStyle(str, Enum):
    """Summarization styles."""
    ABSTRACTIVE = "abstractive"  # Rewrite in new words
    EXTRACTIVE = "extractive"  # Select key sentences
    HYBRID = "hybrid"  # Combine both


@dataclass
class RAPTORConfig:
    """Configuration for RAPTOR tree building."""
    # Model settings (None = load from llm_config)
    summarization_model: Optional[str] = None
    embedding_model: str = "mxbai-embed-large"
    ollama_url: str = "http://localhost:11434"

    def __post_init__(self):
        """Load defaults from central llm_config if not specified."""
        if self.summarization_model is None:
            config = get_llm_config()
            self.summarization_model = config.utility.raptor_summarizer.model

    # Tree structure
    max_levels: int = 5  # Maximum tree depth
    min_cluster_size: int = 2  # Minimum docs per cluster
    max_cluster_size: int = 10  # Maximum docs per cluster
    target_clusters_per_level: int = 5  # Target cluster count per level

    # Clustering
    clustering_method: ClusteringMethod = ClusteringMethod.GMM  # Paper default
    similarity_threshold: float = 0.7  # Minimum similarity for semantic clustering

    # GMM + UMAP settings (paper: arxiv:2401.18059)
    gmm_soft_threshold: float = 0.1  # Assign to cluster if prob > threshold (paper default)
    umap_n_components: int = 10  # Reduce to 10 dimensions before GMM (paper default)
    umap_n_neighbors: int = 15  # UMAP neighborhood size
    umap_min_dist: float = 0.1  # UMAP minimum distance

    # Summarization
    summarization_style: SummarizationStyle = SummarizationStyle.ABSTRACTIVE
    max_summary_tokens: int = 512
    temperature: float = 0.3  # Lower for factual summaries

    # Retrieval
    top_k_per_level: int = 3  # Top-k nodes per level during retrieval
    traverse_to_leaves: bool = True  # Follow paths to original docs
    retrieval_mode: RetrievalMode = RetrievalMode.TREE_GUIDED  # Default mode
    token_budget: int = 2000  # Max tokens for collapsed retrieval (paper default)
    tokens_per_char: float = 0.25  # Approximate tokens per character

    # Performance
    batch_size: int = 5  # Docs per summarization batch
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class TreeNode:
    """Node in the RAPTOR tree."""
    id: str
    content: str
    level: int  # 0 = leaf (original doc), higher = more abstract
    embedding: Optional[np.ndarray] = None
    children: List[str] = field(default_factory=list)  # Child node IDs
    parent: Optional[str] = None  # Parent node ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None and self.level > 0

    @property
    def token_count(self) -> int:
        """Estimate token count (approximately 4 chars per token)."""
        return max(1, len(self.content) // 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "level": self.level,
            "children": self.children,
            "parent": self.parent,
            "is_leaf": self.is_leaf,
            "is_root": self.is_root,
            "metadata": self.metadata,
        }


@dataclass
class RAPTORTree:
    """The complete RAPTOR tree structure."""
    nodes: Dict[str, TreeNode] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)  # May have multiple roots
    max_level: int = 0
    build_time_ms: float = 0.0
    total_documents: int = 0

    def get_nodes_at_level(self, level: int) -> List[TreeNode]:
        """Get all nodes at a specific level."""
        return [n for n in self.nodes.values() if n.level == level]

    def get_path_to_root(self, node_id: str) -> List[TreeNode]:
        """Get path from node to root."""
        path = []
        current = self.nodes.get(node_id)
        while current:
            path.append(current)
            current = self.nodes.get(current.parent) if current.parent else None
        return path

    def get_leaves_under(self, node_id: str) -> List[TreeNode]:
        """Get all leaf nodes under a given node."""
        node = self.nodes.get(node_id)
        if not node:
            return []

        if node.is_leaf:
            return [node]

        leaves = []
        for child_id in node.children:
            leaves.extend(self.get_leaves_under(child_id))
        return leaves

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "total_documents": self.total_documents,
            "max_level": self.max_level,
            "root_count": len(self.root_ids),
            "build_time_ms": self.build_time_ms,
            "levels": {
                level: len(self.get_nodes_at_level(level))
                for level in range(self.max_level + 1)
            },
        }


@dataclass
class RetrievalResult:
    """Result from RAPTOR retrieval."""
    query: str
    relevant_nodes: List[TreeNode]
    leaf_documents: List[TreeNode]
    traversal_path: List[str]  # Node IDs visited
    scores: Dict[str, float]  # Node ID -> relevance score
    retrieval_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "nodes_found": len(self.relevant_nodes),
            "documents_found": len(self.leaf_documents),
            "traversal_depth": len(set(n.level for n in self.relevant_nodes)),
            "retrieval_time_ms": self.retrieval_time_ms,
            "top_scores": sorted(self.scores.items(), key=lambda x: -x[1])[:5],
        }


class RAPTORBuilder:
    """Builds RAPTOR trees from documents."""

    def __init__(self, config: Optional[RAPTORConfig] = None):
        self.config = config or RAPTORConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.ollama_url,
                timeout=120.0,
            )
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _embed_text(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama."""
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "prompt": text[:8000],  # Limit for embedding models
                },
            )
            response.raise_for_status()
            embedding = np.array(response.json()["embedding"])

            # Cache result
            if self.config.enable_cache:
                self._embedding_cache[text_hash] = embedding

            return embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # Return zero vector as fallback
            return np.zeros(1024)

    async def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts in parallel."""
        embeddings = await asyncio.gather(
            *[self._embed_text(text) for text in texts]
        )
        return list(embeddings)

    async def _summarize(self, texts: List[str], context: str = "") -> str:
        """Summarize multiple texts into one."""
        combined = "\n\n---\n\n".join(texts)

        if self.config.summarization_style == SummarizationStyle.EXTRACTIVE:
            prompt = f"""Extract the most important information from these documents.
Keep key facts, entities, and relationships. Be concise but comprehensive.

Context: {context}

Documents:
{combined}

Key information (extractive summary):"""
        else:
            prompt = f"""Summarize the following documents into a coherent, comprehensive summary.
Capture all important information, entities, and relationships.
The summary should stand alone as a representation of the content.

Context: {context}

Documents:
{combined}

Comprehensive summary:"""

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/generate",
                json={
                    "model": self.config.summarization_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_summary_tokens,
                    },
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback: concatenate first sentences
            fallback = " ".join(t.split(".")[0] + "." for t in texts[:3])
            return fallback[:500]

    def _cluster_nodes(
        self,
        nodes: List[TreeNode],
        embeddings: np.ndarray,
    ) -> List[List[TreeNode]]:
        """
        Cluster nodes based on embeddings.

        Uses GMM + UMAP for soft clustering (paper default) or falls back to
        agglomerative clustering if GMM/UMAP not available.
        """
        n_nodes = len(nodes)

        if n_nodes <= self.config.min_cluster_size:
            return [nodes]

        # Calculate target number of clusters
        n_clusters = max(
            1,
            min(
                n_nodes // self.config.min_cluster_size,
                self.config.target_clusters_per_level,
            )
        )

        if not SKLEARN_AVAILABLE:
            # Fallback: simple grouping by similarity
            return self._simple_clustering(nodes, embeddings, n_clusters)

        # Select clustering method
        if self.config.clustering_method == ClusteringMethod.GMM:
            return self._gmm_clustering(nodes, embeddings, n_clusters)
        elif self.config.clustering_method == ClusteringMethod.AGGLOMERATIVE:
            return self._agglomerative_clustering(nodes, embeddings, n_clusters)
        else:
            return self._simple_clustering(nodes, embeddings, n_clusters)

    def _gmm_clustering(
        self,
        nodes: List[TreeNode],
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> List[List[TreeNode]]:
        """
        GMM + UMAP soft clustering (RAPTOR paper default).

        Key differences from hard clustering:
        1. Uses UMAP to reduce dimensionality before clustering
        2. Assigns nodes to MULTIPLE clusters if probability > threshold
        3. Produces overlapping clusters for better tree structure

        Paper reference: arxiv:2401.18059 (Figure 2)
        """
        n_nodes = len(nodes)

        try:
            # Step 1: Reduce dimensionality with UMAP
            reduced = embeddings
            if UMAP_AVAILABLE and n_nodes > self.config.umap_n_components:
                try:
                    # Adjust n_neighbors if we have fewer samples
                    n_neighbors = min(self.config.umap_n_neighbors, n_nodes - 1)
                    if n_neighbors < 2:
                        n_neighbors = 2

                    umap = UMAP(
                        n_components=min(self.config.umap_n_components, n_nodes - 1),
                        n_neighbors=n_neighbors,
                        min_dist=self.config.umap_min_dist,
                        metric="cosine",
                        random_state=42,
                    )
                    reduced = umap.fit_transform(embeddings)
                    logger.debug(f"UMAP reduced {embeddings.shape[1]}d -> {reduced.shape[1]}d")
                except Exception as e:
                    logger.warning(f"UMAP failed, using raw embeddings: {e}")
                    reduced = embeddings
            else:
                logger.debug("UMAP not available or too few samples, using raw embeddings")

            # Step 2: Fit GMM for soft clustering
            # Ensure n_clusters doesn't exceed n_nodes
            actual_n_clusters = min(n_clusters, n_nodes)

            gmm = GaussianMixture(
                n_components=actual_n_clusters,
                covariance_type='full',
                random_state=42,
                max_iter=100,
            )
            gmm.fit(reduced)

            # Step 3: Get soft assignments (probabilities for each cluster)
            probs = gmm.predict_proba(reduced)  # Shape: (n_nodes, n_clusters)

            # Step 4: Assign nodes to clusters where probability > threshold
            # This is the key innovation: nodes can belong to MULTIPLE clusters
            clusters: Dict[int, List[TreeNode]] = {i: [] for i in range(actual_n_clusters)}

            for node_idx, node in enumerate(nodes):
                assigned = False
                for cluster_idx in range(actual_n_clusters):
                    if probs[node_idx, cluster_idx] > self.config.gmm_soft_threshold:
                        clusters[cluster_idx].append(node)
                        assigned = True

                # Fallback: assign to highest probability cluster if none exceed threshold
                if not assigned:
                    best_cluster = int(np.argmax(probs[node_idx]))
                    clusters[best_cluster].append(node)

            # Filter out empty clusters
            result = [cluster for cluster in clusters.values() if len(cluster) > 0]

            # Log soft assignment stats
            multi_assigned = sum(
                1 for i in range(n_nodes)
                if sum(probs[i] > self.config.gmm_soft_threshold) > 1
            )
            logger.debug(
                f"GMM soft clustering: {n_nodes} nodes -> {len(result)} clusters, "
                f"{multi_assigned} nodes assigned to multiple clusters"
            )

            return result

        except Exception as e:
            logger.warning(f"GMM clustering failed: {e}, falling back to agglomerative")
            return self._agglomerative_clustering(nodes, embeddings, n_clusters)

    def _agglomerative_clustering(
        self,
        nodes: List[TreeNode],
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> List[List[TreeNode]]:
        """
        Agglomerative (hierarchical) hard clustering.

        Each node belongs to exactly one cluster.
        """
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)

            # Group nodes by cluster
            clusters: Dict[int, List[TreeNode]] = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(nodes[i])

            return list(clusters.values())
        except Exception as e:
            logger.error(f"Agglomerative clustering failed: {e}, using fallback")
            return self._simple_clustering(nodes, embeddings, n_clusters)

    def _simple_clustering(
        self,
        nodes: List[TreeNode],
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> List[List[TreeNode]]:
        """Simple fallback clustering based on similarity."""
        if len(nodes) <= n_clusters:
            return [[n] for n in nodes]

        # Group by nearest neighbor similarity
        used = set()
        clusters = []

        for i, node in enumerate(nodes):
            if i in used:
                continue

            cluster = [node]
            used.add(i)

            # Find similar nodes
            for j, other in enumerate(nodes):
                if j in used or len(cluster) >= self.config.max_cluster_size:
                    continue

                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )

                if sim >= self.config.similarity_threshold:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

            if len(clusters) >= n_clusters:
                break

        # Add remaining nodes to nearest cluster
        for i, node in enumerate(nodes):
            if i not in used:
                if clusters:
                    clusters[-1].append(node)
                else:
                    clusters.append([node])

        return clusters

    async def build_tree(
        self,
        documents: List[Tuple[str, str]],  # (doc_id, content)
        context: str = "",
    ) -> RAPTORTree:
        """Build a RAPTOR tree from documents."""
        start_time = time.time()

        tree = RAPTORTree(total_documents=len(documents))

        if not documents:
            return tree

        # Level 0: Create leaf nodes from documents
        leaf_nodes = []
        for doc_id, content in documents:
            node = TreeNode(
                id=f"leaf_{doc_id}",
                content=content,
                level=0,
                metadata={"original_id": doc_id},
            )
            tree.nodes[node.id] = node
            leaf_nodes.append(node)

        # Embed all leaf nodes
        logger.info(f"Embedding {len(leaf_nodes)} documents...")
        texts = [n.content for n in leaf_nodes]
        embeddings = await self._embed_batch(texts)
        for node, emb in zip(leaf_nodes, embeddings):
            node.embedding = emb

        # Build tree levels recursively
        current_level_nodes = leaf_nodes
        current_embeddings = np.array(embeddings)
        level = 0

        while len(current_level_nodes) > 1 and level < self.config.max_levels:
            level += 1
            logger.info(f"Building level {level} from {len(current_level_nodes)} nodes...")

            # Cluster current level
            clusters = self._cluster_nodes(current_level_nodes, current_embeddings)

            if len(clusters) == len(current_level_nodes):
                # No more clustering possible
                break

            # Create parent nodes for each cluster
            next_level_nodes = []
            next_embeddings = []

            for i, cluster in enumerate(clusters):
                # Summarize cluster content
                cluster_texts = [n.content for n in cluster]
                summary = await self._summarize(cluster_texts, context)

                # Create parent node
                parent_id = f"level{level}_cluster{i}"
                parent = TreeNode(
                    id=parent_id,
                    content=summary,
                    level=level,
                    children=[n.id for n in cluster],
                    metadata={
                        "cluster_size": len(cluster),
                        "child_levels": list(set(n.level for n in cluster)),
                    },
                )

                # Embed parent
                parent.embedding = await self._embed_text(summary)

                # Update children to point to parent
                for child in cluster:
                    child.parent = parent_id

                tree.nodes[parent_id] = parent
                next_level_nodes.append(parent)
                next_embeddings.append(parent.embedding)

            current_level_nodes = next_level_nodes
            current_embeddings = np.array(next_embeddings)
            tree.max_level = level

        # Mark root nodes
        tree.root_ids = [n.id for n in current_level_nodes if n.parent is None]

        tree.build_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Built RAPTOR tree: {len(tree.nodes)} nodes, "
            f"{tree.max_level + 1} levels, {tree.build_time_ms:.0f}ms"
        )

        return tree


class RAPTORRetriever:
    """Retrieves relevant content from RAPTOR trees."""

    def __init__(
        self,
        tree: RAPTORTree,
        config: Optional[RAPTORConfig] = None,
    ):
        self.tree = tree
        self.config = config or RAPTORConfig()
        self._client: Optional[httpx.AsyncClient] = None

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
        node_emb: np.ndarray,
    ) -> float:
        """Compute cosine similarity."""
        if node_emb is None or query_emb is None:
            return 0.0
        return float(np.dot(query_emb, node_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(node_emb) + 1e-8
        ))

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve relevant content using tree-guided search."""
        start_time = time.time()

        query_emb = await self._embed_query(query)

        # Strategy: Search from root down, collecting best matches at each level
        relevant_nodes = []
        traversal_path = []
        scores: Dict[str, float] = {}

        # Start from roots (or all if no tree structure)
        if self.tree.root_ids:
            current_ids = self.tree.root_ids
        else:
            current_ids = list(self.tree.nodes.keys())

        # Traverse tree
        for level in range(self.tree.max_level, -1, -1):
            level_nodes = [
                self.tree.nodes[nid]
                for nid in current_ids
                if nid in self.tree.nodes and self.tree.nodes[nid].level == level
            ]

            if not level_nodes:
                continue

            # Score nodes at this level
            for node in level_nodes:
                score = self._compute_similarity(query_emb, node.embedding)
                scores[node.id] = score

            # Select top-k at this level
            sorted_nodes = sorted(level_nodes, key=lambda n: scores.get(n.id, 0), reverse=True)
            top_nodes = sorted_nodes[:self.config.top_k_per_level]

            for node in top_nodes:
                relevant_nodes.append(node)
                traversal_path.append(node.id)

            # Prepare next level: children of selected nodes
            next_ids = []
            for node in top_nodes:
                next_ids.extend(node.children)

            if not next_ids:
                break

            current_ids = next_ids

        # Get leaf documents
        leaf_documents = []
        if self.config.traverse_to_leaves:
            seen_leaves = set()
            for node in relevant_nodes:
                leaves = self.tree.get_leaves_under(node.id)
                for leaf in leaves:
                    if leaf.id not in seen_leaves:
                        leaf_documents.append(leaf)
                        seen_leaves.add(leaf.id)

        # Sort by score and limit to top_k
        relevant_nodes = sorted(
            relevant_nodes,
            key=lambda n: scores.get(n.id, 0),
            reverse=True,
        )[:top_k]

        leaf_documents = sorted(
            leaf_documents,
            key=lambda n: max(
                scores.get(p.id, 0)
                for p in self.tree.get_path_to_root(n.id)
            ),
            reverse=True,
        )[:top_k]

        return RetrievalResult(
            query=query,
            relevant_nodes=relevant_nodes,
            leaf_documents=leaf_documents,
            traversal_path=traversal_path,
            scores=scores,
            retrieval_time_ms=(time.time() - start_time) * 1000,
        )

    async def retrieve_collapsed(
        self,
        query: str,
        token_budget: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Collapsed tree retrieval (paper Section 3.3).

        Treats all nodes from all levels equally, ranks by similarity,
        and accumulates until token budget is reached.

        This is an alternative to tree-guided retrieval that can sometimes
        find relevant content at intermediate levels that tree traversal misses.

        Args:
            query: Search query
            token_budget: Maximum tokens to retrieve (default: config.token_budget)
            top_k: Optional hard limit on nodes (in addition to token budget)

        Returns:
            RetrievalResult with nodes ranked by similarity
        """
        start_time = time.time()
        budget = token_budget or self.config.token_budget

        query_emb = await self._embed_query(query)

        # Score ALL nodes in the tree (collapsed view)
        all_nodes = list(self.tree.nodes.values())
        scores: Dict[str, float] = {}

        for node in all_nodes:
            score = self._compute_similarity(query_emb, node.embedding)
            scores[node.id] = score

        # Sort by similarity (descending)
        sorted_nodes = sorted(all_nodes, key=lambda n: scores.get(n.id, 0), reverse=True)

        # Accumulate until token budget is exhausted
        retrieved_nodes = []
        tokens_used = 0
        traversal_path = []

        for node in sorted_nodes:
            node_tokens = node.token_count

            # Check token budget
            if tokens_used + node_tokens > budget:
                # If we have nothing yet, take at least one node
                if not retrieved_nodes:
                    retrieved_nodes.append(node)
                    traversal_path.append(node.id)
                    tokens_used += node_tokens
                break

            retrieved_nodes.append(node)
            traversal_path.append(node.id)
            tokens_used += node_tokens

            # Check top_k limit if specified
            if top_k and len(retrieved_nodes) >= top_k:
                break

        # Separate leaf documents from summary nodes
        leaf_documents = [n for n in retrieved_nodes if n.is_leaf]
        summary_nodes = [n for n in retrieved_nodes if not n.is_leaf]

        # If we have summaries but no leaves, get leaves under top summaries
        if summary_nodes and not leaf_documents and self.config.traverse_to_leaves:
            seen_leaves = set()
            for node in summary_nodes[:3]:  # Top 3 summaries
                leaves = self.tree.get_leaves_under(node.id)
                for leaf in leaves:
                    if leaf.id not in seen_leaves:
                        leaf_documents.append(leaf)
                        seen_leaves.add(leaf.id)

        logger.debug(
            f"Collapsed retrieval: {len(retrieved_nodes)} nodes, "
            f"{tokens_used}/{budget} tokens, "
            f"{len(leaf_documents)} leaves"
        )

        return RetrievalResult(
            query=query,
            relevant_nodes=retrieved_nodes,
            leaf_documents=leaf_documents,
            traversal_path=traversal_path,
            scores=scores,
            retrieval_time_ms=(time.time() - start_time) * 1000,
        )

    async def retrieve_auto(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Automatically select retrieval mode based on configuration.

        Uses config.retrieval_mode to determine whether to use
        tree-guided or collapsed retrieval.
        """
        if self.config.retrieval_mode == RetrievalMode.COLLAPSED:
            return await self.retrieve_collapsed(query, top_k=top_k)
        else:
            return await self.retrieve(query, top_k=top_k)


# Convenience functions
_builder_instance: Optional[RAPTORBuilder] = None
_trees: Dict[str, RAPTORTree] = {}


async def get_raptor_builder(
    config: Optional[RAPTORConfig] = None,
) -> RAPTORBuilder:
    """Get or create RAPTOR builder instance."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = RAPTORBuilder(config)
    return _builder_instance


async def build_raptor_tree(
    documents: List[Tuple[str, str]],
    tree_id: str = "default",
    context: str = "",
    config: Optional[RAPTORConfig] = None,
) -> RAPTORTree:
    """Build and store a RAPTOR tree."""
    builder = await get_raptor_builder(config)
    tree = await builder.build_tree(documents, context)
    _trees[tree_id] = tree
    return tree


async def raptor_retrieve(
    query: str,
    tree_id: str = "default",
    top_k: int = 10,
    config: Optional[RAPTORConfig] = None,
    mode: Optional[RetrievalMode] = None,
) -> RetrievalResult:
    """
    Retrieve from a stored RAPTOR tree.

    Args:
        query: Search query
        tree_id: ID of the tree to search
        top_k: Maximum nodes to retrieve
        config: Optional retrieval configuration
        mode: Retrieval mode (tree_guided or collapsed). If None, uses config default.

    Returns:
        RetrievalResult with relevant nodes and documents
    """
    tree = _trees.get(tree_id)
    if tree is None:
        raise ValueError(f"Tree '{tree_id}' not found. Build it first with build_raptor_tree().")

    retriever = RAPTORRetriever(tree, config)

    # Determine mode
    effective_mode = mode or (config.retrieval_mode if config else RetrievalMode.TREE_GUIDED)

    if effective_mode == RetrievalMode.COLLAPSED:
        result = await retriever.retrieve_collapsed(query, top_k=top_k)
    else:
        result = await retriever.retrieve(query, top_k)

    await retriever.close()
    return result


async def raptor_retrieve_collapsed(
    query: str,
    tree_id: str = "default",
    token_budget: int = 2000,
    top_k: Optional[int] = None,
    config: Optional[RAPTORConfig] = None,
) -> RetrievalResult:
    """
    Collapsed tree retrieval from a stored RAPTOR tree.

    Treats all nodes equally, ranks by similarity, accumulates until token budget.

    Args:
        query: Search query
        tree_id: ID of the tree to search
        token_budget: Maximum tokens to retrieve (default: 2000)
        top_k: Optional hard limit on nodes
        config: Optional retrieval configuration

    Returns:
        RetrievalResult with relevant nodes ranked by similarity
    """
    tree = _trees.get(tree_id)
    if tree is None:
        raise ValueError(f"Tree '{tree_id}' not found. Build it first with build_raptor_tree().")

    retriever = RAPTORRetriever(tree, config)
    result = await retriever.retrieve_collapsed(query, token_budget=token_budget, top_k=top_k)
    await retriever.close()
    return result


def get_raptor_tree(tree_id: str = "default") -> Optional[RAPTORTree]:
    """Get a stored RAPTOR tree."""
    return _trees.get(tree_id)


def list_raptor_trees() -> Dict[str, Dict[str, Any]]:
    """List all stored RAPTOR trees."""
    return {
        tree_id: tree.to_dict()
        for tree_id, tree in _trees.items()
    }
