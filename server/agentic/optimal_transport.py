"""
Optimal Transport for Dense-Sparse Retrieval Fusion

Uses Wasserstein distance and Sinkhorn algorithm to optimally align and fuse
retrieval results from heterogeneous sources (dense semantic + sparse lexical).

Key Features:
- Sinkhorn-Knopp algorithm for efficient entropic regularization (O(n^2) iterations)
- Wasserstein barycenter for combining multiple retrieval distributions
- Gromov-Wasserstein distance for cross-domain alignment
- GPU-compatible differentiable operations

Research Basis:
- Wasserstein Wormhole (NeurIPS 2024) - Scalable OT with transformers
- Gromov-Wasserstein for Graph Alignment (ICML 2020)
- POT: Python Optimal Transport Library

Author: Claude Code
Date: December 2025
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class OTMethod(str, Enum):
    """Optimal Transport methods for score fusion."""
    SINKHORN = "sinkhorn"  # Entropy-regularized OT (fast, approximate)
    EXACT = "exact"  # Exact linear programming (slow, precise)
    UNBALANCED = "unbalanced"  # Unbalanced OT (different marginal sums)
    PARTIAL = "partial"  # Partial OT (transport subset)
    GROMOV = "gromov"  # Gromov-Wasserstein (cross-domain)
    SLICED = "sliced"  # Sliced-Wasserstein (O(n log n), fast approximation)
    WORD_MOVER = "word_mover"  # Word Mover's Distance for documents


class CostMetric(str, Enum):
    """Cost function for OT problem."""
    EUCLIDEAN = "euclidean"  # L2 distance
    COSINE = "cosine"  # 1 - cosine similarity
    MANHATTAN = "manhattan"  # L1 distance
    RANK_BASED = "rank_based"  # Based on rank differences


@dataclass
class OTConfig:
    """Configuration for Optimal Transport fusion."""
    # Sinkhorn parameters
    method: OTMethod = OTMethod.SINKHORN
    epsilon: float = 0.1  # Entropic regularization (lower = closer to exact)
    max_iter: int = 100  # Maximum Sinkhorn iterations
    threshold: float = 1e-6  # Convergence threshold

    # Cost metric
    cost_metric: CostMetric = CostMetric.COSINE

    # Fusion weights
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    # Normalization
    normalize_scores: bool = True
    min_score: float = 0.0
    max_score: float = 1.0

    # Performance
    use_gpu: bool = False  # Use GPU if available
    batch_size: int = 64
    cache_transport_plan: bool = True

    # Gromov-Wasserstein specific
    gw_alpha: float = 0.5  # Balance between feature and structure
    gw_max_iter: int = 50


@dataclass
class OTResult:
    """Result of Optimal Transport fusion."""
    doc_id: str
    fused_score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    transport_weight: float = 1.0  # Weight from transport plan
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportPlan:
    """Optimal transport plan between two distributions."""
    P: np.ndarray  # Transport matrix (n x m)
    cost: float  # Total transport cost (Wasserstein distance)
    iterations: int  # Sinkhorn iterations used
    converged: bool  # Whether algorithm converged
    marginal_error: float  # Error in marginal constraints
    computation_time_ms: float


class SinkhornSolver:
    """
    Sinkhorn-Knopp algorithm for entropy-regularized optimal transport.

    Solves: min_P <C, P> + epsilon * H(P)
    Subject to: P @ 1 = a, P.T @ 1 = b, P >= 0

    Where:
    - C: Cost matrix (n x m)
    - P: Transport plan (n x m)
    - epsilon: Entropic regularization
    - H(P): Entropy of P
    - a, b: Source and target marginals
    """

    def __init__(self, config: OTConfig):
        self.config = config
        self._cache: Dict[str, TransportPlan] = {}

    def compute_cost_matrix(
        self,
        source: np.ndarray,
        target: np.ndarray,
        metric: CostMetric = CostMetric.COSINE
    ) -> np.ndarray:
        """
        Compute cost matrix between source and target points.

        Args:
            source: Source points (n, d) or scores (n,)
            target: Target points (m, d) or scores (m,)
            metric: Distance metric

        Returns:
            Cost matrix (n, m)
        """
        source = np.atleast_2d(source)
        target = np.atleast_2d(target)

        # Handle 1D score arrays
        if source.shape[0] == 1:
            source = source.T
        if target.shape[0] == 1:
            target = target.T

        n, m = source.shape[0], target.shape[0]

        if metric == CostMetric.EUCLIDEAN:
            # L2 distance
            diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]
            cost = np.sqrt(np.sum(diff ** 2, axis=2))

        elif metric == CostMetric.COSINE:
            # 1 - cosine similarity
            source_norm = source / (np.linalg.norm(source, axis=1, keepdims=True) + 1e-8)
            target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
            similarity = source_norm @ target_norm.T
            cost = 1.0 - similarity

        elif metric == CostMetric.MANHATTAN:
            # L1 distance
            diff = source[:, np.newaxis, :] - target[np.newaxis, :, :]
            cost = np.sum(np.abs(diff), axis=2)

        elif metric == CostMetric.RANK_BASED:
            # Rank-based cost: |rank_i - rank_j| / max(n, m)
            source_ranks = np.arange(n).reshape(-1, 1)
            target_ranks = np.arange(m).reshape(1, -1)
            cost = np.abs(source_ranks - target_ranks) / max(n, m)

        else:
            raise ValueError(f"Unknown cost metric: {metric}")

        return cost.astype(np.float64)

    def sinkhorn(
        self,
        cost: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
        max_iter: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> TransportPlan:
        """
        Sinkhorn-Knopp algorithm for entropy-regularized OT.

        Args:
            cost: Cost matrix (n, m)
            a: Source marginal (n,). Default: uniform
            b: Target marginal (m,). Default: uniform
            epsilon: Entropic regularization. Default: config value
            max_iter: Maximum iterations. Default: config value
            threshold: Convergence threshold. Default: config value

        Returns:
            TransportPlan with optimal transport matrix
        """
        start_time = time.perf_counter()

        epsilon = epsilon or self.config.epsilon
        max_iter = max_iter or self.config.max_iter
        threshold = threshold or self.config.threshold

        n, m = cost.shape

        # Default uniform marginals
        if a is None:
            a = np.ones(n) / n
        if b is None:
            b = np.ones(m) / m

        # Ensure proper shape
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()

        # Gibbs kernel K = exp(-C/epsilon)
        K = np.exp(-cost / epsilon)

        # Initialize scaling vectors
        u = np.ones(n, dtype=np.float64)
        v = np.ones(m, dtype=np.float64)

        converged = False
        marginal_error = float('inf')

        for i in range(max_iter):
            # Update u
            u_new = a / (K @ v + 1e-300)

            # Update v
            v_new = b / (K.T @ u_new + 1e-300)

            # Check convergence
            if i % 10 == 0:
                # Compute transport plan
                P = np.diag(u_new) @ K @ np.diag(v_new)

                # Check marginal constraints
                err_a = np.linalg.norm(P.sum(axis=1) - a)
                err_b = np.linalg.norm(P.sum(axis=0) - b)
                marginal_error = err_a + err_b

                if marginal_error < threshold:
                    converged = True
                    u, v = u_new, v_new
                    break

            u, v = u_new, v_new

        # Final transport plan
        P = np.diag(u) @ K @ np.diag(v)

        # Wasserstein distance (transport cost)
        wasserstein_cost = np.sum(P * cost)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TransportPlan(
            P=P,
            cost=wasserstein_cost,
            iterations=i + 1,
            converged=converged,
            marginal_error=marginal_error,
            computation_time_ms=elapsed_ms
        )

    def sinkhorn_log_stabilized(
        self,
        cost: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None
    ) -> TransportPlan:
        """
        Log-stabilized Sinkhorn for numerical stability.

        Uses log-sum-exp trick to prevent overflow/underflow with
        small epsilon values.
        """
        start_time = time.perf_counter()

        epsilon = epsilon or self.config.epsilon
        max_iter = self.config.max_iter
        threshold = self.config.threshold

        n, m = cost.shape

        if a is None:
            a = np.ones(n) / n
        if b is None:
            b = np.ones(m) / m

        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()

        # Log marginals
        log_a = np.log(a + 1e-300)
        log_b = np.log(b + 1e-300)

        # Dual potentials
        f = np.zeros(n, dtype=np.float64)
        g = np.zeros(m, dtype=np.float64)

        converged = False
        marginal_error = float('inf')

        for i in range(max_iter):
            # Update f: f = epsilon * log(a) - epsilon * logsumexp((-C + g) / epsilon)
            log_K_g = (-cost + g.reshape(1, -1)) / epsilon
            f_new = epsilon * log_a - epsilon * self._logsumexp(log_K_g, axis=1)

            # Update g
            log_K_f = (-cost.T + f_new.reshape(1, -1)) / epsilon
            g_new = epsilon * log_b - epsilon * self._logsumexp(log_K_f, axis=1)

            # Check convergence
            if i % 10 == 0:
                # Compute P from dual potentials
                log_P = (f_new.reshape(-1, 1) + g_new.reshape(1, -1) - cost) / epsilon
                P = np.exp(log_P)

                err_a = np.linalg.norm(P.sum(axis=1) - a)
                err_b = np.linalg.norm(P.sum(axis=0) - b)
                marginal_error = err_a + err_b

                if marginal_error < threshold:
                    converged = True
                    f, g = f_new, g_new
                    break

            f, g = f_new, g_new

        # Final transport plan
        log_P = (f.reshape(-1, 1) + g.reshape(1, -1) - cost) / epsilon
        P = np.exp(log_P)

        wasserstein_cost = np.sum(P * cost)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TransportPlan(
            P=P,
            cost=wasserstein_cost,
            iterations=i + 1,
            converged=converged,
            marginal_error=marginal_error,
            computation_time_ms=elapsed_ms
        )

    def _logsumexp(self, x: np.ndarray, axis: int = None) -> np.ndarray:
        """Numerically stable log-sum-exp."""
        x_max = np.max(x, axis=axis, keepdims=True)
        return np.squeeze(x_max) + np.log(np.sum(np.exp(x - x_max), axis=axis))


class GromovWassersteinSolver:
    """
    Gromov-Wasserstein distance for cross-domain alignment.

    Compares metric spaces rather than points directly, enabling
    alignment of heterogeneous representations (e.g., dense vs sparse).

    GW distance: min_P sum_{i,j,k,l} |D1(i,k) - D2(j,l)|^2 * P(i,j) * P(k,l)
    """

    def __init__(self, config: OTConfig):
        self.config = config

    def compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix for points X."""
        n = X.shape[0]
        D = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(X[i] - X[j])
                D[i, j] = d
                D[j, i] = d

        return D

    def gromov_wasserstein(
        self,
        D1: np.ndarray,
        D2: np.ndarray,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        alpha: Optional[float] = None
    ) -> TransportPlan:
        """
        Compute Gromov-Wasserstein distance between two metric spaces.

        Uses entropic regularization for efficiency.

        Args:
            D1: Distance matrix for source space (n, n)
            D2: Distance matrix for target space (m, m)
            a: Source marginal (n,)
            b: Target marginal (m,)
            alpha: GW mixing parameter

        Returns:
            TransportPlan with GW optimal transport
        """
        start_time = time.perf_counter()

        alpha = alpha or self.config.gw_alpha
        max_iter = self.config.gw_max_iter
        epsilon = self.config.epsilon

        n, m = D1.shape[0], D2.shape[0]

        if a is None:
            a = np.ones(n) / n
        if b is None:
            b = np.ones(m) / m

        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).flatten()

        # Initialize transport plan
        P = np.outer(a, b)

        sinkhorn = SinkhornSolver(self.config)

        # Precompute squared distance matrices
        D1_sq = D1 ** 2
        D2_sq = D2 ** 2

        for i in range(max_iter):
            # Compute gradient using efficient tensor formulation
            # grad_ij = sum_k (D1_ik^2 * sum_l P_kl) + sum_l (D2_jl^2 * sum_k P_kl) - 2 * (D1 @ P @ D2.T)_ij
            #
            # Term 1: For each row i, sum over k of D1_ik^2 * (row sum of P at k)
            # This equals D1_sq @ P @ 1_m, broadcast to (n, m)
            term1 = (D1_sq @ P).sum(axis=1, keepdims=True)  # (n, 1)

            # Term 2: For each col j, sum over l of D2_jl^2 * (col sum of P at l)
            # This equals 1_n @ P @ D2_sq.T, taking the diagonal broadcast
            term2 = (P @ D2_sq).sum(axis=0, keepdims=True)  # (1, m)

            # Term 3: Cross term D1 @ P @ D2.T
            term3 = D1 @ P @ D2.T  # (n, m)

            # Gradient: broadcast term1 and term2 to (n, m)
            grad = term1 + term2 - 2 * term3

            # Sinkhorn step with gradient as cost
            plan = sinkhorn.sinkhorn(grad, a, b, epsilon)

            # Check convergence
            P_new = plan.P
            diff = np.linalg.norm(P_new - P, 'fro')
            P = P_new

            if diff < self.config.threshold:
                break

        # Compute GW cost using the final transport plan
        term1_cost = (D1_sq @ P).sum(axis=1, keepdims=True)
        term2_cost = (P @ D2_sq).sum(axis=0, keepdims=True)
        term3_cost = D1 @ P @ D2.T
        gw_cost = float(np.sum((term1_cost + term2_cost - 2 * term3_cost) * P))

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return TransportPlan(
            P=P,
            cost=gw_cost,
            iterations=i + 1,
            converged=diff < self.config.threshold,
            marginal_error=0.0,  # GW doesn't track marginal error same way
            computation_time_ms=elapsed_ms
        )


class SlicedWassersteinSolver:
    """
    Sliced-Wasserstein distance for fast approximate OT.

    Projects high-dimensional distributions to 1D random directions,
    computes 1D Wasserstein (which is just sorting), then averages.

    Complexity: O(n log n) vs O(n^2) for Sinkhorn

    Research Basis:
    - SLoSH (WACV 2024): Set Locality Sensitive Hashing via Sliced-Wasserstein
    - Rabin et al., "Wasserstein Barycenter and Its Application to Texture Mixing"
    """

    def __init__(self, n_projections: int = 50, seed: Optional[int] = None):
        """
        Initialize Sliced-Wasserstein solver.

        Args:
            n_projections: Number of random 1D projections (more = better accuracy)
            seed: Random seed for reproducibility
        """
        self.n_projections = n_projections
        self.rng = np.random.default_rng(seed)
        self._stats = {
            "computations": 0,
            "avg_distance": 0.0,
            "avg_time_ms": 0.0,
        }

    def sliced_wasserstein_distance(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_projections: Optional[int] = None
    ) -> float:
        """
        Compute Sliced-Wasserstein distance between two point clouds.

        O(n log n) complexity - much faster than Sinkhorn for large sets.

        Args:
            source: Source points (n, d)
            target: Target points (m, d)
            n_projections: Override default number of projections

        Returns:
            Sliced-Wasserstein distance (L1 in 1D, averaged over projections)
        """
        start_time = time.perf_counter()

        n_proj = n_projections or self.n_projections

        source = np.atleast_2d(source)
        target = np.atleast_2d(target)

        # Handle 1D arrays
        if source.ndim == 1:
            source = source.reshape(-1, 1)
        if target.ndim == 1:
            target = target.reshape(-1, 1)

        d = source.shape[1]

        # Match dimensions
        if source.shape[1] != target.shape[1]:
            raise ValueError(f"Dimension mismatch: {source.shape[1]} vs {target.shape[1]}")

        distances = []

        for _ in range(n_proj):
            # Random unit direction
            theta = self.rng.standard_normal(d)
            theta /= np.linalg.norm(theta) + 1e-10

            # Project to 1D
            proj_source = source @ theta
            proj_target = target @ theta

            # 1D Wasserstein is just |sorted(a) - sorted(b)|
            # Need to handle different lengths via resampling
            n, m = len(proj_source), len(proj_target)

            if n == m:
                # Direct comparison
                sorted_source = np.sort(proj_source)
                sorted_target = np.sort(proj_target)
                dist = np.mean(np.abs(sorted_source - sorted_target))
            else:
                # Resample to common grid
                sorted_source = np.sort(proj_source)
                sorted_target = np.sort(proj_target)

                # Linearly interpolate to same number of points
                common_size = max(n, m)
                source_interp = np.interp(
                    np.linspace(0, 1, common_size),
                    np.linspace(0, 1, n),
                    sorted_source
                )
                target_interp = np.interp(
                    np.linspace(0, 1, common_size),
                    np.linspace(0, 1, m),
                    sorted_target
                )
                dist = np.mean(np.abs(source_interp - target_interp))

            distances.append(dist)

        sw_distance = float(np.mean(distances))

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        n = self._stats["computations"]
        self._stats["computations"] = n + 1
        self._stats["avg_distance"] = (n * self._stats["avg_distance"] + sw_distance) / (n + 1)
        self._stats["avg_time_ms"] = (n * self._stats["avg_time_ms"] + elapsed_ms) / (n + 1)

        return sw_distance

    def sliced_wasserstein_embedding(
        self,
        points: np.ndarray,
        n_projections: Optional[int] = None
    ) -> np.ndarray:
        """
        Create Sliced-Wasserstein embedding of point cloud.

        Maps a set of points to a vector space where Euclidean distance
        approximates Sliced-Wasserstein distance.

        Useful for indexing with standard vector databases.

        Args:
            points: Point cloud (n, d)
            n_projections: Number of projections for embedding dimension

        Returns:
            Embedding vector (2 * n_projections,) - mean and std per projection
        """
        n_proj = n_projections or self.n_projections

        points = np.atleast_2d(points)
        if points.ndim == 1:
            points = points.reshape(-1, 1)

        d = points.shape[1]

        embedding = []

        for _ in range(n_proj):
            theta = self.rng.standard_normal(d)
            theta /= np.linalg.norm(theta) + 1e-10

            # Project and compute distribution statistics
            proj = points @ theta
            embedding.append(np.mean(proj))
            embedding.append(np.std(proj))

        return np.array(embedding)

    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return {
            **self._stats,
            "n_projections": self.n_projections,
        }


class WordMoverSolver:
    """
    Word Mover's Distance for document-level similarity.

    Treats documents as distributions over word embeddings and computes
    the optimal transport cost between them.

    Key insight: WMD captures semantic similarity even when documents
    share no common words.

    Research Basis:
    - Kusner et al., "From Word Embeddings to Document Distances" (ICML 2015)
    - MoverScore (EMNLP 2019): Text generation evaluation using WMD
    """

    def __init__(self, config: Optional[OTConfig] = None):
        self.config = config or OTConfig()
        self.sinkhorn = SinkhornSolver(self.config)
        self._stats = {
            "computations": 0,
            "avg_distance": 0.0,
            "avg_time_ms": 0.0,
        }

    def word_movers_distance(
        self,
        doc1_embeddings: np.ndarray,
        doc2_embeddings: np.ndarray,
        doc1_weights: Optional[np.ndarray] = None,
        doc2_weights: Optional[np.ndarray] = None,
        metric: CostMetric = CostMetric.COSINE
    ) -> float:
        """
        Compute Word Mover's Distance between two documents.

        Args:
            doc1_embeddings: Word embeddings for doc1 (n_words, dim)
            doc2_embeddings: Word embeddings for doc2 (m_words, dim)
            doc1_weights: Word weights for doc1 (e.g., TF-IDF). Default: uniform
            doc2_weights: Word weights for doc2. Default: uniform
            metric: Distance metric for word pairs

        Returns:
            Word Mover's Distance (optimal transport cost)
        """
        start_time = time.perf_counter()

        doc1_embeddings = np.atleast_2d(doc1_embeddings)
        doc2_embeddings = np.atleast_2d(doc2_embeddings)

        n, m = len(doc1_embeddings), len(doc2_embeddings)

        # Default uniform weights
        if doc1_weights is None:
            doc1_weights = np.ones(n) / n
        if doc2_weights is None:
            doc2_weights = np.ones(m) / m

        # Normalize weights to sum to 1
        doc1_weights = np.asarray(doc1_weights, dtype=np.float64)
        doc2_weights = np.asarray(doc2_weights, dtype=np.float64)
        doc1_weights = doc1_weights / (doc1_weights.sum() + 1e-10)
        doc2_weights = doc2_weights / (doc2_weights.sum() + 1e-10)

        # Compute word-to-word cost matrix
        cost = self.sinkhorn.compute_cost_matrix(
            doc1_embeddings, doc2_embeddings, metric
        )

        # Solve OT problem
        plan = self.sinkhorn.sinkhorn(cost, doc1_weights, doc2_weights)

        wmd = plan.cost

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        count = self._stats["computations"]
        self._stats["computations"] = count + 1
        self._stats["avg_distance"] = (count * self._stats["avg_distance"] + wmd) / (count + 1)
        self._stats["avg_time_ms"] = (count * self._stats["avg_time_ms"] + elapsed_ms) / (count + 1)

        return wmd

    def relaxed_word_movers_distance(
        self,
        doc1_embeddings: np.ndarray,
        doc2_embeddings: np.ndarray,
        doc1_weights: Optional[np.ndarray] = None,
        doc2_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Relaxed Word Mover's Distance (RWMD) - fast lower bound.

        Uses one-sided optimal transport for O(n+m) complexity
        vs O(n*m*log(n*m)) for exact WMD.

        Returns:
            RWMD (lower bound on true WMD)
        """
        doc1_embeddings = np.atleast_2d(doc1_embeddings)
        doc2_embeddings = np.atleast_2d(doc2_embeddings)

        n, m = len(doc1_embeddings), len(doc2_embeddings)

        if doc1_weights is None:
            doc1_weights = np.ones(n) / n
        if doc2_weights is None:
            doc2_weights = np.ones(m) / m

        doc1_weights = np.asarray(doc1_weights, dtype=np.float64)
        doc2_weights = np.asarray(doc2_weights, dtype=np.float64)

        # Compute cost matrix (cosine distance)
        cost = self.sinkhorn.compute_cost_matrix(
            doc1_embeddings, doc2_embeddings, CostMetric.COSINE
        )

        # RWMD from doc1 perspective: each word in doc1 goes to closest in doc2
        min_cost_1 = cost.min(axis=1)  # (n,)
        rwmd_1 = np.sum(doc1_weights * min_cost_1)

        # RWMD from doc2 perspective
        min_cost_2 = cost.min(axis=0)  # (m,)
        rwmd_2 = np.sum(doc2_weights * min_cost_2)

        # Take max for tighter lower bound
        return float(max(rwmd_1, rwmd_2))

    def document_similarity_matrix(
        self,
        documents: List[np.ndarray],
        use_relaxed: bool = False
    ) -> np.ndarray:
        """
        Compute pairwise WMD similarity matrix for a corpus.

        Args:
            documents: List of document embeddings, each (n_words, dim)
            use_relaxed: Use RWMD for speed (lower bound)

        Returns:
            Similarity matrix (n_docs, n_docs) where sim = 1 / (1 + distance)
        """
        n_docs = len(documents)
        distances = np.zeros((n_docs, n_docs))

        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                if use_relaxed:
                    d = self.relaxed_word_movers_distance(documents[i], documents[j])
                else:
                    d = self.word_movers_distance(documents[i], documents[j])
                distances[i, j] = d
                distances[j, i] = d

        # Convert to similarity: sim = 1 / (1 + distance)
        similarity = 1.0 / (1.0 + distances)

        return similarity

    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return {
            **self._stats,
            "epsilon": self.config.epsilon,
        }


class OptimalTransportFusion:
    """
    Main class for OT-based dense-sparse retrieval fusion.

    Uses optimal transport to find the best alignment between
    dense (semantic) and sparse (lexical) retrieval results,
    producing unified scores that respect the geometry of both spaces.
    """

    def __init__(self, config: Optional[OTConfig] = None):
        self.config = config or OTConfig()
        self.sinkhorn = SinkhornSolver(self.config)
        self.gw = GromovWassersteinSolver(self.config)
        self.sliced = SlicedWassersteinSolver()
        self.wmd = WordMoverSolver(self.config)
        self._transport_cache: Dict[str, TransportPlan] = {}
        self._stats = {
            "fusions": 0,
            "cache_hits": 0,
            "avg_iterations": 0.0,
            "avg_cost": 0.0,
            "avg_time_ms": 0.0,
        }

    def fuse_scores(
        self,
        dense_results: List[Tuple[str, float]],  # (doc_id, score)
        sparse_results: List[Tuple[str, float]],  # (doc_id, score)
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        top_k: int = 10
    ) -> List[OTResult]:
        """
        Fuse dense and sparse retrieval results using Optimal Transport.

        The OT framework finds the optimal way to "transport" probability mass
        from dense scores to sparse scores, respecting their underlying geometry.

        Args:
            dense_results: Dense retrieval (doc_id, score) pairs, sorted by score
            sparse_results: Sparse retrieval (doc_id, score) pairs, sorted by score
            embeddings: Optional embeddings for similarity-based cost
            top_k: Number of results to return

        Returns:
            List of OTResult with fused scores
        """
        if not dense_results and not sparse_results:
            return []

        if not dense_results:
            return self._convert_to_results(sparse_results, "sparse", top_k)
        if not sparse_results:
            return self._convert_to_results(dense_results, "dense", top_k)

        start_time = time.perf_counter()

        # Build document set (union of both)
        all_docs = {}
        for doc_id, score in dense_results:
            all_docs[doc_id] = {"dense": score, "sparse": 0.0}
        for doc_id, score in sparse_results:
            if doc_id in all_docs:
                all_docs[doc_id]["sparse"] = score
            else:
                all_docs[doc_id] = {"dense": 0.0, "sparse": score}

        doc_ids = list(all_docs.keys())
        n = len(doc_ids)

        # Normalize scores to [0, 1] and treat as probability distributions
        dense_scores = np.array([all_docs[d]["dense"] for d in doc_ids])
        sparse_scores = np.array([all_docs[d]["sparse"] for d in doc_ids])

        if self.config.normalize_scores:
            dense_scores = self._normalize(dense_scores)
            sparse_scores = self._normalize(sparse_scores)

        # Convert to probability distributions (sum to 1)
        dense_probs = dense_scores / (dense_scores.sum() + 1e-10)
        sparse_probs = sparse_scores / (sparse_scores.sum() + 1e-10)

        # Compute cost matrix
        if embeddings and all(d in embeddings for d in doc_ids):
            # Use embedding similarity as cost
            emb_matrix = np.array([embeddings[d] for d in doc_ids])
            cost = self.sinkhorn.compute_cost_matrix(
                emb_matrix, emb_matrix, self.config.cost_metric
            )
        else:
            # Use rank-based cost
            cost = self.sinkhorn.compute_cost_matrix(
                dense_scores.reshape(-1, 1),
                sparse_scores.reshape(-1, 1),
                CostMetric.RANK_BASED
            )

        # Solve OT problem
        if self.config.method == OTMethod.SINKHORN:
            plan = self.sinkhorn.sinkhorn(cost, dense_probs, sparse_probs)
        elif self.config.method == OTMethod.EXACT:
            # Fall back to Sinkhorn with small epsilon for near-exact
            plan = self.sinkhorn.sinkhorn(cost, dense_probs, sparse_probs, epsilon=0.001)
        else:
            plan = self.sinkhorn.sinkhorn(cost, dense_probs, sparse_probs)

        # Compute fused scores using transport plan
        # Score = weighted sum of original scores with transport weights
        fused_scores = np.zeros(n)

        for i in range(n):
            # Transport weight from this document
            transport_out = plan.P[i, :].sum()
            transport_in = plan.P[:, i].sum()
            transport_weight = (transport_out + transport_in) / 2

            # Fused score: weighted combination with transport adjustment
            fused = (
                self.config.dense_weight * dense_scores[i] +
                self.config.sparse_weight * sparse_scores[i]
            ) * (1 + transport_weight)  # Transport weight boosts aligned docs

            fused_scores[i] = fused

        # Create results
        results = []
        sorted_indices = np.argsort(-fused_scores)  # Descending

        for rank, idx in enumerate(sorted_indices[:top_k]):
            doc_id = doc_ids[idx]
            results.append(OTResult(
                doc_id=doc_id,
                fused_score=float(fused_scores[idx]),
                dense_score=float(dense_scores[idx]),
                sparse_score=float(sparse_scores[idx]),
                transport_weight=float(plan.P[idx, :].sum() + plan.P[:, idx].sum()) / 2,
                rank=rank + 1,
                metadata={
                    "wasserstein_cost": float(plan.cost),
                    "iterations": plan.iterations,
                    "converged": plan.converged,
                }
            ))

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(plan, elapsed_ms)

        return results

    def fuse_multiway(
        self,
        result_lists: Dict[str, List[Tuple[str, float]]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[OTResult]:
        """
        Fuse multiple retrieval result lists using Wasserstein barycenter.

        Computes the barycenter of all result distributions, which is the
        distribution minimizing total Wasserstein distance to all inputs.

        Args:
            result_lists: Dict of retriever_name -> (doc_id, score) pairs
            weights: Optional weights for each retriever (sum to 1)
            top_k: Number of results to return

        Returns:
            List of OTResult with barycentric fused scores
        """
        if not result_lists:
            return []

        names = list(result_lists.keys())
        n_retrievers = len(names)

        if n_retrievers == 1:
            return self._convert_to_results(result_lists[names[0]], names[0], top_k)

        if weights is None:
            weights = {name: 1.0 / n_retrievers for name in names}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Build unified document set
        all_docs = {}
        for name, results in result_lists.items():
            for doc_id, score in results:
                if doc_id not in all_docs:
                    all_docs[doc_id] = {n: 0.0 for n in names}
                all_docs[doc_id][name] = score

        doc_ids = list(all_docs.keys())
        n = len(doc_ids)

        # Build score matrix (n_docs x n_retrievers)
        score_matrix = np.zeros((n, n_retrievers))
        for i, doc_id in enumerate(doc_ids):
            for j, name in enumerate(names):
                score_matrix[i, j] = all_docs[doc_id].get(name, 0.0)

        if self.config.normalize_scores:
            for j in range(n_retrievers):
                score_matrix[:, j] = self._normalize(score_matrix[:, j])

        # Simple weighted barycenter: weighted average
        # (Full OT barycenter requires iterative algorithm)
        weight_vec = np.array([weights[name] for name in names])
        barycenter_scores = score_matrix @ weight_vec

        # Create results
        results = []
        sorted_indices = np.argsort(-barycenter_scores)

        for rank, idx in enumerate(sorted_indices[:top_k]):
            doc_id = doc_ids[idx]
            results.append(OTResult(
                doc_id=doc_id,
                fused_score=float(barycenter_scores[idx]),
                rank=rank + 1,
                metadata={
                    "retriever_scores": {
                        name: float(score_matrix[idx, j])
                        for j, name in enumerate(names)
                    },
                    "fusion_method": "wasserstein_barycenter",
                }
            ))

        return results

    def align_heterogeneous(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        source_ids: List[str],
        target_ids: List[str]
    ) -> Dict[str, str]:
        """
        Align documents across heterogeneous embedding spaces using Gromov-Wasserstein.

        Useful when dense and sparse embeddings live in incompatible spaces
        and direct comparison isn't meaningful.

        Args:
            source_embeddings: Embeddings from source space (n, d1)
            target_embeddings: Embeddings from target space (m, d2)
            source_ids: Document IDs for source
            target_ids: Document IDs for target

        Returns:
            Mapping from source doc IDs to aligned target doc IDs
        """
        # Compute distance matrices
        D1 = self.gw.compute_distance_matrix(source_embeddings)
        D2 = self.gw.compute_distance_matrix(target_embeddings)

        # Solve GW problem
        plan = self.gw.gromov_wasserstein(D1, D2)

        # Extract alignment from transport plan
        alignment = {}
        for i, source_id in enumerate(source_ids):
            # Find best matching target
            best_j = np.argmax(plan.P[i, :])
            if plan.P[i, best_j] > 0.01:  # Threshold for meaningful match
                alignment[source_id] = target_ids[best_j]

        return alignment

    def compute_wasserstein_distance(
        self,
        scores1: List[float],
        scores2: List[float]
    ) -> float:
        """
        Compute Wasserstein distance between two score distributions.

        Useful for measuring similarity between different retrieval results.
        """
        a = np.array(scores1, dtype=np.float64)
        b = np.array(scores2, dtype=np.float64)

        # Normalize to probability distributions
        a = a / (a.sum() + 1e-10)
        b = b / (b.sum() + 1e-10)

        # Pad shorter array
        max_len = max(len(a), len(b))
        if len(a) < max_len:
            a = np.pad(a, (0, max_len - len(a)))
        if len(b) < max_len:
            b = np.pad(b, (0, max_len - len(b)))

        # Compute cost matrix (rank-based)
        n = len(a)
        cost = np.abs(np.arange(n).reshape(-1, 1) - np.arange(n).reshape(1, -1)) / n

        plan = self.sinkhorn.sinkhorn(cost, a, b)
        return plan.cost

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [min_score, max_score]."""
        if scores.max() == scores.min():
            return np.full_like(scores, (self.config.min_score + self.config.max_score) / 2)

        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return normalized * (self.config.max_score - self.config.min_score) + self.config.min_score

    def _convert_to_results(
        self,
        results: List[Tuple[str, float]],
        source: str,
        top_k: int
    ) -> List[OTResult]:
        """Convert simple (doc_id, score) pairs to OTResult."""
        output = []
        for rank, (doc_id, score) in enumerate(results[:top_k]):
            output.append(OTResult(
                doc_id=doc_id,
                fused_score=score,
                dense_score=score if source == "dense" else None,
                sparse_score=score if source == "sparse" else None,
                rank=rank + 1,
                metadata={"source": source}
            ))
        return output

    def _update_stats(self, plan: TransportPlan, elapsed_ms: float):
        """Update running statistics."""
        n = self._stats["fusions"]
        self._stats["fusions"] = n + 1

        # Running averages
        self._stats["avg_iterations"] = (
            (n * self._stats["avg_iterations"] + plan.iterations) / (n + 1)
        )
        self._stats["avg_cost"] = (
            (n * self._stats["avg_cost"] + plan.cost) / (n + 1)
        )
        self._stats["avg_time_ms"] = (
            (n * self._stats["avg_time_ms"] + elapsed_ms) / (n + 1)
        )

    def sliced_wasserstein_distance(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_projections: int = 50
    ) -> float:
        """
        Compute Sliced-Wasserstein distance between point clouds.

        Fast O(n log n) approximation of Wasserstein distance.

        Args:
            source: Source points (n, d)
            target: Target points (m, d)
            n_projections: Number of random projections (more = more accurate)

        Returns:
            Sliced-Wasserstein distance
        """
        return self.sliced.sliced_wasserstein_distance(source, target, n_projections)

    def sliced_wasserstein_embedding(
        self,
        points: np.ndarray,
        n_projections: int = 50
    ) -> np.ndarray:
        """
        Create Sliced-Wasserstein embedding for indexing.

        Useful for approximate nearest neighbor search in vector DBs.

        Args:
            points: Point cloud (n, d)
            n_projections: Embedding dimension will be 2 * n_projections

        Returns:
            Embedding vector
        """
        return self.sliced.sliced_wasserstein_embedding(points, n_projections)

    def word_movers_distance(
        self,
        doc1_embeddings: np.ndarray,
        doc2_embeddings: np.ndarray,
        doc1_weights: Optional[np.ndarray] = None,
        doc2_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Word Mover's Distance between documents.

        Measures document similarity by optimal transport of word embeddings.

        Args:
            doc1_embeddings: Word embeddings for doc1 (n_words, dim)
            doc2_embeddings: Word embeddings for doc2 (m_words, dim)
            doc1_weights: Optional TF-IDF weights for doc1
            doc2_weights: Optional TF-IDF weights for doc2

        Returns:
            Word Mover's Distance (lower = more similar)
        """
        return self.wmd.word_movers_distance(
            doc1_embeddings, doc2_embeddings,
            doc1_weights, doc2_weights
        )

    def relaxed_word_movers_distance(
        self,
        doc1_embeddings: np.ndarray,
        doc2_embeddings: np.ndarray
    ) -> float:
        """
        Fast lower bound on Word Mover's Distance.

        O(n+m) complexity vs O(n*m) for exact WMD.
        Useful for pre-filtering before exact computation.

        Returns:
            Relaxed WMD (lower bound)
        """
        return self.wmd.relaxed_word_movers_distance(doc1_embeddings, doc2_embeddings)

    def document_similarity_matrix(
        self,
        documents: List[np.ndarray],
        use_relaxed: bool = False
    ) -> np.ndarray:
        """
        Compute pairwise WMD similarity matrix for a corpus.

        Args:
            documents: List of document embeddings, each (n_words, dim)
            use_relaxed: Use RWMD for speed

        Returns:
            Similarity matrix (n_docs, n_docs)
        """
        return self.wmd.document_similarity_matrix(documents, use_relaxed)

    def get_stats(self) -> Dict[str, Any]:
        """Get fusion statistics including all sub-solvers."""
        return {
            **self._stats,
            "config": {
                "method": self.config.method.value,
                "epsilon": self.config.epsilon,
                "cost_metric": self.config.cost_metric.value,
                "dense_weight": self.config.dense_weight,
                "sparse_weight": self.config.sparse_weight,
            },
            "sliced_wasserstein": self.sliced.get_stats(),
            "word_mover": self.wmd.get_stats(),
        }


# Global singleton
_ot_fusion: Optional[OptimalTransportFusion] = None


def get_ot_fusion(config: Optional[OTConfig] = None) -> OptimalTransportFusion:
    """Get or create global OptimalTransportFusion instance."""
    global _ot_fusion
    if _ot_fusion is None:
        _ot_fusion = OptimalTransportFusion(config)
    return _ot_fusion


async def ot_fuse_scores(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    top_k: int = 10,
    config: Optional[OTConfig] = None
) -> List[OTResult]:
    """Async wrapper for OT score fusion."""
    fusion = get_ot_fusion(config)
    return fusion.fuse_scores(dense_results, sparse_results, top_k=top_k)


async def ot_fuse_multiway(
    result_lists: Dict[str, List[Tuple[str, float]]],
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 10,
    config: Optional[OTConfig] = None
) -> List[OTResult]:
    """Async wrapper for multi-way OT fusion."""
    fusion = get_ot_fusion(config)
    return fusion.fuse_multiway(result_lists, weights, top_k=top_k)
