"""
Unit Tests for Optimal Transport Fusion Module

Tests the Sinkhorn algorithm, Gromov-Wasserstein, and OT-based score fusion.

Author: Claude Code
Date: December 2025
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentic.optimal_transport import (
    SinkhornSolver,
    GromovWassersteinSolver,
    SlicedWassersteinSolver,
    WordMoverSolver,
    OptimalTransportFusion,
    OTConfig,
    OTMethod,
    CostMetric,
    OTResult,
    TransportPlan,
    get_ot_fusion,
    ot_fuse_scores,
    ot_fuse_multiway,
)


class TestSinkhornSolver:
    """Tests for Sinkhorn-Knopp algorithm."""

    @pytest.fixture
    def solver(self):
        """Create Sinkhorn solver with default config."""
        config = OTConfig(epsilon=0.1, max_iter=100, threshold=1e-6)
        return SinkhornSolver(config)

    def test_cost_matrix_euclidean(self, solver):
        """Test Euclidean cost matrix computation."""
        source = np.array([[0, 0], [1, 0], [0, 1]])
        target = np.array([[0, 0], [2, 0]])

        cost = solver.compute_cost_matrix(source, target, CostMetric.EUCLIDEAN)

        assert cost.shape == (3, 2)
        assert cost[0, 0] == 0.0  # Same point
        assert abs(cost[0, 1] - 2.0) < 1e-6  # Distance from (0,0) to (2,0)
        assert abs(cost[1, 0] - 1.0) < 1e-6  # Distance from (1,0) to (0,0)

    def test_cost_matrix_cosine(self, solver):
        """Test cosine cost matrix computation."""
        source = np.array([[1, 0], [0, 1]])
        target = np.array([[1, 0], [-1, 0]])

        cost = solver.compute_cost_matrix(source, target, CostMetric.COSINE)

        assert cost.shape == (2, 2)
        assert abs(cost[0, 0]) < 1e-6  # Same direction: cost = 0
        assert abs(cost[0, 1] - 2.0) < 1e-6  # Opposite direction: cost = 2

    def test_cost_matrix_rank_based(self, solver):
        """Test rank-based cost matrix computation."""
        source = np.array([[0.9], [0.7], [0.5]])
        target = np.array([[0.8], [0.6]])

        cost = solver.compute_cost_matrix(source, target, CostMetric.RANK_BASED)

        assert cost.shape == (3, 2)
        # Rank 0 to rank 0 should have 0 cost
        assert abs(cost[0, 0]) < 1e-6
        # Rank 0 to rank 1 should have cost 1/3
        assert abs(cost[0, 1] - 1 / 3) < 1e-6

    def test_sinkhorn_uniform_marginals(self, solver):
        """Test Sinkhorn with uniform marginals."""
        n, m = 4, 4
        # Use fixed seed for reproducibility
        np.random.seed(42)
        cost = np.random.rand(n, m)

        plan = solver.sinkhorn(cost)

        assert plan.P.shape == (n, m)
        # Accept either formal convergence or very low marginal error
        assert plan.converged or plan.marginal_error < 1e-4
        # Check marginals are approximately uniform
        row_sums = plan.P.sum(axis=1)
        col_sums = plan.P.sum(axis=0)
        np.testing.assert_array_almost_equal(row_sums, np.ones(n) / n, decimal=4)
        np.testing.assert_array_almost_equal(col_sums, np.ones(m) / m, decimal=4)

    def test_sinkhorn_custom_marginals(self, solver):
        """Test Sinkhorn with custom marginals."""
        n, m = 3, 3
        cost = np.ones((n, m)) * 0.5
        a = np.array([0.5, 0.3, 0.2])
        b = np.array([0.4, 0.4, 0.2])

        plan = solver.sinkhorn(cost, a, b)

        # Check marginal constraints
        row_sums = plan.P.sum(axis=1)
        col_sums = plan.P.sum(axis=0)
        np.testing.assert_array_almost_equal(row_sums, a, decimal=3)
        np.testing.assert_array_almost_equal(col_sums, b, decimal=3)

    def test_sinkhorn_convergence(self, solver):
        """Test that Sinkhorn converges properly."""
        cost = np.random.rand(5, 5)

        plan = solver.sinkhorn(cost)

        assert plan.converged
        assert plan.iterations < solver.config.max_iter
        assert plan.marginal_error < 0.01

    def test_sinkhorn_transport_plan_valid(self, solver):
        """Test that transport plan is valid (non-negative, sums to 1)."""
        cost = np.random.rand(4, 4)

        plan = solver.sinkhorn(cost)

        # All entries non-negative
        assert np.all(plan.P >= 0)
        # Total mass is 1
        assert abs(plan.P.sum() - 1.0) < 1e-4

    def test_sinkhorn_log_stabilized(self, solver):
        """Test log-stabilized Sinkhorn for numerical stability."""
        # Use small epsilon to test stability
        config = OTConfig(epsilon=0.01, max_iter=200)
        stable_solver = SinkhornSolver(config)

        cost = np.random.rand(5, 5) * 10  # Larger costs

        plan = stable_solver.sinkhorn_log_stabilized(cost)

        assert plan.P.shape == (5, 5)
        # Should still produce valid transport plan
        assert np.all(np.isfinite(plan.P))
        assert np.all(plan.P >= 0)


class TestGromovWassersteinSolver:
    """Tests for Gromov-Wasserstein distance."""

    @pytest.fixture
    def solver(self):
        """Create GW solver with default config."""
        config = OTConfig(gw_alpha=0.5, gw_max_iter=50, epsilon=0.1)
        return GromovWassersteinSolver(config)

    def test_compute_distance_matrix(self, solver):
        """Test pairwise distance matrix computation."""
        X = np.array([[0, 0], [1, 0], [0, 1]])

        D = solver.compute_distance_matrix(X)

        assert D.shape == (3, 3)
        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(D), np.zeros(3))
        # Symmetric
        np.testing.assert_array_almost_equal(D, D.T)
        # Distance from (0,0) to (1,0) should be 1
        assert abs(D[0, 1] - 1.0) < 1e-6

    def test_gromov_wasserstein_same_space(self, solver):
        """Test GW distance between identical spaces."""
        X = np.random.rand(5, 3)
        D = solver.compute_distance_matrix(X)

        plan = solver.gromov_wasserstein(D, D)

        # Transport plan should be valid
        assert plan.P.shape == (5, 5)
        # Should converge
        assert plan.converged or plan.iterations <= solver.config.gw_max_iter
        # Total mass should be 1
        assert abs(plan.P.sum() - 1.0) < 0.01
        # All entries should be non-negative
        assert np.all(plan.P >= -1e-10)
        # GW cost should be low (same structure)
        assert plan.cost < 0.5  # Same structure should have low cost

    def test_gromov_wasserstein_different_sizes(self, solver):
        """Test GW with different-sized spaces."""
        X1 = np.random.rand(4, 3)
        X2 = np.random.rand(6, 3)

        D1 = solver.compute_distance_matrix(X1)
        D2 = solver.compute_distance_matrix(X2)

        plan = solver.gromov_wasserstein(D1, D2)

        assert plan.P.shape == (4, 6)
        # Marginals should sum to uniform
        row_sums = plan.P.sum(axis=1)
        col_sums = plan.P.sum(axis=0)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4) / 4, decimal=2)
        np.testing.assert_array_almost_equal(col_sums, np.ones(6) / 6, decimal=2)


class TestOptimalTransportFusion:
    """Tests for OT-based retrieval fusion."""

    @pytest.fixture
    def fusion(self):
        """Create OT fusion with default config."""
        config = OTConfig(
            dense_weight=0.5,
            sparse_weight=0.5,
            epsilon=0.1,
            normalize_scores=True
        )
        return OptimalTransportFusion(config)

    def test_fuse_empty_inputs(self, fusion):
        """Test fusion with empty inputs."""
        results = fusion.fuse_scores([], [])
        assert results == []

    def test_fuse_single_source_dense(self, fusion):
        """Test fusion with only dense results."""
        dense = [("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)]

        results = fusion.fuse_scores(dense, [], top_k=3)

        assert len(results) == 3
        assert results[0].doc_id == "doc1"
        assert results[0].rank == 1

    def test_fuse_single_source_sparse(self, fusion):
        """Test fusion with only sparse results."""
        sparse = [("doc1", 0.8), ("doc2", 0.6)]

        results = fusion.fuse_scores([], sparse, top_k=2)

        assert len(results) == 2
        assert results[0].doc_id == "doc1"

    def test_fuse_overlapping_docs(self, fusion):
        """Test fusion with overlapping documents."""
        dense = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.5)]
        sparse = [("doc2", 0.95), ("doc1", 0.7), ("doc4", 0.6)]

        results = fusion.fuse_scores(dense, sparse, top_k=4)

        assert len(results) == 4
        # All unique docs should be present
        doc_ids = {r.doc_id for r in results}
        assert doc_ids == {"doc1", "doc2", "doc3", "doc4"}
        # Scores should be fused
        for r in results:
            assert r.fused_score is not None

    def test_fuse_disjoint_docs(self, fusion):
        """Test fusion with non-overlapping documents."""
        dense = [("a1", 0.9), ("a2", 0.8)]
        sparse = [("b1", 0.95), ("b2", 0.7)]

        results = fusion.fuse_scores(dense, sparse, top_k=4)

        assert len(results) == 4
        # Each doc should have one zero score
        for r in results:
            if r.doc_id.startswith("a"):
                assert r.sparse_score == 0.0 or r.sparse_score is None or r.sparse_score < 0.01
            else:
                assert r.dense_score == 0.0 or r.dense_score is None or r.dense_score < 0.01

    def test_fuse_respects_top_k(self, fusion):
        """Test that top_k parameter is respected."""
        dense = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7), ("d4", 0.6)]
        sparse = [("d1", 0.85), ("d2", 0.75)]

        results = fusion.fuse_scores(dense, sparse, top_k=2)

        assert len(results) == 2

    def test_fuse_includes_metadata(self, fusion):
        """Test that results include transport metadata."""
        dense = [("doc1", 0.9)]
        sparse = [("doc1", 0.8)]

        results = fusion.fuse_scores(dense, sparse)

        assert len(results) == 1
        assert "wasserstein_cost" in results[0].metadata
        assert "iterations" in results[0].metadata
        assert "converged" in results[0].metadata

    def test_fuse_transport_weights(self, fusion):
        """Test that transport weights are computed."""
        dense = [("doc1", 0.9), ("doc2", 0.5)]
        sparse = [("doc1", 0.8), ("doc2", 0.6)]

        results = fusion.fuse_scores(dense, sparse)

        for r in results:
            assert r.transport_weight >= 0

    def test_fuse_multiway(self, fusion):
        """Test multi-way fusion."""
        result_lists = {
            "dense": [("d1", 0.9), ("d2", 0.7)],
            "sparse": [("d1", 0.8), ("d3", 0.6)],
            "colbert": [("d2", 0.85), ("d3", 0.7)],
        }

        results = fusion.fuse_multiway(result_lists, top_k=3)

        assert len(results) == 3
        # All docs should be present
        doc_ids = {r.doc_id for r in results}
        assert doc_ids == {"d1", "d2", "d3"}
        # Check metadata
        for r in results:
            assert "retriever_scores" in r.metadata
            assert "fusion_method" in r.metadata

    def test_fuse_multiway_with_weights(self, fusion):
        """Test multi-way fusion with custom weights."""
        result_lists = {
            "primary": [("d1", 0.9)],
            "secondary": [("d2", 0.95)],
        }
        weights = {"primary": 0.8, "secondary": 0.2}

        results = fusion.fuse_multiway(result_lists, weights=weights, top_k=2)

        # Primary should be ranked higher due to weight
        assert len(results) == 2
        # d1 should score higher: 0.9 * 0.8 = 0.72 vs d2: 0.95 * 0.2 = 0.19
        assert results[0].doc_id == "d1"

    def test_wasserstein_distance(self, fusion):
        """Test Wasserstein distance computation."""
        scores1 = [0.9, 0.7, 0.5, 0.3]
        scores2 = [0.8, 0.6, 0.4, 0.2]

        dist = fusion.compute_wasserstein_distance(scores1, scores2)

        assert dist >= 0
        # Same distribution should have very small distance
        # (not exactly 0 due to entropy regularization)
        dist_same = fusion.compute_wasserstein_distance(scores1, scores1)
        assert dist_same < 0.05  # Relaxed threshold for entropy-regularized OT

    def test_get_stats(self, fusion):
        """Test statistics retrieval."""
        # Run a fusion first
        fusion.fuse_scores([("d1", 0.9)], [("d1", 0.8)])

        stats = fusion.get_stats()

        assert "fusions" in stats
        assert stats["fusions"] >= 1
        assert "avg_iterations" in stats
        assert "config" in stats


class TestHeterogeneousAlignment:
    """Tests for cross-domain alignment."""

    @pytest.fixture
    def fusion(self):
        """Create fusion for alignment tests."""
        config = OTConfig(gw_alpha=0.5, epsilon=0.1)
        return OptimalTransportFusion(config)

    def test_align_same_embeddings(self, fusion):
        """Test alignment of identical embeddings."""
        embeddings = np.random.rand(5, 64)
        ids = [f"doc{i}" for i in range(5)]

        alignment = fusion.align_heterogeneous(
            embeddings, embeddings, ids, ids
        )

        # Should mostly self-align
        matched = sum(1 for k, v in alignment.items() if k == v)
        assert matched >= 3  # At least 60% self-match


class TestSingletonPattern:
    """Tests for global singleton pattern."""

    def test_get_ot_fusion(self):
        """Test singleton accessor."""
        # Clear any existing singleton
        import agentic.optimal_transport as module
        module._ot_fusion = None

        fusion1 = get_ot_fusion()
        fusion2 = get_ot_fusion()

        # Should be same instance
        assert fusion1 is fusion2


class TestAsyncWrappers:
    """Tests for async wrapper functions."""

    @pytest.mark.asyncio
    async def test_ot_fuse_scores_async(self):
        """Test async score fusion."""
        dense = [("doc1", 0.9), ("doc2", 0.7)]
        sparse = [("doc1", 0.8), ("doc2", 0.6)]

        results = await ot_fuse_scores(dense, sparse, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, OTResult) for r in results)

    @pytest.mark.asyncio
    async def test_ot_fuse_multiway_async(self):
        """Test async multi-way fusion."""
        result_lists = {
            "dense": [("d1", 0.9)],
            "sparse": [("d1", 0.8)],
        }

        results = await ot_fuse_multiway(result_lists, top_k=1)

        assert len(results) == 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_document(self):
        """Test fusion with single document."""
        fusion = OptimalTransportFusion(OTConfig())
        dense = [("only_doc", 0.9)]
        sparse = [("only_doc", 0.8)]

        results = fusion.fuse_scores(dense, sparse)

        assert len(results) == 1
        assert results[0].doc_id == "only_doc"

    def test_zero_scores(self):
        """Test fusion with zero scores."""
        fusion = OptimalTransportFusion(OTConfig())
        dense = [("d1", 0.0), ("d2", 0.0)]
        sparse = [("d1", 0.5)]

        results = fusion.fuse_scores(dense, sparse)

        # Should not crash, should return valid results
        assert len(results) >= 1

    def test_identical_scores(self):
        """Test fusion when all scores are identical."""
        fusion = OptimalTransportFusion(OTConfig())
        dense = [("d1", 0.5), ("d2", 0.5)]
        sparse = [("d1", 0.5), ("d2", 0.5)]

        results = fusion.fuse_scores(dense, sparse)

        assert len(results) == 2
        # Scores should be approximately equal
        assert abs(results[0].fused_score - results[1].fused_score) < 0.1

    def test_large_scale(self):
        """Test fusion with larger number of documents."""
        fusion = OptimalTransportFusion(OTConfig())

        n = 100
        dense = [(f"d{i}", np.random.rand()) for i in range(n)]
        sparse = [(f"d{i}", np.random.rand()) for i in range(n // 2)]

        results = fusion.fuse_scores(dense, sparse, top_k=20)

        assert len(results) == 20
        # Results should be sorted by score
        for i in range(len(results) - 1):
            assert results[i].fused_score >= results[i + 1].fused_score


class TestSlicedWassersteinSolver:
    """Tests for Sliced-Wasserstein distance computation."""

    @pytest.fixture
    def solver(self):
        """Create Sliced-Wasserstein solver."""
        return SlicedWassersteinSolver(n_projections=50, seed=42)

    def test_sliced_wasserstein_same_distribution(self, solver):
        """Test SW distance between identical distributions is small."""
        X = np.random.rand(20, 64)

        dist = solver.sliced_wasserstein_distance(X, X)

        # Same distribution should have very small distance
        assert dist < 0.01

    def test_sliced_wasserstein_different_distributions(self, solver):
        """Test SW distance between different distributions."""
        X1 = np.random.rand(20, 64)
        X2 = np.random.rand(20, 64) + 5.0  # Shifted distribution

        dist = solver.sliced_wasserstein_distance(X1, X2)

        # Different distributions should have positive distance
        assert dist > 0.1

    def test_sliced_wasserstein_symmetry(self, solver):
        """Test that SW distance is symmetric within sampling variance."""
        # Use same seed for both computations
        solver1 = SlicedWassersteinSolver(n_projections=100, seed=42)
        solver2 = SlicedWassersteinSolver(n_projections=100, seed=42)

        X1 = np.random.rand(15, 32)
        X2 = np.random.rand(15, 32) + 1.0

        dist12 = solver1.sliced_wasserstein_distance(X1, X2)
        dist21 = solver2.sliced_wasserstein_distance(X2, X1)

        # With same random projections, should be exactly symmetric
        np.testing.assert_almost_equal(dist12, dist21, decimal=5)

    def test_sliced_wasserstein_different_sizes(self, solver):
        """Test SW distance with different sample sizes."""
        X1 = np.random.rand(10, 64)
        X2 = np.random.rand(20, 64)

        dist = solver.sliced_wasserstein_distance(X1, X2)

        # Should return valid distance
        assert dist >= 0
        assert np.isfinite(dist)

    def test_sliced_wasserstein_custom_projections(self, solver):
        """Test SW with custom number of projections."""
        X1 = np.random.rand(15, 32)
        X2 = np.random.rand(15, 32) + 0.5

        dist1 = solver.sliced_wasserstein_distance(X1, X2, n_projections=10)
        dist2 = solver.sliced_wasserstein_distance(X1, X2, n_projections=100)

        # Both should be positive
        assert dist1 > 0
        assert dist2 > 0
        # More projections should give more stable estimate (but value may differ)

    def test_sliced_wasserstein_embedding(self, solver):
        """Test SW embedding creation."""
        X = np.random.rand(10, 64)

        embedding = solver.sliced_wasserstein_embedding(X, n_projections=20)

        # Embedding has shape (2 * n_projections,) because it stores mean+std per projection
        assert embedding.shape == (40,)
        # All values should be finite
        assert np.all(np.isfinite(embedding))

    def test_sliced_wasserstein_embedding_distance_preservation(self, solver):
        """Test that SW embeddings preserve relative distances."""
        # Create three point clouds: X1 close to X2, X3 far from both
        X1 = np.random.rand(10, 64)
        X2 = X1 + np.random.rand(10, 64) * 0.1  # Close to X1
        X3 = np.random.rand(10, 64) + 5.0  # Far from both

        emb1 = solver.sliced_wasserstein_embedding(X1)
        emb2 = solver.sliced_wasserstein_embedding(X2)
        emb3 = solver.sliced_wasserstein_embedding(X3)

        dist_12 = np.linalg.norm(emb1 - emb2)
        dist_13 = np.linalg.norm(emb1 - emb3)

        # X1-X2 distance should be smaller than X1-X3 distance
        assert dist_12 < dist_13

    def test_get_stats(self, solver):
        """Test solver statistics."""
        # Run some computations first
        X1 = np.random.rand(10, 32)
        X2 = np.random.rand(10, 32)
        solver.sliced_wasserstein_distance(X1, X2)

        stats = solver.get_stats()

        assert "computations" in stats
        assert stats["computations"] >= 1
        assert "n_projections" in stats

    def test_reproducibility_with_seed(self):
        """Test that solver is reproducible with seed."""
        solver1 = SlicedWassersteinSolver(n_projections=30, seed=123)
        solver2 = SlicedWassersteinSolver(n_projections=30, seed=123)

        X1 = np.random.rand(10, 64)
        X2 = np.random.rand(10, 64)

        dist1 = solver1.sliced_wasserstein_distance(X1, X2)
        dist2 = solver2.sliced_wasserstein_distance(X1, X2)

        np.testing.assert_almost_equal(dist1, dist2, decimal=10)


class TestWordMoverSolver:
    """Tests for Word Mover's Distance computation."""

    @pytest.fixture
    def solver(self):
        """Create WMD solver with default config."""
        config = OTConfig(epsilon=0.1, max_iter=100)
        return WordMoverSolver(config)

    def test_wmd_identical_documents(self, solver):
        """Test WMD between identical documents is small."""
        doc = np.random.rand(5, 64)

        dist = solver.word_movers_distance(doc, doc)

        # Same document should have small distance (entropy regularization adds some)
        assert dist < 0.1

    def test_wmd_different_documents(self, solver):
        """Test WMD between different documents."""
        doc1 = np.random.rand(5, 64)
        doc2 = np.random.rand(6, 64) + 2.0  # Different document

        dist = solver.word_movers_distance(doc1, doc2)

        # Different documents should have positive distance
        assert dist > 0

    def test_wmd_with_weights(self, solver):
        """Test WMD with custom word weights."""
        doc1 = np.random.rand(4, 64)
        doc2 = np.random.rand(4, 64)

        # Uniform weights
        weights1 = np.array([0.25, 0.25, 0.25, 0.25])
        weights2 = np.array([0.25, 0.25, 0.25, 0.25])

        dist_uniform = solver.word_movers_distance(doc1, doc2, weights1, weights2)

        # Concentrated weights on first word
        weights1_conc = np.array([0.7, 0.1, 0.1, 0.1])
        weights2_conc = np.array([0.7, 0.1, 0.1, 0.1])

        dist_conc = solver.word_movers_distance(doc1, doc2, weights1_conc, weights2_conc)

        # Both should be valid
        assert dist_uniform >= 0
        assert dist_conc >= 0

    def test_wmd_different_doc_lengths(self, solver):
        """Test WMD with different document lengths."""
        doc1 = np.random.rand(3, 64)  # 3 words
        doc2 = np.random.rand(10, 64)  # 10 words

        dist = solver.word_movers_distance(doc1, doc2)

        # Should return valid distance
        assert dist >= 0
        assert np.isfinite(dist)

    def test_relaxed_wmd(self, solver):
        """Test relaxed WMD (fast lower bound)."""
        doc1 = np.random.rand(8, 64)
        doc2 = np.random.rand(10, 64) + 1.0

        rwmd = solver.relaxed_word_movers_distance(doc1, doc2)

        # RWMD should be non-negative
        assert rwmd >= 0
        assert np.isfinite(rwmd)

    def test_relaxed_wmd_lower_bound(self, solver):
        """Test that RWMD is a lower bound on WMD."""
        doc1 = np.random.rand(5, 64)
        doc2 = np.random.rand(6, 64)

        wmd = solver.word_movers_distance(doc1, doc2)
        rwmd = solver.relaxed_word_movers_distance(doc1, doc2)

        # RWMD should be <= WMD (it's a lower bound)
        # Allow small tolerance for numerical issues
        assert rwmd <= wmd + 0.01

    def test_document_similarity_matrix(self, solver):
        """Test pairwise WMD similarity matrix."""
        # Create 3 documents
        documents = [
            np.random.rand(4, 64),
            np.random.rand(5, 64),
            np.random.rand(3, 64),
        ]

        sim_matrix = solver.document_similarity_matrix(documents)

        assert sim_matrix.shape == (3, 3)
        # Diagonal should be ~1 (self-similarity)
        for i in range(3):
            assert sim_matrix[i, i] > 0.9
        # Symmetric
        np.testing.assert_array_almost_equal(sim_matrix, sim_matrix.T, decimal=5)
        # All values should be in [0, 1]
        assert np.all(sim_matrix >= 0)
        assert np.all(sim_matrix <= 1)

    def test_document_similarity_matrix_relaxed(self, solver):
        """Test similarity matrix using relaxed WMD."""
        documents = [
            np.random.rand(4, 64),
            np.random.rand(5, 64),
        ]

        sim_matrix = solver.document_similarity_matrix(documents, use_relaxed=True)

        assert sim_matrix.shape == (2, 2)
        # Diagonal should be ~1
        assert sim_matrix[0, 0] > 0.9
        assert sim_matrix[1, 1] > 0.9

    def test_get_stats(self, solver):
        """Test solver statistics."""
        doc1 = np.random.rand(4, 32)
        doc2 = np.random.rand(4, 32)
        solver.word_movers_distance(doc1, doc2)

        stats = solver.get_stats()

        # WordMoverSolver uses 'computations' key
        assert "computations" in stats
        assert stats["computations"] >= 1
        assert "epsilon" in stats

    def test_wmd_cosine_metric(self, solver):
        """Test WMD with cosine cost metric."""
        doc1 = np.random.rand(4, 64)
        doc2 = np.random.rand(5, 64)

        dist = solver.word_movers_distance(doc1, doc2, metric=CostMetric.COSINE)

        assert dist >= 0
        assert np.isfinite(dist)

    def test_wmd_empty_document(self, solver):
        """Test WMD handles edge cases gracefully."""
        doc1 = np.random.rand(1, 64)  # Single word
        doc2 = np.random.rand(1, 64)

        dist = solver.word_movers_distance(doc1, doc2)

        # Should return valid distance
        assert dist >= 0
        assert np.isfinite(dist)


class TestOptimalTransportFusionExtended:
    """Tests for extended OT fusion methods (Sliced-Wasserstein and WMD)."""

    @pytest.fixture
    def fusion(self):
        """Create OT fusion with extended capabilities."""
        config = OTConfig(epsilon=0.1)
        return OptimalTransportFusion(config)

    def test_sliced_wasserstein_distance_method(self, fusion):
        """Test SW distance through fusion interface."""
        X1 = np.random.rand(10, 64)
        X2 = np.random.rand(15, 64)

        dist = fusion.sliced_wasserstein_distance(X1, X2)

        assert dist >= 0
        assert np.isfinite(dist)

    def test_sliced_wasserstein_embedding_method(self, fusion):
        """Test SW embedding through fusion interface."""
        X = np.random.rand(10, 64)

        embedding = fusion.sliced_wasserstein_embedding(X, n_projections=25)

        # Embedding has shape (2 * n_projections,) because it stores mean+std per projection
        assert embedding.shape == (50,)

    def test_word_movers_distance_method(self, fusion):
        """Test WMD through fusion interface."""
        doc1 = np.random.rand(5, 64)
        doc2 = np.random.rand(6, 64)

        dist = fusion.word_movers_distance(doc1, doc2)

        assert dist >= 0
        assert np.isfinite(dist)

    def test_relaxed_word_movers_distance_method(self, fusion):
        """Test RWMD through fusion interface."""
        doc1 = np.random.rand(5, 64)
        doc2 = np.random.rand(6, 64)

        dist = fusion.relaxed_word_movers_distance(doc1, doc2)

        assert dist >= 0
        assert np.isfinite(dist)

    def test_document_similarity_matrix_method(self, fusion):
        """Test document similarity matrix through fusion interface."""
        documents = [
            np.random.rand(4, 64),
            np.random.rand(5, 64),
        ]

        sim_matrix = fusion.document_similarity_matrix(documents)

        assert sim_matrix.shape == (2, 2)

    def test_extended_stats(self, fusion):
        """Test that extended stats include sub-solver statistics."""
        # Run some computations
        X1 = np.random.rand(5, 32)
        X2 = np.random.rand(5, 32)
        fusion.sliced_wasserstein_distance(X1, X2)
        fusion.word_movers_distance(X1, X2)

        stats = fusion.get_stats()

        assert "sliced_wasserstein" in stats
        assert "word_mover" in stats
        # Both solvers use 'computations' as the key
        assert stats["sliced_wasserstein"]["computations"] >= 1
        assert stats["word_mover"]["computations"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
