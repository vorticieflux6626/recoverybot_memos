"""
Unit Tests for Hyperbolic Embeddings Module

Tests the Poincaré ball geometry and hyperbolic retrieval for hierarchical documents.

Author: Claude Code
Date: December 2025
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

# Add parent paths for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentic.hyperbolic_embeddings import (
    PoincareBall,
    HyperbolicRetriever,
    HyperbolicDocument,
    HyperbolicSearchResult,
    HierarchyLevel,
    detect_hierarchy_level,
    get_hyperbolic_retriever,
)


class TestPoincareBall:
    """Tests for Poincaré ball geometry operations."""

    def test_initialization(self):
        """Test PoincareBall initialization with default and custom params."""
        # Default
        manifold = PoincareBall()
        assert manifold.dim == 768
        assert manifold.c == 1.0

        # Custom
        manifold = PoincareBall(dim=256, curvature=-2.0)
        assert manifold.dim == 256
        assert manifold.c == 2.0  # Absolute value

    def test_exp_map_origin(self):
        """Test exponential map from origin."""
        manifold = PoincareBall(dim=4)

        # Project a unit vector
        v = np.array([1.0, 0.0, 0.0, 0.0])
        result = manifold.exp_map(v)

        # Should be inside the ball (norm < 1)
        assert np.linalg.norm(result) < 1.0

        # Direction should be preserved
        assert result[0] > 0
        assert np.abs(result[1]) < 1e-5
        assert np.abs(result[2]) < 1e-5
        assert np.abs(result[3]) < 1e-5

    def test_exp_map_clamps_to_ball(self):
        """Test that exp_map keeps points inside the unit ball."""
        manifold = PoincareBall(dim=4)

        # Large vector that would exceed the ball
        v = np.array([10.0, 10.0, 10.0, 10.0])
        result = manifold.exp_map(v)

        # Should be inside the ball
        assert np.linalg.norm(result) < 1.0

    def test_log_map_inverse(self):
        """Test that log_map is inverse of exp_map near origin."""
        manifold = PoincareBall(dim=4)

        # Small vector
        v = np.array([0.1, 0.1, 0.0, 0.0])

        # exp then log should approximate original
        y = manifold.exp_map(v)
        v_recovered = manifold.log_map(y)

        # Should be close (within tolerance)
        np.testing.assert_array_almost_equal(v, v_recovered, decimal=3)

    def test_distance_symmetry(self):
        """Test that distance is symmetric."""
        manifold = PoincareBall(dim=4)

        x = manifold.exp_map(np.array([0.1, 0.0, 0.0, 0.0]))
        y = manifold.exp_map(np.array([0.0, 0.1, 0.0, 0.0]))

        d_xy = manifold.distance(x, y)
        d_yx = manifold.distance(y, x)

        assert abs(d_xy - d_yx) < 1e-6

    def test_distance_positive(self):
        """Test that distance is always positive (except for same point)."""
        manifold = PoincareBall(dim=4)

        x = manifold.exp_map(np.array([0.1, 0.0, 0.0, 0.0]))
        y = manifold.exp_map(np.array([0.0, 0.2, 0.0, 0.0]))

        d = manifold.distance(x, y)
        assert d > 0

        # Same point should have zero distance
        d_same = manifold.distance(x, x)
        assert d_same < 1e-6

    def test_get_depth(self):
        """Test radial depth calculation."""
        manifold = PoincareBall(dim=4)

        # Origin has zero depth
        origin = np.zeros(4)
        assert manifold.get_depth(origin) < 0.01

        # Points farther from origin have higher depth
        near = manifold.exp_map(np.array([0.1, 0.0, 0.0, 0.0]))
        far = manifold.exp_map(np.array([0.5, 0.0, 0.0, 0.0]))

        assert manifold.get_depth(far) > manifold.get_depth(near)

    def test_mobius_addition_identity(self):
        """Test Möbius addition with zero vector."""
        manifold = PoincareBall(dim=4)

        x = np.array([0.3, 0.2, 0.0, 0.0])
        zero = np.zeros(4)

        # x ⊕ 0 should equal x
        result = manifold._mobius_add(x, zero)
        np.testing.assert_array_almost_equal(x, result, decimal=5)


class TestHierarchyLevel:
    """Tests for hierarchy level enumeration and detection."""

    def test_hierarchy_values(self):
        """Test that hierarchy levels have correct ordering."""
        assert HierarchyLevel.CORPUS.value < HierarchyLevel.MANUAL.value
        assert HierarchyLevel.MANUAL.value < HierarchyLevel.CHAPTER.value
        assert HierarchyLevel.CHAPTER.value < HierarchyLevel.SECTION.value
        assert HierarchyLevel.SECTION.value < HierarchyLevel.PROCEDURE.value
        assert HierarchyLevel.PROCEDURE.value < HierarchyLevel.STEP.value

    def test_detect_error_code(self):
        """Test detection of error code content."""
        content = "SRVO-063 is an encoder alarm that occurs when..."
        level = detect_hierarchy_level(content)
        assert level == HierarchyLevel.STEP

    def test_detect_chapter(self):
        """Test detection of chapter-level content."""
        content = "Chapter 5: Introduction to Servo Systems"
        level = detect_hierarchy_level(content)
        assert level == HierarchyLevel.CHAPTER

    def test_detect_procedure(self):
        """Test detection of procedure-level content."""
        content = "Procedure for mastering the robot: Follow these steps carefully."
        level = detect_hierarchy_level(content)
        assert level == HierarchyLevel.PROCEDURE

    def test_detect_from_metadata(self):
        """Test detection from metadata hints."""
        content = "Some generic content"
        metadata = {"hierarchy_level": "MANUAL"}
        level = detect_hierarchy_level(content, metadata)
        assert level == HierarchyLevel.MANUAL

        metadata = {"type": "error_code"}
        level = detect_hierarchy_level(content, metadata)
        assert level == HierarchyLevel.STEP


class TestHyperbolicRetriever:
    """Tests for hyperbolic retrieval operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def retriever(self, temp_db):
        """Create a retriever with temp database."""
        return HyperbolicRetriever(dim=64, db_path=temp_db)

    @pytest.mark.asyncio
    async def test_add_document(self, retriever):
        """Test adding a document."""
        embedding = np.random.randn(64).astype(np.float32)

        doc = await retriever.add_document(
            doc_id="test-001",
            content="SRVO-063 encoder alarm",
            euclidean_embedding=embedding,
            hierarchy_level=HierarchyLevel.STEP,
            metadata={"category": "SRVO"}
        )

        assert doc.doc_id == "test-001"
        assert len(doc.hyperbolic_embedding) == 64
        assert doc.hierarchy_level == HierarchyLevel.STEP
        assert 0 <= doc.depth <= 1

    @pytest.mark.asyncio
    async def test_add_document_dimension_handling(self, retriever):
        """Test dimension mismatch handling."""
        # Too long - should truncate
        long_emb = np.random.randn(128).astype(np.float32)
        doc = await retriever.add_document(
            doc_id="long",
            content="Test",
            euclidean_embedding=long_emb,
        )
        assert len(doc.euclidean_embedding) == 64

        # Too short - should pad
        short_emb = np.random.randn(32).astype(np.float32)
        doc = await retriever.add_document(
            doc_id="short",
            content="Test",
            euclidean_embedding=short_emb,
        )
        assert len(doc.euclidean_embedding) == 64

    @pytest.mark.asyncio
    async def test_search_basic(self, retriever):
        """Test basic search functionality."""
        # Add some documents
        for i in range(5):
            embedding = np.random.randn(64).astype(np.float32)
            await retriever.add_document(
                doc_id=f"doc-{i}",
                content=f"Document {i} content",
                euclidean_embedding=embedding,
                hierarchy_level=HierarchyLevel.STEP,
            )

        # Search
        query_emb = np.random.randn(64).astype(np.float32)
        results = await retriever.search(query_emb, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, HyperbolicSearchResult) for r in results)
        # Results should be sorted by fused_score
        assert results[0].fused_score >= results[1].fused_score >= results[2].fused_score

    @pytest.mark.asyncio
    async def test_search_with_hierarchy_filter(self, retriever):
        """Test search with hierarchy level filtering."""
        # Add documents at different levels
        for i, level in enumerate([
            HierarchyLevel.CHAPTER,
            HierarchyLevel.SECTION,
            HierarchyLevel.STEP,
            HierarchyLevel.STEP,
            HierarchyLevel.PROCEDURE,
        ]):
            embedding = np.random.randn(64).astype(np.float32)
            await retriever.add_document(
                doc_id=f"doc-{i}",
                content=f"Document {i}",
                euclidean_embedding=embedding,
                hierarchy_level=level,
            )

        # Search only STEP level
        query_emb = np.random.randn(64).astype(np.float32)
        results = await retriever.search(
            query_emb,
            top_k=10,
            hierarchy_filter=[HierarchyLevel.STEP]
        )

        assert len(results) == 2
        assert all(r.hierarchy_level == HierarchyLevel.STEP for r in results)

    @pytest.mark.asyncio
    async def test_search_hyperbolic_only(self, retriever):
        """Test search using only hyperbolic scores."""
        for i in range(3):
            embedding = np.random.randn(64).astype(np.float32)
            await retriever.add_document(
                doc_id=f"doc-{i}",
                content=f"Document {i}",
                euclidean_embedding=embedding,
            )

        query_emb = np.random.randn(64).astype(np.float32)
        results = await retriever.search(query_emb, top_k=3, use_fusion=False)

        # When use_fusion=False, fused_score should equal hyperbolic_score
        for r in results:
            assert abs(r.fused_score - r.hyperbolic_score) < 1e-6

    @pytest.mark.asyncio
    async def test_search_by_hierarchy(self, retriever):
        """Test hierarchy-aware search expansion."""
        # Add documents at multiple levels
        for i, level in enumerate([
            HierarchyLevel.MANUAL,
            HierarchyLevel.CHAPTER,
            HierarchyLevel.SECTION,
            HierarchyLevel.STEP,
        ]):
            embedding = np.random.randn(64).astype(np.float32)
            await retriever.add_document(
                doc_id=f"doc-{i}",
                content=f"Document at {level.name}",
                euclidean_embedding=embedding,
                hierarchy_level=level,
            )

        query_emb = np.random.randn(64).astype(np.float32)
        results = await retriever.search_by_hierarchy(
            query_emb,
            target_level=HierarchyLevel.SECTION,
            top_k=5,
            include_parents=True,
            include_children=True,
        )

        # Should have results for multiple levels
        assert len(results) > 0
        # Should include the target level
        assert "SECTION" in results

    @pytest.mark.asyncio
    async def test_get_document_tree(self, retriever):
        """Test document tree retrieval."""
        # Create parent
        parent_emb = np.random.randn(64).astype(np.float32)
        await retriever.add_document(
            doc_id="parent",
            content="Parent chapter",
            euclidean_embedding=parent_emb,
            hierarchy_level=HierarchyLevel.CHAPTER,
        )

        # Create children
        for i in range(3):
            child_emb = np.random.randn(64).astype(np.float32)
            await retriever.add_document(
                doc_id=f"child-{i}",
                content=f"Child section {i}",
                euclidean_embedding=child_emb,
                hierarchy_level=HierarchyLevel.SECTION,
                parent_id="parent",
            )

        tree = await retriever.get_document_tree("parent")

        assert tree["doc_id"] == "parent"
        assert len(tree["children"]) == 3
        assert tree["parent"] is None

    def test_get_stats(self, retriever):
        """Test statistics retrieval."""
        stats = retriever.get_stats()

        assert "total_documents" in stats
        assert "by_hierarchy_level" in stats
        assert "embedding_dim" in stats
        assert "curvature" in stats
        assert "euclidean_weight" in stats
        assert "hyperbolic_weight" in stats

    @pytest.mark.asyncio
    async def test_persistence(self, temp_db):
        """Test that documents persist across retriever instances."""
        # Create first retriever and add documents
        retriever1 = HyperbolicRetriever(dim=64, db_path=temp_db)
        embedding = np.random.randn(64).astype(np.float32)
        await retriever1.add_document(
            doc_id="persist-test",
            content="This should persist",
            euclidean_embedding=embedding,
        )

        # Create new retriever with same database
        retriever2 = HyperbolicRetriever(dim=64, db_path=temp_db)
        retriever2.load_all_documents()

        assert "persist-test" in retriever2.documents
        assert retriever2.documents["persist-test"].content == "This should persist"


class TestHyperbolicProjection:
    """Tests for Euclidean to hyperbolic projection."""

    @pytest.fixture
    def retriever(self):
        """Create a retriever for projection tests."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        retriever = HyperbolicRetriever(dim=64, db_path=db_path)
        yield retriever
        if os.path.exists(db_path):
            os.unlink(db_path)

    def test_projection_stays_in_ball(self, retriever):
        """Test that projections stay inside unit ball."""
        for _ in range(100):
            emb = np.random.randn(64).astype(np.float32) * 10  # Large random vectors
            projected = retriever.project_to_hyperbolic(emb, HierarchyLevel.STEP)
            assert np.linalg.norm(projected) < 1.0

    def test_hierarchy_affects_depth(self, retriever):
        """Test that hierarchy level affects radial depth."""
        emb = np.random.randn(64).astype(np.float32)

        # Project same embedding at different hierarchy levels
        general = retriever.project_to_hyperbolic(emb, HierarchyLevel.CORPUS)
        specific = retriever.project_to_hyperbolic(emb, HierarchyLevel.STEP)

        general_depth = retriever.manifold.get_depth(general)
        specific_depth = retriever.manifold.get_depth(specific)

        # More specific should be deeper (closer to boundary)
        assert specific_depth > general_depth


class TestSingletonRetriever:
    """Tests for singleton retriever pattern."""

    def test_get_hyperbolic_retriever(self):
        """Test singleton accessor."""
        # Clear any existing singleton
        import agentic.hyperbolic_embeddings as module
        module._hyperbolic_retriever = None

        retriever1 = get_hyperbolic_retriever(dim=128)
        retriever2 = get_hyperbolic_retriever(dim=256)  # Dimension ignored

        # Should be same instance
        assert retriever1 is retriever2
        assert retriever1.dim == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
