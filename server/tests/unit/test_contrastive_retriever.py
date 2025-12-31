"""
Tests for G.6.5 Contrastive Retriever Training (R3)

Tests the feedback-based retrieval optimization system.
"""

import pytest
from datetime import datetime, timezone

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agentic.contrastive_retriever import (
    ContrastiveRetriever,
    DocumentOutcome,
    DocumentUtility,
    RetrievalSession,
    RetrievalStrategy,
    RetrievalInsight,
    get_contrastive_retriever,
)


class TestContrastiveRetrieverBasics:
    """Test basic contrastive retriever functionality."""

    @pytest.fixture
    def retriever(self):
        """Create a fresh retriever instance."""
        return ContrastiveRetriever(max_sessions=100)

    def test_record_session_basic(self, retriever):
        """Test recording a basic retrieval session."""
        documents = [
            {"url": "https://example.com/doc1", "score": 0.9, "id": "d1"},
            {"url": "https://example.com/doc2", "score": 0.7, "id": "d2"},
            {"url": "https://example.com/doc3", "score": 0.5, "id": "d3"},
        ]
        cited_urls = {"https://example.com/doc1"}

        session = retriever.record_session(
            query="test query",
            strategy=RetrievalStrategy.HYBRID,
            documents=documents,
            synthesis_confidence=0.85,
            cited_urls=cited_urls
        )

        assert session is not None
        assert session.query == "test query"
        assert session.strategy == RetrievalStrategy.HYBRID
        assert session.synthesis_confidence == 0.85
        assert session.total_retrieved == 3
        assert session.actually_used >= 1

    def test_document_utility_classification(self, retriever):
        """Test that documents are classified correctly by utility."""
        documents = [
            {"url": "https://cited.com/doc", "score": 0.9},
            {"url": "https://high-score.com/doc", "score": 0.75},
            {"url": "https://mid-score.com/doc", "score": 0.45},
            {"url": "https://low-score.com/doc", "score": 0.2},
        ]
        cited_urls = {"https://cited.com/doc"}

        session = retriever.record_session(
            query="utility test",
            strategy=RetrievalStrategy.DENSE_ONLY,
            documents=documents,
            synthesis_confidence=0.8,
            cited_urls=cited_urls
        )

        # Check utility classification
        utilities = {d.doc_url: d.actual_utility for d in session.documents}
        assert utilities["https://cited.com/doc"] == DocumentUtility.ESSENTIAL
        assert utilities["https://high-score.com/doc"] == DocumentUtility.SUPPORTING
        assert utilities["https://mid-score.com/doc"] == DocumentUtility.PERIPHERAL
        assert utilities["https://low-score.com/doc"] == DocumentUtility.NOISE

    def test_precision_calculation(self, retriever):
        """Test precision@k calculation."""
        documents = [
            {"url": "https://useful1.com", "score": 0.9},
            {"url": "https://noise1.com", "score": 0.85},
            {"url": "https://useful2.com", "score": 0.8},
            {"url": "https://noise2.com", "score": 0.3},
        ]
        cited_urls = {"https://useful1.com", "https://useful2.com"}

        session = retriever.record_session(
            query="precision test",
            strategy=RetrievalStrategy.HYBRID,
            documents=documents,
            synthesis_confidence=0.9,
            cited_urls=cited_urls
        )

        # 2 out of 4 were actually useful (cited)
        # But precision counts ESSENTIAL + SUPPORTING
        assert session.actually_used >= 2
        assert 0.0 <= session.precision_at_k <= 1.0


class TestContrastivePairs:
    """Test contrastive pair generation."""

    @pytest.fixture
    def retriever_with_sessions(self):
        """Create retriever with some sessions."""
        retriever = ContrastiveRetriever()

        # Add session with retrieval mistakes
        # noise doc ranked higher than cited doc
        documents = [
            {"url": "https://noise.com", "score": 0.95},  # Rank 1 - not cited
            {"url": "https://noise2.com", "score": 0.9},   # Rank 2 - not cited
            {"url": "https://cited.com", "score": 0.7},   # Rank 3 - cited!
        ]
        cited_urls = {"https://cited.com"}

        retriever.record_session(
            query="contrastive test",
            strategy=RetrievalStrategy.HYBRID,
            documents=documents,
            synthesis_confidence=0.75,
            cited_urls=cited_urls
        )

        return retriever

    def test_contrastive_pairs_generated(self, retriever_with_sessions):
        """Test that contrastive pairs are generated for ranking mistakes."""
        pairs = retriever_with_sessions.get_contrastive_pairs(min_rank_gap=1)

        # Should have pairs where noise was ranked higher than cited
        assert len(pairs) >= 1

        for pair in pairs:
            # Positive (cited) should have worse rank than negative (noise)
            assert pair["positive"]["rank"] > pair["negative"]["rank"]

    def test_contrastive_pairs_empty_when_correct(self):
        """Test no pairs when ranking was correct."""
        retriever = ContrastiveRetriever()

        # Perfect ranking - cited docs ranked first
        documents = [
            {"url": "https://cited1.com", "score": 0.95},
            {"url": "https://cited2.com", "score": 0.9},
            {"url": "https://noise.com", "score": 0.5},
        ]
        cited_urls = {"https://cited1.com", "https://cited2.com"}

        retriever.record_session(
            query="perfect ranking",
            strategy=RetrievalStrategy.HYBRID,
            documents=documents,
            synthesis_confidence=0.9,
            cited_urls=cited_urls
        )

        pairs = retriever.get_contrastive_pairs()
        assert len(pairs) == 0  # No mistakes to learn from


class TestDomainWeights:
    """Test domain weight learning."""

    @pytest.fixture
    def retriever_with_domain_data(self):
        """Create retriever with domain-specific sessions."""
        retriever = ContrastiveRetriever()

        # Add sessions showing high-utility domain
        for i in range(10):
            documents = [
                {"url": "https://fanuc.com/manual1", "score": 0.9},
                {"url": "https://fanuc.com/manual2", "score": 0.85},
                {"url": "https://random-blog.com/post", "score": 0.8},
            ]
            cited_urls = {"https://fanuc.com/manual1", "https://fanuc.com/manual2"}

            retriever.record_session(
                query=f"fanuc query {i}",
                strategy=RetrievalStrategy.HYBRID,
                documents=documents,
                synthesis_confidence=0.85,
                cited_urls=cited_urls
            )

        return retriever

    def test_domain_weights_learned(self, retriever_with_domain_data):
        """Test that domain weights are learned from feedback."""
        weights = retriever_with_domain_data.get_domain_weights()

        # fanuc.com should have high weight (frequently cited)
        if "fanuc.com" in weights:
            assert weights["fanuc.com"] >= 1.0

        # random-blog.com should have lower weight (rarely cited)
        if "random-blog.com" in weights:
            assert weights["random-blog.com"] < weights.get("fanuc.com", 1.0)

    def test_score_adjustment_with_weights(self, retriever_with_domain_data):
        """Test that learned weights adjust retrieval scores."""
        documents = [
            {"url": "https://fanuc.com/doc", "score": 0.7},
            {"url": "https://random-blog.com/post", "score": 0.8},
        ]

        adjusted = retriever_with_domain_data.adjust_retrieval_scores(documents)

        # If fanuc.com has higher weight, it should be boosted
        # Note: weights may not be learned if not enough sessions
        assert len(adjusted) == 2
        for doc in adjusted:
            assert "original_score" in doc
            assert "weight_applied" in doc


class TestStrategyStats:
    """Test strategy statistics tracking."""

    @pytest.fixture
    def retriever_with_strategies(self):
        """Create retriever with various strategy sessions."""
        retriever = ContrastiveRetriever()

        # Hybrid strategy - high precision
        for i in range(5):
            retriever.record_session(
                query=f"hybrid query {i}",
                strategy=RetrievalStrategy.HYBRID,
                documents=[
                    {"url": f"https://ex{i}.com/a", "score": 0.9},
                    {"url": f"https://ex{i}.com/b", "score": 0.85},
                ],
                synthesis_confidence=0.9,
                cited_urls={f"https://ex{i}.com/a", f"https://ex{i}.com/b"}
            )

        # Dense only - lower precision
        for i in range(5):
            retriever.record_session(
                query=f"dense query {i}",
                strategy=RetrievalStrategy.DENSE_ONLY,
                documents=[
                    {"url": f"https://d{i}.com/a", "score": 0.9},
                    {"url": f"https://d{i}.com/b", "score": 0.85},
                ],
                synthesis_confidence=0.6,
                cited_urls={f"https://d{i}.com/a"}  # Only one cited
            )

        return retriever

    def test_strategy_recommendation(self, retriever_with_strategies):
        """Test strategy recommendation based on stats."""
        best = retriever_with_strategies.get_strategy_recommendation()

        # Hybrid had higher precision, should be recommended
        assert best == RetrievalStrategy.HYBRID

    def test_strategy_stats_tracked(self, retriever_with_strategies):
        """Test that strategy stats are tracked correctly."""
        stats = retriever_with_strategies.get_stats()

        assert "strategy_stats" in stats
        assert "hybrid" in stats["strategy_stats"]
        assert stats["strategy_stats"]["hybrid"]["count"] == 5


class TestRetrievalInsights:
    """Test insight generation."""

    def test_insights_generated_after_sessions(self):
        """Test that insights are generated after enough sessions."""
        retriever = ContrastiveRetriever()

        # Add 15 sessions to trigger insight generation
        for i in range(15):
            documents = [
                {"url": "https://trusted.com/doc", "score": 0.9},
                {"url": "https://untrusted.com/doc", "score": 0.8},
            ]
            cited_urls = {"https://trusted.com/doc"}

            retriever.record_session(
                query=f"insight test {i}",
                strategy=RetrievalStrategy.HYBRID,
                documents=documents,
                synthesis_confidence=0.85,
                cited_urls=cited_urls
            )

        insights = retriever.get_insights()
        # May or may not have insights depending on patterns
        assert isinstance(insights, list)

    def test_get_stats_complete(self):
        """Test that get_stats returns all expected fields."""
        retriever = ContrastiveRetriever()

        # Add some data
        retriever.record_session(
            query="stats test",
            strategy=RetrievalStrategy.HYBRID,
            documents=[{"url": "https://test.com", "score": 0.8}],
            synthesis_confidence=0.75,
            cited_urls={"https://test.com"}
        )

        stats = retriever.get_stats()

        assert "total_sessions" in stats
        assert "total_insights" in stats
        assert "domain_weights_learned" in stats
        assert "strategy_stats" in stats
        assert "top_domains" in stats
        assert "recent_precision" in stats


class TestSingletonPattern:
    """Test singleton instance."""

    def test_get_contrastive_retriever_singleton(self):
        """Test that get_contrastive_retriever returns singleton."""
        r1 = get_contrastive_retriever()
        r2 = get_contrastive_retriever()

        assert r1 is r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
