"""
Contrastive Retriever Training (R3)

Based on R3 research: Trial-and-feedback learning for retriever improvement.
Tracks which retrieved documents were actually useful in synthesis vs. which
were retrieved but not used, creating contrastive signals for adaptation.

Key Concepts:
- Retrieval Outcome Tracking: Records which docs contributed to synthesis
- Contrastive Pairs: Positive (used) vs Negative (retrieved but unused)
- Adaptive Weights: Adjusts re-ranking weights based on feedback
- Query Expansion Tuning: Learns which expansion strategies work

Expected Impact: ~5.2% retriever improvement (per R3 arXiv research)
"""

import json
import logging
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import statistics

logger = logging.getLogger("agentic.contrastive_retriever")


class DocumentUtility(Enum):
    """How useful a retrieved document was in synthesis"""
    ESSENTIAL = "essential"      # Directly cited in synthesis
    SUPPORTING = "supporting"    # Contributed context but not cited
    PERIPHERAL = "peripheral"    # Marginally relevant
    NOISE = "noise"              # Retrieved but not used at all


class RetrievalStrategy(Enum):
    """Retrieval strategy used"""
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID = "hybrid"
    HYDE_EXPANDED = "hyde_expanded"
    QUERY_TREE = "query_tree"
    MULTI_HOP = "multi_hop"


@dataclass
class DocumentOutcome:
    """Outcome tracking for a single retrieved document"""
    doc_id: str
    doc_url: str
    retrieval_score: float         # Original retrieval score (0-1)
    retrieval_rank: int            # Position in retrieval results (1-based)
    actual_utility: DocumentUtility # How useful it actually was
    was_cited: bool                # Whether cited in final synthesis
    citation_count: int = 0        # Number of times cited
    key_facts_used: int = 0        # Number of facts extracted and used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_url": self.doc_url,
            "retrieval_score": self.retrieval_score,
            "retrieval_rank": self.retrieval_rank,
            "actual_utility": self.actual_utility.value,
            "was_cited": self.was_cited,
            "citation_count": self.citation_count,
            "key_facts_used": self.key_facts_used
        }


@dataclass
class RetrievalSession:
    """Complete retrieval session with outcomes"""
    session_id: str
    query: str
    query_hash: str
    timestamp: datetime
    strategy: RetrievalStrategy

    # Retrieved documents with outcomes
    documents: List[DocumentOutcome] = field(default_factory=list)

    # Session-level metrics
    synthesis_confidence: float = 0.0
    total_retrieved: int = 0
    actually_used: int = 0
    precision_at_k: float = 0.0   # % of top-k that were useful

    # Strategy-specific metadata
    expansion_terms: List[str] = field(default_factory=list)
    rerank_model: Optional[str] = None
    fusion_weights: Optional[Dict[str, float]] = None

    def calculate_metrics(self) -> None:
        """Calculate session-level metrics from document outcomes"""
        self.total_retrieved = len(self.documents)
        self.actually_used = sum(
            1 for d in self.documents
            if d.actual_utility in (DocumentUtility.ESSENTIAL, DocumentUtility.SUPPORTING)
        )
        if self.total_retrieved > 0:
            self.precision_at_k = self.actually_used / self.total_retrieved

    def get_contrastive_pairs(self) -> List[Tuple[DocumentOutcome, DocumentOutcome]]:
        """
        Generate contrastive pairs for learning.
        Pairs a positive (cited) doc with a negative (not cited) doc.
        """
        positives = [d for d in self.documents if d.was_cited]
        negatives = [d for d in self.documents if not d.was_cited]

        pairs = []
        for pos in positives:
            for neg in negatives:
                # Only pair if negative was ranked higher (a retrieval mistake)
                if neg.retrieval_rank < pos.retrieval_rank:
                    pairs.append((pos, neg))
        return pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query[:100],
            "query_hash": self.query_hash,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy.value,
            "documents": [d.to_dict() for d in self.documents],
            "synthesis_confidence": self.synthesis_confidence,
            "total_retrieved": self.total_retrieved,
            "actually_used": self.actually_used,
            "precision_at_k": self.precision_at_k,
            "expansion_terms": self.expansion_terms,
            "rerank_model": self.rerank_model,
            "fusion_weights": self.fusion_weights
        }


@dataclass
class RetrievalInsight:
    """Learning insight from contrastive analysis"""
    insight_type: str  # "weight_adjustment", "term_boost", "domain_priority"
    description: str
    adjustment: Dict[str, Any]
    confidence: float
    support_count: int  # Number of sessions supporting this insight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "description": self.description,
            "adjustment": self.adjustment,
            "confidence": self.confidence,
            "support_count": self.support_count
        }


class ContrastiveRetriever:
    """
    Contrastive learning manager for retrieval optimization.

    Tracks retrieval outcomes and generates insights for improving
    future retrievals based on what actually worked.
    """

    def __init__(self, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self.sessions: List[RetrievalSession] = []
        self.insights: List[RetrievalInsight] = []

        # Aggregated statistics
        self.strategy_stats: Dict[RetrievalStrategy, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "avg_precision": 0.0, "avg_confidence": 0.0}
        )
        self.domain_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"retrieved": 0, "used": 0, "utility_rate": 0.0}
        )

        # Learned adjustments
        self.domain_weights: Dict[str, float] = {}  # Domain -> weight multiplier
        self.term_boosts: Dict[str, float] = {}     # Term -> boost value
        self.strategy_preferences: Dict[str, RetrievalStrategy] = {}  # Query type -> best strategy

    def record_session(
        self,
        query: str,
        strategy: RetrievalStrategy,
        documents: List[Dict[str, Any]],
        synthesis_confidence: float,
        cited_urls: Set[str] = None,
        expansion_terms: List[str] = None,
        rerank_model: str = None,
        fusion_weights: Dict[str, float] = None
    ) -> RetrievalSession:
        """
        Record a retrieval session with outcome data.

        Args:
            query: The search query
            strategy: Which retrieval strategy was used
            documents: List of retrieved docs with scores and URLs
            synthesis_confidence: Final synthesis confidence score
            cited_urls: Set of URLs that were actually cited
            expansion_terms: Query expansion terms used
            rerank_model: Name of reranking model if used
            fusion_weights: Hybrid fusion weights if used

        Returns:
            The recorded session
        """
        cited_urls = cited_urls or set()

        # Create session
        session_id = hashlib.sha256(
            f"{query}{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        query_hash = hashlib.sha256(query.lower().encode()).hexdigest()[:16]

        session = RetrievalSession(
            session_id=session_id,
            query=query,
            query_hash=query_hash,
            timestamp=datetime.now(timezone.utc),
            strategy=strategy,
            synthesis_confidence=synthesis_confidence,
            expansion_terms=expansion_terms or [],
            rerank_model=rerank_model,
            fusion_weights=fusion_weights
        )

        # Process each document
        for rank, doc in enumerate(documents, 1):
            url = doc.get("url", "")
            was_cited = url in cited_urls

            # Determine utility based on citation and score
            if was_cited:
                utility = DocumentUtility.ESSENTIAL
            elif doc.get("score", 0) >= 0.7:
                utility = DocumentUtility.SUPPORTING
            elif doc.get("score", 0) >= 0.4:
                utility = DocumentUtility.PERIPHERAL
            else:
                utility = DocumentUtility.NOISE

            outcome = DocumentOutcome(
                doc_id=doc.get("id", f"doc_{rank}"),
                doc_url=url,
                retrieval_score=doc.get("score", 0.0),
                retrieval_rank=rank,
                actual_utility=utility,
                was_cited=was_cited,
                citation_count=1 if was_cited else 0,
                key_facts_used=doc.get("facts_used", 0)
            )
            session.documents.append(outcome)

        # Calculate metrics
        session.calculate_metrics()

        # Store session
        self.sessions.append(session)
        if len(self.sessions) > self.max_sessions:
            self.sessions.pop(0)

        # Update aggregated statistics
        self._update_stats(session)

        # Generate insights periodically
        if len(self.sessions) % 10 == 0:
            self._generate_insights()

        logger.debug(
            f"Recorded retrieval session {session_id}: "
            f"{session.actually_used}/{session.total_retrieved} docs used "
            f"(precision: {session.precision_at_k:.2f})"
        )

        return session

    def _update_stats(self, session: RetrievalSession) -> None:
        """Update aggregated statistics from a session"""
        # Strategy stats
        stats = self.strategy_stats[session.strategy]
        n = stats["count"]
        stats["count"] = n + 1
        # Rolling average
        stats["avg_precision"] = (
            (stats["avg_precision"] * n + session.precision_at_k) / (n + 1)
        )
        stats["avg_confidence"] = (
            (stats["avg_confidence"] * n + session.synthesis_confidence) / (n + 1)
        )

        # Domain stats
        for doc in session.documents:
            domain = self._extract_domain(doc.doc_url)
            if domain:
                self.domain_stats[domain]["retrieved"] += 1
                if doc.was_cited:
                    self.domain_stats[domain]["used"] += 1
                # Update utility rate
                total = self.domain_stats[domain]["retrieved"]
                used = self.domain_stats[domain]["used"]
                self.domain_stats[domain]["utility_rate"] = used / total if total > 0 else 0.0

    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return None

    def _generate_insights(self) -> None:
        """Generate insights from accumulated sessions"""
        if len(self.sessions) < 10:
            return

        # Insight 1: Domain weight adjustments
        for domain, stats in self.domain_stats.items():
            if stats["retrieved"] >= 5:  # Minimum sample size
                utility_rate = stats["utility_rate"]
                if utility_rate >= 0.6:
                    # High utility domain - boost weight
                    self.domain_weights[domain] = 1.0 + (utility_rate - 0.5)
                    self.insights.append(RetrievalInsight(
                        insight_type="domain_boost",
                        description=f"Boost {domain} (utility rate: {utility_rate:.2f})",
                        adjustment={"domain": domain, "weight": self.domain_weights[domain]},
                        confidence=min(1.0, utility_rate),
                        support_count=stats["retrieved"]
                    ))
                elif utility_rate < 0.2:
                    # Low utility domain - reduce weight
                    self.domain_weights[domain] = max(0.3, 0.5 + utility_rate)
                    self.insights.append(RetrievalInsight(
                        insight_type="domain_reduce",
                        description=f"Reduce {domain} (utility rate: {utility_rate:.2f})",
                        adjustment={"domain": domain, "weight": self.domain_weights[domain]},
                        confidence=min(1.0, 1.0 - utility_rate),
                        support_count=stats["retrieved"]
                    ))

        # Insight 2: Strategy preferences
        best_strategy = None
        best_precision = 0.0
        for strategy, stats in self.strategy_stats.items():
            if stats["count"] >= 5 and stats["avg_precision"] > best_precision:
                best_precision = stats["avg_precision"]
                best_strategy = strategy

        if best_strategy:
            self.insights.append(RetrievalInsight(
                insight_type="strategy_preference",
                description=f"Best strategy: {best_strategy.value} (precision: {best_precision:.2f})",
                adjustment={"preferred_strategy": best_strategy.value},
                confidence=best_precision,
                support_count=self.strategy_stats[best_strategy]["count"]
            ))

        # Limit insights
        if len(self.insights) > 100:
            # Keep most confident
            self.insights.sort(key=lambda x: x.confidence, reverse=True)
            self.insights = self.insights[:100]

        logger.debug(f"Generated {len(self.insights)} insights from {len(self.sessions)} sessions")

    def get_contrastive_pairs(
        self,
        min_rank_gap: int = 2,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get contrastive pairs for learning.

        Returns pairs where a lower-ranked document was more useful than
        a higher-ranked one (retrieval mistakes).

        Args:
            min_rank_gap: Minimum rank difference for a pair
            limit: Maximum number of pairs to return

        Returns:
            List of contrastive pair dicts
        """
        all_pairs = []

        for session in self.sessions[-50:]:  # Recent sessions only
            pairs = session.get_contrastive_pairs()
            for pos, neg in pairs:
                # pos.retrieval_rank > neg.retrieval_rank (cited doc ranked worse)
                rank_gap = pos.retrieval_rank - neg.retrieval_rank
                if rank_gap >= min_rank_gap:
                    all_pairs.append({
                        "session_id": session.session_id,
                        "query": session.query[:100],
                        "positive": {
                            "url": pos.doc_url,
                            "rank": pos.retrieval_rank,
                            "score": pos.retrieval_score
                        },
                        "negative": {
                            "url": neg.doc_url,
                            "rank": neg.retrieval_rank,
                            "score": neg.retrieval_score
                        },
                        "rank_gap": pos.retrieval_rank - neg.retrieval_rank
                    })

        # Sort by rank gap (worst mistakes first)
        all_pairs.sort(key=lambda x: x["rank_gap"], reverse=True)
        return all_pairs[:limit]

    def get_domain_weights(self) -> Dict[str, float]:
        """Get learned domain weight adjustments"""
        return dict(self.domain_weights)

    def get_strategy_recommendation(self, query_type: str = None) -> Optional[RetrievalStrategy]:
        """
        Get recommended strategy based on historical performance.

        Args:
            query_type: Optional query type for specific recommendation

        Returns:
            Best performing strategy or None
        """
        if not self.strategy_stats:
            return None

        best_strategy = None
        best_score = 0.0

        for strategy, stats in self.strategy_stats.items():
            if stats["count"] >= 3:
                # Weighted score: precision (60%) + confidence (40%)
                score = stats["avg_precision"] * 0.6 + stats["avg_confidence"] * 0.4
                if score > best_score:
                    best_score = score
                    best_strategy = strategy

        return best_strategy

    def adjust_retrieval_scores(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Adjust retrieval scores based on learned domain weights.

        Args:
            documents: Retrieved documents with scores and URLs

        Returns:
            Documents with adjusted scores
        """
        if not self.domain_weights:
            return documents

        adjusted = []
        for doc in documents:
            domain = self._extract_domain(doc.get("url", ""))
            weight = self.domain_weights.get(domain, 1.0)

            adjusted_doc = doc.copy()
            adjusted_doc["original_score"] = doc.get("score", 0.0)
            adjusted_doc["score"] = min(1.0, doc.get("score", 0.0) * weight)
            adjusted_doc["weight_applied"] = weight
            adjusted.append(adjusted_doc)

        # Re-sort by adjusted score
        adjusted.sort(key=lambda x: x["score"], reverse=True)
        return adjusted

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "total_sessions": len(self.sessions),
            "total_insights": len(self.insights),
            "domain_weights_learned": len(self.domain_weights),
            "strategy_stats": {
                s.value: stats for s, stats in self.strategy_stats.items()
            },
            "top_domains": sorted(
                [
                    {"domain": d, **stats}
                    for d, stats in self.domain_stats.items()
                    if stats["retrieved"] >= 3
                ],
                key=lambda x: x["utility_rate"],
                reverse=True
            )[:10],
            "recent_precision": (
                statistics.mean([s.precision_at_k for s in self.sessions[-20:]])
                if self.sessions else 0.0
            ),
            "contrastive_pairs_available": sum(
                len(s.get_contrastive_pairs()) for s in self.sessions[-50:]
            )
        }

    def get_insights(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top insights"""
        sorted_insights = sorted(
            self.insights,
            key=lambda x: (x.confidence, x.support_count),
            reverse=True
        )
        return [i.to_dict() for i in sorted_insights[:limit]]


# Singleton instance
_contrastive_retriever: Optional[ContrastiveRetriever] = None


def get_contrastive_retriever() -> ContrastiveRetriever:
    """Get or create singleton ContrastiveRetriever instance"""
    global _contrastive_retriever
    if _contrastive_retriever is None:
        _contrastive_retriever = ContrastiveRetriever()
    return _contrastive_retriever
