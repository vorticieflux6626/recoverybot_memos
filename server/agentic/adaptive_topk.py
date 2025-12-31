"""
Adaptive Top-K Retrieval using CAR (Calibrated Adaptive Retrieval) Algorithm.

Part of G.2.5: Adaptive top-k selection for efficient retrieval.

Dynamically adjusts the number of documents retrieved based on:
- Query complexity estimation
- Score distribution analysis
- Historical performance tracking
- Early stopping conditions

Research Basis:
- CAR: Calibrated Adaptive Retrieval (NAACL 2024)
- Self-RAG: Early stopping based on confidence
- FLARE: Retrieval budgeting based on uncertainty

Usage:
    from agentic.adaptive_topk import AdaptiveTopK, QueryComplexity

    adapter = AdaptiveTopK()
    top_k = adapter.compute_adaptive_k(
        query="SRVO-063 alarm meaning",
        initial_k=50
    )
    # Returns: AdaptiveKResult(recommended_k=15, complexity=...)

    # Or use with score distribution
    final_k = adapter.apply_early_stopping(
        scores=[0.92, 0.87, 0.85, 0.71, 0.65, ...],
        initial_k=50
    )
    # Returns: 4 (stops when score drops significantly)
"""

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

logger = logging.getLogger("agentic.adaptive_topk")


class QueryComplexity(str, Enum):
    """Query complexity levels for adaptive retrieval."""
    SIMPLE = "simple"           # Single concept, direct lookup
    MODERATE = "moderate"       # Multi-concept, some reasoning
    COMPLEX = "complex"         # Multi-hop, synthesis required
    EXPLORATORY = "exploratory" # Open-ended, research-style


class StoppingReason(str, Enum):
    """Reasons for early stopping."""
    SCORE_PLATEAU = "score_plateau"       # Scores plateaued
    SCORE_CLIFF = "score_cliff"           # Sharp score drop
    DIVERSITY_SATURATED = "diversity_saturated"  # No new information
    CONFIDENCE_THRESHOLD = "confidence_threshold"  # High confidence reached
    BUDGET_EXHAUSTED = "budget_exhausted"  # Max k reached


@dataclass
class ComplexityFeatures:
    """Features used for complexity estimation."""
    token_count: int
    entity_count: int
    question_words: int  # how, why, what, etc.
    conjunction_count: int  # and, or, but
    technical_terms: int
    negation_count: int
    comparison_words: int  # vs, compare, difference
    temporal_words: int  # when, before, after
    quantifier_words: int  # all, some, most

    def to_dict(self) -> Dict[str, int]:
        return {
            "token_count": self.token_count,
            "entity_count": self.entity_count,
            "question_words": self.question_words,
            "conjunction_count": self.conjunction_count,
            "technical_terms": self.technical_terms,
            "negation_count": self.negation_count,
            "comparison_words": self.comparison_words,
            "temporal_words": self.temporal_words,
            "quantifier_words": self.quantifier_words
        }


@dataclass
class ScoreDistribution:
    """Analysis of score distribution for stopping decisions."""
    mean: float
    std: float
    max_score: float
    min_score: float
    score_range: float
    gradient: float  # Rate of score decline
    plateau_detected: bool
    cliff_index: Optional[int]  # Index where cliff occurs
    knee_index: Optional[int]   # Index of optimal cutoff

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "max_score": round(self.max_score, 4),
            "min_score": round(self.min_score, 4),
            "score_range": round(self.score_range, 4),
            "gradient": round(self.gradient, 4),
            "plateau_detected": self.plateau_detected,
            "cliff_index": self.cliff_index,
            "knee_index": self.knee_index
        }


@dataclass
class AdaptiveKResult:
    """Result of adaptive k computation."""
    recommended_k: int
    complexity: QueryComplexity
    features: ComplexityFeatures
    base_k: int
    adjustment_factor: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_k": self.recommended_k,
            "complexity": self.complexity.value,
            "features": self.features.to_dict(),
            "base_k": self.base_k,
            "adjustment_factor": round(self.adjustment_factor, 3),
            "reasoning": self.reasoning
        }


@dataclass
class EarlyStopResult:
    """Result of early stopping analysis."""
    final_k: int
    original_k: int
    reason: StoppingReason
    distribution: ScoreDistribution
    savings_pct: float  # Percentage of docs not needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_k": self.final_k,
            "original_k": self.original_k,
            "reason": self.reason.value,
            "distribution": self.distribution.to_dict(),
            "savings_pct": round(self.savings_pct, 1)
        }


@dataclass
class AdaptiveTopKConfig:
    """Configuration for adaptive top-k."""
    # Base k values per complexity
    simple_base_k: int = 10
    moderate_base_k: int = 25
    complex_base_k: int = 50
    exploratory_base_k: int = 100

    # Adjustment bounds
    min_k: int = 5
    max_k: int = 200

    # Early stopping thresholds
    score_cliff_threshold: float = 0.15  # Drop of 15%+ triggers cliff
    plateau_window: int = 5  # Consecutive similar scores
    plateau_tolerance: float = 0.02  # Max score diff for plateau
    confidence_threshold: float = 0.90  # Stop if top score exceeds

    # Complexity estimation weights
    token_weight: float = 0.1
    entity_weight: float = 0.15
    question_weight: float = 0.2
    conjunction_weight: float = 0.15
    technical_weight: float = 0.2
    comparison_weight: float = 0.1
    temporal_weight: float = 0.05
    negation_weight: float = 0.05


class AdaptiveTopK:
    """
    Adaptive top-k retrieval using CAR algorithm.

    Dynamically adjusts k based on:
    1. Query complexity estimation
    2. Score distribution analysis
    3. Early stopping conditions
    """

    # Patterns for feature extraction
    QUESTION_WORDS = re.compile(
        r"\b(what|why|how|when|where|which|who|whom|whose)\b", re.IGNORECASE
    )
    CONJUNCTION_WORDS = re.compile(
        r"\b(and|or|but|also|furthermore|moreover|however|although|while)\b",
        re.IGNORECASE
    )
    COMPARISON_WORDS = re.compile(
        r"\b(vs\.?|versus|compare|compared|comparing|difference|different|"
        r"better|worse|similar|between|prefer)\b",
        re.IGNORECASE
    )
    TEMPORAL_WORDS = re.compile(
        r"\b(when|before|after|during|while|since|until|previously|later|"
        r"recently|currently|now|then|history|historical)\b",
        re.IGNORECASE
    )
    NEGATION_WORDS = re.compile(
        r"\b(not|no|never|without|neither|nor|cannot|can't|won't|doesn't|"
        r"don't|isn't|aren't|wasn't|weren't)\b",
        re.IGNORECASE
    )
    QUANTIFIER_WORDS = re.compile(
        r"\b(all|every|each|some|any|most|many|few|several|multiple|various)\b",
        re.IGNORECASE
    )
    TECHNICAL_PATTERNS = [
        r"\b[A-Z]{2,5}-\d{3,4}\b",  # Error codes: SRVO-063
        r"\$[A-Z_]+",  # Parameters: $PARAM_GROUP
        r"\b[A-Z]\d{2}[A-Z]-\d{4}-[A-Z]\d{3}\b",  # Part numbers
        r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|lb|N|V|A|Hz|rpm|°C|°F)\b",  # Units
        r"\b(?:J[1-9]|axis\s*[1-9])\b",  # Axes
    ]

    def __init__(self, config: Optional[AdaptiveTopKConfig] = None):
        """Initialize adaptive top-k."""
        self.config = config or AdaptiveTopKConfig()

        # Compile technical patterns
        self._technical_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.TECHNICAL_PATTERNS
        ]

        # Performance tracking
        self._query_history: List[Dict] = []
        self._complexity_stats: Dict[QueryComplexity, List[float]] = defaultdict(list)

        logger.info("AdaptiveTopK initialized")

    def extract_features(self, query: str) -> ComplexityFeatures:
        """Extract complexity features from query."""
        tokens = query.split()

        # Count matches for each feature type
        question_matches = len(self.QUESTION_WORDS.findall(query))
        conjunction_matches = len(self.CONJUNCTION_WORDS.findall(query))
        comparison_matches = len(self.COMPARISON_WORDS.findall(query))
        temporal_matches = len(self.TEMPORAL_WORDS.findall(query))
        negation_matches = len(self.NEGATION_WORDS.findall(query))
        quantifier_matches = len(self.QUANTIFIER_WORDS.findall(query))

        # Count technical terms
        technical_count = 0
        for pattern in self._technical_patterns:
            technical_count += len(pattern.findall(query))

        # Estimate entity count (capitalized words, patterns)
        entity_patterns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        entity_count = len(entity_patterns) + technical_count

        return ComplexityFeatures(
            token_count=len(tokens),
            entity_count=entity_count,
            question_words=question_matches,
            conjunction_count=conjunction_matches,
            technical_terms=technical_count,
            negation_count=negation_matches,
            comparison_words=comparison_matches,
            temporal_words=temporal_matches,
            quantifier_words=quantifier_matches
        )

    def estimate_complexity(
        self,
        features: ComplexityFeatures
    ) -> Tuple[QueryComplexity, float]:
        """
        Estimate query complexity from features.

        Returns:
            Tuple of (complexity level, complexity score 0-1)
        """
        cfg = self.config

        # Calculate weighted complexity score
        score = 0.0

        # Token count contribution (longer = more complex)
        if features.token_count <= 5:
            score += 0.0
        elif features.token_count <= 10:
            score += cfg.token_weight * 0.3
        elif features.token_count <= 20:
            score += cfg.token_weight * 0.6
        else:
            score += cfg.token_weight * 1.0

        # Entity count contribution
        score += min(1.0, features.entity_count / 5) * cfg.entity_weight

        # Question words (more questions = more complex)
        score += min(1.0, features.question_words / 3) * cfg.question_weight

        # Conjunctions (more conjunctions = more complex)
        score += min(1.0, features.conjunction_count / 3) * cfg.conjunction_weight

        # Technical terms (high precision needed)
        score += min(1.0, features.technical_terms / 3) * cfg.technical_weight

        # Comparison (requires multiple perspectives)
        score += min(1.0, features.comparison_words / 2) * cfg.comparison_weight

        # Temporal (requires timeline understanding)
        score += min(1.0, features.temporal_words / 2) * cfg.temporal_weight

        # Negation (requires careful reasoning)
        score += min(1.0, features.negation_count / 2) * cfg.negation_weight

        # Normalize to 0-1
        score = min(1.0, score)

        # Map to complexity level
        if score < 0.25:
            complexity = QueryComplexity.SIMPLE
        elif score < 0.50:
            complexity = QueryComplexity.MODERATE
        elif score < 0.75:
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.EXPLORATORY

        return complexity, score

    def compute_adaptive_k(
        self,
        query: str,
        initial_k: int = 50,
        context: Optional[Dict] = None
    ) -> AdaptiveKResult:
        """
        Compute adaptive k value based on query complexity.

        Args:
            query: Search query
            initial_k: Initial/base k value
            context: Optional context (conversation history, etc.)

        Returns:
            AdaptiveKResult with recommended k and reasoning
        """
        # Extract features
        features = self.extract_features(query)

        # Estimate complexity
        complexity, score = self.estimate_complexity(features)

        # Get base k for complexity level
        base_k_map = {
            QueryComplexity.SIMPLE: self.config.simple_base_k,
            QueryComplexity.MODERATE: self.config.moderate_base_k,
            QueryComplexity.COMPLEX: self.config.complex_base_k,
            QueryComplexity.EXPLORATORY: self.config.exploratory_base_k,
        }
        base_k = base_k_map[complexity]

        # Compute adjustment factor based on specific features
        adjustment = 1.0

        # Technical queries benefit from more precision
        if features.technical_terms >= 2:
            adjustment *= 0.8  # Reduce k, need precision over recall

        # Comparison queries need more diversity
        if features.comparison_words >= 1:
            adjustment *= 1.3

        # Exploratory queries with quantifiers need breadth
        if features.quantifier_words >= 1:
            adjustment *= 1.2

        # Multi-entity queries need more documents
        if features.entity_count >= 3:
            adjustment *= 1.15

        # Apply adjustment
        recommended_k = int(base_k * adjustment)

        # Clamp to bounds
        recommended_k = max(self.config.min_k, min(self.config.max_k, recommended_k))

        # Build reasoning
        reasoning_parts = [f"Complexity: {complexity.value} (score={score:.2f})"]
        if features.technical_terms >= 2:
            reasoning_parts.append("Technical terms detected, prioritizing precision")
        if features.comparison_words >= 1:
            reasoning_parts.append("Comparison query, expanding for diversity")
        if features.entity_count >= 3:
            reasoning_parts.append("Multiple entities, expanding coverage")

        reasoning = "; ".join(reasoning_parts)

        # Track for learning
        self._query_history.append({
            "query": query,
            "complexity": complexity.value,
            "score": score,
            "recommended_k": recommended_k
        })

        return AdaptiveKResult(
            recommended_k=recommended_k,
            complexity=complexity,
            features=features,
            base_k=base_k,
            adjustment_factor=adjustment,
            reasoning=reasoning
        )

    def analyze_score_distribution(
        self,
        scores: List[float]
    ) -> ScoreDistribution:
        """Analyze score distribution for stopping decisions."""
        if not scores:
            return ScoreDistribution(
                mean=0, std=0, max_score=0, min_score=0,
                score_range=0, gradient=0, plateau_detected=False,
                cliff_index=None, knee_index=None
            )

        # Basic stats
        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        # Calculate gradient (average score decline per position)
        if len(scores) > 1:
            gradient = (scores[0] - scores[-1]) / len(scores)
        else:
            gradient = 0

        # Detect plateau (consecutive similar scores)
        plateau_detected = False
        window = self.config.plateau_window
        tolerance = self.config.plateau_tolerance

        for i in range(len(scores) - window + 1):
            window_scores = scores[i:i + window]
            if max(window_scores) - min(window_scores) <= tolerance:
                plateau_detected = True
                break

        # Detect cliff (sharp score drop)
        cliff_index = None
        cliff_threshold = self.config.score_cliff_threshold

        for i in range(1, len(scores)):
            if scores[i-1] > 0:
                drop_pct = (scores[i-1] - scores[i]) / scores[i-1]
                if drop_pct >= cliff_threshold:
                    cliff_index = i
                    break

        # Find knee point (optimal cutoff using curvature)
        knee_index = self._find_knee_point(scores)

        return ScoreDistribution(
            mean=mean_score,
            std=std_score,
            max_score=max_score,
            min_score=min_score,
            score_range=score_range,
            gradient=gradient,
            plateau_detected=plateau_detected,
            cliff_index=cliff_index,
            knee_index=knee_index
        )

    def _find_knee_point(self, scores: List[float]) -> Optional[int]:
        """
        Find knee/elbow point in score curve using maximum curvature.

        This is where the curve bends most sharply, indicating
        diminishing returns.
        """
        if len(scores) < 3:
            return None

        # Normalize scores and indices
        n = len(scores)
        x = [i / (n - 1) for i in range(n)]  # 0 to 1

        if scores[0] == 0:
            return None
        y = [s / scores[0] for s in scores]  # Normalize to first score

        # Calculate curvature at each point
        max_curvature = 0
        knee_idx = None

        for i in range(1, n - 1):
            # Second derivative approximation
            d2y = y[i-1] - 2*y[i] + y[i+1]
            dx = x[1] - x[0]  # Uniform spacing

            # Curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
            dy = (y[i+1] - y[i-1]) / (2 * dx)
            curvature = abs(d2y / (dx * dx)) / ((1 + dy*dy) ** 1.5)

            if curvature > max_curvature:
                max_curvature = curvature
                knee_idx = i

        return knee_idx

    def apply_early_stopping(
        self,
        scores: List[float],
        initial_k: int,
        min_k: int = 3
    ) -> EarlyStopResult:
        """
        Apply early stopping based on score distribution.

        Args:
            scores: List of relevance scores (descending order)
            initial_k: Original k value
            min_k: Minimum k to return

        Returns:
            EarlyStopResult with final k and reasoning
        """
        if not scores:
            return EarlyStopResult(
                final_k=initial_k,
                original_k=initial_k,
                reason=StoppingReason.BUDGET_EXHAUSTED,
                distribution=self.analyze_score_distribution([]),
                savings_pct=0.0
            )

        # Analyze distribution
        distribution = self.analyze_score_distribution(scores)

        # Determine stopping point
        final_k = initial_k
        reason = StoppingReason.BUDGET_EXHAUSTED

        # Check confidence threshold
        if distribution.max_score >= self.config.confidence_threshold:
            # Very high confidence, might need fewer docs
            final_k = max(min_k, min(5, len(scores)))
            reason = StoppingReason.CONFIDENCE_THRESHOLD

        # Check for cliff (sharp drop)
        elif distribution.cliff_index is not None:
            final_k = max(min_k, distribution.cliff_index)
            reason = StoppingReason.SCORE_CLIFF

        # Check for knee point (optimal cutoff)
        elif distribution.knee_index is not None:
            final_k = max(min_k, distribution.knee_index + 1)  # +1 to include knee
            reason = StoppingReason.DIVERSITY_SATURATED

        # Check for plateau
        elif distribution.plateau_detected:
            # Stop at beginning of plateau
            for i in range(1, len(scores)):
                if abs(scores[i] - scores[i-1]) <= self.config.plateau_tolerance:
                    final_k = max(min_k, i)
                    break
            reason = StoppingReason.SCORE_PLATEAU

        # Clamp to available scores
        final_k = min(final_k, len(scores))

        # Calculate savings
        savings_pct = (1 - final_k / initial_k) * 100 if initial_k > 0 else 0

        return EarlyStopResult(
            final_k=final_k,
            original_k=initial_k,
            reason=reason,
            distribution=distribution,
            savings_pct=savings_pct
        )

    def adaptive_retrieve(
        self,
        query: str,
        scores: List[float],
        initial_k: int = 50
    ) -> Tuple[int, AdaptiveKResult, EarlyStopResult]:
        """
        Full adaptive retrieval pipeline.

        1. Compute adaptive k based on query
        2. Apply early stopping based on scores
        3. Return final k

        Args:
            query: Search query
            scores: Relevance scores from retrieval
            initial_k: Initial k value

        Returns:
            Tuple of (final_k, adaptive_result, stopping_result)
        """
        # Step 1: Query-based adaptation
        adaptive_result = self.compute_adaptive_k(query, initial_k)

        # Step 2: Score-based early stopping
        stopping_result = self.apply_early_stopping(
            scores[:adaptive_result.recommended_k],
            adaptive_result.recommended_k
        )

        return (
            stopping_result.final_k,
            adaptive_result,
            stopping_result
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive retrieval statistics."""
        if not self._query_history:
            return {
                "total_queries": 0,
                "complexity_distribution": {},
                "average_k": 0
            }

        # Aggregate complexity distribution
        complexity_counts = defaultdict(int)
        total_k = 0

        for record in self._query_history:
            complexity_counts[record["complexity"]] += 1
            total_k += record["recommended_k"]

        return {
            "total_queries": len(self._query_history),
            "complexity_distribution": dict(complexity_counts),
            "average_k": round(total_k / len(self._query_history), 1),
            "recent_queries": self._query_history[-5:]
        }

    def reset_statistics(self):
        """Reset tracking statistics."""
        self._query_history.clear()
        self._complexity_stats.clear()


# Default configurations for different use cases
PRECISION_CONFIG = AdaptiveTopKConfig(
    simple_base_k=5,
    moderate_base_k=15,
    complex_base_k=30,
    exploratory_base_k=50,
    score_cliff_threshold=0.10,  # Stricter cliff detection
    confidence_threshold=0.95,   # Higher confidence needed
)

BALANCED_ADAPTIVE_CONFIG = AdaptiveTopKConfig(
    simple_base_k=10,
    moderate_base_k=25,
    complex_base_k=50,
    exploratory_base_k=100,
)

RECALL_CONFIG = AdaptiveTopKConfig(
    simple_base_k=20,
    moderate_base_k=50,
    complex_base_k=100,
    exploratory_base_k=200,
    score_cliff_threshold=0.20,  # Allow more score drop
    confidence_threshold=0.85,   # Lower confidence acceptable
)


# Global instance
_adapter: Optional[AdaptiveTopK] = None


def get_adaptive_topk(
    config: Optional[AdaptiveTopKConfig] = None
) -> AdaptiveTopK:
    """Get or create global adaptive top-k instance."""
    global _adapter
    if _adapter is None:
        _adapter = AdaptiveTopK(config)
    return _adapter


def compute_adaptive_k(query: str, initial_k: int = 50) -> AdaptiveKResult:
    """Convenience function to compute adaptive k."""
    return get_adaptive_topk().compute_adaptive_k(query, initial_k)


def apply_early_stopping(
    scores: List[float],
    initial_k: int
) -> EarlyStopResult:
    """Convenience function to apply early stopping."""
    return get_adaptive_topk().apply_early_stopping(scores, initial_k)
