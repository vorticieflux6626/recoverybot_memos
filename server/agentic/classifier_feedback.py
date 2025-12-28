"""
Adaptive Classifier Feedback Loop

Based on Adaptive-RAG research (arXiv:2403.14403) - learns from actual outcomes
to improve query classification over time.

Key Concepts:
- Outcome Tracking: Records classification predictions vs actual search outcomes
- Feedback Integration: Uses past outcomes to refine future classifications
- Adaptive Learning: Identifies patterns where classification was suboptimal

Unlike Adaptive-RAG's trained classifier, this uses outcome-based heuristics
that can be applied to any LLM without retraining.
"""

import json
import logging
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .query_classifier import QueryCategory, RecommendedPipeline, QueryComplexity, QueryClassification

logger = logging.getLogger("agentic.classifier_feedback")


class OutcomeQuality(Enum):
    """Quality assessment of search outcome"""
    EXCELLENT = "excellent"    # Confidence >= 0.85
    GOOD = "good"             # Confidence 0.70-0.85
    ADEQUATE = "adequate"     # Confidence 0.55-0.70
    POOR = "poor"             # Confidence 0.40-0.55
    FAILED = "failed"         # Confidence < 0.40 or error


@dataclass
class ClassificationOutcome:
    """Record of a classification and its outcome"""
    query_hash: str
    query_preview: str  # First 100 chars
    timestamp: datetime

    # Classification
    predicted_category: QueryCategory
    predicted_pipeline: RecommendedPipeline
    predicted_complexity: QueryComplexity

    # Actual outcome
    actual_confidence: float
    actual_iteration_count: int
    actual_source_count: int
    execution_time_ms: int
    outcome_quality: OutcomeQuality

    # Mismatch indicators
    was_overkill: bool = False      # Used agentic when web_search would suffice
    was_underkill: bool = False     # Used web_search when agentic was needed
    missed_web_search: bool = False  # Used direct when web was needed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "query_preview": self.query_preview,
            "timestamp": self.timestamp.isoformat(),
            "predicted_category": self.predicted_category.value,
            "predicted_pipeline": self.predicted_pipeline.value,
            "predicted_complexity": self.predicted_complexity.value,
            "actual_confidence": self.actual_confidence,
            "actual_iteration_count": self.actual_iteration_count,
            "actual_source_count": self.actual_source_count,
            "execution_time_ms": self.execution_time_ms,
            "outcome_quality": self.outcome_quality.value,
            "was_overkill": self.was_overkill,
            "was_underkill": self.was_underkill,
            "missed_web_search": self.missed_web_search
        }


@dataclass
class AdaptiveHint:
    """Hint to adjust future classifications based on past outcomes"""
    pattern_type: str  # "category_override", "pipeline_boost", "complexity_adjust"
    trigger_pattern: str  # Keyword or pattern that triggers this hint
    adjustment: Dict[str, Any]  # What to adjust
    confidence: float  # How confident we are in this hint (0-1)
    support_count: int  # How many outcomes support this hint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "trigger_pattern": self.trigger_pattern,
            "adjustment": self.adjustment,
            "confidence": self.confidence,
            "support_count": self.support_count
        }


class ClassifierFeedback:
    """
    Tracks classification outcomes and provides adaptive feedback.

    Based on Adaptive-RAG principles but uses heuristic learning
    instead of a trained classifier.
    """

    # Quality thresholds
    EXCELLENT_THRESHOLD = 0.85
    GOOD_THRESHOLD = 0.70
    ADEQUATE_THRESHOLD = 0.55
    POOR_THRESHOLD = 0.40

    # Overkill detection: agentic search completed quickly with high confidence
    OVERKILL_TIME_THRESHOLD = 30000  # 30 seconds
    OVERKILL_ITERATION_THRESHOLD = 1

    # Underkill detection: many iterations needed or low confidence
    UNDERKILL_CONFIDENCE_THRESHOLD = 0.60
    UNDERKILL_ITERATION_THRESHOLD = 3

    # Minimum outcomes needed before generating hints
    MIN_OUTCOMES_FOR_HINTS = 5

    # Maximum outcomes to store per category
    MAX_OUTCOMES_PER_CATEGORY = 100

    def __init__(self):
        # Store outcomes by category for pattern analysis
        self.outcomes: Dict[str, List[ClassificationOutcome]] = defaultdict(list)

        # Learned hints for future classifications
        self.hints: List[AdaptiveHint] = []

        # Aggregated stats
        self._stats = {
            "total_outcomes": 0,
            "overkill_count": 0,
            "underkill_count": 0,
            "missed_web_count": 0,
            "excellent_outcomes": 0,
            "poor_outcomes": 0,
            "hints_generated": 0,
            "last_outcome": None,
            "last_hint_generation": None
        }

    def record_outcome(
        self,
        query: str,
        classification: QueryClassification,
        confidence: float,
        iteration_count: int,
        source_count: int,
        execution_time_ms: int,
        was_successful: bool = True
    ) -> ClassificationOutcome:
        """
        Record the outcome of a classification decision.

        Args:
            query: The original query
            classification: The classification result
            confidence: Final confidence score (0-1)
            iteration_count: Number of search iterations needed
            source_count: Number of sources consulted
            execution_time_ms: Total execution time
            was_successful: Whether the search completed without error

        Returns:
            ClassificationOutcome record
        """
        # Determine outcome quality
        if not was_successful:
            quality = OutcomeQuality.FAILED
        elif confidence >= self.EXCELLENT_THRESHOLD:
            quality = OutcomeQuality.EXCELLENT
            self._stats["excellent_outcomes"] += 1
        elif confidence >= self.GOOD_THRESHOLD:
            quality = OutcomeQuality.GOOD
        elif confidence >= self.ADEQUATE_THRESHOLD:
            quality = OutcomeQuality.ADEQUATE
        elif confidence >= self.POOR_THRESHOLD:
            quality = OutcomeQuality.POOR
            self._stats["poor_outcomes"] += 1
        else:
            quality = OutcomeQuality.FAILED
            self._stats["poor_outcomes"] += 1

        # Detect classification mismatches
        was_overkill = self._detect_overkill(
            classification.recommended_pipeline,
            confidence,
            iteration_count,
            execution_time_ms
        )

        was_underkill = self._detect_underkill(
            classification.recommended_pipeline,
            confidence,
            iteration_count
        )

        missed_web_search = self._detect_missed_web(
            classification.recommended_pipeline,
            confidence,
            source_count
        )

        # Update stats
        if was_overkill:
            self._stats["overkill_count"] += 1
        if was_underkill:
            self._stats["underkill_count"] += 1
        if missed_web_search:
            self._stats["missed_web_count"] += 1

        # Create outcome record
        outcome = ClassificationOutcome(
            query_hash=hashlib.md5(query.encode()).hexdigest()[:12],
            query_preview=query[:100],
            timestamp=datetime.now(timezone.utc),
            predicted_category=classification.category,
            predicted_pipeline=classification.recommended_pipeline,
            predicted_complexity=classification.complexity,
            actual_confidence=confidence,
            actual_iteration_count=iteration_count,
            actual_source_count=source_count,
            execution_time_ms=execution_time_ms,
            outcome_quality=quality,
            was_overkill=was_overkill,
            was_underkill=was_underkill,
            missed_web_search=missed_web_search
        )

        # Store by category
        category = classification.category.value
        self.outcomes[category].append(outcome)

        # Trim if too many
        if len(self.outcomes[category]) > self.MAX_OUTCOMES_PER_CATEGORY:
            self.outcomes[category] = self.outcomes[category][-self.MAX_OUTCOMES_PER_CATEGORY:]

        self._stats["total_outcomes"] += 1
        self._stats["last_outcome"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Recorded outcome: category={category}, quality={quality.value}, "
            f"overkill={was_overkill}, underkill={was_underkill}"
        )

        # Check if we should generate new hints
        if self._stats["total_outcomes"] % 10 == 0:  # Every 10 outcomes
            self._generate_hints()

        return outcome

    def _detect_overkill(
        self,
        pipeline: RecommendedPipeline,
        confidence: float,
        iteration_count: int,
        execution_time_ms: int
    ) -> bool:
        """Detect if agentic search was overkill for the query"""
        if pipeline != RecommendedPipeline.AGENTIC_SEARCH:
            return False

        # High confidence achieved quickly with few iterations = overkill
        if (confidence >= self.GOOD_THRESHOLD and
            iteration_count <= self.OVERKILL_ITERATION_THRESHOLD and
            execution_time_ms <= self.OVERKILL_TIME_THRESHOLD):
            return True

        return False

    def _detect_underkill(
        self,
        pipeline: RecommendedPipeline,
        confidence: float,
        iteration_count: int
    ) -> bool:
        """Detect if simpler pipeline was insufficient"""
        if pipeline == RecommendedPipeline.AGENTIC_SEARCH:
            return False

        # Low confidence or many iterations needed = underkill
        if (confidence < self.UNDERKILL_CONFIDENCE_THRESHOLD and
            (pipeline == RecommendedPipeline.WEB_SEARCH or
             pipeline == RecommendedPipeline.DIRECT_ANSWER)):
            return True

        if iteration_count >= self.UNDERKILL_ITERATION_THRESHOLD:
            return True

        return False

    def _detect_missed_web(
        self,
        pipeline: RecommendedPipeline,
        confidence: float,
        source_count: int
    ) -> bool:
        """Detect if direct answer should have used web search"""
        if pipeline != RecommendedPipeline.DIRECT_ANSWER:
            return False

        # Low confidence with no sources = probably needed web search
        if confidence < self.ADEQUATE_THRESHOLD and source_count == 0:
            return True

        return False

    def _generate_hints(self) -> List[AdaptiveHint]:
        """Generate adaptive hints from outcome patterns"""
        new_hints = []

        # Analyze each category
        for category, outcomes in self.outcomes.items():
            if len(outcomes) < self.MIN_OUTCOMES_FOR_HINTS:
                continue

            # Check for consistent overkill pattern
            overkill_outcomes = [o for o in outcomes if o.was_overkill]
            if len(overkill_outcomes) >= 3:
                # Extract common words from overkill queries
                common_words = self._find_common_patterns(
                    [o.query_preview for o in overkill_outcomes]
                )
                if common_words:
                    hint = AdaptiveHint(
                        pattern_type="pipeline_downgrade",
                        trigger_pattern=common_words[0],
                        adjustment={"prefer_pipeline": "web_search"},
                        confidence=len(overkill_outcomes) / len(outcomes),
                        support_count=len(overkill_outcomes)
                    )
                    new_hints.append(hint)

            # Check for consistent underkill pattern
            underkill_outcomes = [o for o in outcomes if o.was_underkill]
            if len(underkill_outcomes) >= 3:
                common_words = self._find_common_patterns(
                    [o.query_preview for o in underkill_outcomes]
                )
                if common_words:
                    hint = AdaptiveHint(
                        pattern_type="pipeline_upgrade",
                        trigger_pattern=common_words[0],
                        adjustment={"prefer_pipeline": "agentic_search"},
                        confidence=len(underkill_outcomes) / len(outcomes),
                        support_count=len(underkill_outcomes)
                    )
                    new_hints.append(hint)

            # Check for consistent complexity underestimate
            low_confidence = [o for o in outcomes
                            if o.actual_confidence < self.ADEQUATE_THRESHOLD
                            and o.predicted_complexity != QueryComplexity.COMPLEX]
            if len(low_confidence) >= 3:
                hint = AdaptiveHint(
                    pattern_type="complexity_upgrade",
                    trigger_pattern=category,
                    adjustment={"min_complexity": "complex"},
                    confidence=len(low_confidence) / len(outcomes),
                    support_count=len(low_confidence)
                )
                new_hints.append(hint)

        # Merge with existing hints (update counts)
        for new_hint in new_hints:
            existing = next(
                (h for h in self.hints
                 if h.pattern_type == new_hint.pattern_type
                 and h.trigger_pattern == new_hint.trigger_pattern),
                None
            )
            if existing:
                # Update existing hint
                existing.support_count = max(existing.support_count, new_hint.support_count)
                existing.confidence = (existing.confidence + new_hint.confidence) / 2
            else:
                self.hints.append(new_hint)
                self._stats["hints_generated"] += 1

        self._stats["last_hint_generation"] = datetime.now(timezone.utc).isoformat()

        if new_hints:
            logger.info(f"Generated {len(new_hints)} adaptive hints")

        return new_hints

    def _find_common_patterns(self, texts: List[str]) -> List[str]:
        """Find common words/patterns in a list of texts"""
        if not texts:
            return []

        # Simple word frequency analysis
        word_counts: Dict[str, int] = defaultdict(int)
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "when", "where", "why", "who"}

        for text in texts:
            words = set(text.lower().split())
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_counts[word] += 1

        # Return words that appear in majority of texts
        threshold = len(texts) / 2
        common = [word for word, count in word_counts.items() if count >= threshold]

        # Sort by frequency
        common.sort(key=lambda w: -word_counts[w])

        return common[:5]

    def get_adjustment_for_query(
        self,
        query: str,
        classification: QueryClassification
    ) -> Dict[str, Any]:
        """
        Get any adjustments that should be applied to a classification
        based on learned patterns.

        Args:
            query: The query being classified
            classification: The initial classification

        Returns:
            Dictionary of suggested adjustments
        """
        adjustments = {}
        query_lower = query.lower()

        for hint in self.hints:
            if hint.confidence < 0.5:
                continue  # Skip low-confidence hints

            if hint.trigger_pattern in query_lower:
                if hint.pattern_type == "pipeline_upgrade":
                    adjustments["recommended_pipeline_hint"] = hint.adjustment.get("prefer_pipeline")
                    adjustments["upgrade_reason"] = f"Pattern '{hint.trigger_pattern}' often needs agentic search"
                elif hint.pattern_type == "pipeline_downgrade":
                    adjustments["recommended_pipeline_hint"] = hint.adjustment.get("prefer_pipeline")
                    adjustments["downgrade_reason"] = f"Pattern '{hint.trigger_pattern}' often works with web search"
                elif hint.pattern_type == "complexity_upgrade":
                    adjustments["complexity_hint"] = hint.adjustment.get("min_complexity")

        if adjustments:
            logger.debug(f"Applied adjustments for query: {adjustments}")

        return adjustments

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        category_stats = {}
        for category, outcomes in self.outcomes.items():
            if outcomes:
                avg_confidence = sum(o.actual_confidence for o in outcomes) / len(outcomes)
                excellent_rate = sum(1 for o in outcomes if o.outcome_quality == OutcomeQuality.EXCELLENT) / len(outcomes)
                category_stats[category] = {
                    "count": len(outcomes),
                    "avg_confidence": round(avg_confidence, 3),
                    "excellent_rate": round(excellent_rate, 3)
                }

        return {
            **self._stats,
            "category_stats": category_stats,
            "hint_count": len(self.hints),
            "hints": [h.to_dict() for h in self.hints[:10]]  # Top 10 hints
        }

    def get_outcomes(
        self,
        category: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recorded outcomes"""
        if category:
            outcomes = self.outcomes.get(category, [])[-limit:]
        else:
            # Flatten all categories
            all_outcomes = []
            for cat_outcomes in self.outcomes.values():
                all_outcomes.extend(cat_outcomes)
            # Sort by timestamp descending
            all_outcomes.sort(key=lambda o: o.timestamp, reverse=True)
            outcomes = all_outcomes[:limit]

        return [o.to_dict() for o in outcomes]

    def clear_outcomes(self, category: Optional[str] = None) -> int:
        """Clear recorded outcomes"""
        if category:
            count = len(self.outcomes.get(category, []))
            self.outcomes[category] = []
            return count

        count = sum(len(o) for o in self.outcomes.values())
        self.outcomes = defaultdict(list)
        return count


# Factory function
def create_classifier_feedback() -> ClassifierFeedback:
    """Create a new ClassifierFeedback instance"""
    return ClassifierFeedback()


# Singleton instance
_classifier_feedback: Optional[ClassifierFeedback] = None


def get_classifier_feedback() -> ClassifierFeedback:
    """Get or create singleton ClassifierFeedback"""
    global _classifier_feedback
    if _classifier_feedback is None:
        _classifier_feedback = create_classifier_feedback()
    return _classifier_feedback
