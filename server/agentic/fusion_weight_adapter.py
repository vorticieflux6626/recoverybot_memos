"""
Query Intent Classifier for Fusion Weight Adaptation.

Part of G.2.3: Query-adaptive fusion weights for hybrid retrieval.

Classifies query intent and returns optimal sparse/dense fusion weights
for the BGE-M3 hybrid retriever.

Research Basis:
- Query-aware hybrid retrieval (various 2023-2025 papers)
- BM25 vs Dense: Lexical queries need more BM25, semantic queries need more dense
- Error code queries: High lexical precision required
- Troubleshooting queries: Semantic understanding required

Usage:
    from agentic.fusion_weight_adapter import FusionWeightAdapter

    adapter = FusionWeightAdapter()
    weights = adapter.get_fusion_weights("SRVO-063 alarm meaning")
    # Returns: FusionWeights(sparse=0.7, dense=0.3)

    weights = adapter.get_fusion_weights("robot arm jerking during movement")
    # Returns: FusionWeights(sparse=0.3, dense=0.7)
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("agentic.fusion_weight_adapter")


class QueryIntent(str, Enum):
    """Query intent categories for fusion weight selection."""
    ERROR_CODE = "error_code"           # Specific error code lookup
    TROUBLESHOOTING = "troubleshooting" # Symptom-based problem solving
    PROCEDURE = "procedure"             # How-to, step-by-step
    CONCEPTUAL = "conceptual"           # Understanding concepts
    COMPARISON = "comparison"           # Comparing options
    PARAMETER = "parameter"             # Parameter/setting queries
    PART_NUMBER = "part_number"         # Part number lookup
    GENERAL = "general"                 # General queries


@dataclass
class FusionWeights:
    """Fusion weights for hybrid retrieval."""
    sparse: float  # BM25 weight (lexical matching)
    dense: float   # Dense embedding weight (semantic matching)
    colbert: float = 0.0  # ColBERT late interaction weight (optional)

    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total = self.sparse + self.dense + self.colbert
        if total > 0:
            self.sparse /= total
            self.dense /= total
            self.colbert /= total

    def to_dict(self) -> Dict[str, float]:
        return {
            "sparse": round(self.sparse, 3),
            "dense": round(self.dense, 3),
            "colbert": round(self.colbert, 3)
        }


@dataclass
class IntentClassification:
    """Result of intent classification for fusion weights."""
    intent: QueryIntent
    confidence: float
    weights: FusionWeights
    matched_patterns: List[str]
    reasoning: str


# Pattern definitions for each intent
INTENT_PATTERNS = {
    QueryIntent.ERROR_CODE: [
        # FANUC patterns
        r"\b[A-Z]{2,5}-\d{3,4}\b",  # SRVO-063, MOTN-023, HOST-001
        r"\balarm\s*(?:code|number|#)?\s*\d+\b",
        r"\berror\s*(?:code|number|#)?\s*\d+\b",
        r"\bfault\s*(?:code|number|#)?\s*\d+\b",
        # Generic patterns
        r"\b(?:E|ERR|ERROR|ALM|ALARM)\d{2,5}\b",
        r"\bcode\s+\d+\b",
    ],
    QueryIntent.PART_NUMBER: [
        r"\b[A-Z]\d{2}[A-Z]-\d{4}-[A-Z]\d{3}\b",  # FANUC part: A06B-0001-B100
        r"\bpart\s*(?:number|#|no\.?)\s*\w+\b",
        r"\b\d{6,12}\b",  # Long numeric part numbers
        r"\b[A-Z]{2,3}\d{5,8}\b",  # Alphanumeric part codes
    ],
    QueryIntent.PARAMETER: [
        r"\$[A-Z_]+",  # FANUC parameters: $PARAM_GROUP
        r"\bparameter\s+\d+\b",
        r"\bsetting\s+\d+\b",
        r"\bconfig(?:uration)?\s+\w+\b",
        r"\b(?:set|adjust|change|modify)\s+(?:parameter|setting)\b",
    ],
    QueryIntent.PROCEDURE: [
        r"\bhow\s+(?:to|do|can)\b",
        r"\bstep(?:s|\s*-?\s*by\s*-?\s*step)?\b",
        r"\bprocedure\b",
        r"\bprocess\b",
        r"\binstruction(?:s)?\b",
        r"\bguide\b",
        r"\btutorial\b",
        r"\bmastering\b",
        r"\bcalibrat(?:e|ion)\b",
        r"\bsetup\b",
        r"\binstall(?:ation)?\b",
    ],
    QueryIntent.TROUBLESHOOTING: [
        r"\b(?:not\s+)?(?:work(?:ing)?|function(?:ing)?)\b",
        r"\bproblem\b",
        r"\bissue\b",
        r"\bfail(?:ing|ure|ed)?\b",
        r"\bwrong\b",
        r"\bbroken\b",
        r"\bstuck\b",
        r"\bjerk(?:ing|y)?\b",
        r"\bvibrat(?:e|ing|ion)\b",
        r"\bnoise\b",
        r"\boverheat(?:ing)?\b",
        r"\bslow\b",
        r"\berratic\b",
        r"\bintermittent\b",
        r"\bunexpected\b",
        r"\bsymptom\b",
        r"\bcause\b",
        r"\bfix\b",
        r"\bresolve\b",
        r"\bsolve\b",
    ],
    QueryIntent.COMPARISON: [
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompare\b",
        r"\bdifference\s+between\b",
        r"\bwhich\s+(?:is|one)\b",
        r"\bbetter\b",
        r"\badvantage\b",
        r"\bdisadvantage\b",
        r"\bpros?\s+(?:and|&)\s+cons?\b",
    ],
    QueryIntent.CONCEPTUAL: [
        r"\bwhat\s+is\b",
        r"\bwhat\s+are\b",
        r"\bdefin(?:e|ition)\b",
        r"\bexplain\b",
        r"\bunderstand\b",
        r"\bmeaning\b",
        r"\bpurpose\b",
        r"\breason\b",
        r"\bwhy\b",
        r"\bhow\s+does\b",
        r"\bconcept\b",
        r"\btheory\b",
        r"\bprinciple\b",
    ],
}

# Default fusion weights per intent
# Higher sparse = more lexical/exact matching
# Higher dense = more semantic understanding
INTENT_WEIGHTS = {
    QueryIntent.ERROR_CODE: FusionWeights(sparse=0.7, dense=0.2, colbert=0.1),
    QueryIntent.PART_NUMBER: FusionWeights(sparse=0.8, dense=0.15, colbert=0.05),
    QueryIntent.PARAMETER: FusionWeights(sparse=0.65, dense=0.25, colbert=0.1),
    QueryIntent.PROCEDURE: FusionWeights(sparse=0.4, dense=0.5, colbert=0.1),
    QueryIntent.TROUBLESHOOTING: FusionWeights(sparse=0.3, dense=0.6, colbert=0.1),
    QueryIntent.COMPARISON: FusionWeights(sparse=0.35, dense=0.55, colbert=0.1),
    QueryIntent.CONCEPTUAL: FusionWeights(sparse=0.25, dense=0.65, colbert=0.1),
    QueryIntent.GENERAL: FusionWeights(sparse=0.4, dense=0.5, colbert=0.1),
}


class FusionWeightAdapter:
    """
    Adapts fusion weights based on query intent.

    Uses pattern matching and heuristics to classify query intent
    and return optimal sparse/dense fusion weights.
    """

    def __init__(
        self,
        custom_weights: Optional[Dict[QueryIntent, FusionWeights]] = None
    ):
        """
        Initialize adapter.

        Args:
            custom_weights: Optional custom weights per intent
        """
        self.weights = INTENT_WEIGHTS.copy()
        if custom_weights:
            self.weights.update(custom_weights)

        # Compile patterns for efficiency
        self._compiled_patterns: Dict[QueryIntent, List[re.Pattern]] = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }

        # Statistics
        self._total_classifications = 0
        self._intent_counts: Dict[QueryIntent, int] = {i: 0 for i in QueryIntent}

        logger.info("FusionWeightAdapter initialized")

    def classify_intent(self, query: str) -> IntentClassification:
        """
        Classify query intent and return fusion weights.

        Args:
            query: User query string

        Returns:
            IntentClassification with intent, confidence, and weights
        """
        self._total_classifications += 1

        # Score each intent based on pattern matches
        intent_scores: Dict[QueryIntent, Tuple[float, List[str]]] = {}

        for intent, patterns in self._compiled_patterns.items():
            matches = []
            score = 0.0

            for pattern in patterns:
                found = pattern.findall(query)
                if found:
                    matches.extend(found)
                    # Weight different patterns differently
                    if intent == QueryIntent.ERROR_CODE:
                        # Error codes are very specific, high confidence
                        score += 2.0 * len(found)
                    elif intent == QueryIntent.PART_NUMBER:
                        score += 1.8 * len(found)
                    elif intent == QueryIntent.PARAMETER:
                        score += 1.5 * len(found)
                    else:
                        score += 1.0 * len(found)

            if matches:
                intent_scores[intent] = (score, matches)

        # Determine winning intent
        if not intent_scores:
            # No patterns matched, use general
            intent = QueryIntent.GENERAL
            confidence = 0.5
            matched_patterns = []
            reasoning = "No specific patterns matched, using general weights"
        else:
            # Find highest scoring intent
            sorted_intents = sorted(
                intent_scores.items(),
                key=lambda x: x[1][0],
                reverse=True
            )
            intent, (score, matched_patterns) = sorted_intents[0]

            # Calculate confidence based on score and competition
            if len(sorted_intents) == 1:
                confidence = min(0.95, 0.6 + score * 0.1)
            else:
                # Reduce confidence if multiple intents scored
                second_score = sorted_intents[1][1][0]
                confidence = min(0.9, 0.5 + (score - second_score) * 0.15 + score * 0.05)

            reasoning = f"Matched {len(matched_patterns)} pattern(s) for {intent.value}"

        # Get weights for the intent
        weights = self.weights[intent]

        # Update statistics
        self._intent_counts[intent] += 1

        return IntentClassification(
            intent=intent,
            confidence=confidence,
            weights=weights,
            matched_patterns=matched_patterns[:5],  # Limit to 5 examples
            reasoning=reasoning
        )

    def get_fusion_weights(self, query: str) -> FusionWeights:
        """
        Get fusion weights for a query (convenience method).

        Args:
            query: User query string

        Returns:
            FusionWeights with sparse/dense/colbert weights
        """
        classification = self.classify_intent(query)
        return classification.weights

    def adapt_weights(
        self,
        base_weights: FusionWeights,
        query: str,
        blend_factor: float = 0.5
    ) -> FusionWeights:
        """
        Blend base weights with query-adaptive weights.

        Args:
            base_weights: Default/baseline weights
            query: User query for adaptation
            blend_factor: How much to weight query-specific (0=all base, 1=all query)

        Returns:
            Blended FusionWeights
        """
        query_weights = self.get_fusion_weights(query)

        return FusionWeights(
            sparse=base_weights.sparse * (1 - blend_factor) + query_weights.sparse * blend_factor,
            dense=base_weights.dense * (1 - blend_factor) + query_weights.dense * blend_factor,
            colbert=base_weights.colbert * (1 - blend_factor) + query_weights.colbert * blend_factor
        )

    def get_weights_for_search_type(
        self,
        query: str,
        search_type: str = "hybrid"
    ) -> Dict[str, float]:
        """
        Get weights formatted for specific search implementations.

        Args:
            query: User query
            search_type: "hybrid", "bge_m3", "rrf"

        Returns:
            Dict with appropriate weight keys for the search type
        """
        weights = self.get_fusion_weights(query)

        if search_type == "bge_m3":
            return {
                "dense_weight": weights.dense,
                "sparse_weight": weights.sparse,
                "colbert_weight": weights.colbert
            }
        elif search_type == "rrf":
            return {
                "bm25_weight": weights.sparse,
                "semantic_weight": weights.dense
            }
        else:  # hybrid (default)
            return {
                "sparse": weights.sparse,
                "dense": weights.dense
            }

    def get_statistics(self) -> Dict[str, any]:
        """Get classification statistics."""
        total = self._total_classifications or 1
        return {
            "total_classifications": self._total_classifications,
            "intent_distribution": {
                intent.value: {
                    "count": count,
                    "percentage": round(count / total * 100, 1)
                }
                for intent, count in self._intent_counts.items()
            },
            "weights_config": {
                intent.value: weights.to_dict()
                for intent, weights in self.weights.items()
            }
        }

    def reset_statistics(self):
        """Reset classification statistics."""
        self._total_classifications = 0
        self._intent_counts = {i: 0 for i in QueryIntent}


# Global instance
_adapter: Optional[FusionWeightAdapter] = None


def get_fusion_weight_adapter() -> FusionWeightAdapter:
    """Get or create global adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = FusionWeightAdapter()
    return _adapter


def get_adaptive_weights(query: str) -> FusionWeights:
    """Convenience function to get adaptive weights."""
    return get_fusion_weight_adapter().get_fusion_weights(query)


def classify_for_fusion(query: str) -> IntentClassification:
    """Convenience function to classify query for fusion."""
    return get_fusion_weight_adapter().classify_intent(query)
