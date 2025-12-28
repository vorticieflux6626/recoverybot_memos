"""
Sufficient Context Classifier

Based on Google's ICLR 2025 research "Sufficient Context: A New Lens on RAG Systems"
(arXiv:2411.06037). Achieves 93% accuracy using chain-of-thought prompting.

Key insight: Context relevance alone is wrong to measure - what matters is whether
the context provides enough information for the LLM to answer correctly.

Benefits:
- Prevents hallucination by detecting insufficient context before synthesis
- Enables selective generation (abstain when context is insufficient)
- Reduces "lost in the middle" effects by filtering low-value content
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class ContextSufficiency(Enum):
    """Classification of context sufficiency"""
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous"


@dataclass
class SufficiencyResult:
    """Result of sufficient context classification"""
    is_sufficient: bool
    confidence: float  # 0-1, how confident the classifier is
    reasoning: str  # Chain-of-thought explanation
    missing_information: List[str] = field(default_factory=list)  # What's missing
    key_facts_found: List[str] = field(default_factory=list)  # What was found
    recommendation: str = ""  # What to do next

    @property
    def sufficiency_level(self) -> ContextSufficiency:
        if self.confidence >= 0.8:
            return ContextSufficiency.SUFFICIENT if self.is_sufficient else ContextSufficiency.INSUFFICIENT
        return ContextSufficiency.AMBIGUOUS


@dataclass
class PositionalAnalysis:
    """Analysis of content position effectiveness"""
    beginning_relevance: float  # 0-1
    middle_relevance: float  # 0-1
    end_relevance: float  # 0-1
    optimal_reorder: List[int]  # Suggested reordering of source indices
    lost_in_middle_risk: str  # low/medium/high


# Chain-of-thought prompt based on Google's research
SUFFICIENT_CONTEXT_PROMPT = '''You are an expert evaluator determining if retrieved context is SUFFICIENT to answer a question.

TASK: Determine if the provided context contains ALL necessary information to craft a definitive, accurate answer.

CRITERIA for SUFFICIENT context:
- Contains all facts needed to answer the question completely
- Provides enough detail to avoid speculation or assumptions
- Includes relevant dates, numbers, names, or specifics when required
- Multi-hop reasoning from the context alone can reach the answer

CRITERIA for INSUFFICIENT context:
- Missing key information required by the question
- Contains only tangentially related information
- Requires external knowledge not in the context
- Has contradictory information without resolution

QUESTION: {question}

CONTEXT:
{context}

Think step by step:
1. What specific information does the question require?
2. What information is present in the context?
3. What information is missing or incomplete?
4. Can a definitive answer be crafted using ONLY this context?

Respond in JSON format:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "step-by-step explanation",
    "missing_information": ["list of missing info"] or [],
    "key_facts_found": ["list of key facts present"],
    "recommendation": "proceed/gather_more/refine_query"
}}'''

# One-shot example for better accuracy (from Google's methodology)
ONE_SHOT_EXAMPLE = '''EXAMPLE:

QUESTION: When was the Eiffel Tower built and who designed it?

CONTEXT:
The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.

ANALYSIS:
{{
    "is_sufficient": false,
    "confidence": 0.95,
    "reasoning": "The context provides when the tower was built (1887-1889) but does NOT mention the designer (Gustave Eiffel). The question asks for BOTH pieces of information, but only one is available.",
    "missing_information": ["designer/architect of the Eiffel Tower"],
    "key_facts_found": ["construction period: 1887-1889", "location: Paris, France", "purpose: 1889 World's Fair centerpiece"],
    "recommendation": "gather_more"
}}

---

Now evaluate the actual question and context:

'''


class SufficientContextClassifier:
    """
    Classifies whether retrieved context is sufficient to answer a query.

    Based on Google's "Sufficient Context" research achieving 93% accuracy.
    Uses chain-of-thought prompting with 1-shot example.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",  # Good reasoning capability
        timeout: float = 60.0
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._stats = {
            "total_classifications": 0,
            "sufficient_count": 0,
            "insufficient_count": 0,
            "ambiguous_count": 0,
            "avg_confidence": 0.0,
            "avg_latency_ms": 0.0
        }

    async def classify(
        self,
        question: str,
        context: str,
        use_one_shot: bool = True
    ) -> SufficiencyResult:
        """
        Classify if context is sufficient to answer the question.

        Args:
            question: The user's query
            context: The retrieved context (concatenated sources)
            use_one_shot: Whether to include the one-shot example

        Returns:
            SufficiencyResult with classification and reasoning
        """
        import time
        start_time = time.time()

        # Build prompt
        prompt = ""
        if use_one_shot:
            prompt = ONE_SHOT_EXAMPLE

        prompt += SUFFICIENT_CONTEXT_PROMPT.format(
            question=question,
            context=context[:50000]  # Limit context to ~12.5K tokens
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Low temp for classification
                            "num_predict": 1024,
                            "num_ctx": 32768
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()

            raw_response = result.get("response", "")
            parsed = self._parse_response(raw_response)

            # Update stats
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(parsed, latency_ms)

            return parsed

        except Exception as e:
            logger.error(f"Sufficient context classification failed: {e}")
            # Return conservative default (insufficient)
            return SufficiencyResult(
                is_sufficient=False,
                confidence=0.5,
                reasoning=f"Classification failed: {str(e)}",
                missing_information=["Unable to determine"],
                recommendation="gather_more"
            )

    def _parse_response(self, response: str) -> SufficiencyResult:
        """Parse LLM response into SufficiencyResult"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*"is_sufficient"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                # Try to find any JSON block
                json_match = re.search(r'\{[\s\S]*?\}', response)

            if json_match:
                data = json.loads(json_match.group())
                return SufficiencyResult(
                    is_sufficient=data.get("is_sufficient", False),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", ""),
                    missing_information=data.get("missing_information", []),
                    key_facts_found=data.get("key_facts_found", []),
                    recommendation=data.get("recommendation", "gather_more")
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse classifier response: {e}")

        # Fallback: try to infer from text
        is_sufficient = "sufficient" in response.lower() and "insufficient" not in response.lower()
        return SufficiencyResult(
            is_sufficient=is_sufficient,
            confidence=0.6,
            reasoning=response[:500],
            recommendation="proceed" if is_sufficient else "gather_more"
        )

    def _update_stats(self, result: SufficiencyResult, latency_ms: float):
        """Update running statistics"""
        self._stats["total_classifications"] += 1

        if result.sufficiency_level == ContextSufficiency.SUFFICIENT:
            self._stats["sufficient_count"] += 1
        elif result.sufficiency_level == ContextSufficiency.INSUFFICIENT:
            self._stats["insufficient_count"] += 1
        else:
            self._stats["ambiguous_count"] += 1

        # Running average
        n = self._stats["total_classifications"]
        self._stats["avg_confidence"] = (
            (self._stats["avg_confidence"] * (n - 1) + result.confidence) / n
        )
        self._stats["avg_latency_ms"] = (
            (self._stats["avg_latency_ms"] * (n - 1) + latency_ms) / n
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Get classifier statistics"""
        return self._stats.copy()


class PositionalOptimizer:
    """
    Optimizes content positioning to mitigate "lost in the middle" effect.

    Research shows LLMs focus on beginning and end of context, neglecting middle.
    This class reorders sources to place most relevant content at optimal positions.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b"  # Fast model for relevance scoring
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model

    async def score_relevance(
        self,
        question: str,
        sources: List[Dict[str, str]]
    ) -> List[Tuple[int, float]]:
        """
        Score each source's relevance to the question.

        Returns list of (source_index, relevance_score) tuples.
        """
        scores = []

        for i, source in enumerate(sources):
            content = source.get("content", source.get("snippet", ""))[:2000]
            title = source.get("title", "")

            # Quick relevance scoring
            prompt = f'''Rate the relevance of this source to the question on a scale of 0-10.

Question: {question}

Source Title: {title}
Source Content: {content[:1000]}

Respond with ONLY a number from 0-10.'''

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.1, "num_predict": 10}
                        }
                    )
                    result = response.json()
                    score_text = result.get("response", "5").strip()
                    # Extract number
                    score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    score = float(score_match.group(1)) / 10.0 if score_match else 0.5
                    scores.append((i, min(1.0, max(0.0, score))))
            except Exception as e:
                logger.warning(f"Failed to score source {i}: {e}")
                scores.append((i, 0.5))  # Default score

        return scores

    def reorder_for_optimal_attention(
        self,
        sources: List[Dict[str, str]],
        relevance_scores: List[Tuple[int, float]]
    ) -> Tuple[List[Dict[str, str]], PositionalAnalysis]:
        """
        Reorder sources to place most relevant at beginning and end.

        Research-backed positioning strategy:
        - Highest relevance sources at BEGINNING (primary attention)
        - Second-highest at END (recency effect)
        - Lower relevance in MIDDLE (least attention)
        """
        # Sort by relevance (descending)
        sorted_scores = sorted(relevance_scores, key=lambda x: x[1], reverse=True)

        n = len(sorted_scores)
        if n <= 2:
            # Too few sources to reorder meaningfully
            reordered_indices = [s[0] for s in sorted_scores]
            return (
                [sources[i] for i in reordered_indices],
                PositionalAnalysis(
                    beginning_relevance=sorted_scores[0][1] if n > 0 else 0,
                    middle_relevance=0,
                    end_relevance=sorted_scores[-1][1] if n > 1 else 0,
                    optimal_reorder=reordered_indices,
                    lost_in_middle_risk="low"
                )
            )

        # Interleave: best at start, second-best at end, rest in middle
        # Pattern: [1st, 3rd, 5th, ..., 6th, 4th, 2nd]
        beginning = []
        end = []

        for i, (idx, score) in enumerate(sorted_scores):
            if i % 2 == 0:
                beginning.append((idx, score))
            else:
                end.insert(0, (idx, score))  # Prepend to reverse order

        reordered = beginning + end
        reordered_indices = [r[0] for r in reordered]
        reordered_sources = [sources[i] for i in reordered_indices]

        # Calculate positional relevance
        third = n // 3
        begin_scores = [r[1] for r in reordered[:third]] if third > 0 else [reordered[0][1]]
        middle_scores = [r[1] for r in reordered[third:2*third]] if third > 0 else []
        end_scores = [r[1] for r in reordered[2*third:]] if third > 0 else [reordered[-1][1]]

        begin_avg = sum(begin_scores) / len(begin_scores) if begin_scores else 0
        middle_avg = sum(middle_scores) / len(middle_scores) if middle_scores else 0
        end_avg = sum(end_scores) / len(end_scores) if end_scores else 0

        # Assess lost-in-middle risk
        if middle_avg > 0.7 and (begin_avg < 0.6 or end_avg < 0.6):
            risk = "high"
        elif middle_avg > 0.5:
            risk = "medium"
        else:
            risk = "low"

        return (
            reordered_sources,
            PositionalAnalysis(
                beginning_relevance=begin_avg,
                middle_relevance=middle_avg,
                end_relevance=end_avg,
                optimal_reorder=reordered_indices,
                lost_in_middle_risk=risk
            )
        )


class DynamicContextAllocator:
    """
    Dynamically allocates context budget based on confidence and quality signals.

    Key insight: Use less of the context budget when confidence is already high.
    This prevents the performance degradation seen with excessive context.
    """

    def __init__(
        self,
        max_context_chars: int = 400000,
        min_context_chars: int = 50000,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.60
    ):
        self.max_context_chars = max_context_chars
        self.min_context_chars = min_context_chars
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold

    def calculate_budget(
        self,
        current_confidence: float,
        sufficiency_result: Optional[SufficiencyResult] = None,
        iteration: int = 0,
        source_count: int = 0
    ) -> Dict[str, int]:
        """
        Calculate dynamic context budget based on current state.

        Returns dict with:
        - max_total_content: Total chars to allocate
        - max_per_source: Chars per source
        - max_sources: Number of sources to include
        """
        # Base allocation
        if current_confidence >= self.high_confidence_threshold:
            # High confidence: use minimal context to avoid degradation
            allocation_ratio = 0.3
        elif current_confidence >= self.low_confidence_threshold:
            # Medium confidence: moderate context
            allocation_ratio = 0.6
        else:
            # Low confidence: use more context to find relevant info
            allocation_ratio = 1.0

        # Adjust based on sufficiency
        if sufficiency_result:
            if sufficiency_result.is_sufficient and sufficiency_result.confidence > 0.8:
                # Already sufficient: reduce further
                allocation_ratio *= 0.7
            elif not sufficiency_result.is_sufficient:
                # Insufficient: increase to find missing info
                allocation_ratio = min(1.0, allocation_ratio * 1.3)

        # Adjust based on iteration (later iterations may need more)
        if iteration > 3:
            allocation_ratio = min(1.0, allocation_ratio * 1.2)

        # Calculate final budget
        total_budget = int(self.min_context_chars +
                          (self.max_context_chars - self.min_context_chars) * allocation_ratio)

        # Per-source allocation
        if source_count > 0:
            max_per_source = min(15000, total_budget // max(source_count, 5))
        else:
            max_per_source = 10000

        max_sources = total_budget // max_per_source

        return {
            "max_total_content": total_budget,
            "max_per_source": max_per_source,
            "max_sources": min(30, max_sources),
            "allocation_ratio": allocation_ratio
        }


# Singleton instances
_classifier: Optional[SufficientContextClassifier] = None
_optimizer: Optional[PositionalOptimizer] = None
_allocator: Optional[DynamicContextAllocator] = None


def get_sufficient_context_classifier() -> SufficientContextClassifier:
    """Get singleton classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = SufficientContextClassifier()
    return _classifier


def get_positional_optimizer() -> PositionalOptimizer:
    """Get singleton optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = PositionalOptimizer()
    return _optimizer


def get_dynamic_allocator() -> DynamicContextAllocator:
    """Get singleton allocator instance"""
    global _allocator
    if _allocator is None:
        _allocator = DynamicContextAllocator()
    return _allocator
