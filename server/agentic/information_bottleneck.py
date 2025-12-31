"""
Information Bottleneck Filtering for RAG

Implements Information Bottleneck theory for effective noise filtering on retrieval-augmented
generation. Based on Zhu et al., ACL 2024: "An Information Bottleneck Perspective for
Effective Noise Filtering on Retrieval-Augmented Generation".

Key Insight:
- L_IB = I(Z; Y) - β·I(Z; X)
- Maximize mutual information between compressed representation Z and correct output Y
- Minimize mutual information between Z and potentially noisy retrieved passage X
- Achieves 2.5% compression rate while improving answer correctness

Integration with CRAG:
- Runs after CRAG quality assessment but before synthesis
- Further filters passages marked as CORRECT or AMBIGUOUS
- Removes noise while preserving task-relevant information
- Returns compressed, high-utility context for synthesis

Architecture:
1. Relevance Scorer: Estimates I(Z; Y) - how useful is content for answering query
2. Noise Estimator: Estimates I(Z; X) - how much noise/redundancy is present
3. IB Optimizer: Balances relevance vs noise to select optimal content
4. Compression Module: Extracts minimal sufficient representation

Research Reference:
- Zhu et al., ACL 2024: https://aclanthology.org/2024.acl-long.59/
- arXiv: https://arxiv.org/abs/2406.01549
"""

import asyncio
import json
import logging
import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

import httpx
import numpy as np

from .metrics import get_performance_metrics
from .context_limits import get_model_context_window

logger = logging.getLogger("agentic.information_bottleneck")


class FilteringLevel(Enum):
    """How aggressively to filter content."""
    MINIMAL = "minimal"       # Keep most content, light filtering
    MODERATE = "moderate"     # Balanced filtering (default)
    AGGRESSIVE = "aggressive" # Maximum compression


class ContentType(Enum):
    """Classification of content relevance."""
    ESSENTIAL = "essential"       # Critical task-relevant information
    SUPPORTING = "supporting"     # Useful context but not critical
    PERIPHERAL = "peripheral"     # Marginally relevant
    NOISE = "noise"               # Irrelevant or harmful to task


@dataclass
class PassageScore:
    """Information Bottleneck score for a single passage."""
    passage_id: str
    content: str
    relevance_score: float      # I(Z; Y) estimate - task relevance
    noise_score: float          # I(Z; X) estimate - noise/redundancy
    ib_score: float             # Final IB score = relevance - β*noise
    content_type: ContentType
    key_sentences: List[str]    # Extracted essential sentences
    compression_ratio: float    # len(key_sentences) / len(content)
    reasoning: str              # Explanation for scoring

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "relevance_score": self.relevance_score,
            "noise_score": self.noise_score,
            "ib_score": self.ib_score,
            "content_type": self.content_type.value,
            "compression_ratio": self.compression_ratio,
            "key_sentence_count": len(self.key_sentences),
            "reasoning": self.reasoning
        }


@dataclass
class IBFilterResult:
    """Result of Information Bottleneck filtering."""
    filtered_passages: List[Dict[str, Any]]  # Passages that passed the filter
    filtered_out_passages: List[Dict[str, Any]]  # Passages that were removed
    passage_scores: List[PassageScore]
    original_count: int
    filtered_count: int
    compression_rate: float  # filtered_count / original_count
    average_ib_score: float
    filtering_reasoning: str
    compressed_content: str  # Concatenated key sentences from all passages

    @property
    def total_compression_rate(self) -> float:
        """Overall compression including sentence extraction."""
        original_chars = sum(len(p["content"]) for p in self.filtered_passages + self.filtered_out_passages)
        compressed_chars = len(self.compressed_content)
        if original_chars == 0:
            return 0.0
        return compressed_chars / original_chars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "filtered_count": self.filtered_count,
            "compression_rate": self.compression_rate,
            "total_compression_rate": self.total_compression_rate,
            "average_ib_score": self.average_ib_score,
            "filtering_reasoning": self.filtering_reasoning,
            "passage_scores": [p.to_dict() for p in self.passage_scores]
        }


class InformationBottleneckFilter:
    """
    Information Bottleneck-based noise filter for RAG pipelines.

    Filters retrieved passages to maximize task-relevant information while
    minimizing noise and redundancy. Achieves significant compression while
    maintaining or improving answer quality.

    Formula: L_IB = I(Z; Y) - β·I(Z; X)
    - Z: compressed representation (selected content)
    - Y: correct/useful output (what we want to generate)
    - X: retrieved passage (potentially noisy)
    - β: tradeoff parameter (higher = more compression)
    """

    # Default β parameter (tradeoff between relevance and noise)
    # Higher β = more aggressive filtering/compression
    BETA_MINIMAL = 0.3      # Keep more content
    BETA_MODERATE = 0.6     # Balanced (default)
    BETA_AGGRESSIVE = 0.9   # Maximum compression

    # Thresholds for content classification
    IB_ESSENTIAL_THRESHOLD = 0.7    # Very high relevance, low noise
    IB_SUPPORTING_THRESHOLD = 0.4   # Decent relevance
    IB_PERIPHERAL_THRESHOLD = 0.2   # Marginal relevance
    # Below PERIPHERAL = NOISE

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",  # Fast model for scoring
        beta: float = 0.6
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.beta = beta

    async def filter(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        decomposed_questions: Optional[List[str]] = None,
        filtering_level: FilteringLevel = FilteringLevel.MODERATE,
        min_passages: int = 2,
        max_passages: int = 10
    ) -> IBFilterResult:
        """
        Apply Information Bottleneck filtering to retrieved passages.

        Args:
            query: Original user query
            passages: List of passages with 'content', 'title', 'url' fields
            decomposed_questions: Optional sub-questions for coverage checking
            filtering_level: How aggressively to filter
            min_passages: Minimum passages to keep regardless of scores
            max_passages: Maximum passages to return

        Returns:
            IBFilterResult with filtered passages and scores
        """
        if not passages:
            return IBFilterResult(
                filtered_passages=[],
                filtered_out_passages=[],
                passage_scores=[],
                original_count=0,
                filtered_count=0,
                compression_rate=0.0,
                average_ib_score=0.0,
                filtering_reasoning="No passages to filter",
                compressed_content=""
            )

        # Adjust β based on filtering level
        beta = self._get_beta_for_level(filtering_level)

        # Score all passages using Information Bottleneck principle
        scores = await self._score_passages(query, passages, decomposed_questions, beta)

        # Classify and filter based on IB scores
        filtered, filtered_out = self._apply_filter(
            scores, min_passages, max_passages, filtering_level
        )

        # Extract compressed content (key sentences only)
        compressed_content = self._extract_compressed_content(filtered)

        # Calculate overall metrics
        compression_rate = len(filtered) / len(passages) if passages else 0.0
        avg_score = np.mean([s.ib_score for s in scores]) if scores else 0.0

        # Generate reasoning summary
        reasoning = self._generate_filtering_reasoning(
            scores, len(filtered), len(filtered_out), compression_rate, filtering_level
        )

        # Map scores back to filtered/filtered_out passages
        score_map = {s.passage_id: s for s in scores}
        filtered_scores = [score_map.get(p["passage_id"], scores[0]) for p in filtered if "passage_id" in p]

        return IBFilterResult(
            filtered_passages=filtered,
            filtered_out_passages=filtered_out,
            passage_scores=scores,
            original_count=len(passages),
            filtered_count=len(filtered),
            compression_rate=compression_rate,
            average_ib_score=avg_score,
            filtering_reasoning=reasoning,
            compressed_content=compressed_content
        )

    def _get_beta_for_level(self, level: FilteringLevel) -> float:
        """Get β parameter for filtering level."""
        if level == FilteringLevel.MINIMAL:
            return self.BETA_MINIMAL
        elif level == FilteringLevel.AGGRESSIVE:
            return self.BETA_AGGRESSIVE
        return self.BETA_MODERATE

    async def _score_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        decomposed_questions: Optional[List[str]],
        beta: float
    ) -> List[PassageScore]:
        """Score each passage using IB principle."""
        scores = []

        # Prepare context for batch evaluation
        questions_context = ""
        if decomposed_questions:
            questions_context = f"\nSub-questions to answer:\n" + "\n".join(
                f"- {q}" for q in decomposed_questions[:5]
            )

        # Score passages in batches of 4 for efficiency
        batch_size = 4
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            batch_scores = await self._score_passage_batch(
                query, batch, questions_context, beta, start_idx=i
            )
            scores.extend(batch_scores)

        return scores

    async def _score_passage_batch(
        self,
        query: str,
        batch: List[Dict[str, Any]],
        questions_context: str,
        beta: float,
        start_idx: int
    ) -> List[PassageScore]:
        """Score a batch of passages."""
        passages_text = ""
        for j, p in enumerate(batch):
            content = p.get("content", p.get("snippet", ""))[:1000]
            title = p.get("title", "Unknown")
            passages_text += f"\n[Passage {start_idx + j + 1}]\nTitle: {title}\nContent: {content}\n"

        prompt = f"""Analyze these passages for answering the query. Apply Information Bottleneck principle:
- Score RELEVANCE (0-10): How directly useful for answering the query
- Score NOISE (0-10): How much irrelevant, redundant, or potentially misleading content
- Extract KEY SENTENCES: 1-3 most important sentences that directly answer the query

QUERY: {query}
{questions_context}

PASSAGES:
{passages_text}

For each passage, output JSON:
[
  {{
    "passage": 1,
    "relevance": 0-10,
    "noise": 0-10,
    "key_sentences": ["sentence 1", "sentence 2"],
    "reason": "brief explanation"
  }},
  ...
]

Rate honestly - low relevance for off-topic content, high noise for redundant/misleading info."""

        try:
            result = await self._call_llm(prompt, max_tokens=1024)

            # Parse JSON from response
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                evaluations = json.loads(json_match.group())

                scores = []
                for j, p in enumerate(batch):
                    eval_data = next(
                        (e for e in evaluations if e.get("passage") == start_idx + j + 1),
                        {"relevance": 5, "noise": 5, "key_sentences": [], "reason": "No evaluation"}
                    )

                    relevance = min(1.0, eval_data.get("relevance", 5) / 10)
                    noise = min(1.0, eval_data.get("noise", 5) / 10)

                    # IB Score: relevance - β * noise
                    ib_score = max(0.0, relevance - beta * noise)

                    # Classify content type based on IB score
                    if ib_score >= self.IB_ESSENTIAL_THRESHOLD:
                        content_type = ContentType.ESSENTIAL
                    elif ib_score >= self.IB_SUPPORTING_THRESHOLD:
                        content_type = ContentType.SUPPORTING
                    elif ib_score >= self.IB_PERIPHERAL_THRESHOLD:
                        content_type = ContentType.PERIPHERAL
                    else:
                        content_type = ContentType.NOISE

                    # Get key sentences
                    key_sentences = eval_data.get("key_sentences", [])
                    if not key_sentences:
                        # Fallback: use first sentence
                        content = p.get("content", p.get("snippet", ""))
                        if content:
                            sentences = re.split(r'[.!?]+', content)
                            key_sentences = [s.strip() for s in sentences[:1] if s.strip()]

                    # Calculate compression ratio
                    content = p.get("content", p.get("snippet", ""))
                    original_len = len(content) if content else 1
                    compressed_len = sum(len(s) for s in key_sentences)
                    compression_ratio = compressed_len / original_len if original_len > 0 else 0.0

                    passage_id = hashlib.md5(
                        (p.get("url", "") + p.get("title", "")).encode()
                    ).hexdigest()[:8]

                    scores.append(PassageScore(
                        passage_id=passage_id,
                        content=content[:500],
                        relevance_score=relevance,
                        noise_score=noise,
                        ib_score=ib_score,
                        content_type=content_type,
                        key_sentences=key_sentences,
                        compression_ratio=compression_ratio,
                        reasoning=eval_data.get("reason", "")
                    ))

                return scores

        except Exception as e:
            logger.warning(f"IB batch scoring failed: {e}, using heuristics")

        # Fallback: heuristic scoring
        return self._heuristic_score_batch(batch, query, beta, start_idx)

    # Noise pattern indicators - obvious non-content
    NOISE_PATTERNS = {
        "cookie", "cookies", "privacy policy", "subscribe", "newsletter",
        "sign up", "click here", "advertisement", "sponsored", "login",
        "accept all", "reject all", "consent", "gdpr", "terms of service",
        "copyright ©", "all rights reserved", "social media", "follow us",
        "share this", "tweet", "facebook", "instagram", "loading...",
        "please wait", "javascript required", "enable javascript"
    }

    # Technical content indicators - likely relevant
    TECHNICAL_PATTERNS = {
        "maintenance", "troubleshoot", "error", "alarm", "fault", "warning",
        "procedure", "step", "instruction", "manual", "guide", "fix",
        "repair", "replace", "calibrat", "reset", "diagnos", "inspect",
        "check", "verify", "ensure", "caution", "safety", "note:",
        "important:", "tip:", "servo", "motor", "encoder", "parameter"
    }

    def _heuristic_score_batch(
        self,
        batch: List[Dict[str, Any]],
        query: str,
        beta: float,
        start_idx: int
    ) -> List[PassageScore]:
        """Heuristic fallback scoring when LLM fails."""
        query_terms = set(query.lower().split())
        scores = []

        for j, p in enumerate(batch):
            content = p.get("content", p.get("snippet", "")).lower()
            title = p.get("title", "").lower()

            # Relevance: term overlap
            content_terms = set(content.split())
            overlap = len(query_terms & content_terms)
            relevance = min(1.0, overlap / max(len(query_terms), 1))

            # Boost relevance for technical content patterns
            technical_matches = sum(1 for pat in self.TECHNICAL_PATTERNS if pat in content)
            if technical_matches > 0:
                relevance = min(1.0, relevance + 0.1 * technical_matches)

            # Noise: estimate from text features
            # High noise indicators: very long, lots of repetition, ads-like patterns
            noise = 0.3  # Base noise
            if len(content) > 2000:
                noise += 0.2  # Long content often has more noise
            if len(set(content.split())) / max(len(content.split()), 1) < 0.5:
                noise += 0.2  # Low vocabulary diversity = redundancy

            # Detect obvious noise patterns (cookie notices, ads, etc.)
            noise_matches = sum(1 for pat in self.NOISE_PATTERNS if pat in content)
            if noise_matches > 0:
                noise = min(1.0, noise + 0.15 * noise_matches)  # Increase noise score

            ib_score = max(0.0, relevance - beta * noise)

            if ib_score >= self.IB_ESSENTIAL_THRESHOLD:
                content_type = ContentType.ESSENTIAL
            elif ib_score >= self.IB_SUPPORTING_THRESHOLD:
                content_type = ContentType.SUPPORTING
            elif ib_score >= self.IB_PERIPHERAL_THRESHOLD:
                content_type = ContentType.PERIPHERAL
            else:
                content_type = ContentType.NOISE

            # Extract first sentence as key
            sentences = re.split(r'[.!?]+', p.get("content", ""))
            key_sentences = [s.strip() for s in sentences[:1] if s.strip()]

            passage_id = hashlib.md5(
                (p.get("url", "") + p.get("title", "")).encode()
            ).hexdigest()[:8]

            scores.append(PassageScore(
                passage_id=passage_id,
                content=p.get("content", "")[:500],
                relevance_score=relevance,
                noise_score=noise,
                ib_score=ib_score,
                content_type=content_type,
                key_sentences=key_sentences,
                compression_ratio=len(" ".join(key_sentences)) / max(len(content), 1),
                reasoning="Heuristic scoring"
            ))

        return scores

    def _apply_filter(
        self,
        scores: List[PassageScore],
        min_passages: int,
        max_passages: int,
        level: FilteringLevel
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply filtering based on IB scores."""
        # Sort by IB score descending
        sorted_scores = sorted(scores, key=lambda s: s.ib_score, reverse=True)

        # Determine threshold based on filtering level
        if level == FilteringLevel.MINIMAL:
            threshold = self.IB_PERIPHERAL_THRESHOLD
        elif level == FilteringLevel.AGGRESSIVE:
            threshold = self.IB_SUPPORTING_THRESHOLD
        else:
            threshold = self.IB_PERIPHERAL_THRESHOLD  # Keep PERIPHERAL and above

        # Filter based on threshold
        above_threshold = [s for s in sorted_scores if s.ib_score >= threshold]
        below_threshold = [s for s in sorted_scores if s.ib_score < threshold]

        # Ensure min/max constraints
        filtered = above_threshold[:max_passages]

        # If below minimum, add from below_threshold
        if len(filtered) < min_passages:
            needed = min_passages - len(filtered)
            filtered.extend(below_threshold[:needed])

        # Convert back to passage dicts with passage_id
        filtered_passages = []
        filtered_out_passages = []

        filtered_ids = {s.passage_id for s in filtered}

        for score in scores:
            passage_dict = {
                "passage_id": score.passage_id,
                "content": score.content,
                "key_sentences": score.key_sentences,
                "ib_score": score.ib_score,
                "content_type": score.content_type.value
            }

            if score.passage_id in filtered_ids:
                filtered_passages.append(passage_dict)
            else:
                filtered_out_passages.append(passage_dict)

        return filtered_passages, filtered_out_passages

    def _extract_compressed_content(self, filtered: List[Dict[str, Any]]) -> str:
        """Extract compressed content from filtered passages."""
        compressed_parts = []

        for p in filtered:
            key_sentences = p.get("key_sentences", [])
            if key_sentences:
                # Join key sentences for this passage
                passage_summary = " ".join(key_sentences)
                compressed_parts.append(passage_summary)
            else:
                # Fallback to truncated content
                content = p.get("content", "")[:200]
                compressed_parts.append(content)

        return "\n\n".join(compressed_parts)

    def _generate_filtering_reasoning(
        self,
        scores: List[PassageScore],
        kept: int,
        removed: int,
        compression_rate: float,
        level: FilteringLevel
    ) -> str:
        """Generate human-readable reasoning for filtering decisions."""
        # Count by content type
        type_counts = {}
        for s in scores:
            type_counts[s.content_type.value] = type_counts.get(s.content_type.value, 0) + 1

        type_summary = ", ".join(f"{t}: {c}" for t, c in type_counts.items())

        avg_relevance = np.mean([s.relevance_score for s in scores]) if scores else 0
        avg_noise = np.mean([s.noise_score for s in scores]) if scores else 0

        return (
            f"IB Filtering ({level.value}): Kept {kept}/{kept + removed} passages "
            f"(compression: {compression_rate:.1%}). "
            f"Content types: {type_summary}. "
            f"Avg relevance: {avg_relevance:.2f}, Avg noise: {avg_noise:.2f}"
        )

    async def _call_llm(self, prompt: str, max_tokens: int = 512, request_id: str = "") -> str:
        """Call Ollama LLM for evaluation."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": max_tokens
                        }
                    }
                )
                if response.status_code == 200:
                    result = response.json().get("response", "")

                    # Track context utilization
                    metrics = get_performance_metrics()
                    req_id = request_id or f"ib_{hash(prompt) % 10000}"
                    metrics.record_context_utilization(
                        request_id=req_id,
                        agent_name="information_bottleneck",
                        model_name=self.model,
                        input_text=prompt,
                        output_text=result,
                        context_window=get_model_context_window(self.model)
                    )

                    return result
        except Exception as e:
            logger.error(f"IB LLM call failed: {e}")
        return ""


# Factory functions
def create_ib_filter(
    ollama_url: str = "http://localhost:11434",
    model: str = "gemma3:4b",
    beta: float = 0.6
) -> InformationBottleneckFilter:
    """Create an InformationBottleneckFilter instance."""
    return InformationBottleneckFilter(
        ollama_url=ollama_url,
        model=model,
        beta=beta
    )


# Singleton instance
_ib_filter: Optional[InformationBottleneckFilter] = None


def get_ib_filter() -> InformationBottleneckFilter:
    """Get the singleton InformationBottleneckFilter instance."""
    global _ib_filter
    if _ib_filter is None:
        _ib_filter = create_ib_filter()
    return _ib_filter
