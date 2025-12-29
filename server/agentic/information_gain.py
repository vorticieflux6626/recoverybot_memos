"""
Document Information Gain (DIG) Scoring Module

Based on InfoGain-RAG (Kuaishou Technology) research achieving +17.9% over naive RAG.

Key insight: Document utility should be measured by its IMPACT on generation confidence,
not just semantic similarity. A document that confuses the model (negative DIG) should
be filtered even if semantically similar.

DIG(d) = P(correct | context + d) - P(correct | context)

References:
- InfoGain-RAG: Kuaishou Technology
- RAGAS: arXiv:2309.15217 (reference-free evaluation)
"""

import asyncio
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import httpx
import json
import re

logger = logging.getLogger(__name__)


class DIGCategory(str, Enum):
    """Document Information Gain categories."""
    POSITIVE = "positive"      # Increases confidence (keep)
    NEUTRAL = "neutral"        # No significant effect (may keep)
    NEGATIVE = "negative"      # Decreases confidence (filter)


@dataclass
class DIGScore:
    """Document Information Gain score result."""
    document_id: str
    content_hash: str
    dig_score: float              # -1.0 to 1.0
    category: DIGCategory
    baseline_confidence: float    # Confidence without this doc
    augmented_confidence: float   # Confidence with this doc
    relevance_score: float        # 0-1 semantic relevance
    quality_score: float          # 0-1 content quality
    reasoning: str                # LLM explanation


@dataclass
class DIGBatchResult:
    """Result of batch DIG scoring."""
    scores: List[DIGScore]
    total_documents: int
    positive_count: int
    neutral_count: int
    negative_count: int
    average_dig: float
    processing_time_ms: float


class DocumentInformationGain:
    """
    Calculate Document Information Gain (DIG) for context curation.

    DIG measures the change in LLM generation confidence when a document
    is added to the context. This helps filter:
    - Noise: Documents that confuse the model
    - Redundancy: Documents that add no new information
    - Contradictions: Documents that conflict with established facts

    Research basis: InfoGain-RAG achieved +17.9% over naive RAG by
    filtering documents with negative or zero information gain.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",  # Fast model for scoring
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.05
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self._cache: Dict[str, DIGScore] = {}

    def _hash_content(self, content: str) -> str:
        """Generate content hash for caching."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.3
    ) -> str:
        """Call Ollama for scoring."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_ctx": 8192
                    }
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")

    async def calculate_dig(
        self,
        query: str,
        document: str,
        existing_context: List[str],
        document_id: Optional[str] = None
    ) -> DIGScore:
        """
        Calculate Document Information Gain for a single document.

        DIG = P(answer_quality | context + doc) - P(answer_quality | context)

        Args:
            query: The original user query
            document: The document to evaluate
            existing_context: Already selected documents (baseline)
            document_id: Optional identifier for the document

        Returns:
            DIGScore with gain value and category
        """
        import time
        start_time = time.time()

        content_hash = self._hash_content(f"{query}:{document[:500]}")
        doc_id = document_id or content_hash

        # Check cache
        cache_key = f"{content_hash}:{len(existing_context)}"
        if cache_key in self._cache:
            logger.debug(f"DIG cache hit for {doc_id}")
            return self._cache[cache_key]

        # Truncate for prompt efficiency
        doc_truncated = document[:2000] if len(document) > 2000 else document
        context_summary = "\n---\n".join([c[:500] for c in existing_context[:5]])

        # Single prompt that evaluates both relevance and information gain
        prompt = f"""Evaluate this document's information gain for answering a query.

QUERY: {query}

EXISTING CONTEXT (what we already know):
{context_summary if context_summary else "[No existing context]"}

NEW DOCUMENT TO EVALUATE:
{doc_truncated}

Analyze and provide scores in this exact JSON format:
{{
    "relevance": <0.0-1.0, how relevant is this document to the query>,
    "quality": <0.0-1.0, how high quality/authoritative is the content>,
    "new_information": <0.0-1.0, how much NEW info does this add beyond existing context>,
    "contradicts_existing": <true/false, does it contradict established facts>,
    "confidence_impact": <-1.0 to 1.0, would adding this HELP or HURT answer quality>,
    "reasoning": "<brief explanation>"
}}

Focus on whether this document IMPROVES our ability to answer the query correctly.
Output ONLY the JSON, no other text."""

        try:
            response = await self._call_llm(prompt)

            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Fallback parsing
                data = {
                    "relevance": 0.5,
                    "quality": 0.5,
                    "new_information": 0.3,
                    "contradicts_existing": False,
                    "confidence_impact": 0.0,
                    "reasoning": "Failed to parse LLM response"
                }

            relevance = float(data.get("relevance", 0.5))
            quality = float(data.get("quality", 0.5))
            new_info = float(data.get("new_information", 0.3))
            contradicts = data.get("contradicts_existing", False)
            confidence_impact = float(data.get("confidence_impact", 0.0))
            reasoning = data.get("reasoning", "")

            # Calculate DIG score
            # Combines relevance, novelty, and direct confidence impact
            if contradicts:
                dig_score = -0.5  # Strong negative for contradictions
            else:
                # Weighted combination
                dig_score = (
                    0.3 * relevance +
                    0.3 * new_info +
                    0.2 * quality +
                    0.2 * confidence_impact
                ) - 0.3  # Normalize to center around 0

            # Clamp to [-1, 1]
            dig_score = max(-1.0, min(1.0, dig_score))

            # Categorize
            if dig_score >= self.positive_threshold:
                category = DIGCategory.POSITIVE
            elif dig_score <= self.negative_threshold:
                category = DIGCategory.NEGATIVE
            else:
                category = DIGCategory.NEUTRAL

            # Calculate baseline vs augmented confidence
            baseline = 0.5  # Assume 50% baseline
            augmented = baseline + dig_score * 0.3  # Scale impact

            result = DIGScore(
                document_id=doc_id,
                content_hash=content_hash,
                dig_score=dig_score,
                category=category,
                baseline_confidence=baseline,
                augmented_confidence=max(0.0, min(1.0, augmented)),
                relevance_score=relevance,
                quality_score=quality,
                reasoning=reasoning
            )

            # Cache result
            self._cache[cache_key] = result

            duration = (time.time() - start_time) * 1000
            logger.debug(f"DIG score for {doc_id}: {dig_score:.3f} ({category.value}) in {duration:.0f}ms")

            return result

        except Exception as e:
            logger.error(f"DIG calculation failed: {e}")
            # Return neutral score on error
            return DIGScore(
                document_id=doc_id,
                content_hash=content_hash,
                dig_score=0.0,
                category=DIGCategory.NEUTRAL,
                baseline_confidence=0.5,
                augmented_confidence=0.5,
                relevance_score=0.5,
                quality_score=0.5,
                reasoning=f"Error: {str(e)}"
            )

    async def batch_dig_ranking(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        existing_context: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        filter_negative: bool = True
    ) -> DIGBatchResult:
        """
        Rank documents by cumulative information gain.

        Uses sliding window approach to mitigate length bias:
        - Score each document against current context
        - Add positive documents to context incrementally
        - Re-score remaining documents with updated context

        Args:
            query: The user query
            documents: List of documents with 'content' and optional 'id' keys
            existing_context: Pre-existing context (if any)
            top_k: Return only top-k documents (None for all positive)
            filter_negative: Whether to filter negative DIG documents

        Returns:
            DIGBatchResult with ranked documents
        """
        import time
        start_time = time.time()

        context = list(existing_context) if existing_context else []
        scores: List[DIGScore] = []
        remaining_docs = list(enumerate(documents))

        logger.info(f"Batch DIG scoring {len(documents)} documents for: {query[:50]}...")

        # Greedy selection with incremental context building
        while remaining_docs:
            # Score all remaining documents against current context
            batch_scores = await asyncio.gather(*[
                self.calculate_dig(
                    query=query,
                    document=doc.get("content", ""),
                    existing_context=context,
                    document_id=doc.get("id") or doc.get("url") or f"doc_{idx}"
                )
                for idx, doc in remaining_docs
            ])

            # Find best positive document
            best_idx = -1
            best_score = float('-inf')
            best_dig = None

            for i, (orig_idx, doc) in enumerate(remaining_docs):
                dig = batch_scores[i]
                if dig.dig_score > best_score and dig.category != DIGCategory.NEGATIVE:
                    best_score = dig.dig_score
                    best_idx = i
                    best_dig = dig

            if best_idx == -1 or best_score <= self.negative_threshold:
                # No more positive documents
                # Add remaining neutral ones if not filtering
                if not filter_negative:
                    for i, (orig_idx, doc) in enumerate(remaining_docs):
                        if batch_scores[i].category == DIGCategory.NEUTRAL:
                            scores.append(batch_scores[i])
                break

            # Add best document to selection and context
            scores.append(best_dig)
            orig_idx, doc = remaining_docs[best_idx]
            context.append(doc.get("content", "")[:1000])
            remaining_docs.pop(best_idx)

            # Stop if we have enough
            if top_k and len(scores) >= top_k:
                break

            # Limit iterations for efficiency
            if len(scores) >= 20:
                logger.info("DIG scoring capped at 20 documents")
                break

        # Sort by DIG score
        scores.sort(key=lambda x: x.dig_score, reverse=True)

        # Apply top_k limit
        if top_k:
            scores = scores[:top_k]

        # Calculate statistics
        positive_count = sum(1 for s in scores if s.category == DIGCategory.POSITIVE)
        neutral_count = sum(1 for s in scores if s.category == DIGCategory.NEUTRAL)
        negative_count = sum(1 for s in scores if s.category == DIGCategory.NEGATIVE)
        avg_dig = sum(s.dig_score for s in scores) / len(scores) if scores else 0.0

        duration = (time.time() - start_time) * 1000

        logger.info(
            f"DIG batch complete: {len(scores)}/{len(documents)} docs, "
            f"avg={avg_dig:.3f}, +{positive_count}/~{neutral_count}/-{negative_count}, "
            f"{duration:.0f}ms"
        )

        return DIGBatchResult(
            scores=scores,
            total_documents=len(documents),
            positive_count=positive_count,
            neutral_count=neutral_count,
            negative_count=negative_count,
            average_dig=avg_dig,
            processing_time_ms=duration
        )

    async def filter_by_dig(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        min_dig: float = 0.0,
        max_documents: int = 10
    ) -> Tuple[List[Dict[str, Any]], DIGBatchResult]:
        """
        Filter documents by minimum DIG threshold.

        Convenience method that returns filtered documents with their scores.

        Args:
            query: User query
            documents: Documents to filter
            min_dig: Minimum DIG score to keep
            max_documents: Maximum documents to return

        Returns:
            Tuple of (filtered_documents, batch_result)
        """
        result = await self.batch_dig_ranking(
            query=query,
            documents=documents,
            top_k=max_documents,
            filter_negative=True
        )

        # Filter by minimum DIG
        passing_scores = [s for s in result.scores if s.dig_score >= min_dig]

        # Map back to original documents
        doc_by_id = {
            (d.get("id") or d.get("url") or f"doc_{i}"): d
            for i, d in enumerate(documents)
        }

        filtered_docs = []
        for score in passing_scores:
            if score.document_id in doc_by_id:
                doc = doc_by_id[score.document_id].copy()
                doc["dig_score"] = score.dig_score
                doc["dig_category"] = score.category.value
                filtered_docs.append(doc)

        return filtered_docs, result

    def clear_cache(self) -> int:
        """Clear the DIG score cache."""
        count = len(self._cache)
        self._cache.clear()
        return count


# Singleton instance
_dig_scorer: Optional[DocumentInformationGain] = None


def get_dig_scorer(
    ollama_url: str = "http://localhost:11434",
    model: str = "gemma3:4b"
) -> DocumentInformationGain:
    """Get or create the DIG scorer singleton."""
    global _dig_scorer
    if _dig_scorer is None:
        _dig_scorer = DocumentInformationGain(ollama_url=ollama_url, model=model)
    return _dig_scorer
