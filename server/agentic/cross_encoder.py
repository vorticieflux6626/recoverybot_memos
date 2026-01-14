"""
Cross-Encoder Reranking Module

Implements LLM-based cross-encoder reranking for improved NDCG (+28% expected).
Unlike bi-encoders (BGE-M3) which encode query and doc separately,
cross-encoders encode them together for richer semantic interaction.

Research Basis:
- Nogueira et al., "Passage Re-ranking with BERT" (arXiv:1901.04085)
- Cross-encoder > Bi-encoder for reranking by ~10-30% on NDCG
- LLM-as-reranker approaches show strong results with proper prompting

Author: Claude Code
Date: December 2025
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import httpx
import json

from .llm_config import get_llm_config
from .prompt_config import get_prompt_config

logger = logging.getLogger(__name__)


@dataclass
class RerankCandidate:
    """A candidate document for reranking."""
    doc_id: str
    title: str
    snippet: str
    url: str
    original_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    """Result of cross-encoder reranking."""
    candidates: List[RerankCandidate]
    rerank_time_ms: int
    model_used: str
    num_reranked: int
    avg_score: float


class CrossEncoderReranker:
    """
    LLM-based cross-encoder for document reranking.

    Uses the LLM to jointly evaluate (query, document) pairs
    and produce relevance scores from 0-10.

    Key features:
    - Batch processing for efficiency (up to 5 docs per prompt)
    - Score normalization to 0-1 range
    - Fallback to original scores on failure
    - Lightweight model (gemma3:4b) for speed
    """

    @staticmethod
    def _get_rerank_prompt() -> str:
        """Get rerank prompt from central config."""
        return get_prompt_config().agent_prompts.cross_encoder.rerank

    # Legacy class variable for backward compatibility
    RERANK_PROMPT = property(lambda self: self._get_rerank_prompt())

    def __init__(
        self,
        ollama_url: str = None,
        model: str = None,
        batch_size: int = 5,
        timeout: float = 30.0
    ):
        llm_config = get_llm_config()
        self.ollama_url = ollama_url or llm_config.ollama.url
        self.model = model or llm_config.utility.cross_encoder.model
        self.batch_size = batch_size
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Statistics
        self.total_reranks = 0
        self.total_docs_reranked = 0
        self.total_time_ms = 0
        self.cache_hits = 0

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout)
            )
        return self._client

    async def _score_batch(
        self,
        query: str,
        candidates: List[RerankCandidate]
    ) -> List[float]:
        """Score a batch of candidates using the LLM."""
        if not candidates:
            return []

        # Format documents for prompt
        doc_lines = []
        for i, c in enumerate(candidates, 1):
            # Truncate snippet for efficiency
            snippet = c.snippet[:300] if c.snippet else ""
            doc_lines.append(f"[{i}] Title: {c.title}\n    Content: {snippet}")

        documents_str = "\n\n".join(doc_lines)

        prompt = self._get_rerank_prompt().format(
            query=query,
            documents=documents_str
        )

        try:
            client = await self._get_client()

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for consistent scoring
                    "num_ctx": 4096,
                    "num_predict": 50  # Just need the score array
                }
            }

            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )
            if response.status_code != 200:
                logger.warning(f"Cross-encoder LLM call failed: {response.status_code}")
                return [c.original_score for c in candidates]

            result = response.json()
            response_text = result.get("response", "").strip()

            # Parse scores from response
            scores = self._parse_scores(response_text, len(candidates))
            return scores

        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            return [c.original_score for c in candidates]

    def _parse_scores(self, response: str, expected_count: int) -> List[float]:
        """Parse score array from LLM response."""
        try:
            # Try to extract JSON array from response
            # Handle various formats: [1,2,3], [ 1, 2, 3 ], etc.
            import re

            # Find array pattern
            match = re.search(r'\[[\d\s,\.]+\]', response)
            if match:
                array_str = match.group()
                scores = json.loads(array_str)

                # Normalize to 0-1 range
                normalized = [min(max(float(s) / 10.0, 0.0), 1.0) for s in scores]

                # Pad or truncate to expected count
                if len(normalized) < expected_count:
                    normalized.extend([0.5] * (expected_count - len(normalized)))
                elif len(normalized) > expected_count:
                    normalized = normalized[:expected_count]

                return normalized

            # Fallback: try to parse numbers from response
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if len(numbers) >= expected_count:
                scores = [float(n) for n in numbers[:expected_count]]
                return [min(max(s / 10.0, 0.0), 1.0) for s in scores]

        except Exception as e:
            logger.debug(f"Failed to parse rerank scores: {e}")

        # Return default scores
        return [0.5] * expected_count

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        top_k: int = 10
    ) -> RerankResult:
        """
        Rerank candidates using cross-encoder scoring.

        Args:
            query: The search query
            candidates: List of candidates to rerank
            top_k: Number of top results to return

        Returns:
            RerankResult with reranked candidates
        """
        start = time.time()

        if not candidates:
            return RerankResult(
                candidates=[],
                rerank_time_ms=0,
                model_used=self.model,
                num_reranked=0,
                avg_score=0.0
            )

        # Limit candidates to rerank
        candidates_to_rerank = candidates[:min(len(candidates), 20)]

        # Process in batches
        all_scores = []
        for i in range(0, len(candidates_to_rerank), self.batch_size):
            batch = candidates_to_rerank[i:i + self.batch_size]
            batch_scores = await self._score_batch(query, batch)
            all_scores.extend(batch_scores)

        # Apply scores to candidates
        for i, candidate in enumerate(candidates_to_rerank):
            if i < len(all_scores):
                candidate.rerank_score = all_scores[i]
            else:
                candidate.rerank_score = candidate.original_score

        # Sort by rerank score
        candidates_to_rerank.sort(key=lambda x: x.rerank_score, reverse=True)

        # Take top_k
        result_candidates = candidates_to_rerank[:top_k]

        # Calculate stats
        rerank_time_ms = int((time.time() - start) * 1000)
        avg_score = sum(c.rerank_score for c in result_candidates) / len(result_candidates) if result_candidates else 0.0

        # Update statistics
        self.total_reranks += 1
        self.total_docs_reranked += len(candidates_to_rerank)
        self.total_time_ms += rerank_time_ms

        logger.info(f"Cross-encoder reranked {len(candidates_to_rerank)} docs in {rerank_time_ms}ms, avg_score={avg_score:.2f}")

        return RerankResult(
            candidates=result_candidates,
            rerank_time_ms=rerank_time_ms,
            model_used=self.model,
            num_reranked=len(candidates_to_rerank),
            avg_score=avg_score
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        avg_time = self.total_time_ms / max(self.total_reranks, 1)
        avg_docs = self.total_docs_reranked / max(self.total_reranks, 1)

        return {
            "total_reranks": self.total_reranks,
            "total_docs_reranked": self.total_docs_reranked,
            "total_time_ms": self.total_time_ms,
            "avg_rerank_time_ms": avg_time,
            "avg_docs_per_rerank": avg_docs,
            "model": self.model
        }

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# =============================================================================
# Singleton and Factory
# =============================================================================

_cross_encoder: Optional[CrossEncoderReranker] = None


def get_cross_encoder(
    ollama_url: str = None,
    model: str = None
) -> CrossEncoderReranker:
    """Get or create the global cross-encoder instance (config from llm_models.yaml)."""
    global _cross_encoder

    if _cross_encoder is None:
        _cross_encoder = CrossEncoderReranker(
            ollama_url=ollama_url,
            model=model
        )

    return _cross_encoder


async def rerank_search_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 10,
    ollama_url: str = "http://localhost:11434"
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query
        results: List of search result dicts with 'title', 'snippet', 'url'
        top_k: Number of results to return
        ollama_url: Ollama API URL

    Returns:
        Reranked list of results with updated scores
    """
    reranker = get_cross_encoder(ollama_url)

    # Convert to candidates
    candidates = []
    for i, r in enumerate(results):
        candidates.append(RerankCandidate(
            doc_id=f"doc_{i}",
            title=r.get("title", ""),
            snippet=r.get("snippet", ""),
            url=r.get("url", ""),
            original_score=r.get("relevance_score", r.get("score", 0.5)),
            metadata=r
        ))

    # Rerank
    result = await reranker.rerank(query, candidates, top_k)

    # Convert back to dicts with updated scores
    reranked_results = []
    for c in result.candidates:
        # Update original dict with new score
        updated = dict(c.metadata)
        updated["relevance_score"] = c.rerank_score
        updated["rerank_score"] = c.rerank_score
        updated["original_score"] = c.original_score
        reranked_results.append(updated)

    return reranked_results
