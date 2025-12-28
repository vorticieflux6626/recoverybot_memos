"""
CRAG-Style Retrieval Evaluator

Implements Corrective RAG (arXiv:2401.15884) pattern for pre-synthesis retrieval quality assessment.

Key Innovation:
- Evaluates retrieval quality BEFORE synthesis (unlike Self-RAG which evaluates AFTER)
- Triggers corrective actions: refine queries, web search fallback, decompose-recompose
- Uses lightweight model for fast evaluation

Corrective Actions:
1. CORRECT: High-quality retrieval → proceed to synthesis
2. AMBIGUOUS: Mixed quality → refine queries and re-retrieve
3. INCORRECT: Poor retrieval → discard and trigger web search fallback

This complements Self-RAG to create a two-stage quality control:
  Search Results → CRAG Eval → [Corrective Action] → Synthesis → Self-RAG Eval → [Refinement]
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

import httpx

logger = logging.getLogger("agentic.retrieval_evaluator")


class RetrievalQuality(Enum):
    """Quality levels for retrieval evaluation"""
    CORRECT = "correct"       # At least one document highly relevant
    AMBIGUOUS = "ambiguous"   # Partial relevance, needs refinement
    INCORRECT = "incorrect"   # Poor retrieval, needs fallback


class CorrectiveAction(Enum):
    """Actions to take based on retrieval quality"""
    PROCEED = "proceed"           # Continue to synthesis
    REFINE_QUERY = "refine_query" # Search with refined queries
    WEB_FALLBACK = "web_fallback" # Discard and search web
    DECOMPOSE = "decompose"       # Break into sub-questions


@dataclass
class DocumentScore:
    """Quality score for a single retrieved document"""
    title: str
    url: str
    relevance_score: float  # 0-1: How relevant to query
    quality_score: float    # 0-1: Source quality/trustworthiness
    coverage_score: float   # 0-1: How much of query it answers
    combined_score: float   # Weighted combination

    @property
    def is_relevant(self) -> bool:
        return self.combined_score >= 0.6


@dataclass
class RetrievalEvaluation:
    """Result of retrieval quality evaluation"""
    quality: RetrievalQuality
    recommended_action: CorrectiveAction
    document_scores: List[DocumentScore]
    overall_relevance: float  # 0-1: Aggregate relevance
    query_coverage: float     # 0-1: How much of query is covered
    reasoning: str            # Explanation for decision
    refined_queries: List[str] = field(default_factory=list)  # If action is REFINE_QUERY
    decomposed_questions: List[str] = field(default_factory=list)  # If action is DECOMPOSE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality": self.quality.value,
            "recommended_action": self.recommended_action.value,
            "overall_relevance": self.overall_relevance,
            "query_coverage": self.query_coverage,
            "reasoning": self.reasoning,
            "document_count": len(self.document_scores),
            "relevant_count": sum(1 for d in self.document_scores if d.is_relevant),
            "refined_queries": self.refined_queries,
            "decomposed_questions": self.decomposed_questions
        }


class RetrievalEvaluator:
    """
    CRAG-Style Retrieval Quality Evaluator.

    Based on: Corrective Retrieval Augmented Generation (arXiv:2401.15884)

    Key differences from original CRAG:
    - Uses LLM instead of fine-tuned T5 (no training data needed)
    - Integrates with existing query decomposition
    - Lighter weight for real-time evaluation

    Evaluation Criteria:
    1. Relevance: Does document relate to query?
    2. Quality: Is source trustworthy?
    3. Coverage: How much of query does it answer?
    """

    # Quality thresholds (CRAG-inspired)
    CORRECT_THRESHOLD = 0.7    # At least one doc above this = CORRECT
    AMBIGUOUS_THRESHOLD = 0.4  # All docs below CORRECT but some above this = AMBIGUOUS
    # Below AMBIGUOUS = INCORRECT

    # Trust scoring for domains
    TRUSTED_DOMAINS = {
        "wikipedia.org": 0.9,
        "arxiv.org": 0.95,
        "github.com": 0.85,
        "stackoverflow.com": 0.8,
        "nature.com": 0.95,
        "ieee.org": 0.9,
        "acm.org": 0.9,
        "docs.python.org": 0.9,
        ".gov": 0.85,
        ".edu": 0.85,
    }

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b"  # Fast model for evaluation
    ):
        self.ollama_url = ollama_url
        self.model = model

    async def evaluate(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        decomposed_questions: Optional[List[str]] = None
    ) -> RetrievalEvaluation:
        """
        Evaluate retrieval quality and recommend corrective action.

        Args:
            query: Original user query
            search_results: List of search results with title, snippet, url
            decomposed_questions: Optional sub-questions for coverage assessment

        Returns:
            RetrievalEvaluation with quality level and recommended action
        """
        if not search_results:
            return RetrievalEvaluation(
                quality=RetrievalQuality.INCORRECT,
                recommended_action=CorrectiveAction.WEB_FALLBACK,
                document_scores=[],
                overall_relevance=0.0,
                query_coverage=0.0,
                reasoning="No search results retrieved",
                refined_queries=[f'"{query}"', f"{query} guide", f"{query} explained"]
            )

        # Score each document
        document_scores = await self._score_documents(query, search_results, decomposed_questions)

        # Calculate aggregate metrics
        overall_relevance = self._calculate_overall_relevance(document_scores)
        query_coverage = await self._calculate_query_coverage(query, search_results, decomposed_questions)

        # Determine quality level (CRAG classification)
        quality = self._classify_quality(document_scores, overall_relevance, query_coverage)

        # Determine corrective action
        action, reasoning, refined_queries, decomposed = await self._determine_action(
            quality, query, document_scores, overall_relevance, query_coverage, decomposed_questions
        )

        return RetrievalEvaluation(
            quality=quality,
            recommended_action=action,
            document_scores=document_scores,
            overall_relevance=overall_relevance,
            query_coverage=query_coverage,
            reasoning=reasoning,
            refined_queries=refined_queries,
            decomposed_questions=decomposed
        )

    async def _score_documents(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        decomposed_questions: Optional[List[str]] = None
    ) -> List[DocumentScore]:
        """Score each document for relevance, quality, and coverage"""
        scores = []

        # Batch evaluate for efficiency
        docs_text = "\n\n".join([
            f"[Doc {i+1}]\nTitle: {r.get('title', 'Unknown')}\n"
            f"URL: {r.get('url', 'Unknown')}\n"
            f"Snippet: {r.get('snippet', '')[:300]}"
            for i, r in enumerate(search_results[:8])
        ])

        prompt = f"""Evaluate how well each document answers this query.

QUERY: {query}

DOCUMENTS:
{docs_text}

For each document, rate on a scale of 0-10:
- relevance: How directly relevant to the query
- coverage: How much of the query it answers

Output JSON array:
[
  {{"doc": 1, "relevance": 0-10, "coverage": 0-10, "reason": "brief explanation"}},
  ...
]"""

        try:
            result = await self._call_llm(prompt, max_tokens=512)
            # Extract JSON array
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                doc_evals = json.loads(json_match.group())

                for i, r in enumerate(search_results[:8]):
                    # Find matching evaluation
                    eval_data = next((e for e in doc_evals if e.get("doc") == i + 1), {})

                    relevance = min(1.0, eval_data.get("relevance", 5) / 10)
                    coverage = min(1.0, eval_data.get("coverage", 5) / 10)
                    quality = self._calculate_source_quality(r.get("url", ""))

                    # Combined score (weighted)
                    combined = (relevance * 0.5) + (quality * 0.2) + (coverage * 0.3)

                    scores.append(DocumentScore(
                        title=r.get("title", "Unknown"),
                        url=r.get("url", ""),
                        relevance_score=relevance,
                        quality_score=quality,
                        coverage_score=coverage,
                        combined_score=combined
                    ))
            else:
                # Fallback to heuristic scoring
                scores = self._heuristic_scoring(query, search_results)

        except Exception as e:
            logger.warning(f"Document scoring failed: {e}, using heuristics")
            scores = self._heuristic_scoring(query, search_results)

        return scores

    def _heuristic_scoring(
        self,
        query: str,
        search_results: List[Dict[str, Any]]
    ) -> List[DocumentScore]:
        """Fallback heuristic scoring when LLM fails"""
        scores = []
        query_terms = set(query.lower().split())

        for r in search_results[:8]:
            title = r.get("title", "").lower()
            snippet = r.get("snippet", "").lower()
            url = r.get("url", "")

            # Term overlap for relevance
            text_terms = set(title.split() + snippet.split())
            overlap = len(query_terms & text_terms)
            relevance = min(1.0, overlap / max(len(query_terms), 1) * 1.5)

            # Source quality
            quality = self._calculate_source_quality(url)

            # Coverage heuristic (length-based)
            coverage = min(1.0, len(snippet) / 300)

            combined = (relevance * 0.5) + (quality * 0.2) + (coverage * 0.3)

            scores.append(DocumentScore(
                title=r.get("title", "Unknown"),
                url=url,
                relevance_score=relevance,
                quality_score=quality,
                coverage_score=coverage,
                combined_score=combined
            ))

        return scores

    def _calculate_source_quality(self, url: str) -> float:
        """Calculate source trustworthiness based on domain"""
        url_lower = url.lower()

        for domain, score in self.TRUSTED_DOMAINS.items():
            if domain in url_lower:
                return score

        # Default moderate trust
        return 0.6

    def _calculate_overall_relevance(self, document_scores: List[DocumentScore]) -> float:
        """Calculate aggregate relevance across all documents"""
        if not document_scores:
            return 0.0

        # Use top-3 documents weighted by position
        top_scores = sorted(document_scores, key=lambda d: d.combined_score, reverse=True)[:3]
        weights = [0.5, 0.3, 0.2]

        total = 0.0
        for i, score in enumerate(top_scores):
            weight = weights[i] if i < len(weights) else 0.1
            total += score.combined_score * weight

        return min(1.0, total / sum(weights[:len(top_scores)]))

    async def _calculate_query_coverage(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        decomposed_questions: Optional[List[str]] = None
    ) -> float:
        """Calculate how much of the query is covered by results"""
        if not decomposed_questions:
            # Simple coverage: average of document coverage scores
            return sum(r.get("coverage_score", 0.5) for r in search_results[:5]) / 5

        # Check which sub-questions are addressed
        covered = 0
        all_text = " ".join([
            r.get("title", "") + " " + r.get("snippet", "")
            for r in search_results[:10]
        ]).lower()

        for question in decomposed_questions:
            # Simple keyword check (could be enhanced with semantic matching)
            q_terms = set(question.lower().split())
            if len(q_terms & set(all_text.split())) >= len(q_terms) * 0.5:
                covered += 1

        return covered / max(len(decomposed_questions), 1)

    def _classify_quality(
        self,
        document_scores: List[DocumentScore],
        overall_relevance: float,
        query_coverage: float
    ) -> RetrievalQuality:
        """Classify retrieval quality using CRAG thresholds"""
        if not document_scores:
            return RetrievalQuality.INCORRECT

        # Check if any document is highly relevant (CRAG CORRECT condition)
        max_score = max(d.combined_score for d in document_scores)

        if max_score >= self.CORRECT_THRESHOLD and query_coverage >= 0.6:
            return RetrievalQuality.CORRECT
        elif max_score >= self.AMBIGUOUS_THRESHOLD or query_coverage >= 0.4:
            return RetrievalQuality.AMBIGUOUS
        else:
            return RetrievalQuality.INCORRECT

    async def _determine_action(
        self,
        quality: RetrievalQuality,
        query: str,
        document_scores: List[DocumentScore],
        overall_relevance: float,
        query_coverage: float,
        decomposed_questions: Optional[List[str]]
    ) -> Tuple[CorrectiveAction, str, List[str], List[str]]:
        """Determine corrective action based on quality assessment"""

        if quality == RetrievalQuality.CORRECT:
            return (
                CorrectiveAction.PROCEED,
                f"Retrieval quality is good (relevance={overall_relevance:.2f}, coverage={query_coverage:.2f})",
                [],
                []
            )

        elif quality == RetrievalQuality.AMBIGUOUS:
            # Generate refined queries
            refined = await self._generate_refined_queries(query, document_scores)
            return (
                CorrectiveAction.REFINE_QUERY,
                f"Retrieval is ambiguous (relevance={overall_relevance:.2f}). Refining queries...",
                refined,
                []
            )

        else:  # INCORRECT
            # Check if decomposition might help
            if len(query.split()) > 10 or "and" in query.lower():
                decomposed = await self._decompose_query(query)
                return (
                    CorrectiveAction.DECOMPOSE,
                    f"Retrieval is poor. Decomposing complex query into sub-questions...",
                    [],
                    decomposed
                )
            else:
                # Web fallback with different terms
                refined = await self._generate_fallback_queries(query)
                return (
                    CorrectiveAction.WEB_FALLBACK,
                    f"Retrieval is poor (relevance={overall_relevance:.2f}). Trying web search fallback...",
                    refined,
                    []
                )

    async def _generate_refined_queries(
        self,
        query: str,
        document_scores: List[DocumentScore]
    ) -> List[str]:
        """Generate refined queries based on what's missing"""
        # Get topics from relevant docs
        relevant_docs = [d for d in document_scores if d.is_relevant]

        prompt = f"""The original query is: "{query}"

The search found some relevant results but coverage is incomplete.

Generate 2-3 refined search queries that:
1. Focus on aspects not covered by current results
2. Use more specific terminology
3. Could find better sources

Output JSON array of queries:
["query1", "query2", "query3"]"""

        try:
            result = await self._call_llm(prompt, max_tokens=256)
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                return json.loads(json_match.group())[:3]
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")

        # Fallback
        return [f"{query} explained", f"{query} guide", f"what is {query}"]

    async def _generate_fallback_queries(self, query: str) -> List[str]:
        """Generate alternative queries for web fallback"""
        prompt = f"""The search for "{query}" returned poor results.

Generate 3 alternative search queries using:
1. Different keywords or synonyms
2. More general or more specific framing
3. Question format if original wasn't

Output JSON array:
["query1", "query2", "query3"]"""

        try:
            result = await self._call_llm(prompt, max_tokens=256)
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                return json.loads(json_match.group())[:3]
        except Exception as e:
            logger.warning(f"Fallback query generation failed: {e}")

        return [f'"{query}"', f"what is {query}", f"{query} definition"]

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-questions"""
        prompt = f"""Break this complex query into 2-4 simpler sub-questions:

Query: "{query}"

Output JSON array of sub-questions:
["question1", "question2", ...]"""

        try:
            result = await self._call_llm(prompt, max_tokens=256)
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                return json.loads(json_match.group())[:4]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")

        return [query]

    async def _call_llm(self, prompt: str, max_tokens: int = 256) -> str:
        """Call Ollama LLM"""
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
                    return response.json().get("response", "")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        return ""


# Factory functions
def create_retrieval_evaluator(
    ollama_url: str = "http://localhost:11434",
    model: str = "gemma3:4b"
) -> RetrievalEvaluator:
    """Create a RetrievalEvaluator instance"""
    return RetrievalEvaluator(ollama_url=ollama_url, model=model)


# Singleton instance
_retrieval_evaluator: Optional[RetrievalEvaluator] = None

def get_retrieval_evaluator() -> RetrievalEvaluator:
    """Get the singleton RetrievalEvaluator instance"""
    global _retrieval_evaluator
    if _retrieval_evaluator is None:
        _retrieval_evaluator = create_retrieval_evaluator()
    return _retrieval_evaluator
