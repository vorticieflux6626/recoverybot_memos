"""
Adaptive Refinement Module for Agentic Search

Implements iterative refinement strategies based on:
- CRAG (Corrective RAG) - arXiv:2401.15884
- Self-RAG - arXiv:2310.11511
- FAIR-RAG - arXiv:2510.22344
- AT-RAG - arXiv:2410.12886

Key Features:
- Gap identification in synthesis
- Targeted refinement query generation
- Confidence-based decision routing
- Iterative loop until quality threshold met

Module Version: 0.28.0
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import httpx

logger = logging.getLogger("agentic.adaptive_refinement")


class RefinementDecision(str, Enum):
    """Decision outcomes for refinement routing."""
    COMPLETE = "complete"           # Confidence >= threshold, answer sufficient
    REFINE_QUERY = "refine_query"   # Generate targeted refinement queries
    WEB_FALLBACK = "web_fallback"   # Discard results, trigger fresh web search
    DECOMPOSE = "decompose"         # Break complex query into sub-questions
    ACCEPT_BEST = "accept_best"     # Max iterations reached, accept best result


class AnswerGrade(str, Enum):
    """Answer quality grades based on AT-RAG."""
    EXCELLENT = "excellent"   # Score 5: Fully answers with specific, actionable details
    GOOD = "good"             # Score 4: Mostly answers, minor gaps
    PARTIAL = "partial"       # Score 3: Partially answers, significant gaps
    TANGENTIAL = "tangential" # Score 2: Tangentially relevant
    INADEQUATE = "inadequate" # Score 1: Does not answer the query


@dataclass
class GapAnalysis:
    """Result of gap identification in synthesis."""
    has_gaps: bool
    gaps: List[str] = field(default_factory=list)
    coverage_score: float = 0.0
    missing_aspects: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)


@dataclass
class AnswerAssessment:
    """Result of answer grading."""
    grade: AnswerGrade
    score: int  # 1-5
    gaps: List[str] = field(default_factory=list)
    refinements: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class RefinementResult:
    """Result of a refinement attempt."""
    decision: RefinementDecision
    confidence_before: float
    confidence_after: float
    gaps_identified: List[str] = field(default_factory=list)
    queries_generated: List[str] = field(default_factory=list)
    iteration: int = 0
    duration_ms: int = 0


class AdaptiveRefinementEngine:
    """
    Engine for adaptive refinement of search results.

    Based on research into agentic RAG patterns:
    - FAIR-RAG's Structured Evidence Assessment for gap identification
    - AT-RAG's Answer Grader for adequacy assessment
    - CRAG's three-tier confidence routing
    - Self-RAG's reflection tokens for quality evaluation
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        evaluation_model: str = "gemma3:4b",
        min_confidence_threshold: float = 0.5,
        max_refinement_attempts: int = 3
    ):
        self.ollama_url = ollama_url
        self.evaluation_model = evaluation_model
        self.min_confidence_threshold = min_confidence_threshold
        self.max_refinement_attempts = max_refinement_attempts

        # Tracking
        self._refinement_history: List[RefinementResult] = []

    async def _call_llm(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """Call Ollama LLM for evaluation."""
        model = model or self.evaluation_model

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": 1024
                        }
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    async def identify_gaps(
        self,
        query: str,
        synthesis: str,
        sources: List[Dict[str, Any]]
    ) -> GapAnalysis:
        """
        Identify specific information gaps in the synthesis.

        Based on FAIR-RAG's Structured Evidence Assessment (SEA).
        Returns actionable gaps that can guide refinement queries.
        """
        if not synthesis or len(synthesis) < 100:
            return GapAnalysis(
                has_gaps=True,
                gaps=["Synthesis too short or missing"],
                coverage_score=0.0
            )

        # Extract source domains for context
        source_domains = []
        for src in sources[:5]:
            url = src.get('url', '') or src.get('link', '')
            if url:
                from urllib.parse import urlparse
                try:
                    domain = urlparse(url).netloc
                    if domain:
                        source_domains.append(domain)
                except (ValueError, AttributeError):
                    pass  # Invalid URL, skip

        prompt = f"""Analyze this Q&A for information gaps.

QUERY: {query}

ANSWER PROVIDED:
{synthesis[:3000]}

SOURCES CONSULTED: {', '.join(source_domains[:5]) if source_domains else 'Various web sources'}

TASK: Identify what specific information is MISSING that would fully answer the query.

Consider:
1. Are there specific steps, procedures, or values that should be included?
2. Are there alternative approaches or solutions not mentioned?
3. Are there important warnings, prerequisites, or requirements missing?
4. Is the answer specific enough for the user to take action?

Return your analysis as JSON:
{{
    "has_gaps": true/false,
    "coverage_score": 0.0-1.0,
    "gaps": ["specific gap 1", "specific gap 2"],
    "missing_aspects": ["aspect that needs more detail"],
    "suggested_queries": ["query to fill gap 1", "query to fill gap 2"]
}}

If the answer is complete and actionable, return:
{{"has_gaps": false, "coverage_score": 1.0, "gaps": [], "missing_aspects": [], "suggested_queries": []}}

IMPORTANT: Only list CONCRETE gaps, not vague suggestions. Be specific.
JSON Response:"""

        response = await self._call_llm(prompt)

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return GapAnalysis(
                    has_gaps=result.get("has_gaps", True),
                    gaps=result.get("gaps", [])[:5],  # Limit to 5 gaps
                    coverage_score=float(result.get("coverage_score", 0.5)),
                    missing_aspects=result.get("missing_aspects", [])[:3],
                    suggested_queries=result.get("suggested_queries", [])[:3]
                )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse gap analysis JSON: {e}")

        # Default: assume some gaps exist
        return GapAnalysis(
            has_gaps=True,
            gaps=["Unable to analyze gaps - assuming incomplete"],
            coverage_score=0.5
        )

    async def grade_answer(
        self,
        query: str,
        synthesis: str
    ) -> AnswerAssessment:
        """
        Grade answer adequacy on 1-5 scale.

        Based on AT-RAG's Answer Grader module.
        Returns grade, score, identified gaps, and suggested refinements.
        """
        if not synthesis or len(synthesis) < 50:
            return AnswerAssessment(
                grade=AnswerGrade.INADEQUATE,
                score=1,
                gaps=["No substantial answer provided"],
                reasoning="Synthesis is missing or too short"
            )

        prompt = f"""Grade this answer's adequacy for the query.

QUERY: {query}

ANSWER: {synthesis[:2500]}

SCORING RUBRIC:
5 (EXCELLENT) = Fully answers with specific, actionable details
4 (GOOD) = Mostly answers, only minor gaps
3 (PARTIAL) = Partially answers, has significant gaps
2 (TANGENTIAL) = Only tangentially relevant to the query
1 (INADEQUATE) = Does not meaningfully answer the query

Provide your assessment as JSON:
{{
    "score": <1-5>,
    "grade": "<excellent|good|partial|tangential|inadequate>",
    "gaps": ["list of missing information"],
    "refinements": ["suggested follow-up queries to fill gaps"],
    "reasoning": "brief explanation of the grade"
}}

JSON Response:"""

        response = await self._call_llm(prompt)

        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = int(result.get("score", 3))
                grade_str = result.get("grade", "partial").lower()

                # Map string to enum
                grade_map = {
                    "excellent": AnswerGrade.EXCELLENT,
                    "good": AnswerGrade.GOOD,
                    "partial": AnswerGrade.PARTIAL,
                    "tangential": AnswerGrade.TANGENTIAL,
                    "inadequate": AnswerGrade.INADEQUATE
                }
                grade = grade_map.get(grade_str, AnswerGrade.PARTIAL)

                return AnswerAssessment(
                    grade=grade,
                    score=min(5, max(1, score)),
                    gaps=result.get("gaps", [])[:5],
                    refinements=result.get("refinements", [])[:3],
                    reasoning=result.get("reasoning", "")
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse answer assessment JSON: {e}")

        # Default: partial grade
        return AnswerAssessment(
            grade=AnswerGrade.PARTIAL,
            score=3,
            reasoning="Unable to assess - defaulting to partial"
        )

    def decide_refinement_action(
        self,
        confidence: float,
        source_count: int,
        query_complexity: str,
        iteration: int,
        gap_analysis: Optional[GapAnalysis] = None,
        answer_assessment: Optional[AnswerAssessment] = None
    ) -> RefinementDecision:
        """
        Decide next action based on current state.

        Based on CRAG's three-tier confidence system and Adaptive-RAG routing.

        Decision Matrix:
        - confidence >= 0.7: COMPLETE (answer is sufficient)
        - confidence 0.4-0.7 + gaps: REFINE_QUERY
        - confidence < 0.4 + few sources: WEB_FALLBACK
        - complex query + low confidence + iteration < 2: DECOMPOSE
        - max iterations reached: ACCEPT_BEST
        """
        # Check if we've hit max iterations
        if iteration >= self.max_refinement_attempts:
            logger.info(f"Max refinement attempts ({self.max_refinement_attempts}) reached")
            return RefinementDecision.ACCEPT_BEST

        # High confidence - answer is sufficient
        if confidence >= 0.7:
            # But check if answer assessment says otherwise
            if answer_assessment and answer_assessment.score <= 2:
                logger.info("High confidence but low answer grade - refining")
                return RefinementDecision.REFINE_QUERY
            return RefinementDecision.COMPLETE

        # Medium confidence - check for gaps
        if confidence >= 0.4:
            if gap_analysis and gap_analysis.has_gaps and gap_analysis.gaps:
                return RefinementDecision.REFINE_QUERY
            if answer_assessment and answer_assessment.score <= 3:
                return RefinementDecision.REFINE_QUERY
            # No clear gaps identified, accept current result
            return RefinementDecision.COMPLETE

        # Low confidence
        if source_count < 3:
            # Very few sources - try web fallback
            return RefinementDecision.WEB_FALLBACK

        if query_complexity == "complex" and iteration < 2:
            # Complex query with low confidence - try decomposition
            return RefinementDecision.DECOMPOSE

        # Default: try to refine
        return RefinementDecision.REFINE_QUERY

    async def generate_refinement_queries(
        self,
        original_query: str,
        gaps: List[str],
        current_synthesis: str
    ) -> List[str]:
        """
        Generate targeted queries to fill identified gaps.

        Based on FAIR-RAG's adaptive query refinement.
        """
        if not gaps:
            return []

        # First try to use suggested queries from gap analysis
        # Then generate new ones if needed

        prompt = f"""Generate search queries to fill these information gaps.

ORIGINAL QUERY: {original_query}

CURRENT ANSWER SUMMARY: {current_synthesis[:500] if current_synthesis else 'No answer yet'}

GAPS TO FILL:
{chr(10).join(f'- {gap}' for gap in gaps[:5])}

Generate 2-3 specific search queries that would help fill these gaps.
Each query should be:
- Focused on ONE specific gap
- Include relevant technical terms from the original query
- Be concise (5-10 words)

Return as JSON array:
["query 1", "query 2", "query 3"]

JSON Response:"""

        response = await self._call_llm(prompt)

        try:
            # Try to extract JSON array
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                if isinstance(queries, list):
                    return [q for q in queries if isinstance(q, str)][:3]
        except json.JSONDecodeError:
            pass

        # Fallback: generate simple queries from gaps
        queries = []
        for gap in gaps[:3]:
            # Extract key terms from original query
            key_terms = ' '.join(original_query.split()[:5])
            query = f"{gap} {key_terms}"
            queries.append(query[:100])  # Limit length

        return queries

    async def decompose_query(
        self,
        query: str
    ) -> List[str]:
        """
        Decompose a complex query into simpler sub-questions.

        Used when confidence is low and query is complex.
        """
        prompt = f"""Break this complex query into simpler sub-questions.

QUERY: {query}

Generate 3-4 simpler, focused questions that together would fully answer the original query.
Each sub-question should be:
- Self-contained and searchable
- Focused on one specific aspect
- Answerable independently

Return as JSON array:
["sub-question 1", "sub-question 2", "sub-question 3"]

JSON Response:"""

        response = await self._call_llm(prompt)

        try:
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
                if isinstance(questions, list):
                    return [q for q in questions if isinstance(q, str)][:4]
        except json.JSONDecodeError:
            pass

        # Fallback: return original query
        return [query]

    def get_stats(self) -> Dict[str, Any]:
        """Get refinement engine statistics."""
        if not self._refinement_history:
            return {
                "total_refinements": 0,
                "avg_confidence_improvement": 0.0,
                "decisions": {}
            }

        decisions = {}
        total_improvement = 0.0

        for result in self._refinement_history:
            decision_name = result.decision.value
            decisions[decision_name] = decisions.get(decision_name, 0) + 1
            total_improvement += result.confidence_after - result.confidence_before

        return {
            "total_refinements": len(self._refinement_history),
            "avg_confidence_improvement": total_improvement / len(self._refinement_history),
            "avg_duration_ms": sum(r.duration_ms for r in self._refinement_history) / len(self._refinement_history),
            "decisions": decisions
        }

    def record_refinement(self, result: RefinementResult):
        """Record a refinement result for statistics."""
        self._refinement_history.append(result)
        # Keep only last 100 results
        if len(self._refinement_history) > 100:
            self._refinement_history = self._refinement_history[-100:]


# Singleton instance
_adaptive_refinement_engine: Optional[AdaptiveRefinementEngine] = None


def get_adaptive_refinement_engine(
    ollama_url: str = "http://localhost:11434",
    **kwargs
) -> AdaptiveRefinementEngine:
    """Get or create the adaptive refinement engine singleton."""
    global _adaptive_refinement_engine
    if _adaptive_refinement_engine is None:
        _adaptive_refinement_engine = AdaptiveRefinementEngine(
            ollama_url=ollama_url,
            **kwargs
        )
    return _adaptive_refinement_engine


def create_adaptive_refinement_engine(**kwargs) -> AdaptiveRefinementEngine:
    """Create a new adaptive refinement engine instance."""
    return AdaptiveRefinementEngine(**kwargs)
