"""
Self-RAG Reflection Module

Implements Self-Reflective RAG patterns based on arXiv:2310.11511:
- ISREL: Is the retrieved content relevant to the query?
- ISSUP: Is the synthesized response supported by sources?
- ISUSE: Is the response useful and answers the query?
- Temporal validation: Cross-check dates/years for consistency

This module catches factual errors like temporal contradictions before
they reach the user.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple

import httpx

from .metrics import get_performance_metrics
from .context_limits import get_model_context_window
from .gateway_client import get_gateway_client, LogicalModel, GatewayResponse

logger = logging.getLogger("agentic.self_reflection")


def extract_json_object(text: str) -> Optional[str]:
    """
    Extract a JSON object from text using proper brace matching.
    Handles nested objects and arrays correctly.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None

    brace_count = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i + 1]

    return None


class ReflectionToken(Enum):
    """Self-RAG reflection tokens"""
    ISREL = "ISREL"  # Is retrieved content relevant?
    ISSUP = "ISSUP"  # Is synthesis supported by sources?
    ISUSE = "ISUSE"  # Is response useful?


class SupportLevel(Enum):
    """How well the synthesis is supported by sources"""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"
    CONTRADICTED = "contradicted"


@dataclass
class TemporalFact:
    """A fact with temporal component extracted from text"""
    entity: str
    event: str
    date_mentioned: str  # Raw date string from text
    normalized_year: Optional[int] = None
    source_idx: Optional[int] = None


@dataclass
class TemporalConflict:
    """A detected temporal inconsistency"""
    entity: str
    event: str
    claim1: str
    claim2: str
    source1_idx: Optional[int] = None
    source2_idx: Optional[int] = None
    severity: str = "high"  # high, medium, low


@dataclass
class ReflectionResult:
    """Result of Self-RAG reflection on synthesis"""
    relevance_score: float  # ISREL: 0-1
    support_level: SupportLevel  # ISSUP
    usefulness_score: float  # ISUSE: 0-1
    temporal_conflicts: List[TemporalConflict] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    needs_refinement: bool = False
    refinement_suggestions: List[str] = field(default_factory=list)
    overall_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevance_score": self.relevance_score,
            "support_level": self.support_level.value,
            "usefulness_score": self.usefulness_score,
            "temporal_conflicts": [
                {
                    "entity": c.entity,
                    "event": c.event,
                    "claim1": c.claim1,
                    "claim2": c.claim2,
                    "severity": c.severity
                }
                for c in self.temporal_conflicts
            ],
            "unsupported_claims": self.unsupported_claims,
            "needs_refinement": self.needs_refinement,
            "refinement_suggestions": self.refinement_suggestions,
            "overall_confidence": self.overall_confidence
        }


class SelfReflectionAgent:
    """
    Implements Self-RAG style reflection on synthesized responses.

    Based on research:
    - Self-RAG: arXiv:2310.11511
    - LangGraph Self-Reflective RAG

    Key improvements over baseline verification:
    1. Reflects on synthesis BEFORE returning to user
    2. Explicitly checks temporal consistency
    3. Surfaces contradictions rather than hiding them
    4. Suggests refinements when quality is low
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b"  # Fast model for reflection
    ):
        self.ollama_url = ollama_url
        self.model = model

    async def reflect(
        self,
        query: str,
        synthesis: str,
        sources: List[Dict[str, Any]],
        scraped_content: Optional[List[str]] = None,
        request_id: str = "",
        use_gateway: bool = False
    ) -> ReflectionResult:
        """
        Perform Self-RAG reflection on synthesized response.

        Args:
            query: Original user query
            synthesis: The synthesized response
            sources: List of sources with title, snippet, url
            scraped_content: Optional full scraped content for deeper analysis
            request_id: Request ID for tracking
            use_gateway: If True, route through LLM Gateway service

        Returns:
            ReflectionResult with scores and issues
        """
        logger.info(f"Self-RAG reflection (gateway={use_gateway}): {query[:50]}...")

        # Run reflection checks in parallel where possible
        relevance_task = self._check_relevance(query, synthesis, request_id, use_gateway)
        support_task = self._check_support(synthesis, sources, scraped_content, request_id, use_gateway)
        temporal_task = self._check_temporal_consistency(synthesis, sources)

        relevance_score, support_result, temporal_conflicts = await asyncio.gather(
            relevance_task, support_task, temporal_task
        )

        support_level, unsupported_claims = support_result

        # Calculate usefulness (how well does synthesis answer the query?)
        usefulness_score = await self._check_usefulness(query, synthesis, request_id, use_gateway)

        # Determine if refinement is needed
        needs_refinement = (
            relevance_score < 0.7 or
            support_level in [SupportLevel.NO_SUPPORT, SupportLevel.CONTRADICTED] or
            usefulness_score < 0.6 or
            len(temporal_conflicts) > 0
        )

        # Generate refinement suggestions
        refinement_suggestions = []
        if temporal_conflicts:
            refinement_suggestions.append(
                f"Fix temporal inconsistency: {temporal_conflicts[0].claim1} vs {temporal_conflicts[0].claim2}"
            )
        if unsupported_claims:
            refinement_suggestions.append(
                f"Remove or verify unsupported claim: {unsupported_claims[0][:100]}"
            )
        if relevance_score < 0.7:
            refinement_suggestions.append("Focus synthesis more directly on the query")

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            relevance_score, support_level, usefulness_score, temporal_conflicts
        )

        return ReflectionResult(
            relevance_score=relevance_score,
            support_level=support_level,
            usefulness_score=usefulness_score,
            temporal_conflicts=temporal_conflicts,
            unsupported_claims=unsupported_claims,
            needs_refinement=needs_refinement,
            refinement_suggestions=refinement_suggestions,
            overall_confidence=confidence
        )

    async def _check_relevance(
        self,
        query: str,
        synthesis: str,
        request_id: str = "",
        use_gateway: bool = False
    ) -> float:
        """ISREL: Check if synthesis is relevant to the query"""
        prompt = f"""Rate how relevant this answer is to the question on a scale of 0-10.

Question: {query}

Answer: {synthesis[:1500]}

Output ONLY a JSON object:
{{"score": <0-10>, "reasoning": "brief explanation"}}"""

        try:
            if use_gateway:
                result = await self._call_via_gateway(prompt, max_tokens=128, request_id=request_id)
            else:
                result = await self._call_llm(prompt, max_tokens=128, request_id=request_id)
            json_str = extract_json_object(result)
            if json_str:
                data = json.loads(json_str)
                return min(1.0, data.get("score", 5) / 10)
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}")

        return 0.7  # Default moderate relevance

    async def _check_support(
        self,
        synthesis: str,
        sources: List[Dict[str, Any]],
        scraped_content: Optional[List[str]] = None,
        request_id: str = "",
        use_gateway: bool = False
    ) -> Tuple[SupportLevel, List[str]]:
        """ISSUP: Check if synthesis claims are supported by sources"""

        # Build source context
        source_text = "\n\n".join([
            f"[Source {i+1}] {s.get('title', 'Unknown')}: {s.get('snippet', '')[:300]}"
            for i, s in enumerate(sources[:8])
        ])

        # Add scraped content if available
        if scraped_content:
            source_text += "\n\n[Full Content Excerpts]:\n"
            # Handle different content types safely
            excerpts = []
            for c in scraped_content[:3]:
                if isinstance(c, str):
                    excerpts.append(c[:500])
                elif isinstance(c, dict):
                    # Handle dict with 'content' or 'text' key
                    content = c.get('content') or c.get('text') or str(c)
                    excerpts.append(str(content)[:500])
                else:
                    excerpts.append(str(c)[:500])
            source_text += "\n---\n".join(excerpts)

        prompt = f"""Analyze if the claims in this synthesis are supported by the sources.

SYNTHESIS:
{synthesis[:2000]}

SOURCES:
{source_text[:3000]}

For each major claim in the synthesis, determine if it's supported.
Output JSON:
{{
  "support_level": "fully_supported|partially_supported|no_support|contradicted",
  "unsupported_claims": ["list of claims not found in sources"],
  "contradictions": ["list of claims that contradict sources"]
}}"""

        unsupported = []
        try:
            if use_gateway:
                result = await self._call_via_gateway(prompt, max_tokens=512, request_id=request_id)
            else:
                result = await self._call_llm(prompt, max_tokens=512, request_id=request_id)
            json_str = extract_json_object(result)
            if json_str:
                data = json.loads(json_str)
                level_str = data.get("support_level", "partially_supported")
                unsupported = data.get("unsupported_claims", [])[:3]

                level_map = {
                    "fully_supported": SupportLevel.FULLY_SUPPORTED,
                    "partially_supported": SupportLevel.PARTIALLY_SUPPORTED,
                    "no_support": SupportLevel.NO_SUPPORT,
                    "contradicted": SupportLevel.CONTRADICTED
                }
                return level_map.get(level_str, SupportLevel.PARTIALLY_SUPPORTED), unsupported
        except Exception as e:
            logger.warning(f"Support check failed: {e}")

        return SupportLevel.PARTIALLY_SUPPORTED, unsupported

    async def _check_temporal_consistency(
        self,
        synthesis: str,
        sources: List[Dict[str, Any]]
    ) -> List[TemporalConflict]:
        """Check for temporal inconsistencies (dates, years, sequences)"""

        # Extract temporal facts from synthesis
        synthesis_facts = self._extract_temporal_facts(synthesis, source_idx=None)

        # Extract temporal facts from sources
        source_facts = []
        for i, source in enumerate(sources[:10]):
            snippet = source.get("snippet", "") + " " + source.get("title", "")
            facts = self._extract_temporal_facts(snippet, source_idx=i)
            source_facts.extend(facts)

        conflicts = []

        # Cross-check synthesis facts against sources
        for s_fact in synthesis_facts:
            for src_fact in source_facts:
                # Same entity, same type of event
                if self._entities_match(s_fact.entity, src_fact.entity):
                    if self._events_match(s_fact.event, src_fact.event):
                        # Check if years conflict
                        if s_fact.normalized_year and src_fact.normalized_year:
                            if s_fact.normalized_year != src_fact.normalized_year:
                                conflicts.append(TemporalConflict(
                                    entity=s_fact.entity,
                                    event=s_fact.event,
                                    claim1=f"{s_fact.entity} {s_fact.event} in {s_fact.date_mentioned}",
                                    claim2=f"Source says {src_fact.entity} {src_fact.event} in {src_fact.date_mentioned}",
                                    source2_idx=src_fact.source_idx,
                                    severity="high"
                                ))

        # Also check for internal synthesis consistency
        for i, fact1 in enumerate(synthesis_facts):
            for fact2 in synthesis_facts[i+1:]:
                if self._entities_match(fact1.entity, fact2.entity):
                    # Check logical timeline issues
                    if fact1.normalized_year and fact2.normalized_year:
                        if self._is_timeline_contradiction(fact1, fact2):
                            conflicts.append(TemporalConflict(
                                entity=fact1.entity,
                                event=f"{fact1.event} vs {fact2.event}",
                                claim1=f"{fact1.event}: {fact1.date_mentioned}",
                                claim2=f"{fact2.event}: {fact2.date_mentioned}",
                                severity="high"
                            ))

        return conflicts[:5]  # Limit to top 5 conflicts

    def _extract_temporal_facts(
        self,
        text: str,
        source_idx: Optional[int] = None
    ) -> List[TemporalFact]:
        """Extract temporal facts (entity + event + date) from text"""
        facts = []

        # Pattern: Entity was/is founded/released/created in YEAR
        patterns = [
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b)\s+(?:was|is|were)\s+(\w+(?:ed)?)\s+(?:in|on)?\s*(\d{4})',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b)\s+(\w+ed)\s+(?:in|on)?\s*(\d{4})',
            r'(?:founded|released|launched|created|established)\s+(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b)\s+(?:in|on)?\s*(\d{4})',
            r'(\bGPT-\d+\b)\s+(?:was|is)?\s*(\w+(?:ed)?)\s+(?:in|on)?\s*(\d{4})',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b).*?(\d{4})',  # Loose pattern
        ]

        for pattern in patterns[:4]:  # Use specific patterns first
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 3:
                    entity, event, year = groups[0], groups[1], groups[2]
                elif len(groups) == 2:
                    entity, year = groups[0], groups[1]
                    event = "mentioned"
                else:
                    continue

                try:
                    normalized_year = int(year)
                    if 1990 <= normalized_year <= 2030:  # Reasonable year range
                        facts.append(TemporalFact(
                            entity=entity,
                            event=event,
                            date_mentioned=year,
                            normalized_year=normalized_year,
                            source_idx=source_idx
                        ))
                except ValueError:
                    pass

        return facts

    def _entities_match(self, e1: str, e2: str) -> bool:
        """Check if two entity mentions refer to the same entity"""
        e1_lower = e1.lower().strip()
        e2_lower = e2.lower().strip()

        # Exact match
        if e1_lower == e2_lower:
            return True

        # One contains the other
        if e1_lower in e2_lower or e2_lower in e1_lower:
            return True

        # GPT model matching
        if "gpt" in e1_lower and "gpt" in e2_lower:
            return True

        return False

    def _events_match(self, ev1: str, ev2: str) -> bool:
        """Check if two events are similar enough to compare"""
        ev1_lower = ev1.lower()
        ev2_lower = ev2.lower()

        # Same event
        if ev1_lower == ev2_lower:
            return True

        # Similar events
        founding_words = {"founded", "started", "established", "created", "began", "launched"}
        release_words = {"released", "launched", "published", "announced"}

        ev1_set = set(ev1_lower.split())
        ev2_set = set(ev2_lower.split())

        if ev1_set & founding_words and ev2_set & founding_words:
            return True
        if ev1_set & release_words and ev2_set & release_words:
            return True

        return False

    def _is_timeline_contradiction(self, fact1: TemporalFact, fact2: TemporalFact) -> bool:
        """Check if two facts create a timeline contradiction"""
        if not (fact1.normalized_year and fact2.normalized_year):
            return False

        # Check for impossible sequences
        # e.g., "founded in 2021" and "existed before 2020 release"
        ev1 = fact1.event.lower()
        ev2 = fact2.event.lower()

        # If one is "before" and one is "after" type event, check order
        before_words = {"before", "prior", "founded", "created"}
        after_words = {"after", "following", "released", "available"}

        if any(w in ev1 for w in before_words) and any(w in ev2 for w in after_words):
            if fact1.normalized_year > fact2.normalized_year:
                return True

        return False

    async def _check_usefulness(
        self,
        query: str,
        synthesis: str,
        request_id: str = "",
        use_gateway: bool = False
    ) -> float:
        """ISUSE: Check if response actually answers the question"""
        prompt = f"""Rate how well this answer addresses the question on a scale of 0-10.
Consider: Does it directly answer what was asked? Is it complete? Is it actionable?

Question: {query}

Answer (excerpt): {synthesis[:1500]}

Output ONLY a JSON object:
{{"score": <0-10>, "missing_aspects": ["list any unanswered parts"]}}"""

        try:
            if use_gateway:
                result = await self._call_via_gateway(prompt, max_tokens=128, request_id=request_id)
            else:
                result = await self._call_llm(prompt, max_tokens=128, request_id=request_id)
            json_str = extract_json_object(result)
            if json_str:
                data = json.loads(json_str)
                return min(1.0, data.get("score", 5) / 10)
        except Exception as e:
            logger.warning(f"Usefulness check failed: {e}")

        return 0.7  # Default moderate usefulness

    def _calculate_confidence(
        self,
        relevance: float,
        support_level: SupportLevel,
        usefulness: float,
        temporal_conflicts: List[TemporalConflict]
    ) -> float:
        """Calculate overall confidence from reflection results"""

        # Base scores
        support_scores = {
            SupportLevel.FULLY_SUPPORTED: 1.0,
            SupportLevel.PARTIALLY_SUPPORTED: 0.7,
            SupportLevel.NO_SUPPORT: 0.3,
            SupportLevel.CONTRADICTED: 0.1
        }
        support_score = support_scores.get(support_level, 0.5)

        # Weighted average
        confidence = (
            relevance * 0.25 +
            support_score * 0.35 +
            usefulness * 0.25 +
            (1.0 if len(temporal_conflicts) == 0 else 0.3) * 0.15
        )

        # Penalty for severe temporal conflicts
        if temporal_conflicts:
            high_severity = sum(1 for c in temporal_conflicts if c.severity == "high")
            confidence *= max(0.3, 1.0 - high_severity * 0.2)

        return round(confidence, 2)

    async def refine_synthesis(
        self,
        original_synthesis: str,
        reflection: ReflectionResult,
        sources: List[Dict[str, Any]],
        request_id: str = "",
        use_gateway: bool = False
    ) -> str:
        """Refine synthesis based on reflection results"""

        if not reflection.needs_refinement:
            return original_synthesis

        issues = []
        if reflection.temporal_conflicts:
            for c in reflection.temporal_conflicts:
                issues.append(f"Temporal conflict: {c.claim1} vs {c.claim2}")
        if reflection.unsupported_claims:
            issues.append(f"Unsupported: {', '.join(reflection.unsupported_claims[:2])}")

        source_text = "\n".join([
            f"[{i+1}] {s.get('title', '')}: {s.get('snippet', '')[:200]}"
            for i, s in enumerate(sources[:5])
        ])

        prompt = f"""Fix the following issues in this synthesis and rewrite it:

ISSUES TO FIX:
{chr(10).join(issues)}

ORIGINAL SYNTHESIS:
{original_synthesis[:2000]}

SOURCES FOR FACT-CHECKING:
{source_text}

Write a corrected version that fixes the identified issues. Be accurate with dates and facts.
Keep the same structure but correct any errors."""

        try:
            if use_gateway:
                refined = await self._call_via_gateway(prompt, max_tokens=2048, request_id=request_id)
            else:
                refined = await self._call_llm(prompt, max_tokens=2048, request_id=request_id)
            if refined and len(refined) > 200:
                return refined
        except Exception as e:
            logger.error(f"Refinement failed: {e}")

        return original_synthesis

    async def _call_llm(self, prompt: str, max_tokens: int = 256, request_id: str = "") -> str:
        """Call Ollama LLM with context utilization tracking"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": max_tokens
                        }
                    }
                )
                if response.status_code == 200:
                    result = response.json().get("response", "")

                    # Track context utilization
                    metrics = get_performance_metrics()
                    req_id = request_id or f"self_rag_{hash(prompt) % 10000}"
                    metrics.record_context_utilization(
                        request_id=req_id,
                        agent_name="self_reflection",
                        model_name=self.model,
                        input_text=prompt,
                        output_text=result,
                        context_window=get_model_context_window(self.model)
                    )

                    return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        return ""

    async def _call_via_gateway(
        self,
        prompt: str,
        max_tokens: int = 256,
        request_id: str = ""
    ) -> str:
        """
        Call LLM via Gateway service with automatic fallback.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            request_id: Request ID for tracking context utilization

        Returns:
            LLM response text
        """
        try:
            gateway = get_gateway_client()

            response: GatewayResponse = await gateway.generate(
                prompt=prompt,
                model=LogicalModel.REFLECTOR,
                timeout=30.0,
                options={
                    "temperature": 0.2,
                    "num_predict": max_tokens,
                }
            )

            result = response.content

            # Track context utilization
            if request_id and result:
                metrics = get_performance_metrics()
                metrics.record_context_utilization(
                    request_id=request_id,
                    agent_name="self_reflection",
                    model_name=response.model,
                    input_text=prompt,
                    output_text=result,
                    context_window=get_model_context_window(response.model)
                )

            if response.fallback_used:
                logger.info(f"Gateway self_reflection used fallback to direct Ollama (model: {response.model})")

            return result

        except Exception as e:
            logger.error(f"Gateway self_reflection call failed: {e}, falling back to direct Ollama")
            return await self._call_llm(prompt, max_tokens, request_id)


# Factory function
def create_self_reflection_agent(
    ollama_url: str = "http://localhost:11434",
    model: str = "gemma3:4b"
) -> SelfReflectionAgent:
    """Create a SelfReflectionAgent instance"""
    return SelfReflectionAgent(ollama_url=ollama_url, model=model)


# Singleton instance
_reflection_agent: Optional[SelfReflectionAgent] = None

def get_self_reflection_agent() -> SelfReflectionAgent:
    """Get the singleton SelfReflectionAgent instance"""
    global _reflection_agent
    if _reflection_agent is None:
        _reflection_agent = create_self_reflection_agent()
    return _reflection_agent
