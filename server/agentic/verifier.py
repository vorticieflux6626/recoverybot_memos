"""
Verifier Agent - Fact Checking and Validation

Cross-checks claims against multiple sources and detects conflicts.
Essential for technical content where accuracy matters.
"""

import asyncio
import json
import logging
import re
from typing import List, Optional, Dict, Any

import httpx

from .models import VerificationResult, VerificationLevel, WebSearchResult
from .llm_config import get_llm_config

logger = logging.getLogger("agentic.verifier")


class VerifierAgent:
    """
    Verifies claims extracted from search results.

    Verification strategies:
    - NONE: Skip verification (fastest)
    - STANDARD: Check if claim appears in multiple sources
    - STRICT: Cross-reference with trusted domains, check for contradictions
    """

    def __init__(
        self,
        ollama_url: str = None,
        model: str = None
    ):
        llm_config = get_llm_config()
        self.ollama_url = ollama_url or llm_config.ollama.url
        self.model = model or llm_config.pipeline.verifier.model

    async def verify(
        self,
        claims: List[str],
        sources: List[WebSearchResult],
        level: VerificationLevel = VerificationLevel.STANDARD,
        scraped_content: Optional[List[str]] = None
    ) -> List[VerificationResult]:
        """
        Verify a list of claims against available sources.

        Args:
            claims: List of claims to verify
            sources: Web search results to check against
            level: How thorough to be
            scraped_content: Full scraped content for term matching (preferred over snippets)

        Returns:
            List of verification results
        """
        if level == VerificationLevel.NONE:
            return [
                VerificationResult(
                    claim=claim,
                    verified=True,  # Assume true when not verifying
                    confidence=0.5,
                    sources=[]
                )
                for claim in claims
            ]

        results = []

        for claim in claims[:10]:  # Limit to 10 claims
            if level == VerificationLevel.STANDARD:
                result = await self._verify_standard(claim, sources, scraped_content)
            else:  # STRICT
                result = await self._verify_strict(claim, sources, scraped_content)

            results.append(result)

        return results

    async def _verify_standard(
        self,
        claim: str,
        sources: List[WebSearchResult],
        scraped_content: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Standard verification: Check if claim appears in multiple sources.
        Uses scraped_content (full text) if available, falls back to snippets.
        """
        supporting_sources = []
        claim_lower = claim.lower()

        # Extract key terms from claim (lowered threshold from 4 to 3 chars)
        key_terms = set(
            word for word in claim_lower.split()
            if len(word) > 3 and word.isalpha()
        )

        # If no key terms found, use all alpha words
        if not key_terms:
            key_terms = set(word for word in claim_lower.split() if word.isalpha())

        if not key_terms:
            logger.warning(f"No key terms extracted from claim: {claim[:100]}")
            return VerificationResult(
                claim=claim,
                verified=False,
                confidence=0.0,
                sources=[],
                conflicts=[]
            )

        # Use scraped content if available (much better for term matching)
        if scraped_content:
            for idx, content in enumerate(scraped_content):
                content_lower = content.lower()

                # Check term overlap in full scraped content
                term_matches = sum(1 for term in key_terms if term in content_lower)
                match_ratio = term_matches / len(key_terms)

                if match_ratio > 0.4:  # 40% threshold for full content (more lenient)
                    # Get domain from corresponding source if available
                    if idx < len(sources):
                        supporting_sources.append(sources[idx].source_domain)
                    else:
                        supporting_sources.append(f"source_{idx}")

                    logger.debug(
                        f"Claim verified in source {idx}: {match_ratio:.1%} term match "
                        f"({term_matches}/{len(key_terms)} terms)"
                    )
        else:
            # Fallback to snippets (less effective but still works)
            for source in sources:
                snippet_lower = source.snippet.lower()
                title_lower = source.title.lower()
                combined = f"{title_lower} {snippet_lower}"

                # Check term overlap
                term_matches = sum(1 for term in key_terms if term in combined)
                match_ratio = term_matches / len(key_terms)

                if match_ratio > 0.5:  # 50% threshold for snippets (stricter)
                    supporting_sources.append(source.source_domain)

        # Calculate confidence based on source count and quality
        unique_domains = list(set(supporting_sources))

        # More nuanced confidence calculation
        if len(unique_domains) >= 3:
            confidence = min(1.0, 0.7 + len(unique_domains) * 0.1)  # 0.8+ for 3+ sources
        elif len(unique_domains) == 2:
            confidence = 0.6
        elif len(unique_domains) == 1:
            confidence = 0.4
        else:
            confidence = 0.1

        verified = len(unique_domains) >= 1  # Verified if at least 1 source supports

        logger.debug(
            f"Claim verification: '{claim[:50]}...' - "
            f"{len(unique_domains)} supporting sources, verified={verified}, conf={confidence:.2f}"
        )

        return VerificationResult(
            claim=claim,
            verified=verified,
            confidence=confidence,
            sources=unique_domains,
            conflicts=[]
        )

    async def _verify_strict(
        self,
        claim: str,
        sources: List[WebSearchResult],
        scraped_content: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Strict verification: Use LLM to analyze claim against sources.
        """
        # First do standard check (pass scraped_content for better term matching)
        standard_result = await self._verify_standard(claim, sources, scraped_content)

        # Then use LLM for deeper analysis
        sources_text = "\n".join([
            f"- {s.title}: {s.snippet[:200]}"
            for s in sources[:5]
        ])

        prompt = f"""<role>TECHNICAL FACT VERIFIER for industrial automation</role>
<expertise>Cross-reference claims against manufacturer documentation. Identify conflicts between sources. Assess source credibility for FANUC, Allen-Bradley, Siemens, and industrial equipment.</expertise>

Think through this step by step, providing thorough reasoning for each consideration.

Analyze if this claim is supported by the sources provided.

Claim: {claim}

Sources:
{sources_text}

Respond in JSON format:
{{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "conflicts": ["any contradicting information"],
  "reasoning": "detailed explanation of verification logic"
}}

JSON:"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 256}
                    }
                )

                if response.status_code == 200:
                    output = response.json().get("response", "")
                    analysis = self._parse_verification_response(output)

                    return VerificationResult(
                        claim=claim,
                        verified=analysis.get("supported", standard_result.verified),
                        confidence=analysis.get("confidence", standard_result.confidence),
                        sources=standard_result.sources,
                        conflicts=analysis.get("conflicts", [])
                    )

        except Exception as e:
            logger.error(f"Strict verification failed: {e}")

        return standard_result

    def _parse_verification_response(self, output: str) -> Dict[str, Any]:
        """Parse LLM verification response"""
        try:
            json_match = re.search(r'\{.*?\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return {}

    async def extract_claims(
        self,
        text: str,
        max_claims: int = 5
    ) -> List[str]:
        """
        Extract verifiable claims from synthesized text.
        Uses simple heuristics to identify factual statements.
        """
        claims = []

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip too short or too long
            if len(sentence) < 20 or len(sentence) > 300:
                continue

            # Skip questions
            if '?' in sentence:
                continue

            # Skip meta-statements
            skip_phrases = [
                'this synthesis', 'the search', 'based on',
                'according to', 'sources indicate', 'it appears'
            ]
            if any(phrase in sentence.lower() for phrase in skip_phrases):
                continue

            # Look for factual indicators
            factual_indicators = [
                'is', 'are', 'was', 'were', 'has', 'have',
                'reduces', 'increases', 'helps', 'causes',
                'provides', 'includes', 'contains'
            ]
            if any(ind in sentence.lower().split() for ind in factual_indicators):
                claims.append(sentence)

            if len(claims) >= max_claims:
                break

        return claims

    def calculate_overall_confidence(
        self,
        verification_results: List[VerificationResult],
        source_count: int = 0,
        unique_domains: int = 0,
        synthesis_length: int = 0,
        scraped_sources: int = 0
    ) -> float:
        """
        Calculate overall confidence score using multiple signals.

        Confidence is calculated from:
        - Verification results (if available): 40% weight
        - Source diversity (unique domains): 25% weight
        - Content depth (scraped sources): 20% weight
        - Synthesis quality (length): 15% weight

        This ensures meaningful confidence even when claim extraction
        doesn't find verifiable statements.
        """
        scores = []
        weights = []

        # 1. Verification score (40% weight when available)
        if verification_results:
            verified_count = sum(1 for r in verification_results if r.verified)
            avg_confidence = sum(r.confidence for r in verification_results) / len(verification_results)
            verification_rate = verified_count / len(verification_results)
            verification_score = (verification_rate * 0.6 + avg_confidence * 0.4)
            scores.append(verification_score)
            weights.append(0.40)

        # 2. Source diversity score (25% weight)
        # More unique domains = higher confidence
        if unique_domains > 0 or source_count > 0:
            domains = unique_domains if unique_domains > 0 else source_count
            # Scale: 1 domain = 0.2, 5 domains = 0.7, 10+ domains = 1.0
            diversity_score = min(1.0, 0.2 + (domains - 1) * 0.09)
            scores.append(diversity_score)
            weights.append(0.25)

        # 3. Content depth score (20% weight)
        # Successfully scraped sources indicate thorough research
        if scraped_sources > 0 or source_count > 0:
            scraped = scraped_sources if scraped_sources > 0 else min(source_count, 5)
            # Scale: 1 source = 0.3, 3 sources = 0.7, 5+ sources = 1.0
            depth_score = min(1.0, 0.3 + (scraped - 1) * 0.175)
            scores.append(depth_score)
            weights.append(0.20)

        # 4. Synthesis quality score (15% weight)
        # Longer, more detailed synthesis = higher confidence
        if synthesis_length > 0:
            # Scale: 500 chars = 0.3, 2000 chars = 0.7, 4000+ chars = 1.0
            quality_score = min(1.0, 0.3 + (synthesis_length / 4000) * 0.7)
            scores.append(quality_score)
            weights.append(0.15)

        # Calculate weighted average
        if not scores:
            return 0.5  # Default when no signals available

        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        confidence = sum(s * w for s, w in zip(scores, normalized_weights))

        logger.debug(
            f"Confidence calculation: scores={[f'{s:.2f}' for s in scores]}, "
            f"weights={[f'{w:.2f}' for w in normalized_weights]}, "
            f"final={confidence:.2f}"
        )

        return round(confidence, 2)
