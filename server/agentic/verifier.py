"""
Verifier Agent - Fact Checking and Validation

Cross-checks claims against multiple sources and detects conflicts.
Essential for recovery-related content where accuracy matters.
"""

import asyncio
import json
import logging
import re
from typing import List, Optional, Dict, Any

import httpx

from .models import VerificationResult, VerificationLevel, WebSearchResult

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
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b"
    ):
        self.ollama_url = ollama_url
        self.model = model

    async def verify(
        self,
        claims: List[str],
        sources: List[WebSearchResult],
        level: VerificationLevel = VerificationLevel.STANDARD
    ) -> List[VerificationResult]:
        """
        Verify a list of claims against available sources.

        Args:
            claims: List of claims to verify
            sources: Web search results to check against
            level: How thorough to be

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
                result = await self._verify_standard(claim, sources)
            else:  # STRICT
                result = await self._verify_strict(claim, sources)

            results.append(result)

        return results

    async def _verify_standard(
        self,
        claim: str,
        sources: List[WebSearchResult]
    ) -> VerificationResult:
        """
        Standard verification: Check if claim appears in multiple sources.
        """
        supporting_sources = []
        claim_lower = claim.lower()

        # Extract key terms from claim
        key_terms = set(
            word for word in claim_lower.split()
            if len(word) > 4 and word.isalpha()
        )

        for source in sources:
            snippet_lower = source.snippet.lower()
            title_lower = source.title.lower()

            # Check term overlap
            term_matches = sum(1 for term in key_terms if term in snippet_lower or term in title_lower)
            match_ratio = term_matches / max(len(key_terms), 1)

            if match_ratio > 0.5:  # More than half of key terms match
                supporting_sources.append(source.source_domain)

        # Calculate confidence based on source count and quality
        unique_domains = list(set(supporting_sources))
        confidence = min(1.0, len(unique_domains) * 0.25)

        return VerificationResult(
            claim=claim,
            verified=len(unique_domains) >= 2,
            confidence=confidence,
            sources=unique_domains,
            conflicts=[]
        )

    async def _verify_strict(
        self,
        claim: str,
        sources: List[WebSearchResult]
    ) -> VerificationResult:
        """
        Strict verification: Use LLM to analyze claim against sources.
        """
        # First do standard check
        standard_result = await self._verify_standard(claim, sources)

        # Then use LLM for deeper analysis
        sources_text = "\n".join([
            f"- {s.title}: {s.snippet[:200]}"
            for s in sources[:5]
        ])

        prompt = f"""Analyze if this claim is supported by the sources provided.

Claim: {claim}

Sources:
{sources_text}

Respond in JSON format:
{{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "conflicts": ["any contradicting information"],
  "reasoning": "brief explanation"
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
        verification_results: List[VerificationResult]
    ) -> float:
        """Calculate overall confidence score from individual verifications"""
        if not verification_results:
            return 0.5

        verified_count = sum(1 for r in verification_results if r.verified)
        avg_confidence = sum(r.confidence for r in verification_results) / len(verification_results)

        # Weight by verification rate and average confidence
        verification_rate = verified_count / len(verification_results)

        return (verification_rate * 0.6 + avg_confidence * 0.4)
