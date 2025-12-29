"""
FLARE Forward-Looking Active REtrieval Module

Based on FLARE research (EMNLP 2023) by Jiang et al.

Key insight: During generation, when the model shows uncertainty (low token
probability), use the tentatively generated content as a retrieval query
to fetch supporting documents before continuing.

This enables proactive retrieval based on what the model PREDICTS it needs
to say next, rather than reactive retrieval based on what it already said.

References:
- FLARE: Active Retrieval Augmented Generation (EMNLP 2023)
- arXiv:2305.06983
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import httpx
import json

logger = logging.getLogger(__name__)


class RetrievalTrigger(str, Enum):
    """Reasons for triggering retrieval."""
    LOW_CONFIDENCE = "low_confidence"      # Token probability below threshold
    UNCERTAIN_PHRASE = "uncertain_phrase"  # Hedging language detected
    FACTUAL_CLAIM = "factual_claim"        # Making verifiable claim
    MISSING_DETAIL = "missing_detail"      # Vague or incomplete content
    CONTINUATION = "continuation"          # Periodic retrieval


@dataclass
class RetrievalPoint:
    """A point where retrieval was triggered during generation."""
    position: int                          # Character position in text
    trigger: RetrievalTrigger
    confidence: float                      # Confidence at this point
    tentative_content: str                 # What model was about to say
    query: str                             # Generated retrieval query
    retrieved_docs: List[str] = field(default_factory=list)


@dataclass
class FLAREResult:
    """Result of FLARE-enhanced generation."""
    final_text: str
    retrieval_points: List[RetrievalPoint]
    total_retrievals: int
    average_confidence: float
    documents_used: List[str]


class FLARERetriever:
    """
    Forward-Looking Active REtrieval.

    During generation, monitors confidence and triggers retrieval when:
    1. Token probability drops below threshold
    2. Hedging language detected (might, perhaps, possibly)
    3. Making factual claims that need verification
    4. Content becomes vague or generic

    Uses tentative generation as retrieval query:
    1. Generate next ~50 tokens tentatively
    2. If low confidence, use as query
    3. Retrieve relevant documents
    4. Regenerate with retrieved context
    """

    # Hedging patterns that indicate uncertainty
    UNCERTAIN_PATTERNS = [
        r'\b(might|may|could|possibly|perhaps|probably|likely|unlikely)\b',
        r'\b(I think|I believe|I assume|supposedly|allegedly)\b',
        r'\b(it seems|appears to|tends to|generally|typically)\b',
        r'\b(not sure|uncertain|unclear|unknown|unverified)\b',
        r'\b(some say|often|sometimes|occasionally)\b',
    ]

    # Factual claim patterns that might need verification
    FACTUAL_PATTERNS = [
        r'\b(according to|research shows|studies indicate|data suggests)\b',
        r'\b(in \d{4}|since \d{4}|from \d{4})\b',  # Date references
        r'\b(\d+(?:\.\d+)?%|\d+(?:,\d{3})*\s+(?:people|users|cases))\b',  # Statistics
        r'\b(official|confirmed|verified|reported)\b',
        r'\b(first|last|only|largest|smallest|most|least)\b',  # Superlatives
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        confidence_threshold: float = 0.6,
        check_interval: int = 50,          # Check every N tokens
        max_retrievals: int = 5,           # Max retrievals per generation
        tentative_tokens: int = 50,        # Tokens to generate tentatively
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.check_interval = check_interval
        self.max_retrievals = max_retrievals
        self.tentative_tokens = tentative_tokens

        # Compile patterns
        self.uncertain_regex = re.compile(
            '|'.join(self.UNCERTAIN_PATTERNS),
            re.IGNORECASE
        )
        self.factual_regex = re.compile(
            '|'.join(self.FACTUAL_PATTERNS),
            re.IGNORECASE
        )

    async def _generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 100
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate text with token log probabilities.

        Returns tuple of (generated_text, token_info_list).
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "num_predict": max_tokens,
                            "logprobs": True,  # Request log probabilities
                            "num_ctx": 8192
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()

                text = result.get("response", "")

                # Extract logprobs if available
                # Note: Ollama's logprobs format may vary
                context = result.get("context", [])
                logprobs = []

                # Estimate confidence from response metadata
                # This is a fallback when logprobs aren't available
                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 1)

                return text, logprobs

        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return "", []

    def _detect_uncertainty_in_text(self, text: str) -> List[Tuple[int, str, float]]:
        """
        Detect uncertainty points in generated text.

        Returns list of (position, matched_pattern, confidence_estimate).
        """
        uncertainties = []

        # Check for hedging language
        for match in self.uncertain_regex.finditer(text):
            uncertainties.append((
                match.start(),
                match.group(),
                0.4  # Low confidence for hedging
            ))

        # Check for unverified factual claims
        for match in self.factual_regex.finditer(text):
            uncertainties.append((
                match.start(),
                match.group(),
                0.5  # Medium confidence - needs verification
            ))

        return sorted(uncertainties, key=lambda x: x[0])

    async def _generate_retrieval_query(
        self,
        original_query: str,
        tentative_content: str,
        trigger: RetrievalTrigger
    ) -> str:
        """
        Generate a retrieval query based on tentative content.

        The key FLARE insight: Use what the model WANTS to say
        as the basis for retrieval queries.
        """
        prompt = f"""Based on the original question and the tentative answer being generated,
create a focused search query to find supporting information.

Original question: {original_query}

Tentative answer being written:
"{tentative_content}"

Trigger: {trigger.value}

Generate a specific search query (max 10 words) to find information that would:
1. Verify or correct the tentative content
2. Fill in missing details
3. Provide supporting evidence

Search query:"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 30}
                    }
                )
                response.raise_for_status()
                query = response.json().get("response", "").strip()

                # Clean up the query
                query = query.replace('"', '').replace("'", "")
                query = query.split('\n')[0][:100]  # First line, max 100 chars

                return query if query else tentative_content[:50]

        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return tentative_content[:50]

    async def _retrieve_documents(
        self,
        query: str,
        top_k: int = 3
    ) -> List[str]:
        """
        Retrieve documents for query.

        This is a placeholder - should integrate with the actual
        search infrastructure (SearXNG, BGE-M3, etc.).
        """
        # In production, this would call the search service
        # For now, return empty - caller should inject retriever
        logger.debug(f"FLARE would retrieve for: {query}")
        return []

    def _estimate_confidence_from_text(self, text: str) -> float:
        """
        Estimate overall confidence from text characteristics.

        Used as fallback when logprobs not available.
        """
        if not text:
            return 0.0

        # Start with neutral confidence
        confidence = 0.7

        # Reduce for hedging language
        hedge_count = len(self.uncertain_regex.findall(text))
        confidence -= hedge_count * 0.08

        # Reduce for short, vague responses
        word_count = len(text.split())
        if word_count < 20:
            confidence -= 0.1

        # Reduce for repetition
        sentences = text.split('.')
        unique_starts = len(set(s[:20].lower() for s in sentences if s.strip()))
        if len(sentences) > 2 and unique_starts < len(sentences) * 0.7:
            confidence -= 0.1

        # Increase for specific details
        if self.factual_regex.search(text):
            confidence += 0.05

        return max(0.1, min(1.0, confidence))

    async def forward_looking_retrieve(
        self,
        query: str,
        partial_synthesis: str,
        context: List[str] = None,
        retrieval_func = None
    ) -> Tuple[List[str], List[RetrievalPoint]]:
        """
        Perform forward-looking retrieval based on partial synthesis.

        Args:
            query: Original user query
            partial_synthesis: Current partial synthesis
            context: Existing context documents
            retrieval_func: Async function to retrieve documents

        Returns:
            Tuple of (additional_documents, retrieval_points)
        """
        context = context or []
        retrieval_points = []
        additional_docs = []

        # Detect uncertainty points in current synthesis
        uncertainties = self._detect_uncertainty_in_text(partial_synthesis)

        if not uncertainties:
            # Generate tentative continuation
            continuation_prompt = f"""Based on the question and partial answer, continue writing:

Question: {query}

Partial answer so far:
{partial_synthesis}

Continue the answer (next 2-3 sentences):"""

            tentative, _ = await self._generate_with_logprobs(
                continuation_prompt,
                max_tokens=self.tentative_tokens
            )

            # Check tentative content for uncertainty
            uncertainties = self._detect_uncertainty_in_text(tentative)

            if uncertainties:
                # Use tentative content as retrieval basis
                for pos, pattern, conf in uncertainties[:3]:
                    if len(retrieval_points) >= self.max_retrievals:
                        break

                    retrieval_query = await self._generate_retrieval_query(
                        query,
                        tentative,
                        RetrievalTrigger.LOW_CONFIDENCE
                    )

                    # Retrieve documents
                    if retrieval_func:
                        docs = await retrieval_func(retrieval_query)
                    else:
                        docs = await self._retrieve_documents(retrieval_query)

                    retrieval_points.append(RetrievalPoint(
                        position=len(partial_synthesis) + pos,
                        trigger=RetrievalTrigger.LOW_CONFIDENCE,
                        confidence=conf,
                        tentative_content=tentative,
                        query=retrieval_query,
                        retrieved_docs=docs
                    ))
                    additional_docs.extend(docs)
        else:
            # Handle uncertainties in existing synthesis
            for pos, pattern, conf in uncertainties[:self.max_retrievals]:
                # Extract context around uncertainty
                start = max(0, pos - 50)
                end = min(len(partial_synthesis), pos + 100)
                surrounding = partial_synthesis[start:end]

                trigger = (
                    RetrievalTrigger.UNCERTAIN_PHRASE
                    if 'might' in pattern.lower() or 'may' in pattern.lower()
                    else RetrievalTrigger.FACTUAL_CLAIM
                )

                retrieval_query = await self._generate_retrieval_query(
                    query,
                    surrounding,
                    trigger
                )

                if retrieval_func:
                    docs = await retrieval_func(retrieval_query)
                else:
                    docs = await self._retrieve_documents(retrieval_query)

                retrieval_points.append(RetrievalPoint(
                    position=pos,
                    trigger=trigger,
                    confidence=conf,
                    tentative_content=surrounding,
                    query=retrieval_query,
                    retrieved_docs=docs
                ))
                additional_docs.extend(docs)

        return additional_docs, retrieval_points

    async def generate_with_flare(
        self,
        query: str,
        context: List[str] = None,
        retrieval_func = None,
        max_length: int = 500
    ) -> FLAREResult:
        """
        Generate response with active FLARE retrieval.

        This iteratively generates content, checking for uncertainty
        and retrieving additional context as needed.

        Args:
            query: User query
            context: Initial context documents
            retrieval_func: Async function(query) -> List[str] for retrieval
            max_length: Maximum response length in words

        Returns:
            FLAREResult with final text and retrieval trace
        """
        context = context or []
        all_context = list(context)
        retrieval_points = []
        current_synthesis = ""
        iteration = 0
        max_iterations = 5

        while iteration < max_iterations:
            iteration += 1

            # Build prompt with current context
            context_text = "\n\n".join(all_context[-10:])  # Last 10 docs

            prompt = f"""Answer the following question using the provided context.
Be specific and detailed. If uncertain, acknowledge it.

Question: {query}

Context:
{context_text}

{"Current partial answer: " + current_synthesis if current_synthesis else ""}

{"Continue the answer:" if current_synthesis else "Answer:"}"""

            # Generate next chunk
            chunk, _ = await self._generate_with_logprobs(prompt, max_tokens=100)

            if not chunk:
                break

            # Check for uncertainty in generated chunk
            confidence = self._estimate_confidence_from_text(chunk)

            if confidence < self.confidence_threshold:
                # Forward-looking retrieval
                additional_docs, points = await self.forward_looking_retrieve(
                    query,
                    current_synthesis + " " + chunk,
                    all_context,
                    retrieval_func
                )

                retrieval_points.extend(points)
                all_context.extend(additional_docs)

                if additional_docs:
                    # Regenerate with new context
                    continue

            current_synthesis += (" " if current_synthesis else "") + chunk.strip()

            # Check if we've reached sufficient length
            word_count = len(current_synthesis.split())
            if word_count >= max_length:
                break

            # Check if response seems complete
            if chunk.strip().endswith(('.', '!', '?')) and word_count > 100:
                # Might be done - check for completeness
                if confidence > 0.7 and not self.uncertain_regex.search(chunk):
                    break

        # Calculate average confidence
        avg_confidence = (
            sum(rp.confidence for rp in retrieval_points) / len(retrieval_points)
            if retrieval_points
            else self._estimate_confidence_from_text(current_synthesis)
        )

        return FLAREResult(
            final_text=current_synthesis.strip(),
            retrieval_points=retrieval_points,
            total_retrievals=len(retrieval_points),
            average_confidence=avg_confidence,
            documents_used=all_context
        )

    async def should_retrieve_now(
        self,
        partial_synthesis: str,
        recent_tokens: str,
        iteration: int
    ) -> Tuple[bool, RetrievalTrigger, float]:
        """
        Decide if retrieval should be triggered at current point.

        Returns:
            Tuple of (should_retrieve, trigger_reason, confidence_estimate)
        """
        # Check for periodic retrieval
        if iteration > 0 and iteration % 3 == 0:
            return True, RetrievalTrigger.CONTINUATION, 0.6

        # Check for uncertainty in recent tokens
        confidence = self._estimate_confidence_from_text(recent_tokens)
        if confidence < self.confidence_threshold:
            return True, RetrievalTrigger.LOW_CONFIDENCE, confidence

        # Check for hedging language
        if self.uncertain_regex.search(recent_tokens):
            return True, RetrievalTrigger.UNCERTAIN_PHRASE, 0.5

        # Check for factual claims
        if self.factual_regex.search(recent_tokens):
            return True, RetrievalTrigger.FACTUAL_CLAIM, 0.6

        return False, RetrievalTrigger.LOW_CONFIDENCE, confidence


# Singleton instance
_flare_retriever: Optional[FLARERetriever] = None


def get_flare_retriever(
    ollama_url: str = "http://localhost:11434"
) -> FLARERetriever:
    """Get or create the FLARE retriever singleton."""
    global _flare_retriever
    if _flare_retriever is None:
        _flare_retriever = FLARERetriever(ollama_url=ollama_url)
    return _flare_retriever
