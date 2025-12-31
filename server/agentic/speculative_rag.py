"""
Speculative RAG - Parallel Draft Generation for Latency Reduction.

Part of G.5.1: Advanced RAG Techniques - 51% latency reduction via parallel drafts.

Based on Google Research "Speculative RAG" (July 2024):
- Smaller drafter model generates multiple answer drafts from document subsets
- Larger verifier model selects best draft by generation probability
- Parallel execution reduces wall-clock time significantly
- Document partitioning mitigates position bias

Key Benefits:
- 51% latency reduction compared to sequential RAG
- Position bias mitigation through diverse document subsets
- Quality maintained via verification step
- Configurable draft count and subset sizes

Research Basis:
- Speculative RAG (Google Research, July 2024)
- Speculative Decoding patterns from LLM inference optimization
- Position bias studies in long-context RAG

Usage:
    from agentic.speculative_rag import (
        SpeculativeRAG,
        SpeculativeRAGConfig,
        get_speculative_rag
    )

    rag = get_speculative_rag()
    result = await rag.generate(
        query="How to troubleshoot SRVO-063?",
        documents=[doc1, doc2, doc3, ...]
    )
"""

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import re

logger = logging.getLogger("agentic.speculative_rag")


class PartitionStrategy(str, Enum):
    """Strategy for partitioning documents into subsets."""
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    RELEVANCE_STRATIFIED = "relevance_stratified"  # Ensure each subset has high-relevance docs
    RANDOM = "random"  # Random shuffle and split
    DIVERSITY = "diversity"  # Maximize diversity within each subset


class SelectionMethod(str, Enum):
    """Method for selecting the best draft."""
    VERIFIER_SCORE = "verifier_score"  # LLM rates each draft
    SELF_CONSISTENCY = "self_consistency"  # Choose most common answer
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by draft confidence
    ENSEMBLE = "ensemble"  # Combine multiple selection methods


@dataclass
class SpeculativeRAGConfig:
    """Configuration for Speculative RAG."""
    # Model configuration
    drafter_model: str = "qwen2.5:7b"  # Smaller, faster model for drafts
    verifier_model: str = "deepseek-r1:14b"  # Larger model for verification
    ollama_url: str = "http://localhost:11434"

    # Draft generation
    num_drafts: int = 4  # Number of parallel drafts to generate
    docs_per_subset: int = 3  # Documents per subset (None = auto)
    partition_strategy: PartitionStrategy = PartitionStrategy.RELEVANCE_STRATIFIED

    # Generation parameters
    drafter_temperature: float = 0.7  # Higher for diversity
    drafter_max_tokens: int = 1024
    verifier_temperature: float = 0.3  # Lower for consistency
    verifier_max_tokens: int = 512

    # Selection
    selection_method: SelectionMethod = SelectionMethod.VERIFIER_SCORE
    min_draft_quality: float = 0.3  # Minimum quality to consider draft

    # Timeouts
    draft_timeout_seconds: int = 30
    verify_timeout_seconds: int = 45

    # Caching
    enable_draft_cache: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class Document:
    """Document for RAG retrieval."""
    id: str
    content: str
    title: str = ""
    url: str = ""
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)


@dataclass
class Draft:
    """A generated draft answer."""
    id: str
    content: str
    subset_ids: List[str]  # Document IDs used
    generation_time_ms: float
    token_count: int = 0
    confidence: float = 0.0
    verifier_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeculativeRAGResult:
    """Result from Speculative RAG generation."""
    query: str
    answer: str
    selected_draft_id: str
    all_drafts: List[Draft]
    documents_used: List[str]  # Document IDs
    total_time_ms: float
    draft_time_ms: float
    verify_time_ms: float
    selection_method: SelectionMethod
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "selected_draft_id": self.selected_draft_id,
            "drafts_count": len(self.all_drafts),
            "documents_used": self.documents_used,
            "total_time_ms": round(self.total_time_ms, 2),
            "draft_time_ms": round(self.draft_time_ms, 2),
            "verify_time_ms": round(self.verify_time_ms, 2),
            "selection_method": self.selection_method.value,
            "metadata": self.metadata,
        }


class SpeculativeRAG:
    """
    Speculative RAG with parallel draft generation.

    Implements the Speculative RAG pattern from Google Research:
    1. Partition documents into subsets
    2. Generate drafts in parallel from each subset
    3. Verify and select the best draft

    This achieves ~51% latency reduction compared to sequential processing.
    """

    def __init__(
        self,
        config: Optional[SpeculativeRAGConfig] = None,
    ):
        """
        Initialize Speculative RAG.

        Args:
            config: Configuration options
        """
        self.config = config or SpeculativeRAGConfig()

        # Statistics
        self._total_generations = 0
        self._total_drafts = 0
        self._cache_hits = 0
        self._avg_latency_ms = 0.0
        self._avg_speedup = 0.0

        # Draft cache
        self._draft_cache: Dict[str, Tuple[Draft, float]] = {}  # hash -> (draft, timestamp)

        logger.info(f"SpeculativeRAG initialized with {self.config.num_drafts} drafts")

    def _partition_documents(
        self,
        documents: List[Document],
        n_subsets: int
    ) -> List[List[Document]]:
        """
        Partition documents into subsets for parallel processing.

        Args:
            documents: All retrieved documents
            n_subsets: Number of subsets to create

        Returns:
            List of document subsets
        """
        if not documents:
            return [[] for _ in range(n_subsets)]

        strategy = self.config.partition_strategy

        if strategy == PartitionStrategy.ROUND_ROBIN:
            # Simple round-robin distribution
            subsets = [[] for _ in range(n_subsets)]
            for i, doc in enumerate(documents):
                subsets[i % n_subsets].append(doc)
            return subsets

        elif strategy == PartitionStrategy.RELEVANCE_STRATIFIED:
            # Sort by relevance, then distribute to ensure each subset has high-relevance docs
            sorted_docs = sorted(documents, key=lambda d: d.relevance_score, reverse=True)
            subsets = [[] for _ in range(n_subsets)]

            # Distribute in zigzag pattern: 0,1,2,3,3,2,1,0,0,1,2,3,...
            direction = 1
            idx = 0
            for doc in sorted_docs:
                subsets[idx].append(doc)
                idx += direction
                if idx >= n_subsets:
                    direction = -1
                    idx = n_subsets - 1
                elif idx < 0:
                    direction = 1
                    idx = 0

            return subsets

        elif strategy == PartitionStrategy.RANDOM:
            import random
            shuffled = documents.copy()
            random.shuffle(shuffled)
            subsets = [[] for _ in range(n_subsets)]
            for i, doc in enumerate(shuffled):
                subsets[i % n_subsets].append(doc)
            return subsets

        elif strategy == PartitionStrategy.DIVERSITY:
            # Try to maximize content diversity within each subset
            # Simple heuristic: use content length variance
            sorted_by_length = sorted(documents, key=lambda d: len(d.content))
            subsets = [[] for _ in range(n_subsets)]

            # Interleave short and long documents
            for i, doc in enumerate(sorted_by_length):
                if i % 2 == 0:
                    subsets[i % n_subsets].append(doc)
                else:
                    subsets[(n_subsets - 1) - (i % n_subsets)].append(doc)

            return subsets

        else:
            # Fallback to round-robin
            subsets = [[] for _ in range(n_subsets)]
            for i, doc in enumerate(documents):
                subsets[i % n_subsets].append(doc)
            return subsets

    def _build_draft_prompt(
        self,
        query: str,
        documents: List[Document],
        draft_index: int
    ) -> str:
        """Build prompt for draft generation."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.title or f"Document {i}"
            context_parts.append(f"[Source {i}: {title}]\n{doc.content}\n")

        context = "\n".join(context_parts)

        return f"""You are generating draft answer {draft_index + 1} for a question.
Use ONLY the provided sources to answer. Be concise and accurate.

Question: {query}

Sources:
{context}

Instructions:
1. Answer based ONLY on the provided sources
2. Be specific and include relevant details
3. If sources don't contain the answer, say so
4. Include source references where appropriate

Draft Answer:"""

    def _build_verification_prompt(
        self,
        query: str,
        drafts: List[Draft],
        documents: List[Document]
    ) -> str:
        """Build prompt for draft verification and selection."""
        # Compile all source content
        all_sources = "\n\n".join([
            f"[{doc.title or doc.id}]: {doc.content[:500]}..."
            for doc in documents[:6]  # Limit for context
        ])

        # Format drafts
        draft_texts = []
        for i, draft in enumerate(drafts, 1):
            draft_texts.append(f"DRAFT {i}:\n{draft.content}\n")

        drafts_section = "\n---\n".join(draft_texts)

        return f"""You are evaluating multiple draft answers to select the best one.

Question: {query}

Available Sources (for fact-checking):
{all_sources}

Draft Answers:
{drafts_section}

Evaluate each draft on:
1. Accuracy: Does it match the source information?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-organized and clear?
4. Relevance: Does it stay focused on the question?

Respond with ONLY a JSON object:
{{
    "best_draft": <1-{len(drafts)}>,
    "scores": {{
        "1": {{"accuracy": 0.0-1.0, "completeness": 0.0-1.0, "clarity": 0.0-1.0, "relevance": 0.0-1.0, "overall": 0.0-1.0}},
        "2": {{...}},
        ...
    }},
    "reasoning": "Brief explanation of why the selected draft is best"
}}"""

    async def _generate_single_draft(
        self,
        query: str,
        documents: List[Document],
        draft_index: int
    ) -> Optional[Draft]:
        """Generate a single draft from a document subset."""
        import httpx

        if not documents:
            return None

        # Check cache
        cache_key = self._compute_cache_key(query, documents)
        if self.config.enable_draft_cache and cache_key in self._draft_cache:
            cached_draft, cached_time = self._draft_cache[cache_key]
            if time.time() - cached_time < self.config.cache_ttl_seconds:
                self._cache_hits += 1
                logger.debug(f"Draft cache hit for subset {draft_index}")
                return cached_draft

        prompt = self._build_draft_prompt(query, documents, draft_index)

        start_time = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self.config.draft_timeout_seconds) as client:
                response = await client.post(
                    f"{self.config.ollama_url}/api/generate",
                    json={
                        "model": self.config.drafter_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.drafter_temperature,
                            "num_predict": self.config.drafter_max_tokens,
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()

            generation_time = (time.perf_counter() - start_time) * 1000

            draft = Draft(
                id=f"draft_{draft_index}_{int(time.time() * 1000)}",
                content=result.get("response", "").strip(),
                subset_ids=[doc.id for doc in documents],
                generation_time_ms=generation_time,
                token_count=result.get("eval_count", 0),
                confidence=0.5,  # Default, updated by verifier
                metadata={
                    "model": self.config.drafter_model,
                    "doc_count": len(documents),
                }
            )

            # Cache the draft
            if self.config.enable_draft_cache:
                self._draft_cache[cache_key] = (draft, time.time())

            self._total_drafts += 1
            return draft

        except Exception as e:
            logger.error(f"Draft generation failed for subset {draft_index}: {e}")
            return None

    async def _verify_and_select(
        self,
        query: str,
        drafts: List[Draft],
        documents: List[Document]
    ) -> Tuple[Draft, Dict[str, Any]]:
        """
        Verify drafts and select the best one.

        Args:
            query: Original query
            drafts: Generated drafts
            documents: All documents

        Returns:
            (selected_draft, verification_details)
        """
        import httpx

        if not drafts:
            raise ValueError("No drafts to verify")

        if len(drafts) == 1:
            drafts[0].verifier_score = 0.7  # Default score for single draft
            return drafts[0], {"method": "single_draft"}

        method = self.config.selection_method

        if method == SelectionMethod.VERIFIER_SCORE:
            # Use verifier LLM to score and select
            prompt = self._build_verification_prompt(query, drafts, documents)

            try:
                async with httpx.AsyncClient(timeout=self.config.verify_timeout_seconds) as client:
                    response = await client.post(
                        f"{self.config.ollama_url}/api/generate",
                        json={
                            "model": self.config.verifier_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": self.config.verifier_temperature,
                                "num_predict": self.config.verifier_max_tokens,
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()

                # Parse verification result
                response_text = result.get("response", "")
                verification = self._parse_verification_response(response_text, len(drafts))

                # Update draft scores
                for i, draft in enumerate(drafts):
                    score_key = str(i + 1)
                    if score_key in verification.get("scores", {}):
                        draft.verifier_score = verification["scores"][score_key].get("overall", 0.5)
                        draft.confidence = draft.verifier_score

                # Select best
                best_idx = verification.get("best_draft", 1) - 1
                best_idx = max(0, min(best_idx, len(drafts) - 1))

                return drafts[best_idx], verification

            except Exception as e:
                logger.warning(f"Verification failed, using confidence fallback: {e}")
                # Fallback to confidence-weighted
                method = SelectionMethod.CONFIDENCE_WEIGHTED

        if method == SelectionMethod.CONFIDENCE_WEIGHTED:
            # Select by generation confidence (token count as proxy)
            best_draft = max(drafts, key=lambda d: d.token_count)
            best_draft.verifier_score = 0.6
            return best_draft, {"method": "confidence_weighted"}

        if method == SelectionMethod.SELF_CONSISTENCY:
            # Simple heuristic: pick longest non-empty draft
            valid_drafts = [d for d in drafts if d.content and len(d.content) > 50]
            if valid_drafts:
                best_draft = max(valid_drafts, key=lambda d: len(d.content))
                best_draft.verifier_score = 0.65
                return best_draft, {"method": "self_consistency"}

        # Default: first draft
        drafts[0].verifier_score = 0.5
        return drafts[0], {"method": "fallback"}

    def _parse_verification_response(
        self,
        response: str,
        num_drafts: int
    ) -> Dict[str, Any]:
        """Parse the verification LLM response."""
        # Try to extract JSON
        try:
            # Look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: look for "DRAFT X" or "best: X" patterns
        best_match = re.search(r'(?:best|select|choose|pick)[:\s]*(?:draft\s*)?(\d+)', response.lower())
        if best_match:
            return {"best_draft": int(best_match.group(1)), "scores": {}}

        # Default to first draft
        return {"best_draft": 1, "scores": {}}

    def _compute_cache_key(self, query: str, documents: List[Document]) -> str:
        """Compute cache key for draft caching."""
        doc_ids = sorted([d.id for d in documents])
        content = f"{query}:{','.join(doc_ids)}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate(
        self,
        query: str,
        documents: List[Document],
        num_drafts: Optional[int] = None
    ) -> SpeculativeRAGResult:
        """
        Generate answer using Speculative RAG.

        Args:
            query: User query
            documents: Retrieved documents
            num_drafts: Override number of drafts

        Returns:
            SpeculativeRAGResult with selected answer and metadata
        """
        start_time = time.perf_counter()

        n_drafts = num_drafts or self.config.num_drafts

        # Partition documents
        subsets = self._partition_documents(documents, n_drafts)

        # Filter empty subsets
        non_empty_subsets = [(i, s) for i, s in enumerate(subsets) if s]

        if not non_empty_subsets:
            # No documents, generate without context
            return SpeculativeRAGResult(
                query=query,
                answer="I don't have enough information to answer this question.",
                selected_draft_id="none",
                all_drafts=[],
                documents_used=[],
                total_time_ms=0,
                draft_time_ms=0,
                verify_time_ms=0,
                selection_method=self.config.selection_method,
                metadata={"error": "no_documents"}
            )

        # Generate drafts in parallel
        draft_start = time.perf_counter()

        draft_tasks = [
            self._generate_single_draft(query, subset, idx)
            for idx, subset in non_empty_subsets
        ]

        draft_results = await asyncio.gather(*draft_tasks, return_exceptions=True)

        draft_time = (time.perf_counter() - draft_start) * 1000

        # Filter successful drafts
        drafts = []
        for result in draft_results:
            if isinstance(result, Draft) and result.content:
                drafts.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Draft generation exception: {result}")

        if not drafts:
            return SpeculativeRAGResult(
                query=query,
                answer="Failed to generate any valid drafts.",
                selected_draft_id="none",
                all_drafts=[],
                documents_used=[d.id for d in documents],
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                draft_time_ms=draft_time,
                verify_time_ms=0,
                selection_method=self.config.selection_method,
                metadata={"error": "all_drafts_failed"}
            )

        # Verify and select best draft
        verify_start = time.perf_counter()
        selected_draft, verification_details = await self._verify_and_select(
            query, drafts, documents
        )
        verify_time = (time.perf_counter() - verify_start) * 1000

        total_time = (time.perf_counter() - start_time) * 1000

        # Update statistics
        self._total_generations += 1
        self._avg_latency_ms = (
            (self._avg_latency_ms * (self._total_generations - 1) + total_time)
            / self._total_generations
        )

        # Estimate speedup vs sequential
        sequential_estimate = sum(d.generation_time_ms for d in drafts) + verify_time
        speedup = sequential_estimate / total_time if total_time > 0 else 1.0
        self._avg_speedup = (
            (self._avg_speedup * (self._total_generations - 1) + speedup)
            / self._total_generations
        )

        return SpeculativeRAGResult(
            query=query,
            answer=selected_draft.content,
            selected_draft_id=selected_draft.id,
            all_drafts=drafts,
            documents_used=list(set(
                doc_id for draft in drafts for doc_id in draft.subset_ids
            )),
            total_time_ms=total_time,
            draft_time_ms=draft_time,
            verify_time_ms=verify_time,
            selection_method=self.config.selection_method,
            metadata={
                "num_drafts_generated": len(drafts),
                "num_drafts_requested": n_drafts,
                "verification": verification_details,
                "speedup_estimate": round(speedup, 2),
                "selected_score": selected_draft.verifier_score,
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "total_generations": self._total_generations,
            "total_drafts": self._total_drafts,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._draft_cache),
            "avg_latency_ms": round(self._avg_latency_ms, 2),
            "avg_speedup": round(self._avg_speedup, 2),
            "config": {
                "drafter_model": self.config.drafter_model,
                "verifier_model": self.config.verifier_model,
                "num_drafts": self.config.num_drafts,
                "partition_strategy": self.config.partition_strategy.value,
                "selection_method": self.config.selection_method.value,
            }
        }

    def clear_cache(self) -> int:
        """Clear the draft cache. Returns number of entries cleared."""
        count = len(self._draft_cache)
        self._draft_cache.clear()
        return count


# Global instance
_speculative_rag: Optional[SpeculativeRAG] = None


def get_speculative_rag(
    config: Optional[SpeculativeRAGConfig] = None
) -> SpeculativeRAG:
    """Get or create global Speculative RAG instance."""
    global _speculative_rag
    if _speculative_rag is None:
        _speculative_rag = SpeculativeRAG(config)
    return _speculative_rag


async def speculative_generate(
    query: str,
    documents: List[Document],
    num_drafts: int = 4
) -> SpeculativeRAGResult:
    """Convenience function for speculative generation."""
    return await get_speculative_rag().generate(query, documents, num_drafts)
