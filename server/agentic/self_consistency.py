"""
Self-Consistency Convergence Module

Based on CISC (Google, arXiv 2502.20233) research achieving >40% sample reduction.

Key insight: Multiple reasoning paths that converge on the same answer indicate
higher confidence than a single path. Use weighted majority voting with
semantic similarity to determine when sufficient agreement has been reached.

Threshold (from research):
- >60% agreement = sufficient confidence to stop sampling
- <60% agreement = continue sampling or flag as low-confidence

References:
- CISC: arXiv 2502.20233 - Confidence-Informed Self-Consistency
- Self-Consistency (Wang et al.): Chain-of-thought prompting
- USC: Universal Self-Consistency for answer extraction
"""

import asyncio
import logging
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import httpx
import json
import re
import numpy as np

logger = logging.getLogger(__name__)


class ConvergenceStatus(str, Enum):
    """Status of convergence check."""
    CONVERGED = "converged"          # Sufficient agreement reached
    PARTIAL = "partial"              # Some agreement, may need more samples
    DIVERGENT = "divergent"          # Answers disagree significantly
    INSUFFICIENT = "insufficient"    # Not enough samples to determine


@dataclass
class SynthesisAttempt:
    """A single synthesis attempt with metadata."""
    content: str
    attempt_id: str
    confidence: float = 0.5
    key_facts: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class AnswerCluster:
    """Cluster of semantically similar answers."""
    representative: SynthesisAttempt
    members: List[SynthesisAttempt]
    centroid_embedding: Optional[List[float]] = None
    cluster_confidence: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def weight(self) -> float:
        """Confidence-weighted cluster size."""
        return sum(m.confidence for m in self.members)


@dataclass
class ConvergenceResult:
    """Result of self-consistency check."""
    status: ConvergenceStatus
    agreement_ratio: float  # 0-1, fraction agreeing with majority
    majority_answer: Optional[str]
    majority_confidence: float
    num_samples: int
    num_clusters: int
    clusters: List[AnswerCluster]
    key_facts_agreement: Dict[str, float]  # Fact -> agreement ratio
    reasoning: str


class SelfConsistencyChecker:
    """
    Weighted majority voting across multiple reasoning paths.

    CISC research shows that:
    - High agreement (>60%) indicates confident, reliable answers
    - Low agreement indicates ambiguity or insufficient information
    - Confidence-weighted voting outperforms simple majority

    The key insight is that if multiple independent reasoning paths
    arrive at the same conclusion, that conclusion is more likely correct.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "gemma3:4b",
        embedding_model: str = "mxbai-embed-large",
        min_agreement: float = 0.6,
        similarity_threshold: float = 0.85,
        min_samples: int = 3,
        max_samples: int = 7
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.embedding_model = embedding_model
        self.min_agreement = min_agreement
        self.similarity_threshold = similarity_threshold
        self.min_samples = min_samples
        self.max_samples = max_samples
        self._embedding_cache: Dict[str, List[float]] = {}

    def _hash_content(self, content: str) -> str:
        """Generate content hash for caching."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text with caching."""
        cache_key = self._hash_content(text[:1000])

        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        text_truncated = text[:1000] if len(text) > 1000 else text

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text_truncated
                }
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])

        self._embedding_cache[cache_key] = embedding
        return embedding

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    async def _extract_key_facts(self, synthesis: str) -> List[str]:
        """Extract key factual claims from a synthesis."""
        prompt = f"""Extract the key factual claims from this text.
Return only verifiable facts, not opinions or hedged statements.
Output as JSON array of strings.

TEXT:
{synthesis[:1500]}

Output ONLY valid JSON array:
["fact 1", "fact 2", ...]"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_ctx": 4096}
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")

            # Parse JSON array
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group())
                return [f for f in facts if isinstance(f, str)][:10]

        except Exception as e:
            logger.warning(f"Key fact extraction failed: {e}")

        return []

    async def _cluster_answers(
        self,
        attempts: List[SynthesisAttempt]
    ) -> List[AnswerCluster]:
        """
        Cluster answers by semantic similarity.

        Uses agglomerative clustering with single-linkage:
        answers with >85% similarity are grouped together.
        """
        if not attempts:
            return []

        # Get embeddings for all attempts
        for attempt in attempts:
            if attempt.embedding is None:
                attempt.embedding = await self._get_embedding(attempt.content)

        # Build similarity matrix
        n = len(attempts)
        similarity_matrix = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(
                    attempts[i].embedding,
                    attempts[j].embedding
                )
                if sim >= self.similarity_threshold:
                    similarity_matrix[(i, j)] = sim

        # Agglomerative clustering
        clusters: List[Set[int]] = [{i} for i in range(n)]

        changed = True
        while changed:
            changed = False
            new_clusters = []
            merged = set()

            for i, cluster_i in enumerate(clusters):
                if i in merged:
                    continue

                merged_cluster = set(cluster_i)

                for j, cluster_j in enumerate(clusters):
                    if j <= i or j in merged:
                        continue

                    should_merge = any(
                        (min(a, b), max(a, b)) in similarity_matrix
                        for a in cluster_i for b in cluster_j
                    )

                    if should_merge:
                        merged_cluster.update(cluster_j)
                        merged.add(j)
                        changed = True

                new_clusters.append(merged_cluster)
                merged.add(i)

            clusters = new_clusters

        # Build AnswerCluster objects
        result_clusters = []
        for indices in clusters:
            members = [attempts[i] for i in sorted(indices)]

            # Representative: highest confidence member
            representative = max(members, key=lambda m: m.confidence)

            # Calculate centroid
            embeddings = [m.embedding for m in members if m.embedding]
            centroid = list(np.mean(embeddings, axis=0)) if embeddings else None

            # Cluster confidence: weighted average
            total_weight = sum(m.confidence for m in members)
            cluster_conf = total_weight / len(members) if members else 0.0

            cluster = AnswerCluster(
                representative=representative,
                members=members,
                centroid_embedding=centroid,
                cluster_confidence=cluster_conf
            )
            result_clusters.append(cluster)

        # Sort by size (largest first)
        result_clusters.sort(key=lambda c: c.weight, reverse=True)

        return result_clusters

    def _calculate_key_fact_agreement(
        self,
        attempts: List[SynthesisAttempt]
    ) -> Dict[str, float]:
        """
        Calculate agreement ratio for each key fact across attempts.

        Fact normalization: lowercase, strip whitespace
        """
        all_facts: Dict[str, int] = {}

        for attempt in attempts:
            for fact in attempt.key_facts:
                normalized = fact.lower().strip()
                if len(normalized) > 10:  # Skip very short facts
                    all_facts[normalized] = all_facts.get(normalized, 0) + 1

        # Calculate agreement ratios
        total_attempts = len(attempts)
        agreement = {
            fact: count / total_attempts
            for fact, count in all_facts.items()
        }

        # Sort by agreement (highest first)
        return dict(sorted(
            agreement.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    async def check_convergence(
        self,
        query: str,
        synthesis_attempts: List[str],
        attempt_confidences: Optional[List[float]] = None
    ) -> ConvergenceResult:
        """
        Check if multiple synthesis attempts converge on the same answer.

        Args:
            query: Original user query
            synthesis_attempts: List of synthesis attempts to compare
            attempt_confidences: Optional confidence scores for each attempt

        Returns:
            ConvergenceResult with agreement analysis
        """
        import time
        start_time = time.time()

        if len(synthesis_attempts) < self.min_samples:
            return ConvergenceResult(
                status=ConvergenceStatus.INSUFFICIENT,
                agreement_ratio=0.0,
                majority_answer=None,
                majority_confidence=0.0,
                num_samples=len(synthesis_attempts),
                num_clusters=0,
                clusters=[],
                key_facts_agreement={},
                reasoning=f"Need at least {self.min_samples} samples, got {len(synthesis_attempts)}"
            )

        # Create SynthesisAttempt objects
        if attempt_confidences is None:
            attempt_confidences = [0.5] * len(synthesis_attempts)

        attempts = []
        for i, (content, conf) in enumerate(zip(synthesis_attempts, attempt_confidences)):
            attempt = SynthesisAttempt(
                content=content,
                attempt_id=f"attempt_{i}",
                confidence=conf
            )
            attempts.append(attempt)

        # Extract key facts from each attempt (parallel)
        logger.debug(f"Extracting key facts from {len(attempts)} attempts...")
        fact_tasks = [self._extract_key_facts(a.content) for a in attempts]
        all_facts = await asyncio.gather(*fact_tasks)

        for attempt, facts in zip(attempts, all_facts):
            attempt.key_facts = facts

        # Cluster answers by semantic similarity
        logger.debug("Clustering answers by similarity...")
        clusters = await self._cluster_answers(attempts)

        if not clusters:
            return ConvergenceResult(
                status=ConvergenceStatus.INSUFFICIENT,
                agreement_ratio=0.0,
                majority_answer=None,
                majority_confidence=0.0,
                num_samples=len(attempts),
                num_clusters=0,
                clusters=[],
                key_facts_agreement={},
                reasoning="Could not cluster answers"
            )

        # Calculate agreement ratio
        largest_cluster = clusters[0]
        total_weight = sum(c.weight for c in clusters)
        agreement_ratio = largest_cluster.weight / total_weight if total_weight > 0 else 0.0

        # Calculate key fact agreement
        key_facts_agreement = self._calculate_key_fact_agreement(attempts)

        # Determine convergence status
        if agreement_ratio >= self.min_agreement:
            status = ConvergenceStatus.CONVERGED
            reasoning = f"High agreement ({agreement_ratio:.1%}) with {largest_cluster.size}/{len(attempts)} answers in majority cluster"
        elif agreement_ratio >= 0.4:
            status = ConvergenceStatus.PARTIAL
            reasoning = f"Partial agreement ({agreement_ratio:.1%}), may benefit from more samples"
        else:
            status = ConvergenceStatus.DIVERGENT
            reasoning = f"Low agreement ({agreement_ratio:.1%}) across {len(clusters)} distinct answer clusters"

        # Majority answer and confidence
        majority_answer = largest_cluster.representative.content
        majority_confidence = (
            agreement_ratio * largest_cluster.cluster_confidence
        )

        duration = (time.time() - start_time) * 1000
        logger.info(
            f"Self-consistency check: {status.value} "
            f"(agreement={agreement_ratio:.1%}, clusters={len(clusters)}) "
            f"in {duration:.0f}ms"
        )

        return ConvergenceResult(
            status=status,
            agreement_ratio=agreement_ratio,
            majority_answer=majority_answer,
            majority_confidence=majority_confidence,
            num_samples=len(attempts),
            num_clusters=len(clusters),
            clusters=clusters,
            key_facts_agreement=key_facts_agreement,
            reasoning=reasoning
        )

    async def generate_and_check(
        self,
        query: str,
        context: List[str],
        synthesis_prompt: str,
        num_samples: int = 5,
        temperature_variance: bool = True
    ) -> ConvergenceResult:
        """
        Generate multiple synthesis attempts and check convergence.

        This combines generation and consistency checking in one call.

        Args:
            query: Original query
            context: Retrieved context
            synthesis_prompt: Prompt template for synthesis
            num_samples: Number of samples to generate
            temperature_variance: Whether to vary temperature across samples

        Returns:
            ConvergenceResult with best answer if converged
        """
        num_samples = min(num_samples, self.max_samples)

        # Generate multiple attempts with varying temperatures
        temperatures = (
            [0.3, 0.5, 0.7, 0.5, 0.6, 0.4, 0.8][:num_samples]
            if temperature_variance
            else [0.5] * num_samples
        )

        async def generate_one(temp: float, idx: int) -> str:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": synthesis_prompt,
                            "stream": False,
                            "options": {
                                "temperature": temp,
                                "num_ctx": 8192
                            }
                        }
                    )
                    response.raise_for_status()
                    return response.json().get("response", "")
            except Exception as e:
                logger.warning(f"Generation attempt {idx} failed: {e}")
                return ""

        logger.info(f"Generating {num_samples} synthesis samples...")
        tasks = [generate_one(t, i) for i, t in enumerate(temperatures)]
        results = await asyncio.gather(*tasks)

        # Filter out failed attempts
        valid_attempts = [r for r in results if r and len(r) > 50]

        if len(valid_attempts) < self.min_samples:
            return ConvergenceResult(
                status=ConvergenceStatus.INSUFFICIENT,
                agreement_ratio=0.0,
                majority_answer=valid_attempts[0] if valid_attempts else None,
                majority_confidence=0.0,
                num_samples=len(valid_attempts),
                num_clusters=0,
                clusters=[],
                key_facts_agreement={},
                reasoning=f"Only {len(valid_attempts)} valid samples generated"
            )

        return await self.check_convergence(query, valid_attempts)

    def clear_cache(self) -> int:
        """Clear embedding cache."""
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        return count


# Singleton instance
_consistency_checker: Optional[SelfConsistencyChecker] = None


def get_consistency_checker(
    ollama_url: str = "http://localhost:11434"
) -> SelfConsistencyChecker:
    """Get or create the self-consistency checker singleton."""
    global _consistency_checker
    if _consistency_checker is None:
        _consistency_checker = SelfConsistencyChecker(ollama_url=ollama_url)
    return _consistency_checker
