"""
Entropy-Based Confidence Signals Module

Based on UALA (ACL Findings 2024) research achieving >50% reduction in tool calls.

Key insight: Track generation entropy to detect confident vs uncertain outputs.
High entropy indicates the model is uncertain and may benefit from more retrieval.
Low entropy indicates confident generation where further iteration adds little value.

Thresholds (from research):
- High confidence: entropy < 0.2 → stop iterating
- Medium confidence: 0.2 <= entropy <= 0.5 → continue cautiously
- Low confidence: entropy > 0.5 → continue iterating

References:
- UALA: ACL Findings 2024 - Uncertainty-Aware LLM Agents
- arXiv:2310.13766 - Self-Evaluation Guided Decoding
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import httpx
import json
import re

logger = logging.getLogger(__name__)


class HaltDecision(str, Enum):
    """Decision on whether to halt iteration."""
    CONTINUE = "continue"              # Keep iterating, model uncertain
    HALT_CONFIDENT = "halt_confident"  # Stop, model is confident
    HALT_MAX_ITERATIONS = "halt_max"   # Stop, reached max iterations
    HALT_CONVERGENCE = "halt_converge" # Stop, entropy converged


@dataclass
class EntropyState:
    """Track entropy across iterations."""
    iteration: int
    entropy: float
    confidence: float  # 1 - normalized_entropy
    token_logprobs: Optional[List[float]] = None
    decision: HaltDecision = HaltDecision.CONTINUE
    trajectory: List[float] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        return self.entropy < 0.2

    @property
    def is_uncertain(self) -> bool:
        return self.entropy > 0.5


@dataclass
class EntropyResult:
    """Result of entropy analysis."""
    current_entropy: float
    confidence_score: float
    decision: HaltDecision
    iteration: int
    entropy_trajectory: List[float]
    reasoning: str
    convergence_detected: bool
    convergence_rate: float  # How fast entropy is changing


class EntropyMonitor:
    """
    Monitor generation entropy to detect confident vs uncertain outputs.

    UALA research shows that tracking entropy across iterations can:
    - Reduce unnecessary tool calls by >50%
    - Identify when the model has sufficient information
    - Detect when more retrieval would actually help

    The key insight is that low entropy (confident generation) means
    the model's probability mass is concentrated on few tokens,
    indicating it "knows" the answer. High entropy means diffuse
    probability, indicating uncertainty.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",  # Upgraded from gemma3:4b for better entropy estimation
        high_confidence_threshold: float = 0.2,
        low_confidence_threshold: float = 0.5,
        convergence_threshold: float = 0.05,  # Entropy change < 5% = converged
        min_iterations_before_halt: int = 1
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.convergence_threshold = convergence_threshold
        self.min_iterations_before_halt = min_iterations_before_halt
        self._entropy_history: Dict[str, List[float]] = {}

    def _calculate_entropy_from_logprobs(
        self,
        logprobs: List[float]
    ) -> float:
        """
        Calculate Shannon entropy from token log probabilities.

        H = -Σ p(x) * log(p(x))

        Since we have logprobs, p(x) = exp(logprob)
        H = -Σ exp(logprob) * logprob
        """
        if not logprobs:
            return 1.0  # Maximum uncertainty

        entropy = 0.0
        for logprob in logprobs:
            if logprob < -100:  # Skip effectively zero probabilities
                continue
            prob = math.exp(logprob)
            if prob > 0:
                entropy -= prob * logprob

        # Normalize to [0, 1] range
        # Max entropy for n tokens = log(n)
        max_entropy = math.log(len(logprobs)) if len(logprobs) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, max(0.0, normalized_entropy))

    async def _estimate_entropy_via_llm(
        self,
        query: str,
        synthesis: str,
        context: List[str]
    ) -> Tuple[float, str]:
        """
        Estimate entropy/uncertainty via LLM self-evaluation.

        Since Ollama doesn't always provide logprobs, we use a
        self-evaluation approach where the model assesses its own
        confidence in the answer.

        This follows the "self-evaluation guided decoding" pattern.
        """
        context_summary = "\n---\n".join([c[:500] for c in context[:5]])

        prompt = f"""Evaluate the confidence level of this synthesis.

QUERY: {query}

CONTEXT USED:
{context_summary if context_summary else "[No context available]"}

SYNTHESIS TO EVALUATE:
{synthesis[:1500]}

Rate the synthesis on these dimensions (each 0.0-1.0):
1. Information Completeness: How much of the query is answered?
2. Source Confidence: How reliable/authoritative are the sources?
3. Hedging Level: How much hedging/uncertainty language is used? (0=lots, 1=none)
4. Specificity: How specific vs vague is the answer?

Output ONLY valid JSON:
{{"completeness": <0.0-1.0>, "source_confidence": <0.0-1.0>, "hedging_inverse": <0.0-1.0>, "specificity": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""

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
                            "num_ctx": 4096
                        }
                    }
                )
                response.raise_for_status()
                result = response.json().get("response", "")

            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                completeness = float(data.get("completeness", 0.5))
                source_conf = float(data.get("source_confidence", 0.5))
                hedging_inv = float(data.get("hedging_inverse", 0.5))
                specificity = float(data.get("specificity", 0.5))
                reasoning = data.get("reasoning", "")

                # Confidence = weighted average
                confidence = (
                    0.35 * completeness +
                    0.25 * source_conf +
                    0.20 * hedging_inv +
                    0.20 * specificity
                )

                # Convert confidence to entropy (inverse relationship)
                # High confidence = low entropy
                entropy = 1.0 - confidence

                return (entropy, reasoning)

        except Exception as e:
            logger.warning(f"Entropy estimation failed: {e}")

        # Default: moderate uncertainty
        return (0.5, "Could not estimate entropy")

    def _detect_convergence(
        self,
        trajectory: List[float]
    ) -> Tuple[bool, float]:
        """
        Detect if entropy has converged (stopped changing significantly).

        Convergence indicates the model has reached a stable confidence
        level - either confidently knowing or confidently uncertain.
        """
        if len(trajectory) < 2:
            return (False, 0.0)

        # Calculate rate of change
        recent = trajectory[-3:] if len(trajectory) >= 3 else trajectory
        if len(recent) < 2:
            return (False, 0.0)

        changes = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        avg_change = sum(changes) / len(changes)

        # Converged if change rate below threshold
        converged = avg_change < self.convergence_threshold

        return (converged, avg_change)

    async def calculate_entropy(
        self,
        query: str,
        synthesis: str,
        context: List[str],
        iteration: int,
        max_iterations: int,
        session_id: Optional[str] = None,
        logprobs: Optional[List[float]] = None
    ) -> EntropyResult:
        """
        Calculate entropy and make halt decision.

        Args:
            query: Original user query
            synthesis: Current synthesis attempt
            context: Retrieved context used
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            session_id: Optional session ID for trajectory tracking
            logprobs: Optional token logprobs if available

        Returns:
            EntropyResult with decision and reasoning
        """
        import time
        start_time = time.time()

        # Track trajectory per session
        if session_id:
            if session_id not in self._entropy_history:
                self._entropy_history[session_id] = []
            trajectory = self._entropy_history[session_id]
        else:
            trajectory = []

        # Calculate entropy
        if logprobs:
            entropy = self._calculate_entropy_from_logprobs(logprobs)
            reasoning = "Calculated from token logprobs"
        else:
            entropy, reasoning = await self._estimate_entropy_via_llm(
                query, synthesis, context
            )

        # Update trajectory
        trajectory.append(entropy)

        # Detect convergence
        converged, convergence_rate = self._detect_convergence(trajectory)

        # Make halt decision
        if iteration >= max_iterations:
            decision = HaltDecision.HALT_MAX_ITERATIONS
        elif iteration < self.min_iterations_before_halt:
            decision = HaltDecision.CONTINUE
        elif entropy < self.high_confidence_threshold:
            decision = HaltDecision.HALT_CONFIDENT
        elif converged and entropy < self.low_confidence_threshold:
            decision = HaltDecision.HALT_CONVERGENCE
        elif entropy > self.low_confidence_threshold:
            decision = HaltDecision.CONTINUE
        else:
            # Medium entropy - check trajectory trend
            if len(trajectory) >= 2 and trajectory[-1] > trajectory[-2]:
                # Entropy increasing, continue to try to reduce
                decision = HaltDecision.CONTINUE
            else:
                # Entropy stable or decreasing, may halt soon
                decision = HaltDecision.CONTINUE

        confidence_score = 1.0 - entropy

        duration = (time.time() - start_time) * 1000
        logger.debug(
            f"Entropy analysis: {entropy:.3f} (conf={confidence_score:.3f}) "
            f"decision={decision.value} iter={iteration}/{max_iterations} "
            f"in {duration:.0f}ms"
        )

        return EntropyResult(
            current_entropy=entropy,
            confidence_score=confidence_score,
            decision=decision,
            iteration=iteration,
            entropy_trajectory=list(trajectory),
            reasoning=reasoning,
            convergence_detected=converged,
            convergence_rate=convergence_rate
        )

    def should_halt(
        self,
        entropy: float,
        iteration: int,
        max_iterations: int
    ) -> HaltDecision:
        """
        Simple halt decision based on entropy value.

        Use this for quick decisions without full analysis.
        """
        if iteration >= max_iterations:
            return HaltDecision.HALT_MAX_ITERATIONS

        if iteration < self.min_iterations_before_halt:
            return HaltDecision.CONTINUE

        if entropy < self.high_confidence_threshold:
            return HaltDecision.HALT_CONFIDENT

        if entropy > self.low_confidence_threshold:
            return HaltDecision.CONTINUE

        return HaltDecision.CONTINUE

    def clear_session(self, session_id: str) -> bool:
        """Clear entropy trajectory for a session."""
        if session_id in self._entropy_history:
            del self._entropy_history[session_id]
            return True
        return False

    def get_session_stats(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get entropy statistics for a session."""
        if session_id not in self._entropy_history:
            return None

        trajectory = self._entropy_history[session_id]
        if not trajectory:
            return None

        return {
            "session_id": session_id,
            "iterations": len(trajectory),
            "current_entropy": trajectory[-1],
            "min_entropy": min(trajectory),
            "max_entropy": max(trajectory),
            "avg_entropy": sum(trajectory) / len(trajectory),
            "trajectory": trajectory,
            "trend": "decreasing" if len(trajectory) >= 2 and trajectory[-1] < trajectory[0] else "stable_or_increasing"
        }


# Singleton instance
_entropy_monitor: Optional[EntropyMonitor] = None


def get_entropy_monitor(
    ollama_url: str = "http://localhost:11434"
) -> EntropyMonitor:
    """Get or create the entropy monitor singleton."""
    global _entropy_monitor
    if _entropy_monitor is None:
        _entropy_monitor = EntropyMonitor(ollama_url=ollama_url)
    return _entropy_monitor
