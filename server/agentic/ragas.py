"""
RAGAS (Retrieval-Augmented Generation Assessment) Evaluation Pipeline.

RAGAS evaluates RAG pipeline quality without requiring ground truth labels.
Uses LLM-as-judge to assess:

1. **Faithfulness**: Are claims in response supported by retrieved context?
2. **Answer Relevancy**: Does the response actually answer the question?
3. **Context Relevancy**: Is retrieved context relevant to the question?
4. **Context Precision**: Is relevant context ranked higher than irrelevant?

Key Insight:
- Traditional evaluation requires expensive human labels
- RAGAS uses LLMs to extract claims and verify support
- Produces normalized 0-1 scores for each dimension

Research:
- Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (EMNLP 2024)
- arXiv:2309.15217

Benefits:
- Reference-free evaluation (no ground truth needed)
- Granular component-level assessment
- Composable with any RAG pipeline
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import httpx

from .llm_config import get_llm_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class EvaluationMetric(Enum):
    """RAGAS evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RELEVANCY = "context_relevancy"
    CONTEXT_PRECISION = "context_precision"
    OVERALL = "overall"


@dataclass
class RAGASInput:
    """Input for RAGAS evaluation."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None  # Optional for context recall


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""
    claim: str
    supported: bool
    supporting_context: Optional[str] = None
    confidence: float = 0.0


@dataclass
class RAGASResult:
    """Result of RAGAS evaluation."""
    faithfulness: float = 0.0         # 0-1: claims supported by context
    answer_relevancy: float = 0.0     # 0-1: answer addresses question
    context_relevancy: float = 0.0    # 0-1: context relevant to question
    context_precision: float = 0.0    # 0-1: relevant context ranked higher
    overall_score: float = 0.0        # Weighted average

    claims: List[str] = field(default_factory=list)
    claim_verifications: List[ClaimVerification] = field(default_factory=list)

    evaluation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": {
                "faithfulness": round(self.faithfulness, 4),
                "answer_relevancy": round(self.answer_relevancy, 4),
                "context_relevancy": round(self.context_relevancy, 4),
                "context_precision": round(self.context_precision, 4),
                "overall": round(self.overall_score, 4)
            },
            "claims": {
                "total": len(self.claims),
                "supported": sum(1 for cv in self.claim_verifications if cv.supported),
                "list": self.claims
            },
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
            "metadata": self.metadata
        }


@dataclass
class RAGASConfig:
    """Configuration for RAGAS evaluation."""
    # Metric weights for overall score
    faithfulness_weight: float = 0.30
    answer_relevancy_weight: float = 0.30
    context_relevancy_weight: float = 0.20
    context_precision_weight: float = 0.20

    # LLM settings
    temperature: float = 0.3  # Low temp for consistent evaluation
    max_claims: int = 10      # Max claims to extract per answer

    # Thresholds
    support_threshold: float = 0.7    # Min confidence for "supported"
    relevancy_threshold: float = 0.5  # Min for relevant context


# =============================================================================
# RAGAS Prompts
# =============================================================================

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer.
A claim is a specific statement that can be verified against source material.

Answer: {answer}

List each claim on a separate line. Only include verifiable factual claims.
Do not include opinions, questions, or vague statements.

Claims:"""

CLAIM_VERIFICATION_PROMPT = """Determine if the following claim is supported by the context.

Claim: {claim}

Context:
{context}

Is this claim supported by the context? Answer with:
- SUPPORTED: if the claim is clearly supported by the context
- NOT_SUPPORTED: if the claim is not mentioned or contradicted
- PARTIALLY: if only part of the claim is supported

Also rate your confidence (0.0 to 1.0).

Response format:
{{
    "verdict": "SUPPORTED|NOT_SUPPORTED|PARTIALLY",
    "confidence": 0.85,
    "explanation": "Brief explanation"
}}"""

ANSWER_RELEVANCY_PROMPT = """Rate how well the answer addresses the question.

Question: {question}

Answer: {answer}

Rate the answer relevancy from 0.0 to 1.0:
- 1.0: Answer completely and directly addresses the question
- 0.5: Answer partially addresses the question or is tangential
- 0.0: Answer is completely irrelevant to the question

Response format:
{{
    "score": 0.85,
    "explanation": "Brief explanation"
}}"""

CONTEXT_RELEVANCY_PROMPT = """Rate how relevant this context passage is to the question.

Question: {question}

Context: {context}

Rate the relevancy from 0.0 to 1.0:
- 1.0: Context directly helps answer the question
- 0.5: Context is somewhat related but not directly helpful
- 0.0: Context is completely irrelevant

Response format:
{{
    "score": 0.85,
    "explanation": "Brief explanation"
}}"""

QUESTION_GENERATION_PROMPT = """Generate a question that would be answered by this response.

Response: {answer}

Generate a clear, specific question that this response would answer:"""


# =============================================================================
# RAGAS Evaluator
# =============================================================================

class RAGASEvaluator:
    """
    RAGAS-style evaluation for RAG pipelines.

    Evaluates:
    - Faithfulness: Claim extraction + verification against context
    - Answer Relevancy: Semantic similarity of generated question
    - Context Relevancy: LLM rating of context relevance
    - Context Precision: Position-weighted relevancy
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        judge_model: Optional[str] = None,
        embedding_model: str = "bge-m3",
        config: Optional[RAGASConfig] = None
    ):
        llm_config = get_llm_config()
        self.ollama_url = ollama_url or llm_config.ollama.url
        self.judge_model = judge_model or llm_config.utility.ragas_judge.model
        self.embedding_model = embedding_model
        self.config = config or RAGASConfig()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Evaluation history
        self._history: List[RAGASResult] = []

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def _llm_generate(self, prompt: str) -> str:
        """Generate text using LLM."""
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.judge_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": 500
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return ""

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embedding"], dtype=np.float32)

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(1024, dtype=np.float32)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to find JSON in response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: extract key values
        result = {"score": 0.5, "verdict": "NOT_SUPPORTED", "confidence": 0.5}

        if "SUPPORTED" in text.upper():
            result["verdict"] = "SUPPORTED"
            result["confidence"] = 0.8
        elif "PARTIALLY" in text.upper():
            result["verdict"] = "PARTIALLY"
            result["confidence"] = 0.5

        # Try to extract numeric score
        score_match = re.search(r'(?:score|rating)[:\s]*([0-9.]+)', text.lower())
        if score_match:
            try:
                result["score"] = float(score_match.group(1))
            except ValueError:
                pass

        return result

    async def extract_claims(self, answer: str) -> List[str]:
        """Extract verifiable claims from an answer."""
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        response = await self._llm_generate(prompt)

        # Parse claims from response
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            # Remove numbering and bullets
            line = re.sub(r'^[\d\-\*\.\)]+\s*', '', line)
            if line and len(line) > 10:  # Filter very short lines
                claims.append(line)

        return claims[:self.config.max_claims]

    async def verify_claim(
        self,
        claim: str,
        contexts: List[str]
    ) -> ClaimVerification:
        """Verify if a claim is supported by contexts."""
        combined_context = "\n\n".join(contexts)
        prompt = CLAIM_VERIFICATION_PROMPT.format(
            claim=claim,
            context=combined_context
        )

        response = await self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        verdict = parsed.get("verdict", "NOT_SUPPORTED")
        confidence = float(parsed.get("confidence", 0.5))

        supported = verdict == "SUPPORTED" or (
            verdict == "PARTIALLY" and confidence >= self.config.support_threshold
        )

        return ClaimVerification(
            claim=claim,
            supported=supported,
            supporting_context=combined_context[:200] if supported else None,
            confidence=confidence
        )

    async def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> Tuple[float, List[str], List[ClaimVerification]]:
        """
        Evaluate faithfulness by extracting and verifying claims.

        Faithfulness = (# supported claims) / (# total claims)
        """
        claims = await self.extract_claims(answer)

        if not claims:
            return 1.0, [], []  # No claims = fully faithful (vacuously true)

        verifications = []
        for claim in claims:
            verification = await self.verify_claim(claim, contexts)
            verifications.append(verification)

        supported_count = sum(1 for v in verifications if v.supported)
        faithfulness = supported_count / len(claims)

        return faithfulness, claims, verifications

    async def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate answer relevancy using question regeneration.

        Method 1: Direct rating
        Method 2: Generate question from answer, measure semantic similarity
        """
        # Direct rating approach (faster)
        prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        response = await self._llm_generate(prompt)
        parsed = self._parse_json_response(response)

        direct_score = float(parsed.get("score", 0.5))

        # Question regeneration approach (more robust)
        gen_prompt = QUESTION_GENERATION_PROMPT.format(answer=answer)
        generated_question = await self._llm_generate(gen_prompt)

        if generated_question:
            # Compute semantic similarity
            orig_emb = await self._get_embedding(question)
            gen_emb = await self._get_embedding(generated_question)

            similarity = float(
                np.dot(orig_emb, gen_emb) /
                (np.linalg.norm(orig_emb) * np.linalg.norm(gen_emb) + 1e-8)
            )

            # Combine both approaches
            return 0.5 * direct_score + 0.5 * max(0, similarity)

        return direct_score

    async def evaluate_context_relevancy(
        self,
        question: str,
        contexts: List[str]
    ) -> Tuple[float, List[float]]:
        """
        Evaluate context relevancy for each retrieved context.

        Returns average relevancy and per-context scores.
        """
        if not contexts:
            return 0.0, []

        scores = []
        for context in contexts:
            prompt = CONTEXT_RELEVANCY_PROMPT.format(
                question=question,
                context=context[:500]  # Truncate for efficiency
            )
            response = await self._llm_generate(prompt)
            parsed = self._parse_json_response(response)
            scores.append(float(parsed.get("score", 0.0)))

        avg_score = sum(scores) / len(scores)
        return avg_score, scores

    async def evaluate_context_precision(
        self,
        question: str,
        contexts: List[str]
    ) -> float:
        """
        Evaluate context precision (relevant contexts ranked higher).

        Uses position-weighted scoring: contexts early in list weighted more.
        """
        _, relevancy_scores = await self.evaluate_context_relevancy(question, contexts)

        if not relevancy_scores:
            return 0.0

        # Binary relevancy (above threshold = relevant)
        binary = [1 if s >= self.config.relevancy_threshold else 0 for s in relevancy_scores]

        # Average Precision @ K
        precision_at_k = []
        relevant_count = 0

        for k, is_relevant in enumerate(binary, 1):
            relevant_count += is_relevant
            if is_relevant:
                precision_at_k.append(relevant_count / k)

        if not precision_at_k:
            return 0.0

        return sum(precision_at_k) / len(precision_at_k)

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        metrics: Optional[List[EvaluationMetric]] = None
    ) -> RAGASResult:
        """
        Perform full RAGAS evaluation.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: Retrieved context passages
            metrics: Specific metrics to evaluate (default: all)
        """
        start_time = time.time()

        result = RAGASResult()
        metrics = metrics or [
            EvaluationMetric.FAITHFULNESS,
            EvaluationMetric.ANSWER_RELEVANCY,
            EvaluationMetric.CONTEXT_RELEVANCY,
            EvaluationMetric.CONTEXT_PRECISION
        ]

        # Evaluate each metric
        if EvaluationMetric.FAITHFULNESS in metrics:
            faithfulness, claims, verifications = await self.evaluate_faithfulness(
                answer, contexts
            )
            result.faithfulness = faithfulness
            result.claims = claims
            result.claim_verifications = verifications

        if EvaluationMetric.ANSWER_RELEVANCY in metrics:
            result.answer_relevancy = await self.evaluate_answer_relevancy(
                question, answer
            )

        if EvaluationMetric.CONTEXT_RELEVANCY in metrics:
            result.context_relevancy, _ = await self.evaluate_context_relevancy(
                question, contexts
            )

        if EvaluationMetric.CONTEXT_PRECISION in metrics:
            result.context_precision = await self.evaluate_context_precision(
                question, contexts
            )

        # Compute overall score
        result.overall_score = (
            self.config.faithfulness_weight * result.faithfulness +
            self.config.answer_relevancy_weight * result.answer_relevancy +
            self.config.context_relevancy_weight * result.context_relevancy +
            self.config.context_precision_weight * result.context_precision
        )

        result.evaluation_time_ms = (time.time() - start_time) * 1000
        result.metadata = {
            "judge_model": self.judge_model,
            "metrics_evaluated": [m.value for m in metrics],
            "num_contexts": len(contexts)
        }

        # Store in history
        self._history.append(result)

        logger.info(
            f"RAGAS evaluation: faith={result.faithfulness:.2f}, "
            f"ans_rel={result.answer_relevancy:.2f}, "
            f"ctx_rel={result.context_relevancy:.2f}, "
            f"overall={result.overall_score:.2f}"
        )

        return result

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all evaluations."""
        if not self._history:
            return {"evaluations": 0}

        return {
            "evaluations": len(self._history),
            "average_scores": {
                "faithfulness": round(
                    sum(r.faithfulness for r in self._history) / len(self._history), 4
                ),
                "answer_relevancy": round(
                    sum(r.answer_relevancy for r in self._history) / len(self._history), 4
                ),
                "context_relevancy": round(
                    sum(r.context_relevancy for r in self._history) / len(self._history), 4
                ),
                "context_precision": round(
                    sum(r.context_precision for r in self._history) / len(self._history), 4
                ),
                "overall": round(
                    sum(r.overall_score for r in self._history) / len(self._history), 4
                )
            },
            "average_evaluation_time_ms": round(
                sum(r.evaluation_time_ms for r in self._history) / len(self._history), 2
            )
        }

    def clear_history(self):
        """Clear evaluation history."""
        self._history.clear()

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Singleton and Factory
# =============================================================================

_ragas_evaluator: Optional[RAGASEvaluator] = None


def get_ragas_evaluator(
    ollama_url: Optional[str] = None,
    judge_model: Optional[str] = None,
    embedding_model: str = "bge-m3"
) -> RAGASEvaluator:
    """Get or create the global RAGAS evaluator instance."""
    global _ragas_evaluator

    if _ragas_evaluator is None:
        _ragas_evaluator = RAGASEvaluator(
            ollama_url=ollama_url,
            judge_model=judge_model,
            embedding_model=embedding_model
        )

    return _ragas_evaluator


async def create_ragas_evaluator(
    ollama_url: Optional[str] = None,
    judge_model: Optional[str] = None,
    embedding_model: str = "bge-m3",
    config: Optional[RAGASConfig] = None
) -> RAGASEvaluator:
    """Create a new RAGAS evaluator instance."""
    return RAGASEvaluator(
        ollama_url=ollama_url,
        judge_model=judge_model,
        embedding_model=embedding_model,
        config=config
    )
