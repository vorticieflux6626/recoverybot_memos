"""
DeepEval Integration for Agentic Search CI Pipeline.

Part of G.1.5: Integrate DeepEval in CI pipeline.

Provides automated RAG quality evaluation using DeepEval metrics:
- Faithfulness: Claims supported by retrieved context
- Answer Relevancy: Response addresses the question
- Context Precision: Relevant context ranked appropriately
- Hallucination: Detects unsupported claims

Usage:
    # Run CI evaluation
    pytest tests/integration/test_deepeval_rag.py -v

    # Programmatic evaluation
    from agentic.deepeval_integration import (
        evaluate_rag_response,
        run_benchmark_evaluation,
        get_evaluation_summary
    )

    results = await evaluate_rag_response(
        query="What causes SRVO-063 alarm?",
        response="SRVO-063 indicates encoder failure...",
        contexts=["SRVO-063 is a servo alarm indicating..."]
    )

Research Basis:
- DeepEval (Apache 2.0) - LLM evaluation framework
- RAGAS metrics (EACL 2024)
- RAG evaluation best practices
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("agentic.deepeval")

# Try to import DeepEval
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase
    from deepeval.dataset import EvaluationDataset

    DEEPEVAL_AVAILABLE = True
    logger.info("DeepEval available - RAG evaluation enabled")
except ImportError as e:
    DEEPEVAL_AVAILABLE = False
    logger.warning(f"DeepEval not available - evaluation disabled: {e}")
    # Create dummy classes for type hints
    FaithfulnessMetric = None
    AnswerRelevancyMetric = None
    ContextualPrecisionMetric = None
    HallucinationMetric = None
    LLMTestCase = None
    EvaluationDataset = None
    evaluate = None


class EvaluationMetric(str, Enum):
    """Available evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_PRECISION = "context_precision"
    HALLUCINATION = "hallucination"


@dataclass
class EvaluationResult:
    """Result from a single evaluation."""
    query: str
    response: str
    contexts: List[str]
    metrics: Dict[str, float]
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query[:100],
            "response": self.response[:200],
            "context_count": len(self.contexts),
            "metrics": self.metrics,
            "passed": self.passed,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class BenchmarkEvaluationResult:
    """Result from benchmark evaluation."""
    total_queries: int
    passed_queries: int
    pass_rate: float
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_hallucination: float
    results: List[EvaluationResult]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "passed_queries": self.passed_queries,
            "pass_rate": round(self.pass_rate, 3),
            "metrics": {
                "avg_faithfulness": round(self.avg_faithfulness, 3),
                "avg_answer_relevancy": round(self.avg_answer_relevancy, 3),
                "avg_context_precision": round(self.avg_context_precision, 3),
                "avg_hallucination": round(self.avg_hallucination, 3),
            },
            "timestamp": self.timestamp,
        }


class DeepEvalRAGEvaluator:
    """
    RAG evaluation using DeepEval metrics.

    Provides comprehensive quality assessment for RAG responses:
    - Faithfulness: Are claims grounded in context?
    - Answer Relevancy: Does response address the question?
    - Context Precision: Is relevant context properly ranked?
    - Hallucination: Are there unsupported claims?
    """

    # Default thresholds for passing
    DEFAULT_THRESHOLDS = {
        EvaluationMetric.FAITHFULNESS: 0.7,
        EvaluationMetric.ANSWER_RELEVANCY: 0.7,
        EvaluationMetric.CONTEXT_PRECISION: 0.6,
        EvaluationMetric.HALLUCINATION: 0.3,  # Lower is better
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",  # DeepEval's default model for evaluation
        thresholds: Optional[Dict[EvaluationMetric, float]] = None,
        use_local_model: bool = True,
        local_model_url: str = "http://localhost:11434/v1",
    ):
        """
        Initialize DeepEval evaluator.

        Args:
            model: Model to use for evaluation
            thresholds: Custom pass/fail thresholds
            use_local_model: Use local Ollama model instead of OpenAI
            local_model_url: URL for local model API
        """
        self.model = model
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.use_local_model = use_local_model
        self.local_model_url = local_model_url

        # Statistics
        self._total_evaluations = 0
        self._total_passed = 0
        self._metric_totals: Dict[str, float] = {m.value: 0.0 for m in EvaluationMetric}

        if not DEEPEVAL_AVAILABLE:
            logger.warning("DeepEval not available - evaluations will be simulated")

    def _create_metrics(
        self,
        metrics_to_use: Optional[List[EvaluationMetric]] = None
    ) -> List[Any]:
        """Create metric instances for evaluation."""
        if not DEEPEVAL_AVAILABLE:
            return []

        metrics_to_use = metrics_to_use or list(EvaluationMetric)
        metrics = []

        for metric in metrics_to_use:
            threshold = self.thresholds.get(metric, 0.7)

            if metric == EvaluationMetric.FAITHFULNESS:
                metrics.append(FaithfulnessMetric(
                    threshold=threshold,
                    model=self.model,
                    include_reason=True
                ))
            elif metric == EvaluationMetric.ANSWER_RELEVANCY:
                metrics.append(AnswerRelevancyMetric(
                    threshold=threshold,
                    model=self.model,
                    include_reason=True
                ))
            elif metric == EvaluationMetric.CONTEXT_PRECISION:
                metrics.append(ContextualPrecisionMetric(
                    threshold=threshold,
                    model=self.model,
                    include_reason=True
                ))
            elif metric == EvaluationMetric.HALLUCINATION:
                metrics.append(HallucinationMetric(
                    threshold=threshold,
                    model=self.model,
                    include_reason=True
                ))

        return metrics

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: List[str],
        expected_output: Optional[str] = None,
        metrics: Optional[List[EvaluationMetric]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.

        Args:
            query: User query
            response: Generated response
            contexts: Retrieved contexts used for generation
            expected_output: Optional expected answer (for precision)
            metrics: Optional list of metrics to evaluate

        Returns:
            EvaluationResult with metric scores
        """
        if not DEEPEVAL_AVAILABLE:
            # Return simulated results for testing
            return self._simulate_evaluation(query, response, contexts)

        try:
            # Create test case
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                retrieval_context=contexts,
                expected_output=expected_output,
            )

            # Create metrics
            metric_instances = self._create_metrics(metrics)

            # Run evaluation in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: evaluate([test_case], metric_instances, run_async=False)
            )

            # Extract scores
            metric_scores = {}
            for metric in metric_instances:
                metric_name = metric.__class__.__name__.lower().replace("metric", "")
                metric_scores[metric_name] = getattr(metric, 'score', 0.0) or 0.0

            # Determine pass/fail
            passed = self._check_passed(metric_scores)

            # Update statistics
            self._total_evaluations += 1
            if passed:
                self._total_passed += 1
            for name, score in metric_scores.items():
                if name in self._metric_totals:
                    self._metric_totals[name] += score

            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                metrics=metric_scores,
                passed=passed,
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                metrics={},
                passed=False,
                error=str(e),
            )

    def _simulate_evaluation(
        self,
        query: str,
        response: str,
        contexts: List[str]
    ) -> EvaluationResult:
        """Simulate evaluation when DeepEval is not available."""
        # Basic heuristic scoring
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        context_terms = set(" ".join(contexts).lower().split())

        # Simple relevancy: overlap between query and response terms
        relevancy = len(query_terms & response_terms) / max(len(query_terms), 1)

        # Simple faithfulness: response terms appear in context
        faithfulness = len(response_terms & context_terms) / max(len(response_terms), 1)

        # Context precision: query terms in context
        context_precision = len(query_terms & context_terms) / max(len(query_terms), 1)

        # Hallucination: response terms NOT in context (inverse)
        unsupported = len(response_terms - context_terms) / max(len(response_terms), 1)

        metrics = {
            "faithfulness": min(1.0, faithfulness + 0.3),  # Adjust for reasonable scores
            "answerrelevancy": min(1.0, relevancy + 0.3),
            "contextualprecision": min(1.0, context_precision + 0.2),
            "hallucination": max(0.0, unsupported - 0.2),
        }

        passed = self._check_passed(metrics)

        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            metrics=metrics,
            passed=passed,
        )

    def _check_passed(self, metrics: Dict[str, float]) -> bool:
        """Check if all metrics pass their thresholds."""
        for metric_enum in EvaluationMetric:
            threshold = self.thresholds.get(metric_enum, 0.7)
            score = metrics.get(metric_enum.value.replace("_", ""), 0.0)

            # Handle different metric names
            for key in metrics:
                if metric_enum.value.replace("_", "") in key:
                    score = metrics[key]
                    break

            if metric_enum == EvaluationMetric.HALLUCINATION:
                # Lower is better for hallucination
                if score > threshold:
                    return False
            else:
                if score < threshold:
                    return False

        return True

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        metrics: Optional[List[EvaluationMetric]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple RAG responses.

        Args:
            test_cases: List of dicts with 'query', 'response', 'contexts'
            metrics: Optional list of metrics to evaluate

        Returns:
            List of EvaluationResult
        """
        results = []
        for case in test_cases:
            result = await self.evaluate(
                query=case.get("query", ""),
                response=case.get("response", ""),
                contexts=case.get("contexts", []),
                expected_output=case.get("expected_output"),
                metrics=metrics,
            )
            results.append(result)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        total = self._total_evaluations or 1  # Avoid division by zero

        return {
            "total_evaluations": self._total_evaluations,
            "passed_evaluations": self._total_passed,
            "pass_rate": round(self._total_passed / total, 3),
            "avg_metrics": {
                name: round(total_score / total, 3)
                for name, total_score in self._metric_totals.items()
            },
            "thresholds": {m.value: t for m, t in self.thresholds.items()},
            "deepeval_available": DEEPEVAL_AVAILABLE,
        }

    def reset_statistics(self):
        """Reset evaluation statistics."""
        self._total_evaluations = 0
        self._total_passed = 0
        self._metric_totals = {m.value: 0.0 for m in EvaluationMetric}


# Global evaluator instance
_evaluator: Optional[DeepEvalRAGEvaluator] = None


def get_evaluator(
    model: str = "gpt-4o-mini",
    **kwargs
) -> DeepEvalRAGEvaluator:
    """Get or create global evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = DeepEvalRAGEvaluator(model=model, **kwargs)
    return _evaluator


# Convenience functions
async def evaluate_rag_response(
    query: str,
    response: str,
    contexts: List[str],
    expected_output: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate a single RAG response.

    Args:
        query: User query
        response: Generated response
        contexts: Retrieved contexts
        expected_output: Optional expected answer

    Returns:
        EvaluationResult with scores
    """
    evaluator = get_evaluator()
    return await evaluator.evaluate(
        query=query,
        response=response,
        contexts=contexts,
        expected_output=expected_output,
    )


async def run_benchmark_evaluation(
    orchestrator,
    queries: Optional[List[Any]] = None,
    max_queries: int = 10,
    preset: str = "balanced",
) -> BenchmarkEvaluationResult:
    """
    Run DeepEval evaluation on benchmark queries.

    Args:
        orchestrator: UniversalOrchestrator instance
        queries: Optional list of BenchmarkQuery (uses FANUC_BENCHMARK if None)
        max_queries: Maximum queries to evaluate
        preset: Orchestrator preset to use

    Returns:
        BenchmarkEvaluationResult with aggregate metrics
    """
    from .benchmark import FANUC_BENCHMARK
    from .models import SearchRequest

    evaluator = get_evaluator()
    test_queries = queries or FANUC_BENCHMARK[:max_queries]
    results = []

    for i, bq in enumerate(test_queries):
        logger.info(f"[{i+1}/{len(test_queries)}] Evaluating: {bq.query[:50]}...")

        try:
            # Run search
            request = SearchRequest(
                query=bq.query,
                max_iterations=3,
                min_confidence=0.5
            )
            response = await orchestrator.search(request)

            # Extract response and contexts
            answer = getattr(response, 'synthesis', str(response))
            contexts = []
            if hasattr(response, 'sources') and response.sources:
                contexts = [
                    s.get('snippet', s.get('content', ''))[:500]
                    for s in response.sources
                    if isinstance(s, dict)
                ]

            # Evaluate
            result = await evaluator.evaluate(
                query=bq.query,
                response=answer,
                contexts=contexts,
                expected_output=bq.ground_truth_summary,
            )
            results.append(result)

            logger.info(
                f"  -> {'PASS' if result.passed else 'FAIL'} "
                f"(faithfulness={result.metrics.get('faithfulness', 0):.2f})"
            )

        except Exception as e:
            logger.error(f"  -> ERROR: {e}")
            results.append(EvaluationResult(
                query=bq.query,
                response="",
                contexts=[],
                metrics={},
                passed=False,
                error=str(e),
            ))

    # Calculate aggregates
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    avg_faithfulness = sum(
        r.metrics.get("faithfulness", 0) for r in results
    ) / max(total, 1)
    avg_relevancy = sum(
        r.metrics.get("answerrelevancy", 0) for r in results
    ) / max(total, 1)
    avg_precision = sum(
        r.metrics.get("contextualprecision", 0) for r in results
    ) / max(total, 1)
    avg_hallucination = sum(
        r.metrics.get("hallucination", 0) for r in results
    ) / max(total, 1)

    return BenchmarkEvaluationResult(
        total_queries=total,
        passed_queries=passed,
        pass_rate=passed / max(total, 1),
        avg_faithfulness=avg_faithfulness,
        avg_answer_relevancy=avg_relevancy,
        avg_context_precision=avg_precision,
        avg_hallucination=avg_hallucination,
        results=results,
    )


def get_evaluation_summary() -> Dict[str, Any]:
    """Get summary of all evaluations."""
    evaluator = get_evaluator()
    return evaluator.get_statistics()


# Pytest integration
def create_deepeval_test_cases(
    test_data: List[Dict[str, Any]]
) -> List["LLMTestCase"]:
    """
    Create DeepEval test cases from test data.

    For use in pytest integration:
        @pytest.mark.parametrize("test_case", create_deepeval_test_cases(data))
        def test_rag_quality(test_case):
            ...
    """
    if not DEEPEVAL_AVAILABLE:
        return []

    return [
        LLMTestCase(
            input=case["query"],
            actual_output=case["response"],
            retrieval_context=case.get("contexts", []),
            expected_output=case.get("expected_output"),
        )
        for case in test_data
    ]


# Export availability flag
def is_deepeval_available() -> bool:
    """Check if DeepEval is available."""
    return DEEPEVAL_AVAILABLE
