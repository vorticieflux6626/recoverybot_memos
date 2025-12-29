"""
Performance Metrics for Agentic Search Pipeline

Tracks key performance indicators:
- Query response times (TTFT, total time)
- Cache hit rates
- Token usage and savings
- Tool latencies
- Success/failure rates

Ref: KV_CACHE_IMPLEMENTATION_PLAN.md - Monitoring Metrics section
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict

logger = logging.getLogger("agentic.metrics")


class MetricType(str, Enum):
    """Types of metrics tracked"""
    QUERY_TIME = "query_time"
    CACHE_HIT = "cache_hit"
    TOKEN_USAGE = "token_usage"
    TOOL_LATENCY = "tool_latency"
    SYNTHESIS_TIME = "synthesis_time"
    SEARCH_ITERATION = "search_iteration"
    CONTEXT_UTILIZATION = "context_utilization"


@dataclass
class ContextUtilization:
    """Tracks context window utilization per agent/model call"""
    agent_name: str
    model_name: str
    context_window: int  # Total context window size (tokens)
    input_chars: int  # Characters used in input
    output_chars: int  # Characters in output (estimated)
    estimated_input_tokens: int  # Estimated input tokens (chars / 4)
    estimated_output_tokens: int  # Estimated output tokens
    utilization_pct: float  # Percentage of context window used

    @classmethod
    def calculate(
        cls,
        agent_name: str,
        model_name: str,
        input_text: str,
        output_text: str = "",
        context_window: int = 32768
    ) -> "ContextUtilization":
        """Calculate context utilization from input/output text"""
        input_chars = len(input_text)
        output_chars = len(output_text)
        # Rough estimate: ~4 characters per token
        estimated_input_tokens = input_chars // 4
        estimated_output_tokens = output_chars // 4
        total_tokens = estimated_input_tokens + estimated_output_tokens
        utilization_pct = (total_tokens / context_window) * 100 if context_window > 0 else 0

        return cls(
            agent_name=agent_name,
            model_name=model_name,
            context_window=context_window,
            input_chars=input_chars,
            output_chars=output_chars,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            utilization_pct=round(utilization_pct, 1)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "agent": self.agent_name,
            "model": self.model_name,
            "context_window": self.context_window,
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
            "est_input_tokens": self.estimated_input_tokens,
            "est_output_tokens": self.estimated_output_tokens,
            "utilization_pct": self.utilization_pct
        }


@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""
    request_id: str
    query: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Time breakdown (ms)
    ttft_ms: float = 0  # Time to first token
    analysis_ms: float = 0
    search_ms: float = 0
    scrape_ms: float = 0
    synthesis_ms: float = 0
    total_ms: float = 0

    # Cache performance
    cache_hit: bool = False
    semantic_cache_hit: bool = False
    content_cache_hits: int = 0

    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens_saved: int = 0  # From Chain-of-Draft

    # Search stats
    iterations: int = 0
    queries_executed: int = 0
    urls_scraped: int = 0
    sources_consulted: int = 0

    # PDF API stats (technical documentation)
    pdf_api_calls: int = 0
    pdf_api_hits: int = 0  # Successful responses
    pdf_api_ms: float = 0  # Total PDF API time
    pdf_api_sources: int = 0  # Sources from PDF API
    pdf_api_error: str = ""  # Last error if any

    # Quality
    confidence_score: float = 0.0
    success: bool = True
    error_message: str = ""

    # Context utilization tracking per agent
    context_utilization: List["ContextUtilization"] = field(default_factory=list)

    def add_context_utilization(
        self,
        agent_name: str,
        model_name: str,
        input_text: str,
        output_text: str = "",
        context_window: int = 32768
    ):
        """Track context utilization for an agent call"""
        utilization = ContextUtilization.calculate(
            agent_name=agent_name,
            model_name=model_name,
            input_text=input_text,
            output_text=output_text,
            context_window=context_window
        )
        self.context_utilization.append(utilization)

        # Log if utilization is below 50% (opportunity for improvement)
        if utilization.utilization_pct < 50:
            logger.debug(
                f"Low context utilization: {agent_name} using {utilization.utilization_pct}% "
                f"of {context_window} tokens ({utilization.estimated_input_tokens} input)"
            )

    def get_avg_context_utilization(self) -> float:
        """Get average context utilization across all agent calls"""
        if not self.context_utilization:
            return 0.0
        return sum(u.utilization_pct for u in self.context_utilization) / len(self.context_utilization)

    def complete(self, success: bool = True, error: str = ""):
        """Mark query as complete"""
        self.end_time = time.time()
        self.total_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_message = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "request_id": self.request_id,
            "query": self.query[:100],
            "total_ms": round(self.total_ms, 1),
            "ttft_ms": round(self.ttft_ms, 1),
            "analysis_ms": round(self.analysis_ms, 1),
            "search_ms": round(self.search_ms, 1),
            "scrape_ms": round(self.scrape_ms, 1),
            "synthesis_ms": round(self.synthesis_ms, 1),
            "cache_hit": self.cache_hit,
            "semantic_cache_hit": self.semantic_cache_hit,
            "content_cache_hits": self.content_cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens_saved": self.thinking_tokens_saved,
            "iterations": self.iterations,
            "queries_executed": self.queries_executed,
            "urls_scraped": self.urls_scraped,
            "pdf_api_calls": self.pdf_api_calls,
            "pdf_api_hits": self.pdf_api_hits,
            "pdf_api_ms": round(self.pdf_api_ms, 1),
            "pdf_api_sources": self.pdf_api_sources,
            "pdf_api_error": self.pdf_api_error,
            "confidence": round(self.confidence_score, 2),
            "success": self.success,
            "error": self.error_message,
            "avg_context_utilization_pct": round(self.get_avg_context_utilization(), 1),
            "context_utilization": [u.to_dict() for u in self.context_utilization]
        }


class PerformanceMetrics:
    """
    Track optimization effectiveness across all queries.

    Provides aggregated statistics and individual query metrics
    for monitoring the agentic search pipeline performance.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics tracker.

        Args:
            max_history: Maximum number of queries to keep in history
        """
        self.max_history = max_history

        # Active queries
        self._active_queries: Dict[str, QueryMetrics] = {}

        # Query history
        self._query_history: List[QueryMetrics] = []

        # Aggregated stats
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "semantic_cache_hits": 0,
            "total_tokens_used": 0,
            "thinking_tokens_saved": 0,
            "total_time_ms": 0,
            "total_synthesis_ms": 0,
            # PDF API stats
            "pdf_api_total_calls": 0,
            "pdf_api_total_hits": 0,
            "pdf_api_total_ms": 0,
            "pdf_api_total_sources": 0,
        }

        # Rolling averages (exponential moving average)
        self._ema_alpha = 0.1  # Weight for new values
        self.rolling_avg = {
            "ttft_ms": 0.0,
            "total_ms": 0.0,
            "synthesis_ms": 0.0,
            "tokens_per_query": 0.0,
            "context_utilization_pct": 0.0,
        }

        # Tool-specific latency tracking
        self._tool_latencies: Dict[str, List[float]] = defaultdict(list)

        # Context utilization per agent type
        self._agent_context_utilization: Dict[str, List[float]] = defaultdict(list)

    def start_query(self, request_id: str, query: str) -> QueryMetrics:
        """Start tracking a new query"""
        metrics = QueryMetrics(request_id=request_id, query=query)
        self._active_queries[request_id] = metrics
        return metrics

    def get_active_query(self, request_id: str) -> Optional[QueryMetrics]:
        """Get metrics for an active query"""
        return self._active_queries.get(request_id)

    def record_ttft(self, request_id: str, ttft_ms: float):
        """Record time to first token"""
        if request_id in self._active_queries:
            self._active_queries[request_id].ttft_ms = ttft_ms

    def record_phase_time(
        self,
        request_id: str,
        phase: str,
        time_ms: float
    ):
        """Record time for a specific phase"""
        if request_id not in self._active_queries:
            return

        metrics = self._active_queries[request_id]
        if phase == "analysis":
            metrics.analysis_ms = time_ms
        elif phase == "search":
            metrics.search_ms = time_ms
        elif phase == "scrape":
            metrics.scrape_ms = time_ms
        elif phase == "synthesis":
            metrics.synthesis_ms = time_ms

    def record_cache_hit(
        self,
        request_id: str,
        cache_type: str = "exact"
    ):
        """Record a cache hit"""
        if request_id in self._active_queries:
            if cache_type == "exact":
                self._active_queries[request_id].cache_hit = True
            elif cache_type == "semantic":
                self._active_queries[request_id].semantic_cache_hit = True
            elif cache_type == "content":
                self._active_queries[request_id].content_cache_hits += 1

    def record_tokens(
        self,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        thinking_saved: int = 0
    ):
        """Record token usage"""
        if request_id in self._active_queries:
            metrics = self._active_queries[request_id]
            metrics.input_tokens += input_tokens
            metrics.output_tokens += output_tokens
            metrics.thinking_tokens_saved += thinking_saved

    def record_iteration(self, request_id: str):
        """Record a search iteration"""
        if request_id in self._active_queries:
            self._active_queries[request_id].iterations += 1

    def record_search(self, request_id: str, num_queries: int = 1):
        """Record search queries executed"""
        if request_id in self._active_queries:
            self._active_queries[request_id].queries_executed += num_queries

    def record_scrape(self, request_id: str, num_urls: int = 1):
        """Record URLs scraped"""
        if request_id in self._active_queries:
            self._active_queries[request_id].urls_scraped += num_urls

    def record_pdf_api_call(
        self,
        request_id: str,
        success: bool,
        latency_ms: float,
        sources_count: int = 0,
        error: str = ""
    ):
        """Record a PDF API call (technical documentation)"""
        if request_id in self._active_queries:
            metrics = self._active_queries[request_id]
            metrics.pdf_api_calls += 1
            metrics.pdf_api_ms += latency_ms
            if success:
                metrics.pdf_api_hits += 1
                metrics.pdf_api_sources += sources_count
            else:
                metrics.pdf_api_error = error

        # Track tool latency
        self.record_tool_latency("pdf_api", latency_ms)

    def record_tool_latency(self, tool_name: str, latency_ms: float):
        """Record latency for a specific tool"""
        self._tool_latencies[tool_name].append(latency_ms)
        # Keep last 100 samples
        if len(self._tool_latencies[tool_name]) > 100:
            self._tool_latencies[tool_name].pop(0)

    def record_context_utilization(
        self,
        request_id: str,
        agent_name: str,
        model_name: str,
        input_text: str,
        output_text: str = "",
        context_window: int = 32768
    ):
        """Record context utilization for an agent call"""
        if request_id in self._active_queries:
            self._active_queries[request_id].add_context_utilization(
                agent_name=agent_name,
                model_name=model_name,
                input_text=input_text,
                output_text=output_text,
                context_window=context_window
            )
            # Also track per-agent utilization
            utilization = ContextUtilization.calculate(
                agent_name=agent_name,
                model_name=model_name,
                input_text=input_text,
                output_text=output_text,
                context_window=context_window
            )
            self._agent_context_utilization[agent_name].append(utilization.utilization_pct)
            # Keep last 100 samples per agent
            if len(self._agent_context_utilization[agent_name]) > 100:
                self._agent_context_utilization[agent_name].pop(0)

    def complete_query(
        self,
        request_id: str,
        success: bool = True,
        error: str = "",
        confidence: float = 0.0
    ) -> Optional[QueryMetrics]:
        """
        Complete tracking for a query and update aggregates.

        Returns the final QueryMetrics object.
        """
        if request_id not in self._active_queries:
            return None

        metrics = self._active_queries.pop(request_id)
        metrics.complete(success=success, error=error)
        metrics.confidence_score = confidence

        # Add to history
        self._query_history.append(metrics)
        if len(self._query_history) > self.max_history:
            self._query_history.pop(0)

        # Update aggregates
        self._update_aggregates(metrics)

        # Log summary including context utilization
        avg_ctx_util = metrics.get_avg_context_utilization()
        logger.info(
            f"Query {request_id[:8]} completed: "
            f"total={metrics.total_ms:.0f}ms, "
            f"synthesis={metrics.synthesis_ms:.0f}ms, "
            f"cache_hit={metrics.cache_hit}, "
            f"tokens={metrics.input_tokens + metrics.output_tokens}, "
            f"saved={metrics.thinking_tokens_saved}, "
            f"ctx_util={avg_ctx_util:.1f}%"
        )

        return metrics

    def _update_aggregates(self, metrics: QueryMetrics):
        """Update aggregate statistics"""
        self.stats["total_queries"] += 1

        if metrics.success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1

        if metrics.cache_hit or metrics.semantic_cache_hit:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1

        if metrics.semantic_cache_hit:
            self.stats["semantic_cache_hits"] += 1

        total_tokens = metrics.input_tokens + metrics.output_tokens
        self.stats["total_tokens_used"] += total_tokens
        self.stats["thinking_tokens_saved"] += metrics.thinking_tokens_saved
        self.stats["total_time_ms"] += metrics.total_ms
        self.stats["total_synthesis_ms"] += metrics.synthesis_ms

        # Update PDF API stats
        self.stats["pdf_api_total_calls"] += metrics.pdf_api_calls
        self.stats["pdf_api_total_hits"] += metrics.pdf_api_hits
        self.stats["pdf_api_total_ms"] += metrics.pdf_api_ms
        self.stats["pdf_api_total_sources"] += metrics.pdf_api_sources

        # Update rolling averages
        self._update_ema("ttft_ms", metrics.ttft_ms)
        self._update_ema("total_ms", metrics.total_ms)
        self._update_ema("synthesis_ms", metrics.synthesis_ms)
        self._update_ema("tokens_per_query", total_tokens)
        self._update_ema("context_utilization_pct", metrics.get_avg_context_utilization())

    def _update_ema(self, key: str, value: float):
        """Update exponential moving average"""
        if self.rolling_avg[key] == 0:
            self.rolling_avg[key] = value
        else:
            self.rolling_avg[key] = (
                self._ema_alpha * value +
                (1 - self._ema_alpha) * self.rolling_avg[key]
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        total = self.stats["total_queries"] or 1  # Avoid division by zero

        # Calculate cache hit rate
        cache_checks = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = self.stats["cache_hits"] / max(cache_checks, 1)

        # Calculate average times
        avg_total_ms = self.stats["total_time_ms"] / total
        avg_synthesis_ms = self.stats["total_synthesis_ms"] / total
        avg_tokens = self.stats["total_tokens_used"] / total

        # Estimated time saved from cache hits and thinking tokens
        estimated_time_saved_ms = (
            self.stats["cache_hits"] * avg_total_ms +  # Full cache hits
            self.stats["thinking_tokens_saved"] * 50   # ~50ms per token saved
        )

        # Calculate PDF API stats
        pdf_calls = self.stats["pdf_api_total_calls"] or 1
        pdf_hit_rate = self.stats["pdf_api_total_hits"] / pdf_calls if pdf_calls > 0 else 0
        avg_pdf_ms = self.stats["pdf_api_total_ms"] / pdf_calls if pdf_calls > 0 else 0
        avg_pdf_sources = self.stats["pdf_api_total_sources"] / pdf_calls if pdf_calls > 0 else 0

        return {
            "aggregate_stats": {
                **self.stats,
                "cache_hit_rate": round(cache_hit_rate, 3),
                "success_rate": round(self.stats["successful_queries"] / total, 3),
                "avg_total_ms": round(avg_total_ms, 1),
                "avg_synthesis_ms": round(avg_synthesis_ms, 1),
                "avg_tokens_per_query": round(avg_tokens, 0),
                "estimated_time_saved_ms": round(estimated_time_saved_ms, 0),
            },
            "pdf_api_stats": {
                "total_calls": self.stats["pdf_api_total_calls"],
                "total_hits": self.stats["pdf_api_total_hits"],
                "hit_rate": round(pdf_hit_rate, 3),
                "avg_latency_ms": round(avg_pdf_ms, 1),
                "avg_sources_per_call": round(avg_pdf_sources, 1),
                "total_sources": self.stats["pdf_api_total_sources"],
            },
            "rolling_averages": {
                k: round(v, 1) for k, v in self.rolling_avg.items()
            },
            "tool_latencies": self._get_tool_latency_stats(),
            "context_utilization": self._get_context_utilization_stats(),
            "recent_queries": [
                q.to_dict() for q in self._query_history[-5:]
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _get_tool_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics per tool"""
        stats = {}
        for tool, latencies in self._tool_latencies.items():
            if not latencies:
                continue
            sorted_latencies = sorted(latencies)
            stats[tool] = {
                "avg_ms": round(sum(latencies) / len(latencies), 1),
                "p50_ms": round(sorted_latencies[len(sorted_latencies) // 2], 1),
                "p95_ms": round(sorted_latencies[int(len(sorted_latencies) * 0.95)], 1),
                "count": len(latencies)
            }
        return stats

    def _get_context_utilization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get context utilization statistics per agent"""
        stats = {
            "avg_utilization_pct": round(self.rolling_avg.get("context_utilization_pct", 0), 1),
            "per_agent": {}
        }
        for agent, utilizations in self._agent_context_utilization.items():
            if not utilizations:
                continue
            sorted_utils = sorted(utilizations)
            stats["per_agent"][agent] = {
                "avg_pct": round(sum(utilizations) / len(utilizations), 1),
                "min_pct": round(min(utilizations), 1),
                "max_pct": round(max(utilizations), 1),
                "p50_pct": round(sorted_utils[len(sorted_utils) // 2], 1),
                "count": len(utilizations)
            }
        return stats

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query metrics"""
        return [q.to_dict() for q in self._query_history[-limit:]]

    def reset(self):
        """Reset all metrics (for testing)"""
        self._active_queries.clear()
        self._query_history.clear()
        self._tool_latencies.clear()
        self._agent_context_utilization.clear()
        self.stats = {k: 0 for k in self.stats}
        self.rolling_avg = {k: 0.0 for k in self.rolling_avg}


# Global metrics instance
_performance_metrics: Optional[PerformanceMetrics] = None


def get_performance_metrics() -> PerformanceMetrics:
    """Get the global performance metrics instance"""
    global _performance_metrics
    if _performance_metrics is None:
        _performance_metrics = PerformanceMetrics()
    return _performance_metrics


# Context manager for timing phases
class PhaseTimer:
    """Context manager for timing query phases"""

    def __init__(
        self,
        request_id: str,
        phase: str,
        metrics: Optional[PerformanceMetrics] = None
    ):
        self.request_id = request_id
        self.phase = phase
        self.metrics = metrics or get_performance_metrics()
        self.start_time: float = 0

    def __enter__(self) -> "PhaseTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.metrics.record_phase_time(self.request_id, self.phase, elapsed_ms)
        return False
