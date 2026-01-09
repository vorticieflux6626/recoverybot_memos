"""
Observability Dashboard - Aggregate views for pipeline monitoring.

Part of P3 Observability Enhancement (OBSERVABILITY_IMPROVEMENT_PLAN.md).

Provides:
- Request-level observability aggregation
- Cross-request statistics
- Pipeline health metrics
- Technician audit trail generation
- **Database persistence for historical queries** (Added 2026-01-07)

Created: 2026-01-02
Updated: 2026-01-07 - Added database persistence for Agent Console
"""

import logging
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RequestObservability:
    """
    Aggregated observability data for a single request.

    Combines all P0-P2 observability modules into a unified view.
    """
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    query: str = ""
    preset: str = "balanced"

    # From DecisionLogger (P0)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    decision_count: int = 0

    # From ContextFlowTracker (P0)
    context_transfers: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens_transferred: int = 0

    # From LLMCallLogger (P1)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    total_llm_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # From ScratchpadObserver (P1)
    scratchpad_changes: List[Dict[str, Any]] = field(default_factory=list)
    findings_count: int = 0
    questions_answered: int = 0

    # From TechnicianLog (P1)
    technician_log_markdown: str = ""

    # From ConfidenceLogger (P2)
    confidence_breakdown: Optional[Dict[str, Any]] = None
    final_confidence: float = 0.0
    confidence_level: str = "unknown"

    # Pipeline metrics
    total_duration_ms: int = 0
    agents_executed: List[str] = field(default_factory=list)
    features_enabled: List[str] = field(default_factory=list)
    features_skipped: List[str] = field(default_factory=list)

    # Outcome
    success: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "preset": self.preset,
            "summary": {
                "success": self.success,
                "duration_ms": self.total_duration_ms,
                "confidence": self.final_confidence,
                "confidence_level": self.confidence_level,
            },
            "decisions": {
                "count": self.decision_count,
                "items": self.decisions[:10],  # Limit for response size
            },
            "context_flow": {
                "transfers": len(self.context_transfers),
                "total_tokens": self.total_tokens_transferred,
            },
            "llm_calls": {
                "count": len(self.llm_calls),
                "total_latency_ms": self.total_llm_latency_ms,
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
            },
            "scratchpad": {
                "changes": len(self.scratchpad_changes),
                "findings": self.findings_count,
                "questions_answered": self.questions_answered,
            },
            "confidence": self.confidence_breakdown,
            "agents": self.agents_executed,
            "features": {
                "enabled": self.features_enabled,
                "skipped": self.features_skipped,
            },
            "error": self.error_message,
        }

    def to_technician_summary(self) -> str:
        """Generate a technician-friendly summary."""
        lines = [
            f"# Request Summary: {self.request_id}",
            f"",
            f"**Query**: {self.query}",
            f"**Preset**: {self.preset}",
            f"**Status**: {'✅ Success' if self.success else '❌ Failed'}",
            f"",
            f"## Performance",
            f"- Total Duration: {self.total_duration_ms}ms",
            f"- LLM Calls: {len(self.llm_calls)} ({self.total_llm_latency_ms}ms total)",
            f"- Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out",
            f"",
            f"## Confidence: {self.final_confidence:.0%} ({self.confidence_level})",
        ]

        if self.confidence_breakdown:
            signals = self.confidence_breakdown.get("signals", [])
            if signals:
                lines.append("")
                lines.append("### Signal Breakdown")
                for signal in signals:
                    bar = "█" * int(signal.get("raw", 0) * 5) + "░" * (5 - int(signal.get("raw", 0) * 5))
                    lines.append(f"- {signal.get('signal', 'unknown')}: {bar} {signal.get('raw', 0):.0%}")

        if self.findings_count > 0:
            lines.append(f"")
            lines.append(f"## Findings: {self.findings_count}")
            lines.append(f"- Questions Answered: {self.questions_answered}")

        if self.agents_executed:
            lines.append(f"")
            lines.append(f"## Agents Executed")
            lines.append(f"- {' → '.join(self.agents_executed)}")

        if self.features_skipped:
            lines.append(f"")
            lines.append(f"## Features Skipped")
            for feat in self.features_skipped[:5]:
                lines.append(f"- {feat}")

        if self.error_message:
            lines.append(f"")
            lines.append(f"## ⚠️ Error")
            lines.append(f"```")
            lines.append(self.error_message[:500])
            lines.append(f"```")

        return "\n".join(lines)


@dataclass
class DashboardStats:
    """Aggregate statistics across multiple requests."""

    # Time window
    window_start: datetime = field(default_factory=datetime.now)
    window_end: datetime = field(default_factory=datetime.now)

    # Request counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Performance
    avg_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0

    # LLM usage
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_llm_latency_ms: float = 0.0

    # Confidence
    avg_confidence: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)

    # Agent activity
    agent_call_counts: Dict[str, int] = field(default_factory=dict)

    # Feature usage
    feature_usage_counts: Dict[str, int] = field(default_factory=dict)
    feature_skip_counts: Dict[str, int] = field(default_factory=dict)

    # Errors
    error_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": {
                "start": self.window_start.isoformat(),
                "end": self.window_end.isoformat(),
            },
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": self.successful_requests / max(1, self.total_requests),
            },
            "performance": {
                "avg_duration_ms": round(self.avg_duration_ms, 1),
                "p50_duration_ms": round(self.p50_duration_ms, 1),
                "p95_duration_ms": round(self.p95_duration_ms, 1),
            },
            "llm_usage": {
                "total_calls": self.total_llm_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "avg_latency_ms": round(self.avg_llm_latency_ms, 1),
            },
            "confidence": {
                "average": round(self.avg_confidence, 3),
                "distribution": self.confidence_distribution,
            },
            "agents": self.agent_call_counts,
            "features": {
                "enabled": self.feature_usage_counts,
                "skipped": self.feature_skip_counts,
            },
            "errors": self.error_counts,
        }


class ObservabilityDashboard:
    """
    Central dashboard for aggregating and querying observability data.

    Usage:
        dashboard = ObservabilityDashboard()

        # Store request observability
        dashboard.store_request(request_obs)

        # Query
        stats = dashboard.get_stats(last_hours=1)
        request = dashboard.get_request("req-123")
        recent = dashboard.get_recent_requests(limit=10)

    Note: As of 2026-01-07, this class also persists data to PostgreSQL
    for historical queries by the Agent Console.
    """

    def __init__(self, max_requests: int = 1000):
        self.max_requests = max_requests
        self._requests: Dict[str, RequestObservability] = {}
        self._request_order: List[str] = []  # For FIFO eviction
        self._db_persist_enabled = True  # Can be disabled for testing

    def store_request(self, obs: RequestObservability, sse_events: Optional[List[Any]] = None):
        """Store a request's observability data (in-memory and database).

        Args:
            obs: RequestObservability data from the agent pipeline
            sse_events: Optional list of SearchEvent objects from EventEmitter.get_history()
                       These contain the detailed SSE events (classifying_query, scraping_url, etc.)
        """
        # Evict oldest if at capacity (in-memory cache)
        while len(self._requests) >= self.max_requests and self._request_order:
            oldest_id = self._request_order.pop(0)
            self._requests.pop(oldest_id, None)

        self._requests[obs.request_id] = obs
        self._request_order.append(obs.request_id)

        logger.info(
            f"[Dashboard] Stored request {obs.request_id}: "
            f"conf={obs.final_confidence:.2f}, duration={obs.total_duration_ms}ms, "
            f"llm_calls={len(obs.llm_calls)}, sse_events={len(sse_events) if sse_events else 0}"
        )

        # Persist to database using background thread (always works, sync or async context)
        if self._db_persist_enabled:
            import threading
            try:
                thread = threading.Thread(
                    target=self._persist_to_database_sync,
                    args=(obs, sse_events),
                    daemon=True,
                    name=f"persist-{obs.request_id[:8]}"
                )
                thread.start()
                logger.info(f"[Dashboard] Queued persistence for {obs.request_id}")
            except Exception as e:
                logger.warning(f"[Dashboard] Could not queue persistence: {e}")

    async def _persist_to_database(self, obs: RequestObservability):
        """Persist observability data to PostgreSQL for historical queries."""
        try:
            from config.database import AsyncSessionLocal
            from models.agent_run import AgentRun, AgentRunStatus

            async with AsyncSessionLocal() as session:
                # Check if already exists (update case)
                from sqlalchemy import select
                result = await session.execute(
                    select(AgentRun).where(AgentRun.request_id == obs.request_id)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing record
                    existing.status = AgentRunStatus.COMPLETED if obs.success else AgentRunStatus.FAILED
                    existing.completed_at = datetime.now(timezone.utc)
                    existing.duration_ms = obs.total_duration_ms
                    existing.success = obs.success
                    existing.error_message = obs.error_message
                    existing.final_confidence = obs.final_confidence
                    existing.confidence_level = obs.confidence_level
                    existing.confidence_breakdown = obs.confidence_breakdown
                    existing.llm_calls_count = len(obs.llm_calls)
                    existing.total_llm_latency_ms = obs.total_llm_latency_ms
                    existing.total_input_tokens = obs.total_input_tokens
                    existing.total_output_tokens = obs.total_output_tokens
                    existing.agents_executed = obs.agents_executed
                    existing.features_enabled = obs.features_enabled
                    existing.features_skipped = obs.features_skipped
                    existing.decision_count = obs.decision_count
                    existing.decisions = obs.decisions
                    existing.context_transfers_count = len(obs.context_transfers)
                    existing.total_tokens_transferred = obs.total_tokens_transferred
                    existing.context_transfers = obs.context_transfers
                    existing.llm_calls = obs.llm_calls
                    existing.scratchpad_changes = obs.scratchpad_changes
                    existing.findings_count = obs.findings_count
                    existing.questions_answered = obs.questions_answered
                    existing.technician_log_markdown = obs.technician_log_markdown
                    existing.events = self._build_events_timeline(obs)
                else:
                    # Create new record
                    agent_run = AgentRun(
                        request_id=obs.request_id,
                        query=obs.query,
                        preset=obs.preset,
                        status=AgentRunStatus.COMPLETED if obs.success else AgentRunStatus.FAILED,
                        started_at=obs.timestamp,
                        completed_at=datetime.now(timezone.utc),
                        duration_ms=obs.total_duration_ms,
                        success=obs.success,
                        error_message=obs.error_message,
                        final_confidence=obs.final_confidence,
                        confidence_level=obs.confidence_level,
                        confidence_breakdown=obs.confidence_breakdown,
                        llm_calls_count=len(obs.llm_calls),
                        total_llm_latency_ms=obs.total_llm_latency_ms,
                        total_input_tokens=obs.total_input_tokens,
                        total_output_tokens=obs.total_output_tokens,
                        agents_executed=obs.agents_executed,
                        features_enabled=obs.features_enabled,
                        features_skipped=obs.features_skipped,
                        decision_count=obs.decision_count,
                        decisions=obs.decisions,
                        context_transfers_count=len(obs.context_transfers),
                        total_tokens_transferred=obs.total_tokens_transferred,
                        context_transfers=obs.context_transfers,
                        llm_calls=obs.llm_calls,
                        scratchpad_changes=obs.scratchpad_changes,
                        findings_count=obs.findings_count,
                        questions_answered=obs.questions_answered,
                        technician_log_markdown=obs.technician_log_markdown,
                        events=self._build_events_timeline(obs),
                    )
                    session.add(agent_run)

                await session.commit()
                logger.debug(f"[Dashboard] Persisted request {obs.request_id} to database")

        except Exception as e:
            logger.error(f"[Dashboard] Failed to persist to database: {e}")

    def _persist_to_database_sync(self, obs: RequestObservability, sse_events: Optional[List[Any]] = None):
        """Synchronous database persistence using a fresh sync session.

        Args:
            obs: RequestObservability data
            sse_events: Optional list of SearchEvent objects from EventEmitter.get_history()
        """
        try:
            from config.database import SyncSessionLocal
            from models.agent_run import AgentRun, AgentRunStatus
            from sqlalchemy import select

            # Build events timeline from SSE events if available, otherwise from obs
            events_timeline = self._build_events_timeline(obs, sse_events)

            with SyncSessionLocal() as session:
                # Check if already exists (update case)
                result = session.execute(
                    select(AgentRun).where(AgentRun.request_id == obs.request_id)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing record
                    existing.status = AgentRunStatus.COMPLETED if obs.success else AgentRunStatus.FAILED
                    existing.completed_at = datetime.now(timezone.utc)
                    existing.duration_ms = obs.total_duration_ms
                    existing.success = obs.success
                    existing.error_message = obs.error_message
                    existing.final_confidence = obs.final_confidence
                    existing.confidence_level = obs.confidence_level
                    existing.confidence_breakdown = obs.confidence_breakdown
                    existing.llm_calls_count = len(obs.llm_calls)
                    existing.total_llm_latency_ms = obs.total_llm_latency_ms
                    existing.total_input_tokens = obs.total_input_tokens
                    existing.total_output_tokens = obs.total_output_tokens
                    existing.agents_executed = obs.agents_executed
                    existing.features_enabled = obs.features_enabled
                    existing.features_skipped = obs.features_skipped
                    existing.decision_count = obs.decision_count
                    existing.decisions = obs.decisions
                    existing.context_transfers_count = len(obs.context_transfers)
                    existing.total_tokens_transferred = obs.total_tokens_transferred
                    existing.context_transfers = obs.context_transfers
                    existing.llm_calls = obs.llm_calls
                    existing.scratchpad_changes = obs.scratchpad_changes
                    existing.findings_count = obs.findings_count
                    existing.questions_answered = obs.questions_answered
                    existing.technician_log_markdown = obs.technician_log_markdown
                    existing.events = events_timeline
                else:
                    # Create new record
                    agent_run = AgentRun(
                        request_id=obs.request_id,
                        query=obs.query,
                        preset=obs.preset,
                        status=AgentRunStatus.COMPLETED if obs.success else AgentRunStatus.FAILED,
                        started_at=obs.timestamp,
                        completed_at=datetime.now(timezone.utc),
                        duration_ms=obs.total_duration_ms,
                        success=obs.success,
                        error_message=obs.error_message,
                        final_confidence=obs.final_confidence,
                        confidence_level=obs.confidence_level,
                        confidence_breakdown=obs.confidence_breakdown,
                        llm_calls_count=len(obs.llm_calls),
                        total_llm_latency_ms=obs.total_llm_latency_ms,
                        total_input_tokens=obs.total_input_tokens,
                        total_output_tokens=obs.total_output_tokens,
                        agents_executed=obs.agents_executed,
                        features_enabled=obs.features_enabled,
                        features_skipped=obs.features_skipped,
                        decision_count=obs.decision_count,
                        decisions=obs.decisions,
                        context_transfers_count=len(obs.context_transfers),
                        total_tokens_transferred=obs.total_tokens_transferred,
                        context_transfers=obs.context_transfers,
                        llm_calls=obs.llm_calls,
                        scratchpad_changes=obs.scratchpad_changes,
                        findings_count=obs.findings_count,
                        questions_answered=obs.questions_answered,
                        technician_log_markdown=obs.technician_log_markdown,
                        events=events_timeline,
                    )
                    session.add(agent_run)

                session.commit()
                logger.info(f"[Dashboard] Persisted request {obs.request_id} to database with {len(events_timeline)} events")

        except Exception as e:
            logger.error(f"[Dashboard] Sync persist failed: {e}", exc_info=True)

    def _build_events_timeline(self, obs: RequestObservability, sse_events: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Build a unified events timeline from SSE events or observability data.

        If sse_events is provided (from EventEmitter.get_history()), use those as the
        primary source since they contain detailed pipeline events (classifying_query,
        scraping_url, verifying_claims, etc.). Otherwise fall back to obs data.

        Args:
            obs: RequestObservability data
            sse_events: Optional list of SearchEvent objects from EventEmitter.get_history()

        Returns:
            List of event dictionaries for the timeline
        """
        events = []

        # If we have SSE events from the event emitter, use them - they have all the detail
        if sse_events:
            for i, event in enumerate(sse_events):
                try:
                    # SearchEvent is a dataclass, convert to dict
                    event_dict = {
                        "type": event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                        "timestamp": event.timestamp,
                        "message": event.message or "",
                        "order": i,
                        "data": {},
                    }

                    # Add optional fields if present
                    if hasattr(event, 'query') and event.query:
                        event_dict["data"]["query"] = event.query
                    if hasattr(event, 'url') and event.url:
                        event_dict["data"]["url"] = event.url
                    if hasattr(event, 'url_index') and event.url_index:
                        event_dict["data"]["url_index"] = event.url_index
                    if hasattr(event, 'url_total') and event.url_total:
                        event_dict["data"]["url_total"] = event.url_total
                    if hasattr(event, 'results_count') and event.results_count is not None:
                        event_dict["data"]["results_count"] = event.results_count
                    if hasattr(event, 'sources_count') and event.sources_count is not None:
                        event_dict["data"]["sources_count"] = event.sources_count
                    if hasattr(event, 'progress_percent') and event.progress_percent is not None:
                        event_dict["data"]["progress"] = event.progress_percent
                    if hasattr(event, 'confidence') and event.confidence is not None:
                        event_dict["data"]["confidence"] = event.confidence
                    if hasattr(event, 'engines') and event.engines:
                        event_dict["data"]["engines"] = event.engines
                    if hasattr(event, 'iteration') and event.iteration is not None:
                        event_dict["data"]["iteration"] = event.iteration
                    if hasattr(event, 'max_iterations') and event.max_iterations is not None:
                        event_dict["data"]["max_iterations"] = event.max_iterations
                    if hasattr(event, 'graph_line') and event.graph_line:
                        event_dict["data"]["graph_line"] = event.graph_line

                    # Include any additional data from the event's data dict
                    if hasattr(event, 'data') and event.data:
                        event_dict["data"].update(event.data)

                    events.append(event_dict)
                except Exception as e:
                    logger.warning(f"[Dashboard] Failed to serialize SSE event: {e}")
                    continue

            logger.debug(f"[Dashboard] Built timeline with {len(events)} SSE events")
            return events

        # Fallback: Build from observability summary data
        # Add decisions as events
        for i, decision in enumerate(obs.decisions):
            events.append({
                "type": "decision",
                "timestamp": decision.get("timestamp", obs.timestamp.isoformat()),
                "agent": decision.get("agent_name", decision.get("agent", "unknown")),
                "action": decision.get("decision_type", "decision"),
                "data": decision,
                "order": i,
            })

        # Add LLM calls as events
        for i, call in enumerate(obs.llm_calls):
            events.append({
                "type": "llm_call",
                "timestamp": call.get("timestamp", obs.timestamp.isoformat()),
                "agent": call.get("caller", "unknown"),
                "action": "llm_call",
                "data": {
                    "model": call.get("model", "unknown"),
                    "latency_ms": call.get("latency_ms", 0),
                    "input_tokens": call.get("input_tokens", 0),
                    "output_tokens": call.get("output_tokens", 0),
                    "purpose": call.get("purpose", ""),
                },
                "order": len(obs.decisions) + i,
            })

        # Add context transfers as events
        for i, transfer in enumerate(obs.context_transfers):
            events.append({
                "type": "context_transfer",
                "timestamp": transfer.get("timestamp", obs.timestamp.isoformat()),
                "agent": transfer.get("from_agent", "unknown"),
                "action": "context_transfer",
                "data": {
                    "from": transfer.get("from_agent"),
                    "to": transfer.get("to_agent"),
                    "tokens": transfer.get("tokens_transferred", 0),
                },
                "order": len(obs.decisions) + len(obs.llm_calls) + i,
            })

        # Add scratchpad changes as events
        for i, change in enumerate(obs.scratchpad_changes):
            events.append({
                "type": "scratchpad_change",
                "timestamp": change.get("timestamp", obs.timestamp.isoformat()),
                "agent": change.get("agent", "unknown"),
                "action": change.get("change_type", "update"),
                "data": change,
                "order": len(obs.decisions) + len(obs.llm_calls) + len(obs.context_transfers) + i,
            })

        # Sort by order
        events.sort(key=lambda e: e.get("order", 0))

        return events

    def get_request(self, request_id: str) -> Optional[RequestObservability]:
        """Get observability data for a specific request."""
        return self._requests.get(request_id)

    def get_recent_requests(
        self,
        limit: int = 10,
        preset: Optional[str] = None,
        success_only: bool = False
    ) -> List[RequestObservability]:
        """Get recent requests with optional filtering."""
        results = []
        for request_id in reversed(self._request_order):
            obs = self._requests.get(request_id)
            if not obs:
                continue

            # Apply filters
            if preset and obs.preset != preset:
                continue
            if success_only and not obs.success:
                continue

            results.append(obs)
            if len(results) >= limit:
                break

        return results

    def get_stats(
        self,
        last_hours: int = 1,
        preset: Optional[str] = None
    ) -> DashboardStats:
        """Get aggregate statistics for a time window."""
        cutoff = datetime.now() - timedelta(hours=last_hours)

        # Filter requests
        requests = [
            obs for obs in self._requests.values()
            if obs.timestamp >= cutoff
            and (preset is None or obs.preset == preset)
        ]

        if not requests:
            return DashboardStats(
                window_start=cutoff,
                window_end=datetime.now()
            )

        # Calculate stats
        stats = DashboardStats(
            window_start=cutoff,
            window_end=datetime.now(),
            total_requests=len(requests),
            successful_requests=sum(1 for r in requests if r.success),
            failed_requests=sum(1 for r in requests if not r.success),
        )

        # Performance metrics
        durations = [r.total_duration_ms for r in requests]
        stats.avg_duration_ms = sum(durations) / len(durations)
        sorted_durations = sorted(durations)
        stats.p50_duration_ms = sorted_durations[len(sorted_durations) // 2]
        stats.p95_duration_ms = sorted_durations[int(len(sorted_durations) * 0.95)]

        # LLM usage
        stats.total_llm_calls = sum(len(r.llm_calls) for r in requests)
        stats.total_input_tokens = sum(r.total_input_tokens for r in requests)
        stats.total_output_tokens = sum(r.total_output_tokens for r in requests)
        llm_latencies = [r.total_llm_latency_ms for r in requests if r.total_llm_latency_ms > 0]
        if llm_latencies:
            stats.avg_llm_latency_ms = sum(llm_latencies) / len(llm_latencies)

        # Confidence
        confidences = [r.final_confidence for r in requests]
        stats.avg_confidence = sum(confidences) / len(confidences)

        # Confidence distribution
        for r in requests:
            level = r.confidence_level
            stats.confidence_distribution[level] = stats.confidence_distribution.get(level, 0) + 1

        # Agent activity
        for r in requests:
            for agent in r.agents_executed:
                stats.agent_call_counts[agent] = stats.agent_call_counts.get(agent, 0) + 1

        # Feature usage
        for r in requests:
            for feat in r.features_enabled:
                stats.feature_usage_counts[feat] = stats.feature_usage_counts.get(feat, 0) + 1
            for feat in r.features_skipped:
                stats.feature_skip_counts[feat] = stats.feature_skip_counts.get(feat, 0) + 1

        # Errors
        for r in requests:
            if r.error_message:
                error_type = r.error_message.split(":")[0][:50]
                stats.error_counts[error_type] = stats.error_counts.get(error_type, 0) + 1

        return stats

    def get_technician_audit(
        self,
        request_id: str
    ) -> Optional[str]:
        """Get a technician-friendly audit trail for a request."""
        obs = self._requests.get(request_id)
        if not obs:
            return None

        lines = [
            obs.to_technician_summary(),
            "",
            "---",
            "",
            "## Decision Trail",
            "",
        ]

        for i, decision in enumerate(obs.decisions[:15], 1):
            lines.append(
                f"{i}. **{decision.get('agent', 'unknown')}.{decision.get('decision_type', 'unknown')}**: "
                f"{decision.get('decision_made', 'N/A')}"
            )
            if decision.get('reasoning'):
                lines.append(f"   - Reason: {decision.get('reasoning')[:100]}...")

        if obs.technician_log_markdown:
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append(obs.technician_log_markdown)

        return "\n".join(lines)

    def get_health(self) -> Dict[str, Any]:
        """Get dashboard health status."""
        return {
            "status": "healthy",
            "stored_requests": len(self._requests),
            "max_requests": self.max_requests,
            "memory_usage_percent": len(self._requests) / self.max_requests * 100,
        }


# Global dashboard instance
_dashboard: Optional[ObservabilityDashboard] = None


def get_observability_dashboard() -> ObservabilityDashboard:
    """Get the global observability dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = ObservabilityDashboard()
    return _dashboard


def create_request_observability(
    request_id: str,
    query: str,
    preset: str = "balanced"
) -> RequestObservability:
    """Factory function to create a new RequestObservability."""
    return RequestObservability(
        request_id=request_id,
        query=query,
        preset=preset
    )


class ObservabilityAggregator:
    """
    Helper class to aggregate observability from multiple sources.

    Usage:
        aggregator = ObservabilityAggregator(request_id, query, preset)

        # Add data from various loggers
        aggregator.add_decisions(decision_logger.decisions)
        aggregator.add_context_flow(context_tracker.get_flow_summary())
        aggregator.add_llm_calls(llm_logger.get_call_summary())
        aggregator.add_scratchpad(scratchpad_observer.get_change_summary())
        aggregator.add_confidence(confidence_breakdown)
        aggregator.add_technician_log(tech_log.to_markdown())

        # Finalize and store
        obs = aggregator.finalize(success=True, duration_ms=1234)
        dashboard.store_request(obs)
    """

    def __init__(self, request_id: str, query: str, preset: str = "balanced"):
        self.obs = RequestObservability(
            request_id=request_id,
            query=query,
            preset=preset
        )
        self._agents_seen: set = set()

    def add_decisions(self, decisions: List[Any]):
        """Add decisions from DecisionLogger."""
        for d in decisions:
            if hasattr(d, 'to_log_dict'):
                self.obs.decisions.append(d.to_log_dict())
            elif isinstance(d, dict):
                self.obs.decisions.append(d)

            # Track agent
            agent = d.agent_name if hasattr(d, 'agent_name') else d.get('agent_name', '')
            if agent and agent not in self._agents_seen:
                self._agents_seen.add(agent)
                self.obs.agents_executed.append(agent)

        self.obs.decision_count = len(self.obs.decisions)

    def add_context_flow(self, flow_summary: Dict[str, Any]):
        """Add context flow from ContextFlowTracker."""
        self.obs.context_transfers = flow_summary.get("transfers", [])
        self.obs.total_tokens_transferred = flow_summary.get("total_tokens_transferred", 0)

    def add_llm_calls(self, call_summary: Dict[str, Any]):
        """Add LLM call data from LLMCallLogger."""
        self.obs.llm_calls = call_summary.get("call_chain", [])
        self.obs.total_llm_latency_ms = call_summary.get("total_latency_ms", 0)
        self.obs.total_input_tokens = call_summary.get("total_input_tokens", 0)
        self.obs.total_output_tokens = call_summary.get("total_output_tokens", 0)

    def add_scratchpad(self, scratchpad_summary: Dict[str, Any]):
        """Add scratchpad data from ScratchpadObserver."""
        self.obs.scratchpad_changes = scratchpad_summary.get("change_timeline", [])
        final_state = scratchpad_summary.get("final_state", {})
        self.obs.findings_count = final_state.get("findings", 0)
        self.obs.questions_answered = final_state.get("questions_answered", 0)

    def add_confidence(self, breakdown: Any):
        """Add confidence data from ConfidenceLogger."""
        if hasattr(breakdown, 'to_dict'):
            self.obs.confidence_breakdown = breakdown.to_dict()
            self.obs.final_confidence = breakdown.final_confidence
            self.obs.confidence_level = breakdown.confidence_level
        elif isinstance(breakdown, dict):
            self.obs.confidence_breakdown = breakdown
            self.obs.final_confidence = breakdown.get("final_confidence", 0.0)
            self.obs.confidence_level = breakdown.get("confidence_level", "unknown")

    def add_technician_log(self, markdown: str):
        """Add technician log markdown."""
        self.obs.technician_log_markdown = markdown

    def add_feature_status(self, feature: str, enabled: bool, reason: str = ""):
        """Track feature enabled/skipped."""
        if enabled:
            if feature not in self.obs.features_enabled:
                self.obs.features_enabled.append(feature)
        else:
            if feature not in self.obs.features_skipped:
                self.obs.features_skipped.append(f"{feature}: {reason}" if reason else feature)

    def finalize(
        self,
        success: bool,
        duration_ms: int,
        error: Optional[str] = None
    ) -> RequestObservability:
        """Finalize and return the aggregated observability."""
        self.obs.success = success
        self.obs.total_duration_ms = duration_ms
        self.obs.error_message = error
        return self.obs
