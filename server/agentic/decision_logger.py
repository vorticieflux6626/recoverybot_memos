"""
Decision Logger - Centralized agent decision tracking for observability.

Part of P0 Observability Enhancement (OBSERVABILITY_IMPROVEMENT_PLAN.md).

Provides structured logging for all agent decision points in the agentic pipeline,
enabling:
- Real-time SSE events for decision visibility
- Engineering-level debugging with reasoning traces
- Technician-friendly decision summaries

Created: 2026-01-02
"""

import logging
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Types of decisions agents can make."""
    CLASSIFICATION = "classification"  # Routing, categorization
    ACTION = "action"  # Executing an operation
    SKIP = "skip"  # Deciding not to act
    FALLBACK = "fallback"  # Using alternative path
    REFINEMENT = "refinement"  # Improving previous output
    HALT = "halt"  # Stopping iteration
    EVALUATION = "evaluation"  # Quality assessment


class AgentName(str, Enum):
    """Known agent names in the pipeline."""
    QUERY_CLASSIFIER = "query_classifier"
    ANALYZER = "analyzer"
    SEARCHER = "searcher"
    SCRAPER = "scraper"
    VERIFIER = "verifier"
    SYNTHESIZER = "synthesizer"
    CRAG = "crag"
    SELF_RAG = "self_rag"
    DYLAN = "dylan"
    ENTROPY_MONITOR = "entropy_monitor"
    EXPERIENCE_DISTILLER = "experience_distiller"
    FLARE = "flare"
    ENTITY_TRACKER = "entity_tracker"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentDecision:
    """Structured logging for any agent decision point."""
    request_id: str
    agent_name: str  # analyzer, synthesizer, verifier, crag, self_rag, etc.
    decision_type: str  # classification, action, skip, fallback, refinement
    decision_made: str  # The actual decision
    reasoning: str  # Why this decision was made
    alternatives_considered: List[str]  # What else was considered
    confidence: float  # 0-1 confidence in decision
    timestamp: datetime = field(default_factory=datetime.now)

    # Context metrics
    context_size_tokens: int = 0
    context_size_chars: int = 0

    # Decision-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "request_id": self.request_id,
            "agent": self.agent_name,
            "decision_type": self.decision_type,
            "decision": self.decision_made,
            "reasoning": self.reasoning[:500] if self.reasoning else "",
            "alternatives": self.alternatives_considered[:5],
            "confidence": round(self.confidence, 3),
            "context_tokens": self.context_size_tokens,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    def to_sse_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SSE event (truncated for real-time display)."""
        return {
            "agent": self.agent_name,
            "decision_type": self.decision_type,
            "decision": self.decision_made[:100],
            "reasoning": self.reasoning[:200] if self.reasoning else "",
            "confidence": round(self.confidence, 3)
        }


class DecisionLogger:
    """
    Centralized decision logging with SSE emission.

    Usage:
        logger = DecisionLogger(request_id="req-123")

        # Log a decision
        await logger.log_decision(
            agent_name="analyzer",
            decision_type=DecisionType.CLASSIFICATION,
            decision_made="agentic_search",
            reasoning="Query requires multi-step research",
            alternatives=["direct_answer", "web_search"],
            confidence=0.85
        )

        # Get summary for response
        summary = logger.get_decision_summary()
    """

    def __init__(
        self,
        request_id: str,
        emitter: Optional[Any] = None,  # EventEmitter from models.py
        verbose: bool = False
    ):
        self.request_id = request_id
        self.emitter = emitter
        self.verbose = verbose
        self.decisions: List[AgentDecision] = []
        self._decision_count_by_agent: Dict[str, int] = {}

    async def log_decision(
        self,
        agent_name: str,
        decision_type: str,
        decision_made: str,
        reasoning: str = "",
        alternatives: Optional[List[str]] = None,
        confidence: float = 1.0,
        context_size_tokens: int = 0,
        context_size_chars: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentDecision:
        """
        Log an agent decision with full context.

        Args:
            agent_name: Name of the agent making the decision
            decision_type: Type of decision (use DecisionType enum)
            decision_made: The actual decision value
            reasoning: Human-readable explanation of why
            alternatives: Other options that were considered
            confidence: 0-1 confidence in the decision
            context_size_tokens: Approximate input token count
            context_size_chars: Input character count
            metadata: Additional decision-specific data

        Returns:
            The logged AgentDecision
        """
        decision = AgentDecision(
            request_id=self.request_id,
            agent_name=agent_name,
            decision_type=decision_type,
            decision_made=decision_made,
            reasoning=reasoning,
            alternatives_considered=alternatives or [],
            confidence=confidence,
            context_size_tokens=context_size_tokens,
            context_size_chars=context_size_chars,
            metadata=metadata or {}
        )

        # 1. Structured log
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logger.log(
            log_level,
            f"[{self.request_id}] DECISION: {agent_name}.{decision_type} → {decision_made} "
            f"(conf={confidence:.2f}, reason={reasoning[:100]}...)",
            extra={"decision": decision.to_log_dict(), "structured": True}
        )

        # 2. SSE event for real-time visibility
        if self.emitter:
            try:
                # Import here to avoid circular imports
                from agentic.models import SearchEvent, EventType
                await self.emitter.emit(SearchEvent(
                    event_type=EventType.DECISION_POINT,
                    request_id=self.request_id,
                    data=decision.to_sse_dict()
                ))
            except Exception as e:
                logger.debug(f"Failed to emit decision SSE event: {e}")

        # 3. Store for request summary
        self.decisions.append(decision)
        self._decision_count_by_agent[agent_name] = \
            self._decision_count_by_agent.get(agent_name, 0) + 1

        return decision

    def log_decision_sync(
        self,
        agent_name: str,
        decision_type: str,
        decision_made: str,
        reasoning: str = "",
        alternatives: Optional[List[str]] = None,
        confidence: float = 1.0,
        context_size_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentDecision:
        """
        Synchronous version of log_decision (no SSE emission).

        Use this in synchronous contexts where async is not available.
        """
        decision = AgentDecision(
            request_id=self.request_id,
            agent_name=agent_name,
            decision_type=decision_type,
            decision_made=decision_made,
            reasoning=reasoning,
            alternatives_considered=alternatives or [],
            confidence=confidence,
            context_size_tokens=context_size_tokens,
            metadata=metadata or {}
        )

        logger.info(
            f"[{self.request_id}] DECISION: {agent_name}.{decision_type} → {decision_made} "
            f"(conf={confidence:.2f})"
        )

        self.decisions.append(decision)
        self._decision_count_by_agent[agent_name] = \
            self._decision_count_by_agent.get(agent_name, 0) + 1

        return decision

    async def log_feature_status(
        self,
        feature_name: str,
        enabled: bool,
        reason: str = ""
    ):
        """
        Log whether a feature was enabled or skipped.

        Args:
            feature_name: Name of the feature/modality
            enabled: Whether the feature will run
            reason: Why it was enabled/disabled
        """
        decision_type = DecisionType.ACTION if enabled else DecisionType.SKIP
        decision_made = "enabled" if enabled else "disabled"

        await self.log_decision(
            agent_name=AgentName.ORCHESTRATOR,
            decision_type=decision_type,
            decision_made=f"feature:{feature_name}={decision_made}",
            reasoning=reason or ("feature enabled in config" if enabled else "feature disabled in config"),
            confidence=1.0,
            metadata={"feature": feature_name, "enabled": enabled}
        )

    def get_decision_summary(self) -> Dict[str, Any]:
        """
        Generate summary of all decisions for this request.

        Returns:
            Summary dict suitable for response metadata
        """
        if not self.decisions:
            return {
                "total_decisions": 0,
                "by_agent": {},
                "decision_chain": []
            }

        # Build decision chain (simplified view)
        chain = []
        for d in self.decisions:
            chain.append({
                "agent": d.agent_name,
                "type": d.decision_type,
                "decision": d.decision_made[:50],
                "confidence": round(d.confidence, 2)
            })

        # Calculate confidence stats
        confidences = [d.confidence for d in self.decisions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_decisions": len(self.decisions),
            "by_agent": dict(self._decision_count_by_agent),
            "avg_confidence": round(avg_confidence, 3),
            "min_confidence": round(min(confidences), 3) if confidences else 0,
            "decision_chain": chain[:20]  # Limit to first 20 for response size
        }

    def get_decisions_by_agent(self, agent_name: str) -> List[AgentDecision]:
        """Get all decisions made by a specific agent."""
        return [d for d in self.decisions if d.agent_name == agent_name]

    def get_decisions_by_type(self, decision_type: str) -> List[AgentDecision]:
        """Get all decisions of a specific type."""
        return [d for d in self.decisions if d.decision_type == decision_type]

    def export_for_debugging(self) -> List[Dict[str, Any]]:
        """Export all decisions in full detail for debugging."""
        return [d.to_log_dict() for d in self.decisions]


# Global instance for simple usage (optional)
_default_logger: Optional[DecisionLogger] = None


def get_decision_logger(
    request_id: str,
    emitter: Optional[Any] = None,
    verbose: bool = False
) -> DecisionLogger:
    """
    Factory function to get a DecisionLogger instance.

    Args:
        request_id: Unique request identifier
        emitter: Optional SSE event emitter
        verbose: Enable verbose logging

    Returns:
        DecisionLogger instance
    """
    return DecisionLogger(request_id, emitter, verbose)
