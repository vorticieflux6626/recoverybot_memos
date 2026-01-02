"""
Scratchpad Observer - Track state changes in the agentic scratchpad.

Part of P1 Observability Enhancement (OBSERVABILITY_IMPROVEMENT_PLAN.md).

Provides detailed logging for all scratchpad state changes:
- Finding additions and updates
- Question status transitions
- Entity extractions
- Gap detection events
- Contradiction tracking
- Public/private space operations

Created: 2026-01-02
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ScratchpadOperation(str, Enum):
    """Types of operations on the scratchpad."""
    ADD_FINDING = "add_finding"
    UPDATE_FINDING = "update_finding"
    ADD_QUESTION = "add_question"
    UPDATE_QUESTION = "update_question"
    ANSWER_QUESTION = "answer_question"
    ADD_ENTITY = "add_entity"
    UPDATE_ENTITY = "update_entity"
    ADD_ENTITY_RELATION = "add_entity_relation"
    DETECT_GAP = "detect_gap"
    DETECT_CONTRADICTION = "detect_contradiction"
    RESOLVE_CONTRADICTION = "resolve_contradiction"
    WRITE_PUBLIC = "write_public"
    WRITE_PRIVATE = "write_private"
    ADD_NOTE = "add_note"
    SET_STATUS = "set_status"
    INITIALIZE = "initialize"


@dataclass
class ScratchpadChange:
    """Record of a single scratchpad state change."""
    request_id: str
    operation: str
    agent: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Change details
    details: Dict[str, Any] = field(default_factory=dict)

    # State snapshot after change
    state_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "agent": self.agent,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "state": self.state_snapshot
        }

    def to_sse_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SSE event (compact)."""
        return {
            "operation": self.operation,
            "agent": self.agent,
            "details": {k: v for k, v in self.details.items() if k not in ['content', 'full_text']},
            "state": self.state_snapshot
        }


class ScratchpadObserver:
    """
    Observe and log all scratchpad state changes.

    Usage:
        observer = ScratchpadObserver(request_id="req-123")

        # Attach to scratchpad
        scratchpad.observer = observer

        # Changes are automatically logged via scratchpad methods
        # Or manually record:
        await observer.record_change(
            operation=ScratchpadOperation.ADD_FINDING,
            agent="verifier",
            details={"finding_id": "f1", "confidence": 0.9}
        )

        # Get summary
        summary = observer.get_change_summary()
    """

    def __init__(
        self,
        request_id: str,
        emitter: Optional[Any] = None,
        verbose: bool = False
    ):
        self.request_id = request_id
        self.emitter = emitter
        self.verbose = verbose
        self.changes: List[ScratchpadChange] = []

        # Aggregate metrics
        self._changes_by_operation: Dict[str, int] = {}
        self._changes_by_agent: Dict[str, int] = {}

        # Current state trackers
        self._findings_count = 0
        self._questions_total = 0
        self._questions_answered = 0
        self._entities_count = 0
        self._gaps_count = 0
        self._contradictions_count = 0

    def _get_state_snapshot(self) -> Dict[str, Any]:
        """Get current state snapshot for logging."""
        return {
            "findings": self._findings_count,
            "questions_total": self._questions_total,
            "questions_answered": self._questions_answered,
            "entities": self._entities_count,
            "gaps": self._gaps_count,
            "contradictions": self._contradictions_count
        }

    async def record_change(
        self,
        operation: str,
        agent: str,
        details: Optional[Dict[str, Any]] = None,
        state_update: Optional[Dict[str, Any]] = None
    ) -> ScratchpadChange:
        """
        Record a scratchpad state change.

        Args:
            operation: Type of operation (use ScratchpadOperation enum)
            agent: Name of the agent making the change
            details: Operation-specific details
            state_update: Optional state counters to update

        Returns:
            The recorded ScratchpadChange
        """
        # Update state trackers if provided
        if state_update:
            for key, value in state_update.items():
                if hasattr(self, f"_{key}"):
                    setattr(self, f"_{key}", value)

        change = ScratchpadChange(
            request_id=self.request_id,
            operation=operation,
            agent=agent,
            details=details or {},
            state_snapshot=self._get_state_snapshot()
        )

        # Update aggregates
        self._changes_by_operation[operation] = self._changes_by_operation.get(operation, 0) + 1
        self._changes_by_agent[agent] = self._changes_by_agent.get(agent, 0) + 1

        # Store change
        self.changes.append(change)

        # Log
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logger.log(
            log_level,
            f"[{self.request_id}] Scratchpad: {agent}.{operation} | "
            f"findings={self._findings_count} | "
            f"questions={self._questions_answered}/{self._questions_total} | "
            f"entities={self._entities_count}"
        )

        # Emit SSE event
        if self.emitter:
            try:
                from agentic.events import SearchEvent, EventType
                await self.emitter.emit(SearchEvent(
                    event_type=EventType.SCRATCHPAD_UPDATED,
                    request_id=self.request_id,
                    data=change.to_sse_dict()
                ))
            except Exception as e:
                logger.debug(f"Failed to emit scratchpad event: {e}")

        return change

    def record_change_sync(
        self,
        operation: str,
        agent: str,
        details: Optional[Dict[str, Any]] = None,
        state_update: Optional[Dict[str, Any]] = None
    ) -> ScratchpadChange:
        """
        Synchronous version of record_change (no SSE emission).
        """
        if state_update:
            for key, value in state_update.items():
                if hasattr(self, f"_{key}"):
                    setattr(self, f"_{key}", value)

        change = ScratchpadChange(
            request_id=self.request_id,
            operation=operation,
            agent=agent,
            details=details or {},
            state_snapshot=self._get_state_snapshot()
        )

        self._changes_by_operation[operation] = self._changes_by_operation.get(operation, 0) + 1
        self._changes_by_agent[agent] = self._changes_by_agent.get(agent, 0) + 1
        self.changes.append(change)

        logger.info(
            f"[{self.request_id}] Scratchpad: {agent}.{operation} | "
            f"findings={self._findings_count} | "
            f"questions={self._questions_answered}/{self._questions_total}"
        )

        return change

    # Convenience methods for common operations

    async def on_finding_added(
        self,
        agent: str,
        finding_id: str,
        content_preview: str,
        confidence: float,
        source_count: int,
        conflicts_with: Optional[List[str]] = None
    ):
        """Record a finding addition."""
        self._findings_count += 1
        if conflicts_with:
            self._contradictions_count += len(conflicts_with)

        await self.record_change(
            operation=ScratchpadOperation.ADD_FINDING,
            agent=agent,
            details={
                "finding_id": finding_id,
                "content_preview": content_preview[:100] if content_preview else "",
                "confidence": confidence,
                "source_count": source_count,
                "conflicts_with": conflicts_with or []
            }
        )

    async def on_question_added(
        self,
        agent: str,
        question_id: str,
        question_text: str,
        priority: str
    ):
        """Record a question addition."""
        self._questions_total += 1

        await self.record_change(
            operation=ScratchpadOperation.ADD_QUESTION,
            agent=agent,
            details={
                "question_id": question_id,
                "question_text": question_text[:100] if question_text else "",
                "priority": priority
            }
        )

    async def on_question_answered(
        self,
        agent: str,
        question_id: str,
        question_text: str,
        old_status: str,
        new_status: str
    ):
        """Record a question status change."""
        if new_status == "answered":
            self._questions_answered += 1

        await self.record_change(
            operation=ScratchpadOperation.ANSWER_QUESTION,
            agent=agent,
            details={
                "question_id": question_id,
                "question_text": question_text[:50] if question_text else "",
                "old_status": old_status,
                "new_status": new_status
            }
        )

    async def on_entity_added(
        self,
        agent: str,
        entity_id: str,
        entity_type: str,
        entity_name: str
    ):
        """Record an entity addition."""
        self._entities_count += 1

        await self.record_change(
            operation=ScratchpadOperation.ADD_ENTITY,
            agent=agent,
            details={
                "entity_id": entity_id,
                "entity_type": entity_type,
                "entity_name": entity_name
            }
        )

    async def on_gap_detected(
        self,
        agent: str,
        gap_description: str,
        related_questions: Optional[List[str]] = None
    ):
        """Record a gap detection."""
        self._gaps_count += 1

        await self.record_change(
            operation=ScratchpadOperation.DETECT_GAP,
            agent=agent,
            details={
                "gap_description": gap_description[:200] if gap_description else "",
                "related_questions": related_questions or []
            }
        )

    async def on_contradiction_detected(
        self,
        agent: str,
        finding_a: str,
        finding_b: str,
        description: str
    ):
        """Record a contradiction detection."""
        self._contradictions_count += 1

        await self.record_change(
            operation=ScratchpadOperation.DETECT_CONTRADICTION,
            agent=agent,
            details={
                "finding_a": finding_a,
                "finding_b": finding_b,
                "description": description[:200] if description else ""
            }
        )

    def get_change_summary(self) -> Dict[str, Any]:
        """
        Generate summary of all scratchpad changes.

        Returns:
            Summary dict suitable for response metadata
        """
        if not self.changes:
            return {
                "total_changes": 0,
                "final_state": self._get_state_snapshot(),
                "by_operation": {},
                "by_agent": {}
            }

        return {
            "total_changes": len(self.changes),
            "final_state": self._get_state_snapshot(),
            "by_operation": dict(self._changes_by_operation),
            "by_agent": dict(self._changes_by_agent),
            "change_timeline": [
                {
                    "op": c.operation,
                    "agent": c.agent,
                    "ts": c.timestamp.isoformat()
                }
                for c in self.changes[:30]  # Limit for response size
            ]
        }

    def get_changes_by_operation(self, operation: str) -> List[ScratchpadChange]:
        """Get all changes of a specific operation type."""
        return [c for c in self.changes if c.operation == operation]

    def get_changes_by_agent(self, agent: str) -> List[ScratchpadChange]:
        """Get all changes made by a specific agent."""
        return [c for c in self.changes if c.agent == agent]

    def get_finding_history(self) -> List[ScratchpadChange]:
        """Get all finding-related changes."""
        ops = {ScratchpadOperation.ADD_FINDING, ScratchpadOperation.UPDATE_FINDING}
        return [c for c in self.changes if c.operation in ops]

    def get_question_history(self) -> List[ScratchpadChange]:
        """Get all question-related changes."""
        ops = {
            ScratchpadOperation.ADD_QUESTION,
            ScratchpadOperation.UPDATE_QUESTION,
            ScratchpadOperation.ANSWER_QUESTION
        }
        return [c for c in self.changes if c.operation in ops]

    def export_for_debugging(self) -> List[Dict[str, Any]]:
        """Export all changes in full detail for debugging."""
        return [c.to_log_dict() for c in self.changes]


def get_scratchpad_observer(
    request_id: str,
    emitter: Optional[Any] = None,
    verbose: bool = False
) -> ScratchpadObserver:
    """
    Factory function to get a ScratchpadObserver instance.

    Args:
        request_id: Unique request identifier
        emitter: Optional SSE event emitter
        verbose: Enable verbose logging

    Returns:
        ScratchpadObserver instance
    """
    return ScratchpadObserver(request_id, emitter, verbose)
