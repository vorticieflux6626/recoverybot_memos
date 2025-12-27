"""
Progress Update Tools for AIME-Style Agent Progress Reporting.

This module provides tools that agents can use to report their progress
back to the DynamicPlanner, enabling real-time task tracking and
adaptive re-planning.

Based on AIME (ByteDance) research:
- Agents proactively report status changes
- Progress Management is the SSOT for task state
- Enables tight feedback loop for dynamic planning

Usage:
    # In agent execution
    from agentic.progress_tools import ProgressReporter

    reporter = ProgressReporter(scratchpad, planner)

    # Report progress
    await reporter.report_progress(
        task_id="t1",
        status="in_progress",
        message="Found 3 relevant sources",
        artifacts=["search_results.json"]
    )

    # Report completion
    await reporter.report_completed(
        task_id="t1",
        output={"sources": [...]},
        artifacts=["analysis.json"]
    )

    # Report failure with reason
    await reporter.report_failed(
        task_id="t1",
        reason="No relevant results found",
        should_retry=True
    )
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
import logging
import json

logger = logging.getLogger("agentic.progress_tools")


class ProgressStatus(str, Enum):
    """Status values for progress updates"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class ProgressUpdate:
    """A progress update from an agent"""
    task_id: str
    status: ProgressStatus
    message: str = ""
    output: Optional[Any] = None
    artifacts: List[str] = field(default_factory=list)
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "message": self.message,
            "output": self.output,
            "artifacts": self.artifacts,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class ProgressReporter:
    """
    Agent-side progress reporter that updates scratchpad and optionally triggers replanning.

    This is the primary interface agents use to report their progress.
    """

    def __init__(
        self,
        scratchpad: 'AgenticScratchpad',
        planner: Optional['DynamicPlanner'] = None,
        on_progress: Optional[Callable[[ProgressUpdate], Awaitable[None]]] = None
    ):
        """
        Args:
            scratchpad: The shared scratchpad to update
            planner: Optional DynamicPlanner for triggering re-planning
            on_progress: Optional async callback for progress events
        """
        self.scratchpad = scratchpad
        self.planner = planner
        self.on_progress = on_progress
        self._updates: List[ProgressUpdate] = []

    async def report_progress(
        self,
        task_id: str,
        status: ProgressStatus,
        message: str = "",
        output: Optional[Any] = None,
        artifacts: Optional[List[str]] = None,
        duration_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressUpdate:
        """
        Report progress on a task.

        This updates the scratchpad and optionally triggers the planner.

        Args:
            task_id: ID of the task being updated
            status: New status of the task
            message: Human-readable progress message
            output: Optional output/result data
            artifacts: Optional list of artifact IDs produced
            duration_ms: Time spent so far in milliseconds
            metadata: Additional metadata

        Returns:
            The ProgressUpdate that was recorded
        """
        update = ProgressUpdate(
            task_id=task_id,
            status=status,
            message=message,
            output=output,
            artifacts=artifacts or [],
            duration_ms=duration_ms,
            metadata=metadata or {}
        )

        self._updates.append(update)

        # Update scratchpad
        self.scratchpad.update_task_status(
            task_id=task_id,
            status=status.value,
            message=message,
            artifacts=artifacts
        )

        # Add agent note for visibility
        self.scratchpad.add_agent_note(
            agent="progress_reporter",
            action_taken=f"Updated task {task_id} to {status.value}",
            observation=message,
            recommendation=""
        )

        # Call optional callback
        if self.on_progress:
            try:
                await self.on_progress(update)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

        logger.debug(f"Progress update: {task_id} -> {status.value}: {message}")

        return update

    async def report_started(
        self,
        task_id: str,
        message: str = "Starting execution"
    ) -> ProgressUpdate:
        """Convenience method to report task started"""
        return await self.report_progress(
            task_id=task_id,
            status=ProgressStatus.IN_PROGRESS,
            message=message
        )

    async def report_completed(
        self,
        task_id: str,
        output: Any,
        message: str = "Task completed successfully",
        artifacts: Optional[List[str]] = None,
        duration_ms: int = 0
    ) -> ProgressUpdate:
        """
        Report task completion.

        This also records the execution in scratchpad history.
        """
        update = await self.report_progress(
            task_id=task_id,
            status=ProgressStatus.COMPLETED,
            message=message,
            output=output,
            artifacts=artifacts,
            duration_ms=duration_ms
        )

        # Record execution for planner feedback
        self.scratchpad.record_execution(
            task_id=task_id,
            action_type="task",
            success=True,
            output=output,
            duration_ms=duration_ms,
            artifacts=artifacts
        )

        return update

    async def report_failed(
        self,
        task_id: str,
        reason: str,
        error: Optional[Exception] = None,
        should_retry: bool = False,
        duration_ms: int = 0
    ) -> ProgressUpdate:
        """
        Report task failure.

        Args:
            task_id: ID of the failed task
            reason: Human-readable failure reason
            error: Optional exception that caused failure
            should_retry: Whether the task should be retried
            duration_ms: Time spent before failure
        """
        metadata = {
            "should_retry": should_retry,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None
        }

        update = await self.report_progress(
            task_id=task_id,
            status=ProgressStatus.FAILED,
            message=reason,
            metadata=metadata,
            duration_ms=duration_ms
        )

        # Record failed execution
        self.scratchpad.record_execution(
            task_id=task_id,
            action_type="task",
            success=False,
            output={"reason": reason, "should_retry": should_retry},
            duration_ms=duration_ms
        )

        return update

    async def report_blocked(
        self,
        task_id: str,
        blocking_task_ids: List[str],
        message: str = ""
    ) -> ProgressUpdate:
        """Report task is blocked by dependencies"""
        return await self.report_progress(
            task_id=task_id,
            status=ProgressStatus.BLOCKED,
            message=message or f"Blocked by: {', '.join(blocking_task_ids)}",
            metadata={"blocking_tasks": blocking_task_ids}
        )

    async def report_skipped(
        self,
        task_id: str,
        reason: str = "No longer needed"
    ) -> ProgressUpdate:
        """Report task was skipped"""
        return await self.report_progress(
            task_id=task_id,
            status=ProgressStatus.SKIPPED,
            message=reason
        )

    def get_all_updates(self) -> List[ProgressUpdate]:
        """Get all progress updates recorded by this reporter"""
        return self._updates.copy()

    def get_updates_for_task(self, task_id: str) -> List[ProgressUpdate]:
        """Get progress updates for a specific task"""
        return [u for u in self._updates if u.task_id == task_id]


class ProgressTool:
    """
    Tool definition for agent function calling.

    This defines the tool schema that LLM agents can use to report progress.
    Compatible with Ollama/OpenAI function calling format.
    """

    TOOL_DEFINITION = {
        "type": "function",
        "function": {
            "name": "report_progress",
            "description": "Report progress on the current task to the planner. Use this to update task status, report findings, or signal completion/failure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the task being updated (e.g., 't1', 't2.1')"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["in_progress", "completed", "failed", "blocked", "skipped"],
                        "description": "New status of the task"
                    },
                    "message": {
                        "type": "string",
                        "description": "Human-readable progress message describing what was done or found"
                    },
                    "findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of key findings or results"
                    },
                    "should_retry": {
                        "type": "boolean",
                        "description": "If failed, whether the task should be retried"
                    }
                },
                "required": ["task_id", "status", "message"]
            }
        }
    }

    @staticmethod
    def parse_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a tool call from LLM output"""
        args = tool_call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return {
            "task_id": args.get("task_id", ""),
            "status": args.get("status", "in_progress"),
            "message": args.get("message", ""),
            "findings": args.get("findings", []),
            "should_retry": args.get("should_retry", False)
        }


class ProgressAggregator:
    """
    Aggregates progress updates across multiple agents for dashboard/UI.

    This can be used by the orchestrator to track overall progress
    and provide updates to clients (e.g., via SSE).
    """

    def __init__(self):
        self._updates: List[ProgressUpdate] = []
        self._task_states: Dict[str, ProgressStatus] = {}
        self._listeners: List[Callable[[ProgressUpdate], Awaitable[None]]] = []

    def add_listener(self, listener: Callable[[ProgressUpdate], Awaitable[None]]) -> None:
        """Add a listener for progress updates"""
        self._listeners.append(listener)

    async def record_update(self, update: ProgressUpdate) -> None:
        """Record a progress update and notify listeners"""
        self._updates.append(update)
        self._task_states[update.task_id] = update.status

        # Notify all listeners
        for listener in self._listeners:
            try:
                await listener(update)
            except Exception as e:
                logger.warning(f"Progress listener failed: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all progress"""
        total = len(self._task_states)
        if total == 0:
            return {"total": 0, "by_status": {}, "progress": 0.0}

        by_status = {}
        for status in self._task_states.values():
            by_status[status.value] = by_status.get(status.value, 0) + 1

        completed = by_status.get("completed", 0)
        skipped = by_status.get("skipped", 0)
        progress = (completed + skipped) / total if total > 0 else 0.0

        return {
            "total": total,
            "by_status": by_status,
            "progress": progress,
            "completed": completed,
            "failed": by_status.get("failed", 0),
            "in_progress": by_status.get("in_progress", 0),
            "pending": by_status.get("pending", 0)
        }

    def get_recent_updates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent updates as dicts"""
        return [u.to_dict() for u in self._updates[-limit:]]

    def clear(self) -> None:
        """Clear all recorded updates"""
        self._updates.clear()
        self._task_states.clear()


# Export tool definition for agent prompts
PROGRESS_TOOL_PROMPT = """
You have access to a progress reporting tool:

**report_progress**: Report your progress on the current task
Parameters:
- task_id (required): The ID of the task you're working on
- status (required): One of: in_progress, completed, failed, blocked, skipped
- message (required): Describe what you did or found
- findings (optional): List of key findings or results
- should_retry (optional): If failed, whether to retry

Use this tool to keep the planner informed of your progress.
Report 'in_progress' when starting, 'completed' when done, or 'failed' with reason if something goes wrong.
"""
