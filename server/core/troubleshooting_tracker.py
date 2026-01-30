"""
Troubleshooting Task Tracker

Automatically tracks pipeline stage execution within troubleshooting sessions.
Integrates with the agentic orchestrator via hooks to record task metrics.

Usage:
    from core.troubleshooting_tracker import TroubleshootingTaskTracker

    # In orchestrator initialization
    tracker = TroubleshootingTaskTracker(session_id="...")

    # Track pipeline stages
    async with tracker.track_task("_search_technical_docs") as task_context:
        results = await self._search_technical_docs(query)
        task_context.set_output({"result_count": len(results)})
        task_context.set_metrics({"latency_ms": 1234})

See: QUEST_SYSTEM_REIMPLEMENTATION_PLAN.md for architecture details.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.troubleshooting import TaskState, TaskExecutionType
from config.database import get_async_db

# Type hint for EventEmitter - actual import is deferred to avoid circular imports
if TYPE_CHECKING:
    from agentic.events import EventEmitter

logger = logging.getLogger(__name__)

# Number of workflow tasks that the Android client expects
WORKFLOW_TASK_COUNT = 6  # Query Analysis, Entity Extraction, Technical Doc Search, Cross-Domain Validation, Synthesis, User Verification


# =============================================================================
# PIPELINE HOOK DEFINITIONS
# =============================================================================

# Maps pipeline method names to task metadata
PIPELINE_HOOKS = {
    "_analyze_query": {
        "name": "Query Analysis",
        "workflow_task": "Query Analysis",  # Maps to workflow task name
        "description": "Analyze query to extract entities and intent",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 30,
        "verification_criteria": {},
    },
    "_extract_entities": {
        "name": "Entity Extraction",
        "workflow_task": "Entity Extraction",
        "description": "Extract technical entities from query",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 30,
        "verification_criteria": {},
    },
    "_search_technical_docs": {
        "name": "Technical Doc Search",
        "workflow_task": "Technical Doc Search",  # Maps to workflow task name
        "description": "Search technical documentation for relevant context",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 60,
        "verification_criteria": {
            "min_results": 1,
            "min_context_length": 100,
        },
    },
    "_traverse_graph": {
        "name": "Graph Traversal",
        "workflow_task": "Technical Doc Search",  # Part of doc search
        "description": "Navigate knowledge graph to find diagnostic paths",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 30,
        "verification_criteria": {
            "min_paths": 1,
        },
    },
    "_ground_entities": {
        "name": "Entity Validation",
        "workflow_task": "Entity Extraction",  # Part of entity processing
        "description": "Verify entities against knowledge base",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 30,
        "verification_criteria": {
            "max_fabricated_ratio": 0.2,
        },
    },
    "_validate_cross_domain": {
        "name": "Cross-Domain Validation",
        "workflow_task": "Cross-Domain Validation",  # Direct match
        "description": "Check for invalid cross-domain claims",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 20,
        "verification_criteria": {
            "max_violations": 0,
        },
    },
    "_synthesize": {
        "name": "Synthesis",
        "workflow_task": "Synthesis",  # Direct match
        "description": "Generate comprehensive response from context",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 120,
        "verification_criteria": {
            "min_confidence": 0.5,
            "min_citations": 1,
        },
    },
    "_generate_diagram": {
        "name": "Diagram Generation",
        "workflow_task": "Synthesis",  # Part of synthesis output
        "description": "Generate relevant technical diagrams",
        "execution_type": TaskExecutionType.AUTOMATIC,
        "timeout_seconds": 30,
        "verification_criteria": {},
    },
    "path_selection": {
        "name": "Path Selection",
        "workflow_task": "User Verification",
        "description": "User selects diagnostic path to follow",
        "execution_type": TaskExecutionType.USER_ACTION,
        "timeout_seconds": 300,
        "verification_criteria": {},
    },
    "step_completion": {
        "name": "Step Completion",
        "workflow_task": "User Verification",
        "description": "User marks diagnostic step as complete",
        "execution_type": TaskExecutionType.USER_ACTION,
        "timeout_seconds": 600,
        "verification_criteria": {},
    },
    "resolution_verification": {
        "name": "Resolution Verification",
        "workflow_task": "User Verification",
        "description": "User confirms issue is resolved",
        "execution_type": TaskExecutionType.HYBRID,
        "timeout_seconds": 300,
        "verification_criteria": {},
    },
}

# Mapping from workflow task names to the primary pipeline hook that completes them
WORKFLOW_TASK_TO_HOOK = {
    "Query Analysis": "_analyze_query",
    "Entity Extraction": "_extract_entities",
    "Technical Doc Search": "_search_technical_docs",
    "Cross-Domain Validation": "_validate_cross_domain",
    "Synthesis": "_synthesize",
    "User Verification": "resolution_verification",
}


# =============================================================================
# TASK CONTEXT
# =============================================================================

@dataclass
class TaskContext:
    """
    Context object for tracking a single task execution.

    Collects input, output, and metrics during task execution.
    """
    hook_name: str
    task_name: str
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    error: Optional[str] = None
    verification_passed: Optional[bool] = None

    def set_input(self, data: Dict[str, Any]):
        """Set input data for the task."""
        self.input_data.update(data)

    def set_output(self, data: Dict[str, Any]):
        """Set output data for the task."""
        self.output_data.update(data)

    def set_metrics(self, data: Dict[str, Any]):
        """Set metrics for the task."""
        self.metrics.update(data)

    def set_error(self, error: str):
        """Record an error."""
        self.error = error

    def complete(self, verification_passed: Optional[bool] = None):
        """Mark task as complete."""
        self.completed_at = time.time()
        self.verification_passed = verification_passed

        # Auto-calculate latency
        if "latency_ms" not in self.metrics:
            self.metrics["latency_ms"] = int((self.completed_at - self.started_at) * 1000)

    @property
    def duration_seconds(self) -> float:
        """Get task duration in seconds."""
        end = self.completed_at or time.time()
        return end - self.started_at


# =============================================================================
# TROUBLESHOOTING TASK TRACKER
# =============================================================================

class TroubleshootingTaskTracker:
    """
    Tracks pipeline task execution within a troubleshooting session.

    Can be injected into the orchestrator to automatically record:
    - Task start/completion times
    - Input/output data summaries
    - Performance metrics
    - Verification results

    The tracker operates independently of the database session to avoid
    holding transactions open during long-running pipeline operations.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        enabled: bool = True,
        emitter: Optional["EventEmitter"] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize the task tracker.

        Args:
            session_id: Troubleshooting session ID to track
            user_id: User ID for attribution
            enabled: Whether tracking is enabled
            emitter: Optional EventEmitter for real-time SSE updates
            request_id: Request ID for SSE events
        """
        self.session_id = session_id
        self.user_id = user_id
        self.enabled = enabled and session_id is not None
        self.emitter = emitter
        self.request_id = request_id or session_id or "unknown"

        # In-memory tracking (flushed to DB periodically)
        self._task_contexts: List[TaskContext] = []
        self._current_task: Optional[TaskContext] = None

        # Track which workflow tasks have been completed (for SSE progress)
        self._completed_workflow_tasks: set = set()

        # Pipeline metadata
        self._pipeline_started_at: Optional[float] = None
        self._pipeline_completed_at: Optional[float] = None
        self._pipeline_metadata: Dict[str, Any] = {}

        logger.debug(
            f"TroubleshootingTaskTracker initialized: "
            f"session={session_id}, enabled={self.enabled}, has_emitter={emitter is not None}"
        )

    # =========================================================================
    # PIPELINE LIFECYCLE
    # =========================================================================

    async def start_pipeline(self, metadata: Optional[Dict[str, Any]] = None, query: str = ""):
        """Mark the start of a pipeline run and emit SSE event."""
        self._pipeline_started_at = time.time()
        self._pipeline_metadata = metadata or {}
        self._task_contexts = []
        self._completed_workflow_tasks = set()

        logger.debug(f"Pipeline started for session {self.session_id}")

        # Emit SSE event for real-time tracking
        if self.emitter and self.session_id:
            try:
                # Import locally to avoid circular import
                from agentic.events import troubleshooting_pipeline_started
                await self.emitter.emit(troubleshooting_pipeline_started(
                    request_id=self.request_id,
                    session_id=self.session_id,
                    task_total=WORKFLOW_TASK_COUNT,
                    query=query,
                    graph_line=self._build_graph_line()
                ))
            except Exception as e:
                logger.warning(f"Failed to emit pipeline started event: {e}")

    async def complete_pipeline(self, success: bool = True):
        """
        Mark pipeline completion, emit SSE event, and flush all tracked data to database.
        """
        self._pipeline_completed_at = time.time()

        # Calculate pipeline duration
        duration_ms = 0
        if self._pipeline_started_at:
            duration_ms = int((self._pipeline_completed_at - self._pipeline_started_at) * 1000)

        # Emit pipeline completed event for real-time tracking
        if self.emitter and self.session_id:
            try:
                # Import locally to avoid circular import
                from agentic.events import troubleshooting_pipeline_completed
                await self.emitter.emit(troubleshooting_pipeline_completed(
                    request_id=self.request_id,
                    session_id=self.session_id,
                    completed_count=len(self._completed_workflow_tasks),
                    task_total=WORKFLOW_TASK_COUNT,
                    success=success,
                    duration_ms=duration_ms,
                    graph_line=self._build_graph_line()
                ))
            except Exception as e:
                logger.warning(f"Failed to emit pipeline completed event: {e}")

        if not self.enabled:
            return

        try:
            await self._flush_to_database(success)
        except Exception as e:
            logger.error(f"Failed to flush tracking data: {e}")

    # =========================================================================
    # TASK TRACKING
    # =========================================================================

    @asynccontextmanager
    async def track_task(
        self,
        hook_name: str,
        input_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracking a single task execution.

        Usage:
            async with tracker.track_task("_search_technical_docs") as ctx:
                results = await search(...)
                ctx.set_output({"count": len(results)})
                ctx.set_metrics({"latency_ms": 123})

        Args:
            hook_name: Pipeline method name (must be in PIPELINE_HOOKS)
            input_data: Optional input data to record

        Yields:
            TaskContext for recording output and metrics
        """
        hook_config = PIPELINE_HOOKS.get(hook_name, {})
        task_name = hook_config.get("name", hook_name)
        workflow_task_name = hook_config.get("workflow_task", task_name)

        ctx = TaskContext(
            hook_name=hook_name,
            task_name=task_name,
        )

        if input_data:
            ctx.set_input(input_data)

        self._current_task = ctx

        # Calculate task index for SSE progress
        task_index = self._get_workflow_task_index(workflow_task_name)

        # Emit task started event (only if not already started for this workflow task)
        if self.emitter and self.session_id and workflow_task_name not in self._completed_workflow_tasks:
            try:
                # Import locally to avoid circular import
                from agentic.events import troubleshooting_task_started
                await self.emitter.emit(troubleshooting_task_started(
                    request_id=self.request_id,
                    session_id=self.session_id,
                    task_name=workflow_task_name,
                    task_index=task_index,
                    task_total=WORKFLOW_TASK_COUNT,
                    graph_line=self._build_graph_line(active_task=workflow_task_name)
                ))
            except Exception as e:
                logger.warning(f"Failed to emit task started event: {e}")

        try:
            yield ctx
            ctx.complete()

            # Mark workflow task as completed and emit event
            if workflow_task_name not in self._completed_workflow_tasks:
                self._completed_workflow_tasks.add(workflow_task_name)

                if self.emitter and self.session_id:
                    try:
                        # Import locally to avoid circular import
                        from agentic.events import troubleshooting_task_completed
                        duration_ms = int(ctx.duration_seconds * 1000)
                        await self.emitter.emit(troubleshooting_task_completed(
                            request_id=self.request_id,
                            session_id=self.session_id,
                            task_name=workflow_task_name,
                            task_index=task_index,
                            task_total=WORKFLOW_TASK_COUNT,
                            duration_ms=duration_ms,
                            graph_line=self._build_graph_line()
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to emit task completed event: {e}")

        except Exception as e:
            ctx.set_error(str(e))
            ctx.complete(verification_passed=False)

            # Emit task failed event
            if self.emitter and self.session_id:
                try:
                    # Import locally to avoid circular import
                    from agentic.events import troubleshooting_task_failed
                    duration_ms = int(ctx.duration_seconds * 1000)
                    await self.emitter.emit(troubleshooting_task_failed(
                        request_id=self.request_id,
                        session_id=self.session_id,
                        task_name=workflow_task_name,
                        task_index=task_index,
                        task_total=WORKFLOW_TASK_COUNT,
                        error=str(e),
                        duration_ms=duration_ms,
                        graph_line=self._build_graph_line()
                    ))
                except Exception as emit_err:
                    logger.warning(f"Failed to emit task failed event: {emit_err}")

            raise
        finally:
            self._task_contexts.append(ctx)
            self._current_task = None

            logger.debug(
                f"Task {task_name} completed: "
                f"duration={ctx.duration_seconds:.2f}s, "
                f"error={ctx.error is not None}"
            )

    def record_task_sync(
        self,
        hook_name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ):
        """
        Record a task execution synchronously (for non-async contexts).

        Use this when the task has already completed and you just need
        to record the results.
        """
        hook_config = PIPELINE_HOOKS.get(hook_name, {})
        task_name = hook_config.get("name", hook_name)

        ctx = TaskContext(
            hook_name=hook_name,
            task_name=task_name,
        )

        if input_data:
            ctx.set_input(input_data)
        if output_data:
            ctx.set_output(output_data)
        if metrics:
            ctx.set_metrics(metrics)
        if error:
            ctx.set_error(error)
        if duration_ms:
            ctx.metrics["latency_ms"] = duration_ms

        ctx.complete(verification_passed=error is None)
        self._task_contexts.append(ctx)

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    def verify_task(self, ctx: TaskContext) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify task output against criteria defined in PIPELINE_HOOKS.

        Returns:
            Tuple of (passed, details)
        """
        hook_config = PIPELINE_HOOKS.get(ctx.hook_name, {})
        criteria = hook_config.get("verification_criteria", {})

        if not criteria:
            return True, {"reason": "no_criteria"}

        details = {}
        passed = True

        # Check minimum results
        if "min_results" in criteria:
            result_count = ctx.output_data.get("result_count", 0)
            if result_count < criteria["min_results"]:
                passed = False
                details["min_results"] = {
                    "expected": criteria["min_results"],
                    "actual": result_count,
                }

        # Check minimum context length
        if "min_context_length" in criteria:
            context_len = ctx.output_data.get("context_length", 0)
            if context_len < criteria["min_context_length"]:
                passed = False
                details["min_context_length"] = {
                    "expected": criteria["min_context_length"],
                    "actual": context_len,
                }

        # Check minimum confidence
        if "min_confidence" in criteria:
            confidence = ctx.output_data.get("confidence", 0)
            if confidence < criteria["min_confidence"]:
                passed = False
                details["min_confidence"] = {
                    "expected": criteria["min_confidence"],
                    "actual": confidence,
                }

        # Check minimum paths
        if "min_paths" in criteria:
            path_count = ctx.output_data.get("path_count", 0)
            if path_count < criteria["min_paths"]:
                passed = False
                details["min_paths"] = {
                    "expected": criteria["min_paths"],
                    "actual": path_count,
                }

        # Check max fabricated ratio
        if "max_fabricated_ratio" in criteria:
            total = ctx.output_data.get("total_entities", 1)
            fabricated = ctx.output_data.get("fabricated_count", 0)
            ratio = fabricated / max(total, 1)
            if ratio > criteria["max_fabricated_ratio"]:
                passed = False
                details["max_fabricated_ratio"] = {
                    "expected": criteria["max_fabricated_ratio"],
                    "actual": ratio,
                }

        return passed, details

    # =========================================================================
    # SSE HELPER METHODS
    # =========================================================================

    # Ordered list of workflow task names for progress tracking
    WORKFLOW_TASK_ORDER = [
        "Query Analysis",
        "Entity Extraction",
        "Technical Doc Search",
        "Cross-Domain Validation",
        "Synthesis",
        "User Verification",
    ]

    def _get_workflow_task_index(self, workflow_task_name: str) -> int:
        """Get the index of a workflow task in the standard order."""
        try:
            return self.WORKFLOW_TASK_ORDER.index(workflow_task_name)
        except ValueError:
            return -1  # Unknown task

    def _build_graph_line(self, active_task: Optional[str] = None) -> str:
        """
        Build an ASCII graph line showing pipeline progress.

        Format: ●─●─●─◎─○─○
        - ● = completed
        - ◎ = active/in-progress
        - ○ = pending
        """
        symbols = []
        for task_name in self.WORKFLOW_TASK_ORDER:
            if task_name in self._completed_workflow_tasks:
                symbols.append("●")
            elif task_name == active_task:
                symbols.append("◎")
            else:
                symbols.append("○")
        return "─".join(symbols)

    # =========================================================================
    # DATABASE PERSISTENCE
    # =========================================================================

    async def _flush_to_database(self, pipeline_success: bool):
        """
        Flush all tracked task data to the database.

        Creates a new database session to avoid transaction conflicts
        with the main pipeline session.

        Updates both:
        1. Session metadata with pipeline summary
        2. Individual TaskExecution records with actual progress
        """
        if not self.session_id or not self._task_contexts:
            return

        from core.troubleshooting_service import troubleshooting_service
        from models.troubleshooting import TaskState

        try:
            async for db_session in get_async_db():
                # Get the troubleshooting session with task_executions loaded
                ts_session = await troubleshooting_service.get_session(
                    db_session, self.session_id
                )
                if not ts_session:
                    logger.warning(f"Session not found for flush: {self.session_id}")
                    return

                # Build mapping of workflow task names to TaskExecution records
                task_exec_by_name: Dict[str, Any] = {}
                for task_exec in ts_session.task_executions:
                    if task_exec.task:
                        task_exec_by_name[task_exec.task.name] = task_exec
                        logger.debug(f"Mapped task: {task_exec.task.name} -> {task_exec.id}")

                # Update session metadata with pipeline info
                metadata = ts_session.session_metadata or {}
                metadata["pipeline"] = {
                    "started_at": self._pipeline_started_at,
                    "completed_at": self._pipeline_completed_at,
                    "success": pipeline_success,
                    "task_count": len(self._task_contexts),
                    "total_duration_ms": int(
                        (self._pipeline_completed_at - self._pipeline_started_at) * 1000
                    ) if self._pipeline_started_at and self._pipeline_completed_at else None,
                }

                # Track which workflow tasks have been updated
                updated_tasks = set()
                task_summaries = []

                # Update individual TaskExecution records
                for ctx in self._task_contexts:
                    passed, details = self.verify_task(ctx)

                    # Get the workflow task name this hook maps to
                    hook_config = PIPELINE_HOOKS.get(ctx.hook_name, {})
                    workflow_task_name = hook_config.get("workflow_task", ctx.task_name)

                    task_summaries.append({
                        "hook": ctx.hook_name,
                        "name": ctx.task_name,
                        "workflow_task": workflow_task_name,
                        "duration_ms": ctx.metrics.get("latency_ms"),
                        "success": ctx.error is None,
                        "verification_passed": passed,
                        "error": ctx.error,
                    })

                    # Find and update the matching TaskExecution
                    if workflow_task_name in task_exec_by_name:
                        task_exec = task_exec_by_name[workflow_task_name]

                        # Only update if not already updated by a previous hook
                        if workflow_task_name not in updated_tasks:
                            # Update TaskExecution fields
                            task_exec.state = TaskState.COMPLETED if ctx.error is None else TaskState.FAILED
                            task_exec.started_at = datetime.fromtimestamp(
                                ctx.started_at, tz=timezone.utc
                            ) if ctx.started_at else None
                            task_exec.completed_at = datetime.fromtimestamp(
                                ctx.completed_at, tz=timezone.utc
                            ) if ctx.completed_at else None
                            task_exec.input_data = ctx.input_data or {}
                            task_exec.output_data = ctx.output_data or {}
                            task_exec.metrics = ctx.metrics or {}
                            task_exec.verification_passed = passed
                            task_exec.verification_details = details
                            task_exec.error_message = ctx.error

                            updated_tasks.add(workflow_task_name)
                            logger.debug(
                                f"Updated TaskExecution '{workflow_task_name}': "
                                f"state={task_exec.state}, duration={ctx.metrics.get('latency_ms')}ms"
                            )
                    else:
                        logger.debug(
                            f"No matching TaskExecution for workflow_task '{workflow_task_name}' "
                            f"(hook: {ctx.hook_name})"
                        )

                metadata["tasks"] = task_summaries
                ts_session.session_metadata = metadata

                # Count completed tasks from actual TaskExecution states
                completed_count = sum(
                    1 for te in ts_session.task_executions
                    if te.state == TaskState.COMPLETED
                )
                ts_session.completed_tasks = completed_count

                # Update session state if pipeline completed
                if pipeline_success and completed_count > 0:
                    from models.troubleshooting import SessionState
                    if ts_session.state == SessionState.INITIATED:
                        ts_session.state = SessionState.IN_PROGRESS

                await db_session.commit()

                logger.info(
                    f"Flushed {len(self._task_contexts)} task records "
                    f"for session {self.session_id}, "
                    f"updated {len(updated_tasks)} TaskExecutions, "
                    f"completed: {completed_count}/{ts_session.total_tasks}"
                )
                break  # Exit the async generator

        except Exception as e:
            logger.error(f"Failed to flush to database: {e}", exc_info=True)

    # =========================================================================
    # METRICS & REPORTING
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked tasks."""
        if not self._task_contexts:
            return {"tasks": [], "total_duration_ms": 0}

        tasks = []
        total_duration = 0

        for ctx in self._task_contexts:
            duration = ctx.metrics.get("latency_ms", 0)
            total_duration += duration

            tasks.append({
                "name": ctx.task_name,
                "hook": ctx.hook_name,
                "duration_ms": duration,
                "success": ctx.error is None,
                "error": ctx.error,
            })

        return {
            "session_id": self.session_id,
            "tasks": tasks,
            "task_count": len(tasks),
            "successful_tasks": sum(1 for t in tasks if t["success"]),
            "failed_tasks": sum(1 for t in tasks if not t["success"]),
            "total_duration_ms": total_duration,
        }

    def get_metrics_for_observability(self) -> Dict[str, Any]:
        """
        Get metrics in a format suitable for observability tools.

        Returns structured data for logging, tracing, or dashboards.
        """
        return {
            "troubleshooting.session_id": self.session_id,
            "troubleshooting.user_id": self.user_id,
            "troubleshooting.task_count": len(self._task_contexts),
            "troubleshooting.pipeline_duration_ms": int(
                (self._pipeline_completed_at - self._pipeline_started_at) * 1000
            ) if self._pipeline_started_at and self._pipeline_completed_at else None,
            "troubleshooting.tasks": [
                {
                    "name": ctx.task_name,
                    "duration_ms": ctx.metrics.get("latency_ms"),
                    "success": ctx.error is None,
                }
                for ctx in self._task_contexts
            ],
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_tracker(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    emitter: Optional["EventEmitter"] = None,
    request_id: Optional[str] = None,
) -> TroubleshootingTaskTracker:
    """
    Factory function to create a task tracker.

    Returns a disabled tracker if session_id is not provided.

    Args:
        session_id: Troubleshooting session ID
        user_id: User ID for attribution
        emitter: Optional EventEmitter for real-time SSE updates
        request_id: Request ID for SSE event correlation
    """
    return TroubleshootingTaskTracker(
        session_id=session_id,
        user_id=user_id,
        enabled=session_id is not None,
        emitter=emitter,
        request_id=request_id,
    )


# =============================================================================
# DECORATOR FOR AUTOMATIC TRACKING
# =============================================================================

def track_pipeline_task(hook_name: str):
    """
    Decorator for automatically tracking pipeline tasks.

    Usage:
        @track_pipeline_task("_search_technical_docs")
        async def _search_technical_docs(self, query: str):
            ...

    The decorated method must be part of a class with a `tracker` attribute
    that is a TroubleshootingTaskTracker instance.
    """
    def decorator(func: Callable):
        async def wrapper(self, *args, **kwargs):
            tracker = getattr(self, "tracker", None)

            if tracker and tracker.enabled:
                async with tracker.track_task(hook_name) as ctx:
                    result = await func(self, *args, **kwargs)

                    # Try to extract metrics from result
                    if isinstance(result, dict):
                        if "results" in result:
                            ctx.set_output({"result_count": len(result["results"])})
                        if "confidence" in result:
                            ctx.set_output({"confidence": result["confidence"]})
                        if "paths" in result:
                            ctx.set_output({"path_count": len(result["paths"])})

                    return result
            else:
                return await func(self, *args, **kwargs)

        return wrapper
    return decorator
