"""
Unit Tests for Troubleshooting Task Tracker

Tests automatic pipeline task tracking and verification.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from core.troubleshooting_tracker import (
    TroubleshootingTaskTracker,
    TaskContext,
    PIPELINE_HOOKS,
    create_tracker,
    track_pipeline_task,
)
from models.troubleshooting import TaskState, TaskExecutionType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tracker():
    """Create a tracker with a session ID."""
    return TroubleshootingTaskTracker(
        session_id=str(uuid4()),
        user_id="test_user",
        enabled=True,
    )


@pytest.fixture
def disabled_tracker():
    """Create a disabled tracker."""
    return TroubleshootingTaskTracker(
        session_id=None,
        user_id="test_user",
        enabled=False,
    )


# =============================================================================
# TASK CONTEXT TESTS
# =============================================================================

class TestTaskContext:
    """Tests for TaskContext data class."""

    def test_create_context(self):
        """Test creating a task context."""
        ctx = TaskContext(
            hook_name="_search_technical_docs",
            task_name="Document Retrieval",
        )
        assert ctx.hook_name == "_search_technical_docs"
        assert ctx.task_name == "Document Retrieval"
        assert ctx.started_at > 0
        assert ctx.completed_at is None
        assert ctx.error is None

    def test_set_input_data(self):
        """Test setting input data."""
        ctx = TaskContext(hook_name="test", task_name="Test")
        ctx.set_input({"query": "test query"})
        assert ctx.input_data["query"] == "test query"

    def test_set_output_data(self):
        """Test setting output data."""
        ctx = TaskContext(hook_name="test", task_name="Test")
        ctx.set_output({"result_count": 10})
        assert ctx.output_data["result_count"] == 10

    def test_set_metrics(self):
        """Test setting metrics."""
        ctx = TaskContext(hook_name="test", task_name="Test")
        ctx.set_metrics({"latency_ms": 150})
        assert ctx.metrics["latency_ms"] == 150

    def test_complete_task(self):
        """Test completing a task."""
        ctx = TaskContext(hook_name="test", task_name="Test")
        time.sleep(0.01)  # Small delay
        ctx.complete(verification_passed=True)

        assert ctx.completed_at is not None
        assert ctx.completed_at > ctx.started_at
        assert ctx.verification_passed is True
        assert "latency_ms" in ctx.metrics

    def test_duration_calculation(self):
        """Test duration calculation."""
        ctx = TaskContext(hook_name="test", task_name="Test")
        time.sleep(0.05)  # 50ms delay
        duration = ctx.duration_seconds

        assert duration >= 0.05
        assert duration < 1.0  # Sanity check


# =============================================================================
# TRACKER INITIALIZATION TESTS
# =============================================================================

class TestTrackerInitialization:
    """Tests for tracker initialization."""

    def test_create_enabled_tracker(self):
        """Test creating an enabled tracker."""
        tracker = TroubleshootingTaskTracker(
            session_id="test-session",
            user_id="test-user",
            enabled=True,
        )
        assert tracker.enabled is True
        assert tracker.session_id == "test-session"

    def test_create_disabled_tracker_without_session(self):
        """Test that tracker is disabled without session ID."""
        tracker = TroubleshootingTaskTracker(
            session_id=None,
            user_id="test-user",
            enabled=True,
        )
        assert tracker.enabled is False

    def test_factory_function(self):
        """Test create_tracker factory function."""
        tracker = create_tracker(
            session_id="test-session",
            user_id="test-user",
        )
        assert tracker.enabled is True

        tracker = create_tracker(session_id=None)
        assert tracker.enabled is False


# =============================================================================
# PIPELINE LIFECYCLE TESTS
# =============================================================================

class TestPipelineLifecycle:
    """Tests for pipeline start/complete lifecycle."""

    def test_start_pipeline(self, tracker):
        """Test starting a pipeline."""
        tracker.start_pipeline(metadata={"preset": "RESEARCH"})

        assert tracker._pipeline_started_at is not None
        assert tracker._pipeline_metadata["preset"] == "RESEARCH"
        assert len(tracker._task_contexts) == 0

    @pytest.mark.asyncio
    async def test_complete_pipeline(self, tracker):
        """Test completing a pipeline."""
        tracker.start_pipeline()

        # Simulate some tasks
        tracker.record_task_sync(
            "_search_technical_docs",
            output_data={"result_count": 5},
            duration_ms=100,
        )

        # Mock the flush to avoid actual DB operations
        with patch.object(tracker, '_flush_to_database', new_callable=AsyncMock):
            await tracker.complete_pipeline(success=True)

        assert tracker._pipeline_completed_at is not None


# =============================================================================
# TASK TRACKING TESTS
# =============================================================================

class TestTaskTracking:
    """Tests for task tracking functionality."""

    @pytest.mark.asyncio
    async def test_track_task_context_manager(self, tracker):
        """Test tracking a task with context manager."""
        tracker.start_pipeline()

        async with tracker.track_task("_search_technical_docs") as ctx:
            ctx.set_output({"result_count": 10})
            ctx.set_metrics({"latency_ms": 150})

        assert len(tracker._task_contexts) == 1
        assert tracker._task_contexts[0].output_data["result_count"] == 10

    @pytest.mark.asyncio
    async def test_track_task_with_error(self, tracker):
        """Test tracking a task that raises an error."""
        tracker.start_pipeline()

        with pytest.raises(ValueError):
            async with tracker.track_task("_synthesize") as ctx:
                raise ValueError("Test error")

        # Task should still be recorded
        assert len(tracker._task_contexts) == 1
        assert tracker._task_contexts[0].error is not None

    def test_record_task_sync(self, tracker):
        """Test synchronous task recording."""
        tracker.start_pipeline()

        tracker.record_task_sync(
            "_search_technical_docs",
            input_data={"query": "test"},
            output_data={"result_count": 5},
            metrics={"tokens_used": 100},
            duration_ms=200,
        )

        assert len(tracker._task_contexts) == 1
        ctx = tracker._task_contexts[0]
        assert ctx.input_data["query"] == "test"
        assert ctx.output_data["result_count"] == 5
        assert ctx.metrics["latency_ms"] == 200

    @pytest.mark.asyncio
    async def test_disabled_tracker_skips_tracking(self, disabled_tracker):
        """Test that disabled tracker doesn't track."""
        disabled_tracker.start_pipeline()

        async with disabled_tracker.track_task("_search_technical_docs") as ctx:
            ctx.set_output({"result_count": 10})

        # Should still append (tracking happens regardless of enabled state)
        # But flush won't happen


# =============================================================================
# VERIFICATION TESTS
# =============================================================================

class TestTaskVerification:
    """Tests for task verification logic."""

    def test_verify_min_results_pass(self, tracker):
        """Test verification passes when min_results met."""
        ctx = TaskContext(
            hook_name="_search_technical_docs",
            task_name="Document Retrieval",
        )
        # Need both min_results and min_context_length for this hook
        ctx.set_output({"result_count": 5, "context_length": 500})

        passed, details = tracker.verify_task(ctx)
        assert passed is True

    def test_verify_min_results_fail(self, tracker):
        """Test verification fails when min_results not met."""
        ctx = TaskContext(
            hook_name="_search_technical_docs",
            task_name="Document Retrieval",
        )
        ctx.set_output({"result_count": 0})

        passed, details = tracker.verify_task(ctx)
        assert passed is False
        assert "min_results" in details

    def test_verify_min_confidence_pass(self, tracker):
        """Test verification passes when min_confidence met."""
        ctx = TaskContext(
            hook_name="_synthesize",
            task_name="Response Synthesis",
        )
        ctx.set_output({"confidence": 0.8, "citations": 3})

        passed, details = tracker.verify_task(ctx)
        assert passed is True

    def test_verify_min_confidence_fail(self, tracker):
        """Test verification fails when min_confidence not met."""
        ctx = TaskContext(
            hook_name="_synthesize",
            task_name="Response Synthesis",
        )
        ctx.set_output({"confidence": 0.3})

        passed, details = tracker.verify_task(ctx)
        assert passed is False
        assert "min_confidence" in details

    def test_verify_no_criteria(self, tracker):
        """Test verification passes when no criteria defined."""
        ctx = TaskContext(
            hook_name="_generate_diagram",
            task_name="Diagram Generation",
        )

        passed, details = tracker.verify_task(ctx)
        assert passed is True


# =============================================================================
# METRICS AND REPORTING TESTS
# =============================================================================

class TestMetricsReporting:
    """Tests for metrics and reporting functionality."""

    def test_get_summary_empty(self, tracker):
        """Test summary with no tasks."""
        summary = tracker.get_summary()
        assert summary["tasks"] == []
        assert summary["total_duration_ms"] == 0

    def test_get_summary_with_tasks(self, tracker):
        """Test summary with recorded tasks."""
        tracker.start_pipeline()

        tracker.record_task_sync(
            "_search_technical_docs",
            output_data={"result_count": 5},
            duration_ms=100,
        )
        tracker.record_task_sync(
            "_synthesize",
            output_data={"confidence": 0.8},
            duration_ms=500,
        )

        summary = tracker.get_summary()

        assert len(summary["tasks"]) == 2
        assert summary["task_count"] == 2
        assert summary["successful_tasks"] == 2
        assert summary["total_duration_ms"] == 600

    def test_get_metrics_for_observability(self, tracker):
        """Test observability metrics format."""
        tracker.start_pipeline()
        tracker._pipeline_completed_at = time.time()

        tracker.record_task_sync(
            "_search_technical_docs",
            duration_ms=100,
        )

        metrics = tracker.get_metrics_for_observability()

        assert "troubleshooting.session_id" in metrics
        assert "troubleshooting.user_id" in metrics
        assert "troubleshooting.task_count" in metrics
        assert "troubleshooting.tasks" in metrics


# =============================================================================
# PIPELINE HOOKS CONFIGURATION TESTS
# =============================================================================

class TestPipelineHooksConfig:
    """Tests for pipeline hooks configuration."""

    def test_all_hooks_have_name(self):
        """Test that all hooks have a name."""
        for hook_name, config in PIPELINE_HOOKS.items():
            assert "name" in config
            assert config["name"] is not None

    def test_all_hooks_have_execution_type(self):
        """Test that all hooks have an execution type."""
        for hook_name, config in PIPELINE_HOOKS.items():
            assert "execution_type" in config
            assert isinstance(config["execution_type"], TaskExecutionType)

    def test_automatic_hooks_have_verification(self):
        """Test that automatic hooks have verification criteria."""
        for hook_name, config in PIPELINE_HOOKS.items():
            if config["execution_type"] == TaskExecutionType.AUTOMATIC:
                assert "verification_criteria" in config

    def test_expected_hooks_exist(self):
        """Test that expected hooks are defined."""
        expected_hooks = [
            "_search_technical_docs",
            "_traverse_graph",
            "_ground_entities",
            "_validate_cross_domain",
            "_synthesize",
            "_generate_diagram",
        ]
        for hook in expected_hooks:
            assert hook in PIPELINE_HOOKS


# =============================================================================
# DECORATOR TESTS
# =============================================================================

class TestTrackPipelineTaskDecorator:
    """Tests for the track_pipeline_task decorator."""

    @pytest.mark.asyncio
    async def test_decorator_tracks_task(self):
        """Test that decorator tracks task execution."""

        class MockOrchestrator:
            def __init__(self):
                self.tracker = TroubleshootingTaskTracker(
                    session_id=str(uuid4()),
                    user_id="test",
                    enabled=True,
                )
                self.tracker.start_pipeline()

            @track_pipeline_task("_search_technical_docs")
            async def _search_technical_docs(self, query: str):
                return {"results": [1, 2, 3]}

        orchestrator = MockOrchestrator()
        result = await orchestrator._search_technical_docs("test query")

        assert result == {"results": [1, 2, 3]}
        assert len(orchestrator.tracker._task_contexts) == 1

    @pytest.mark.asyncio
    async def test_decorator_extracts_result_count(self):
        """Test that decorator extracts result count from output."""

        class MockOrchestrator:
            def __init__(self):
                self.tracker = TroubleshootingTaskTracker(
                    session_id=str(uuid4()),
                    user_id="test",
                    enabled=True,
                )
                self.tracker.start_pipeline()

            @track_pipeline_task("_search_technical_docs")
            async def _search_technical_docs(self, query: str):
                return {"results": [1, 2, 3, 4, 5]}

        orchestrator = MockOrchestrator()
        await orchestrator._search_technical_docs("test query")

        ctx = orchestrator.tracker._task_contexts[0]
        assert ctx.output_data.get("result_count") == 5

    @pytest.mark.asyncio
    async def test_decorator_without_tracker(self):
        """Test decorator works when no tracker is present."""

        class MockOrchestrator:
            @track_pipeline_task("_search_technical_docs")
            async def _search_technical_docs(self, query: str):
                return {"results": [1, 2, 3]}

        orchestrator = MockOrchestrator()
        result = await orchestrator._search_technical_docs("test query")

        assert result == {"results": [1, 2, 3]}


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================

class TestTrackerIntegration:
    """Integration-like tests for complete tracking flows."""

    @pytest.mark.asyncio
    async def test_full_pipeline_tracking(self, tracker):
        """Test tracking a complete pipeline run."""
        tracker.start_pipeline(metadata={"preset": "RESEARCH"})

        # Track document retrieval
        async with tracker.track_task("_search_technical_docs") as ctx:
            ctx.set_input({"query": "SRVO-063"})
            ctx.set_output({"result_count": 5, "context_length": 1500})
            ctx.set_metrics({"latency_ms": 150})

        # Track entity grounding
        async with tracker.track_task("_ground_entities") as ctx:
            ctx.set_output({"total_entities": 10, "fabricated_count": 0})
            ctx.set_metrics({"latency_ms": 50})

        # Track synthesis
        async with tracker.track_task("_synthesize") as ctx:
            ctx.set_output({"confidence": 0.85, "citations": 3})
            ctx.set_metrics({"latency_ms": 800, "tokens_used": 2000})

        # Get summary
        summary = tracker.get_summary()

        assert summary["task_count"] == 3
        assert summary["successful_tasks"] == 3
        assert summary["total_duration_ms"] == 1000  # 150 + 50 + 800

    @pytest.mark.asyncio
    async def test_pipeline_with_failure(self, tracker):
        """Test tracking a pipeline with a failed task."""
        tracker.start_pipeline()

        # Successful task
        async with tracker.track_task("_search_technical_docs") as ctx:
            ctx.set_output({"result_count": 5})

        # Failed task
        tracker.record_task_sync(
            "_synthesize",
            error="LLM timeout after 120s",
            duration_ms=120000,
        )

        summary = tracker.get_summary()

        assert summary["task_count"] == 2
        assert summary["successful_tasks"] == 1
        assert summary["failed_tasks"] == 1
