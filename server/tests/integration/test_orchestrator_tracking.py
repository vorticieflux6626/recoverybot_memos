"""
Integration tests for orchestrator troubleshooting task tracking (Phase 4).

Tests the integration between UniversalOrchestrator and TroubleshootingTaskTracker.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentic.orchestrator_universal import UniversalOrchestrator, OrchestratorPreset
from agentic.models import SearchRequest
from core.troubleshooting_tracker import TroubleshootingTaskTracker, TaskContext


class TestOrchestratorTrackerIntegration:
    """Test that orchestrator correctly initializes and uses the troubleshooting tracker."""

    def test_tracker_initialization_with_session_id(self):
        """Test that tracker is initialized when session_id is provided."""
        orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.MINIMAL)

        request = SearchRequest(
            query="SRVO-063 encoder disconnect error",
            session_id="test-session-123",
            troubleshooting_mode=True,
            user_id="test-user"
        )

        # Initialize tracker
        tracker = orchestrator._init_troubleshooting_tracker(request, "req-001")

        assert tracker is not None
        assert tracker.enabled is True
        assert tracker.session_id == "test-session-123"

    def test_tracker_not_initialized_without_session(self):
        """Test that tracker is not initialized when session_id is missing."""
        orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.MINIMAL)

        request = SearchRequest(
            query="SRVO-063 encoder disconnect error",
            # No session_id
            troubleshooting_mode=True
        )

        tracker = orchestrator._init_troubleshooting_tracker(request, "req-001")

        assert tracker is None

    def test_tracker_not_initialized_without_troubleshooting_mode(self):
        """Test that tracker is not initialized when troubleshooting_mode is False."""
        orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.MINIMAL)

        request = SearchRequest(
            query="SRVO-063 encoder disconnect error",
            session_id="test-session-123",
            troubleshooting_mode=False  # Explicitly disabled
        )

        tracker = orchestrator._init_troubleshooting_tracker(request, "req-001")

        assert tracker is None

    def test_get_troubleshooting_tracker_returns_current_tracker(self):
        """Test that _get_troubleshooting_tracker returns the current tracker."""
        orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.MINIMAL)

        # No tracker initially
        assert orchestrator._get_troubleshooting_tracker() is None

        # Set tracker
        request = SearchRequest(
            query="Test query",
            session_id="test-session",
            troubleshooting_mode=True,
            user_id="user-1"
        )
        tracker = orchestrator._init_troubleshooting_tracker(request, "req-001")
        orchestrator._troubleshooting_tracker = tracker

        # Should return the tracker
        retrieved = orchestrator._get_troubleshooting_tracker()
        assert retrieved is tracker


class TestTrackerRecordTaskSync:
    """Test the record_task_sync functionality used by pipeline phases."""

    def test_record_task_sync_with_input_output(self):
        """Test recording a task with input and output data."""
        tracker = TroubleshootingTaskTracker(
            session_id="test-session",
            user_id="test-user",
            enabled=True
        )
        tracker.start_pipeline(metadata={"test": True})

        tracker.record_task_sync(
            "_analyze_query",
            input_data={"query_length": 50, "has_context": True},
            output_data={
                "requires_search": True,
                "query_type": "troubleshooting",
                "complexity": "moderate"
            },
            duration_ms=150
        )

        # Check task was recorded
        assert len(tracker._task_contexts) == 1
        ctx = tracker._task_contexts[0]
        assert ctx.hook_name == "_analyze_query"
        assert ctx.input_data["query_length"] == 50
        assert ctx.output_data["requires_search"] is True
        assert ctx.metrics["latency_ms"] == 150
        assert ctx.error is None

    def test_record_task_sync_with_error(self):
        """Test recording a failed task with error."""
        tracker = TroubleshootingTaskTracker(
            session_id="test-session",
            user_id="test-user",
            enabled=True
        )
        tracker.start_pipeline(metadata={})

        tracker.record_task_sync(
            "_hyde_expand",
            input_data={"query_length": 30},
            error="LLM service unavailable",
            duration_ms=5000
        )

        ctx = tracker._task_contexts[0]
        assert ctx.hook_name == "_hyde_expand"
        assert ctx.error == "LLM service unavailable"
        assert ctx.output_data == {}  # Empty dict, not None

    def test_record_task_sync_disabled_tracker(self):
        """Test that disabled tracker still records but marks disabled."""
        tracker = TroubleshootingTaskTracker(
            session_id=None,  # None session_id makes enabled=False internally
            user_id=None,
            enabled=True  # enabled flag gets overridden when session_id is None
        )

        # With no session_id, tracker.enabled becomes False
        assert tracker.enabled is False

        # Record a task anyway (recording still happens, just won't flush to DB)
        tracker.record_task_sync(
            "_synthesize",
            input_data={"sources_count": 5},
            output_data={"synthesis_length": 2000}
        )

        # Tasks are still recorded in memory
        assert len(tracker._task_contexts) == 1


class TestTrackerSummary:
    """Test tracker summary and metrics."""

    def test_get_summary_with_multiple_tasks(self):
        """Test getting summary after multiple tasks."""
        tracker = TroubleshootingTaskTracker(
            session_id="test-session",
            user_id="test-user",
            enabled=True
        )
        tracker.start_pipeline(metadata={"preset": "ENHANCED"})

        # Record multiple tasks
        tracker.record_task_sync(
            "_analyze_query",
            input_data={"query_length": 50},
            output_data={"requires_search": True},
            duration_ms=100
        )
        tracker.record_task_sync(
            "_search_web",
            input_data={"queries_count": 3},
            output_data={"results_found": 25},
            duration_ms=2500
        )
        tracker.record_task_sync(
            "_synthesize",
            input_data={"sources_count": 5},
            output_data={"synthesis_length": 3000, "confidence": 0.75},
            duration_ms=8000
        )

        summary = tracker.get_summary()

        assert summary["session_id"] == "test-session"
        assert summary["task_count"] == 3
        assert summary["successful_tasks"] == 3
        assert summary["failed_tasks"] == 0
        assert summary["total_duration_ms"] == 10600  # 100 + 2500 + 8000

    def test_get_summary_with_failed_task(self):
        """Test summary includes failed task count."""
        tracker = TroubleshootingTaskTracker(
            session_id="test-session",
            user_id="test-user",
            enabled=True
        )
        tracker.start_pipeline(metadata={})

        tracker.record_task_sync(
            "_analyze_query",
            input_data={"query_length": 50},
            output_data={"requires_search": True},
            duration_ms=100
        )
        tracker.record_task_sync(
            "_scrape_content",
            input_data={"urls_to_scrape": 5},
            error="Connection timeout",
            duration_ms=30000
        )

        summary = tracker.get_summary()

        assert summary["successful_tasks"] == 1
        assert summary["failed_tasks"] == 1


class TestSearchRequestFields:
    """Test that SearchRequest has the troubleshooting fields."""

    def test_search_request_has_session_id(self):
        """Test SearchRequest has session_id field."""
        request = SearchRequest(
            query="Test query",
            session_id="session-123"
        )
        assert request.session_id == "session-123"

    def test_search_request_has_troubleshooting_mode(self):
        """Test SearchRequest has troubleshooting_mode field."""
        request = SearchRequest(
            query="Test query",
            troubleshooting_mode=True
        )
        assert request.troubleshooting_mode is True

    def test_search_request_defaults(self):
        """Test SearchRequest has sensible defaults."""
        request = SearchRequest(query="Test query")
        assert request.session_id is None
        assert request.troubleshooting_mode is False


# Run with: pytest tests/integration/test_orchestrator_tracking.py -v
