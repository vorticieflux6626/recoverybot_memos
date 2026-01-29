"""
Integration Tests for Troubleshooting API

Tests the full API endpoints with database interactions.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from uuid import uuid4

# Use session-scoped event loop for all tests in this module
# This is required because the ASGI app's lifespan creates database connections
# that are tied to the event loop at initialization time.
pytestmark = pytest.mark.asyncio(loop_scope="session")

from main import app
from models.troubleshooting import (
    TroubleshootingCategory,
    SessionState,
    TaskState,
    ExpertiseLevel,
    TroubleshootingDomain,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def test_user_id():
    """Generate a unique test user ID."""
    return f"test_user_{uuid4().hex[:8]}"


@pytest_asyncio.fixture(loop_scope="session", scope="session")
async def async_client():
    """Create an async test client (session-scoped to share event loop)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://localhost:8001",
        headers={"Host": "localhost:8001"},
    ) as client:
        yield client


# =============================================================================
# DOMAIN & CATEGORY ENDPOINT TESTS
# =============================================================================

class TestMetadataEndpoints:
    """Tests for domain, category, and pipeline hook endpoints."""

    @pytest.mark.asyncio
    async def test_list_domains(self, async_client):
        """Test listing all troubleshooting domains."""
        response = await async_client.get("/api/v1/troubleshooting/domains")

        assert response.status_code == 200
        domains = response.json()

        assert isinstance(domains, list)
        assert len(domains) >= 8
        assert "fanuc_servo" in domains
        assert "imm_defects" in domains

    @pytest.mark.asyncio
    async def test_list_categories(self, async_client):
        """Test listing all troubleshooting categories."""
        response = await async_client.get("/api/v1/troubleshooting/categories")

        assert response.status_code == 200
        categories = response.json()

        assert isinstance(categories, list)
        assert len(categories) >= 5
        assert "error_diagnosis" in categories
        assert "symptom_analysis" in categories

    @pytest.mark.asyncio
    async def test_list_pipeline_hooks(self, async_client):
        """Test listing pipeline hooks."""
        response = await async_client.get("/api/v1/troubleshooting/pipeline-hooks")

        assert response.status_code == 200
        hooks = response.json()

        assert isinstance(hooks, dict)
        assert "_search_technical_docs" in hooks
        assert "_synthesize" in hooks

        # Check hook structure
        search_hook = hooks["_search_technical_docs"]
        assert "name" in search_hook
        assert "execution_type" in search_hook
        assert "timeout_seconds" in search_hook


# =============================================================================
# SESSION ENDPOINT TESTS
# =============================================================================

class TestSessionEndpoints:
    """Tests for session CRUD endpoints."""

    @pytest.mark.asyncio
    async def test_create_session_with_error_code(self, async_client, test_user_id):
        """Test creating a session with an error code query."""
        response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "SRVO-063 pulsecoder error on axis 1"},
        )

        assert response.status_code == 200
        session = response.json()

        assert "id" in session
        assert session["user_id"] == test_user_id
        assert session["original_query"] == "SRVO-063 pulsecoder error on axis 1"
        assert "SRVO-063" in session["detected_error_codes"]
        assert session["entry_type"] == "error_code"
        assert session["domain"] == "fanuc_servo"
        assert session["state"] == "initiated"

    @pytest.mark.asyncio
    async def test_create_session_with_symptom(self, async_client, test_user_id):
        """Test creating a session with a symptom-based query."""
        response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "Robot arm is vibrating and jerky during movement"},
        )

        assert response.status_code == 200
        session = response.json()

        assert session["entry_type"] == "symptom"
        assert len(session["detected_symptoms"]) > 0

    @pytest.mark.asyncio
    async def test_create_session_requires_query(self, async_client, test_user_id):
        """Test that session creation requires a query."""
        response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "ab"},  # Too short
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_session(self, async_client, test_user_id):
        """Test getting a session by ID."""
        # Create a session first
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "SRVO-063 encoder error"},
        )
        session_id = create_response.json()["id"]

        # Get the session
        response = await async_client.get(
            f"/api/v1/troubleshooting/sessions/{session_id}"
        )

        assert response.status_code == 200
        session = response.json()
        assert session["id"] == session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, async_client):
        """Test getting a non-existent session."""
        fake_id = str(uuid4())
        response = await async_client.get(
            f"/api/v1/troubleshooting/sessions/{fake_id}"
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_list_sessions(self, async_client, test_user_id):
        """Test listing user sessions."""
        # Create multiple sessions
        for i in range(3):
            await async_client.post(
                "/api/v1/troubleshooting/sessions",
                params={"user_id": test_user_id},
                json={"query": f"SRVO-{60+i:03d} test error"},
            )

        # List sessions
        response = await async_client.get(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
        )

        assert response.status_code == 200
        data = response.json()

        assert "sessions" in data
        assert len(data["sessions"]) >= 3
        assert data["total"] >= 3

    @pytest.mark.asyncio
    async def test_list_sessions_filter_by_state(self, async_client, test_user_id):
        """Test listing sessions filtered by state."""
        # Create a session
        await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "Test query for filter"},
        )

        # List only initiated sessions
        response = await async_client.get(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id, "state": "initiated"},
        )

        assert response.status_code == 200
        data = response.json()

        for session in data["sessions"]:
            assert session["state"] == "initiated"

    @pytest.mark.asyncio
    async def test_get_active_session(self, async_client, test_user_id):
        """Test getting the active session for a user."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "Active session test"},
        )
        session_id = create_response.json()["id"]

        # Get active session
        response = await async_client.get(
            "/api/v1/troubleshooting/sessions/active",
            params={"user_id": test_user_id},
        )

        assert response.status_code == 200
        # May be the session we created or null if no active


# =============================================================================
# SESSION RESOLUTION TESTS
# =============================================================================

class TestSessionResolution:
    """Tests for session resolution endpoints."""

    @pytest.mark.asyncio
    async def test_resolve_session_success(self, async_client, test_user_id):
        """Test resolving a session successfully."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "SRVO-063 for resolution test"},
        )
        session_id = create_response.json()["id"]

        # Resolve the session
        response = await async_client.post(
            f"/api/v1/troubleshooting/sessions/{session_id}/resolve",
            json={
                "resolution_type": "self_resolved",
                "rating": 5,
                "feedback": "Very helpful!",
            },
        )

        assert response.status_code == 200
        session = response.json()

        assert session["state"] == "resolved"
        assert session["expertise_points_earned"] > 0

    @pytest.mark.asyncio
    async def test_resolve_session_escalated(self, async_client, test_user_id):
        """Test escalating a session."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "Complex issue for escalation"},
        )
        session_id = create_response.json()["id"]

        # Escalate the session
        response = await async_client.post(
            f"/api/v1/troubleshooting/sessions/{session_id}/resolve",
            json={
                "resolution_type": "escalated",
                "feedback": "Need human expert",
            },
        )

        assert response.status_code == 200
        session = response.json()

        assert session["state"] == "escalated"

    @pytest.mark.asyncio
    async def test_resolve_session_abandoned(self, async_client, test_user_id):
        """Test abandoning a session."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "Query for abandoned session"},
        )
        session_id = create_response.json()["id"]

        # Abandon the session
        response = await async_client.post(
            f"/api/v1/troubleshooting/sessions/{session_id}/resolve",
            json={"resolution_type": "abandoned"},
        )

        assert response.status_code == 200
        session = response.json()

        assert session["state"] == "abandoned"


# =============================================================================
# EXPERTISE ENDPOINT TESTS
# =============================================================================

class TestExpertiseEndpoints:
    """Tests for expertise and achievement endpoints."""

    @pytest.mark.asyncio
    async def test_get_user_expertise(self, async_client, test_user_id):
        """Test getting user expertise."""
        response = await async_client.get(
            f"/api/v1/troubleshooting/users/{test_user_id}/expertise"
        )

        assert response.status_code == 200
        expertise = response.json()

        assert expertise["user_id"] == test_user_id
        assert "total_expertise_points" in expertise
        assert "expertise_level" in expertise
        assert "domain_points" in expertise
        assert "resolution_rate" in expertise

    @pytest.mark.asyncio
    async def test_expertise_accumulates(self, async_client, test_user_id):
        """Test that expertise accumulates after resolutions."""
        # Create and resolve a session
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "SRVO-063 for expertise test"},
        )
        session_id = create_response.json()["id"]

        await async_client.post(
            f"/api/v1/troubleshooting/sessions/{session_id}/resolve",
            json={"resolution_type": "self_resolved"},
        )

        # Check expertise
        response = await async_client.get(
            f"/api/v1/troubleshooting/users/{test_user_id}/expertise"
        )

        assert response.status_code == 200
        expertise = response.json()

        assert expertise["total_expertise_points"] > 0
        assert expertise["total_sessions"] >= 1
        assert expertise["successful_resolutions"] >= 1

    @pytest.mark.asyncio
    async def test_get_user_achievements(self, async_client, test_user_id):
        """Test getting user achievements."""
        response = await async_client.get(
            f"/api/v1/troubleshooting/users/{test_user_id}/achievements"
        )

        assert response.status_code == 200
        achievements = response.json()

        assert isinstance(achievements, list)


# =============================================================================
# LEADERBOARD TESTS
# =============================================================================

class TestLeaderboardEndpoints:
    """Tests for leaderboard endpoint."""

    @pytest.mark.asyncio
    async def test_get_leaderboard(self, async_client):
        """Test getting the leaderboard."""
        response = await async_client.get("/api/v1/troubleshooting/leaderboard")

        assert response.status_code == 200
        data = response.json()

        assert "entries" in data
        assert "total_participants" in data
        assert isinstance(data["entries"], list)

    @pytest.mark.asyncio
    async def test_leaderboard_filter_by_domain(self, async_client):
        """Test filtering leaderboard by domain."""
        response = await async_client.get(
            "/api/v1/troubleshooting/leaderboard",
            params={"domain": "fanuc_servo"},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["domain"] == "fanuc_servo"


# =============================================================================
# WORKFLOW ENDPOINT TESTS
# =============================================================================

class TestWorkflowEndpoints:
    """Tests for workflow listing endpoints."""

    @pytest.mark.asyncio
    async def test_list_workflows(self, async_client):
        """Test listing all workflows."""
        response = await async_client.get("/api/v1/troubleshooting/workflows")

        assert response.status_code == 200
        data = response.json()

        assert "workflows" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_list_workflows_filter_by_domain(self, async_client):
        """Test filtering workflows by domain."""
        response = await async_client.get(
            "/api/v1/troubleshooting/workflows",
            params={"domain": "fanuc_servo"},
        )

        assert response.status_code == 200
        data = response.json()

        for workflow in data["workflows"]:
            assert workflow["domain"] == "fanuc_servo"


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

class TestBackwardCompatibility:
    """Tests for backward-compatible quest API endpoints."""

    @pytest.mark.asyncio
    async def test_compat_get_available_quests(self, async_client, test_user_id):
        """Test backward-compatible available quests endpoint."""
        response = await async_client.get(
            "/api/v1/troubleshooting/compat/quests/available",
            params={"user_id": test_user_id},
        )

        assert response.status_code == 200
        data = response.json()

        assert "quests" in data
        assert "total" in data
        assert "hasMore" in data

    @pytest.mark.asyncio
    async def test_compat_get_user_stats(self, async_client, test_user_id):
        """Test backward-compatible user stats endpoint."""
        response = await async_client.get(
            f"/api/v1/troubleshooting/compat/users/{test_user_id}/stats"
        )

        assert response.status_code == 200
        data = response.json()

        # Check old quest stats format
        assert "user_id" in data
        assert "total_points" in data
        assert "current_streak_days" in data
        assert "level" in data
        assert "total_quests_completed" in data


# =============================================================================
# DIAGNOSTIC PATH TESTS
# =============================================================================

class TestDiagnosticPathEndpoints:
    """Tests for diagnostic path endpoints."""

    @pytest.mark.asyncio
    async def test_get_diagnostic_paths(self, async_client, test_user_id):
        """Test getting diagnostic paths for a session."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "SRVO-063 encoder error"},
        )
        session_id = create_response.json()["id"]

        # Get diagnostic paths
        response = await async_client.get(
            f"/api/v1/troubleshooting/sessions/{session_id}/paths"
        )

        assert response.status_code == 200
        paths = response.json()

        assert isinstance(paths, list)
        # Paths may be empty if PDF API is not available


# =============================================================================
# STEP COMPLETION TESTS
# =============================================================================

class TestStepCompletion:
    """Tests for step completion endpoints."""

    @pytest.mark.asyncio
    async def test_complete_step(self, async_client, test_user_id):
        """Test completing a diagnostic step."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/troubleshooting/sessions",
            params={"user_id": test_user_id},
            json={"query": "SRVO-063 for step test"},
        )
        session = create_response.json()
        session_id = session["id"]

        # We need to set up the session with steps first
        # For now, test that the endpoint exists and handles requests

        # Complete step 0 (may fail if no steps defined, but endpoint should work)
        response = await async_client.post(
            f"/api/v1/troubleshooting/sessions/{session_id}/steps/0/complete",
            json={
                "user_notes": "Checked encoder cable",
                "evidence_data": {"checked": True},
            },
        )

        # May be 400 (invalid step) or 200 (success) depending on session setup
        assert response.status_code in [200, 400]
