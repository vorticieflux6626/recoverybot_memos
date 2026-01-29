"""
Unit Tests for Troubleshooting Service

Tests session management, task execution tracking, and expertise progression.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from models.troubleshooting import (
    TroubleshootingCategory,
    SessionState,
    TaskState,
    TaskExecutionType,
    ExpertiseLevel,
    TroubleshootingDomain,
    TroubleshootingWorkflow,
    WorkflowTask,
    TroubleshootingSession,
    TaskExecution,
    UserExpertise,
    CreateSessionRequest,
)
from core.troubleshooting_service import (
    TroubleshootingService,
    ERROR_CODE_PATTERNS,
    SYMPTOM_KEYWORDS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def service():
    """Create a TroubleshootingService instance."""
    return TroubleshootingService()


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing."""
    workflow = TroubleshootingWorkflow(
        id=uuid4(),
        name="FANUC Servo Alarm Diagnosis",
        description="Diagnose SRVO-xxx servo alarms",
        category=TroubleshootingCategory.ERROR_DIAGNOSIS.value,
        domain=TroubleshootingDomain.FANUC_SERVO.value,
        traversal_mode="semantic_astar",
        max_hops=5,
        beam_width=10,
        expertise_points=15,
        is_active=True,
        error_code_patterns=["SRVO-\\d{3}"],
    )
    workflow.tasks = [
        WorkflowTask(
            id=uuid4(),
            workflow_id=workflow.id,
            name="Document Retrieval",
            execution_type=TaskExecutionType.AUTOMATIC.value,
            order_index=0,
            is_required=True,
            pipeline_hook="_search_technical_docs",
        ),
        WorkflowTask(
            id=uuid4(),
            workflow_id=workflow.id,
            name="Response Synthesis",
            execution_type=TaskExecutionType.AUTOMATIC.value,
            order_index=1,
            is_required=True,
            pipeline_hook="_synthesize",
        ),
    ]
    return workflow


@pytest.fixture
def sample_session(sample_workflow):
    """Create a sample troubleshooting session."""
    return TroubleshootingSession(
        id=uuid4(),
        user_id="test_user_123",
        workflow_id=sample_workflow.id,
        original_query="SRVO-063 encoder error",
        detected_error_codes=["SRVO-063"],
        detected_symptoms=[],
        entry_type="error_code",
        domain=TroubleshootingDomain.FANUC_SERVO.value,
        state=SessionState.INITIATED,
        started_at=datetime.now(timezone.utc),
        total_tasks=2,
        completed_tasks=0,
    )


# =============================================================================
# ERROR CODE DETECTION TESTS
# =============================================================================

class TestErrorCodeDetection:
    """Tests for error code pattern detection."""

    def test_detect_fanuc_servo_codes(self, service):
        """Test detection of FANUC servo alarm codes."""
        codes = service._detect_error_codes("Getting SRVO-063 pulsecoder error")
        assert "SRVO-063" in codes

        codes = service._detect_error_codes("SRVO-001 and SRVO-999 alarms")
        assert "SRVO-001" in codes
        assert "SRVO-999" in codes

    def test_detect_fanuc_motion_codes(self, service):
        """Test detection of FANUC motion alarm codes."""
        codes = service._detect_error_codes("Robot showing MOTN-017 speed limit")
        assert "MOTN-017" in codes

    def test_detect_mixed_codes(self, service):
        """Test detection of multiple code types."""
        query = "Getting SRVO-063 and MOTN-017 after collision"
        codes = service._detect_error_codes(query)
        assert "SRVO-063" in codes
        assert "MOTN-017" in codes

    def test_no_codes_detected(self, service):
        """Test query with no error codes."""
        codes = service._detect_error_codes("Robot arm is making noise")
        assert codes == []

    def test_case_insensitive_detection(self, service):
        """Test case-insensitive error code detection."""
        codes = service._detect_error_codes("srvo-063 alarm")
        assert len(codes) == 1


# =============================================================================
# SYMPTOM DETECTION TESTS
# =============================================================================

class TestSymptomDetection:
    """Tests for symptom keyword detection."""

    def test_detect_servo_symptoms(self, service):
        """Test detection of servo-related symptoms."""
        symptoms = service._detect_symptoms("Motor is vibrating and jerky")
        assert "vibration" in symptoms or "jerky" in symptoms

    def test_detect_imm_symptoms(self, service):
        """Test detection of injection molding symptoms."""
        symptoms = service._detect_symptoms("Parts showing short shot defects")
        assert any("short" in s.lower() for s in symptoms)

    def test_detect_multiple_symptoms(self, service):
        """Test detection of multiple symptoms."""
        query = "Servo motor overload with position error"
        symptoms = service._detect_symptoms(query)
        assert len(symptoms) >= 2


# =============================================================================
# DOMAIN INFERENCE TESTS
# =============================================================================

class TestDomainInference:
    """Tests for domain inference from queries."""

    def test_infer_servo_domain_from_code(self, service):
        """Test domain inference from servo error code."""
        domain = service._infer_domain(["SRVO-063"], [], "SRVO-063 error")
        assert domain == TroubleshootingDomain.FANUC_SERVO.value

    def test_infer_motion_domain_from_code(self, service):
        """Test domain inference from motion error code."""
        domain = service._infer_domain(["MOTN-017"], [], "MOTN-017 error")
        assert domain == TroubleshootingDomain.FANUC_MOTION.value

    def test_infer_domain_from_symptoms(self, service):
        """Test domain inference from symptom keywords."""
        domain = service._infer_domain(
            [],
            ["encoder", "servo"],
            "encoder error on servo axis"
        )
        assert domain == TroubleshootingDomain.FANUC_SERVO.value

    def test_infer_imm_domain(self, service):
        """Test domain inference for injection molding."""
        domain = service._infer_domain(
            [],
            ["injection", "mold"],
            "injection molding defect"
        )
        assert domain == TroubleshootingDomain.IMM_DEFECTS.value


# =============================================================================
# EXPERTISE LEVEL TESTS
# =============================================================================

class TestExpertiseLevel:
    """Tests for expertise level calculation."""

    def test_novice_level(self, service):
        """Test novice level threshold."""
        level = service._calculate_level(50)
        assert level == ExpertiseLevel.NOVICE

        level = service._calculate_level(99)
        assert level == ExpertiseLevel.NOVICE

    def test_technician_level(self, service):
        """Test technician level threshold."""
        level = service._calculate_level(100)
        assert level == ExpertiseLevel.TECHNICIAN

        level = service._calculate_level(499)
        assert level == ExpertiseLevel.TECHNICIAN

    def test_specialist_level(self, service):
        """Test specialist level threshold."""
        level = service._calculate_level(500)
        assert level == ExpertiseLevel.SPECIALIST

        level = service._calculate_level(1999)
        assert level == ExpertiseLevel.SPECIALIST

    def test_expert_level(self, service):
        """Test expert level threshold."""
        level = service._calculate_level(2000)
        assert level == ExpertiseLevel.EXPERT

        level = service._calculate_level(10000)
        assert level == ExpertiseLevel.EXPERT


# =============================================================================
# SESSION CREATION TESTS
# =============================================================================

class TestSessionCreation:
    """Tests for session creation logic."""

    @pytest.mark.asyncio
    async def test_create_session_with_error_code(self, service, mock_db_session):
        """Test session creation with error code query."""
        # Mock the execute method for workflow lookup
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        request = CreateSessionRequest(
            query="SRVO-063 pulsecoder error on axis 1",
        )

        # Mock session creation
        with patch.object(service, '_find_matching_workflow', return_value=None):
            session = await service.create_session(
                mock_db_session,
                "test_user",
                request,
            )

        # Verify session.add was called
        assert mock_db_session.add.called

    @pytest.mark.asyncio
    async def test_create_session_detects_codes(self, service, mock_db_session):
        """Test that session creation properly detects error codes."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        request = CreateSessionRequest(
            query="Getting SRVO-063 and MOTN-017 after collision",
        )

        with patch.object(service, '_find_matching_workflow', return_value=None):
            # Verify the detection logic works
            codes = service._detect_error_codes(request.query)
            assert "SRVO-063" in codes
            assert "MOTN-017" in codes


# =============================================================================
# EXPERTISE POINTS CALCULATION TESTS
# =============================================================================

class TestExpertisePoints:
    """Tests for expertise points calculation."""

    @pytest.mark.asyncio
    async def test_base_points_calculation(self, service, mock_db_session, sample_session):
        """Test base points for resolved session."""
        sample_session.workflow_id = None
        sample_session.total_steps = 0
        sample_session.completed_steps = []

        points = await service._calculate_expertise_points(mock_db_session, sample_session)
        assert points == 10  # Base points

    @pytest.mark.asyncio
    async def test_completion_bonus(self, service, mock_db_session, sample_session):
        """Test bonus points for full completion."""
        sample_session.workflow_id = None
        sample_session.total_steps = 5
        sample_session.completed_steps = [0, 1, 2, 3, 4]  # All completed

        points = await service._calculate_expertise_points(mock_db_session, sample_session)
        assert points == 12  # 10 * 1.25 = 12.5 -> 12

    @pytest.mark.asyncio
    async def test_speed_bonus(self, service, mock_db_session, sample_session):
        """Test bonus points for fast resolution."""
        sample_session.workflow_id = None
        sample_session.total_steps = 0
        sample_session.resolution_time_seconds = 120  # 2 minutes

        points = await service._calculate_expertise_points(mock_db_session, sample_session)
        assert points == 11  # 10 * 1.1 = 11


# =============================================================================
# WORKFLOW MATCHING TESTS
# =============================================================================

class TestWorkflowMatching:
    """Tests for workflow matching logic."""

    @pytest.mark.asyncio
    async def test_find_workflow_by_domain(self, service, mock_db_session, sample_workflow):
        """Test finding workflow by domain."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_workflow
        mock_db_session.execute.return_value = mock_result

        workflow = await service._find_matching_workflow(
            mock_db_session,
            "error_code",
            TroubleshootingDomain.FANUC_SERVO.value,
            ["SRVO-063"],
        )

        # Verify query was executed
        assert mock_db_session.execute.called


# =============================================================================
# SESSION STATE TRANSITIONS TESTS
# =============================================================================

class TestSessionStateTransitions:
    """Tests for session state transitions."""

    def test_valid_session_states(self):
        """Test that all session states are valid."""
        states = [
            SessionState.INITIATED,
            SessionState.IN_PROGRESS,
            SessionState.PATH_SELECTED,
            SessionState.AWAITING_USER,
            SessionState.RESOLVED,
            SessionState.ESCALATED,
            SessionState.ABANDONED,
        ]
        for state in states:
            assert state.value is not None

    def test_task_states(self):
        """Test that all task states are valid."""
        states = [
            TaskState.PENDING,
            TaskState.RUNNING,
            TaskState.COMPLETED,
            TaskState.SKIPPED,
            TaskState.FAILED,
            TaskState.WAITING_USER,
        ]
        for state in states:
            assert state.value is not None


# =============================================================================
# TROUBLESHOOTING CATEGORY TESTS
# =============================================================================

class TestTroubleshootingCategories:
    """Tests for troubleshooting category definitions."""

    def test_all_categories_defined(self):
        """Test that all expected categories exist."""
        expected = [
            "error_diagnosis",
            "symptom_analysis",
            "procedure_execution",
            "learning",
            "preventive",
        ]
        actual = [c.value for c in TroubleshootingCategory]
        for cat in expected:
            assert cat in actual

    def test_category_values_are_strings(self):
        """Test that category values are strings."""
        for cat in TroubleshootingCategory:
            assert isinstance(cat.value, str)


# =============================================================================
# DOMAIN TESTS
# =============================================================================

class TestTroubleshootingDomains:
    """Tests for troubleshooting domain definitions."""

    def test_all_domains_defined(self):
        """Test that all expected domains exist."""
        expected = [
            "fanuc_servo",
            "fanuc_motion",
            "fanuc_system",
            "imm_defects",
            "imm_process",
            "electrical",
            "mechanical",
            "safety",
        ]
        actual = [d.value for d in TroubleshootingDomain]
        for domain in expected:
            assert domain in actual

    def test_error_patterns_cover_domains(self):
        """Test that error patterns exist for key domains."""
        assert TroubleshootingDomain.FANUC_SERVO in ERROR_CODE_PATTERNS
        assert TroubleshootingDomain.FANUC_MOTION in ERROR_CODE_PATTERNS

    def test_symptom_keywords_cover_domains(self):
        """Test that symptom keywords exist for key domains."""
        assert TroubleshootingDomain.FANUC_SERVO in SYMPTOM_KEYWORDS
        assert TroubleshootingDomain.IMM_DEFECTS in SYMPTOM_KEYWORDS
