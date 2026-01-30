"""
Troubleshooting Service for Recovery Bot memOS

Manages troubleshooting sessions, task execution tracking, and expertise progression.
Integrates with PDF_Extraction_Tools knowledge graph for diagnostic workflows.

See: QUEST_SYSTEM_REIMPLEMENTATION_PLAN.md for architecture details.
"""

import asyncio
import logging
import re
from datetime import datetime, date, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.troubleshooting import (
    # Enumerations
    TroubleshootingCategory,
    SessionState,
    TaskState,
    TaskExecutionType,
    ExpertiseLevel,
    TroubleshootingDomain,
    # SQLAlchemy Models
    TroubleshootingWorkflow,
    WorkflowTask,
    TroubleshootingSession,
    TaskExecution,
    UserExpertise,
    TroubleshootingAchievement,
    UserTroubleshootingAchievement,
    # Request/Response Models
    CreateSessionRequest,
    SelectPathRequest,
    CompleteStepRequest,
    ResolveSessionRequest,
)
from core.document_graph_service import document_graph_service, TraversalPath, TroubleshootingStep
from config.settings import get_settings
from config.logging_config import get_audit_logger

logger = logging.getLogger(__name__)
audit_logger = get_audit_logger()
settings = get_settings()


# =============================================================================
# ERROR CODE PATTERNS FOR AUTO-DETECTION
# =============================================================================

ERROR_CODE_PATTERNS = {
    TroubleshootingDomain.FANUC_SERVO: [
        r"SRVO-\d{3}",
        r"SV-\d{3}",
    ],
    TroubleshootingDomain.FANUC_MOTION: [
        r"MOTN-\d{3}",
        r"MO-\d{3}",
    ],
    TroubleshootingDomain.FANUC_SYSTEM: [
        r"SYST-\d{3}",
        r"SYS-\d{3}",
        r"INTP-\d{3}",
    ],
    TroubleshootingDomain.IMM_DEFECTS: [
        r"short\s*shot",
        r"flash",
        r"sink\s*mark",
        r"weld\s*line",
        r"burn\s*mark",
        r"warpage",
        r"jetting",
    ],
    TroubleshootingDomain.ELECTRICAL: [
        r"E-\d{3,4}",
        r"EL-\d{3}",
    ],
}

SYMPTOM_KEYWORDS = {
    TroubleshootingDomain.FANUC_SERVO: [
        "servo", "motor", "encoder", "pulsecoder", "overload", "overcurrent",
        "position error", "torque", "vibration", "jerky", "hunting",
    ],
    TroubleshootingDomain.FANUC_MOTION: [
        "motion", "movement", "speed", "acceleration", "trajectory",
        "collision", "singularity", "axis", "joint",
    ],
    TroubleshootingDomain.IMM_DEFECTS: [
        "injection", "molding", "plastic", "mold", "defect", "quality",
        "pressure", "temperature", "cycle", "screw", "barrel",
    ],
    TroubleshootingDomain.ELECTRICAL: [
        "circuit", "wire", "connector", "voltage", "current", "ground",
        "short", "open", "resistance",
    ],
}


class TroubleshootingService:
    """
    Service for managing troubleshooting sessions and task execution.

    Integrates with:
    - PDF_Extraction_Tools for graph traversal
    - Orchestrator for automatic task tracking
    - Memory service for expertise tracking
    """

    def __init__(self):
        self.document_graph = document_graph_service

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    async def create_session(
        self,
        session: AsyncSession,
        user_id: str,
        request: CreateSessionRequest,
    ) -> TroubleshootingSession:
        """
        Create a new troubleshooting session.

        Analyzes the query to:
        1. Detect error codes and symptoms
        2. Determine entry type and domain
        3. Find matching workflow (if any)
        4. Initialize session with detected context
        """
        try:
            query = request.query.strip()

            # Detect error codes and symptoms
            detected_codes = self._detect_error_codes(query)
            detected_symptoms = self._detect_symptoms(query)

            # Determine entry type
            if detected_codes:
                entry_type = "error_code"
            elif detected_symptoms:
                entry_type = "symptom"
            else:
                entry_type = "general"

            # Determine domain
            domain = request.domain
            if not domain:
                domain = self._infer_domain(detected_codes, detected_symptoms, query)

            # Find matching workflow
            workflow_id = None
            if request.workflow_id:
                workflow_id = UUID(request.workflow_id)
            else:
                workflow = await self._find_matching_workflow(
                    session, entry_type, domain, detected_codes
                )
                if workflow:
                    workflow_id = workflow.id

            # Create session
            ts_session = TroubleshootingSession(
                user_id=user_id,
                workflow_id=workflow_id,
                original_query=query,
                detected_error_codes=detected_codes,
                detected_symptoms=detected_symptoms,
                entry_type=entry_type,
                domain=domain,
                state=SessionState.INITIATED,
                started_at=datetime.now(timezone.utc),
                session_metadata={
                    "user_agent": "memOS/1.0",
                    "api_version": "v1",
                }
            )
            session.add(ts_session)
            await session.flush()

            # Create task executions if workflow exists
            if workflow_id:
                workflow = await self._get_workflow_with_tasks(session, workflow_id)
                if workflow:
                    ts_session.total_tasks = len(workflow.tasks)
                    for task in workflow.tasks:
                        execution = TaskExecution(
                            session_id=ts_session.id,
                            task_id=task.id,
                            state=TaskState.PENDING,
                        )
                        session.add(execution)

            await session.commit()
            await session.refresh(ts_session)

            # Audit log
            audit_logger.info(
                "Troubleshooting session created",
                extra={
                    "user_id": user_id,
                    "session_id": str(ts_session.id),
                    "entry_type": entry_type,
                    "domain": domain,
                    "detected_codes": detected_codes,
                    "action": "session_created",
                }
            )

            return ts_session

        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to create troubleshooting session: {e}")
            raise

    async def get_session(
        self,
        session: AsyncSession,
        session_id: str,
    ) -> Optional[TroubleshootingSession]:
        """Get a troubleshooting session by ID with related data."""
        try:
            result = await session.execute(
                select(TroubleshootingSession)
                .where(TroubleshootingSession.id == UUID(session_id))
                .options(
                    selectinload(TroubleshootingSession.workflow)
                    .selectinload(TroubleshootingWorkflow.tasks),
                    selectinload(TroubleshootingSession.task_executions)
                    .selectinload(TaskExecution.task),
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    async def get_user_sessions(
        self,
        session: AsyncSession,
        user_id: str,
        state: Optional[SessionState] = None,
        domain: Optional[str] = None,
        limit: int = 20,
        include_completed: bool = True,
    ) -> List[TroubleshootingSession]:
        """Get troubleshooting sessions for a user."""
        try:
            query = select(TroubleshootingSession).where(
                TroubleshootingSession.user_id == user_id
            )

            if state:
                query = query.where(TroubleshootingSession.state == state)
            elif not include_completed:
                query = query.where(
                    TroubleshootingSession.state.notin_([
                        SessionState.RESOLVED,
                        SessionState.ESCALATED,
                        SessionState.ABANDONED,
                    ])
                )

            if domain:
                query = query.where(TroubleshootingSession.domain == domain)

            query = query.options(
                selectinload(TroubleshootingSession.workflow),
                selectinload(TroubleshootingSession.task_executions),
            ).order_by(TroubleshootingSession.started_at.desc()).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get sessions for user {user_id}: {e}")
            return []

    async def get_active_session(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> Optional[TroubleshootingSession]:
        """Get the user's currently active session (if any)."""
        sessions = await self.get_user_sessions(
            session, user_id, include_completed=False, limit=1
        )
        return sessions[0] if sessions else None

    # =========================================================================
    # GRAPH TRAVERSAL & PATH SELECTION
    # =========================================================================

    async def get_diagnostic_paths(
        self,
        session: AsyncSession,
        session_id: str,
        max_paths: int = 3,
    ) -> List[TraversalPath]:
        """
        Get diagnostic paths from the knowledge graph for a session.

        Uses PDF_Extraction_Tools PathRAG to find relevant troubleshooting paths.
        """
        ts_session = await self.get_session(session, session_id)
        if not ts_session:
            raise ValueError(f"Session not found: {session_id}")

        # Update state
        ts_session.state = SessionState.IN_PROGRESS
        await session.commit()

        # Query the document graph
        paths = []

        try:
            if ts_session.detected_error_codes:
                # Use error code as entry point
                for code in ts_session.detected_error_codes[:2]:  # Limit to 2 codes
                    path_results = await self.document_graph.query_troubleshooting_path(
                        error_code=code,
                        max_hops=5,
                    )
                    if path_results:
                        paths.extend(path_results[:max_paths])

            elif ts_session.detected_symptoms:
                # Use symptom-based traversal
                symptom_query = " ".join(ts_session.detected_symptoms[:3])
                path_results = await self.document_graph.query_by_symptom(
                    symptom_description=symptom_query,
                    domain=ts_session.domain,
                    max_hops=5,
                )
                if path_results:
                    paths.extend(path_results[:max_paths])

            else:
                # General search
                search_results = await self.document_graph.search_documentation(
                    query=ts_session.original_query,
                    max_results=10,
                )
                # Convert search results to simple paths
                for result in search_results[:max_paths]:
                    path = TraversalPath(
                        path_id=str(uuid4()),
                        steps=[TroubleshootingStep(
                            node_id=result.node_id,
                            title=result.title,
                            content=result.content_preview,
                            step_type="info",
                            relevance_score=result.score,
                            hop_number=0,
                        )],
                        total_score=result.score,
                        path_type="search",
                    )
                    paths.append(path)

            # Store paths in session
            ts_session.paths_presented = [
                {
                    "path_id": p.path_id,
                    "total_score": p.total_score,
                    "path_type": p.path_type,
                    "step_count": len(p.steps),
                }
                for p in paths
            ]
            await session.commit()

        except Exception as e:
            logger.error(f"Failed to get diagnostic paths: {e}")
            # Return empty list on failure

        return paths[:max_paths]

    async def select_path(
        self,
        session: AsyncSession,
        session_id: str,
        path_id: str,
        steps: List[TroubleshootingStep],
    ) -> TroubleshootingSession:
        """
        Record user's selection of a diagnostic path.

        Updates session state and tracks the selected path for progress monitoring.
        """
        ts_session = await self.get_session(session, session_id)
        if not ts_session:
            raise ValueError(f"Session not found: {session_id}")

        ts_session.selected_path_id = path_id
        ts_session.state = SessionState.PATH_SELECTED
        ts_session.total_steps = len(steps)
        ts_session.current_step_index = 0
        ts_session.completed_steps = []

        await session.commit()
        await session.refresh(ts_session)

        audit_logger.info(
            "Diagnostic path selected",
            extra={
                "user_id": ts_session.user_id,
                "session_id": session_id,
                "path_id": path_id,
                "total_steps": len(steps),
                "action": "path_selected",
            }
        )

        return ts_session

    async def complete_step(
        self,
        session: AsyncSession,
        session_id: str,
        step_index: int,
        user_notes: Optional[str] = None,
        evidence_data: Optional[Dict[str, Any]] = None,
    ) -> TroubleshootingSession:
        """Mark a diagnostic step as completed."""
        ts_session = await self.get_session(session, session_id)
        if not ts_session:
            raise ValueError(f"Session not found: {session_id}")

        if step_index < 0 or step_index >= ts_session.total_steps:
            raise ValueError(f"Invalid step index: {step_index}")

        # Update completed steps
        completed = ts_session.completed_steps or []
        if step_index not in completed:
            completed.append(step_index)
            ts_session.completed_steps = sorted(completed)

        # Update current step
        if step_index >= ts_session.current_step_index:
            ts_session.current_step_index = step_index + 1

        # Store notes/evidence in metadata
        if user_notes or evidence_data:
            metadata = ts_session.session_metadata or {}
            step_data = metadata.get("step_data", {})
            step_data[str(step_index)] = {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "user_notes": user_notes,
                "evidence": evidence_data,
            }
            metadata["step_data"] = step_data
            ts_session.session_metadata = metadata

        await session.commit()
        await session.refresh(ts_session)

        return ts_session

    # =========================================================================
    # TASK EXECUTION TRACKING
    # =========================================================================

    async def start_task_execution(
        self,
        session: AsyncSession,
        session_id: str,
        task_id: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> TaskExecution:
        """Start execution of a task within a session."""
        result = await session.execute(
            select(TaskExecution).where(
                and_(
                    TaskExecution.session_id == UUID(session_id),
                    TaskExecution.task_id == UUID(task_id),
                )
            )
        )
        execution = result.scalar_one_or_none()

        if not execution:
            raise ValueError(f"Task execution not found: {task_id}")

        execution.state = TaskState.RUNNING
        execution.started_at = datetime.now(timezone.utc)
        execution.input_data = input_data or {}

        await session.commit()
        await session.refresh(execution)

        return execution

    async def complete_task_execution(
        self,
        session: AsyncSession,
        session_id: str,
        task_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        verification_passed: Optional[bool] = None,
        error_message: Optional[str] = None,
    ) -> TaskExecution:
        """Complete execution of a task."""
        result = await session.execute(
            select(TaskExecution).where(
                and_(
                    TaskExecution.session_id == UUID(session_id),
                    TaskExecution.task_id == UUID(task_id),
                )
            ).options(selectinload(TaskExecution.task))
        )
        execution = result.scalar_one_or_none()

        if not execution:
            raise ValueError(f"Task execution not found: {task_id}")

        execution.completed_at = datetime.now(timezone.utc)
        execution.output_data = output_data or {}
        execution.metrics = metrics or {}
        execution.verification_passed = verification_passed

        if error_message:
            execution.state = TaskState.FAILED
            execution.error_message = error_message
        else:
            execution.state = TaskState.COMPLETED

        # Update session task count
        ts_session = await self.get_session(session, session_id)
        if ts_session:
            ts_session.completed_tasks = (ts_session.completed_tasks or 0) + 1

        await session.commit()
        await session.refresh(execution)

        return execution

    async def record_user_action(
        self,
        session: AsyncSession,
        session_id: str,
        task_id: str,
        user_input: Dict[str, Any],
        user_notes: Optional[str] = None,
    ) -> TaskExecution:
        """Record user action for a USER_ACTION or HYBRID task."""
        result = await session.execute(
            select(TaskExecution).where(
                and_(
                    TaskExecution.session_id == UUID(session_id),
                    TaskExecution.task_id == UUID(task_id),
                )
            )
        )
        execution = result.scalar_one_or_none()

        if not execution:
            raise ValueError(f"Task execution not found: {task_id}")

        execution.user_input = user_input
        execution.user_notes = user_notes
        execution.state = TaskState.COMPLETED
        execution.completed_at = datetime.now(timezone.utc)

        await session.commit()
        await session.refresh(execution)

        return execution

    # =========================================================================
    # SESSION RESOLUTION
    # =========================================================================

    async def resolve_session(
        self,
        session: AsyncSession,
        session_id: str,
        resolution_type: str,
        rating: Optional[int] = None,
        feedback: Optional[str] = None,
    ) -> TroubleshootingSession:
        """
        Resolve a troubleshooting session.

        resolution_type: "self_resolved", "escalated", "abandoned"
        """
        ts_session = await self.get_session(session, session_id)
        if not ts_session:
            raise ValueError(f"Session not found: {session_id}")

        ts_session.completed_at = datetime.now(timezone.utc)
        ts_session.resolution_type = resolution_type
        ts_session.user_rating = rating
        ts_session.user_feedback = feedback

        # Set final state
        if resolution_type == "self_resolved":
            ts_session.state = SessionState.RESOLVED
        elif resolution_type == "escalated":
            ts_session.state = SessionState.ESCALATED
        else:
            ts_session.state = SessionState.ABANDONED

        # Calculate resolution time (handle timezone-aware/naive comparison)
        if ts_session.started_at:
            started = ts_session.started_at
            completed = ts_session.completed_at
            # Normalize both to UTC if needed
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if completed.tzinfo is None:
                completed = completed.replace(tzinfo=timezone.utc)
            delta = completed - started
            ts_session.resolution_time_seconds = delta.total_seconds()

        # Award expertise points for successful resolution
        if resolution_type == "self_resolved":
            points = await self._calculate_expertise_points(session, ts_session)
            ts_session.expertise_points_earned = points
            await self._update_user_expertise(session, ts_session.user_id, ts_session)

        await session.commit()
        await session.refresh(ts_session)

        audit_logger.info(
            "Troubleshooting session resolved",
            extra={
                "user_id": ts_session.user_id,
                "session_id": session_id,
                "resolution_type": resolution_type,
                "points_earned": ts_session.expertise_points_earned,
                "resolution_time": ts_session.resolution_time_seconds,
                "action": "session_resolved",
            }
        )

        return ts_session

    # =========================================================================
    # EXPERTISE TRACKING
    # =========================================================================

    async def get_user_expertise(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> UserExpertise:
        """Get or create user expertise record."""
        result = await session.execute(
            select(UserExpertise).where(UserExpertise.user_id == user_id)
        )
        expertise = result.scalar_one_or_none()

        if not expertise:
            expertise = UserExpertise(
                user_id=user_id,
                total_expertise_points=0,
                expertise_level=ExpertiseLevel.NOVICE,
                domain_points={},
                domains_mastered=[],
                total_sessions=0,
                successful_resolutions=0,
            )
            session.add(expertise)
            await session.commit()
            await session.refresh(expertise)

        return expertise

    async def _update_user_expertise(
        self,
        session: AsyncSession,
        user_id: str,
        ts_session: TroubleshootingSession,
    ) -> UserExpertise:
        """Update user expertise after session resolution."""
        expertise = await self.get_user_expertise(session, user_id)

        # Update totals
        expertise.total_sessions = (expertise.total_sessions or 0) + 1
        expertise.total_expertise_points = (
            (expertise.total_expertise_points or 0) +
            (ts_session.expertise_points_earned or 0)
        )

        if ts_session.state == SessionState.RESOLVED:
            expertise.successful_resolutions = (expertise.successful_resolutions or 0) + 1

        # Update domain-specific points
        if ts_session.domain:
            domain_points = expertise.domain_points or {}
            current = domain_points.get(ts_session.domain, 0)
            domain_points[ts_session.domain] = current + (ts_session.expertise_points_earned or 0)
            expertise.domain_points = domain_points

            # Check for domain mastery
            if domain_points[ts_session.domain] >= 100:
                mastered = expertise.domains_mastered or []
                if ts_session.domain not in mastered:
                    mastered.append(ts_session.domain)
                    expertise.domains_mastered = mastered

        # Update resolution time average
        if ts_session.resolution_time_seconds:
            if expertise.avg_resolution_time_seconds:
                # Running average
                n = expertise.successful_resolutions
                current_avg = expertise.avg_resolution_time_seconds
                expertise.avg_resolution_time_seconds = (
                    (current_avg * (n - 1) + ts_session.resolution_time_seconds) / n
                )
            else:
                expertise.avg_resolution_time_seconds = ts_session.resolution_time_seconds

            # Track fastest
            if not expertise.fastest_resolution_seconds or \
               ts_session.resolution_time_seconds < expertise.fastest_resolution_seconds:
                expertise.fastest_resolution_seconds = ts_session.resolution_time_seconds

        # Update streak
        await self._update_streak(expertise)

        # Calculate new level
        expertise.expertise_level = self._calculate_level(expertise.total_expertise_points)

        # Update periodic counters
        expertise.weekly_sessions = (expertise.weekly_sessions or 0) + 1
        expertise.monthly_sessions = (expertise.monthly_sessions or 0) + 1
        if ts_session.state == SessionState.RESOLVED:
            expertise.weekly_resolutions = (expertise.weekly_resolutions or 0) + 1
            expertise.monthly_resolutions = (expertise.monthly_resolutions or 0) + 1

        expertise.updated_at = datetime.now(timezone.utc)

        # Check for achievements
        await self._check_achievements(session, user_id, expertise, ts_session)

        await session.commit()
        return expertise

    async def _calculate_expertise_points(
        self,
        session: AsyncSession,
        ts_session: TroubleshootingSession,
    ) -> int:
        """Calculate expertise points for a resolved session."""
        base_points = 10

        # Workflow bonus
        if ts_session.workflow_id:
            workflow = await self._get_workflow_with_tasks(session, ts_session.workflow_id)
            if workflow:
                base_points = workflow.expertise_points or 10

        # Completion bonus (all steps completed)
        if ts_session.total_steps > 0:
            completion_rate = len(ts_session.completed_steps or []) / ts_session.total_steps
            if completion_rate >= 1.0:
                base_points = int(base_points * 1.25)  # 25% bonus

        # Speed bonus (under 5 minutes)
        if ts_session.resolution_time_seconds and ts_session.resolution_time_seconds < 300:
            base_points = int(base_points * 1.1)  # 10% bonus

        return base_points

    def _calculate_level(self, total_points: int) -> ExpertiseLevel:
        """Calculate expertise level based on total points."""
        if total_points >= 2000:
            return ExpertiseLevel.EXPERT
        elif total_points >= 500:
            return ExpertiseLevel.SPECIALIST
        elif total_points >= 100:
            return ExpertiseLevel.TECHNICIAN
        else:
            return ExpertiseLevel.NOVICE

    async def _update_streak(self, expertise: UserExpertise):
        """Update user's activity streak."""
        today = date.today()

        if expertise.last_activity_date:
            days_diff = (today - expertise.last_activity_date).days

            if days_diff == 0:
                pass  # Same day activity
            elif days_diff == 1:
                expertise.current_streak_days = (expertise.current_streak_days or 0) + 1
            else:
                expertise.current_streak_days = 1
        else:
            expertise.current_streak_days = 1

        expertise.last_activity_date = today
        expertise.longest_streak_days = max(
            expertise.longest_streak_days or 0,
            expertise.current_streak_days or 0
        )

    # =========================================================================
    # ACHIEVEMENTS
    # =========================================================================

    async def _check_achievements(
        self,
        session: AsyncSession,
        user_id: str,
        expertise: UserExpertise,
        ts_session: TroubleshootingSession,
    ):
        """Check and award achievements based on user progress."""
        try:
            # Get all active achievements
            result = await session.execute(
                select(TroubleshootingAchievement).where(
                    TroubleshootingAchievement.is_active == True
                )
            )
            achievements = result.scalars().all()

            # Get user's existing achievements
            result = await session.execute(
                select(UserTroubleshootingAchievement.achievement_id).where(
                    UserTroubleshootingAchievement.user_id == user_id
                )
            )
            existing_ids = [row[0] for row in result.fetchall()]

            for achievement in achievements:
                if achievement.id in existing_ids:
                    continue

                criteria_met = False

                if achievement.criteria_type == "sessions_completed":
                    criteria_met = (expertise.total_sessions or 0) >= achievement.criteria_value
                elif achievement.criteria_type == "points_earned":
                    criteria_met = (expertise.total_expertise_points or 0) >= achievement.criteria_value
                elif achievement.criteria_type == "streak_days":
                    criteria_met = (expertise.current_streak_days or 0) >= achievement.criteria_value
                elif achievement.criteria_type == "domain_mastered":
                    if achievement.domain:
                        domain_pts = (expertise.domain_points or {}).get(achievement.domain, 0)
                        criteria_met = domain_pts >= achievement.criteria_value

                if criteria_met:
                    user_achievement = UserTroubleshootingAchievement(
                        user_id=user_id,
                        achievement_id=achievement.id,
                        session_id=ts_session.id,
                        domain=ts_session.domain,
                        points_awarded=achievement.criteria_value // 10,  # Bonus points
                    )
                    session.add(user_achievement)

                    # Add bonus points
                    expertise.total_expertise_points += user_achievement.points_awarded

                    logger.info(
                        f"Achievement unlocked for user {user_id}: {achievement.title}"
                    )

        except Exception as e:
            logger.error(f"Failed to check achievements: {e}")

    async def get_user_achievements(
        self,
        session: AsyncSession,
        user_id: str,
    ) -> List[UserTroubleshootingAchievement]:
        """Get user's earned achievements."""
        result = await session.execute(
            select(UserTroubleshootingAchievement)
            .where(UserTroubleshootingAchievement.user_id == user_id)
            .options(selectinload(UserTroubleshootingAchievement.achievement))
            .order_by(UserTroubleshootingAchievement.earned_at.desc())
        )
        return list(result.scalars().all())

    # =========================================================================
    # WORKFLOW MANAGEMENT
    # =========================================================================

    async def get_available_workflows(
        self,
        session: AsyncSession,
        domain: Optional[str] = None,
        category: Optional[TroubleshootingCategory] = None,
    ) -> List[TroubleshootingWorkflow]:
        """Get available troubleshooting workflows."""
        query = select(TroubleshootingWorkflow).where(
            TroubleshootingWorkflow.is_active == True
        )

        if domain:
            query = query.where(TroubleshootingWorkflow.domain == domain)

        if category:
            query = query.where(TroubleshootingWorkflow.category == category.value)

        query = query.options(selectinload(TroubleshootingWorkflow.tasks))

        result = await session.execute(query)
        return list(result.scalars().all())

    async def _get_workflow_with_tasks(
        self,
        session: AsyncSession,
        workflow_id: UUID,
    ) -> Optional[TroubleshootingWorkflow]:
        """Get workflow with its tasks."""
        result = await session.execute(
            select(TroubleshootingWorkflow)
            .where(TroubleshootingWorkflow.id == workflow_id)
            .options(selectinload(TroubleshootingWorkflow.tasks))
        )
        return result.scalar_one_or_none()

    async def get_workflow_name(
        self,
        session: AsyncSession,
        workflow_id: Optional[UUID],
    ) -> Optional[str]:
        """Get workflow name by ID (lightweight, no relationships loaded)."""
        if not workflow_id:
            return None
        result = await session.execute(
            select(TroubleshootingWorkflow.name)
            .where(TroubleshootingWorkflow.id == workflow_id)
        )
        return result.scalar_one_or_none()

    async def _find_matching_workflow(
        self,
        session: AsyncSession,
        entry_type: str,
        domain: Optional[str],
        error_codes: List[str],
    ) -> Optional[TroubleshootingWorkflow]:
        """Find a workflow matching the query context."""
        query = select(TroubleshootingWorkflow).where(
            TroubleshootingWorkflow.is_active == True
        )

        if domain:
            query = query.where(TroubleshootingWorkflow.domain == domain)

        if entry_type == "error_code":
            query = query.where(
                TroubleshootingWorkflow.category == TroubleshootingCategory.ERROR_DIAGNOSIS.value
            )
        elif entry_type == "symptom":
            query = query.where(
                TroubleshootingWorkflow.category == TroubleshootingCategory.SYMPTOM_ANALYSIS.value
            )

        result = await session.execute(query.limit(1))
        return result.scalar_one_or_none()

    # =========================================================================
    # DETECTION HELPERS
    # =========================================================================

    def _detect_error_codes(self, query: str) -> List[str]:
        """Detect error codes in query using regex patterns."""
        detected = []
        query_upper = query.upper()

        for domain, patterns in ERROR_CODE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_upper, re.IGNORECASE)
                detected.extend(matches)

        return list(set(detected))

    def _detect_symptoms(self, query: str) -> List[str]:
        """Detect symptom keywords in query."""
        detected = []
        query_lower = query.lower()

        for domain, keywords in SYMPTOM_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    detected.append(keyword)

        return list(set(detected))

    def _infer_domain(
        self,
        error_codes: List[str],
        symptoms: List[str],
        query: str,
    ) -> Optional[str]:
        """Infer the troubleshooting domain from detected patterns."""
        query_upper = query.upper()

        # Check error code patterns
        for domain, patterns in ERROR_CODE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_upper):
                    return domain.value

        # Check symptom keywords
        query_lower = query.lower()
        domain_scores = {}
        for domain, keywords in SYMPTOM_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get).value

        return None

    # =========================================================================
    # LEADERBOARD
    # =========================================================================

    async def get_leaderboard(
        self,
        session: AsyncSession,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> List[UserExpertise]:
        """Get expertise leaderboard."""
        query = select(UserExpertise)

        if domain:
            # Filter by domain expertise (JSON field)
            # Note: This is a simplified version; production would use PostgreSQL JSON operators
            pass

        query = query.order_by(
            UserExpertise.total_expertise_points.desc()
        ).limit(limit)

        result = await session.execute(query)
        return list(result.scalars().all())


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

troubleshooting_service = TroubleshootingService()
