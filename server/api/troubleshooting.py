"""
Troubleshooting API Endpoints for memOS Server

Provides REST API for troubleshooting task tracking system:
- Session management (create, get, list, resolve)
- Diagnostic path traversal
- Task execution tracking
- Expertise and achievements

See: QUEST_SYSTEM_REIMPLEMENTATION_PLAN.md for architecture details.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config.database import get_db_dependency
from core.troubleshooting_service import troubleshooting_service
from core.troubleshooting_tracker import create_tracker, PIPELINE_HOOKS
from core.document_graph_service import document_graph_service, TraversalPath
from models.troubleshooting import (
    # Enumerations
    TroubleshootingCategory,
    SessionState,
    TaskState,
    ExpertiseLevel,
    TroubleshootingDomain,
    # Response Models
    WorkflowResponse,
    WorkflowTaskResponse,
    SessionResponse,
    SessionDetailResponse,
    TaskExecutionResponse,
    TraversalPathResponse,
    TraversalStepResponse,
    ExpertiseResponse,
    AchievementResponse,
    WorkflowListResponse,
    SessionListResponse,
    LeaderboardEntry,
    LeaderboardResponse,
    # Request Models
    CreateSessionRequest,
    SelectPathRequest,
    CompleteStepRequest,
    ResolveSessionRequest,
    UserActionRequest,
)
from config.logging_config import get_audit_logger

router = APIRouter(prefix="/api/v1/troubleshooting", tags=["troubleshooting"])
audit_logger = get_audit_logger()
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _session_to_response(session, include_tasks: bool = False) -> SessionResponse:
    """Convert SQLAlchemy session to response model."""
    task_executions = []
    if include_tasks and session.task_executions:
        for exec in session.task_executions:
            task_executions.append(TaskExecutionResponse(
                id=str(exec.id),
                task_id=str(exec.task_id) if exec.task_id else "",
                task_name=exec.task.name if exec.task else "Unknown",
                execution_type=exec.task.execution_type if exec.task else "automatic",
                state=exec.state,
                started_at=exec.started_at,
                completed_at=exec.completed_at,
                duration_seconds=(exec.completed_at - exec.started_at).total_seconds()
                    if exec.completed_at and exec.started_at else None,
                verification_passed=exec.verification_passed,
                output_summary=exec.output_data if exec.output_data else None,
                error_message=exec.error_message,
            ))

    return SessionResponse(
        id=str(session.id),
        user_id=session.user_id,
        workflow_id=str(session.workflow_id) if session.workflow_id else None,
        workflow_name=session.workflow.name if session.workflow else None,
        original_query=session.original_query,
        detected_error_codes=session.detected_error_codes or [],
        detected_symptoms=session.detected_symptoms or [],
        entry_type=session.entry_type,
        domain=session.domain,
        state=session.state,
        started_at=session.started_at,
        completed_at=session.completed_at,
        resolution_type=session.resolution_type,
        progress_percentage=session.progress_percentage,
        total_tasks=session.total_tasks or 0,
        completed_tasks=session.completed_tasks or 0,
        current_step_index=session.current_step_index or 0,
        total_steps=session.total_steps or 0,
        expertise_points_earned=session.expertise_points_earned or 0,
        task_executions=task_executions,
    )


def _expertise_to_response(expertise) -> ExpertiseResponse:
    """Convert SQLAlchemy expertise to response model."""
    # Calculate next level info
    level_thresholds = {
        ExpertiseLevel.NOVICE: (ExpertiseLevel.TECHNICIAN, 100),
        ExpertiseLevel.TECHNICIAN: (ExpertiseLevel.SPECIALIST, 500),
        ExpertiseLevel.SPECIALIST: (ExpertiseLevel.EXPERT, 2000),
        ExpertiseLevel.EXPERT: (None, 0),
    }

    current_level = ExpertiseLevel(expertise.expertise_level) if expertise.expertise_level else ExpertiseLevel.NOVICE
    next_level, threshold = level_thresholds.get(current_level, (None, 0))
    points_to_next = max(0, threshold - (expertise.total_expertise_points or 0)) if threshold else 0

    return ExpertiseResponse(
        user_id=expertise.user_id,
        total_expertise_points=expertise.total_expertise_points or 0,
        expertise_level=current_level,
        domain_points=expertise.domain_points or {},
        domains_mastered=expertise.domains_mastered or [],
        total_sessions=expertise.total_sessions or 0,
        successful_resolutions=expertise.successful_resolutions or 0,
        resolution_rate=expertise.resolution_rate if hasattr(expertise, 'resolution_rate') else 0.0,
        avg_resolution_time_seconds=expertise.avg_resolution_time_seconds,
        current_streak_days=expertise.current_streak_days or 0,
        longest_streak_days=expertise.longest_streak_days or 0,
        next_level=next_level,
        points_to_next_level=points_to_next,
    )


def _path_to_response(path: TraversalPath) -> TraversalPathResponse:
    """Convert TraversalPath to response model."""
    steps = [
        TraversalStepResponse(
            node_id=step.node_id,
            title=step.title,
            content=step.content,
            step_type=step.step_type,
            relevance_score=step.relevance_score,
            hop_number=step.hop_number,
            is_completed=False,
        )
        for step in path.steps
    ]

    return TraversalPathResponse(
        path_id=path.path_id,
        steps=steps,
        total_score=path.total_score,
        path_type=path.path_type,
        estimated_steps=len(steps),
        has_solution=any(s.step_type in ("solution", "remedy", "procedure") for s in path.steps),
    )


# =============================================================================
# SESSION ENDPOINTS
# =============================================================================

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    user_id: str = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db_dependency),
) -> SessionResponse:
    """
    Create a new troubleshooting session.

    Analyzes the query to detect error codes, symptoms, and domain.
    Automatically matches a workflow if available.
    """
    try:
        session = await troubleshooting_service.create_session(db, user_id, request)
        return _session_to_response(session)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create session for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_dependency),
) -> SessionDetailResponse:
    """
    Get detailed information about a troubleshooting session.

    Includes task executions, selected path, and completed steps.
    """
    try:
        session = await troubleshooting_service.get_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Build base response
        base = _session_to_response(session, include_tasks=True)

        # Add detailed fields
        return SessionDetailResponse(
            **base.model_dump(),
            paths_presented=[],  # Would need to reconstruct from metadata
            selected_path=None,  # Would need to fetch from graph service
            completed_steps=session.completed_steps or [],
            user_rating=session.user_rating,
            user_feedback=session.user_feedback,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: str = Query(..., description="User ID"),
    state: Optional[SessionState] = Query(None, description="Filter by state"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    include_completed: bool = Query(True, description="Include completed sessions"),
    limit: int = Query(20, ge=1, le=100, description="Max sessions to return"),
    db: AsyncSession = Depends(get_db_dependency),
) -> SessionListResponse:
    """
    List troubleshooting sessions for a user.
    """
    try:
        sessions = await troubleshooting_service.get_user_sessions(
            db, user_id,
            state=state,
            domain=domain,
            limit=limit,
            include_completed=include_completed,
        )

        return SessionListResponse(
            sessions=[_session_to_response(s) for s in sessions],
            total=len(sessions),
            page=1,
            has_more=len(sessions) >= limit,
        )

    except Exception as e:
        logger.error(f"Failed to list sessions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/sessions/active", response_model=Optional[SessionResponse])
async def get_active_session(
    user_id: str = Query(..., description="User ID"),
    db: AsyncSession = Depends(get_db_dependency),
) -> Optional[SessionResponse]:
    """
    Get the user's currently active troubleshooting session.

    Returns null if no active session exists.
    """
    try:
        session = await troubleshooting_service.get_active_session(db, user_id)
        if not session:
            return None
        return _session_to_response(session)

    except Exception as e:
        logger.error(f"Failed to get active session for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active session: {str(e)}")


# =============================================================================
# DIAGNOSTIC PATH ENDPOINTS
# =============================================================================

@router.get("/sessions/{session_id}/paths", response_model=List[TraversalPathResponse])
async def get_diagnostic_paths(
    session_id: str,
    max_paths: int = Query(3, ge=1, le=10, description="Max paths to return"),
    db: AsyncSession = Depends(get_db_dependency),
) -> List[TraversalPathResponse]:
    """
    Get diagnostic paths for a troubleshooting session.

    Queries the PDF_Extraction_Tools knowledge graph to find
    relevant troubleshooting paths based on detected error codes
    or symptoms.
    """
    try:
        paths = await troubleshooting_service.get_diagnostic_paths(
            db, session_id, max_paths=max_paths
        )
        return [_path_to_response(p) for p in paths]

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get diagnostic paths for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get paths: {str(e)}")


@router.post("/sessions/{session_id}/paths/select")
async def select_path(
    session_id: str,
    request: SelectPathRequest,
    db: AsyncSession = Depends(get_db_dependency),
) -> SessionResponse:
    """
    Select a diagnostic path to follow.

    Records the user's choice and prepares the session for
    step-by-step progression through the selected path.
    """
    try:
        # Get the path details from the document graph service
        # For now, we'll create a placeholder - in production, this would
        # fetch the actual path from the graph service cache
        from core.document_graph_service import TroubleshootingStep

        # Placeholder steps - in production, fetch from graph service
        steps = []

        session = await troubleshooting_service.select_path(
            db, session_id, request.path_id, steps
        )
        return _session_to_response(session)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to select path for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to select path: {str(e)}")


@router.post("/sessions/{session_id}/steps/{step_index}/complete")
async def complete_step(
    session_id: str,
    step_index: int,
    request: Optional[CompleteStepRequest] = None,
    db: AsyncSession = Depends(get_db_dependency),
) -> SessionResponse:
    """
    Mark a diagnostic step as completed.

    Records user notes and evidence data if provided.
    """
    try:
        user_notes = request.user_notes if request else None
        evidence_data = request.evidence_data if request else None

        session = await troubleshooting_service.complete_step(
            db, session_id, step_index,
            user_notes=user_notes,
            evidence_data=evidence_data,
        )
        return _session_to_response(session)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to complete step {step_index} for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete step: {str(e)}")


# =============================================================================
# SESSION RESOLUTION ENDPOINTS
# =============================================================================

@router.post("/sessions/{session_id}/resolve")
async def resolve_session(
    session_id: str,
    request: ResolveSessionRequest,
    db: AsyncSession = Depends(get_db_dependency),
) -> SessionResponse:
    """
    Resolve a troubleshooting session.

    resolution_type options:
    - "self_resolved": Issue was successfully resolved
    - "escalated": Issue escalated to human expert
    - "abandoned": User left without resolution

    Awards expertise points for successful resolutions.
    """
    try:
        session = await troubleshooting_service.resolve_session(
            db, session_id,
            resolution_type=request.resolution_type,
            rating=request.rating,
            feedback=request.feedback,
        )
        return _session_to_response(session)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to resolve session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve session: {str(e)}")


# =============================================================================
# EXPERTISE ENDPOINTS
# =============================================================================

@router.get("/users/{user_id}/expertise", response_model=ExpertiseResponse)
async def get_user_expertise(
    user_id: str,
    db: AsyncSession = Depends(get_db_dependency),
) -> ExpertiseResponse:
    """
    Get user's troubleshooting expertise and statistics.
    """
    try:
        expertise = await troubleshooting_service.get_user_expertise(db, user_id)
        return _expertise_to_response(expertise)

    except Exception as e:
        logger.error(f"Failed to get expertise for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get expertise: {str(e)}")


@router.get("/users/{user_id}/achievements", response_model=List[AchievementResponse])
async def get_user_achievements(
    user_id: str,
    db: AsyncSession = Depends(get_db_dependency),
) -> List[AchievementResponse]:
    """
    Get user's earned troubleshooting achievements.
    """
    try:
        user_achievements = await troubleshooting_service.get_user_achievements(db, user_id)

        return [
            AchievementResponse(
                id=str(ua.achievement.id),
                title=ua.achievement.title,
                description=ua.achievement.description,
                icon_url=ua.achievement.icon_url,
                domain=ua.achievement.domain,
                badge_color=ua.achievement.badge_color,
                tier=ua.achievement.tier or "bronze",
                earned=True,
                earned_at=ua.earned_at,
                points_awarded=ua.points_awarded or 0,
            )
            for ua in user_achievements
        ]

    except Exception as e:
        logger.error(f"Failed to get achievements for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get achievements: {str(e)}")


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    limit: int = Query(10, ge=1, le=50, description="Max entries to return"),
    user_id: Optional[str] = Query(None, description="Current user ID for rank"),
    db: AsyncSession = Depends(get_db_dependency),
) -> LeaderboardResponse:
    """
    Get the troubleshooting expertise leaderboard.
    """
    try:
        leaders = await troubleshooting_service.get_leaderboard(db, domain=domain, limit=limit)

        entries = []
        for rank, expertise in enumerate(leaders, start=1):
            entries.append(LeaderboardEntry(
                rank=rank,
                user_id=expertise.user_id,
                expertise_level=ExpertiseLevel(expertise.expertise_level) if expertise.expertise_level else ExpertiseLevel.NOVICE,
                total_points=expertise.total_expertise_points or 0,
                successful_resolutions=expertise.successful_resolutions or 0,
                domain=domain,
                domain_points=(expertise.domain_points or {}).get(domain) if domain else None,
            ))

        # Find current user's rank if provided
        user_rank = None
        if user_id:
            for entry in entries:
                if entry.user_id == user_id:
                    user_rank = entry.rank
                    break

        return LeaderboardResponse(
            domain=domain,
            entries=entries,
            total_participants=len(entries),
            user_rank=user_rank,
        )

    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get leaderboard: {str(e)}")


# =============================================================================
# WORKFLOW ENDPOINTS
# =============================================================================

@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    category: Optional[TroubleshootingCategory] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db_dependency),
) -> WorkflowListResponse:
    """
    List available troubleshooting workflows.
    """
    try:
        workflows = await troubleshooting_service.get_available_workflows(
            db, domain=domain, category=category
        )

        workflow_responses = []
        for w in workflows:
            tasks = [
                WorkflowTaskResponse(
                    id=str(t.id),
                    name=t.name,
                    description=t.description,
                    execution_type=t.execution_type,
                    order_index=t.order_index,
                    is_required=t.is_required,
                    pipeline_hook=t.pipeline_hook,
                    user_prompt=t.user_prompt,
                    verification_criteria=t.verification_criteria or {},
                )
                for t in (w.tasks or [])
            ]

            workflow_responses.append(WorkflowResponse(
                id=str(w.id),
                name=w.name,
                description=w.description,
                category=TroubleshootingCategory(w.category),
                domain=w.domain,
                traversal_mode=w.traversal_mode or "semantic_astar",
                max_hops=w.max_hops or 5,
                expertise_points=w.expertise_points or 10,
                estimated_duration_minutes=w.estimated_duration_minutes,
                is_active=w.is_active,
                task_count=len(tasks),
                tasks=tasks,
                created_at=w.created_at,
                updated_at=w.updated_at,
            ))

        return WorkflowListResponse(
            workflows=workflow_responses,
            total=len(workflow_responses),
            page=1,
            has_more=False,
        )

    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")


# =============================================================================
# DOMAIN & CATEGORY ENDPOINTS
# =============================================================================

@router.get("/domains")
async def list_domains() -> List[str]:
    """
    List all troubleshooting domains.
    """
    return [d.value for d in TroubleshootingDomain]


@router.get("/categories")
async def list_categories() -> List[str]:
    """
    List all troubleshooting categories.
    """
    return [c.value for c in TroubleshootingCategory]


@router.get("/pipeline-hooks")
async def list_pipeline_hooks() -> Dict[str, Any]:
    """
    List available pipeline hooks for task tracking.

    Returns hook configurations including names, execution types,
    and verification criteria.
    """
    return {
        hook_name: {
            "name": config["name"],
            "description": config["description"],
            "execution_type": config["execution_type"].value,
            "timeout_seconds": config["timeout_seconds"],
            "verification_criteria": config["verification_criteria"],
        }
        for hook_name, config in PIPELINE_HOOKS.items()
    }


# =============================================================================
# SSE STREAMING ENDPOINT
# =============================================================================

@router.get("/sessions/{session_id}/stream")
async def stream_session_progress(
    session_id: str,
    db: AsyncSession = Depends(get_db_dependency),
) -> StreamingResponse:
    """
    Stream session progress updates via Server-Sent Events (SSE).

    Sends real-time updates as tasks complete during pipeline execution.

    Event types:
    - task_started: A task has begun execution
    - task_completed: A task has finished (success or failure)
    - path_found: Diagnostic paths have been found
    - session_updated: Session state has changed
    - error: An error occurred
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for session progress."""
        try:
            # Send initial session state
            session = await troubleshooting_service.get_session(db, session_id)
            if not session:
                yield f"event: error\ndata: {json.dumps({'error': 'Session not found'})}\n\n"
                return

            yield f"event: session_updated\ndata: {json.dumps({'session_id': session_id, 'state': session.state})}\n\n"

            # Poll for updates (in production, use Redis pub/sub or similar)
            last_state = session.state
            last_completed = session.completed_tasks or 0
            poll_count = 0
            max_polls = 300  # 5 minutes at 1 second intervals

            while poll_count < max_polls:
                await asyncio.sleep(1)
                poll_count += 1

                # Refresh session state
                session = await troubleshooting_service.get_session(db, session_id)
                if not session:
                    break

                # Check for state changes
                if session.state != last_state:
                    yield f"event: session_updated\ndata: {json.dumps({'session_id': session_id, 'state': session.state, 'previous_state': last_state})}\n\n"
                    last_state = session.state

                # Check for task completions
                current_completed = session.completed_tasks or 0
                if current_completed > last_completed:
                    yield f"event: task_completed\ndata: {json.dumps({'session_id': session_id, 'completed_tasks': current_completed, 'total_tasks': session.total_tasks})}\n\n"
                    last_completed = current_completed

                # Check for terminal states
                if session.state in [SessionState.RESOLVED, SessionState.ESCALATED, SessionState.ABANDONED]:
                    yield f"event: session_completed\ndata: {json.dumps({'session_id': session_id, 'state': session.state, 'expertise_points': session.expertise_points_earned})}\n\n"
                    break

                # Send heartbeat every 30 seconds
                if poll_count % 30 == 0:
                    yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"

        except Exception as e:
            logger.error(f"SSE stream error for session {session_id}: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================

# These endpoints provide compatibility with the old quest API format
# to allow gradual migration of Android clients

@router.get("/compat/quests/available")
async def compat_get_available_quests(
    user_id: str = Query(..., description="User ID"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_dependency),
) -> Dict[str, Any]:
    """
    Backward-compatible endpoint returning workflows as quests.

    Maps troubleshooting workflows to the old quest format for
    Android clients that haven't migrated yet.
    """
    try:
        # Map old categories to new
        category_map = {
            "DAILY": TroubleshootingCategory.ERROR_DIAGNOSIS,
            "WEEKLY": TroubleshootingCategory.SYMPTOM_ANALYSIS,
            "MILESTONE": TroubleshootingCategory.LEARNING,
        }

        ts_category = category_map.get(category) if category else None
        workflows = await troubleshooting_service.get_available_workflows(
            db, category=ts_category
        )

        # Convert to old quest format
        quests = []
        for w in workflows[:limit]:
            quests.append({
                "id": str(w.id),
                "title": w.name,
                "description": w.description,
                "category": category or "DAILY",
                "points": w.expertise_points or 10,
                "task_count": len(w.tasks) if w.tasks else 0,
                "is_active": w.is_active,
                "created_at": w.created_at.isoformat() if w.created_at else None,
                "updated_at": w.updated_at.isoformat() if w.updated_at else None,
            })

        return {
            "quests": quests,
            "total": len(quests),
            "page": 1,
            "hasMore": False,
        }

    except Exception as e:
        logger.error(f"Compat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compat/users/{user_id}/stats")
async def compat_get_user_stats(
    user_id: str,
    db: AsyncSession = Depends(get_db_dependency),
) -> Dict[str, Any]:
    """
    Backward-compatible endpoint returning expertise as quest stats.
    """
    try:
        expertise = await troubleshooting_service.get_user_expertise(db, user_id)

        # Map expertise level to old user level
        level_map = {
            ExpertiseLevel.NOVICE: "newcomer",
            ExpertiseLevel.TECHNICIAN: "seeker",
            ExpertiseLevel.SPECIALIST: "warrior",
            ExpertiseLevel.EXPERT: "mentor",
        }

        level = ExpertiseLevel(expertise.expertise_level) if expertise.expertise_level else ExpertiseLevel.NOVICE

        return {
            "user_id": expertise.user_id,
            "total_points": expertise.total_expertise_points or 0,
            "current_streak_days": expertise.current_streak_days or 0,
            "longest_streak_days": expertise.longest_streak_days or 0,
            "last_activity_date": expertise.last_activity_date.isoformat() if expertise.last_activity_date else None,
            "total_quests_completed": expertise.successful_resolutions or 0,
            "level": level_map.get(level, "newcomer"),
            "weekly_points": expertise.weekly_sessions or 0,
            "monthly_points": expertise.monthly_sessions or 0,
            "next_level_points": 100,  # Placeholder
        }

    except Exception as e:
        logger.error(f"Compat stats endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
