"""
Quest API Endpoints for memOS Server
Gamification system for recovery journey tracking
"""

import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from config.database import get_db_dependency
from core.quest_service_fixed import quest_service_fixed as quest_service
from models.quest import (
    Quest, UserQuest, UserTask,  # Add SQLAlchemy models
    QuestCategory, QuestState, TaskState,
    QuestCreate, QuestUpdate, QuestResponse, UserQuestResponse,
    TaskResponse, UserTaskResponse, QuestStatsResponse, DailyQuestsResponse,
    QuestListResponse, UserProgressResponse  # Android-compatible wrappers
)
from config.logging_config import get_audit_logger

router = APIRouter(prefix="/api/v1/quests", tags=["quests"])
audit_logger = get_audit_logger()
logger = logging.getLogger(__name__)


@router.get("/available", response_model=QuestListResponse)
async def get_available_quests(
    user_id: str = Query(..., description="User ID"),
    category: Optional[QuestCategory] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100, description="Max quests to return"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> QuestListResponse:
    """
    Get quests available for a user based on prerequisites and recovery stage
    """
    try:
        quests = await quest_service.get_available_quests(db, user_id, category, limit)
        
        quest_responses = []
        for quest in quests:
            # Build task responses for this quest
            task_responses = [
                TaskResponse(
                    id=str(task.id),
                    title=task.title,
                    description=task.description,
                    order_index=task.order_index,
                    is_required=task.is_required,
                    verification_data=task.verification_data or {}
                )
                for task in quest.tasks
            ]
            
            quest_responses.append(QuestResponse(
                id=str(quest.id),
                title=quest.title,
                description=quest.description,
                category=quest.category,
                points=quest.points,
                min_recovery_stage=quest.min_recovery_stage,
                max_active_days=quest.max_active_days,
                cooldown_hours=quest.cooldown_hours,
                prerequisites=quest.prerequisites or [],
                verification_type=quest.verification_type,
                task_count=len(quest.tasks),
                is_active=quest.is_active,
                created_at=quest.created_at,
                updated_at=quest.updated_at,
                tasks=task_responses  # Added actual tasks
            ))
        
        return QuestListResponse(
            quests=quest_responses,
            total=len(quest_responses),
            page=1,
            hasMore=False
        )
        
    except Exception as e:
        logger.error(f"Failed to get available quests for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve available quests: {str(e)}"
        )


@router.get("/categories")
async def get_quest_categories() -> List[str]:
    """
    Get all quest categories (Android expects List<String>)
    """
    categories = [
        QuestCategory.DAILY,
        QuestCategory.WEEKLY,
        QuestCategory.MILESTONE,
        QuestCategory.LIFE_SKILLS,
        QuestCategory.COMMUNITY,
        QuestCategory.WELLNESS,
        QuestCategory.SPIRITUAL
    ]
    
    return categories


@router.get("/{quest_id}", response_model=QuestResponse)
async def get_quest_details(
    quest_id: str,
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> QuestResponse:
    """
    Get detailed information about a specific quest
    """
    try:
        from models.quest import Quest
        from sqlalchemy.orm import selectinload
        from sqlalchemy import select
        import uuid

        result = await db.execute(
            select(Quest)
            .where(Quest.id == uuid.UUID(quest_id))
            .options(selectinload(Quest.tasks))
        )
        quest = result.scalar_one_or_none()

        if not quest:
            raise HTTPException(status_code=404, detail="Quest not found")

        return QuestResponse(
            id=str(quest.id),
            title=quest.title,
            description=quest.description,
            category=quest.category,
            points=quest.points,
            min_recovery_stage=quest.min_recovery_stage,
            max_active_days=quest.max_active_days,
            cooldown_hours=quest.cooldown_hours,
            prerequisites=quest.prerequisites or [],
            verification_type=quest.verification_type,
            task_count=len(quest.tasks),
            is_active=quest.is_active,
            created_at=quest.created_at,
            updated_at=quest.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quest {quest_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve quest details: {str(e)}"
        )


@router.post("/{quest_id}/assign", response_model=UserQuestResponse)
async def assign_quest(
    quest_id: str,
    user_id: str = Query(..., description="User ID to assign quest to"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> UserQuestResponse:
    """
    Assign a quest to a user
    """
    try:
        user_quest = await quest_service.assign_quest(db, user_id, quest_id)
        
        # Calculate progress
        completed_tasks = sum(1 for task in user_quest.tasks if task.state == TaskState.COMPLETED)
        total_tasks = len(user_quest.tasks)
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        return UserQuestResponse(
            id=str(user_quest.id),
            quest_id=str(user_quest.quest.id),  # Added quest_id field
            quest=QuestResponse(
                id=str(user_quest.quest.id),
                title=user_quest.quest.title,
                description=user_quest.quest.description,
                category=user_quest.quest.category,
                points=user_quest.quest.points,
                min_recovery_stage=user_quest.quest.min_recovery_stage,
                max_active_days=user_quest.quest.max_active_days,
                cooldown_hours=user_quest.quest.cooldown_hours,
                prerequisites=user_quest.quest.prerequisites or [],
                verification_type=user_quest.quest.verification_type,
                task_count=len(user_quest.quest.tasks),
                is_active=user_quest.quest.is_active,
                created_at=user_quest.quest.created_at,
                updated_at=user_quest.quest.updated_at
            ),
            state=user_quest.state,
            started_at=user_quest.started_at,
            completed_at=user_quest.completed_at,
            verified_at=user_quest.verified_at,
            progress_percentage=progress_percentage,
            points_earned=user_quest.points_earned,
            tasks_completed=completed_tasks,
            total_tasks=total_tasks
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to assign quest {quest_id} to user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign quest: {str(e)}"
        )


@router.get("/users/{user_id}/quests", response_model=UserProgressResponse)
async def get_user_quests(
    user_id: str,
    state: Optional[QuestState] = Query(None, description="Filter by quest state"),
    include_completed: bool = Query(True, description="Include completed quests"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> UserProgressResponse:
    """
    Get all quests for a user (Android expects UserProgressResponse)
    """
    try:
        from models.quest import UserQuest
        from sqlalchemy.orm import selectinload
        from sqlalchemy import select, and_

        # Build query
        conditions = [UserQuest.user_id == user_id]

        if state:
            conditions.append(UserQuest.state == state)
        elif not include_completed:
            conditions.append(UserQuest.state.notin_([
                QuestState.COMPLETED,
                QuestState.VERIFIED,
                QuestState.REWARDED
            ]))

        result = await db.execute(
            select(UserQuest)
            .where(and_(*conditions))
            .options(
                selectinload(UserQuest.quest).selectinload(Quest.tasks),
                selectinload(UserQuest.tasks)
            )
            .order_by(UserQuest.started_at.desc())
        )

        user_quests = result.scalars().all()

        user_quest_responses = []
        for uq in user_quests:
            completed_tasks = sum(1 for task in uq.tasks if task.state == TaskState.COMPLETED)
            total_tasks = len(uq.quest.tasks)
            progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

            # Build user task responses
            user_task_responses = []
            for user_task in uq.tasks:
                # Find the corresponding quest task
                quest_task = next((qt for qt in uq.quest.tasks if qt.id == user_task.task_id), None)
                if quest_task:
                    user_task_responses.append(UserTaskResponse(
                        id=str(user_task.id),
                        task_id=str(user_task.task_id),
                        task=TaskResponse(
                            id=str(quest_task.id),
                            title=quest_task.title,
                            description=quest_task.description,
                            order_index=quest_task.order_index,
                            is_required=quest_task.is_required,
                            verification_data=quest_task.verification_data or {}
                        ),
                        state=user_task.state,
                        completed_at=user_task.completed_at,
                        evidence_data=user_task.evidence_data
                    ))

            # Build quest task responses for the nested quest
            quest_task_responses = [
                TaskResponse(
                    id=str(task.id),
                    title=task.title,
                    description=task.description,
                    order_index=task.order_index,
                    is_required=task.is_required,
                    verification_data=task.verification_data or {}
                )
                for task in uq.quest.tasks
            ]

            user_quest_responses.append(UserQuestResponse(
                id=str(uq.id),
                quest_id=str(uq.quest.id),  # Added quest_id field
                quest=QuestResponse(
                    id=str(uq.quest.id),
                    title=uq.quest.title,
                    description=uq.quest.description,
                    category=uq.quest.category,
                    points=uq.quest.points,
                    min_recovery_stage=uq.quest.min_recovery_stage,
                    max_active_days=uq.quest.max_active_days,
                    cooldown_hours=uq.quest.cooldown_hours,
                    prerequisites=uq.quest.prerequisites or [],
                    verification_type=uq.quest.verification_type,
                    task_count=len(uq.quest.tasks),
                    is_active=uq.quest.is_active,
                    created_at=uq.quest.created_at,
                    updated_at=uq.quest.updated_at,
                    tasks=quest_task_responses  # Added actual quest tasks
                ),
                state=uq.state,
                started_at=uq.started_at,
                completed_at=uq.completed_at,
                verified_at=uq.verified_at,
                progress_percentage=progress_percentage,
                points_earned=uq.points_earned,
                tasks_completed=completed_tasks,
                total_tasks=total_tasks,
                tasks=user_task_responses  # Added actual user tasks
            ))

        # Get user stats
        stats = await quest_service.get_user_stats(db, user_id)
        stats_response = QuestStatsResponse(
            user_id=stats.user_id,
            total_points=stats.total_points,
            current_streak_days=stats.current_streak_days,
            longest_streak_days=stats.longest_streak_days,
            last_activity_date=stats.last_activity_date,
            total_quests_completed=stats.total_quests_completed,
            level=stats.level,
            weekly_points=stats.weekly_points,
            monthly_points=stats.monthly_points,
            next_level_points=0  # TODO: Calculate next level points
        )

        return UserProgressResponse(
            userQuests=user_quest_responses,
            stats=stats_response,
            achievements=[],  # TODO: Add achievements
            newAchievements=[]
        )

    except Exception as e:
        logger.error(f"Failed to get quests for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user quests: {str(e)}"
        )


@router.put("/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    user_id: str = Query(..., description="User ID"),
    evidence: Optional[Dict[str, Any]] = None,
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, Any]:
    """
    Mark a task as complete
    """
    try:
        user_task = await quest_service.complete_task(db, user_id, task_id, evidence)
        
        # Audit log
        audit_logger.info(
            f"Task completed",
            extra={
                "user_id": user_id,
                "quest_id": str(user_task.user_quest.quest_id),
                "event_type": "TASK_COMPLETED",
                "task_id": task_id,
                "action": "task_completed"
            }
        )
        
        return {
            "message": "Task completed successfully",
            "task_id": str(user_task.id),
            "state": user_task.state,
            "completed_at": user_task.completed_at.isoformat() if user_task.completed_at else None
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to complete task {task_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete task: {str(e)}"
        )


# Quest completion is handled automatically when all tasks are completed
# @router.post("/{quest_id}/complete")
async def complete_quest(
    quest_id: str,
    user_id: str = Query(..., description="User ID"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, Any]:
    """
    Submit a quest for completion
    """
    try:
        user_quest = await quest_service.complete_quest(user_id, quest_id)
        
        return {
            "message": "Quest completed successfully",
            "quest_id": str(user_quest.quest.id),
            "state": user_quest.state,
            "points_earned": user_quest.points_earned,
            "completed_at": user_quest.completed_at.isoformat() if user_quest.completed_at else None,
            "needs_verification": user_quest.state == QuestState.COMPLETED
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to complete quest {quest_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete quest: {str(e)}"
        )


@router.get("/users/{user_id}/stats", response_model=QuestStatsResponse)
async def get_user_quest_stats(
    user_id: str,
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> QuestStatsResponse:
    """
    Get user's quest statistics
    """
    try:
        stats = await quest_service.get_user_stats(db, user_id)
        
        # Calculate next level points
        level_thresholds = {
            "newcomer": 1000,
            "seeker": 5000,
            "warrior": 15000,
            "mentor": 30000,
            "elder": 50000,
            "guide": 100000
        }
        
        next_level_points = level_thresholds.get(stats.level, 0)
        
        return QuestStatsResponse(
            user_id=stats.user_id,
            total_points=stats.total_points,
            current_streak_days=stats.current_streak_days,
            longest_streak_days=stats.longest_streak_days,
            last_activity_date=stats.last_activity_date,
            total_quests_completed=stats.total_quests_completed,
            level=stats.level,
            weekly_points=stats.weekly_points,
            monthly_points=stats.monthly_points,
            next_level_points=next_level_points
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user statistics: {str(e)}"
        )


@router.get("/users/{user_id}/daily", response_model=UserProgressResponse)
async def get_daily_quests(
    user_id: str,
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> UserProgressResponse:
    """
    Get user's daily quests and tasks (Android expects UserProgressResponse)
    """
    try:
        # Get active daily quests
        active_daily_response = await get_user_quests(
            user_id=user_id,
            state=QuestState.IN_PROGRESS,
            include_completed=False,
            request=request,
            db=db
        )
        
        # Filter for daily quests only
        daily_user_quests = [
            uq for uq in active_daily_response.userQuests 
            if uq.quest.category == QuestCategory.DAILY
        ]
        
        # Return user progress with daily focus
        return UserProgressResponse(
            userQuests=daily_user_quests,
            stats=active_daily_response.stats,
            achievements=active_daily_response.achievements,
            newAchievements=active_daily_response.newAchievements
        )
        
    except Exception as e:
        logger.error(f"Failed to get daily quests for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve daily quests: {str(e)}"
        )


# Admin endpoints would go here for quest creation/management