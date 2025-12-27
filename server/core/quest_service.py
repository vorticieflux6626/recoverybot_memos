"""
Quest Service for Recovery Bot memOS
Handles quest assignment, progress tracking, and completion
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.quest import (
    Quest, QuestTask, UserQuest, UserTask, Achievement, UserAchievement, UserQuestStats,
    QuestCategory, QuestState, TaskState, VerificationType, UserLevel,
    QuestCreate, QuestUpdate, QuestResponse, UserQuestResponse, QuestStatsResponse
)
from models.user import UserMemorySettings
from config.database import AsyncSessionLocal
from config.settings import get_settings
from core.memory_service import MemoryService
from config.logging_config import get_audit_logger

logger = logging.getLogger(__name__)
audit_logger = get_audit_logger()
settings = get_settings()


class QuestService:
    """
    Service for managing quests and user progress
    Integrates with memory service for milestone tracking
    """
    
    def __init__(self):
        self.memory_service = MemoryService()
        
    async def create_quest(self, quest_data: QuestCreate) -> Quest:
        """Create a new quest with tasks"""
        try:
            async with AsyncSessionLocal() as session:
                # Create quest
                quest = Quest(
                    title=quest_data.title,
                    description=quest_data.description,
                    category=quest_data.category,
                    points=quest_data.points,
                    min_recovery_stage=quest_data.min_recovery_stage,
                    max_active_days=quest_data.max_active_days,
                    cooldown_hours=quest_data.cooldown_hours,
                    prerequisites=quest_data.prerequisites,
                    verification_type=quest_data.verification_type,
                    quest_metadata=quest_data.metadata
                )
                
                session.add(quest)
                await session.flush()  # Get quest ID
                
                # Create tasks
                for idx, task_data in enumerate(quest_data.tasks):
                    task = QuestTask(
                        quest_id=quest.id,
                        title=task_data['title'],
                        description=task_data.get('description'),
                        order_index=idx,
                        is_required=task_data.get('is_required', True),
                        verification_data=task_data.get('verification_data', {})
                    )
                    session.add(task)
                
                await session.commit()
                await session.refresh(quest, ['tasks'])
                
                logger.info(f"Created quest: {quest.id} - {quest.title}")
                return quest
                
        except Exception as e:
            logger.error(f"Failed to create quest: {e}")
            raise
    
    async def get_available_quests(
        self,
        user_id: str,
        category: Optional[QuestCategory] = None,
        limit: int = 20
    ) -> List[Quest]:
        """Get quests available for a user based on their stage and prerequisites"""
        try:
            async with AsyncSessionLocal() as session:
                # Get user settings and completed quests
                user_settings = await self._get_user_settings(session, user_id)
                completed_quest_ids = await self._get_user_completed_quest_ids(session, user_id)
                active_quest_ids = await self._get_user_active_quest_ids(session, user_id)
                
                # Build query
                query = select(Quest).where(
                    and_(
                        Quest.is_active == True,
                        Quest.id.notin_(completed_quest_ids + active_quest_ids)
                    )
                ).options(selectinload(Quest.tasks))
                
                # Filter by category
                if category:
                    query = query.where(Quest.category == category)
                
                # Filter by recovery stage
                if user_settings and user_settings.recovery_stage:
                    # This would need more sophisticated stage comparison
                    # For now, we'll include all quests
                    pass
                
                query = query.limit(limit)
                result = await session.execute(query)
                quests = result.scalars().all()
                
                # Filter by prerequisites
                available_quests = []
                for quest in quests:
                    if await self._check_prerequisites(quest, completed_quest_ids):
                        available_quests.append(quest)
                
                return available_quests
                
        except Exception as e:
            logger.error(f"Failed to get available quests for user {user_id}: {e}")
            return []
    
    async def assign_quest(self, user_id: str, quest_id: str) -> UserQuest:
        """Assign a quest to a user"""
        try:
            async with AsyncSessionLocal() as session:
                # Check if quest exists and is available
                quest = await session.get(Quest, UUID(quest_id))
                if not quest or not quest.is_active:
                    raise ValueError("Quest not found or inactive")
                
                # Check prerequisites
                completed_quest_ids = await self._get_user_completed_quest_ids(session, user_id)
                if not await self._check_prerequisites(quest, completed_quest_ids):
                    raise ValueError("Prerequisites not met")
                
                # Check if user already has this quest active
                existing = await session.execute(
                    select(UserQuest).where(
                        and_(
                            UserQuest.user_id == user_id,
                            UserQuest.quest_id == quest.id,
                            UserQuest.state.in_([QuestState.ASSIGNED, QuestState.IN_PROGRESS])
                        )
                    )
                )
                if existing.scalar():
                    raise ValueError("Quest already active")
                
                # Create user quest
                user_quest = UserQuest(
                    user_id=user_id,
                    quest_id=quest.id,
                    state=QuestState.ASSIGNED
                )
                session.add(user_quest)
                await session.flush()
                
                # Create user tasks
                for task in quest.tasks:
                    user_task = UserTask(
                        user_quest_id=user_quest.id,
                        task_id=task.id,
                        user_id=user_id,
                        state=TaskState.PENDING
                    )
                    session.add(user_task)
                
                await session.commit()
                await session.refresh(user_quest, ['quest', 'tasks'])
                
                # Audit log
                audit_logger.log_quest_event(
                    user_id=user_id,
                    quest_id=str(quest.id),
                    event_type="QUEST_ASSIGNED",
                    quest_title=quest.title,
                    points_possible=quest.points
                )
                
                logger.info(f"Assigned quest {quest_id} to user {user_id}")
                return user_quest
                
        except Exception as e:
            logger.error(f"Failed to assign quest {quest_id} to user {user_id}: {e}")
            raise
    
    async def update_task_progress(
        self,
        user_id: str,
        task_id: str,
        evidence_data: Optional[Dict[str, Any]] = None
    ) -> UserTask:
        """Update progress on a specific task"""
        try:
            async with AsyncSessionLocal() as session:
                # Get user task
                result = await session.execute(
                    select(UserTask).where(
                        and_(
                            UserTask.id == UUID(task_id),
                            UserTask.user_id == user_id
                        )
                    ).options(
                        selectinload(UserTask.user_quest).selectinload(UserQuest.quest),
                        selectinload(UserTask.task)
                    )
                )
                user_task = result.scalar_one_or_none()
                
                if not user_task:
                    raise ValueError("Task not found")
                
                # Update task state
                user_task.state = TaskState.COMPLETED
                user_task.completed_at = datetime.utcnow()
                if evidence_data:
                    user_task.evidence_data = evidence_data
                
                # Check if quest is complete
                await self._check_quest_completion(session, user_task.user_quest)
                
                await session.commit()
                
                logger.info(f"Updated task {task_id} for user {user_id}")
                return user_task
                
        except Exception as e:
            logger.error(f"Failed to update task {task_id} for user {user_id}: {e}")
            raise
    
    async def complete_quest(self, user_id: str, quest_id: str) -> UserQuest:
        """Mark a quest as completed and award points"""
        try:
            async with AsyncSessionLocal() as session:
                # Get user quest
                result = await session.execute(
                    select(UserQuest).where(
                        and_(
                            UserQuest.id == UUID(quest_id),
                            UserQuest.user_id == user_id
                        )
                    ).options(
                        selectinload(UserQuest.quest),
                        selectinload(UserQuest.tasks)
                    )
                )
                user_quest = result.scalar_one_or_none()
                
                if not user_quest:
                    raise ValueError("Quest not found")
                
                if user_quest.state != QuestState.IN_PROGRESS:
                    raise ValueError("Quest not in progress")
                
                # Verify all required tasks are complete
                required_complete = all(
                    task.state == TaskState.COMPLETED
                    for task in user_quest.tasks
                    if task.task.is_required
                )
                
                if not required_complete:
                    raise ValueError("Not all required tasks completed")
                
                # Update quest state
                user_quest.state = QuestState.COMPLETED
                user_quest.completed_at = datetime.utcnow()
                
                # Award points based on verification type
                if user_quest.quest.verification_type == VerificationType.SELF_REPORT:
                    user_quest.state = QuestState.VERIFIED
                    user_quest.verified_at = datetime.utcnow()
                    user_quest.points_earned = user_quest.quest.points
                    
                    # Update user stats
                    await self._update_user_stats(session, user_id, user_quest.quest.points)
                
                await session.commit()
                
                # Store as memory
                await self._store_quest_memory(user_quest)
                
                # Check achievements
                await self._check_achievements(user_id)
                
                # Audit log
                audit_logger.log_quest_event(
                    user_id=user_id,
                    quest_id=str(user_quest.quest.id),
                    event_type="QUEST_COMPLETED",
                    quest_title=user_quest.quest.title,
                    points_earned=user_quest.points_earned
                )
                
                logger.info(f"Completed quest {quest_id} for user {user_id}")
                return user_quest
                
        except Exception as e:
            logger.error(f"Failed to complete quest {quest_id} for user {user_id}: {e}")
            raise
    
    async def get_user_stats(self, user_id: str) -> UserQuestStats:
        """Get or create user quest statistics"""
        try:
            async with AsyncSessionLocal() as session:
                stats = await session.get(UserQuestStats, user_id)
                
                if not stats:
                    stats = UserQuestStats(user_id=user_id)
                    session.add(stats)
                    await session.commit()
                    await session.refresh(stats)
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get stats for user {user_id}: {e}")
            raise
    
    async def calculate_streak(self, user_id: str) -> int:
        """Calculate current streak days"""
        try:
            async with AsyncSessionLocal() as session:
                stats = await session.get(UserQuestStats, user_id)
                if not stats or not stats.last_activity_date:
                    return 0
                
                # Check if streak is still active (activity within last day)
                days_since_activity = (date.today() - stats.last_activity_date).days
                
                if days_since_activity <= 1:
                    return stats.current_streak_days
                else:
                    # Streak broken - reset
                    stats.current_streak_days = 0
                    await session.commit()
                    return 0
                    
        except Exception as e:
            logger.error(f"Failed to calculate streak for user {user_id}: {e}")
            return 0
    
    # Private helper methods
    
    async def _get_user_settings(self, session: AsyncSession, user_id: str) -> Optional[UserMemorySettings]:
        """Get user memory settings"""
        from models.user import UserMemorySettings
        result = await session.execute(
            select(UserMemorySettings).where(UserMemorySettings.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def _get_user_completed_quest_ids(self, session: AsyncSession, user_id: str) -> List[UUID]:
        """Get IDs of quests user has completed"""
        result = await session.execute(
            select(UserQuest.quest_id).where(
                and_(
                    UserQuest.user_id == user_id,
                    UserQuest.state.in_([QuestState.COMPLETED, QuestState.VERIFIED, QuestState.REWARDED])
                )
            )
        )
        return [row[0] for row in result.fetchall()]
    
    async def _get_user_active_quest_ids(self, session: AsyncSession, user_id: str) -> List[UUID]:
        """Get IDs of quests user has active"""
        result = await session.execute(
            select(UserQuest.quest_id).where(
                and_(
                    UserQuest.user_id == user_id,
                    UserQuest.state.in_([QuestState.ASSIGNED, QuestState.IN_PROGRESS])
                )
            )
        )
        return [row[0] for row in result.fetchall()]
    
    async def _check_prerequisites(self, quest: Quest, completed_quest_ids: List[UUID]) -> bool:
        """Check if quest prerequisites are met"""
        if not quest.prerequisites:
            return True
        
        prerequisite_ids = [UUID(pid) for pid in quest.prerequisites]
        return all(pid in completed_quest_ids for pid in prerequisite_ids)
    
    async def _check_quest_completion(self, session: AsyncSession, user_quest: UserQuest):
        """Check if all tasks are complete and update quest state"""
        # Get all tasks for this quest
        result = await session.execute(
            select(UserTask).where(UserTask.user_quest_id == user_quest.id)
        )
        tasks = result.scalars().all()
        
        # Check if all required tasks are complete
        all_complete = all(
            task.state == TaskState.COMPLETED
            for task in tasks
            if task.task.is_required
        )
        
        if all_complete and user_quest.state == QuestState.ASSIGNED:
            user_quest.state = QuestState.IN_PROGRESS
    
    async def _update_user_stats(self, session: AsyncSession, user_id: str, points: int):
        """Update user statistics with new points"""
        stats = await session.get(UserQuestStats, user_id)
        if not stats:
            stats = UserQuestStats(user_id=user_id)
            session.add(stats)
        
        # Update points
        stats.total_points += points
        stats.weekly_points += points
        stats.monthly_points += points
        stats.total_quests_completed += 1
        
        # Update streak
        today = date.today()
        if stats.last_activity_date:
            days_since = (today - stats.last_activity_date).days
            if days_since == 1:
                stats.current_streak_days += 1
            elif days_since > 1:
                stats.current_streak_days = 1
        else:
            stats.current_streak_days = 1
            
        stats.last_activity_date = today
        stats.longest_streak_days = max(stats.longest_streak_days, stats.current_streak_days)
        
        # Update level
        stats.level = self._calculate_user_level(stats.total_points)
        
        await session.commit()
    
    def _calculate_user_level(self, total_points: int) -> str:
        """Calculate user level based on points"""
        if total_points < 1000:
            return UserLevel.NEWCOMER
        elif total_points < 5000:
            return UserLevel.SEEKER
        elif total_points < 15000:
            return UserLevel.WARRIOR
        elif total_points < 30000:
            return UserLevel.MENTOR
        elif total_points < 50000:
            return UserLevel.ELDER
        else:
            return UserLevel.GUIDE
    
    async def _store_quest_memory(self, user_quest: UserQuest):
        """Store quest completion as a memory"""
        try:
            content = f"Completed quest: {user_quest.quest.title}. {user_quest.quest.description or ''}"
            
            await self.memory_service.store_memory(
                user_id=user_quest.user_id,
                content=content,
                memory_type="recovery",
                privacy_level="balanced",
                tags=["quest", "achievement", user_quest.quest.category],
                recovery_stage="milestone",
                therapeutic_relevance=0.9,
                consent_given=True
            )
        except Exception as e:
            logger.warning(f"Failed to store quest memory: {e}")
    
    async def _check_achievements(self, user_id: str):
        """Check if user has earned any new achievements"""
        # This would be implemented with achievement checking logic
        # For now, just log
        logger.info(f"Checking achievements for user {user_id}")


# Global quest service instance
quest_service = QuestService()