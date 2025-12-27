"""
Quest Service for Recovery Bot memOS (Fixed version)
Handles quest assignment, progress tracking, and completion
Accepts database session as parameter to avoid async context issues
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
from config.settings import get_settings
from core.memory_service_fixed import memory_service_fixed
from config.logging_config import get_audit_logger

logger = logging.getLogger(__name__)
audit_logger = get_audit_logger()
settings = get_settings()


class QuestServiceFixed:
    """
    Service for managing quests and user progress
    Integrates with memory service for milestone tracking
    """
    
    def __init__(self):
        self.memory_service = memory_service_fixed
        
    async def _get_user_completed_quest_ids(self, session: AsyncSession, user_id: str) -> List[UUID]:
        """Get list of quest IDs the user has completed"""
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
        """Get list of quest IDs the user currently has active"""
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
        """Check if all prerequisites for a quest are met"""
        if not quest.prerequisites:
            return True
        
        required_ids = [UUID(id_str) if isinstance(id_str, str) else id_str 
                       for id_str in quest.prerequisites]
        return all(req_id in completed_quest_ids for req_id in required_ids)
    
    async def get_available_quests(
        self,
        session: AsyncSession,
        user_id: str,
        category: Optional[QuestCategory] = None,
        limit: int = 20
    ) -> List[Quest]:
        """Get quests available for a user to start"""
        try:
            # Get user's completed and active quests
            completed_quest_ids = await self._get_user_completed_quest_ids(session, user_id)
            active_quest_ids = await self._get_user_active_quest_ids(session, user_id)
            
            # Get user settings for recovery stage filtering
            user_settings = await session.execute(
                select(UserMemorySettings).where(UserMemorySettings.user_id == user_id)
            )
            user_settings = user_settings.scalar_one_or_none()
            
            # Query available quests
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
    
    async def assign_quest(self, session: AsyncSession, user_id: str, quest_id: str) -> UserQuest:
        """Assign a quest to a user"""
        try:
            # Check if quest exists and is available (eagerly load tasks)
            result = await session.execute(
                select(Quest)
                .where(Quest.id == UUID(quest_id))
                .options(selectinload(Quest.tasks))
            )
            quest = result.scalar_one_or_none()
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
            if existing.scalar_one_or_none():
                raise ValueError("Quest already active for user")
            
            # Create user quest
            user_quest = UserQuest(
                user_id=user_id,
                quest_id=quest.id,
                state=QuestState.ASSIGNED,
                started_at=datetime.utcnow(),
                progress_data={}
            )
            session.add(user_quest)
            await session.flush()  # Flush to get the user_quest.id
            
            # Create user tasks
            for task in quest.tasks:
                user_task = UserTask(
                    user_quest_id=user_quest.id,
                    task_id=task.id,
                    user_id=user_id,
                    state=TaskState.PENDING
                )
                session.add(user_task)
            
            # Update user stats
            stats = await self._get_or_create_user_stats(session, user_id)
            stats.updated_at = datetime.utcnow()
            
            await session.commit()
            
            # Reload with relationships
            await session.refresh(user_quest)
            result = await session.execute(
                select(UserQuest)
                .options(
                    selectinload(UserQuest.quest).selectinload(Quest.tasks),
                    selectinload(UserQuest.tasks)
                )
                .where(UserQuest.id == user_quest.id)
            )
            user_quest = result.scalar_one()
            
            # Audit log
            audit_logger.info(
                f"Quest assigned",
                extra={
                    "user_id": user_id,
                    "quest_id": quest_id,
                    "quest_title": quest.title,
                    "action": "quest_assigned"
                }
            )
            
            return user_quest
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to assign quest {quest_id} to user {user_id}: {e}")
            raise
    
    async def _get_or_create_user_stats(self, session: AsyncSession, user_id: str) -> UserQuestStats:
        """Get or create user quest statistics"""
        stats = await session.execute(
            select(UserQuestStats).where(UserQuestStats.user_id == user_id)
        )
        stats = stats.scalar_one_or_none()
        
        if not stats:
            stats = UserQuestStats(
                user_id=user_id,
                total_points=0,
                current_streak_days=0,
                longest_streak_days=0,
                total_quests_completed=0,
                level=UserLevel.NEWCOMER,
                weekly_points=0,
                monthly_points=0
            )
            session.add(stats)
        
        return stats
    
    async def get_user_quests(
        self,
        session: AsyncSession,
        user_id: str,
        state: Optional[QuestState] = None,
        include_completed: bool = True
    ) -> List[UserQuest]:
        """Get user's quests with optional state filter"""
        try:
            query = select(UserQuest).where(UserQuest.user_id == user_id)
            
            if state:
                query = query.where(UserQuest.state == state)
            elif not include_completed:
                query = query.where(
                    UserQuest.state.notin_([QuestState.COMPLETED, QuestState.VERIFIED, QuestState.REWARDED])
                )
            
            query = query.options(
                selectinload(UserQuest.quest).selectinload(Quest.tasks),
                selectinload(UserQuest.tasks)
            ).order_by(UserQuest.started_at.desc())
            
            result = await session.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Failed to get user quests for {user_id}: {e}")
            return []
    
    async def complete_task(
        self,
        session: AsyncSession,
        user_id: str,
        task_id: str,
        evidence_data: Optional[Dict[str, Any]] = None
    ) -> UserTask:
        """Mark a task as completed"""
        try:
            # Get the user task
            result = await session.execute(
                select(UserTask).where(
                    and_(
                        UserTask.task_id == UUID(task_id),
                        UserTask.user_id == user_id
                    )
                ).options(
                    selectinload(UserTask.user_quest).selectinload(UserQuest.quest),
                    selectinload(UserTask.user_quest).selectinload(UserQuest.tasks),
                    selectinload(UserTask.task)
                )
            )
            user_task = result.scalar_one_or_none()
            
            if not user_task:
                raise ValueError("Task not found for user")
            
            if user_task.state == TaskState.COMPLETED:
                return user_task  # Already completed
            
            # Update task state
            user_task.state = TaskState.COMPLETED
            user_task.completed_at = datetime.utcnow()
            user_task.evidence_data = evidence_data or {}
            
            # Check if all tasks in quest are completed
            user_quest = user_task.user_quest
            all_tasks_completed = all(
                task.state == TaskState.COMPLETED 
                for task in user_quest.tasks
            )
            
            # Update quest state if needed
            if user_quest.state == QuestState.ASSIGNED:
                user_quest.state = QuestState.IN_PROGRESS
            
            if all_tasks_completed:
                await self._complete_quest(session, user_quest)
            
            await session.commit()
            await session.refresh(user_task)
            
            return user_task
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to complete task {task_id} for user {user_id}: {e}")
            raise
    
    async def _complete_quest(self, session: AsyncSession, user_quest: UserQuest):
        """Handle quest completion"""
        try:
            user_quest.state = QuestState.COMPLETED
            user_quest.completed_at = datetime.utcnow()
            user_quest.points_earned = user_quest.quest.points
            
            # Update user stats
            stats = await self._get_or_create_user_stats(session, user_quest.user_id)
            stats.total_points += user_quest.quest.points
            stats.weekly_points += user_quest.quest.points
            stats.monthly_points += user_quest.quest.points
            stats.total_quests_completed += 1
            stats.updated_at = datetime.utcnow()
            
            # Update streak
            await self._update_streak(session, stats)
            
            # Calculate new level
            stats.level = self._calculate_level(stats.total_points)
            
            # Store as memory milestone
            try:
                await self.memory_service.store_memory(
                    session,
                    user_id=user_quest.user_id,
                    content=f"Completed quest: {user_quest.quest.title}",
                    memory_type="milestone",
                    metadata={
                        "quest_id": str(user_quest.quest_id),
                        "quest_title": user_quest.quest.title,
                        "points_earned": user_quest.points_earned,
                        "category": user_quest.quest.category.value,
                        "therapeutic_relevance": 0.8,  # Quest completions are highly relevant
                        "crisis_level": 0.0,
                        "consent_given": True,  # Assume consent for quest tracking
                        "recovery_stage": "maintenance"  # Quest completion implies progress
                    },
                    is_milestone=True
                )
            except Exception as e:
                logger.error(f"Failed to store quest completion memory: {e}")
            
            # Check for achievements
            await self._check_achievements(session, user_quest.user_id, stats)
            
        except Exception as e:
            logger.error(f"Failed to complete quest: {e}")
            raise
    
    async def _update_streak(self, session: AsyncSession, stats: UserQuestStats):
        """Update user's streak information"""
        today = date.today()
        
        if stats.last_activity_date:
            days_diff = (today - stats.last_activity_date).days
            
            if days_diff == 0:
                # Activity today, streak continues
                pass
            elif days_diff == 1:
                # Consecutive day
                stats.current_streak_days += 1
            else:
                # Streak broken
                stats.current_streak_days = 1
        else:
            # First activity
            stats.current_streak_days = 1
        
        stats.last_activity_date = today
        stats.longest_streak_days = max(stats.longest_streak_days, stats.current_streak_days)
    
    def _calculate_level(self, total_points: int) -> UserLevel:
        """Calculate user level based on total points"""
        if total_points < 100:
            return UserLevel.NEWCOMER
        elif total_points < 500:
            return UserLevel.SEEKER
        elif total_points < 1000:
            return UserLevel.PATHFINDER
        elif total_points < 2500:
            return UserLevel.WARRIOR
        elif total_points < 5000:
            return UserLevel.CHAMPION
        elif total_points < 10000:
            return UserLevel.MENTOR
        else:
            return UserLevel.LEGEND
    
    async def _check_achievements(self, session: AsyncSession, user_id: str, stats: UserQuestStats):
        """Check and award achievements based on user progress"""
        try:
            # Get all achievements
            result = await session.execute(
                select(Achievement).where(Achievement.is_active == True)
            )
            achievements = result.scalars().all()
            
            # Get user's existing achievements
            result = await session.execute(
                select(UserAchievement.achievement_id).where(
                    UserAchievement.user_id == user_id
                )
            )
            existing_achievement_ids = [row[0] for row in result.fetchall()]
            
            for achievement in achievements:
                if achievement.id in existing_achievement_ids:
                    continue
                
                # Check if criteria met
                criteria_met = False
                
                if achievement.criteria_type == "points_earned":
                    criteria_met = stats.total_points >= achievement.criteria_value
                elif achievement.criteria_type == "quests_completed":
                    criteria_met = stats.total_quests_completed >= achievement.criteria_value
                elif achievement.criteria_type == "streak_days":
                    criteria_met = stats.current_streak_days >= achievement.criteria_value
                
                if criteria_met:
                    user_achievement = UserAchievement(
                        user_id=user_id,
                        achievement_id=achievement.id,
                        points_awarded=achievement.criteria_value // 10  # Bonus points
                    )
                    session.add(user_achievement)
                    
                    # Add bonus points
                    stats.total_points += user_achievement.points_awarded
                    
                    logger.info(f"Achievement unlocked for user {user_id}: {achievement.title}")
            
        except Exception as e:
            logger.error(f"Failed to check achievements: {e}")
    
    async def get_user_stats(self, session: AsyncSession, user_id: str) -> UserQuestStats:
        """Get user quest statistics"""
        try:
            stats = await self._get_or_create_user_stats(session, user_id)
            await session.commit()
            return stats
        except Exception as e:
            logger.error(f"Failed to get user stats for {user_id}: {e}")
            raise
    
    async def get_daily_quests(
        self,
        session: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """Get user's daily quest information"""
        try:
            # Get active daily quests
            result = await session.execute(
                select(UserQuest).join(Quest).where(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.state.in_([QuestState.ASSIGNED, QuestState.IN_PROGRESS]),
                        Quest.category == QuestCategory.DAILY
                    )
                ).options(
                    selectinload(UserQuest.quest).selectinload(Quest.tasks),
                    selectinload(UserQuest.tasks)
                )
            )
            active_daily = result.scalars().all()
            
            # Get available daily quests
            available_daily = await self.get_available_quests(
                session, user_id, category=QuestCategory.DAILY, limit=5
            )
            
            # Get today's completed daily quests
            today_start = datetime.combine(date.today(), datetime.min.time())
            result = await session.execute(
                select(UserQuest).join(Quest).where(
                    and_(
                        UserQuest.user_id == user_id,
                        UserQuest.completed_at >= today_start,
                        Quest.category == QuestCategory.DAILY
                    )
                )
            )
            completed_today = result.scalars().all()
            
            # Get user stats
            stats = await self._get_or_create_user_stats(session, user_id)
            
            return {
                "active_daily_quests": active_daily,
                "available_daily_quests": available_daily,
                "completed_today": len(completed_today),
                "current_streak": stats.current_streak_days,
                "last_activity": stats.last_activity_date
            }
            
        except Exception as e:
            logger.error(f"Failed to get daily quests for user {user_id}: {e}")
            return {
                "active_daily_quests": [],
                "available_daily_quests": [],
                "completed_today": 0,
                "current_streak": 0,
                "last_activity": None
            }


# Create singleton instance
quest_service_fixed = QuestServiceFixed()