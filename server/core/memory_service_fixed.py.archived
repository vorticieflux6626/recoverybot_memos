"""
Memory Service for Recovery Bot memOS (Fixed version)
Handles memory storage, retrieval, and semantic search
Accepts database session as parameter to avoid async context issues
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from sqlalchemy import select, update, delete, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from pgvector.sqlalchemy import Vector

from models.memory import Memory
from models.user import UserMemorySettings
from config.settings import get_settings
from core.memory_encryption import encrypt_data, decrypt_data
from core.memory_embeddings import get_embedding
from config.logging_config import get_audit_logger

logger = logging.getLogger(__name__)
audit_logger = get_audit_logger()
settings = get_settings()


class MemoryServiceFixed:
    """
    Fixed memory service that accepts database session as parameter
    Provides HIPAA-compliant memory storage with semantic search
    """
    
    def __init__(self):
        """Initialize memory service"""
        self.embedding_model = settings.ollama_embedding_model
        self.mem0_client = None  # Mem0 integration disabled for now
        
    async def store_memory(
        self,
        session: AsyncSession,
        user_id: str,
        content: str,
        memory_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        is_milestone: bool = False
    ) -> Memory:
        """
        Store a new memory for a user
        
        Args:
            session: Database session
            user_id: User identifier
            content: Memory content
            memory_type: Type of memory (general, milestone, medical, etc.)
            metadata: Additional metadata
            tags: Optional tags for categorization
            is_milestone: Whether this is a milestone memory
            
        Returns:
            Created memory object
        """
        try:
            # Get user settings
            settings_result = await session.execute(
                select(UserMemorySettings).where(UserMemorySettings.user_id == user_id)
            )
            user_settings = settings_result.scalar_one_or_none()
            
            if not user_settings:
                # Create default settings
                user_settings = UserMemorySettings(
                    user_id=user_id,
                    default_privacy_level="balanced",
                    retention_days=365 * 7,  # 7 years for HIPAA
                    allow_care_team_access=True,
                    memory_enabled=True
                )
                session.add(user_settings)
                await session.flush()
            
            # Check if memory is enabled
            if not user_settings.memory_enabled:
                logger.warning(f"User {user_id} has not enabled memory storage")
                raise ValueError("Memory storage is not enabled for this user")
            
            # Encrypt content if privacy level is restricted
            stored_content = content
            if user_settings.default_privacy_level == "restricted":
                stored_content = encrypt_data(content)
            
            # Generate embedding
            embedding = None
            if user_settings.memory_enabled and user_settings.memory_insights_enabled:
                try:
                    embedding = await get_embedding(content)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
            
            # Create memory
            memory = Memory(
                id=uuid4(),
                user_id=user_id,
                content_hash="placeholder_hash",  # TODO: implement proper hashing
                encrypted_content=stored_content.encode() if isinstance(stored_content, str) else stored_content,
                content_summary=content[:100] if len(content) > 100 else content,  # First 100 chars as summary
                memory_type=memory_type,
                privacy_level=user_settings.default_privacy_level,
                embedding_vector=embedding,
                recovery_stage=metadata.get("recovery_stage") if metadata else None,
                therapeutic_relevance=metadata.get("therapeutic_relevance", 0.5) if metadata else 0.5,
                crisis_level=metadata.get("crisis_level", 0.0) if metadata else 0.0,
                source_conversation_id=metadata.get("source_conversation_id") if metadata else None,
                tags=tags or [],
                entities=metadata.get("entities", {}) if metadata else {},
                consent_given=metadata.get("consent_given", False) if metadata else False,
                consent_date=datetime.now(timezone.utc) if metadata and metadata.get("consent_given") else None,
                created_at=datetime.now(timezone.utc)
            )
            
            session.add(memory)
            await session.commit()
            await session.refresh(memory)
            
            # Audit log
            audit_logger.info(
                f"Memory created for user {user_id}",
                extra={
                    "user_id": user_id,
                    "memory_id": str(memory.id),
                    "memory_type": memory_type,
                    "is_milestone": is_milestone,
                    "action": "memory_created"
                }
            )
            
            return memory
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def get_memory(
        self,
        session: AsyncSession,
        memory_id: str,
        user_id: str
    ) -> Optional[Memory]:
        """Get a specific memory by ID"""
        try:
            result = await session.execute(
                select(Memory).where(
                    and_(
                        Memory.id == UUID(memory_id),
                        Memory.user_id == user_id,
                        Memory.is_deleted == False
                    )
                )
            )
            memory = result.scalar_one_or_none()
            
            if memory and memory.privacy_level == "restricted":
                # Decrypt content and store in a temporary attribute
                memory._decrypted_content = decrypt_data(memory.encrypted_content)
            elif memory:
                memory._decrypted_content = memory.encrypted_content.decode() if isinstance(memory.encrypted_content, bytes) else str(memory.encrypted_content)
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    async def get_user_memories(
        self,
        session: AsyncSession,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False
    ) -> List[Memory]:
        """Get all memories for a user"""
        try:
            query = select(Memory).where(Memory.user_id == user_id)
            
            if not include_deleted:
                query = query.where(Memory.is_deleted == False)
            
            if memory_type:
                query = query.where(Memory.memory_type == memory_type)
            
            query = query.order_by(desc(Memory.created_at)).limit(limit).offset(offset)
            
            result = await session.execute(query)
            memories = result.scalars().all()
            
            # Decrypt encrypted memories
            for memory in memories:
                if memory.privacy_level == "restricted":
                    memory._decrypted_content = decrypt_data(memory.encrypted_content)
                else:
                    memory._decrypted_content = memory.encrypted_content.decode() if isinstance(memory.encrypted_content, bytes) else str(memory.encrypted_content)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get user memories: {e}")
            return []
    
    async def search_memories(
        self,
        session: AsyncSession,
        user_id: str,
        query: str,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search memories using semantic similarity
        
        Args:
            session: Database session
            user_id: User identifier
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of memories with similarity scores
        """
        try:
            # Get user settings
            settings_result = await session.execute(
                select(UserMemorySettings).where(UserMemorySettings.user_id == user_id)
            )
            user_settings = settings_result.scalar_one_or_none()
            
            if not user_settings or not user_settings.memory_enabled or not user_settings.memory_insights_enabled:
                logger.warning(f"Semantic search not allowed for user {user_id}")
                return []
            
            # Generate embedding for query
            query_embedding = await get_embedding(query)
            
            # Search using pgvector
            results = await session.execute(
                select(
                    Memory,
                    Memory.embedding_vector.cosine_distance(query_embedding).label("distance")
                ).where(
                    and_(
                        Memory.user_id == user_id,
                        Memory.is_deleted == False,
                        Memory.embedding_vector.isnot(None)
                    )
                ).order_by("distance").limit(limit)
            )
            
            memories_with_scores = []
            for memory, distance in results:
                similarity = 1 - distance  # Convert distance to similarity
                
                if similarity >= similarity_threshold:
                    # Decrypt if needed
                    content = decrypt_data(memory.encrypted_content) if memory.privacy_level == "restricted" else memory.encrypted_content.decode() if isinstance(memory.encrypted_content, bytes) else str(memory.encrypted_content)
                    
                    memories_with_scores.append({
                        "memory": {
                            "id": str(memory.id),
                            "content": content,
                            "memory_type": memory.memory_type,
                            "created_at": memory.created_at.isoformat(),
                            "tags": memory.tags,
                            "metadata": memory.metadata
                        },
                        "similarity_score": similarity
                    })
            
            return memories_with_scores
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def update_memory(
        self,
        session: AsyncSession,
        memory_id: str,
        user_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Memory]:
        """Update an existing memory"""
        try:
            memory = await self.get_memory(session, memory_id, user_id)
            
            if not memory:
                return None
            
            # Get user settings for encryption
            settings_result = await session.execute(
                select(UserMemorySettings).where(UserMemorySettings.user_id == user_id)
            )
            user_settings = settings_result.scalar_one_or_none()
            
            # Update fields
            if content is not None:
                if user_settings and user_settings.default_privacy_level == "restricted":
                    memory.content = encrypt_data(content)
                    memory.is_encrypted = True
                else:
                    memory.content = content
                
                # Update embedding
                if user_settings and user_settings.memory_enabled and user_settings.memory_insights_enabled:
                    try:
                        memory.embedding = await get_embedding(content)
                    except Exception as e:
                        logger.warning(f"Failed to update embedding: {e}")
            
            if metadata is not None:
                memory.metadata = metadata
            
            if tags is not None:
                memory.tags = tags
            
            memory.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            await session.refresh(memory)
            
            # Audit log
            audit_logger.info(
                f"Memory updated",
                extra={
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "action": "memory_updated"
                }
            )
            
            return memory
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to update memory: {e}")
            return None
    
    async def delete_memory(
        self,
        session: AsyncSession,
        memory_id: str,
        user_id: str,
        hard_delete: bool = False
    ) -> bool:
        """Delete a memory (soft delete by default)"""
        try:
            memory = await self.get_memory(session, memory_id, user_id)
            
            if not memory:
                return False
            
            if hard_delete:
                await session.delete(memory)
            else:
                memory.is_deleted = True
                memory.deleted_at = datetime.now(timezone.utc)
            
            await session.commit()
            
            # Audit log
            audit_logger.info(
                f"Memory deleted",
                extra={
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "hard_delete": hard_delete,
                    "action": "memory_deleted"
                }
            )
            
            return True
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    async def get_memory_stats(
        self,
        session: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """Get memory statistics for user dashboard"""
        try:
            # Total memories
            total_result = await session.execute(
                select(func.count(Memory.id)).where(
                    and_(Memory.user_id == user_id, Memory.is_deleted == False)
                )
            )
            total_memories = total_result.scalar() or 0
            
            # Memory types breakdown
            type_result = await session.execute(
                select(Memory.memory_type, func.count(Memory.id))
                .where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
                .group_by(Memory.memory_type)
            )
            memory_types = {row[0]: row[1] for row in type_result}
            
            # Milestone count
            milestone_result = await session.execute(
                select(func.count(Memory.id)).where(
                    and_(
                        Memory.user_id == user_id,
                        or_(
                            Memory.memory_type == "recovery",
                            Memory.tags.contains(["milestone"])
                        ),
                        Memory.is_deleted == False
                    )
                )
            )
            milestone_count = milestone_result.scalar() or 0
            
            # Recent memories (last 7 days)
            from datetime import timedelta
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            recent_result = await session.execute(
                select(func.count(Memory.id)).where(
                    and_(
                        Memory.user_id == user_id,
                        Memory.created_at >= week_ago,
                        Memory.is_deleted == False
                    )
                )
            )
            recent_count = recent_result.scalar() or 0
            
            return {
                "total_memories": total_memories,
                "memory_types": memory_types,
                "milestone_count": milestone_count,
                "recent_memories": recent_count,
                "storage_used_mb": round(total_memories * 0.1, 2)  # Rough estimate
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {
                "total_memories": 0,
                "memory_types": {},
                "milestone_count": 0,
                "recent_memories": 0,
                "storage_used_mb": 0
            }
    
    async def get_user_settings(
        self,
        session: AsyncSession,
        user_id: str
    ) -> UserMemorySettings:
        """Get or create user memory settings"""
        try:
            result = await session.execute(
                select(UserMemorySettings).where(UserMemorySettings.user_id == user_id)
            )
            settings = result.scalar_one_or_none()
            
            if not settings:
                # Create default settings
                settings = UserMemorySettings(
                    user_id=user_id,
                    default_privacy_level="balanced",
                    retention_days=365 * 7,  # 7 years for HIPAA
                    allow_care_team_access=True,
                    memory_enabled=True,
                    memory_insights_enabled=True
                )
                session.add(settings)
                await session.commit()
                await session.refresh(settings)
            
            return settings
            
        except Exception as e:
            logger.error(f"Failed to get user settings: {e}")
            raise
    
    async def update_user_settings(
        self,
        session: AsyncSession,
        user_id: str,
        **kwargs
    ) -> UserMemorySettings:
        """Update user memory settings"""
        try:
            settings = await self.get_user_settings(session, user_id)
            
            # Update allowed fields
            allowed_fields = [
                "default_privacy_level", "retention_days", "allow_care_team_access",
                "memory_enabled", "memory_insights_enabled", "offline_sync_enabled",
                "push_notifications", "recovery_stage", "therapy_goals"
            ]
            
            for field, value in kwargs.items():
                if field in allowed_fields and value is not None:
                    setattr(settings, field, value)
            
            settings.updated_at = datetime.now(timezone.utc)
            
            await session.commit()
            await session.refresh(settings)
            
            # Audit log
            audit_logger.info(
                f"User settings updated",
                extra={
                    "user_id": user_id,
                    "updated_fields": list(kwargs.keys()),
                    "action": "settings_updated"
                }
            )
            
            return settings
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to update user settings: {e}")
            raise


# Create singleton instance
memory_service_fixed = MemoryServiceFixed()