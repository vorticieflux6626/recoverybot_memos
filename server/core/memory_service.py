"""
Core Memory Service for Recovery Bot memOS
Integrates Mem0 framework with therapeutic context and HIPAA compliance
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from mem0 import Memory as Mem0Memory
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from models.memory import Memory, MemoryType, MemoryPrivacyLevel, MemorySearchRequest
from models.user import UserMemorySettings
from config.database import get_async_db, AsyncSessionLocal
from config.settings import get_settings
from core.embedding_service import EmbeddingService
from core.privacy_service import PrivacyService
from core.encryption_service import EncryptionService

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class MemorySearchResult:
    """Search result with relevance scoring"""
    memory: Memory
    similarity_score: float
    therapeutic_relevance: float
    decay_factor: float
    final_score: float


class MemoryService:
    """
    Core memory service integrating Mem0 with Recovery Bot
    Provides HIPAA-compliant memory operations with therapeutic context
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.privacy_service = PrivacyService()
        self.encryption_service = EncryptionService()
        
        # Initialize Mem0 client
        # Note: Mem0 0.1.114+ requires specific config structure
        # - Use ollama_base_url (not base_url) for Ollama provider
        # - Use Memory.from_config() instead of Memory(config=)
        self.mem0_config = {
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "host": self.settings.postgres_host,
                    "port": self.settings.postgres_port,
                    "user": self.settings.postgres_user,
                    "password": self.settings.postgres_password,
                    "dbname": self.settings.postgres_db,
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self.settings.ollama_embedding_model,
                    "ollama_base_url": self.settings.ollama_base_url,
                }
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self.settings.ollama_model,
                    "ollama_base_url": self.settings.ollama_base_url,
                }
            },
            "version": "v1.1"
        }

        try:
            self.mem0_client = Mem0Memory.from_config(config_dict=self.mem0_config)
            logger.info("Mem0 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 client: {e}")
            self.mem0_client = None
    
    async def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.CONVERSATIONAL,
        privacy_level: MemoryPrivacyLevel = MemoryPrivacyLevel.BALANCED,
        metadata: Optional[Dict[str, Any]] = None,
        source_conversation_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        recovery_stage: Optional[str] = None,
        therapeutic_relevance: Optional[float] = None,
        consent_given: bool = True
    ) -> Memory:
        """
        Store a new memory with therapeutic context and HIPAA compliance
        """
        try:
            # Check user consent and settings
            async with AsyncSessionLocal() as session:
                user_settings = await self._get_user_settings(session, user_id)
                if not user_settings:
                    # Create default settings for new users if consent is given
                    if consent_given:
                        logger.warning(f"Creating default settings for user {user_id}")
                        # Create temporary settings object for processing
                        from models.user import UserMemorySettings
                        user_settings = UserMemorySettings(
                            user_id=user_id,
                            memory_enabled=True,
                            recovery_stage=recovery_stage or "maintenance",
                            retention_days=2555  # 7 years default
                        )
                    else:
                        raise ValueError(f"No consent given for memory storage for user {user_id}")
                elif not user_settings.memory_enabled:
                    raise ValueError(f"Memory storage not enabled for user {user_id}")
                
                # Privacy and content validation
                if not await self.privacy_service.validate_memory_content(content, privacy_level):
                    raise ValueError("Memory content violates privacy policy")
                
                # Anonymize content if required
                anonymized_content = await self.privacy_service.anonymize_content(
                    content, privacy_level
                )
                
                # Generate embedding
                embedding_vector = await self.embedding_service.generate_embedding(
                    anonymized_content
                )
                
                # Encrypt content
                encrypted_content = self.encryption_service.encrypt(content)
                content_hash = self.encryption_service.generate_hash(content)
                
                # Extract therapeutic context
                therapeutic_context = await self._extract_therapeutic_context(
                    content, user_settings.recovery_stage
                )
                
                # Create memory record
                now = datetime.now(timezone.utc)
                memory = Memory(
                    id=uuid.uuid4(),
                    user_id=user_id,
                    content_hash=content_hash,
                    encrypted_content=encrypted_content,
                    content_summary=await self._generate_summary(anonymized_content),
                    memory_type=memory_type,
                    privacy_level=privacy_level,
                    embedding_vector=embedding_vector,
                    embedding_model=self.settings.embedding_model,
                    recovery_stage=recovery_stage or user_settings.recovery_stage,
                    therapeutic_relevance=therapeutic_relevance or therapeutic_context.get('relevance', 0.5),
                    crisis_level=therapeutic_context.get('crisis_level', 0.0),
                    source_conversation_id=source_conversation_id,
                    tags=tags or therapeutic_context.get('tags', []),
                    entities=therapeutic_context.get('entities', {}),
                    consent_given=consent_given,
                    consent_date=now,
                    created_at=now,
                    updated_at=now
                )
                
                # Set expiration based on retention policy
                memory.set_expiration_date(user_settings.retention_days // 365)
                
                # Store in database
                session.add(memory)
                await session.commit()
                await session.refresh(memory)
                
                # Store in Mem0 for enhanced retrieval
                if self.mem0_client:
                    try:
                        await self._store_in_mem0(memory, anonymized_content, metadata)
                    except Exception as e:
                        logger.warning(f"Mem0 storage failed, continuing with database: {e}")
                
                logger.info(f"Memory stored successfully for user {user_id}")
                return memory
                
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            raise
    
    async def retrieve_memories(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        min_similarity: float = 0.7,
        include_content: bool = False
    ) -> List[MemorySearchResult]:
        """
        Retrieve relevant memories using semantic search and therapeutic context
        """
        try:
            async with AsyncSessionLocal() as session:
                user_settings = await self._get_user_settings(session, user_id)
                if not user_settings or not user_settings.memory_enabled:
                    return []
                
                # Generate query embedding
                query_embedding = await self.embedding_service.generate_embedding(query)
                
                # Search database with therapeutic weighting
                memories = await self._search_memories_database(
                    session,
                    user_id=user_id,
                    query_embedding=query_embedding,
                    memory_types=memory_types,
                    limit=limit,
                    min_similarity=min_similarity,
                    recovery_stage=user_settings.recovery_stage
                )
                
                # Enhance with Mem0 if available
                if self.mem0_client:
                    try:
                        mem0_results = await self._search_mem0(
                            user_id, query, limit=limit // 2
                        )
                        memories = await self._merge_search_results(memories, mem0_results)
                    except Exception as e:
                        logger.warning(f"Mem0 search failed, using database only: {e}")
                
                # Decrypt content if requested and authorized
                if include_content:
                    for result in memories:
                        if await self.privacy_service.can_access_content(
                            user_id, result.memory.privacy_level
                        ):
                            try:
                                decrypted = self.encryption_service.decrypt(
                                    result.memory.encrypted_content
                                )
                                result.memory.content_summary = decrypted[:200] + "..."
                            except Exception as e:
                                logger.warning(f"Content decryption failed: {e}")
                
                # Update access tracking
                for result in memories:
                    result.memory.update_access_tracking()
                await session.commit()
                
                return memories
                
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    async def search_memories(
        self,
        search_request: MemorySearchRequest
    ) -> List[MemorySearchResult]:
        """Search memories with comprehensive filtering"""
        return await self.retrieve_memories(
            user_id=search_request.user_id,
            query=search_request.query,
            memory_types=search_request.memory_types,
            limit=search_request.limit,
            min_similarity=search_request.min_relevance,
            include_content=search_request.include_content
        )
    
    async def update_memory(
        self,
        memory_id: uuid.UUID,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Memory]:
        """Update memory with audit trail"""
        try:
            async with AsyncSessionLocal() as session:
                # Get existing memory
                result = await session.execute(
                    select(Memory).where(
                        and_(Memory.id == memory_id, Memory.user_id == user_id)
                    )
                )
                memory = result.scalar_one_or_none()
                
                if not memory:
                    return None
                
                # Update fields
                for field, value in updates.items():
                    if hasattr(memory, field):
                        setattr(memory, field, value)
                
                memory.updated_at = datetime.now(timezone.utc)
                await session.commit()
                await session.refresh(memory)
                
                logger.info(f"Memory {memory_id} updated for user {user_id}")
                return memory
                
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            raise
    
    async def delete_memory(
        self,
        memory_id: uuid.UUID,
        user_id: str,
        deletion_reason: str = "user_request"
    ) -> bool:
        """Soft delete memory with HIPAA compliance"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    update(Memory)
                    .where(and_(Memory.id == memory_id, Memory.user_id == user_id))
                    .values(
                        is_deleted=True,
                        deleted_at=datetime.now(timezone.utc),
                        deletion_reason=deletion_reason
                    )
                )
                
                if result.rowcount == 0:
                    return False
                
                await session.commit()
                
                # Remove from Mem0 if available
                if self.mem0_client:
                    try:
                        await self._delete_from_mem0(memory_id, user_id)
                    except Exception as e:
                        logger.warning(f"Mem0 deletion failed: {e}")
                
                logger.info(f"Memory {memory_id} deleted for user {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Memory deletion failed: {e}")
            return False
    
    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for user dashboard"""
        try:
            async with AsyncSessionLocal() as session:
                # Total memories
                total_result = await session.execute(
                    select(func.count(Memory.id)).where(
                        and_(Memory.user_id == user_id, Memory.is_deleted == False)
                    )
                )
                total_memories = total_result.scalar()
                
                # Memory types breakdown
                type_result = await session.execute(
                    select(Memory.memory_type, func.count(Memory.id))
                    .where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
                    .group_by(Memory.memory_type)
                )
                by_type = dict(type_result.fetchall())
                
                # Privacy levels breakdown
                privacy_result = await session.execute(
                    select(Memory.privacy_level, func.count(Memory.id))
                    .where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
                    .group_by(Memory.privacy_level)
                )
                by_privacy = dict(privacy_result.fetchall())
                
                # Date range
                date_result = await session.execute(
                    select(
                        func.min(Memory.created_at),
                        func.max(Memory.created_at),
                        func.avg(Memory.therapeutic_relevance)
                    ).where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
                )
                oldest, newest, avg_relevance = date_result.first()
                
                # Crisis memories
                crisis_result = await session.execute(
                    select(func.count(Memory.id)).where(
                        and_(
                            Memory.user_id == user_id,
                            Memory.is_deleted == False,
                            Memory.crisis_level > 0.7
                        )
                    )
                )
                crisis_count = crisis_result.scalar()
                
                return {
                    'total_memories': total_memories,
                    'by_type': by_type,
                    'by_privacy_level': by_privacy,
                    'storage_usage_mb': total_memories * 0.1,  # Estimate
                    'oldest_memory': oldest,
                    'newest_memory': newest,
                    'avg_therapeutic_relevance': float(avg_relevance or 0),
                    'crisis_memories_count': crisis_count
                }
                
        except Exception as e:
            logger.error(f"Memory stats retrieval failed: {e}")
            return {}
    
    # Private helper methods
    
    async def _get_user_settings(
        self, session: AsyncSession, user_id: str
    ) -> Optional[UserMemorySettings]:
        """Get user memory settings"""
        result = await session.execute(
            select(UserMemorySettings).where(UserMemorySettings.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def _extract_therapeutic_context(
        self, content: str, recovery_stage: Optional[str]
    ) -> Dict[str, Any]:
        """Extract therapeutic context from memory content"""
        # This would integrate with NLP models to extract:
        # - Emotional indicators
        # - Crisis level assessment
        # - Therapeutic relevance
        # - Recovery-related entities
        
        # Simplified implementation for now
        crisis_keywords = ['crisis', 'emergency', 'relapse', 'urgent', 'help', 'suicide']
        recovery_keywords = ['milestone', 'sober', 'clean', 'recovery', 'progress']
        
        content_lower = content.lower()
        
        crisis_level = 0.0
        for keyword in crisis_keywords:
            if keyword in content_lower:
                crisis_level = max(crisis_level, 0.8)
        
        therapeutic_relevance = 0.5
        for keyword in recovery_keywords:
            if keyword in content_lower:
                therapeutic_relevance = max(therapeutic_relevance, 0.9)
        
        # Enhanced relevance based on recovery stage
        stage_multipliers = {
            'detox': 1.2,
            'early_recovery': 1.1,
            'maintenance': 0.9,
            'relapse_prevention': 1.0
        }
        
        if recovery_stage in stage_multipliers:
            therapeutic_relevance *= stage_multipliers[recovery_stage]
        
        return {
            'relevance': min(therapeutic_relevance, 1.0),
            'crisis_level': crisis_level,
            'tags': self._extract_tags(content),
            'entities': self._extract_entities(content)
        }
    
    async def _generate_summary(self, content: str) -> str:
        """Generate non-sensitive summary for search"""
        # Simple implementation - would use LLM for better summaries
        sentences = content.split('.')[:2]
        summary = '. '.join(sentences).strip()
        return summary[:200] if len(summary) > 200 else summary
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        # Simplified tag extraction
        recovery_tags = {
            'sobriety', 'meetings', 'sponsor', 'steps', 'recovery',
            'therapy', 'counseling', 'support', 'family', 'work',
            'housing', 'healthcare', 'medication', 'goals'
        }
        
        content_lower = content.lower()
        found_tags = []
        
        for tag in recovery_tags:
            if tag in content_lower:
                found_tags.append(tag)
        
        return found_tags[:5]  # Limit to 5 tags
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content"""
        # Simplified entity extraction - would use NER models
        return {
            'locations': [],
            'organizations': [],
            'persons': [],
            'dates': []
        }
    
    async def _search_memories_database(
        self,
        session: AsyncSession,
        user_id: str,
        query_embedding: List[float],
        memory_types: Optional[List[MemoryType]],
        limit: int,
        min_similarity: float,
        recovery_stage: Optional[str]
    ) -> List[MemorySearchResult]:
        """Search memories in database with therapeutic scoring"""
        
        # Build query conditions
        conditions = [
            Memory.user_id == user_id,
            Memory.is_deleted == False,
            Memory.embedding_vector.isnot(None)
        ]
        
        if memory_types:
            conditions.append(Memory.memory_type.in_(memory_types))
        
        # Execute search with similarity calculation
        query = select(Memory).where(and_(*conditions)).limit(limit * 2)
        result = await session.execute(query)
        memories = result.scalars().all()
        
        # Calculate similarity scores and rank
        search_results = []
        for memory in memories:
            if memory.embedding_vector:
                # Calculate cosine similarity (simplified)
                similarity = await self.embedding_service.calculate_similarity(
                    query_embedding, memory.embedding_vector
                )
                
                if similarity >= min_similarity:
                    # Calculate therapeutic decay
                    decay_factor = self._calculate_memory_decay(
                        memory.created_at,
                        memory.therapeutic_relevance,
                        memory.crisis_level > 0.8
                    )
                    
                    # Calculate final score
                    final_score = (
                        similarity * 
                        memory.therapeutic_relevance * 
                        decay_factor
                    )
                    
                    search_results.append(MemorySearchResult(
                        memory=memory,
                        similarity_score=similarity,
                        therapeutic_relevance=memory.therapeutic_relevance,
                        decay_factor=decay_factor,
                        final_score=final_score
                    ))
        
        # Sort by final score and limit results
        search_results.sort(key=lambda x: x.final_score, reverse=True)
        return search_results[:limit]
    
    def _calculate_memory_decay(
        self,
        created_at: datetime,
        importance: float,
        is_milestone: bool
    ) -> float:
        """Calculate memory decay factor for therapeutic relevance"""
        days_old = (datetime.now(timezone.utc) - created_at).days
        
        # Base forgetting curve
        base_retention = 2.71828 ** (-days_old / 7.0)  # exp(-days_old / 7.0)
        
        # Importance multiplier
        importance_factor = 1 + importance * 2
        
        # Milestone protection
        milestone_factor = 10 if is_milestone else 1
        
        # Recovery-specific adjustments
        if days_old < 30:
            recency_boost = 1.5
        elif days_old > 365:
            recency_boost = 0.5
        else:
            recency_boost = 1.0
        
        final_retention = (
            base_retention * importance_factor * 
            milestone_factor * recency_boost
        )
        
        return min(final_retention, 1.0)
    
    async def _store_in_mem0(
        self, memory: Memory, content: str, metadata: Optional[Dict[str, Any]]
    ):
        """Store memory in Mem0 for enhanced retrieval"""
        if not self.mem0_client:
            return
        
        mem0_metadata = {
            'memory_id': str(memory.id),
            'user_id': memory.user_id,
            'memory_type': memory.memory_type,
            'privacy_level': memory.privacy_level,
            'recovery_stage': memory.recovery_stage,
            'therapeutic_relevance': memory.therapeutic_relevance,
            'crisis_level': memory.crisis_level,
            'created_at': memory.created_at.isoformat()
        }
        
        if metadata:
            mem0_metadata.update(metadata)
        
        self.mem0_client.add(content, user_id=memory.user_id, metadata=mem0_metadata)
    
    async def _search_mem0(
        self, user_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Search memories using Mem0"""
        if not self.mem0_client:
            return []
        
        try:
            results = self.mem0_client.search(query, user_id=user_id, limit=limit)
            return results
        except Exception as e:
            logger.warning(f"Mem0 search failed: {e}")
            return []
    
    async def _merge_search_results(
        self,
        db_results: List[MemorySearchResult],
        mem0_results: List[Dict[str, Any]]
    ) -> List[MemorySearchResult]:
        """Merge database and Mem0 search results"""
        # Simple merge - would implement more sophisticated ranking
        return db_results  # For now, prioritize database results
    
    async def _delete_from_mem0(self, memory_id: uuid.UUID, user_id: str):
        """Delete memory from Mem0"""
        if not self.mem0_client:
            return
        
        try:
            # Mem0 deletion by memory_id in metadata
            memories = self.mem0_client.get_all(user_id=user_id)
            for memory in memories:
                if memory.get('metadata', {}).get('memory_id') == str(memory_id):
                    self.mem0_client.delete(memory['id'])
                    break
        except Exception as e:
            logger.warning(f"Mem0 deletion failed: {e}")


    async def get_user_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        privacy_level: Optional[MemoryPrivacyLevel] = None,
        recovery_stage: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_content: bool = False
    ) -> List[Memory]:
        """Get user memories with filtering and pagination"""
        try:
            async with AsyncSessionLocal() as session:
                # Build query conditions
                conditions = [
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                ]
                
                if memory_type:
                    conditions.append(Memory.memory_type == memory_type)
                if privacy_level:
                    conditions.append(Memory.privacy_level == privacy_level)
                if recovery_stage:
                    conditions.append(Memory.recovery_stage == recovery_stage)
                
                # Execute query with pagination
                query = (
                    select(Memory)
                    .where(and_(*conditions))
                    .order_by(Memory.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
                
                result = await session.execute(query)
                memories = result.scalars().all()
                
                return list(memories)
                
        except Exception as e:
            logger.error(f"Failed to get user memories: {e}")
            return []
    
    async def get_user_memory_count(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        privacy_level: Optional[MemoryPrivacyLevel] = None,
        recovery_stage: Optional[str] = None
    ) -> int:
        """Get total count of user memories with filters"""
        try:
            async with AsyncSessionLocal() as session:
                # Build query conditions
                conditions = [
                    Memory.user_id == user_id,
                    Memory.is_deleted == False
                ]
                
                if memory_type:
                    conditions.append(Memory.memory_type == memory_type)
                if privacy_level:
                    conditions.append(Memory.privacy_level == privacy_level)
                if recovery_stage:
                    conditions.append(Memory.recovery_stage == recovery_stage)
                
                # Count query
                result = await session.execute(
                    select(func.count(Memory.id)).where(and_(*conditions))
                )
                
                return result.scalar() or 0
                
        except Exception as e:
            logger.error(f"Failed to count user memories: {e}")
            return 0
    
    async def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Get a specific memory by ID with authorization check"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Memory).where(
                        and_(
                            Memory.id == uuid.UUID(memory_id),
                            Memory.user_id == user_id,
                            Memory.is_deleted == False
                        )
                    )
                )
                
                return result.scalar_one_or_none()
                
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    async def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive memory statistics for user"""
        try:
            async with AsyncSessionLocal() as session:
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
                memory_types = dict(type_result.fetchall())
                
                # Privacy levels breakdown
                privacy_result = await session.execute(
                    select(Memory.privacy_level, func.count(Memory.id))
                    .where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
                    .group_by(Memory.privacy_level)
                )
                privacy_levels = dict(privacy_result.fetchall())
                
                # Recovery stages breakdown
                stage_result = await session.execute(
                    select(Memory.recovery_stage, func.count(Memory.id))
                    .where(and_(
                        Memory.user_id == user_id, 
                        Memory.is_deleted == False,
                        Memory.recovery_stage.isnot(None)
                    ))
                    .group_by(Memory.recovery_stage)
                )
                recovery_stages = dict(stage_result.fetchall())
                
                # Average therapeutic relevance
                avg_result = await session.execute(
                    select(func.avg(Memory.therapeutic_relevance))
                    .where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
                )
                avg_therapeutic_relevance = float(avg_result.scalar() or 0.0)
                
                # Last memory created
                last_result = await session.execute(
                    select(Memory.created_at)
                    .where(and_(Memory.user_id == user_id, Memory.is_deleted == False))
                    .order_by(Memory.created_at.desc())
                    .limit(1)
                )
                last_created = last_result.scalar()
                
                # Count memories with embeddings
                embedding_result = await session.execute(
                    select(func.count(Memory.id))
                    .where(and_(
                        Memory.user_id == user_id,
                        Memory.is_deleted == False,
                        Memory.embedding_vector.isnot(None)
                    ))
                )
                embeddings_count = embedding_result.scalar() or 0
                
                return {
                    "total_memories": total_memories,
                    "memory_types": memory_types,
                    "privacy_levels": privacy_levels,
                    "recovery_stages": recovery_stages,
                    "avg_therapeutic_relevance": avg_therapeutic_relevance,
                    "last_created": last_created.isoformat() if last_created else None,
                    "embeddings_count": embeddings_count
                }
                
        except Exception as e:
            logger.error(f"Failed to get user memory stats: {e}")
            return {
                "total_memories": 0,
                "memory_types": {},
                "privacy_levels": {},
                "recovery_stages": {},
                "avg_therapeutic_relevance": 0.0,
                "last_created": None,
                "embeddings_count": 0
            }


# Global memory service instance
memory_service = MemoryService()