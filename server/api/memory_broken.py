"""
Memory API Endpoints for memOS Server
REST API for Recovery Bot Android client memory operations
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config.database import get_db_dependency
from core.memory_service_fixed import memory_service_fixed as memory_service
from core.privacy_service import PrivacyService
from core.encryption_service import EncryptionService
from models.memory import (
    MemoryCreate, MemoryUpdate, MemoryResponse, MemorySearchRequest, 
    MemorySearchResponse, MemoryType, MemoryPrivacyLevel
)
from config.logging_config import get_audit_logger, get_memory_logger

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])

# Initialize services
privacy_service = PrivacyService()
encryption_service = EncryptionService()
audit_logger = get_audit_logger()
memory_logger = get_memory_logger()

logger = logging.getLogger(__name__)


@router.post("/store", response_model=MemoryResponse)
async def store_memory(
    memory_data: MemoryCreate,
    request: Request,
    db: AsyncSession = Depends(get_db_dependency)
) -> MemoryResponse:
    """
    Store a new memory from Android client
    Includes privacy validation, encryption, and embedding generation
    """
    try:
        # Audit log the request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id": memory_data.user_id,
                "memory_id": "pending",
                "operation": "CREATE",
                "requester_id": memory_data.user_id,
                "requester_role": "user",
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
                "action": "memory_access"
            ,
                "action": "memory_access"
            }
        )
        
        # Privacy validation
        is_content_valid = await privacy_service.validate_memory_content(
            memory_data.content, 
            MemoryPrivacyLevel(memory_data.privacy_level)
        )
        
        if not is_content_valid:
            raise HTTPException(
                status_code=400,
                detail="Content violates privacy policies for the specified privacy level"
            )
        
        # Store memory using memory service
        stored_memory = await memory_service.store_memory(
            session=db,
            user_id=memory_data.user_id,
            content=memory_data.content,
            memory_type=memory_data.memory_type,
            metadata={
                "privacy_level": memory_data.privacy_level,
                "recovery_stage": memory_data.recovery_stage,
                "therapeutic_relevance": memory_data.therapeutic_relevance,
                "consent_given": memory_data.consent_given,
                "source_conversation_id": memory_data.source_conversation_id
            },
            tags=memory_data.tags,
            is_milestone=memory_data.memory_type == "milestone"
        )
        
        # Log successful creation
        memory_logger.info(
            f"Memory created",
            extra={
                "user_id": memory_data.user_id,
                "memory_id": str(stored_memory.id),
                "memory_type": memory_data.memory_type,
                "privacy_level": memory_data.privacy_level,
                "content_length": len(memory_data.content),
                "therapeutic_relevance": memory_data.therapeutic_relevance,
                "action": "memory_created"
            }
        )
        
        return MemoryResponse(
            id=str(stored_memory.id),
            user_id=stored_memory.user_id,
            memory_type=stored_memory.memory_type,
            privacy_level=stored_memory.privacy_level,
            created_at=stored_memory.created_at,
            updated_at=stored_memory.updated_at,
            tags=stored_memory.tags or [],
            recovery_stage=stored_memory.recovery_stage,
            therapeutic_relevance=stored_memory.therapeutic_relevance,
            crisis_level=stored_memory.crisis_level,
            embedding_generated=stored_memory.embedding_vector is not None,
            content_length=len(memory_data.content)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store memory for user {memory_data.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store memory: {str(e)}"
        )


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    "user_id": str = Query(..., description="User ID to search memories for"),
    query: str = Query(..., description="Search query text"),
    memory_types: Optional[List[str]] = Query(None, description="Filter by memory types"),
    privacy_levels: Optional[List[str]] = Query(None, description="Filter by privacy levels"),
    min_relevance: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    include_content: bool = Query(False, description="Whether to decrypt and return content"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> MemorySearchResponse:
    """
    Search memories using semantic similarity
    Returns ranked results based on embedding similarity
    """
    try:
        start_time = datetime.utcnow()
        
        # Audit log the search request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id"=user_id,
            memory_id="search_operation",
            operation="SEARCH",
            requester_id=user_id,
            requester_role="user",
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None,
            details={"query_length": len(query), "include_content": include_content,
                "action": "memory_access"
            }
        )
        
        # Create search request
        search_request = MemorySearchRequest(
            user_id=user_id,
            query=query,
            memory_types=memory_types,
            privacy_levels=privacy_levels,
            min_relevance=min_relevance,
            limit=limit,
            include_content=include_content
        )
        
        # Perform search using memory service
        search_results = await memory_service.search_memories(
            session=db,
            user_id=user_id,
            query=query,
            limit=limit,
            similarity_threshold=min_relevance or 0.7
        )
        
        # Calculate search time
        search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Log search operation
        avg_relevance = sum(r.get("similarity_score", 0) for r in search_results) / len(search_results) if search_results else 0.0
        memory_logger.info(
            f"Memory retrieved",
            extra={
                "user_id": user_id,
                "query": query,
                "results_count": len(search_results),
                "search_time_ms": search_time,
                "avg_relevance": avg_relevance,
                "action": "memory_retrieved"
            }
        )
        
        return MemorySearchResponse(
            query=query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=round(search_time, 2),
            user_id=user_id
        )
        
    except Exception as e:
        logger.error(f"Memory search failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search operation failed: {str(e)}"
        )


@router.get("/list/{user_id}")
async def list_user_memories(
    "user_id": str,
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    privacy_level: Optional[str] = Query(None, description="Filter by privacy level"),
    recovery_stage: Optional[str] = Query(None, description="Filter by recovery stage"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of memories"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    include_content: bool = Query(False, description="Whether to decrypt and return content"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, Any]:
    """
    List memories for a user with optional filtering
    Supports pagination for Android client
    """
    try:
        # Audit log the list request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id"=user_id,
            memory_id="list_operation",
            operation="READ",
            requester_id=user_id,
            requester_role="user",
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None,
            details={"limit": limit, "offset": offset, "include_content": include_content,
                "action": "memory_access"
            }
        )
        
        # Get memories using memory service
        memories = await memory_service.get_user_memories(
            session=db,
            user_id=user_id,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
            include_deleted=False
        )
        
        # Get total count for pagination (simplified)
        total_count = len(memories)
        
        return {
            "memories": [
                MemoryResponse(
                    id=str(memory.id),
                    user_id=memory.user_id,
                    memory_type=memory.memory_type,
                    privacy_level=memory.privacy_level,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at,
                    tags=memory.tags or [],
                    recovery_stage=memory.recovery_stage,
                    therapeutic_relevance=memory.therapeutic_relevance,
                    crisis_level=memory.crisis_level,
                    embedding_generated=memory.embedding_vector is not None,
                    content=encryption_service.decrypt(memory.encrypted_content) if include_content and memory.encrypted_content else None,
                    content_length=len(encryption_service.decrypt(memory.encrypted_content)) if memory.encrypted_content else 0
                ) for memory in memories
            ],
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            },
            "filters": {
                "memory_type": memory_type,
                "privacy_level": privacy_level,
                "recovery_stage": recovery_stage
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list memories for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memories: {str(e)}"
        )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    "memory_id": str,
    "user_id": str = Query(..., description="User ID for authorization"),
    include_content: bool = Query(False, description="Whether to decrypt and return content"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> MemoryResponse:
    """
    Retrieve a specific memory by ID
    Includes authorization check and content decryption
    """
    try:
        # Audit log the access request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id"=user_id,
            memory_id=memory_id,
            operation="READ",
            requester_id=user_id,
            requester_role="user",
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None,
            details={"include_content": include_content,
                "action": "memory_access"
            }
        )
        
        # Get memory using memory service
        memory = await memory_service.get_memory(db, memory_id, user_id)
        
        if not memory:
            raise HTTPException(
                status_code=404,
                detail="Memory not found or access denied"
            )
        
        # Decrypt content if requested
        content = None
        content_length = 0
        if include_content and memory.encrypted_content:
            content = encryption_service.decrypt(memory.encrypted_content)
            content_length = len(content)
        
        return MemoryResponse(
            id=str(memory.id),
            user_id=memory.user_id,
            memory_type=memory.memory_type,
            privacy_level=memory.privacy_level,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            tags=memory.tags or [],
            recovery_stage=memory.recovery_stage,
            therapeutic_relevance=memory.therapeutic_relevance,
            crisis_level=memory.crisis_level,
            embedding_generated=memory.embedding_vector is not None,
            content=content,
            content_length=content_length
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory {memory_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memory: {str(e)}"
        )


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    "memory_id": str,
    memory_update: MemoryUpdate,
    "user_id": str = Query(..., description="User ID for authorization"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> MemoryResponse:
    """
    Update an existing memory
    Supports partial updates of metadata and tags
    """
    try:
        # Audit log the update request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id"=user_id,
            memory_id=memory_id,
            operation="UPDATE",
            requester_id=user_id,
            requester_role="user",
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None,
            details={"fields_updated": list(memory_update.dict(exclude_unset=True).keys()),
                "action": "memory_access"
            }
        )
        
        # Update memory using memory service
        update_data = memory_update.dict(exclude_unset=True)
        updated_memory = await memory_service.update_memory(
            session=db,
            memory_id=memory_id,
            user_id=user_id,
            content=update_data.get("content"),
            metadata=update_data.get("metadata"),
            tags=update_data.get("tags")
        )
        
        if not updated_memory:
            raise HTTPException(
                status_code=404,
                detail="Memory not found or access denied"
            )
        
        return MemoryResponse(
            id=str(updated_memory.id),
            user_id=updated_memory.user_id,
            memory_type=updated_memory.memory_type,
            privacy_level=updated_memory.privacy_level,
            created_at=updated_memory.created_at,
            updated_at=updated_memory.updated_at,
            tags=updated_memory.tags or [],
            recovery_stage=updated_memory.recovery_stage,
            therapeutic_relevance=updated_memory.therapeutic_relevance,
            crisis_level=updated_memory.crisis_level,
            embedding_generated=updated_memory.embedding_vector is not None,
            content_length=len(encryption_service.decrypt(updated_memory.encrypted_content)) if updated_memory.encrypted_content else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update memory {memory_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update memory: {str(e)}"
        )


@router.delete("/{memory_id}")
async def delete_memory(
    "memory_id": str,
    "user_id": str = Query(..., description="User ID for authorization"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, str]:
    """
    Delete a memory (soft delete with audit trail)
    Maintains HIPAA compliance with audit logging
    """
    try:
        # Audit log the delete request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id"=user_id,
            memory_id=memory_id,
            operation="DELETE",
            requester_id=user_id,
            requester_role="user",
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None
        )
        
        # Delete memory using memory service
        success = await memory_service.delete_memory(db, memory_id, user_id, hard_delete=False)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Memory not found or access denied"
            )
        
        return {
            "message": "Memory deleted successfully",
            "memory_id": memory_id,
            "deleted_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete memory: {str(e)}"
        )


@router.get("/stats/{user_id}")
async def get_memory_stats(
    "user_id": str,
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, Any]:
    """
    Get memory statistics for a user
    Provides insights for Android client dashboard
    """
    try:
        # Audit log the stats request
        audit_logger.info(
            f"Memory access",
            extra={
                "user_id"=user_id,
            memory_id="stats_operation",
            operation="READ",
            requester_id=user_id,
            requester_role="user",
            ip_address=request.client.host if request and request.client else None,
            user_agent=request.headers.get("user-agent") if request else None
        )
        
        # Get statistics using memory service
        stats = await memory_service.get_memory_stats(db, user_id)
        
        return {
            "user_id": user_id,
            "total_memories": stats.get("total_memories", 0),
            "memory_types": stats.get("memory_types", {,
                "action": "memory_access"
            }),
            "privacy_levels": stats.get("privacy_levels", {}),
            "recovery_stages": stats.get("recovery_stages", {}),
            "average_therapeutic_relevance": stats.get("avg_therapeutic_relevance", 0.0),
            "last_memory_created": stats.get("last_created", None),
            "embeddings_generated": stats.get("embeddings_count", 0),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory stats for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memory statistics: {str(e)}"
        )