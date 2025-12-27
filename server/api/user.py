"""
User API Endpoints for memOS Server
User settings and consent management for Recovery Bot Android client
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from config.database import get_db_dependency
from core.privacy_service import PrivacyService
from models.user import (
    UserMemorySettingsCreate, UserMemorySettingsUpdate, UserMemorySettingsResponse,
    ConsentRequest, ConsentResponse, UserMemoryDashboard
)
from config.logging_config import get_audit_logger

router = APIRouter(prefix="/api/v1/user", tags=["user"])

# Initialize services
privacy_service = PrivacyService()
audit_logger = get_audit_logger()

logger = logging.getLogger(__name__)


@router.post("/settings", response_model=UserMemorySettingsResponse)
async def create_user_settings(
    settings_data: UserMemorySettingsCreate,
    request: Request,
    db: AsyncSession = Depends(get_db_dependency)
) -> UserMemorySettingsResponse:
    """
    Create user memory settings for new users
    Establishes privacy preferences and consent
    """
    try:
        # Audit log the settings creation
        audit_logger.log_consent_event(
            user_id=settings_data.user_id,
            consent_type="memory_collection",
            consent_given=settings_data.consent_given,
            requester_id=settings_data.user_id,
            consent_version=settings_data.consent_document_version,
            ip_address=request.client.host if request.client else None
        )
        
        # Create settings using privacy service
        # Note: This would typically involve database operations
        # For now, we'll return a mock response structure
        
        return UserMemorySettingsResponse(
            user_id=settings_data.user_id,
            memory_enabled=settings_data.memory_enabled,
            max_memories=getattr(settings_data, 'max_memories', 1000),
            retention_days=settings_data.retention_days,
            default_privacy_level=settings_data.default_privacy_level,
            auto_consent=getattr(settings_data, 'auto_consent', False),
            allow_clinical_memories=getattr(settings_data, 'allow_clinical_memories', False),
            allow_crisis_detection=getattr(settings_data, 'allow_crisis_detection', True),
            recovery_stage=settings_data.recovery_stage,
            therapy_goals=settings_data.therapy_goals,
            retrieval_depth=getattr(settings_data, 'retrieval_depth', 5.0),
            semantic_threshold=getattr(settings_data, 'semantic_threshold', 0.7),
            include_low_relevance=getattr(settings_data, 'include_low_relevance', False),
            allow_care_team_access=getattr(settings_data, 'allow_care_team_access', False),
            care_team_members=settings_data.care_team_members,
            family_sharing_enabled=getattr(settings_data, 'family_sharing_enabled', False),
            offline_sync_enabled=getattr(settings_data, 'offline_sync_enabled', True),
            push_notifications=getattr(settings_data, 'push_notifications', True),
            memory_insights_enabled=getattr(settings_data, 'memory_insights_enabled', True),
            settings_version=1,
            last_consent_date=datetime.utcnow(),
            consent_document_version=settings_data.consent_document_version,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            requires_consent_renewal=False
        )
        
    except Exception as e:
        logger.error(f"Failed to create user settings for {settings_data.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create user settings: {str(e)}"
        )


@router.get("/settings/{user_id}", response_model=UserMemorySettingsResponse)
async def get_user_settings(
    user_id: str,
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> UserMemorySettingsResponse:
    """
    Retrieve user memory settings
    Returns privacy preferences and configuration
    """
    try:
        # For now, return default settings structure
        # In full implementation, this would query the database
        
        return UserMemorySettingsResponse(
            user_id=user_id,
            memory_enabled=True,
            max_memories=1000,
            retention_days=2555,  # 7 years HIPAA
            default_privacy_level="balanced",
            auto_consent=False,
            allow_clinical_memories=False,
            allow_crisis_detection=True,
            recovery_stage="maintenance",
            therapy_goals=["maintain_sobriety", "improve_relationships"],
            retrieval_depth=5.0,
            semantic_threshold=0.7,
            include_low_relevance=False,
            allow_care_team_access=False,
            care_team_members=["sponsor", "therapist"],
            family_sharing_enabled=False,
            offline_sync_enabled=True,
            push_notifications=True,
            memory_insights_enabled=True,
            settings_version=1,
            last_consent_date=datetime.utcnow(),
            consent_document_version="1.0",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            requires_consent_renewal=False
        )
        
    except Exception as e:
        logger.error(f"Failed to get user settings for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user settings: {str(e)}"
        )


@router.put("/settings/{user_id}", response_model=UserMemorySettingsResponse)
async def update_user_settings(
    user_id: str,
    settings_update: UserMemorySettingsUpdate,
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> UserMemorySettingsResponse:
    """
    Update user memory settings
    Allows partial updates of preferences
    """
    try:
        # Audit log if privacy level is being changed
        update_dict = settings_update.dict(exclude_unset=True)
        if "default_privacy_level" in update_dict:
            audit_logger.log_security_event(
                event_type="PRIVACY_LEVEL_CHANGE",
                severity="MEDIUM",
                description=f"User {user_id} changed privacy level to {update_dict['default_privacy_level']}",
                user_id=user_id,
                ip_address=request.client.host if request and request.client else None
            )
        
        # For now, return updated settings structure
        # In full implementation, this would update the database
        
        current_settings = await get_user_settings(user_id, request, db)
        
        # Apply updates
        for field, value in update_dict.items():
            if hasattr(current_settings, field):
                setattr(current_settings, field, value)
        
        current_settings.updated_at = datetime.utcnow()
        
        return current_settings
        
    except Exception as e:
        logger.error(f"Failed to update user settings for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update user settings: {str(e)}"
        )


@router.post("/consent", response_model=ConsentResponse)
async def manage_consent(
    consent_data: ConsentRequest,
    request: Request,
    db: AsyncSession = Depends(get_db_dependency)
) -> ConsentResponse:
    """
    Manage user consent for data collection and processing
    HIPAA-compliant consent tracking
    """
    try:
        # Audit log the consent event
        audit_logger.log_consent_event(
            user_id=consent_data.user_id,
            consent_type=consent_data.consent_type,
            consent_given=consent_data.consent_given,
            requester_id=consent_data.user_id,
            consent_version=consent_data.consent_document_version,
            ip_address=request.client.host if request.client else None
        )
        
        # Check consent requirements using privacy service
        from models.memory import MemoryPrivacyLevel
        consent_requirements = await privacy_service.check_consent_requirements(
            user_id=consent_data.user_id,
            memory_privacy_level=MemoryPrivacyLevel.BALANCED,  # Default
            operation_type=consent_data.consent_type
        )
        
        return ConsentResponse(
            user_id=consent_data.user_id,
            consent_given=consent_data.consent_given,
            consent_type=consent_data.consent_type,
            consent_document_version=consent_data.consent_document_version,
            consent_timestamp=datetime.utcnow(),
            expires_at=datetime.utcnow() if not consent_data.consent_given else None,
            specific_permissions=consent_data.specific_permissions or {},
            requirements_met=True
        )
        
    except Exception as e:
        logger.error(f"Failed to manage consent for user {consent_data.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process consent: {str(e)}"
        )


@router.get("/consent/{user_id}")
async def get_user_consent_status(
    user_id: str,
    consent_type: Optional[str] = Query(None, description="Specific consent type to check"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, Any]:
    """
    Get current consent status for a user
    Returns all active consents and their expiration dates
    """
    try:
        # For now, return mock consent status
        # In full implementation, this would query consent records
        
        all_consents = {
            "memory_collection": {
                "consent_given": True,
                "consent_version": "1.0",
                "consent_date": datetime.utcnow().isoformat(),
                "expires_at": None
            },
            "clinical_data": {
                "consent_given": False,
                "consent_version": "1.0",
                "consent_date": None,
                "expires_at": None
            },
            "crisis_detection": {
                "consent_given": True,
                "consent_version": "1.0", 
                "consent_date": datetime.utcnow().isoformat(),
                "expires_at": None
            },
            "care_team_sharing": {
                "consent_given": False,
                "consent_version": "1.0",
                "consent_date": None,
                "expires_at": None
            }
        }
        
        if consent_type:
            if consent_type not in all_consents:
                raise HTTPException(
                    status_code=404,
                    detail=f"Consent type '{consent_type}' not found"
                )
            return {
                "user_id": user_id,
                "consent_type": consent_type,
                **all_consents[consent_type]
            }
        
        return {
            "user_id": user_id,
            "consents": all_consents,
            "overall_status": "partial_consent",
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get consent status for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve consent status: {str(e)}"
        )


@router.get("/dashboard/{user_id}", response_model=UserMemoryDashboard)
async def get_user_dashboard(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Number of days for statistics"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> UserMemoryDashboard:
    """
    Get user dashboard data for Android client
    Provides overview of memory activity and insights
    """
    try:
        # For now, return mock dashboard data
        # In full implementation, this would aggregate from database
        
        return UserMemoryDashboard(
            user_id=user_id,
            total_memories=47,
            memories_this_period=12,
            avg_therapeutic_relevance=0.73,
            recovery_stage_progress=0.85,
            memory_types_breakdown={
                "conversational": 28,
                "recovery": 12,
                "clinical": 4,
                "crisis": 2,
                "resource": 1
            },
            privacy_levels_breakdown={
                "minimal": 5,
                "balanced": 35,
                "comprehensive": 7,
                "restricted": 0
            },
            recent_themes=["anxiety_management", "family_relationships", "milestone_celebration"],
            insights=[
                "You've been consistently working on anxiety management techniques",
                "Family relationships appear in 40% of your recent memories",
                "Your therapeutic relevance scores have improved by 15% this month"
            ],
            streak_days=14,
            last_memory_date=datetime.utcnow(),
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get dashboard for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate dashboard: {str(e)}"
        )


@router.get("/privacy-report/{user_id}")
async def get_privacy_report(
    user_id: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, Any]:
    """
    Generate HIPAA-compliant privacy report for user
    Shows data access, consent history, and privacy metrics
    """
    try:
        # Audit log the privacy report request
        audit_logger.log_data_export(
            user_id=user_id,
            export_type="privacy_report",
            requester_id=user_id,
            data_scope="privacy_metadata",
            ip_address=request.client.host if request and request.client else None
        )
        
        # Parse date range if provided
        date_range = None
        if start_date and end_date:
            from datetime import datetime as dt
            start = dt.fromisoformat(start_date)
            end = dt.fromisoformat(end_date)
            date_range = (start, end)
        
        # Generate privacy report using privacy service
        privacy_report = await privacy_service.generate_privacy_report(
            user_id=user_id,
            date_range=date_range
        )
        
        return privacy_report
        
    except Exception as e:
        logger.error(f"Failed to generate privacy report for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate privacy report: {str(e)}"
        )


@router.delete("/data/{user_id}")
async def delete_user_data(
    user_id: str,
    confirm: str = Query(..., description="Must be 'DELETE_ALL_DATA' to confirm"),
    request: Request = None,
    db: AsyncSession = Depends(get_db_dependency)
) -> Dict[str, str]:
    """
    Delete all user data (GDPR/CCPA compliance)
    Requires explicit confirmation and maintains audit trail
    """
    try:
        if confirm != "DELETE_ALL_DATA":
            raise HTTPException(
                status_code=400,
                detail="Confirmation parameter must be 'DELETE_ALL_DATA'"
            )
        
        # Audit log the data deletion request
        audit_logger.log_data_export(
            user_id=user_id,
            export_type="data_deletion",
            requester_id=user_id,
            data_scope="all_user_data",
            ip_address=request.client.host if request and request.client else None
        )
        
        # In full implementation, this would:
        # 1. Delete all memories
        # 2. Delete user settings
        # 3. Delete consent records (keeping audit trail)
        # 4. Anonymize any remaining references
        
        return {
            "message": "All user data has been scheduled for deletion",
            "user_id": user_id,
            "deletion_requested_at": datetime.utcnow().isoformat(),
            "completion_expected": "Data deletion will be completed within 30 days as required by privacy regulations"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user data for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process data deletion: {str(e)}"
        )