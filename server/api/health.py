"""
Health Check API Endpoints for memOS Server
Provides health monitoring for Android client
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Dict, Any

from config.database import db_manager
from core.embedding_service import EmbeddingService
from core.encryption_service import EncryptionService

router = APIRouter(prefix="/api/v1/health", tags=["health"])

embedding_service = EmbeddingService()
encryption_service = EncryptionService()


@router.get("/")
async def comprehensive_health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for all memOS services
    Used by Android client to verify server availability
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "services": {},
        "performance": {}
    }
    
    overall_healthy = True
    
    try:
        # Database health check
        db_healthy = await db_manager.health_check()
        health_status["services"]["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "type": "PostgreSQL with pgvector"
        }
        if not db_healthy:
            overall_healthy = False
        
        # Ollama health check
        ollama_healthy = await embedding_service.health_check()
        health_status["services"]["ollama"] = {
            "status": "healthy" if ollama_healthy else "degraded",
            "embedding_model": "mxbai-embed-large"
        }
        
        # Encryption service check
        encryption_info = encryption_service.get_encryption_info()
        health_status["services"]["encryption"] = {
            "status": "healthy",
            "algorithm": encryption_info["symmetric_algorithm"],
            "hipaa_compliant": encryption_info["hipaa_compliant"]
        }
        
        # Performance metrics
        start_time = datetime.now(timezone.utc)
        test_embedding = await embedding_service.generate_embedding("health check test")
        embedding_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        health_status["performance"] = {
            "embedding_generation_ms": round(embedding_time, 2),
            "embedding_dimensions": len(test_embedding) if test_embedding else 0
        }
        
        health_status["status"] = "healthy" if overall_healthy else "degraded"
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        overall_healthy = False
    
    status_code = 200 if overall_healthy else 503
    if status_code == 503:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@router.get("/quick")
async def quick_health_check() -> Dict[str, str]:
    """
    Quick health check for basic availability
    Minimal response for frequent polling by Android client
    """
    try:
        # Just check database connection
        db_healthy = await db_manager.health_check()
        
        if db_healthy:
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(
                status_code=503, 
                detail={
                    "status": "unhealthy",
                    "reason": "database_unavailable",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy", 
                "reason": "service_error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@router.get("/services/{service_name}")
async def service_specific_health(service_name: str) -> Dict[str, Any]:
    """
    Health check for specific service component
    Allows Android client to check individual services
    """
    service_checks = {
        "database": lambda: db_manager.health_check(),
        "ollama": lambda: embedding_service.health_check(),
        "encryption": lambda: True  # Encryption is always available
    }
    
    if service_name not in service_checks:
        raise HTTPException(
            status_code=404,
            detail=f"Service '{service_name}' not found. Available: {list(service_checks.keys())}"
        )
    
    try:
        is_healthy = await service_checks[service_name]()
        
        response = {
            "service": service_name,
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add service-specific details
        if service_name == "ollama" and is_healthy:
            response["embedding_model"] = "mxbai-embed-large"
            response["embedding_dimensions"] = 1024
        elif service_name == "encryption":
            encryption_info = encryption_service.get_encryption_info()
            response.update(encryption_info)
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "service": service_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )