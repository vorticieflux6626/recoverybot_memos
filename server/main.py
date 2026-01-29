"""
memOS Server Main Application
FastAPI server for Recovery Bot memory system with HIPAA compliance
"""

# MUST be first import to suppress third-party deprecation warnings
import suppress_warnings  # noqa: F401

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exception_handlers import http_exception_handler
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import configuration and setup
from config import settings, setup_logging
from config.database import db_manager, init_database
from core.memory_service import memory_service
from core.embedding_service import embedding_service
from core.privacy_service import privacy_service
from core.encryption_service import encryption_service
from core.exceptions import AppException

# Import API routers
from api.memory import router as memory_router
from api.user import router as user_router
from api.health import router as health_router
from api.quest import router as quest_router
from api.auth import router as auth_router

# Agentic search router (isolated, optional)
try:
    from api.search import router as search_router
    SEARCH_ENABLED = True
except ImportError as e:
    search_router = None
    SEARCH_ENABLED = False
    logging.getLogger("memos_server").warning(f"Agentic search disabled: {e}")

# Model specs router (for Ollama model management)
try:
    from api.models import router as models_router
    MODELS_ENABLED = True
except ImportError as e:
    models_router = None
    MODELS_ENABLED = False
    logging.getLogger("memos_server").warning(f"Model specs API disabled: {e}")

# TTS router (for text-to-speech using edge-tts)
try:
    from api.tts import router as tts_router
    TTS_ENABLED = True
except ImportError as e:
    tts_router = None
    TTS_ENABLED = False
    logging.getLogger("memos_server").warning(f"TTS API disabled: {e}")

# System health aggregator router (reads SYSTEM_MANIFEST.yaml)
try:
    from api.system_health import router as system_health_router
    SYSTEM_HEALTH_ENABLED = True
except ImportError as e:
    system_health_router = None
    SYSTEM_HEALTH_ENABLED = False
    logging.getLogger("memos_server").warning(f"System health API disabled: {e}")

# Observability router (for Unified Dashboard)
try:
    from api.observability import router as observability_router
    OBSERVABILITY_ENABLED = True
except ImportError as e:
    observability_router = None
    OBSERVABILITY_ENABLED = False
    logging.getLogger("memos_server").warning(f"Observability API disabled: {e}")

# LLM Config router (for model configuration management)
try:
    from api.config import router as config_router
    CONFIG_ENABLED = True
except ImportError as e:
    config_router = None
    CONFIG_ENABLED = False
    logging.getLogger("memos_server").warning(f"LLM Config API disabled: {e}")

# Troubleshooting Task Tracker router (replaces quest system)
try:
    from api.troubleshooting import router as troubleshooting_router
    TROUBLESHOOTING_ENABLED = True
except ImportError as e:
    troubleshooting_router = None
    TROUBLESHOOTING_ENABLED = False
    logging.getLogger("memos_server").warning(f"Troubleshooting API disabled: {e}")

# Configure logging
setup_logging()
logger = logging.getLogger("memos_server")


# Lifespan event handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events
    """
    # Startup
    logger.info("Starting memOS Server for Recovery Bot")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    try:
        # Initialize database
        await db_manager.initialize()
        logger.info("Database initialized successfully")
        
        # Test Ollama connection
        ollama_available = await embedding_service.health_check()
        if ollama_available:
            logger.info("Ollama service connection verified")
        else:
            logger.warning("Ollama service not available - embeddings will fail")
        
        # Initialize core services
        logger.info("Core services initialized")

        # Warm KV cache with common system prompts for faster first queries
        if ollama_available:
            try:
                from agentic.kv_cache_service import warm_system_prompts
                await warm_system_prompts()
                logger.info("KV cache warmed with system prompts")
            except Exception as e:
                logger.warning(f"KV cache warming failed (non-fatal): {e}")

        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down memOS Server")
        await db_manager.close()
        logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="memOS Server for Recovery Bot",
    description="HIPAA-compliant memory system for therapeutic AI interactions",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "192.168.1.58", "*.sparkonelabs.com", "sparkonelabs.duckdns.org", "testserver"]
)

# Include API routers
app.include_router(auth_router)
app.include_router(memory_router)
app.include_router(user_router)
app.include_router(health_router)
app.include_router(quest_router)

# Include agentic search router if available (isolated module)
if SEARCH_ENABLED and search_router:
    app.include_router(search_router)
    logger.info("Agentic search endpoints enabled at /api/v1/search/*")

# Include model specs router if available
if MODELS_ENABLED and models_router:
    app.include_router(models_router)
    logger.info("Model specs endpoints enabled at /api/v1/models/*")

# Include TTS router if available
if TTS_ENABLED and tts_router:
    app.include_router(tts_router, prefix="/api")
    logger.info("TTS endpoints enabled at /api/tts/*")

# Include system health aggregator if available
if SYSTEM_HEALTH_ENABLED and system_health_router:
    app.include_router(system_health_router)
    logger.info("System health endpoints enabled at /api/v1/system/*")

# Include observability router if available (for Unified Dashboard)
if OBSERVABILITY_ENABLED and observability_router:
    app.include_router(observability_router)
    logger.info("Observability endpoints enabled at /api/v1/observability/*")

# Include LLM config router if available (for model configuration)
if CONFIG_ENABLED and config_router:
    app.include_router(config_router)
    logger.info("LLM config endpoints enabled at /api/v1/config/*")

# Include troubleshooting task tracker router (replaces quest system)
if TROUBLESHOOTING_ENABLED and troubleshooting_router:
    app.include_router(troubleshooting_router)
    logger.info("Troubleshooting endpoints enabled at /api/v1/troubleshooting/*")

# Mount static files for Sherpa-ONNX TTS models
# Models are served from: /api/models/sherpa-onnx/{model_dir}/{file}
MODELS_DIR = Path(__file__).parent.parent / "models" / "sherpa-onnx"
if MODELS_DIR.exists():
    app.mount("/api/models/sherpa-onnx", StaticFiles(directory=str(MODELS_DIR)), name="sherpa_models")
    logger.info(f"Sherpa-ONNX models served from /api/models/sherpa-onnx/* ({MODELS_DIR})")
else:
    logger.warning(f"Sherpa-ONNX models directory not found: {MODELS_DIR}")


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    return f"""
    <html>
        <head>
            <title>memOS Server</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .status {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .healthy {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .error {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1 class="header">memOS Server for Recovery Bot</h1>
            <div class="status">
                <h2>Service Status</h2>
                <p><strong>Version:</strong> 1.0.0</p>
                <p><strong>Environment:</strong> {settings.environment}</p>
                <p><strong>Started:</strong> {datetime.now(timezone.utc).isoformat()}</p>
                <p><strong>Database:</strong> <span class="healthy">Connected</span></p>
                <p><strong>Encryption:</strong> <span class="healthy">AES-256 Enabled</span></p>
                <p><strong>HIPAA Compliance:</strong> <span class="healthy">Active</span></p>
            </div>
            <div class="status">
                <h2>API Documentation</h2>
                <p><a href="/docs">Swagger UI Documentation</a></p>
                <p><a href="/redoc">ReDoc Documentation</a></p>
                <p><a href="/health">Health Check Endpoint</a></p>
            </div>
        </body>
    </html>
    """


# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check database connection
        db_healthy = await db_manager.health_check()
        
        # Check Ollama connection
        ollama_healthy = await embedding_service.health_check()
        
        # Check encryption service
        encryption_info = encryption_service.get_encryption_info()
        
        health_status = {
            "status": "healthy" if db_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "environment": settings.environment,
            "services": {
                "database": {
                    "status": "healthy" if db_healthy else "unhealthy",
                    "type": "PostgreSQL with pgvector"
                },
                "ollama": {
                    "status": "healthy" if ollama_healthy else "degraded",
                    "url": settings.ollama_base_url,
                    "embedding_model": settings.ollama_embedding_model
                },
                "encryption": {
                    "status": "healthy",
                    "algorithm": encryption_info["symmetric_algorithm"],
                    "hipaa_compliant": encryption_info["hipaa_compliant"]
                },
                "memory_service": {
                    "status": "healthy",
                    "features": ["storage", "retrieval", "privacy", "audit"]
                },
                "agentic_search": {
                    "status": "enabled" if SEARCH_ENABLED else "disabled",
                    "features": ["planning", "web_search", "verification", "synthesis"] if SEARCH_ENABLED else []
                }
            },
            "configuration": {
                "debug_mode": settings.debug,
                "cors_enabled": settings.enable_cors,
                "audit_logging": settings.enable_audit_logging,
                "encryption_at_rest": settings.enable_encryption_at_rest
            }
        }
        
        status_code = 200 if db_healthy else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=503
        )


# Test endpoint for development
@app.get("/test")
async def test_endpoint():
    """Test endpoint for verifying setup"""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Test embedding generation
        test_embedding = await embedding_service.generate_embedding("Hello, Recovery Bot!")
        
        # Test encryption
        test_data = "This is sensitive recovery information"
        encrypted = encryption_service.encrypt(test_data)
        decrypted = encryption_service.decrypt(encrypted)
        
        # Test privacy validation
        privacy_valid = await privacy_service.validate_memory_content(
            test_data, "balanced"
        )
        
        return {
            "message": "memOS Server test successful",
            "tests": {
                "embedding_generation": {
                    "status": "success" if test_embedding else "failed",
                    "dimensions": len(test_embedding) if test_embedding else 0
                },
                "encryption": {
                    "status": "success" if decrypted == test_data else "failed",
                    "encrypted_length": len(encrypted)
                },
                "privacy_validation": {
                    "status": "success",
                    "content_valid": privacy_valid
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Test endpoint failed: {e}")
        return JSONResponse(
            content={
                "message": "memOS Server test failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            status_code=500
        )


# =============================================================================
# Exception Handlers - Unified Response Format (Phase 7)
# =============================================================================

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """
    Handle all AppException instances with unified response format.

    Response format:
    {
        "success": false,
        "data": null,
        "meta": {"timestamp": "...", "request_id": "...", "path": "..."},
        "errors": [{"code": "ERR_xxxx", "message": "...", "details": {...}}]
    }
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": getattr(request.state, "request_id", None),
                "path": str(request.url.path)
            },
            "errors": [exc.to_dict()]
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Logs the error and returns unified error response format.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    error_details = {"type": type(exc).__name__}
    if settings.debug:
        error_details["detail"] = str(exc)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": getattr(request.state, "request_id", None),
                "path": str(request.url.path)
            },
            "errors": [{
                "code": "ERR_9001",
                "message": "An unexpected error occurred" if not settings.debug else str(exc),
                "details": error_details
            }]
        }
    )


# Graceful shutdown handler
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )