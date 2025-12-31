"""
Model Specs API Router

Endpoints for managing Ollama model specifications.
Provides CRUD operations and refresh functionality for model metadata.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config.database import get_db_dependency
from models.ollama_model import OllamaModelSpec, ModelSpecResponse, ModelSpecsRefreshResponse
from services.model_scraper import OllamaModelScraper

logger = logging.getLogger("api.models")

router = APIRouter(prefix="/api/v1/models", tags=["Models"])

# Global scraper instance
_scraper: Optional[OllamaModelScraper] = None


def get_scraper() -> OllamaModelScraper:
    """Get or create the scraper instance"""
    global _scraper
    if _scraper is None:
        import os
        _scraper = OllamaModelScraper(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    return _scraper


@router.post("/refresh", response_model=ModelSpecsRefreshResponse)
async def refresh_model_specs(
    force: bool = Query(default=False, description="Force refresh all models, ignoring cache"),
    models: Optional[str] = Query(default=None, description="Comma-separated list of specific models to refresh"),
    synthesize: bool = Query(default=True, description="Use LLM to synthesize optimized descriptions for new/missing models"),
    resynthesize_all: bool = Query(default=False, description="Force re-synthesis of ALL descriptions (one-time operation)"),
    synthesis_model: str = Query(default="gemma3:4b", description="Model to use for description synthesis (use non-thinking models)"),
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Refresh model specifications from ollama.com

    Scrapes model pages for each locally installed model and updates
    the database with context windows, capabilities, and other metadata.

    By default, only refreshes models whose cache is older than 24 hours.
    Use force=true to refresh all models.

    Description synthesis:
    - Only runs for NEW models or models with MISSING descriptions
    - Uses a local LLM to create tool-selection-optimized descriptions
    - Does NOT re-synthesize existing descriptions on every refresh
    - Use resynthesize_all=true to force re-synthesis of ALL descriptions (one-time operation)
    """
    logger.info(f"Refreshing model specs (force={force}, synthesize={synthesize}, resynthesize_all={resynthesize_all})")

    try:
        scraper = get_scraper()
        model_list = models.split(",") if models else None

        stats = await scraper.refresh_model_specs(
            session=db,
            force=force,
            models=model_list,
            synthesize_new=synthesize,
            resynthesize_all=resynthesize_all,
            synthesis_model=synthesis_model
        )

        return ModelSpecsRefreshResponse(
            success=True,
            models_updated=stats.get("models_updated", 0),
            models_added=stats.get("models_added", 0),
            descriptions_synthesized=stats.get("descriptions_synthesized", 0),
            errors=stats.get("errors", []),
            duration_seconds=stats.get("duration_seconds", 0)
        )

    except Exception as e:
        logger.error(f"Failed to refresh model specs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh model specs: {str(e)}"
        )


@router.get("/specs")
async def get_model_specs(
    capability: Optional[str] = Query(default=None, description="Filter by capability (e.g., 'reasoning', 'code', 'vision')"),
    max_vram: Optional[float] = Query(default=None, description="Max VRAM in GB"),
    specialization: Optional[str] = Query(default=None, description="Filter by specialization"),
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get all model specifications from the database.

    Optionally filter by capability, max VRAM, or specialization.
    """
    try:
        query = select(OllamaModelSpec)

        if max_vram:
            query = query.where(OllamaModelSpec.vram_min_gb <= max_vram)

        if specialization:
            query = query.where(OllamaModelSpec.specialization == specialization)

        result = await db.execute(query)
        specs = result.scalars().all()

        specs_list = [ModelSpecResponse.model_validate(s) for s in specs]

        if capability:
            specs_list = [
                s for s in specs_list
                if capability in s.capabilities
            ]

        return {
            "success": True,
            "data": {
                "models": [s.model_dump() for s in specs_list],
                "count": len(specs_list)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        }

    except Exception as e:
        logger.error(f"Failed to get model specs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model specs: {str(e)}"
        )


@router.get("/specs/{model_name}", response_model=ModelSpecResponse)
async def get_model_spec(
    model_name: str,
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get specification for a specific model.
    """
    try:
        scraper = get_scraper()
        spec = await scraper.get_model_spec(db, model_name)

        if not spec:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found in specs database. Try /refresh first."
            )

        return ModelSpecResponse(**spec)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model spec: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model spec: {str(e)}"
        )


@router.get("/local")
async def get_local_models():
    """
    Get list of locally installed Ollama models.

    Returns raw model info from the local Ollama instance.
    """
    try:
        scraper = get_scraper()
        models = await scraper.get_local_models()

        return {
            "success": True,
            "data": {
                "models": models,
                "count": len(models)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get local models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get local models: {str(e)}"
        )


@router.get("/capabilities")
async def get_available_capabilities(
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get list of all unique capabilities across all models.
    """
    try:
        result = await db.execute(select(OllamaModelSpec.capabilities))
        all_caps = result.scalars().all()

        # Flatten and deduplicate
        unique_caps = set()
        for caps in all_caps:
            if caps:
                unique_caps.update(caps)

        return {
            "success": True,
            "data": {
                "capabilities": sorted(list(unique_caps))
            }
        }

    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get capabilities: {str(e)}"
        )


@router.get("/specializations")
async def get_available_specializations(
    db: AsyncSession = Depends(get_db_dependency)
):
    """
    Get list of all unique specializations across all models.
    """
    try:
        result = await db.execute(
            select(OllamaModelSpec.specialization).distinct()
        )
        specs = [s for s in result.scalars().all() if s]

        return {
            "success": True,
            "data": {
                "specializations": sorted(specs)
            }
        }

    except Exception as e:
        logger.error(f"Failed to get specializations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get specializations: {str(e)}"
        )


# ============== GPU Monitoring Endpoints ==============

from services.gpu_monitor import get_gpu_monitor


@router.get("/gpu/status")
async def get_gpu_status():
    """
    Get real-time GPU status including VRAM usage.

    Returns information about all NVIDIA GPUs:
    - Total, free, and used VRAM
    - GPU utilization
    - Temperature and power draw
    - Currently loaded Ollama models
    """
    try:
        monitor = get_gpu_monitor()
        summary = await monitor.get_summary()
        return {
            "success": True,
            "data": summary,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GPU status: {str(e)}"
        )


@router.get("/gpu/loaded")
async def get_loaded_models():
    """
    Get currently loaded Ollama models and their VRAM usage.

    Uses 'ollama ps' to get actual loaded model information.
    """
    try:
        monitor = get_gpu_monitor()
        models = await monitor.get_loaded_models(force_refresh=True)
        gpus = await monitor.get_gpu_info(force_refresh=True)

        return {
            "success": True,
            "data": {
                "loaded_models": [m.to_dict() for m in models],
                "total_model_vram_gb": round(sum(m.vram_gb for m in models), 2),
                "gpu_vram_used_gb": round(sum(g.used_memory_gb for g in gpus), 2),
                "gpu_vram_free_gb": round(sum(g.free_memory_gb for g in gpus), 2)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get loaded models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get loaded models: {str(e)}"
        )


@router.get("/gpu/can-load")
async def check_can_load_model(
    vram_gb: float = Query(..., description="Required VRAM in GB"),
    buffer_gb: float = Query(default=2.0, description="Buffer VRAM to leave free")
):
    """
    Check if a model with given VRAM requirement can be loaded.

    Returns which GPU(s) have sufficient free VRAM and a recommendation.
    """
    try:
        monitor = get_gpu_monitor()
        result = await monitor.can_load_model(vram_gb, buffer_gb)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"Failed to check model loading: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check model loading: {str(e)}"
        )


@router.post("/gpu/snapshot")
async def record_gpu_snapshot(
    model_name: Optional[str] = Query(default=None, description="Model being used"),
    context_tokens: int = Query(default=0, description="Number of context tokens"),
    image_count: int = Query(default=0, description="Number of images processed")
):
    """
    Record a VRAM usage snapshot for overhead calculation.

    Call this before/after model operations to track overhead from
    context windows and image processing.
    """
    try:
        monitor = get_gpu_monitor()
        await monitor.record_usage_snapshot(
            model_name=model_name,
            context_tokens=context_tokens,
            image_count=image_count
        )
        return {
            "success": True,
            "message": "Snapshot recorded"
        }
    except Exception as e:
        logger.error(f"Failed to record snapshot: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record snapshot: {str(e)}"
        )


@router.get("/gpu/overhead")
async def get_overhead_stats():
    """
    Get calculated VRAM overhead statistics.

    Returns estimated VRAM usage per 1K tokens and per image,
    based on recorded usage history.
    """
    try:
        monitor = get_gpu_monitor()
        stats = monitor.calculate_overhead_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Failed to calculate overhead: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate overhead: {str(e)}"
        )


# ============== Model Validation Endpoints ==============

from config.settings import (
    validate_configured_models,
    get_resolved_model_config,
    get_available_ollama_models
)


@router.get("/validation/status")
async def get_model_validation_status():
    """
    Get validation status of all configured models.

    Returns:
        - Which models are configured
        - Whether each model is available in Ollama
        - Resolved model (fallback applied if needed)
        - Whether a fallback is being used

    Use this endpoint to verify model configuration is correct
    before running agentic search operations.
    """
    try:
        report = validate_configured_models()

        # Add summary statistics
        models_info = report.get("models", {})
        all_available = all(m.get("available", False) for m in models_info.values())
        using_fallbacks = any(m.get("using_fallback", False) for m in models_info.values())

        return {
            "success": True,
            "data": {
                "all_models_available": all_available,
                "using_fallbacks": using_fallbacks,
                "ollama_models_count": report.get("available_count", 0),
                "models": models_info
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Failed to validate models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate models: {str(e)}"
        )


@router.get("/validation/resolved")
async def get_resolved_models():
    """
    Get the resolved model configuration with fallbacks applied.

    This shows which models will actually be used for each function,
    after applying fallback logic for unavailable models.
    """
    try:
        resolved = get_resolved_model_config()

        return {
            "success": True,
            "data": resolved,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Failed to get resolved models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resolved models: {str(e)}"
        )


@router.post("/validation/refresh")
async def refresh_available_models():
    """
    Force refresh the list of available Ollama models.

    Clears the cache and re-fetches from Ollama API.
    Use this after pulling new models to update the validation cache.
    """
    try:
        models = get_available_ollama_models(force_refresh=True)

        return {
            "success": True,
            "data": {
                "models_count": len(models),
                "message": "Model cache refreshed"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Failed to refresh models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh models: {str(e)}"
        )
