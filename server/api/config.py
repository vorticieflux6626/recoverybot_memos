"""
LLM Configuration API

Provides endpoints for viewing and modifying LLM model configurations.
Used by the unified dashboard for pipeline tuning.

Endpoints:
    GET  /api/v1/config/llm-models     - Get current LLM config
    PUT  /api/v1/config/llm-models     - Update LLM config
    POST /api/v1/config/llm-models/reload  - Reload config from file
    GET  /api/v1/config/llm-models/presets - List available presets
    POST /api/v1/config/llm-models/presets/{name} - Apply a preset
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yaml

from agentic.llm_config import get_llm_config, reload_llm_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/config", tags=["config"])


class ModelUpdateRequest(BaseModel):
    """Request to update a model assignment."""
    task: str  # e.g., "pipeline.url_evaluator", "utility.entity_extractor"
    model: str  # e.g., "ministral-3:3b"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class PresetApplyRequest(BaseModel):
    """Request to apply a preset."""
    preset_name: str


@router.get("/llm-models")
async def get_llm_models_config() -> Dict[str, Any]:
    """
    Get the current LLM model configuration.

    Returns the full configuration including:
    - Pipeline stage model assignments
    - Utility model assignments
    - Embedding model assignments
    - Available presets
    - Benchmark reference data
    """
    try:
        config = get_llm_config()
        return {
            "success": True,
            "data": config.to_dict(),
            "meta": {
                "config_path": str(config.config_path),
                "version": config.version,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get LLM config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/llm-models")
async def update_llm_model(request: ModelUpdateRequest) -> Dict[str, Any]:
    """
    Update a single model assignment.

    This updates the in-memory config and optionally persists to the YAML file.

    Args:
        request: ModelUpdateRequest with task, model, and optional parameters
    """
    try:
        config = get_llm_config()

        # Parse task identifier
        if "." in request.task:
            category, task_name = request.task.split(".", 1)
        else:
            category = "pipeline"
            task_name = request.task

        # Get the appropriate category
        if category == "pipeline" and hasattr(config.pipeline, task_name):
            model_config = getattr(config.pipeline, task_name)
        elif category == "utility" and hasattr(config.utility, task_name):
            model_config = getattr(config.utility, task_name)
        elif category == "corpus" and hasattr(config.corpus, task_name):
            model_config = getattr(config.corpus, task_name)
        else:
            raise HTTPException(status_code=404, detail=f"Unknown task: {request.task}")

        # Update the config
        old_model = model_config.model
        model_config.model = request.model
        if request.temperature is not None:
            model_config.temperature = request.temperature
        if request.max_tokens is not None:
            model_config.max_tokens = request.max_tokens

        logger.info(f"Updated {request.task}: {old_model} -> {request.model}")

        return {
            "success": True,
            "data": {
                "task": request.task,
                "old_model": old_model,
                "new_model": request.model,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
            },
            "meta": {
                "note": "Change is in-memory only. Use /llm-models/save to persist."
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update LLM config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm-models/reload")
async def reload_llm_models_config() -> Dict[str, Any]:
    """
    Reload the LLM configuration from the YAML file.

    Discards any in-memory changes and reloads from disk.
    """
    try:
        reload_llm_config()
        config = get_llm_config()

        return {
            "success": True,
            "data": {
                "version": config.version,
                "last_updated": config.last_updated,
            },
            "meta": {
                "message": "Configuration reloaded from file"
            }
        }
    except Exception as e:
        logger.error(f"Failed to reload LLM config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm-models/save")
async def save_llm_models_config() -> Dict[str, Any]:
    """
    Save the current in-memory configuration to the YAML file.
    """
    try:
        config = get_llm_config()
        config_data = config.to_dict()

        # Read existing file to preserve structure and comments
        with open(config.config_path, "r") as f:
            existing = yaml.safe_load(f)

        # Update model assignments in existing structure
        if existing and "pipeline" in existing:
            for task_name, task_config in config_data.get("pipeline", {}).items():
                if task_name in existing["pipeline"]:
                    existing["pipeline"][task_name]["model"] = task_config["model"]
                    existing["pipeline"][task_name]["temperature"] = task_config["temperature"]
                    existing["pipeline"][task_name]["max_tokens"] = task_config["max_tokens"]

        if existing and "utility" in existing:
            for task_name, task_config in config_data.get("utility", {}).items():
                if task_name in existing["utility"]:
                    existing["utility"][task_name]["model"] = task_config["model"]

        # Write back
        with open(config.config_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved LLM config to {config.config_path}")

        return {
            "success": True,
            "data": {
                "config_path": str(config.config_path),
            },
            "meta": {
                "message": "Configuration saved to file"
            }
        }
    except Exception as e:
        logger.error(f"Failed to save LLM config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-models/presets")
async def list_presets() -> Dict[str, Any]:
    """
    List available configuration presets.

    Presets are predefined model assignments for different use cases:
    - speed: Fastest models for low latency
    - quality: Best models for highest accuracy
    - balanced: Production-recommended balance
    - low_vram: Minimal VRAM usage
    """
    try:
        config = get_llm_config()

        return {
            "success": True,
            "data": {
                "presets": list(config.presets.keys()),
                "details": config.presets,
            }
        }
    except Exception as e:
        logger.error(f"Failed to list presets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm-models/presets/{preset_name}")
async def apply_preset(preset_name: str) -> Dict[str, Any]:
    """
    Apply a configuration preset.

    Args:
        preset_name: Name of preset (speed, quality, balanced, low_vram)
    """
    try:
        config = get_llm_config()

        if preset_name not in config.presets:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown preset: {preset_name}. Available: {list(config.presets.keys())}"
            )

        success = config.apply_preset(preset_name)

        if success:
            return {
                "success": True,
                "data": {
                    "preset": preset_name,
                    "applied_changes": config.presets[preset_name],
                },
                "meta": {
                    "message": f"Applied preset: {preset_name}"
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to apply preset")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to apply preset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm-models/raw")
async def get_raw_config() -> Dict[str, Any]:
    """
    Get the raw YAML configuration file content.

    Useful for viewing the full configuration with comments.
    """
    try:
        config = get_llm_config()

        if not config.config_path.exists():
            raise HTTPException(status_code=404, detail="Config file not found")

        with open(config.config_path, "r") as f:
            content = f.read()

        return {
            "success": True,
            "data": {
                "content": content,
                "path": str(config.config_path),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read raw config: {e}")
        raise HTTPException(status_code=500, detail=str(e))
