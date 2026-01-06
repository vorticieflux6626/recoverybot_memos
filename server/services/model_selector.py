#!/usr/bin/env python3
"""
Model Selector - Dynamic model selection based on capabilities and VRAM availability.

This module provides intelligent model selection that:
1. Queries available Ollama models
2. Categorizes them by capability (vision, text, embedding, etc.)
3. Selects the most powerful model that fits in available VRAM
4. Provides fallback options for resource-constrained scenarios
"""

import asyncio
import logging
import subprocess
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Model capability types"""
    VISION = "vision"           # Can process images
    TEXT = "text"               # Text generation/chat
    EMBEDDING = "embedding"     # Vector embeddings
    CODE = "code"               # Code-specialized
    REASONING = "reasoning"     # Chain-of-thought/reasoning
    FUNCTION = "function"       # Function calling


@dataclass
class ModelInfo:
    """Information about an available model"""
    name: str
    size_gb: float
    capabilities: List[ModelCapability] = field(default_factory=list)
    quality_tier: int = 1  # 1-5, higher is better
    context_length: int = 4096
    quantization: Optional[str] = None  # e.g., "q8_0", "fp16"

    @property
    def is_vision(self) -> bool:
        return ModelCapability.VISION in self.capabilities

    @property
    def is_text(self) -> bool:
        return ModelCapability.TEXT in self.capabilities


# ============================================================================
# Vision Model Test Results (2026-01-05)
# ============================================================================
# Models were tested with real screenshots for JSON extraction quality.
# Quality tiers updated based on actual parse rate and content richness.
#
# RECOMMENDED (tested, working):
#   - qwen2.5vl:7b      - 100% parse, 6.7s, best quality
#   - granite3.2-vision - 100% parse, 2.9s, fastest
#   - qwen3-vl:*        - 67% parse, 11.5s, huge context (256K)
#   - llama3.2-vision   - 33% parse, 6.9s, large context (128K)
#
# NOT RECOMMENDED (broken on this system):
#   - llava:*           - Model runner crash (CUDA/Ollama compatibility)
#   - llava-llama3:*    - Model runner crash
#   - minicpm-v:*       - Model runner crash
#   - deepseek-ocr:*    - Produces garbage output
# ============================================================================

# Priority order for vision model selection (tested models first)
VISION_MODEL_PRIORITY = [
    # Tier 1: Best quality + speed balance (tested, 100% parse rate)
    "qwen2.5vl:7b",
    "qwen2.5vl:7b-q8_0",
    "qwen2.5vl:7b-fp16",
    "granite3.2-vision:2b",
    "granite3.2-vision:2b-q8_0",
    "granite3.2-vision:2b-fp16",
    # Tier 2: Good quality, huge context (tested, 67% parse rate)
    "qwen3-vl:8b-instruct-bf16",
    "qwen3-vl:8b",
    "qwen3-vl:4b-instruct-bf16",
    "qwen3-vl:4b",
    "qwen3-vl:2b-instruct-bf16",
    "qwen3-vl:2b",
    # Tier 3: Usable but inconsistent (tested, 33% parse rate)
    "llama3.2-vision:11b-instruct-q8_0",
    "llama3.2-vision:11b",
    # Tier 4: Large models (untested, may work)
    "qwen2.5vl:32b",
    "qwen3-vl:32b",
    "qwen3-vl:8b-thinking-bf16",
    "qwen3-vl:4b-thinking-bf16",
    "qwen3-vl:2b-thinking-bf16",
]

# Models known to be broken (excluded from selection)
BROKEN_VISION_MODELS = {
    "llava:latest",
    "llava:7b",
    "llava:13b",
    "llava:7b-v1.6-mistral-fp16",
    "llava:7b-v1.6-mistral-q8_0",
    "llava-llama3:8b",
    "minicpm-v:8b",
    "minicpm-v:8b-2.6-fp16",
    "minicpm-v:8b-2.6-q8_0",
    "deepseek-ocr:3b",
}

# Known model patterns and their capabilities
# Quality tiers updated based on test results (2026-01-05)
MODEL_PATTERNS = {
    # =========================================================================
    # VISION MODELS - Quality based on actual JSON extraction tests
    # =========================================================================

    # Qwen2.5-VL - BEST TESTED (100% parse rate, 6.7s avg)
    r"qwen2\.5vl:32b": (ModelCapability.VISION, ModelCapability.TEXT, 5, 128000),
    r"qwen2\.5vl:7b-fp16": (ModelCapability.VISION, ModelCapability.TEXT, 5, 128000),
    r"qwen2\.5vl:7b-q8_0": (ModelCapability.VISION, ModelCapability.TEXT, 5, 128000),
    r"qwen2\.5vl:7b": (ModelCapability.VISION, ModelCapability.TEXT, 5, 128000),

    # Granite Vision - FASTEST TESTED (100% parse rate, 2.9s avg)
    r"granite3\.2-vision:2b-fp16": (ModelCapability.VISION, ModelCapability.TEXT, 4, 16384),
    r"granite3\.2-vision:2b-q8_0": (ModelCapability.VISION, ModelCapability.TEXT, 4, 16384),
    r"granite3\.2-vision:2b": (ModelCapability.VISION, ModelCapability.TEXT, 4, 16384),

    # Qwen3-VL - HUGE CONTEXT (67% parse rate, 256K context)
    r"qwen3-vl:32b": (ModelCapability.VISION, ModelCapability.TEXT, 4, 262144),
    r"qwen3-vl:8b-thinking-bf16": (ModelCapability.VISION, ModelCapability.REASONING, 4, 262144),
    r"qwen3-vl:8b-instruct-bf16": (ModelCapability.VISION, ModelCapability.TEXT, 4, 262144),
    r"qwen3-vl:8b": (ModelCapability.VISION, ModelCapability.TEXT, 4, 262144),
    r"qwen3-vl:4b-thinking-bf16": (ModelCapability.VISION, ModelCapability.REASONING, 3, 262144),
    r"qwen3-vl:4b-instruct-bf16": (ModelCapability.VISION, ModelCapability.TEXT, 3, 262144),
    r"qwen3-vl:4b": (ModelCapability.VISION, ModelCapability.TEXT, 3, 262144),
    r"qwen3-vl:2b-thinking-bf16": (ModelCapability.VISION, ModelCapability.REASONING, 3, 262144),
    r"qwen3-vl:2b-instruct-bf16": (ModelCapability.VISION, ModelCapability.TEXT, 3, 262144),
    r"qwen3-vl:2b": (ModelCapability.VISION, ModelCapability.TEXT, 3, 262144),

    # Llama Vision - LARGE CONTEXT (33% parse rate, 128K context)
    r"llama3\.2-vision:11b-instruct-q8_0": (ModelCapability.VISION, ModelCapability.TEXT, 3, 131072),
    r"llama3\.2-vision:11b": (ModelCapability.VISION, ModelCapability.TEXT, 3, 131072),

    # BROKEN MODELS - Listed for reference but excluded from selection (quality=0)
    # These crash with "model runner has unexpectedly stopped" on this system
    r"llava:7b-v1\.6-mistral-fp16": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"llava:7b-v1\.6-mistral-q8_0": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"llava-llama3:8b": (ModelCapability.VISION, ModelCapability.TEXT, 0, 8192),
    r"llava:latest": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"llava:7b": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"llava:13b": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"minicpm-v:8b-2\.6-fp16": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"minicpm-v:8b-2\.6-q8_0": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"minicpm-v:8b": (ModelCapability.VISION, ModelCapability.TEXT, 0, 32768),
    r"deepseek-ocr:3b": (ModelCapability.VISION, ModelCapability.TEXT, 0, 8192),  # Garbage output

    # Llama 4 MoE Models (Mixture of Experts - huge but efficient)
    r"llama4:128x17b": (ModelCapability.TEXT, ModelCapability.REASONING, 5, 1048576),  # 1M context!
    r"llama4:16x17b": (ModelCapability.TEXT, ModelCapability.REASONING, 5, 262144),   # 256K context

    # Large Text Models
    r"llama3\.3:70b": (ModelCapability.TEXT, ModelCapability.REASONING, 5, 131072),
    r"qwen3:32b": (ModelCapability.TEXT, ModelCapability.REASONING, 5, 32768),
    r"devstral.*:24b": (ModelCapability.TEXT, ModelCapability.CODE, 5, 32768),
    r"aya-expanse:32b": (ModelCapability.TEXT, ModelCapability.REASONING, 4, 8192),
    r"deepseek-r1:32b": (ModelCapability.TEXT, ModelCapability.REASONING, 5, 65536),
    r"qwq:32b": (ModelCapability.TEXT, ModelCapability.REASONING, 5, 32768),

    # Medium Text Models
    r"qwen3:14b": (ModelCapability.TEXT, ModelCapability.REASONING, 4, 32768),
    r"gemma3:12b": (ModelCapability.TEXT, None, 3, 8192),
    r"gemma3:27b": (ModelCapability.TEXT, None, 4, 8192),
    r"qwen3:8b": (ModelCapability.TEXT, None, 3, 32768),
    r"ministral.*:8b": (ModelCapability.TEXT, None, 3, 32768),
    r"deepseek-r1:8b": (ModelCapability.TEXT, ModelCapability.REASONING, 3, 65536),

    # Llama 3.2 Text Models (efficient small models)
    r"llama3\.2:3b-instruct-fp16": (ModelCapability.TEXT, None, 3, 131072),  # Full precision instruct
    r"llama3\.2:3b-text-fp16": (ModelCapability.TEXT, None, 3, 131072),      # Full precision base
    r"llama3\.2:3b": (ModelCapability.TEXT, None, 2, 131072),                # Quantized 3B
    r"llama3\.2:1b-instruct-fp16": (ModelCapability.TEXT, None, 2, 131072),  # Full precision 1B instruct
    r"llama3\.2:1b-text-fp16": (ModelCapability.TEXT, None, 2, 131072),      # Full precision 1B base
    r"llama3\.2:1b": (ModelCapability.TEXT, None, 1, 131072),                # Quantized 1B

    # Small Text Models
    r"phi4-mini:3\.8b": (ModelCapability.TEXT, None, 2, 16384),
    r"gemma3:4b": (ModelCapability.TEXT, None, 2, 8192),
    r"smollm2:1\.7b": (ModelCapability.TEXT, None, 1, 8192),

    # Embedding Models
    r"mxbai-embed": (ModelCapability.EMBEDDING, None, 3, 512),
    r"snowflake-arctic-embed": (ModelCapability.EMBEDDING, None, 3, 512),
    r"granite-embedding": (ModelCapability.EMBEDDING, None, 2, 512),
    r"qwen3-embedding": (ModelCapability.EMBEDDING, None, 4, 40960),  # High quality, long context

    # Function/Tool Models
    r"functiongemma": (ModelCapability.FUNCTION, ModelCapability.TEXT, 2, 8192),
}


class ModelSelector:
    """
    Intelligent model selection based on capabilities and resources.

    Prioritizes the most powerful model available that fits the task
    requirements and available VRAM.
    """

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self._model_cache: Dict[str, ModelInfo] = {}
        self._available_models: List[str] = []
        self._last_refresh = 0

    async def refresh_models(self) -> List[ModelInfo]:
        """Refresh the list of available models from Ollama"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                data = response.json()

            models = []
            for model_data in data.get("models", []):
                name = model_data.get("name", "")
                size_bytes = model_data.get("size", 0)
                size_gb = size_bytes / (1024 ** 3)

                model_info = self._classify_model(name, size_gb)
                models.append(model_info)
                self._model_cache[name] = model_info

            self._available_models = [m.name for m in models]
            logger.info(f"Refreshed model list: {len(models)} models available")
            return models

        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
            return []

    def _classify_model(self, name: str, size_gb: float) -> ModelInfo:
        """Classify a model based on its name pattern"""
        capabilities = []
        quality_tier = 2  # Default
        context_length = 4096  # Default
        quantization = None

        # Check for quantization in name
        if "q8_0" in name:
            quantization = "q8_0"
        elif "q4_0" in name:
            quantization = "q4_0"
        elif "fp16" in name:
            quantization = "fp16"

        # Match against known patterns
        for pattern, info in MODEL_PATTERNS.items():
            if re.search(pattern, name, re.IGNORECASE):
                caps = [c for c in info[:2] if c is not None]
                capabilities.extend(caps)
                quality_tier = info[2]
                context_length = info[3]
                break
        else:
            # Default to text capability if no pattern matches
            capabilities = [ModelCapability.TEXT]

        return ModelInfo(
            name=name,
            size_gb=size_gb,
            capabilities=capabilities,
            quality_tier=quality_tier,
            context_length=context_length,
            quantization=quantization
        )

    async def get_available_vram(self) -> float:
        """Get available VRAM in GB using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Sum up free memory from all GPUs
                free_mb = sum(int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip())
                return free_mb / 1024  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not get VRAM info: {e}")
        return 24.0  # Assume 24GB if we can't detect

    async def get_loaded_models(self) -> List[str]:
        """Get currently loaded models in Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_host}/api/ps")
                response.raise_for_status()
                data = response.json()

            return [m.get("name", "") for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not get loaded models: {e}")
            return []

    async def select_best_model(
        self,
        capability: ModelCapability,
        max_size_gb: Optional[float] = None,
        min_quality: int = 1,
        prefer_loaded: bool = True
    ) -> Optional[ModelInfo]:
        """
        Select the best available model for a given capability.

        Args:
            capability: Required capability (VISION, TEXT, etc.)
            max_size_gb: Maximum model size in GB (defaults to available VRAM)
            min_quality: Minimum quality tier (1-5)
            prefer_loaded: Prefer already-loaded models to avoid load time

        Returns:
            Best matching ModelInfo or None
        """
        # Ensure we have model info
        if not self._model_cache:
            await self.refresh_models()

        # Get constraints
        if max_size_gb is None:
            max_size_gb = await self.get_available_vram()

        # Get loaded models for preference
        loaded_models = set()
        if prefer_loaded:
            loaded_models = set(await self.get_loaded_models())

        # Filter and sort candidates
        candidates = []
        for model_info in self._model_cache.values():
            # Check capability
            if capability not in model_info.capabilities:
                continue

            # Check size constraint
            if model_info.size_gb > max_size_gb:
                continue

            # Check quality
            if model_info.quality_tier < min_quality:
                continue

            candidates.append(model_info)

        if not candidates:
            logger.warning(f"No models found for capability {capability} within {max_size_gb}GB")
            return None

        # Sort by: loaded status (if preferring), then quality, then size (bigger is better for quality)
        def sort_key(m: ModelInfo) -> Tuple[int, int, float]:
            is_loaded = 1 if m.name in loaded_models else 0
            return (is_loaded, m.quality_tier, m.size_gb)

        candidates.sort(key=sort_key, reverse=True)

        best = candidates[0]
        logger.info(f"Selected model '{best.name}' (tier {best.quality_tier}, {best.size_gb:.1f}GB) for {capability.value}")
        return best

    async def select_vision_model(
        self,
        max_size_gb: Optional[float] = None,
        min_quality: int = 2
    ) -> Optional[ModelInfo]:
        """Convenience method to select the best vision model"""
        return await self.select_best_model(
            ModelCapability.VISION,
            max_size_gb=max_size_gb,
            min_quality=min_quality
        )

    async def select_text_model(
        self,
        max_size_gb: Optional[float] = None,
        min_quality: int = 2,
        prefer_reasoning: bool = False
    ) -> Optional[ModelInfo]:
        """Convenience method to select the best text model"""
        if prefer_reasoning:
            # Try reasoning model first
            model = await self.select_best_model(
                ModelCapability.REASONING,
                max_size_gb=max_size_gb,
                min_quality=min_quality
            )
            if model:
                return model

        return await self.select_best_model(
            ModelCapability.TEXT,
            max_size_gb=max_size_gb,
            min_quality=min_quality
        )

    async def get_model_for_task(
        self,
        task_type: str,
        content_size: int = 0
    ) -> Optional[ModelInfo]:
        """
        Get the best model for a specific task type.

        Args:
            task_type: Type of task (screenshot_extraction, relevance_check, summarization)
            content_size: Size of content to process (for context length considerations)

        Returns:
            Best ModelInfo for the task
        """
        available_vram = await self.get_available_vram()

        if task_type == "screenshot_extraction":
            # Need vision model, prioritize quality
            return await self.select_vision_model(
                max_size_gb=available_vram,
                min_quality=3  # Want good quality for extraction
            )

        elif task_type == "relevance_check":
            # Need text model, can use smaller one
            return await self.select_text_model(
                max_size_gb=min(available_vram, 10),  # Don't use huge model for simple check
                min_quality=2
            )

        elif task_type == "summarization":
            # Need good text model
            return await self.select_text_model(
                max_size_gb=available_vram,
                min_quality=3,
                prefer_reasoning=True
            )

        else:
            # Default to text
            return await self.select_text_model(max_size_gb=available_vram)


# Singleton instance
_selector: Optional[ModelSelector] = None


def get_model_selector(ollama_host: str = "http://localhost:11434") -> ModelSelector:
    """Get the global ModelSelector instance"""
    global _selector
    if _selector is None:
        _selector = ModelSelector(ollama_host)
    return _selector


async def main():
    """Test the model selector"""
    selector = get_model_selector()
    await selector.refresh_models()

    print("\n=== Available Models ===")
    for name, info in selector._model_cache.items():
        caps = [c.value for c in info.capabilities]
        print(f"{name}: {info.size_gb:.1f}GB, tier={info.quality_tier}, caps={caps}")

    print("\n=== Best Vision Model ===")
    vision_model = await selector.select_vision_model()
    if vision_model:
        print(f"Selected: {vision_model.name} ({vision_model.size_gb:.1f}GB)")

    print("\n=== Best Text Model ===")
    text_model = await selector.select_text_model()
    if text_model:
        print(f"Selected: {text_model.name} ({text_model.size_gb:.1f}GB)")

    print("\n=== Available VRAM ===")
    vram = await selector.get_available_vram()
    print(f"Available: {vram:.1f}GB")


if __name__ == "__main__":
    asyncio.run(main())
