"""
GPU Monitor Service

Provides real-time GPU monitoring for intelligent model selection:
- Tracks VRAM usage across all NVIDIA GPUs
- Monitors loaded Ollama models and their actual VRAM consumption
- Calculates overhead from context windows and image processing
- Provides metrics for model selection decisions
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

import httpx

logger = logging.getLogger("services.gpu_monitor")


@dataclass
class GPUDevice:
    """Information about a single GPU"""
    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    utilization_percent: float
    temperature_c: Optional[int] = None
    power_draw_w: Optional[float] = None

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_mb / 1024

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory_mb / 1024

    @property
    def used_memory_gb(self) -> float:
        return self.used_memory_mb / 1024

    @property
    def usage_percent(self) -> float:
        return (self.used_memory_mb / self.total_memory_mb * 100) if self.total_memory_mb > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "free_memory_gb": round(self.free_memory_gb, 2),
            "used_memory_gb": round(self.used_memory_gb, 2),
            "usage_percent": round(self.usage_percent, 1),
            "utilization_percent": round(self.utilization_percent, 1),
            "temperature_c": self.temperature_c,
            "power_draw_w": round(self.power_draw_w, 1) if self.power_draw_w else None
        }


@dataclass
class LoadedModel:
    """Information about a model currently loaded in Ollama"""
    name: str
    size_mb: int
    vram_mb: int
    until: Optional[str] = None  # When model will be unloaded
    digest: Optional[str] = None

    @property
    def size_gb(self) -> float:
        return self.size_mb / 1024

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size_gb": round(self.size_gb, 2),
            "vram_gb": round(self.vram_gb, 2),
            "until": self.until,
            "digest": self.digest[:12] if self.digest else None
        }


@dataclass
class VRAMUsageSnapshot:
    """Snapshot of VRAM usage at a point in time"""
    timestamp: datetime
    gpu_index: int
    total_vram_mb: int
    used_vram_mb: int
    model_name: Optional[str] = None
    context_tokens: int = 0
    image_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "gpu_index": self.gpu_index,
            "total_vram_gb": round(self.total_vram_mb / 1024, 2),
            "used_vram_gb": round(self.used_vram_mb / 1024, 2),
            "model_name": self.model_name,
            "context_tokens": self.context_tokens,
            "image_count": self.image_count
        }


class GPUMonitor:
    """
    Centralized GPU monitoring service.

    Features:
    - Real-time GPU memory tracking via nvidia-smi
    - Ollama model tracking via ollama ps
    - Historical VRAM usage for overhead calculation
    - Smart caching with configurable TTL
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        cache_ttl_seconds: float = 2.0
    ):
        self.ollama_url = ollama_url
        self.cache_ttl = cache_ttl_seconds

        # Cached data
        self._gpus: List[GPUDevice] = []
        self._loaded_models: List[LoadedModel] = []
        self._last_gpu_refresh: float = 0
        self._last_models_refresh: float = 0

        # Historical data for overhead calculation
        self._usage_history: List[VRAMUsageSnapshot] = []
        self._max_history_size = 1000

        # Calculated overhead stats
        self._model_overhead_stats: Dict[str, Dict[str, float]] = {}

    async def get_gpu_info(self, force_refresh: bool = False) -> List[GPUDevice]:
        """
        Get current GPU information from nvidia-smi.

        Returns list of GPUDevice with memory and utilization stats.
        """
        now = time.time()
        if not force_refresh and (now - self._last_gpu_refresh < self.cache_ttl) and self._gpus:
            return self._gpus

        gpus = []
        try:
            # Query nvidia-smi for comprehensive GPU info
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 6:
                            try:
                                gpu = GPUDevice(
                                    index=int(parts[0]),
                                    name=parts[1],
                                    total_memory_mb=int(float(parts[2])),
                                    free_memory_mb=int(float(parts[3])),
                                    used_memory_mb=int(float(parts[4])),
                                    utilization_percent=float(parts[5]) if parts[5] not in ['[N/A]', '[Not Supported]'] else 0.0,
                                    temperature_c=int(float(parts[6])) if len(parts) > 6 and parts[6] not in ['[N/A]', '[Not Supported]'] else None,
                                    power_draw_w=float(parts[7]) if len(parts) > 7 and parts[7] not in ['[N/A]', '[Not Supported]'] else None
                                )
                                gpus.append(gpu)
                            except (ValueError, IndexError) as e:
                                logger.debug(f"Error parsing GPU line: {line} - {e}")

                self._gpus = gpus
                self._last_gpu_refresh = now

                if gpus:
                    total_free = sum(g.free_memory_gb for g in gpus)
                    total_used = sum(g.used_memory_gb for g in gpus)
                    logger.debug(f"GPU info: {len(gpus)} GPUs, {total_free:.1f}GB free, {total_used:.1f}GB used")

        except FileNotFoundError:
            logger.warning("nvidia-smi not found - GPU monitoring disabled")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
        except Exception as e:
            logger.error(f"GPU query failed: {e}")

        return gpus

    async def get_loaded_models(self, force_refresh: bool = False) -> List[LoadedModel]:
        """
        Get currently loaded Ollama models and their VRAM usage.

        Uses 'ollama ps' command or API to get loaded model info.
        """
        now = time.time()
        if not force_refresh and (now - self._last_models_refresh < self.cache_ttl) and self._loaded_models:
            return self._loaded_models

        models = []
        try:
            # Try ollama ps command first (more reliable for VRAM info)
            result = subprocess.run(
                ['ollama', 'ps'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Skip header line
                for line in lines[1:]:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            name = parts[0]
                            # Parse size (e.g., "5.2 GB" or "523 MB")
                            size_str = parts[2] + parts[3] if len(parts) > 3 else parts[2]
                            size_mb = self._parse_size_to_mb(size_str)

                            # VRAM is typically shown as the size for loaded models
                            vram_mb = size_mb

                            # Parse until time if present
                            until = None
                            for i, p in enumerate(parts):
                                if 'minute' in p.lower() or 'hour' in p.lower() or 'second' in p.lower():
                                    until = ' '.join(parts[i-1:i+1]) if i > 0 else p
                                    break

                            models.append(LoadedModel(
                                name=name,
                                size_mb=size_mb,
                                vram_mb=vram_mb,
                                until=until
                            ))

                self._loaded_models = models
                self._last_models_refresh = now

                if models:
                    total_vram = sum(m.vram_gb for m in models)
                    logger.debug(f"Loaded models: {len(models)}, total VRAM: {total_vram:.1f}GB")

        except FileNotFoundError:
            logger.debug("ollama command not found, trying API")
            # Fallback to API (may not have VRAM info)
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.ollama_url}/api/ps")
                    if response.status_code == 200:
                        data = response.json()
                        for model_info in data.get("models", []):
                            models.append(LoadedModel(
                                name=model_info.get("name", ""),
                                size_mb=model_info.get("size", 0) // (1024 * 1024),
                                vram_mb=model_info.get("size_vram", 0) // (1024 * 1024),
                                digest=model_info.get("digest"),
                                until=model_info.get("expires_at")
                            ))
                        self._loaded_models = models
                        self._last_models_refresh = now
            except Exception as e:
                logger.debug(f"Ollama API query failed: {e}")

        except Exception as e:
            logger.error(f"Failed to get loaded models: {e}")

        return models

    def _parse_size_to_mb(self, size_str: str) -> int:
        """Parse size string like '5.2 GB' or '523 MB' to MB"""
        try:
            size_str = size_str.upper().replace(' ', '')
            if 'GB' in size_str:
                return int(float(size_str.replace('GB', '')) * 1024)
            elif 'MB' in size_str:
                return int(float(size_str.replace('MB', '')))
            elif 'KB' in size_str:
                return int(float(size_str.replace('KB', '')) / 1024)
            else:
                return int(float(size_str))
        except:
            return 0

    async def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU and model summary.

        Returns combined stats useful for model selection.
        """
        gpus = await self.get_gpu_info()
        models = await self.get_loaded_models()

        total_vram = sum(g.total_memory_gb for g in gpus)
        free_vram = sum(g.free_memory_gb for g in gpus)
        used_vram = sum(g.used_memory_gb for g in gpus)
        model_vram = sum(m.vram_gb for m in models)

        # Calculate overhead (difference between GPU used and model sizes)
        # This includes CUDA context, driver overhead, etc.
        overhead_vram = used_vram - model_vram if model_vram > 0 else 0

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "gpu_count": len(gpus),
            "gpus": [g.to_dict() for g in gpus],
            "total_vram_gb": round(total_vram, 2),
            "free_vram_gb": round(free_vram, 2),
            "used_vram_gb": round(used_vram, 2),
            "loaded_models": [m.to_dict() for m in models],
            "model_count": len(models),
            "model_vram_gb": round(model_vram, 2),
            "overhead_vram_gb": round(overhead_vram, 2),
            "overhead_percent": round((overhead_vram / used_vram * 100) if used_vram > 0 else 0, 1),
            "can_load_additional_gb": round(free_vram - 2.0, 2),  # 2GB buffer
        }

    async def record_usage_snapshot(
        self,
        model_name: Optional[str] = None,
        context_tokens: int = 0,
        image_count: int = 0
    ):
        """
        Record a VRAM usage snapshot for overhead calculation.

        Call this before/after loading models or processing to track overhead.
        """
        gpus = await self.get_gpu_info(force_refresh=True)

        for gpu in gpus:
            snapshot = VRAMUsageSnapshot(
                timestamp=datetime.utcnow(),
                gpu_index=gpu.index,
                total_vram_mb=gpu.total_memory_mb,
                used_vram_mb=gpu.used_memory_mb,
                model_name=model_name,
                context_tokens=context_tokens,
                image_count=image_count
            )
            self._usage_history.append(snapshot)

        # Trim history if too large
        if len(self._usage_history) > self._max_history_size:
            self._usage_history = self._usage_history[-self._max_history_size:]

    def calculate_overhead_stats(self) -> Dict[str, Any]:
        """
        Calculate overhead statistics from usage history.

        Returns estimated VRAM overhead per context token and per image.
        """
        if len(self._usage_history) < 2:
            return {
                "samples": 0,
                "message": "Insufficient data for overhead calculation"
            }

        # Group by model and calculate deltas
        model_deltas: Dict[str, List[Dict]] = {}

        for i in range(1, len(self._usage_history)):
            prev = self._usage_history[i-1]
            curr = self._usage_history[i]

            if prev.model_name and curr.model_name == prev.model_name:
                delta_vram = curr.used_vram_mb - prev.used_vram_mb
                delta_tokens = curr.context_tokens - prev.context_tokens
                delta_images = curr.image_count - prev.image_count

                if prev.model_name not in model_deltas:
                    model_deltas[prev.model_name] = []

                model_deltas[prev.model_name].append({
                    "delta_vram_mb": delta_vram,
                    "delta_tokens": delta_tokens,
                    "delta_images": delta_images
                })

        # Calculate per-token and per-image overhead
        stats = {
            "samples": len(self._usage_history),
            "models": {}
        }

        for model_name, deltas in model_deltas.items():
            token_deltas = [(d["delta_vram_mb"], d["delta_tokens"])
                           for d in deltas if d["delta_tokens"] > 0]
            image_deltas = [(d["delta_vram_mb"], d["delta_images"])
                           for d in deltas if d["delta_images"] > 0]

            model_stats = {}

            if token_deltas:
                avg_mb_per_token = sum(d[0]/d[1] for d in token_deltas) / len(token_deltas)
                model_stats["mb_per_1k_tokens"] = round(avg_mb_per_token * 1000, 2)

            if image_deltas:
                avg_mb_per_image = sum(d[0]/d[1] for d in image_deltas) / len(image_deltas)
                model_stats["mb_per_image"] = round(avg_mb_per_image, 2)

            if model_stats:
                stats["models"][model_name] = model_stats

        return stats

    async def can_load_model(self, required_vram_gb: float, buffer_gb: float = 2.0) -> Dict[str, Any]:
        """
        Check if a model with given VRAM requirement can be loaded.

        Returns which GPU(s) can accommodate the model.
        """
        gpus = await self.get_gpu_info()

        results = {
            "can_load": False,
            "required_gb": required_vram_gb,
            "buffer_gb": buffer_gb,
            "total_required_gb": required_vram_gb + buffer_gb,
            "suitable_gpus": [],
            "recommendation": None
        }

        for gpu in gpus:
            available = gpu.free_memory_gb - buffer_gb
            if available >= required_vram_gb:
                results["suitable_gpus"].append({
                    "index": gpu.index,
                    "name": gpu.name,
                    "free_gb": round(gpu.free_memory_gb, 2),
                    "headroom_gb": round(available - required_vram_gb, 2)
                })

        results["can_load"] = len(results["suitable_gpus"]) > 0

        if results["suitable_gpus"]:
            # Recommend GPU with most headroom
            best = max(results["suitable_gpus"], key=lambda x: x["headroom_gb"])
            results["recommendation"] = f"Use GPU {best['index']} ({best['name']}) with {best['headroom_gb']:.1f}GB headroom"
        else:
            total_free = sum(g.free_memory_gb for g in gpus)
            results["recommendation"] = f"Insufficient VRAM. Need {required_vram_gb + buffer_gb:.1f}GB, only {total_free:.1f}GB free across all GPUs"

        return results


# Global monitor instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor(ollama_url: str = "http://localhost:11434") -> GPUMonitor:
    """Get or create the global GPU monitor instance"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(ollama_url=ollama_url)
    return _gpu_monitor


async def get_gpu_summary() -> Dict[str, Any]:
    """Convenience function to get GPU summary"""
    monitor = get_gpu_monitor()
    return await monitor.get_summary()
