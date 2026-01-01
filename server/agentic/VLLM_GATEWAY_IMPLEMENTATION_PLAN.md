# vLLM Gateway Implementation Plan

> **Version**: 1.0 | **Date**: 2026-01-01 | **Status**: Planning Complete

## Executive Summary

This document presents a comprehensive plan for implementing an LLM Gateway sub-project that will:
1. **Track all LLM calls** across memOS, Android, and PDF Tools subsystems
2. **Manage VRAM allocation** with queue-based scheduling
3. **Route requests** between Ollama (current) and vLLM (target)
4. **Support incremental migration** without breaking existing functionality
5. **Monitor and optimize** inference performance

**Key Finding from Audit**: The system currently has 100+ LLM call sites across 45+ files with **API format mismatch** between memOS (Ollama native `/api/generate`) and Android (OpenAI-compatible `/v1/chat/completions`). The gateway must handle translation between these formats.

---

## Table of Contents

1. [Current Architecture Audit](#1-current-architecture-audit)
2. [Gateway Architecture Design](#2-gateway-architecture-design)
3. [API Translation Layer](#3-api-translation-layer)
4. [VRAM Management & Scheduling](#4-vram-management--scheduling)
5. [Multi-Tenant Request Tracking](#5-multi-tenant-request-tracking)
6. [vLLM Integration Strategy](#6-vllm-integration-strategy)
7. [Migration Phases](#7-migration-phases)
8. [Monitoring & Observability](#8-monitoring--observability)
9. [Risk Mitigation](#9-risk-mitigation)
10. [Implementation Timeline](#10-implementation-timeline)

---

## 1. Current Architecture Audit

### 1.1 LLM Call Sites Summary

| Subsystem | Files | Call Sites | Primary Endpoint | Models Used |
|-----------|-------|------------|------------------|-------------|
| **memOS Server** | 45+ | 100+ | `/api/generate` (Ollama native) | qwen3:8b, deepseek-r1:14b, gemma3:4b |
| **Android Client** | 8+ | 20+ | `/v1/chat/completions` (OpenAI) | qwen3:8b, llama3.2:3b |
| **PDF Tools** | 10+ | 15+ | `/api/generate` (Ollama native) | qwen3:8b, GLiNER |
| **Agentic Modules** | 15+ | 50+ | Mixed | Multiple per task |

### 1.2 Current LLM Call Pattern (memOS)

```python
# Pattern found across memOS codebase
async with httpx.AsyncClient(timeout=30-120.0) as client:
    response = await client.post(
        f"{self.ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_ctx": 32768}
        }
    )
```

### 1.3 Current LLM Call Pattern (Android)

```kotlin
// OpenAI-compatible format
@POST("v1/chat/completions")
suspend fun chatCompletion(@Body request: ChatCompletionRequest): ChatCompletionResponse

data class ChatCompletionRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val temperature: Float = 0.7f,
    val max_tokens: Int = 4096,
    val stream: Boolean = false
)
```

### 1.4 30-Phase Pipeline LLM Calls

| Phase | LLM Call # | Model | Timeout | Purpose |
|-------|-----------|-------|---------|---------|
| 0 | #1 | DeepSeek R1 | 30s | Query Classification |
| 1 | #2 | qwen3:8b/gemma3:4b | 60s | Query Analysis |
| 3 | #3 | qwen3:8b | 30s | HyDE Expansion |
| 4 | #4 | qwen3:8b | 30s | Query Tree Decoding |
| 9 | #5-6 | qwen3:8b/gemma3:4b | 60s | CRAG Evaluation |
| 12 | #7 | qwen3:8b | 30s | Content Curation |
| 14 | #8 | qwen3:8b | 30s | URL Evaluation |
| 17 | #9 | qwen3:8b/gemma3:4b | 30s | Verification |
| 20 | #10 | DeepSeek R1/qwen3:8b | 180-600s | Synthesis |
| 22 | #11 | gemma3:4b | 30s | Quality Evaluation |
| 23 | #12 | qwen3:8b/gemma3:4b | 60s | Self-RAG Reflection |
| 25 | #13-14 | qwen3:8b | 60s | Experience Distillation |
| 27 | #15 | gemma3:4b | 30s | Classifier Feedback |

### 1.5 Model Inventory

| Model | Parameters | Primary Use | Context Window | VRAM (FP16) |
|-------|------------|-------------|----------------|-------------|
| deepseek-r1:14b-qwen-distill-q8_0 | 14B Q8 | Reasoning, Synthesis | 32K | ~14GB |
| qwen3:8b | 8B | Analysis, Classification | 32K | ~16GB |
| gemma3:4b | 4B | Fast evaluation | 8K | ~8GB |
| llama3.2:3b | 3B | Edge inference | 8K | ~6GB |
| mxbai-embed-large | 335M | Embeddings | 512 | ~0.7GB |
| bge-m3 | 568M | Hybrid retrieval | 8K | ~1.2GB |

---

## 2. Gateway Architecture Design

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM GATEWAY SERVICE                                │
│                              Port 8100                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      REQUEST ROUTER                                  │   │
│  │  - API format detection (Ollama native vs OpenAI)                   │   │
│  │  - Request ID generation and tracking                                │   │
│  │  - Source system identification                                      │   │
│  │  - Priority queue placement                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   REQUEST QUEUE & SCHEDULER                          │   │
│  │  - Priority-based scheduling (by source, model, urgency)            │   │
│  │  - VRAM-aware batching                                               │   │
│  │  - Preemption for high-priority requests                            │   │
│  │  - Model loading/unloading orchestration                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│              ┌─────────────────────┴─────────────────────┐                  │
│              ▼                                           ▼                   │
│  ┌───────────────────────┐                ┌───────────────────────┐         │
│  │   OLLAMA BACKEND      │                │    vLLM BACKEND       │         │
│  │   (Port 11434)        │                │    (Port 8000)        │         │
│  │   - Primary (current) │                │    - Target (future)  │         │
│  │   - KV cache q8_0     │                │    - PagedAttention   │         │
│  │   - Flash attention   │                │    - Prefix caching   │         │
│  └───────────────────────┘                └───────────────────────┘         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    METRICS & MONITORING                              │   │
│  │  - VRAM utilization tracking                                         │   │
│  │  - Request latency histograms                                        │   │
│  │  - Token throughput counters                                         │   │
│  │  - Queue depth gauges                                                │   │
│  │  - Model state tracking                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Request Router** | API detection, request parsing, source identification | FastAPI middleware |
| **Queue Manager** | Priority queuing, VRAM-aware scheduling | Redis + custom scheduler |
| **Backend Manager** | Model loading, health checks, failover | Async controller |
| **Metrics Collector** | Prometheus metrics, request tracing | OpenTelemetry |
| **API Translator** | Ollama ↔ OpenAI format conversion | Pydantic models |

### 2.3 Directory Structure

```
memOS/server/
├── gateway/                      # NEW: LLM Gateway sub-project
│   ├── __init__.py
│   ├── main.py                   # FastAPI app (port 8100)
│   ├── router.py                 # Request routing logic
│   ├── queue_manager.py          # Priority queue + scheduler
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract backend interface
│   │   ├── ollama_backend.py     # Ollama adapter
│   │   └── vllm_backend.py       # vLLM adapter
│   ├── api_translator.py         # Format conversion
│   ├── vram_manager.py           # VRAM tracking + allocation
│   ├── metrics.py                # Prometheus metrics
│   ├── models.py                 # Pydantic request/response models
│   ├── config.py                 # Gateway configuration
│   └── tests/
│       ├── test_router.py
│       ├── test_queue.py
│       └── test_backends.py
└── docker-compose.gateway.yml    # Gateway deployment
```

---

## 3. API Translation Layer

### 3.1 Format Conversion Matrix

| Source Format | Target Format | Translation Required |
|---------------|---------------|---------------------|
| Ollama `/api/generate` | Ollama `/api/generate` | None |
| Ollama `/api/generate` | vLLM `/v1/completions` | Yes |
| OpenAI `/v1/chat/completions` | Ollama `/api/generate` | Yes |
| OpenAI `/v1/chat/completions` | vLLM `/v1/chat/completions` | None |

### 3.2 Request Models

```python
# gateway/models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class SourceSystem(str, Enum):
    MEMOS = "memos"
    ANDROID = "android"
    PDF_TOOLS = "pdf_tools"
    EXTERNAL = "external"

class APIFormat(str, Enum):
    OLLAMA_NATIVE = "ollama_native"      # /api/generate
    OLLAMA_CHAT = "ollama_chat"          # /api/chat
    OPENAI_CHAT = "openai_chat"          # /v1/chat/completions
    OPENAI_COMPLETION = "openai_completion"  # /v1/completions

class GatewayRequest(BaseModel):
    """Unified gateway request format."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_system: SourceSystem
    source_api_format: APIFormat
    priority: int = Field(default=5, ge=1, le=10)  # 1=highest, 10=lowest

    # Content (one of these will be populated)
    prompt: Optional[str] = None                    # For generate endpoints
    messages: Optional[List[dict]] = None           # For chat endpoints

    # Model configuration
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = False

    # Context
    context_window: int = 32768
    stop_sequences: Optional[List[str]] = None

    # Tracking
    metadata: dict = Field(default_factory=dict)

class GatewayResponse(BaseModel):
    """Unified gateway response format."""
    request_id: str
    success: bool

    # Content (format matches source_api_format)
    content: str

    # Metrics
    tokens_in: int
    tokens_out: int
    latency_ms: float
    ttft_ms: float  # Time to first token

    # Routing info
    backend_used: str  # "ollama" or "vllm"
    model_used: str
    queue_wait_ms: float

    # Error handling
    error: Optional[str] = None
```

### 3.3 Translation Functions

```python
# gateway/api_translator.py

class APITranslator:
    """Translates between API formats."""

    def to_unified(self, request: dict, source_format: APIFormat) -> GatewayRequest:
        """Convert any format to unified GatewayRequest."""
        if source_format == APIFormat.OLLAMA_NATIVE:
            return self._from_ollama_native(request)
        elif source_format == APIFormat.OPENAI_CHAT:
            return self._from_openai_chat(request)
        # ... other formats

    def from_unified(self, request: GatewayRequest, target_format: APIFormat) -> dict:
        """Convert unified request to target backend format."""
        if target_format == APIFormat.OLLAMA_NATIVE:
            return self._to_ollama_native(request)
        elif target_format == APIFormat.OPENAI_CHAT:
            return self._to_openai_chat(request)
        # ... other formats

    def _from_ollama_native(self, request: dict) -> GatewayRequest:
        """Convert Ollama /api/generate format."""
        return GatewayRequest(
            prompt=request.get("prompt"),
            model=request["model"],
            temperature=request.get("options", {}).get("temperature", 0.7),
            context_window=request.get("options", {}).get("num_ctx", 32768),
            stream=request.get("stream", False),
            source_api_format=APIFormat.OLLAMA_NATIVE
        )

    def _from_openai_chat(self, request: dict) -> GatewayRequest:
        """Convert OpenAI /v1/chat/completions format."""
        return GatewayRequest(
            messages=request.get("messages"),
            model=request["model"],
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens", 4096),
            stream=request.get("stream", False),
            source_api_format=APIFormat.OPENAI_CHAT
        )

    def _to_ollama_native(self, request: GatewayRequest) -> dict:
        """Convert to Ollama /api/generate format."""
        if request.messages:
            # Convert chat messages to single prompt
            prompt = self._messages_to_prompt(request.messages)
        else:
            prompt = request.prompt

        return {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_ctx": request.context_window,
            }
        }

    def _to_openai_chat(self, request: GatewayRequest) -> dict:
        """Convert to OpenAI /v1/chat/completions format."""
        if request.prompt:
            messages = [{"role": "user", "content": request.prompt}]
        else:
            messages = request.messages

        return {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }

    def _messages_to_prompt(self, messages: List[dict]) -> str:
        """Convert chat messages to single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)
```

---

## 4. VRAM Management & Scheduling

### 4.1 VRAM Budget (TITAN RTX 24GB)

| Component | Allocation | Notes |
|-----------|------------|-------|
| **Inference Models** | 18GB | Primary allocation |
| **KV Cache** | 4GB | Dynamic allocation |
| **System Overhead** | 2GB | CUDA, PyTorch |

### 4.2 Model Memory Requirements

| Model | FP16 | Q8_0 | Q4_K_M | Max Context |
|-------|------|------|--------|-------------|
| deepseek-r1:14b | N/A | ~14GB | ~8GB | 32K |
| qwen3:8b | ~16GB | ~8GB | ~4GB | 32K |
| gemma3:4b | ~8GB | ~4GB | ~2GB | 8K |
| llama3.2:3b | ~6GB | ~3GB | ~1.5GB | 8K |

### 4.3 KV Cache Formula

```python
def calculate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
    batch_size: int = 1,
    precision: str = "fp16"
) -> int:
    """
    KV Cache Size = 2 × L × H × D × S × B × bytes_per_element

    Where:
    - 2 = Key + Value
    - L = Number of layers
    - H = Number of attention heads
    - D = Head dimension
    - S = Sequence length
    - B = Batch size
    """
    bytes_per_element = {"fp32": 4, "fp16": 2, "q8_0": 1, "q4_0": 0.5}
    return 2 * num_layers * num_heads * head_dim * sequence_length * batch_size * bytes_per_element[precision]

# Example: qwen3:8b with 32K context
# 2 × 32 layers × 32 heads × 128 dim × 32768 seq × 1 batch × 2 bytes
# = ~8.6 GB KV cache for full context
```

### 4.4 VRAM Manager

```python
# gateway/vram_manager.py

from dataclasses import dataclass
from typing import Dict, Optional
import asyncio
import subprocess
import re

@dataclass
class VRAMState:
    total_mb: int
    used_mb: int
    free_mb: int
    models_loaded: Dict[str, int]  # model -> memory_mb

    @property
    def utilization(self) -> float:
        return self.used_mb / self.total_mb

class VRAMManager:
    """Manages GPU VRAM allocation for model inference."""

    def __init__(self, target_utilization: float = 0.85):
        self.target_utilization = target_utilization
        self.model_memory_estimates: Dict[str, int] = {
            "deepseek-r1:14b-qwen-distill-q8_0": 14000,
            "qwen3:8b": 8000,
            "gemma3:4b": 4000,
            "llama3.2:3b": 3000,
            "mxbai-embed-large": 700,
            "bge-m3": 1200,
        }
        self._lock = asyncio.Lock()

    async def get_vram_state(self) -> VRAMState:
        """Get current VRAM state from nvidia-smi."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        total, used, free = map(int, result.stdout.strip().split(", "))

        # Get loaded models from Ollama
        models_loaded = await self._get_ollama_loaded_models()

        return VRAMState(
            total_mb=total,
            used_mb=used,
            free_mb=free,
            models_loaded=models_loaded
        )

    async def can_load_model(self, model: str) -> bool:
        """Check if there's enough VRAM to load a model."""
        state = await self.get_vram_state()
        required = self.model_memory_estimates.get(model, 8000)  # Default 8GB
        available = state.total_mb * self.target_utilization - state.used_mb
        return available >= required

    async def suggest_eviction(self, required_mb: int) -> Optional[str]:
        """Suggest a model to evict to free space."""
        state = await self.get_vram_state()

        # Find smallest model that frees enough space
        candidates = sorted(
            state.models_loaded.items(),
            key=lambda x: x[1]  # Sort by memory size
        )

        for model, mem in candidates:
            if mem >= required_mb:
                return model

        return None

    async def _get_ollama_loaded_models(self) -> Dict[str, int]:
        """Get currently loaded models from Ollama."""
        # Ollama ps command shows loaded models
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True
        )
        # Parse output to get model names and sizes
        # Format: NAME  ID  SIZE  PROCESSOR  UNTIL
        models = {}
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                size_str = parts[2]
                # Parse size (e.g., "8.0 GB" -> 8000)
                size_mb = self._parse_size(size_str)
                models[name] = size_mb
        return models

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '8.0 GB' to MB."""
        match = re.match(r"([\d.]+)\s*(GB|MB)", size_str, re.I)
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            return int(value * 1000) if unit == "GB" else int(value)
        return 0
```

### 4.5 Priority Queue Scheduler

```python
# gateway/queue_manager.py

import asyncio
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Optional
import time

@dataclass(order=True)
class QueuedRequest:
    priority: int
    timestamp: float = field(compare=False)
    request: GatewayRequest = field(compare=False)

    def __init__(self, request: GatewayRequest):
        # Lower number = higher priority
        # Adjust for source system and urgency
        base_priority = request.priority

        # Boost priority for Android (user-facing)
        if request.source_system == SourceSystem.ANDROID:
            base_priority -= 2

        # Boost priority for streaming requests
        if request.stream:
            base_priority -= 1

        self.priority = max(1, base_priority)
        self.timestamp = time.time()
        self.request = request

class QueueManager:
    """Priority queue with VRAM-aware scheduling."""

    def __init__(self, vram_manager: VRAMManager, max_concurrent: int = 4):
        self.vram_manager = vram_manager
        self.max_concurrent = max_concurrent
        self.queue: list = []  # heapq
        self._lock = asyncio.Lock()
        self._active_count = 0
        self._request_complete = asyncio.Event()

        # Metrics
        self.total_queued = 0
        self.total_processed = 0
        self.total_wait_time_ms = 0

    async def enqueue(self, request: GatewayRequest) -> str:
        """Add request to priority queue. Returns request_id."""
        async with self._lock:
            queued = QueuedRequest(request)
            heappush(self.queue, queued)
            self.total_queued += 1
        return request.request_id

    async def dequeue(self) -> Optional[GatewayRequest]:
        """Get next request if capacity available."""
        async with self._lock:
            if self._active_count >= self.max_concurrent:
                return None

            if not self.queue:
                return None

            queued = heappop(self.queue)

            # Check VRAM availability
            can_run = await self.vram_manager.can_load_model(queued.request.model)
            if not can_run:
                # Put back in queue and try to evict
                heappush(self.queue, queued)
                return None

            self._active_count += 1
            wait_time = (time.time() - queued.timestamp) * 1000
            self.total_wait_time_ms += wait_time
            queued.request.metadata["queue_wait_ms"] = wait_time

            return queued.request

    async def complete(self, request_id: str):
        """Mark request as complete."""
        async with self._lock:
            self._active_count -= 1
            self.total_processed += 1
            self._request_complete.set()

    async def wait_for_capacity(self, timeout: float = 30.0):
        """Wait until there's capacity to process."""
        start = time.time()
        while time.time() - start < timeout:
            if self._active_count < self.max_concurrent:
                return True
            self._request_complete.clear()
            try:
                await asyncio.wait_for(
                    self._request_complete.wait(),
                    timeout=timeout - (time.time() - start)
                )
            except asyncio.TimeoutError:
                break
        return False

    @property
    def stats(self) -> dict:
        return {
            "queue_depth": len(self.queue),
            "active_requests": self._active_count,
            "total_queued": self.total_queued,
            "total_processed": self.total_processed,
            "avg_wait_time_ms": (
                self.total_wait_time_ms / self.total_processed
                if self.total_processed > 0 else 0
            ),
        }
```

---

## 5. Multi-Tenant Request Tracking

### 5.1 Request Tracking Schema

```python
# gateway/tracking.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class RequestRecord:
    """Complete record of a gateway request."""
    request_id: str
    source_system: SourceSystem
    source_ip: Optional[str]
    user_id: Optional[str]

    # Request details
    model: str
    prompt_length: int
    max_tokens: int
    stream: bool

    # Timing
    received_at: datetime
    queued_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Results
    success: bool
    tokens_in: int
    tokens_out: int
    latency_ms: float
    ttft_ms: float
    queue_wait_ms: float

    # Backend
    backend_used: str
    model_loaded_before: bool

    # Error
    error: Optional[str] = None
```

### 5.2 Per-System Quotas

```python
# gateway/quotas.py

from dataclasses import dataclass
from typing import Dict

@dataclass
class SystemQuota:
    """Per-system request limits."""
    requests_per_minute: int
    tokens_per_minute: int
    max_concurrent: int
    max_context_window: int
    priority_boost: int  # Negative = higher priority

DEFAULT_QUOTAS: Dict[SourceSystem, SystemQuota] = {
    SourceSystem.ANDROID: SystemQuota(
        requests_per_minute=60,
        tokens_per_minute=100_000,
        max_concurrent=4,
        max_context_window=32768,
        priority_boost=-2  # Highest priority
    ),
    SourceSystem.MEMOS: SystemQuota(
        requests_per_minute=120,
        tokens_per_minute=500_000,
        max_concurrent=8,
        max_context_window=65536,
        priority_boost=0
    ),
    SourceSystem.PDF_TOOLS: SystemQuota(
        requests_per_minute=30,
        tokens_per_minute=50_000,
        max_concurrent=2,
        max_context_window=16384,
        priority_boost=2  # Lower priority (batch processing)
    ),
    SourceSystem.EXTERNAL: SystemQuota(
        requests_per_minute=10,
        tokens_per_minute=10_000,
        max_concurrent=1,
        max_context_window=8192,
        priority_boost=5  # Lowest priority
    ),
}
```

### 5.3 Cost Tracking

```python
# gateway/cost_tracker.py

from datetime import datetime, timedelta
from collections import defaultdict

class CostTracker:
    """Track token usage and costs per system."""

    # Cost per 1M tokens (example rates)
    COST_PER_MILLION_INPUT = 0.15
    COST_PER_MILLION_OUTPUT = 0.60

    def __init__(self):
        self.usage: Dict[SourceSystem, Dict[str, int]] = defaultdict(
            lambda: {"tokens_in": 0, "tokens_out": 0, "requests": 0}
        )
        self.reset_time = datetime.now()

    def record(self, source: SourceSystem, tokens_in: int, tokens_out: int):
        self.usage[source]["tokens_in"] += tokens_in
        self.usage[source]["tokens_out"] += tokens_out
        self.usage[source]["requests"] += 1

    def get_cost(self, source: SourceSystem) -> float:
        data = self.usage[source]
        input_cost = (data["tokens_in"] / 1_000_000) * self.COST_PER_MILLION_INPUT
        output_cost = (data["tokens_out"] / 1_000_000) * self.COST_PER_MILLION_OUTPUT
        return input_cost + output_cost

    def get_report(self) -> dict:
        report = {}
        for source in SourceSystem:
            data = self.usage[source]
            report[source.value] = {
                "requests": data["requests"],
                "tokens_in": data["tokens_in"],
                "tokens_out": data["tokens_out"],
                "cost_usd": self.get_cost(source),
            }
        report["period_start"] = self.reset_time.isoformat()
        report["period_end"] = datetime.now().isoformat()
        return report
```

---

## 6. vLLM Integration Strategy

### 6.1 vLLM vs Ollama Comparison

| Feature | Ollama | vLLM | Winner |
|---------|--------|------|--------|
| **Throughput** | 41 TPS | 793 TPS | **vLLM (19x)** |
| **P99 Latency** | 673ms | 80ms | **vLLM (8x)** |
| **Concurrent Users** | 4 | 50+ | **vLLM** |
| **Model Format** | GGUF | HuggingFace | Ollama |
| **Setup Complexity** | Simple | Complex | Ollama |
| **Prefix Caching** | Limited | Native | **vLLM** |
| **Speculative Decoding** | No | Yes | **vLLM** |
| **Memory Efficiency** | Good | Excellent | **vLLM** |

### 6.2 Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM GATEWAY                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Request Router                                                  │
│       │                                                          │
│       ├──── [Simple requests] ──────► Ollama (fast start)       │
│       │     (< 4K context, quick models)                        │
│       │                                                          │
│       ├──── [Complex requests] ─────► vLLM (high throughput)    │
│       │     (long context, reasoning models)                    │
│       │                                                          │
│       └──── [Embedding requests] ──► Ollama (dedicated)         │
│             (mxbai-embed-large, bge-m3)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 Model Routing Rules

```python
# gateway/routing_rules.py

def get_backend(request: GatewayRequest, config: GatewayConfig) -> str:
    """Determine which backend to use for a request."""

    # Always use Ollama for embeddings
    if request.model in ["mxbai-embed-large", "bge-m3", "nomic-embed-text"]:
        return "ollama"

    # Use Ollama for streaming requests (lower latency start)
    if request.stream and request.context_window < 8192:
        return "ollama"

    # Use vLLM for:
    # - Long context (> 16K)
    # - Reasoning models (deepseek-r1)
    # - High-throughput batch requests
    if request.context_window > 16384:
        return "vllm"

    if request.model.startswith("deepseek"):
        return "vllm"

    if request.source_system == SourceSystem.PDF_TOOLS:  # Batch processing
        return "vllm"

    # Default: Use configured primary
    return config.primary_backend
```

### 6.4 vLLM Configuration

```yaml
# docker-compose.vllm.yml

services:
  vllm:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    command:
      - --model=Qwen/Qwen2.5-7B-Instruct
      - --port=8000
      - --tensor-parallel-size=1
      - --gpu-memory-utilization=0.85
      - --max-model-len=32768
      - --enable-prefix-caching
      - --enable-chunked-prefill
      - --max-num-seqs=64
      - --disable-log-requests
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 6.5 HuggingFace Model Equivalents

| Ollama Model | HuggingFace Equivalent | Notes |
|--------------|------------------------|-------|
| qwen3:8b | Qwen/Qwen2.5-7B-Instruct | Direct replacement |
| deepseek-r1:14b-qwen-distill-q8_0 | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | May need AWQ |
| gemma3:4b | google/gemma-2-2b-it | Smaller alternative |
| llama3.2:3b | meta-llama/Llama-3.2-3B-Instruct | Exact match |

---

## 7. Migration Phases

### Phase 1: Gateway Foundation (Week 1-2)

**Goal**: Implement basic gateway routing without changing existing behavior.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 1: GATEWAY FOUNDATION                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [memOS] ──► [Gateway:8100] ──► [Ollama:11434]                  │
│  [Android] ────────────────────► [Ollama:11434] (unchanged)     │
│                                                                  │
│  Features:                                                       │
│  - API translation (Ollama native ↔ OpenAI)                     │
│  - Request tracking and logging                                  │
│  - Basic metrics (latency, tokens)                              │
│  - Health checks                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Create gateway FastAPI application
- [ ] Implement API translator
- [ ] Implement request tracking
- [ ] Add Prometheus metrics
- [ ] Create health check endpoints
- [ ] Update memOS to use gateway (feature flag)

### Phase 2: Queue & VRAM Management (Week 3-4)

**Goal**: Add intelligent scheduling and VRAM tracking.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 2: QUEUE MANAGEMENT                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [memOS] ──► [Gateway:8100] ──┬──► [Queue] ──► [Ollama:11434]  │
│  [Android] ──► [Gateway:8100] ┘                                 │
│                                                                  │
│  Features:                                                       │
│  - Priority queue scheduling                                     │
│  - VRAM utilization tracking                                     │
│  - Per-system quotas                                             │
│  - Request preemption                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Implement priority queue manager
- [ ] Implement VRAM manager
- [ ] Add per-system quotas
- [ ] Create queue monitoring dashboard
- [ ] Update Android to use gateway

### Phase 3: vLLM Integration (Week 5-6)

**Goal**: Add vLLM as secondary backend with shadow mode.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 3: VLLM SHADOW MODE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Gateway] ──┬──► [Ollama:11434] (primary, returns response)    │
│              │                                                   │
│              └──► [vLLM:8000] (shadow, logged only)             │
│                                                                  │
│  Features:                                                       │
│  - vLLM backend adapter                                          │
│  - Shadow mode (compare outputs)                                 │
│  - Quality comparison metrics                                    │
│  - Latency comparison metrics                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Deploy vLLM container
- [ ] Implement vLLM backend adapter
- [ ] Add shadow mode routing
- [ ] Create comparison dashboard
- [ ] Download HuggingFace models

### Phase 4: Traffic Splitting (Week 7-8)

**Goal**: Gradually shift traffic to vLLM.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 4: TRAFFIC SPLITTING                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Week 7: 10% vLLM, 90% Ollama                                   │
│  Week 8: 30% vLLM, 70% Ollama                                   │
│                                                                  │
│  Criteria for increase:                                          │
│  - Error rate < 1%                                               │
│  - P95 latency <= Ollama baseline                               │
│  - Quality score >= Ollama baseline                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Implement traffic splitting
- [ ] Create gradual rollout controls
- [ ] Set up automated rollback triggers
- [ ] Monitor quality metrics
- [ ] Document rollback procedures

### Phase 5: Full Migration (Week 9-10)

**Goal**: vLLM as primary, Ollama as fallback.

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE 5: VLLM PRIMARY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Gateway] ──┬──► [vLLM:8000] (primary, 90%)                    │
│              │                                                   │
│              └──► [Ollama:11434] (fallback, embeddings, 10%)    │
│                                                                  │
│  Features:                                                       │
│  - vLLM as primary for generation                               │
│  - Ollama for embeddings only                                   │
│  - Automatic failover                                            │
│  - Cost optimization                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Set vLLM as primary
- [ ] Configure Ollama as fallback
- [ ] Optimize model loading
- [ ] Performance tuning
- [ ] Documentation update

---

## 8. Monitoring & Observability

### 8.1 Prometheus Metrics

```python
# gateway/metrics.py

from prometheus_client import Counter, Histogram, Gauge, Info

# Request metrics
requests_total = Counter(
    "llm_gateway_requests_total",
    "Total LLM gateway requests",
    ["source_system", "model", "backend", "status"]
)

request_latency = Histogram(
    "llm_gateway_request_latency_seconds",
    "Request latency in seconds",
    ["source_system", "model", "backend"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]
)

ttft = Histogram(
    "llm_gateway_ttft_seconds",
    "Time to first token",
    ["model", "backend"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5]
)

# Token metrics
tokens_total = Counter(
    "llm_gateway_tokens_total",
    "Total tokens processed",
    ["source_system", "model", "direction"]  # direction: input/output
)

# Queue metrics
queue_depth = Gauge(
    "llm_gateway_queue_depth",
    "Current queue depth",
    ["priority"]
)

active_requests = Gauge(
    "llm_gateway_active_requests",
    "Currently processing requests",
    ["backend"]
)

# VRAM metrics
vram_usage = Gauge(
    "llm_gateway_vram_usage_bytes",
    "GPU VRAM usage",
    ["gpu_id"]
)

vram_utilization = Gauge(
    "llm_gateway_vram_utilization",
    "GPU VRAM utilization (0-1)",
    ["gpu_id"]
)

# Model metrics
model_loaded = Gauge(
    "llm_gateway_model_loaded",
    "Model loading state (1=loaded)",
    ["model", "backend"]
)

# Backend health
backend_health = Gauge(
    "llm_gateway_backend_health",
    "Backend health status (1=healthy)",
    ["backend"]
)
```

### 8.2 Grafana Dashboards

**Dashboard: LLM Gateway Overview**
- Requests per second by source system
- Latency percentiles (p50, p95, p99)
- Token throughput
- Queue depth
- VRAM utilization
- Error rate

**Dashboard: Model Performance**
- Latency by model
- Throughput by model
- Cache hit rate
- Model load time

**Dashboard: Cost Analysis**
- Tokens by source system
- Estimated cost by source system
- Usage trends

### 8.3 Alerting Rules

```yaml
# alerting/rules.yml

groups:
  - name: llm_gateway
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(llm_gateway_request_latency_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency exceeds 30s"

      - alert: HighErrorRate
        expr: rate(llm_gateway_requests_total{status="error"}[5m]) / rate(llm_gateway_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate exceeds 5%"

      - alert: HighVRAMUtilization
        expr: llm_gateway_vram_utilization > 0.95
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "VRAM utilization exceeds 95%"

      - alert: QueueBacklog
        expr: llm_gateway_queue_depth > 50
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Queue depth exceeds 50 requests"

      - alert: BackendUnhealthy
        expr: llm_gateway_backend_health == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Backend {{ $labels.backend }} is unhealthy"
```

---

## 9. Risk Mitigation

### 9.1 Rollback Triggers

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Error rate | > 5% | Automatic rollback to Ollama |
| P95 latency | > 60s | Alert + manual review |
| VRAM OOM | Any | Automatic model eviction |
| Backend unhealthy | 30s | Failover to alternate |

### 9.2 Rollback Procedure

```bash
# Immediate rollback via environment variable
export LLM_GATEWAY_PRIMARY_BACKEND=ollama
export LLM_GATEWAY_VLLM_ENABLED=false

# Restart gateway
docker-compose restart gateway

# Or via API
curl -X POST http://localhost:8100/admin/rollback \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"target_backend": "ollama", "reason": "High error rate"}'
```

### 9.3 Data Loss Prevention

- All requests logged to PostgreSQL before processing
- Retry queue for failed requests
- Dead letter queue for unrecoverable failures
- Audit trail for all configuration changes

### 9.4 Go/No-Go Criteria

**Phase 3 (vLLM Shadow Mode)**:
- [ ] vLLM output quality matches Ollama (manual review of 50 samples)
- [ ] vLLM latency within 2x of Ollama
- [ ] No OOM errors in 24 hours
- [ ] All golden queries pass

**Phase 4 (Traffic Splitting)**:
- [ ] Shadow mode quality >= 95% match
- [ ] P95 latency < 30s
- [ ] Error rate < 1%
- [ ] VRAM stable over 24 hours

**Phase 5 (Full Migration)**:
- [ ] 70% traffic on vLLM for 7 days
- [ ] Error rate < 0.5%
- [ ] P95 latency < 20s
- [ ] No manual rollbacks in 7 days

---

## 10. Implementation Timeline

### Gantt Chart

```
Week     1    2    3    4    5    6    7    8    9   10
         |    |    |    |    |    |    |    |    |    |
Phase 1  ████████
Gateway
Foundation

Phase 2       ████████
Queue &
VRAM Mgmt

Phase 3            ████████
vLLM
Integration

Phase 4                 ████████
Traffic
Splitting

Phase 5                      ████████
Full
Migration

Testing  ─────────────────────────────────────────────
         Continuous Integration Testing Throughout
```

### Weekly Milestones

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| 1 | Gateway foundation | FastAPI app, API translator, basic metrics |
| 2 | Request tracking | Logging, per-system tracking, health checks |
| 3 | Queue manager | Priority queue, VRAM manager, quotas |
| 4 | Queue optimization | Preemption, monitoring dashboard |
| 5 | vLLM setup | Container deployment, backend adapter |
| 6 | Shadow mode | Comparison metrics, quality validation |
| 7 | 10% traffic split | Gradual rollout, monitoring |
| 8 | 30% traffic split | Performance tuning, rollback testing |
| 9 | Primary switch | vLLM as primary, Ollama fallback |
| 10 | Optimization | Cost analysis, documentation, handoff |

### Resource Requirements

| Resource | Allocation | Notes |
|----------|------------|-------|
| **Developer** | 0.5 FTE | Part-time during implementation |
| **GPU** | 1x TITAN RTX 24GB | Existing hardware |
| **Storage** | 100GB SSD | HuggingFace models |
| **Redis** | Existing | Queue management |
| **PostgreSQL** | Existing | Request logging |

---

## Appendix A: API Endpoint Reference

### Gateway Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/completions` | POST | OpenAI-compatible completion |
| `/api/generate` | POST | Ollama-compatible generation |
| `/api/chat` | POST | Ollama-compatible chat |
| `/health` | GET | Gateway health check |
| `/metrics` | GET | Prometheus metrics |
| `/admin/stats` | GET | Detailed statistics |
| `/admin/queue` | GET | Queue status |
| `/admin/backends` | GET | Backend status |
| `/admin/rollback` | POST | Trigger rollback |

### Example Requests

```bash
# OpenAI-compatible chat (Android)
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Source-System: android" \
  -d '{
    "model": "qwen3:8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7
  }'

# Ollama-compatible generate (memOS)
curl -X POST http://localhost:8100/api/generate \
  -H "Content-Type: application/json" \
  -H "X-Source-System: memos" \
  -d '{
    "model": "qwen3:8b",
    "prompt": "Analyze this query...",
    "options": {"temperature": 0.7, "num_ctx": 32768}
  }'
```

---

## Appendix B: Configuration Reference

```python
# gateway/config.py

from pydantic_settings import BaseSettings

class GatewayConfig(BaseSettings):
    """Gateway configuration."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8100
    workers: int = 4

    # Backends
    ollama_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"
    primary_backend: str = "ollama"  # "ollama" or "vllm"
    vllm_enabled: bool = False

    # Traffic splitting
    vllm_traffic_percentage: float = 0.0  # 0.0-1.0
    shadow_mode_enabled: bool = False

    # Queue
    max_concurrent_requests: int = 8
    max_queue_depth: int = 100
    request_timeout: int = 600  # seconds

    # VRAM
    target_vram_utilization: float = 0.85

    # Quotas
    default_requests_per_minute: int = 60
    default_tokens_per_minute: int = 100_000

    # Monitoring
    metrics_enabled: bool = True
    request_logging_enabled: bool = True

    # Database (for request logging)
    database_url: str = "postgresql://..."

    class Config:
        env_prefix = "LLM_GATEWAY_"
```

---

## Appendix C: Research Sources

1. **vLLM Documentation**: https://docs.vllm.ai/
2. **Ollama Documentation**: https://ollama.com/docs
3. **Local LLM Hosting Guide 2025**: https://medium.com/@rosgluk/local-llm-hosting-complete-2025-guide
4. **Ollama vs vLLM Benchmark**: https://developers.redhat.com/articles/2025/08/08/ollama-vs-vllm
5. **vLLM Production Stack**: https://github.com/vllm-project/production-stack
6. **LiteLLM (Unified Router)**: https://docs.litellm.ai/
7. **KV Cache Optimization**: See `agentic/KV_CACHE_IMPLEMENTATION_PLAN.md`
8. **SGLang (Alternative)**: https://docs.sglang.ai/

---

*Document Version 1.0 - 2026-01-01*
*Authors: Claude Code + Recovery Bot Engineering Team*
