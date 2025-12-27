# KV Cache Optimization Implementation Plan

## Executive Summary

Based on comprehensive research from 5 specialized agents, this document provides a prioritized implementation plan for optimizing the memOS agentic search pipeline. The goal is to reduce the current ~133s query time (with 75s spent in thinking model synthesis) by 40-80%.

**Current Performance Baseline:**
- Total query time: ~133s
- Thinking model synthesis: ~75s (56% of total)
- Refinement rounds: 1 (reduced from 2)
- Sources consulted: ~18

**Target Performance:**
- Total query time: <60s (55% reduction)
- Thinking model synthesis: <30s (60% reduction)
- Token reduction: 50-80% via Chain-of-Draft

---

## Phase 1: Immediate Optimizations (1-2 Days)

### 1.1 Ollama Environment Configuration

Add these environment variables to the Ollama service:

```bash
# /etc/systemd/system/ollama.service.d/override.conf
# OR export in shell before starting Ollama

# KV Cache Quantization (50% VRAM reduction)
export OLLAMA_KV_CACHE_TYPE=q8_0  # NOT q4_0 for reasoning models

# Flash Attention (faster attention computation)
export OLLAMA_FLASH_ATTENTION=1

# Keep models loaded longer for subsequent queries
export OLLAMA_KEEP_ALIVE=30m

# Parallel request handling
export OLLAMA_NUM_PARALLEL=2

# Context window (balance between capability and VRAM)
export OLLAMA_CONTEXT_LENGTH=16384
```

**Expected Impact:** 10-20% TTFT reduction from Flash Attention + reduced VRAM pressure

### 1.2 Chain-of-Draft Prompting for DeepSeek R1

Update the synthesizer to use Chain-of-Draft prompting, which reduces thinking tokens by up to 80%:

```python
# In synthesizer.py, update the synthesis prompt

CHAIN_OF_DRAFT_INSTRUCTION = """
Think step by step, but only keep a minimum draft for each thinking step.
Provide your final answer with citations.
"""

# Prepend to all synthesis prompts
prompt = CHAIN_OF_DRAFT_INSTRUCTION + "\n\n" + base_prompt
```

**Expected Impact:** 50-80% reduction in thinking tokens (40-60s → 15-25s)

### 1.3 Optimal DeepSeek R1 Sampling Parameters

Update synthesizer.py with research-validated parameters:

```python
# In synthesizer.py, THINKING_MODELS configuration

THINKING_MODELS = {
    "deepseek-r1:14b-qwen-distill-q8_0": {
        "vram_gb": 15,
        "context_window": 16384,
        "max_tokens": 4096,
        "temperature": 0.6,  # VALIDATED: Prevents repetition
        "top_p": 0.95,       # VALIDATED: Good diversity
        "description": "Qwen-distilled 14B thinking model"
    }
}
```

### 1.4 Prompt Template Registry

Create a centralized prompt registry for maximum prefix cache hits:

```python
# New file: agentic/prompts.py

"""
Prompt Registry for Cache Optimization

DESIGN PRINCIPLES:
1. Static content at the beginning of every prompt (cacheable)
2. Consistent ordering: system → tools → context → query
3. Modular composition for different agent types
"""

# Core system prompt (shared across all agents ~1000 tokens)
CORE_SYSTEM_PREFIX = """You are an AI research assistant for Recovery Bot,
helping people in addiction recovery and underprivileged communities access
vital community services in Morehead, Kentucky and surrounding areas.

Core Values:
- Compassionate, non-judgmental support
- Evidence-based information with citations
- Clear, actionable guidance
- Privacy-first approach

Available Information Categories:
- Addiction recovery centers and treatment facilities
- Mental health services and counseling
- Homeless shelters and transitional housing
- Food pantries and nutrition programs
- Career centers and job training
- Healthcare facilities for underserved populations
"""

ANALYZER_SUFFIX = """
Your role: Query Analyzer
Determine if web search is needed and decompose complex queries.
Think step by step, but only keep a minimum draft for each thinking step.
Output JSON: {requires_search, query_type, decomposed_questions}
"""

SYNTHESIZER_SUFFIX = """
Your role: Information Synthesizer
Think step by step, but only keep a minimum draft for each thinking step.
Combine search results into comprehensive, cited responses.
ALWAYS include [Source N] citations for key facts.
"""

def build_prompt(agent_type: str, dynamic_context: str = "") -> str:
    """Build prompt with consistent prefix structure for cache optimization."""
    suffixes = {
        "analyzer": ANALYZER_SUFFIX,
        "synthesizer": SYNTHESIZER_SUFFIX,
        "planner": PLANNER_SUFFIX,
        "coverage": COVERAGE_SUFFIX,
    }
    return CORE_SYSTEM_PREFIX + suffixes.get(agent_type, "") + "\n\n" + dynamic_context
```

---

## Phase 2: Application-Level Caching (3-5 Days)

### 2.1 Query Deduplication Cache

Implement semantic caching to avoid redundant LLM calls:

```python
# New file: agentic/cache.py

import hashlib
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

class AgenticCache:
    """
    Multi-layer caching for agentic search pipeline.

    Layers:
    1. Exact match cache (hash-based)
    2. Semantic cache (embedding similarity)
    3. Tool result cache (web scrapes)
    """

    def __init__(self, ttl_seconds: int = 3600, similarity_threshold: float = 0.95):
        self._exact_cache: Dict[str, Dict] = {}
        self._semantic_cache: Dict[str, Dict] = {}
        self._tool_cache: Dict[str, Dict] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._similarity_threshold = similarity_threshold

    def get_cache_key(self, prompt: str, model: str) -> str:
        """Generate deterministic cache key."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get_exact(self, prompt: str, model: str) -> Optional[str]:
        """Check exact match cache."""
        key = self.get_cache_key(prompt, model)
        entry = self._exact_cache.get(key)
        if entry and datetime.now() - entry["timestamp"] < self._ttl:
            return entry["response"]
        return None

    def set_exact(self, prompt: str, model: str, response: str):
        """Store in exact match cache."""
        key = self.get_cache_key(prompt, model)
        self._exact_cache[key] = {
            "response": response,
            "timestamp": datetime.now()
        }

    async def get_semantic(self, query: str, embedder) -> Optional[Dict]:
        """Check semantic similarity cache."""
        query_embedding = await embedder.embed(query)

        for key, entry in self._semantic_cache.items():
            if datetime.now() - entry["timestamp"] > self._ttl:
                continue
            similarity = np.dot(query_embedding, entry["embedding"])
            if similarity >= self._similarity_threshold:
                return entry["result"]
        return None

    async def set_semantic(self, query: str, result: Dict, embedder):
        """Store with semantic embedding."""
        embedding = await embedder.embed(query)
        key = hashlib.md5(query.encode()).hexdigest()
        self._semantic_cache[key] = {
            "embedding": embedding,
            "result": result,
            "timestamp": datetime.now()
        }

    def get_scraped_content(self, url: str) -> Optional[Dict]:
        """Check scraped content cache."""
        entry = self._tool_cache.get(url)
        if entry and datetime.now() - entry["timestamp"] < self._ttl:
            return entry["content"]
        return None

    def set_scraped_content(self, url: str, content: Dict):
        """Cache scraped content."""
        self._tool_cache[url] = {
            "content": content,
            "timestamp": datetime.now()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Cache statistics for monitoring."""
        now = datetime.now()
        return {
            "exact_cache_entries": len(self._exact_cache),
            "semantic_cache_entries": len(self._semantic_cache),
            "tool_cache_entries": len(self._tool_cache),
            "valid_exact": sum(1 for e in self._exact_cache.values()
                              if now - e["timestamp"] < self._ttl),
            "valid_semantic": sum(1 for e in self._semantic_cache.values()
                                 if now - e["timestamp"] < self._ttl),
        }
```

### 2.2 Blackboard Architecture Integration

Enhance the existing scratchpad with blackboard patterns:

```python
# Update: agentic/scratchpad.py

class EnhancedScratchpad(AgenticScratchpad):
    """
    Enhanced scratchpad with blackboard architecture patterns.

    Based on LbMAS research showing 13-57% improvement over RAG.
    """

    def __init__(self):
        super().__init__()
        self.public_space: Dict[str, Any] = {}     # Shared across all agents
        self.private_spaces: Dict[str, Dict] = {}  # Per-agent private state
        self.kv_cache_refs: Dict[str, str] = {}    # References to cached KV states

    def write_public(self, agent_id: str, key: str, value: Any):
        """Write to shared space with provenance."""
        self.public_space[key] = {
            "value": value,
            "author": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }

    def read_context(self, agent_id: str) -> Dict:
        """Build context for agent including public + own private."""
        return {
            "public": self.public_space,
            "private": self.private_spaces.get(agent_id, {}),
            "findings": self.findings[-20:],  # Last 20 findings
            "history": self.search_history[-10:]  # Last 10 searches
        }

    def register_kv_cache(self, content_hash: str, cache_id: str):
        """Track KV cache references for reuse."""
        self.kv_cache_refs[content_hash] = cache_id

    def get_kv_cache_id(self, content_hash: str) -> Optional[str]:
        """Retrieve cached KV state if available."""
        return self.kv_cache_refs.get(content_hash)
```

### 2.3 Artifact-Based Communication

Implement Anthropic's artifact pattern for reducing token transfer:

```python
# New file: agentic/artifacts.py

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib

class ArtifactStore:
    """
    Filesystem-based artifact storage for agent communication.

    Based on Anthropic's multi-agent research system:
    - Subagents write to files, pass lightweight references
    - Prevents information loss in multi-stage processing
    - Reduces token overhead significantly
    """

    def __init__(self, base_path: str = "/tmp/memos_artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def store(
        self,
        session_id: str,
        artifact_type: str,
        content: Any,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store artifact and return lightweight reference."""
        artifact_id = hashlib.md5(
            f"{session_id}:{artifact_type}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        session_path = self.base_path / session_id
        session_path.mkdir(exist_ok=True)

        artifact = {
            "id": artifact_id,
            "type": artifact_type,
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }

        artifact_path = session_path / f"{artifact_id}.json"
        artifact_path.write_text(json.dumps(artifact, default=str))

        return artifact_id

    def retrieve(self, session_id: str, artifact_id: str) -> Optional[Dict]:
        """Retrieve artifact by reference."""
        artifact_path = self.base_path / session_id / f"{artifact_id}.json"
        if artifact_path.exists():
            return json.loads(artifact_path.read_text())
        return None

    def cleanup_session(self, session_id: str):
        """Clean up session artifacts."""
        session_path = self.base_path / session_id
        if session_path.exists():
            for file in session_path.glob("*.json"):
                file.unlink()
            session_path.rmdir()
```

---

## Phase 3: vLLM Migration (1-2 Weeks)

### 3.1 vLLM Server Setup

vLLM provides 40-60% TTFT improvement through automatic prefix caching:

```bash
# Install vLLM
pip install vllm

# Start vLLM server with prefix caching enabled
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --port 8000
```

### 3.2 vLLM Client Integration

```python
# New file: agentic/vllm_client.py

from openai import AsyncOpenAI
from typing import Optional

class VLLMClient:
    """
    vLLM client leveraging automatic prefix caching.

    Performance benefits:
    - Automatic cross-request prefix caching
    - PagedAttention for efficient memory
    - 40-60% TTFT reduction for cached prefixes
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="dummy"  # vLLM doesn't require auth
        )

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        temperature: float = 0.6,
        max_tokens: int = 4096
    ) -> str:
        """Generate with automatic prefix caching."""
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
```

### 3.3 LMCache Integration

For distributed KV caching with 3-10x speedup:

```bash
# Install LMCache
pip install lmcache

# Start vLLM with LMCache backend
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --enable-prefix-caching \
    --kv-connector-config lmcache_config.yaml
```

```yaml
# lmcache_config.yaml
backend: cpu  # or gpu, disk
max_size: 32GB
eviction_policy: lru
connector_type: lmcache
```

---

## Phase 4: Full MemOS Integration (2-4 Weeks)

### 4.1 Three-Tier Memory Architecture

Based on MemOS MemCube research (80-94% TTFT reduction):

```python
# New file: agentic/memory_tiers.py

"""
Three-tier memory model based on MemOS research (arxiv:2501.09136):

Tier 1: PLAINTEXT MEMORY (Cold)
  - Documents, knowledge graphs, prompt templates
  - Storage: PostgreSQL + Redis

Tier 2: ACTIVATION MEMORY (Warm)
  - KV cache, hidden states, precomputed attention
  - Storage: LMCache / GPU VRAM

Tier 3: PARAMETRIC MEMORY (Hot)
  - LoRA weights, fine-tuned parameters
  - Storage: Model weights in VRAM
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

class MemoryTierManager:
    """
    Manages memory promotion/demotion between tiers.

    Promotion: Plaintext → Activation (on frequent access)
    Demotion: Activation → Plaintext (on eviction)
    """

    def __init__(self, kv_cache_service, plaintext_storage):
        self.kv_cache = kv_cache_service
        self.plaintext = plaintext_storage
        self.access_counts: Dict[str, int] = {}
        self.promotion_threshold = 3  # Promote after 3 accesses

    async def get_context(
        self,
        content_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Retrieve context, checking warm tier first.
        Automatically promotes frequently accessed content.
        """
        # Check warm tier (activation memory)
        kv_cached = await self.kv_cache.get(content_id)
        if kv_cached:
            return {"source": "activation", "content": kv_cached}

        # Fall back to cold tier (plaintext)
        plaintext = await self.plaintext.get(content_id)
        if plaintext:
            # Track access for potential promotion
            self.access_counts[content_id] = self.access_counts.get(content_id, 0) + 1

            # Promote if threshold reached
            if self.access_counts[content_id] >= self.promotion_threshold:
                await self.promote_to_activation(content_id, plaintext)

            return {"source": "plaintext", "content": plaintext}

        return None

    async def promote_to_activation(self, content_id: str, content: str):
        """Precompute KV cache for frequently accessed content."""
        cache_id = await self.kv_cache.precompute(content)
        self.kv_cache.register(content_id, cache_id)

    async def store(
        self,
        content_id: str,
        content: str,
        tier: str = "plaintext"
    ):
        """Store content at specified tier."""
        if tier == "plaintext":
            await self.plaintext.store(content_id, content)
        elif tier == "activation":
            await self.promote_to_activation(content_id, content)
```

### 4.2 KV Cache Service Wrapper

```python
# New file: agentic/kv_cache_service.py

"""
KV Cache service wrapping inference engine capabilities.

Supports:
- Ollama (via keep_alive and prompt caching)
- vLLM (via automatic prefix caching)
- Future: Direct MemOS integration
"""

from typing import Optional, Dict
import hashlib

class KVCacheService:
    """
    Unified KV cache interface across inference backends.
    """

    def __init__(self, backend: str = "ollama"):
        self.backend = backend
        self.cache_registry: Dict[str, str] = {}  # content_hash → cache_id

    async def precompute(self, content: str) -> str:
        """
        Precompute KV cache for content.

        For Ollama: Make minimal inference call to warm cache
        For vLLM: Leverage automatic prefix caching
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]

        if self.backend == "ollama":
            # Ollama: Use keep_alive to persist model + minimal inference
            # The KV cache is implicitly created on first use
            return content_hash

        elif self.backend == "vllm":
            # vLLM: Automatic prefix caching handles this
            return content_hash

        return content_hash

    async def get(self, content_id: str) -> Optional[str]:
        """Check if KV cache exists for content."""
        return self.cache_registry.get(content_id)

    def register(self, content_id: str, cache_id: str):
        """Register content with its cache ID."""
        self.cache_registry[content_id] = cache_id
```

---

## Implementation Priority Matrix

| Priority | Task | Impact | Effort | Dependencies |
|----------|------|--------|--------|--------------|
| **P0** | Ollama environment config | 10-20% TTFT | 1 hour | None |
| **P0** | Chain-of-Draft prompting | 50-80% tokens | 2 hours | None |
| **P0** | DeepSeek R1 sampling params | Quality + speed | 30 min | None |
| **P1** | Prompt template registry | Cache hits | 1 day | None |
| **P1** | Application-level caching | 20-30% speedup | 2 days | None |
| **P1** | Blackboard enhancement | 13-57% over RAG | 2 days | Scratchpad |
| **P2** | vLLM migration | 40-60% TTFT | 1 week | Testing env |
| **P2** | LMCache integration | 3-10x speedup | 3 days | vLLM |
| **P3** | Full MemOS integration | 80-94% TTFT | 2-4 weeks | vLLM, LMCache |

---

## Expected Results

### After Phase 1 (Immediate)
- Query time: 133s → ~80s (40% reduction)
- Thinking tokens: 75s worth → ~20s worth (75% reduction)
- VRAM usage: 15GB → 10GB (33% reduction)

### After Phase 2 (1 Week)
- Query time: ~80s → ~50s (37% additional reduction)
- Cache hit rate: 0% → 30-50%
- Redundant LLM calls: eliminated

### After Phase 3 (2-3 Weeks)
- Query time: ~50s → ~30s (40% additional reduction)
- Automatic prefix caching: enabled
- Distributed caching: available

### After Phase 4 (4-6 Weeks)
- Query time: ~30s → <20s (33% additional reduction)
- Three-tier memory: operational
- Context reuse: maximized

---

## Monitoring Metrics

```python
# Add to orchestrator.py

class PerformanceMetrics:
    """Track optimization effectiveness."""

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_ttft_ms": 0,
            "avg_synthesis_ms": 0,
            "thinking_tokens_saved": 0,
        }

    def record_query(
        self,
        ttft_ms: float,
        synthesis_ms: float,
        cache_hit: bool,
        tokens_used: int,
        tokens_saved: int
    ):
        n = self.metrics["total_queries"]
        self.metrics["total_queries"] = n + 1

        if cache_hit:
            self.metrics["cache_hits"] += 1
        else:
            self.metrics["cache_misses"] += 1

        # Rolling average
        self.metrics["avg_ttft_ms"] = (
            (self.metrics["avg_ttft_ms"] * n + ttft_ms) / (n + 1)
        )
        self.metrics["avg_synthesis_ms"] = (
            (self.metrics["avg_synthesis_ms"] * n + synthesis_ms) / (n + 1)
        )
        self.metrics["thinking_tokens_saved"] += tokens_saved

    def get_summary(self) -> Dict:
        total = self.metrics["total_queries"]
        hits = self.metrics["cache_hits"]
        return {
            **self.metrics,
            "cache_hit_rate": hits / max(total, 1),
            "estimated_time_saved_ms": self.metrics["thinking_tokens_saved"] * 50  # ~50ms per token
        }
```

---

## Sources & References

### Research Papers
- [MemOS: Memory OS for AI Systems](https://arxiv.org/abs/2501.09136)
- [LbMAS: Blackboard Multi-Agent Systems](https://arxiv.org/abs/2507.01701)
- [DroidSpeak: Cross-LLM KV Cache Sharing](https://arxiv.org/abs/2411.02820)
- [Prompt Cache: Modular Attention Reuse](https://arxiv.org/abs/2311.04934)
- [ThinKV: Thought-Adaptive KV Compression](https://arxiv.org/abs/2510.01290)

### Documentation
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
- [LMCache Technical Report](https://lmcache.ai/tech_report.pdf)
- [DeepSeek R1 API Parameters](https://api-docs.deepseek.com/quick_start/parameter_settings)

### Industry Resources
- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Ollama KV Cache Quantization](https://mitjamartini.com/posts/ollama-kv-cache-quantization/)
- [Helicone Thinking Model Prompting](https://www.helicone.ai/blog/prompt-thinking-models)
