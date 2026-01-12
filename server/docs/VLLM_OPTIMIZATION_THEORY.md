# vLLM High-Throughput Optimization Theory Report

**Date**: 2026-01-12
**Based on**: FANUC SRVO-062 query benchmark (195s, 73% confidence)
**Scope**: Agentic pipeline + VL scraping LLM call analysis

---

## Executive Summary

Analysis of the current Ollama-based pipeline reveals **10 LLM calls per search query** using **4 distinct models**. The sequential nature of Ollama requests creates a bottleneck where:

- **gemma3:4b** handles 60% of calls (6/10)
- **qwen3:8b** consumes 45% of LLM time (35.9s of 79.2s)
- Model loading/unloading overhead adds ~2-5s per model switch

**Theoretical speedup with vLLM**: **2.8x-4.2x** (79s → 19-28s for LLM operations)

---

## Current Pipeline LLM Call Analysis

### Per-Query Model Usage

| Model | Calls | Time (ms) | % Time | Phases |
|-------|-------|-----------|--------|--------|
| **gemma3:4b** | 6 | 33,734 | 42.6% | Analysis, Planning, URL Filter, Verify (×3) |
| **qwen3:8b** | 2 | 35,913 | 45.4% | HyDE, Synthesis |
| **qwen3:4b-instruct-q8_0** | 1 | 4,513 | 5.7% | CRAG Evaluation |
| **cogito:8b** | 1 | 5,018 | 6.3% | Self-Reflection |
| **TOTAL** | **10** | **79,178** | 100% | |

### VL Scraping (Additional)

| Model | Calls/Query | Time/Call | Usage Rate |
|-------|-------------|-----------|------------|
| qwen2.5-vl:7b | ~6 (of 20 URLs) | 8,000ms | ~30% of URLs |

With VL scraping: **+48,000ms** potential (6 × 8s)

---

## Ollama vs vLLM Architecture Comparison

### Current: Ollama (Sequential)

```
┌─────────────────────────────────────────────────────────────────┐
│                        OLLAMA SERVER                            │
│                                                                 │
│  Request 1 ─────► [Load gemma3:4b] ─────► Generate ─────►      │
│                          ↓                                      │
│  Request 2 ─────► [Unload/Load qwen3:8b] ─────► Generate ─────►│
│                          ↓                                      │
│  Request 3 ─────► [Keep qwen3:8b] ─────► Generate ─────►       │
│                          ↓                                      │
│  Request 4 ─────► [Unload/Load cogito:8b] ─────► Generate ─────►│
│                                                                 │
│  ⚠️ Model switching overhead: 2-5s per switch                   │
│  ⚠️ Sequential processing: No parallelism                       │
│  ⚠️ Single GPU utilization: ~60-70%                             │
└─────────────────────────────────────────────────────────────────┘

Timeline (79s LLM time):
[====gemma3:4b====][==qwen3:8b==][==qwen3:8b==][cogito][q8_0]
     33.7s              5s          30.9s       5s     4.5s
```

### Proposed: vLLM (Continuous Batching)

```
┌─────────────────────────────────────────────────────────────────┐
│                     vLLM INFERENCE SERVER                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              ALWAYS-LOADED MODEL POOL                     │  │
│  │                                                           │  │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │  │
│  │   │  gemma3:4b  │  │  qwen3:8b   │  │  cogito:8b  │     │  │
│  │   │   (3.2GB)   │  │   (5.6GB)   │  │   (4.5GB)   │     │  │
│  │   └─────────────┘  └─────────────┘  └─────────────┘     │  │
│  │                                                           │  │
│  │   Total VRAM: ~13.3GB (fits in 24GB GPU)                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              CONTINUOUS BATCHING ENGINE                   │  │
│  │                                                           │  │
│  │  Incoming requests → Dynamic batching → Parallel decode  │  │
│  │                                                           │  │
│  │  • Batch size: 8-32 requests                             │  │
│  │  • PagedAttention: Efficient KV cache                    │  │
│  │  • Speculative decoding: 2x token throughput             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ✅ No model loading overhead                                   │
│  ✅ Parallel inference across models                            │
│  ✅ GPU utilization: 90-95%                                     │
└─────────────────────────────────────────────────────────────────┘

Timeline (theoretical 19-28s):
[gemma3:4b──────────────────]  (parallel phases)
[qwen3:8b───────────────────]  (batched HyDE+Synth)
[cogito:8b──]                   (no load overhead)
[q8_0─]
```

---

## Optimization Strategies

### Strategy 1: Model Consolidation (Reduce Distinct Models)

**Current**: 4 distinct models
**Proposed**: 2-3 models

| Current Model | Proposed Replacement | Rationale |
|---------------|---------------------|-----------|
| gemma3:4b | qwen3:8b | Consolidate to single high-quality model |
| qwen3:4b-instruct-q8_0 | qwen3:8b | Same family, better quality |
| cogito:8b | qwen3:8b OR keep | cogito:8b has unique thinking capability |

**Benefit**: Fewer model switches, better batching opportunities

**Trade-off**: Slightly slower per-call for simple tasks (8B vs 4B)

### Strategy 2: Parallel Phase Execution

Currently sequential phases that could run in parallel:

```
CURRENT (Sequential):
[Analyze] → [Plan] → [Search] → [Scrape] → [Evaluate] → [Verify] → [Synthesize] → [Reflect]
   8.7s      3s        10s       145s        4.5s        16s         31s           5s

OPTIMIZED (Parallel where possible):
[Analyze + Plan]  → [Search] → [Scrape + Evaluate(batch)] → [Verify(batch)] → [Synthesize] → [Reflect]
     5s               10s              145s                     8s              25s            5s
                                       (parallel VL)
```

**Parallelizable Pairs**:
1. Query Analysis + Search Planning (same model, can batch)
2. URL Evaluation + Scraping (evaluation runs while scraping starts)
3. Verification claims (currently 3 calls → 1 batched call)

### Strategy 3: vLLM Continuous Batching

vLLM's key advantage is **continuous batching** where:

1. **No model loading**: Models stay in VRAM permanently
2. **Request batching**: Multiple requests processed together
3. **Speculative decoding**: Draft model generates candidates, main model verifies

**Configuration for our pipeline**:

```python
# vLLM server configuration
vllm_config = {
    "models": [
        {"name": "gemma3:4b", "tensor_parallel": 1, "gpu_memory_utilization": 0.25},
        {"name": "qwen3:8b", "tensor_parallel": 1, "gpu_memory_utilization": 0.35},
        {"name": "cogito:8b", "tensor_parallel": 1, "gpu_memory_utilization": 0.30},
    ],
    "max_batch_size": 16,
    "enable_speculative_decoding": True,
    "speculative_draft_model": "qwen3:4b",  # Fast draft for qwen3:8b
}
```

### Strategy 4: VL Model Batching

Current VL scraping is **per-URL sequential**. With vLLM:

```
CURRENT (20 URLs, 30% need VL = 6 VL calls):
[VL-1] → [VL-2] → [VL-3] → [VL-4] → [VL-5] → [VL-6]
  8s       8s       8s       8s       8s       8s    = 48s total

VLLM BATCHED:
[VL-1, VL-2, VL-3, VL-4, VL-5, VL-6]  (single batch)
              12-15s total              = 3x speedup
```

---

## Theoretical Speedup Calculations

### LLM Operations Only (79s → 19-28s)

| Optimization | Savings | Remaining |
|--------------|---------|-----------|
| **Baseline** | - | 79s |
| Eliminate model loading (4 switches × 3s) | -12s | 67s |
| Parallel analysis+planning | -6s | 61s |
| Batch verification (3→1 call) | -11s | 50s |
| Continuous batching efficiency (+40%) | -20s | 30s |
| Speculative decoding synthesis (+30%) | -9s | 21s |

**Theoretical LLM time**: **19-28s** (2.8x-4.2x speedup)

### Full Pipeline (195s → 85-110s)

| Phase | Current | Optimized | Savings |
|-------|---------|-----------|---------|
| LLM Operations | 79s | 21s | 58s |
| Scraping (VL batched) | 145s | 100s | 45s |
| Search/Network | 10s | 10s | 0s |
| Overhead | 5s | 3s | 2s |
| **TOTAL** | **195s** | **~100s** | **~49% faster** |

---

## Implementation Roadmap

### Phase 1: Model Consolidation (Low Risk)
**Effort**: 2-4 hours
**Speedup**: 10-15%

1. Consolidate gemma3:4b and qwen3:4b-instruct-q8_0 → qwen3:8b
2. Keep cogito:8b for self-reflection (unique capability)
3. Result: 2 models instead of 4

### Phase 2: Request Batching (Medium Risk)
**Effort**: 1-2 days
**Speedup**: 20-30%

1. Batch verification claims (3 calls → 1)
2. Parallel analyze + plan
3. Implement request queue with batching

### Phase 3: vLLM Migration (High Effort)
**Effort**: 1-2 weeks
**Speedup**: 40-60%

1. Deploy vLLM server alongside Ollama
2. Create abstraction layer (gateway) for model routing
3. Migrate high-throughput models to vLLM
4. Keep Ollama for development/fallback

### Phase 4: VL Batching (Medium Effort)
**Effort**: 3-5 days
**Speedup**: 15-20% (of scraping time)

1. Collect screenshots in parallel
2. Batch VL inference (requires vLLM with vision support OR separate VL server)
3. Async result collection

---

## VRAM Requirements

### Current (Ollama, Sequential Loading)
- Peak: ~8GB (one model at a time + overhead)
- Average: ~5GB

### Proposed (vLLM, All Models Loaded)
| Model | VRAM |
|-------|------|
| gemma3:4b | 3.2GB |
| qwen3:8b | 5.6GB |
| cogito:8b | 4.5GB |
| qwen2.5-vl:7b (optional) | 5.0GB |
| **Overhead** | 2.0GB |
| **Total** | **15.3-20.3GB** |

**Requirement**: 24GB GPU (RTX 4090, A6000, etc.)

### Multi-GPU Option
With 2× 12GB GPUs:
- GPU 0: gemma3:4b + cogito:8b (7.7GB)
- GPU 1: qwen3:8b + vl:7b (10.6GB)

---

## Benchmark Predictions

### After Full Optimization (vLLM + Batching)

| Metric | Current | Predicted | Change |
|--------|---------|-----------|--------|
| Total Time | 195s | 85-110s | **-44% to -56%** |
| LLM Time | 79s | 19-28s | **-65% to -76%** |
| Throughput | 0.3 queries/min | 0.7 queries/min | **+133%** |
| GPU Utilization | 60-70% | 90-95% | **+35%** |
| VRAM Usage | 5-8GB (peak) | 15-20GB (constant) | +150% |

### Cost-Benefit Analysis

| Optimization | Effort | Speedup | ROI |
|--------------|--------|---------|-----|
| Model Consolidation | 2h | 15% | ⭐⭐⭐⭐⭐ |
| Request Batching | 2d | 25% | ⭐⭐⭐⭐ |
| vLLM Migration | 2w | 50% | ⭐⭐⭐ |
| VL Batching | 5d | 15% | ⭐⭐ |

---

## Conclusion

The current Ollama-based pipeline has significant optimization potential through:

1. **Model consolidation**: Reduce 4 models to 2-3 (-15% time)
2. **Request batching**: Parallel/batched LLM calls (-25% time)
3. **vLLM migration**: Continuous batching, no loading (-50% time)
4. **VL batching**: Parallel screenshot analysis (-15% scraping time)

**Total theoretical improvement**: 44-56% faster pipeline (195s → 85-110s)

The highest ROI optimization is **model consolidation** followed by **request batching**, both achievable without major infrastructure changes. vLLM migration offers the largest speedup but requires dedicated GPU resources and infrastructure investment.

---

## Appendix: Model Call Sequence Diagram

```
Search Query: "FANUC SRVO-062 troubleshooting"
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: QUERY ANALYSIS                                         │
│ Model: gemma3:4b | Call #1 | 8.7s                               │
│ Input: Query text (500 tokens)                                  │
│ Output: Query type, complexity, search strategy (800 tokens)    │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: SEARCH PLANNING                                        │
│ Model: gemma3:4b | Call #2 | 3.0s                               │
│ Input: Analysis result (800 tokens)                             │
│ Output: Search plan, queries (400 tokens)                       │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: HYDE EXPANSION                                         │
│ Model: qwen3:8b | Call #3 | 5.0s                                │
│ Input: Query (300 tokens)                                       │
│ Output: Hypothetical document (500 tokens)                      │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
        [Web Search - 10s, no LLM]
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: CRAG EVALUATION                                        │
│ Model: qwen3:4b-instruct-q8_0 | Call #4 | 4.5s                  │
│ Input: Search results (3000 tokens)                             │
│ Output: Quality assessment (300 tokens)                         │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: URL RELEVANCE FILTER                                   │
│ Model: gemma3:4b | Call #5 | 5.7s                               │
│ Input: URLs + snippets (2000 tokens)                            │
│ Output: Filtered URLs (200 tokens)                              │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
        [Web Scraping - 145s, includes VL calls]
        ┌─────────────────────────────────────────┐
        │ VL Calls (30% of URLs = ~6 calls)       │
        │ Model: qwen2.5-vl:7b | 6 × 8s = 48s    │
        └─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: CONTENT VERIFICATION                                   │
│ Model: gemma3:4b | Calls #6-8 | 3 × 5.4s = 16.2s               │
│ Input: Claims to verify (1500 tokens each)                      │
│ Output: Verification results (400 tokens each)                  │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 7: SYNTHESIS                                              │
│ Model: qwen3:8b | Call #9 | 30.9s                               │
│ Input: All context (40000 tokens)                               │
│ Output: Synthesized answer (2000 tokens)                        │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 8: SELF-REFLECTION                                        │
│ Model: cogito:8b | Call #10 | 5.0s                              │
│ Input: Answer + sources (3000 tokens)                           │
│ Output: Quality assessment (500 tokens)                         │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
        [Response - Total: 195s]
```

---

## Implementation Path: LLM Gateway Sub-Project

> **Note**: The optimizations described in this report are planned for implementation via the **LLM Gateway** sub-project, not direct modification of the memOS agentic pipeline.

### Gateway Architecture (Port 8100)

The Gateway sub-project (`/home/sparkone/sdd/Recovery_Bot/gateway/`) provides a unified routing layer that will implement vLLM integration:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          LLM GATEWAY SERVICE                                  │
│                             Port 8100                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      REQUEST ROUTER                                      │ │
│  │  • API format detection (Ollama native vs OpenAI-compatible)            │ │
│  │  • Source system identification (memOS, Android, PDF Tools)             │ │
│  │  • Request batching and queueing                                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│              ┌─────────────────────┴─────────────────────┐                   │
│              ▼                                           ▼                    │
│  ┌───────────────────────┐                ┌───────────────────────┐          │
│  │   OLLAMA BACKEND      │                │    vLLM BACKEND       │          │
│  │   (Port 11434)        │                │    (Port 8000)        │          │
│  │   ✅ Current primary   │                │    🎯 Future target   │          │
│  │   • Sequential        │                │    • Continuous batch │          │
│  │   • Model swapping    │                │    • PagedAttention   │          │
│  └───────────────────────┘                └───────────────────────┘          │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Migration Strategy

When vLLM is deployed via the Gateway, the migration will be **transparent** to the agentic pipeline:

| Phase | Action | Impact on memOS |
|-------|--------|-----------------|
| **Phase 1** | Gateway intercepts all LLM requests | No code changes required |
| **Phase 2** | High-throughput models route to vLLM | Automatic speedup |
| **Phase 3** | Request batching enabled | Further optimization |
| **Phase 4** | Full vLLM migration (Ollama fallback) | Maximum performance |

### Gateway Integration Points

The Gateway will automatically apply the optimizations from this report:

```python
# Gateway routing logic (conceptual)
class ModelRouter:
    def route_request(self, request: GatewayRequest) -> Backend:
        model = request.model

        # High-throughput models → vLLM (when available)
        if model in ["qwen3:8b", "gemma3:4b"] and self.vllm_available:
            return Backend.VLLM

        # Vision models → Ollama (vLLM vision support limited)
        if "vl" in model or "vision" in model:
            return Backend.OLLAMA

        # Thinking models → Ollama (better streaming support)
        if model in ["cogito:8b", "deepseek-r1:8b"]:
            return Backend.OLLAMA

        # Default: Ollama
        return Backend.OLLAMA
```

### Benchmark-Optimized Model Routing Table

Based on the benchmarks from 2026-01-12, the Gateway will route as follows:

| Model | Backend | Rationale |
|-------|---------|-----------|
| `gemma3:4b` | **vLLM** | 6 calls/query, highest frequency |
| `qwen3:8b` | **vLLM** | Synthesis (30s), benefits from batching |
| `qwen3:4b-instruct-2507-q8_0` | **vLLM** | Fast evaluator, good batch candidate |
| `cogito:8b` | Ollama | Thinking model, streaming preferred |
| `qwen2.5-vl:7b` | Ollama | Vision model, vLLM support limited |

### Expected Timeline

| Milestone | Status | ETA |
|-----------|--------|-----|
| Gateway v0.1 (Ollama proxy) | ✅ Complete | - |
| Gateway v0.2 (vLLM backend) | 🔄 Planned | TBD |
| Request batching | 🔄 Planned | TBD |
| Full vLLM migration | 🔄 Planned | TBD |

### No Action Required for memOS

Once the Gateway implements vLLM, the speedups described in this report will be achieved **without any changes to the memOS agentic pipeline**. The pipeline currently calls Ollama directly via `http://localhost:11434`, which can be redirected to the Gateway (`http://localhost:8100`) via:

1. **Environment variable**: `OLLAMA_BASE_URL=http://localhost:8100/ollama`
2. **Config update**: `llm_config.yaml` → `ollama.url: http://localhost:8100/ollama`

The Gateway will handle all routing, batching, and backend selection transparently.

---

*Report generated from benchmark data: 2026-01-12*
*Gateway integration planned via: `/home/sparkone/sdd/Recovery_Bot/gateway/`*
