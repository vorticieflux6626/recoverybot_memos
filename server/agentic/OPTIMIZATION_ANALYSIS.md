# Agentic Search Optimization Analysis

> **Updated**: 2025-12-30 | **Parent**: [AGENTIC_OVERVIEW.md](./AGENTIC_OVERVIEW.md) | **Status**: Complete (Phase 1-2)

### Test Results Summary (Updated 2025-12-26)

| Test Query | Before | After Phase 1 | Improvement |
|------------|--------|---------------|-------------|
| K8s OOMKilled Debugging | 133s | 115.9s | **-12.8%** |
| Homeless Veterans Resources | N/A | 87.6s | Baseline |
| PostgreSQL vs MongoDB | N/A | 148.7s | (complex) |
| Alcohol Relapse Coping | N/A | 103.6s | Warm cache |

| Metric | Before Optimization | After Phase 1 | Change |
|--------|--------------------|--------------------|--------|
| Complex Query Time | 133s | 115.9s | **-17s (-12.8%)** |
| Coverage Evaluation | 21s | 11s | **-48%** |
| Coverage Checks | 2 | 1 | -1 |
| Refinement Rounds | 2 max | 1 max | -1 |
| Model Persistence | 5m | 30m | +25m |

### Bottleneck Analysis (Measured)

```
Total Time: ~116s breakdown (MEASURED via server logs):
├── Query Analysis (qwen3:8b)        ~8s
├── Search Planning (qwen3:8b)       ~6s
├── Web Search (DuckDuckGo)          ~3-7s (3 queries)
├── URL Evaluation (qwen3:8b)        ~6s
├── Web Scraping                     ~9-10s
├── Coverage Evaluation (gemma3:4b)  ~11s  ← OPTIMIZED from 21s
├── Refinement Search                ~5s
└── THINKING MODEL SYNTHESIS         ~50-70s ← MAIN BOTTLENECK
    (deepseek-r1:14b-qwen-distill)

With warm model cache: Synthesis drops to ~40-50s
```

### Implemented Optimizations (December 2025)

1. **Reduced Refinement Iterations**: 2 → 1
   - Saves one LLM call (~15s) and one search round (~5s)
   - Trade-off: May miss some edge case information

2. **Keep-Alive for Thinking Model**
   - `keep_alive: "30m"` keeps model in VRAM (increased from 5m)
   - Subsequent queries benefit from cached model weights
   - **Measured**: ~10-15% reduction on warm queries (103s vs 118s)

6. **Coverage Evaluation Model Optimization** (NEW - 2025-12-26)
   - Changed from `qwen3:8b` to `gemma3:4b` (50% smaller model)
   - Reduced context window from 32K to 16K tokens
   - Added `keep_alive: "10m"` for coverage model
   - **Measured**: 48% faster (21s → 11s)

3. **Chain-of-Draft Prompting** (NEW)
   - Added `CHAIN_OF_DRAFT_INSTRUCTION` to synthesis prompts
   - Reduces thinking tokens by up to 80%
   - Expected: 40-60s reduction in synthesis time
   - Source: [Helicone Thinking Model Prompts](https://www.helicone.ai/blog/prompt-thinking-models)

4. **Validated DeepSeek R1 Parameters** (NEW)
   - `temperature: 0.6` - prevents repetition, maintains coherence
   - `top_p: 0.95` - good diversity while filtering improbable tokens
   - Source: [DeepSeek API Documentation](https://api-docs.deepseek.com/quick_start/parameter_settings)

5. **Ollama Environment Configuration** (NEW - `setup_ollama_optimization.sh`)
   - `OLLAMA_KV_CACHE_TYPE=q8_0` - 50% VRAM reduction
   - `OLLAMA_FLASH_ATTENTION=1` - faster attention computation
   - `OLLAMA_KEEP_ALIVE=30m` - longer model persistence
   - `OLLAMA_NUM_PARALLEL=2` - parallel request handling

**See `KV_CACHE_IMPLEMENTATION_PLAN.md` for comprehensive optimization roadmap.**

### Research Findings: MemOS KV Cache Architecture

Based on the [MemOS/MemCube architecture](https://arxiv.org/abs/2501.09136), there are significant optimization opportunities:

#### MemOS Three-Tier Memory Model
```
┌─────────────────────────────────────────────────────────────────┐
│                    MemCube Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────┐ │
│  │  PLAINTEXT      │   │  ACTIVATION     │   │  PARAMETRIC   │ │
│  │  MEMORY         │   │  MEMORY         │   │  MEMORY       │ │
│  │  (Hard Drive)   │   │  (RAM/KV Cache) │   │  (CPU Cache)  │ │
│  ├─────────────────┤   ├─────────────────┤   ├───────────────┤ │
│  │ • Text/docs     │   │ • KV-cache      │   │ • LoRA weights│ │
│  │ • Knowledge     │   │ • Hidden states │   │ • Fine-tuned  │ │
│  │   graphs        │   │ • Attention     │   │   parameters  │ │
│  │ • Prompt        │   │   patterns      │   │               │ │
│  │   templates     │   │                 │   │               │ │
│  └─────────────────┘   └─────────────────┘   └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

#### Published Performance Improvements

| System | Context Length | TTFT Improvement |
|--------|---------------|------------------|
| MemOS + Qwen3-8B | 8K tokens | 8.3x faster |
| MemOS + Qwen2.5-72B | 8K tokens | 10x faster |
| vLLM + prefix caching | 10K tokens | 7x faster |
| LMCache + vLLM | Multi-turn QA | 3-10x faster |

### Optimization Roadmap

#### Phase 1: Ollama-Native Optimizations (No Infrastructure Change) ✅ COMPLETE
- [x] Reduce refinement iterations (2→1)
- [x] Add keep_alive for model persistence (30m for thinking, 10m for coverage)
- [x] Chain-of-Draft prompting for thinking models (50-80% token reduction)
- [x] Validated DeepSeek R1 sampling parameters (temperature=0.6, top_p=0.95)
- [x] Ollama environment optimization script (KV quantization, Flash Attention)
- [x] Use smaller model (gemma3:4b) for coverage evaluation - **48% faster**
- [ ] Batch multiple LLM calls where possible (future enhancement)

#### Phase 2: Enhanced Caching ✅ COMPLETE
- [x] Content hash cache for scraped pages (avoid re-scraping) - **30% hit rate**
- [x] Result cache with TTL for identical queries - **99.9% speedup**
- [x] Query embedding cache for semantic deduplication - **0.919 similarity threshold**
- [x] Scratchpad persistence for session continuity - **memOS integration**

#### Phase 3: vLLM Migration (Significant Improvement)
- [ ] Replace Ollama with vLLM for thinking model
- [ ] Enable KV cache prefix sharing
- [ ] Implement prompt caching for common patterns
- [ ] Expected: 40-60% TTFT reduction

#### Phase 4: Full MemOS Integration
- [ ] Deploy MemOS alongside memOS server
- [ ] Implement KVCacheMemory for activation storage
- [ ] Create shared MemCube for multi-agent coordination
- [ ] Enable MemScheduler for automatic tier promotion
- [ ] Expected: 80-94% TTFT reduction

### Architecture Comparison

#### Current Architecture
```
User Query → Analyzer → Searcher → Scraper → Synthesizer → Response
     ↓           ↓          ↓          ↓           ↓
  (LLM call) (LLM call) (HTTP)   (HTTP)    (LLM call)

Each LLM call re-processes:
- System prompt (~1000 tokens)
- Query context (~500 tokens)
- Scraped content (~4000 tokens)
= ~5500 tokens re-tokenized per call
```

#### MemOS-Optimized Architecture
```
User Query → [KV Cache Check] → Activation Memory → Response
                   ↓
           [Cache Miss]
                   ↓
    Analyzer → Searcher → Scraper → Synthesizer
        ↓                                ↓
   KV Cache                         KV Cache
   (extracted)                      (injected)

With KV caching:
- System prompt: Pre-computed KV (~0 tokens to process)
- Query context: Incrementally added
- Common patterns: Cached activations
= ~1000-2000 tokens actually processed
```

### Key Insights from Literature

1. **Agentic RAG Patterns** (arxiv:2501.09136)
   - Reflection: Agents evaluate and refine outputs
   - Planning: Multi-step task management
   - Tool Use: Leveraging external resources
   - Multi-agent Collaboration: Coordinated complex tasks

2. **Adaptive RAG** (LlamaIndex)
   - Simple queries: Skip retrieval
   - Moderate: Single-step retrieval
   - Complex: Multi-step with iterative refinement

3. **Plan Optimization** (VLDB 2025)
   - Rule-based plan rewriting inspired by DB query optimization
   - Predicate pushdown: Front-load constraints
   - Plan verification: Ensure transformed plans satisfy intent

### Phase 1 Results Summary

**Achieved:**
- 12.8% overall improvement on complex queries (133s → 116s)
- 48% faster coverage evaluation (21s → 11s)
- Warm cache queries ~15% faster (118s → 103s)

**Main Remaining Bottleneck:**
- DeepSeek R1 synthesis still takes 50-70s
- Further optimization requires vLLM or MemOS integration (Phase 3/4)

---

### Phase 2 Results Summary (2025-12-26)

**Implemented: Content Hash Cache + Semantic Deduplication**

| Test | Time | Details |
|------|------|---------|
| First query (cold) | 85.0s | 7 URLs cached, embedding stored |
| Semantically similar query | 1.3s | **0.919 similarity match** |
| Identical query | 0.1s | Full query cache hit |

**Key Metrics:**
- **Content cache savings**: ~15s reduction on overlapping queries
- **Query cache**: 99.9% improvement for identical queries (85s → 0.1s)
- **Semantic cache**: 98.5% speedup for similar queries (85s → 1.3s)
- **Similarity threshold**: 0.88 for search results

**New Features:**
- `agentic/content_cache.py` - SQLite-backed content + semantic cache
- Embedding-based query matching via `mxbai-embed-large`
- Scratchpad persistence to memOS PostgreSQL
- Updated `agentic/scraper.py` - Content cache integration
- Updated `agentic/scratchpad.py` - memOS persistence
- Updated `agentic/orchestrator.py` - Semantic cache lookup
- Updated `api/search.py` - Cache stats endpoint

**API Endpoints:**
- `GET /api/v1/search/cache/stats` - View cache statistics
- `DELETE /api/v1/search/cache` - Clear all caches

**Cache Configuration:**
- Content TTL: 1 hour (3600s)
- Query TTL: 15 minutes (900s)
- Semantic similarity threshold: 0.88

### Recommendations

1. **Completed** (Phase 1 + Phase 2)
   - [x] Deployed Ollama-native optimizations
   - [x] Benchmarked with multiple query types
   - [x] Implemented content and query caching
   - [x] Documented results

2. **Next Steps** (Future Enhancements)
   - Query embedding cache for semantic deduplication
   - Scratchpad persistence for session continuity
   - Batch LLM calls where possible
   - Expected: Additional 10-15% improvement

3. **Long-Term** (Q1 2026 - Phase 3/4)
   - vLLM migration for prefix caching (40-60% improvement)
   - Full MemOS integration (80-94% TTFT reduction)
   - Production deployment with monitoring

### Sources

- [Agentic RAG Survey](https://arxiv.org/abs/2501.09136)
- [MemOS GitHub](https://github.com/MemTensor/MemOS)
- [LlamaIndex Agentic Retrieval](https://www.llamaindex.ai/blog/rag-is-dead-long-live-agentic-retrieval)
- [MA-RAG: Multi-Agent RAG](https://arxiv.org/html/2511.11788)
- [vLLM KV Cache Optimization](https://docs.vllm.ai/en/latest/)
