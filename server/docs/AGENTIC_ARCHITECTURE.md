# Agentic Search Architecture

> **Updated**: 2026-01-02 | **Status**: Reference Documentation | **Parent**: [memOS CLAUDE.md](../../CLAUDE.md)

This document contains detailed architecture documentation for the memOS agentic search system.

---

## Overview

The agentic search system implements a **ReAct (Reasoning + Acting)** pattern for intelligent web search and context injection. This enables multi-step reasoning, query decomposition, and verification before injecting search results into the main LLM conversation.

### Recent Fixes (2025-12-25)
- **Fixed empty synthesis**: Increased `num_ctx` from 16K to 32K to accommodate large prompts
- **Added URL scraping to non-streaming endpoint**: Both `/agentic` and `/stream` now scrape content
- **Improved source citations**: Synthesis now includes `[Source X]` citations throughout
- **Enhanced logging**: Added prompt length and response length tracking for debugging

---

## Performance Optimizations (2025-12-26)

**IMPORTANT**: Before starting Ollama, apply the optimization configuration:

```bash
# Apply Ollama KV cache and performance optimizations
source /home/sparkone/sdd/Recovery_Bot/memOS/server/setup_ollama_optimization.sh
systemctl restart ollama  # or: pkill ollama && ollama serve
```

### Implemented Optimizations

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| **Chain-of-Draft Prompting** | 50-80% thinking token reduction | `synthesizer.py` - prepends CoD instruction for DeepSeek R1 |
| **DeepSeek R1 Parameters** | Improved reasoning quality | `temperature=0.6`, `top_p=0.95` (validated by DeepSeek) |
| **KV Cache Quantization** | 50% VRAM reduction | `OLLAMA_KV_CACHE_TYPE=q8_0` |
| **Flash Attention** | 10-20% faster attention | `OLLAMA_FLASH_ATTENTION=1` |
| **Model Persistence** | Faster subsequent queries | `OLLAMA_KEEP_ALIVE=30m` |
| **Reduced Refinements** | 20s saved per query | `max_scrape_refinements=1` |

### Performance Results

| Phase | Optimization | Impact |
|-------|--------------|--------|
| Phase 1 | Ollama-native optimizations | 12.8% faster (133s → 116s) |
| Phase 1 | Coverage evaluation model | 48% faster (21s → 11s) |
| Phase 2 | Content hash cache | 30% hit rate on similar queries |
| Phase 2 | Query result cache | 99.9% speedup on identical queries |
| Phase 2 | Semantic query cache | 98.5% speedup for similar queries (0.88+ similarity) |
| Phase 2 | Prompt template registry | Maximizes KV cache prefix hits |
| Phase 2 | Artifact-based communication | Reduces agent token transfer |
| Phase 2 | Performance metrics tracking | Real-time TTFT/cache/token monitoring |
| Phase 3 | TTL-based cache pinning | Prevents KV eviction during 3-90s tool calls |
| Phase 4 | KV cache service | Unified interface for cache warming |
| Phase 4 | Three-tier memory (MemOS) | Cold→warm auto-promotion (80-94% TTFT target) |
| Phase 4 | System prompt pre-warming | Near-zero TTFT for common prompts |

### Key Files

- `agentic/synthesizer.py` - Chain-of-Draft prompting, validated sampling parameters
- `agentic/analyzer.py` - Coverage evaluation optimization
- `agentic/content_cache.py` - SQLite-backed content and query cache
- `agentic/ttl_cache_manager.py` - Continuum-inspired TTL-based KV cache pinning
- `agentic/prompts.py` - Centralized prompt registry for KV cache hits
- `agentic/artifacts.py` - Filesystem-based artifact store for token reduction
- `agentic/metrics.py` - Performance metrics tracking (TTFT, cache hits, tokens)
- `agentic/scratchpad.py` - Enhanced with public/private spaces, KV cache refs
- `agentic/kv_cache_service.py` - Phase 4: Unified KV cache interface for Ollama/vLLM
- `agentic/memory_tiers.py` - Phase 4: Three-tier memory (cold/warm/hot) architecture
- `agentic/OPTIMIZATION_ANALYSIS.md` - Test results and bottleneck analysis
- `agentic/KV_CACHE_IMPLEMENTATION_PLAN.md` - Full 4-phase optimization roadmap
- `setup_ollama_optimization.sh` - Ollama environment configuration

---

## API Endpoints

### Cache & Performance Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/cache/stats` | GET | View content cache statistics |
| `/api/v1/search/ttl/stats` | GET | View TTL pinning statistics and tool latencies |
| `/api/v1/search/metrics` | GET | View performance metrics (TTFT, tokens, cache hits) |
| `/api/v1/search/artifacts/stats` | GET | View artifact store statistics |
| `/api/v1/search/cache` | DELETE | Clear all caches |
| `/api/v1/search/artifacts/{session_id}` | DELETE | Clean up session artifacts |

### Phase 4 Memory Tier Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/memory/tiers/stats` | GET | View three-tier memory statistics |
| `/api/v1/search/memory/kv-cache/stats` | GET | View KV cache service stats |
| `/api/v1/search/memory/kv-cache/warm` | GET | List warm cache entries |
| `/api/v1/search/memory/kv-cache/warm` | POST | Warm a prefix in KV cache |
| `/api/v1/search/memory/tiers/store` | POST | Store content in memory tiers |
| `/api/v1/search/memory/tiers/{content_id}` | GET | Retrieve content |
| `/api/v1/search/memory/tiers/{content_id}/promote` | POST | Promote cold→warm |
| `/api/v1/search/memory/tiers/{content_id}/demote` | POST | Demote warm→cold |
| `/api/v1/search/memory/initialize` | POST | Initialize and warm system prompts |

---

## SGLang Evaluation (G.8.5 - 2025-12-31)

Evaluating SGLang as a high-performance alternative to Ollama for LLM inference with speculative decoding.

### Installation Status
- SGLang v0.5.7 installed with FlashInfer backend
- TITAN RTX (24GB, sm75) meets hardware requirements
- Benchmark script: `scripts/benchmark_sglang_vs_ollama.py`

### Ollama Baseline Results (qwen3:8b)

| Metric | Value | Notes |
|--------|-------|-------|
| Avg TTFT | 31.7s | Time to first token (includes prefill) |
| P50 TTFT | 25.8s | Median latency |
| P95 TTFT | 88.7s | Long tail for complex prompts |
| Avg Tokens/sec | 24.2 | Generation throughput |
| Success Rate | 100% | All requests completed |

### SGLang Expected Benefits

| Feature | Expected Impact |
|---------|-----------------|
| RadixAttention | Automatic KV cache prefix sharing (reduces TTFT) |
| Speculative Decoding | 2-5x throughput via draft model verification |
| FlashInfer | Optimized attention kernels for sm75+ GPUs |
| Continuous Batching | Better GPU utilization under concurrent load |

### Benchmark Usage

```bash
# Activate venv first
source venv/bin/activate

# Run Ollama-only benchmark
python scripts/benchmark_sglang_vs_ollama.py --ollama-only --runs 3

# Run SGLang-only benchmark (requires SGLang server running)
python scripts/benchmark_sglang_vs_ollama.py --sglang-only --runs 3

# Compare both
python scripts/benchmark_sglang_vs_ollama.py --compare --runs 3 --output results.json
```

### Starting SGLang Server

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000 \
    --mem-fraction-static 0.85
```

**Status:** Baseline complete. Full SGLang comparison pending HuggingFace model download (~15GB).

---

## Design Rationale

### Current vs. Agentic Approach

Current Android implementation uses a simple web search pattern:
```
User Query → Extract Search Keywords → Single Web Search → Inject Results
```

The agentic approach:
```
User Query → Planner Agent → [Decomposed Queries] → Searcher Agents →
Verifier Agent → Synthesizer Agent → Verified Context → Main LLM
```

---

## MCP Node Editor Integration

memOS leverages the **MCP Node Editor** (`/home/sparkone/sdd/MCP_Node_Editor`) as the underlying workflow orchestration engine. This provides:

- **27 Node Types**: Including `agent_orchestrator`, `web_search`, `rag_pipeline`, `memory`
- **Cyclic Workflows**: Iterative refinement until convergence
- **Event-Driven Architecture**: 1000+ events/sec throughput
- **Code Sandboxing**: Safe execution of generated code
- **Circuit Breakers**: Automatic error loop prevention

See `mcp_node_editor_integration.md` for full API reference.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    memOS Agentic Search Service                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Orchestrator │───▶│   Planner    │───▶│   Searcher   │       │
│  │    Agent     │    │    Agent     │    │    Agent(s)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Memory     │◀───│   Verifier   │◀───│  Synthesizer │       │
│  │   Service    │    │    Agent     │    │    Agent     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                  MCP Node Editor (Port 7777)                     │
│              Pipeline Orchestration & Execution                  │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Responsibility | LLM Model |
|-------|---------------|-----------|
| **Orchestrator** | Receives query + history, routes to appropriate pipeline | llama3.2:3b |
| **Planner** | Decomposes complex queries, generates search strategy | llama3.2:3b |
| **Searcher** | Executes web searches, scrapes pages | (no LLM, uses APIs) |
| **Verifier** | Cross-checks facts, detects contradictions | llama3.2:3b |
| **Synthesizer** | Combines results, formats for injection | llama3.2:3b |

---

## ReAct Loop Implementation

```python
class AgenticSearchService:
    """
    Implements ReAct pattern for intelligent web search.

    Loop: THINK → ACT → OBSERVE → THINK → ...
    Until: Sufficient information gathered OR max iterations reached
    """

    async def execute_search(self, query: str, context: dict) -> SearchResult:
        state = SearchState(query=query, context=context)

        for iteration in range(self.max_iterations):
            # THINK: Planner decides next action
            action = await self.planner.decide(state)

            if action.type == "SEARCH":
                # ACT: Execute search
                results = await self.searcher.search(action.queries)
                # OBSERVE: Update state with results
                state.add_results(results)

            elif action.type == "VERIFY":
                # ACT: Cross-check claims
                verified = await self.verifier.verify(state.claims)
                # OBSERVE: Mark verified/unverified
                state.update_verification(verified)

            elif action.type == "SYNTHESIZE":
                # ACT: Combine and format
                synthesis = await self.synthesizer.synthesize(state)
                return synthesis

            elif action.type == "DONE":
                break

        return await self.synthesizer.synthesize(state)
```

---

## API Endpoints (Agentic Search)

```python
POST /api/v1/search/agentic
    """
    Execute multi-step agentic search.

    Request:
        {
            "query": "What treatment options exist for opioid addiction?",
            "user_id": "uuid",
            "context": {
                "conversation_history": [...],
                "user_preferences": {...}
            },
            "max_iterations": 3,
            "verification_level": "standard"  # none|standard|strict
        }

    Response:
        {
            "success": true,
            "data": {
                "synthesized_context": "...",
                "sources": [...],
                "confidence_score": 0.85,
                "verification_status": "verified",
                "search_trace": [...]  # For debugging
            },
            "meta": {
                "iterations": 2,
                "queries_executed": 4,
                "sources_consulted": 8
            }
        }
    """

GET /api/v1/search/status/{search_id}
    """Get status of running agentic search (for async execution)."""

POST /api/v1/search/simple
    """
    Lightweight single-query search (fallback for simple queries).
    Used when orchestrator determines agentic approach is overkill.
    """

POST /api/v1/context/inject
    """
    Inject verified context into memory for session use.
    Stores search results for potential reuse.
    """
```

---

## Hybrid Scoring Algorithm

Search results are scored using a hybrid approach:

```python
def calculate_relevance_score(result: SearchResult) -> float:
    """
    Hybrid scoring: BM25 (40%) + Semantic (40%) + Entity (20%)
    """
    bm25_score = calculate_bm25(result.text, query_terms)
    semantic_score = cosine_similarity(result.embedding, query_embedding)
    entity_score = entity_overlap(result.entities, query_entities)

    return (
        0.40 * normalize(bm25_score) +
        0.40 * semantic_score +
        0.20 * entity_score
    )
```

---

## Edge Model Query Optimization

For queries originating from Android edge models (1B parameters), memOS can pre-optimize:

```python
async def optimize_query_for_edge(
    raw_query: str,
    edge_model_context: str
) -> str:
    """
    Use memOS's larger model (3B) to refine queries from
    Android's smaller edge model (1B) before searching.
    """
    optimization_prompt = f"""
    Original edge model query: {raw_query}
    Context: {edge_model_context}

    Refine this into optimal web search queries.
    Output: JSON array of 1-3 search queries.
    """
    return await self.llm.generate(optimization_prompt)
```

---

## File Structure

```
memOS/server/
├── agentic/
│   ├── __init__.py
│   ├── orchestrator.py      # Main routing logic
│   ├── planner.py           # Query decomposition
│   ├── searcher.py          # Web search execution
│   ├── verifier.py          # Fact verification
│   ├── synthesizer.py       # Result synthesis
│   ├── state.py             # Search state management
│   └── scoring.py           # Hybrid relevance scoring
├── api/
│   └── search.py            # New API endpoints
└── pipelines/
    ├── agentic_search.json  # MCP Node Editor pipeline
    └── simple_search.json   # Fallback pipeline
```

---

## Integration with Memory Service

Search results are optionally cached in memory for:
- Avoiding redundant searches within session
- Building user-specific knowledge base
- Training personalized ranking models

```python
async def store_search_memory(
    user_id: str,
    query: str,
    results: List[SearchResult]
) -> None:
    memory_content = {
        "type": "search_result",
        "query": query,
        "results": [r.to_dict() for r in results],
        "timestamp": datetime.utcnow().isoformat()
    }
    await memory_service.store(
        user_id=user_id,
        content=memory_content,
        memory_type=MemoryType.PROCEDURAL,
        privacy_level=PrivacyLevel.MINIMAL
    )
```

---

## Security Considerations

1. **Source Allowlisting**: Only search trusted domains for recovery-related content
2. **Content Filtering**: PHI detection before injecting search results
3. **Rate Limiting**: Per-user limits on agentic search operations
4. **Audit Logging**: Track all search queries and results for compliance

## NOTES:
## The following are added by sparkone admin, excerpts from claude code chat regarding how the URL searching and scraping process works. It seems that we could expose the parameters
## of the existing process for configuration by the user, or intelligent use by an LLM, in a multi-step approach to gathering the context data to proceed. Or perhaps use a small suitable
## LLM.

● URL Selection Process Analysis

  Key Finding: Heuristic-Based, Not LLM-Based

  The pipeline uses zero LLM calls for URL selection. Speed and determinism are prioritized over AI-based relevance evaluation.

  7-Stage Selection Pipeline

  Search Results (SearXNG)
      ↓
  1. Query Classification (regex patterns)
      ↓
  2. Domain Trust Scoring (+0.25 premium, +0.15 trusted)
      ↓
  3. Query-Domain Alignment (+0.30 for category match)
      ↓
  4. Keyword Relevance Filter (≥15% overlap)
      ↓
  5. Multi-Engine Bonus (+5% per extra engine)
      ↓
  6. Sort by Combined Score
      ↓
  7. Take top N (preset-dependent: 5/10/15)

  Scoring Components
  ┌────────────────────┬──────────┬──────────────────────────────────────────────────────────┐
  │     Component      │  Boost   │                          Source                          │
  ├────────────────────┼──────────┼──────────────────────────────────────────────────────────┤
  │ Base score         │ 0.0-0.9  │ SearXNG relevance                                        │
  ├────────────────────┼──────────┼──────────────────────────────────────────────────────────┤
  │ Premium domain     │ +0.25    │ 62 curated domains (arxiv.org, fanuc.eu, manualslib.com) │
  ├────────────────────┼──────────┼──────────────────────────────────────────────────────────┤
  │ Trusted domain     │ +0.15    │ 316 industrial domains                                   │
  ├────────────────────┼──────────┼──────────────────────────────────────────────────────────┤
  │ Category alignment │ +0.30    │ Query type matches domain category                       │
  ├────────────────────┼──────────┼──────────────────────────────────────────────────────────┤
  │ Multi-engine       │ +5% each │ Result appears in multiple engines                       │
  └────────────────────┴──────────┴──────────────────────────────────────────────────────────┘
  Domain Categories (12 total)

  The system has 316+ trusted domains organized by category:
  ┌───────────────┬───────────────────────────────────┬────────────────────────────────────┐
  │   Category    │          Example Domains          │            Query Match             │
  ├───────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │ FANUC         │ fanuc.eu, fanucamerica.com        │ SRVO-, MOTN-, A06B-*               │
  ├───────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │ Allen-Bradley │ literature.rockwellautomation.com │ 1756-*, PLC, Logix                 │
  ├───────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │ Siemens       │ support.industry.siemens.com      │ 6ES7*, S7-*, TIA Portal            │
  ├───────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │ IMM           │ engel.at, arburg.com, husky.co    │ barrel, screw, clamp, plasticizing │
  ├───────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │ Sensors       │ bannerengineering.com, sick.com   │ proximity, photoelectric           │
  ├───────────────┼───────────────────────────────────┼────────────────────────────────────┤
  │ Hot Runners   │ moldmasters.com, husky.co         │ nozzle, manifold, heater           │
  └───────────────┴───────────────────────────────────┴────────────────────────────────────┘
  Keyword Relevance Filter

  # Minimum 15% keyword overlap required
  # Includes stemming and hyphenated codes
  query: "SRVO-063 encoder alarm"
  result_title: "FANUC Servo Alarm SRVO-063 Troubleshooting"

  keywords_in_query = {"srvo", "063", "encoder", "alarm"}
  keywords_in_result = {"fanuc", "servo", "alarm", "srvo", "063", "troubleshooting"}
  overlap = {"srvo", "063", "alarm"} = 3/4 = 75% ✓ (passes filter)

  URLs Selected Per Preset
  ┌──────────┬────────────────────┬───────────────────┐
  │  Preset  │ max_urls_to_scrape │     Rationale     │
  ├──────────┼────────────────────┼───────────────────┤
  │ MINIMAL  │ 5                  │ Fast response     │
  ├──────────┼────────────────────┼───────────────────┤
  │ BALANCED │ 10                 │ Default quality   │
  ├──────────┼────────────────────┼───────────────────┤
  │ ENHANCED │ 10                 │ Same as balanced  │
  ├──────────┼────────────────────┼───────────────────┤
  │ RESEARCH │ 15                 │ Thorough coverage │
  ├──────────┼────────────────────┼───────────────────┤
  │ FULL     │ 15                 │ Maximum coverage  │
  └──────────┴────────────────────┴───────────────────┘
  Why No LLM for URL Selection?

  1. Latency: LLM call adds 2-5s per evaluation
  2. Cost: 10 URLs × LLM call = significant overhead
  3. Determinism: Heuristics are reproducible and debuggable
  4. Domain expertise: Curated trust lists encode industrial knowledge
  5. Keyword filter: Catches obvious mismatches cheaply

  Where LLM IS Used (Post-Scrape)

  The LLM enters the pipeline after content is scraped:

  - CRAG Evaluator: Assesses retrieved content quality
  - Synthesizer: Generates final answer from scraped content
  - Self-Reflection: Validates synthesis quality

  This design scrapes first (fast, cheap), then evaluates content quality with LLM (expensive, but fewer items).

---

*Last Updated: 2026-01-02 by Claude Code*
