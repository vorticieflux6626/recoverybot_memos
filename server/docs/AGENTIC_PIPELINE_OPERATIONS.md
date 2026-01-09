# Agentic Pipeline Operations Guide

> **Updated**: 2026-01-08 | **Status**: Reference Documentation | **Parent**: [memOS CLAUDE.md](../../CLAUDE.md)

This document provides detailed documentation of each operation in the agentic search pipeline, including inputs, outputs, decision points, API endpoints, and optimization strategies.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [API Endpoints](#2-api-endpoints)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Phase-by-Phase Documentation](#4-phase-by-phase-documentation)
5. [Preset Configurations](#5-preset-configurations)
6. [Relevance Filtering Mechanisms](#6-relevance-filtering-mechanisms)
7. [Common Causes of Irrelevant Results](#7-common-causes-of-irrelevant-results)
8. [Optimization Strategies](#8-optimization-strategies)
9. [Decision Flow Diagram](#9-decision-flow-diagram)
10. [Key Files Reference](#10-key-files-reference)

---

## 1. Executive Summary

The memOS agentic search pipeline is a sophisticated multi-agent system with **12 phases** and **50+ configurable features** organized into 5 preset levels. The pipeline implements advanced RAG patterns (CRAG, Self-RAG, FLARE, HyDE) with extensive quality control mechanisms.

**Key Characteristics:**
- **Lazy component initialization** - Components only created when needed based on config flags
- **Defense-in-depth** - Multiple filtering stages prevent irrelevant results
- **Configurable presets** - MINIMAL (8 features) → FULL (42+ features)
- **Real-time observability** - SSE events and decision logging throughout

**Main Orchestrator:** `orchestrator_universal.py` (6,200+ lines)

---

## 2. API Endpoints

### Endpoint Architecture Overview

All orchestrators have been consolidated into `UniversalOrchestrator` with presets. Legacy endpoints redirect to Universal.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          API ENDPOINT HIERARCHY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RECOMMENDED (Current):                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ /gateway/stream (ChatGatewayRequest)                                 │   │
│  │   └─ ENTRY POINT for Android - Classifies and routes automatically  │   │
│  │   └─ Routes to: direct_answer | web_search | agentic_search          │   │
│  │   └─ Uses: UniversalOrchestrator.search() with event emitter set     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ /universal (UniversalSearchRequest) - Non-streaming                  │   │
│  │   └─ Direct search without classification                            │   │
│  │   └─ Supports preset selection and feature overrides                 │   │
│  │   └─ Uses: UniversalOrchestrator.search()                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ /universal/stream (UniversalSearchRequest) - Streaming with SSE      │   │
│  │   └─ Real-time progress updates                                      │   │
│  │   └─ Uses: UniversalOrchestrator.search() with event emitter         │   │
│  │   └─ NOTE: Emits wrapper events, not detailed phase events           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LEGACY (Deprecated - redirect to Universal):                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ /agentic       → get_orchestrator() → UniversalOrchestrator(balanced)│   │
│  │ /stream        → search_with_events() - DETAILED phase events        │   │
│  │ /enhanced      → get_enhanced_orchestrator() → Universal(enhanced)   │   │
│  │ /graph-enhanced → get_graph_orchestrator() → Universal(research)     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Endpoint Comparison

| Endpoint | Method | Streaming | Classification | Event Detail | Recommended |
|----------|--------|-----------|----------------|--------------|-------------|
| `/gateway/stream` | POST | Yes (SSE) | Yes | Medium | ✅ **Primary** |
| `/universal` | POST | No | No | None | ✅ For scripts |
| `/universal/stream` | POST | Yes (SSE) | No | Low | ⚠️ Use gateway |
| `/stream` | POST | Yes (SSE) | No | **High** | ⚠️ Legacy |
| `/agentic` | POST | No | No | None | ❌ Deprecated |

### Critical Implementation Note: Event Detail Levels

There's an architectural inconsistency between streaming endpoints:

**`/stream` (Legacy) - HIGH Detail:**
- Uses `search_with_events()` which manually implements pipeline with SSE events
- Emits 50+ event types including per-phase completion, LLM calls, graph nodes
- Full pipeline visualization support

**`/universal/stream` and `/gateway/stream` - LOW/MEDIUM Detail:**
- Uses `search()` with event emitter set, NOT `search_with_events()`
- Emits wrapper events (STARTED, ANALYZING, SYNTHESIZING, COMPLETED)
- Does NOT emit detailed per-phase events from `_execute_pipeline()`

**Implication:** If you need detailed SSE events for debugging or rich UI feedback, consider using the legacy `/stream` endpoint temporarily, or request enhancement of the Universal streaming path.

---

### Detailed Endpoint Documentation

#### 2.1 `/gateway/stream` (ChatGatewayRequest) - **RECOMMENDED**

The unified chat gateway with server-side routing. This is the primary entry point for Android.

**Request Model:**
```python
class ChatGatewayRequest(BaseModel):
    query: str                                    # The user's query
    user_id: Optional[str] = None                 # For personalization
    conversation_history: Optional[List[Dict]] = None  # Previous messages
    force_agentic: bool = False                   # Force full pipeline
    model: str = "qwen3:8b"                       # LLM for direct answers
    preset: str = "full"                          # Preset: minimal|balanced|enhanced|research|full
```

**Processing Flow:**
1. **Classify Query** - Uses `QueryClassifier` to determine optimal pipeline
2. **Route to Pipeline:**
   - `direct_answer` - Query can be answered from LLM knowledge alone
   - `web_search` - Simple web search sufficient
   - `agentic_search` - Full multi-agent pipeline needed
   - `code_assistant` - Technical/code analysis mode
3. **Execute Pipeline** - Calls appropriate handler with SSE events

**SSE Events Emitted:**
| Event | Description |
|-------|-------------|
| `classifying_query` | Classification started |
| `query_classified` | Classification complete with pipeline recommendation |
| `pipeline_routed` | Pipeline selection confirmed |
| `searching` | Web search in progress |
| `synthesizing` | LLM synthesis in progress |
| `gateway_complete` / `search_completed` | Final response |

**Upgrade Logic:**
- If `classification.use_thinking_model` and pipeline is `web_search`, auto-upgrades to `agentic_search`
- If `force_agentic=True`, always uses `agentic_search`

---

#### 2.2 `/universal` (UniversalSearchRequest) - Non-Streaming

Direct access to UniversalOrchestrator without classification.

**Request Model:**
```python
class UniversalSearchRequest(BaseModel):
    query: str                                    # Search query (min 3 chars)
    user_id: Optional[str] = None
    context: Optional[Dict] = None                # Conversation context
    max_iterations: int = 5                       # Max search iterations
    search_mode: str = "adaptive"                 # fixed|adaptive|exhaustive
    analyze_query: bool = True                    # Use LLM to analyze query
    verification_level: str = "standard"          # none|standard|strict
    cache_results: bool = True                    # Cache for reuse
    min_sources: int = 3                          # Minimum sources
    max_sources: int = 15                         # Maximum sources
    preset: str = "balanced"                      # Preset configuration

    # Feature overrides (optional)
    enable_hyde: Optional[bool] = None
    enable_hybrid_reranking: Optional[bool] = None
    enable_ragas: Optional[bool] = None
    enable_entity_tracking: Optional[bool] = None
    enable_thought_library: Optional[bool] = None
    enable_pre_act_planning: Optional[bool] = None
    enable_parallel_execution: Optional[bool] = None
```

**Response:**
```python
class SearchResponse(BaseModel):
    success: bool
    data: SearchData
    meta: ResponseMeta
    errors: List[ErrorDetail] = []

class SearchData(BaseModel):
    synthesized_context: str        # The generated answer
    sources: List[Source]           # Consulted sources
    search_queries: List[str]       # Queries executed
    confidence_score: float         # 0.0-1.0 confidence
    search_trace: List[Dict]        # Debug trace
```

---

#### 2.3 `/universal/stream` (UniversalSearchRequest) - Streaming

Same request model as `/universal` but returns SSE stream.

**SSE Events:**
| Event | Progress % | Description |
|-------|------------|-------------|
| `search_started` | 0% | Search initiated |
| `analyzing_query` | 5% | Query analysis begun |
| `planning_search` | 10% | Creating search strategy |
| `searching` | 20% | Executing web searches |
| `synthesizing` | 90% | Combining results |
| `search_completed` | 100% | Final results in data field |

**Response Headers:**
```
Content-Type: text/event-stream
X-Request-Id: <uuid>
X-Universal: true
X-Preset: <preset_name>
```

---

#### 2.4 `/stream` (SearchRequest) - Legacy Streaming with DETAILED Events

**IMPORTANT:** This is the only endpoint that uses `search_with_events()` which provides detailed per-phase SSE events.

**SSE Events (50+ types):**
```
# Phase Events
analyzing_query, query_analyzed
planning_search, search_planned
searching, search_results
hybrid_search_start, hybrid_search_complete
crag_evaluating, crag_evaluation_complete
evaluating_urls, urls_evaluated
verifying_claims, claims_verified
synthesizing, synthesis_complete
self_reflecting, self_reflection_complete

# Graph Visualization
graph_node_entered, graph_node_completed

# LLM Debug
llm_call_start, llm_call_complete

# Entity Tracking
entities_extracted, entity_relation_found

# Reasoning
reasoning_branch_created, reasoning_node_verified
thought_template_matched, reasoning_strategy_composed

# Advanced Features
hyde_generating, hyde_complete
dylan_complexity_classified, dylan_agent_skipped
ib_filtering_start, ib_filtering_complete
iteration_start_detailed, iteration_complete_detailed
```

**Use Case:** Rich debugging UI, detailed progress visualization.

---

#### 2.5 Helper Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/universal/presets` | GET | List available presets with feature counts |
| `/universal/stats` | GET | Orchestrator statistics (cache hits, searches) |
| `/events/{request_id}` | GET | Reconnect to in-progress search SSE stream |
| `/active` | GET | List active searches |
| `/cache/stats` | GET | Content cache statistics |
| `/metrics` | GET | Performance metrics (TTFT, tokens) |

---

### Request Model Comparison

| Field | ChatGatewayRequest | UniversalSearchRequest | SearchRequest (Legacy) |
|-------|-------------------|------------------------|----------------------|
| query | ✅ | ✅ (min 3 chars) | ✅ |
| user_id | ✅ | ✅ | ✅ |
| conversation_history | ✅ | ❌ | ❌ |
| context | ❌ | ✅ | ✅ |
| force_agentic | ✅ | ❌ | ❌ |
| model | ✅ | ❌ | ❌ |
| preset | ✅ (default: full) | ✅ (default: balanced) | ❌ |
| max_iterations | ❌ | ✅ (default: 5) | ✅ (default: 5) |
| search_mode | ❌ | ✅ | ✅ |
| verification_level | ❌ | ✅ | ✅ |
| Feature overrides | ❌ | ✅ (7 flags) | ❌ |

---

## 3. Pipeline Overview

### Complete Data Flow

```
[User Query]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1: Query Analysis                                          │
│   • Classify query type (factual, troubleshooting, etc.)        │
│   • Assess complexity (simple → expert)                          │
│   • Extract entities and expand acronyms                         │
│   • Decision: Does this query require web search?                │
└────────────────────────┬────────────────────────────────────────┘
                         │ YES
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2: Search Planning                                         │
│   • Decompose into sub-questions                                │
│   • Retrieve reasoning templates (Meta-Buffer)                  │
│   • HyDE query expansion for better semantic matching           │
│   • Create search strategy                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 3: Search Execution (ReAct Loop)                          │
│   • Execute web searches via SearXNG                            │
│   • Hybrid re-ranking with BGE-M3 (dense + sparse)             │
│   • FLARE uncertainty-aware retrieval                           │
│   • Query tree expansion for parallel exploration               │
│   • Loop: Refine queries if results insufficient                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 4: Context Curation                                        │
│   • Deduplication (>0.85 similarity)                            │
│   • DIG scoring (Document Information Gain)                     │
│   • Two-stage filtering (recall → precision)                    │
│   • Clustering and representative selection                      │
│   • Output: 5-20 curated documents                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 5: CRAG Evaluation (Pre-Synthesis)                        │
│   • Score each document: relevance, quality, coverage           │
│   • Classify: CORRECT (≥0.7) | AMBIGUOUS | INCORRECT (<0.4)     │
│   • Decision: PROCEED | REFINE_QUERY | WEB_FALLBACK             │
└────────────────────────┬──────────┬─────────────────────────────┘
                         │ PROCEED  │ REFINE/FALLBACK
                         ▼          └──────► (Loop to Phase 3)
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 6: Scraping & Deep Reading                                │
│   • Full-page content scraping (not just snippets)              │
│   • Vision-language analysis for JS-heavy sites                 │
│   • Deep reading for technical documentation                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 7: Technical Document Integration (Conditional)           │
│   • Query PDF Tools API (port 8002)                             │
│   • HSEA three-stratum retrieval for domain knowledge           │
│   • Symptom-based entry via INDICATES edges                     │
│   • Structured troubleshooting chains                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 8: Synthesis                                              │
│   • Combine scraped content with LLM                            │
│   • Model selection based on complexity                         │
│   • FLARE integration for uncertainty detection                 │
│   • Output: Synthesized answer with citations                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 9: Cross-Domain Validation (Phase 48)                     │
│   • Extract causal chains from synthesis                        │
│   • Validate multi-hop relationships                            │
│   • Entity grounding via PDF Tools lookup                       │
│   • Decision: Revise synthesis if critical issues               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 10: Self-Reflection (Self-RAG)                            │
│   • ISREL: Is retrieved content relevant?                       │
│   • ISSUP: Is synthesis supported by sources?                   │
│   • ISUSE: Is response useful?                                  │
│   • Temporal consistency checking                                │
│   • Output: ReflectionResult with confidence                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 11: RAGAS Evaluation (Optional)                           │
│   • Faithfulness: Is answer supported by sources?               │
│   • Answer relevancy: Does answer address query?                │
│   • Claim verification                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 12: Experience Distillation                               │
│   • Store successful patterns for template creation             │
│   • Update classifier feedback loop                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                  [Final Response]
```

---

## 4. Phase-by-Phase Documentation

### Phase 1: Query Analysis

**File:** `agentic/analyzer.py` (1,400+ lines)

**Purpose:** Classify and understand the incoming query before deciding how to process it.

#### Input
| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Raw user query |
| `context` | dict (optional) | Conversation history, user preferences |
| `config` | FeatureConfig | Pipeline configuration flags |

#### Processing Steps

1. **Search Necessity Check** - Determines if web search is needed vs direct answer
2. **Query Type Classification:**
   - `factual` - Fact-based questions (dates, definitions)
   - `opinion` - Subjective questions
   - `troubleshooting` - Problem-solving queries
   - `code` - Programming questions
   - `research` - Academic/thorough research
   - `creative` - Creative writing requests

3. **Complexity Assessment:**
   - `simple` (1-3) - Single-step answer
   - `moderate` (4-6) - Multi-step reasoning
   - `complex` (7-8) - Deep research needed
   - `expert` (9-10) - Specialized knowledge required

4. **Entity Extraction** (if `enable_entity_tracking`) - Extracts named entities, part numbers, error codes

#### Output: `QueryAnalysis`
```python
@dataclass
class QueryAnalysis:
    requires_search: bool          # Should we search the web?
    query_type: str                # Classification category
    complexity: str                # simple|moderate|complex|expert
    estimated_complexity: int      # 1-10 scale
    requires_thinking_model: bool  # Needs reasoning model?
    key_topics: List[str]          # Extracted topics
```

#### Decision Points
| Condition | Decision | Next Phase |
|-----------|----------|------------|
| `requires_search = False` | Direct answer | Phase 8 (skip search) |
| `requires_search = True` | Web search | Phase 2 |
| `requires_thinking_model = True` | Use DeepSeek R1 | Affects Phase 8 |

---

### Phase 3: Search Execution

**Files:** `agentic/searxng_search.py`, `agentic/bge_m3_hybrid.py`

#### Processing Steps

1. **SearXNG Search** - Queries multiple engines: Brave (1.5), Bing (1.2), Reddit (25), Wikipedia, arXiv
2. **Hybrid Re-ranking** (if `enable_hybrid_reranking`) - BGE-M3 dense+sparse fusion
3. **FLARE Enhancement** (if `enable_flare_retrieval`) - Monitors for uncertainty, triggers additional retrieval
4. **Query Tree Expansion** (if `enable_query_tree`) - Parallel sub-query exploration

---

### Phase 5: CRAG Evaluation (Pre-Synthesis)

**File:** `agentic/retrieval_evaluator.py`

**Purpose:** Assess retrieval quality BEFORE synthesis to catch problems early.

#### Scoring System
| Quality | Score Range | Action |
|---------|-------------|--------|
| **CORRECT** | ≥0.7 | PROCEED to synthesis |
| **AMBIGUOUS** | 0.4-0.7 | REFINE_QUERY |
| **INCORRECT** | <0.4 | WEB_FALLBACK or DECOMPOSE |

**Trusted Domain Scoring:**
```
0.95: arxiv.org, nature.com, ieee.org
0.90: wikipedia.org, docs.python.org
0.85: github.com, .gov, .edu
0.80: stackoverflow.com
```

---

### Phase 8: Synthesis

**File:** `agentic/synthesizer.py`

#### Model Selection Strategy
```python
if config.force_thinking_model:
    model = thinking_model
elif query_analysis.requires_thinking_model:
    model = thinking_model
elif query_analysis.complexity in ["complex", "expert"]:
    model = thinking_model
else:
    model = default_synthesis_model  # qwen3:8b
```

**Available Thinking Models:**
| Model | VRAM | Use Case |
|-------|------|----------|
| ministral-3:3b | ~3GB | Default (93.3% coverage, 17s) |
| deepseek-r1:8b | 5GB | Better reasoning |
| deepseek-r1:14b | 15GB | Complex queries |

---

### Phase 10: Self-Reflection (Self-RAG)

**File:** `agentic/self_reflection.py`

**Three Reflection Tokens:**
1. **ISREL** (0-1): Is retrieved content relevant?
2. **ISSUP**: Is synthesis supported? (FULLY_SUPPORTED, PARTIALLY_SUPPORTED, NO_SUPPORT, CONTRADICTED)
3. **ISUSE** (0-1): Is response useful?

---

## 5. Preset Configurations

### Summary Table

| Preset | Features | Typical Latency | Use Case |
|--------|----------|-----------------|----------|
| **MINIMAL** | 8 | 15-30s | Fast, simple queries |
| **BALANCED** | 18 | 60-90s | Default for most queries |
| **ENHANCED** | 28 | 90-120s | Complex research |
| **RESEARCH** | 35 | 120-180s | Academic/thorough |
| **FULL** | 42+ | 180-300s | Maximum capability |

### Key Features by Preset

| Feature | MIN | BAL | ENH | RES | FULL |
|---------|-----|-----|-----|-----|------|
| query_analysis | ✅ | ✅ | ✅ | ✅ | ✅ |
| self_reflection | ❌ | ✅ | ✅ | ✅ | ✅ |
| crag_evaluation | ❌ | ✅ | ✅ | ✅ | ✅ |
| hybrid_reranking | ❌ | ✅ | ✅ | ✅ | ✅ |
| hyde | ❌ | ❌ | ✅ | ✅ | ✅ |
| cross_encoder | ❌ | ❌ | ✅ | ✅ | ✅ |
| ragas | ❌ | ❌ | ✅ | ✅ | ✅ |
| technical_docs | ❌ | ❌ | ✅ | ✅ | ✅ |
| cross_domain_validation | ❌ | ❌ | ✅ | ✅ | ✅ |
| entropy_halting | ❌ | ❌ | ❌ | ✅ | ✅ |
| flare_retrieval | ❌ | ❌ | ❌ | ✅ | ✅ |
| query_tree | ❌ | ❌ | ❌ | ✅ | ✅ |
| vision_analysis | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 6. Relevance Filtering Mechanisms

### Defense-in-Depth Summary

| Phase | Filter | Threshold | Action on Failure |
|-------|--------|-----------|-------------------|
| 3 | Hybrid re-ranking | Top-k | Lower results excluded |
| 4 | Deduplication | >0.85 similarity | Duplicates removed |
| 4 | DIG scoring | min_dig_score | Low-utility removed |
| 5 | CRAG relevance | ≥0.6 | Flag or refine |
| 5 | CRAG overall | ≥0.7 CORRECT | Loop back if low |
| 9 | Cross-domain | Relationship registry | Revise if invalid |
| 10 | ISREL | ≥0.5 | Flag |
| 10 | ISSUP | SUPPORTED | Flag unsupported |

---

## 7. Common Causes of Irrelevant Results

### Problem 1: Disabled Quality Filters
**Symptom:** Low-quality results pass through to synthesis
**Cause:** Using MINIMAL preset or manually disabling filters
**Fix:** Use BALANCED or higher preset

### Problem 2: Query Misclassification
**Symptom:** Wrong search engines selected
**Fix:** Enable `enable_entity_tracking=True`, `enable_acronym_expansion=True`

### Problem 3: Broad Search Queries
**Symptom:** Too many tangentially related results
**Fix:** Enable HyDE (`enable_hyde=True`), context curation

### Problem 4: Synthesis Drift
**Symptom:** Answer addresses different topic than query
**Fix:** Enable self-reflection, check ISREL/ISUSE scores

---

## 8. Optimization Strategies

### Strategy 1: Reduce Latency
| Technique | Impact | Configuration |
|-----------|--------|---------------|
| Use MINIMAL preset | 70-80% faster | `preset="MINIMAL"` |
| Reduce iterations | 20-30% faster | `max_iterations=2` |
| Enable caching | 90%+ on repeat | `enable_semantic_cache=True` |

### Strategy 2: Improve Relevance
| Technique | Impact | Configuration |
|-----------|--------|---------------|
| Enable CRAG | Catches 30-40% bad results | `enable_crag_evaluation=True` |
| Enable hybrid reranking | +28% NDCG | `enable_hybrid_reranking=True` |
| Use ENHANCED preset | All above combined | `preset="ENHANCED"` |

---

## 9. Decision Flow Diagram

```
Query → Analysis → requires_search?
                   ├─ NO → Direct Synthesis
                   └─ YES → Planning → Search Loop
                                        │
                            ┌───────────┴───────────┐
                            │ CRAG Evaluation       │
                            │ quality?              │
                            ├─ CORRECT → Scrape     │
                            ├─ AMBIGUOUS → Refine ──┘
                            └─ INCORRECT → Fallback─┘
                                        │
                            Synthesis → Cross-Domain Validation
                                        │
                            Self-Reflection → RAGAS → Response
```

---

## 10. Key Files Reference

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Orchestrator** | `orchestrator_universal.py` | 6,200+ | Main pipeline coordinator |
| **API Endpoints** | `api/search.py` | 7,000+ | All search endpoints |
| **Query Analysis** | `analyzer.py` | 1,400+ | Query classification |
| **CRAG Evaluation** | `retrieval_evaluator.py` | 600+ | Pre-synthesis quality |
| **Synthesis** | `synthesizer.py` | 600+ | LLM answer generation |
| **Self-Reflection** | `self_reflection.py` | 900+ | Self-RAG post-synthesis |

### Key Methods in UniversalOrchestrator

| Method | Purpose | Event Emission |
|--------|---------|----------------|
| `search()` | Main non-streaming search | None (stores observability at end) |
| `search_with_events()` | Streaming with detailed events | 50+ event types |
| `_execute_pipeline()` | Core pipeline logic (called by search()) | None |
| `set_event_emitter()` | Set emitter for event emission | N/A |

---

## 11. Pipeline Bypass Modalities

Before the agentic pipeline executes, the Gateway Classifier determines whether the full pipeline is needed. There are **4 distinct processing modalities**:

### 11.1 Modality Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         GATEWAY CLASSIFICATION ROUTING                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User Query                                                                      │
│       │                                                                          │
│       ▼                                                                          │
│  ┌────────────────────┐                                                         │
│  │ QueryClassifier    │  (qwen3:8b, temp=0.2)                                   │
│  │ - Category         │                                                         │
│  │ - Complexity       │                                                         │
│  │ - Capabilities     │                                                         │
│  │ - Pipeline         │                                                         │
│  └─────────┬──────────┘                                                         │
│            │                                                                     │
│   ┌────────┴────────┬────────────────┬─────────────────┐                        │
│   │                 │                │                 │                        │
│   ▼                 ▼                ▼                 ▼                        │
│ ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│ │ DIRECT   │  │ WEB_SEARCH   │  │ AGENTIC      │  │ CODE         │             │
│ │ ANSWER   │  │              │  │ SEARCH       │  │ ASSISTANT    │             │
│ ├──────────┤  ├──────────────┤  ├──────────────┤  ├──────────────┤             │
│ │ LLM only │  │ Search +     │  │ Full 12-     │  │ Technical    │             │
│ │ No search│  │ Synthesis    │  │ phase pipe   │  │ code mode    │             │
│ │ ~2-5s    │  │ ~30-60s      │  │ ~60-180s     │  │ ~30-90s      │             │
│ └──────────┘  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Modality Decision Matrix

| Modality | Trigger Conditions | Phases Executed | Typical Latency |
|----------|-------------------|-----------------|-----------------|
| **DIRECT_ANSWER** | Simple factual, conversational, in-model knowledge | 0 (LLM only) | 2-5s |
| **WEB_SEARCH** | Current events, specific facts, verifiable info | 3, 6, 8 only | 30-60s |
| **AGENTIC_SEARCH** | Complex research, troubleshooting, multi-source | All 12 phases | 60-180s |
| **CODE_ASSISTANT** | Programming, debugging, technical implementation | Subset + code tools | 30-90s |

### 11.3 Classification Rules

**QueryClassifier** (`agentic/query_classifier.py`) uses these rules:

```
INDUSTRIAL TROUBLESHOOTING (forces AGENTIC_SEARCH + thinking model):
├── Error codes: SRVO-xxx, MOTN-xxx, SYST-xxx, INTP-xxx, HOST-xxx
├── Diagnostic: "intermittent", "encoder problem", "servo alarm"
├── Robot comparisons and technical evaluations
└── Complexity: EXPERT

PROCEDURAL QUERIES (AGENTIC_SEARCH without thinking model):
├── Mastering procedures, calibration, backup/restore
├── How-to for industrial equipment
└── Complexity: COMPLEX

SIMPLE FACTUAL (DIRECT_ANSWER):
├── Single-fact questions
├── Conversational responses
├── Answer exists in model knowledge
└── Complexity: SIMPLE

CURRENT INFORMATION (WEB_SEARCH):
├── Current events, recent news
├── Specific verifiable facts
├── Time-sensitive information
└── Complexity: MODERATE
```

### 11.4 Modality Processing Details

#### DIRECT_ANSWER (Pipeline Bypass)
```
Query → Classifier → DIRECT_ANSWER
                          │
                          ▼
                    ┌─────────────┐
                    │ Ollama Chat │
                    │ (qwen3:8b)  │
                    └──────┬──────┘
                           │
                           ▼
                      Response
                    (confidence: 0.9 fixed)
```
- **Phases skipped:** ALL (1-12)
- **No web search, no verification, no quality control**
- **Use case:** Greetings, simple questions, conversational

#### WEB_SEARCH (Partial Pipeline)
```
Query → Classifier → WEB_SEARCH
                          │
                          ▼
                 ┌──────────────────┐
                 │ UniversalOrch    │
                 │ (preset from req)│
                 └────────┬─────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
 Phase 3:             Phase 6:             Phase 8:
 Search               Scraping             Synthesis
 (SearXNG)            (Full pages)         (LLM combine)
```
- **Phases executed:** 3 (Search), 6 (Scraping), 8 (Synthesis)
- **Phases skipped:** 1, 2, 4, 5, 7, 9, 10, 11, 12
- **Quality control:** Depends on preset (CRAG may still run)

#### AGENTIC_SEARCH (Full Pipeline)
```
Query → Classifier → AGENTIC_SEARCH
                          │
                          ▼
                 ┌──────────────────┐
                 │ UniversalOrch    │
                 │ (preset from req)│
                 └────────┬─────────┘
                          │
                    All 12 Phases
                    (per preset config)
```
- **Phases executed:** All enabled by preset
- **Upgrade trigger:** `use_thinking_model=true` in classification

#### CODE_ASSISTANT (Technical Mode)
```
Query → Classifier → CODE_ASSISTANT
                          │
                          ▼
                 ┌──────────────────┐
                 │ UniversalOrch    │
                 │ (agentic mode)   │
                 └────────┬─────────┘
                          │
                  Technical Pipeline
                  + Code Analysis Tools
```
- **Treated as:** AGENTIC_SEARCH with technical focus
- **Additional:** Code analysis capabilities enabled

---

## 12. Preset Decision Trees & Operation Nodes

Each preset defines which operations execute. Below are complete decision trees showing the conditional execution paths.

### 12.1 MINIMAL Preset Decision Tree

**Features Enabled:** 8
**Typical Latency:** 15-30s
**Use Case:** Fast, simple queries where speed > quality

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MINIMAL PRESET DECISION TREE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  START                                                                           │
│    │                                                                             │
│    ▼                                                                             │
│  ┌─────────────────┐                                                            │
│  │ PHASE 1: Query  │ ◄── enable_query_analysis=TRUE (always)                    │
│  │ Analysis        │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ├── requires_search?                                                   │
│           │   ├── NO ───► Direct Synthesis (Phase 8 only)                       │
│           │   └── YES ──┐                                                        │
│           │             │                                                        │
│           ▼             ▼                                                        │
│  ┌─────────────────┐                                                            │
│  │ PHASE 3: Search │ ◄── Basic SearXNG search                                   │
│  │ Execution       │     NO hybrid reranking                                    │
│  └────────┬────────┘     NO CRAG evaluation                                     │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 6: Scrape │ ◄── Basic content extraction                               │
│  │ (Basic)         │     NO deep reading                                        │
│  └────────┬────────┘     NO vision analysis                                     │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 8: Synth  │ ◄── Direct synthesis                                       │
│  │ (Basic)         │     NO verification pass                                   │
│  └────────┬────────┘     NO self-reflection                                     │
│           │                                                                      │
│           ▼                                                                      │
│         END                                                                      │
│                                                                                  │
│  DISABLED PHASES: 2, 4, 5, 7, 9, 10, 11, 12                                     │
│  DISABLED FEATURES:                                                              │
│    ✗ self_reflection        ✗ crag_evaluation                                   │
│    ✗ adaptive_refinement    ✗ semantic_cache                                    │
│    ✗ experience_distillation ✗ classifier_feedback                             │
│    ✗ ttl_pinning            ✗ metrics                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**MINIMAL Operation Nodes:**
| Node | Operation | Enabled | Input | Output |
|------|-----------|---------|-------|--------|
| N1 | Query Analysis | ✅ | query | QueryAnalysis |
| N2 | Search Planning | ❌ | - | - |
| N3 | SearXNG Search | ✅ | queries | raw_results |
| N4 | Hybrid Rerank | ❌ | - | - |
| N5 | CRAG Eval | ❌ | - | - |
| N6 | Content Scrape | ✅ | urls | scraped_content |
| N7 | Context Curation | ❌ | - | - |
| N8 | Verification | ❌ | - | - |
| N9 | Technical Docs | ❌ | - | - |
| N10 | Synthesis | ✅ | content | synthesis |
| N11 | Cross-Domain Val | ❌ | - | - |
| N12 | Self-Reflection | ❌ | - | - |
| N13 | RAGAS | ❌ | - | - |

---

### 12.2 BALANCED Preset Decision Tree

**Features Enabled:** 18
**Typical Latency:** 60-90s
**Use Case:** Default for most queries - good quality/speed tradeoff

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          BALANCED PRESET DECISION TREE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  START                                                                           │
│    │                                                                             │
│    ▼                                                                             │
│  ┌─────────────────┐                                                            │
│  │ PHASE 1: Query  │ ◄── enable_query_analysis=TRUE                             │
│  │ Analysis        │     enable_verification=TRUE                               │
│  └────────┬────────┘     enable_scratchpad=TRUE                                 │
│           │                                                                      │
│           ├── requires_search?                                                   │
│           │   ├── NO ───► Direct Synthesis + Self-Reflection                    │
│           │   └── YES ──┐                                                        │
│           │             │                                                        │
│           ▼             ▼                                                        │
│  ┌─────────────────┐                                                            │
│  │ PHASE 2: Plan   │ ◄── Basic decomposition                                    │
│  │ (Basic)         │     Scratchpad initialized                                 │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐        ┌─────────────────┐                                 │
│  │ PHASE 3: Search │───────►│ PHASE 3.5:      │ ◄── enable_hybrid_reranking=TRUE│
│  │ Execution       │        │ BGE-M3 Rerank   │     (+2-5% NDCG)                │
│  └─────────────────┘        └────────┬────────┘                                 │
│                                      │                                          │
│                                      ▼                                          │
│                             ┌─────────────────┐                                 │
│                             │ PHASE 5: CRAG   │ ◄── enable_crag_evaluation=TRUE │
│                             │ Evaluation      │                                 │
│                             └────────┬────────┘                                 │
│                                      │                                          │
│                  ┌───────────────────┼───────────────────┐                      │
│                  │                   │                   │                      │
│            CORRECT (≥0.7)      AMBIGUOUS (0.4-0.7)  INCORRECT (<0.4)            │
│                  │                   │                   │                      │
│                  ▼                   ▼                   ▼                      │
│              PROCEED           REFINE_QUERY         WEB_FALLBACK                │
│                  │                   │                   │                      │
│                  │                   └───────┬───────────┘                      │
│                  │                           │                                  │
│                  │                   Loop back to Phase 3                       │
│                  │                   (max 3 attempts)                           │
│                  ▼                                                              │
│  ┌─────────────────┐                                                            │
│  │ PHASE 6: Scrape │ ◄── Standard content extraction                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 7: HSEA   │ ◄── enable_hsea_context=TRUE                               │
│  │ (FANUC docs)    │     enable_domain_corpus=TRUE                              │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 8: Synth  │ ◄── Standard synthesis                                     │
│  │ (Standard)      │     With verification                                      │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 10: Self  │ ◄── enable_self_reflection=TRUE                            │
│  │ Reflection      │     ISREL, ISSUP, ISUSE checks                             │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ├── needs_refinement?                                                  │
│           │   ├── YES ──► Log suggestions (don't block)                         │
│           │   └── NO ───┐                                                        │
│           │             │                                                        │
│           ▼             ▼                                                        │
│  ┌─────────────────┐                                                            │
│  │ PHASE 12: Learn │ ◄── enable_experience_distillation=TRUE                    │
│  │ (Distillation)  │     enable_classifier_feedback=TRUE                        │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│         END                                                                      │
│                                                                                  │
│  ENABLED FEATURES (Layer 1 + selective Layer 2):                                │
│    ✓ query_analysis         ✓ verification         ✓ scratchpad                 │
│    ✓ self_reflection        ✓ crag_evaluation      ✓ sufficient_context         │
│    ✓ experience_distillation ✓ classifier_feedback ✓ adaptive_refinement        │
│    ✓ hybrid_reranking       ✓ domain_corpus        ✓ hsea_context               │
│    ✓ content_cache          ✓ semantic_cache       ✓ ttl_pinning                │
│                                                                                  │
│  DISABLED FEATURES:                                                              │
│    ✗ hyde                   ✗ cross_encoder        ✗ ragas                      │
│    ✗ context_curation       ✗ entity_tracking      ✗ technical_docs             │
│    ✗ cross_domain_validation ✗ entropy_halting     ✗ flare_retrieval            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**BALANCED Operation Nodes:**
| Node | Operation | Enabled | Conditional | Decision Point |
|------|-----------|---------|-------------|----------------|
| N1 | Query Analysis | ✅ | - | requires_search? |
| N2 | Search Planning | ✅ | - | - |
| N3 | SearXNG Search | ✅ | - | - |
| N4 | BGE-M3 Rerank | ✅ | - | - |
| N5 | CRAG Eval | ✅ | - | quality: CORRECT/AMBIGUOUS/INCORRECT |
| N6 | Content Scrape | ✅ | - | - |
| N7 | HSEA Retrieval | ✅ | - | - |
| N8 | Verification | ✅ | - | - |
| N9 | Synthesis | ✅ | - | - |
| N10 | Self-Reflection | ✅ | - | needs_refinement? |
| N11 | Experience Distill | ✅ | confidence ≥0.8 | - |

---

### 12.3 ENHANCED Preset Decision Tree

**Features Enabled:** 28
**Typical Latency:** 90-120s
**Use Case:** Complex research requiring accuracy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ENHANCED PRESET DECISION TREE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  START                                                                           │
│    │                                                                             │
│    ▼                                                                             │
│  ┌─────────────────┐                                                            │
│  │ PHASE 1: Query  │ ◄── Full analysis suite                                    │
│  │ Analysis        │     + Entity Extraction (enable_entity_tracking=TRUE)      │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 1.5:      │ ◄── enable_entity_tracking=TRUE                            │
│  │ Entity Extract  │     GSW-style entity tracking                              │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 2: Plan   │ ◄── enable_thought_library=TRUE                            │
│  │ + Templates     │     Retrieve thought templates                             │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 2.1: HyDE │ ◄── enable_hyde=TRUE                                       │
│  │ Query Expansion │     Generate hypothetical docs                             │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 3: Search │ ◄── Multi-engine search                                    │
│  │ + Entity-Aware  │     enable_entity_enhanced_retrieval=TRUE                  │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 3.5:      │ ◄── enable_hybrid_reranking=TRUE                           │
│  │ BGE-M3 + Cross  │     enable_cross_encoder=TRUE (+28% NDCG)                  │
│  │ Encoder         │     enable_mixed_precision=TRUE                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 4: Context│ ◄── enable_context_curation=TRUE                           │
│  │ Curation (DIG)  │     preset="balanced"                                      │
│  │ - Deduplication │     - Dedup >0.85 similarity                               │
│  │ - DIG Scoring   │     - DIG utility scoring                                  │
│  │ - Two-Stage     │     - Recall → Precision                                   │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐     ┌────────────────┐                                     │
│  │ PHASE 5: CRAG   │────►│ CRAG Decision  │                                     │
│  │ Evaluation      │     │ CORRECT/AMBIG/ │                                     │
│  └─────────────────┘     │ INCORRECT      │                                     │
│                          └───────┬────────┘                                     │
│                                  │                                               │
│  ┌─────────────────┐             │                                               │
│  │ PHASE 6: Deep   │◄────────────┤ PROCEED                                      │
│  │ Reading         │             │                                               │
│  │ (Technical)     │◄── enable_deep_reading=TRUE                                │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 7: Tech   │ ◄── enable_technical_docs=TRUE                             │
│  │ Docs + HSEA     │     enable_symptom_entry=TRUE                              │
│  │ + Symptom Entry │     enable_structured_causal_chain=TRUE                    │
│  │                 │     traversal_mode="semantic_astar"                        │
│  │ Query: symptom──┼───► INDICATES → error → RESOLVED_BY → remedy               │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 8: Synth  │ ◄── Model selection by complexity                          │
│  │ (Enhanced)      │     Thinking model if complex/expert                       │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 9: Cross  │ ◄── enable_cross_domain_validation=TRUE                    │
│  │ Domain Valid    │     enable_entity_grounding=TRUE                           │
│  │ - Chain Extract │     severity_threshold="warning"                           │
│  │ - Relationship  │                                                            │
│  │   Validation    │     Invalid: "servo→hydraulic" ✗                           │
│  │ - Entity Ground │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ├── Critical issues?                                                   │
│           │   ├── YES ──► Revise synthesis                                      │
│           │   └── NO ───┐                                                        │
│           │             │                                                        │
│           ▼             ▼                                                        │
│  ┌─────────────────┐                                                            │
│  │ PHASE 10: Self  │ ◄── Full Self-RAG                                          │
│  │ Reflection      │     + Temporal check                                       │
│  └────────┬────────┘     + Cross-domain check                                   │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 11: RAGAS │ ◄── enable_ragas=TRUE                                      │
│  │ Evaluation      │     Faithfulness + Relevancy scoring                       │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│         END                                                                      │
│                                                                                  │
│  ADDITIONAL ENABLED (beyond BALANCED):                                          │
│    ✓ hyde                   ✓ cross_encoder        ✓ ragas                      │
│    ✓ context_curation       ✓ entity_tracking      ✓ thought_library            │
│    ✓ technical_docs         ✓ hsea_context         ✓ symptom_entry              │
│    ✓ structured_causal_chain ✓ cross_domain_validation ✓ entity_grounding       │
│    ✓ deep_reading           ✓ mixed_precision      ✓ entity_enhanced_retrieval  │
│    ✓ embedding_aggregator   ✓ domain_corpus                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**ENHANCED Operation Nodes:**
| Node | Operation | Conditional | Decision Point |
|------|-----------|-------------|----------------|
| N1 | Query Analysis | - | requires_search? |
| N2 | Entity Extraction | - | - |
| N3 | Thought Template Retrieval | - | - |
| N4 | HyDE Expansion | - | - |
| N5 | Entity-Enhanced Search | - | - |
| N6 | BGE-M3 + Cross-Encoder | - | - |
| N7 | Context Curation (DIG) | - | - |
| N8 | CRAG Evaluation | - | quality level? |
| N9 | Deep Reading | - | - |
| N10 | Technical Docs (HSEA) | - | - |
| N11 | Synthesis | - | - |
| N12 | Cross-Domain Validation | - | critical issues? |
| N13 | Self-Reflection | - | needs_refinement? |
| N14 | RAGAS Evaluation | - | - |

---

### 12.4 RESEARCH Preset Decision Tree

**Features Enabled:** 35
**Typical Latency:** 120-180s
**Use Case:** Academic/thorough research, maximum quality

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RESEARCH PRESET DECISION TREE                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  START                                                                           │
│    │                                                                             │
│    ▼                                                                             │
│  ┌─────────────────┐                                                            │
│  │ PHASE 1: Query  │ ◄── Full analysis                                          │
│  │ Analysis        │     + DyLAN complexity classification                      │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 1.4:      │ ◄── enable_dylan_agent_skipping=TRUE                       │
│  │ DyLAN Classify  │     Determine skippable agents                             │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 2: Plan   │ ◄── enable_pre_act_planning=TRUE                           │
│  │ (Pre-Act)       │     enable_dynamic_planning=TRUE                           │
│  │ + Meta-Buffer   │     enable_meta_buffer=TRUE                                │
│  │ + Reasoning     │     enable_reasoning_composer=TRUE                         │
│  │   Composer      │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 2.2:      │ ◄── enable_reasoning_dag=TRUE                              │
│  │ Reasoning DAG   │     Initialize multi-path reasoning                        │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 3: Search │ ◄── enable_parallel_execution=TRUE                         │
│  │ (Parallel)      │     Concurrent query execution                             │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 3.1:      │ ◄── enable_query_tree=TRUE                                 │
│  │ Query Tree      │     RQ-RAG tree decoding for expansion                     │
│  │ Expansion       │     Parallel sub-query exploration                         │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 3.2:      │ ◄── enable_flare_retrieval=TRUE                            │
│  │ FLARE Retrieval │     Monitor synthesis for uncertainty                      │
│  │                 │     Trigger proactive retrieval                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  [... All ENHANCED phases ...]                                                  │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 5.9: IB   │ ◄── enable_information_bottleneck=TRUE                     │
│  │ Filtering       │     ib_filtering_level="moderate"                          │
│  │ - Noise reduce  │     Reduce noise, preserve task-relevant                   │
│  │ - Key sentences │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 7: Tech   │ ◄── traversal_mode="flow_based" (PathRAG)                  │
│  │ Docs (PathRAG)  │     technical_max_hops=5                                   │
│  │                 │     technical_beam_width=20                                │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 8: Synth  │ ◄── Check DyLAN skip decision                              │
│  │ (Conditional)   │     May skip if simple + high confidence                   │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 9: Cross  │ ◄── severity_threshold="critical"                          │
│  │ Domain Valid    │     Stricter validation for research                       │
│  │ (Critical)      │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ ENTROPY HALTING │ ◄── enable_entropy_halting=TRUE                            │
│  │ DECISION        │     enable_iteration_bandit=TRUE                           │
│  │ - Entropy check │     UCB action selection                                   │
│  │ - Confidence    │                                                            │
│  │   calibration   │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ├── entropy_low && confidence_high?                                    │
│           │   ├── YES ──► Early termination                                     │
│           │   └── NO ───► Continue iteration                                    │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE L.5:      │ ◄── enable_constraint_verification=TRUE                    │
│  │ Constraint      │     Validate output against directives                     │
│  │ Verification    │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│         END                                                                      │
│                                                                                  │
│  ADDITIONAL ENABLED (beyond ENHANCED):                                          │
│    ✓ entropy_halting        ✓ iteration_bandit     ✓ flare_retrieval            │
│    ✓ query_tree             ✓ semantic_memory      ✓ raise_structure            │
│    ✓ meta_buffer            ✓ reasoning_composer   ✓ reasoning_dag              │
│    ✓ pre_act_planning       ✓ stuck_detection      ✓ parallel_execution         │
│    ✓ contradiction_detection ✓ vision_analysis     ✓ dynamic_planning           │
│    ✓ progress_tracking      ✓ graph_cache          ✓ prefetching                │
│    ✓ kv_cache_service       ✓ artifacts            ✓ dylan_agent_skipping       │
│    ✓ information_bottleneck ✓ contrastive_learning ✓ constraint_verification    │
│    PathRAG: flow_based, max_hops=5, beam_width=20                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 12.5 FULL Preset Decision Tree

**Features Enabled:** 42+
**Typical Latency:** 180-300s
**Use Case:** Maximum capability, exhaustive search

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FULL PRESET DECISION TREE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  START                                                                           │
│    │                                                                             │
│    ▼                                                                             │
│  ┌─────────────────┐                                                            │
│  │ ALL RESEARCH    │ ◄── Everything from RESEARCH preset                        │
│  │ PHASES          │     PLUS:                                                  │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 3.X:      │ ◄── enable_self_consistency=TRUE                           │
│  │ Self-Consistency│     Multi-path answer convergence                          │
│  │ Sampling        │     (expensive but thorough)                               │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 4.X:      │ ◄── enable_memory_tiers=TRUE                               │
│  │ Three-Tier      │     Cold → Warm → Hot memory                               │
│  │ Memory          │     Auto-promotion based on access                         │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 6.X:      │ ◄── enable_vision_analysis=TRUE                            │
│  │ Vision-Language │     Screenshot capture + VL model analysis                 │
│  │ Analysis        │     For JS-heavy pages                                     │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE 7: Tech   │ ◄── traversal_mode="multi_hop"                             │
│  │ Docs (Multi-Hop)│     technical_max_hops=6                                   │
│  │                 │     technical_beam_width=50                                │
│  │ Cross-document  │     Cross-document reasoning                               │
│  │ reasoning       │                                                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ PHASE X: Multi  │ ◄── enable_multi_agent=TRUE                                │
│  │ Agent Coord     │     enable_actor_factory=TRUE                              │
│  │ - Actor Factory │     Dynamic agent creation                                 │
│  │ - Parallel exec │     Parallel specialized agents                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ CONTEXT CURATION│ ◄── context_curation_preset="technical"                    │
│  │ (Technical)     │     Maximum precision filtering                            │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ IB FILTERING    │ ◄── ib_filtering_level="aggressive"                        │
│  │ (Aggressive)    │     Maximum compression                                    │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐                                                            │
│  │ LLM DEBUG       │ ◄── enable_llm_debug=TRUE                                  │
│  │ EVENTS          │     Detailed LLM call events                               │
│  └────────┬────────┘                                                            │
│           │                                                                      │
│           ▼                                                                      │
│         END                                                                      │
│                                                                                  │
│  ADDITIONAL ENABLED (beyond RESEARCH):                                          │
│    ✓ self_consistency       ✓ memory_tiers         ✓ actor_factory              │
│    ✓ multi_agent            ✓ llm_debug                                         │
│    Multi-hop: max_hops=6, beam_width=50                                         │
│    Context curation: "technical" preset                                         │
│    IB filtering: "aggressive"                                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Complete Feature Matrix by Preset

### 13.1 Core & Quality Features (Layers 1-2)

| Feature | MIN | BAL | ENH | RES | FULL | Layer |
|---------|-----|-----|-----|-----|------|-------|
| enable_query_analysis | ✅ | ✅ | ✅ | ✅ | ✅ | Core |
| enable_verification | ❌ | ✅ | ✅ | ✅ | ✅ | Core |
| enable_scratchpad | ❌ | ✅ | ✅ | ✅ | ✅ | Core |
| enable_self_reflection | ❌ | ✅ | ✅ | ✅ | ✅ | L1 |
| enable_crag_evaluation | ❌ | ✅ | ✅ | ✅ | ✅ | L1 |
| enable_sufficient_context | ❌ | ✅ | ✅ | ✅ | ✅ | L1 |
| enable_positional_optimization | ❌ | ✅ | ✅ | ✅ | ✅ | L1 |
| enable_experience_distillation | ❌ | ✅ | ✅ | ✅ | ✅ | L1 |
| enable_classifier_feedback | ❌ | ✅ | ✅ | ✅ | ✅ | L1 |
| enable_adaptive_refinement | ❌ | ✅ | ✅ | ✅ | ✅ | L1.5 |
| enable_answer_grading | ❌ | ✅ | ✅ | ✅ | ✅ | L1.5 |
| enable_gap_detection | ❌ | ✅ | ✅ | ✅ | ✅ | L1.5 |

### 13.2 Performance & Retrieval Features (Layer 2)

| Feature | MIN | BAL | ENH | RES | FULL | Layer |
|---------|-----|-----|-----|-----|------|-------|
| enable_content_cache | ✅ | ✅ | ✅ | ✅ | ✅ | L2 |
| enable_semantic_cache | ❌ | ✅ | ✅ | ✅ | ✅ | L2 |
| enable_ttl_pinning | ❌ | ✅ | ✅ | ✅ | ✅ | L2 |
| enable_kv_cache_service | ❌ | ❌ | ❌ | ✅ | ✅ | L2 |
| enable_memory_tiers | ❌ | ❌ | ❌ | ❌ | ✅ | L2 |
| enable_artifacts | ❌ | ❌ | ❌ | ✅ | ✅ | L2 |
| enable_hyde | ❌ | ❌ | ✅ | ✅ | ✅ | L2 |
| enable_hybrid_reranking | ❌ | ✅ | ✅ | ✅ | ✅ | L2 |
| enable_cross_encoder | ❌ | ❌ | ✅ | ✅ | ✅ | L2 |
| enable_mixed_precision | ❌ | ❌ | ✅ | ✅ | ✅ | L2 |
| enable_entity_enhanced_retrieval | ❌ | ❌ | ✅ | ✅ | ✅ | L2 |
| enable_ragas | ❌ | ❌ | ✅ | ✅ | ✅ | L2 |
| enable_context_curation | ❌ | ❌ | ✅ | ✅ | ✅ | L2 |
| enable_entropy_halting | ❌ | ❌ | ❌ | ✅ | ✅ | L2 |
| enable_iteration_bandit | ❌ | ❌ | ❌ | ✅ | ✅ | L2 |
| enable_self_consistency | ❌ | ❌ | ❌ | ❌ | ✅ | L2 |
| enable_flare_retrieval | ❌ | ❌ | ❌ | ✅ | ✅ | L2 |
| enable_query_tree | ❌ | ❌ | ❌ | ✅ | ✅ | L2 |

### 13.3 Advanced Reasoning & Domain Features (Layer 3)

| Feature | MIN | BAL | ENH | RES | FULL | Layer |
|---------|-----|-----|-----|-----|------|-------|
| enable_semantic_memory | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_raise_structure | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_meta_buffer | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_reasoning_composer | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_entity_tracking | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_thought_library | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_reasoning_dag | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_domain_corpus | ❌ | ✅ | ✅ | ✅ | ✅ | L3 |
| enable_embedding_aggregator | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_technical_docs | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_hsea_context | ❌ | ✅ | ✅ | ✅ | ✅ | L3 |
| enable_symptom_entry | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_structured_causal_chain | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_cross_domain_validation | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_entity_grounding | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |
| enable_pre_act_planning | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_stuck_detection | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_parallel_execution | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_contradiction_detection | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_vision_analysis | ❌ | ❌ | ❌ | ✅ | ✅ | L3 |
| enable_deep_reading | ❌ | ❌ | ✅ | ✅ | ✅ | L3 |

### 13.4 Multi-Agent & Graph Features (Layer 4)

| Feature | MIN | BAL | ENH | RES | FULL | Layer |
|---------|-----|-----|-----|-----|------|-------|
| enable_dynamic_planning | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_progress_tracking | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_actor_factory | ❌ | ❌ | ❌ | ❌ | ✅ | L4 |
| enable_multi_agent | ❌ | ❌ | ❌ | ❌ | ✅ | L4 |
| enable_dylan_agent_skipping | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_information_bottleneck | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_contrastive_learning | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_constraint_verification | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_graph_cache | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_prefetching | ❌ | ❌ | ❌ | ✅ | ✅ | L4 |
| enable_llm_debug | ❌ | ❌ | ❌ | ❌ | ✅ | L4 |

### 13.5 Diagnostic Path Configuration by Preset

| Parameter | MIN | BAL | ENH | RES | FULL |
|-----------|-----|-----|-----|-----|------|
| traversal_mode | - | - | semantic_astar | flow_based | multi_hop |
| max_hops | - | - | 4 | 5 | 6 |
| beam_width | - | - | 10 | 20 | 50 |
| severity_threshold | - | - | warning | critical | critical |
| context_curation_preset | - | - | balanced | thorough | technical |
| ib_filtering_level | - | - | - | moderate | aggressive |

---

## 14. Strategic Orchestration Recommendations

### 14.1 Query-to-Pipeline Routing Optimization

The current architecture has **three levels of routing decisions**:

```
Level 1: Gateway Classification (coarse)
    │
    └── Determines: direct_answer | web_search | agentic_search | code_assistant

Level 2: Preset Selection (medium)
    │
    └── Determines: MINIMAL | BALANCED | ENHANCED | RESEARCH | FULL

Level 3: Feature Flags (fine)
    │
    └── 50+ individual enable_* flags within preset
```

**Current Issue:** Level 1 and Level 2 are not well-coordinated.

### 14.2 Recommended Routing Strategy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDED QUERY ROUTING STRATEGY                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Query → Classification                                                          │
│            │                                                                     │
│            ├── SIMPLE + factual ──────────────► DIRECT_ANSWER                   │
│            │                                    (no preset needed)               │
│            │                                                                     │
│            ├── MODERATE + current_info ───────► WEB_SEARCH                      │
│            │                                    + BALANCED preset                │
│            │                                                                     │
│            ├── MODERATE + research ───────────► AGENTIC_SEARCH                  │
│            │                                    + BALANCED preset                │
│            │                                                                     │
│            ├── COMPLEX + research ────────────► AGENTIC_SEARCH                  │
│            │                                    + ENHANCED preset                │
│            │                                                                     │
│            ├── EXPERT + troubleshooting ──────► AGENTIC_SEARCH                  │
│            │                                    + RESEARCH preset                │
│            │                                    + thinking_model=TRUE            │
│            │                                                                     │
│            └── EXPERT + multi_faceted ────────► AGENTIC_SEARCH                  │
│                                                 + FULL preset                    │
│                                                 + thinking_model=TRUE            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 14.3 Text Transformation Pipeline Views

The agentic pipeline can be viewed as a **text transformation system**:

```
                        TEXT TRANSFORMATION PIPELINE

Input: Raw Query
   │
   ▼
┌─────────────┐
│ EXPANSION   │  Query → [Expanded Query, HyDE Docs, Sub-Questions]
│ (Phase 1-2) │  Transform: 1 query → N queries
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ RETRIEVAL   │  Queries → [Search Results, Scraped Content]
│ (Phase 3-6) │  Transform: N queries → M documents
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ FILTERING   │  M documents → K curated documents (K << M)
│ (Phase 4-5) │  Transform: Dedup, DIG, CRAG, IB filtering
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ ENRICHMENT  │  K documents + Domain Knowledge
│ (Phase 7)   │  Transform: Add FANUC/industrial context
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ SYNTHESIS   │  [K documents + context] → Answer
│ (Phase 8)   │  Transform: Compression + Generation
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ VALIDATION  │  Answer → Validated Answer
│ (Phase 9-11)│  Transform: Quality gates + Revision
└──────┬──────┘
       │
       ▼
Output: Validated Answer with Citations
```

### 14.4 Bottleneck Identification

Based on the decision trees, key bottlenecks are:

| Bottleneck | Location | Impact | Mitigation |
|------------|----------|--------|------------|
| **LLM Calls** | Phases 1, 5, 8, 9, 10, 11 | 60-70% of latency | DyLAN skipping, caching |
| **Web Search** | Phase 3 | 10-20% of latency | Parallel queries |
| **Scraping** | Phase 6 | 10-15% of latency | Selective deep reading |
| **Reranking** | Phase 3.5 | 5-10% of latency | Cross-encoder only when needed |

### 14.5 Feature Correlation Analysis

Features that **should be enabled together**:

| Primary Feature | Correlated Features | Reason |
|-----------------|---------------------|--------|
| `enable_technical_docs` | `enable_hsea_context`, `enable_symptom_entry` | Domain knowledge chain |
| `enable_hyde` | `enable_hybrid_reranking` | HyDE needs good reranking |
| `enable_crag_evaluation` | `enable_adaptive_refinement` | CRAG triggers refinement |
| `enable_entity_tracking` | `enable_entity_grounding`, `enable_cross_domain_validation` | Entity validation chain |
| `enable_reasoning_dag` | `enable_meta_buffer`, `enable_reasoning_composer` | Reasoning template use |
| `enable_entropy_halting` | `enable_iteration_bandit` | Confidence-calibrated stopping |

Features that are **mutually exclusive or redundant**:

| Feature A | Feature B | Relationship |
|-----------|-----------|--------------|
| `enable_self_consistency` | MINIMAL preset | Never together (expensive) |
| `enable_llm_debug` | Production use | Debug only |

---

## 15. Future Orchestration Improvements

### 15.1 Proposed: Dynamic Preset Selection

Instead of fixed presets, dynamically select features based on:
- Query complexity (from classifier)
- Available latency budget
- Domain detection (industrial vs general)
- User preferences

```python
# Proposed dynamic preset selection
def select_dynamic_preset(classification, latency_budget_ms, domain):
    if latency_budget_ms < 30000:
        return "MINIMAL"
    elif domain == "industrial" and classification.complexity == "expert":
        return "RESEARCH"
    elif classification.requires_thinking_model:
        return "ENHANCED"
    else:
        return "BALANCED"
```

### 15.2 Proposed: Feature Cascading

Enable features incrementally based on intermediate results:

```
Start with MINIMAL
    │
    ├── If CRAG quality < 0.5 → Enable HyDE + reranking
    │
    ├── If entity_detected → Enable entity_tracking + grounding
    │
    ├── If confidence < 0.6 → Enable self_reflection + RAGAS
    │
    └── If industrial_domain → Enable technical_docs + HSEA
```

### 15.3 Proposed: Unified Event Emission

Fix the streaming event inconsistency by having `_execute_pipeline()` emit events internally when emitter is set, eliminating the need for separate `search_with_events()`.

---

*Last Updated: 2026-01-08 by Claude Code*
