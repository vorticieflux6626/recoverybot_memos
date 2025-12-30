# Agentic Search Module Overview

> **Updated**: 2025-12-30 | **Parent**: [memOS CLAUDE.md](../../CLAUDE.md) | **Version**: 0.39.0

## Purpose

The agentic search module implements an advanced multi-agent search and synthesis system for the RecoveryBot platform. It provides intelligent web search, content scraping, verification, and LLM-powered synthesis with real-time progress streaming.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UniversalOrchestrator                         │
│                    (Single Source of Truth)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Analyzer │→ │ Planner  │→ │ Searcher │→ │ Verifier │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│       ↓              ↓             ↓             ↓               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Scratchpad (Working Memory)              │       │
│  └──────────────────────────────────────────────────────┘       │
│       ↓                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │Synthesizr│→ │Self-RAG  │→ │  RAGAS   │                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Categories

### Core Orchestration

| Module | File | Purpose |
|--------|------|---------|
| **UniversalOrchestrator** | `orchestrator_universal.py` | Single source of truth for all search pipelines |
| **BaseSearchPipeline** | `base_pipeline.py` | Shared pipeline functionality |
| **Models** | `models.py` | Pydantic models for requests/responses |
| **Events** | `events.py` | SSE event types and helpers |

### Agent Components

| Agent | File | Purpose |
|-------|------|---------|
| **Analyzer** | `analyzer.py` | Query analysis and decomposition |
| **Planner** | `planner.py` | Search strategy planning |
| **Searcher** | `searcher.py` | Web search execution (SearXNG, Brave, DDG) |
| **Verifier** | `verifier.py` | Claim verification |
| **Synthesizer** | `synthesizer.py` | Answer synthesis with sources |
| **Scraper** | `scraper.py` | Web content extraction |

### Quality Assurance

| Module | File | Purpose | Research Basis |
|--------|------|---------|----------------|
| **SelfReflection** | `self_reflection.py` | Post-synthesis evaluation | Self-RAG (arXiv:2310.11511) |
| **RetrievalEvaluator** | `retrieval_evaluator.py` | Pre-synthesis quality | CRAG (arXiv:2401.15884) |
| **RAGASEvaluator** | `ragas.py` | Faithfulness/relevancy scoring | RAGAS (arXiv:2309.15217) |

### Advanced Features

| Module | File | Purpose | Research Basis |
|--------|------|---------|----------------|
| **DynamicPlanner** | `dynamic_planner.py` | Strategic/tactical planning | AIME (ByteDance) |
| **EntityTracker** | `entity_tracker.py` | Entity extraction/tracking | GSW (arXiv:2405.xxxxx) |
| **ReasoningDAG** | `reasoning_dag.py` | Graph-of-thoughts reasoning | GoT (arXiv:2308.09687) |
| **ThoughtLibrary** | `thought_library.py` | Reusable reasoning templates | BoT (arXiv:2406.04271) |
| **ActorFactory** | `actor_factory.py` | Dynamic agent creation | AIME (ByteDance) |

### Retrieval Enhancement

| Module | File | Purpose |
|--------|------|---------|
| **BGEM3HybridRetriever** | `bge_m3_hybrid.py` | Dense + Sparse hybrid search |
| **HyDEExpander** | `hyde.py` | Hypothetical document embeddings |
| **MixedPrecisionEmbeddings** | `mixed_precision_embeddings.py` | Multi-precision embedding search |
| **EntityEnhancedRetriever** | `entity_enhanced_retrieval.py` | Entity-aware retrieval |
| **EmbeddingAggregator** | `embedding_aggregator.py` | Multi-source embedding fusion |

### Domain Knowledge

| Module | File | Purpose |
|--------|------|---------|
| **DomainCorpus** | `domain_corpus.py` | Domain-specific knowledge base |
| **QueryClassifier** | `query_classifier.py` | Query type and complexity detection |
| **SufficientContext** | `sufficient_context.py` | Context sufficiency evaluation |

### Caching & Performance

| Module | File | Purpose |
|--------|------|---------|
| **ContentCache** | `content_cache.py` | Query and content caching |
| **TTLCacheManager** | `ttl_cache_manager.py` | Time-based cache management |
| **KVCacheService** | `kv_cache_service.py` | LLM KV cache management |
| **MemoryTiers** | `memory_tiers.py` | Three-tier memory (hot/warm/cold) |
| **Metrics** | `metrics.py` | Performance tracking |

### Utilities

| Module | File | Purpose |
|--------|------|---------|
| **Prompts** | `prompts.py` | Centralized prompt registry |
| **Artifacts** | `artifacts.py` | Inter-agent artifact storage |
| **Scratchpad** | `scratchpad.py` | Working memory for agents |
| **ProgressTools** | `progress_tools.py` | Progress reporting |

### Recent Additions (Phases 21-26)

| Module | File | Purpose | Research Basis |
|--------|------|---------|----------------|
| **ContextCurator** | `context_curator.py` | Document selection and gap analysis | DIG scoring |
| **RedundancyDetector** | `redundancy_detector.py` | Semantic deduplication | Clustering |
| **EntropyMonitor** | `entropy_monitor.py` | Confidence-based halting | Information theory |
| **SelfConsistencyChecker** | `self_consistency.py` | Multi-synthesis convergence | Self-consistency prompting |
| **IterationBandit** | `iteration_bandit.py` | Adaptive refinement decisions | Multi-armed bandit |
| **FLARERetriever** | `flare_retriever.py` | Active retrieval during synthesis | FLARE (arXiv:2305.06983) |
| **QueryTreeDecoder** | `query_tree.py` | Hierarchical query expansion | Tree-structured QA |
| **SemanticMemory** | `semantic_memory.py` | Concept-linked memory network | Knowledge graphs |
| **RAISEScratchpad** | `raise_scratchpad.py` | Rich working memory | RAISE (arXiv:2024.xxxxx) |
| **MetaBuffer** | `meta_buffer.py` | Template distillation and reuse | Buffer of Thoughts |
| **ReasoningComposer** | `reasoning_composer.py` | Modular reasoning assembly | Chain-of-Thought |

### Industrial Domain Support

| Module | File | Purpose |
|--------|------|---------|
| **HSEAController** | `hsea_controller.py` | Hierarchical semantic search for industrial troubleshooting |
| **FANUCCorpusBuilder** | `fanuc_corpus_builder.py` | FANUC robotics knowledge extraction |
| **IMMCorpusBuilder** | `imm_corpus_builder.py` | Injection molding machine documentation |
| **DocumentGraphService** | `core/document_graph_service.py` | PDF extraction API integration |

## Preset System

The UniversalOrchestrator supports 5 presets with increasing feature sets:

| Preset | Features | Use Case | Android Icon |
|--------|----------|----------|--------------|
| `minimal` | 8 | Fast, simple queries | FlashOn (green) |
| `balanced` | 18 | Default for most queries | Balance (blue) |
| `enhanced` | 28 | Complex research | TrendingUp (purple) |
| `research` | 39 | Academic/thorough | Science (orange) |
| `full` | 42+ | Maximum capability | AllInclusive (pink) |

**Note**: Android app default changed from `full` to `balanced` on 2025-12-29 to reduce default resource consumption.

## SSE Event Flow

Real-time progress streaming via Server-Sent Events:

```
search_started → analyzing_query → query_analyzed → planning_search →
search_planned → iteration_start → searching → search_results →
crag_evaluating → evaluating_urls → scraping_url → url_scraped →
verifying_claims → claims_verified → synthesizing → synthesis_complete →
self_rag_reflecting → self_rag_complete → search_completed
```

## Graph Visualization

Pipeline progress shown as:
```
[A✓]→[P✓]→[S✓]→[E✓]→[W✓]→[V✓]→[Σ✓]→[R✓]→[✓✓]
```

| Symbol | Agent |
|--------|-------|
| A | Analyze |
| P | Plan |
| S | Search |
| E | CRAG Evaluate |
| W | Scrape (Web) |
| V | Verify |
| Σ | Synthesize |
| R | Reflect (Self-RAG) |
| ✓ | Complete |

## API Endpoints

### Primary Endpoints

```bash
# Gateway (auto-routes based on query)
POST /api/v1/search/gateway
POST /api/v1/search/gateway/stream

# Universal Orchestrator
POST /api/v1/search/universal
POST /api/v1/search/universal/stream

# Query Classification
POST /api/v1/search/classify
```

### Metrics & Stats

```bash
GET /api/v1/search/metrics
GET /api/v1/search/cache/stats
GET /api/v1/search/ttl/stats
GET /api/v1/search/artifacts/stats
```

### Domain Corpus

```bash
GET /api/v1/search/corpus/domains
POST /api/v1/search/corpus/{domain_id}/query
GET /api/v1/search/corpus/{domain_id}/troubleshoot/{code}
```

## Configuration

### Environment Variables

```bash
# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_FLASH_ATTENTION=1

# Search
BRAVE_API_KEY=<key>  # Optional
SEARXNG_URL=http://localhost:8888
```

### Default Models

| Purpose | Model |
|---------|-------|
| Query Classification | deepseek-r1:14b |
| Analysis/Synthesis | qwen3:8b |
| Embeddings | mxbai-embed-large |
| Quick Evaluation | gemma3:4b |

## Testing

```bash
# Full system test
./test_system.sh

# Specific tests
./test_system.sh hybrid    # BGE-M3 hybrid retrieval
./test_system.sh hyde      # HyDE query expansion
./test_system.sh ragas     # RAGAS evaluation

# FANUC queries
python -c "
import asyncio
from agentic import UniversalOrchestrator, SearchRequest

async def test():
    orch = UniversalOrchestrator(preset='balanced')
    resp = await orch.search(SearchRequest(
        query='FANUC SRVO-063 alarm resolution',
        max_iterations=3
    ))
    print(f'Sources: {len(resp.data.sources)}, Confidence: {resp.data.confidence_score:.1%}')

asyncio.run(test())
"
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.39.0 | 2025-12-29 | Phase 26: Feature synergy (Query Tree + CRAG, FLARE + Synthesis, Meta-Buffer) |
| 0.38.0 | 2025-12-29 | Phase 25: Context curation (DIG scoring, redundancy detection) |
| 0.37.0 | 2025-12-29 | Phase 24: Confidence-calibrated halting (entropy, self-consistency, bandit) |
| 0.36.0 | 2025-12-29 | Phase 23: Enhanced query generation (FLARE, query tree) |
| 0.35.0 | 2025-12-29 | Phase 22: Scratchpad enhancement (semantic memory, RAISE) |
| 0.34.0 | 2025-12-29 | Phase 21: Template optimization (meta-buffer, reasoning composer) |
| 0.33.0 | 2025-12-29 | HSEA controller for industrial troubleshooting |
| 0.32.0 | 2025-12-29 | IMM corpus builder for injection molding |
| 0.31.0 | 2025-12-29 | PDF API integration via DocumentGraphService |
| 0.30.0 | 2025-12-29 | Adaptive refinement engine |
| 0.29.0 | 2025-12-29 | Base pipeline abstraction |
| 0.28.0 | 2025-12-29 | Orchestrator archival and backward-compat shims |
| 0.27.1 | 2025-12-29 | Android SSE streaming, FANUC testing |
| 0.27.0 | 2025-12-28 | Orchestrator consolidation to UniversalOrchestrator |
| 0.26.0 | 2025-12-28 | Context utilization tracking |
| 0.25.0 | 2025-12-28 | Universal Orchestrator + 17 bug fixes |
| 0.20.0 | 2025-12-27 | SSE graph visualization |
| 0.19.0 | 2025-12-27 | RAGAS evaluation pipeline |
| 0.18.0 | 2025-12-27 | HyDE query expansion |
| 0.17.0 | 2025-12-27 | BGE-M3 hybrid retrieval |
| 0.16.0 | 2025-12-27 | Mixed-precision embeddings |
| 0.14.0 | 2025-12-27 | Domain corpus system |

## Related Documentation

- `memOS/CLAUDE.md` - Full server documentation
- `FANUC_TEST_QUERIES.md` - Test query set
- `KV_CACHE_IMPLEMENTATION_PLAN.md` - KV cache optimization
- `ENHANCEMENT_IMPLEMENTATION_PLAN.md` - Enhancement roadmap
