# memOS Implementation Phases Reference

> **Updated**: 2026-01-02 | **Status**: Reference Documentation | **Parent**: [memOS CLAUDE.md](../../CLAUDE.md)

This document contains detailed implementation documentation for all completed phases of the memOS agentic search system. For current status and quick reference, see the main [CLAUDE.md](../../CLAUDE.md).

---

## Table of Contents

### Core RAG Architecture (Part G)
- [G.1: RAG Foundation](#-part-g1-rag-foundation-completed-2025-12-30)
- [G.2: Hierarchical Retrieval](#-part-g2-hierarchical-retrieval-optimization-completed-2025-12-30)
- [G.6: Agent Coordination](#-part-g6-agent-coordination-completed-2025-12-31)
- [G.7.2: Hyperbolic Embeddings](#-part-g72-hyperbolic-embeddings-completed-2025-12-31)
- [G.7.3: Optimal Transport Fusion](#-part-g73-optimal-transport-fusion-completed-2025-12-31)
- [G.7.4: TSDAE Domain Adaptation](#-part-g74-tsdae-domain-adaptation-completed-2025-12-31)

### Document Processing (Part K)
- [K.2: Docling Document Processor](#-part-k2-docling-document-processor-integration-completed-2026-01-01)
- [K.3: Table Complexity Routing](#-part-k3-table-complexity-routing-completed-2026-01-01)

### Enhancement Phases (1-27)
- [Phase 1: AIME-Style Dynamic Planning](#-phase-1-aime-style-dynamic-planning-completed-2025-12-27)
- [Phase 2: GSW Entity Tracking](#-phase-2-gsw-style-entity-tracking-completed-2025-12-27)
- [Phase 3: DAG-Based Reasoning](#-phase-3-dag-based-reasoning-completed-2025-12-27)
- [Phase 4: Buffer of Thoughts](#-phase-4-buffer-of-thoughts-completed-2025-12-27)
- [Phase 5: Actor Factory](#-phase-5-actor-factory-completed-2025-12-27)
- [Phase 6: Self-RAG Reflection](#-phase-6-self-rag-reflection-completed-2025-12-27)
- [Phase 7: CRAG Retrieval Evaluator](#-phase-7-crag-retrieval-evaluator-completed-2025-12-27)
- [Phase 8: Experience Distillation](#-phase-8-experience-distillation-completed-2025-12-27)
- [Phase 9: Classifier Feedback Loop](#-phase-9-classifier-feedback-loop-completed-2025-12-27)
- [Phase 10: SSE Visibility](#-phase-10-sse-visibility--thorough-search-completed-2025-12-27)
- [Phase 11: Domain Corpus](#-phase-11-domain-specific-persistent-scratchpad-completed-2025-12-27)
- [Phase 12: SSE Graph Visualization](#-phase-12-sse-graph-visualization--enhanced-events-completed-2025-12-27)
- [Phase 13: Universal Orchestrator](#-phase-13-universal-orchestrator--bug-fixes-completed-2025-12-28)
- [Phase 14: Context Utilization](#-phase-14-context-utilization-tracking-completed-2025-12-28)
- [Phase 15: Orchestrator Consolidation](#-phase-15-orchestrator-consolidation-completed-2025-12-28)
- [Phase 16: Android SSE Integration](#-phase-16-android-sse-streaming-integration-completed-2025-12-29)
- [Phase 17: Context Curation](#-phase-17-context-curation-pipeline-completed-2025-12-29)
- [Phase 18: Confidence Halting](#-phase-18-confidence-calibrated-halting-completed-2025-12-29)
- [Phase 19: Query Generation](#-phase-19-enhanced-query-generation-completed-2025-12-29)
- [Phase 20: Scratchpad Enhancement](#-phase-20-scratchpad-enhancement-completed-2025-12-29)
- [Phase 21: Template Optimization](#-phase-21-template-reuse-optimization-completed-2025-12-29)
- [Phase 22: PDF Tools Integration](#-phase-22-pdf-extraction-tools-integration-completed-2025-12-29)
- [Phase 23: HSEA Indexing](#-phase-23-hsea-three-stratum-indexing-completed-2025-12-29)
- [Phase 24: Engineering Remediation](#-phase-24-engineering-remediation-completed-2025-12-29)
- [Phase 25: Feature Audit](#-phase-25-feature-combination-audit-completed-2025-12-29)
- [Phase 26: Feature Synergy](#-phase-26-feature-synergy-integration-completed-2025-12-29)
- [Phase 27: FANUC Ingestion](#-phase-27-fanuc-ingestion-pipeline-design-research-complete-2025-12-30)

### Other Components
- [Part F: Benchmark Suite](#-part-f-benchmark-test-suite-completed-2025-12-30)
- [B.10: Gateway Integration](#-b10-llm-gateway-client-integration-completed-2026-01-01)

---

#### ✅ Part G.7.2: Hyperbolic Embeddings (Completed 2025-12-31)

Based on HyperbolicRAG (arXiv:2511.18808) achieving +5.6% Recall@5 via Poincaré ball geometry.

**Implementation Files:**
- `agentic/hyperbolic_embeddings.py` (~744 lines): Core Poincaré ball geometry and retriever
- `tests/unit/test_hyperbolic_embeddings.py` (25 tests): Comprehensive test suite

**Key Components:**
| Component | Description |
|-----------|-------------|
| `PoincareBall` | Poincaré ball manifold with exp/log maps and geodesic distance |
| `HyperbolicRetriever` | Hierarchical document retrieval with hyperbolic embeddings |
| `HyperbolicDocument` | Document with Euclidean and hyperbolic embeddings |
| `HierarchyLevel` | Enum: CORPUS→MANUAL→CHAPTER→SECTION→PROCEDURE→STEP |

**Hierarchy-Aware Depth Encoding:**
| Level | Value | Radial Position | Description |
|-------|-------|-----------------|-------------|
| CORPUS | 0 | Near origin | Entire corpus (most general) |
| MANUAL | 1 | 0.1-0.26 | Manual/document level |
| CHAPTER | 2 | 0.26-0.42 | Chapter/major section |
| SECTION | 3 | 0.42-0.58 | Subsection |
| PROCEDURE | 4 | 0.58-0.74 | Procedure/topic |
| STEP | 5 | Near boundary | Individual step (most specific) |

**Score Fusion:**
```python
fused_score = euclidean_weight * cosine_similarity + hyperbolic_weight * exp(-geodesic_distance)
# Default: 50% Euclidean + 50% Hyperbolic
```

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hyperbolic/stats` | GET | Retriever statistics |
| `/hyperbolic/index` | POST | Index with hyperbolic embedding |
| `/hyperbolic/search` | POST | Hierarchy-aware search |
| `/hyperbolic/search-by-hierarchy` | POST | Multi-level search |
| `/hyperbolic/tree/{doc_id}` | GET | Document hierarchy tree |

**Test Results:** 25/25 tests passing

**Module Version**: `agentic/__init__.py` → v0.72.0

#### ✅ Part G.7.3: Optimal Transport Fusion (Completed 2025-12-31)

Based on Wasserstein distance and Sinkhorn algorithm for superior dense-sparse alignment.

**Implementation Files:**
- `agentic/optimal_transport.py` (~700 lines): Core OT implementation
- `tests/unit/test_optimal_transport.py` (31 tests): Comprehensive test suite

**Key Components:**
| Component | Description |
|-----------|-------------|
| `SinkhornSolver` | Entropy-regularized OT with Sinkhorn-Knopp algorithm |
| `GromovWassersteinSolver` | Cross-domain alignment for heterogeneous spaces |
| `OptimalTransportFusion` | Main class for dense-sparse retrieval fusion |
| `OTConfig` | Configuration for epsilon, weights, cost metrics |

**Fusion Methods:**
| Method | Description |
|--------|-------------|
| `fuse_scores()` | Two-way OT fusion (dense + sparse) |
| `fuse_multiway()` | Wasserstein barycenter for multiple retrievers |
| `align_heterogeneous()` | Gromov-Wasserstein cross-domain alignment |

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ot/stats` | GET | OT fusion statistics |
| `/ot/fuse` | POST | Two-way OT fusion |
| `/ot/fuse-multiway` | POST | Multi-way barycentric fusion |
| `/ot/wasserstein-distance` | POST | Compute Wasserstein distance |

**Test Results:** 57/57 tests passing (26 new tests for SW + WMD)

**Module Version**: `agentic/__init__.py` → v0.74.0

#### ✅ Part G.7.4: TSDAE Domain Adaptation (Completed 2025-12-31)

Based on Wang, Reimers, Gurevych (EMNLP 2021): "TSDAE: Using Transformer-based Sequential
Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning".

**Implementation Files:**
- `agentic/tsdae_adapter.py` (~530 lines): TSDAE domain adapter with multi-domain support
- `tests/unit/test_tsdae_adapter.py` (34 tests): Comprehensive test suite including training tests

**Key Components:**
| Component | Description |
|-----------|-------------|
| `TSDaeAdapter` | Main adapter for training and encoding |
| `MultiDomainAdapter` | Combines embeddings from multiple domains |
| `DomainConfig` | Configuration for domain adaptation tasks |
| `AdaptationResult` | Training outcome with metrics |

**Key Features:**
| Feature | Description |
|---------|-------------|
| Unsupervised Adaptation | No labeled data needed - works with raw domain text |
| Denoising Objective | Corrupts sentences and trains encoder to reconstruct |
| 93.1% Performance | Achieves up to 93.1% of supervised fine-tuning |
| Minimal Data | Requires only ~10K domain sentences |
| Incremental Updates | Add new domain data without full retraining |

**Predefined Domain Configs:**
| Config | Domain | Description |
|--------|--------|-------------|
| `FANUC_DOMAIN_CONFIG` | FANUC Robotics | Error codes, procedures, components |
| `SIEMENS_DOMAIN_CONFIG` | Siemens PLC | PLC alarms, ladder logic |
| `ROCKWELL_DOMAIN_CONFIG` | Rockwell/Allen-Bradley | Controller faults, I/O |

**Noise Types:**
- DELETE (default, 0.6 ratio): Random word deletion
- SWAP: Random word swapping
- INSERT: Random word insertion
- SUBSTITUTE: Random word substitution

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tsdae/stats` | GET | Adapter statistics |
| `/tsdae/train` | POST | Train domain adapter |
| `/tsdae/encode` | POST | Encode text with domain adapter |
| `/tsdae/domains` | GET | List available domains |

**Usage:**
```python
from agentic import TSDaeAdapter, DomainConfig, FANUC_DOMAIN_CONFIG

adapter = TSDaeAdapter()

# Train on FANUC documentation
result = await adapter.train_adapter(
    sentences=fanuc_sentences,  # List of domain sentences
    config=FANUC_DOMAIN_CONFIG
)

# Encode with domain-adapted model
embeddings = await adapter.encode(
    ["SRVO-063 alarm detected"],
    domain_id="fanuc"
)
```

**Research Reference:**
- arXiv: https://arxiv.org/abs/2104.06979
- EMNLP 2021

**Test Results:** 34/34 tests passing

**Module Version**: `agentic/__init__.py` → v0.75.0

#### ✅ Part K.2: Docling Document Processor Integration (Completed 2026-01-01)

Based on arXiv:2408.09869 - Docling Technical Report with 97.9% TEDS-S table extraction accuracy.

**Implementation Files:**
- `agentic/docling_adapter.py` (~525 lines): Core Docling adapter with circuit breaker
- `api/search.py`: 6 new API endpoints

**Key Components:**
| Component | Description |
|-----------|-------------|
| `DoclingAdapter` | Async HTTP client with circuit breaker pattern |
| `DoclingFormat` | Output formats: markdown, json, text, html |
| `ExtractionQuality` | Quality levels: fast, standard, accurate |
| `TableData` | Extracted table with structure metadata |
| `ExtractedDocument` | Full document with content, tables, sections |

**Key Features:**
| Feature | Description |
|---------|-------------|
| Table Extraction | 97.9% TEDS-S accuracy via TableFormer |
| Multi-Format | PDF, HTML, DOCX, PPTX, images |
| Circuit Breaker | Auto-open after 5 failures, 60s timeout |
| LRU Cache | 100-item document cache |
| Complex Detection | Auto-detect multi-level headers, merged cells |

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docling/health` | GET | Health check with circuit breaker status |
| `/docling/stats` | GET | Adapter statistics and cache metrics |
| `/docling/convert` | POST | Convert document to markdown/json/text/html |
| `/docling/extract-tables` | POST | Extract tables with TableFormer |
| `/docling/is-complex` | POST | Check if document needs Docling |
| `/docling/cache` | DELETE | Clear adapter cache |

**Usage:**
```python
from agentic import DoclingAdapter, DoclingFormat, ExtractionQuality, get_docling_adapter

adapter = get_docling_adapter()

# Convert document
result = await adapter.convert(
    source="https://example.com/manual.pdf",
    output_format=DoclingFormat.MARKDOWN,
    quality=ExtractionQuality.ACCURATE,
    extract_tables=True
)

# Extract only tables
tables = await adapter.extract_tables(source="manual.pdf")

# Check complexity
is_complex = await adapter.is_complex_document(source="document.html")
```

**Docker Configuration:**
```bash
# Start Docling service (optional profile)
docker compose --profile docling up -d
```

**Research Reference:**
- arXiv: https://arxiv.org/abs/2408.09869
- TableFormer for complex table structure recognition

**Module Version**: `agentic/__init__.py` → v0.76.0

#### ✅ Part K.3: Table Complexity Routing (Completed 2026-01-01)

Routes complex tables to Docling for high-accuracy extraction (97.9% TEDS-S) based on structural complexity scoring.

**Implementation Files:**
- `agentic/table_complexity.py` (~400 lines): TableComplexityScorer with weighted scoring
- `agentic/scraper.py` (lines 438-443): Integration after VL scraper fallback

**Key Components:**
| Component | Description |
|-----------|-------------|
| `TableComplexityScorer` | Analyzes HTML for table complexity indicators |
| `TableComplexity` | Enum: SIMPLE, MODERATE, COMPLEX, VERY_COMPLEX |
| `ComplexityResult` | Analysis result with score and reasons |
| `TableInfo` | Per-table details (rows, cols, merged cells, etc.) |

**Complexity Scoring Weights:**
| Indicator | Weight | Cap |
|-----------|--------|-----|
| Merged cells (colspan/rowspan) | +0.25 per occurrence | 1.0 |
| Multi-level headers | +0.40 | - |
| Nested tables | +0.50 | - |
| Large tables (>20 rows) | +0.20 | - |
| Very large tables (>50 rows) | +0.30 | - |
| Wide tables (>10 columns) | +0.20 | - |
| High empty cell ratio (>30%) | +0.15 | - |
| Technical patterns (FANUC codes) | +0.02 per match | 0.20 |

**Complexity Thresholds:**
| Score | Level | Docling? |
|-------|-------|----------|
| < 0.3 | SIMPLE | No |
| 0.3-0.5 | MODERATE | Optional |
| 0.5-0.7 | COMPLEX | Recommended |
| ≥ 0.7 | VERY_COMPLEX | Required |

**Technical Patterns Detected:**
- Error codes: `SRVO-063`, `MOTN-023`
- Parameters: `$PARAM_GROUP.$ITEM`
- Axes: `J1`, `J2`, `A1`
- Part numbers: `A06B-6079-H101`
- Measurements: `10.5mm`, `45°`

**Integration Flow:**
```
HTML Scraped → Check enable_docling flag
    ↓
[TableComplexityScorer.analyze_html()]
    ↓
score ≥ 0.5? → [DoclingAdapter.convert()]
    ↓             ↓
No             Yes: Re-extract with TableFormer
    ↓             ↓
Use original   Use Docling result (97.9% accuracy)
```

**Usage:**
```python
from agentic import TableComplexityScorer, get_table_complexity_scorer

scorer = get_table_complexity_scorer()
should_use, result = scorer.should_use_docling(html_content, "html")

if should_use:
    print(f"Complex tables detected (score: {result.overall_score:.2f})")
    print(f"Reasons: {result.reasons}")
```

**Test Results:**
- Simple tables: NOT routed to Docling ✅
- Complex FANUC tables (merged + multi-header): Routed with score 1.00 ✅
- Large tables only: NOT routed (score 0.20 < threshold) ✅

**Research Basis:**
- arXiv:2408.09869 - Docling Technical Report
- TableFormer for multi-level header recognition
- TEDS-S benchmark for table structure accuracy

**Module Version**: `agentic/__init__.py` → v0.78.0

#### ✅ B.10: LLM Gateway Client Integration (Completed 2026-01-01)

Unified routing layer integration between memOS agentic agents and LLM backends (Ollama, vLLM).

**Implementation Files:**
- `agentic/gateway_client.py` (~700 lines): Core gateway client with fallback
- `agentic/synthesizer.py`: Gateway integration for synthesis
- `agentic/query_classifier.py`: Gateway integration for classification
- `agentic/analyzer.py`: Gateway integration for query analysis
- `agentic/self_reflection.py`: Gateway integration for Self-RAG reflection
- `agentic/retrieval_evaluator.py`: Gateway integration for CRAG evaluation

**Key Components:**
| Component | Description |
|-----------|-------------|
| `GatewayClient` | Async HTTP client with circuit breaker pattern |
| `LogicalModel` | Enum mapping purpose to physical models |
| `GatewayResponse` | Unified response with fallback tracking |
| `get_gateway_client()` | Singleton factory function |

**Logical Model Mappings:**
| Logical Name | Purpose | Default Model |
|--------------|---------|---------------|
| SYNTHESIZER | Answer synthesis | llama3.3:70b |
| SYNTHESIZER_FAST | Fast synthesis | qwen3:8b |
| ANALYZER | Query analysis | gemma3:4b |
| CLASSIFIER | Query classification | qwen3:8b |
| THINKING | Complex reasoning | deepseek-r1:14b |
| VERIFIER | Claim verification | gemma3:4b |
| REFLECTOR | Self-RAG reflection | gemma3:4b |
| VISION | Image processing | qwen3-vl:7b |
| EMBEDDINGS | Text embeddings | mxbai-embed-large |

**Feature Flag:**
```python
enable_gateway_routing: bool = False  # Route LLM calls through gateway (port 8100)
```

**Usage:**
```python
from agentic import get_gateway_client, LogicalModel

gateway = get_gateway_client()
response = await gateway.generate(
    prompt="Synthesize these findings...",
    model=LogicalModel.SYNTHESIZER,
    timeout=60.0,
    options={"temperature": 0.7}
)

if response.fallback_used:
    logger.info(f"Used fallback model: {response.model}")
```

**Integrated Agents:**
| Agent | Status |
|-------|--------|
| Synthesizer | ✅ Complete |
| QueryClassifier | ✅ Complete |
| Analyzer | ✅ Complete |
| SelfReflection | ✅ Complete |
| RetrievalEvaluator | ✅ Complete |

**Module Version**: `agentic/__init__.py` → v0.80.0

#### ✅ Part G.2: Hierarchical Retrieval Optimization (Completed 2025-12-30)

Phase 2 of 8-week RAG Architecture Improvement Roadmap:

**New Components:**
- **`agentic/cascade_retriever.py`** (~500 lines): FunnelRAG-style cascade retrieval
- **`agentic/fusion_weight_adapter.py`** (~400 lines): Query intent classification
- **`agentic/qdrant_storage.py`** (~560 lines): Qdrant vector storage with VRAM optimization
- **`agentic/adaptive_topk.py`** (~500 lines): CAR algorithm for dynamic k selection

**G.2.1-G.2.2 Cascade Retriever:**
| Feature | Description |
|---------|-------------|
| Binary Oversampling | 3x oversampling with rescoring for coarse retrieval |
| MRL Cascade | 64→256→1024→4096 dimension progression |
| Early Exit | Score threshold, entropy, ranking stability checks |
| Promotion Thresholds | Binary→Int8: 0.65, Int8→FP16: 0.80 |

**G.2.3 Query Intent Classifier:**
| Intent | Sparse Weight | Dense Weight | Use Case |
|--------|---------------|--------------|----------|
| ERROR_CODE | 0.70 | 0.20 | Exact matching critical |
| PART_NUMBER | 0.80 | 0.15 | Alphanumeric lookup |
| TROUBLESHOOTING | 0.30 | 0.60 | Semantic understanding |
| CONCEPTUAL | 0.25 | 0.65 | Concept explanation |
| PROCEDURE | 0.40 | 0.50 | Step-by-step guides |

**G.2.4 Qdrant Storage Configurations:**
| Config | Quantization | HNSW m | Use Case |
|--------|--------------|--------|----------|
| VRAM_EFFICIENT | Scalar (4x) | 16 | Default, balanced |
| MAXIMUM_COMPRESSION | Binary (32x) | 12 | Large collections |
| BALANCED | Scalar (4x) | 32 | Higher recall |

**G.2.5 Adaptive Top-K (CAR Algorithm):**
| Complexity | Base K | Description |
|------------|--------|-------------|
| SIMPLE | 10 | Single concept, direct lookup |
| MODERATE | 25 | Multi-concept, some reasoning |
| COMPLEX | 50 | Multi-hop, synthesis required |
| EXPLORATORY | 100 | Open-ended, research-style |

**Early Stopping Reasons:**
- `score_cliff`: Sharp score drop (>15%)
- `score_plateau`: Consecutive similar scores
- `confidence_threshold`: Top score >0.90
- `diversity_saturated`: No new information (knee point)

**Module Version**: `agentic/__init__.py` → v0.50.0

#### ✅ Part G.1: RAG Foundation (Completed 2025-12-30)

Phase 1 of 8-week RAG Architecture Improvement Roadmap:

**New Components:**
- **`agentic/cross_encoder_reranker.py`** (~300 lines): bge-reranker-v2-m3 reranking
- **`agentic/redis_embeddings_cache.py`** (~400 lines): 3-tier embeddings cache
- **`agentic/tracing.py`** (~350 lines): OpenTelemetry instrumentation
- **`agentic/deepeval_integration.py`** (~450 lines): RAG evaluation metrics

**G.1.1 BGE-M3 ColBERT Mode:**
- Enabled `return_colbert_vecs=True` for late interaction
- FP16 precision with `use_fp16=True`
- Hybrid scoring: `0.4*dense + 0.3*sparse + 0.3*colbert`

**G.1.2 Redis 3-Tier Cache:**
| Tier | Precision | MRL Dim | TTL | Compression |
|------|-----------|---------|-----|-------------|
| Hot | Binary | 64 | Session | 32x |
| Warm | Int8 | 256 | 24h | 4x |
| Cold | FP16 | 4096 | On-demand | 1x |

**G.1.3 OpenTelemetry Tracing:**
- Span tracking for all agentic operations
- Automatic latency measurement
- Integration with Jaeger/Zipkin exporters

**G.1.5 DeepEval Metrics:**
| Metric | Description |
|--------|-------------|
| Faithfulness | Claims supported by context |
| Answer Relevancy | Addresses the question |
| Hallucination | Unsupported claims detected |
| Context Precision | Relevant context ranked higher |

**G.1.6 Cross-Encoder Reranking:**
- Model: `BAAI/bge-reranker-v2-m3`
- Reranks top-50 → top-10
- Latency: 100-300ms per batch

**Module Version**: `agentic/__init__.py` → v0.46.0

#### ✅ Part G.6: Agent Coordination (Completed 2025-12-31)

Phase 6 of 8-week RAG Architecture Improvement Roadmap:

**New Components:**
- **`agentic/semantic_memory.py`** (~1018 lines): A-MEM with SQLite persistence
- **`agentic/information_bottleneck.py`** (~550 lines): IB theory-based noise filtering
- **`agentic/contrastive_retriever.py`** (~600 lines): Trial-and-feedback learning
- **`agentic/dylan_agent.py`** (~400 lines): Query complexity classification

**G.6.1 A-MEM Semantic Memory:**
| Feature | Description |
|---------|-------------|
| SQLite Persistence | Cross-session memory in `data/semantic_memory.db` |
| Auto-Connection | Similarity-based links (threshold 0.7) |
| Access Tracking | Ebbinghaus decay + access count |
| Graph Traversal | BFS with strength-weighted paths |

**G.6.2 DyLAN Agent Importance Scores:**
| Complexity | Skippable Agents | Use Case |
|------------|------------------|----------|
| SIMPLE | Verifier, Reflector | Direct lookups |
| MODERATE | Reflector | Multi-step queries |
| COMPLEX | None | Research queries |

**G.6.4 Information Bottleneck Filtering:**
| Content Type | IB Score | Action |
|--------------|----------|--------|
| ESSENTIAL | ≥0.7 | Keep, extract key sentences |
| SUPPORTING | 0.4-0.7 | Keep if space allows |
| PERIPHERAL | 0.2-0.4 | Filter if compressing |
| NOISE | <0.2 | Always filter |

**G.6.5 Contrastive Retriever Training:**
- Session recording with cited URL tracking
- Contrastive pairs from ranking mistakes
- Per-domain weight learning
- Strategy performance statistics

**Module Version**: `agentic/__init__.py` → v0.40.0

#### ✅ Part G.6-SSE: G.6 Streaming Integration (Completed 2025-12-31)

Added G.6 Agent Coordination features to `search_with_events()` streaming method:

**Features Added to Streaming:**
| Feature | Location | SSE Events |
|---------|----------|------------|
| DyLAN Classification | After PHASE 1 | `dylan_complexity_classified` |
| DyLAN Agent Skipping | PHASE 5, 7 | `dylan_agent_skipped` |
| IB Filtering | PHASE 5.9 | `ib_filtering_start`, `ib_filtering_complete` |
| Contrastive Learning | PHASE 9.5 | `contrastive_session_recorded` |

**SSE Event Flow (Extended):**
```
... → dylan_complexity_classified → ... → dylan_agent_skipped (optional)
→ ib_filtering_start → ib_filtering_complete → synthesizing → ...
→ contrastive_session_recorded → search_completed
```

**Preset Configuration:**
| Preset | DyLAN | IB Filter | Contrastive |
|--------|-------|-----------|-------------|
| minimal | ❌ | ❌ | ❌ |
| balanced | ❌ | ❌ | ❌ |
| enhanced | ❌ | ❌ | ❌ |
| research | ✅ | ✅ | ✅ |
| full | ✅ | ✅ | ✅ |

**Test Results:**
- All 22 G.6 unit tests passing
- Streaming/non-streaming feature parity achieved
- SSE events properly emitted for Android client

**Module Version**: `agentic/__init__.py` → v0.71.0

#### ✅ Part F: Benchmark Test Suite (Completed 2025-12-30)

Comprehensive domain-specific benchmark system for FANUC troubleshooting quality evaluation:

**New Components:**
- **`agentic/benchmark.py`** (~900 lines): Complete benchmark framework

**Benchmark Test Set (F.1):**
| Category | Count | Examples |
|----------|-------|----------|
| error_code | 8 | SRVO-063, MOTN-023, SYST-032 |
| troubleshooting | 7 | Vibration, noise, intermittent issues |
| procedure | 5 | Mastering, calibration, backup |
| comparison | 4 | Control modes, safety features |
| conceptual | 4 | DCS, RCAL, servo systems |
| parameter | 3 | $PARAM_GROUP adjustments |

**Difficulty Distribution:**
- EASY: 8 queries (single error code lookup)
- MEDIUM: 10 queries (multi-step reasoning)
- HARD: 8 queries (cross-domain knowledge)
- EXPERT: 5 queries (system integration)

**Technical Accuracy Scorer (F.2):**
| Metric | Weight | Description |
|--------|--------|-------------|
| entity_coverage | 0.25 | Expected entities found |
| concept_coverage | 0.25 | Required concepts present |
| domain_match | 0.20 | Sources from trusted domains |
| safety_present | 0.10 | Safety terms when required |
| procedure_completeness | 0.10 | Step/action density |
| term_accuracy | 0.10 | Technical term presence |

**Usage:**
```python
from agentic import FANUC_BENCHMARK, TechnicalAccuracyScorer, run_benchmark

# Run full benchmark
report = await run_benchmark(orchestrator)
print(f"Pass rate: {report.pass_rate:.1%}")

# Filter by difficulty
hard_queries = filter_benchmark(difficulty=QueryDifficulty.HARD)

# Score single answer
scorer = TechnicalAccuracyScorer()
score = scorer.score(answer, benchmark_query, sources)
```

**Domain Matching Fix (F.3 - 2025-12-30):**
The benchmark now uses `FANUC_TRUSTED_DOMAINS` (22 domains) instead of narrow 3-domain lists:

| Category | Count | Examples |
|----------|-------|----------|
| Official FANUC | 4 | fanucamerica.com, techtransfer, crc2.frc.com |
| Industrial Forums | 5 | robot-forum.com, plctalk.net, emastercam.com |
| Integrators | 4 | 2rirobotics.com, aerobotix.net, robotworx.com |
| Doc Aggregators | 3 | everythingaboutrobots.com, manualslib.com, pdfcoffee.com |
| Tech Communities | 6 | reddit.com, stackoverflow.com, cnczone.com |

**Impact:**
- Domain match: 10% → 80% (+70%)
- Overall accuracy: 48% → 60.3% (+12%)
- SRVO-063 benchmark: FAIL → PASS

**Module Version**: `agentic/__init__.py` → v0.40.0

#### ✅ Phase 27: FANUC Ingestion Pipeline Design (Research Complete 2025-12-30)

Comprehensive research and pipeline design for optimal FANUC technical manual ingestion:

**System Audit Findings (6 Critical Gaps):**
| Gap | Impact | Priority |
|-----|--------|----------|
| No embeddings generated | Vector search unusable | P0 |
| Incomplete HSEA export | Graph relationships lost | P0 |
| Sparse entity graph (99.986%) | Multi-hop reasoning impossible | P1 |
| Dormant learning modules | No automatic expansion | P1 |
| BM25-only search | Missing semantic retrieval | P1 |
| No pattern matching | Generic troubleshooting | P2 |

**7-Stage Ingestion Pipeline Designed:**
1. **Semantic Chunking**: Structure-aware with atomic units (tables, error sections)
2. **Hybrid Entity Extraction**: Pattern matching (100% precision) + GLiNER (zero-shot)
3. **Causal Relation Extraction**: Diagnostic chains (symptom → error → remedy)
4. **Graph Assembly**: Matryoshka embeddings at 3 granularities (128/256/768d)
5. **HybridRAG Search**: BM25 + Vector + PathRAG with RRF fusion (k=60)
6. **Enhanced HSEA Export**: Relationships + embeddings + diagnostic paths
7. **Continual Learning**: Curriculum Levels 2-5 activation

**Research Sources:**
- GLiNER (NAACL 2024) for zero-shot NER
- PathRAG (2025) for causal chain retrieval
- Matryoshka embeddings (NeurIPS 2022) for multi-resolution
- HybridRAG (ACM ICAIF 2024) for vector+graph fusion

**Implementation Artifacts (PDF_Extraction_Tools):**
- `FANUC_INGESTION_PIPELINE_DESIGN.md`: Full architecture
- `docs/FANUC_NLP_INGESTION_BEST_PRACTICES.md`: Research synthesis
- `scripts/generate_embeddings.py`: Matryoshka generation
- `scripts/bootstrap_symptoms.py`: Level 2 curriculum bootstrap

**Target Metrics:**
| Metric | Current | Target |
|--------|---------|--------|
| Total entities | 8,626 | 15,000+ |
| Graph edges | ~9,716 | 40,000+ |
| Embeddings | 0 | 8,449+ |
| Curriculum stages | 1/5 | 5/5 |

#### ✅ Phase 26: Feature Synergy Integration (Completed 2025-12-29)

Integrated isolated features that were defined but not connected in the pipeline:

**Query Tree + CRAG Integration:**
- When CRAG recommends `REFINE_QUERY`, queries now expanded via QueryTreeDecoder
- Parallel exploration of query variations (REWRITE, DECOMPOSE, DISAMBIGUATE)
- Estimated 10-15% performance improvement realized
- Both SSE streaming and non-streaming paths updated

**FLARE + Synthesis Integration:**
- FLARE now monitors synthesis for uncertainty (hedging patterns, low-confidence markers)
- Triggers proactive retrieval when uncertainty detected
- Re-synthesizes with augmented context (up to 3 additional docs)
- Enabled in RESEARCH/FULL presets

**Meta-Buffer Templates in Synthesis:**
- `state.retrieved_template` and `state.composed_reasoning_strategy` now applied
- Templates passed as `thought_context` to synthesis phase
- Includes adapted prompts from reasoning composer
- Previously dead code now active

**Files Modified:**
- `agentic/orchestrator_universal.py`: CRAG handler, synthesis phase, SSE streaming

**Module Version**: `agentic/__init__.py` → v0.39.0

#### ✅ Phase 25: Feature Combination Audit (Completed 2025-12-29)

6-agent parallel research validated the feature combination architecture against 2025 research best practices.

**Validation Results:**
| Category | Status | Confidence |
|----------|--------|------------|
| RAG Technique Combinations | **CORRECT** | High |
| Reasoning Framework Combinations | **CORRECT** | High |
| Retrieval Optimization | **CORRECT** | High |
| Quality Control Patterns | **MOSTLY CORRECT** | Medium-High |
| Memory & Caching Architecture | **CORRECT** | High |
| Multi-Agent Orchestration | **CORRECT** | High |

**Key Findings:**
1. **Pipeline order is correct**: `Query → HyDE → Search → CRAG → Scrape → Synthesize → Self-RAG → Response`
2. **Reasoning layers are NOT redundant** - they operate at different abstraction levels:
   - Layer 4: AIME (orchestration)
   - Layer 3: Pre-Act (execution planning)
   - Layer 2: GoT (reasoning structure)
   - Layer 1: BoT (reasoning content)
3. **Semantic cache threshold 0.88** is appropriate for mxbai-embed-large
4. **A-MEM + RAISE are complementary** - cross-session vs per-request
5. **Preset-based configuration aligns** with LangGraph, CrewAI, AutoGen patterns

**Calibration Recommendations (P1):**
- Align confidence weights: 40% verification, 25% diversity, 20% depth, 15% synthesis
- UCB bandit c: 2.0 → 1.5 with warm-start priors
- Self-consistency: Conditional 3-5 samples with early termination

**Optimization Recommendations (P2):**
- RAISE → A-MEM promotion for high-confidence findings (≥0.8)
- STE-based cache eviction (KVFlow-inspired)
- Time-decay for semantic cache freshness

**Enhancement Recommendations (P3):**
- Feature flag bundling (group related flags)
- Circuit breaker for parallel search
- LangGraph-style checkpointing

**Research Validation Sources:**
- CRAG: arXiv:2401.15884
- Self-RAG: arXiv:2310.11511
- HyDE: arXiv:2212.10496
- GoT: arXiv:2308.09687
- BoT: arXiv:2406.04271
- AIME: ByteDance 2025 (77.6% GAIA)

**Module Version**: `agentic/__init__.py` → v0.38.0

#### ✅ Phase 24: Engineering Remediation (Completed 2025-12-29)

Comprehensive technical debt cleanup based on COMPREHENSIVE_ENGINEERING_AUDIT_REPORT.md:

**Part A - Orchestrator Consolidation:**
- Archived 5 legacy orchestrators (~120K lines) to `archive/legacy_orchestrators/`
- UniversalOrchestrator is now **Single Source of Truth** with 50+ feature flags
- 5 presets: MINIMAL (8), BALANCED (18), ENHANCED (28), RESEARCH (39+), FULL (42+)
- Backward-compatible shims in `__init__.py` with deprecation warnings
- Full API compatibility maintained

**Archived Files:**
| File | Lines | Migrated To |
|------|-------|-------------|
| `orchestrator.py` | 2,445 | `UniversalOrchestrator(preset=BALANCED)` |
| `orchestrator_dynamic.py` | 631+ | `UniversalOrchestrator(preset=RESEARCH)` |
| `orchestrator_enhanced.py` | 707+ | `UniversalOrchestrator(preset=ENHANCED)` |
| `orchestrator_unified.py` | 756+ | `UniversalOrchestrator(preset=ENHANCED)` |
| `orchestrator_graph_enhanced.py` | 886+ | `UniversalOrchestrator(preset=RESEARCH)` |

**Part B - Phase 7 Error Response Standardization:**
- Created `core/exceptions.py` with `AppException` and `ErrorCode` enum
- 25+ standardized error codes across 6 categories
- Unified error response format for all endpoints
- Global exception handlers in `main.py`

**Error Response Format:**
```json
{
  "success": false,
  "data": null,
  "meta": {"timestamp": "...", "request_id": "...", "path": "..."},
  "errors": [{"code": "ERR_xxxx", "message": "...", "details": {...}}]
}
```

**Error Code Categories:**
| Range | Category |
|-------|----------|
| 1xxx | Validation errors |
| 2xxx | Authentication errors |
| 3xxx | Resource errors |
| 4xxx | Search/Agentic errors |
| 5xxx | External service errors |
| 6xxx | Memory/Quest errors |
| 9xxx | System errors |

**Impact Summary:**
| Metric | Before | After |
|--------|--------|-------|
| Active orchestrator files | 6 | 1 |
| Orchestrator code lines | ~312K | ~4.7K |
| Error response formats | Mixed | Unified |
| Maintenance burden | 6x | 1x |

**Module Version**: `agentic/__init__.py` → v0.37.0

#### ✅ Phase 23: HSEA Three-Stratum Indexing (Completed 2025-12-29)

Implemented **Hierarchical Stratified Embedding Architecture (HSEA)** for three-layer contextual search:

**Architecture Overview:**
```
┌─────────────────────────────────────────────────────────────────┐
│  π₁ SYSTEMIC (17%)     │  π₂ STRUCTURAL (17%)  │  π₃ SUBSTANTIVE (66%)  │
│  Category anchors      │  Auto-connections     │  Full content          │
│  Troubleshooting       │  Semantic memory      │  BGE-M3 Hybrid         │
│  patterns              │  network              │  Dense + BM25          │
│  Binary index (32x)    │  Int8 index (4x)      │  FP16 store            │
│  MRL 64d               │  MRL 256d             │  MRL 1024-4096d        │
└─────────────────────────────────────────────────────────────────┘
```

**New Components:**
- **`agentic/hsea_controller.py`** (~700 lines): Core HSEA orchestration
  - ErrorCodeEntity and CrossStratumContext dataclasses
  - HSEASearchMode: systemic, structural, substantive, contextual, mrl
  - 7 built-in troubleshooting patterns
  - Lazy-loaded integration with existing memOS embedding systems
  - RRF merge for multi-source ranking

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/hsea/search` | POST | Multi-stratum semantic search |
| `/api/v1/search/hsea/troubleshoot/{code}` | GET | Troubleshooting context retrieval |
| `/api/v1/search/hsea/similar/{code}` | GET | Similar error code search |
| `/api/v1/search/hsea/index/batch` | POST | Batch entity indexing |
| `/api/v1/search/hsea/stats` | GET | System statistics |

**Three-Stage Retrieval Pipeline:**
```
Query → [Binary Index] → 500 candidates (Hamming distance)
           ↓
      [Int8 Index] → 50 candidates (Cosine similarity)
           ↓
      [FP16 Store] → 10 results (Full precision + enrichment)
```

**MRL Dimension Progression:**
- Stage 1: 64 dimensions (coarse semantics, fast filtering)
- Stage 2: 256 dimensions (balanced precision)
- Stage 3: 1024 dimensions (fine-grained ranking)
- Stage 4: 4096 dimensions (full precision final)

**Troubleshooting Patterns:**
| Pattern | Categories | Keywords |
|---------|------------|----------|
| encoder_replacement | SRVO | encoder, pulsecoder, RCAL, mastering |
| calibration | SRVO, MOTN | calibration, mastering, zero, origin |
| communication_reset | HOST, COMM | network, timeout, ethernet, IP |
| parameter_adjustment | SYST, SVGN, MOTN | parameter, setting, $PARAM |
| safety_interlock | SYST, PRIO | safety, e-stop, fence, DCS |
| servo_power_cycle | SRVO, SVGN | servo, power, amplifier, motor |
| vision_calibration | CVIS | vision, camera, iRVision, lens |

**Integration with Existing Systems:**
- `MixedPrecisionEmbeddingService`: Binary + Int8 + FP16 indexing
- `SemanticMemoryNetwork`: Auto-connection graph (0.7 threshold)
- `BGEM3HybridRetriever`: Dense + BM25 fusion via RRF
- `HyDEExpander`: +15-25% recall via hypothetical documents

**Cross-Stratum Context:**
```python
CrossStratumContext:
  - error_code: "SRVO-063"
  - title: "SRVO-063 RCAL alarm"
  - score: 0.87
  - layer_1_context:
      - category_anchor: "SRVO Alarms"
      - patterns: ["Encoder Replacement", "Calibration"]
  - layer_2_context:
      - related_codes: ["SRVO-068", "SRVO-069"]
      - cluster_members: ["SRVO-064", "SRVO-065"]
      - auto_connections: 5
  - layer_3_context:
      - similar_by_cause: [...]
      - bm25_score: 0.72
      - dense_score: 0.91
```

**Performance Targets:**
| Metric | Target |
|--------|--------|
| Contextual search latency | 10-50ms |
| Three-stage compression | 3.4x |
| Recall@10 improvement | +15-25% |
| Auto-connection threshold | 0.7 |

**Research Basis:**
- Kusupati et al., 2022: Matryoshka Representation Learning
- Cormack et al., 2009: Reciprocal Rank Fusion
- Gao et al., 2023: HyDE Query Expansion
- Microsoft, 2024: GraphRAG
- Sarmah et al., 2024: HybridRAG

**Module Version**: `agentic/__init__.py` → v0.36.0

#### ✅ Phase 22: PDF Extraction Tools Integration (Completed 2025-12-29)

Integrated PDF Extraction Tools API for FANUC technical documentation RAG:

**New Components:**
- **`core/document_graph_service.py`**: Bridge to PDF Extraction Tools API
  - Async HTTP client with connection pooling (aiohttp)
  - Circuit breaker pattern with health check
  - In-memory caching with configurable TTL
  - Graceful degradation when API unavailable
- **`agentic/schemas/fanuc_schema.py`**: FANUC entity extraction patterns
  - 15+ error code pattern categories (SRVO, MOTN, SYST, INTP, HOST, etc.)
  - Component patterns (axes J1-J9, motors, encoders, cables)
  - Parameter patterns ($PARAM_GROUP, etc.)
  - Helper functions: `is_fanuc_query()`, `extract_error_codes()`, `get_error_category()`

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search/technical/health` | GET | Check PDF API health status |
| `/api/v1/search/technical/search` | POST | Search FANUC technical documentation |
| `/api/v1/search/technical/troubleshoot` | POST | Get troubleshooting path for error code |
| `/api/v1/search/technical/context` | GET | Get formatted context for RAG injection |

**Configuration Settings (config/settings.py):**
```python
pdf_api_url: str = "http://localhost:8002"
pdf_api_timeout: int = 30
pdf_api_enabled: bool = True
pdf_api_max_results: int = 10
pdf_api_cache_ttl: int = 300  # 5 minutes
```

**Feature Flag:**
| Flag | Default | Description |
|------|---------|-------------|
| `enable_technical_docs` | False | FANUC manual RAG via PDF API |

**Preset Configuration:**
| Preset | enable_technical_docs |
|--------|----------------------|
| minimal | False |
| balanced | False |
| enhanced | True |
| research | True |
| full | True |

**Orchestrator Integration:**
- `_document_graph_service`: Lazy-loaded service component
- `_get_document_graph_service()`: Getter for lazy initialization
- `_search_technical_docs()`: Helper method for automatic FANUC query detection

**Key Features:**
- **Automatic Detection**: Uses regex patterns to detect FANUC-related queries
- **Error Code Extraction**: Extracts error codes like SRVO-063, MOTN-023
- **PathRAG Traversal**: Builds troubleshooting paths through document graph
- **Context Formatting**: Returns LLM-ready context strings with source citations
- **Circuit Breaker**: Prevents cascade failures when PDF API is down

**External Dependency:**
- **PDF Extraction Tools API** running on port 8002
- Location: `/home/sparkone/sdd/PDF_Extraction_Tools`
- Integration plan: `PDF_Extraction_Tools/MEMOS_INTEGRATION_PLAN.md`

**Corpus Status (2026-01-01 Audit):**
- 137/137 FANUC manuals ingested (100%)
- 8,471 indexed error codes across 104 categories
- 268,886 nodes, 269,466 edges
- HNSW indices: 1.4 GB, 0.17ms P95 latency
- See `PDF_Extraction_Tools/CLAUDE.md` for full audit

**Module Version**: `agentic/__init__.py` → v0.35.0

#### ✅ Phase 21: Template Reuse Optimization (Completed 2025-12-29)

Implemented Cross-Session Meta-Buffer and Self-Discover Reasoning Composition:

**New Modules:**
- **`meta_buffer.py`**: Cross-session template persistence
  - SQLite-backed storage for reasoning templates
  - Template distillation from successful searches (confidence ≥ 0.75)
  - Semantic retrieval via embedding similarity (threshold 0.7)
  - Performance tracking (success rate, usage count, avg execution time)
  - Template types: DECOMPOSITION, SEARCH_STRATEGY, SYNTHESIS, VERIFICATION, REFINEMENT
  - Based on Buffer of Thoughts (NeurIPS 2024) - 12% cost of ToT, 11-51% accuracy improvement
- **`reasoning_composer.py`**: Self-Discover style reasoning composition
  - 12+ atomic reasoning modules (critical analysis, step-by-step, compare/contrast, etc.)
  - SELECT → ADAPT → IMPLEMENT meta-action pipeline
  - Module definitions with prompt templates and example applications
  - Composes task-specific reasoning strategies from atomic modules
  - Based on Self-Discover (Google DeepMind, NeurIPS 2024)

**Key Features:**
- **Template Distillation**: Extract reusable patterns from successful searches
- **Semantic Retrieval**: Find relevant templates via embedding similarity
- **Cross-Session Persistence**: SQLite storage survives server restarts
- **Module Selection**: LLM selects relevant reasoning modules for each task
- **Module Adaptation**: Customizes generic modules for specific tasks
- **Strategy Composition**: Structures selected modules into executable plan

**Reasoning Module Categories:**
| Module | Description |
|--------|-------------|
| critical_analysis | Evaluate claims, identify assumptions |
| step_by_step | Systematic approach for complex problems |
| compare_contrast | Side-by-side analysis of options |
| causal_reasoning | Trace cause-and-effect chains |
| synthesis | Combine multiple sources coherently |
| hypothesis_testing | Formulate and test predictions |
| abstraction | Extract generalizable patterns |
| decomposition | Break into manageable sub-problems |
| verification | Cross-check against evidence |
| temporal_reasoning | Analyze timelines and sequences |
| counterfactual | Consider alternative scenarios |
| meta_cognitive | Monitor and adjust reasoning process |

**Orchestrator Integration:**
- `_get_meta_buffer()`: Lazy-loaded SQLite-backed template store
- `_get_reasoning_composer()`: Lazy-loaded reasoning composer
- `_retrieve_template()`: Find relevant templates for query
- `_distill_successful_search()`: Extract templates from high-confidence results
- `_compose_reasoning_strategy()`: Build task-specific reasoning plan

**Feature Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `enable_meta_buffer` | False | Cross-session template persistence |
| `enable_reasoning_composer` | False | Self-Discover reasoning composition |

**Preset Configuration:**
| Preset | meta_buffer | reasoning_composer |
|--------|-------------|-------------------|
| minimal | False | False |
| balanced | False | False |
| enhanced | False | False |
| research | True | True |
| full | True | True |

**Research Basis:**
- Buffer of Thoughts (NeurIPS 2024): 12% cost of ToT, templates transfer across task types
- Self-Discover (Google DeepMind): SELECT → ADAPT → IMPLEMENT for reasoning composition

**Module Version**: `agentic/__init__.py` → v0.34.0

#### 🔧 Phase 21 Bug Fixes (Applied 2025-12-29)

Two critical bugs discovered during FANUC challenging query testing:

**Bug #1: Missing `calculate_confidence` Method**
- **Error**: `'UniversalOrchestrator' object has no attribute 'calculate_confidence'`
- **Location**: `orchestrator_universal.py:1813`
- **Fix**: Changed to `calculate_heuristic_confidence(sources, synthesis, request.query)`

**Bug #2: Extra `scratchpad` Argument**
- **Error**: `TypeError: unhashable type: 'list'` (misleading - appeared in metrics.py)
- **Location**: `orchestrator_universal.py:1806`
- **Root Cause**: `_phase_synthesis` was called with 6 args but expects 5
- **Fix**: Removed `scratchpad` from the call

**Test Results After Fixes:**
| Query | Confidence | Status |
|-------|------------|--------|
| Q1: Mastering procedures | 76.2% | ✅ |
| Q2: Servo alarms | 68.4% | ✅ |
| Q3: DCS Safe Position | 50.5% | ✅ (was crashing) |
| Q4: iRVision drift | 80.7% | ✅ |
| Q5: KAREL upgrade | 75.7% | ✅ |

**Average Confidence**: 70.3% (5/5 passing)

**Audit Report**: See `agentic/PHASE_21_AUDIT_REPORT.md` for full details.

**Integration Gap Note**: Phase 21 helper methods (`_retrieve_template`, `_distill_successful_search`, `_compose_reasoning_strategy`) are implemented but not yet called in the main search flow. This is documented in the audit report with recommendations.

#### ✅ Phase 20: Scratchpad Enhancement (Completed 2025-12-29)

Implemented A-MEM semantic memory network and RAISE four-component structure:

**New Modules:**
- **`semantic_memory.py`**: A-MEM Zettelkasten-inspired memory network
  - Automatic connection establishment based on embedding similarity (>0.7)
  - Bidirectional links for graph traversal
  - 8 memory types: finding, source, entity, reasoning, observation, example, question, answer
  - 7 connection types: semantic, reference, supports, contradicts, derived_from, answers, related
  - Based on A-MEM (arXiv 2502.12110) - 35% F1 improvement
- **`raise_scratchpad.py`**: RAISE four-component structure
  - Observations: Tool outputs, retrieved documents (6 types)
  - Reasoning: Intermediate conclusions with confidence (7 types)
  - Examples: Successful patterns for reuse
  - Trajectory: Execution history with timing
  - Quality signal extraction from scratchpad contents
  - Based on RAISE (arXiv 2401.02777)

**Key Features:**
- **Semantic Memory**: Embedding-based similarity, auto-connection, graph traversal
- **Memory Traversal**: BFS exploration with strength-weighted paths
- **Quality Signal**: Evidence quality, reasoning quality, coverage, uncertainty indicators
- **Observation Types**: TOOL_OUTPUT, DOCUMENT, SEARCH_RESULT, USER_INPUT, SYSTEM_DATA, SCRAPED_CONTENT
- **Reasoning Types**: DEDUCTION, INDUCTION, ABDUCTION, COMPARISON, SYNTHESIS, CRITIQUE, HYPOTHESIS
- **Uncertainty Indicators**: HIGH/MEDIUM/LOW_CONFIDENCE, CONFLICTING_EVIDENCE, MISSING_INFORMATION, UNVERIFIED

**Orchestrator Integration:**
- `_get_semantic_memory()`: Lazy-loaded A-MEM network
- `_get_raise_scratchpad()`: Per-request RAISE scratchpad
- `_add_to_semantic_memory()`: Add content with auto-connections
- `_record_observation()`: Record observations during pipeline
- `_record_reasoning()`: Record reasoning steps
- `_get_quality_signal()`: Extract quality assessment

**Feature Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `enable_semantic_memory` | False | A-MEM Zettelkasten-style memory network |
| `enable_raise_structure` | False | RAISE four-component scratchpad |

**Preset Configuration:**
| Preset | semantic_memory | raise_structure |
|--------|-----------------|-----------------|
| minimal | False | False |
| balanced | False | False |
| enhanced | False | False |
| research | True | True |
| full | True | True |

**Research Basis:**
- A-MEM (arXiv 2502.12110): Zettelkasten-inspired memory with 35% F1 improvement
- RAISE (arXiv 2401.02777): Observations/Reasoning/Examples/Trajectory structure

**Module Version**: `agentic/__init__.py` → v0.33.0

#### ✅ Phase 19: Enhanced Query Generation (Completed 2025-12-29)

Implemented FLARE forward-looking retrieval and RQ-RAG query tree decoding for improved query coverage:

**New Modules:**
- **`flare_retriever.py`**: Forward-Looking Active REtrieval
  - Detects uncertainty during synthesis via hedging patterns and low-confidence markers
  - Triggers proactive retrieval based on what the model PREDICTS it needs
  - Regenerates with retrieved context when uncertainty detected
  - Based on FLARE (EMNLP 2023) - retrieve when uncertain, not after-the-fact
- **`query_tree.py`**: RQ-RAG Query Tree Decoder
  - Explores multiple query variations via tree decoding
  - Operations: REWRITE, DECOMPOSE, DISAMBIGUATE, EXPAND, NARROW, NEGATE
  - Parallel exploration of different phrasings retrieves different relevant docs
  - Confidence-weighted aggregation across tree branches
  - Based on RQ-RAG (arXiv 2404.00610) - +33.5% on QA benchmarks

**Key Features:**
- **Uncertainty Detection**: Patterns for hedging (might, may, possibly) and factual claims (statistics, dates)
- **Tentative Generation**: Generate 50 tokens tentatively, use as retrieval query if uncertain
- **Tree Decoding**: Generate 4+ query variations at depth 2, retrieve in parallel
- **Retrieval Triggers**: LOW_CONFIDENCE, UNCERTAIN_PHRASE, FACTUAL_CLAIM, MISSING_DETAIL
- **Branch Aggregation**: Weight documents by node confidence × retrieval score

**Orchestrator Integration:**
- `_expand_queries_with_tree()`: Expands query into 8 variations for parallel search
- `_flare_enhanced_retrieval()`: Adds docs when synthesis shows uncertainty
- Lazy-loaded components for minimal overhead when disabled

**Feature Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `enable_flare_retrieval` | False | Forward-looking active retrieval |
| `enable_query_tree` | False | RQ-RAG tree decoding for query expansion |

**Preset Configuration:**
| Preset | flare_retrieval | query_tree |
|--------|-----------------|------------|
| minimal | False | False |
| balanced | False | False |
| enhanced | False | False |
| research | True | True |
| full | True | True |

**Research Basis:**
- FLARE (EMNLP 2023): Active retrieval on uncertainty (arXiv:2305.06983)
- RQ-RAG (arXiv 2404.00610): +33.5% on QA benchmarks via tree decoding

**Module Version**: `agentic/__init__.py` → v0.32.0

#### ✅ Phase 18: Confidence-Calibrated Halting (Completed 2025-12-29)

Implemented adaptive iteration control based on UALA, CISC, and REFRAIN research:

**New Modules:**
- **`entropy_monitor.py`**: UALA-style entropy-based halting
  - Tracks generation entropy to detect confident vs uncertain outputs
  - Thresholds: entropy < 0.2 → halt confident, entropy > 0.5 → continue
  - Session-based trajectory tracking for convergence detection
  - Based on UALA (ACL 2024) - >50% reduction in tool calls
- **`self_consistency.py`**: CISC multi-path convergence checking
  - Weighted majority voting across multiple synthesis attempts
  - Semantic similarity clustering of answers
  - Key fact agreement analysis across attempts
  - Based on CISC (Google, arXiv 2502.20233) - >40% sample reduction
- **`iteration_bandit.py`**: UCB-based action selection
  - Multi-armed bandit for exploration/exploitation tradeoff
  - Actions: search_more, refine_query, synthesize_now, decompose, verify, broaden, narrow
  - Context-aware statistics for improved learning
  - Based on REFRAIN (arXiv 2510.10103) - 20-55% token reduction

**Key Features:**
- **Entropy Monitoring**: LLM self-evaluation of confidence via completeness, source quality, hedging, specificity
- **Halt Decisions**: CONTINUE, HALT_CONFIDENT, HALT_MAX_ITERATIONS, HALT_CONVERGENCE
- **Convergence Status**: CONVERGED (>60% agreement), PARTIAL, DIVERGENT, INSUFFICIENT
- **UCB Action Selection**: `UCB(a) = Q(a) + c * sqrt(log(t) / N(a))`
- **Bandit Learning**: Persistent stats across sessions, context-aware rewards

**Orchestrator Integration:**
- Phase 7.5 (Entropy Check): Between reflection and adaptive refinement
- If entropy indicates confidence, skips refinement loop (saves time)
- UCB bandit optionally replaces default refinement decision logic
- Bandit outcomes recorded for continuous learning

**Feature Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `enable_entropy_halting` | False | UALA-style entropy monitoring |
| `enable_iteration_bandit` | False | UCB action selection for iterations |
| `enable_self_consistency` | False | Multi-path answer convergence |

**Preset Configuration:**
| Preset | entropy_halting | iteration_bandit | self_consistency |
|--------|-----------------|------------------|------------------|
| minimal | False | False | False |
| balanced | False | False | False |
| enhanced | False | False | False |
| research | True | True | False |
| full | True | True | True |

**Research Basis:**
- UALA (ACL 2024): Uncertainty-aware tool call reduction
- CISC (arXiv 2502.20233): Confidence-informed self-consistency
- REFRAIN (arXiv 2510.10103): UCB-based iteration decisions

**Module Version**: `agentic/__init__.py` → v0.31.0

#### ✅ Phase 17: Context Curation Pipeline (Completed 2025-12-29)

Implemented InfoGain-RAG and Context-Picker inspired context curation for high-quality retrieval:

**New Modules:**
- **`information_gain.py`**: Document Information Gain (DIG) scoring
  - Measures document utility by impact on generation confidence
  - Categories: positive, neutral, negative
  - Based on Kuaishou InfoGain-RAG (+17.9% over naive RAG)
- **`redundancy_detector.py`**: Semantic similarity clustering
  - Agglomerative clustering with 0.85 similarity threshold
  - Selection methods: central, quality, authority, DIG, length
- **`context_curator.py`**: Main curation pipeline
  - 4 presets: fast, balanced, thorough, technical
  - Two-stage filtering (recall-oriented → precision prune)
  - Coverage analysis against decomposed questions

**Key Features:**
- **DIG Scoring**: LLM-evaluated document relevance, quality, novelty, and contradiction detection
- **Redundancy Detection**: Embedding-based clustering with best representative selection
- **Coverage Analysis**: Tracks which decomposed questions are answered by curated content
- **Confidence Estimation**: Combined metric from DIG scores and coverage ratio

**Orchestrator Integration:**
- New feature flag: `enable_context_curation` (default: False)
- New config: `context_curation_preset` ("fast", "balanced", "thorough", "technical")
- Enabled by default in ENHANCED, RESEARCH, FULL presets
- Runs between scraping and verification phases

**Preset Configuration:**
| Preset | enable_context_curation | context_curation_preset |
|--------|------------------------|-------------------------|
| minimal | False | balanced |
| balanced | False | balanced |
| enhanced | True | balanced |
| research | True | thorough |
| full | True | technical |

**Research Basis:**
- InfoGain-RAG: Document utility measurement (+17.9% improvement)
- Context-Picker (arXiv 2512.14465): Two-stage filtering with Leave-One-Out pruning
- RAGAS: Deduplication patterns

**Module Version**: `agentic/__init__.py` → v0.30.0

#### ✅ Phase 16: Android SSE Streaming Integration (Completed 2025-12-29)

Added `search_with_events()` method to UniversalOrchestrator for Android CLI integration:

**New Method:**
- **`search_with_events(request, emitter)`**: Full pipeline with SSE event emissions

**Bug Fixes:**
1. SupportLevel enum JSON serialization → Convert to string value
2. Config attribute mismatches (enable_crag → enable_crag_evaluation, etc.)
3. Method signature alignment with UniversalOrchestrator conventions

**SSE Event Flow:**
```
search_started → analyzing_query → query_analyzed → planning_search → search_planned
→ iteration_start → searching → search_results → crag_evaluating → evaluating_urls
→ scraping_url → url_scraped → verifying_claims → claims_verified → synthesizing
→ synthesis_complete → self_rag_reflecting → self_rag_complete → search_completed
```

**Graph Visualization:**
```
[A✓]→[P✓]→[S✓]→[E✓]→[W✓]→[V✓]→[Σ✓]→[R✓]→[✓✓]
```

**Android Integration:**
- SSE events received and parsed by `AgenticSearchService.kt`
- Real-time progress shown in `ToolExecutionIndicator.kt`
- Graph line tracked across events for visualization

**Test Results (FANUC SRVO-063 Query):**
- Sources: 10
- Confidence: 49-51%
- Execution Time: 60-270s depending on preset
- All SSE events properly emitted and received by Android client

**Module Version**: `agentic/__init__.py` → v0.27.1

#### ✅ Phase 15: Orchestrator Consolidation (Completed 2025-12-28)

Consolidated all orchestrators into UniversalOrchestrator as the single source of truth:

**DEPRECATED Orchestrators:**
| Class | Replacement |
|-------|-------------|
| `AgenticOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)` |
| `EnhancedAgenticOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)` |
| `DynamicOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH)` |
| `GraphEnhancedOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH)` |
| `UnifiedOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)` |

**Code Reduction:**
- 5 deprecated orchestrator files (~5,342 lines)
- Single source of truth: `orchestrator_universal.py`
- 42+ feature flags via presets

**API Changes:**
- `get_orchestrator()` → redirects to `get_universal_orchestrator("balanced")`
- `get_enhanced_orchestrator()` → redirects to `get_universal_orchestrator("enhanced")`
- `get_graph_orchestrator()` → redirects to `get_universal_orchestrator("research")`
- `get_unified_orchestrator_instance()` → redirects to `get_universal_orchestrator("enhanced")`

**Preset Quick Reference:**
| Preset | Features | Use Case |
|--------|----------|----------|
| `minimal` | 8 | Fast, simple queries |
| `balanced` | 18 | Default for most queries |
| `enhanced` | 28 | Complex research |
| `research` | 39 | Academic/thorough (includes dynamic planning + graph cache) |
| `full` | 42+ | Maximum capability (adds multi-agent) |

**Module Version**: `agentic/__init__.py` → v0.27.0

#### ✅ Phase 14: Context Utilization Tracking (Completed 2025-12-28)

Comprehensive instrumentation for tracking context window utilization across all agentic pipeline agents:

**New Components:**
- **Context Utilization Metrics**: Per-agent tracking in `metrics.py`
- **Request ID Propagation**: Unique request_id passed through entire pipeline
- **Query Lifecycle Tracking**: `start_query()` / `complete_query()` for orchestrator

**Instrumented Agents:**
| Agent | File | Method | Tracking |
|-------|------|--------|----------|
| Analyzer | `analyzer.py` | `analyze()`, `create_search_plan()` | ✅ |
| QueryClassifier | `query_classifier.py` | `classify_query()` | ✅ |
| RetrievalEvaluator | `retrieval_evaluator.py` | `evaluate()` | ✅ |
| SelfReflection | `self_reflection.py` | `reflect()` | ✅ |

**Key Features:**
- **Per-Agent Metrics**: Tracks input tokens, output tokens, utilization percentage
- **Rolling Averages**: Maintains historical context usage for optimization
- **Tool Latency Tracking**: Per-tool response time for bottleneck analysis
- **API Endpoint**: `GET /api/v1/search/metrics` returns context utilization summary

**Audit Findings (2025-12-28):**
- Analyzer context utilization: 0.7% (optimal for analysis tasks per research)
- Content integrity: 100% source preservation verified
- KV-cache prefix optimization identified as next optimization target

**Module Version**: `agentic/__init__.py` → v0.26.0

#### ✅ Gateway Preset Integration (Completed 2025-12-28)

Fixed preset propagation from Android client through gateway endpoint:

- **ChatGatewayRequest Model**: Added `preset` field (default: "full")
- **_execute_simple_search()**: Uses `request.preset` instead of hardcoded "full"
- **_execute_agentic_pipeline()**: Uses `request.preset` for orchestrator instantiation
- **Android Integration**: Preset selection flows: AppSettings → ChatToolIntegration → AgenticSearchService → Server

**Test Results:**
- Minimal preset: 4 features activated (`content_cache`, `query_analysis`, `scratchpad`, `verification`)
- Full preset: 38+ features activated
- Logs confirm preset routing: `Simple search using preset: minimal`

#### ✅ Phase 13: Universal Orchestrator + Bug Fixes (Completed 2025-12-28)

Consolidated all 40+ features into a single UniversalOrchestrator with preset-based configuration and fixed all method signature errors:

**New Components:**
- **UniversalOrchestrator** (`agentic/orchestrator_universal.py`): Single orchestrator with 5 presets
- **BaseSearchPipeline** (`agentic/base_pipeline.py`): Shared pipeline functionality
- **UniversalGraphState**: Real-time agent progress visualization

**5 Presets (feature configurations):**
| Preset | Features | Use Case |
|--------|----------|----------|
| `minimal` | 8 core features | Fast, simple queries |
| `balanced` | 18 features | Default for most queries |
| `enhanced` | 28 features | Complex research |
| `research` | 39 features | Academic/thorough (dynamic planning + graph cache) |
| `full` | 42+ features | Maximum capability (adds multi-agent) |

**17 Bug Fixes Applied:**
1. ContentCache `get_cached_query_result` → `get_query_result`
2. ToolCallContext request_id argument removed
3. get_kv_cache_service argument error fixed
4. AgenticScratchpad `add_entities` → `add_entity` loop
5. DynamicPlanner `create_initial_plan` → `initial_decomposition`
6. ThoughtLibrary `retrieve` → `retrieve_templates`
7. EmbeddingAggregator `aggregate` → `retrieve`
8. ActorFactory `analyze_task` removed (not needed)
9. DomainCorpusManager async initialization fixed
10. cross_domain_query `top_k` argument removed
11. EntityEnhancedRetriever method signature fixed
12. MixedPrecisionEmbeddingService method fixed
13. detect_contradictions synthesis argument fixed
14. Artifact storage await expression fixed
15. Query classifier model selection fixed (skip embedding models)
16. Graph visualization added to all SSE events
17. ProgressAggregator `start_tracking`/`complete_tracking` methods added

**Graph Visualization:**
Real-time agent progress shown in SSE events:
```
[A✓]→[P✓]→[S•]→[E]→[W]→[V]→[Σ]→[R]→[✓]
```
- `A` = Analyze, `P` = Plan, `S` = Search, `E` = Evaluate (CRAG)
- `W` = Scrape, `V` = Verify, `Σ` = Synthesize, `R` = Reflect
- `✓` = Complete, `•` = Active

**Test Results:**
```
Direct Answer Pipeline: ✅ Works
Web Search Pipeline: ✅ Works
Agentic Search Pipeline: ✅ Works
Graph Visualization: ✅ All events include graph_line
Confidence Score: 72% (research query)
Execution Time: ~119s (full preset)
```

**Module Version**: `agentic/__init__.py` → v0.25.0

**Documentation**: `agentic/ENHANCEMENT_IMPLEMENTATION_PLAN.md`

#### ✅ Phase 6: Self-RAG Reflection (Completed 2025-12-27)

Implemented Self-Reflective RAG based on arXiv:2310.11511 for synthesis quality assurance:

**New Components:**
- **SelfReflectionAgent** (`agentic/self_reflection.py`): ISREL/ISSUP/ISUSE reflection pattern
- **TemporalFactValidator**: Cross-checks dates/years for consistency
- **ReflectionResult**: Detailed quality assessment with refinement suggestions

**Key Features:**
- **ISREL (Relevance)**: Scores how relevant synthesis is to query (0-1)
- **ISSUP (Support)**: Checks if claims are supported by sources (fully_supported/partially_supported/no_support/contradicted)
- **ISUSE (Usefulness)**: Evaluates if response actually answers the question (0-1)
- **Temporal Validation**: Extracts and cross-checks dates/years for contradictions
- **Auto-Refinement**: Automatically refines synthesis when temporal conflicts detected
- **Blended Confidence**: Combines verifier confidence (60%) with reflection confidence (40%)

**Benchmark Results (Before/After Self-RAG):**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Temporal Accuracy | ❌ Errors | ✅ Correct | Fixed |
| Confidence Score | 0.61 | 0.74 | +21% |
| Relevance | N/A | 1.00 | New metric |
| Support Level | N/A | fully_supported | New metric |

**Usage:**
```python
from agentic import SelfReflectionAgent, get_self_reflection_agent

reflector = get_self_reflection_agent()
result = await reflector.reflect(
    query="When was GPT-3 released vs Anthropic founded?",
    synthesis="...",
    sources=[{"title": "...", "snippet": "...", "url": "..."}],
    scraped_content=["..."]
)

if result.needs_refinement:
    refined = await reflector.refine_synthesis(synthesis, result, sources)
```

**Module Version**: `agentic/__init__.py` → v0.9.0

#### ✅ Phase 7: CRAG Retrieval Evaluator (Completed 2025-12-27)

Implemented Corrective RAG based on arXiv:2401.15884 for pre-synthesis retrieval quality assessment:

**New Components:**
- **RetrievalEvaluator** (`agentic/retrieval_evaluator.py`): Pre-synthesis quality assessment
- **RetrievalQuality**: Quality levels (CORRECT/AMBIGUOUS/INCORRECT)
- **CorrectiveAction**: Actions (PROCEED/REFINE_QUERY/WEB_FALLBACK/DECOMPOSE)
- **DocumentScore**: Per-document relevance, quality, and coverage scoring

**Two-Stage Quality Control Pipeline:**
```
Search Results → CRAG Eval → [Corrective Action] → Synthesis → Self-RAG Eval → [Refinement]
                  ^Stage 1                          ^Stage 2
```

**Key Features:**
- **Pre-Synthesis Evaluation**: Assesses retrieval quality BEFORE LLM synthesis (unlike Self-RAG which evaluates AFTER)
- **Three Quality Levels**:
  - `CORRECT` (relevance ≥ 0.7): At least one document highly relevant → proceed
  - `AMBIGUOUS` (0.4-0.7): Mixed quality → refine queries and re-retrieve
  - `INCORRECT` (< 0.4): Poor retrieval → discard and trigger web fallback
- **Corrective Actions**:
  - `PROCEED`: Continue to synthesis
  - `REFINE_QUERY`: Generate targeted refinement queries
  - `WEB_FALLBACK`: Discard results, trigger fresh web search
  - `DECOMPOSE`: Break complex query into sub-questions
- **Document Scoring**: Per-document relevance, quality (source trust), coverage
- **Query Coverage Analysis**: Maps which parts of query are answered by results
- **LLM-Based Assessment**: Uses lightweight model (gemma3:4b) for fast evaluation

**Integration with Orchestrator:**
- Runs immediately after search phase, before synthesis
- Adds up to 3 refined queries if quality is AMBIGUOUS
- Triggers web fallback if quality is INCORRECT
- Logs CRAG evaluation results for debugging

**Test Results:**
```
CRAG Evaluation Output:
- Quality: ambiguous
- Relevance: 0.82
- Coverage: 0.00 (gaps detected)
- Action: refine_query
- Refined Queries: ["topic X specific aspect", "topic Y details", ...]
```

**Usage:**
```python
from agentic import RetrievalEvaluator, get_retrieval_evaluator

evaluator = get_retrieval_evaluator()
result = await evaluator.evaluate(
    query="What are the latest developments in topic X?",
    search_results=[{"title": "...", "snippet": "...", "url": "..."}],
    decomposed_questions=["sub-question 1", "sub-question 2"]
)

if result.recommended_action == CorrectiveAction.REFINE_QUERY:
    for refined_q in result.refined_queries:
        # Add to search queue
        state.pending_queries.append(refined_q)
```

**Module Version**: `agentic/__init__.py` → v0.10.0

#### ✅ Phase 8: Experience Distillation (Completed 2025-12-27)

Implemented MetaAgent-style experience distillation based on arXiv:2402.11904:

**New Components:**
- **ExperienceDistiller** (`agentic/experience_distiller.py`): Captures and distills successful search experiences
- **SearchExperience**: Structured capture of successful search patterns
- **DistillationResult**: Result of template extraction attempt

**Key Features:**
- **Automatic Experience Capture**: Captures successful searches (confidence ≥ 0.75) for learning
- **Structure Extraction**: Extracts abstract structure (headers, lists, patterns) from synthesis
- **Insight Extraction**: Identifies key insights using emphasis patterns (bold, "Key", "Important")
- **LLM-Based Distillation**: Uses gemma3:4b to generalize patterns into reusable templates
- **ThoughtLibrary Integration**: Adds distilled templates directly to the ThoughtLibrary
- **Duplicate Detection**: Checks for similar existing templates before creation
- **Per-Type Memory**: Stores experiences by query type for targeted distillation

**Experience Capture Flow:**
```
Successful Search (conf ≥ 0.75)
    ↓
Extract: structure, insights, queries, sources
    ↓
Store in ExperienceDistiller.experiences[query_type]
    ↓
If experiences ≥ 3: Trigger Distillation
    ↓
LLM analyzes patterns → Creates ThoughtTemplate
```

**API Endpoints:**
- `GET /api/v1/search/distillation/stats` - Distillation statistics
- `GET /api/v1/search/distillation/experiences` - View captured experiences
- `POST /api/v1/search/distillation/distill?query_type=X` - Trigger distillation
- `DELETE /api/v1/search/distillation/experiences` - Clear experiences

**Integration with Orchestrator:**
```python
# Automatic capture after successful search (orchestrator.py:994-1015)
if response.success and confidence_score >= 0.75:
    await self.experience_distiller.capture_experience(
        query=request.query,
        response=response,
        query_type=query_type,
        decomposed_questions=decomposed
    )
```

**Benefits (from MetaAgent research):**
- Continuous learning from successful searches
- Reduces token usage by reusing proven patterns
- Improves over time as more experiences are captured
- Combines with Buffer of Thoughts (Phase 4) for template reuse

**Module Version**: `agentic/__init__.py` → v0.11.0

#### ✅ Phase 9: Classifier Feedback Loop (Completed 2025-12-27)

Implemented Adaptive-RAG style feedback loop based on arXiv:2403.14403:

**New Components:**
- **ClassifierFeedback** (`agentic/classifier_feedback.py`): Tracks classification outcomes
- **ClassificationOutcome**: Record of prediction vs actual outcome
- **AdaptiveHint**: Learned adjustment for future classifications
- **OutcomeQuality**: Quality levels (EXCELLENT/GOOD/ADEQUATE/POOR/FAILED)

**Key Features:**
- **Outcome Tracking**: Records classification predictions vs actual search outcomes
- **Mismatch Detection**:
  - `was_overkill`: Used agentic search when web_search would suffice
  - `was_underkill`: Used simple pipeline when agentic was needed
  - `missed_web_search`: Used direct answer when web search was needed
- **Adaptive Hint Generation**: Learns patterns from outcomes every 10 searches
- **Per-Category Statistics**: Tracks success rates by query category
- **Pattern-Based Adjustments**: Suggests pipeline upgrades/downgrades based on query patterns

**Feedback Flow:**
```
Search Completes
    ↓
Record: predicted_category, predicted_pipeline, actual_confidence
    ↓
Detect: overkill? underkill? missed_web?
    ↓
Store in outcomes[category]
    ↓
Every 10 outcomes: Generate adaptive hints
    ↓
Future queries: Apply hints to adjust classifications
```

**API Endpoints:**
- `GET /api/v1/search/classifier/stats` - Feedback statistics
- `GET /api/v1/search/classifier/outcomes` - View outcome history
- `GET /api/v1/search/classifier/hints` - View learned hints
- `DELETE /api/v1/search/classifier/outcomes` - Clear history

**Integration with Orchestrator:**
```python
# Automatic recording after search (orchestrator.py:1023-1050)
if state.query_analysis:
    self.classifier_feedback.record_outcome(
        query=request.query,
        classification=pseudo_classification,
        confidence=confidence_score,
        iteration_count=state.iteration,
        source_count=len(state.raw_results),
        execution_time_ms=execution_time_ms
    )
```

**Mismatch Thresholds:**
- Overkill: High confidence (≥0.70) + fast completion (<30s) + few iterations (≤1)
- Underkill: Low confidence (<0.60) with simple pipeline
- Missed Web: Direct answer with low confidence (<0.55) and no sources

**Module Version**: `agentic/__init__.py` → v0.12.0

#### ✅ Phase 10: SSE Visibility + Thorough Search (Completed 2025-12-27)

Implemented comprehensive SSE event visibility for debugging and Android app display, plus increased iteration/refinement limits for thorough multi-direction exploration:

**Configuration Changes (`agentic/models.py`):**
- `max_iterations`: 5 → 10 (allows more ReAct loop cycles)
- `min_sources`: 3 → 5 (requires more source diversity)
- `max_sources`: 15 → 25 (allows comprehensive research)
- `min_confidence`: 0.70 (new - minimum quality threshold)
- `max_scrape_refinements`: 3 (new - configurable refinement cycles)

**New SSE Event Types (`agentic/events.py`):**
40+ new event types for complete agent processing visibility:

| Category | Events |
|----------|--------|
| **Query Classification** | `CLASSIFYING_QUERY`, `QUERY_CLASSIFIED` |
| **CRAG Evaluation** | `CRAG_EVALUATING`, `CRAG_EVALUATION_COMPLETE`, `CRAG_REFINING` |
| **Self-RAG Reflection** | `SELF_RAG_REFLECTING`, `SELF_RAG_COMPLETE`, `SELF_RAG_REFINING` |
| **Scratchpad** | `SCRATCHPAD_INITIALIZED`, `SCRATCHPAD_UPDATED`, `SCRATCHPAD_FINDING_ADDED`, `SCRATCHPAD_QUESTION_ANSWERED`, `SCRATCHPAD_GAP_DETECTED` |
| **Coverage** | `COVERAGE_EVALUATING`, `COVERAGE_EVALUATED`, `COVERAGE_INSUFFICIENT` |
| **Refinement** | `REFINEMENT_CYCLE_START`, `REFINEMENT_CYCLE_COMPLETE`, `REFINEMENT_QUERIES_GENERATED` |
| **LLM Calls** | `LLM_CALL_START`, `LLM_CALL_COMPLETE` |
| **Quality** | `CORPUS_QUALITY_ASSESSED` |
| **Web Search** | `WEB_SEARCH_START`, `WEB_SEARCH_COMPLETE`, `WEB_SEARCH_FALLBACK` |
| **Decision Points** | `DECISION_POINT`, `PIPELINE_ROUTED` |
| **Learning** | `EXPERIENCE_CAPTURED`, `OUTCOME_RECORDED` |

**Orchestrator Enhancements (`agentic/orchestrator.py`):**
- Both `search()` and `search_with_events()` now have CRAG + Self-RAG
- Streaming method now uses configurable `max_scrape_refinements` (was hardcoded 1)
- Blended confidence scoring with reflection in both methods
- Full experience distillation and classifier feedback in streaming method
- Comprehensive event emissions at every processing step

**Event Helper Functions:**
```python
from agentic import events

# Query classification events
await emitter.emit(events.classifying_query(request_id, query))
await emitter.emit(events.query_classified(request_id, category, pipeline, complexity, capabilities))

# CRAG events
await emitter.emit(events.crag_evaluating(request_id, document_count))
await emitter.emit(events.crag_evaluation_complete(request_id, quality, relevance, action))
await emitter.emit(events.crag_refining(request_id, refined_queries))

# Self-RAG events
await emitter.emit(events.self_rag_reflecting(request_id, synthesis_length))
await emitter.emit(events.self_rag_complete(request_id, relevance, support_level, usefulness, temporal_conflicts))

# Coverage events
await emitter.emit(events.coverage_evaluating(request_id))
await emitter.emit(events.coverage_evaluated(request_id, score, is_sufficient, gaps))
await emitter.emit(events.coverage_insufficient(request_id, gaps))

# Quality assessment
await emitter.emit(events.corpus_quality_assessed(request_id, confidence, sources, domains, iterations))
```

**Benefits:**
- Full visibility into all agent processing steps in Android app
- Comprehensive debug logging for engineering analysis
- Thorough multi-direction exploration with configurable depth
- Quality-gated corpus generation with refinement cycles
- Real-time progress updates for user feedback

**Module Version**: `agentic/__init__.py` → v0.13.0

#### ✅ Phase 11: Domain-Specific Persistent Scratchpad (Completed 2025-12-27)

Implemented a general-purpose framework for building domain-specific knowledge corpuses that persist across sessions. Designed for technical troubleshooting domains like FANUC robotics, Raspberry Pi, industrial equipment, etc.

**Research Basis:**
- **HybridRAG (2025)**: Entity-focused retrieval with knowledge graphs (97.5% accuracy)
- **GSW (2025)**: Actor-centric memory for 51% token reduction
- **Industrial Knowledge Graphs**: Proven in manufacturing troubleshooting
- **Incremental Learning**: Delta indexing without full corpus rebuild

**New Components (`agentic/domain_corpus.py`):**
- **DomainSchema**: Define domain-specific entity types and relationships
- **DomainCorpus**: SQLite-backed persistent knowledge store with embeddings
- **CorpusBuilder**: Incremental document indexing with LLM entity extraction
- **CorpusRetriever**: Hybrid search (semantic + graph traversal)
- **DomainCorpusManager**: Multi-domain support with unified API

**Pre-built Domain Schemas:**
| Domain | Entity Types | Relationships | Description |
|--------|--------------|---------------|-------------|
| `fanuc_robotics` | 8 types | 6 relations | FANUC robot troubleshooting |
| `raspberry_pi` | 8 types | 7 relations | Raspberry Pi projects/issues |

**Troubleshooting Entity Types:**
```python
TroubleshootingEntityType:
  - error_code    # SRVO-001, GPIO Error
  - component     # J1 motor, GPIO pin
  - symptom       # overcurrent, overheating
  - cause         # worn gearbox, voltage spike
  - solution      # replace component, recalibrate
  - procedure     # mastering, backup/restore
  - parameter     # $PARAM_GROUP, config.txt
  - part_number   # A06B-6079-H101
```

**Key Features:**
- **Incremental Building**: Content hashing prevents duplicate indexing
- **Entity Deduplication**: Canonical names merge similar entities
- **Graph Traversal**: Navigate error → symptom → cause → solution chains
- **Semantic Search**: Embedding-based relevance scoring
- **Persistence**: SQLite-backed with hot cache for performance
- **Cross-Domain Queries**: Search across all registered corpuses

**API Endpoints:**
```
GET  /api/v1/search/corpus/domains                    - List registered domains
GET  /api/v1/search/corpus/{domain_id}/stats          - Domain statistics
POST /api/v1/search/corpus/{domain_id}/documents      - Add document with extraction
POST /api/v1/search/corpus/{domain_id}/query          - Hybrid search query
GET  /api/v1/search/corpus/{domain_id}/entities       - List entities
GET  /api/v1/search/corpus/{domain_id}/graph          - Export knowledge graph
GET  /api/v1/search/corpus/{domain_id}/troubleshoot/{code} - Get troubleshooting path
POST /api/v1/search/corpus/cross-domain/query         - Query all domains
POST /api/v1/search/corpus/register                   - Register custom domain
```

**Usage Example:**
```python
from agentic import (
    DomainCorpus,
    CorpusBuilder,
    CorpusRetriever,
    create_fanuc_schema
)

# Create corpus
corpus = DomainCorpus(schema=create_fanuc_schema(), db_path="fanuc.db")

# Build incrementally
builder = CorpusBuilder(corpus)
result = await builder.add_document(
    content="SRVO-001 servo overcurrent alarm...",
    source_url="manual.pdf",
    source_type="manual"
)
# Result: {"status": "indexed", "entities": 5, "relations": 3}

# Query with hybrid search
retriever = CorpusRetriever(corpus)
result = await retriever.query("motor overcurrent error")
# Result: {entities: [...], related: [...], context: "..."}

# Get troubleshooting path
path = await retriever.get_troubleshooting_path("SRVO-001")
# Result: {error_code: {...}, symptoms: [...], causes: [...], solutions: [...]}
```

**Test Results:**
```
python test_domain_corpus.py
  Schema Creation: PASS
  Corpus Initialization: PASS
  Corpus Persistence: PASS
  Content Deduplication: PASS
  Corpus Manager: PASS
  Semantic Search: PASS
  Troubleshooting Path: PASS
  LLM Entity Extraction: PASS
Passed: 9/9
```

**Module Version**: `agentic/__init__.py` → v0.14.0

#### ✅ Phase 12: SSE Graph Visualization + Enhanced Events (Completed 2025-12-27)

Implemented comprehensive SSE event system with real-time agent graph visualization for Android app display:

**New Components (`agentic/events.py`):**
- **AgentGraphState**: Tracks agent traversal state for visualization
- **Graph Event Types**: `graph_node_entered`, `graph_node_completed`, `graph_state_update`, `graph_edge_traversed`, `graph_branch_created`, `graph_paths_merged`
- **BGE-M3 Events**: `hybrid_search_start/complete`, `bm25_search`, `dense_embedding`, `rrf_fusion`
- **HyDE Events**: `hyde_generating`, `hyde_hypothetical_generated`, `hyde_embedding`, `hyde_complete`
- **RAGAS Events**: `ragas_evaluating`, `ragas_claims_extracted`, `ragas_claim_verified`, `ragas_evaluation_complete`

**Graph Visualization Formats:**
```
Simple:  [A✓]→[P✓]→[S•]→[V]→[Σ]     (active step marked with •)
Dots:    ●─●─◎─○─○                    (● completed, ◎ active, ○ pending)
Names:   Analyze→Plan→*Search*→(Verify)→(Synthesize)
```

**Agent Symbols:**
| Symbol | Agent | Description |
|--------|-------|-------------|
| A | Analyze | Query analysis |
| P | Plan | Search planning |
| S | Search | Web search |
| E | CRAG | Retrieval evaluation |
| V | Verify | Claim verification |
| W | Scrape | Web scraping |
| Σ | Synthesize | Answer synthesis |
| R | Reflect | Self-RAG reflection |
| H | HyDE | Query expansion |
| M | Hybrid | BGE-M3 retrieval |
| Q | RAGAS | Quality evaluation |
| ✓ | Complete | Pipeline complete |

**Full Pipeline Graph Example:**
```
[A✓]→[P✓]→[S✓]→[E✓]→[V✓]→[W✓]→[Σ✓]→[R✓]→[✓✓]
```

**Key Features:**
- **Real-Time Visualization**: Graph line included in every SSE event for live UI updates
- **Agent Status Tracking**: Pending, active (•), completed (✓), or failed (✗)
- **Multi-Path Support**: Branch and merge visualization for parallel exploration
- **Orchestrator Integration**: All major pipeline steps emit graph events
- **Android SSE Parsing**: `graph_line` field in event data for easy UI rendering

**SSE Event Example:**
```json
{
  "event": "graph_node_completed",
  "request_id": "abc123",
  "message": "Analyze ✓ (245ms)",
  "graph_line": "[A✓]→[P•]",
  "data": {
    "agent": "analyze",
    "success": true,
    "duration_ms": 245,
    "graph": {
      "nodes": [...],
      "line_simple": "[A✓]→[P•]",
      "line_dots": "●─◎",
      "line_names": "Analyze→*Plan*"
    }
  }
}
```

**Test Results:**
```
Full Pipeline Test:
  Events: 201
  Graph Events: 19
  Duration: 153.4s
  Final Graph: [A✓]→[P✓]→[S✓]→[E✓]→[V✓]→[W✓]→[Σ✓]→[R✓]→[✓✓]
```

**Module Version**: `agentic/__init__.py` → v0.20.0

#### ✅ Phase 3: DAG-Based Reasoning (Completed 2025-12-27)

Implemented Graph of Thoughts (GoT) style multi-path reasoning with DAG structure:

**New Components:**
- **ReasoningDAG** (`agentic/reasoning_dag.py`): Full DAG-based reasoning with branching, aggregation, and verification
- **ReasoningNode**: Immutable node with parent/child tracking, confidence scores, and evidence linking
- **Topological Verification**: GoV-style verification ensuring premises validated before conclusions

**Key Features:**
- **8 Node Types**: ROOT, HYPOTHESIS, EVIDENCE, CRITIQUE, REFINEMENT, AGGREGATION, CONCLUSION, CONTRADICTION
- **6 Node Statuses**: PENDING, EXPLORING, VALIDATED, INVALIDATED, MERGED, PRUNED
- **LLM-Based Operations**:
  - `branch()`: Generate multiple reasoning paths from a node
  - `aggregate()`: Combine insights from multiple nodes
  - `critique()`: Analyze a node for weaknesses
  - `refine()`: Improve a node based on critique
- **Topological Verification**: Verify nodes in dependency order, propagate confidence
- **Path Pruning**: Automatically prune invalidated reasoning branches
- **Convergent Answer**: Extract final answer from validated sink nodes
- **Full Trace Export**: JSON export for debugging and visualization

**DAG Operations:**
```python
# Create reasoning DAG
dag = ReasoningDAG(ollama_url="http://localhost:11434", model="qwen3:8b")

# Add root query
root_id = dag.add_node("What are the benefits of FastAPI?", NodeType.ROOT)

# Branch into hypotheses (GoT branching)
hypotheses = await dag.branch(root_id, num_branches=3)

# Add evidence
evidence_id = dag.add_evidence(hypotheses[0], "Benchmarks show 300% faster", "https://...")

# Aggregate findings (GoT aggregation)
synthesis_id = await dag.aggregate(hypotheses)

# Verify topologically (GoV verification)
results = dag.verify_topologically()

# Get convergent answer
answer = dag.get_convergent_answer()
```

**Test Results:**
```
python test_reasoning_dag.py
  imports: PASS (Module v0.6.0)
  basic_operations: PASS (Node management, depth, topological sort)
  verification: PASS (Topological verification, path pruning)
  aggregation: PASS (Manual aggregation, trace export)
  serialization: PASS (Node serialization, status updates)
Passed: 5/5 (6th test requires LLM)
```

**Module Version**: `agentic/__init__.py` → v0.6.0

#### ✅ Phase 4: Buffer of Thoughts (Completed 2025-12-27)

Implemented ThoughtLibrary for reusable reasoning templates:

**New Components:**
- **ThoughtLibrary** (`agentic/thought_library.py`): Meta-buffer of reusable reasoning patterns
- **ThoughtTemplate**: Reusable templates with placeholders, embeddings, and success tracking
- **InstantiatedThought**: Template customized with specific context

**Key Features:**
- **8 Default Templates**: source_credibility, compare_options, step_by_step_solution, synthesize_sources, extract_key_info, causal_analysis, research_plan, contradiction_resolution
- **8 Template Categories**: ANALYSIS, VERIFICATION, SYNTHESIS, COMPARISON, PROBLEM_SOLVING, INFORMATION_EXTRACTION, REASONING, PLANNING
- **Semantic Retrieval**: Embedding-based template matching via Ollama
- **Keyword Fallback**: Works without embeddings using keyword overlap
- **Buffer-Manager Learning**: Track success/failure rates per template
- **Create From Success**: Generate new templates from successful reasoning traces
- **JSON Serialization**: Export/import library state

**Research Basis (Buffer of Thoughts paper):**
- Llama3-8B + BoT can surpass Llama3-70B
- Reduces token usage by reusing proven patterns
- Continuous improvement via buffer-manager

**Test Results:**
```
python test_thought_library.py --full
  imports: PASS (Module v0.7.0)
  default_templates: PASS (8 templates, 8 categories)
  instantiation: PASS (Context application, partial fill)
  buffer_manager: PASS (Success/failure tracking)
  keyword_retrieval: PASS (Fallback matching)
  serialization: PASS (JSON export/import)
  create_from_success: PASS (New template creation)
  llm_embedding: PASS (Semantic retrieval, 0.709 similarity)
  top_performers: PASS (Success rate ranking)
Passed: 9/9
```

**Module Version**: `agentic/__init__.py` → v0.7.0

#### ✅ Phase 5: Actor Factory (Completed 2025-12-27)

Implemented AIME-style dynamic agent specialization with tool bundles:

**New Components:**
- **ActorFactory** (`agentic/actor_factory.py`): Create purpose-built agents on demand
- **DynamicActor**: Agent with LLM, toolkit, persona, and memory context
- **ToolBundle**: Pre-packaged tool collections for functional completeness
- **ActorPersona**: Customized role and expertise for each agent

**Key Features:**
- **8 Default Tool Bundles**: web_research, vision_extraction, verification, synthesis, analysis, code_analysis, memory_ops, quick_response
- **7 Model Capabilities**: text_generation, long_context, reasoning, vision, code, embedding, fast
- **Dynamic Assembly**: Actors created per-subtask, not pre-defined
- **Task Analysis**: Automatic requirement detection from task descriptions
- **Capability Matching**: Select optimal model based on bundle requirements
- **Persona Generation**: Role, expertise, and constraints from task context

**AIME Formula**: A_t = {LLM_t, T_t, P_t, M_t}
- LLM_t: Cognitive engine (model selection based on capabilities)
- T_t: Toolkit (selected tool bundles, not individual tools)
- P_t: Persona (customized system prompt)
- M_t: Memory (relevant context from scratchpad)

**Test Results:**
```
python test_actor_factory.py --full
  imports: PASS (Module v0.8.0)
  default_bundles: PASS (8 bundles, 4 tools per bundle)
  task_analysis: PASS (8 task types detected)
  bundle_selection: PASS (Automatic selection)
  capability_matching: PASS (Model requirements)
  persona_generation: PASS (Role, expertise, constraints)
  actor_creation_sync: PASS (Prompt composition)
  tool_registration: PASS (Custom tools/bundles)
  llm_actor_creation: PASS (Full actor with tools)
  actor_serialization: PASS (Dict export, stats)
Passed: 10/10
```

**Module Version**: `agentic/__init__.py` → v0.8.0

#### ✅ Phase 2: GSW-Style Entity Tracking (Completed 2025-12-27)

Implemented GSW (Generative Semantic Workspace) entity extraction and tracking:

**New Components:**
- **EntityTracker** (`agentic/entity_tracker.py`): LLM-based entity extraction with coreference resolution
- **Scratchpad Entity Integration** (`agentic/scratchpad.py`): GSW entity tracking in working memory
- **Analyzer Entity Extraction** (`agentic/analyzer.py`): Entity extraction during content analysis

**Key Features:**
- GSW Operator pattern: Extracts ACTORS, ROLES, STATES, VERBS from content
- Reconciler pattern: Coreference resolution, entity merging, timeline ordering
- 51% token reduction via entity-centric summaries vs full document retrieval
- Query-relevant entity context generation
- Semantic verb frames (Subject-Predicate-Object-Time-Place)
- Entity relations with typed relationships (created_by, depends_on, part_of)

**Entity Types:**
`person`, `organization`, `product`, `technology`, `concept`, `location`, `event`, `date`, `quantity`, `other`

**Role Types:**
`creator`, `maintainer`, `user`, `competitor`, `component`, `feature`, `benefit`, `drawback`, `alternative`, `dependency`

**Test Results:**
```
python test_entity_tracker.py --full
  imports: PASS (Module v0.5.0)
  entity_tracker: PASS (Basic operations, merging, summaries)
  scratchpad: PASS (Entity storage, relations, relevance scoring)
  analyzer: PASS (Enable/disable, context generation)
  llm_extraction: PASS (7 entities, 4 relations in 29s)
Passed: 5/5
```

**Module Version**: `agentic/__init__.py` → v0.5.0

#### ✅ Phase 1: AIME-Style Dynamic Planning (Completed 2025-12-27)

Implemented dual strategic/tactical planning based on AIME (ByteDance) research:

**New Components:**
- **DynamicPlanner** (`agentic/dynamic_planner.py`): Dual-output planning with hierarchical TaskNode structure
- **Progress Tools** (`agentic/progress_tools.py`): ProgressReporter, ProgressAggregator for agent progress reporting
- **DynamicOrchestrator** (`agentic/orchestrator_dynamic.py`): Full integration with iterative replanning

**Key Features:**
- Dual outputs: Strategic task hierarchy + Tactical next action per iteration
- Formula: `(L_{t+1}, g_{t+1}) = LLM(goal, L_t, H_t)` where L=task list, g=action, H=history
- Hierarchical TaskNode with dependencies, completion criteria, and artifacts
- Real-time progress tracking via ProgressReporter
- Scratchpad integration for shared working memory

**Test Results:**
```
python test_dynamic_orchestrator.py --full
  dynamic_planner: PASS (16.2s decomposition, 30.2s replan)
  scratchpad: PASS (Task hierarchy + progress tracking)
  progress_tools: PASS (Async events + aggregation)
  orchestrator: PASS (23s end-to-end execution)
Passed: 4/4
```

**Module Version**: `agentic/__init__.py` → v0.4.0

## Current Status (2025-12-30)

**Module Version**: `agentic/__init__.py` → v0.40.0 (Part F: Benchmark Test Suite + Technical Accuracy Scorer)
