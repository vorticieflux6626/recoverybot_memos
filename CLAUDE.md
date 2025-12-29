# memOS Server Development Status

## Project Overview
memOS is a memory management, quest/gamification, and **intelligent data injection** system for the Recovery Bot Android application. It provides REST APIs for storing user memories, tracking progress through quests and achievements, and **orchestrating agentic AI workflows for enhanced research and troubleshooting assistance**.

## Strategic Vision: Intelligent Research Hub

memOS is the **central intelligence layer** for the Recovery Bot ecosystem, responsible for:

1. **Memory Management** (Current) - Secure storage and semantic search
2. **Quest Gamification** (Current) - Progress tracking and achievements
3. **Agentic Search Orchestration** (Active) - Multi-agent web search and context enhancement
4. **Context Injection** (Active) - Intelligent data augmentation for LLM conversations

### Core Architecture Principle
memOS serves as the **Single Source of Truth (SSOT)** for user context, memory, and intelligent data retrieval. All context augmentation flows through memOS before reaching the primary LLM.

## Current Status (2025-12-28)

### Next-Gen Enhancement Plan (December 2025)

Comprehensive research into cutting-edge agentic AI frameworks has produced a detailed enhancement roadmap:

**Research Sources:**
| Framework | Innovation | Expected Impact |
|-----------|------------|-----------------|
| **AIME** (ByteDance) | Dynamic Planner, Actor Factory, Progress Management | 77.6% GAIA benchmark |
| **GSW** (Hippocampal Memory) | Actor-centric episodic memory, entity tracking | 51% token reduction |
| **DAG-Math/DoT** | Graph-structured reasoning | Perfect reasoning rate |
| **Graph of Thoughts** | Multi-path exploration with convergence | 200-300% over ToT |
| **Buffer of Thoughts** | Reusable thought-template library | 8B model surpasses 70B |

**Implementation Status:**
| Phase | Component | Status |
|-------|-----------|--------|
| **Phase 1** | AIME-Style Dynamic Planning | ✅ **COMPLETE** |
| **Phase 2** | GSW Entity Tracker | ✅ **COMPLETE** |
| **Phase 3** | Reasoning DAG | ✅ **COMPLETE** |
| **Phase 4** | Thought Template Library | ✅ **COMPLETE** |
| **Phase 5** | Actor Factory | ✅ **COMPLETE** |
| **Phase 6** | Self-RAG Reflection | ✅ **COMPLETE** |
| **Phase 7** | CRAG Retrieval Evaluator | ✅ **COMPLETE** |
| **Phase 8** | Experience Distillation | ✅ **COMPLETE** |
| **Phase 9** | Classifier Feedback Loop | ✅ **COMPLETE** |
| **Phase 10** | SSE Visibility + Thorough Search | ✅ **COMPLETE** |
| **Phase 11** | Domain-Specific Persistent Scratchpad | ✅ **COMPLETE** |
| **Phase 12** | SSE Graph Visualization + Enhanced Events | ✅ **COMPLETE** |
| **Phase 13** | Universal Orchestrator + Bug Fixes | ✅ **COMPLETE** |
| **Phase 14** | Context Utilization Tracking | ✅ **COMPLETE** |

#### ✅ Phase 15: Orchestrator Consolidation (Completed 2025-12-28)

Consolidated all orchestrators into UniversalOrchestrator as the single source of truth:

**DEPRECATED Orchestrators:**
| Class | Replacement |
|-------|-------------|
| `AgenticOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)` |
| `EnhancedAgenticOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)` |
| `DynamicOrchestrator` | `UniversalOrchestrator(enable_dynamic_planning=True)` |
| `GraphEnhancedOrchestrator` | `UniversalOrchestrator(enable_graph_cache=True)` |
| `UnifiedOrchestrator` | `UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)` |

**Code Reduction:**
- 5 deprecated orchestrator files (~5,342 lines)
- Single source of truth: `orchestrator_universal.py`
- 38+ feature flags via presets

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
| `research` | 35 | Academic/thorough |
| `full` | 38+ | Maximum capability |

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
| `research` | 35 features | Academic/thorough research |
| `full` | 38+ features | Maximum capability |

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

## Current Status (2025-12-26)

### ✅ Completed Components

#### 0. Sherpa-ONNX TTS Model Serving (December 2025)
- **Static File Serving**: Models served from `/api/models/sherpa-onnx/{model_dir}/{file}`
- **Models Directory**: `/home/sparkone/sdd/Recovery_Bot/memOS/models/sherpa-onnx/`
- **Available Models**:
  - `vits-piper-en_US-lessac-medium` (65MB) - US Female
  - `vits-piper-en_US-kristin-medium` (65MB) - US Female
  - `vits-piper-en_US-glados` (65MB) - GLaDOS AI voice
  - `vits-piper-en_US-libritts_r-medium` (79MB) - Multi-speaker (904 voices)
  - `vits-piper-en_GB-cori-medium` (65MB) - UK Female
  - `vits-piper-en_GB-jenny_dioco-medium` (65MB) - UK Female
  - `vits-piper-en_GB-alan-medium` (65MB) - UK Male
- **Android Access URL**: `https://technobot.sparkonelabs.com:8443/memOS/models/sherpa-onnx/`
- **nginx Routing**: `/memOS/` → `/api/`, so `/memOS/models/` → `/api/models/`

#### 0.5. Advanced TTS Engines (NEW - December 2025)
Three TTS engines now available with emotion control and voice cloning:

**EmotiVoice (Apache 2.0 - Commercial OK)**
- **Endpoint**: `POST /api/tts/emotivoice/synthesize`
- **Features**: Prompt-based emotion control, 2000+ built-in speakers from LibriTTS/HiFiTTS
- **Emotions**: Happy, Sad, Angry, Empathetic, Encouraging, Calm, Excited, Gentle, Soothing, Seductive, etc.
- **Languages**: English, Chinese
- **Output**: 16-bit PCM mono 22050Hz
- **VRAM**: ~1.5GB when loaded (hot-swappable)

**Verified Charming Female Voices (from EmotiVoice wiki):**
| Voice Key | Speaker ID | Name | Description |
|-----------|------------|------|-------------|
| `female_inviting` | 3559 | Kerry Hiles | Soothing, clear, inviting - BEST for seductive |
| `female_soothing` | 8051 | Maria Kasper | Clear, soothing, expressive (277 LibriVox audiobooks!) |
| `female_melodic` | 11614 | Sylviamb | Crisp, melodic, captivating |
| `female_lively` | 92 | Cori Samuel | Lively, expressive, energetic |
| `female_warm` | 1088 | - | Recommended warm female |
| `female_gentle` | 1093 | - | Recommended gentle female |
| `female_soft` | 225 | - | Recommended soft female |
| `female_sweet` | 102 | - | Recommended sweet female |
| `female_breathy` | 65 | - | Recommended breathy female |

**Sources:** [EmotiVoice Wiki](https://github.com/netease-youdao/EmotiVoice/wiki/%F0%9F%98%8A-voice-wiki-page), [LibriVox](https://librivox.org/reader/8051)

**OpenVoice (MIT License - Commercial OK)**
- **Endpoint**: `POST /api/tts/openvoice/synthesize`
- **Features**: Voice cloning from 10-30s samples, style transfer
- **Styles**: default, friendly, cheerful, excited, sad, angry, terrified, shouting, whispering
- **Voice Registration**: `POST /api/tts/openvoice/register-voice`
- **Languages**: English, Chinese
- **Output**: 16-bit PCM mono 22050Hz
- **VRAM**: ~3GB when loaded (hot-swappable)

**Edge-TTS (Microsoft - Free for personal use)**
- **Endpoint**: `GET /api/tts/base_tts/` or `GET /api/tts/synthesize_speech/`
- **Features**: 322 neural voices, personality presets, speed/pitch control
- **Voices**: Various accents (US, UK, AU, etc.)
- **Output**: 16-bit PCM mono 22050Hz
- **VRAM**: None (cloud-based)

**TTS API Endpoints:**
```
GET  /api/tts/engines                    - List all TTS engines and availability
GET  /api/tts/emotivoice/emotions        - List EmotiVoice emotion prompts
GET  /api/tts/emotivoice/speakers        - List EmotiVoice speaker presets
POST /api/tts/emotivoice/synthesize      - Synthesize with emotion control
GET  /api/tts/openvoice/styles           - List OpenVoice styles
GET  /api/tts/openvoice/voices           - List registered voice clones
POST /api/tts/openvoice/register-voice   - Register voice sample for cloning
POST /api/tts/openvoice/synthesize       - Synthesize with style/cloned voice
POST /api/tts/models/unload              - Unload models to free VRAM
GET  /api/tts/models/status              - Check which models are loaded
```

**CLI Testing:**
```bash
./test_tts.sh engines              # List available engines
./test_tts.sh emotivoice "Hello" Empathetic
./test_tts.sh openvoice "Hello" friendly
./test_tts.sh all-emotions         # Test all EmotiVoice emotions
./test_tts.sh all-styles           # Test all OpenVoice styles
./test_tts.sh compare "Test text"  # Compare all engines
./test_tts.sh unload               # Free VRAM
```

#### 1. Agentic Search System (December 2025)
- **Full Implementation**: ReAct-based multi-step search with URL evaluation
- **Intelligent URL Scraping**: LLM evaluates URL relevance before scraping (up to 8 URLs)
- **Content Synthesis**: Full-content synthesis with source citations using qwen3:8b
- **SSE Streaming**: Real-time progress events for Android client integration
- **Search Modes**: Fixed, Adaptive, and Exhaustive search strategies
- **Context Window**: 32K tokens for large content synthesis
- **Confidence Scoring**: Multi-signal confidence calculation using verification (40%), source diversity (25%), content depth (20%), and synthesis quality (15%)
- **SearXNG Integration** (NEW - December 2025):
  - Self-hosted metasearch engine replacing rate-limited DuckDuckGo
  - Aggregates results from Google, Bing, DuckDuckGo, Brave, Wikipedia, arXiv
  - No rate limits, caching allowed, self-hosted privacy
  - Provider priority: SearXNG → Brave → DuckDuckGo (cascading fallback)
  - Location: `/home/sparkone/sdd/Recovery_Bot/searxng/` (separate git repo)
  - Key files:
    - `agentic/searcher.py` - `SearXNGSearchProvider` class with multi-provider fallback
    - `agentic/searxng_search.py` - Async client for direct SearXNG API access
    - `../searxng/searxng_client.py` - Standalone Python client
  - API: `http://localhost:8888/search?q=query&format=json&engines=google,bing`
- **Post-Scrape Content Coverage Evaluation** (NEW - December 2025):
  - Evaluates scraped content against decomposed questions using qwen3:8b
  - Identifies specific information gaps (e.g., missing costs, requirements, contact info)
  - Generates targeted refinement queries to fill gaps
  - Loops up to 2 refinement rounds for convergence
  - Returns coverage score (0-1) and list of unanswered questions
- **Scratchpad/Blackboard Architecture** (NEW - December 2025):
  - Shared working memory for multi-agent coordination
  - Enables intelligent direction from higher-order processes
  - Features:
    - Mission decomposition with explicit completion criteria per question
    - Finding repository with source attribution and confidence scores
    - Gap detection for incomplete answers
    - Contradiction tracking between conflicting sources
    - Agent-to-agent communication via notes
    - Search history to avoid redundant queries/scrapes
    - Priority queue for next actions
  - Based on research into LLM-based Multi-Agent Systems (LbMAS) patterns:
    - Serialized turns prevent concurrent write conflicts
    - 96.5% convergence within 3 iterations empirically
    - Message cleaning for token efficiency
  - Key files:
    - `agentic/scratchpad.py` - Core AgenticScratchpad and ScratchpadManager classes
    - `agentic/SCRATCHPAD_INTEGRATION.md` - Integration documentation
    - `agentic/orchestrator.py` - Integrated with scratchpad lifecycle

- **Graph-Based KV Cache System** (NEW - December 2025):
  - Based on cutting-edge research: KVFlow (NeurIPS 2025), ROG (2025), LbMAS (2025)
  - Features:
    - **Agent Step Graph**: DAG representing workflow dependencies with steps-to-execution (STE) for eviction priority
    - **Proactive Prefetching**: Loads KV cache for likely next agents during current agent execution
    - **Scratchpad Cache**: ROG-style intermediate answer caching + semantic finding deduplication
    - **Mission Decomposition Cache**: Reuses query decomposition patterns for similar queries
    - **Prefix-Optimized Prompts**: Hierarchical prompt structure (system→role→scratchpad→task) for maximum cache reuse
  - Key files:
    - `agentic/agent_step_graph.py` - Workflow-aware cache eviction (KVFlow-inspired)
    - `agentic/scratchpad_cache.py` - Intermediate caching (ROG-inspired)
    - `agentic/prefix_optimized_prompts.py` - Prompt templates for cache hits
    - `agentic/graph_cache_integration.py` - Integration wrapper for orchestrator

- **DeepSeek-R1 Query Classification** (NEW - December 2025):
  - Initial query classification using DeepSeek-R1 14B Q8 thinking model
  - Chain-of-Draft prompting for 50-80% reduction in thinking tokens
  - Query Categories:
    - `research`: Information gathering, learning about topics
    - `problem_solving`: Debugging, troubleshooting, finding solutions
    - `factual`: Direct questions with verifiable answers
    - `creative`: Open-ended brainstorming, ideation
    - `technical`: Code, engineering, scientific analysis
    - `comparative`: Evaluating options, comparing alternatives
    - `how_to`: Step-by-step guidance, tutorials
  - Pipeline Routing:
    - `direct_answer`: Simple LLM response, no search needed
    - `web_search`: Basic web search + synthesis
    - `agentic_search`: Full multi-agent pipeline
    - `code_assistant`: Technical/code analysis mode
  - API Endpoint: `POST /api/v1/search/classify`
  - Key file: `agentic/query_classifier.py`

- **Generalized System Prompts** (NEW - December 2025):
  - Converted from recovery-focused to general research/problem-solving
  - Updated CORE_SYSTEM_PREFIX for broad applicability
  - Updated agent suffixes (ANALYZER, SYNTHESIZER, VERIFIER)
  - Removed domain-specific constraints for flexibility

- **Enhanced Reasoning Patterns** (NEW - December 2025):
  - Research-backed improvements from 2025 agentic AI literature
  - **Performance Improvements** (vs baseline):
    - Confidence: 0.87 avg vs 0.65 baseline (34% improvement)
    - Duration: 86s avg vs 95s baseline (10% faster)
    - Aspect coverage: 100% vs 100% (maintained)
  - Features:
    - **Pre-Act Pattern** (arXiv 2505.09970): Creates multi-step execution plans BEFORE acting
      - Enables parallel execution of independent actions
      - 70% accuracy improvement over standard ReAct in research
    - **Self-Reflection Loop**: Critiques synthesis quality, refines if score < 0.85
      - Uses gemma3:4b for fast quality evaluation
      - Up to 2 refinement iterations
    - **Stuck State Detection**: Detects loops and attempts recovery
      - Monitors repeated queries and synthesis similarity
      - Recovery strategies: broaden, narrow, rephrase, simplify, accept
    - **Parallel Action Execution**: Runs independent searches concurrently
      - Up to 4 parallel queries per batch
      - Reduces wall-clock time significantly
    - **Contradiction Detection**: Surfaces conflicting information from sources
      - Presents both viewpoints rather than arbitrarily choosing
      - Includes resolution suggestions
  - Key files:
    - `agentic/enhanced_reasoning.py` - Core enhanced reasoning engine
    - `agentic/orchestrator_enhanced.py` - Enhanced orchestrator integrating all patterns
  - API Endpoints:
    - `POST /api/v1/search/enhanced` - Enhanced agentic search
    - `GET /api/v1/search/enhanced/stats` - Enhanced orchestrator statistics
    - `GET /api/v1/search/graph/stats` - Comprehensive cache statistics
    - `GET /api/v1/search/graph/agent-step-graph` - Agent transition probabilities
    - `GET /api/v1/search/graph/scratchpad-cache` - Finding/subquery cache stats
    - `GET /api/v1/search/graph/eviction-candidates` - STE-based eviction candidates
    - `POST /api/v1/search/graph/initialize` - Initialize graph cache system

#### 2. VL Screenshot Scraper (NEW - December 2025)
- **Vision-Language Web Scraping**: Uses VL models (qwen3-vl, llama3.2-vision) to extract data from JS-rendered pages
- **Playwright Integration**: Lightweight screenshot capture with scroll-and-capture for lazy loading
- **Dynamic Model Selection**: Automatically selects most powerful available VL model within VRAM constraints
- **Relevance Evaluation**: Second-pass LLM evaluates extracted data for relevance to recovery services
- **Multiple Extraction Types**: RECOVERY_SERVICE, CONTACT_INFO, MEETING_SCHEDULE, GENERAL_INFO

#### 3. Intelligent Model Selection (NEW - December 2025)
- **LLM-Synthesized Descriptions**: Uses gemma3:4b to generate tool-optimized descriptions for all 89 models
- **Smart Refresh Logic**: Synthesis only runs for NEW models or those with MISSING descriptions
- **Description Content**: Primary use case, capabilities, context window, trade-offs
- **VRAM-Aware Selection**: ModelSelector queries GPU status for optimal model fitting
- **Capability Detection**: Auto-detects vision, reasoning, code, embedding capabilities from model names and descriptions

#### 4. Memory Management System
- **REST API Endpoints**: Full CRUD operations for memory storage
- **Semantic Search**: Ollama-powered vector embeddings for intelligent memory retrieval
- **HIPAA Compliance**: Encrypted storage with audit logging
- **User Settings**: Privacy controls and consent management
- **Database**: PostgreSQL with pgvector extension

#### 5. Quest System Implementation
- **Database Schema**: Complete schema for quests, tasks, achievements, and user progress
- **Core Models**: SQLAlchemy models with Pydantic validation
- **Service Layer**: Quest assignment, progress tracking, and achievement logic
- **Sample Data**: 16 quests across 8 categories (Daily, Weekly, Milestone, Life Skills, Community, Emergency, Wellness, Spiritual)
- **REST API**: Read endpoints working correctly

### 🔧 Current Issues

#### 1. Async/Greenlet Event Loop Conflict
- **Issue**: SQLAlchemy async sessions conflict with FastAPI's event loop
- **Error**: `greenlet_spawn has not been called; can't call await_only() here`
- **Impact**: Quest assignment and other write operations fail
- **Workaround**: Created `quest_service_fixed.py` that accepts session as parameter
- **Status**: Partial fix implemented, needs testing

#### 2. Memory Service Integration
- **Issue**: MemoryService also creates its own AsyncSession instances
- **Impact**: Similar greenlet errors when storing quest completion memories
- **Next Step**: Refactor MemoryService to accept session parameter

### 📊 Test Results

#### ✅ Working Endpoints
- `GET /api/v1/quests/categories` - Returns 7 quest categories
- `GET /api/v1/quests/available` - Returns available quests for user
- `GET /api/v1/quests/users/{user_id}/stats` - Returns user statistics
- `GET /api/v1/quests/users/{user_id}/daily` - Returns daily quest info
- `GET /api/v1/memory/health` - Health check endpoint
- All memory read endpoints

#### ❌ Failing Endpoints
- `POST /api/v1/quests/{quest_id}/assign` - Greenlet error
- `PUT /api/v1/quests/tasks/{task_id}/complete` - Greenlet error
- Memory write operations when called from quest service

### 🚀 Next Steps

1. **Fix Async Issues**
   - Refactor MemoryService to accept session parameter
   - Update all service methods to use dependency-injected sessions
   - Test quest assignment and task completion

2. **Complete Quest System**
   - Fix quest assignment functionality
   - Implement quest verification workflow
   - Add achievement checking
   - Create admin endpoints for quest management

3. **Android Integration**
   - Document API endpoints in TELEPHONE.md
   - Create Kotlin data models
   - Design quest UI components
   - Implement offline sync strategy

4. **Deployment**
   - Deploy to Ollama server
   - Configure production database
   - Set up monitoring and logging
   - Performance testing

## Development Commands

```bash
# ============================================
# IMPORTANT: Apply Ollama optimizations first!
# ============================================
source /home/sparkone/sdd/Recovery_Bot/memOS/server/setup_ollama_optimization.sh
# Then restart Ollama if running: systemctl restart ollama

# Start the server
cd /home/sparkone/sdd/Recovery_Bot/memOS/server
source venv/bin/activate
python -m uvicorn main:app --reload --port 8001

# Or use convenience scripts
./start_server.sh                # Start server in background
./stop_server.sh                 # Stop server
./restart_server.sh              # Restart server
./status_server.sh               # Check server status
./logs_server.sh                 # Tail server logs

# Run tests
./test_system.sh                 # Full system test suite
./test_system.sh quick           # Quick tests (no LLM calls)
./test_system.sh hybrid          # Test BGE-M3 hybrid retrieval
./test_system.sh hyde            # Test HyDE query expansion
./test_system.sh ragas           # Test RAGAS evaluation
./test_system.sh api             # Test API endpoints
python test_quest_simple.py      # Test read operations (working)
python test_quest_assignment.py  # Test full workflow (has issues)
python test_vl_scraper.py        # Test VL screenshot scraper

# Model management
curl -X POST "http://localhost:8001/api/v1/models/refresh?force=true"  # Refresh model specs
curl -X POST "http://localhost:8001/api/v1/models/refresh?resynthesize_all=true"  # Re-synthesize descriptions
curl "http://localhost:8001/api/v1/models/specs?capability=vision"  # Get vision models
curl "http://localhost:8001/api/v1/models/gpu/status"  # Check GPU/VRAM status

# Agentic search testing
curl -X POST "http://localhost:8001/api/v1/search/agentic" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "max_iterations": 3}'

# Database management
python init_database.py          # Initialize schema
python create_sample_quests.py   # Populate sample quests

# Verify Ollama optimization settings
env | grep OLLAMA
```

## Technical Stack

- **Framework**: FastAPI with async support
- **Database**: PostgreSQL 15 with pgvector
- **ORM**: SQLAlchemy 2.0 with async sessions
- **Validation**: Pydantic v2
- **AI/Embeddings**: Ollama with llama3.2:3b
- **Authentication**: JWT tokens with refresh
- **Logging**: Python logging with audit trail

## Architecture Notes

- Service-oriented architecture with clear separation of concerns
- Repository pattern for data access
- Dependency injection for database sessions
- HIPAA-compliant data handling throughout
- Event-driven updates for real-time features

---

## Agentic Search Architecture (December 2025) - IMPLEMENTED

### Overview

The agentic search system implements a **ReAct (Reasoning + Acting)** pattern for intelligent web search and context injection. This enables multi-step reasoning, query decomposition, and verification before injecting search results into the main LLM conversation.

### Recent Fixes (2025-12-25)
- **Fixed empty synthesis**: Increased `num_ctx` from 16K to 32K to accommodate large prompts
- **Added URL scraping to non-streaming endpoint**: Both `/agentic` and `/stream` now scrape content
- **Improved source citations**: Synthesis now includes `[Source X]` citations throughout
- **Enhanced logging**: Added prompt length and response length tracking for debugging

### Performance Optimizations (2025-12-26)

**IMPORTANT**: Before starting Ollama, apply the optimization configuration:

```bash
# Apply Ollama KV cache and performance optimizations
source /home/sparkone/sdd/Recovery_Bot/memOS/server/setup_ollama_optimization.sh
systemctl restart ollama  # or: pkill ollama && ollama serve
```

**Implemented Optimizations:**

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| **Chain-of-Draft Prompting** | 50-80% thinking token reduction | `synthesizer.py` - prepends CoD instruction for DeepSeek R1 |
| **DeepSeek R1 Parameters** | Improved reasoning quality | `temperature=0.6`, `top_p=0.95` (validated by DeepSeek) |
| **KV Cache Quantization** | 50% VRAM reduction | `OLLAMA_KV_CACHE_TYPE=q8_0` |
| **Flash Attention** | 10-20% faster attention | `OLLAMA_FLASH_ATTENTION=1` |
| **Model Persistence** | Faster subsequent queries | `OLLAMA_KEEP_ALIVE=30m` |
| **Reduced Refinements** | 20s saved per query | `max_scrape_refinements=1` |

**Performance Results:**

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

**Key Files:**
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

**Cache & Performance API Endpoints:**
- `GET /api/v1/search/cache/stats` - View content cache statistics
- `GET /api/v1/search/ttl/stats` - View TTL pinning statistics and tool latencies
- `GET /api/v1/search/metrics` - View performance metrics (TTFT, tokens, cache hits)
- `GET /api/v1/search/artifacts/stats` - View artifact store statistics
- `DELETE /api/v1/search/cache` - Clear all caches
- `DELETE /api/v1/search/artifacts/{session_id}` - Clean up session artifacts

**Phase 4 Memory Tier API Endpoints:**
- `GET /api/v1/search/memory/tiers/stats` - View three-tier memory statistics
- `GET /api/v1/search/memory/kv-cache/stats` - View KV cache service stats
- `GET /api/v1/search/memory/kv-cache/warm` - List warm cache entries
- `POST /api/v1/search/memory/kv-cache/warm` - Warm a prefix in KV cache
- `POST /api/v1/search/memory/tiers/store` - Store content in memory tiers
- `GET /api/v1/search/memory/tiers/{content_id}` - Retrieve content
- `POST /api/v1/search/memory/tiers/{content_id}/promote` - Promote cold→warm
- `POST /api/v1/search/memory/tiers/{content_id}/demote` - Demote warm→cold
- `POST /api/v1/search/memory/initialize` - Initialize and warm system prompts

**Future Optimizations (See KV_CACHE_IMPLEMENTATION_PLAN.md):**
- Phase 3: vLLM migration (40-60% additional TTFT reduction) - skipped for now

### Design Rationale

Current Android implementation uses a simple web search pattern:
```
User Query → Extract Search Keywords → Single Web Search → Inject Results
```

The new agentic approach:
```
User Query → Planner Agent → [Decomposed Queries] → Searcher Agents →
Verifier Agent → Synthesizer Agent → Verified Context → Main LLM
```

### MCP Node Editor Integration

memOS leverages the **MCP Node Editor** (`/home/sparkone/sdd/MCP_Node_Editor`) as the underlying workflow orchestration engine. This provides:

- **27 Node Types**: Including `agent_orchestrator`, `web_search`, `rag_pipeline`, `memory`
- **Cyclic Workflows**: Iterative refinement until convergence
- **Event-Driven Architecture**: 1000+ events/sec throughput
- **Code Sandboxing**: Safe execution of generated code
- **Circuit Breakers**: Automatic error loop prevention

See `mcp_node_editor_integration.md` for full API reference.

### Agentic Pipeline Architecture

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

### ReAct Loop Implementation

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

### API Endpoints (New)

```python
# Agentic Search Endpoints (to be implemented)
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

### Hybrid Scoring Algorithm

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

### Edge Model Query Optimization

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

### Implementation Phases

| Phase | Components | Effort |
|-------|-----------|--------|
| **Phase 1** | Simple search endpoint, basic MCP integration | 3-5 days |
| **Phase 2** | Planner + Searcher agents, ReAct loop | 5-7 days |
| **Phase 3** | Verifier agent, confidence scoring | 3-5 days |
| **Phase 4** | Full integration with Android client | 3-5 days |

### File Structure (New)

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

### Integration with Memory Service

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

### Security Considerations

1. **Source Allowlisting**: Only search trusted domains for recovery-related content
2. **Content Filtering**: PHI detection before injecting search results
3. **Rate Limiting**: Per-user limits on agentic search operations
4. **Audit Logging**: Track all search queries and results for compliance

---

## Unified Architecture Integration

memOS follows the Recovery Bot **Unified Architecture Recommendations** (see `/UNIFIED_ARCHITECTURE_RECOMMENDATIONS.md`):

### Response Format Compliance

All memOS endpoints return the unified response envelope:
```json
{
  "success": true,
  "data": { ... },
  "meta": {
    "timestamp": "2025-12-24T00:00:00Z",
    "request_id": "uuid",
    "version": "1.0.0"
  },
  "errors": []
}
```

### SSOT Data Ownership

| Domain | Owner |
|--------|-------|
| User Memories | memOS PostgreSQL |
| Quest Progress | memOS PostgreSQL |
| Search Context | memOS PostgreSQL |
| User Settings | memOS PostgreSQL |

### Cross-Service Communication

memOS publishes events for other services:
```python
# Event types
"memory.stored"      # New memory added
"quest.completed"    # Quest milestone reached
"search.completed"   # Agentic search finished
"context.injected"   # Context added to session
```

---

## Embedding Models Research (December 2025)

### Available Ollama Embedding Models

| Model | Parameters | Dimensions | Context | Size | Multilingual | MTEB Score |
|-------|------------|------------|---------|------|--------------|------------|
| **qwen3-embedding:8b** | 8B | 4096 (MRL: 32-4096) | 40K | 4.7GB | 100+ languages | **70.58** |
| **qwen3-embedding:4b** | 4B | 2560 | 40K | 2.5GB | 100+ languages | 81.20 (Code) |
| **qwen3-embedding:0.6b** | 0.6B | 1024 | 32K | 639MB | 100+ languages | 64.33 |
| **mxbai-embed-large** | 335M | 1024 (MRL: 64-1024) | 512 | 670MB | English | 64.68 |
| **snowflake-arctic-embed2** | 568M | 768 (MRL: 128+) | 8K | 1.2GB | Multilingual | 55.98 |
| **nomic-embed-text** | 137M | 768 (MRL: 64-768) | 8K | 274MB | English | 53.01 |
| **granite-embedding:278m** | 278M | 768 | - | 560MB | Multilingual | - |

**MRL** = Matryoshka Representation Learning (flexible dimension truncation)

### Locally Available Models

```bash
# Current local embedding models
qwen3-embedding:latest     # 4.68GB - Primary high-quality model
snowflake-arctic-embed2    # 1.16GB - Good multilingual
mxbai-embed-large          # 670MB - Fast English-only
nomic-embed-text           # 274MB - Lightweight with long context
granite-embedding:278m     # 560MB - IBM Granite
granite-embedding:30m      # 60MB - Ultra-lightweight
```

### Recommended Configuration

For the domain corpus system (FANUC robotics, Raspberry Pi troubleshooting):

| Use Case | Model | Rationale |
|----------|-------|-----------|
| **Primary** | qwen3-embedding:4b | 81.20 MTEB-Code, best for technical docs |
| **Fast Fallback** | mxbai-embed-large | 1024d, fast inference, good English |
| **Long Documents** | nomic-embed-text | 8K context window |
| **Maximum Quality** | qwen3-embedding:8b | Highest MTEB score (70.58) |

### Dimension Alignment Techniques

When mixing embeddings from different models:

1. **MRL Truncation** (Recommended): Truncate larger to match smaller
   ```python
   qwen_emb = model.encode(text)[:1024]  # Truncate 4096 → 1024
   qwen_emb = qwen_emb / np.linalg.norm(qwen_emb)  # Re-normalize
   ```

2. **Linear Projection**: Project smaller to larger dimension
   ```python
   projector = nn.Linear(768, 1024)  # Train on paired data
   ```

3. **Orthogonal Procrustes**: Align embedding spaces while preserving geometry
   ```python
   R, _ = orthogonal_procrustes(model_a_anchors, model_b_anchors)
   aligned = embeddings @ R
   ```

### Quantization Effects

| Quantization | Compression | Quality Impact |
|-------------|-------------|----------------|
| float32 → float16 | 2x | Negligible |
| float32 → int8 | 4x | Minor (needs calibration) |
| float32 → binary | 32x | Significant (needs re-ranking) |
| FP8 (E4M3) | 4x | Better than INT8 for NLP |

**Recommended Pipeline**: Binary search → INT8 re-scoring → Cross-encoder re-ranking

### VRAM Usage (24GB TITAN RTX)

| Configuration | VRAM Used | Room for LLM |
|---------------|-----------|--------------|
| qwen3-embedding:8b | ~5.5GB | 18.5GB |
| qwen3-embedding:4b | ~3.0GB | 21GB |
| mxbai-embed-large | ~0.9GB | 23.1GB |
| nomic-embed-text | ~0.4GB | 23.6GB |

### Research References

- **RouterRetriever** (arXiv:2409.02685): Similarity-based routing to domain experts
- **HF-RAG** (arXiv:2509.02837): Z-score normalization for cross-source fusion
- **MRL** (NeurIPS 2022): Matryoshka Representation Learning
- **Procrustes Alignment** (arXiv:2510.13406): Cross-model embedding alignment
- **PCA-RAG** (arXiv:2504.08386): 28.6x index reduction with moderate accuracy loss

### Mixed-Precision Embedding System (December 2025)

Implements precision-stratified embedding retrieval based on the "bounding hyperspace" hypothesis:
- Higher-precision embeddings (fp16) serve as semantic reference frames
- Lower-precision embeddings (int8/binary) for efficient coarse retrieval
- Semantic residuals capture what's lost in quantization

**Key Components:**
- **MixedPrecisionEmbeddingService** (`agentic/mixed_precision_embeddings.py`)
- **Three-Tier Indexing**: Binary (32x compression) → Int8 (4x) → FP16 (full quality)
- **Semantic Residual Manager**: Precision-guided operations
- **Anchor Embeddings**: Category-specific semantic reference frames

**Three-Stage Search Pipeline:**
```
User Query
    |
    v
[Qwen3-Embedding (4096-dim fp16)]
    |
    +---> [Binary Index] ---> Top-500 (Hamming distance)
    |            |
    |            v
    +---> [Int8 Index] ----> Top-50 (cosine similarity)
    |            |
    |            v
    +---> [FP16 Store] ----> Top-10 (high-precision)
                 |
                 v
          Retrieved Documents
```

**Compression Ratios:**
| Precision | Compression | Accuracy | Memory/doc (4096d) |
|-----------|-------------|----------|-------------------|
| Binary | 32x | ~92.5% | 512 bytes |
| Int8 | 4x | 95-99% | 4 KB |
| FP16 | 1x | 100% | 8 KB |

**MRL Hierarchical Search:**
Exploits Matryoshka Representation Learning for progressive refinement:
- Stage 1: 64 dimensions (coarse semantics, fast filtering)
- Stage 2: 256 dimensions (balanced precision)
- Stage 3: 1024 dimensions (fine-grained ranking)
- Stage 4: 4096 dimensions (full precision final)

**Semantic Operations:**
```python
# Semantic arithmetic (Word2Vec-style analogies)
result = service.semantic_arithmetic(
    base=homeless_shelter_emb,
    add=addiction_recovery_emb,
    subtract=basic_housing_emb
)
# Result: embedding closer to "recovery center"

# Anchor-guided interpolation
result = service.guided_interpolation(
    emb_a, emb_b,
    alpha=0.5,
    anchor_category="fanuc_errors"  # Validates semantic validity
)
```

**API Endpoints:**
- `GET /api/v1/search/mixed-precision/stats` - Service statistics
- `POST /api/v1/search/mixed-precision/index` - Index at all precision levels
- `POST /api/v1/search/mixed-precision/search` - Three-stage search
- `POST /api/v1/search/mixed-precision/mrl-search` - MRL hierarchical search
- `POST /api/v1/search/mixed-precision/anchor` - Create category anchor
- `POST /api/v1/search/mixed-precision/semantic-arithmetic` - Vector arithmetic

**Research Basis:**
- ResQ: Mixed-Precision Quantization with Low-Rank Residuals (arXiv 2024)
- R2Q: Residual Refinement Quantization (arXiv 2025)
- Binary and Scalar Embedding Quantization (HuggingFace 2024)
- 4bit-Quantization in Vector-Embedding for RAG (arXiv 2025)

**Module Version**: `agentic/__init__.py` → v0.16.0

### BGE-M3 Hybrid Retrieval (December 2025)

Implements hybrid retrieval combining dense and sparse methods for improved recall:

**Key Components:**
- **BGEM3HybridRetriever** (`agentic/bge_m3_hybrid.py`): Main hybrid retrieval class
- **BM25Index**: Efficient sparse lexical matching (no LLM required)
- **RRF Fusion**: Reciprocal Rank Fusion for combining scores
- **SQLite Persistence**: Large corpus support with lazy loading

**Retrieval Modes:**
| Mode | Description | Use Case |
|------|-------------|----------|
| `dense_only` | BGE-M3 semantic similarity | Semantic meaning |
| `sparse_only` | BM25 lexical matching | Exact terms |
| `hybrid` | Dense + Sparse with RRF | Best overall |

**Architecture:**
```
User Query
    |
    +---> [BGE-M3 Dense] ---> Top-100 semantic
    |
    +---> [BM25 Sparse] ----> Top-100 lexical
    |
    v
[Reciprocal Rank Fusion]
    |
    v
Top-K Combined Results
```

**RRF Formula:** `score(d) = Σ 1/(k + rank_i(d))` where k=60 (constant)

**API Endpoints:**
- `GET /api/v1/search/hybrid/stats` - Index statistics
- `POST /api/v1/search/hybrid/index` - Index documents
- `POST /api/v1/search/hybrid/search` - Hybrid search
- `POST /api/v1/search/hybrid/weights` - Update fusion weights
- `GET /api/v1/search/hybrid/bm25-stats` - BM25 statistics
- `DELETE /api/v1/search/hybrid/clear` - Clear index

**Usage:**
```python
from agentic import BGEM3HybridRetriever, RetrievalMode

retriever = BGEM3HybridRetriever()
await retriever.add_document("d1", "FANUC robot servo alarm...")

results = await retriever.search(
    query="robot alarm",
    top_k=10,
    mode=RetrievalMode.HYBRID
)
```

**Model Specs:**
- BGE-M3: 568M params, 1024 dimensions, 8K context, MIT license
- BM25: k1=1.5, b=0.75 (tuned for technical docs)

**Module Version**: `agentic/__init__.py` → v0.17.0

### HyDE Query Expansion (December 2025)

Implements Hypothetical Document Embeddings for improved retrieval by generating
hypothetical answers before searching.

**Key Insight:**
- Queries are short and abstract
- Documents are long and detailed
- Hypothetical documents bridge this semantic gap

**Key Components:**
- **HyDEExpander** (`agentic/hyde.py`): Main query expansion class
- **5 Document Types**: answer, passage, explanation, summary, technical
- **Embedding Fusion**: Mean, max, or weighted combination
- **Query Caching**: Avoids redundant LLM calls

**Pipeline:**
```
User Query
    |
    v
[LLM generates hypothetical answer]
    |
    v
[Embed hypothetical document]
    |
    v
[Search with hypothetical embedding]
    |
    v
Real Documents (better matching)
```

**HyDE Modes:**
| Mode | Description | Use Case |
|------|-------------|----------|
| `single` | One hypothetical | Fast, default |
| `multi` | Multiple hypotheticals | Better coverage |
| `contrastive` | Positive + negative | Disambiguation |

**API Endpoints:**
- `GET /api/v1/search/hyde/stats` - Expander statistics
- `POST /api/v1/search/hyde/expand` - Expand query
- `POST /api/v1/search/hyde/search` - Search with HyDE
- `DELETE /api/v1/search/hyde/cache` - Clear cache

**Usage:**
```python
from agentic import HyDEExpander, HyDEMode

expander = HyDEExpander()
result = await expander.expand(
    query="How to reset robot alarm?",
    mode=HyDEMode.SINGLE
)

# result.hypothetical_documents[0] contains generated text
# result.fused_embedding for search
```

**Research Basis:**
- Gao et al., "Precise Zero-Shot Dense Retrieval" (ACL 2023)
- arXiv:2212.10496
- 10-20% improvement in recall@10 on benchmarks

**Module Version**: `agentic/__init__.py` → v0.18.0

### RAGAS Evaluation Pipeline (December 2025)

Implements reference-free RAG evaluation using LLM-as-judge for quality assessment.

**Key Components:**
- **RAGASEvaluator** (`agentic/ragas.py`): Main evaluation class
- **Claim Extraction**: Extract verifiable facts from answers
- **Claim Verification**: Check support against context
- **Question Regeneration**: For answer relevancy scoring

**Metrics:**
| Metric | Range | Description |
|--------|-------|-------------|
| **Faithfulness** | 0-1 | Claims supported by context |
| **Answer Relevancy** | 0-1 | Answer addresses question |
| **Context Relevancy** | 0-1 | Retrieved context is relevant |
| **Context Precision** | 0-1 | Relevant context ranked higher |

**Evaluation Pipeline:**
```
(Question, Answer, Contexts)
    |
    +---> Extract claims from answer
    |         |
    |         v
    |     Verify each claim against contexts
    |         |
    |         v
    |     Faithfulness = supported/total
    |
    +---> Generate question from answer
    |         |
    |         v
    |     Compare with original (embedding similarity)
    |         |
    |         v
    |     Answer Relevancy = similarity
    |
    +---> Rate each context for relevance
    |         |
    |         v
    |     Context Relevancy = avg(scores)
    |
    +---> Position-weighted relevance
              |
              v
          Context Precision (AP@K)
```

**API Endpoints:**
- `GET /api/v1/search/ragas/stats` - Aggregate statistics
- `POST /api/v1/search/ragas/evaluate` - Single evaluation
- `POST /api/v1/search/ragas/batch-evaluate` - Batch evaluation
- `DELETE /api/v1/search/ragas/history` - Clear history
- `POST /api/v1/search/ragas/evaluate-search` - Evaluate search response

**Usage:**
```python
from agentic import RAGASEvaluator

evaluator = RAGASEvaluator()
result = await evaluator.evaluate(
    question="How to reset alarm?",
    answer="Press RESET button on teach pendant.",
    contexts=["To reset alarms, press RESET..."]
)

print(f"Faithfulness: {result.faithfulness:.2f}")
print(f"Answer Relevancy: {result.answer_relevancy:.2f}")
print(f"Overall: {result.overall_score:.2f}")
```

**Test Results:**
| Metric | Score |
|--------|-------|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.80 |
| Context Relevancy | 0.88 |
| Overall | 0.93 |

**Research Basis:**
- Es et al., "RAGAS: Automated Evaluation of RAG" (EMNLP 2024)
- arXiv:2309.15217

**Module Version**: `agentic/__init__.py` → v0.19.0