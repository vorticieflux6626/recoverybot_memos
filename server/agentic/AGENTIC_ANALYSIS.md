# Agentic Pipeline Analysis

> **Generated**: 2026-01-05 | **Version**: 0.88.0 | **Status**: Production

This document outlines the model selection strategy and agent activation sequence in the memOS agentic search pipeline.

---

## Table of Contents

1. [Model Allocation](#model-allocation)
2. [Pipeline Phase Sequence](#pipeline-phase-sequence)
3. [Agent Inventory](#agent-inventory)
4. [Preset Configurations](#preset-configurations)
5. [Feature Activation Matrix](#feature-activation-matrix)
6. [VRAM Budget Analysis](#vram-budget-analysis)

---

## Model Allocation

### Primary Pipeline Models

| Role | Model | Parameters | Context | VRAM | Purpose |
|------|-------|------------|---------|------|---------|
| **Analyzer** | qwen3:8b | 8B | 40K | ~5.7GB | Query analysis, URL evaluation, entity extraction |
| **Planner** | qwen3:8b | 8B | 40K | ~5.7GB | Search strategy, task decomposition |
| **Synthesizer** | ministral-3:3b | 3B | 32K | ~3.0GB | Final answer synthesis |
| **Thinking** | ministral-3:3b | 3B | 32K | ~3.0GB | Complex reasoning (when `use_thinking_model=true`) |
| **Evaluator** | qwen3:8b | 8B | 128K | ~5.7GB | CRAG evaluation, quality assessment |
| **Verifier** | qwen3:8b | 8B | 40K | ~5.7GB | Claim verification, fact checking |

### Auxiliary Models

| Role | Model | Parameters | Context | VRAM | Purpose |
|------|-------|------------|---------|------|---------|
| **HyDE Generator** | gemma3:4b | 4B | 128K | ~3.2GB | Hypothetical document generation |
| **Info Bottleneck** | gemma3:4b | 4B | 128K | ~3.2GB | Context compression (IB filtering) |
| **Info Gain** | gemma3:4b | 4B | 128K | ~3.2GB | Document relevance scoring |
| **KV Cache Warmer** | llama3.2:3b | 3B | 128K | ~2.0GB | System prompt caching |
| **Embedding** | qwen3-embedding:* | 0.6-8B | - | 0.5-6GB | Semantic embeddings |

### Vision Models (VL Scraper)

| Model | Parameters | VRAM | Quality | Context |
|-------|------------|------|---------|---------|
| qwen3-vl:2b-instruct-bf16 | 2B | 3.97GB | 3 | 32K |
| qwen3-vl:4b | 4B | 4.5GB | 3 | 32K |
| qwen3-vl:8b | 8B | 5.7GB | 4 | 32K |
| llava:7b-v1.6-mistral-q8_0 | 7B | 7.75GB | 4 | 8K |
| llama3.2-vision:11b-instruct-q8_0 | 11B | 11.4GB | 4 | 131K |
| qwen3-vl:8b-instruct-bf16 | 8B | 16.3GB | 5 | 32K |

---

## Pipeline Phase Sequence

### Phase Execution Order

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC SEARCH PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 0: Initialization                                                     │
│  ├── 0.1 TTL Pinning (cache protection)                                     │
│  └── 0.5 KV Cache Warming                                                   │
│                                                                              │
│  PHASE 1: Query Understanding                                               │
│  ├── 1.0 Query Analysis ──────────────────────────────► [Analyzer]          │
│  ├── 1.4 DyLAN Complexity Classification                                    │
│  ├── 1.5 Entity Extraction ───────────────────────────► [EntityTracker]     │
│  ├── 1.6 Dynamic Planning (AIME-style) ───────────────► [DynamicPlanner]    │
│  └── 1.7 Reasoning DAG Initialization                                       │
│                                                                              │
│  PHASE 2: Query Expansion                                                   │
│  ├── 2.1 HyDE Expansion ──────────────────────────────► [HyDEGenerator]     │
│  ├── 2.2 Query Tree Expansion                                               │
│  └── 2.3 Meta-Buffer Template Retrieval                                     │
│                                                                              │
│  PHASE 3: Search Execution (ReAct Loop)                                     │
│  ├── 3.0 Web Search ──────────────────────────────────► [Searcher]          │
│  ├── 3.2 Technical Docs Search (PDF API) ─────────────► [DocGraphService]   │
│  ├── 3.3 HSEA Context Retrieval                                             │
│  ├── 3.5 CRAG Evaluation ─────────────────────────────► [RetrievalEvaluator]│
│  └── 3.7 Hybrid Reranking (BGE-M3) ───────────────────► [CrossEncoder]      │
│                                                                              │
│  PHASE 4: Content Processing                                                │
│  ├── 4.0 URL Scraping ────────────────────────────────► [Scraper]           │
│  ├── 4.3 VL Screenshot Extraction ────────────────────► [VLScraper]         │
│  └── 4.5 Context Curation (DIG filtering) ────────────► [InfoBottleneck]    │
│                                                                              │
│  PHASE 5: Verification                                                      │
│  ├── 5.0 Claim Verification ──────────────────────────► [Verifier]          │
│  └── 5.9 Information Bottleneck Filtering                                   │
│                                                                              │
│  PHASE 6: Synthesis                                                         │
│  ├── 6.0 Answer Synthesis ────────────────────────────► [Synthesizer]       │
│  └── 6.5 Cross-Domain Validation ─────────────────────► [CrossDomainValidator]│
│                                                                              │
│  PHASE 7: Quality Assessment                                                │
│  ├── 7.0 Self-RAG Reflection ─────────────────────────► [SelfReflection]    │
│  ├── 7.2 RAGAS Evaluation ────────────────────────────► [RAGASEvaluator]    │
│  └── 7.5 Entropy-Based Halting Check                                        │
│                                                                              │
│  PHASE 8: Adaptive Refinement (if confidence < threshold)                   │
│  └── 8.0 Refinement Loop ─────────────────────────────► [AdaptiveRefinement]│
│                                                                              │
│  PHASE 9: Learning & Feedback                                               │
│  ├── 9.0 Experience Distillation ─────────────────────► [ExperienceDistiller]│
│  ├── 9.5 Contrastive Retriever Recording                                    │
│  └── 9.9 Classifier Feedback Update                                         │
│                                                                              │
│  PHASE 12: Constraint Verification Gate                                     │
│  └── 12.0 Output Validation ──────────────────────────► [ConstraintVerifier]│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Details

#### Phase 1: Query Understanding

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 1.0 | QueryAnalyzer | qwen3:8b | Always | QueryAnalysis (intent, entities, keywords) |
| 1.4 | DyLANAgentNetwork | - | `enable_dylan_agent_skipping` | Complexity score, agent skip decisions |
| 1.5 | EntityTracker | qwen3:8b | `enable_entity_tracking` | Extracted entities (GSW format) |
| 1.6 | DynamicPlanner | qwen3:8b | `enable_dynamic_planning` | Strategic + tactical plans |
| 1.7 | ReasoningDAG | qwen3:8b | `enable_reasoning_dag` | Initialized reasoning graph |

#### Phase 2: Query Expansion

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 2.1 | HyDEGenerator | gemma3:4b | `enable_hyde` | Hypothetical documents |
| 2.2 | QueryTreeDecoder | qwen3:8b | `enable_query_tree` | Expanded query tree |
| 2.3 | MetaBuffer | - | `enable_meta_buffer` | Retrieved reasoning templates |

#### Phase 3: Search Execution

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 3.0 | Searcher | - | Always | Web search results |
| 3.2 | DocumentGraphService | - | `enable_technical_docs` | PDF knowledge graph context |
| 3.3 | HSEAContext | - | `enable_hsea_context` | Three-stratum FANUC knowledge |
| 3.5 | RetrievalEvaluator | qwen3:8b | `enable_crag_evaluation` | CRAG action (CORRECT/INCORRECT/AMBIGUOUS) |
| 3.7 | CrossEncoder | MiniLM | `enable_cross_encoder` | Reranked results |

#### Phase 4: Content Processing

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 4.0 | Scraper | - | Always | Scraped page content |
| 4.3 | VLScraper | qwen3-vl:* | JS-rendered pages | Structured extraction |
| 4.5 | InfoBottleneck | gemma3:4b | `enable_context_curation` | Filtered/deduplicated context |

#### Phase 5: Verification

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 5.0 | Verifier | qwen3:8b | `enable_verification` (default: true) | Verified claims |
| 5.9 | InfoBottleneck | gemma3:4b | `enable_information_bottleneck` | IB-compressed context |

#### Phase 6: Synthesis

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 6.0 | Synthesizer | ministral-3:3b | Always | Synthesized answer |
| 6.0* | Synthesizer | ministral-3:3b | `use_thinking_model=true` | Chain-of-thought answer |
| 6.5 | CrossDomainValidator | - | `enable_cross_domain_validation` | Validated claims, hedged synthesis |

#### Phase 7: Quality Assessment

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 7.0 | SelfReflection | qwen3:8b | `enable_self_reflection` | ISREL/ISSUP/ISUSE scores |
| 7.2 | RAGASEvaluator | qwen3:8b | `enable_ragas` (if Self-RAG skipped) | RAGAS metrics |
| 7.5 | EntropyMonitor | - | `enable_entropy_halting` | Halt/continue decision |

#### Phase 8: Adaptive Refinement

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 8.0 | AdaptiveRefinement | qwen3:8b | `enable_adaptive_refinement` AND confidence < threshold | Refined search + synthesis |

#### Phase 9: Learning

| Step | Agent | Model | Condition | Output |
|------|-------|-------|-----------|--------|
| 9.0 | ExperienceDistiller | qwen3:8b | `enable_experience_distillation` AND success | Distilled template |
| 9.5 | ContrastiveRecorder | - | `enable_contrastive_learning` | Training signal |
| 9.9 | ClassifierFeedback | - | `enable_classifier_feedback` | Updated classifier weights |

---

## Agent Inventory

### Core Agents (Always Initialized)

| Agent | File | Model | Purpose |
|-------|------|-------|---------|
| `QueryClassifier` | `query_classifier.py` | qwen3:8b | Classify query type, recommend pipeline |
| `QueryAnalyzer` | `analyzer.py` | qwen3:8b | Deep query analysis |
| `Searcher` | `searcher.py` | - | Execute web searches |
| `Scraper` | `scraper.py` | - | Extract page content |
| `Synthesizer` | `synthesizer.py` | ministral-3:3b | Generate final answers |
| `Verifier` | `verifier.py` | qwen3:8b | Verify claims against sources |
| `SelfReflection` | `self_reflection.py` | qwen3:8b | Self-RAG quality assessment |
| `RetrievalEvaluator` | `retrieval_evaluator.py` | qwen3:8b | CRAG retrieval evaluation |

### Conditional Agents (Feature-Gated)

| Agent | File | Model | Feature Flag |
|-------|------|-------|--------------|
| `EntityTracker` | `entity_tracker.py` | qwen3:8b | `enable_entity_tracking` |
| `DynamicPlanner` | `dynamic_planner.py` | qwen3:8b | `enable_dynamic_planning` |
| `HyDEGenerator` | `hyde.py` | gemma3:4b | `enable_hyde` |
| `ReasoningDAG` | `reasoning_dag.py` | qwen3:8b | `enable_reasoning_dag` |
| `ReasoningComposer` | `reasoning_composer.py` | qwen3:8b | `enable_reasoning_composer` |
| `CrossEncoder` | `cross_encoder.py` | MiniLM | `enable_cross_encoder` |
| `InfoBottleneck` | `information_bottleneck.py` | gemma3:4b | `enable_information_bottleneck` |
| `VLScraper` | `vl_scraper.py` | qwen3-vl:* | `enable_vision_analysis` |
| `CrossDomainValidator` | `cross_domain_validator.py` | - | `enable_cross_domain_validation` |
| `EntityGroundingAgent` | `entity_grounding.py` | - | `enable_entity_grounding` |
| `AdaptiveRefinement` | `adaptive_refinement.py` | qwen3:8b | `enable_adaptive_refinement` |
| `ExperienceDistiller` | `experience_distiller.py` | qwen3:8b | `enable_experience_distillation` |
| `RAGASEvaluator` | `ragas_evaluator.py` | qwen3:8b | `enable_ragas` |
| `EntropyMonitor` | `entropy_monitor.py` | qwen3:8b | `enable_entropy_halting` |
| `FLARERetriever` | `flare_retriever.py` | qwen3:8b | `enable_flare_retrieval` |
| `MetaBuffer` | `meta_buffer.py` | - | `enable_meta_buffer` |
| `ActorFactory` | `actor_factory.py` | qwen3:8b | `enable_actor_factory` |
| `DyLANAgentNetwork` | `dylan_agents.py` | - | `enable_dylan_agent_skipping` |

### Support Components

| Component | File | Purpose |
|-----------|------|---------|
| `AgenticScratchpad` | `scratchpad.py` | Shared working memory (blackboard pattern) |
| `KVCacheService` | `kv_cache_service.py` | System prompt caching |
| `DocumentGraphService` | `document_graph_service.py` | PDF knowledge graph bridge |
| `SemanticCache` | `semantic_cache.py` | Query result caching |
| `AgentStepGraph` | `agent_step_graph.py` | KVFlow DAG tracking |

---

## Preset Configurations

### Feature Count by Preset

| Preset | Features Enabled | Primary Use Case |
|--------|------------------|------------------|
| **MINIMAL** | 8 | Fast, simple queries |
| **BALANCED** | 18 | Default for most queries |
| **ENHANCED** | 28 | Complex research |
| **RESEARCH** | 35 | Academic/thorough investigation |
| **FULL** | 42+ | Maximum capability |

### MINIMAL Preset (8 features)

```
Core pipeline only:
- Query Analysis → Search → Scrape → Verify → Synthesize → Reflect
- No caching, no adaptive refinement, no advanced retrieval
```

### BALANCED Preset (18 features)

```
Core + Light Enhancements:
+ enable_hybrid_reranking (BGE-M3 dense+sparse)
+ enable_domain_corpus
+ enable_hsea_context (FANUC knowledge)
```

### ENHANCED Preset (28 features)

```
Quality-Focused:
+ enable_hyde (query expansion)
+ enable_cross_encoder (neural reranking)
+ enable_ragas (evaluation)
+ enable_context_curation (DIG filtering)
+ enable_entity_tracking
+ enable_technical_docs (PDF API)
+ enable_cross_domain_validation (Phase 48)
+ enable_entity_grounding
```

### RESEARCH Preset (35 features)

```
Thorough Investigation:
+ enable_entropy_halting (confidence-calibrated)
+ enable_flare_retrieval (proactive retrieval)
+ enable_query_tree (parallel exploration)
+ enable_semantic_memory (A-MEM/RAISE)
+ enable_meta_buffer (template reuse)
+ enable_reasoning_dag (GoT structure)
+ enable_dynamic_planning (AIME-style)
+ enable_kv_cache_service
```

### FULL Preset (42+ features)

```
Everything Enabled:
+ enable_self_consistency (expensive)
+ enable_actor_factory (dynamic agents)
+ enable_multi_agent
+ enable_information_bottleneck (aggressive)
+ enable_contrastive_learning
+ enable_constraint_verification
+ enable_llm_debug
```

---

## Feature Activation Matrix

| Feature | MIN | BAL | ENH | RES | FULL |
|---------|-----|-----|-----|-----|------|
| **Core Pipeline** |
| Query Analysis | ✅ | ✅ | ✅ | ✅ | ✅ |
| Search Execution | ✅ | ✅ | ✅ | ✅ | ✅ |
| Content Scraping | ✅ | ✅ | ✅ | ✅ | ✅ |
| Verification | ✅ | ✅ | ✅ | ✅ | ✅ |
| Synthesis | ✅ | ✅ | ✅ | ✅ | ✅ |
| Self-Reflection | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Retrieval Enhancement** |
| HyDE Expansion | ❌ | ❌ | ✅ | ✅ | ✅ |
| Hybrid Reranking | ❌ | ✅ | ✅ | ✅ | ✅ |
| Cross-Encoder | ❌ | ❌ | ✅ | ✅ | ✅ |
| CRAG Evaluation | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Domain Knowledge** |
| Domain Corpus | ❌ | ✅ | ✅ | ✅ | ✅ |
| HSEA Context | ❌ | ✅ | ✅ | ✅ | ✅ |
| Technical Docs (PDF) | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Reasoning** |
| Entity Tracking | ❌ | ❌ | ✅ | ✅ | ✅ |
| Reasoning DAG | ❌ | ❌ | ❌ | ✅ | ✅ |
| Dynamic Planning | ❌ | ❌ | ❌ | ✅ | ✅ |
| Meta-Buffer | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Quality Control** |
| RAGAS Evaluation | ❌ | ❌ | ✅ | ✅ | ✅ |
| Entropy Halting | ❌ | ❌ | ❌ | ✅ | ✅ |
| Cross-Domain Validation | ❌ | ❌ | ✅ | ✅ | ✅ |
| Entity Grounding | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Advanced** |
| FLARE Retrieval | ❌ | ❌ | ❌ | ✅ | ✅ |
| Query Tree | ❌ | ❌ | ❌ | ✅ | ✅ |
| Semantic Memory | ❌ | ❌ | ❌ | ✅ | ✅ |
| Actor Factory | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-Agent | ❌ | ❌ | ❌ | ❌ | ✅ |
| Self-Consistency | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## VRAM Budget Analysis

### Minimum VRAM Requirements by Preset

| Preset | Concurrent Models | Peak VRAM | Notes |
|--------|-------------------|-----------|-------|
| **MINIMAL** | 2 | ~9GB | qwen3:8b + ministral-3:3b |
| **BALANCED** | 2 | ~9GB | Same as minimal |
| **ENHANCED** | 3 | ~12GB | + gemma3:4b for HyDE/IB |
| **RESEARCH** | 3 | ~12GB | Same as enhanced |
| **FULL** | 4+ | ~16GB | + vision models, multiple simultaneous |

### Model Loading Strategy

```
1. Always Loaded (Ollama keep_alive=30m):
   - qwen3:8b (primary fast model)
   - ministral-3:3b (synthesizer)

2. Loaded On-Demand:
   - gemma3:4b (HyDE, IB filtering)
   - qwen3-vl:* (vision analysis)
   - llama3.2:3b (KV cache warming)

3. Embedding Models (Separate):
   - qwen3-embedding:0.6b-fp16 (fast)
   - nomic-embed-text:latest (fallback)
```

### Concurrent Model Constraints

```
With 24GB VRAM:
├── qwen3:8b ────────────── 5.7GB
├── ministral-3:3b ──────── 3.0GB
├── gemma3:4b ───────────── 3.2GB
├── qwen3-embedding:0.6b ── 0.5GB
└── Available ───────────── 11.6GB (for vision or additional models)

With 12GB VRAM:
├── qwen3:8b ────────────── 5.7GB
├── ministral-3:3b ──────── 3.0GB
└── Available ───────────── 3.3GB (limited vision model options)
```

---

## Appendix: Agent Communication Flow

### Scratchpad (Blackboard) Pattern

All agents communicate through the `AgenticScratchpad`:

```python
@dataclass
class AgenticScratchpad:
    # Query understanding
    original_query: str
    expanded_queries: List[str]
    entities: List[Entity]

    # Search state
    search_results: List[SearchResult]
    scraped_content: List[ScrapedContent]

    # Reasoning state
    reasoning_trace: List[ReasoningStep]
    verification_results: Dict[str, VerificationResult]

    # Quality metrics
    confidence_score: float
    source_diversity: float
    content_depth: float

    # Agent notes (RAISE structure)
    agent_notes: List[AgentNote]
    mission: str
    sub_goals: List[str]
```

### Event Flow (SSE)

```
Client ──► POST /search/chat-gateway
                │
                ▼
        ┌───────────────┐
        │  Orchestrator │
        └───────┬───────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
[analyzing] [planning] [searching] ──► SSE Events
    │           │           │
    ▼           ▼           ▼
[scraping] [verifying] [synthesizing]
    │           │           │
    ▼           ▼           ▼
[reflecting] ─────────────► [complete]
                                │
                                ▼
                          Final Response
```

---

*Last Updated: 2026-01-05*
