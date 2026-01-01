"""
Agentic Search Module for memOS

Implements multi-agent search with ReAct pattern:
- Orchestrator: Routes queries to appropriate pipelines
- Planner: Decomposes queries into search terms
- Searcher: Executes web searches (Brave API, DuckDuckGo fallback)
- Verifier: Cross-checks facts and validates claims
- Synthesizer: Combines results into coherent answers

Performance optimizations:
- Phase 1-2: TTL-based KV cache pinning, prompt registry, artifacts, metrics
- Phase 4: Three-tier memory architecture (MemOS MemCube pattern)

This module is isolated from core memOS services and can be
enabled/disabled independently.

=============================================================================
ORCHESTRATOR CONSOLIDATION (December 2025)
=============================================================================
UniversalOrchestrator is the SINGLE SOURCE OF TRUTH for all orchestrator needs.

DEPRECATED orchestrators (maintained for backward compatibility):
- AgenticOrchestrator -> Use UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)
- EnhancedAgenticOrchestrator -> Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)
- DynamicOrchestrator -> Use UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH)
- GraphEnhancedOrchestrator -> Use UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH)
- UnifiedOrchestrator -> Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)

Presets:
- minimal: 8 features - fast, simple queries
- balanced: 18 features - default for most queries
- enhanced: 28 features - complex research
- research: 39 features - academic/thorough (includes dynamic planning + graph cache)
- full: 42+ features - maximum capability (adds multi-agent)
=============================================================================
"""

# =============================================================================
# BACKWARD COMPATIBILITY SHIMS (Legacy orchestrators archived 2025-12-29)
# =============================================================================
# The following classes are defined as shim functions that redirect to
# UniversalOrchestrator with appropriate presets. They emit deprecation warnings.
#
# See: agentic/archive/legacy_orchestrators/README.md
# =============================================================================

import warnings

def AgenticOrchestrator(*args, **kwargs):
    """
    DEPRECATED: Use UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)

    This shim maintains backward compatibility with code importing AgenticOrchestrator.
    """
    warnings.warn(
        "AgenticOrchestrator is deprecated and archived. "
        "Use UniversalOrchestrator(preset=OrchestratorPreset.BALANCED) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from .orchestrator_universal import UniversalOrchestrator, OrchestratorPreset
    return UniversalOrchestrator(preset=OrchestratorPreset.BALANCED, *args, **kwargs)
from .models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    VerificationResult,
    AgentAction,
    SearchState
)
from .events import (
    EventType,
    SearchEvent,
    EventEmitter,
    EventManager,
    get_event_manager
)
from .ttl_cache_manager import (
    TTLCacheManager,
    ToolType,
    ToolCallContext,
    get_ttl_cache_manager
)
from .prompts import (
    build_prompt,
    get_system_prompt,
    get_template,
    CHAIN_OF_DRAFT_INSTRUCTION
)
from .artifacts import (
    ArtifactStore,
    ArtifactType,
    get_artifact_store,
    store_search_results,
    store_scraped_content,
    store_synthesis
)
from .metrics import (
    PerformanceMetrics,
    QueryMetrics,
    PhaseTimer,
    get_performance_metrics
)
from .search_metrics import (
    SearchMetrics,
    get_search_metrics
)
from .kv_cache_service import (
    KVCacheService,
    CacheBackend,
    CacheState,
    get_kv_cache_service,
    warm_system_prompts
)
from .memory_tiers import (
    MemoryTierManager,
    MemoryTier,
    ContentType,
    PlaintextStorage,
    get_memory_tier_manager,
    initialize_memory_tiers
)
from .dynamic_planner import (
    DynamicPlanner,
    TaskNode,
    TaskStatus,
    ActionType,
    TacticalAction,
    ExecutionResult,
    PlannerOutput
)
from .progress_tools import (
    ProgressReporter,
    ProgressAggregator,
    ProgressStatus,
    ProgressUpdate,
    ProgressTool,
    PROGRESS_TOOL_PROMPT
)
# DynamicOrchestrator shim (archived 2025-12-29)
def DynamicOrchestrator(*args, **kwargs):
    """DEPRECATED: Use UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH)"""
    warnings.warn(
        "DynamicOrchestrator is deprecated and archived. "
        "Use UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from .orchestrator_universal import UniversalOrchestrator, OrchestratorPreset
    return UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH, *args, **kwargs)

def create_dynamic_orchestrator(*args, **kwargs):
    """DEPRECATED: Use UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH)"""
    return DynamicOrchestrator(*args, **kwargs)
from .entity_tracker import (
    EntityTracker,
    EntityState,
    EntityType,
    RoleType,
    RelationType,
    RoleAssignment,
    EntityEvent,
    EntityRelation,
    VerbFrame,
    create_entity_tracker
)
from .reasoning_dag import (
    ReasoningDAG,
    ReasoningNode,
    NodeType,
    NodeStatus,
    VerificationResult,
    create_reasoning_dag
)
from .thought_library import (
    ThoughtLibrary,
    ThoughtTemplate,
    TemplateCategory,
    InstantiatedThought,
    create_thought_library,
    get_thought_library
)
from .actor_factory import (
    ActorFactory,
    DynamicActor,
    ActorPersona,
    ToolBundle,
    Tool,
    ToolCategory,
    ModelCapability,
    create_actor_factory,
    get_actor_factory
)
from .self_reflection import (
    SelfReflectionAgent,
    ReflectionResult,
    ReflectionToken,
    SupportLevel,
    TemporalFact,
    TemporalConflict,
    create_self_reflection_agent,
    get_self_reflection_agent
)
from .retrieval_evaluator import (
    RetrievalEvaluator,
    RetrievalEvaluation,
    RetrievalQuality,
    CorrectiveAction,
    DocumentScore,
    create_retrieval_evaluator,
    get_retrieval_evaluator
)
from .experience_distiller import (
    ExperienceDistiller,
    SearchExperience,
    DistillationResult,
    create_experience_distiller,
    get_experience_distiller
)
from .classifier_feedback import (
    ClassifierFeedback,
    ClassificationOutcome,
    AdaptiveHint,
    OutcomeQuality,
    create_classifier_feedback,
    get_classifier_feedback
)
from .domain_corpus import (
    DomainSchema,
    DomainEntityDef,
    DomainRelationDef,
    DomainEntity,
    DomainRelation,
    DomainCorpus,
    CorpusBuilder,
    CorpusRetriever,
    DomainCorpusManager,
    TroubleshootingEntityType,
    TroubleshootingRelationType,
    create_fanuc_schema,
    create_raspberry_pi_schema,
    get_corpus_manager,
    initialize_default_corpuses
)
from .rjg_corpus_scraper import (
    RJGCorpusScraper,
    ScrapeResult,
    create_rjg_corpus_schema,
    get_rjg_scraper,
    RJG_SEED_URLS,
    RJG_ARTICLE_URLS
)
from .plc_corpus_scraper import (
    PLCCorpusScraper,
    ScrapeResult as PLCScrapeResult,
    create_plc_corpus_schema,
    get_plc_scraper,
    PLC_SEED_URLS,
    PLC_ARTICLE_URLS
)
from .acronym_dictionary import (
    AcronymInfo,
    INDUSTRIAL_ACRONYMS,
    FANUC_ERROR_CODES,
    expand_acronyms,
    expand_acronym,
    expand_error_code_prefixes,
    get_acronym_info,
    get_related_terms,
    get_category_acronyms,
    get_dictionary_stats
)
from .query_classifier import (
    QueryClassifier,
    QueryClassification,
    QueryCategory,
    QueryComplexity,
    RecommendedPipeline,
    classify_query,
    get_query_classifier
)
from .embedding_aggregator import (
    EmbeddingAggregator,
    DomainExpert,
    AggregatedEmbedding,
    SubManifoldResult,
    RetrievalResult,
    get_embedding_aggregator,
    retrieve_with_entities
)
from .entity_enhanced_retrieval import (
    EntityEnhancedRetriever,
    EnhancedRetrievalResult,
    entity_enhanced_retrieve,
    get_entity_enhanced_retriever
)
from .mixed_precision_embeddings import (
    MixedPrecisionEmbeddingService,
    PrecisionLevel,
    QuantizedEmbedding,
    SearchResult as MixedPrecisionSearchResult,
    RetrievalStats,
    EmbeddingModelSpec,
    QWEN3_EMBEDDING_MODELS,
    MODEL_TIERS,
    get_model_spec,
    get_model_dimension,
    get_mixed_precision_service
)
from .bge_m3_hybrid import (
    BGEM3HybridRetriever,
    HybridDocument,
    HybridSearchResult,
    HybridRetrievalStats,
    RetrievalMode,
    BM25Index,
    get_hybrid_retriever,
    create_hybrid_retriever,
    # G.1.1: ColBERT support
    BGEM3Embeddings,
    get_bge_m3_model,
    _FLAG_EMBEDDING_AVAILABLE as COLBERT_AVAILABLE
)
# G.1.2: Redis EmbeddingsCache with 3-tier strategy
from .redis_embeddings_cache import (
    RedisEmbeddingsCache,
    CacheTier,
    TierConfig,
    CachedEmbedding,
    CacheStats,
    DEFAULT_TIER_CONFIGS,
    get_redis_embeddings_cache,
    get_redis_embeddings_cache_async,
    REDIS_AVAILABLE
)
# G.1.6: Cross-encoder reranker (bge-reranker-v2-m3)
from .cross_encoder_reranker import (
    CrossEncoderReranker,
    RerankedResult,
    RerankerStats,
    get_cross_encoder_reranker,
    get_cross_encoder_reranker_async,
    _FLAG_RERANKER_AVAILABLE as RERANKER_AVAILABLE
)
from .hyde import (
    HyDEExpander,
    HyDEConfig,
    HyDEMode,
    DocumentType as HyDEDocumentType,
    HyDEResult,
    get_hyde_expander,
    create_hyde_expander
)
from .ragas import (
    RAGASEvaluator,
    RAGASConfig,
    RAGASResult,
    EvaluationMetric,
    ClaimVerification,
    get_ragas_evaluator,
    create_ragas_evaluator
)
from .adaptive_refinement import (
    AdaptiveRefinementEngine,
    RefinementDecision,
    AnswerGrade,
    GapAnalysis,
    AnswerAssessment,
    RefinementResult,
    get_adaptive_refinement_engine,
    create_adaptive_refinement_engine
)
from . import events
from .events import (
    AgentGraphState,
    get_graph_state,
    reset_graph_state,
    graph_node_entered,
    graph_node_completed,
    graph_state_update,
    graph_edge_traversed,
    graph_branch_created,
    graph_paths_merged
)
from .sufficient_context import (
    SufficientContextClassifier,
    PositionalOptimizer,
    DynamicContextAllocator,
    SufficiencyResult,
    ContextSufficiency,
    PositionalAnalysis,
    get_sufficient_context_classifier,
    get_positional_optimizer,
    get_dynamic_allocator
)
# UnifiedOrchestrator shims (archived 2025-12-29)
def UnifiedOrchestrator(*args, **kwargs):
    """DEPRECATED: Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)"""
    warnings.warn(
        "UnifiedOrchestrator is deprecated and archived. "
        "Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from .orchestrator_universal import UniversalOrchestrator, OrchestratorPreset
    return UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED, *args, **kwargs)

def get_unified_orchestrator(*args, **kwargs):
    """DEPRECATED: Use get_universal_orchestrator('enhanced')"""
    return UnifiedOrchestrator(*args, **kwargs)

def create_unified_orchestrator(*args, **kwargs):
    """DEPRECATED: Use UniversalOrchestrator(preset=ENHANCED)"""
    return UnifiedOrchestrator(*args, **kwargs)
from .base_pipeline import (
    BaseSearchPipeline
)
from .orchestrator_universal import (
    UniversalOrchestrator,
    FeatureConfig,
    OrchestratorPreset,
    PRESET_CONFIGS
)
# Phase 1 Context Curation (December 2025)
from .information_gain import (
    DocumentInformationGain,
    DIGScore,
    DIGCategory,
    DIGBatchResult,
    get_dig_scorer
)
from .redundancy_detector import (
    RedundancyDetector,
    DocumentCluster,
    DeduplicationResult,
    SelectionMethod,
    get_redundancy_detector
)
from .context_curator import (
    ContextCurator,
    CuratedContext,
    CurationConfig,
    CurationPreset,
    CoverageAnalysis,
    CurationTrace,
    get_context_curator,
    curate_context
)
# Phase 2 Confidence-Calibrated Halting (December 2025)
from .entropy_monitor import (
    EntropyMonitor,
    EntropyResult,
    EntropyState,
    HaltDecision,
    get_entropy_monitor
)
from .self_consistency import (
    SelfConsistencyChecker,
    ConvergenceResult,
    ConvergenceStatus,
    SynthesisAttempt,
    AnswerCluster,
    get_consistency_checker
)
from .iteration_bandit import (
    IterationBandit,
    BanditDecision,
    RefinementAction,
    RefinementState,
    ActionOutcome,
    ArmStats,
    get_iteration_bandit
)
# Phase 3 Enhanced Query Generation (December 2025)
from .flare_retriever import (
    FLARERetriever,
    FLAREResult,
    RetrievalPoint,
    RetrievalTrigger,
    get_flare_retriever
)
from .query_tree import (
    QueryTreeDecoder,
    QueryTree,
    QueryNode,
    QueryOperation,
    TreeDecodingResult,
    NodeStatus as QueryNodeStatus,
    get_query_tree_decoder
)
# Phase 4 Scratchpad Enhancement (December 2025)
from .semantic_memory import (
    SemanticMemoryNetwork,
    Memory,
    MemoryConnection,
    MemoryType,
    ConnectionType,
    TraversalResult,
    get_semantic_memory
)
from .raise_scratchpad import (
    RAISEScratchpad,
    Observation,
    ReasoningStep,
    Example,
    TrajectoryStep,
    QualitySignal,
    ObservationType,
    ReasoningType,
    UncertaintyIndicator,
    create_raise_scratchpad
)

# Phase 5 Template Reuse Optimization (December 2025)
from .meta_buffer import (
    MetaBuffer,
    DistilledTemplate,
    TemplateType,
    InstantiatedTemplate,
    get_meta_buffer
)
from .reasoning_composer import (
    ReasoningComposer,
    ReasoningModule,
    ComposedStrategy,
    ModuleDefinition,
    get_reasoning_composer
)

# Part F: Evaluation & Metrics (December 2025)
from .benchmark import (
    BenchmarkQuery,
    BenchmarkResult,
    BenchmarkReport,
    TechnicalAccuracyScorer,
    BenchmarkRunner,
    QueryDifficulty,
    QueryCategory,
    FANUC_BENCHMARK,
    run_benchmark,
    get_benchmark_stats,
    filter_benchmark
)

# G.1.3: OpenTelemetry Tracing (December 2025)
from .tracing import (
    configure_tracing,
    get_tracer,
    get_agentic_tracer,
    trace_operation,
    trace_span,
    is_tracing_enabled,
    get_tracing_status,
    shutdown_tracing,
    TracingConfig,
    AgenticTracer,
    OTEL_AVAILABLE,
    OTLP_AVAILABLE
)

# G.1.5: DeepEval CI Pipeline Integration (December 2025)
from .deepeval_integration import (
    DeepEvalRAGEvaluator,
    EvaluationMetric as DeepEvalMetric,
    EvaluationResult as DeepEvalResult,
    BenchmarkEvaluationResult,
    get_evaluator as get_deepeval_evaluator,
    evaluate_rag_response,
    run_benchmark_evaluation,
    get_evaluation_summary,
    create_deepeval_test_cases,
    is_deepeval_available,
    DEEPEVAL_AVAILABLE
)

# G.2.1-G.2.2: Cascade Retriever (Binary Oversampling + MRL Cascade)
from .cascade_retriever import (
    CascadeRetriever,
    CascadeConfig,
    CascadeStage,
    CascadeResult,
    CascadeStats,
    get_cascade_retriever,
    get_cascade_retriever_async
)

# G.2.3: Query Intent Classifier for Fusion Weight Adaptation
from .fusion_weight_adapter import (
    FusionWeightAdapter,
    FusionWeights,
    QueryIntent,
    IntentClassification,
    get_fusion_weight_adapter,
    get_adaptive_weights,
    classify_for_fusion
)

# G.2.4: Qdrant On-Disk Storage for VRAM Management
from .qdrant_storage import (
    QdrantStorage,
    StorageConfig,
    QuantizationType,
    DistanceMetric,
    SearchResult as QdrantSearchResult,
    CollectionInfo,
    get_qdrant_storage,
    get_qdrant_storage_async,
    is_qdrant_available,
    QDRANT_AVAILABLE,
    VRAM_EFFICIENT_CONFIG,
    MAXIMUM_COMPRESSION_CONFIG,
    BALANCED_CONFIG
)

# G.2.5: Adaptive Top-K using CAR Algorithm
from .adaptive_topk import (
    AdaptiveTopK,
    AdaptiveTopKConfig,
    QueryComplexity as AdaptiveQueryComplexity,
    StoppingReason,
    ComplexityFeatures,
    ScoreDistribution,
    AdaptiveKResult,
    EarlyStopResult,
    get_adaptive_topk,
    compute_adaptive_k,
    apply_early_stopping,
    PRECISION_CONFIG,
    BALANCED_ADAPTIVE_CONFIG,
    RECALL_CONFIG
)

# G.3.1-G.3.4: NanoGraphRAG with PPR + Leiden Communities (December 2025)
from .nano_graphrag import (
    NanoGraphRAG,
    GraphRAGConfig,
    GraphRAGIntegration,
    QueryMode as GraphQueryMode,
    EntityType as GraphEntityType,
    Entity as GraphEntity,
    Relationship as GraphRelationship,
    Community as GraphCommunity,
    QueryResult as GraphQueryResult,
    get_nano_graphrag,
    get_graphrag_integration,
    initialize_graphrag,
    NETWORKX_AVAILABLE,
    LEIDEN_AVAILABLE
)

# G.3.3: GLiNER + Regex Hybrid Entity Extraction (December 2025)
from .gliner_extractor import (
    GLiNERHybridExtractor,
    TroubleshootingExtractor,
    ExtractedEntity,
    ExtractionResult,
    EntitySource,
    EntityCategory,
    PatternRegistry,
    get_gliner_extractor,
    get_troubleshooting_extractor,
    is_gliner_available,
    GLINER_AVAILABLE as GLINER_MODEL_AVAILABLE
)

# G.3.5: Late Chunking for Context-Aware Embedding (December 2025)
from .late_chunking import (
    LateChunker,
    ChunkingConfig,
    ChunkingStrategy,
    Chunk as LateChunk,
    TokenSpan,
    LateChunkingResult,
    get_late_chunker,
    create_late_chunker,
    compare_chunking_methods
)

# G.4.1: Circuit Breakers for Production Reliability (December 2025)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    CircuitMetrics,
    CircuitBreakerError,
    CircuitOpenError,
    AllCircuitsOpenError,
    get_circuit_breaker_registry,
    with_circuit_breaker,
    create_llm_circuit_breaker,
    create_search_circuit_breaker,
    create_embedding_circuit_breaker,
    create_scraping_circuit_breaker,
    execute_with_fallback,
    get_circuit_health,
    get_circuit_status,
    reset_circuit,
    LLM_CIRCUIT_CONFIG,
    SEARCH_CIRCUIT_CONFIG,
    EMBEDDING_CIRCUIT_CONFIG,
    SCRAPING_CIRCUIT_CONFIG,
)

# G.4.2: Shadow Mode for Embedding Model Testing (December 2025)
from .shadow_embeddings import (
    ShadowEmbeddingTester,
    ShadowEmbeddingRegistry,
    ShadowConfig,
    ShadowMode,
    ShadowMetrics,
    ComparisonResult,
    EmbeddingProvider,
    OllamaEmbeddingAdapter,
    get_shadow_registry,
    create_ollama_shadow_tester,
)

# G.4.3: Feature Flags for Gradual Rollout (December 2025)
from .feature_flags import (
    FeatureFlagManager,
    Flag,
    FlagType,
    FlagStatus,
    FlagBundle,
    FlagEvaluation,
    FeatureFlagOverride,
    get_feature_flags,
    is_enabled as is_flag_enabled,
    get_flag_value,
)

# G.4.4: Embedding Drift Monitoring (December 2025)
from .embedding_drift import (
    EmbeddingDriftMonitor,
    DriftConfig,
    DriftType,
    DriftSeverity,
    DriftMetrics,
    DriftResult,
    AlertEvent,
    get_drift_monitor,
    record_embedding_for_drift,
    check_embedding_drift,
)

# G.4.5: Blue-Green Deployment for Zero-Downtime Migrations (December 2025)
from .blue_green import (
    BlueGreenManager,
    BlueGreenConfig,
    Deployment,
    DeploymentSlot,
    DeploymentStatus,
    DeploymentConfig,
    DeploymentEvent,
    HealthCheck,
    HealthStatus,
    ModelProvider,
    MockModelProvider,
    get_blue_green_manager,
    deploy_model,
    switch_to_slot,
    rollback_deployment,
)

# G.5.1: Speculative RAG for 51% Latency Reduction (December 2025)
from .speculative_rag import (
    SpeculativeRAG,
    SpeculativeRAGConfig,
    SpeculativeRAGResult,
    Document as SpeculativeDocument,
    Draft,
    PartitionStrategy,
    SelectionMethod,
    get_speculative_rag,
    speculative_generate,
)

# G.5.2: LLMLingua-2 Prompt Compression (December 2025)
from .prompt_compressor import (
    PromptCompressor,
    CompressionConfig,
    CompressionResult,
    CompressionMethod,
    CompressionLevel,
    get_prompt_compressor,
    compress_prompt,
    LIGHT_COMPRESSION,
    MODERATE_COMPRESSION,
    AGGRESSIVE_COMPRESSION,
    FANUC_COMPRESSION,
    LLMLINGUA_AVAILABLE,
)

# G.5.3: Jina-ColBERT-v2 (8K Context, MRL Support) (December 2025)
from .jina_colbert import (
    JinaColBERT,
    ColBERTConfig,
    ColBERTEmbedding,
    ColBERTScore,
    ColBERTSearchResult,
    MRLDimension,
    ScoringMethod,
    get_jina_colbert,
    colbert_encode,
    colbert_search,
    SENTENCE_TRANSFORMERS_AVAILABLE,
    COLBERT_AVAILABLE,
)

# G.5.4: RAPTOR Recursive Summarization (December 2025)
# Based on ICLR 2024: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
# 20% accuracy improvement on QuALITY benchmark
try:
    from .raptor import (
        RAPTORConfig,
        RAPTORBuilder,
        RAPTORRetriever,
        RAPTORTree,
        TreeNode,
        RetrievalResult as RAPTORRetrievalResult,
        ClusteringMethod,
        SummarizationStyle,
        get_raptor_builder,
        build_raptor_tree,
        raptor_retrieve,
        get_raptor_tree,
        list_raptor_trees,
    )
    RAPTOR_AVAILABLE = True
except ImportError as e:
    RAPTOR_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"RAPTOR not available: {e}")

# G.5.5: HopRAG Multi-Hop Passage Graphs (December 2025)
# Based on February 2025: "HopRAG: Multi-Hop Reasoning via Knowledge Graph Retrieval"
# 76% higher answer metric, 65% retrieval F1 improvement
try:
    from .hoprag import (
        HopRAGConfig,
        HopRAGBuilder,
        HopRAGRetriever,
        PassageGraph,
        Passage,
        Edge,
        ReasoningPath,
        HopRAGResult,
        EdgeType,
        HopStrategy,
        get_hoprag_builder,
        build_hoprag_graph,
        hoprag_retrieve,
        get_hoprag_graph,
        list_hoprag_graphs,
    )
    HOPRAG_AVAILABLE = True
except ImportError as e:
    HOPRAG_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"HopRAG not available: {e}")

# G.5.6: Three-Way Hybrid Fusion (December 2025)
# Combines BM25 + Dense + ColBERT for +15-25% nDCG@10 improvement
try:
    from .hybrid_fusion import (
        HybridFusionConfig,
        HybridFusionRetriever,
        HybridFusionResult,
        FusedResult,
        FusionMethod,
        RetrieverType,
        Document as FusionDocument,
        RetrievalScore,
        BM25Index,
        get_hybrid_fusion_retriever,
        hybrid_fusion_search,
        add_to_fusion_index,
        get_fusion_stats,
    )
    HYBRID_FUSION_AVAILABLE = True
except ImportError as e:
    HYBRID_FUSION_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Hybrid Fusion not available: {e}")

# G.6.2: DyLAN Agent Importance Scores (December 2025)
# Based on DyLAN: Dynamic Language Agent Network (2025)
# Implements conditional agent skipping for 25% accuracy improvement
try:
    from .dylan_agent_network import (
        DyLANAgentNetwork,
        QueryComplexity,
        AgentRole,
        AgentContribution,
        AgentStats,
        SkipDecision,
        QueryComplexityResult,
        get_dylan_network,
        reset_dylan_network,
    )
    DYLAN_AVAILABLE = True
except ImportError as e:
    DYLAN_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"DyLAN not available: {e}")

# G.6.4: Information Bottleneck Filtering (December 2025)
# Based on Zhu et al., ACL 2024: "An Information Bottleneck Perspective for
# Effective Noise Filtering on Retrieval-Augmented Generation"
# Achieves 2.5% compression rate while improving answer correctness
try:
    from .information_bottleneck import (
        InformationBottleneckFilter,
        FilteringLevel,
        ContentType as IBContentType,
        PassageScore,
        IBFilterResult,
        get_ib_filter,
        create_ib_filter,
    )
    IB_FILTER_AVAILABLE = True
except ImportError as e:
    IB_FILTER_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Information Bottleneck not available: {e}")

# G.6.5 Contrastive Retriever (R3)
try:
    from .contrastive_retriever import (
        ContrastiveRetriever,
        DocumentOutcome,
        DocumentUtility,
        RetrievalSession,
        RetrievalStrategy,
        RetrievalInsight,
        get_contrastive_retriever,
    )
    CONTRASTIVE_RETRIEVER_AVAILABLE = True
except ImportError as e:
    CONTRASTIVE_RETRIEVER_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Contrastive Retriever not available: {e}")

# G.7.2: Hyperbolic Embeddings for Hierarchical Documents (December 2025)
# Based on HyperbolicRAG (arXiv:2511.18808) - +5.6% Recall@5 improvement
from .hyperbolic_embeddings import (
    PoincareBall,
    HyperbolicRetriever,
    HyperbolicDocument,
    HyperbolicSearchResult,
    HierarchyLevel,
    detect_hierarchy_level,
    get_hyperbolic_retriever,
)

# G.7.3: Optimal Transport for Dense-Sparse Fusion (December 2025)
# Based on Wasserstein distance and Sinkhorn algorithm for superior alignment
# Extended with Sliced-Wasserstein (O(n log n)) and Word Mover's Distance
from .optimal_transport import (
    SinkhornSolver,
    GromovWassersteinSolver,
    SlicedWassersteinSolver,
    WordMoverSolver,
    OptimalTransportFusion,
    OTConfig,
    OTMethod,
    CostMetric,
    OTResult,
    TransportPlan,
    get_ot_fusion,
    ot_fuse_scores,
    ot_fuse_multiway,
)

# G.7.4: TSDAE Domain Adaptation (unsupervised, no fine-tuning)
from .tsdae_adapter import (
    TSDaeAdapter,
    MultiDomainAdapter,
    DomainConfig,
    AdaptationResult,
    AdaptationStatus,
    DomainEmbeddingResult,
    NoiseType,
    PoolingMode,
    FANUC_DOMAIN_CONFIG,
    SIEMENS_DOMAIN_CONFIG,
    ROCKWELL_DOMAIN_CONFIG,
    get_tsdae_adapter,
    get_multi_domain_adapter,
)

# K.2: Docling Document Processor Integration (December 2025)
# Based on arXiv:2408.09869 - Docling Technical Report
# 97.9% TEDS-S table extraction accuracy
from .docling_adapter import (
    DoclingAdapter,
    DoclingFormat,
    DocumentType as DoclingDocumentType,
    ExtractionQuality,
    TableData,
    ExtractedDocument,
    DoclingStats,
    get_docling_adapter,
    cleanup_docling_adapter,
)

__all__ = [
    # ==========================================================================
    # PRIMARY: UniversalOrchestrator - SINGLE SOURCE OF TRUTH
    # ==========================================================================
    "UniversalOrchestrator",
    "OrchestratorPreset",
    "FeatureConfig",
    "PRESET_CONFIGS",

    # ==========================================================================
    # DEPRECATED Orchestrators (maintained for backward compatibility)
    # ==========================================================================
    "AgenticOrchestrator",  # DEPRECATED: Use UniversalOrchestrator(preset=BALANCED)
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "VerificationResult",
    "AgentAction",
    "SearchState",
    # Events
    "EventType",
    "SearchEvent",
    "EventEmitter",
    "EventManager",
    "get_event_manager",
    "events",
    # Graph Visualization
    "AgentGraphState",
    "get_graph_state",
    "reset_graph_state",
    "graph_node_entered",
    "graph_node_completed",
    "graph_state_update",
    "graph_edge_traversed",
    "graph_branch_created",
    "graph_paths_merged",
    # TTL Cache Management
    "TTLCacheManager",
    "ToolType",
    "ToolCallContext",
    "get_ttl_cache_manager",
    # Prompt Registry
    "build_prompt",
    "get_system_prompt",
    "get_template",
    "CHAIN_OF_DRAFT_INSTRUCTION",
    # Artifact Store
    "ArtifactStore",
    "ArtifactType",
    "get_artifact_store",
    "store_search_results",
    "store_scraped_content",
    "store_synthesis",
    # Performance Metrics
    "PerformanceMetrics",
    "QueryMetrics",
    "PhaseTimer",
    "get_performance_metrics",
    # Search/Scrape Metrics
    "SearchMetrics",
    "get_search_metrics",
    # Phase 4: KV Cache Service
    "KVCacheService",
    "CacheBackend",
    "CacheState",
    "get_kv_cache_service",
    "warm_system_prompts",
    # Phase 4: Memory Tiers
    "MemoryTierManager",
    "MemoryTier",
    "ContentType",
    "PlaintextStorage",
    "get_memory_tier_manager",
    "initialize_memory_tiers",
    # AIME-Style Dynamic Planning (Phase 1 Enhancement)
    "DynamicPlanner",
    "TaskNode",
    "TaskStatus",
    "ActionType",
    "TacticalAction",
    "ExecutionResult",
    "PlannerOutput",
    # Progress Tools
    "ProgressReporter",
    "ProgressAggregator",
    "ProgressStatus",
    "ProgressUpdate",
    "ProgressTool",
    "PROGRESS_TOOL_PROMPT",
    # Dynamic Orchestrator (DEPRECATED: Use UniversalOrchestrator with preset=RESEARCH)
    "DynamicOrchestrator",  # DEPRECATED
    "create_dynamic_orchestrator",  # DEPRECATED
    # GSW-Style Entity Tracking (Phase 2 Enhancement)
    "EntityTracker",
    "EntityState",
    "EntityType",
    "RoleType",
    "RelationType",
    "RoleAssignment",
    "EntityEvent",
    "EntityRelation",
    "VerbFrame",
    "create_entity_tracker",
    # DAG-Based Reasoning (Phase 3 Enhancement)
    "ReasoningDAG",
    "ReasoningNode",
    "NodeType",
    "NodeStatus",
    "VerificationResult",
    "create_reasoning_dag",
    # Buffer of Thoughts (Phase 4 Enhancement)
    "ThoughtLibrary",
    "ThoughtTemplate",
    "TemplateCategory",
    "InstantiatedThought",
    "create_thought_library",
    "get_thought_library",
    # Actor Factory (Phase 5 Enhancement)
    "ActorFactory",
    "DynamicActor",
    "ActorPersona",
    "ToolBundle",
    "Tool",
    "ToolCategory",
    "ModelCapability",
    "create_actor_factory",
    "get_actor_factory",
    # Self-RAG Reflection (Phase 6 Enhancement)
    "SelfReflectionAgent",
    "ReflectionResult",
    "ReflectionToken",
    "SupportLevel",
    "TemporalFact",
    "TemporalConflict",
    "create_self_reflection_agent",
    "get_self_reflection_agent",
    # CRAG Retrieval Evaluator (Phase 7 Enhancement)
    "RetrievalEvaluator",
    "RetrievalEvaluation",
    "RetrievalQuality",
    "CorrectiveAction",
    "DocumentScore",
    "create_retrieval_evaluator",
    "get_retrieval_evaluator",
    # Experience Distillation (Phase 8 Enhancement)
    "ExperienceDistiller",
    "SearchExperience",
    "DistillationResult",
    "create_experience_distiller",
    "get_experience_distiller",
    # Classifier Feedback (Phase 9 Enhancement)
    "ClassifierFeedback",
    "ClassificationOutcome",
    "AdaptiveHint",
    "OutcomeQuality",
    "create_classifier_feedback",
    "get_classifier_feedback",
    # Domain Corpus System (December 2025)
    "DomainSchema",
    "DomainEntityDef",
    "DomainRelationDef",
    "DomainEntity",
    "DomainRelation",
    "DomainCorpus",
    "CorpusBuilder",
    "CorpusRetriever",
    "DomainCorpusManager",
    "TroubleshootingEntityType",
    "TroubleshootingRelationType",
    "create_fanuc_schema",
    "create_raspberry_pi_schema",
    "get_corpus_manager",
    "initialize_default_corpuses",
    # RJG Scientific Molding Corpus (December 2025)
    "RJGCorpusScraper",
    "ScrapeResult",
    "create_rjg_corpus_schema",
    "get_rjg_scraper",
    "RJG_SEED_URLS",
    "RJG_ARTICLE_URLS",
    # PLC/Automation Corpus (December 2025)
    "PLCCorpusScraper",
    "PLCScrapeResult",
    "create_plc_corpus_schema",
    "get_plc_scraper",
    "PLC_SEED_URLS",
    "PLC_ARTICLE_URLS",
    # Industrial Acronym Dictionary (December 2025)
    "AcronymInfo",
    "INDUSTRIAL_ACRONYMS",
    "FANUC_ERROR_CODES",
    "expand_acronyms",
    "expand_acronym",
    "expand_error_code_prefixes",
    "get_acronym_info",
    "get_related_terms",
    "get_category_acronyms",
    "get_dictionary_stats",
    # Query Classification (December 2025)
    "QueryClassifier",
    "QueryClassification",
    "QueryCategory",
    "QueryComplexity",
    "RecommendedPipeline",
    "classify_query",
    "get_query_classifier",
    # Master Embedding Aggregation (December 2025)
    "EmbeddingAggregator",
    "DomainExpert",
    "AggregatedEmbedding",
    "SubManifoldResult",
    "RetrievalResult",
    "get_embedding_aggregator",
    "retrieve_with_entities",
    # Entity-Enhanced Retrieval (December 2025)
    "EntityEnhancedRetriever",
    "EnhancedRetrievalResult",
    "entity_enhanced_retrieve",
    "get_entity_enhanced_retriever",
    # Mixed-Precision Embeddings (December 2025)
    "MixedPrecisionEmbeddingService",
    "PrecisionLevel",
    "QuantizedEmbedding",
    "MixedPrecisionSearchResult",
    "RetrievalStats",
    "EmbeddingModelSpec",
    "QWEN3_EMBEDDING_MODELS",
    "MODEL_TIERS",
    "get_model_spec",
    "get_model_dimension",
    "get_mixed_precision_service",
    # BGE-M3 Hybrid Retrieval (December 2025, G.1.1: ColBERT support)
    "BGEM3HybridRetriever",
    "HybridDocument",
    "HybridSearchResult",
    "HybridRetrievalStats",
    "RetrievalMode",
    "BM25Index",
    "get_hybrid_retriever",
    "create_hybrid_retriever",
    "BGEM3Embeddings",  # G.1.1
    "get_bge_m3_model",  # G.1.1
    "COLBERT_AVAILABLE",  # G.1.1
    # HyDE Query Expansion (December 2025)
    "HyDEExpander",
    "HyDEConfig",
    "HyDEMode",
    "HyDEDocumentType",
    "HyDEResult",
    "get_hyde_expander",
    "create_hyde_expander",
    # RAGAS Evaluation (December 2025)
    "RAGASEvaluator",
    "RAGASConfig",
    "RAGASResult",
    "EvaluationMetric",
    "ClaimVerification",
    "get_ragas_evaluator",
    "create_ragas_evaluator",
    # Sufficient Context & Mitigations (December 2025)
    "SufficientContextClassifier",
    "PositionalOptimizer",
    "DynamicContextAllocator",
    "SufficiencyResult",
    "ContextSufficiency",
    "PositionalAnalysis",
    "get_sufficient_context_classifier",
    "get_positional_optimizer",
    "get_dynamic_allocator",
    # Unified Orchestrator (DEPRECATED: Use UniversalOrchestrator with preset=ENHANCED)
    "UnifiedOrchestrator",  # DEPRECATED
    "get_unified_orchestrator",  # DEPRECATED
    "create_unified_orchestrator",  # DEPRECATED
    # Base Pipeline (December 2025)
    "BaseSearchPipeline",
    # Adaptive Refinement (December 2025)
    "AdaptiveRefinementEngine",
    "RefinementDecision",
    "AnswerGrade",
    "GapAnalysis",
    "AnswerAssessment",
    "RefinementResult",
    "get_adaptive_refinement_engine",
    "create_adaptive_refinement_engine",
    # Universal Orchestrator (December 2025) - SINGLE SOURCE OF TRUTH
    # (Also exported at top of __all__ for visibility)
    # "UniversalOrchestrator",
    # "FeatureConfig",
    # "OrchestratorPreset",
    # "PRESET_CONFIGS",
    # Phase 1 Context Curation (December 2025)
    "DocumentInformationGain",
    "DIGScore",
    "DIGCategory",
    "DIGBatchResult",
    "get_dig_scorer",
    "RedundancyDetector",
    "DocumentCluster",
    "DeduplicationResult",
    "SelectionMethod",
    "get_redundancy_detector",
    "ContextCurator",
    "CuratedContext",
    "CurationConfig",
    "CurationPreset",
    "CoverageAnalysis",
    "CurationTrace",
    "get_context_curator",
    "curate_context",
    # Phase 2 Confidence-Calibrated Halting (December 2025)
    "EntropyMonitor",
    "EntropyResult",
    "EntropyState",
    "HaltDecision",
    "get_entropy_monitor",
    "SelfConsistencyChecker",
    "ConvergenceResult",
    "ConvergenceStatus",
    "SynthesisAttempt",
    "AnswerCluster",
    "get_consistency_checker",
    "IterationBandit",
    "BanditDecision",
    "RefinementAction",
    "RefinementState",
    "ActionOutcome",
    "ArmStats",
    "get_iteration_bandit",
    # Phase 3 Enhanced Query Generation (December 2025)
    "FLARERetriever",
    "FLAREResult",
    "RetrievalPoint",
    "RetrievalTrigger",
    "get_flare_retriever",
    "QueryTreeDecoder",
    "QueryTree",
    "QueryNode",
    "QueryOperation",
    "TreeDecodingResult",
    "get_query_tree_decoder",
    "QueryNodeStatus",
    # Phase 4 Scratchpad Enhancement (December 2025)
    "SemanticMemoryNetwork",
    "Memory",
    "MemoryConnection",
    "MemoryType",
    "ConnectionType",
    "TraversalResult",
    "get_semantic_memory",
    "RAISEScratchpad",
    "Observation",
    "ReasoningStep",
    "Example",
    "TrajectoryStep",
    "QualitySignal",
    "ObservationType",
    "ReasoningType",
    "UncertaintyIndicator",
    "create_raise_scratchpad",
    # Phase 5 Template Reuse Optimization (December 2025)
    "MetaBuffer",
    "DistilledTemplate",
    "TemplateType",
    "InstantiatedTemplate",
    "get_meta_buffer",
    "ReasoningComposer",
    "ReasoningModule",
    "ComposedStrategy",
    "ModuleDefinition",
    "get_reasoning_composer",
    # Part F: Evaluation & Metrics (December 2025)
    "BenchmarkQuery",
    "BenchmarkResult",
    "BenchmarkReport",
    "TechnicalAccuracyScorer",
    "BenchmarkRunner",
    "QueryDifficulty",
    "QueryCategory",
    "FANUC_BENCHMARK",
    "run_benchmark",
    "get_benchmark_stats",
    "filter_benchmark",
    # G.1.3: OpenTelemetry Tracing (December 2025)
    "configure_tracing",
    "get_tracer",
    "get_agentic_tracer",
    "trace_operation",
    "trace_span",
    "is_tracing_enabled",
    "get_tracing_status",
    "shutdown_tracing",
    "TracingConfig",
    "AgenticTracer",
    "OTEL_AVAILABLE",
    "OTLP_AVAILABLE",
    # G.1.5: DeepEval CI Pipeline Integration (December 2025)
    "DeepEvalRAGEvaluator",
    "DeepEvalMetric",
    "DeepEvalResult",
    "BenchmarkEvaluationResult",
    "get_deepeval_evaluator",
    "evaluate_rag_response",
    "run_benchmark_evaluation",
    "get_evaluation_summary",
    "create_deepeval_test_cases",
    "is_deepeval_available",
    "DEEPEVAL_AVAILABLE",
    # G.2.1-G.2.2: Cascade Retriever (December 2025)
    "CascadeRetriever",
    "CascadeConfig",
    "CascadeStage",
    "CascadeResult",
    "CascadeStats",
    "get_cascade_retriever",
    "get_cascade_retriever_async",
    # G.2.3: Fusion Weight Adapter (December 2025)
    "FusionWeightAdapter",
    "FusionWeights",
    "QueryIntent",
    "IntentClassification",
    "get_fusion_weight_adapter",
    "get_adaptive_weights",
    "classify_for_fusion",
    # G.2.4: Qdrant On-Disk Storage (December 2025)
    "QdrantStorage",
    "StorageConfig",
    "QuantizationType",
    "DistanceMetric",
    "QdrantSearchResult",
    "CollectionInfo",
    "get_qdrant_storage",
    "get_qdrant_storage_async",
    "is_qdrant_available",
    "QDRANT_AVAILABLE",
    "VRAM_EFFICIENT_CONFIG",
    "MAXIMUM_COMPRESSION_CONFIG",
    "BALANCED_CONFIG",
    # G.2.5: Adaptive Top-K CAR Algorithm (December 2025)
    "AdaptiveTopK",
    "AdaptiveTopKConfig",
    "AdaptiveQueryComplexity",
    "StoppingReason",
    "ComplexityFeatures",
    "ScoreDistribution",
    "AdaptiveKResult",
    "EarlyStopResult",
    "get_adaptive_topk",
    "compute_adaptive_k",
    "apply_early_stopping",
    "PRECISION_CONFIG",
    "BALANCED_ADAPTIVE_CONFIG",
    "RECALL_CONFIG",
    # G.3.1-G.3.4: NanoGraphRAG with PPR + Leiden (December 2025)
    "NanoGraphRAG",
    "GraphRAGConfig",
    "GraphRAGIntegration",
    "GraphQueryMode",
    "GraphEntityType",
    "GraphEntity",
    "GraphRelationship",
    "GraphCommunity",
    "GraphQueryResult",
    "get_nano_graphrag",
    "get_graphrag_integration",
    "initialize_graphrag",
    "NETWORKX_AVAILABLE",
    "LEIDEN_AVAILABLE",
    # G.3.3: GLiNER + Regex Hybrid Entity Extraction (December 2025)
    "GLiNERHybridExtractor",
    "TroubleshootingExtractor",
    "ExtractedEntity",
    "ExtractionResult",
    "EntitySource",
    "EntityCategory",
    "PatternRegistry",
    "get_gliner_extractor",
    "get_troubleshooting_extractor",
    "is_gliner_available",
    "GLINER_MODEL_AVAILABLE",
    # G.3.5: Late Chunking for Context-Aware Embedding (December 2025)
    "LateChunker",
    "ChunkingConfig",
    "ChunkingStrategy",
    "LateChunk",
    "TokenSpan",
    "LateChunkingResult",
    "get_late_chunker",
    "create_late_chunker",
    "compare_chunking_methods",

    # G.4.1: Circuit Breakers for Production Reliability (December 2025)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitState",
    "CircuitMetrics",
    "CircuitBreakerError",
    "CircuitOpenError",
    "AllCircuitsOpenError",
    "get_circuit_breaker_registry",
    "with_circuit_breaker",
    "create_llm_circuit_breaker",
    "create_search_circuit_breaker",
    "create_embedding_circuit_breaker",
    "create_scraping_circuit_breaker",
    "execute_with_fallback",
    "get_circuit_health",
    "get_circuit_status",
    "reset_circuit",
    "LLM_CIRCUIT_CONFIG",
    "SEARCH_CIRCUIT_CONFIG",
    "EMBEDDING_CIRCUIT_CONFIG",
    "SCRAPING_CIRCUIT_CONFIG",

    # G.4.2: Shadow Mode for Embedding Model Testing (December 2025)
    "ShadowEmbeddingTester",
    "ShadowEmbeddingRegistry",
    "ShadowConfig",
    "ShadowMode",
    "ShadowMetrics",
    "ComparisonResult",
    "EmbeddingProvider",
    "OllamaEmbeddingAdapter",
    "get_shadow_registry",
    "create_ollama_shadow_tester",

    # G.4.3: Feature Flags for Gradual Rollout (December 2025)
    "FeatureFlagManager",
    "Flag",
    "FlagType",
    "FlagStatus",
    "FlagBundle",
    "FlagEvaluation",
    "FeatureFlagOverride",
    "get_feature_flags",
    "is_flag_enabled",
    "get_flag_value",

    # G.4.4: Embedding Drift Monitoring (December 2025)
    "EmbeddingDriftMonitor",
    "DriftConfig",
    "DriftType",
    "DriftSeverity",
    "DriftMetrics",
    "DriftResult",
    "AlertEvent",
    "get_drift_monitor",
    "record_embedding_for_drift",
    "check_embedding_drift",

    # G.4.5: Blue-Green Deployment (December 2025)
    "BlueGreenManager",
    "BlueGreenConfig",
    "Deployment",
    "DeploymentSlot",
    "DeploymentStatus",
    "DeploymentConfig",
    "DeploymentEvent",
    "HealthCheck",
    "HealthStatus",
    "ModelProvider",
    "MockModelProvider",
    "get_blue_green_manager",
    "deploy_model",
    "switch_to_slot",
    "rollback_deployment",

    # G.5.1: Speculative RAG (December 2025)
    "SpeculativeRAG",
    "SpeculativeRAGConfig",
    "SpeculativeRAGResult",
    "SpeculativeDocument",
    "Draft",
    "PartitionStrategy",
    "SelectionMethod",
    "get_speculative_rag",
    "speculative_generate",

    # G.5.2: LLMLingua-2 Prompt Compression (December 2025)
    "PromptCompressor",
    "CompressionConfig",
    "CompressionResult",
    "CompressionMethod",
    "CompressionLevel",
    "get_prompt_compressor",
    "compress_prompt",
    "LIGHT_COMPRESSION",
    "MODERATE_COMPRESSION",
    "AGGRESSIVE_COMPRESSION",
    "FANUC_COMPRESSION",
    "LLMLINGUA_AVAILABLE",

    # G.5.3: Jina-ColBERT-v2 (December 2025)
    "JinaColBERT",
    "ColBERTConfig",
    "ColBERTEmbedding",
    "ColBERTScore",
    "ColBERTSearchResult",
    "MRLDimension",
    "ScoringMethod",
    "get_jina_colbert",
    "colbert_encode",
    "colbert_search",
    "SENTENCE_TRANSFORMERS_AVAILABLE",
    "COLBERT_AVAILABLE",

    # G.5.4: RAPTOR Recursive Summarization (December 2025)
    "RAPTORConfig",
    "RAPTORBuilder",
    "RAPTORRetriever",
    "RAPTORTree",
    "TreeNode",
    "RAPTORRetrievalResult",
    "ClusteringMethod",
    "SummarizationStyle",
    "get_raptor_builder",
    "build_raptor_tree",
    "raptor_retrieve",
    "get_raptor_tree",
    "list_raptor_trees",
    "RAPTOR_AVAILABLE",

    # G.5.5: HopRAG Multi-Hop Passage Graphs (December 2025)
    "HopRAGConfig",
    "HopRAGBuilder",
    "HopRAGRetriever",
    "PassageGraph",
    "Passage",
    "Edge",
    "ReasoningPath",
    "HopRAGResult",
    "EdgeType",
    "HopStrategy",
    "get_hoprag_builder",
    "build_hoprag_graph",
    "hoprag_retrieve",
    "get_hoprag_graph",
    "list_hoprag_graphs",
    "HOPRAG_AVAILABLE",

    # G.5.6: Three-Way Hybrid Fusion (December 2025)
    "HybridFusionConfig",
    "HybridFusionRetriever",
    "HybridFusionResult",
    "FusedResult",
    "FusionMethod",
    "RetrieverType",
    "FusionDocument",
    "RetrievalScore",
    "BM25Index",
    "get_hybrid_fusion_retriever",
    "hybrid_fusion_search",
    "add_to_fusion_index",
    "get_fusion_stats",
    "HYBRID_FUSION_AVAILABLE",

    # G.6.2: DyLAN Agent Importance Scores (December 2025)
    "DyLANAgentNetwork",
    "QueryComplexity",
    "AgentRole",
    "AgentContribution",
    "AgentStats",
    "SkipDecision",
    "QueryComplexityResult",
    "get_dylan_network",
    "reset_dylan_network",
    "DYLAN_AVAILABLE",
    # G.6.4: Information Bottleneck Filtering
    "InformationBottleneckFilter",
    "FilteringLevel",
    "IBContentType",
    "PassageScore",
    "IBFilterResult",
    "get_ib_filter",
    "create_ib_filter",
    "IB_FILTER_AVAILABLE",
    # Contrastive Retriever (G.6.5)
    "ContrastiveRetriever",
    "DocumentOutcome",
    "DocumentUtility",
    "RetrievalSession",
    "RetrievalStrategy",
    "RetrievalInsight",
    "get_contrastive_retriever",
    "CONTRASTIVE_RETRIEVER_AVAILABLE",

    # G.7.2: Hyperbolic Embeddings for Hierarchical Documents
    "PoincareBall",
    "HyperbolicRetriever",
    "HyperbolicDocument",
    "HyperbolicSearchResult",
    "HierarchyLevel",
    "detect_hierarchy_level",
    "get_hyperbolic_retriever",

    # G.7.3: Optimal Transport for Dense-Sparse Fusion
    # Extended with Sliced-Wasserstein (O(n log n)) and Word Mover's Distance
    "SinkhornSolver",
    "GromovWassersteinSolver",
    "SlicedWassersteinSolver",
    "WordMoverSolver",
    "OptimalTransportFusion",
    "OTConfig",
    "OTMethod",
    "CostMetric",
    "OTResult",
    "TransportPlan",
    "get_ot_fusion",
    "ot_fuse_scores",
    "ot_fuse_multiway",

    # G.7.4: TSDAE Domain Adaptation
    "TSDaeAdapter",
    "MultiDomainAdapter",
    "DomainConfig",
    "AdaptationResult",
    "AdaptationStatus",
    "DomainEmbeddingResult",
    "NoiseType",
    "PoolingMode",
    "FANUC_DOMAIN_CONFIG",
    "SIEMENS_DOMAIN_CONFIG",
    "ROCKWELL_DOMAIN_CONFIG",
    "get_tsdae_adapter",
    "get_multi_domain_adapter",

    # K.2: Docling Document Processor (December 2025)
    # Based on arXiv:2408.09869 - 97.9% TEDS-S table extraction accuracy
    "DoclingAdapter",
    "DoclingFormat",
    "DoclingDocumentType",
    "ExtractionQuality",
    "TableData",
    "ExtractedDocument",
    "DoclingStats",
    "get_docling_adapter",
    "cleanup_docling_adapter",
]

__version__ = "0.78.0"  # K.3 Table Complexity Routing to Docling
