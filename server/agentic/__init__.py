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

# DEPRECATED: Use UniversalOrchestrator instead
from .orchestrator import AgenticOrchestrator  # DEPRECATED
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
from .orchestrator_dynamic import (
    DynamicOrchestrator,
    create_dynamic_orchestrator
)
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
    create_hybrid_retriever
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
from .orchestrator_unified import (
    UnifiedOrchestrator,
    get_unified_orchestrator,
    create_unified_orchestrator
)
from .base_pipeline import (
    BaseSearchPipeline
)
from .orchestrator_universal import (
    UniversalOrchestrator,
    FeatureConfig,
    OrchestratorPreset,
    PRESET_CONFIGS
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
    # BGE-M3 Hybrid Retrieval (December 2025)
    "BGEM3HybridRetriever",
    "HybridDocument",
    "HybridSearchResult",
    "HybridRetrievalStats",
    "RetrievalMode",
    "BM25Index",
    "get_hybrid_retriever",
    "create_hybrid_retriever",
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
    # Universal Orchestrator (December 2025) - SINGLE SOURCE OF TRUTH
    # (Also exported at top of __all__ for visibility)
    # "UniversalOrchestrator",
    # "FeatureConfig",
    # "OrchestratorPreset",
    # "PRESET_CONFIGS",
]

__version__ = "0.27.0"  # Orchestrator consolidation: UniversalOrchestrator as single source of truth, deprecated legacy orchestrators
