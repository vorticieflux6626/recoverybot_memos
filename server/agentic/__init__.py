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
"""

from .orchestrator import AgenticOrchestrator
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
from . import events

__all__ = [
    # Core orchestration
    "AgenticOrchestrator",
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
    # Dynamic Orchestrator
    "DynamicOrchestrator",
    "create_dynamic_orchestrator",
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
]

__version__ = "0.15.0"  # Entity-enhanced retrieval with master embedding aggregation
