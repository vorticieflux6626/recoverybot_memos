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
]

__version__ = "0.6.0"  # Updated for DAG-Based Reasoning (Phase 3)
