"""
Universal Agentic Orchestrator

Consolidates ALL orchestrator functionality into a single configurable class:
- Base ReAct pipeline (AgenticOrchestrator)
- Enhanced reasoning patterns (EnhancedAgenticOrchestrator)
- AIME-style dynamic planning (DynamicOrchestrator)
- Graph-based KV cache optimization (GraphEnhancedOrchestrator)
- Unified feature integration (UnifiedOrchestrator)

Feature flags allow enabling/disabling any combination of features.
Presets provide sensible defaults for common use cases.

Usage:
    # Quick minimal search
    orchestrator = UniversalOrchestrator.minimal()

    # Full research mode
    orchestrator = UniversalOrchestrator.research()

    # Custom configuration
    orchestrator = UniversalOrchestrator(
        enable_hyde=True,
        enable_reasoning_dag=True,
        enable_parallel_execution=True
    )
"""

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Set
import uuid

from .base_pipeline import BaseSearchPipeline
from .models import (
    SearchRequest,
    SearchResponse,
    SearchResultData,
    SearchMeta,
    SearchState,
    SearchMode,
    ConfidenceLevel,
    WebSearchResult,
    QueryAnalysis
)
from .events import (
    EventEmitter, EventType, SearchEvent,
    get_graph_state, reset_graph_state,
    graph_node_entered, graph_node_completed,
    llm_call_start, llm_call_complete
)

# Core agents (always available)
from .query_classifier import QueryClassifier, QueryClassification, RecommendedPipeline
from .self_reflection import SelfReflectionAgent, ReflectionResult, get_self_reflection_agent
from .retrieval_evaluator import RetrievalEvaluator, RetrievalEvaluation, CorrectiveAction
from .experience_distiller import ExperienceDistiller, get_experience_distiller
from .classifier_feedback import ClassifierFeedback, get_classifier_feedback
from .adaptive_refinement import (
    AdaptiveRefinementEngine, RefinementDecision, GapAnalysis, AnswerAssessment,
    get_adaptive_refinement_engine
)
from .sufficient_context import (
    SufficientContextClassifier,
    get_sufficient_context_classifier
)

# Enhanced features (optional)
from .hyde import HyDEExpander, HyDEMode, HyDEResult, get_hyde_expander
from .bge_m3_hybrid import (
    BGEM3HybridRetriever, HybridSearchResult, RetrievalMode, get_hybrid_retriever
)
from .ragas import RAGASEvaluator, RAGASResult, get_ragas_evaluator
from .entity_tracker import EntityTracker, EntityState, create_entity_tracker
from .thought_library import ThoughtLibrary, ThoughtTemplate, get_thought_library
from .embedding_aggregator import EmbeddingAggregator, get_embedding_aggregator
from .domain_corpus import DomainCorpusManager, get_corpus_manager, initialize_default_corpuses
from .reasoning_dag import ReasoningDAG, NodeType, create_reasoning_dag
from .metrics import PerformanceMetrics, QueryMetrics, get_performance_metrics
from .prefix_optimized_prompts import SYSTEM_PREFIX, build_scratchpad_context

# Enhanced reasoning (from EnhancedAgenticOrchestrator)
from .enhanced_reasoning import (
    EnhancedReasoningEngine,
    get_enhanced_reasoning,
    PreActPlan,
    StuckStateMetrics
)

# Dynamic planning (from DynamicOrchestrator)
from .dynamic_planner import DynamicPlanner, TaskNode, TaskStatus, PlannerOutput
from .progress_tools import ProgressReporter, ProgressAggregator

# Graph cache (from GraphEnhancedOrchestrator)
from .agent_step_graph import AgentType, get_agent_step_graph
from .scratchpad_cache import get_scratchpad_cache
from .graph_cache_integration import GraphCacheIntegration, get_graph_cache_integration

# Additional features for full integration
from .mixed_precision_embeddings import (
    MixedPrecisionEmbeddingService,
    get_mixed_precision_service
)
from .entity_enhanced_retrieval import (
    EntityEnhancedRetriever,
    get_entity_enhanced_retriever
)
from .actor_factory import (
    ActorFactory,
    DynamicActor,
    get_actor_factory
)
from .multi_agent import MultiAgentOrchestrator
from .scraper import VisionAnalyzer, DeepReader
from .scratchpad import AgenticScratchpad, FindingType, QuestionStatus
from .ttl_cache_manager import get_ttl_cache_manager, ToolType, ToolCallContext
from .kv_cache_service import KVCacheService, get_kv_cache_service
from .memory_tiers import MemoryTierManager, get_memory_tier_manager
from .artifacts import ArtifactStore, get_artifact_store, ArtifactType
from .content_cache import get_content_cache
from .adaptive_refinement import (
    AdaptiveRefinementEngine,
    RefinementDecision,
    GapAnalysis,
    AnswerAssessment,
    RefinementResult,
    get_adaptive_refinement_engine
)
from .context_limits import (
    calculate_context_budget,
    get_synthesizer_limits,
    get_search_result_limits,
    get_model_context_window,
    format_context_utilization_report,
    PipelineContextConfig,
    DEFAULT_PIPELINE_CONFIG
)
from .context_curator import (
    ContextCurator,
    CuratedContext,
    CurationConfig,
    CurationPreset,
    get_context_curator
)
# Phase 2: Confidence-Calibrated Halting
from .entropy_monitor import (
    EntropyMonitor,
    EntropyResult,
    HaltDecision,
    get_entropy_monitor
)
from .self_consistency import (
    SelfConsistencyChecker,
    ConvergenceResult,
    ConvergenceStatus,
    get_consistency_checker
)
from .iteration_bandit import (
    IterationBandit,
    BanditDecision,
    RefinementAction,
    RefinementState,
    get_iteration_bandit
)
# Phase 3: Enhanced Query Generation
from .flare_retriever import (
    FLARERetriever,
    FLAREResult,
    RetrievalPoint,
    get_flare_retriever
)
from .query_tree import (
    QueryTreeDecoder,
    QueryTree,
    TreeDecodingResult,
    get_query_tree_decoder
)
# Phase 4: Scratchpad Enhancement
from .semantic_memory import (
    SemanticMemoryNetwork,
    MemoryType,
    ConnectionType,
    get_semantic_memory
)
from .raise_scratchpad import (
    RAISEScratchpad,
    ObservationType,
    ReasoningType,
    create_raise_scratchpad
)

# Phase 5: Template Reuse Optimization
from .meta_buffer import (
    MetaBuffer,
    TemplateType,
    get_meta_buffer
)
from .reasoning_composer import (
    ReasoningComposer,
    get_reasoning_composer
)

logger = logging.getLogger("agentic.orchestrator_universal")


class UniversalGraphState:
    """
    Tracks agent progress for graph visualization in SSE events.

    Graph format: [A✓]→[P✓]→[S•]→[E]→[W]→[V]→[Σ]
    Where:
        A = Analyze, P = Plan, S = Search, E = Evaluate URLs
        W = Web scrape, V = Verify, Σ = Synthesize, ✓ = Complete
        • = In progress (current step), (no mark) = Pending
    """

    # Agent symbols and their full names
    AGENTS = [
        ("A", "Analyze"),
        ("P", "Plan"),
        ("S", "Search"),
        ("E", "Evaluate"),
        ("W", "Scrape"),
        ("V", "Verify"),
        ("Σ", "Synthesize"),
        ("R", "Reflect"),
        ("✓", "Complete")
    ]

    def __init__(self):
        # Track state: 0=pending, 1=active, 2=completed
        self._state = {sym: 0 for sym, _ in self.AGENTS}
        self._active = None

    def enter(self, agent_symbol: str):
        """Mark an agent as active (in progress)."""
        if self._active and self._active != agent_symbol:
            # Complete the previous agent
            self._state[self._active] = 2
        self._active = agent_symbol
        self._state[agent_symbol] = 1
        # Log at INFO for key phases
        if agent_symbol in ["A", "S", "Σ", "✓"]:
            agent_name = next((name for sym, name in self.AGENTS if sym == agent_symbol), agent_symbol)
            logger.info(f"Graph: {agent_name} started → {self.to_line()}")

    def complete(self, agent_symbol: str = None):
        """Mark an agent as completed."""
        symbol = agent_symbol or self._active
        if symbol:
            self._state[symbol] = 2
            if self._active == symbol:
                self._active = None
            # Log graph state change for visibility
            agent_name = next((name for sym, name in self.AGENTS if sym == symbol), symbol)
            # Log at INFO for key phases, DEBUG for others
            if symbol in ["A", "S", "Σ", "✓"]:
                logger.info(f"Graph: {agent_name} completed → {self.to_line()}")
            else:
                logger.debug(f"Graph: {agent_name} completed → {self.to_line()}")

    def to_line(self) -> str:
        """Generate the graph line string."""
        parts = []
        for sym, _ in self.AGENTS:
            state = self._state.get(sym, 0)
            if state == 2:  # Completed
                parts.append(f"[{sym}✓]")
            elif state == 1:  # Active
                parts.append(f"[{sym}•]")
            else:  # Pending
                parts.append(f"[{sym}]")
        return "→".join(parts)

    def reset(self):
        """Reset all state."""
        self._state = {sym: 0 for sym, _ in self.AGENTS}
        self._active = None


class OrchestratorPreset(str, Enum):
    """Preset configurations for common use cases."""
    MINIMAL = "minimal"         # Fast, basic search - no enhancements
    BALANCED = "balanced"       # Good quality/speed trade-off
    ENHANCED = "enhanced"       # All quality features enabled
    RESEARCH = "research"       # Thorough multi-direction exploration
    FULL = "full"               # Everything enabled (expensive)


@dataclass
class FeatureConfig:
    """Feature configuration with all flags."""
    # Core pipeline (always on)
    enable_query_analysis: bool = True
    enable_verification: bool = True
    enable_scratchpad: bool = True

    # Quality control (Layer 1)
    enable_self_reflection: bool = True     # Self-RAG post-synthesis
    enable_crag_evaluation: bool = True     # CRAG pre-synthesis
    enable_sufficient_context: bool = True  # Context sufficiency check
    enable_positional_optimization: bool = True  # Lost-in-the-middle mitigation

    # Learning (Layer 1)
    enable_experience_distillation: bool = True
    enable_classifier_feedback: bool = True

    # Adaptive Refinement (Layer 1.5)
    enable_adaptive_refinement: bool = True     # Iterative gap-filling loop
    enable_answer_grading: bool = True          # Answer quality assessment
    enable_gap_detection: bool = True           # Structured gap identification
    min_confidence_threshold: float = 0.5       # Minimum acceptable confidence
    max_refinement_attempts: int = 3            # Max refinement iterations

    # Performance (Layer 2)
    enable_content_cache: bool = True
    enable_semantic_cache: bool = True
    enable_ttl_pinning: bool = True
    enable_kv_cache_service: bool = False   # KV cache warming
    enable_memory_tiers: bool = False       # Three-tier memory
    enable_artifacts: bool = False          # Artifact storage for token reduction

    # Enhanced retrieval (Layer 2)
    enable_hyde: bool = False              # Query expansion
    enable_hybrid_reranking: bool = False  # BGE-M3 dense+sparse
    enable_mixed_precision: bool = False   # Quantized embeddings
    enable_entity_enhanced_retrieval: bool = False  # Entity-based search

    # Quality scoring (Layer 2)
    enable_ragas: bool = False             # Faithfulness/relevancy
    enable_context_curation: bool = False  # DIG-based context filtering/dedup
    context_curation_preset: str = "balanced"  # fast/balanced/thorough/technical

    # Confidence-Calibrated Halting (Phase 2 - Layer 2)
    enable_entropy_halting: bool = False   # UALA-style entropy monitoring
    enable_iteration_bandit: bool = False  # UCB action selection for iterations
    enable_self_consistency: bool = False  # Multi-path answer convergence

    # Enhanced Query Generation (Phase 3 - Layer 2)
    enable_flare_retrieval: bool = False   # Forward-looking active retrieval
    enable_query_tree: bool = False        # RQ-RAG tree decoding for query expansion

    # Scratchpad Enhancement (Phase 4 - Layer 3)
    enable_semantic_memory: bool = False   # A-MEM Zettelkasten-style memory network
    enable_raise_structure: bool = False   # RAISE four-component scratchpad

    # Template Reuse Optimization (Phase 5 - Layer 3)
    enable_meta_buffer: bool = False       # Cross-session template persistence
    enable_reasoning_composer: bool = False  # Self-Discover reasoning composition

    # Advanced reasoning (Layer 3)
    enable_entity_tracking: bool = False   # GSW entity extraction
    enable_thought_library: bool = False   # Reusable patterns
    enable_reasoning_dag: bool = False     # Multi-path reasoning

    # Domain knowledge (Layer 3)
    enable_domain_corpus: bool = False     # Domain-specific knowledge
    enable_embedding_aggregator: bool = False  # Domain routing

    # Enhanced patterns (Layer 3)
    enable_pre_act_planning: bool = False  # Multi-step planning
    enable_stuck_detection: bool = False   # Loop recovery
    enable_parallel_execution: bool = False  # Concurrent searches
    enable_contradiction_detection: bool = False  # Surface conflicts

    # Vision/Deep analysis (Layer 3)
    enable_vision_analysis: bool = False   # Vision-language analysis
    enable_deep_reading: bool = False      # Deep document analysis

    # Dynamic planning (Layer 4)
    enable_dynamic_planning: bool = False  # AIME-style task decomposition
    enable_progress_tracking: bool = False  # Real-time progress

    # Multi-agent (Layer 4)
    enable_actor_factory: bool = False     # Dynamic agent creation
    enable_multi_agent: bool = False       # Parallel agent execution

    # Graph cache (Layer 4)
    enable_graph_cache: bool = False       # Agent step graph
    enable_prefetching: bool = False       # Proactive prefetching

    # Metrics (always available)
    enable_metrics: bool = True

    # Debug mode (Layer 4 - for development/troubleshooting)
    enable_llm_debug: bool = False  # Emit detailed LLM call events


# Preset configurations
PRESET_CONFIGS = {
    OrchestratorPreset.MINIMAL: FeatureConfig(
        enable_self_reflection=False,
        enable_crag_evaluation=False,
        enable_sufficient_context=False,
        enable_positional_optimization=False,
        enable_experience_distillation=False,
        enable_classifier_feedback=False,
        enable_adaptive_refinement=False,  # Disabled in minimal
        enable_answer_grading=False,
        enable_gap_detection=False,
        enable_semantic_cache=False,
        enable_ttl_pinning=False,
        enable_metrics=False
    ),
    OrchestratorPreset.BALANCED: FeatureConfig(
        # All Layer 1 defaults are True
        # Layer 2+ features disabled for balance
        enable_hyde=False,
        enable_hybrid_reranking=False,
        enable_ragas=False
    ),
    OrchestratorPreset.ENHANCED: FeatureConfig(
        # Layer 2 quality features
        enable_hyde=True,
        enable_hybrid_reranking=True,
        enable_ragas=True,
        enable_context_curation=True,  # DIG-based context filtering
        context_curation_preset="balanced",
        enable_mixed_precision=True,
        enable_entity_enhanced_retrieval=True,
        # Layer 3 reasoning features
        enable_entity_tracking=True,
        enable_thought_library=True,
        enable_domain_corpus=True,
        enable_embedding_aggregator=True,
        enable_deep_reading=True
    ),
    OrchestratorPreset.RESEARCH: FeatureConfig(
        # All enhanced features
        enable_hyde=True,
        enable_hybrid_reranking=True,
        enable_ragas=True,
        enable_context_curation=True,  # DIG-based context filtering
        context_curation_preset="thorough",  # Thorough for research
        # Phase 2: Confidence-Calibrated Halting
        enable_entropy_halting=True,
        enable_iteration_bandit=True,
        enable_self_consistency=False,  # Expensive, disabled by default
        # Phase 3: Enhanced Query Generation
        enable_flare_retrieval=True,
        enable_query_tree=True,
        # Phase 4: Scratchpad Enhancement
        enable_semantic_memory=True,
        enable_raise_structure=True,
        # Phase 5: Template Reuse Optimization
        enable_meta_buffer=True,
        enable_reasoning_composer=True,
        enable_mixed_precision=True,
        enable_entity_enhanced_retrieval=True,
        enable_entity_tracking=True,
        enable_thought_library=True,
        enable_reasoning_dag=True,
        enable_domain_corpus=True,
        enable_embedding_aggregator=True,
        # Enhanced patterns
        enable_pre_act_planning=True,
        enable_stuck_detection=True,
        enable_parallel_execution=True,
        enable_contradiction_detection=True,
        # Vision/Deep analysis
        enable_vision_analysis=True,
        enable_deep_reading=True,
        # Performance
        enable_kv_cache_service=True,
        enable_artifacts=True,
        # Dynamic planning (AIME-style hierarchical task decomposition)
        enable_dynamic_planning=True,
        enable_progress_tracking=True,
        # Graph cache (workflow-aware KV cache optimization)
        enable_graph_cache=True,
        enable_prefetching=True
    ),
    OrchestratorPreset.FULL: FeatureConfig(
        # ALL features enabled
        # Layer 2 - Performance
        enable_kv_cache_service=True,
        enable_memory_tiers=True,
        enable_artifacts=True,
        # Layer 2 - Enhanced retrieval
        enable_hyde=True,
        enable_hybrid_reranking=True,
        enable_mixed_precision=True,
        enable_entity_enhanced_retrieval=True,
        # Layer 2 - Quality
        enable_ragas=True,
        enable_context_curation=True,  # DIG-based context filtering
        context_curation_preset="technical",  # Technical for full precision
        # Phase 2: Confidence-Calibrated Halting
        enable_entropy_halting=True,
        enable_iteration_bandit=True,
        enable_self_consistency=True,  # Full enables all
        # Phase 3: Enhanced Query Generation
        enable_flare_retrieval=True,
        enable_query_tree=True,
        # Phase 4: Scratchpad Enhancement
        enable_semantic_memory=True,
        enable_raise_structure=True,
        # Phase 5: Template Reuse Optimization
        enable_meta_buffer=True,
        enable_reasoning_composer=True,
        # Layer 3 - Advanced reasoning
        enable_entity_tracking=True,
        enable_thought_library=True,
        enable_reasoning_dag=True,
        # Layer 3 - Domain knowledge
        enable_domain_corpus=True,
        enable_embedding_aggregator=True,
        # Layer 3 - Enhanced patterns
        enable_pre_act_planning=True,
        enable_stuck_detection=True,
        enable_parallel_execution=True,
        enable_contradiction_detection=True,
        # Layer 3 - Vision/Deep analysis
        enable_vision_analysis=True,
        enable_deep_reading=True,
        # Layer 4 - Dynamic planning
        enable_dynamic_planning=True,
        enable_progress_tracking=True,
        # Layer 4 - Multi-agent
        enable_actor_factory=True,
        enable_multi_agent=True,
        # Layer 4 - Graph cache
        enable_graph_cache=True,
        enable_prefetching=True,
        # Layer 4 - Debug
        enable_llm_debug=True
    )
}


class UniversalOrchestrator(BaseSearchPipeline):
    """
    Universal orchestrator consolidating all agentic search functionality.

    This is the recommended entry point for all search operations, replacing:
    - AgenticOrchestrator
    - EnhancedAgenticOrchestrator
    - DynamicOrchestrator
    - GraphEnhancedOrchestrator
    - UnifiedOrchestrator
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        mcp_url: str = "http://localhost:7777",
        brave_api_key: Optional[str] = None,
        memory_service: Optional[Any] = None,
        config: Optional[FeatureConfig] = None,
        preset: Optional[OrchestratorPreset] = None,
        db_path: Optional[str] = None,
        **feature_overrides
    ):
        """
        Initialize the universal orchestrator.

        Args:
            ollama_url: Ollama API URL
            mcp_url: MCP Node Editor URL
            brave_api_key: Brave Search API key
            memory_service: Optional memory service
            config: Explicit FeatureConfig (takes precedence over preset)
            preset: Preset configuration (MINIMAL, BALANCED, ENHANCED, RESEARCH, FULL)
            db_path: Database path for persistent components
            **feature_overrides: Override specific feature flags
        """
        super().__init__(ollama_url, mcp_url, brave_api_key, memory_service)

        # Determine configuration
        if config is not None:
            self.config = config
        elif preset is not None:
            self.config = PRESET_CONFIGS.get(preset, PRESET_CONFIGS[OrchestratorPreset.BALANCED])
        else:
            # Default to balanced
            self.config = PRESET_CONFIGS[OrchestratorPreset.BALANCED]

        # Apply any feature overrides
        for key, value in feature_overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self.db_path = db_path or "/home/sparkone/sdd/Recovery_Bot/memOS/data"

        # Core components (always initialized)
        self.classifier = QueryClassifier(ollama_url=ollama_url)
        self.reflector = get_self_reflection_agent()
        self.retrieval_evaluator = RetrievalEvaluator(ollama_url=ollama_url)
        self.experience_distiller = get_experience_distiller()
        self.classifier_feedback = get_classifier_feedback()
        self.context_classifier = get_sufficient_context_classifier()

        # Adaptive refinement engine (Phase 2)
        self.adaptive_refinement = get_adaptive_refinement_engine(
            ollama_url=ollama_url,
            min_confidence_threshold=self.config.min_confidence_threshold,
            max_refinement_attempts=self.config.max_refinement_attempts
        )

        # Lazy-loaded components (initialized on demand based on config)
        self._hyde_expander: Optional[HyDEExpander] = None
        self._hybrid_retriever: Optional[BGEM3HybridRetriever] = None
        self._ragas_evaluator: Optional[RAGASEvaluator] = None
        self._entity_tracker: Optional[EntityTracker] = None
        self._thought_library: Optional[ThoughtLibrary] = None
        self._embedding_aggregator: Optional[EmbeddingAggregator] = None
        self._domain_corpus_manager: Optional[DomainCorpusManager] = None
        self._reasoning_dag: Optional[ReasoningDAG] = None
        self._metrics: Optional[PerformanceMetrics] = None
        self._reasoning_engine: Optional[EnhancedReasoningEngine] = None
        self._dynamic_planner: Optional[DynamicPlanner] = None
        self._progress_aggregator: Optional[ProgressAggregator] = None
        self._graph_cache: Optional[GraphCacheIntegration] = None
        self._agent_graph = None
        self._scratchpad_cache = None

        # Additional feature instances
        self._mixed_precision_service: Optional[MixedPrecisionEmbeddingService] = None
        self._entity_enhanced_retriever: Optional[EntityEnhancedRetriever] = None
        self._actor_factory: Optional[ActorFactory] = None
        self._multi_agent_orchestrator: Optional[MultiAgentOrchestrator] = None
        self._vision_analyzer: Optional[VisionAnalyzer] = None
        self._deep_reader: Optional[DeepReader] = None
        self._kv_cache_service: Optional[KVCacheService] = None
        self._memory_tier_manager: Optional[MemoryTierManager] = None
        self._artifact_store: Optional[ArtifactStore] = None
        self._ttl_manager = None
        self._stuck_metrics: Optional[StuckStateMetrics] = None
        self._context_curator: Optional[ContextCurator] = None

        # Phase 2: Confidence-Calibrated Halting components
        self._entropy_monitor: Optional[EntropyMonitor] = None
        self._self_consistency_checker: Optional[SelfConsistencyChecker] = None
        self._iteration_bandit: Optional[IterationBandit] = None

        # Phase 3: Enhanced Query Generation components
        self._flare_retriever: Optional[FLARERetriever] = None
        self._query_tree_decoder: Optional[QueryTreeDecoder] = None

        # Phase 4: Scratchpad Enhancement components
        self._semantic_memory: Optional[SemanticMemoryNetwork] = None
        self._raise_scratchpad: Optional[RAISEScratchpad] = None

        # Phase 5: Template Reuse Optimization components
        self._meta_buffer: Optional[MetaBuffer] = None
        self._reasoning_composer: Optional[ReasoningComposer] = None

        # Graph visualization state for SSE events
        self._graph_state = UniversalGraphState()

        # Statistics
        self._stats = {
            "total_searches": 0,
            "feature_usage": {},
            "avg_latency_ms": 0.0,
            "cache_hits": 0
        }

        # Feature timings for performance analysis
        self._feature_timings: Dict[str, List[float]] = {}

    # ===== Factory Methods (Presets) =====

    @classmethod
    def minimal(cls, **kwargs) -> "UniversalOrchestrator":
        """Create minimal orchestrator - fast, basic search."""
        return cls(preset=OrchestratorPreset.MINIMAL, **kwargs)

    @classmethod
    def balanced(cls, **kwargs) -> "UniversalOrchestrator":
        """Create balanced orchestrator - good quality/speed trade-off."""
        return cls(preset=OrchestratorPreset.BALANCED, **kwargs)

    @classmethod
    def enhanced(cls, **kwargs) -> "UniversalOrchestrator":
        """Create enhanced orchestrator - all quality features."""
        return cls(preset=OrchestratorPreset.ENHANCED, **kwargs)

    @classmethod
    def research(cls, **kwargs) -> "UniversalOrchestrator":
        """Create research orchestrator - thorough exploration."""
        return cls(preset=OrchestratorPreset.RESEARCH, **kwargs)

    @classmethod
    def full(cls, **kwargs) -> "UniversalOrchestrator":
        """Create full orchestrator - everything enabled."""
        return cls(preset=OrchestratorPreset.FULL, **kwargs)

    # ===== LLM Debug Tracking =====

    def _get_model_classification(self, model: str) -> str:
        """Classify model by its primary use case for debugging."""
        model_lower = model.lower()
        if "deepseek" in model_lower or "r1" in model_lower:
            return "reasoning"
        elif "vision" in model_lower or "vl" in model_lower:
            return "vision"
        elif "embed" in model_lower:
            return "embedding"
        elif "gemma" in model_lower and "4b" in model_lower:
            return "fast_evaluator"
        elif "qwen" in model_lower:
            return "general"
        else:
            return "unknown"

    def _get_model_context_window_size(self, model: str) -> int:
        """Get model context window size for utilization tracking."""
        model_lower = model.lower()
        if "deepseek" in model_lower:
            return 32768  # 32K for DeepSeek R1
        elif "qwen3:8b" in model_lower:
            return 32768  # 32K for qwen3:8b
        elif "gemma" in model_lower:
            return 8192  # 8K for gemma3:4b
        elif "llama" in model_lower:
            return 8192  # 8K for llama models
        else:
            return 16384  # Default 16K

    async def _emit_llm_start(
        self,
        emitter: EventEmitter,
        request_id: str,
        model: str,
        task: str,
        agent_phase: str,
        prompt: str = ""
    ) -> float:
        """
        Emit LLM call start event if debug mode enabled.
        Returns start timestamp for duration calculation.
        """
        start_time = time.time()
        if self.config.enable_llm_debug:
            classification = self._get_model_classification(model)
            context_window = self._get_model_context_window_size(model)
            # Estimate input tokens (roughly 4 chars per token)
            input_tokens = len(prompt) // 4 if prompt else 0

            await emitter.emit(llm_call_start(
                request_id=request_id,
                model=model,
                task=task,
                agent_phase=agent_phase,
                classification=classification,
                input_tokens=input_tokens,
                context_window=context_window,
                prompt_preview=prompt[:200] if prompt else ""
            ))
        return start_time

    async def _emit_llm_complete(
        self,
        emitter: EventEmitter,
        request_id: str,
        model: str,
        task: str,
        agent_phase: str,
        start_time: float,
        output: str = "",
        input_prompt: str = "",
        cache_hit: bool = False,
        thinking_tokens: int = 0
    ):
        """Emit LLM call complete event if debug mode enabled."""
        if self.config.enable_llm_debug:
            duration_ms = int((time.time() - start_time) * 1000)
            classification = self._get_model_classification(model)
            context_window = self._get_model_context_window_size(model)
            # Estimate tokens (roughly 4 chars per token)
            input_tokens = len(input_prompt) // 4 if input_prompt else 0
            output_tokens = len(output) // 4 if output else 0

            await emitter.emit(llm_call_complete(
                request_id=request_id,
                model=model,
                task=task,
                agent_phase=agent_phase,
                classification=classification,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_window=context_window,
                output_preview=output[:300] if output else "",
                cache_hit=cache_hit,
                thinking_tokens=thinking_tokens
            ))

    # ===== Lazy Component Initialization =====

    def _get_hyde_expander(self) -> HyDEExpander:
        """Lazy initialize HyDE expander."""
        if self._hyde_expander is None:
            self._hyde_expander = get_hyde_expander(self.ollama_url)
        return self._hyde_expander

    def _get_hybrid_retriever(self) -> BGEM3HybridRetriever:
        """Lazy initialize hybrid retriever."""
        if self._hybrid_retriever is None:
            self._hybrid_retriever = get_hybrid_retriever(self.ollama_url)
        return self._hybrid_retriever

    def _get_ragas_evaluator(self) -> RAGASEvaluator:
        """Lazy initialize RAGAS evaluator."""
        if self._ragas_evaluator is None:
            self._ragas_evaluator = get_ragas_evaluator(self.ollama_url)
        return self._ragas_evaluator

    def _get_entity_tracker(self) -> EntityTracker:
        """Lazy initialize entity tracker."""
        if self._entity_tracker is None:
            self._entity_tracker = create_entity_tracker(self.ollama_url)
        return self._entity_tracker

    def _get_thought_library(self) -> ThoughtLibrary:
        """Lazy initialize thought library."""
        if self._thought_library is None:
            self._thought_library = get_thought_library(self.ollama_url)
        return self._thought_library

    def _get_domain_corpus_manager(self) -> DomainCorpusManager:
        """Lazy initialize domain corpus manager."""
        if self._domain_corpus_manager is None:
            self._domain_corpus_manager = get_corpus_manager()
            # Initialize default corpuses (synchronous function)
            try:
                initialize_default_corpuses(ollama_url=self.ollama_url)
            except Exception as e:
                logger.warning(f"Failed to initialize default corpuses: {e}")
        return self._domain_corpus_manager

    def _get_reasoning_dag(self) -> ReasoningDAG:
        """Lazy initialize reasoning DAG."""
        if self._reasoning_dag is None:
            self._reasoning_dag = create_reasoning_dag(self.ollama_url)
        return self._reasoning_dag

    def _get_metrics(self) -> PerformanceMetrics:
        """Lazy initialize metrics."""
        if self._metrics is None:
            self._metrics = get_performance_metrics()
        return self._metrics

    def _get_reasoning_engine(self) -> EnhancedReasoningEngine:
        """Lazy initialize enhanced reasoning engine."""
        if self._reasoning_engine is None:
            self._reasoning_engine = get_enhanced_reasoning(self.ollama_url)
        return self._reasoning_engine

    def _get_graph_cache(self) -> GraphCacheIntegration:
        """Lazy initialize graph cache."""
        if self._graph_cache is None:
            self._graph_cache = get_graph_cache_integration(self.ollama_url)
            self._agent_graph = get_agent_step_graph()
            self._scratchpad_cache = get_scratchpad_cache()
        return self._graph_cache

    def _get_mixed_precision_service(self) -> MixedPrecisionEmbeddingService:
        """Lazy initialize mixed precision embedding service."""
        if self._mixed_precision_service is None:
            self._mixed_precision_service = get_mixed_precision_service(self.ollama_url)
        return self._mixed_precision_service

    def _get_entity_enhanced_retriever(self) -> EntityEnhancedRetriever:
        """Lazy initialize entity-enhanced retriever."""
        if self._entity_enhanced_retriever is None:
            self._entity_enhanced_retriever = get_entity_enhanced_retriever(self.ollama_url)
        return self._entity_enhanced_retriever

    def _get_actor_factory(self) -> ActorFactory:
        """Lazy initialize actor factory."""
        if self._actor_factory is None:
            self._actor_factory = get_actor_factory(self.ollama_url)
        return self._actor_factory

    def _get_multi_agent_orchestrator(self) -> MultiAgentOrchestrator:
        """Lazy initialize multi-agent orchestrator."""
        if self._multi_agent_orchestrator is None:
            self._multi_agent_orchestrator = MultiAgentOrchestrator(
                ollama_url=self.ollama_url
            )
        return self._multi_agent_orchestrator

    def _get_vision_analyzer(self) -> VisionAnalyzer:
        """Lazy initialize vision analyzer."""
        if self._vision_analyzer is None:
            self._vision_analyzer = VisionAnalyzer(ollama_url=self.ollama_url)
        return self._vision_analyzer

    def _get_deep_reader(self) -> DeepReader:
        """Lazy initialize deep reader."""
        if self._deep_reader is None:
            self._deep_reader = DeepReader(ollama_url=self.ollama_url)
        return self._deep_reader

    def _get_kv_cache_service(self) -> KVCacheService:
        """Lazy initialize KV cache service."""
        if self._kv_cache_service is None:
            self._kv_cache_service = get_kv_cache_service()  # Uses env vars internally
        return self._kv_cache_service

    def _get_memory_tier_manager(self) -> MemoryTierManager:
        """Lazy initialize memory tier manager."""
        if self._memory_tier_manager is None:
            self._memory_tier_manager = get_memory_tier_manager()
        return self._memory_tier_manager

    def _get_artifact_store(self) -> ArtifactStore:
        """Lazy initialize artifact store."""
        if self._artifact_store is None:
            self._artifact_store = get_artifact_store()
        return self._artifact_store

    def _get_ttl_manager(self):
        """Lazy initialize TTL cache manager."""
        if self._ttl_manager is None:
            self._ttl_manager = get_ttl_cache_manager()
        return self._ttl_manager

    def _get_dynamic_planner(self) -> DynamicPlanner:
        """Lazy initialize dynamic planner."""
        if self._dynamic_planner is None:
            self._dynamic_planner = DynamicPlanner(ollama_url=self.ollama_url)
        return self._dynamic_planner

    def _get_progress_aggregator(self) -> ProgressAggregator:
        """Lazy initialize progress aggregator."""
        if self._progress_aggregator is None:
            self._progress_aggregator = ProgressAggregator()
        return self._progress_aggregator

    def _get_embedding_aggregator(self) -> EmbeddingAggregator:
        """Lazy initialize embedding aggregator."""
        if self._embedding_aggregator is None:
            self._embedding_aggregator = get_embedding_aggregator(self.ollama_url)
        return self._embedding_aggregator

    def _get_context_curator(self) -> ContextCurator:
        """Lazy initialize context curator."""
        if self._context_curator is None:
            # Map preset string to CurationPreset enum
            preset_map = {
                "fast": CurationPreset.FAST,
                "balanced": CurationPreset.BALANCED,
                "thorough": CurationPreset.THOROUGH,
                "technical": CurationPreset.TECHNICAL,
            }
            preset = preset_map.get(
                self.config.context_curation_preset,
                CurationPreset.BALANCED
            )
            self._context_curator = get_context_curator(
                ollama_url=self.ollama_url,
                preset=preset
            )
        return self._context_curator

    def _get_entropy_monitor(self) -> EntropyMonitor:
        """Lazy initialize entropy monitor for confidence-calibrated halting."""
        if self._entropy_monitor is None:
            self._entropy_monitor = get_entropy_monitor(self.ollama_url)
        return self._entropy_monitor

    def _get_consistency_checker(self) -> SelfConsistencyChecker:
        """Lazy initialize self-consistency checker."""
        if self._self_consistency_checker is None:
            self._self_consistency_checker = get_consistency_checker(self.ollama_url)
        return self._self_consistency_checker

    def _get_iteration_bandit(self) -> IterationBandit:
        """Lazy initialize iteration bandit for UCB action selection."""
        if self._iteration_bandit is None:
            # Persistence path for learning across sessions
            persistence_path = f"{self.db_path}/iteration_bandit_stats.json"
            self._iteration_bandit = get_iteration_bandit(persistence_path)
        return self._iteration_bandit

    def _create_refinement_state(
        self,
        query: str,
        state,
        entropy: float = 0.5,
        coverage_score: float = 0.0,
        confidence: float = 0.5
    ) -> RefinementState:
        """Create a RefinementState for bandit decision-making."""
        unanswered = []
        if hasattr(state, 'search_plan') and state.search_plan:
            # Get unanswered from decomposed questions vs answered
            all_qs = getattr(state.search_plan, 'decomposed_questions', [])
            # Simplified: assume coverage ratio determines unanswered
            if coverage_score < 1.0 and all_qs:
                unanswered_ratio = 1.0 - coverage_score
                unanswered_count = int(len(all_qs) * unanswered_ratio)
                unanswered = all_qs[:unanswered_count]

        return RefinementState(
            query=query,
            iteration=state.iteration,
            max_iterations=state.max_iterations if hasattr(state, 'max_iterations') else 5,
            current_confidence=confidence,
            num_sources=state.sources_consulted if hasattr(state, 'sources_consulted') else 0,
            coverage_score=coverage_score,
            entropy=entropy,
            has_contradictions=False,  # TODO: integrate contradiction detection
            unanswered_questions=unanswered
        )

    # ===== Phase 3: Enhanced Query Generation =====

    def _get_flare_retriever(self) -> FLARERetriever:
        """Lazy initialize FLARE retriever for forward-looking active retrieval."""
        if self._flare_retriever is None:
            self._flare_retriever = get_flare_retriever(self.ollama_url)
        return self._flare_retriever

    def _get_query_tree_decoder(self) -> QueryTreeDecoder:
        """Lazy initialize query tree decoder for RQ-RAG tree decoding."""
        if self._query_tree_decoder is None:
            self._query_tree_decoder = get_query_tree_decoder(self.ollama_url)
        return self._query_tree_decoder

    async def _expand_queries_with_tree(
        self,
        query: str,
        retrieval_func=None
    ) -> List[str]:
        """
        Use RQ-RAG query tree to expand the query into variations.

        Returns list of expanded queries for parallel search.
        """
        if not self.config.enable_query_tree:
            return [query]

        try:
            decoder = self._get_query_tree_decoder()
            result = await decoder.tree_decode(query, retrieval_func)

            # Return all unique queries from the tree
            all_queries = [query]  # Original first
            for node_id, node in result.tree.nodes.items():
                if node.query != query and node.query not in all_queries:
                    all_queries.append(node.query)

            logger.debug(f"Query tree expanded {query!r} into {len(all_queries)} variations")
            return all_queries[:8]  # Limit to 8 total

        except Exception as e:
            logger.warning(f"Query tree expansion failed: {e}")
            return [query]

    async def _flare_enhanced_retrieval(
        self,
        query: str,
        partial_synthesis: str,
        context: List[str],
        retrieval_func=None
    ) -> List[str]:
        """
        Use FLARE for forward-looking retrieval during synthesis.

        If the partial synthesis shows uncertainty, retrieve more documents.
        """
        if not self.config.enable_flare_retrieval:
            return []

        try:
            flare = self._get_flare_retriever()
            additional_docs, retrieval_points = await flare.forward_looking_retrieve(
                query=query,
                partial_synthesis=partial_synthesis,
                context=context,
                retrieval_func=retrieval_func
            )

            if retrieval_points:
                logger.info(
                    f"FLARE retrieved {len(additional_docs)} docs at "
                    f"{len(retrieval_points)} uncertainty points"
                )

            return additional_docs

        except Exception as e:
            logger.warning(f"FLARE retrieval failed: {e}")
            return []

    # ===== Phase 4: Scratchpad Enhancement =====

    def _get_semantic_memory(self) -> SemanticMemoryNetwork:
        """Lazy initialize semantic memory network for A-MEM connections."""
        if self._semantic_memory is None:
            self._semantic_memory = get_semantic_memory(self.ollama_url)
        return self._semantic_memory

    def _get_raise_scratchpad(self, request_id: str, query: str) -> RAISEScratchpad:
        """Get or create a RAISE scratchpad for structured working memory."""
        if self._raise_scratchpad is None or self._raise_scratchpad.request_id != request_id:
            self._raise_scratchpad = create_raise_scratchpad(request_id, query)
        return self._raise_scratchpad

    async def _add_to_semantic_memory(
        self,
        content: str,
        memory_type: MemoryType,
        attributes: Dict[str, Any] = None,
        explicit_connections: List = None
    ):
        """
        Add content to the semantic memory network.

        Creates connections based on semantic similarity automatically.
        """
        if not self.config.enable_semantic_memory:
            return

        try:
            memory = self._get_semantic_memory()
            await memory.add_memory(
                content=content,
                memory_type=memory_type,
                attributes=attributes or {},
                explicit_connections=explicit_connections or []
            )
            logger.debug(f"Added to semantic memory: {memory_type.value}")
        except Exception as e:
            logger.warning(f"Failed to add to semantic memory: {e}")

    def _record_observation(
        self,
        request_id: str,
        query: str,
        content: str,
        observation_type: ObservationType,
        source: str,
        quality_score: float = 0.5
    ):
        """Record an observation in the RAISE scratchpad."""
        if not self.config.enable_raise_structure:
            return

        scratchpad = self._get_raise_scratchpad(request_id, query)
        scratchpad.add_observation(
            content=content,
            observation_type=observation_type,
            source=source,
            quality_score=quality_score
        )

    def _record_reasoning(
        self,
        request_id: str,
        query: str,
        conclusion: str,
        reasoning_type: ReasoningType,
        confidence: float = 0.5,
        agent: str = ""
    ):
        """Record a reasoning step in the RAISE scratchpad."""
        if not self.config.enable_raise_structure:
            return

        scratchpad = self._get_raise_scratchpad(request_id, query)
        scratchpad.add_reasoning_step(
            conclusion=conclusion,
            reasoning_type=reasoning_type,
            confidence=confidence,
            agent=agent
        )

    def _get_quality_signal(self, request_id: str, query: str):
        """Get quality signal from RAISE scratchpad."""
        if not self.config.enable_raise_structure:
            return None

        scratchpad = self._get_raise_scratchpad(request_id, query)
        return scratchpad.get_quality_signal()

    # ===== Phase 5: Template Reuse Optimization =====

    def _get_meta_buffer(self) -> MetaBuffer:
        """Lazy initialize meta-buffer for cross-session template persistence."""
        if self._meta_buffer is None:
            db_path = f"{self.db_path}/meta_buffer.db"
            self._meta_buffer = get_meta_buffer(
                ollama_url=self.ollama_url,
                db_path=db_path
            )
        return self._meta_buffer

    def _get_reasoning_composer(self) -> ReasoningComposer:
        """Lazy initialize reasoning composer for Self-Discover composition."""
        if self._reasoning_composer is None:
            self._reasoning_composer = get_reasoning_composer(
                ollama_url=self.ollama_url
            )
        return self._reasoning_composer

    async def _retrieve_template(self, query: str, template_type: TemplateType = None):
        """
        Retrieve a relevant template from the meta-buffer if available.

        Returns template and similarity score if found, None otherwise.
        """
        if not self.config.enable_meta_buffer:
            return None

        meta_buffer = self._get_meta_buffer()
        result = await meta_buffer.retrieve_template(
            query=query,
            template_type=template_type,
            min_success_rate=0.6
        )
        return result

    async def _distill_successful_search(
        self,
        query: str,
        decomposed_questions: List[str],
        search_queries: List[str],
        synthesis: str,
        sources: List[Dict],
        confidence: float,
        execution_time_ms: float
    ):
        """
        Distill a successful search into a reusable template.

        Only distills if confidence >= 0.75 (high quality result).
        """
        if not self.config.enable_meta_buffer:
            return None

        if confidence < 0.75:
            return None

        meta_buffer = self._get_meta_buffer()
        template = await meta_buffer.distill_from_search(
            query=query,
            decomposed_questions=decomposed_questions,
            search_queries=search_queries,
            synthesis=synthesis,
            sources=sources,
            confidence=confidence,
            execution_time_ms=execution_time_ms
        )
        return template

    async def _compose_reasoning_strategy(self, task: str, max_modules: int = 4):
        """
        Compose a task-specific reasoning strategy using Self-Discover.

        Returns a ComposedStrategy with selected and adapted reasoning modules.
        """
        if not self.config.enable_reasoning_composer:
            return None

        composer = self._get_reasoning_composer()
        strategy = await composer.compose_strategy(task, max_modules=max_modules)
        return strategy

    # ===== Core Search Method =====

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute search with all enabled features.

        This is the main entry point that orchestrates all enabled features
        in the correct order.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        self._stats["total_searches"] += 1

        logger.info(f"[{request_id}] Starting UNIVERSAL search: {request.query[:50]}...")
        logger.info(f"[{request_id}] Features: {self._get_enabled_features()}")

        # Track metrics if enabled
        if self.config.enable_metrics:
            metrics = self._get_metrics()
            metrics.start_query(request_id, request.query)

        # Check caches
        if self.config.enable_content_cache:
            cached = await self.check_cache(request)
            if cached:
                self._stats["cache_hits"] += 1
                return cached

        if self.config.enable_semantic_cache:
            semantic_cached = await self.check_semantic_cache(request, request_id)
            if semantic_cached:
                self._stats["cache_hits"] += 1
                return semantic_cached

        # Initialize graph cache workflow if enabled
        if self.config.enable_graph_cache:
            graph_cache = self._get_graph_cache()
            await graph_cache.start_workflow(request_id, request.query)

        try:
            # Execute search pipeline
            response = await self._execute_pipeline(request, request_id, start_time)

            # Store in cache
            if self.config.enable_content_cache:
                self.store_in_cache(request, response)

            # End graph cache workflow
            if self.config.enable_graph_cache:
                await graph_cache.end_workflow(request_id, success=response.success)

            # Track metrics
            if self.config.enable_metrics:
                execution_time = int((time.time() - start_time) * 1000)
                metrics.complete_query(request_id, response.data.confidence_score)

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Search failed: {e}", exc_info=True)

            if self.config.enable_graph_cache:
                await graph_cache.end_workflow(request_id, success=False)

            return self.build_error_response(
                str(e),
                request_id,
                int((time.time() - start_time) * 1000)
            )

    async def search_with_events(
        self,
        request: SearchRequest,
        emitter: EventEmitter
    ) -> SearchResponse:
        """
        Execute search with real-time SSE event emissions for Android streaming.

        This method wraps the standard search() method with event emissions
        at major pipeline phases, enabling real-time progress updates to clients.

        Args:
            request: SearchRequest with query and options
            emitter: EventEmitter for sending progress events

        Returns:
            SearchResponse with synthesized results
        """
        from . import events

        start_time = time.time()
        request_id = emitter.request_id
        self._stats["total_searches"] += 1

        logger.info(f"[{request_id}] Starting STREAMING search: {request.query[:50]}...")

        # Initialize graph state for visualization
        reset_graph_state()
        graph = get_graph_state()

        # Track metrics if enabled
        if self.config.enable_metrics:
            metrics = self._get_metrics()
            metrics.start_query(request_id, request.query)

        # Check caches first
        if self.config.enable_content_cache:
            cached = await self.check_cache(request)
            if cached:
                self._stats["cache_hits"] += 1
                await emitter.emit(events.progress_update(request_id, 100, "Returning cached result"))
                return cached

        if self.config.enable_semantic_cache:
            semantic_cached = await self.check_semantic_cache(request, request_id)
            if semantic_cached:
                self._stats["cache_hits"] += 1
                await emitter.emit(events.progress_update(request_id, 100, "Returning semantically cached result"))
                return semantic_cached

        # Initialize graph cache workflow if enabled
        graph_cache = None
        if self.config.enable_graph_cache:
            graph_cache = self._get_graph_cache()
            await graph_cache.start_workflow(request_id, request.query)

        try:
            # PHASE 1: Query Analysis
            await emitter.emit(graph_node_entered(request_id, "analyze", graph))
            await emitter.emit(events.analyzing_query(request_id, request.query))

            state = self.create_search_state(request)
            scratchpad = self.create_scratchpad(request, request_id)
            search_trace = []

            analyze_start = time.time()
            query_analysis = None
            if self.config.enable_query_analysis and request.analyze_query:
                # LLM Debug: Track query analysis call
                llm_start = await self._emit_llm_start(
                    emitter, request_id,
                    model=DEFAULT_PIPELINE_CONFIG.analyzer_model,
                    task="query_analysis",
                    agent_phase="PHASE_1_ANALYZE",
                    prompt=request.query
                )

                query_analysis = await self._phase_query_analysis(
                    request, state, search_trace, request_id
                )

                # LLM Debug: Complete query analysis tracking
                analysis_summary = ""
                if query_analysis:
                    analysis_summary = f"type={query_analysis.query_type}, requires_search={query_analysis.requires_search}"
                await self._emit_llm_complete(
                    emitter, request_id,
                    model=DEFAULT_PIPELINE_CONFIG.analyzer_model,
                    task="query_analysis",
                    agent_phase="PHASE_1_ANALYZE",
                    start_time=llm_start,
                    output=analysis_summary,
                    input_prompt=request.query
                )

                await emitter.emit(events.query_analyzed(
                    request_id,
                    query_analysis.requires_search if query_analysis else True,
                    query_analysis.query_type if query_analysis else "research"
                ))

            analyze_ms = int((time.time() - analyze_start) * 1000)
            await emitter.emit(graph_node_completed(request_id, "analyze", True, graph, analyze_ms))

            # PHASE 1.5: Entity Extraction (if enabled)
            if self.config.enable_entity_tracking and scratchpad:
                try:
                    entity_start = time.time()
                    tracker = self._get_entity_tracker()
                    entities = await tracker.extract_entities(request.query)
                    if entities:
                        for entity in entities:
                            scratchpad.add_entity(entity.to_dict())
                        await emitter.emit(events.entities_extracted(
                            request_id, len(entities),
                            [e.name for e in entities[:5]]  # First 5 entity names
                        ))

                        # Emit relations if any
                        relations = []
                        for entity in entities:
                            if hasattr(entity, 'relations') and entity.relations:
                                relations.extend(entity.relations)
                        if relations:
                            for rel in relations[:3]:  # First 3 relations
                                await emitter.emit(events.entity_relation_found(
                                    request_id,
                                    rel.get('source', 'unknown'),
                                    rel.get('target', 'unknown'),
                                    rel.get('relation_type', 'related_to')
                                ))

                        entity_ms = int((time.time() - entity_start) * 1000)
                        logger.info(f"[{request_id}] Extracted {len(entities)} entities in {entity_ms}ms")
                except Exception as e:
                    logger.debug(f"[{request_id}] Entity extraction failed: {e}")

            # PHASE 21: Meta-Buffer Template Retrieval & Reasoning Composition
            template_applied = False
            composed_strategy = None

            if self.config.enable_meta_buffer:
                try:
                    template_result = await self._retrieve_template(request.query)
                    if template_result:
                        template, similarity = template_result
                        logger.info(f"[{request_id}] Meta-Buffer: Found template (similarity={similarity:.2f})")
                        await emitter.emit(events.thought_template_matched(
                            request_id, template.id, similarity
                        ))
                        # Store template for potential use in search planning
                        state.retrieved_template = template
                        template_applied = True
                except Exception as e:
                    logger.debug(f"[{request_id}] Meta-Buffer template retrieval failed: {e}")

            if self.config.enable_reasoning_composer:
                try:
                    composed_strategy = await self._compose_reasoning_strategy(request.query)
                    if composed_strategy:
                        module_names = [m.name for m in composed_strategy.selected_modules] if hasattr(composed_strategy, 'selected_modules') else []
                        logger.info(f"[{request_id}] Reasoning Composer: Strategy composed with {len(module_names)} modules")
                        await emitter.emit(events.reasoning_strategy_composed(
                            request_id, len(module_names), module_names
                        ))
                        # Store strategy for use in synthesis
                        state.composed_reasoning_strategy = composed_strategy
                except Exception as e:
                    logger.debug(f"[{request_id}] Reasoning composition failed: {e}")

            # PHASE 2.1: HyDE Query Expansion (if enabled)
            expanded_query = request.query
            if self.config.enable_hyde:
                try:
                    await emitter.emit(events.hyde_generating(request_id, request.query))
                    expander = self._get_hyde_expander()
                    hyde_result = await expander.expand(request.query, mode=HyDEMode.SINGLE)
                    if hyde_result.hypothetical_documents:
                        expanded_query = hyde_result.hypothetical_documents[0]
                        await emitter.emit(events.hyde_complete(
                            request_id,
                            len(hyde_result.hypothetical_documents),
                            hyde_result.fused_embedding is not None
                        ))
                        logger.info(f"[{request_id}] HyDE expanded query: {expanded_query[:100]}...")
                except Exception as e:
                    logger.debug(f"[{request_id}] HyDE expansion failed: {e}")

            # PHASE 2.2: Reasoning DAG Initialization (if enabled)
            reasoning_dag = None
            if self.config.enable_reasoning_dag:
                try:
                    await emitter.emit(events.reasoning_branch_created(
                        request_id, "root", request.query, 0
                    ))
                    dag_start = time.time()
                    reasoning_dag = await self._phase_init_reasoning_dag(request, request_id)
                    dag_ms = int((time.time() - dag_start) * 1000)

                    if reasoning_dag:
                        node_count = len(reasoning_dag.nodes) if hasattr(reasoning_dag, 'nodes') else 1
                        await emitter.emit(events.reasoning_node_verified(
                            request_id, "root", True, 1.0
                        ))
                        logger.info(f"[{request_id}] Reasoning DAG initialized with {node_count} nodes in {dag_ms}ms")
                except Exception as e:
                    logger.debug(f"[{request_id}] Reasoning DAG initialization failed: {e}")

            # Handle direct answer if no search needed
            if query_analysis and not query_analysis.requires_search:
                await emitter.emit(events.synthesizing(request_id, 0))
                return await self._handle_direct_answer(
                    request, query_analysis, request_id, start_time
                )

            # PHASE 2: Search Planning
            await emitter.emit(graph_node_entered(request_id, "plan", graph))
            await emitter.emit(events.planning_search(request_id))

            plan_start = time.time()
            # Use analyzer to create search plan
            if query_analysis and hasattr(self, 'analyzer') and self.analyzer:
                state.search_plan = await self.analyzer.create_search_plan(
                    request.query, query_analysis, request.context,
                    request_id=request_id
                )
            else:
                # Fallback: create simple plan
                from .models import SearchPlan
                state.search_plan = SearchPlan(
                    original_query=request.query,
                    decomposed_questions=[request.query],
                    search_phases=[{"phase": "initial", "queries": [request.query]}],
                    priority_order=[0],
                    fallback_strategies=["broaden search"],
                    estimated_iterations=request.max_iterations,
                    reasoning="Direct query search"
                )

            initial_queries = []
            for phase in state.search_plan.search_phases:
                initial_queries.extend(phase.get("queries", []))
            state.add_pending_queries(initial_queries)

            plan_ms = int((time.time() - plan_start) * 1000)
            await emitter.emit(events.search_planned(
                request_id, initial_queries, len(state.search_plan.search_phases)
            ))
            await emitter.emit(graph_node_completed(request_id, "plan", True, graph, plan_ms))

            # Initialize scratchpad mission
            if self.config.enable_scratchpad:
                decomposed_qs = state.search_plan.decomposed_questions
                completion_criteria = {
                    f"q{i+1}": f"Find information about: {q}"
                    for i, q in enumerate(decomposed_qs)
                }
                scratchpad.set_mission(decomposed_qs, completion_criteria)
                await emitter.emit(events.scratchpad_initialized(request_id, decomposed_qs))

            # PHASE 3: Search Execution (ReAct Loop)
            for iteration in range(request.max_iterations):
                state.iteration = iteration + 1

                await emitter.emit(events.iteration_start_detailed(
                    request_id,
                    state.iteration,
                    request.max_iterations,
                    len(state.pending_queries),
                    state.sources_consulted
                ))

                if not state.pending_queries:
                    break

                # Execute searches
                await emitter.emit(graph_node_entered(request_id, "search", graph))
                queries_to_execute = state.pending_queries[:3]
                await emitter.emit(events.searching(
                    request_id, queries_to_execute, state.iteration, request.max_iterations
                ))

                search_start = time.time()
                await self._phase_search_execution(
                    request, state, scratchpad, search_trace, request_id,
                    pre_act_plan=None
                )
                search_ms = int((time.time() - search_start) * 1000)

                await emitter.emit(events.search_results(
                    request_id, len(state.raw_results), state.sources_consulted
                ))
                await emitter.emit(graph_node_completed(request_id, "search", True, graph, search_ms))

                # PHASE 3.5: Hybrid Re-ranking with BGE-M3 (if enabled)
                if self.config.enable_hybrid_reranking and state.raw_results:
                    try:
                        await emitter.emit(events.hybrid_search_start(
                            request_id, len(state.raw_results), "hybrid"
                        ))
                        hybrid_start = time.time()
                        retriever = self._get_hybrid_retriever()

                        # Index current results for hybrid search
                        for i, result in enumerate(state.raw_results[:20]):
                            content = getattr(result, 'snippet', '') or getattr(result, 'title', '')
                            if content:
                                await retriever.add_document(f"doc_{i}", content)

                        # Perform hybrid search to re-rank
                        hybrid_results = await retriever.search(
                            request.query, top_k=min(len(state.raw_results), 10)
                        )
                        hybrid_ms = int((time.time() - hybrid_start) * 1000)

                        await emitter.emit(events.hybrid_search_complete(
                            request_id, len(hybrid_results), hybrid_ms
                        ))
                        logger.info(f"[{request_id}] Hybrid re-ranking complete: {len(hybrid_results)} results in {hybrid_ms}ms")
                    except Exception as e:
                        logger.warning(f"[{request_id}] Hybrid re-ranking failed: {e}")

                # CRAG evaluation if enabled
                if self.config.enable_crag_evaluation:
                    await emitter.emit(graph_node_entered(request_id, "crag", graph))
                    await emitter.emit(events.crag_evaluating(request_id, len(state.raw_results)))

                    crag_start = time.time()

                    # LLM Debug: Track CRAG evaluation call
                    crag_prompt = f"CRAG pre-synthesis eval: {len(state.raw_results)} results"
                    llm_start = await self._emit_llm_start(
                        emitter, request_id,
                        model=DEFAULT_PIPELINE_CONFIG.evaluator_model,
                        task="crag_evaluation",
                        agent_phase="PHASE_3.5_CRAG",
                        prompt=crag_prompt
                    )

                    crag_result = await self._phase_crag_evaluation(
                        request, state, search_trace, request_id
                    )

                    # LLM Debug: Complete CRAG tracking
                    crag_output = ""
                    if crag_result:
                        crag_output = f"quality={crag_result.quality.value}, relevance={crag_result.relevance_score:.2f}"
                    await self._emit_llm_complete(
                        emitter, request_id,
                        model=DEFAULT_PIPELINE_CONFIG.evaluator_model,
                        task="crag_evaluation",
                        agent_phase="PHASE_3.5_CRAG",
                        start_time=llm_start,
                        output=crag_output,
                        input_prompt=crag_prompt
                    )

                    crag_ms = int((time.time() - crag_start) * 1000)

                    if crag_result:
                        await emitter.emit(events.crag_evaluation_complete(
                            request_id,
                            crag_result.quality.value,
                            crag_result.relevance_score,
                            crag_result.recommended_action.value if hasattr(crag_result.recommended_action, 'value') else str(crag_result.recommended_action)
                        ))
                        if crag_result.refined_queries:
                            await emitter.emit(events.crag_refining(request_id, crag_result.refined_queries))
                            state.add_pending_queries(crag_result.refined_queries)

                    await emitter.emit(graph_node_completed(request_id, "crag", True, graph, crag_ms))

                # Update iteration complete
                await emitter.emit(events.iteration_complete_detailed(
                    request_id, state.iteration,
                    len(state.raw_results), state.sources_consulted,
                    "continuing" if state.pending_queries else "complete"
                ))

            # PHASE 4: URL Scraping
            scraped_content = []
            if state.raw_results:
                await emitter.emit(graph_node_entered(request_id, "scrape", graph))

                urls_to_scrape = [r.url for r in state.raw_results[:request.max_urls_to_scrape] if hasattr(r, 'url') and r.url]
                await emitter.emit(events.evaluating_urls(request_id, len(urls_to_scrape)))

                scrape_start = time.time()
                scraped_content = await self._phase_content_scraping(
                    request, state, scratchpad, search_trace, request_id
                )
                scrape_ms = int((time.time() - scrape_start) * 1000)

                await emitter.emit(events.urls_evaluated(
                    request_id, len(scraped_content), len(urls_to_scrape)
                ))
                await emitter.emit(graph_node_completed(request_id, "scrape", True, graph, scrape_ms))

            # PHASE 4.5: Context Curation (DIG-based filtering and deduplication)
            if self.config.enable_context_curation and scraped_content:
                curation_start = time.time()
                try:
                    curator = self._get_context_curator()

                    # Convert scraped_content (List[str]) to document format for curation
                    documents = [
                        {"content": content, "id": f"doc_{i}"}
                        for i, content in enumerate(scraped_content)
                    ]

                    # Get decomposed questions from state if available
                    decomposed_questions = None
                    if hasattr(state, 'search_plan') and state.search_plan:
                        decomposed_questions = getattr(
                            state.search_plan, 'decomposed_questions', None
                        )

                    # Run context curation
                    curated = await curator.curate(
                        query=request.query,
                        documents=documents,
                        decomposed_questions=decomposed_questions
                    )

                    # Replace scraped_content with curated content
                    scraped_content = [doc.get("content", "") for doc in curated.documents]

                    curation_ms = int((time.time() - curation_start) * 1000)
                    logger.info(
                        f"[{request_id}] Context curation: "
                        f"{curated.original_count} → {curated.curated_count} docs "
                        f"({curated.reduction_ratio:.1%} reduction) in {curation_ms}ms"
                    )

                    # Record in search trace
                    search_trace.append({
                        "step": "context_curation",
                        "original_count": curated.original_count,
                        "curated_count": curated.curated_count,
                        "reduction_ratio": curated.reduction_ratio,
                        "avg_dig_score": curated.dig_summary.get("average_dig", 0.0) if curated.dig_summary else 0.0,
                        "coverage": curated.coverage.coverage_ratio if curated.coverage else None,
                        "duration_ms": curation_ms
                    })
                except Exception as e:
                    logger.warning(f"[{request_id}] Context curation failed: {e}")
                    # Continue with original scraped_content

            # PHASE 5: Verification
            if self.config.enable_verification and scraped_content:
                await emitter.emit(graph_node_entered(request_id, "verify", graph))
                await emitter.emit(events.verifying_claims(request_id, len(scraped_content)))

                verify_start = time.time()

                # LLM Debug: Track verification call
                verify_prompt = f"Verifying {len(scraped_content)} sources"
                llm_start = await self._emit_llm_start(
                    emitter, request_id,
                    model=DEFAULT_PIPELINE_CONFIG.verifier_model,
                    task="claim_verification",
                    agent_phase="PHASE_5_VERIFY",
                    prompt=verify_prompt
                )

                verification_result = await self._phase_verification(state, scraped_content, search_trace, request_id)

                # LLM Debug: Complete verification tracking
                verify_output = ""
                if verification_result:
                    verified = getattr(verification_result, 'verified_count', 0)
                    total = getattr(verification_result, 'total_claims', 0)
                    verify_output = f"verified={verified}/{total}"
                await self._emit_llm_complete(
                    emitter, request_id,
                    model=DEFAULT_PIPELINE_CONFIG.verifier_model,
                    task="claim_verification",
                    agent_phase="PHASE_5_VERIFY",
                    start_time=llm_start,
                    output=verify_output,
                    input_prompt=verify_prompt
                )

                verify_ms = int((time.time() - verify_start) * 1000)

                if verification_result:
                    # AggregateVerification has verified_count and total_claims attributes
                    verified_count = getattr(verification_result, 'verified_count', 0)
                    total_claims = getattr(verification_result, 'total_claims', len(scraped_content))
                    await emitter.emit(events.claims_verified(request_id, verified_count, total_claims))

                await emitter.emit(graph_node_completed(request_id, "verify", True, graph, verify_ms))

            # PHASE 6: Synthesis
            await emitter.emit(graph_node_entered(request_id, "synthesize", graph))
            await emitter.emit(events.synthesizing(request_id, len(scraped_content)))

            synthesis_start = time.time()

            # LLM Debug: Track synthesis call (reasoning model)
            synthesis_prompt = f"Query: {request.query}\nSources: {len(scraped_content)}"
            llm_start = await self._emit_llm_start(
                emitter, request_id,
                model=DEFAULT_PIPELINE_CONFIG.synthesizer_model,
                task="synthesis",
                agent_phase="PHASE_6_SYNTHESIZE",
                prompt=synthesis_prompt
            )

            synthesis = await self._phase_synthesis(
                request, state, scraped_content, search_trace, request_id
            )

            # LLM Debug: Complete synthesis tracking
            await self._emit_llm_complete(
                emitter, request_id,
                model=DEFAULT_PIPELINE_CONFIG.synthesizer_model,
                task="synthesis",
                agent_phase="PHASE_6_SYNTHESIZE",
                start_time=llm_start,
                output=synthesis[:500] if synthesis else "",
                input_prompt=synthesis_prompt,
                thinking_tokens=500 if "deepseek" in DEFAULT_PIPELINE_CONFIG.synthesizer_model.lower() else 0
            )

            synthesis_ms = int((time.time() - synthesis_start) * 1000)

            # Calculate confidence using heuristic baseline
            # This provides meaningful scores even when evaluation features are disabled
            source_diversity = len(state.unique_domains) / max(request.max_sources, 1)
            sources = self._get_sources(state)

            # Use heuristic confidence as baseline (works without LLM evaluation)
            heuristic_conf = self.calculate_heuristic_confidence(
                sources=sources,
                synthesis=synthesis or "",
                query=request.query,
                max_sources=request.max_sources
            )

            # Simple confidence as fallback
            simple_conf = 0.5 + (0.3 * min(source_diversity, 1.0))

            # Use the better of heuristic or simple confidence
            confidence = max(heuristic_conf, simple_conf)

            await emitter.emit(events.synthesis_complete(
                request_id, len(synthesis) if synthesis else 0, confidence
            ))
            await emitter.emit(graph_node_completed(request_id, "synthesize", True, graph, synthesis_ms))

            # PHASE 7: Self-RAG Reflection (if enabled)
            reflection_result = None
            if self.config.enable_self_reflection and synthesis:
                await emitter.emit(graph_node_entered(request_id, "reflect", graph))
                await emitter.emit(events.self_rag_reflecting(request_id, len(synthesis)))

                reflect_start = time.time()

                # LLM Debug: Track self-reflection call (fast evaluator model)
                reflect_prompt = f"Query: {request.query[:100]}\nSynthesis length: {len(synthesis)}"
                llm_start = await self._emit_llm_start(
                    emitter, request_id,
                    model=DEFAULT_PIPELINE_CONFIG.evaluator_model,
                    task="self_reflection",
                    agent_phase="PHASE_7_REFLECT",
                    prompt=reflect_prompt
                )

                reflection_result = await self._phase_self_reflection(
                    request.query, synthesis, state, scraped_content, request_id
                )

                # LLM Debug: Complete reflection tracking
                reflect_output = ""
                if reflection_result:
                    reflect_output = f"relevance={reflection_result.relevance_score:.2f}, useful={reflection_result.usefulness_score:.2f}"
                await self._emit_llm_complete(
                    emitter, request_id,
                    model=DEFAULT_PIPELINE_CONFIG.evaluator_model,
                    task="self_reflection",
                    agent_phase="PHASE_7_REFLECT",
                    start_time=llm_start,
                    output=reflect_output,
                    input_prompt=reflect_prompt
                )

                reflect_ms = int((time.time() - reflect_start) * 1000)

                if reflection_result:
                    # Convert enum to string for JSON serialization
                    support_level = reflection_result.support_level
                    if hasattr(support_level, 'value'):
                        support_level = support_level.value
                    await emitter.emit(events.self_rag_complete(
                        request_id,
                        reflection_result.relevance_score,
                        str(support_level),
                        reflection_result.usefulness_score,
                        reflection_result.temporal_conflicts
                    ))

                    # Blend confidence with reflection
                    reflection_conf = reflection_result.overall_confidence if hasattr(reflection_result, 'overall_confidence') else reflection_result.relevance_score
                    if reflection_conf > 0:
                        confidence = self.calculate_blended_confidence(
                            confidence, reflection_conf, None, source_diversity
                        )

                await emitter.emit(graph_node_completed(request_id, "reflect", True, graph, reflect_ms))

            # PHASE 7.2: RAGAS Evaluation (if enabled)
            ragas_result = None
            if self.config.enable_ragas and synthesis:
                try:
                    await emitter.emit(events.ragas_evaluating(request_id, len(scraped_content)))
                    ragas_start = time.time()

                    # LLM Debug: Track RAGAS evaluation call
                    ragas_prompt = f"RAGAS eval: {len(scraped_content)} contexts"
                    llm_start = await self._emit_llm_start(
                        emitter, request_id,
                        model=DEFAULT_PIPELINE_CONFIG.evaluator_model,
                        task="ragas_evaluation",
                        agent_phase="PHASE_7.2_RAGAS",
                        prompt=ragas_prompt
                    )

                    ragas_result = await self._phase_ragas_evaluation(
                        request.query, synthesis, state, scraped_content, request_id
                    )

                    # LLM Debug: Complete RAGAS tracking
                    ragas_output = ""
                    if ragas_result:
                        ragas_output = f"faith={ragas_result.faithfulness:.2f}, relevancy={ragas_result.answer_relevancy:.2f}"
                    await self._emit_llm_complete(
                        emitter, request_id,
                        model=DEFAULT_PIPELINE_CONFIG.evaluator_model,
                        task="ragas_evaluation",
                        agent_phase="PHASE_7.2_RAGAS",
                        start_time=llm_start,
                        output=ragas_output,
                        input_prompt=ragas_prompt
                    )

                    ragas_ms = int((time.time() - ragas_start) * 1000)

                    if ragas_result:
                        await emitter.emit(events.ragas_evaluation_complete(
                            request_id,
                            ragas_result.faithfulness,
                            ragas_result.answer_relevancy,
                            ragas_result.overall_score
                        ))
                        logger.info(
                            f"[{request_id}] RAGAS evaluation complete: "
                            f"faith={ragas_result.faithfulness:.2f}, "
                            f"relevancy={ragas_result.answer_relevancy:.2f} in {ragas_ms}ms"
                        )
                        # Factor RAGAS into confidence
                        if ragas_result.overall_score > 0:
                            confidence = self.calculate_blended_confidence(
                                confidence, None, ragas_result.overall_score, source_diversity
                            )
                except Exception as e:
                    logger.warning(f"[{request_id}] RAGAS evaluation failed: {e}")

            # PHASE 7.5: Entropy-Based Halting Check (if enabled)
            entropy_result = None
            should_skip_refinement = False
            if self.config.enable_entropy_halting and synthesis:
                entropy_start = time.time()
                try:
                    entropy_monitor = self._get_entropy_monitor()
                    entropy_result = await entropy_monitor.calculate_entropy(
                        query=request.query,
                        synthesis=synthesis,
                        context=scraped_content,
                        iteration=state.iteration,
                        max_iterations=request.max_iterations,
                        session_id=request_id
                    )

                    entropy_ms = int((time.time() - entropy_start) * 1000)
                    logger.info(
                        f"[{request_id}] Entropy check: {entropy_result.current_entropy:.3f} "
                        f"(conf={entropy_result.confidence_score:.2f}) "
                        f"decision={entropy_result.decision.value} in {entropy_ms}ms"
                    )

                    # If entropy indicates high confidence, skip refinement
                    if entropy_result.decision in [HaltDecision.HALT_CONFIDENT, HaltDecision.HALT_CONVERGENCE]:
                        should_skip_refinement = True
                        # Boost confidence based on entropy analysis
                        confidence = max(confidence, entropy_result.confidence_score)
                        logger.info(
                            f"[{request_id}] Entropy-based halting: skipping refinement "
                            f"(decision={entropy_result.decision.value})"
                        )

                    # Record in search trace
                    search_trace.append({
                        "step": "entropy_halting",
                        "entropy": entropy_result.current_entropy,
                        "confidence_score": entropy_result.confidence_score,
                        "decision": entropy_result.decision.value,
                        "skip_refinement": should_skip_refinement,
                        "duration_ms": entropy_ms
                    })
                except Exception as e:
                    logger.warning(f"[{request_id}] Entropy monitoring failed: {e}")

            # PHASE 8: Adaptive Refinement Loop (if enabled, not skipped by entropy, and confidence below threshold)
            should_refine = (
                self.config.enable_adaptive_refinement and
                not should_skip_refinement and
                confidence < self.config.min_confidence_threshold
            )
            logger.info(f"[{request_id}] Adaptive refinement check: enabled={self.config.enable_adaptive_refinement}, "
                       f"confidence={confidence:.2%}, threshold={self.config.min_confidence_threshold:.2%}, "
                       f"entropy_skip={should_skip_refinement}, trigger={should_refine}")
            if should_refine:
                refine_loop_start = time.time()
                initial_confidence = confidence
                refinement_attempt = 0
                all_scraped_content = scraped_content.copy()  # Accumulate all content
                best_synthesis = synthesis
                best_confidence = confidence
                best_sources = sources.copy()

                await emitter.emit(events.adaptive_refinement_start(
                    request_id, confidence, self.config.min_confidence_threshold,
                    self.config.max_refinement_attempts
                ))

                decision = RefinementDecision.REFINE_QUERY  # Default decision

                # Iterative refinement loop
                while (refinement_attempt < self.config.max_refinement_attempts and
                       confidence < self.config.min_confidence_threshold):

                    refinement_attempt += 1
                    refine_iter_start = time.time()

                    logger.info(f"[{request_id}] Refinement attempt {refinement_attempt}/{self.config.max_refinement_attempts}, "
                               f"current confidence: {confidence:.2%}")

                    # Step 1: Identify gaps in synthesis
                    gap_analysis = None
                    if self.config.enable_gap_detection and synthesis:
                        gap_analysis = await self.adaptive_refinement.identify_gaps(
                            request.query, synthesis, sources
                        )
                        await emitter.emit(events.gaps_identified(
                            request_id, gap_analysis.gaps, gap_analysis.coverage_score
                        ))

                    # Step 2: Grade answer quality
                    answer_assessment = None
                    if self.config.enable_answer_grading and synthesis:
                        answer_assessment = await self.adaptive_refinement.grade_answer(
                            request.query, synthesis
                        )
                        await emitter.emit(events.answer_graded(
                            request_id,
                            answer_assessment.grade.value,
                            answer_assessment.score,
                            answer_assessment.gaps
                        ))

                    # Step 3: Decide refinement action (optionally via UCB bandit)
                    bandit_decision = None
                    if self.config.enable_iteration_bandit:
                        try:
                            bandit = self._get_iteration_bandit()
                            coverage_score = gap_analysis.coverage_score if gap_analysis else 0.5
                            entropy_val = entropy_result.current_entropy if entropy_result else 0.5

                            refinement_state = self._create_refinement_state(
                                query=request.query,
                                state=state,
                                entropy=entropy_val,
                                coverage_score=coverage_score,
                                confidence=confidence
                            )

                            bandit_decision = bandit.select_action(refinement_state)
                            logger.info(
                                f"[{request_id}] Bandit decision: {bandit_decision.action.value} "
                                f"(UCB={bandit_decision.ucb_score:.3f}, conf={bandit_decision.confidence:.2f})"
                            )

                            # Map bandit action to RefinementDecision
                            bandit_to_refinement = {
                                RefinementAction.SYNTHESIZE_NOW: RefinementDecision.COMPLETE,
                                RefinementAction.SEARCH_MORE: RefinementDecision.REFINE_QUERY,
                                RefinementAction.REFINE_QUERY: RefinementDecision.REFINE_QUERY,
                                RefinementAction.DECOMPOSE: RefinementDecision.DECOMPOSE,
                                RefinementAction.VERIFY_CLAIMS: RefinementDecision.COMPLETE,  # Maps to verify then stop
                                RefinementAction.BROADEN_SCOPE: RefinementDecision.WEB_FALLBACK,
                                RefinementAction.NARROW_FOCUS: RefinementDecision.REFINE_QUERY,
                            }
                            decision = bandit_to_refinement.get(
                                bandit_decision.action,
                                RefinementDecision.REFINE_QUERY
                            )
                        except Exception as e:
                            logger.warning(f"[{request_id}] Iteration bandit failed: {e}, falling back to default")
                            bandit_decision = None

                    # Fall back to standard adaptive refinement decision if bandit not used
                    if bandit_decision is None:
                        decision = self.adaptive_refinement.decide_refinement_action(
                            confidence=confidence,
                            source_count=len(sources),
                            query_complexity=state.query_analysis.estimated_complexity if state.query_analysis else "medium",
                            iteration=refinement_attempt,
                            gap_analysis=gap_analysis,
                            answer_assessment=answer_assessment
                        )

                    decision_source = "bandit" if bandit_decision else "adaptive"
                    await emitter.emit(events.adaptive_refinement_decision(
                        request_id, decision.value, confidence, refinement_attempt,
                        f"Source: {decision_source}, Gap count: {len(gap_analysis.gaps) if gap_analysis else 0}, "
                        f"Grade: {answer_assessment.grade.value if answer_assessment else 'N/A'}"
                    ))

                    # Step 4: Execute refinement based on decision
                    if decision == RefinementDecision.COMPLETE:
                        logger.info(f"[{request_id}] Refinement complete - confidence sufficient")
                        break

                    elif decision == RefinementDecision.ACCEPT_BEST:
                        logger.info(f"[{request_id}] Accepting best result after {refinement_attempt} attempts")
                        synthesis = best_synthesis
                        confidence = best_confidence
                        sources = best_sources
                        break

                    elif decision == RefinementDecision.REFINE_QUERY and gap_analysis and gap_analysis.gaps:
                        # Generate and execute gap-filling queries
                        refinement_queries = await self.adaptive_refinement.generate_refinement_queries(
                            request.query, gap_analysis.gaps, synthesis
                        )

                        if not refinement_queries:
                            logger.info(f"[{request_id}] No refinement queries generated, accepting current result")
                            break

                        await emitter.emit(events.refinement_queries_generated(
                            request_id, refinement_queries
                        ))

                        # Execute searches for refinement queries
                        await emitter.emit(events.iteration_start_detailed(
                            request_id, refinement_attempt, self.config.max_refinement_attempts,
                            len(refinement_queries), state.sources_consulted
                        ))

                        # Add refinement queries to pending and execute
                        state.pending_queries = refinement_queries
                        await self._phase_search_execution(
                            request, state, scratchpad, search_trace, request_id, pre_act_plan=None
                        )

                        await emitter.emit(events.search_results(
                            request_id, len(state.raw_results), state.sources_consulted
                        ))

                        # Scrape new URLs (only new ones not already scraped)
                        new_urls = [r.url for r in state.raw_results[-10:] if hasattr(r, 'url') and r.url]
                        scraped_urls = {getattr(r, 'url', '') for r in state.raw_results[:-10] if hasattr(r, 'url')}
                        urls_to_scrape = [u for u in new_urls if u not in scraped_urls][:5]

                        if urls_to_scrape:
                            await emitter.emit(events.evaluating_urls(request_id, len(urls_to_scrape)))
                            new_scraped = await self._phase_content_scraping(
                                request, state, scratchpad, search_trace, request_id
                            )
                            all_scraped_content.extend(new_scraped)
                            await emitter.emit(events.urls_evaluated(
                                request_id, len(new_scraped), len(urls_to_scrape)
                            ))

                        # Re-synthesize with all accumulated content
                        if all_scraped_content:
                            await emitter.emit(graph_node_entered(request_id, "synthesize", graph))
                            synth_start = time.time()
                            synthesis = await self._phase_synthesis(
                                request, state, all_scraped_content, search_trace, request_id
                            )
                            synth_ms = int((time.time() - synth_start) * 1000)
                            await emitter.emit(graph_node_completed(request_id, "synthesize", True, graph, synth_ms))

                            # Re-calculate confidence with new synthesis
                            sources = self._get_sources(state)
                            confidence = self.calculate_heuristic_confidence(sources, synthesis, request.query)

                            # Track best result
                            if confidence > best_confidence:
                                best_synthesis = synthesis
                                best_confidence = confidence
                                best_sources = sources.copy()

                            logger.info(f"[{request_id}] Refinement {refinement_attempt}: new confidence {confidence:.2%} "
                                       f"(best: {best_confidence:.2%})")

                    elif decision == RefinementDecision.WEB_FALLBACK:
                        # Trigger fresh web search with reformulated query
                        reformulated = f"{request.query} detailed technical information"
                        state.pending_queries = [reformulated]
                        await self._phase_search_execution(
                            request, state, scratchpad, search_trace, request_id, pre_act_plan=None
                        )
                        logger.info(f"[{request_id}] Web fallback executed")

                    elif decision == RefinementDecision.DECOMPOSE:
                        # Decompose into sub-questions and search each
                        sub_questions = await self.adaptive_refinement.decompose_query(request.query)
                        if sub_questions:
                            state.pending_queries = sub_questions[:3]
                            await self._phase_search_execution(
                                request, state, scratchpad, search_trace, request_id, pre_act_plan=None
                            )
                            logger.info(f"[{request_id}] Query decomposed into {len(sub_questions)} sub-questions")

                    search_trace.append({
                        "step": f"adaptive_refinement_{refinement_attempt}",
                        "decision": decision.value,
                        "gaps": gap_analysis.gaps if gap_analysis else [],
                        "confidence_before": initial_confidence if refinement_attempt == 1 else confidence,
                        "confidence_after": confidence
                    })

                    refine_iter_ms = int((time.time() - refine_iter_start) * 1000)
                    logger.debug(f"[{request_id}] Refinement iteration {refinement_attempt} took {refine_iter_ms}ms")

                # Use best result if current is worse
                if best_confidence > confidence:
                    synthesis = best_synthesis
                    confidence = best_confidence
                    sources = best_sources

                refine_total_ms = int((time.time() - refine_loop_start) * 1000)
                await emitter.emit(events.adaptive_refinement_complete(
                    request_id, decision.value,
                    initial_confidence, confidence, refinement_attempt, refine_total_ms
                ))

                logger.info(f"[{request_id}] Adaptive refinement complete: {initial_confidence:.2%} → {confidence:.2%} "
                           f"in {refinement_attempt} attempts ({refine_total_ms}ms)")

                # Record bandit outcome for learning (if bandit was used)
                if self.config.enable_iteration_bandit and bandit_decision:
                    try:
                        bandit = self._get_iteration_bandit()
                        # Calculate reward based on confidence improvement
                        improvement = confidence - initial_confidence
                        # Reward is normalized: 0.5 = no change, 1.0 = large improvement, 0.0 = degraded
                        reward = min(1.0, max(0.0, 0.5 + improvement))
                        # Boost reward if we reached threshold
                        if confidence >= self.config.min_confidence_threshold:
                            reward = min(1.0, reward + 0.2)

                        bandit.record_outcome(
                            action=bandit_decision.action,
                            reward=reward,
                            state=refinement_state if 'refinement_state' in dir() else None,
                            context={
                                "initial_confidence": initial_confidence,
                                "final_confidence": confidence,
                                "iterations": refinement_attempt,
                                "duration_ms": refine_total_ms
                            }
                        )
                        logger.debug(f"[{request_id}] Bandit outcome recorded: action={bandit_decision.action.value}, reward={reward:.2f}")
                    except Exception as e:
                        logger.warning(f"[{request_id}] Failed to record bandit outcome: {e}")

            # Build final response
            execution_time_ms = int((time.time() - start_time) * 1000)

            response = self.build_response(
                synthesis=synthesis or "",
                sources=self._get_sources(state),
                queries=state.executed_queries,
                confidence=confidence,
                state=state,
                request_id=request_id,
                search_trace=search_trace,
                execution_time_ms=execution_time_ms
            )

            # Store in cache
            if self.config.enable_content_cache:
                self.store_in_cache(request, response)

            # End graph cache workflow
            if graph_cache:
                await graph_cache.end_workflow(request_id, success=response.success)

            # Track metrics
            if self.config.enable_metrics:
                metrics.complete_query(request_id, confidence)

            # Experience distillation (if enabled)
            if self.config.enable_experience_distillation and confidence >= 0.75:
                try:
                    await self._phase_experience_distillation(
                        request, synthesis or "", confidence, state, request_id
                    )
                    await emitter.emit(events.experience_captured(
                        request_id,
                        state.query_analysis.query_type if state.query_analysis else "research",
                        confidence
                    ))
                except Exception as e:
                    logger.debug(f"Experience capture failed: {e}")

            # Meta-Buffer template distillation (Phase 21)
            if self.config.enable_meta_buffer and confidence >= 0.75:
                try:
                    # Gather search queries from state
                    search_queries = state.executed_queries if hasattr(state, 'executed_queries') else []
                    decomposed = state.search_plan.decomposed_questions if state.search_plan else [request.query]

                    template = await self._distill_successful_search(
                        query=request.query,
                        decomposed_questions=decomposed,
                        search_queries=search_queries,
                        synthesis=synthesis or "",
                        sources=self._get_sources(state),
                        confidence=confidence,
                        execution_time_ms=execution_time_ms
                    )
                    if template:
                        logger.info(f"[{request_id}] Meta-Buffer: Template distilled (id={template.id})")
                        await emitter.emit(events.template_created(request_id, template.id))
                except Exception as e:
                    logger.debug(f"[{request_id}] Meta-Buffer template distillation failed: {e}")

            # Final graph complete
            await emitter.emit(graph_node_completed(request_id, "complete", True, graph, execution_time_ms))

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Streaming search failed: {e}", exc_info=True)

            if graph_cache:
                await graph_cache.end_workflow(request_id, success=False)

            return self.build_error_response(
                str(e),
                request_id,
                int((time.time() - start_time) * 1000)
            )

    async def _execute_pipeline(
        self,
        request: SearchRequest,
        request_id: str,
        start_time: float
    ) -> SearchResponse:
        """Execute the main search pipeline with all enabled features."""

        # Initialize state
        state = self.create_search_state(request)
        scratchpad = self.create_scratchpad(request, request_id)
        search_trace = []
        enhancement_metadata = {"features_used": []}

        # Apply dynamic context limits based on model's context window
        context_info = self._apply_dynamic_context_limits(request, request_id)
        enhancement_metadata["context_limits"] = context_info

        # PHASE 0: Initialize TTL pinning for long-running operations
        ttl_context = None
        if self.config.enable_ttl_pinning:
            try:
                ttl_manager = self._get_ttl_manager()
                # ToolCallContext signature: program_id, tool_type, cache_id=None, manager=None
                ttl_context = ToolCallContext(
                    program_id=request_id,
                    tool_type=ToolType.OLLAMA_GENERATE,  # Synthesis uses LLM generation
                    manager=ttl_manager
                )
                # Pin using manager's pin_for_tool method instead of non-existent pin_cache
                ttl_manager.pin_for_tool(request_id, ToolType.OLLAMA_GENERATE)
                enhancement_metadata["features_used"].append("ttl_pinning")
            except Exception as e:
                logger.warning(f"[{request_id}] TTL pinning failed: {e}")

        # PHASE 0.5: Warm KV cache with common prefixes
        if self.config.enable_kv_cache_service:
            try:
                kv_service = self._get_kv_cache_service()
                await kv_service.warm_prefix(SYSTEM_PREFIX[:500])
                enhancement_metadata["features_used"].append("kv_cache_service")
            except Exception as e:
                logger.warning(f"[{request_id}] KV cache warming failed: {e}")

        # PHASE 1: Query Understanding
        query_analysis = None
        if self.config.enable_query_analysis and request.analyze_query:
            query_analysis = await self._phase_query_analysis(
                request, state, search_trace, request_id
            )
            if query_analysis and not query_analysis.requires_search:
                # Direct answer, no search needed
                return await self._handle_direct_answer(
                    request, query_analysis, request_id, start_time
                )

        # PHASE 1.5: Entity Extraction (if enabled)
        if self.config.enable_entity_tracking:
            await self._phase_entity_extraction(request, scratchpad, request_id)
            enhancement_metadata["features_used"].append("entity_tracking")

        # PHASE 1.6: Dynamic Planning (AIME-style task decomposition)
        dynamic_plan = None
        if self.config.enable_dynamic_planning:
            dynamic_plan = await self._phase_dynamic_planning(request, request_id)
            enhancement_metadata["features_used"].append("dynamic_planning")
            if self.config.enable_progress_tracking and dynamic_plan:
                progress = self._get_progress_aggregator()
                progress.start_tracking(request_id, len(dynamic_plan.task_tree) if hasattr(dynamic_plan, 'task_tree') else 3)
                enhancement_metadata["features_used"].append("progress_tracking")

        # PHASE 1.7: Initialize Reasoning DAG for complex queries
        reasoning_dag = None
        if self.config.enable_reasoning_dag:
            reasoning_dag = await self._phase_init_reasoning_dag(request, request_id)
            enhancement_metadata["features_used"].append("reasoning_dag")

        # PHASE 2: Query Expansion (HyDE)
        expanded_query = request.query
        if self.config.enable_hyde:
            expanded_query = await self._phase_hyde_expansion(
                request.query, request_id
            )
            enhancement_metadata["features_used"].append("hyde")
            enhancement_metadata["hyde_expanded"] = expanded_query != request.query

        # PHASE 2.5: Thought Library (if enabled)
        thought_context = None
        if self.config.enable_thought_library:
            thought_context = await self._phase_thought_library(
                request.query, request_id
            )
            enhancement_metadata["features_used"].append("thought_library")

        # PHASE 2.6: Embedding Aggregator for domain routing
        aggregated_embedding = None
        if self.config.enable_embedding_aggregator:
            aggregated_embedding = await self._phase_embedding_aggregation(
                request.query, request_id
            )
            enhancement_metadata["features_used"].append("embedding_aggregator")

        # PHASE 3: Pre-Act Planning (if enabled)
        pre_act_plan = None
        if self.config.enable_pre_act_planning:
            pre_act_plan = await self._phase_pre_act_planning(
                request, search_trace, request_id
            )
            enhancement_metadata["features_used"].append("pre_act_planning")

        # PHASE 3.5: Multi-Agent Execution (if enabled)
        multi_agent_results = None
        if self.config.enable_multi_agent:
            multi_agent_results = await self._phase_multi_agent_execution(
                request, state, search_trace, request_id
            )
            enhancement_metadata["features_used"].append("multi_agent")
            if multi_agent_results:
                enhancement_metadata["agents_used"] = multi_agent_results.get("agents", [])

        # PHASE 4: Search Execution
        await self._phase_search_execution(
            request, state, scratchpad, search_trace, request_id,
            pre_act_plan=pre_act_plan
        )

        # PHASE 4.5: Domain Corpus Augmentation (if enabled)
        domain_context = None
        if self.config.enable_domain_corpus:
            domain_context = await self._phase_domain_corpus(
                request.query, request_id
            )
            enhancement_metadata["features_used"].append("domain_corpus")

        # PHASE 5: CRAG Evaluation (if enabled)
        if self.config.enable_crag_evaluation:
            await self._phase_crag_evaluation(
                request, state, search_trace, request_id
            )
            enhancement_metadata["features_used"].append("crag")

        # PHASE 6: Hybrid Re-ranking (if enabled)
        if self.config.enable_hybrid_reranking:
            await self._phase_hybrid_reranking(
                state, request_id
            )
            enhancement_metadata["features_used"].append("hybrid_reranking")

        # PHASE 6.5: Entity-Enhanced Retrieval
        if self.config.enable_entity_enhanced_retrieval:
            await self._phase_entity_enhanced_retrieval(
                request, state, request_id
            )
            enhancement_metadata["features_used"].append("entity_enhanced_retrieval")

        # PHASE 6.6: Mixed Precision Indexing (index search results)
        if self.config.enable_mixed_precision:
            await self._phase_mixed_precision_indexing(
                state, request_id
            )
            enhancement_metadata["features_used"].append("mixed_precision")

        # PHASE 7: Content Scraping
        scraped_content = await self._phase_content_scraping(
            request, state, scratchpad, search_trace, request_id
        )

        # PHASE 7.5: Deep Reading (extract detailed info from scraped content)
        if self.config.enable_deep_reading and scraped_content:
            deep_insights = await self._phase_deep_reading(
                scraped_content, request.query, request_id
            )
            if deep_insights:
                scraped_content.append(f"[Deep Analysis]\n{deep_insights}")
            enhancement_metadata["features_used"].append("deep_reading")

        # PHASE 7.6: Vision Analysis (analyze any images in results)
        if self.config.enable_vision_analysis:
            vision_insights = await self._phase_vision_analysis(
                state, request_id
            )
            if vision_insights:
                scraped_content.append(f"[Visual Analysis]\n{vision_insights}")
            enhancement_metadata["features_used"].append("vision_analysis")

        # PHASE 7.7: Stuck Detection (check for loops)
        is_stuck = False
        if self.config.enable_stuck_detection:
            is_stuck, recovery_action = await self._phase_stuck_detection(
                state, scratchpad, request_id
            )
            if is_stuck:
                enhancement_metadata["stuck_detected"] = True
                enhancement_metadata["recovery_action"] = recovery_action
                enhancement_metadata["features_used"].append("stuck_detection")
                # Apply recovery if stuck
                if recovery_action == "broaden":
                    expanded_results = await self.searcher.search([f"general {request.query}"])
                    state.add_results(expanded_results)

        # PHASE 7.8: Context Curation (DIG-based filtering and deduplication)
        if self.config.enable_context_curation and scraped_content:
            curation_start = time.time()
            try:
                curator = self._get_context_curator()

                # Convert scraped_content (List[str]) to document format for curation
                documents = [
                    {"content": content, "id": f"doc_{i}"}
                    for i, content in enumerate(scraped_content)
                ]

                # Get decomposed questions from state if available
                decomposed_questions = None
                if hasattr(state, 'search_plan') and state.search_plan:
                    decomposed_questions = getattr(
                        state.search_plan, 'decomposed_questions', None
                    )

                # Run context curation
                curated = await curator.curate(
                    query=request.query,
                    documents=documents,
                    decomposed_questions=decomposed_questions
                )

                # Replace scraped_content with curated content
                scraped_content = [doc.get("content", "") for doc in curated.documents]

                curation_ms = int((time.time() - curation_start) * 1000)
                logger.info(
                    f"[{request_id}] Context curation: "
                    f"{curated.original_count} → {curated.curated_count} docs "
                    f"({curated.reduction_ratio:.1%} reduction) in {curation_ms}ms"
                )
                enhancement_metadata["features_used"].append("context_curation")
                enhancement_metadata["curation_reduction"] = curated.reduction_ratio

                # Record in search trace
                search_trace.append({
                    "step": "context_curation",
                    "original_count": curated.original_count,
                    "curated_count": curated.curated_count,
                    "reduction_ratio": curated.reduction_ratio,
                    "avg_dig_score": curated.dig_summary.get("average_dig", 0.0) if curated.dig_summary else 0.0,
                    "coverage": curated.coverage.coverage_ratio if curated.coverage else None,
                    "duration_ms": curation_ms
                })
            except Exception as e:
                logger.warning(f"[{request_id}] Context curation failed: {e}")
                # Continue with original scraped_content

        # PHASE 8: Verification
        verification_result = None
        if self.config.enable_verification:
            verification_result = await self._phase_verification(
                state, scraped_content, search_trace, request_id
            )

        # PHASE 8.5: Positional Optimization (lost-in-the-middle mitigation)
        if self.config.enable_positional_optimization and scraped_content:
            scraped_content = await self._phase_positional_optimization(
                scraped_content, request.query, request_id
            )
            enhancement_metadata["features_used"].append("positional_optimization")

        # PHASE 9: Synthesis
        synthesis = await self._phase_synthesis(
            request, state, scraped_content, search_trace, request_id,
            thought_context=thought_context,
            domain_context=domain_context
        )

        # PHASE 9.5: Contradiction Detection
        contradictions = None
        if self.config.enable_contradiction_detection:
            contradictions = await self._phase_contradiction_detection(
                synthesis, scraped_content, request_id
            )
            if contradictions:
                enhancement_metadata["contradictions_found"] = len(contradictions)
                enhancement_metadata["features_used"].append("contradiction_detection")
                # Append contradiction summary to synthesis
                if contradictions:
                    synthesis += "\n\n**Note:** Some sources provide conflicting information:\n"
                    for c in contradictions[:3]:
                        synthesis += f"- {c}\n"

        # PHASE 10: Self-Reflection (if enabled)
        reflection_result = None
        if self.config.enable_self_reflection:
            reflection_result = await self._phase_self_reflection(
                request.query, synthesis, state, scraped_content, request_id
            )
            enhancement_metadata["features_used"].append("self_reflection")

            # Refine if needed
            if reflection_result and reflection_result.needs_refinement:
                synthesis = await self.reflector.refine_synthesis(
                    synthesis, reflection_result,
                    [{"snippet": s.get("snippet", "")} for s in self._get_sources(state)]
                )

        # PHASE 11: RAGAS Evaluation (if enabled)
        ragas_result = None
        if self.config.enable_ragas:
            ragas_result = await self._phase_ragas_evaluation(
                request.query, synthesis, state, scraped_content, request_id
            )
            enhancement_metadata["features_used"].append("ragas")
            enhancement_metadata["ragas_scores"] = {
                "faithfulness": ragas_result.faithfulness if ragas_result else None,
                "relevancy": ragas_result.answer_relevancy if ragas_result else None
            }

        # Calculate final confidence
        verifier_conf = verification_result.confidence if verification_result else 0.6
        reflection_conf = reflection_result.overall_confidence if reflection_result else None
        ragas_score = ragas_result.overall_score if ragas_result else None
        source_diversity = len(state.unique_domains) / max(request.max_sources, 1)

        # Calculate heuristic confidence as baseline (works without LLM evaluation)
        sources = self._get_sources(state)
        heuristic_conf = self.calculate_heuristic_confidence(
            sources=sources,
            synthesis=synthesis or "",
            query=request.query,
            max_sources=request.max_sources
        )

        # Use heuristic as floor for confidence when evaluation features are disabled
        if reflection_conf is None and ragas_score is None:
            # No evaluation features - use max of heuristic and verifier
            final_confidence = max(heuristic_conf, verifier_conf)
        else:
            # Evaluation features enabled - blend normally, but ensure floor
            blended = self.calculate_blended_confidence(
                verifier_conf, reflection_conf, ragas_score, source_diversity
            )
            final_confidence = max(heuristic_conf, blended)

        # PHASE 11.5: Adaptive Refinement Loop (if enabled and confidence below threshold)
        logger.info(f"[{request_id}] Adaptive refinement check: enabled={self.config.enable_adaptive_refinement}, "
                   f"confidence={final_confidence:.2%}, threshold={self.config.min_confidence_threshold:.2%}")

        if self.config.enable_adaptive_refinement and final_confidence < self.config.min_confidence_threshold:
            refine_loop_start = time.time()
            initial_confidence = final_confidence
            refinement_attempt = 0
            all_scraped_content = scraped_content.copy()
            best_synthesis = synthesis
            best_confidence = final_confidence
            best_sources = sources.copy()

            logger.info(f"[{request_id}] Starting adaptive refinement loop "
                       f"(confidence {final_confidence:.2%} < threshold {self.config.min_confidence_threshold:.2%})")

            decision = RefinementDecision.REFINE_QUERY  # Default

            while (refinement_attempt < self.config.max_refinement_attempts and
                   final_confidence < self.config.min_confidence_threshold):

                refinement_attempt += 1

                logger.info(f"[{request_id}] Refinement attempt {refinement_attempt}/{self.config.max_refinement_attempts}")

                # Identify gaps
                gap_analysis = None
                if self.config.enable_gap_detection and synthesis:
                    gap_analysis = await self.adaptive_refinement.identify_gaps(
                        request.query, synthesis, sources
                    )

                # Grade answer
                answer_assessment = None
                if self.config.enable_answer_grading and synthesis:
                    answer_assessment = await self.adaptive_refinement.grade_answer(
                        request.query, synthesis
                    )

                # Decide action
                decision = self.adaptive_refinement.decide_refinement_action(
                    confidence=final_confidence,
                    source_count=len(sources),
                    query_complexity=query_analysis.estimated_complexity if query_analysis else "medium",
                    iteration=refinement_attempt,
                    gap_analysis=gap_analysis,
                    answer_assessment=answer_assessment
                )

                logger.info(f"[{request_id}] Refinement decision: {decision.value}")

                if decision == RefinementDecision.COMPLETE:
                    break

                elif decision == RefinementDecision.ACCEPT_BEST:
                    synthesis = best_synthesis
                    final_confidence = best_confidence
                    sources = best_sources
                    break

                elif decision == RefinementDecision.REFINE_QUERY and gap_analysis and gap_analysis.gaps:
                    refinement_queries = await self.adaptive_refinement.generate_refinement_queries(
                        request.query, gap_analysis.gaps, synthesis
                    )

                    if not refinement_queries:
                        break

                    # Execute refinement searches
                    state.pending_queries = refinement_queries
                    await self._phase_search_execution(
                        request, state, scratchpad, search_trace, request_id, pre_act_plan=None
                    )

                    # Scrape new content
                    new_scraped = await self._phase_content_scraping(
                        request, state, scratchpad, search_trace, request_id
                    )
                    all_scraped_content.extend(new_scraped)

                    # Re-synthesize
                    synthesis = await self._phase_synthesis(
                        request, state, all_scraped_content, search_trace, request_id,
                        thought_context=thought_context, domain_context=domain_context
                    )

                    # Re-calculate confidence
                    sources = self._get_sources(state)
                    heuristic_conf = self.calculate_heuristic_confidence(
                        sources, synthesis or "", request.query, request.max_sources
                    )
                    final_confidence = max(heuristic_conf, verifier_conf)

                    if final_confidence > best_confidence:
                        best_synthesis = synthesis
                        best_confidence = final_confidence
                        best_sources = sources.copy()

                    logger.info(f"[{request_id}] Refinement {refinement_attempt}: "
                               f"confidence {final_confidence:.2%} (best: {best_confidence:.2%})")

                elif decision == RefinementDecision.WEB_FALLBACK:
                    state.pending_queries = [f"{request.query} detailed technical information"]
                    await self._phase_search_execution(
                        request, state, scratchpad, search_trace, request_id, pre_act_plan=None
                    )

                elif decision == RefinementDecision.DECOMPOSE:
                    sub_questions = await self.adaptive_refinement.decompose_query(request.query)
                    if sub_questions:
                        state.pending_queries = sub_questions[:3]
                        await self._phase_search_execution(
                            request, state, scratchpad, search_trace, request_id, pre_act_plan=None
                        )

                search_trace.append({
                    "step": f"adaptive_refinement_{refinement_attempt}",
                    "decision": decision.value,
                    "gaps": gap_analysis.gaps if gap_analysis else [],
                    "confidence_before": initial_confidence if refinement_attempt == 1 else final_confidence,
                    "confidence_after": final_confidence
                })

            # Use best result
            if best_confidence > final_confidence:
                synthesis = best_synthesis
                final_confidence = best_confidence
                sources = best_sources

            refine_total_ms = int((time.time() - refine_loop_start) * 1000)
            logger.info(f"[{request_id}] Adaptive refinement complete: {initial_confidence:.2%} → {final_confidence:.2%} "
                       f"in {refinement_attempt} attempts ({refine_total_ms}ms)")

            enhancement_metadata["adaptive_refinement"] = {
                "enabled": True,
                "attempts": refinement_attempt,
                "initial_confidence": initial_confidence,
                "final_confidence": final_confidence,
                "duration_ms": refine_total_ms
            }

        # PHASE 12: Learning (if enabled)
        if self.config.enable_experience_distillation and final_confidence >= 0.75:
            await self._phase_experience_distillation(
                request, synthesis, final_confidence, state, request_id
            )

        if self.config.enable_classifier_feedback and query_analysis:
            await self._phase_classifier_feedback(
                request, query_analysis, final_confidence, state, request_id, start_time
            )

        # PHASE 12.5: Extract Reasoning DAG conclusion
        if self.config.enable_reasoning_dag and reasoning_dag:
            dag_conclusion = await self._phase_reasoning_dag_conclusion(
                reasoning_dag, synthesis, request_id
            )
            if dag_conclusion:
                enhancement_metadata["dag_paths"] = dag_conclusion.get("paths", 0)
                synthesis = dag_conclusion.get("enhanced_synthesis", synthesis)

        # PHASE 12.6: Store in Memory Tiers
        if self.config.enable_memory_tiers:
            await self._phase_memory_tier_storage(
                request, synthesis, state, request_id
            )
            enhancement_metadata["features_used"].append("memory_tiers")

        # PHASE 12.7: Store Artifact for token reduction
        if self.config.enable_artifacts:
            await self._phase_artifact_storage(
                request_id, synthesis, state, request_id
            )
            enhancement_metadata["features_used"].append("artifacts")

        # PHASE 12.8: Progress Tracking completion
        if self.config.enable_progress_tracking:
            try:
                progress = self._get_progress_aggregator()
                progress.complete_tracking(request_id)
            except Exception as e:
                logger.debug(f"Progress completion failed: {e}")

        # PHASE 12.9: TTL Unpin
        if self.config.enable_ttl_pinning and ttl_context:
            try:
                ttl_manager = self._get_ttl_manager()
                await ttl_manager.unpin_cache(ttl_context)
            except Exception as e:
                logger.debug(f"TTL unpin failed: {e}")

        # Build response
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Update graph: complete Synthesize, mark Complete
        self._graph_state.complete("Σ")
        self._graph_state.complete("✓")

        # Emit completion event with final graph
        await self.emit_event(
            EventType.COMPLETE,
            {
                "success": True,
                "confidence": final_confidence,
                "sources_count": len(self._get_sources(state)),
                "execution_time_ms": execution_time_ms
            },
            request_id,
            message=f"Found {len(self._get_sources(state))} sources (confidence: {int(final_confidence * 100)}%)",
            graph_line=self._graph_state.to_line()
        )

        return self.build_response(
            synthesis=synthesis,
            sources=self._get_sources(state),
            queries=state.executed_queries,
            confidence=final_confidence,
            state=state,
            request_id=request_id,
            search_trace=search_trace,
            execution_time_ms=execution_time_ms,
            enhancement_metadata=enhancement_metadata
        )

    # ===== Phase Implementations =====

    async def _phase_query_analysis(
        self,
        request: SearchRequest,
        state: SearchState,
        search_trace: List[Dict],
        request_id: str
    ) -> Optional[QueryAnalysis]:
        """Phase 1: Query analysis."""
        start = time.time()
        try:
            # Reset and start graph visualization
            self._graph_state.reset()
            self._graph_state.enter("A")  # Analyze
            await self.emit_event(
                EventType.ANALYZING_QUERY,
                {"query": request.query},
                request_id,
                message="Analyzing query...",
                graph_line=self._graph_state.to_line()
            )

            analysis = await self.analyzer.analyze(
                request.query, request.context, request_id=request_id
            )
            state.query_analysis = analysis

            search_trace.append({
                "step": "analyze",
                "requires_search": analysis.requires_search,
                "query_type": analysis.query_type,
                "complexity": analysis.estimated_complexity,
                "duration_ms": int((time.time() - start) * 1000)
            })

            self._record_timing("query_analysis", time.time() - start)
            return analysis
        except Exception as e:
            logger.warning(f"[{request_id}] Query analysis failed: {e}")
            return None

    async def _phase_entity_extraction(
        self,
        request: SearchRequest,
        scratchpad,
        request_id: str
    ):
        """Phase 1.5: Entity extraction."""
        start = time.time()
        try:
            tracker = self._get_entity_tracker()
            entities = await tracker.extract_entities(request.query)
            if entities:
                # add_entity() expects Dict, EntityState.to_dict() provides that
                for entity in entities:
                    scratchpad.add_entity(entity.to_dict())
            self._record_timing("entity_tracking", time.time() - start)
        except Exception as e:
            logger.warning(f"[{request_id}] Entity extraction failed: {e}")

    async def _phase_hyde_expansion(
        self,
        query: str,
        request_id: str
    ) -> str:
        """Phase 2: HyDE query expansion."""
        start = time.time()
        try:
            expander = self._get_hyde_expander()
            result = await expander.expand(query, mode=HyDEMode.SINGLE)
            self._record_timing("hyde", time.time() - start)
            if result.hypothetical_documents:
                return result.hypothetical_documents[0]
            return query
        except Exception as e:
            logger.warning(f"[{request_id}] HyDE expansion failed: {e}")
            return query

    async def _phase_thought_library(
        self,
        query: str,
        request_id: str
    ) -> Optional[str]:
        """Phase 2.5: Thought library retrieval."""
        start = time.time()
        try:
            library = self._get_thought_library()
            # Correct method is retrieve_templates(), not retrieve()
            templates = await library.retrieve_templates(query, top_k=1)
            self._record_timing("thought_library", time.time() - start)
            if templates:
                return templates[0].template if hasattr(templates[0], 'template') else None
            return None
        except Exception as e:
            logger.warning(f"[{request_id}] Thought library retrieval failed: {e}")
            return None

    async def _phase_pre_act_planning(
        self,
        request: SearchRequest,
        search_trace: List[Dict],
        request_id: str
    ) -> Optional[PreActPlan]:
        """Phase 3: Pre-Act planning."""
        start = time.time()
        try:
            engine = self._get_reasoning_engine()
            plan = await engine.create_pre_act_plan(
                request.query,
                request.context,
                max_actions=6
            )
            search_trace.append({
                "step": "pre_act_plan",
                "actions_planned": len(plan.actions),
                "confidence": plan.confidence
            })
            self._record_timing("pre_act_planning", time.time() - start)
            return plan
        except Exception as e:
            logger.warning(f"[{request_id}] Pre-Act planning failed: {e}")
            return None

    async def _phase_search_execution(
        self,
        request: SearchRequest,
        state: SearchState,
        scratchpad,
        search_trace: List[Dict],
        request_id: str,
        pre_act_plan: Optional[PreActPlan] = None
    ):
        """Phase 4: Search execution."""
        start = time.time()

        # Update graph: complete Analyze, enter Plan
        self._graph_state.complete("A")
        self._graph_state.enter("P")

        # Emit iteration start event
        await self.emit_event(
            EventType.ITERATION_START,
            {"iteration": 1, "max_iterations": request.max_iterations},
            request_id,
            message=f"Planning search strategy...",
            graph_line=self._graph_state.to_line()
        )

        # Get context-aware search limits for maximum context utilization
        search_limits = get_search_result_limits()
        max_queries = search_limits.get("max_queries_per_iteration", 5)
        max_results_per_query = search_limits.get("max_results_per_query", 10)
        logger.info(f"[{request_id}] Search limits: max_queries={max_queries}, max_results_per_query={max_results_per_query}")

        # Determine queries to execute - always try analyzer for query decomposition
        queries = []

        # First, try analyzer for search plan decomposition
        if state.query_analysis:
            try:
                plan = await self.analyzer.create_search_plan(
                    request.query, state.query_analysis, request.context,
                    request_id=request_id
                )
                # Ensure decomposed_questions is a list of strings
                decomposed = plan.decomposed_questions if plan.decomposed_questions else []
                if isinstance(decomposed, str):
                    decomposed = [decomposed]
                queries = [q for q in decomposed if q and isinstance(q, str)][:max_queries]
                if queries:
                    state.search_plan = plan
                    logger.info(f"[{request_id}] Search queries from analyzer: {queries}")
            except Exception as e:
                logger.warning(f"[{request_id}] Analyzer search plan failed: {e}")

        # If analyzer didn't produce queries, try Pre-Act plan
        if not queries and pre_act_plan and self.config.enable_parallel_execution:
            queries = [a.inputs.get("query", "") for a in pre_act_plan.actions
                       if a.action_type.value == "search" and a.inputs.get("query")][:max_queries]
            if queries:
                logger.info(f"[{request_id}] Search queries from Pre-Act: {queries}")

        # If only 1 query, expand with topic variations to maximize context utilization
        if len(queries) <= 1:
            base_query = queries[0] if queries else request.query
            # Generate query variations for broader coverage
            expanded_queries = [base_query]
            key_topics = state.query_analysis.key_topics if state.query_analysis else []
            query_lower = base_query.lower()

            # Add topic-specific variations
            if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
                expanded_queries.extend([
                    f"{base_query} advantages disadvantages",
                    f"{base_query} pros cons 2025",
                    f"{base_query} which is better"
                ])
            elif any(word in query_lower for word in ["how", "what", "why"]):
                expanded_queries.extend([
                    f"{base_query} explained",
                    f"{base_query} latest research 2025",
                    f"{base_query} practical examples"
                ])
            else:
                # General expansion
                expanded_queries.extend([
                    f"{base_query} 2025",
                    f"{base_query} latest developments",
                    f"{base_query} benefits challenges"
                ])

            queries = expanded_queries[:max_queries]
            logger.info(f"[{request_id}] Expanded to {len(queries)} queries for context utilization: {queries}")

        # Final fallback: use original query
        if not queries:
            queries = [request.query]
            logger.info(f"[{request_id}] Using original query (no decomposition)")

        # Update graph: complete Plan, enter Search
        self._graph_state.complete("P")
        self._graph_state.enter("S")

        # Emit searching event with queries
        await self.emit_event(
            EventType.SEARCHING,
            {"queries": queries, "iteration": 1, "max_iterations": request.max_iterations},
            request_id,
            message=f"Searching {len(queries)} queries...",
            graph_line=self._graph_state.to_line()
        )

        # Execute searches with context-aware result limits
        if self.config.enable_parallel_execution and len(queries) > 1:
            # Parallel execution - searcher.search() expects a list of queries
            tasks = [self.searcher.search([q], max_results_per_query=max_results_per_query) for q in queries]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            for i, results in enumerate(results_list):
                if isinstance(results, Exception):
                    logger.warning(f"Search {i} failed: {results}")
                    continue
                state.add_results(results)
                state.mark_query_executed(queries[i])
        else:
            # Sequential execution - searcher.search() expects a list of queries
            for query in queries:
                results = await self.searcher.search([query], max_results_per_query=max_results_per_query)
                state.add_results(results)
                state.mark_query_executed(query)

        # Emit search results event (still in Search phase)
        await self.emit_event(
            EventType.SEARCH_RESULTS,
            {
                "results_count": len(state.raw_results),
                "sources_count": len(set(r.url for r in state.raw_results if hasattr(r, 'url')))
            },
            request_id,
            message=f"Found {len(state.raw_results)} results",
            graph_line=self._graph_state.to_line()
        )

        search_trace.append({
            "step": "search",
            "queries_executed": len(state.executed_queries),
            "results_found": len(state.raw_results),
            "duration_ms": int((time.time() - start) * 1000)
        })
        self._record_timing("search", time.time() - start)

    async def _phase_domain_corpus(
        self,
        query: str,
        request_id: str
    ) -> Optional[str]:
        """Phase 4.5: Domain corpus augmentation."""
        start = time.time()
        try:
            manager = self._get_domain_corpus_manager()
            # cross_domain_query signature: (query, domain_ids=None) - no top_k argument
            results = await manager.cross_domain_query(query)
            self._record_timing("domain_corpus", time.time() - start)
            if results and results.get("results"):
                context_parts = []
                # Limit to 3 results per domain manually
                for domain_id, domain_results in list(results["results"].items())[:3]:
                    if domain_results and domain_results.get("context"):
                        context_parts.append(f"[{domain_id}] {domain_results.get('context', '')[:500]}")
                return "\n\n".join(context_parts) if context_parts else None
            return None
        except Exception as e:
            logger.warning(f"[{request_id}] Domain corpus search failed: {e}")
            return None

    async def _phase_crag_evaluation(
        self,
        request: SearchRequest,
        state: SearchState,
        search_trace: List[Dict],
        request_id: str
    ):
        """Phase 5: CRAG retrieval evaluation."""
        start = time.time()
        try:
            search_results = [
                {"title": r.title, "snippet": r.snippet, "url": r.url}
                for r in state.raw_results[:10]
            ]
            evaluation = await self.retrieval_evaluator.evaluate(
                request.query,
                search_results,
                state.search_plan.decomposed_questions if state.search_plan else []
            )

            search_trace.append({
                "step": "crag_evaluation",
                "quality": evaluation.quality.value,
                "action": evaluation.recommended_action.value
            })

            # Handle corrective actions
            if evaluation.recommended_action == CorrectiveAction.REFINE_QUERY:
                state.add_pending_queries(evaluation.refined_queries[:3])
            elif evaluation.recommended_action == CorrectiveAction.WEB_FALLBACK:
                # Trigger additional web search - searcher.search() expects a list
                fallback_results = await self.searcher.search([f"detailed {request.query}"])
                state.add_results(fallback_results)

            self._record_timing("crag", time.time() - start)
        except Exception as e:
            logger.warning(f"[{request_id}] CRAG evaluation failed: {e}")

    async def _phase_hybrid_reranking(
        self,
        state: SearchState,
        request_id: str
    ):
        """Phase 6: Hybrid re-ranking with BGE-M3."""
        start = time.time()
        try:
            retriever = self._get_hybrid_retriever()

            # Index documents
            for i, result in enumerate(state.raw_results[:20]):
                await retriever.add_document(
                    doc_id=f"doc_{i}",
                    content=f"{result.title} {result.snippet}",
                    metadata={"url": result.url, "title": result.title}
                )

            # Search with hybrid mode
            reranked = await retriever.search(
                query=state.query,
                top_k=10,
                mode=RetrievalMode.HYBRID
            )

            # Update scores in state
            url_to_score = {}
            for r in reranked:
                if r.metadata and "url" in r.metadata:
                    url_to_score[r.metadata["url"]] = r.combined_score

            for result in state.raw_results:
                if result.url in url_to_score:
                    result.relevance_score = url_to_score[result.url]

            # Sort by new scores
            state.raw_results.sort(key=lambda x: x.relevance_score, reverse=True)

            self._record_timing("hybrid_reranking", time.time() - start)
        except Exception as e:
            logger.warning(f"[{request_id}] Hybrid re-ranking failed: {e}")

    async def _phase_content_scraping(
        self,
        request: SearchRequest,
        state: SearchState,
        scratchpad,
        search_trace: List[Dict],
        request_id: str
    ) -> List[str]:
        """Phase 7: Content scraping."""
        start = time.time()
        scraped_content = []

        # Deduplicate URLs before scraping (preserve order, take first occurrence)
        seen_urls = set()
        unique_urls = []
        for r in state.raw_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_urls.append(r.url)
                if len(unique_urls) >= request.max_urls_to_scrape:
                    break
        urls_to_scrape = unique_urls
        logger.info(f"[{request_id}] Scraping {len(urls_to_scrape)} unique URLs (from {len(state.raw_results)} total results)")

        # Update graph: complete Search, enter Evaluate
        self._graph_state.complete("S")
        self._graph_state.enter("E")

        # Emit evaluating URLs event
        await self.emit_event(
            EventType.EVALUATING_URLS,
            {"results_count": len(urls_to_scrape)},
            request_id,
            message=f"Evaluating {len(urls_to_scrape)} URLs...",
            graph_line=self._graph_state.to_line()
        )

        # Update graph: complete Evaluate, enter Scrape
        self._graph_state.complete("E")
        self._graph_state.enter("W")

        for i, url in enumerate(urls_to_scrape):
            # Emit scraping URL event
            await self.emit_event(
                EventType.SCRAPING_URL,
                {"url": url, "url_index": i + 1, "url_total": len(urls_to_scrape)},
                request_id,
                message=f"Scraping {i + 1}/{len(urls_to_scrape)}...",
                graph_line=self._graph_state.to_line()
            )
            try:
                result = await self.scraper.scrape_url(url)
                if result.get("success") and result.get("content"):
                    content = result["content"]
                    scraped_content.append(content[:request.max_content_per_source])
                    # Emit URL scraped event
                    await self.emit_event(
                        EventType.URL_SCRAPED,
                        {"url": url, "content_length": len(content)},
                        request_id,
                        message=f"Scraped {len(content):,} chars from {url[:40]}...",
                        graph_line=self._graph_state.to_line()
                    )
                    logger.info(f"[{request_id}] Scraped {len(content):,} chars from {url[:60]}")
                else:
                    logger.debug(f"[{request_id}] Scrape returned no content for {url[:60]}: {result.get('error', 'unknown')}")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to scrape {url[:60]}: {e}")

        search_trace.append({
            "step": "scrape",
            "urls_attempted": len(urls_to_scrape),
            "content_scraped": len(scraped_content),
            "duration_ms": int((time.time() - start) * 1000)
        })
        self._record_timing("scraping", time.time() - start)
        return scraped_content

    async def _phase_verification(
        self,
        state: SearchState,
        scraped_content: List[str],
        search_trace: List[Dict],
        request_id: str
    ):
        """Phase 8: Verification."""
        start = time.time()

        # Update graph: complete Scrape, enter Verify
        self._graph_state.complete("W")
        self._graph_state.enter("V")

        # Emit verifying event
        await self.emit_event(
            EventType.VERIFYING_CLAIMS,
            {"content_count": len(scraped_content)},
            request_id,
            message=f"Verifying {len(scraped_content)} sources...",
            graph_line=self._graph_state.to_line()
        )

        try:
            # Extract claims from scraped content
            claims = []
            for content in scraped_content[:5]:
                extracted = await self.verifier.extract_claims(content[:2000])
                claims.extend(extracted[:5])

            if claims:
                # Pass WebSearchResult objects, not raw content strings
                sources_for_verification = state.raw_results[:5]
                verification_results = await self.verifier.verify(claims[:10], sources_for_verification)

                # Calculate aggregate confidence from list of VerificationResult
                if verification_results:
                    verified_count = sum(1 for v in verification_results if v.verified)
                    avg_confidence = sum(v.confidence for v in verification_results) / len(verification_results)

                    # Create a composite result for return
                    from dataclasses import dataclass

                    @dataclass
                    class AggregateVerification:
                        confidence: float
                        verified_count: int
                        total_claims: int
                        results: list

                    aggregate = AggregateVerification(
                        confidence=avg_confidence,
                        verified_count=verified_count,
                        total_claims=len(verification_results),
                        results=verification_results
                    )

                    search_trace.append({
                        "step": "verify",
                        "claims_checked": len(claims),
                        "verified_count": verified_count,
                        "confidence": avg_confidence
                    })
                    self._record_timing("verification", time.time() - start)
                    return aggregate
        except Exception as e:
            logger.warning(f"[{request_id}] Verification failed: {e}")
        return None

    async def _phase_synthesis(
        self,
        request: SearchRequest,
        state: SearchState,
        scraped_content: List[str],
        search_trace: List[Dict],
        request_id: str,
        thought_context: Optional[str] = None,
        domain_context: Optional[str] = None
    ) -> str:
        """Phase 9: Synthesis."""
        start = time.time()

        # Update graph: complete Verify, enter Synthesize
        self._graph_state.complete("V")
        self._graph_state.enter("Σ")

        # Emit synthesizing event
        await self.emit_event(
            EventType.SYNTHESIZING,
            {"sources_count": len(scraped_content), "results_count": len(state.raw_results)},
            request_id,
            message=f"Synthesizing from {len(scraped_content)} sources...",
            graph_line=self._graph_state.to_line()
        )

        # Build additional context
        additional_context_parts = []
        if thought_context:
            additional_context_parts.append(f"Reasoning Template:\n{thought_context}")
        if domain_context:
            additional_context_parts.append(f"Domain Knowledge:\n{domain_context}")
        additional_context = "\n\n".join(additional_context_parts) if additional_context_parts else None

        # Convert raw_results to WebSearchResult-like format
        search_results = state.raw_results[:10]

        # Convert scraped content strings to Dict format for synthesize_with_content
        # The scraped_content list contains raw content strings, need to pair with URLs
        scraped_content_dicts = []
        for i, content in enumerate(scraped_content[:request.max_urls_to_scrape]):
            # Get corresponding URL from raw_results if available
            url = state.raw_results[i].url if i < len(state.raw_results) else f"source_{i+1}"
            scraped_content_dicts.append({
                "url": url,
                "content": content[:request.max_content_per_source]
            })

        # Use synthesize_with_content for full content synthesis
        synthesis = await self.synthesizer.synthesize_with_content(
            query=request.query,
            search_results=search_results,
            scraped_content=scraped_content_dicts,
            verifications=None,  # Verifications handled separately if enabled
            context={"additional_context": additional_context} if additional_context else None,
            request_id=request_id
        )

        # Calculate and log context utilization
        total_content_chars = sum(len(sc.get("content", "")) for sc in scraped_content_dicts)
        additional_context_chars = len(additional_context) if additional_context else 0
        total_input_chars = total_content_chars + additional_context_chars + len(request.query)

        utilization_report = format_context_utilization_report(
            DEFAULT_PIPELINE_CONFIG.synthesizer_model,
            total_input_chars,
            len(synthesis)
        )
        logger.info(f"[{request_id}] Context utilization:\n{utilization_report}")

        # Track context utilization in metrics
        self._get_metrics().record_context_utilization(
            request_id=request_id,
            agent_name="synthesizer",
            model_name=DEFAULT_PIPELINE_CONFIG.synthesizer_model,
            input_text="." * total_input_chars,  # Placeholder for length calculation
            output_text=synthesis,
            context_window=get_model_context_window(DEFAULT_PIPELINE_CONFIG.synthesizer_model)
        )

        search_trace.append({
            "step": "synthesize",
            "sources_count": len(scraped_content_dicts),
            "synthesis_length": len(synthesis),
            "total_input_chars": total_input_chars,
            "context_utilization": total_input_chars / (get_model_context_window(DEFAULT_PIPELINE_CONFIG.synthesizer_model) * 4),
            "duration_ms": int((time.time() - start) * 1000)
        })
        self._record_timing("synthesis", time.time() - start)
        return synthesis

    async def _phase_self_reflection(
        self,
        query: str,
        synthesis: str,
        state: SearchState,
        scraped_content: List[str],
        request_id: str
    ) -> Optional[ReflectionResult]:
        """Phase 10: Self-reflection."""
        start = time.time()

        # Update graph: complete Synthesize, enter Reflect
        self._graph_state.complete("Σ")
        self._graph_state.enter("R")

        await self.emit_event(
            EventType.SELF_RAG_REFLECTING,
            {"synthesis_length": len(synthesis)},
            request_id,
            message="Reflecting on synthesis quality...",
            graph_line=self._graph_state.to_line()
        )

        try:
            result = await self.reflector.reflect(
                query=query,
                synthesis=synthesis,
                sources=[{"snippet": s.get("snippet", "")} for s in self._get_sources(state)],
                scraped_content=scraped_content[:5]
            )

            # Mark reflect complete
            self._graph_state.complete("R")

            self._record_timing("self_reflection", time.time() - start)
            return result
        except Exception as e:
            logger.warning(f"[{request_id}] Self-reflection failed: {e}")
            self._graph_state.complete("R")  # Still mark complete even on failure
            return None

    async def _phase_ragas_evaluation(
        self,
        query: str,
        synthesis: str,
        state: SearchState,
        scraped_content: List[str],
        request_id: str
    ) -> Optional[RAGASResult]:
        """Phase 11: RAGAS evaluation.

        Uses scraped_content (full page content) for claim verification,
        not just search result snippets.
        """
        start = time.time()
        try:
            evaluator = self._get_ragas_evaluator()

            # Use scraped content if available (contains full page text)
            # Fall back to snippets only if no scraped content
            if scraped_content and any(len(c) > 100 for c in scraped_content):
                # Use first 5 scraped pages, each truncated to 5000 chars for efficiency
                contexts = [c[:5000] for c in scraped_content[:5] if len(c) > 100]
                logger.info(f"[{request_id}] RAGAS using {len(contexts)} scraped content pieces "
                           f"(total chars: {sum(len(c) for c in contexts)})")
            else:
                # Fall back to snippets if no scraped content
                contexts = [
                    s.get("snippet", s.get("title", ""))
                    for s in self._get_sources(state)[:5]
                    if isinstance(s, dict)
                ]
                logger.warning(f"[{request_id}] RAGAS falling back to snippets (no scraped content)")

            result = await evaluator.evaluate(
                question=query,
                answer=synthesis,
                contexts=contexts
            )

            logger.info(f"[{request_id}] RAGAS evaluation: faith={result.faithfulness:.2f}, "
                       f"claims={len(result.claims)}/{sum(1 for v in result.claim_verifications if v.supported)} supported")

            self._record_timing("ragas", time.time() - start)
            return result
        except Exception as e:
            logger.warning(f"[{request_id}] RAGAS evaluation failed: {e}")
            return None

    async def _phase_experience_distillation(
        self,
        request: SearchRequest,
        synthesis: str,
        confidence: float,
        state: SearchState,
        request_id: str
    ):
        """Phase 12a: Experience distillation."""
        try:
            response = SearchResponse(
                success=True,
                data=SearchResultData(
                    synthesized_context=synthesis,
                    sources=self._get_sources(state),
                    search_queries=state.executed_queries,
                    confidence_score=confidence
                ),
                meta=SearchMeta(request_id=request_id)
            )
            await self.experience_distiller.capture_experience(
                query=request.query,
                response=response,
                query_type=state.query_analysis.query_type if state.query_analysis else "research",
                decomposed_questions=state.search_plan.decomposed_questions if state.search_plan else []
            )
        except Exception as e:
            logger.debug(f"Experience capture failed: {e}")

    async def _phase_classifier_feedback(
        self,
        request: SearchRequest,
        analysis: QueryAnalysis,
        confidence: float,
        state: SearchState,
        request_id: str,
        start_time: float
    ):
        """Phase 12b: Classifier feedback."""
        try:
            from .query_classifier import QueryClassification, RecommendedPipeline, QueryCategory

            classification = QueryClassification(
                category=QueryCategory.RESEARCH,
                capabilities=["web_search"],
                complexity=analysis.estimated_complexity,
                recommended_pipeline=RecommendedPipeline.AGENTIC_SEARCH
            )
            self.classifier_feedback.record_outcome(
                query=request.query,
                classification=classification,
                confidence=confidence,
                iteration_count=state.iteration,
                source_count=len(state.raw_results),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            logger.debug(f"Classifier feedback failed: {e}")

    async def _handle_direct_answer(
        self,
        request: SearchRequest,
        analysis: QueryAnalysis,
        request_id: str,
        start_time: float
    ) -> SearchResponse:
        """Handle queries that don't need search."""
        # For no-search queries, use the basic synthesize with empty results
        # Just create a simple response without web search
        synthesis = await self.synthesizer.synthesize(
            query=request.query,
            search_results=[],  # No search results
            verifications=None,
            context={
                "query_type": analysis.query_type,
                "reasoning": analysis.search_reasoning
            },
            request_id=request_id
        )
        return self.build_response(
            synthesis=synthesis,
            sources=[],
            queries=[],
            confidence=analysis.confidence,
            state=self.create_search_state(request),
            request_id=request_id,
            search_trace=[{"step": "direct_answer", "reason": analysis.search_reasoning}],
            execution_time_ms=int((time.time() - start_time) * 1000)
        )

    # ===== Additional Phase Implementations (Layer 3-4 Features) =====

    async def _phase_dynamic_planning(
        self,
        request: SearchRequest,
        request_id: str
    ) -> Optional[PlannerOutput]:
        """Phase 1.6: AIME-style dynamic planning."""
        start = time.time()
        try:
            planner = self._get_dynamic_planner()
            # Correct method is initial_decomposition(), not create_initial_plan()
            plan = await planner.initial_decomposition(
                goal=request.query,
                context=request.context
            )
            self._record_timing("dynamic_planning", time.time() - start)
            return plan
        except Exception as e:
            logger.warning(f"[{request_id}] Dynamic planning failed: {e}")
            return None

    async def _phase_init_reasoning_dag(
        self,
        request: SearchRequest,
        request_id: str
    ) -> Optional[ReasoningDAG]:
        """Phase 1.7: Initialize reasoning DAG for complex queries."""
        start = time.time()
        try:
            dag = self._get_reasoning_dag()
            # Add root node for the query
            dag.add_node(request.query, NodeType.ROOT)
            self._record_timing("reasoning_dag_init", time.time() - start)
            return dag
        except Exception as e:
            logger.warning(f"[{request_id}] Reasoning DAG init failed: {e}")
            return None

    async def _phase_embedding_aggregation(
        self,
        query: str,
        request_id: str
    ) -> Optional[Any]:
        """Phase 2.6: Embedding aggregation for domain routing."""
        start = time.time()
        try:
            aggregator = self._get_embedding_aggregator()
            # Correct method is retrieve(), not aggregate()
            result = await aggregator.retrieve(query)
            self._record_timing("embedding_aggregation", time.time() - start)
            return result
        except Exception as e:
            logger.warning(f"[{request_id}] Embedding aggregation failed: {e}")
            return None

    async def _phase_multi_agent_execution(
        self,
        request: SearchRequest,
        state: SearchState,
        search_trace: List[Dict],
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Phase 3.5: Multi-agent parallel execution using ActorFactory."""
        start = time.time()
        try:
            # Create specialized actors for different aspects of the search
            factory = self._get_actor_factory()
            orchestrator = self._get_multi_agent_orchestrator()

            # Create actor directly with the query (analyze_task doesn't exist)
            # ActorFactory.create_actor() analyzes task requirements internally
            actors = []
            # Create a web research actor for the main query
            actor = await factory.create_actor(
                task_description=request.query,
                force_bundles=["web_research"]  # Use bundle name, not tool names
            )
            if actor:
                actors.append(actor)

            # Execute in parallel if we have multiple actors
            if len(actors) > 1:
                results = await orchestrator.execute_parallel(
                    actors=actors,
                    query=request.query,
                    timeout=30.0
                )
                # Merge results into state
                for result in results:
                    if result.get("search_results"):
                        for r in result["search_results"][:5]:
                            state.add_result(r)

                search_trace.append({
                    "step": "multi_agent",
                    "actors": len(actors),
                    "results": len(results)
                })

                self._record_timing("multi_agent", time.time() - start)
                return {
                    "agents": [a.persona.name if hasattr(a, 'persona') else "agent" for a in actors],
                    "results_count": sum(len(r.get("search_results", [])) for r in results)
                }

            return None
        except Exception as e:
            logger.warning(f"[{request_id}] Multi-agent execution failed: {e}")
            return None

    async def _phase_entity_enhanced_retrieval(
        self,
        request: SearchRequest,
        state: SearchState,
        request_id: str
    ):
        """Phase 6.5: Entity-enhanced retrieval."""
        start = time.time()
        try:
            # Use the standalone retrieve_with_entities function from embedding_aggregator
            from .embedding_aggregator import retrieve_with_entities
            enhanced_results = await retrieve_with_entities(
                query=request.query,
                k=10
            )
            # Boost results with high entity overlap
            if enhanced_results and hasattr(enhanced_results, 'context'):
                # Enhanced results are in RetrievalResult format
                logger.info(f"[{request_id}] Entity-enhanced retrieval: {len(enhanced_results.entities)} entities found")
            self._record_timing("entity_enhanced_retrieval", time.time() - start)
        except Exception as e:
            logger.warning(f"[{request_id}] Entity-enhanced retrieval failed: {e}")

    async def _phase_mixed_precision_indexing(
        self,
        state: SearchState,
        request_id: str
    ):
        """Phase 6.6: Index results at multiple precision levels."""
        start = time.time()
        try:
            service = self._get_mixed_precision_service()
            for i, result in enumerate(state.raw_results[:15]):
                text = f"{result.title} {result.snippet}"
                # Correct method is index_document(), not index_at_all_precisions()
                await service.index_document(
                    doc_id=f"{request_id}_{i}",
                    text=text,
                    content=result.snippet,
                    metadata={"url": result.url, "title": result.title}
                )
            self._record_timing("mixed_precision_indexing", time.time() - start)
        except Exception as e:
            logger.warning(f"[{request_id}] Mixed precision indexing failed: {e}")

    async def _phase_deep_reading(
        self,
        scraped_content: List[str],
        query: str,
        request_id: str
    ) -> Optional[str]:
        """Phase 7.5: Deep reading analysis."""
        start = time.time()
        try:
            deep_reader = self._get_deep_reader()
            # Analyze scraped content using DeepReader
            if scraped_content:
                # Convert string content to dict format expected by DeepReader
                content_dicts = [
                    {"content": c[:10000], "url": f"source_{i}", "success": True}
                    for i, c in enumerate(scraped_content[:5])
                ]
                insights = await deep_reader.analyze_content(
                    question=query,
                    scraped_content=content_dicts,
                    request_id=request_id
                )
                self._record_timing("deep_reading", time.time() - start)
                # Return the synthesis from the insights dict
                return insights.get("synthesis", "") if isinstance(insights, dict) else str(insights)
            return None
        except Exception as e:
            logger.warning(f"[{request_id}] Deep reading failed: {e}")
            return None

    async def _phase_vision_analysis(
        self,
        state: SearchState,
        request_id: str
    ) -> Optional[str]:
        """Phase 7.6: Vision analysis of any images in results."""
        start = time.time()
        try:
            analyzer = self._get_vision_analyzer()
            # Look for images in scraped PDF content
            image_insights = []
            for result in state.raw_results[:5]:
                # Check if result has extracted images (from PDF scraping)
                if hasattr(result, 'images') and result.images:
                    for img in result.images[:3]:  # Limit to 3 images per result
                        if 'base64' in img:
                            insight_result = await analyzer.analyze_image(
                                img['base64'],
                                context=f"Image from {result.title}",
                                request_id=request_id
                            )
                            if insight_result.get('success') and insight_result.get('description'):
                                image_insights.append(insight_result['description'])
            self._record_timing("vision_analysis", time.time() - start)
            return "\n".join(image_insights) if image_insights else None
        except Exception as e:
            logger.warning(f"[{request_id}] Vision analysis failed: {e}")
            return None

    async def _phase_stuck_detection(
        self,
        state: SearchState,
        scratchpad,
        request_id: str
    ) -> tuple:
        """Phase 7.7: Detect if search is stuck in a loop."""
        start = time.time()
        try:
            engine = self._get_reasoning_engine()
            # Check for repeated queries
            query_counts = {}
            for q in state.executed_queries:
                query_counts[q] = query_counts.get(q, 0) + 1

            is_stuck = any(count >= 2 for count in query_counts.values())
            recovery_action = None

            if is_stuck:
                # Determine recovery strategy
                if state.iteration >= 3:
                    recovery_action = "accept"  # Accept what we have
                else:
                    recovery_action = "broaden"  # Try broader search

            self._record_timing("stuck_detection", time.time() - start)
            return is_stuck, recovery_action
        except Exception as e:
            logger.warning(f"[{request_id}] Stuck detection failed: {e}")
            return False, None

    async def _phase_positional_optimization(
        self,
        scraped_content: List[str],
        query: str,
        request_id: str
    ) -> List[str]:
        """Phase 8.5: Optimize content positions to mitigate lost-in-the-middle."""
        start = time.time()
        try:
            from .sufficient_context import get_positional_optimizer
            optimizer = get_positional_optimizer()

            # Convert scraped_content strings to source dicts for the optimizer
            sources = [
                {"content": content, "title": f"Source {i+1}"}
                for i, content in enumerate(scraped_content)
            ]

            # Score all sources at once (correct API: question, sources)
            relevance_scores = await optimizer.score_relevance(query, sources)

            # Use the optimizer's reorder method for lost-in-middle mitigation
            reordered_sources, analysis = optimizer.reorder_for_optimal_attention(
                sources, relevance_scores
            )

            # Extract content back from reordered sources
            optimized = [s.get("content", "") for s in reordered_sources]

            logger.debug(f"[{request_id}] Positional optimization: risk={analysis.lost_in_middle_risk}")
            self._record_timing("positional_optimization", time.time() - start)
            return optimized
        except Exception as e:
            logger.warning(f"[{request_id}] Positional optimization failed: {e}")
            return scraped_content

    async def _phase_contradiction_detection(
        self,
        synthesis: str,
        scraped_content: List[str],
        request_id: str
    ) -> Optional[List[str]]:
        """Phase 9.5: Detect contradictions in sources."""
        start = time.time()
        try:
            engine = self._get_reasoning_engine()
            # Correct signature: detect_contradictions(sources: List[Dict], key_claims: List[str])
            # Convert scraped_content to source dicts
            sources = [
                {"url": f"source_{i}", "content": content[:500]}
                for i, content in enumerate(scraped_content[:5])
            ]
            # Extract key claims from synthesis (simple sentence extraction)
            key_claims = [s.strip() for s in synthesis.split('.') if len(s.strip()) > 20][:5]

            contradictions = await engine.detect_contradictions(
                sources=sources,
                key_claims=key_claims
            )
            self._record_timing("contradiction_detection", time.time() - start)
            return contradictions if contradictions else None
        except Exception as e:
            logger.warning(f"[{request_id}] Contradiction detection failed: {e}")
            return None

    async def _phase_reasoning_dag_conclusion(
        self,
        dag: ReasoningDAG,
        synthesis: str,
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """Phase 12.5: Extract conclusion from reasoning DAG."""
        start = time.time()
        try:
            # Add synthesis as conclusion node
            dag.add_node(synthesis[:500], NodeType.CONCLUSION)

            # Get convergent answer
            conclusion = dag.get_convergent_answer()

            result = {
                "paths": len(dag.nodes),
                "enhanced_synthesis": conclusion if conclusion else synthesis
            }
            self._record_timing("reasoning_dag_conclusion", time.time() - start)
            return result
        except Exception as e:
            logger.warning(f"[{request_id}] Reasoning DAG conclusion failed: {e}")
            return None

    async def _phase_memory_tier_storage(
        self,
        request: SearchRequest,
        synthesis: str,
        state: SearchState,
        request_id: str
    ):
        """Phase 12.6: Store in three-tier memory system."""
        start = time.time()
        try:
            from .memory_tiers import ContentType
            manager = self._get_memory_tier_manager()

            # Store synthesis in warm tier
            await manager.store(
                content_id=f"search_{request_id}",
                content=synthesis,
                content_type=ContentType.SEARCH_RESULT,
                metadata={
                    "query": request.query,
                    "sources": len(state.raw_results),
                    "confidence": state.confidence if hasattr(state, 'confidence') else 0.5
                }
            )
            self._record_timing("memory_tier_storage", time.time() - start)
        except Exception as e:
            logger.warning(f"[{request_id}] Memory tier storage failed: {e}")

    async def _phase_artifact_storage(
        self,
        session_id: str,
        synthesis: str,
        state: SearchState,
        request_id: str
    ):
        """Phase 12.7: Store as artifact for token reduction."""
        start = time.time()
        try:
            artifact_store = self._get_artifact_store()
            # store() is not async, so don't use await
            artifact_store.store(
                session_id=session_id,
                artifact_type=ArtifactType.SYNTHESIS,
                content=synthesis,
                metadata={
                    "query": state.query,
                    "source_count": len(state.raw_results)
                }
            )
            self._record_timing("artifact_storage", time.time() - start)
        except Exception as e:
            logger.warning(f"[{request_id}] Artifact storage failed: {e}")

    # ===== Helper Methods =====

    def _get_sources(self, state: SearchState) -> List[Dict[str, str]]:
        """Extract sources from state."""
        return [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in state.raw_results[:10]
        ]

    def _get_enabled_features(self) -> List[str]:
        """Get list of enabled features."""
        features = []
        for field_name in dir(self.config):
            if field_name.startswith("enable_") and getattr(self.config, field_name):
                features.append(field_name.replace("enable_", ""))
        return features

    def _record_timing(self, feature: str, duration: float):
        """Record feature timing for performance analysis."""
        if feature not in self._feature_timings:
            self._feature_timings[feature] = []
        self._feature_timings[feature].append(duration * 1000)  # ms

    def _apply_dynamic_context_limits(
        self,
        request: SearchRequest,
        request_id: str
    ) -> Dict[str, Any]:
        """
        Calculate and apply dynamic context limits based on the synthesizer model.

        Uses the model's recorded context window from the database to maximize
        context utilization for better results.

        Returns:
            Dict with applied limits and utilization info for logging
        """
        # Get limits based on synthesizer model (DeepSeek-R1 or configured model)
        synthesizer_model = DEFAULT_PIPELINE_CONFIG.synthesizer_model
        is_thinking = "deepseek" in synthesizer_model.lower() or "r1" in synthesizer_model.lower()

        limits = get_synthesizer_limits(
            model_name=synthesizer_model,
            is_thinking_model=is_thinking
        )

        # Calculate context budget for detailed logging
        budget = calculate_context_budget(
            synthesizer_model,
            system_prompt_chars=2500,
            response_reserve_chars=8000 if is_thinking else 4000
        )

        # Apply dynamic limits - override request defaults with model-aware values
        original_limits = {
            "max_urls_to_scrape": request.max_urls_to_scrape,
            "max_content_per_source": request.max_content_per_source,
            "max_sources": request.max_sources
        }

        # Update request with calculated limits
        request.max_urls_to_scrape = limits["max_urls_to_scrape"]
        request.max_content_per_source = limits["max_content_per_source"]
        request.max_sources = min(request.max_sources, limits["max_urls_to_scrape"] * 2)

        applied_limits = {
            "max_urls_to_scrape": request.max_urls_to_scrape,
            "max_content_per_source": request.max_content_per_source,
            "max_sources": request.max_sources,
            "max_total_content": limits["max_total_content"],
            "num_ctx": limits["num_ctx"]
        }

        # Log context utilization info
        logger.info(f"[{request_id}] Context limits applied for model '{synthesizer_model}':")
        logger.info(f"[{request_id}]   Context window: {budget.context_window_tokens:,} tokens ({budget.context_window_chars:,} chars)")
        logger.info(f"[{request_id}]   Available for content: {budget.available_chars:,} chars ({budget.target_utilization*100:.0f}% utilization)")
        logger.info(f"[{request_id}]   Max URLs to scrape: {original_limits['max_urls_to_scrape']} -> {applied_limits['max_urls_to_scrape']}")
        logger.info(f"[{request_id}]   Max content/source: {original_limits['max_content_per_source']:,} -> {applied_limits['max_content_per_source']:,} chars")
        logger.info(f"[{request_id}]   Total capacity: {applied_limits['max_total_content']:,} chars (~{applied_limits['max_total_content']//4:,} tokens)")

        return {
            "model": synthesizer_model,
            "context_window_tokens": budget.context_window_tokens,
            "available_chars": budget.available_chars,
            "original_limits": original_limits,
            "applied_limits": applied_limits,
            "is_thinking_model": is_thinking
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        base_stats = super().get_stats()

        # Add feature-specific stats
        feature_stats = {}
        for feature, timings in self._feature_timings.items():
            if timings:
                feature_stats[feature] = {
                    "count": len(timings),
                    "avg_ms": sum(timings) / len(timings),
                    "max_ms": max(timings)
                }

        return {
            **base_stats,
            "preset": self._get_preset_name(),
            "features_enabled": self._get_enabled_features(),
            "total_searches": self._stats["total_searches"],
            "cache_hits": self._stats["cache_hits"],
            "feature_timings": feature_stats,
            "config": {
                field: getattr(self.config, field)
                for field in dir(self.config)
                if field.startswith("enable_")
            }
        }

    def _get_preset_name(self) -> str:
        """Determine which preset matches current config."""
        for preset, config in PRESET_CONFIGS.items():
            match = True
            for field in dir(config):
                if field.startswith("enable_"):
                    if getattr(config, field) != getattr(self.config, field):
                        match = False
                        break
            if match:
                return preset.value
        return "custom"
