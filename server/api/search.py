"""
Search API Router

REST endpoints for agentic search functionality.
Isolated from core memOS services.

Endpoints:
- POST /api/v1/search/agentic - Multi-step agentic search
- POST /api/v1/search/stream - Streaming agentic search with SSE progress
- POST /api/v1/search/simple - Simple web search
- GET /api/v1/search/status/{search_id} - Search status (for async)
- GET /api/v1/search/events/{request_id} - SSE event stream for a search
- DELETE /api/v1/search/cache - Clear search cache
"""

import asyncio
import logging
import os
from datetime import datetime, timezone, UTC
from typing import Optional, List, Dict, Any

from config.settings import get_settings

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError as PydanticValidationError

# Phase 7: Unified exception handling
from core.exceptions import (
    AppException,
    ErrorCode,
    ValidationError,
    SearchError,
    SearchTimeoutError,
    ExternalServiceError,
    ServiceUnavailableError
)

from agentic import UniversalOrchestrator, OrchestratorPreset, AgenticOrchestrator
from agentic.models import (
    SearchRequest,
    SearchResponse,
    SimpleSearchRequest,
    VerificationLevel,
    SearchMode
)
from agentic.multi_agent import MultiAgentOrchestrator

# =============================================================================
# DEPRECATION NOTICE: Legacy Orchestrators
# =============================================================================
# The following orchestrators are DEPRECATED and will be removed in a future release:
# - AgenticOrchestrator → Use UniversalOrchestrator(preset=OrchestratorPreset.BALANCED)
# - EnhancedAgenticOrchestrator → Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)
# - GraphEnhancedOrchestrator → Use UniversalOrchestrator(preset=OrchestratorPreset.RESEARCH)
# - UnifiedOrchestrator → Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)
# - DynamicOrchestrator → Use UniversalOrchestrator with enable_dynamic_planning=True
#
# SINGLE SOURCE OF TRUTH: UniversalOrchestrator with preset-based configuration
# =============================================================================
from agentic.events import (
    get_event_manager,
    EventType,
    SearchEvent
)
from agentic.query_classifier import (
    QueryClassifier,
    QueryClassification,
    QueryCategory,
    RecommendedPipeline,
    QueryComplexity
)
from pydantic import BaseModel, Field

logger = logging.getLogger("api.search")


# Request/Response models for classification
class ClassifyRequest(BaseModel):
    """Request model for query classification"""
    query: str
    context: Optional[dict] = None


class ClassifyResponse(BaseModel):
    """Response model for query classification"""
    success: bool
    category: str
    capabilities: list[str]
    complexity: str
    urgency: str
    use_thinking_model: bool
    recommended_pipeline: str
    reasoning: str

router = APIRouter(prefix="/api/v1/search", tags=["Search"])

# =============================================================================
# UNIFIED ORCHESTRATOR SYSTEM
# =============================================================================
# All orchestrators are now consolidated into UniversalOrchestrator with presets.
# Legacy getters are maintained for backward compatibility but redirect to Universal.
# =============================================================================

# Cache of UniversalOrchestrator instances by preset
_universal_orchestrators: Dict[str, UniversalOrchestrator] = {}

# Legacy orchestrator instances (DEPRECATED - use get_universal_orchestrator instead)
_orchestrator: Optional[AgenticOrchestrator] = None  # DEPRECATED
_graph_orchestrator = None  # DEPRECATED
_enhanced_orchestrator = None  # DEPRECATED
_multi_orchestrator: Optional[MultiAgentOrchestrator] = None


async def get_universal_orchestrator(preset: str = "balanced") -> UniversalOrchestrator:
    """
    Get or create a UniversalOrchestrator for the given preset.

    This is the SINGLE SOURCE OF TRUTH for all orchestrator access.

    Args:
        preset: One of 'minimal', 'balanced', 'enhanced', 'research', 'full'

    Returns:
        UniversalOrchestrator instance configured with the specified preset
    """
    global _universal_orchestrators

    if preset not in _universal_orchestrators:
        settings = get_settings()
        preset_enum = OrchestratorPreset(preset) if preset in [p.value for p in OrchestratorPreset] else OrchestratorPreset.BALANCED
        _universal_orchestrators[preset] = UniversalOrchestrator(
            ollama_url=settings.ollama_base_url,
            mcp_url=settings.mcp_url,
            brave_api_key=settings.brave_api_key,
            preset=preset_enum,
            db_path=settings.data_dir
        )
        await _universal_orchestrators[preset].initialize()
        logger.info(f"UniversalOrchestrator initialized with preset: {preset}")
    return _universal_orchestrators[preset]


async def get_orchestrator() -> UniversalOrchestrator:
    """
    DEPRECATED: Use get_universal_orchestrator('balanced') instead.

    Maintained for backward compatibility - redirects to UniversalOrchestrator.
    """
    import warnings
    warnings.warn(
        "get_orchestrator() is deprecated. Use get_universal_orchestrator('balanced') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return await get_universal_orchestrator("balanced")


async def get_graph_orchestrator() -> UniversalOrchestrator:
    """
    DEPRECATED: Use get_universal_orchestrator('research') instead.

    GraphEnhancedOrchestrator features are available via:
    - enable_graph_cache=True
    - enable_prefetching=True
    """
    import warnings
    warnings.warn(
        "get_graph_orchestrator() is deprecated. Use get_universal_orchestrator('research') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return await get_universal_orchestrator("research")


async def get_enhanced_orchestrator() -> UniversalOrchestrator:
    """
    DEPRECATED: Use get_universal_orchestrator('enhanced') instead.

    EnhancedAgenticOrchestrator features are available via:
    - enable_pre_act_planning=True
    - enable_stuck_detection=True
    - enable_contradiction_detection=True
    - enable_parallel_execution=True
    """
    import warnings
    warnings.warn(
        "get_enhanced_orchestrator() is deprecated. Use get_universal_orchestrator('enhanced') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return await get_universal_orchestrator("enhanced")


# Global classifier instance
_classifier: Optional[QueryClassifier] = None


async def get_classifier() -> QueryClassifier:
    """Get or create the query classifier instance"""
    global _classifier
    if _classifier is None:
        import os
        _classifier = QueryClassifier(
            ollama_url=get_settings().ollama_base_url
        )
        logger.info("Query classifier initialized (DeepSeek-R1 based)")
    return _classifier


@router.post("/classify", response_model=ClassifyResponse)
async def classify_query(request: ClassifyRequest):
    """
    Classify a query to determine optimal processing pipeline.

    Uses DeepSeek-R1 14B with Chain-of-Draft prompting for efficient
    classification with reasoning capabilities.

    Query Categories:
    - research: Information gathering, learning about topics
    - problem_solving: Debugging, troubleshooting, solutions
    - factual: Direct questions with verifiable answers
    - creative: Open-ended brainstorming, ideation
    - technical: Code, engineering, scientific analysis
    - comparative: Evaluating options, comparing alternatives
    - how_to: Step-by-step guidance, tutorials

    Recommended Pipelines:
    - direct_answer: Simple LLM response, no search needed
    - web_search: Basic web search + synthesis
    - agentic_search: Full multi-agent pipeline
    - code_assistant: Technical/code analysis mode

    Args:
        request: ClassifyRequest with query and optional context

    Returns:
        ClassifyResponse with category, capabilities, and pipeline recommendation
    """
    logger.info(f"Classification request: {request.query[:50]}...")

    try:
        classifier = await get_classifier()
        result = await classifier.classify(request.query, request.context)

        return ClassifyResponse(
            success=True,
            category=result.category.value,
            capabilities=result.capabilities,
            complexity=result.complexity.value,
            urgency=result.urgency,
            use_thinking_model=result.use_thinking_model,
            recommended_pipeline=result.recommended_pipeline.value,
            reasoning=result.reasoning
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        # Phase 7: Use unified exception format
        raise SearchError(
            message=f"Query classification failed: {str(e)}",
            code=ErrorCode.CLASSIFICATION_FAILED,
            query=request.query
        )


@router.post("/agentic", response_model=SearchResponse)
async def agentic_search(request: SearchRequest):
    """
    Execute multi-step agentic search with intelligent query analysis.

    This endpoint uses an enhanced ReAct (Reasoning + Acting) pattern to:
    1. Analyze: LLM determines if web search would be beneficial for the query
    2. Plan: Decompose query into targeted search terms with multi-phase strategy
    3. Search: Execute adaptive web searches until leads are exhausted
    4. Verify: Cross-check facts (if verification_level != 'none')
    5. Synthesize: Combine results into a coherent answer

    Search Modes:
    - FIXED: Stop after max_iterations (default behavior)
    - ADAPTIVE: Continue until min_sources reached AND information is sufficient
    - EXHAUSTIVE: Search all possible leads regardless of sufficiency

    Args:
        request: SearchRequest with:
            - query: The search query
            - user_id: Optional user ID for personalization
            - context: Optional conversation context
            - search_mode: FIXED, ADAPTIVE, or EXHAUSTIVE (default: ADAPTIVE)
            - analyze_query: Whether to use LLM to analyze query first (default: True)
            - max_iterations: Maximum search iterations (default: 5, max: 50)
            - min_sources: Minimum sources before stopping (default: 3)
            - max_sources: Maximum sources to consult (default: 15)
            - verification_level: NONE, STANDARD, or STRICT

    Returns:
        SearchResponse with:
            - synthesized_context: The AI-generated answer
            - sources: List of consulted sources
            - search_queries: All queries executed
            - confidence_score: 0.0-1.0 confidence rating
            - search_trace: Full trace of agent actions for debugging
    """
    logger.info(f"Agentic search request: {request.query[:50]}...")

    try:
        orchestrator = await get_orchestrator()
        response = await orchestrator.search(request)
        return response

    except Exception as e:
        logger.error(f"Agentic search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/graph-enhanced", response_model=SearchResponse)
async def graph_enhanced_search(request: SearchRequest):
    """
    Execute GRAPH-ENHANCED agentic search with KV cache optimization.

    This endpoint uses the graph-based KV cache system inspired by:
    - KVFlow (NeurIPS 2025): Agent Step Graph for workflow-aware cache eviction
    - ROG (2025): Chain-style reasoning with intermediate answer caching
    - LbMAS (2025): Blackboard architecture for multi-agent coordination

    Key optimizations:
    - Steps-to-execution (STE) based cache eviction priority
    - Proactive prefetching for likely next agents
    - Mission decomposition caching (reuse query decompositions)
    - Sub-query result caching (ROG-style intermediate answers)
    - Semantic finding deduplication

    Expected improvements over standard agentic search:
    - 50-80% reduction in inference latency for repeated patterns
    - 80%+ cache hit rate for similar workflows
    - Intelligent eviction based on workflow position

    Args:
        request: Same as /agentic endpoint

    Returns:
        SearchResponse with additional graph cache metadata in search_trace
    """
    logger.info(f"Graph-enhanced search request: {request.query[:50]}...")

    try:
        orchestrator = await get_graph_orchestrator()
        response = await orchestrator.search(request)
        return response

    except Exception as e:
        logger.error(f"Graph-enhanced search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Graph-enhanced search failed: {str(e)}"
        )


@router.get("/graph-enhanced/stats")
async def get_graph_enhanced_stats():
    """
    Get comprehensive statistics from the graph-enhanced orchestrator.

    Returns stats on:
    - Total searches and cache hits
    - Mission decomposition reuse count
    - Sub-query cache reuse count
    - Agent Step Graph transition probabilities
    - Scratchpad cache hit rates
    """
    try:
        orchestrator = await get_graph_orchestrator()
        stats = orchestrator.get_graph_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get graph stats failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph stats: {str(e)}"
        )


@router.post("/enhanced", response_model=SearchResponse)
async def enhanced_agentic_search(request: SearchRequest):
    """
    Execute ENHANCED agentic search with research-backed improvements.

    This endpoint implements cutting-edge patterns from 2025 research:

    1. **Pre-Act Planning** (arXiv 2505.09970): Creates multi-step execution
       plans BEFORE acting, enabling parallel execution and 70% accuracy
       improvement over standard ReAct.

    2. **Self-Reflection Loop**: After synthesis, the system critiques its
       own output and refines if quality is insufficient.

    3. **Stuck State Detection**: Detects when the agent is in a loop and
       attempts recovery strategies (broaden, narrow, rephrase, simplify).

    4. **Parallel Search Execution**: Executes independent searches
       concurrently for faster results.

    5. **Contradiction Detection**: Surfaces conflicting information from
       different sources rather than arbitrarily choosing.

    Expected improvements over /agentic:
    - Higher confidence scores through multi-signal calibration
    - Better synthesis quality via reflection
    - Faster execution via parallel search
    - More transparent results with contradiction surfacing

    Args:
        request: SearchRequest with query and options

    Returns:
        SearchResponse with enhanced search results
    """
    try:
        orchestrator = await get_enhanced_orchestrator()
        response = await orchestrator.search(request)
        return response

    except Exception as e:
        logger.error(f"Enhanced search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced search failed: {str(e)}"
        )


@router.get("/enhanced/stats")
async def get_enhanced_stats():
    """
    Get statistics from the enhanced agentic orchestrator.

    Returns stats on:
    - Pre-Act plans created
    - Reflections triggered
    - Stuck state recoveries
    - Parallel batches executed
    - Contradictions surfaced
    - Average quality improvement
    """
    try:
        orchestrator = await get_enhanced_orchestrator()
        stats = orchestrator.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features_enabled": {
                    "pre_act": orchestrator.enable_pre_act,
                    "reflection": orchestrator.enable_reflection,
                    "stuck_detection": orchestrator.enable_stuck_detection,
                    "contradiction_detection": orchestrator.enable_contradiction_detection
                }
            }
        }

    except Exception as e:
        logger.error(f"Get enhanced stats failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get enhanced stats: {str(e)}"
        )


@router.post("/enhanced/stream")
async def streaming_enhanced_search(request: SearchRequest):
    """
    Execute ENHANCED agentic search with real-time SSE progress updates.

    Combines the research-backed enhanced reasoning patterns with streaming:
    - Pre-Act planning events
    - Self-reflection progress
    - Contradiction detection alerts
    - Parallel search batch events

    See /enhanced endpoint for full feature description.
    See /stream endpoint for SSE event format.
    """
    import uuid

    request_id = str(uuid.uuid4())
    logger.info(f"Enhanced streaming search request [{request_id}]: {request.query[:50]}...")

    # Create event emitter for this search
    event_manager = get_event_manager()
    emitter = event_manager.create_emitter(request_id)

    async def generate_events():
        """Generator for SSE events"""
        queue = emitter.subscribe()

        try:
            # Start the enhanced search in a background task
            search_task = asyncio.create_task(
                _execute_enhanced_streaming_search(request, request_id, emitter)
            )

            # Stream events until search completes
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=90.0)

                    if event is None:  # End of stream signal
                        break

                    yield event.to_sse()

                    # If this is the final event, break
                    if event.event_type in [EventType.SEARCH_COMPLETED, EventType.SEARCH_FAILED]:
                        break

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"

            # Wait for search to finish
            await search_task

        except asyncio.CancelledError:
            logger.info(f"[{request_id}] Enhanced stream cancelled by client")
        except Exception as e:
            logger.error(f"[{request_id}] Enhanced stream error: {e}")
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            emitter.unsubscribe(queue)
            event_manager.remove_emitter(request_id)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Request-Id": request_id,
            "X-Enhanced": "true"  # Marker for enhanced endpoint
        }
    )


async def _execute_enhanced_streaming_search(
    request: SearchRequest,
    request_id: str,
    emitter
):
    """Execute enhanced search while emitting progress events"""
    from agentic import events

    try:
        orchestrator = await get_enhanced_orchestrator()

        # Emit start event with enhanced marker
        await emitter.emit(events.search_started(
            request_id=request_id,
            query=request.query,
            max_iterations=request.max_iterations,
            enhanced=True
        ))

        # Execute the enhanced search
        response = await orchestrator.search(request)

        # Emit completion event with enhanced stats
        await emitter.emit(events.search_completed(
            request_id=request_id,
            response=response,
            enhanced=True
        ))

        await emitter.emit(None)  # Signal end of stream

    except Exception as e:
        logger.error(f"[{request_id}] Enhanced streaming search failed: {e}", exc_info=True)
        await emitter.emit(events.search_failed(
            request_id=request_id,
            error=str(e),
            enhanced=True
        ))
        await emitter.emit(None)


# =============================================================================
# UNIFIED ORCHESTRATOR ENDPOINTS (DEPRECATED - Use /universal instead)
# =============================================================================

async def get_unified_orchestrator_instance() -> UniversalOrchestrator:
    """
    DEPRECATED: Use get_universal_orchestrator('enhanced') instead.

    UnifiedOrchestrator features are available via the 'enhanced' preset:
    - enable_hyde=True
    - enable_hybrid_reranking=True
    - enable_ragas=True
    - enable_entity_tracking=True
    """
    import warnings
    warnings.warn(
        "get_unified_orchestrator_instance() is deprecated. Use get_universal_orchestrator('enhanced') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return await get_universal_orchestrator("enhanced")


@router.post("/unified", response_model=SearchResponse)
async def unified_agentic_search(request: SearchRequest):
    """
    Execute UNIFIED agentic search integrating ALL advanced features.

    This endpoint uses the UnifiedOrchestrator which combines:

    **Query Enhancement:**
    - HyDE Query Expansion: Generates hypothetical documents for better retrieval
    - Entity Tracking: Extracts and tracks entities across sources

    **Search & Retrieval:**
    - SearXNG/DuckDuckGo/Brave web search (existing)
    - BGE-M3 Hybrid Re-ranking: Dense + sparse fusion for better precision

    **Quality Assurance:**
    - CRAG Evaluation (existing): Pre-synthesis quality check
    - Self-RAG Reflection (existing): Post-synthesis quality check
    - RAGAS Evaluation: Faithfulness and relevancy scoring

    **Learning:**
    - ThoughtLibrary: Reuses successful reasoning patterns
    - ExperienceDistiller (existing): Learns from successful searches

    Expected improvements over /agentic:
    - 15-20% better confidence scores via RAGAS-blended scoring
    - Better recall via HyDE query expansion
    - Better precision via hybrid re-ranking
    - More consistent entity handling

    Args:
        request: SearchRequest with query and options

    Returns:
        SearchResponse with enhanced metadata including:
        - entities_extracted: Number of entities tracked
        - hyde_expansion: Whether HyDE was applied
        - hybrid_reranked: Whether hybrid re-ranking was applied
        - ragas_faithfulness: RAGAS faithfulness score
        - ragas_relevancy: RAGAS relevancy score
        - ragas_overall: RAGAS overall score
    """
    try:
        orchestrator = await get_unified_orchestrator_instance()
        response = await orchestrator.search(request)
        return response

    except Exception as e:
        logger.error(f"Unified search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unified search failed: {str(e)}"
        )


@router.get("/unified/stats")
async def get_unified_stats():
    """
    Get statistics from the unified agentic orchestrator.

    Returns stats on:
    - Features enabled (HyDE, hybrid, RAGAS, entity tracking, thought library)
    - Feature timing statistics (avg/min/max ms per feature)
    - Thought library templates and categories
    - Hybrid retriever document count
    """
    try:
        orchestrator = await get_unified_orchestrator_instance()
        stats = orchestrator.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "0.22.0"
            }
        }

    except Exception as e:
        logger.error(f"Get unified stats failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get unified stats: {str(e)}"
        )


# ============================================================================
# UNIVERSAL ORCHESTRATOR - Consolidates ALL orchestrators with feature flags
# ============================================================================
# NOTE: get_universal_orchestrator is defined at the top of this file (line ~101)
# This section contains only the API endpoints and models for the universal search.
# ============================================================================

class UniversalSearchRequest(BaseModel):
    """Request model for universal search."""
    query: str = Field(..., min_length=3, description="The search query (min 3 characters)")
    user_id: Optional[str] = None
    context: Optional[dict] = None
    max_iterations: int = 10
    search_mode: str = "adaptive"
    analyze_query: bool = True
    verification_level: str = "standard"
    cache_results: bool = True
    min_sources: int = 5
    max_sources: int = 25
    # Universal-specific
    preset: Optional[str] = None  # minimal, balanced, enhanced, research, full
    # Feature overrides (optional)
    enable_hyde: Optional[bool] = None
    enable_hybrid_reranking: Optional[bool] = None
    enable_ragas: Optional[bool] = None
    enable_entity_tracking: Optional[bool] = None
    enable_thought_library: Optional[bool] = None
    enable_pre_act_planning: Optional[bool] = None
    enable_parallel_execution: Optional[bool] = None


@router.post("/universal")
async def universal_search(request: UniversalSearchRequest):
    """
    Execute search with the UNIVERSAL orchestrator - consolidates ALL orchestrators.

    The Universal Orchestrator replaces:
    - /agentic (AgenticOrchestrator)
    - /enhanced (EnhancedAgenticOrchestrator)
    - /graph-enhanced (GraphEnhancedOrchestrator)
    - /unified (UnifiedOrchestrator)

    **Presets:**
    - `minimal`: Fast basic search, no enhancements
    - `balanced`: (default) Good quality/speed trade-off with core features
    - `enhanced`: All quality features (HyDE, hybrid, RAGAS, entities, thought library)
    - `research`: Thorough exploration with planning and parallel execution
    - `full`: Everything enabled (expensive but comprehensive)

    **Feature Overrides:**
    You can override individual features by passing enable_* parameters.

    **Example Request:**
    ```json
    {
      "query": "Compare FastAPI vs Django for REST APIs",
      "preset": "research",
      "enable_parallel_execution": true
    }
    ```

    Returns:
        SearchResponse with enhancement_metadata showing which features were used.
    """
    try:
        # Determine preset
        preset = request.preset or "balanced"

        # Get or create orchestrator
        from agentic import UniversalOrchestrator, OrchestratorPreset, FeatureConfig

        # Build feature overrides
        overrides = {}
        for field in ["enable_hyde", "enable_hybrid_reranking", "enable_ragas",
                      "enable_entity_tracking", "enable_thought_library",
                      "enable_pre_act_planning", "enable_parallel_execution"]:
            value = getattr(request, field, None)
            if value is not None:
                overrides[field] = value

        # If overrides provided, create a custom orchestrator
        if overrides:
            orchestrator = UniversalOrchestrator(
                ollama_url=get_settings().ollama_base_url,
                mcp_url=get_settings().mcp_url,
                brave_api_key=os.getenv("BRAVE_API_KEY"),
                preset=OrchestratorPreset(preset) if preset in [p.value for p in OrchestratorPreset] else OrchestratorPreset.BALANCED,
                db_path="/home/sparkone/sdd/Recovery_Bot/memOS/data",
                **overrides
            )
            await orchestrator.initialize()
        else:
            orchestrator = await get_universal_orchestrator(preset)

        # Convert to SearchRequest
        search_request = SearchRequest(
            query=request.query,
            user_id=request.user_id,
            context=request.context,
            max_iterations=request.max_iterations,
            search_mode=SearchMode(request.search_mode) if request.search_mode else SearchMode.ADAPTIVE,
            analyze_query=request.analyze_query,
            verification_level=VerificationLevel(request.verification_level) if request.verification_level else VerificationLevel.STANDARD,
            cache_results=request.cache_results,
            min_sources=request.min_sources,
            max_sources=request.max_sources
        )

        response = await orchestrator.search(search_request)
        return response

    except PydanticValidationError as e:
        # Pydantic validation errors (min_length, etc.)
        logger.warning(f"Universal search validation failed: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Validation error: {str(e)}"
        )
    except ValueError as e:
        # Other validation errors
        logger.warning(f"Universal search validation failed: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        error_msg = str(e)
        # Check if it's a validation error wrapped in another exception
        if "validation error" in error_msg.lower() or "string_too_short" in error_msg.lower():
            logger.warning(f"Universal search validation failed: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Validation error: {error_msg}"
            )
        logger.error(f"Universal search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Universal search failed: {error_msg}"
        )


@router.get("/universal/presets")
async def get_universal_presets():
    """
    Get available presets and their configurations.

    Returns a list of presets with their enabled features, allowing clients
    to understand the trade-offs between speed and quality.
    """
    from agentic import PRESET_CONFIGS, OrchestratorPreset

    presets = {}
    for preset in OrchestratorPreset:
        config = PRESET_CONFIGS[preset]
        enabled_features = [
            field.replace("enable_", "")
            for field in dir(config)
            if field.startswith("enable_") and getattr(config, field)
        ]
        presets[preset.value] = {
            "name": preset.value,
            "description": {
                "minimal": "Fast basic search, no enhancements",
                "balanced": "Good quality/speed trade-off",
                "enhanced": "All quality features enabled",
                "research": "Thorough multi-direction exploration",
                "full": "Everything enabled (expensive)"
            }.get(preset.value, ""),
            "enabled_features": enabled_features,
            "feature_count": len(enabled_features)
        }

    return {
        "success": True,
        "data": {
            "presets": presets,
            "default": "balanced",
            "recommended": {
                "quick_lookup": "minimal",
                "general_search": "balanced",
                "quality_research": "research",
                "deep_analysis": "full"
            }
        },
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "0.24.0"
        }
    }


@router.get("/universal/stats")
async def get_universal_stats(preset: str = "balanced"):
    """
    Get statistics from a universal orchestrator instance.

    Args:
        preset: Which preset's stats to return (default: balanced)

    Returns:
        Statistics including:
        - features_enabled: List of enabled features
        - total_searches: Number of searches performed
        - cache_hits: Number of cache hits
        - feature_timings: Per-feature timing statistics
    """
    try:
        orchestrator = await get_universal_orchestrator(preset)
        stats = orchestrator.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "0.24.0",
                "preset": preset
            }
        }

    except Exception as e:
        logger.error(f"Get universal stats failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get universal stats: {str(e)}"
        )


@router.post("/universal/stream")
async def streaming_universal_search(request: UniversalSearchRequest):
    """
    Execute UNIVERSAL search with real-time SSE progress updates.

    This is the streaming version of the /universal endpoint with 40+ features
    and 5 configurable presets.

    **Presets:**
    - `minimal`: Fast basic search (4 features)
    - `balanced`: (default) Good quality/speed trade-off (13 features)
    - `enhanced`: All quality features (23 features)
    - `research`: Thorough exploration (31 features)
    - `full`: Everything enabled (38 features)

    **SSE Events emitted:**
    - search_started: Search has begun
    - analyzing_query: LLM is analyzing the query
    - planning_search: Creating search strategy
    - searching: Executing web searches
    - scraping_url: Scraping content from a URL
    - verifying_claims: Cross-checking facts
    - synthesizing: Combining results
    - search_completed: Final results

    See /universal endpoint for full feature description.
    """
    import uuid

    request_id = str(uuid.uuid4())
    preset = request.preset or "balanced"
    logger.info(f"Universal streaming search [{request_id}] (preset={preset}): {request.query[:50]}...")

    # Create event emitter for this search
    event_manager = get_event_manager()
    emitter = event_manager.create_emitter(request_id)

    async def generate_events():
        """Generator for SSE events"""
        queue = emitter.subscribe()

        try:
            # Start the universal search in a background task
            search_task = asyncio.create_task(
                _execute_universal_streaming_search(request, request_id, emitter)
            )

            # Stream events until search completes
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120.0)  # Longer timeout for full preset

                    if event is None:  # End of stream signal
                        break

                    yield event.to_sse()

                    # If this is the final event, break
                    if event.event_type in [EventType.SEARCH_COMPLETED, EventType.SEARCH_FAILED]:
                        break

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"

            # Wait for search to finish
            await search_task

        except asyncio.CancelledError:
            logger.info(f"[{request_id}] Universal stream cancelled by client")
        except Exception as e:
            logger.error(f"[{request_id}] Universal stream error: {e}")
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            emitter.unsubscribe(queue)
            event_manager.remove_emitter(request_id)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Request-Id": request_id,
            "X-Universal": "true",  # Marker for universal endpoint
            "X-Preset": preset
        }
    )


async def _execute_universal_streaming_search(
    request: UniversalSearchRequest,
    request_id: str,
    emitter
):
    """Execute the Universal orchestrator search with event emission."""
    from agentic import UniversalOrchestrator, OrchestratorPreset
    from agentic.events import SearchEvent
    import time

    preset = request.preset or "balanced"

    try:
        # Emit search started
        await emitter.emit(SearchEvent(
            event_type=EventType.SEARCH_STARTED,
            request_id=request_id,
            message=f"Starting universal search (preset={preset})",
            progress_percent=0,
            data={"query": request.query, "preset": preset}
        ))

        # Emit analyzing event
        await emitter.emit(SearchEvent(
            event_type=EventType.ANALYZING_QUERY,
            request_id=request_id,
            message="Analyzing query with Universal orchestrator",
            progress_percent=5,
            data={"query": request.query}
        ))

        # Build feature overrides
        overrides = {}
        for field in ["enable_hyde", "enable_hybrid_reranking", "enable_ragas",
                      "enable_entity_tracking", "enable_thought_library",
                      "enable_pre_act_planning", "enable_parallel_execution"]:
            value = getattr(request, field, None)
            if value is not None:
                overrides[field] = value

        # Create or get orchestrator
        if overrides:
            orchestrator = UniversalOrchestrator(
                ollama_url=get_settings().ollama_base_url,
                mcp_url=get_settings().mcp_url,
                brave_api_key=os.getenv("BRAVE_API_KEY"),
                preset=OrchestratorPreset(preset) if preset in [p.value for p in OrchestratorPreset] else OrchestratorPreset.BALANCED,
                db_path="/home/sparkone/sdd/Recovery_Bot/memOS/data",
                **overrides
            )
            await orchestrator.initialize()
        else:
            orchestrator = await get_universal_orchestrator(preset)

        # Pass the event emitter to the orchestrator so it can emit detailed phase events
        orchestrator.set_event_emitter(emitter)

        # Emit planning event
        await emitter.emit(SearchEvent(
            event_type=EventType.PLANNING_SEARCH,
            request_id=request_id,
            message=f"Planning search with {preset} preset",
            progress_percent=10,
            data={"preset": preset, "features_enabled": orchestrator.get_stats().get("features_enabled", [])}
        ))

        # Convert to SearchRequest
        search_request = SearchRequest(
            query=request.query,
            user_id=request.user_id,
            context=request.context,
            max_iterations=request.max_iterations,
            search_mode=SearchMode(request.search_mode) if request.search_mode else SearchMode.ADAPTIVE,
            analyze_query=request.analyze_query,
            verification_level=VerificationLevel(request.verification_level) if request.verification_level else VerificationLevel.STANDARD,
            cache_results=request.cache_results,
            min_sources=request.min_sources,
            max_sources=request.max_sources
        )

        # Check which search engines are available
        engines = []
        if await orchestrator.searcher.searxng.check_availability():
            engines.append("SearXNG")
        else:
            if orchestrator.searcher.duckduckgo:
                engines.append("DuckDuckGo")
            if orchestrator.searcher.brave and orchestrator.searcher.brave.available:
                engines.append("Brave")
            if not engines:
                engines.append("DuckDuckGo")

        # Emit searching event with engines info
        await emitter.emit(SearchEvent(
            event_type=EventType.SEARCHING,
            request_id=request_id,
            message=f"Searching via {', '.join(engines)}",
            progress_percent=20,
            engines=engines,
            data={"queries": [request.query]}
        ))

        # Execute search (the orchestrator handles all phases internally)
        start_time = time.time()
        response = await orchestrator.search(search_request)
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Get stats for completion event
        stats = orchestrator.get_stats()
        sources_count = len(response.data.sources) if response.data and response.data.sources else 0
        confidence = response.data.confidence_score if response.data else 0.0

        # Emit synthesizing event
        await emitter.emit(SearchEvent(
            event_type=EventType.SYNTHESIZING,
            request_id=request_id,
            message="Synthesizing results",
            progress_percent=90,
            data={"sources_count": sources_count}
        ))

        # Emit completion
        await emitter.emit(SearchEvent(
            event_type=EventType.SEARCH_COMPLETED,
            request_id=request_id,
            message="Universal search completed",
            progress_percent=100,
            data={
                "response": response.dict() if hasattr(response, 'dict') else {
                    "success": response.success,
                    "data": response.data.dict() if hasattr(response.data, 'dict') else None,
                    "meta": response.meta.dict() if hasattr(response.meta, 'dict') else None
                },
                "preset": preset,
                "features_used": stats.get("features_enabled", []),
                "sources_count": sources_count,
                "confidence_score": confidence,
                "execution_time_ms": execution_time_ms
            }
        ))

    except Exception as e:
        logger.error(f"[{request_id}] Universal search failed: {e}", exc_info=True)
        await emitter.emit(SearchEvent(
            event_type=EventType.SEARCH_FAILED,
            request_id=request_id,
            message=f"Search failed: {str(e)}",
            progress_percent=0,
            data={"error": str(e)}
        ))


@router.post("/stream")
async def streaming_agentic_search(request: SearchRequest):
    """
    Execute agentic search with real-time SSE progress updates.

    This endpoint returns a Server-Sent Events stream that provides
    real-time updates as the search progresses:

    Events emitted:
    - search_started: Search has begun
    - analyzing_query: LLM is analyzing the query
    - planning_search: Creating search strategy
    - searching: Executing web searches
    - scraping_url: Scraping content from a URL
    - verifying_claims: Cross-checking facts
    - synthesizing: Combining results
    - search_completed: Final results

    Each event contains:
    - event: Event type
    - request_id: Unique search ID
    - message: Human-readable status
    - progress: 0-100 percentage
    - Additional context-specific data

    The final event (search_completed or search_failed) includes
    the full search response in the data field.

    Usage with JavaScript:
    ```javascript
    const eventSource = new EventSource('/api/v1/search/stream?query=...');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(data.message, data.progress + '%');
    };
    ```
    """
    import uuid

    request_id = str(uuid.uuid4())
    logger.info(f"Streaming search request [{request_id}]: {request.query[:50]}...")

    # Create event emitter for this search
    event_manager = get_event_manager()
    emitter = event_manager.create_emitter(request_id)

    async def generate_events():
        """Generator for SSE events"""
        queue = emitter.subscribe()

        try:
            # Start the search in a background task
            search_task = asyncio.create_task(
                _execute_streaming_search(request, request_id, emitter)
            )

            # Stream events until search completes
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)

                    if event is None:  # End of stream signal
                        break

                    yield event.to_sse()

                    # If this is the final event, break
                    if event.event_type in [EventType.SEARCH_COMPLETED, EventType.SEARCH_FAILED]:
                        break

                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"

            # Wait for search to finish
            await search_task

        except asyncio.CancelledError:
            logger.info(f"[{request_id}] Stream cancelled by client")
        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}")
            yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            emitter.unsubscribe(queue)
            event_manager.remove_emitter(request_id)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "X-Request-Id": request_id
        }
    )


async def _execute_streaming_search(
    request: SearchRequest,
    request_id: str,
    emitter
):
    """Execute search while emitting progress events"""
    from agentic import events

    try:
        orchestrator = await get_orchestrator()

        # Emit start event
        await emitter.emit(events.search_started(
            request_id=request_id,
            query=request.query,
            max_iterations=request.max_iterations
        ))

        # Execute search with event emitter
        response = await orchestrator.search_with_events(request, emitter)

        # Emit completion event with full response
        await emitter.emit(SearchEvent(
            event_type=EventType.SEARCH_COMPLETED,
            request_id=request_id,
            message=f"Search complete: {len(response.data.sources)} sources",
            progress_percent=100,
            sources_count=len(response.data.sources),
            data={
                "response": response.model_dump(),
                "execution_time_ms": response.meta.execution_time_ms
            }
        ))

    except Exception as e:
        logger.error(f"[{request_id}] Streaming search failed: {e}", exc_info=True)
        await emitter.emit(events.search_failed(request_id, str(e)))

    finally:
        emitter.close()


@router.get("/events/{request_id}")
async def get_search_events(request_id: str):
    """
    Subscribe to events for an existing search.

    Returns an SSE stream for real-time updates on a search in progress.
    Use this to reconnect to a search if the original connection was lost.

    Args:
        request_id: The search request ID from X-Request-Id header

    Returns:
        Server-Sent Events stream
    """
    event_manager = get_event_manager()
    emitter = event_manager.get_emitter(request_id)

    if not emitter:
        raise HTTPException(
            status_code=404,
            detail=f"Search {request_id} not found or already completed"
        )

    async def generate_events():
        queue = emitter.subscribe()

        try:
            # First, send event history
            for event in emitter.get_history():
                yield event.to_sse()

            # Then stream new events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=60.0)

                    if event is None:
                        break

                    yield event.to_sse()

                    if event.event_type in [EventType.SEARCH_COMPLETED, EventType.SEARCH_FAILED]:
                        break

                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            emitter.unsubscribe(queue)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/active")
async def get_active_searches():
    """
    Get list of currently active searches.

    Returns request IDs of searches currently in progress.
    """
    event_manager = get_event_manager()

    return {
        "success": True,
        "data": {
            "active_searches": event_manager.get_active_request_ids(),
            "count": event_manager.active_searches
        }
    }


@router.post("/simple")
async def simple_search(
    query: str = Query(..., min_length=3, description="Search query"),
    max_results: int = Query(default=5, ge=1, le=20, description="Maximum results")
):
    """
    Execute simple web search without synthesis.

    Returns raw search results for cases where the full
    agentic pipeline is not needed.

    Args:
        query: Search query string
        max_results: Maximum number of results (1-20)

    Returns:
        List of search results with title, url, snippet
    """
    logger.info(f"Simple search: {query[:50]}...")

    try:
        orchestrator = await get_orchestrator()
        results = await orchestrator.simple_search(query, max_results)

        return {
            "success": True,
            "data": {
                "query": query,
                "results": results,
                "count": len(results)
            }
        }

    except Exception as e:
        logger.error(f"Simple search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/quick")
async def quick_search(
    query: str = Query(..., min_length=3),
    user_id: Optional[str] = Query(default=None)
):
    """
    Quick agentic search with adaptive mode.

    Convenience endpoint that uses:
    - ADAPTIVE search mode (continues until information is sufficient)
    - Query analysis enabled
    - Standard verification
    - Result caching
    """
    request = SearchRequest(
        query=query,
        user_id=user_id,
        search_mode=SearchMode.ADAPTIVE,
        analyze_query=True,
        max_iterations=5,
        min_sources=3,
        max_sources=10,
        verification_level=VerificationLevel.STANDARD,
        cache_results=True
    )

    return await agentic_search(request)


@router.post("/exhaustive")
async def exhaustive_search(
    query: str = Query(..., min_length=3),
    user_id: Optional[str] = Query(default=None),
    max_sources: int = Query(default=20, ge=5, le=50)
):
    """
    Exhaustive agentic search that explores all leads.

    This endpoint searches as many websites as necessary to find
    comprehensive information, continuing until:
    - All search leads are exhausted, OR
    - max_sources limit is reached

    Use this for complex queries requiring thorough research.
    """
    request = SearchRequest(
        query=query,
        user_id=user_id,
        search_mode=SearchMode.EXHAUSTIVE,
        analyze_query=True,
        max_iterations=15,  # Higher limit for exhaustive
        min_sources=5,
        max_sources=max_sources,
        verification_level=VerificationLevel.STRICT,  # Strict for thorough verification
        cache_results=True
    )

    return await agentic_search(request)


@router.post("/deep")
async def deep_search(
    query: str = Query(..., min_length=3, description="The question to research"),
    max_sources: int = Query(default=5, ge=1, le=10, description="Max sources to scrape"),
    include_pdfs: bool = Query(default=True, description="Whether to download and analyze PDFs")
):
    """
    Execute deep search with content scraping and LLM analysis.

    This endpoint goes beyond basic search by:
    1. Finding relevant sources via web search
    2. Scraping full content from top sources (including PDFs)
    3. Using the smartest available reasoning model to analyze content
    4. Extracting specific answers with key findings

    Use this for complex technical questions that require reading documentation.

    Returns:
        - answer: Detailed answer from the reasoning model
        - key_findings: Bullet points of key information found
        - sources_used: URLs that were scraped and analyzed
        - confidence: HIGH/MEDIUM/LOW confidence rating
        - limitations: What wasn't found or unclear
    """
    logger.info(f"Deep search request: {query[:50]}...")

    try:
        orchestrator = await get_orchestrator()
        result = await orchestrator.deep_search(
            query=query,
            max_sources_to_scrape=max_sources,
            include_pdfs=include_pdfs
        )

        return {
            "success": result.get("success", False),
            "data": {
                "answer": result.get("answer", ""),
                "key_findings": result.get("key_findings", []),
                "confidence": result.get("confidence", 0.0),
                "limitations": result.get("limitations", ""),
                "sources_used": result.get("sources_used", []),
                "model_used": result.get("model_used", "unknown")
            },
            "meta": {
                "sources_scraped": result.get("sources_scraped", 0),
                "search_results_count": result.get("search_results_count", 0),
                "execution_time_ms": result.get("execution_time_ms", 0)
            }
        }

    except Exception as e:
        logger.error(f"Deep search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Deep search failed: {str(e)}"
        )


@router.delete("/cache")
async def clear_cache():
    """
    Clear the search result cache.

    Useful for forcing fresh searches or after
    configuration changes.
    """
    try:
        orchestrator = await get_orchestrator()
        orchestrator.clear_cache()

        # Also clear content cache (Phase 2)
        from agentic.content_cache import get_content_cache
        content_cache = get_content_cache()
        content_cache.clear()

        return {
            "success": True,
            "message": "Search cache and content cache cleared"
        }

    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics.

    Returns stats for both content cache (scraped pages)
    and query result cache.

    Phase 2 Optimization: Provides visibility into cache effectiveness.
    """
    try:
        from agentic.content_cache import get_content_cache
        content_cache = get_content_cache()
        stats = content_cache.get_stats()

        # Also cleanup expired entries
        content_removed, query_removed = content_cache.cleanup_expired()

        return {
            "success": True,
            "data": {
                **stats,
                "cleanup": {
                    "content_removed": content_removed,
                    "query_removed": query_removed
                }
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "2.0.0"
            }
        }

    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.get("/ttl/stats")
async def get_ttl_stats():
    """
    Get TTL cache manager statistics.

    Returns per-tool latency statistics and active pin information.
    This enables monitoring of tool call durations and cache pinning effectiveness.

    Phase 3 Optimization: TTL-based KV cache pinning (Continuum-inspired).
    """
    try:
        from agentic.ttl_cache_manager import get_ttl_cache_manager
        ttl_manager = get_ttl_cache_manager()
        stats = ttl_manager.get_all_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "description": "TTL-based KV cache pinning statistics"
            }
        }

    except Exception as e:
        logger.error(f"TTL stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get TTL stats: {str(e)}"
        )


@router.get("/metrics")
async def get_performance_metrics_endpoint():
    """
    Get performance metrics for the agentic search pipeline.

    Returns comprehensive statistics including:
    - Query response times (TTFT, total, synthesis)
    - Cache hit rates
    - Token usage and savings
    - Tool latencies
    - Recent query history

    Phase 2 Optimization: Performance metrics tracking.
    """
    try:
        from agentic.metrics import get_performance_metrics
        metrics = get_performance_metrics()
        summary = metrics.get_summary()

        return {
            "success": True,
            "data": summary,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "description": "Agentic search performance metrics"
            },
            "errors": []
        }

    except Exception as e:
        logger.error(f"Performance metrics failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get("/artifacts/stats")
async def get_artifacts_stats():
    """
    Get artifact store statistics.

    Returns information about stored artifacts used for
    reducing token transfer between agents.

    Phase 2.3 Optimization: Artifact-based communication.
    """
    try:
        from agentic.artifacts import get_artifact_store
        store = get_artifact_store()
        stats = store.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "description": "Artifact store statistics"
            }
        }

    except Exception as e:
        logger.error(f"Artifacts stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get artifacts stats: {str(e)}"
        )


@router.delete("/artifacts/{session_id}")
async def cleanup_session_artifacts(session_id: str):
    """
    Clean up artifacts for a specific session.

    Use this to free up disk space after a search session is complete.
    """
    try:
        from agentic.artifacts import get_artifact_store
        store = get_artifact_store()
        store.cleanup_session(session_id)

        return {
            "success": True,
            "data": {"session_id": session_id, "status": "cleaned"},
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Artifact cleanup failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup artifacts: {str(e)}"
        )


async def get_multi_orchestrator() -> MultiAgentOrchestrator:
    """Get or create the multi-agent orchestrator instance"""
    global _multi_orchestrator, _orchestrator
    if _multi_orchestrator is None:
        import os
        # Ensure main orchestrator is initialized first (we need its scraper/searcher)
        orchestrator = await get_orchestrator()

        _multi_orchestrator = MultiAgentOrchestrator(
            ollama_url=get_settings().ollama_base_url,
            scraper=orchestrator.scraper,
            searcher=orchestrator.searcher,
            max_iterations=5
        )
        logger.info("Multi-agent orchestrator initialized")
    return _multi_orchestrator


@router.post("/multi")
async def multi_agent_search(
    query: str = Query(..., min_length=3, description="The question to research"),
    max_iterations: int = Query(default=5, ge=1, le=10, description="Maximum refinement iterations")
):
    """
    Execute multi-agent search with smart planning and iterative refinement.

    This is the most intelligent search mode, using:
    1. **Smart Planner (14B+ model)**: Analyzes query and creates optimal task plan
    2. **Sub-Agents (smaller models)**: Execute search/scrape/analyze tasks in parallel when VRAM permits
    3. **Smart Evaluator (14B+ model)**: Reviews results and decides if more work needed
    4. **Iterative Refinement**: Continues until answer found or max iterations reached

    Use this for complex questions that may require:
    - Multiple search queries from different angles
    - Scraping and analyzing full document content
    - Cross-verification of findings
    - Synthesis across multiple sources

    Returns:
        - answer: Comprehensive answer from the evaluator
        - key_findings: Important facts discovered
        - confidence: 0.0-1.0 confidence rating
        - sources: All URLs consulted
        - iterations: How many refinement cycles were needed
        - models_used: Which models were employed
        - trace: Full execution trace for debugging
    """
    logger.info(f"Multi-agent search request: {query[:50]}...")

    try:
        multi_orch = await get_multi_orchestrator()

        # Set max_iterations before search
        multi_orch.max_iterations = max_iterations

        result = await multi_orch.search(
            query=query,
            context=None
        )

        return {
            "success": result.get("success", False),
            "data": {
                "answer": result.get("answer", ""),
                "key_findings": result.get("key_findings", []),
                "confidence": result.get("confidence", 0.0),
                "missing_info": result.get("missing_info", ""),
                "sources": result.get("sources", [])
            },
            "meta": {
                "iterations": result.get("iterations", 0),
                "tasks_executed": result.get("tasks_executed", 0),
                "execution_time_ms": result.get("execution_time_ms", 0),
                "models_used": result.get("models_used", [])
            },
            "trace": result.get("trace", [])
        }

    except Exception as e:
        logger.error(f"Multi-agent search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Multi-agent search failed: {str(e)}"
        )


@router.get("/health")
async def search_health():
    """
    Health check for search service.

    Returns status of search providers and agents.
    """
    try:
        orchestrator = await get_orchestrator()

        # Check agent availability
        mcp_available = await orchestrator.planner.check_mcp_available()
        brave_available = orchestrator.searcher.brave.available

        return {
            "status": "healthy",
            "providers": {
                "mcp_node_editor": "available" if mcp_available else "unavailable",
                "brave_search": "available" if brave_available else "unavailable",
                "duckduckgo": "available",  # Always available as fallback
                "ollama": "available"  # Assumed available if server is running
            },
            "cache_size": len(orchestrator._cache)
        }

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# ========================================
# PHASE 4: MEMORY TIER ENDPOINTS
# ========================================

@router.get("/memory/tiers/stats")
async def get_memory_tier_stats():
    """
    Get comprehensive memory tier statistics.

    Returns information about cold/warm tier usage, hit rates,
    promotions, and demotions.

    Phase 4 Optimization: Three-tier memory architecture (MemOS MemCube).
    """
    try:
        from agentic.memory_tiers import get_memory_tier_manager
        manager = get_memory_tier_manager()
        stats = manager.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "description": "Three-tier memory statistics"
            }
        }

    except Exception as e:
        logger.error(f"Memory tier stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get memory tier stats: {str(e)}"
        )


@router.get("/memory/kv-cache/stats")
async def get_kv_cache_stats():
    """
    Get KV cache service statistics.

    Returns information about cached prefixes, warm entries,
    hit rates, and warm times.

    Phase 4 Optimization: KV cache service for activation memory.
    """
    try:
        from agentic.kv_cache_service import get_kv_cache_service
        service = get_kv_cache_service()
        stats = service.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0.0",
                "description": "KV cache service statistics"
            }
        }

    except Exception as e:
        logger.error(f"KV cache stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get KV cache stats: {str(e)}"
        )


@router.get("/memory/kv-cache/warm")
async def get_warm_entries():
    """
    Get list of currently warm (cached) entries.

    Returns details about entries in the activation memory tier.
    """
    try:
        from agentic.kv_cache_service import get_kv_cache_service
        service = get_kv_cache_service()
        entries = service.get_warm_entries()

        return {
            "success": True,
            "data": {
                "entries": entries,
                "count": len(entries)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get warm entries failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get warm entries: {str(e)}"
        )


@router.post("/memory/kv-cache/warm")
async def warm_prefix(
    prefix: str,
    model: Optional[str] = None,
    content_id: Optional[str] = None
):
    """
    Manually warm a prefix in the KV cache.

    This precomputes the KV cache for the given prefix,
    reducing TTFT on subsequent use.

    Args:
        prefix: The text prefix to warm
        model: Optional model to use (defaults to llama3.2:3b)
        content_id: Optional custom ID for this cache entry
    """
    try:
        from agentic.kv_cache_service import get_kv_cache_service
        service = get_kv_cache_service()

        cache_id = await service.warm_prefix(
            prefix=prefix,
            model=model,
            content_id=content_id
        )

        return {
            "success": True,
            "data": {
                "cache_id": cache_id,
                "status": "warm"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Warm prefix failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to warm prefix: {str(e)}"
        )


@router.post("/memory/tiers/store")
async def store_in_memory_tier(
    content_id: str,
    content: str,
    content_type: str = "context_chunk",
    promote: bool = False
):
    """
    Store content in the memory tier system.

    Content starts in cold tier (plaintext storage).
    Set promote=True to immediately warm the KV cache.

    Args:
        content_id: Unique identifier for the content
        content: The text content to store
        content_type: Type of content (system_prompt, context_chunk, etc.)
        promote: Whether to immediately promote to warm tier
    """
    try:
        from agentic.memory_tiers import (
            get_memory_tier_manager,
            ContentType,
            MemoryTier
        )

        manager = get_memory_tier_manager()

        # Parse content type
        try:
            ctype = ContentType(content_type)
        except ValueError:
            ctype = ContentType.CONTEXT_CHUNK

        entry = await manager.store(
            content_id=content_id,
            content=content,
            content_type=ctype,
            initial_tier=MemoryTier.WARM if promote else MemoryTier.COLD
        )

        return {
            "success": True,
            "data": entry.to_dict(),
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Store in memory tier failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store content: {str(e)}"
        )


@router.get("/memory/tiers/{content_id}")
async def get_from_memory_tier(content_id: str):
    """
    Retrieve content from the memory tier system.

    Checks warm tier first, falls back to cold tier.
    Tracks access for automatic promotion.
    """
    try:
        from agentic.memory_tiers import get_memory_tier_manager

        manager = get_memory_tier_manager()
        result = await manager.get_context(content_id)

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Content not found: {content_id}"
            )

        return {
            "success": True,
            "data": result,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get from memory tier failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get content: {str(e)}"
        )


@router.post("/memory/tiers/{content_id}/promote")
async def promote_to_warm(content_id: str):
    """
    Manually promote content from cold to warm tier.

    This precomputes the KV cache for the content.
    """
    try:
        from agentic.memory_tiers import get_memory_tier_manager

        manager = get_memory_tier_manager()
        success = await manager.promote_to_warm(content_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Content not found or already warm: {content_id}"
            )

        return {
            "success": True,
            "data": {
                "content_id": content_id,
                "status": "promoted to warm"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Promote to warm failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to promote content: {str(e)}"
        )


@router.post("/memory/tiers/{content_id}/demote")
async def demote_to_cold(content_id: str):
    """
    Demote content from warm back to cold tier.

    This marks the KV cache as evicted.
    """
    try:
        from agentic.memory_tiers import get_memory_tier_manager

        manager = get_memory_tier_manager()
        success = await manager.demote_to_cold(content_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Content not found or not warm: {content_id}"
            )

        return {
            "success": True,
            "data": {
                "content_id": content_id,
                "status": "demoted to cold"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demote to cold failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to demote content: {str(e)}"
        )


@router.post("/memory/initialize")
async def initialize_memory_system():
    """
    Initialize the memory tier system and warm system prompts.

    This should be called at server startup to pre-warm
    commonly used prompts.
    """
    try:
        from agentic.memory_tiers import initialize_memory_tiers

        manager = await initialize_memory_tiers()
        stats = manager.get_stats()

        return {
            "success": True,
            "data": {
                "status": "initialized",
                "stats": stats
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Initialize memory system failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize memory system: {str(e)}"
        )


# =============================================================================
# Graph-Based Cache Endpoints (KVFlow + ROG inspired)
# =============================================================================

@router.get("/graph/stats")
async def get_graph_cache_stats():
    """
    Get comprehensive statistics from the graph-based cache system.

    Returns statistics from:
    - Agent Step Graph: Workflow-aware cache management
    - Scratchpad Cache: Intermediate answer caching (ROG-style)
    - Workflow tracking: Active workflows and transitions
    """
    try:
        from agentic.graph_cache_integration import get_graph_cache_integration

        integration = get_graph_cache_integration()
        stats = integration.get_comprehensive_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get graph cache stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph cache stats: {str(e)}"
        )


@router.get("/graph/agent-step-graph")
async def get_agent_step_graph_stats():
    """
    Get Agent Step Graph statistics including transition probabilities.

    The Agent Step Graph tracks:
    - Agent transitions and their probabilities
    - Steps-to-execution (STE) values for cache eviction
    - Agent execution statistics (duration, token count)
    """
    try:
        from agentic.agent_step_graph import get_agent_step_graph

        graph = get_agent_step_graph()
        stats = graph.get_graph_stats()

        # Add visualization
        stats['visualization'] = graph.visualize_graph()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get agent step graph failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent step graph: {str(e)}"
        )


@router.get("/graph/scratchpad-cache")
async def get_scratchpad_cache_stats():
    """
    Get Scratchpad Cache statistics.

    The Scratchpad Cache tracks:
    - Finding cache: Deduplicated scraped content
    - Sub-query cache: Intermediate reasoning results (ROG-style)
    - Mission cache: Query decomposition patterns
    """
    try:
        from agentic.scratchpad_cache import get_scratchpad_cache

        cache = get_scratchpad_cache()
        stats = cache.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get scratchpad cache stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get scratchpad cache stats: {str(e)}"
        )


@router.get("/graph/eviction-candidates")
async def get_eviction_candidates(memory_pressure: float = Query(0.8, ge=0.0, le=1.0)):
    """
    Get agents whose KV cache should be evicted based on steps-to-execution.

    Args:
        memory_pressure: Current memory pressure level (0.0-1.0)

    Returns:
        List of agent types that are candidates for cache eviction
    """
    try:
        from agentic.graph_cache_integration import get_graph_cache_integration

        integration = get_graph_cache_integration()
        candidates = integration.get_eviction_candidates(memory_pressure)

        return {
            "success": True,
            "data": {
                "memory_pressure": memory_pressure,
                "eviction_candidates": [c.value for c in candidates],
                "candidate_count": len(candidates)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get eviction candidates failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get eviction candidates: {str(e)}"
        )


@router.delete("/graph/scratchpad-cache")
async def clear_scratchpad_cache():
    """
    Clear all scratchpad caches (finding, sub-query, and mission caches).
    """
    try:
        from agentic.scratchpad_cache import get_scratchpad_cache

        cache = get_scratchpad_cache()
        cache.clear_all()

        return {
            "success": True,
            "data": {
                "status": "cleared",
                "message": "All scratchpad caches cleared"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Clear scratchpad cache failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear scratchpad cache: {str(e)}"
        )


@router.post("/graph/initialize")
async def initialize_graph_cache():
    """
    Initialize the graph-based cache system.

    This should be called at server startup to:
    1. Load cached data from SQLite
    2. Initialize agent step graph
    3. Pre-warm high-priority prompt templates
    """
    try:
        from agentic.graph_cache_integration import initialize_graph_cache as init_graph

        import os
        ollama_url = get_settings().ollama_base_url

        integration = await init_graph(ollama_url)
        stats = integration.get_comprehensive_stats()

        return {
            "success": True,
            "data": {
                "status": "initialized",
                "stats": stats
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Initialize graph cache failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize graph cache: {str(e)}"
        )


@router.get("/graph/prefix-estimate")
async def estimate_prefix_reuse(
    agent_type: str = Query(..., description="Agent type (analyzer, planner, etc.)"),
    scratchpad_size: int = Query(0, ge=0, description="Number of items in scratchpad")
):
    """
    Estimate KV cache reuse potential for a prompt configuration.

    Useful for understanding cache efficiency before sending requests.
    """
    try:
        from agentic.prefix_optimized_prompts import estimate_prefix_reuse as estimate

        valid_agents = ['analyzer', 'planner', 'searcher', 'scraper', 'verifier', 'synthesizer']
        if agent_type.lower() not in valid_agents:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type. Must be one of: {valid_agents}"
            )

        estimate_data = estimate(agent_type.lower(), scratchpad_size)

        return {
            "success": True,
            "data": estimate_data,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Estimate prefix reuse failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to estimate prefix reuse: {str(e)}"
        )


# =============================================================================
# Experience Distillation Endpoints (MetaAgent-inspired)
# =============================================================================

@router.get("/distillation/stats")
async def get_distillation_stats():
    """
    Get experience distillation statistics.

    Returns information about captured experiences, distillation attempts,
    and templates created from successful searches.
    """
    try:
        from agentic import get_experience_distiller

        distiller = get_experience_distiller()
        stats = distiller.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get distillation stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get distillation stats: {str(e)}"
        )


@router.get("/distillation/experiences")
async def get_experiences(
    query_type: Optional[str] = Query(None, description="Filter by query type")
):
    """
    Get captured search experiences.

    These are successful searches that have been captured for potential
    distillation into reusable templates.
    """
    try:
        from agentic import get_experience_distiller

        distiller = get_experience_distiller()
        experiences = distiller.get_experiences(query_type)

        return {
            "success": True,
            "data": {
                "experiences": experiences,
                "count": len(experiences)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_type_filter": query_type
            }
        }

    except Exception as e:
        logger.error(f"Get experiences failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get experiences: {str(e)}"
        )


@router.post("/distillation/distill")
async def trigger_distillation(
    query_type: str = Query(..., description="Query type to distill")
):
    """
    Trigger distillation for a specific query type.

    Analyzes captured experiences and attempts to extract a reusable
    template for the ThoughtLibrary.
    """
    try:
        from agentic import get_experience_distiller

        distiller = get_experience_distiller()
        result = await distiller.attempt_distillation(query_type)

        return {
            "success": True,
            "data": result.to_dict(),
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query_type": query_type
            }
        }

    except Exception as e:
        logger.error(f"Trigger distillation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger distillation: {str(e)}"
        )


@router.delete("/distillation/experiences")
async def clear_experiences(
    query_type: Optional[str] = Query(None, description="Query type to clear (all if not specified)")
):
    """
    Clear captured experiences.

    Args:
        query_type: If specified, only clear experiences of this type.
                   If not specified, clear all experiences.
    """
    try:
        from agentic import get_experience_distiller

        distiller = get_experience_distiller()
        count = distiller.clear_experiences(query_type)

        return {
            "success": True,
            "data": {
                "cleared_count": count,
                "query_type": query_type or "all"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Clear experiences failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear experiences: {str(e)}"
        )


# =============================================================================
# Phase 21: Meta-Buffer & Reasoning Composer Endpoints (Template Reuse)
# =============================================================================

@router.get("/meta-buffer/stats")
async def get_meta_buffer_stats():
    """
    Get Meta-Buffer (cross-session template) statistics.

    Returns information about stored templates, retrieval patterns,
    success rates, and template performance metrics.
    """
    try:
        from agentic.meta_buffer import get_meta_buffer

        buffer = get_meta_buffer()
        stats = buffer.get_stats()  # Sync method

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "21",
                "feature": "meta_buffer"
            }
        }

    except Exception as e:
        logger.error(f"Get meta-buffer stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get meta-buffer stats: {str(e)}"
        )


@router.get("/meta-buffer/templates")
async def get_meta_buffer_templates(
    limit: int = Query(20, ge=1, le=100, description="Maximum templates to return")
):
    """
    List stored templates in Meta-Buffer.

    Templates are reasoning patterns distilled from successful searches.
    """
    try:
        from agentic.meta_buffer import get_meta_buffer

        buffer = get_meta_buffer()
        templates = buffer.get_top_templates(limit=limit)  # Use existing method

        # Serialize templates to dicts
        template_dicts = [
            {
                "id": t.id,
                "template_type": t.template_type.value if hasattr(t.template_type, 'value') else str(t.template_type),
                "abstract_pattern": t.abstract_pattern[:200] + "..." if len(t.abstract_pattern) > 200 else t.abstract_pattern,
                "usage_count": t.usage_count,
                "success_count": t.success_count,
                "avg_confidence": t.avg_confidence,
                "success_rate": t.success_count / max(t.usage_count, 1)
            }
            for t in templates
        ]

        return {
            "success": True,
            "data": {
                "templates": template_dicts,
                "count": len(template_dicts)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "limit": limit
            }
        }

    except Exception as e:
        logger.error(f"List meta-buffer templates failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list templates: {str(e)}"
        )


@router.get("/reasoning-composer/stats")
async def get_reasoning_composer_stats():
    """
    Get Reasoning Composer (Self-Discover) statistics.

    Returns information about reasoning module usage, composition patterns,
    and strategy effectiveness.
    """
    try:
        from agentic.reasoning_composer import get_reasoning_composer

        composer = get_reasoning_composer()
        stats = composer.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "21",
                "feature": "reasoning_composer"
            }
        }

    except Exception as e:
        logger.error(f"Get reasoning-composer stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get reasoning-composer stats: {str(e)}"
        )


@router.get("/reasoning-composer/modules")
async def get_reasoning_modules():
    """
    List available reasoning modules.

    Modules are atomic reasoning patterns (e.g., critical_analysis, step_by_step)
    that can be composed into task-specific strategies.
    """
    try:
        from agentic.reasoning_composer import get_reasoning_composer, ReasoningModule

        composer = get_reasoning_composer()
        # List all modules from MODULE_DEFINITIONS
        modules = [
            {
                "name": m.value,
                "description": d.description,
                "examples": d.examples[:2] if d.examples else []
            }
            for m, d in composer.MODULE_DEFINITIONS.items()
        ]

        return {
            "success": True,
            "data": {
                "modules": modules,
                "count": len(modules)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"List reasoning modules failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list modules: {str(e)}"
        )


@router.get("/phase21/summary")
async def get_phase21_summary():
    """
    Get comprehensive Phase 21 (Template Reuse Optimization) summary.

    Combines Meta-Buffer and Reasoning Composer statistics into a single view
    for debugging and performance monitoring.
    """
    try:
        from agentic.meta_buffer import get_meta_buffer
        from agentic.reasoning_composer import get_reasoning_composer

        buffer = get_meta_buffer()
        composer = get_reasoning_composer()

        buffer_stats = buffer.get_stats()  # Sync method
        composer_stats = composer.get_stats()  # Sync method

        return {
            "success": True,
            "data": {
                "meta_buffer": buffer_stats,
                "reasoning_composer": composer_stats,
                "combined": {
                    "total_templates": buffer_stats.get("total_templates", 0),
                    "available_modules": composer_stats.get("available_modules", 12),
                    "template_hit_rate": buffer_stats.get("hit_rate", 0.0),
                    "compositions_performed": composer_stats.get("compositions_attempted", 0)
                }
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "21",
                "feature": "template_reuse_optimization"
            }
        }

    except Exception as e:
        logger.error(f"Get Phase 21 summary failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Phase 21 summary: {str(e)}"
        )


# =============================================================================
# Classifier Feedback Endpoints (Adaptive-RAG inspired)
# =============================================================================

@router.get("/classifier/stats")
async def get_classifier_stats():
    """
    Get classifier feedback statistics.

    Returns information about classification outcomes, mismatch patterns,
    and adaptive hints generated from learning.
    """
    try:
        from agentic import get_classifier_feedback

        feedback = get_classifier_feedback()
        stats = feedback.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get classifier stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get classifier stats: {str(e)}"
        )


@router.get("/classifier/outcomes")
async def get_classifier_outcomes(
    category: Optional[str] = Query(None, description="Filter by query category"),
    limit: int = Query(50, ge=1, le=200, description="Maximum outcomes to return")
):
    """
    Get classification outcome history.

    Shows past classifications and their actual outcomes for analysis.
    """
    try:
        from agentic import get_classifier_feedback

        feedback = get_classifier_feedback()
        outcomes = feedback.get_outcomes(category, limit)

        return {
            "success": True,
            "data": {
                "outcomes": outcomes,
                "count": len(outcomes)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category_filter": category,
                "limit": limit
            }
        }

    except Exception as e:
        logger.error(f"Get classifier outcomes failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get classifier outcomes: {str(e)}"
        )


@router.get("/classifier/hints")
async def get_classifier_hints():
    """
    Get adaptive hints learned from classification outcomes.

    These hints are used to adjust future classifications based on
    observed patterns in past outcomes.
    """
    try:
        from agentic import get_classifier_feedback

        feedback = get_classifier_feedback()
        stats = feedback.get_stats()

        return {
            "success": True,
            "data": {
                "hints": stats.get("hints", []),
                "hint_count": stats.get("hint_count", 0),
                "last_generation": stats.get("last_hint_generation")
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Get classifier hints failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get classifier hints: {str(e)}"
        )


@router.delete("/classifier/outcomes")
async def clear_classifier_outcomes(
    category: Optional[str] = Query(None, description="Category to clear (all if not specified)")
):
    """
    Clear classification outcome history.

    Args:
        category: If specified, only clear outcomes of this category.
                 If not specified, clear all outcomes.
    """
    try:
        from agentic import get_classifier_feedback

        feedback = get_classifier_feedback()
        count = feedback.clear_outcomes(category)

        return {
            "success": True,
            "data": {
                "cleared_count": count,
                "category": category or "all"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Clear classifier outcomes failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear classifier outcomes: {str(e)}"
        )


# ============================================
# DOMAIN CORPUS ENDPOINTS (Phase 2: December 2025)
# Persistent domain-specific knowledge bases
# ============================================

# Global corpus manager instance
_corpus_manager = None


async def get_corpus_manager():
    """Get or create the domain corpus manager"""
    global _corpus_manager
    if _corpus_manager is None:
        import os
        from agentic.domain_corpus import (
            get_corpus_manager as _get_manager,
            create_fanuc_schema,
            create_raspberry_pi_schema
        )
        _corpus_manager = _get_manager()

        # Register default schemas
        ollama_url = get_settings().ollama_base_url
        _corpus_manager.register_corpus(create_fanuc_schema(), ollama_url)
        _corpus_manager.register_corpus(create_raspberry_pi_schema(), ollama_url)

        logger.info(f"Domain corpus manager initialized with {len(_corpus_manager.corpuses)} domains")
    return _corpus_manager


class CorpusDocumentRequest(BaseModel):
    """Request to add document to corpus"""
    content: str
    source_url: str = ""
    source_type: str = "unknown"
    title: str = ""
    extract_entities: bool = True


class CorpusQueryRequest(BaseModel):
    """Request to query corpus"""
    query: str
    entity_types: Optional[list[str]] = None
    include_relations: bool = True
    max_results: int = 10


class CorpusSchemaRequest(BaseModel):
    """Request to create custom domain schema"""
    domain_id: str
    domain_name: str
    description: str
    entity_types: list[dict]
    relationships: list[dict]
    extraction_hints: dict = {}
    priority_patterns: list[str] = []


@router.get("/corpus/domains")
async def list_corpus_domains():
    """
    List all registered domain corpuses.

    Returns domain IDs, names, and entity/relation counts.
    """
    try:
        manager = await get_corpus_manager()
        domains = manager.list_domains()

        return {
            "success": True,
            "data": {
                "domains": domains,
                "count": len(domains)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"List corpus domains failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list domains: {str(e)}"
        )


@router.get("/corpus/{domain_id}/stats")
async def get_corpus_stats(domain_id: str):
    """
    Get statistics for a specific domain corpus.

    Args:
        domain_id: Domain identifier (e.g., 'fanuc_robotics', 'raspberry_pi')

    Returns:
        Entity counts, relation counts, document counts, extraction stats
    """
    try:
        manager = await get_corpus_manager()
        corpus = manager.get_corpus(domain_id)

        if not corpus:
            raise HTTPException(
                status_code=404,
                detail=f"Domain corpus '{domain_id}' not found"
            )

        return {
            "success": True,
            "data": corpus.get_stats(),
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get corpus stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get corpus stats: {str(e)}"
        )


@router.post("/corpus/{domain_id}/documents")
async def add_corpus_document(domain_id: str, request: CorpusDocumentRequest):
    """
    Add document to domain corpus with entity extraction.

    The document will be:
    1. Checked for duplicates via content hashing
    2. Chunked if necessary
    3. Processed to extract domain-specific entities
    4. Indexed for semantic retrieval

    Args:
        domain_id: Domain identifier
        request: Document content and metadata

    Returns:
        Extraction results (entities and relations found)
    """
    try:
        manager = await get_corpus_manager()
        builder = manager.get_builder(domain_id)

        if not builder:
            raise HTTPException(
                status_code=404,
                detail=f"Domain corpus '{domain_id}' not found"
            )

        result = await builder.add_document(
            content=request.content,
            source_url=request.source_url,
            source_type=request.source_type,
            title=request.title,
            extract_entities=request.extract_entities
        )

        return {
            "success": True,
            "data": result,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain_id": domain_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add corpus document failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add document: {str(e)}"
        )


@router.post("/corpus/{domain_id}/query")
async def query_corpus(domain_id: str, request: CorpusQueryRequest):
    """
    Query domain corpus with hybrid search.

    Combines:
    - Semantic search via embeddings
    - Graph traversal for related entities
    - Contextual synthesis for LLM consumption

    Args:
        domain_id: Domain identifier
        request: Query parameters

    Returns:
        Matching entities, related entities, synthesized context
    """
    try:
        manager = await get_corpus_manager()
        retriever = manager.get_retriever(domain_id)

        if not retriever:
            raise HTTPException(
                status_code=404,
                detail=f"Domain corpus '{domain_id}' not found"
            )

        result = await retriever.query(
            query=request.query,
            entity_types=request.entity_types,
            include_relations=request.include_relations
        )

        return {
            "success": True,
            "data": result,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain_id": domain_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query corpus failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query corpus: {str(e)}"
        )


@router.get("/corpus/{domain_id}/troubleshoot/{error_code}")
async def get_troubleshooting_path(domain_id: str, error_code: str):
    """
    Get complete troubleshooting path for an error code.

    Traverses the knowledge graph:
    error_code → symptoms → causes → solutions

    Args:
        domain_id: Domain identifier
        error_code: Error code to troubleshoot (e.g., 'SRVO-001')

    Returns:
        Complete troubleshooting path with symptoms, causes, and solutions
    """
    try:
        manager = await get_corpus_manager()
        retriever = manager.get_retriever(domain_id)

        if not retriever:
            raise HTTPException(
                status_code=404,
                detail=f"Domain corpus '{domain_id}' not found"
            )

        result = await retriever.get_troubleshooting_path(error_code)

        if "error" in result:
            raise HTTPException(
                status_code=404,
                detail=result["error"]
            )

        return {
            "success": True,
            "data": result,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain_id": domain_id,
                "error_code": error_code
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get troubleshooting path failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get troubleshooting path: {str(e)}"
        )


@router.get("/corpus/{domain_id}/entities")
async def list_corpus_entities(
    domain_id: str,
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(50, ge=1, le=500, description="Maximum entities to return")
):
    """
    List entities in a domain corpus.

    Args:
        domain_id: Domain identifier
        entity_type: Optional filter by entity type
        limit: Maximum entities to return

    Returns:
        List of entities with their attributes
    """
    try:
        manager = await get_corpus_manager()
        corpus = manager.get_corpus(domain_id)

        if not corpus:
            raise HTTPException(
                status_code=404,
                detail=f"Domain corpus '{domain_id}' not found"
            )

        if entity_type:
            entities = corpus.get_entities_by_type(entity_type)
        else:
            entities = list(corpus.entities.values())

        # Sort by mention count and limit
        entities = sorted(entities, key=lambda e: e.mention_count, reverse=True)[:limit]

        return {
            "success": True,
            "data": {
                "entities": [e.to_dict() for e in entities],
                "count": len(entities),
                "total": len(corpus.entities)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain_id": domain_id,
                "filter": entity_type
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List corpus entities failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list entities: {str(e)}"
        )


@router.get("/corpus/{domain_id}/graph")
async def export_knowledge_graph(domain_id: str):
    """
    Export corpus as knowledge graph for visualization.

    Returns nodes (entities) and edges (relations) in a format
    suitable for graph visualization libraries.

    Args:
        domain_id: Domain identifier

    Returns:
        Nodes, edges, and schema definition
    """
    try:
        manager = await get_corpus_manager()
        corpus = manager.get_corpus(domain_id)

        if not corpus:
            raise HTTPException(
                status_code=404,
                detail=f"Domain corpus '{domain_id}' not found"
            )

        return {
            "success": True,
            "data": corpus.export_knowledge_graph(),
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "domain_id": domain_id
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export knowledge graph failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export graph: {str(e)}"
        )


@router.post("/corpus/cross-domain/query")
async def cross_domain_query(request: CorpusQueryRequest):
    """
    Query across all registered domain corpuses.

    Useful when the domain is unknown or the query spans multiple domains.

    Args:
        request: Query parameters

    Returns:
        Results from each domain corpus
    """
    try:
        manager = await get_corpus_manager()
        result = await manager.cross_domain_query(request.query)

        return {
            "success": True,
            "data": result,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Cross-domain query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute cross-domain query: {str(e)}"
        )


@router.post("/corpus/register")
async def register_custom_corpus(request: CorpusSchemaRequest):
    """
    Register a custom domain corpus with a new schema.

    Allows creating domain-specific knowledge bases beyond the
    default FANUC and Raspberry Pi schemas.

    Args:
        request: Domain schema definition

    Returns:
        Confirmation of registration
    """
    try:
        import os
        from agentic.domain_corpus import (
            DomainSchema,
            DomainEntityDef,
            DomainRelationDef
        )

        manager = await get_corpus_manager()

        # Check if domain already exists
        if manager.get_corpus(request.domain_id):
            raise HTTPException(
                status_code=409,
                detail=f"Domain '{request.domain_id}' already exists"
            )

        # Build entity type definitions
        entity_types = [
            DomainEntityDef(
                entity_type=et.get("type", "concept"),
                description=et.get("description", ""),
                extraction_patterns=et.get("patterns", []),
                examples=et.get("examples", []),
                attributes=et.get("attributes", [])
            )
            for et in request.entity_types
        ]

        # Build relationship definitions
        relationships = [
            DomainRelationDef(
                relation_type=rel.get("type", "related_to"),
                source_types=rel.get("source_types", []),
                target_types=rel.get("target_types", []),
                description=rel.get("description", ""),
                bidirectional=rel.get("bidirectional", False)
            )
            for rel in request.relationships
        ]

        # Create schema
        schema = DomainSchema(
            domain_id=request.domain_id,
            domain_name=request.domain_name,
            description=request.description,
            entity_types=entity_types,
            relationships=relationships,
            extraction_hints=request.extraction_hints,
            priority_patterns=request.priority_patterns
        )

        # Register corpus
        ollama_url = get_settings().ollama_base_url
        manager.register_corpus(schema, ollama_url)

        return {
            "success": True,
            "data": {
                "domain_id": request.domain_id,
                "domain_name": request.domain_name,
                "entity_types": len(entity_types),
                "relationships": len(relationships)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Register custom corpus failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register corpus: {str(e)}"
        )


# ============================================================================
# Entity-Enhanced Retrieval Endpoints (December 2025)
# ============================================================================

class EntityEnhancedQueryRequest(BaseModel):
    """Request for entity-enhanced retrieval."""
    query: str
    context: Optional[Dict[str, Any]] = None
    max_results: int = 10
    include_relations: bool = True
    enable_classification: bool = True
    enable_entity_extraction: bool = True
    enable_domain_corpus: bool = True
    enable_embedding_aggregation: bool = True


@router.post("/entity-enhanced/query")
async def entity_enhanced_query(request: EntityEnhancedQueryRequest):
    """
    Entity-enhanced retrieval with full pipeline.

    Combines:
    - Query classification (DeepSeek-R1)
    - Entity extraction (GSW-style)
    - Domain corpus retrieval (FANUC, Raspberry Pi, etc.)
    - Master embedding aggregation (RouterRetriever pattern)
    - Sub-manifold navigation (entity-guided)

    Returns comprehensive retrieval result with context from all sources.
    """
    from agentic.entity_enhanced_retrieval import get_entity_enhanced_retriever

    try:
        ollama_url = get_settings().ollama_base_url
        retriever = get_entity_enhanced_retriever(ollama_url)

        # Configure feature flags
        retriever.enable_classification = request.enable_classification
        retriever.enable_entity_extraction = request.enable_entity_extraction
        retriever.enable_domain_corpus = request.enable_domain_corpus
        retriever.enable_embedding_aggregation = request.enable_embedding_aggregation

        result = await retriever.retrieve(
            query=request.query,
            context=request.context,
            max_results=request.max_results,
            include_relations=request.include_relations
        )

        return {
            "success": True,
            "data": {
                "query": result.query,
                "synthesized_context": result.synthesized_context,
                "confidence": result.confidence,
                "pipeline_used": result.pipeline_used,
                "domains_queried": result.domains_queried,
                "sources_used": result.sources_used,
                "extracted_entities": result.extracted_entities,
                "classification": {
                    "category": result.classification.category.value if result.classification else None,
                    "complexity": result.classification.complexity.value if result.classification else None,
                    "recommended_pipeline": result.classification.recommended_pipeline.value if result.classification else None,
                    "reasoning": result.classification.reasoning if result.classification else None
                } if result.classification else None,
                "embedding_results": {
                    "result_count": len(result.embedding_results.sub_manifold_results) if result.embedding_results else 0,
                    "domains_used": result.embedding_results.domains_used if result.embedding_results else [],
                    "retrieval_time_ms": result.embedding_results.retrieval_time_ms if result.embedding_results else 0
                } if result.embedding_results else None
            },
            "meta": {
                "total_time_ms": result.total_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Entity-enhanced query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Entity-enhanced query failed: {str(e)}"
        )


@router.get("/entity-enhanced/stats")
async def get_entity_enhanced_stats():
    """Get entity-enhanced retriever statistics."""
    from agentic.entity_enhanced_retrieval import get_entity_enhanced_retriever

    try:
        ollama_url = get_settings().ollama_base_url
        retriever = get_entity_enhanced_retriever(ollama_url)
        stats = retriever.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


class ClassifyQueryRequest(BaseModel):
    """Request for query classification."""
    query: str
    context: Optional[Dict[str, Any]] = None


@router.post("/classify")
async def classify_query_endpoint(request: ClassifyQueryRequest):
    """
    Classify a query using DeepSeek-R1 or fallback model.

    Returns:
    - category: research, problem_solving, factual, technical, etc.
    - capabilities: web_search, reasoning, code_analysis, etc.
    - complexity: simple, moderate, complex, expert
    - recommended_pipeline: direct_answer, web_search, agentic_search, code_assistant
    """
    from agentic.query_classifier import get_query_classifier

    try:
        ollama_url = get_settings().ollama_base_url
        classifier = get_query_classifier(ollama_url)

        result = await classifier.classify(request.query, request.context)

        return {
            "success": True,
            "data": {
                "category": result.category.value,
                "capabilities": result.capabilities,
                "complexity": result.complexity.value,
                "urgency": result.urgency,
                "use_thinking_model": result.use_thinking_model,
                "recommended_pipeline": result.recommended_pipeline.value,
                "reasoning": result.reasoning
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.get("/classifier/stats")
async def get_classifier_stats():
    """Get query classifier statistics."""
    from agentic.query_classifier import get_query_classifier

    try:
        ollama_url = get_settings().ollama_base_url
        classifier = get_query_classifier(ollama_url)

        return {
            "success": True,
            "data": classifier.get_stats() if hasattr(classifier, 'get_stats') else {},
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get classifier stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/embedding-aggregator/stats")
async def get_embedding_aggregator_stats():
    """Get embedding aggregator statistics."""
    from agentic.embedding_aggregator import get_embedding_aggregator

    try:
        ollama_url = get_settings().ollama_base_url
        aggregator = get_embedding_aggregator(ollama_url)
        stats = aggregator.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get embedding aggregator stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


class IndexEntityRequest(BaseModel):
    """Request to index an entity in the master embedding."""
    domain: str
    entity_id: str
    entity_name: str
    entity_type: str
    context: str
    source_url: Optional[str] = ""


@router.post("/embedding-aggregator/index")
async def index_entity_in_aggregator(request: IndexEntityRequest):
    """Index an entity in the master embedding aggregator."""
    from agentic.embedding_aggregator import get_embedding_aggregator

    try:
        ollama_url = get_settings().ollama_base_url
        aggregator = get_embedding_aggregator(ollama_url)

        success = await aggregator.index_entity(
            domain=request.domain,
            entity_id=request.entity_id,
            entity_name=request.entity_name,
            entity_type=request.entity_type,
            context=request.context,
            source_url=request.source_url
        )

        return {
            "success": success,
            "data": {
                "entity_id": request.entity_id,
                "domain": request.domain,
                "indexed": success
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to index entity: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


class SubManifoldQueryRequest(BaseModel):
    """Request for sub-manifold retrieval."""
    anchor_entities: List[Dict[str, str]]
    domains: List[str] = []
    k: int = 10
    hop_radius: int = 2


@router.post("/embedding-aggregator/sub-manifold")
async def query_sub_manifold(request: SubManifoldQueryRequest):
    """
    Query sub-manifold around anchor entities.

    Uses entity-guided navigation through embedding space.
    """
    from agentic.embedding_aggregator import get_embedding_aggregator

    try:
        ollama_url = get_settings().ollama_base_url
        aggregator = get_embedding_aggregator(ollama_url)

        results = await aggregator.retrieve_sub_manifold(
            anchor_entities=request.anchor_entities,
            domains=request.domains,
            k=request.k,
            hop_radius=request.hop_radius
        )

        return {
            "success": True,
            "data": {
                "results": [
                    {
                        "entity_id": r.entity_id,
                        "entity_name": r.entity_name,
                        "entity_type": r.entity_type,
                        "distance": r.distance,
                        "domain": r.domain,
                        "context": r.context[:500] if r.context else "",
                        "related_entities": r.related_entities
                    }
                    for r in results
                ],
                "count": len(results)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Sub-manifold query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =============================================================================
# Mixed-Precision Embedding Endpoints
# =============================================================================


class MixedPrecisionIndexRequest(BaseModel):
    """Request to index a document at multiple precision levels."""
    doc_id: str
    text: str
    content: str = ""
    metadata: Optional[dict] = None
    instruction: Optional[str] = None
    store_residual: bool = True


class MixedPrecisionSearchRequest(BaseModel):
    """Request for three-stage mixed-precision search."""
    query: str
    top_k: int = 10
    instruction: Optional[str] = None
    binary_candidates: int = 500
    int8_candidates: int = 50


class MRLSearchRequest(BaseModel):
    """Request for MRL hierarchical search."""
    query: str
    top_k: int = 10
    instruction: Optional[str] = None
    stages: List[int] = [64, 256, 1024, 4096]


class CreateAnchorRequest(BaseModel):
    """Request to create a category anchor from examples."""
    category: str
    example_texts: List[str]
    instruction: Optional[str] = None


class SemanticArithmeticRequest(BaseModel):
    """Request for semantic arithmetic operation."""
    base_text: str
    add_text: str
    subtract_text: str
    anchor_category: Optional[str] = None


@router.get("/mixed-precision/stats")
async def get_mixed_precision_stats():
    """Get mixed-precision embedding service statistics."""
    from agentic.mixed_precision_embeddings import get_mixed_precision_service

    try:
        ollama_url = get_settings().ollama_base_url
        service = get_mixed_precision_service(ollama_url)

        stats = service.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get mixed-precision stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mixed-precision/index")
async def index_mixed_precision(request: MixedPrecisionIndexRequest):
    """
    Index a document at all precision levels (binary, int8, fp16).

    Creates:
    - Binary embedding (32x compression) for fast coarse search
    - Int8 embedding (4x compression) for medium precision
    - FP16 embedding for high-precision rescoring
    - Optional semantic residual for precision correction
    """
    from agentic.mixed_precision_embeddings import get_mixed_precision_service

    try:
        ollama_url = get_settings().ollama_base_url
        service = get_mixed_precision_service(ollama_url)

        result = await service.index_document(
            doc_id=request.doc_id,
            text=request.text,
            content=request.content,
            metadata=request.metadata,
            instruction=request.instruction,
            store_residual=request.store_residual
        )

        return {
            "success": True,
            "data": {
                "doc_id": result.doc_id,
                "dimension": result.dimension,
                "binary_bytes": len(result.binary) if result.binary else 0,
                "has_residual": result.residual is not None,
                "compression_ratios": {
                    "binary": round(result.dimension * 2 / (len(result.binary) if result.binary else 1), 1),
                    "int8": round(result.dimension * 2 / result.dimension, 1),
                    "fp16": 1.0
                }
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mixed-precision/search")
async def mixed_precision_search(request: MixedPrecisionSearchRequest):
    """
    Three-stage precision-stratified search.

    Pipeline:
    1. Binary search: Ultra-fast Hamming distance (top 500 candidates)
    2. Int8 rescore: Cosine similarity on quantized embeddings (top 50)
    3. FP16 final: High-precision rescoring (final top_k)

    Memory savings: 32x for binary stage, 4x for int8 stage.
    """
    from agentic.mixed_precision_embeddings import get_mixed_precision_service

    try:
        ollama_url = get_settings().ollama_base_url
        service = get_mixed_precision_service(ollama_url)

        results, stats = await service.search(
            query=request.query,
            top_k=request.top_k,
            instruction=request.instruction,
            binary_candidates=request.binary_candidates,
            int8_candidates=request.int8_candidates
        )

        return {
            "success": True,
            "data": {
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "score": r.score,
                        "precision_used": r.precision_used.value,
                        "content": r.content[:500] if r.content else "",
                        "metadata": r.metadata
                    }
                    for r in results
                ],
                "count": len(results)
            },
            "meta": {
                "stats": {
                    "binary_candidates": stats.binary_candidates,
                    "int8_candidates": stats.int8_candidates,
                    "final_results": stats.final_results,
                    "binary_time_ms": round(stats.binary_time_ms, 1),
                    "int8_time_ms": round(stats.int8_time_ms, 1),
                    "fp16_time_ms": round(stats.fp16_time_ms, 1),
                    "total_time_ms": round(stats.total_time_ms, 1)
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Mixed-precision search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mixed-precision/mrl-search")
async def mrl_hierarchical_search(request: MRLSearchRequest):
    """
    Matryoshka Representation Learning (MRL) hierarchical search.

    Uses dimension truncation for progressive refinement:
    - Early dimensions (64): Coarse semantics, fast filtering
    - Middle dimensions (256, 1024): Balanced precision
    - Full dimensions (4096): Fine-grained final ranking

    Based on: Kusupati et al., "Matryoshka Representation Learning" (NeurIPS 2022)
    """
    from agentic.mixed_precision_embeddings import get_mixed_precision_service

    try:
        ollama_url = get_settings().ollama_base_url
        service = get_mixed_precision_service(ollama_url)

        results, stats = await service.mrl_hierarchical_search(
            query=request.query,
            top_k=request.top_k,
            instruction=request.instruction,
            stages=request.stages
        )

        return {
            "success": True,
            "data": {
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "score": r.score,
                        "content": r.content[:500] if r.content else "",
                        "metadata": r.metadata
                    }
                    for r in results
                ],
                "count": len(results)
            },
            "meta": {
                "stages": request.stages,
                "total_time_ms": round(stats.total_time_ms, 1),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"MRL search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mixed-precision/anchor")
async def create_anchor(request: CreateAnchorRequest):
    """
    Create a category anchor from example texts.

    Anchors provide semantic reference frames for:
    - Guided interpolation between embeddings
    - Semantic arithmetic validation
    - Precision-guided operations

    The anchor is computed as the normalized mean of example embeddings.
    """
    from agentic.mixed_precision_embeddings import get_mixed_precision_service

    try:
        ollama_url = get_settings().ollama_base_url
        service = get_mixed_precision_service(ollama_url)

        anchor = await service.create_anchor_from_examples(
            category=request.category,
            example_texts=request.example_texts,
            instruction=request.instruction
        )

        return {
            "success": True,
            "data": {
                "category": request.category,
                "dimension": len(anchor),
                "example_count": len(request.example_texts),
                "anchor_norm": float(sum(x**2 for x in anchor)**0.5)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to create anchor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mixed-precision/semantic-arithmetic")
async def semantic_arithmetic(request: SemanticArithmeticRequest):
    """
    Perform semantic arithmetic: base - subtract + add.

    Example:
        base = "homeless shelter"
        add = "addiction recovery"
        subtract = "basic housing"
        Result: embedding closer to "recovery center"

    If anchor_category is provided, validates result stays in valid semantic region.

    Based on Word2Vec analogy principle: king - man + woman = queen
    """
    from agentic.mixed_precision_embeddings import get_mixed_precision_service

    try:
        ollama_url = get_settings().ollama_base_url
        service = get_mixed_precision_service(ollama_url)

        # Get embeddings
        base_emb = await service.get_embedding(request.base_text)
        add_emb = await service.get_embedding(request.add_text)
        sub_emb = await service.get_embedding(request.subtract_text)

        # Perform arithmetic
        result_emb = service.semantic_arithmetic(
            base=base_emb,
            add=add_emb,
            subtract=sub_emb,
            anchor_category=request.anchor_category
        )

        # Find nearest neighbors to result
        results, _ = await service.search(
            query=request.base_text,  # Dummy - we'll use result_emb directly
            top_k=5
        )

        return {
            "success": True,
            "data": {
                "operation": f"({request.base_text}) - ({request.subtract_text}) + ({request.add_text})",
                "result_norm": float(sum(x**2 for x in result_emb)**0.5),
                "anchor_used": request.anchor_category,
                "dimension": len(result_emb)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Semantic arithmetic failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BGE-M3 Hybrid Retrieval Endpoints
# =============================================================================

from agentic.bge_m3_hybrid import (
    BGEM3HybridRetriever,
    HybridDocument,
    HybridSearchResult,
    RetrievalMode,
    get_hybrid_retriever,
    create_hybrid_retriever
)

# Global hybrid retriever instance
_hybrid_retriever: Optional[BGEM3HybridRetriever] = None


async def get_hybrid_retriever_instance() -> BGEM3HybridRetriever:
    """Get or create the global hybrid retriever instance."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = await create_hybrid_retriever(
            db_path="/home/sparkone/sdd/Recovery_Bot/memOS/data/bge_m3_hybrid.db",
            load_existing=True
        )
    return _hybrid_retriever


class HybridIndexRequest(BaseModel):
    """Request to index documents in hybrid retriever."""
    documents: List[Dict[str, Any]]  # [{"doc_id": str, "content": str, "metadata": dict}]


class HybridSearchRequest(BaseModel):
    """Request for hybrid search."""
    query: str
    top_k: int = 10
    mode: str = "hybrid"  # dense_only, sparse_only, hybrid
    dense_candidates: int = 100
    sparse_candidates: int = 100


class HybridWeightsRequest(BaseModel):
    """Request to update fusion weights."""
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    multivec_weight: float = 0.0
    use_rrf: bool = True


@router.get("/hybrid/stats")
async def get_hybrid_stats():
    """Get BGE-M3 hybrid retrieval statistics."""
    try:
        retriever = await get_hybrid_retriever_instance()
        stats = retriever.get_stats()

        return {
            "success": True,
            "data": {
                "documents_indexed": stats.documents_indexed,
                "vocabulary_size": stats.vocabulary_size,
                "avg_doc_length": stats.avg_doc_length,
                "dense_index_size_mb": stats.dense_index_size_mb,
                "sparse_index_size_mb": stats.sparse_index_size_mb,
                "mode": stats.mode,
                "model": retriever.model,
                "weights": {
                    "dense": retriever.dense_weight,
                    "sparse": retriever.sparse_weight,
                    "multivec": retriever.multivec_weight
                },
                "use_rrf": retriever.use_rrf
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get hybrid stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid/index")
async def index_hybrid_documents(request: HybridIndexRequest):
    """
    Index documents for hybrid retrieval.

    Creates both dense (BGE-M3) and sparse (BM25) representations.
    """
    try:
        retriever = await get_hybrid_retriever_instance()

        indexed_docs = []
        for doc_data in request.documents:
            doc = await retriever.add_document(
                doc_id=doc_data.get("doc_id", hashlib.md5(doc_data["content"].encode()).hexdigest()),
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {})
            )
            indexed_docs.append(doc.to_dict())

        return {
            "success": True,
            "data": {
                "indexed": len(indexed_docs),
                "documents": indexed_docs
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Hybrid indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid/search")
async def hybrid_search(request: HybridSearchRequest):
    """
    Execute hybrid search combining dense and sparse retrieval.

    Modes:
    - dense_only: BGE-M3 semantic similarity only
    - sparse_only: BM25 lexical matching only
    - hybrid: Combined dense + sparse with RRF fusion
    """
    try:
        retriever = await get_hybrid_retriever_instance()

        # Map mode string to enum
        mode_map = {
            "dense_only": RetrievalMode.DENSE_ONLY,
            "sparse_only": RetrievalMode.SPARSE_ONLY,
            "hybrid": RetrievalMode.HYBRID,
            "full_hybrid": RetrievalMode.FULL_HYBRID
        }
        mode = mode_map.get(request.mode, RetrievalMode.HYBRID)

        results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            mode=mode,
            dense_candidates=request.dense_candidates,
            sparse_candidates=request.sparse_candidates
        )

        return {
            "success": True,
            "data": {
                "query": request.query,
                "mode": request.mode,
                "results": [r.to_dict() for r in results],
                "total_results": len(results)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hybrid/weights")
async def update_hybrid_weights(request: HybridWeightsRequest):
    """Update fusion weights for hybrid retrieval."""
    try:
        retriever = await get_hybrid_retriever_instance()

        # Normalize weights
        total = request.dense_weight + request.sparse_weight + request.multivec_weight
        if total > 0:
            retriever.dense_weight = request.dense_weight / total
            retriever.sparse_weight = request.sparse_weight / total
            retriever.multivec_weight = request.multivec_weight / total
        retriever.use_rrf = request.use_rrf

        return {
            "success": True,
            "data": {
                "weights": {
                    "dense": retriever.dense_weight,
                    "sparse": retriever.sparse_weight,
                    "multivec": retriever.multivec_weight
                },
                "use_rrf": retriever.use_rrf
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to update weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hybrid/bm25-stats")
async def get_bm25_stats():
    """Get BM25 sparse index statistics."""
    try:
        retriever = await get_hybrid_retriever_instance()
        bm25_stats = retriever.bm25_index.get_stats()

        return {
            "success": True,
            "data": bm25_stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to get BM25 stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/hybrid/clear")
async def clear_hybrid_index():
    """Clear all documents from hybrid index."""
    global _hybrid_retriever

    try:
        if _hybrid_retriever:
            await _hybrid_retriever.close()
            _hybrid_retriever = None

        # Remove database file
        import os
        db_path = "/home/sparkone/sdd/Recovery_Bot/memOS/data/bge_m3_hybrid.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        return {
            "success": True,
            "data": {"message": "Hybrid index cleared"},
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to clear hybrid index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HyDE (Hypothetical Document Embeddings) Endpoints
# =============================================================================

from agentic.hyde import (
    HyDEExpander,
    HyDEConfig,
    HyDEMode,
    DocumentType,
    HyDEResult,
    get_hyde_expander,
    create_hyde_expander
)

# Global HyDE expander instance
_hyde_expander: Optional[HyDEExpander] = None


async def get_hyde_expander_instance() -> HyDEExpander:
    """Get or create the global HyDE expander instance."""
    global _hyde_expander
    if _hyde_expander is None:
        _hyde_expander = await create_hyde_expander()
    return _hyde_expander


class HyDEExpandRequest(BaseModel):
    """Request for HyDE query expansion."""
    query: str
    mode: str = "single"  # single, multi
    num_hypotheticals: int = 1
    document_type: str = "passage"  # answer, passage, explanation, summary, technical
    include_query_embedding: bool = True


class HyDESearchRequest(BaseModel):
    """Request for HyDE-enhanced search."""
    query: str
    top_k: int = 10
    mode: str = "single"
    num_hypotheticals: int = 1
    use_hybrid: bool = True  # Combine with hybrid retrieval


@router.get("/hyde/stats")
async def get_hyde_stats():
    """Get HyDE expander statistics."""
    try:
        expander = await get_hyde_expander_instance()
        stats = expander.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get HyDE stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyde/expand")
async def expand_with_hyde(request: HyDEExpandRequest):
    """
    Expand a query using HyDE (Hypothetical Document Embeddings).

    Generates hypothetical documents that would answer the query,
    then creates a fused embedding for improved retrieval.
    """
    try:
        expander = await get_hyde_expander_instance()

        # Map mode string to enum
        mode_map = {
            "single": HyDEMode.SINGLE,
            "multi": HyDEMode.MULTI,
            "contrastive": HyDEMode.CONTRASTIVE
        }
        mode = mode_map.get(request.mode, HyDEMode.SINGLE)

        # Map document type
        type_map = {
            "answer": DocumentType.ANSWER,
            "passage": DocumentType.PASSAGE,
            "explanation": DocumentType.EXPLANATION,
            "summary": DocumentType.SUMMARY,
            "technical": DocumentType.TECHNICAL
        }
        doc_type = type_map.get(request.document_type, DocumentType.PASSAGE)

        # Update config
        expander.config.document_type = doc_type

        result = await expander.expand(
            query=request.query,
            mode=mode,
            num_hypotheticals=request.num_hypotheticals,
            include_query_embedding=request.include_query_embedding
        )

        return {
            "success": True,
            "data": {
                "query": result.original_query,
                "hypothetical_documents": result.hypothetical_documents,
                "generation_time_ms": result.generation_time_ms,
                "embedding_time_ms": result.embedding_time_ms,
                "fused_embedding_norm": float(np.linalg.norm(result.fused_embedding)),
                "metadata": result.metadata
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"HyDE expansion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyde/search")
async def search_with_hyde(request: HyDESearchRequest):
    """
    Search using HyDE-expanded query.

    Combines HyDE expansion with hybrid retrieval for best results.
    """
    try:
        expander = await get_hyde_expander_instance()

        # Map mode
        mode_map = {
            "single": HyDEMode.SINGLE,
            "multi": HyDEMode.MULTI
        }
        mode = mode_map.get(request.mode, HyDEMode.SINGLE)

        # First expand the query with HyDE
        hyde_result = await expander.expand(
            query=request.query,
            mode=mode,
            num_hypotheticals=request.num_hypotheticals
        )

        # If hybrid search is enabled and we have documents indexed
        if request.use_hybrid:
            retriever = await get_hybrid_retriever_instance()

            if retriever.documents:
                # Use the fused embedding for search
                fused_emb = hyde_result.fused_embedding
                query_norm = np.linalg.norm(fused_emb)

                # Compute scores against indexed documents
                results = []
                for doc_id, doc in retriever.documents.items():
                    dense_score = 0.0
                    if doc.dense_embedding is not None:
                        doc_norm = np.linalg.norm(doc.dense_embedding)
                        if query_norm > 0 and doc_norm > 0:
                            dense_score = float(
                                np.dot(fused_emb, doc.dense_embedding) /
                                (query_norm * doc_norm)
                            )

                    # Also get BM25 score using original query
                    sparse_results = retriever.bm25_index.search(request.query, 100)
                    sparse_scores = {d: s for d, s in sparse_results}
                    max_sparse = max(sparse_scores.values()) if sparse_scores else 1
                    sparse_score = sparse_scores.get(doc_id, 0) / max_sparse if max_sparse > 0 else 0

                    # Combine scores
                    combined = 0.6 * dense_score + 0.4 * sparse_score

                    results.append({
                        "doc_id": doc_id,
                        "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        "dense_score": round(dense_score, 4),
                        "sparse_score": round(sparse_score, 4),
                        "combined_score": round(combined, 4),
                        "metadata": doc.metadata
                    })

                # Sort by combined score
                results.sort(key=lambda x: x["combined_score"], reverse=True)
                results = results[:request.top_k]

                return {
                    "success": True,
                    "data": {
                        "query": request.query,
                        "hypothetical_documents": hyde_result.hypothetical_documents,
                        "results": results,
                        "total_results": len(results),
                        "hyde_generation_time_ms": hyde_result.generation_time_ms,
                        "hyde_embedding_time_ms": hyde_result.embedding_time_ms
                    },
                    "meta": {
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                }

        # If no hybrid or no documents, return just the HyDE result
        return {
            "success": True,
            "data": {
                "query": request.query,
                "hypothetical_documents": hyde_result.hypothetical_documents,
                "results": [],
                "total_results": 0,
                "message": "No documents indexed for search",
                "hyde_generation_time_ms": hyde_result.generation_time_ms,
                "hyde_embedding_time_ms": hyde_result.embedding_time_ms
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"HyDE search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/hyde/cache")
async def clear_hyde_cache():
    """Clear HyDE expansion cache."""
    try:
        expander = await get_hyde_expander_instance()
        cache_size = len(expander._cache)
        expander._cache.clear()

        return {
            "success": True,
            "data": {
                "cleared_entries": cache_size,
                "message": "HyDE cache cleared"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to clear HyDE cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RAGAS Evaluation Endpoints
# =============================================================================

from agentic.ragas import (
    RAGASEvaluator,
    RAGASConfig,
    RAGASResult,
    EvaluationMetric,
    get_ragas_evaluator,
    create_ragas_evaluator
)

# Global RAGAS evaluator instance
_ragas_evaluator: Optional[RAGASEvaluator] = None


async def get_ragas_evaluator_instance() -> RAGASEvaluator:
    """Get or create the global RAGAS evaluator instance."""
    global _ragas_evaluator
    if _ragas_evaluator is None:
        _ragas_evaluator = await create_ragas_evaluator()
    return _ragas_evaluator


class RAGASEvaluateRequest(BaseModel):
    """Request for RAGAS evaluation."""
    question: str
    answer: str
    contexts: List[str]
    metrics: Optional[List[str]] = None  # faithfulness, answer_relevancy, context_relevancy, context_precision


class RAGASBatchEvaluateRequest(BaseModel):
    """Request for batch RAGAS evaluation."""
    samples: List[Dict[str, Any]]  # [{"question": str, "answer": str, "contexts": list}]


@router.get("/ragas/stats")
async def get_ragas_stats():
    """Get RAGAS evaluation statistics."""
    try:
        evaluator = await get_ragas_evaluator_instance()
        stats = evaluator.get_aggregate_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get RAGAS stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ragas/evaluate")
async def evaluate_with_ragas(request: RAGASEvaluateRequest):
    """
    Evaluate a RAG response using RAGAS metrics.

    Metrics:
    - faithfulness: Are claims in answer supported by context?
    - answer_relevancy: Does answer address the question?
    - context_relevancy: Is retrieved context relevant?
    - context_precision: Is relevant context ranked higher?
    """
    try:
        evaluator = await get_ragas_evaluator_instance()

        # Map metric strings to enums
        metrics = None
        if request.metrics:
            metric_map = {
                "faithfulness": EvaluationMetric.FAITHFULNESS,
                "answer_relevancy": EvaluationMetric.ANSWER_RELEVANCY,
                "context_relevancy": EvaluationMetric.CONTEXT_RELEVANCY,
                "context_precision": EvaluationMetric.CONTEXT_PRECISION
            }
            metrics = [metric_map[m] for m in request.metrics if m in metric_map]

        result = await evaluator.evaluate(
            question=request.question,
            answer=request.answer,
            contexts=request.contexts,
            metrics=metrics
        )

        return {
            "success": True,
            "data": result.to_dict(),
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ragas/batch-evaluate")
async def batch_evaluate_with_ragas(request: RAGASBatchEvaluateRequest):
    """
    Evaluate multiple RAG responses using RAGAS.

    Returns individual scores and aggregate statistics.
    """
    try:
        evaluator = await get_ragas_evaluator_instance()

        results = []
        for sample in request.samples:
            result = await evaluator.evaluate(
                question=sample["question"],
                answer=sample["answer"],
                contexts=sample.get("contexts", [])
            )
            results.append({
                "question": sample["question"][:100],
                "scores": result.to_dict()["scores"]
            })

        # Get aggregate stats
        stats = evaluator.get_aggregate_stats()

        return {
            "success": True,
            "data": {
                "results": results,
                "aggregate": stats
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "samples_evaluated": len(results)
            }
        }

    except Exception as e:
        logger.error(f"RAGAS batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/ragas/history")
async def clear_ragas_history():
    """Clear RAGAS evaluation history."""
    try:
        evaluator = await get_ragas_evaluator_instance()
        history_size = len(evaluator._history)
        evaluator.clear_history()

        return {
            "success": True,
            "data": {
                "cleared_evaluations": history_size,
                "message": "RAGAS history cleared"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to clear RAGAS history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ragas/evaluate-search")
async def evaluate_search_with_ragas(request: Dict[str, Any]):
    """
    Evaluate a search response from the agentic search pipeline.

    Expects the standard search response format with synthesized_context and sources.
    """
    try:
        evaluator = await get_ragas_evaluator_instance()

        # Extract from search response format
        question = request.get("query", "")
        answer = request.get("synthesized_context", "")
        sources = request.get("sources", [])

        # Extract context from sources
        contexts = []
        for source in sources:
            if isinstance(source, dict):
                content = source.get("snippet", "") or source.get("content", "")
                if content:
                    contexts.append(content)
            elif isinstance(source, str):
                contexts.append(source)

        if not contexts:
            return {
                "success": False,
                "error": "No contexts found in search response",
                "meta": {"timestamp": datetime.now(timezone.utc).isoformat()}
            }

        result = await evaluator.evaluate(
            question=question,
            answer=answer,
            contexts=contexts
        )

        return {
            "success": True,
            "data": {
                "evaluation": result.to_dict(),
                "input_summary": {
                    "question_length": len(question),
                    "answer_length": len(answer),
                    "num_contexts": len(contexts)
                }
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Search RAGAS evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# UNIFIED CHAT GATEWAY - Server-side routing and agentic activation
# =============================================================================

class ChatGatewayRequest(BaseModel):
    """Request model for the unified chat gateway"""
    query: str
    user_id: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    force_agentic: bool = False  # Force agentic search even if classifier says otherwise
    model: str = "qwen3:8b"  # LLM model for direct responses
    preset: str = "full"  # UniversalOrchestrator preset: minimal, balanced, enhanced, research, full


class ChatGatewayResponse(BaseModel):
    """Response model for chat gateway"""
    success: bool
    pipeline_used: str  # direct_answer, web_search, agentic_search
    response: str
    sources: Optional[List[Dict]] = None
    confidence_score: Optional[float] = None
    classification: Optional[Dict] = None


@router.post("/gateway/stream")
async def chat_gateway_stream(request: ChatGatewayRequest):
    """
    Unified Chat Gateway with Server-Side Routing.

    This endpoint is the single entry point for all chat queries. It:
    1. Classifies the query using DeepSeek-R1 to determine optimal pipeline
    2. Routes to the appropriate handler (direct, web_search, or agentic)
    3. Streams SSE events for EVERY processing step for rich UI feedback

    Classification determines:
    - direct_answer: Query can be answered from LLM knowledge alone
    - web_search: Simple web search sufficient
    - agentic_search: Full multi-agent pipeline needed
    - code_assistant: Technical/code analysis mode

    SSE Events emitted:
    - classifying_query: Query classification started
    - query_classified: Classification complete with pipeline recommendation
    - pipeline_routed: Pipeline selection confirmed
    - [All agentic events if agentic pipeline selected]
    - gateway_complete: Final response ready

    Args:
        request: ChatGatewayRequest with query and options

    Returns:
        StreamingResponse with SSE events
    """
    import uuid
    import httpx
    import os

    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Chat gateway request: {request.query[:50]}...")

    # Create event emitter
    event_manager = get_event_manager()
    emitter = event_manager.create_emitter(request_id)

    async def generate_events():
        """Generator for SSE events"""
        queue = emitter.subscribe()

        try:
            # Start the gateway processing in background
            gateway_task = asyncio.create_task(
                _execute_chat_gateway(request, request_id, emitter)
            )

            # Stream events
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120.0)

                    if event is None:
                        break

                    yield event.to_sse()

                    # Check for terminal events
                    if event.event_type in [
                        EventType.SEARCH_COMPLETED,
                        EventType.SEARCH_FAILED,
                        "gateway_complete",
                        "gateway_error"
                    ]:
                        break

                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"

            await gateway_task

        except asyncio.CancelledError:
            logger.info(f"[{request_id}] Gateway stream cancelled")
        except Exception as e:
            logger.error(f"[{request_id}] Gateway stream error: {e}")
            yield f"event: gateway_error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"
        finally:
            emitter.unsubscribe(queue)
            event_manager.remove_emitter(request_id)

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
            "X-Gateway": "true"
        }
    )


async def _execute_chat_gateway(
    request: ChatGatewayRequest,
    request_id: str,
    emitter
):
    """Execute the chat gateway logic with classification and routing."""
    import httpx
    import os
    from agentic import events
    from agentic.events import AgentGraphState

    try:
        # Initialize graph state for visualization
        graph_state = AgentGraphState()

        # ===== STEP 1: CLASSIFY THE QUERY =====
        await emitter.emit(events.classifying_query(request_id, request.query))

        classifier = await get_classifier()
        classification = await classifier.classify(
            request.query,
            {"conversation_history": request.conversation_history} if request.conversation_history else None
        )

        # Emit classification result
        await emitter.emit(events.query_classified(
            request_id=request_id,
            category=classification.category.value,
            pipeline=classification.recommended_pipeline.value,
            complexity=classification.complexity.value,
            capabilities=classification.capabilities
        ))

        logger.info(f"[{request_id}] Classification: category={classification.category.value}, "
                   f"pipeline={classification.recommended_pipeline.value}, "
                   f"complexity={classification.complexity.value}")

        # ===== STEP 2: DETERMINE PIPELINE =====
        pipeline = classification.recommended_pipeline.value

        # Override if force_agentic is set
        if request.force_agentic:
            pipeline = "agentic_search"
            logger.info(f"[{request_id}] Force agentic override applied")

        # Emit pipeline routing decision
        await emitter.emit(events.pipeline_routed(
            request_id=request_id,
            pipeline=pipeline,
            reason=classification.reasoning
        ))

        # ===== STEP 3: EXECUTE APPROPRIATE PIPELINE =====

        if pipeline == "direct_answer":
            # Direct LLM response without search
            await _execute_direct_answer(request, request_id, emitter, graph_state)

        elif pipeline == "web_search":
            # Simple web search + synthesis
            await _execute_simple_search(request, request_id, emitter, graph_state)

        elif pipeline in ["agentic_search", "code_assistant"]:
            # Full agentic pipeline
            await _execute_agentic_pipeline(request, request_id, emitter, graph_state, classification)

        else:
            # Default to agentic for unknown
            logger.warning(f"[{request_id}] Unknown pipeline '{pipeline}', defaulting to agentic")
            await _execute_agentic_pipeline(request, request_id, emitter, graph_state, classification)

    except Exception as e:
        logger.error(f"[{request_id}] Gateway execution failed: {e}", exc_info=True)
        await emitter.emit(SearchEvent(
            event_type=EventType.SEARCH_FAILED,
            request_id=request_id,
            message=f"Gateway error: {str(e)}",
            progress_percent=0,
            data={"error": str(e), "gateway_error": True}
        ))
    finally:
        await emitter.emit(None)  # Signal end of stream


async def _execute_direct_answer(
    request: ChatGatewayRequest,
    request_id: str,
    emitter,
    graph_state
):
    """Execute direct LLM answer without web search."""
    import httpx
    import os

    await emitter.emit(SearchEvent(
        event_type=EventType.SYNTHESIZING,
        request_id=request_id,
        message="Generating response from knowledge...",
        progress_percent=50,
        data={"model": request.model, "pipeline": "direct_answer"}
    ))

    ollama_url = get_settings().ollama_base_url

    # Build messages
    messages = []
    if request.conversation_history:
        for msg in request.conversation_history:
            messages.append(msg)
    messages.append({"role": "user", "content": request.query})

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ollama_url}/api/chat",
            json={
                "model": request.model,
                "messages": messages,
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()

    answer = result.get("message", {}).get("content", "")

    # Emit completion
    await emitter.emit(SearchEvent(
        event_type=EventType.SEARCH_COMPLETED,
        request_id=request_id,
        message="Response generated",
        progress_percent=100,
        confidence=0.9,  # Top-level for Android client
        sources_count=0,
        data={
            "gateway_complete": True,
            "pipeline_used": "direct_answer",
            "response": {"synthesized_context": answer},
            "sources": [],
            "confidence_score": 0.9,
            "classification": {
                "category": "direct",
                "pipeline": "direct_answer"
            }
        }
    ))


async def _execute_simple_search(
    request: ChatGatewayRequest,
    request_id: str,
    emitter,
    graph_state
):
    """Execute simple web search with synthesis using UniversalOrchestrator."""
    from agentic import events
    import time

    # Use preset from request (defaults to "full" for maximum features)
    preset = request.preset
    logger.info(f"[{request_id}] Simple search using preset: {preset}")
    orchestrator = await get_universal_orchestrator(preset)

    # Pass the event emitter to the orchestrator for detailed phase events
    orchestrator.set_event_emitter(emitter)

    # Initialize and update graph state for visualization
    search_idx = graph_state.add_node("search", "active")

    await emitter.emit(SearchEvent(
        event_type=EventType.SEARCHING,
        request_id=request_id,
        message=f"Searching the web (universal/{preset})...",
        progress_percent=30,
        query=request.query,
        data={"pipeline": "web_search", "preset": preset},
        graph_line=graph_state.to_line_simple()
    ))

    search_request = SearchRequest(
        query=request.query,
        user_id=request.user_id,
        search_mode=SearchMode.ADAPTIVE,
        max_iterations=10,
        min_sources=5,
        max_sources=25,
        verification_level=VerificationLevel.STANDARD,
        cache_results=True
    )

    start_time = time.time()
    response = await orchestrator.search(search_request)
    execution_time_ms = int((time.time() - start_time) * 1000)

    # Get stats for completion event
    stats = orchestrator.get_stats()
    confidence = response.data.confidence_score if response.data else 0.5

    # Mark search complete in graph
    graph_state.complete_node(search_idx, success=True)
    complete_idx = graph_state.add_node("complete", "completed")

    sources_list = (response.data.sources or []) if response.data else []
    sources_count = len(sources_list)

    await emitter.emit(SearchEvent(
        event_type=EventType.SEARCH_COMPLETED,
        request_id=request_id,
        message=f"Universal search complete (preset={preset})",
        progress_percent=100,
        confidence=confidence,  # Top-level for Android client
        sources_count=sources_count,  # Top-level for Android client
        data={
            "gateway_complete": True,
            "pipeline_used": "universal_search",
            "preset": preset,
            "response": {"synthesized_context": response.data.synthesized_context if response.data else ""},
            "sources": [s.model_dump() if hasattr(s, 'model_dump') else s for s in sources_list],
            "confidence_score": confidence,
            "search_queries": response.data.search_queries if response.data else [],
            "execution_time_ms": execution_time_ms,
            "stats": stats,
            "classification": {
                "category": "search",
                "pipeline": "web_search"
            }
        },
        graph_line=graph_state.to_line_simple()
    ))


async def _execute_agentic_pipeline(
    request: ChatGatewayRequest,
    request_id: str,
    emitter,
    graph_state,
    classification
):
    """Execute full agentic search pipeline using UniversalOrchestrator."""
    from agentic import events
    import time

    # Use preset from request (defaults to "full" for maximum features)
    # Full provides 38 features including HyDE, CRAG, Self-RAG, and experience distillation
    preset = request.preset
    logger.info(f"[{request_id}] Agentic pipeline using preset: {preset}")
    orchestrator = await get_universal_orchestrator(preset)

    # Pass the event emitter to the orchestrator so it can emit detailed phase events
    orchestrator.set_event_emitter(emitter)

    # Initialize graph state for visualization
    plan_idx = graph_state.add_node("plan", "active")

    # Emit planning event
    await emitter.emit(SearchEvent(
        event_type=EventType.PLANNING_SEARCH,
        request_id=request_id,
        message=f"Planning universal search with {preset} preset",
        progress_percent=10,
        data={"preset": preset, "features_enabled": orchestrator.get_stats().get("features_enabled", [])},
        graph_line=graph_state.to_line_simple()
    ))

    # Create SearchRequest for the UniversalOrchestrator
    search_request = SearchRequest(
        query=request.query,
        user_id=request.user_id,
        search_mode=SearchMode.ADAPTIVE,
        analyze_query=True,
        max_iterations=10,
        min_sources=5,
        max_sources=25,
        verification_level=VerificationLevel.STANDARD,
        cache_results=True
    )

    # Update graph state
    graph_state.complete_node(plan_idx, success=True)
    search_idx = graph_state.add_node("search", "active")

    # Emit searching event
    await emitter.emit(SearchEvent(
        event_type=EventType.SEARCHING,
        request_id=request_id,
        message="Executing universal search pipeline",
        progress_percent=20,
        data={"query": request.query},
        graph_line=graph_state.to_line_simple()
    ))

    # Execute search using UniversalOrchestrator
    start_time = time.time()
    response = await orchestrator.search(search_request)
    execution_time_ms = int((time.time() - start_time) * 1000)

    # Get stats for completion event
    stats = orchestrator.get_stats()
    sources_count = len(response.data.sources) if response.data and response.data.sources else 0
    confidence = response.data.confidence_score if response.data else 0.0

    # Update graph state for synthesizing
    graph_state.complete_node(search_idx, success=True)
    synth_idx = graph_state.add_node("synthesize", "active")

    # Emit synthesizing event
    await emitter.emit(SearchEvent(
        event_type=EventType.SYNTHESIZING,
        request_id=request_id,
        message="Synthesizing results",
        progress_percent=90,
        data={"sources_count": sources_count},
        graph_line=graph_state.to_line_simple()
    ))

    # Mark complete
    graph_state.complete_node(synth_idx, success=True)
    complete_idx = graph_state.add_node("complete", "completed")

    # Emit final gateway completion
    await emitter.emit(SearchEvent(
        event_type=EventType.SEARCH_COMPLETED,
        request_id=request_id,
        message=f"Universal search complete (preset={preset})",
        progress_percent=100,
        confidence=confidence,  # Top-level for Android client
        sources_count=sources_count,  # Top-level for Android client
        data={
            "gateway_complete": True,
            "pipeline_used": "universal_search",
            "preset": preset,
            "response": {"synthesized_context": response.data.synthesized_context if response.data else ""},
            "sources": [s.model_dump() if hasattr(s, 'model_dump') else s for s in (response.data.sources or [])] if response.data else [],
            "confidence_score": confidence,
            "search_queries": response.data.search_queries if response.data else [],
            "execution_time_ms": execution_time_ms,
            "stats": stats,
            "classification": {
                "category": classification.category.value,
                "pipeline": classification.recommended_pipeline.value,
                "complexity": classification.complexity.value,
                "reasoning": classification.reasoning
            }
        },
        graph_line=graph_state.to_line_simple()
    ))


# =============================================================================
# TECHNICAL DOCUMENTATION ENDPOINTS (PDF Extraction Tools Integration)
# =============================================================================
# These endpoints provide access to FANUC technical documentation via the
# PDF Extraction Tools API running on port 8002.
# =============================================================================

from core.document_graph_service import (
    DocumentGraphService,
    get_document_graph_service,
    DocumentSearchResult,
    TroubleshootingStep
)
from agentic.schemas.fanuc_schema import (
    is_fanuc_query,
    extract_error_codes,
    get_error_category
)


class TechnicalSearchRequest(BaseModel):
    """Request model for technical documentation search."""
    query: str
    max_results: int = 10
    include_related: bool = True


class TechnicalSearchResponse(BaseModel):
    """Response model for technical documentation search."""
    success: bool
    results: List[dict]
    total_results: int
    query: str
    is_fanuc_query: bool
    error_codes_detected: List[str]


class TroubleshootRequest(BaseModel):
    """Request model for troubleshooting path query."""
    error_code: str
    context: Optional[str] = None


class TroubleshootResponse(BaseModel):
    """Response model for troubleshooting path."""
    success: bool
    error_code: str
    category: Optional[str]
    steps: List[dict]
    related_errors: List[str]


# Cache for document graph service
_document_graph_service: Optional[DocumentGraphService] = None


async def get_doc_graph_service() -> DocumentGraphService:
    """Get or create the document graph service instance."""
    global _document_graph_service
    if _document_graph_service is None:
        _document_graph_service = get_document_graph_service()
    return _document_graph_service


@router.get("/technical/health")
async def technical_health():
    """
    Check health status of PDF Extraction Tools API.

    Returns:
        Health status including API availability and response time.
    """
    try:
        doc_service = await get_doc_graph_service()
        health = await doc_service.health_check()
        return JSONResponse(content={
            "success": True,
            "data": health,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "pdf_extraction_tools"
            }
        })
    except Exception as e:
        logger.error(f"Technical health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": str(e),
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "service": "pdf_extraction_tools"
                }
            }
        )


@router.post("/technical/search", response_model=TechnicalSearchResponse)
async def search_technical_docs(request: TechnicalSearchRequest):
    """
    Search FANUC technical documentation.

    This endpoint queries the PDF Extraction Tools API for relevant
    documentation based on the query. Automatically detects error codes
    and routes to appropriate search methods.

    Args:
        request: Search request with query and options

    Returns:
        Technical documentation results with source attribution.
    """
    try:
        doc_service = await get_doc_graph_service()

        # Detect if query is FANUC-related
        fanuc_query = is_fanuc_query(request.query)
        error_codes = extract_error_codes(request.query)

        # Search documentation
        results = await doc_service.search_documentation(
            query=request.query,
            max_results=request.max_results
        )

        return TechnicalSearchResponse(
            success=True,
            results=[r.model_dump() for r in results] if results else [],
            total_results=len(results) if results else 0,
            query=request.query,
            is_fanuc_query=fanuc_query,
            error_codes_detected=error_codes
        )

    except Exception as e:
        logger.error(f"Technical search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/technical/troubleshoot", response_model=TroubleshootResponse)
async def get_troubleshooting_path(request: TroubleshootRequest):
    """
    Get step-by-step troubleshooting path for a FANUC error code.

    Proxies to PDF Extraction Tools HSEA troubleshoot endpoint
    so innovations there propagate naturally to memOS.

    Args:
        request: Error code and optional context

    Returns:
        Troubleshooting context from PDF Tools.
    """
    try:
        settings = get_settings()
        pdf_api_url = getattr(settings, 'pdf_api_url', 'http://localhost:8002')

        async with aiohttp.ClientSession() as session:
            url = f"{pdf_api_url}/api/v1/search/hsea/troubleshoot/{request.error_code}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    if data.get("success"):
                        result = data.get("data", {})
                        context = result.get("context", {})
                        metadata = context.get("metadata", {})

                        # Build troubleshooting steps from the SRVO error info
                        # The actual cause/remedy are in metadata
                        steps = []

                        # Step 1: Diagnosis (cause from metadata)
                        cause = metadata.get("cause") or context.get("cause")
                        if cause:
                            steps.append({
                                "node_id": f"{request.error_code}_diagnosis",
                                "title": f"{metadata.get('title', request.error_code)} - Cause",
                                "content": cause,
                                "step_type": "diagnosis",
                                "relevance_score": 1.0,
                                "hop_number": 0,
                                "metadata": {"error_code": request.error_code}
                            })

                        # Step 2: Solution (remedy from metadata)
                        remedy = metadata.get("remedy") or context.get("remedy")
                        if remedy:
                            steps.append({
                                "node_id": f"{request.error_code}_remedy",
                                "title": "Remedy",
                                "content": remedy,
                                "step_type": "solution",
                                "relevance_score": 1.0,
                                "hop_number": 1,
                                "metadata": {"error_code": request.error_code}
                            })

                        # Step 3: Additional notes if present
                        note = metadata.get("note")
                        if note:
                            steps.append({
                                "node_id": f"{request.error_code}_note",
                                "title": "Note",
                                "content": note,
                                "step_type": "info",
                                "relevance_score": 0.9,
                                "hop_number": 2,
                                "metadata": {"error_code": request.error_code}
                            })

                        # Extract related from PDF Tools response
                        related = []
                        for r in result.get("related_codes", []):
                            code = r.get("error_code")
                            if code:
                                related.append(code)

                        return TroubleshootResponse(
                            success=True,
                            error_code=request.error_code,
                            category=context.get("category"),
                            steps=steps,
                            related_errors=related
                        )
                    else:
                        raise HTTPException(status_code=500, detail="PDF Tools returned error")
                elif resp.status == 404:
                    raise HTTPException(status_code=404, detail=f"Error code {request.error_code} not found")
                else:
                    raise HTTPException(status_code=resp.status, detail=await resp.text())

    except aiohttp.ClientError as e:
        logger.error(f"PDF Tools connection failed: {e}")
        raise HTTPException(status_code=503, detail=f"PDF Tools unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Troubleshooting path query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/technical/context")
async def get_technical_context(
    query: str = Query(..., description="Query to get technical context for"),
    error_code: Optional[str] = Query(None, description="Optional error code"),
    max_results: int = Query(5, description="Maximum results to include")
):
    """
    Get formatted technical context for RAG injection.

    Returns a formatted context string suitable for injection into
    LLM prompts alongside web search results.

    Args:
        query: The user's query
        error_code: Optional specific error code
        max_results: Maximum documents to include

    Returns:
        Formatted context string ready for LLM injection.
    """
    try:
        doc_service = await get_doc_graph_service()

        # Check if query is FANUC-related
        if not is_fanuc_query(query) and not error_code:
            return JSONResponse(content={
                "success": True,
                "data": {
                    "context": None,
                    "reason": "Query not detected as FANUC-related"
                },
                "meta": {
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            })

        # Get context for RAG
        context = await doc_service.get_context_for_rag(
            query=query,
            error_code=error_code,
            max_results=max_results
        )

        return JSONResponse(content={
            "success": True,
            "data": {
                "context": context,
                "query": query,
                "error_code": error_code,
                "max_results": max_results
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Technical context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CORPUS SYNCHRONIZATION ENDPOINTS
# =============================================================================
# These endpoints enable synchronization between the local FANUC corpus
# and the PDF Extraction Tools knowledge base.
# =============================================================================

from agentic.fanuc_corpus_builder import get_fanuc_builder, FANUCCorpusBuilder

# Cache for corpus builder
_corpus_builder: Optional[FANUCCorpusBuilder] = None


async def get_corpus_builder() -> FANUCCorpusBuilder:
    """Get or create the FANUC corpus builder instance."""
    global _corpus_builder
    if _corpus_builder is None:
        _corpus_builder = get_fanuc_builder()
    return _corpus_builder


class CorpusSyncRequest(BaseModel):
    """Request model for corpus synchronization."""
    error_codes: Optional[List[str]] = None
    pdf_api_url: str = "http://localhost:8002"
    sync_type: str = "incremental"  # incremental, full, or specific


class CorpusSyncResponse(BaseModel):
    """Response model for corpus synchronization."""
    success: bool
    synced: int
    new_entities: int
    updated_entities: int
    failed: int
    pdf_api_available: bool


class EnrichmentRequest(BaseModel):
    """Request model for entity enrichment."""
    error_code: str
    pdf_api_url: str = "http://localhost:8002"


class EnrichmentResponse(BaseModel):
    """Response model for entity enrichment."""
    success: bool
    error_code: str
    steps_added: int
    relations_added: int


@router.post("/corpus/sync", response_model=CorpusSyncResponse)
async def sync_corpus_with_pdf(request: CorpusSyncRequest):
    """
    Synchronize local FANUC corpus with PDF Extraction Tools API.

    This endpoint pulls entities from the PDF knowledge base and merges
    them with the local corpus, enabling cross-referencing between
    web-sourced knowledge and official documentation.

    Sync types:
    - incremental: Only sync new/changed entities
    - full: Re-sync all entities
    - specific: Only sync provided error_codes list
    """
    try:
        builder = await get_corpus_builder()

        result = await builder.sync_with_pdf_api(
            pdf_api_url=request.pdf_api_url,
            error_codes=request.error_codes
        )

        return CorpusSyncResponse(
            success=True,
            synced=result.get("synced", 0),
            new_entities=result.get("new", 0),
            updated_entities=result.get("updated", 0),
            failed=result.get("failed", 0),
            pdf_api_available=result.get("pdf_api_available", False)
        )

    except Exception as e:
        logger.error(f"Corpus sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/corpus/enrich", response_model=EnrichmentResponse)
async def enrich_entity_from_pdf(request: EnrichmentRequest):
    """
    Enrich a specific error code entity with data from PDF API.

    Fetches troubleshooting path and related entities from the PDF
    knowledge base and adds them as relations in the local corpus.
    """
    try:
        builder = await get_corpus_builder()

        result = await builder.enrich_from_pdf_api(
            error_code=request.error_code,
            pdf_api_url=request.pdf_api_url
        )

        if result is None:
            return EnrichmentResponse(
                success=False,
                error_code=request.error_code,
                steps_added=0,
                relations_added=0
            )

        return EnrichmentResponse(
            success=True,
            error_code=request.error_code,
            steps_added=result.get("steps_added", 0),
            relations_added=result.get("relations_added", 0)
        )

    except Exception as e:
        logger.error(f"Entity enrichment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/corpus/cross-reference")
async def cross_reference_pdf_nodes(
    pdf_api_url: str = Query("http://localhost:8002", description="PDF API URL")
):
    """
    Create cross-references between local corpus entities and PDF graph nodes.

    This enables queries to seamlessly blend local knowledge with
    PDF documentation by linking equivalent entities.
    """
    try:
        builder = await get_corpus_builder()

        result = await builder.cross_reference_pdf_nodes(
            pdf_api_url=pdf_api_url
        )

        return JSONResponse(content={
            "success": True,
            "data": result,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Cross-referencing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/corpus/stats")
async def get_corpus_stats():
    """
    Get statistics about the FANUC corpus.

    Returns entity counts, relation counts, and sync status.
    """
    try:
        builder = await get_corpus_builder()
        stats = builder.get_stats()

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to get corpus stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/corpus/entities")
async def list_corpus_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(100, description="Maximum entities to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    List entities in the FANUC corpus.

    Supports filtering by entity type and pagination.
    """
    try:
        builder = await get_corpus_builder()

        entities = list(builder.corpus.entities.values())

        # Filter by type if specified
        if entity_type:
            entities = [
                e for e in entities
                if (e.entity_type == entity_type if isinstance(e.entity_type, str)
                    else e.entity_type.value == entity_type)
            ]

        # Paginate
        total = len(entities)
        entities = entities[offset:offset + limit]

        return JSONResponse(content={
            "success": True,
            "data": {
                "entities": [
                    {
                        "id": e.id,
                        "type": e.entity_type if isinstance(e.entity_type, str) else e.entity_type.value,
                        "name": e.name,
                        "description": e.description[:200] if e.description else None,
                        "has_pdf_link": bool(hasattr(e, 'attributes') and e.attributes and e.attributes.get("pdf_node_id"))
                    }
                    for e in entities
                ],
                "total": total,
                "offset": offset,
                "limit": limit
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to list corpus entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HSEA (Hierarchical Stratified Embedding Architecture) ENDPOINTS
# =============================================================================
# Three-stratum semantic search with cross-traversal capabilities.
# Research basis: Matryoshka Representation Learning, RRF, HyDE
# =============================================================================

from agentic.hsea_controller import (
    get_hsea_controller,
    HSEAController,
    HSEAConfig,
    HSEASearchMode,
    ErrorCodeEntity,
    CrossStratumContext,
    HSEASearchResult
)


class HSEASearchRequest(BaseModel):
    """Request model for HSEA search."""
    query: str
    mode: str = "contextual"  # systemic, structural, substantive, contextual, mrl
    top_k: int = 10
    category_filter: Optional[str] = None
    enable_hyde: bool = True


class HSEAIndexRequest(BaseModel):
    """Request model for HSEA batch indexing."""
    entities: List[Dict[str, Any]]


@router.post("/hsea/search")
async def hsea_search(request: HSEASearchRequest):
    """
    HSEA semantic search across three strata.

    Modes:
        - systemic: Binary index (32x compression, ~2ms latency)
        - structural: Int8 index + memory graph (4x compression, ~5ms)
        - substantive: FP16 + hybrid (full precision, 10-50ms)
        - contextual: All strata with cross-traversal (recommended)
        - mrl: Progressive 64->256->1024->4096 refinement

    Mathematical operation:
        search(q) = argmax_e [a*sim_pi1(q,e) + b*sim_pi2(q,e) + g*sim_pi3(q,e)]

    Research basis:
        - MRL (Kusupati et al., 2022): Hierarchical dimension progression
        - RRF (Cormack et al., 2009): Multi-retriever fusion
        - HyDE (Gao et al., 2022): Query expansion
    """
    try:
        controller = get_hsea_controller()

        # Map string mode to enum
        try:
            mode = HSEASearchMode(request.mode)
        except ValueError:
            mode = HSEASearchMode.CONTEXTUAL

        result = await controller.search(
            query=request.query,
            mode=mode,
            top_k=request.top_k,
            category_filter=request.category_filter,
            enable_hyde=request.enable_hyde
        )

        return JSONResponse(content={
            "success": True,
            "data": {
                "query": result.query,
                "search_time_ms": round(result.search_time_ms, 2),
                "mode": result.mode.value,
                "dominant_categories": result.dominant_categories,
                "suggested_patterns": [
                    {
                        "name": p.name,
                        "description": p.description,
                        "steps": p.steps,
                        "applicable_categories": p.applicable_categories
                    }
                    for p in result.suggested_patterns
                ],
                "results": [
                    {
                        "error_code": ctx.entity.canonical_form,
                        "title": ctx.entity.title,
                        "score": round(ctx.score, 4),
                        "category": ctx.entity.category,
                        "cause": ctx.entity.cause[:500] if ctx.entity.cause else None,
                        "remedy": ctx.entity.remedy[:500] if ctx.entity.remedy else None,
                        "layer_1": {
                            "category_anchor": ctx.category_anchor.name if ctx.category_anchor else None,
                            "patterns": [p.name for p in ctx.troubleshooting_patterns],
                            "mrl_64d_score": round(ctx.mrl_64d_score, 4)
                        },
                        "layer_2": {
                            "related_codes": [e.canonical_form for e in ctx.related_codes[:5]],
                            "cluster_members": [e.canonical_form for e in ctx.cluster_members[:5]],
                            "auto_connections": len(ctx.auto_connections),
                            "mrl_256d_score": round(ctx.mrl_256d_score, 4)
                        },
                        "layer_3": {
                            "dense_score": round(ctx.dense_score, 4),
                            "mrl_1024d_score": round(ctx.mrl_1024d_score, 4)
                        }
                    }
                    for ctx in result.results
                ],
                "statistics": {
                    "binary_candidates": result.binary_candidates,
                    "int8_candidates": result.int8_candidates,
                    "fp16_results": result.fp16_results,
                    "mrl_progression": result.mrl_progression
                }
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"HSEA search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hsea/troubleshoot/{error_code}")
async def hsea_troubleshoot(error_code: str):
    """
    Get complete HSEA troubleshooting context.

    Cross-stratum traversal: pi_3 -> pi_2 -> pi_1
    """
    try:
        controller = get_hsea_controller()
        ctx = await controller.get_troubleshooting_context(error_code)

        if not ctx:
            raise HTTPException(404, f"Error code {error_code} not found")

        return JSONResponse(content={
            "success": True,
            "data": {
                "error_code": ctx.entity.canonical_form,
                "title": ctx.entity.title,
                "category": ctx.entity.category,
                "severity": ctx.entity.severity,
                "cause": ctx.entity.cause,
                "remedy": ctx.entity.remedy,
                "layer_1_context": {
                    "category": {
                        "name": ctx.category_anchor.name if ctx.category_anchor else None,
                        "description": ctx.category_anchor.description if ctx.category_anchor else None,
                        "error_count": ctx.category_anchor.error_count if ctx.category_anchor else 0
                    },
                    "troubleshooting_patterns": [
                        {"name": p.name, "steps": p.steps}
                        for p in ctx.troubleshooting_patterns
                    ]
                },
                "layer_2_context": {
                    "related_codes": [
                        {"code": e.canonical_form, "title": e.title}
                        for e in ctx.related_codes
                    ],
                    "cluster_members": [
                        {"code": e.canonical_form, "title": e.title}
                        for e in ctx.cluster_members[:10]
                    ],
                    "auto_connections": ctx.auto_connections[:10]
                }
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HSEA troubleshoot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hsea/similar/{error_code}")
async def hsea_similar(
    error_code: str,
    top_k: int = Query(5, description="Number of similar codes to return")
):
    """Find semantically similar error codes."""
    try:
        controller = get_hsea_controller()
        similar = await controller.find_similar(error_code, top_k)

        return JSONResponse(content={
            "success": True,
            "data": {
                "source_code": error_code.upper(),
                "similar_codes": [
                    {
                        "code": ctx.entity.canonical_form,
                        "title": ctx.entity.title,
                        "category": ctx.entity.category,
                        "score": round(ctx.score, 4),
                        "shared_patterns": [p.name for p in ctx.troubleshooting_patterns]
                    }
                    for ctx in similar
                ]
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"HSEA similar failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hsea/index/batch")
async def hsea_index_batch(request: HSEAIndexRequest):
    """Batch index error codes into HSEA system."""
    try:
        controller = get_hsea_controller()

        entity_objects = [
            ErrorCodeEntity(
                entity_id=e["entity_id"],
                canonical_form=e["canonical_form"],
                title=e["title"],
                category=e["category"],
                code_number=e.get("code_number", 0),
                cause=e.get("cause", ""),
                remedy=e.get("remedy", ""),
                severity=e.get("severity", "alarm"),
                related_codes=e.get("related_codes", []),
                page_number=e.get("page_number")
            )
            for e in request.entities
        ]

        stats = await controller.index_batch(entity_objects)

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"HSEA batch index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hsea/index/loaded")
async def hsea_index_loaded(
    limit: int = 100,
    category: Optional[str] = None
):
    """
    Index entities that were loaded from database but not yet in embedding indices.

    Args:
        limit: Maximum number of entities to index (default 100)
        category: Only index entities from this category (e.g., "SRVO")
    """
    try:
        controller = get_hsea_controller()
        stats = await controller.index_loaded_entities(limit=limit, category_filter=category)

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        })

    except Exception as e:
        logger.error(f"HSEA index loaded failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hsea/stats")
async def hsea_stats():
    """Get HSEA system statistics."""
    try:
        controller = get_hsea_controller()
        stats = controller.get_stats()

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"HSEA stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HSEA Sync from PDF Extraction Tools
# =============================================================================

import aiohttp

class HSEASyncRequest(BaseModel):
    """Request model for syncing HSEA from PDF Tools."""
    limit: int = 100
    offset: int = 0
    category: Optional[str] = None
    generate_embeddings: bool = True


@router.post("/hsea/sync/pdf-tools")
async def hsea_sync_from_pdf_tools(
    request: HSEASyncRequest,
    background_tasks: BackgroundTasks
):
    """
    Sync error code entities from PDF Extraction Tools into HSEA.

    This endpoint:
    1. Fetches entities from PDF Tools export endpoint
    2. Generates Matryoshka embeddings for each entity
    3. Indexes them in the HSEA system

    Args:
        limit: Maximum number of entities per batch (default 100)
        offset: Starting offset for pagination
        category: Filter to specific category (e.g., "SRVO")
        generate_embeddings: Generate new embeddings via PDF Tools API
    """
    settings = get_settings()
    pdf_api_url = getattr(settings, 'pdf_api_url', 'http://localhost:8002')

    try:
        # Step 1: Fetch entities from PDF Tools
        async with aiohttp.ClientSession() as session:
            params = {
                "limit": request.limit,
                "offset": request.offset,
                "include_embeddings": "true" if not request.generate_embeddings else "false"
            }
            if request.category:
                params["category"] = request.category

            async with session.get(
                f"{pdf_api_url}/api/v1/search/hsea/export/entities",
                params=params,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=resp.status,
                        detail=f"PDF Tools export failed: {await resp.text()}"
                    )
                export_data = await resp.json()

        if not export_data.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"PDF Tools export failed: {export_data.get('errors', [])}"
            )

        entities = export_data["data"]["entities"]
        logger.info(f"Fetched {len(entities)} entities from PDF Tools")

        # Step 2: Generate embeddings if needed
        if request.generate_embeddings:
            async with aiohttp.ClientSession() as session:
                for entity in entities:
                    # Create text for embedding from entity fields
                    embed_text = f"{entity['canonical_form']} {entity['title']} {entity.get('cause', '')} {entity.get('remedy', '')}"
                    embed_text = embed_text.strip()

                    try:
                        async with session.post(
                            f"{pdf_api_url}/api/v1/embeddings/generate",
                            json={"text": embed_text, "dimensions": [128, 256, 768]},
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status == 200:
                                embed_data = await resp.json()
                                if embed_data.get("success"):
                                    embeddings = embed_data["data"]["embeddings"]
                                    entity["embeddings"] = {
                                        "systemic": embeddings.get("128d", []),
                                        "structural": embeddings.get("256d", []),
                                        "substantive": embeddings.get("768d", [])
                                    }
                    except Exception as e:
                        logger.warning(f"Failed to generate embeddings for {entity['canonical_form']}: {e}")

        # Step 3: Index into HSEA
        controller = get_hsea_controller()
        entity_objects = []

        for e in entities:
            try:
                entity_obj = ErrorCodeEntity(
                    entity_id=e["entity_id"],
                    canonical_form=e["canonical_form"],
                    title=e["title"],
                    category=e["category"],
                    code_number=e.get("code_number", 0),
                    cause=e.get("cause", ""),
                    remedy=e.get("remedy", ""),
                    severity=e.get("severity", "alarm"),
                    related_codes=e.get("related_codes", [])
                )
                entity_objects.append(entity_obj)
            except Exception as ex:
                logger.warning(f"Failed to create entity object for {e.get('canonical_form', 'unknown')}: {ex}")

        if entity_objects:
            stats = await controller.index_batch(entity_objects)
        else:
            stats = {"indexed": 0, "skipped": 0}

        return JSONResponse(content={
            "success": True,
            "data": {
                "fetched": len(entities),
                "with_embeddings": sum(1 for e in entities if e.get("embeddings")),
                "indexed": stats.get("indexed", 0),
                "skipped": stats.get("skipped", 0),
                "offset": request.offset,
                "has_more": len(entities) == request.limit
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pdf_api_url": pdf_api_url
            },
            "errors": []
        })

    except aiohttp.ClientError as e:
        logger.error(f"HSEA sync connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to PDF Tools API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"HSEA sync failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hsea/sync/status")
async def hsea_sync_status():
    """
    Get the current sync status between memOS HSEA and PDF Tools.

    Compares entity counts and coverage between systems.
    """
    settings = get_settings()
    pdf_api_url = getattr(settings, 'pdf_api_url', 'http://localhost:8002')

    try:
        # Get PDF Tools stats
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{pdf_api_url}/api/v1/search/hsea/export/stats",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    pdf_stats = await resp.json()
                else:
                    pdf_stats = {"success": False, "error": f"Status {resp.status}"}

        # Get memOS HSEA stats
        controller = get_hsea_controller()
        memos_stats = controller.get_stats()

        return JSONResponse(content={
            "success": True,
            "data": {
                "pdf_tools": {
                    "total_entities": pdf_stats.get("data", {}).get("error_code_entities", 0),
                    "with_embeddings": pdf_stats.get("data", {}).get("entities_with_embeddings", 0),
                    "categories": pdf_stats.get("data", {}).get("category_count", 0)
                },
                "memos_hsea": {
                    "indexed_entities": memos_stats.get("layer_3_entities", 0),
                    "systemic_anchors": memos_stats.get("layer_1_anchors", 0),
                    "memory_connections": memos_stats.get("layer_2_connections", 0)
                },
                "sync_gap": max(
                    0,
                    pdf_stats.get("data", {}).get("error_code_entities", 0) -
                    memos_stats.get("layer_3_entities", 0)
                )
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pdf_api_url": pdf_api_url
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"HSEA sync status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# G.1.2: Redis Embeddings Cache Endpoints
# =============================================================================

@router.get("/embeddings-cache/stats")
async def get_embeddings_cache_stats():
    """
    Get statistics for the Redis embeddings cache.

    Returns tier-level statistics including hit rates, entry counts,
    and compression ratios.
    """
    try:
        from agentic.redis_embeddings_cache import get_redis_embeddings_cache_async

        cache = await get_redis_embeddings_cache_async()
        stats = await cache.get_stats()

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Embeddings cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings-cache/store")
async def store_embedding_in_cache(
    key: str = Query(..., description="Unique key for the embedding"),
    tier: str = Query("cold", description="Cache tier: hot, warm, or cold"),
    text: Optional[str] = Body(None, description="Optional text to embed")
):
    """
    Store an embedding in the cache at the specified tier.

    If text is provided, generates embedding first. Otherwise expects
    embedding data in request body.
    """
    try:
        from agentic.redis_embeddings_cache import (
            get_redis_embeddings_cache_async,
            CacheTier
        )
        import numpy as np

        cache = await get_redis_embeddings_cache_async()

        # Map tier string to enum
        tier_map = {
            "hot": CacheTier.HOT,
            "warm": CacheTier.WARM,
            "cold": CacheTier.COLD
        }
        cache_tier = tier_map.get(tier.lower(), CacheTier.COLD)

        if text:
            # Generate embedding from text
            from agentic.mixed_precision_embeddings import get_mixed_precision_service
            service = get_mixed_precision_service()
            embedding = await service.get_embedding(text)
        else:
            raise HTTPException(
                status_code=400,
                detail="text parameter is required"
            )

        success = await cache.put(key, embedding, cache_tier)

        return JSONResponse(content={
            "success": success,
            "data": {
                "key": key,
                "tier": tier,
                "dimension": len(embedding)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Store embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embeddings-cache/get/{key}")
async def get_embedding_from_cache(
    key: str,
    min_tier: str = Query("hot", description="Minimum acceptable tier")
):
    """
    Retrieve an embedding from the cache.

    Searches from hot to cold tier, returning the first match.
    """
    try:
        from agentic.redis_embeddings_cache import (
            get_redis_embeddings_cache_async,
            CacheTier
        )

        cache = await get_redis_embeddings_cache_async()

        tier_map = {
            "hot": CacheTier.HOT,
            "warm": CacheTier.WARM,
            "cold": CacheTier.COLD
        }
        cache_tier = tier_map.get(min_tier.lower(), CacheTier.HOT)

        result = await cache.get(key, cache_tier)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Embedding not found: {key}"
            )

        embedding, found_tier = result

        return JSONResponse(content={
            "success": True,
            "data": {
                "key": key,
                "tier": found_tier.value,
                "dimension": len(embedding),
                "embedding": embedding.tolist()[:10] + ["..."]  # Truncated for display
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/embeddings-cache/{key}")
async def invalidate_embedding(
    key: str,
    tier: Optional[str] = Query(None, description="Specific tier to clear, or all if None")
):
    """
    Invalidate a cached embedding.

    Can target a specific tier or all tiers.
    """
    try:
        from agentic.redis_embeddings_cache import (
            get_redis_embeddings_cache_async,
            CacheTier
        )

        cache = await get_redis_embeddings_cache_async()

        cache_tier = None
        if tier:
            tier_map = {
                "hot": CacheTier.HOT,
                "warm": CacheTier.WARM,
                "cold": CacheTier.COLD
            }
            cache_tier = tier_map.get(tier.lower())

        count = await cache.invalidate(key, cache_tier)

        return JSONResponse(content={
            "success": True,
            "data": {
                "key": key,
                "tier": tier or "all",
                "entries_invalidated": count
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Invalidate embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/embeddings-cache/tier/{tier}")
async def clear_cache_tier(tier: str):
    """
    Clear all entries from a specific cache tier.
    """
    try:
        from agentic.redis_embeddings_cache import (
            get_redis_embeddings_cache_async,
            CacheTier
        )

        cache = await get_redis_embeddings_cache_async()

        tier_map = {
            "hot": CacheTier.HOT,
            "warm": CacheTier.WARM,
            "cold": CacheTier.COLD
        }
        cache_tier = tier_map.get(tier.lower())

        if cache_tier is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tier: {tier}. Use hot, warm, or cold."
            )

        count = await cache.clear_tier(cache_tier)

        return JSONResponse(content={
            "success": True,
            "data": {
                "tier": tier,
                "entries_cleared": count
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear cache tier failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings-cache/warm")
async def warm_embeddings_cache(
    texts: List[str] = Body(..., description="List of texts to warm"),
    tier: str = Query("warm", description="Target tier for warming")
):
    """
    Pre-warm the cache with embeddings for frequently accessed texts.

    Generates embeddings for all provided texts and stores them
    in the specified tier.
    """
    try:
        from agentic.redis_embeddings_cache import (
            get_redis_embeddings_cache_async,
            CacheTier
        )
        from agentic.mixed_precision_embeddings import get_mixed_precision_service
        import numpy as np

        cache = await get_redis_embeddings_cache_async()
        service = get_mixed_precision_service()

        tier_map = {
            "hot": CacheTier.HOT,
            "warm": CacheTier.WARM,
            "cold": CacheTier.COLD
        }
        cache_tier = tier_map.get(tier.lower(), CacheTier.WARM)

        stored = 0
        for text in texts:
            embedding = await service.get_embedding(text)
            key = cache._hash_text(text)
            if await cache.put(key, embedding, cache_tier):
                stored += 1

        return JSONResponse(content={
            "success": True,
            "data": {
                "texts_provided": len(texts),
                "embeddings_stored": stored,
                "tier": tier
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Warm embeddings cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings-cache/demote-stale")
async def demote_stale_embeddings(
    max_age_seconds: int = Query(3600, description="Max age before demotion"),
    tier: str = Query("hot", description="Tier to check for stale entries")
):
    """
    Demote stale entries from a tier to a colder tier.

    Entries not accessed within max_age_seconds are demoted.
    """
    try:
        from agentic.redis_embeddings_cache import (
            get_redis_embeddings_cache_async,
            CacheTier
        )

        cache = await get_redis_embeddings_cache_async()

        tier_map = {
            "hot": CacheTier.HOT,
            "warm": CacheTier.WARM
        }
        cache_tier = tier_map.get(tier.lower())

        if cache_tier is None:
            raise HTTPException(
                status_code=400,
                detail="Only hot and warm tiers can be demoted"
            )

        count = await cache.demote_stale_entries(max_age_seconds, cache_tier)

        return JSONResponse(content={
            "success": True,
            "data": {
                "tier": tier,
                "max_age_seconds": max_age_seconds,
                "entries_demoted": count
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demote stale embeddings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# G.1.6: Cross-Encoder Reranker Endpoints
# =============================================================================

@router.get("/reranker/stats")
async def get_reranker_stats():
    """
    Get statistics for the cross-encoder reranker.

    Returns model status, usage stats, and latency metrics.
    """
    try:
        from agentic.cross_encoder_reranker import get_cross_encoder_reranker

        reranker = get_cross_encoder_reranker()
        stats = reranker.get_stats()

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Reranker stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class RerankerRequest(BaseModel):
    """Request model for reranking."""
    query: str
    documents: List[Dict[str, Any]]
    top_k: int = 10
    score_threshold: float = 0.0


@router.post("/reranker/rerank")
async def rerank_documents(request: RerankerRequest):
    """
    Rerank documents using cross-encoder.

    Expects a query and list of documents with 'doc_id' and 'content' fields.
    Returns reranked documents with original and reranked scores.
    """
    try:
        from agentic.cross_encoder_reranker import get_cross_encoder_reranker

        reranker = get_cross_encoder_reranker()
        results, stats = await reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
            score_threshold=request.score_threshold
        )

        return JSONResponse(content={
            "success": True,
            "data": {
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "original_score": r.original_score,
                        "rerank_score": r.rerank_score,
                        "content": r.content[:200] + "..." if len(r.content) > 200 else r.content
                    }
                    for r in results
                ],
                "stats": {
                    "input_count": stats.input_count,
                    "output_count": stats.output_count,
                    "rerank_time_ms": stats.rerank_time_ms,
                    "max_score": stats.max_score,
                    "min_score": stats.min_score,
                    "avg_score": stats.avg_score
                }
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reranker/load")
async def load_reranker_model():
    """
    Pre-load the reranker model.

    Loads the model into VRAM for faster subsequent reranking.
    Model size: ~1GB VRAM.
    """
    try:
        from agentic.cross_encoder_reranker import get_cross_encoder_reranker_async

        reranker = await get_cross_encoder_reranker_async()
        stats = reranker.get_stats()

        return JSONResponse(content={
            "success": True,
            "data": {
                "status": "loaded" if stats["model_loaded"] else "failed",
                "model_name": stats["model_name"],
                "use_fp16": stats["use_fp16"]
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Load reranker failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reranker/unload")
async def unload_reranker_model():
    """
    Unload the reranker model to free VRAM.
    """
    try:
        from agentic.cross_encoder_reranker import get_cross_encoder_reranker

        reranker = get_cross_encoder_reranker()
        reranker.unload_model()

        return JSONResponse(content={
            "success": True,
            "data": {
                "status": "unloaded"
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Unload reranker failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Search Provider Circuit Breaker Endpoints
# =============================================================================

@router.get("/providers/status")
async def get_search_providers_status():
    """
    Get status of all search providers and their circuit breaker state.

    Returns:
        - Provider availability and rate limit status
        - SearXNG engine status (google, bing, etc.)
        - Domain scrape success rates
        - Recent query history

    Use this endpoint to monitor search provider health and
    diagnose rate limiting issues.
    """
    try:
        from agentic.search_metrics import get_search_metrics

        metrics = get_search_metrics()
        summary = metrics.get_summary()

        # Add availability status for each provider
        providers = summary.get("providers", {})
        for provider_name in providers.keys():
            available, reason = metrics.is_provider_available(provider_name)
            providers[provider_name]["available"] = available
            providers[provider_name]["availability_reason"] = reason

        return JSONResponse(content={
            "success": True,
            "data": summary,
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": "Search provider and engine circuit breaker status"
            },
            "errors": []
        })

    except Exception as e:
        logger.error(f"Get provider status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers/{provider}/status")
async def get_provider_status(provider: str):
    """
    Get detailed status for a specific search provider.

    Args:
        provider: Provider name (searxng, duckduckgo, brave)
    """
    try:
        from agentic.search_metrics import get_search_metrics

        metrics = get_search_metrics()
        stats = metrics.get_provider_stats(provider)

        if stats is None:
            return JSONResponse(content={
                "success": True,
                "data": {
                    "name": provider,
                    "status": "no_data",
                    "message": f"No usage data for provider '{provider}'"
                },
                "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
                "errors": []
            })

        available, reason = metrics.is_provider_available(provider)
        stats["available"] = available
        stats["availability_reason"] = reason

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": []
        })

    except Exception as e:
        logger.error(f"Get provider status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engines/status")
async def get_engines_status():
    """
    Get status of SearXNG internal engines with circuit breaker state.

    Shows which engines are available and which are in backoff
    due to rate limits, CAPTCHAs, or timeouts.
    """
    try:
        from agentic.search_metrics import get_search_metrics

        metrics = get_search_metrics()

        # Get all engine stats
        engine_list = ["google", "bing", "duckduckgo", "brave", "wikipedia",
                       "arxiv", "semantic_scholar", "google_scholar", "pubmed",
                       "github", "stackoverflow", "pypi", "npm", "dockerhub"]

        available, skipped = metrics.get_available_engines(engine_list)

        engines_detail = {}
        for engine in engine_list:
            stats = metrics.get_engine_stats(engine)
            if stats:
                engines_detail[engine] = stats
            else:
                engines_detail[engine] = {
                    "name": engine,
                    "total_queries": 0,
                    "status": "unused"
                }

        return JSONResponse(content={
            "success": True,
            "data": {
                "available_count": len(available),
                "skipped_count": len(skipped),
                "available_engines": available,
                "skipped_engines": skipped,
                "engine_details": engines_detail
            },
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": []
        })

    except Exception as e:
        logger.error(f"Get engines status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/providers/reset")
async def reset_provider_rate_limits(provider: Optional[str] = Query(default=None)):
    """
    Reset rate limit counters for a provider or all providers.

    Args:
        provider: Optional provider name. If not specified, resets all.
    """
    try:
        from agentic.search_metrics import get_search_metrics

        metrics = get_search_metrics()

        if provider:
            metrics.reset_rate_limit(provider)
            return JSONResponse(content={
                "success": True,
                "data": {"message": f"Reset rate limits for {provider}"},
                "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
                "errors": []
            })
        else:
            for p in ["searxng", "duckduckgo", "brave"]:
                metrics.reset_rate_limit(p)
            return JSONResponse(content={
                "success": True,
                "data": {"message": "Reset rate limits for all providers"},
                "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
                "errors": []
            })

    except Exception as e:
        logger.error(f"Reset rate limits failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Hyperbolic Embeddings Endpoints (G.7.2)
# =============================================================================

@router.get("/hyperbolic/stats")
async def get_hyperbolic_stats():
    """
    Get hyperbolic retriever statistics.

    Returns statistics about indexed documents, hierarchy distribution,
    embedding dimensions, and curvature settings.
    """
    try:
        from agentic.hyperbolic_embeddings import get_hyperbolic_retriever

        retriever = get_hyperbolic_retriever()
        stats = retriever.get_stats()

        return JSONResponse(content={
            "success": True,
            "data": stats,
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": []
        })

    except Exception as e:
        logger.error(f"Hyperbolic stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperbolic/index")
async def index_hyperbolic_document(
    doc_id: str = Body(..., description="Unique document ID"),
    content: str = Body(..., description="Document text content"),
    embedding: List[float] = Body(..., description="Euclidean embedding vector"),
    hierarchy_level: str = Body("STEP", description="Hierarchy level: CORPUS, MANUAL, CHAPTER, SECTION, PROCEDURE, STEP"),
    parent_id: Optional[str] = Body(None, description="Parent document ID for tree structure"),
    metadata: Optional[Dict[str, Any]] = Body(None, description="Additional metadata")
):
    """
    Index a document with hyperbolic embedding.

    Projects the Euclidean embedding to Poincaré ball space with hierarchy-aware
    depth encoding. More specific documents (STEP) are placed closer to the ball
    boundary, while general documents (CORPUS) are near the origin.
    """
    try:
        from agentic.hyperbolic_embeddings import (
            get_hyperbolic_retriever,
            HierarchyLevel
        )
        import numpy as np

        retriever = get_hyperbolic_retriever()
        level = HierarchyLevel[hierarchy_level.upper()]

        doc = await retriever.add_document(
            doc_id=doc_id,
            content=content,
            euclidean_embedding=np.array(embedding, dtype=np.float32),
            hierarchy_level=level,
            parent_id=parent_id,
            metadata=metadata or {}
        )

        return JSONResponse(content={
            "success": True,
            "data": {
                "doc_id": doc.doc_id,
                "hierarchy_level": doc.hierarchy_level.name,
                "depth": float(doc.depth),
                "euclidean_dim": len(doc.euclidean_embedding),
                "hyperbolic_dim": len(doc.hyperbolic_embedding)
            },
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": []
        })

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hierarchy level: {hierarchy_level}. Valid: CORPUS, MANUAL, CHAPTER, SECTION, PROCEDURE, STEP"
        )
    except Exception as e:
        logger.error(f"Hyperbolic index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperbolic/search")
async def search_hyperbolic(
    query_embedding: List[float] = Body(..., description="Query embedding vector"),
    top_k: int = Body(10, description="Number of results to return"),
    hierarchy_filter: Optional[List[str]] = Body(None, description="Filter by hierarchy levels"),
    use_fusion: bool = Body(True, description="Use Euclidean-Hyperbolic score fusion")
):
    """
    Search using hyperbolic geometry for hierarchy-aware retrieval.

    Combines Euclidean cosine similarity (40%) with hyperbolic geodesic distance (60%)
    for improved retrieval of hierarchically structured documents.
    """
    try:
        from agentic.hyperbolic_embeddings import (
            get_hyperbolic_retriever,
            HierarchyLevel
        )
        import numpy as np

        retriever = get_hyperbolic_retriever()

        # Convert hierarchy filter strings to enums
        level_filter = None
        if hierarchy_filter:
            level_filter = [HierarchyLevel[level.upper()] for level in hierarchy_filter]

        results = await retriever.search(
            query_embedding=np.array(query_embedding, dtype=np.float32),
            top_k=top_k,
            hierarchy_filter=level_filter,
            use_fusion=use_fusion
        )

        return JSONResponse(content={
            "success": True,
            "data": {
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "content": r.content[:500] + "..." if len(r.content) > 500 else r.content,
                        "euclidean_score": float(r.euclidean_score),
                        "hyperbolic_score": float(r.hyperbolic_score),
                        "fused_score": float(r.fused_score),
                        "hierarchy_level": r.hierarchy_level.name,
                        "depth": float(r.depth),
                        "metadata": r.metadata
                    }
                    for r in results
                ],
                "total": len(results)
            },
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "use_fusion": use_fusion
            },
            "errors": []
        })

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hierarchy level in filter. Valid: CORPUS, MANUAL, CHAPTER, SECTION, PROCEDURE, STEP"
        )
    except Exception as e:
        logger.error(f"Hyperbolic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hyperbolic/search-by-hierarchy")
async def search_hyperbolic_by_hierarchy(
    query_embedding: List[float] = Body(..., description="Query embedding vector"),
    target_level: str = Body(..., description="Primary hierarchy level to search"),
    top_k: int = Body(5, description="Number of results per level"),
    include_parents: bool = Body(True, description="Include parent levels in results"),
    include_children: bool = Body(True, description="Include child levels in results")
):
    """
    Search with hierarchy-aware expansion.

    Returns results grouped by hierarchy level, optionally including
    parent levels (more general) and child levels (more specific).
    """
    try:
        from agentic.hyperbolic_embeddings import (
            get_hyperbolic_retriever,
            HierarchyLevel
        )
        import numpy as np

        retriever = get_hyperbolic_retriever()
        level = HierarchyLevel[target_level.upper()]

        results = await retriever.search_by_hierarchy(
            query_embedding=np.array(query_embedding, dtype=np.float32),
            target_level=level,
            top_k=top_k,
            include_parents=include_parents,
            include_children=include_children
        )

        # Format results
        formatted = {}
        for level_name, level_results in results.items():
            formatted[level_name] = [
                {
                    "doc_id": r.doc_id,
                    "content": r.content[:300] + "..." if len(r.content) > 300 else r.content,
                    "fused_score": float(r.fused_score),
                    "depth": float(r.depth)
                }
                for r in level_results
            ]

        return JSONResponse(content={
            "success": True,
            "data": {
                "results_by_level": formatted,
                "target_level": target_level.upper(),
                "include_parents": include_parents,
                "include_children": include_children
            },
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": []
        })

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid hierarchy level: {target_level}. Valid: CORPUS, MANUAL, CHAPTER, SECTION, PROCEDURE, STEP"
        )
    except Exception as e:
        logger.error(f"Hyperbolic hierarchy search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hyperbolic/tree/{doc_id}")
async def get_hyperbolic_document_tree(
    doc_id: str,
    max_depth: int = Query(3, description="Maximum tree depth to traverse")
):
    """
    Get document with its hierarchy tree.

    Returns the document along with its parent and children,
    useful for navigating hierarchical documentation.
    """
    try:
        from agentic.hyperbolic_embeddings import get_hyperbolic_retriever

        retriever = get_hyperbolic_retriever()
        tree = await retriever.get_document_tree(doc_id, max_depth=max_depth)

        if not tree:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        return JSONResponse(content={
            "success": True,
            "data": tree,
            "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
            "errors": []
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hyperbolic tree failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# G.7.3: Optimal Transport Fusion Endpoints (December 2025)
# Based on Wasserstein distance and Sinkhorn algorithm for dense-sparse fusion
# =============================================================================

@router.get("/ot/stats")
async def get_ot_stats():
    """
    Get Optimal Transport fusion statistics.

    Returns:
        OT fusion statistics and configuration
    """
    try:
        from agentic.optimal_transport import get_ot_fusion

        fusion = get_ot_fusion()
        stats = fusion.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"OT stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ot/fuse")
async def fuse_with_ot(
    dense_results: List[Dict[str, float]] = Body(..., description="Dense retrieval results [{doc_id: score}]"),
    sparse_results: List[Dict[str, float]] = Body(..., description="Sparse retrieval results [{doc_id: score}]"),
    top_k: int = Body(10, description="Number of results to return"),
    epsilon: float = Body(0.1, description="Sinkhorn entropy regularization"),
    dense_weight: float = Body(0.5, description="Weight for dense scores"),
    sparse_weight: float = Body(0.5, description="Weight for sparse scores")
):
    """
    Fuse dense and sparse retrieval results using Optimal Transport.

    Uses Sinkhorn algorithm to find optimal alignment between retrieval distributions.

    Args:
        dense_results: List of {doc_id: score} from dense retrieval
        sparse_results: List of {doc_id: score} from sparse retrieval
        top_k: Number of results to return
        epsilon: Entropy regularization (lower = closer to exact OT)
        dense_weight: Weight for dense scores in fusion
        sparse_weight: Weight for sparse scores in fusion

    Returns:
        Fused results with transport-weighted scores
    """
    try:
        from agentic.optimal_transport import OptimalTransportFusion, OTConfig

        # Convert dict format to (doc_id, score) pairs
        dense_pairs = [(list(d.keys())[0], list(d.values())[0]) for d in dense_results]
        sparse_pairs = [(list(d.keys())[0], list(d.values())[0]) for d in sparse_results]

        config = OTConfig(
            epsilon=epsilon,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )

        fusion = OptimalTransportFusion(config)
        results = fusion.fuse_scores(dense_pairs, sparse_pairs, top_k=top_k)

        return {
            "success": True,
            "data": {
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "fused_score": r.fused_score,
                        "dense_score": r.dense_score,
                        "sparse_score": r.sparse_score,
                        "transport_weight": r.transport_weight,
                        "rank": r.rank,
                        "metadata": r.metadata
                    }
                    for r in results
                ],
                "stats": fusion.get_stats()
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"OT fusion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ot/fuse-multiway")
async def fuse_multiway_with_ot(
    result_lists: Dict[str, List[Dict[str, float]]] = Body(..., description="Retriever results {name: [{doc_id: score}]}"),
    weights: Optional[Dict[str, float]] = Body(None, description="Weights for each retriever"),
    top_k: int = Body(10, description="Number of results to return")
):
    """
    Fuse multiple retrieval result lists using Wasserstein barycenter.

    Computes the barycenter distribution that minimizes total Wasserstein distance
    to all input distributions.

    Args:
        result_lists: Dict mapping retriever name to list of {doc_id: score}
        weights: Optional weights for each retriever (sum to 1)
        top_k: Number of results to return

    Returns:
        Barycentric fused results
    """
    try:
        from agentic.optimal_transport import OptimalTransportFusion

        # Convert dict format to (doc_id, score) pairs
        converted_lists = {}
        for name, results in result_lists.items():
            converted_lists[name] = [(list(d.keys())[0], list(d.values())[0]) for d in results]

        fusion = OptimalTransportFusion()
        results = fusion.fuse_multiway(converted_lists, weights=weights, top_k=top_k)

        return {
            "success": True,
            "data": {
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "fused_score": r.fused_score,
                        "rank": r.rank,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"OT multiway fusion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ot/wasserstein-distance")
async def compute_wasserstein_distance(
    scores1: List[float] = Body(..., description="First score distribution"),
    scores2: List[float] = Body(..., description="Second score distribution")
):
    """
    Compute Wasserstein distance between two score distributions.

    Useful for measuring similarity between retrieval result distributions.

    Args:
        scores1: First score distribution
        scores2: Second score distribution

    Returns:
        Wasserstein distance
    """
    try:
        from agentic.optimal_transport import OptimalTransportFusion

        fusion = OptimalTransportFusion()
        distance = fusion.compute_wasserstein_distance(scores1, scores2)

        return {
            "success": True,
            "data": {
                "distance": float(distance),
                "interpretation": (
                    "low" if distance < 0.1 else
                    "moderate" if distance < 0.3 else
                    "high"
                )
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Wasserstein distance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ot/sliced-wasserstein")
async def compute_sliced_wasserstein_distance(
    source: List[List[float]] = Body(..., description="Source point cloud (n x d)"),
    target: List[List[float]] = Body(..., description="Target point cloud (m x d)"),
    n_projections: int = Body(50, description="Number of random projections")
):
    """
    Compute Sliced-Wasserstein distance between two point clouds.

    O(n log n) approximation via random 1D projections.
    Based on SLoSH (WACV 2024) for efficient distribution comparison.

    Returns:
        Sliced-Wasserstein distance
    """
    try:
        import numpy as np
        from agentic.optimal_transport import SlicedWassersteinSolver

        solver = SlicedWassersteinSolver(n_projections=n_projections)
        source_arr = np.array(source)
        target_arr = np.array(target)

        distance = solver.sliced_wasserstein_distance(source_arr, target_arr)

        return {
            "success": True,
            "data": {
                "distance": float(distance),
                "n_projections": n_projections,
                "source_shape": list(source_arr.shape),
                "target_shape": list(target_arr.shape),
                "interpretation": (
                    "identical" if distance < 0.01 else
                    "similar" if distance < 0.1 else
                    "moderate" if distance < 0.5 else
                    "different"
                )
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0",
                "algorithm": "sliced_wasserstein"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Sliced-Wasserstein distance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ot/sliced-wasserstein-embedding")
async def create_sliced_wasserstein_embedding(
    points: List[List[float]] = Body(..., description="Point cloud (n x d)"),
    n_projections: int = Body(50, description="Number of projections (embedding dim = 2*n)")
):
    """
    Create Sliced-Wasserstein embedding for indexing.

    Maps a point cloud to a vector where Euclidean distance
    approximates Sliced-Wasserstein distance. Useful for
    approximate nearest neighbor search in vector databases.

    Returns:
        Embedding vector of dimension (2 * n_projections)
    """
    try:
        import numpy as np
        from agentic.optimal_transport import SlicedWassersteinSolver

        solver = SlicedWassersteinSolver(n_projections=n_projections)
        points_arr = np.array(points)

        embedding = solver.sliced_wasserstein_embedding(points_arr, n_projections)

        return {
            "success": True,
            "data": {
                "embedding": embedding.tolist(),
                "dimension": len(embedding),
                "n_projections": n_projections,
                "input_shape": list(points_arr.shape)
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Sliced-Wasserstein embedding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ot/word-movers-distance")
async def compute_word_movers_distance(
    doc1_embeddings: List[List[float]] = Body(..., description="Document 1 word embeddings (n x d)"),
    doc2_embeddings: List[List[float]] = Body(..., description="Document 2 word embeddings (m x d)"),
    doc1_weights: Optional[List[float]] = Body(None, description="Word weights for document 1"),
    doc2_weights: Optional[List[float]] = Body(None, description="Word weights for document 2"),
    use_relaxed: bool = Body(False, description="Use fast O(n+m) relaxed WMD lower bound")
):
    """
    Compute Word Mover's Distance between two documents.

    Based on Kusner et al. ICML 2015 - optimal transport of word embeddings.
    Can use full WMD (more accurate) or relaxed WMD (faster, lower bound).

    Returns:
        WMD or RWMD distance and similarity score
    """
    try:
        import numpy as np
        from agentic.optimal_transport import WordMoverSolver, OTConfig

        config = OTConfig(epsilon=0.1, max_iter=100)
        solver = WordMoverSolver(config)

        doc1_arr = np.array(doc1_embeddings)
        doc2_arr = np.array(doc2_embeddings)

        weights1 = np.array(doc1_weights) if doc1_weights else None
        weights2 = np.array(doc2_weights) if doc2_weights else None

        if use_relaxed:
            distance = solver.relaxed_word_movers_distance(doc1_arr, doc2_arr, weights1, weights2)
            method = "relaxed_wmd"
        else:
            distance = solver.word_movers_distance(doc1_arr, doc2_arr, weights1, weights2)
            method = "wmd"

        # Convert distance to similarity (1 / (1 + distance))
        similarity = 1.0 / (1.0 + distance)

        return {
            "success": True,
            "data": {
                "distance": float(distance),
                "similarity": float(similarity),
                "method": method,
                "doc1_words": doc1_arr.shape[0],
                "doc2_words": doc2_arr.shape[0],
                "interpretation": (
                    "very_similar" if similarity > 0.8 else
                    "similar" if similarity > 0.6 else
                    "moderate" if similarity > 0.4 else
                    "different"
                )
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Word Mover's Distance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ot/document-similarity-matrix")
async def compute_document_similarity_matrix(
    documents: List[List[List[float]]] = Body(..., description="List of documents, each as word embeddings"),
    use_relaxed: bool = Body(True, description="Use fast RWMD (default True for large corpora)")
):
    """
    Compute pairwise document similarity matrix using Word Mover's Distance.

    Returns:
        n x n similarity matrix where entry (i,j) is similarity between doc i and doc j
    """
    try:
        import numpy as np
        from agentic.optimal_transport import WordMoverSolver, OTConfig

        config = OTConfig(epsilon=0.1, max_iter=100)
        solver = WordMoverSolver(config)

        # Convert to numpy arrays
        doc_arrays = [np.array(doc) for doc in documents]

        sim_matrix = solver.document_similarity_matrix(doc_arrays, use_relaxed=use_relaxed)

        return {
            "success": True,
            "data": {
                "similarity_matrix": sim_matrix.tolist(),
                "n_documents": len(documents),
                "method": "relaxed_wmd" if use_relaxed else "wmd",
                "avg_similarity": float(np.mean(sim_matrix)),
                "min_similarity": float(np.min(sim_matrix)),
                "max_similarity": float(np.max(sim_matrix))
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Document similarity matrix failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# K.2: Docling Document Processor Endpoints (December 2025)
# Based on arXiv:2408.09869 - 97.9% TEDS-S table extraction accuracy
# =============================================================================

@router.get("/docling/health")
async def check_docling_health():
    """
    Check if Docling service is healthy and reachable.

    Returns:
        Health status with last check timestamp
    """
    try:
        from agentic.docling_adapter import get_docling_adapter

        adapter = get_docling_adapter()
        is_healthy = await adapter.health_check()

        stats = adapter.get_stats()

        return {
            "success": True,
            "data": {
                "is_healthy": is_healthy,
                "last_health_check": stats.get("last_health_check"),
                "circuit_breaker_open": stats.get("circuit_breaker_open", False),
                "consecutive_failures": stats.get("consecutive_failures", 0),
                "base_url": adapter.base_url
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Docling health check failed: {e}")
        return {
            "success": False,
            "data": {"is_healthy": False, "error": str(e)},
            "meta": {"timestamp": datetime.now(UTC).isoformat(), "version": "1.0.0"},
            "errors": [{"code": "DOCLING_ERROR", "message": str(e)}]
        }


@router.get("/docling/stats")
async def get_docling_stats():
    """
    Get Docling adapter statistics.

    Returns:
        Request counts, cache stats, and performance metrics
    """
    try:
        from agentic.docling_adapter import get_docling_adapter

        adapter = get_docling_adapter()
        stats = adapter.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Docling stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/docling/convert")
async def convert_document(
    source: str = Body(..., description="URL or file path to document"),
    output_format: str = Body("markdown", description="Output format: markdown, json, text, html"),
    quality: str = Body("standard", description="Extraction quality: fast, standard, accurate"),
    extract_tables: bool = Body(True, description="Extract tables separately"),
    use_cache: bool = Body(True, description="Use cached results if available")
):
    """
    Convert a document using Docling.

    Supports PDF, HTML, DOCX, PPTX, and images.
    Uses TableFormer for 97.9% accurate table extraction.

    Returns:
        Extracted document with content, tables, sections, and metadata
    """
    try:
        from agentic.docling_adapter import (
            get_docling_adapter,
            DoclingFormat,
            ExtractionQuality
        )

        adapter = get_docling_adapter()

        # Map string to enum
        format_map = {
            "markdown": DoclingFormat.MARKDOWN,
            "json": DoclingFormat.JSON,
            "text": DoclingFormat.TEXT,
            "html": DoclingFormat.HTML
        }
        quality_map = {
            "fast": ExtractionQuality.FAST,
            "standard": ExtractionQuality.STANDARD,
            "accurate": ExtractionQuality.ACCURATE
        }

        doc_format = format_map.get(output_format.lower(), DoclingFormat.MARKDOWN)
        doc_quality = quality_map.get(quality.lower(), ExtractionQuality.STANDARD)

        result = await adapter.convert(
            source=source,
            output_format=doc_format,
            quality=doc_quality,
            extract_tables=extract_tables,
            use_cache=use_cache
        )

        if not result:
            return {
                "success": False,
                "data": None,
                "meta": {"timestamp": datetime.now(UTC).isoformat(), "version": "1.0.0"},
                "errors": [{"code": "DOCLING_CONVERSION_FAILED", "message": "Conversion failed"}]
            }

        return {
            "success": True,
            "data": {
                "document_id": result.document_id,
                "source_url": result.source_url,
                "document_type": result.document_type.value,
                "title": result.title,
                "content": result.content,
                "tables": [
                    {
                        "table_id": t.table_id,
                        "headers": t.headers,
                        "row_count": t.row_count,
                        "col_count": t.col_count,
                        "has_merged_cells": t.has_merged_cells,
                        "has_multi_level_header": t.has_multi_level_header,
                        "confidence": t.confidence,
                        "content": t.content
                    }
                    for t in result.tables
                ],
                "sections": result.sections,
                "metadata": result.metadata,
                "extraction_quality": result.extraction_quality.value,
                "processing_time_ms": result.processing_time_ms
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Docling conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/docling/extract-tables")
async def extract_tables(
    source: str = Body(..., description="URL or file path to document"),
    quality: str = Body("accurate", description="Extraction quality: fast, standard, accurate")
):
    """
    Extract only tables from a document with maximum accuracy.

    Uses TableFormer for complex table structure recognition.
    Handles merged cells, multi-level headers, and nested tables.

    Returns:
        List of extracted tables with structure metadata
    """
    try:
        from agentic.docling_adapter import get_docling_adapter, ExtractionQuality

        adapter = get_docling_adapter()

        quality_map = {
            "fast": ExtractionQuality.FAST,
            "standard": ExtractionQuality.STANDARD,
            "accurate": ExtractionQuality.ACCURATE
        }
        doc_quality = quality_map.get(quality.lower(), ExtractionQuality.ACCURATE)

        tables = await adapter.extract_tables(source=source, quality=doc_quality)

        return {
            "success": True,
            "data": {
                "table_count": len(tables),
                "tables": [
                    {
                        "table_id": t.table_id,
                        "headers": t.headers,
                        "content": t.content,
                        "row_count": t.row_count,
                        "col_count": t.col_count,
                        "has_merged_cells": t.has_merged_cells,
                        "has_multi_level_header": t.has_multi_level_header,
                        "confidence": t.confidence,
                        "source_page": t.source_page
                    }
                    for t in tables
                ]
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Docling table extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/docling/is-complex")
async def check_document_complexity(
    source: str = Body(..., description="URL or file path to document")
):
    """
    Check if a document has complex structure requiring Docling.

    Detects:
    - Multi-level headers
    - Merged cells
    - Complex layouts
    - Nested tables

    Returns:
        Whether document needs Docling (vs simpler tools like BeautifulSoup)
    """
    try:
        from agentic.docling_adapter import get_docling_adapter

        adapter = get_docling_adapter()
        is_complex = await adapter.is_complex_document(source=source)

        return {
            "success": True,
            "data": {
                "is_complex": is_complex,
                "recommendation": "docling" if is_complex else "simple_extraction"
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Docling complexity check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/docling/cache")
async def clear_docling_cache():
    """
    Clear the Docling adapter cache.

    Returns:
        Number of items cleared
    """
    try:
        from agentic.docling_adapter import get_docling_adapter

        adapter = get_docling_adapter()
        cleared_count = adapter.clear_cache()

        return {
            "success": True,
            "data": {
                "cleared_count": cleared_count,
                "message": f"Cleared {cleared_count} cached items"
            },
            "meta": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0"
            },
            "errors": []
        }
    except Exception as e:
        logger.error(f"Docling cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
