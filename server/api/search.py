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
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse

from agentic import AgenticOrchestrator
from agentic.models import (
    SearchRequest,
    SearchResponse,
    SimpleSearchRequest,
    VerificationLevel,
    SearchMode
)
from agentic.multi_agent import MultiAgentOrchestrator
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
from pydantic import BaseModel

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

# Global orchestrator instances (initialized on first use)
_orchestrator: Optional[AgenticOrchestrator] = None
_graph_orchestrator = None  # GraphEnhancedOrchestrator
_enhanced_orchestrator = None  # EnhancedAgenticOrchestrator
_multi_orchestrator: Optional[MultiAgentOrchestrator] = None


async def get_orchestrator() -> AgenticOrchestrator:
    """Get or create the orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        import os
        _orchestrator = AgenticOrchestrator(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            mcp_url=os.getenv("MCP_URL", "http://localhost:7777"),
            brave_api_key=os.getenv("BRAVE_API_KEY")
        )
        await _orchestrator.initialize()
        logger.info("Agentic search orchestrator initialized")
    return _orchestrator


async def get_graph_orchestrator():
    """Get or create the graph-enhanced orchestrator instance"""
    global _graph_orchestrator
    if _graph_orchestrator is None:
        import os
        from agentic.orchestrator_graph_enhanced import GraphEnhancedOrchestrator
        _graph_orchestrator = GraphEnhancedOrchestrator(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            mcp_url=os.getenv("MCP_URL", "http://localhost:7777"),
            brave_api_key=os.getenv("BRAVE_API_KEY")
        )
        await _graph_orchestrator.initialize()
        logger.info("Graph-enhanced orchestrator initialized")
    return _graph_orchestrator


async def get_enhanced_orchestrator():
    """Get or create the enhanced agentic orchestrator instance"""
    global _enhanced_orchestrator
    if _enhanced_orchestrator is None:
        import os
        from agentic.orchestrator_enhanced import EnhancedAgenticOrchestrator
        _enhanced_orchestrator = EnhancedAgenticOrchestrator(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            mcp_url=os.getenv("MCP_URL", "http://localhost:7777"),
            brave_api_key=os.getenv("BRAVE_API_KEY"),
            enable_reflection=True,
            enable_pre_act=True,
            enable_stuck_detection=True,
            enable_contradiction_detection=True,
            max_reflection_iterations=2
        )
        await _enhanced_orchestrator.initialize()
        logger.info("Enhanced agentic orchestrator initialized")
    return _enhanced_orchestrator


# Global classifier instance
_classifier: Optional[QueryClassifier] = None


async def get_classifier() -> QueryClassifier:
    """Get or create the query classifier instance"""
    global _classifier
    if _classifier is None:
        import os
        _classifier = QueryClassifier(
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "Agentic search performance metrics"
            }
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
            ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        integration = await init_graph(ollama_url)
        stats = integration.get_comprehensive_stats()

        return {
            "success": True,
            "data": {
                "status": "initialized",
                "stats": stats
            },
            "meta": {
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Clear experiences failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear experiences: {str(e)}"
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": __import__("datetime").datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        retriever = get_entity_enhanced_retriever(ollama_url)
        stats = retriever.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        classifier = get_query_classifier(ollama_url)

        return {
            "success": True,
            "data": classifier.get_stats() if hasattr(classifier, 'get_stats') else {},
            "meta": {
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        aggregator = get_embedding_aggregator(ollama_url)
        stats = aggregator.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        service = get_mixed_precision_service(ollama_url)

        stats = service.get_stats()

        return {
            "success": True,
            "data": stats,
            "meta": {
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                        "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
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
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Failed to clear HyDE cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))
