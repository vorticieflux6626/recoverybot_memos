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
from typing import Optional

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
