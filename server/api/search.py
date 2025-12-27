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

logger = logging.getLogger("api.search")

router = APIRouter(prefix="/api/v1/search", tags=["Search"])

# Global orchestrator instances (initialized on first use)
_orchestrator: Optional[AgenticOrchestrator] = None
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
