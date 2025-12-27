"""
Agentic Search Orchestrator

Coordinates the multi-agent search pipeline:
Analyzer → Planner → Searcher → Verifier → Synthesizer

Implements an enhanced ReAct (Reasoning + Acting) pattern with:
- Query analysis to determine if search is needed
- Adaptive search that continues until information is found or leads exhausted
- Dynamic search refinement based on results
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from .models import (
    SearchRequest,
    SearchResponse,
    SearchResultData,
    SearchMeta,
    SearchState,
    SearchMode,
    VerificationLevel,
    ConfidenceLevel,
    ActionType,
    QueryAnalysis,
    SearchPlan
)
from .analyzer import QueryAnalyzer
from .planner import PlannerAgent
from .searcher import SearcherAgent
from .verifier import VerifierAgent
from .synthesizer import SynthesizerAgent, DEFAULT_THINKING_MODEL
from .scraper import ContentScraper, DeepReader, VisionAnalyzer
from .scratchpad import AgenticScratchpad, ScratchpadManager, QuestionStatus, FindingType
from .content_cache import get_content_cache
from .ttl_cache_manager import (
    get_ttl_cache_manager, ToolType, ToolCallContext
)
from . import events
from .events import EventEmitter, EventType, SearchEvent

logger = logging.getLogger("agentic.orchestrator")

# Embedding service for semantic cache (lazy loaded)
_embedding_service = None

async def get_embedding_service():
    """Get or create the embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        from core.embedding_service import EmbeddingService
        _embedding_service = EmbeddingService()
    return _embedding_service


class AgenticOrchestrator:
    """
    Main orchestrator for agentic search.

    Coordinates the multi-agent pipeline and implements
    the ReAct loop for iterative refinement.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        mcp_url: str = "http://localhost:7777",
        brave_api_key: Optional[str] = None,
        memory_service: Optional[Any] = None  # Optional memOS memory service
    ):
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url

        # Initialize agents
        self.analyzer = QueryAnalyzer(ollama_url=ollama_url)
        self.planner = PlannerAgent(ollama_url=ollama_url, mcp_url=mcp_url)
        self.searcher = SearcherAgent(brave_api_key=brave_api_key)
        self.verifier = VerifierAgent(ollama_url=ollama_url)
        self.synthesizer = SynthesizerAgent(ollama_url=ollama_url, mcp_url=mcp_url)

        # Content scraper and deep reader for document analysis
        self.scraper = ContentScraper()
        self.deep_reader = DeepReader(ollama_url=ollama_url)

        # Vision analyzer for charts, graphs, and images
        self.vision_analyzer = VisionAnalyzer(ollama_url=ollama_url)

        # Optional memory service for caching
        self.memory_service = memory_service

        # Scratchpad manager for multi-agent coordination
        self.scratchpad_manager = ScratchpadManager(memory_service=memory_service)

        # TTL cache manager for tool call latency tracking
        self.ttl_manager = get_ttl_cache_manager()

        # Search cache (in-memory, cleared on restart)
        self._cache: Dict[str, SearchResponse] = {}
        self._cache_ttl = 3600  # 1 hour

    async def initialize(self):
        """Initialize orchestrator and check service availability"""
        # Check MCP availability for agents that use it
        mcp_available = await self.planner.check_mcp_available()
        await self.synthesizer.check_mcp_available()

        logger.info(f"Orchestrator initialized. MCP available: {mcp_available}")

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute enhanced agentic search pipeline with query analysis.

        Implements a ReAct (Reasoning + Acting) pattern that:
        1. Analyzes the query to determine if web search is beneficial
        2. Creates a comprehensive search plan if search is needed
        3. Adaptively searches until information is found or leads are exhausted

        Args:
            request: SearchRequest with query and options

        Returns:
            SearchResponse with synthesized results
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.info(f"[{request_id}] Starting agentic search: {request.query[:50]}...")
        logger.info(f"[{request_id}] Mode: {request.search_mode}, Max iterations: {request.max_iterations}")

        # Check in-memory cache first (fastest)
        cache_key = self._get_cache_key(request)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.meta.cache_hit = True
            logger.info(f"[{request_id}] In-memory cache hit")
            return cached

        # Check semantic cache for similar queries
        try:
            content_cache = get_content_cache()
            embedding_service = await get_embedding_service()

            # Generate query embedding
            query_embedding = await embedding_service.generate_embedding(request.query)

            # Look for semantically similar cached queries
            similar_result = content_cache.find_similar_query(
                query_embedding,
                similarity_threshold=0.88  # High threshold for search results
            )

            if similar_result:
                logger.info(f"[{request_id}] Semantic cache hit "
                           f"(matched: '{similar_result.get('matched_query', '')[:30]}...', "
                           f"similarity: {similar_result.get('similarity_score', 0):.3f})")

                # Convert cached result back to SearchResponse
                cached_data = similar_result
                response = SearchResponse(
                    success=cached_data.get("success", True),
                    data=SearchResultData(
                        synthesized_context=cached_data.get("data", {}).get("synthesized_context", ""),
                        sources=cached_data.get("data", {}).get("sources", []),
                        search_queries=cached_data.get("data", {}).get("search_queries", []),
                        search_trace=cached_data.get("data", {}).get("search_trace", []),
                        confidence_score=cached_data.get("data", {}).get("confidence_score", 0.0),
                        confidence_level=ConfidenceLevel(cached_data.get("data", {}).get("confidence_level", "medium"))
                    ),
                    meta=SearchMeta(
                        request_id=request_id,
                        timestamp=datetime.now().isoformat(),
                        execution_time_ms=0,
                        iterations=0,
                        queries_executed=0,
                        sources_consulted=0,
                        cache_hit=True,
                        semantic_match=True,
                        matched_query=similar_result.get("matched_query", ""),
                        similarity_score=similar_result.get("similarity_score", 0)
                    )
                )
                return response

        except Exception as e:
            logger.debug(f"[{request_id}] Semantic cache check failed: {e}")

        try:
            # Initialize search state with search mode
            state = SearchState(
                query=request.query,
                max_iterations=request.max_iterations,
                search_mode=request.search_mode.value
            )

            # Create scratchpad for multi-agent coordination
            scratchpad = self.scratchpad_manager.create(
                query=request.query,
                request_id=request_id,
                user_id=request.user_id
            )
            logger.info(f"[{request_id}] Created scratchpad for multi-agent coordination")

            search_trace = []

            # STEP 1: ANALYZE - Determine if web search is beneficial
            if request.analyze_query:
                logger.info(f"[{request_id}] Analyzing query to determine search necessity...")
                state.query_analysis = await self.analyzer.analyze(
                    request.query,
                    request.context
                )
                search_trace.append({
                    "step": "analyze",
                    "requires_search": state.query_analysis.requires_search,
                    "query_type": state.query_analysis.query_type,
                    "complexity": state.query_analysis.estimated_complexity,
                    "reasoning": state.query_analysis.search_reasoning,
                    "confidence": state.query_analysis.confidence
                })

                # If analysis indicates search is not needed, return early with synthesis
                if not state.query_analysis.requires_search:
                    logger.info(f"[{request_id}] Query analysis indicates no search needed")
                    synthesis = await self.synthesizer.synthesize(
                        request.query,
                        [],  # No search results
                        None,
                        request.context
                    )
                    search_trace.append({
                        "step": "skip_search",
                        "reason": state.query_analysis.search_reasoning
                    })

                    execution_time_ms = int((time.time() - start_time) * 1000)
                    return SearchResponse(
                        success=True,
                        data=SearchResultData(
                            synthesized_context=synthesis,
                            sources=[],
                            search_queries=[],
                            confidence_score=state.query_analysis.confidence,
                            confidence_level=ConfidenceLevel.MEDIUM,
                            verification_status="skipped",
                            search_trace=search_trace
                        ),
                        meta=SearchMeta(
                            request_id=request_id,
                            iterations=0,
                            queries_executed=0,
                            sources_consulted=0,
                            execution_time_ms=execution_time_ms,
                            cache_hit=False
                        )
                    )

            # STEP 2: PLAN - Create comprehensive search strategy
            logger.info(f"[{request_id}] Creating search plan...")
            if state.query_analysis:
                state.search_plan = await self.analyzer.create_search_plan(
                    request.query,
                    state.query_analysis,
                    request.context
                )
            else:
                # Create basic plan if analysis was skipped
                action = await self.planner.plan(request.query, request.context)
                state.search_plan = SearchPlan(
                    original_query=request.query,
                    decomposed_questions=[request.query],
                    search_phases=[{"phase": "initial", "queries": action.queries}],
                    priority_order=list(range(len(action.queries))),
                    fallback_strategies=["broaden search terms"],
                    estimated_iterations=request.max_iterations,
                    reasoning=action.reasoning
                )

            # Initialize pending queries from search plan
            initial_queries = []
            for phase in state.search_plan.search_phases:
                initial_queries.extend(phase.get("queries", []))
            state.add_pending_queries(initial_queries)

            # Set scratchpad mission with decomposed questions
            decomposed_qs = state.search_plan.decomposed_questions
            completion_criteria = {
                f"q{i+1}": f"Find comprehensive information about: {q}"
                for i, q in enumerate(decomposed_qs)
            }
            scratchpad.set_mission(decomposed_qs, completion_criteria)
            logger.info(f"[{request_id}] Scratchpad mission set: {len(decomposed_qs)} questions")

            search_trace.append({
                "step": "plan",
                "decomposed_questions": state.search_plan.decomposed_questions,
                "phases": len(state.search_plan.search_phases),
                "initial_queries": len(initial_queries),
                "estimated_iterations": state.search_plan.estimated_iterations,
                "reasoning": state.search_plan.reasoning,
                "scratchpad_mission_set": True
            })

            logger.info(f"[{request_id}] Search plan created: {len(initial_queries)} queries, "
                       f"{len(state.search_plan.decomposed_questions)} questions")

            # STEP 3: ADAPTIVE ReAct LOOP
            for iteration in range(request.max_iterations):
                state.iteration = iteration + 1
                logger.info(f"[{request_id}] Iteration {state.iteration}/{state.max_iterations} "
                           f"(sources: {state.sources_consulted}, domains: {len(state.unique_domains)})")

                # ACT: Execute pending searches
                if state.pending_queries:
                    queries_to_execute = state.pending_queries[:3]  # Execute up to 3 queries at a time

                    logger.info(f"[{request_id}] Executing {len(queries_to_execute)} queries: {queries_to_execute}")

                    # Record queries in scratchpad
                    for q in queries_to_execute:
                        scratchpad.record_search(q)

                    # TTL pin cache during web search (prevents eviction during ~1-3s operation)
                    async with ToolCallContext(request_id, ToolType.WEB_SEARCH, manager=self.ttl_manager):
                        results = await self.searcher.search(queries_to_execute)
                    state.add_results(results)

                    # Record preliminary findings in scratchpad (from snippets)
                    for result in results:
                        # Match result to question based on query
                        question_id = self._match_result_to_question(result, scratchpad)
                        if question_id:
                            scratchpad.add_finding(
                                question_id=question_id,
                                content=result.snippet[:500],
                                source_url=result.url,
                                source_title=result.title,
                                finding_type=FindingType.FACT,
                                confidence=result.relevance_score if hasattr(result, 'relevance_score') else 0.5
                            )

                    # Mark queries as executed
                    for q in queries_to_execute:
                        state.mark_query_executed(q)

                    # Log scratchpad status
                    sp_status = scratchpad.get_completion_status()
                    logger.info(f"[{request_id}] Scratchpad progress: {sp_status['overall']:.0%}")

                    search_trace.append({
                        "step": "search",
                        "iteration": iteration + 1,
                        "queries": queries_to_execute,
                        "results_count": len(results),
                        "total_sources": state.sources_consulted,
                        "unique_domains": len(state.unique_domains),
                        "scratchpad_progress": sp_status['overall']
                    })

                # VERIFY: If requested and we have results
                if (request.verification_level != VerificationLevel.NONE and
                    state.raw_results and
                    not state.verified_claims):

                    combined_text = " ".join(r.snippet for r in state.raw_results[:10])
                    state.claims = await self.verifier.extract_claims(combined_text)

                    if state.claims:
                        state.verified_claims = await self.verifier.verify(
                            state.claims,
                            state.raw_results,
                            request.verification_level
                        )
                        search_trace.append({
                            "step": "verify",
                            "iteration": iteration + 1,
                            "claims_verified": len(state.verified_claims),
                            "verified_count": sum(1 for v in state.verified_claims if v.verified)
                        })

                # OBSERVE: Evaluate if we should continue searching
                if request.search_mode == SearchMode.FIXED:
                    # Fixed mode: continue until max iterations
                    if iteration >= request.max_iterations - 1:
                        logger.info(f"[{request_id}] Fixed mode: max iterations reached")
                        break

                elif request.search_mode == SearchMode.ADAPTIVE:
                    # Adaptive mode: continue until information is sufficient or leads exhausted
                    if not state.can_continue_search(request.max_sources):
                        logger.info(f"[{request_id}] Cannot continue: leads exhausted or max sources reached")
                        break

                    # Check if we have enough information
                    if state.has_sufficient_sources(request.min_sources):
                        # Ask analyzer if information is sufficient
                        should_continue, reason, new_queries = await self.analyzer.should_continue_search(
                            request.query,
                            [{"title": r.title, "snippet": r.snippet} for r in state.raw_results],
                            state.search_plan,
                            iteration
                        )

                        if not should_continue:
                            logger.info(f"[{request_id}] Adaptive mode: sufficient information ({reason})")
                            state.information_sufficient = True
                            break

                        # Add new queries if provided
                        if new_queries:
                            state.add_pending_queries(new_queries)
                            state.refinement_attempts += 1
                            search_trace.append({
                                "step": "refine",
                                "iteration": iteration + 1,
                                "reason": reason,
                                "new_queries": new_queries
                            })
                            logger.info(f"[{request_id}] Adding refined queries: {new_queries}")

                elif request.search_mode == SearchMode.EXHAUSTIVE:
                    # Exhaustive mode: search all possible leads
                    if not state.pending_queries and not state.can_continue_search(request.max_sources):
                        logger.info(f"[{request_id}] Exhaustive mode: all leads explored")
                        state.leads_exhausted = True
                        break

                    # Even if sufficient, try to find more with refinements
                    if not state.pending_queries and state.refinement_attempts < state.max_refinements:
                        _, _, new_queries = await self.analyzer.should_continue_search(
                            request.query,
                            [{"title": r.title, "snippet": r.snippet} for r in state.raw_results],
                            state.search_plan,
                            iteration
                        )
                        if new_queries:
                            state.add_pending_queries(new_queries)
                            state.refinement_attempts += 1

                # Check if no more pending queries
                if not state.pending_queries:
                    logger.info(f"[{request_id}] No more pending queries")
                    state.leads_exhausted = True
                    break

            # STEP 4: URL EVALUATION, SCRAPING, AND COVERAGE CHECK
            # This step may loop back to search if content coverage is insufficient
            # OPTIMIZATION: Reduced from 2 to 1 refinement (saves ~30s per search)
            max_scrape_refinements = 1  # Allow 1 additional search round after scraping
            scrape_refinement = 0
            scraped_content = []

            while scrape_refinement <= max_scrape_refinements:
                if state.raw_results and len(state.raw_results) > 0:
                    # Convert WebSearchResult objects to dicts for evaluation
                    results_for_eval = [
                        {
                            "url": r.url,
                            "title": r.title,
                            "snippet": r.snippet,
                            "source_domain": r.source_domain
                        }
                        for r in state.raw_results
                    ]

                    # Use LLM to evaluate which URLs are worth scraping
                    logger.info(f"[{request_id}] Evaluating {len(results_for_eval)} URLs for relevance...")
                    relevant_urls = await self.analyzer.evaluate_urls_for_scraping(
                        request.query,
                        results_for_eval,
                        max_urls=8
                    )
                    logger.info(f"[{request_id}] {len(relevant_urls)} URLs deemed relevant for scraping")

                    # Scrape all relevant URLs
                    if relevant_urls:
                        urls_to_scrape = [
                            (ru["url"], ru["title"])
                            for ru in relevant_urls
                            if not ru["url"].lower().endswith('.pdf')
                        ]

                        logger.info(f"[{request_id}] Scraping {len(urls_to_scrape)} relevant sources for full content...")

                        for idx, (url, title) in enumerate(urls_to_scrape, 1):
                            # Skip URLs we've already scraped (check both local list and scratchpad)
                            if any(s.get("url") == url for s in scraped_content) or scratchpad.has_scraped(url):
                                continue
                            try:
                                # TTL pin cache during web scrape (prevents eviction during ~3-8s operation)
                                async with ToolCallContext(request_id, ToolType.WEB_SCRAPE, manager=self.ttl_manager):
                                    scraped = await self.scraper.scrape_url(url)
                                if scraped.get("success") and scraped.get("content"):
                                    content_len = len(scraped["content"])
                                    scraped_content.append({
                                        "url": url,
                                        "title": title,
                                        "content": scraped["content"][:12000],
                                        "content_type": scraped.get("content_type", "html")
                                    })
                                    # Record in scratchpad
                                    scratchpad.record_scrape(url)
                                    logger.info(f"[{request_id}] Scraped {content_len} chars from {url[:50]}")
                            except Exception as e:
                                logger.warning(f"[{request_id}] Failed to scrape {url}: {e}")

                        logger.info(f"[{request_id}] Successfully scraped {len(scraped_content)} sources")

                # STEP 4b: EVALUATE CONTENT COVERAGE (NEW)
                # Check if scraped content answers the decomposed questions
                if (scraped_content and
                    state.search_plan and
                    state.search_plan.decomposed_questions and
                    scrape_refinement < max_scrape_refinements):

                    logger.info(f"[{request_id}] Evaluating content coverage (refinement {scrape_refinement})...")

                    # TTL pin cache during LLM analysis (prevents eviction during ~5-15s operation)
                    async with ToolCallContext(request_id, ToolType.OLLAMA_GENERATE, manager=self.ttl_manager):
                        coverage = await self.analyzer.evaluate_content_coverage(
                            query=request.query,
                            decomposed_questions=state.search_plan.decomposed_questions,
                            scraped_content=scraped_content
                        )

                    search_trace.append({
                        "step": "coverage_check",
                        "refinement": scrape_refinement,
                        "coverage_score": coverage.get("coverage_score", 0),
                        "is_sufficient": coverage.get("is_sufficient", True),
                        "gaps": coverage.get("information_gaps", []),
                        "reasoning": coverage.get("reasoning", "")
                    })

                    if coverage.get("is_sufficient", True):
                        logger.info(f"[{request_id}] Content coverage sufficient ({coverage.get('coverage_score', 0):.0%})")
                        break

                    # Content not sufficient - perform additional searches
                    new_queries = coverage.get("suggested_queries", [])
                    if new_queries and state.iteration < request.max_iterations:
                        logger.info(f"[{request_id}] Content coverage insufficient ({coverage.get('coverage_score', 0):.0%})")
                        logger.info(f"[{request_id}] Gaps: {coverage.get('information_gaps', [])}")
                        logger.info(f"[{request_id}] Adding {len(new_queries)} refined queries: {new_queries}")

                        # Add new queries and perform additional search iteration
                        state.add_pending_queries(new_queries)
                        state.refinement_attempts += 1

                        # Execute the new searches
                        for iteration in range(min(2, request.max_iterations - state.iteration)):
                            if not state.pending_queries:
                                break

                            state.iteration += 1
                            queries_to_execute = state.pending_queries[:3]

                            logger.info(f"[{request_id}] Coverage refinement search: {queries_to_execute}")

                            results = await self.searcher.search(queries_to_execute)
                            state.add_results(results)

                            for q in queries_to_execute:
                                state.mark_query_executed(q)

                            search_trace.append({
                                "step": "coverage_refinement_search",
                                "refinement": scrape_refinement,
                                "queries": queries_to_execute,
                                "results_count": len(results),
                                "total_sources": state.sources_consulted
                            })

                        scrape_refinement += 1
                        continue  # Loop back to scrape new URLs

                # No more refinements needed or possible
                break

            # STEP 5: SYNTHESIZE - Use scraped content if available
            # Include scratchpad context for the synthesizer
            scratchpad_context = scratchpad.to_context_for_agent("synthesizer")

            # Determine if we should use a thinking model for synthesis
            thinking_model = None
            if state.query_analysis and state.query_analysis.requires_thinking_model:
                thinking_model = DEFAULT_THINKING_MODEL
                logger.info(f"[{request_id}] Using THINKING MODEL ({thinking_model}) for complex reasoning: "
                           f"type={state.query_analysis.query_type}, "
                           f"complexity={state.query_analysis.reasoning_complexity}")
            else:
                logger.info(f"[{request_id}] Using standard model for synthesis")

            logger.info(f"[{request_id}] Synthesizing {len(state.raw_results)} results with scratchpad context...")

            # Merge scratchpad context with user context
            synthesis_context = request.context.copy() if request.context else {}
            synthesis_context["scratchpad_summary"] = scratchpad_context

            # TTL pin cache during synthesis (prevents eviction during ~30-90s for thinking models)
            async with ToolCallContext(request_id, ToolType.OLLAMA_GENERATE, manager=self.ttl_manager):
                if scraped_content:
                    synthesis = await self.synthesizer.synthesize_with_content(
                        request.query,
                        state.raw_results,
                        scraped_content,
                        state.verified_claims if request.verification_level != VerificationLevel.NONE else None,
                        synthesis_context,
                        model_override=thinking_model
                    )
                else:
                    synthesis = await self.synthesizer.synthesize(
                        request.query,
                        state.raw_results,
                        state.verified_claims if request.verification_level != VerificationLevel.NONE else None,
                        synthesis_context
                    )

            # Mark scratchpad as complete
            scratchpad.mark_complete(f"Synthesized {len(state.raw_results)} sources from {len(state.unique_domains)} domains")

            # Get final scratchpad status
            final_scratchpad_status = scratchpad.get_completion_status()

            search_trace.append({
                "step": "synthesize",
                "iteration": state.iteration,
                "synthesis_length": len(synthesis),
                "total_sources": state.sources_consulted,
                "unique_domains": len(state.unique_domains),
                "scratchpad_status": {
                    "overall_completion": final_scratchpad_status["overall"],
                    "questions_answered": sum(1 for q in scratchpad.questions.values() if q.status == QuestionStatus.ANSWERED),
                    "findings_count": len(scratchpad.findings),
                    "contradictions_unresolved": final_scratchpad_status.get("unresolved_contradictions", 0)
                }
            })

            # Calculate confidence with multiple signals
            scraped_count = len(scraped_content) if scraped_content else 0
            confidence_score = self.verifier.calculate_overall_confidence(
                state.verified_claims,
                source_count=len(state.raw_results),
                unique_domains=len(state.unique_domains),
                synthesis_length=len(synthesis),
                scraped_sources=scraped_count
            )
            confidence_level = self.synthesizer.determine_confidence_level(
                state.verified_claims,
                len(state.raw_results)
            )

            # Determine verification status
            if request.verification_level == VerificationLevel.NONE:
                verification_status = "skipped"
            elif state.verified_claims:
                verified_ratio = sum(1 for v in state.verified_claims if v.verified) / len(state.verified_claims)
                verification_status = "verified" if verified_ratio >= 0.7 else "partial"
            else:
                # Even without explicit verification, consider it partial if we have good coverage
                if scraped_count >= 3 and len(state.unique_domains) >= 3:
                    verification_status = "partial"
                else:
                    verification_status = "unverified"

            # Build response
            execution_time_ms = int((time.time() - start_time) * 1000)

            response = SearchResponse(
                success=True,
                data=SearchResultData(
                    synthesized_context=synthesis,
                    sources=[
                        {"title": r.title, "url": r.url, "domain": r.source_domain}
                        for r in state.raw_results[:20]  # Return up to 20 sources
                    ],
                    search_queries=state.executed_queries,
                    confidence_score=confidence_score,
                    confidence_level=confidence_level,
                    verification_status=verification_status,
                    search_trace=search_trace
                ),
                meta=SearchMeta(
                    request_id=request_id,
                    iterations=state.iteration,
                    queries_executed=len(state.executed_queries),
                    sources_consulted=state.sources_consulted,
                    execution_time_ms=execution_time_ms,
                    cache_hit=False
                )
            )

            # Cache successful results
            if request.cache_results:
                self._cache[cache_key] = response

                # Also store in semantic cache with embedding for deduplication
                try:
                    content_cache = get_content_cache()
                    embedding_service = await get_embedding_service()
                    query_embedding = await embedding_service.generate_embedding(request.query)

                    # Store as JSON-serializable dict
                    cache_data = {
                        "success": response.success,
                        "data": {
                            "synthesized_context": response.data.synthesized_context,
                            "sources": response.data.sources,
                            "search_queries": response.data.search_queries,
                            "search_trace": response.data.search_trace,
                            "confidence_score": response.data.confidence_score,
                            "confidence_level": response.data.confidence_level.value
                        }
                    }
                    content_cache.set_query_result(
                        query=request.query,
                        result=cache_data,
                        embedding=query_embedding
                    )
                    logger.debug(f"[{request_id}] Stored in semantic cache with embedding")
                except Exception as cache_err:
                    logger.debug(f"[{request_id}] Semantic cache store failed: {cache_err}")

            # Store in memory service if available
            if self.memory_service and request.user_id and request.cache_results:
                await self._store_in_memory(request.user_id, request.query, response)

            logger.info(
                f"[{request_id}] Search complete: "
                f"{state.sources_consulted} sources from {len(state.unique_domains)} domains, "
                f"{state.iteration} iterations, "
                f"{execution_time_ms}ms, "
                f"confidence={confidence_score:.2f}, "
                f"mode={request.search_mode.value}"
            )

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Search failed: {e}", exc_info=True)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return SearchResponse(
                success=False,
                data=SearchResultData(
                    synthesized_context=f"I encountered an error while searching. Please try again or contact support.",
                    sources=[],
                    search_queries=[request.query],
                    confidence_score=0.0,
                    confidence_level=ConfidenceLevel.UNKNOWN,
                    verification_status="error"
                ),
                meta=SearchMeta(
                    request_id=request_id,
                    execution_time_ms=execution_time_ms
                ),
                errors=[{"type": "search_error", "message": str(e)}]
            )

    async def search_with_events(
        self,
        request: SearchRequest,
        emitter: EventEmitter
    ) -> SearchResponse:
        """
        Execute agentic search pipeline with real-time event emissions.

        This method is identical to search() but emits events at each step
        for real-time progress updates to connected clients.

        Args:
            request: SearchRequest with query and options
            emitter: EventEmitter to send progress updates to

        Returns:
            SearchResponse with synthesized results
        """
        start_time = time.time()
        request_id = emitter.request_id

        logger.info(f"[{request_id}] Starting streaming agentic search: {request.query[:50]}...")

        # Check cache
        cache_key = self._get_cache_key(request)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.meta.cache_hit = True
            await emitter.emit(events.progress_update(request_id, 100, "Returning cached result"))
            return cached

        try:
            # Initialize search state
            state = SearchState(
                query=request.query,
                max_iterations=request.max_iterations,
                search_mode=request.search_mode.value
            )

            # Create scratchpad for multi-agent coordination
            scratchpad = self.scratchpad_manager.create(
                query=request.query,
                request_id=request_id,
                user_id=request.user_id
            )

            search_trace = []

            # STEP 1: ANALYZE
            if request.analyze_query:
                await emitter.emit(events.analyzing_query(request_id, request.query))

                state.query_analysis = await self.analyzer.analyze(
                    request.query,
                    request.context
                )

                await emitter.emit(events.query_analyzed(
                    request_id,
                    state.query_analysis.requires_search,
                    state.query_analysis.query_type
                ))

                search_trace.append({
                    "step": "analyze",
                    "requires_search": state.query_analysis.requires_search,
                    "query_type": state.query_analysis.query_type,
                    "complexity": state.query_analysis.estimated_complexity,
                    "reasoning": state.query_analysis.search_reasoning,
                    "confidence": state.query_analysis.confidence
                })

                # If no search needed, synthesize and return
                if not state.query_analysis.requires_search:
                    await emitter.emit(events.synthesizing(request_id, 0))

                    synthesis = await self.synthesizer.synthesize(
                        request.query, [], None, request.context
                    )

                    execution_time_ms = int((time.time() - start_time) * 1000)
                    return SearchResponse(
                        success=True,
                        data=SearchResultData(
                            synthesized_context=synthesis,
                            sources=[],
                            search_queries=[],
                            confidence_score=state.query_analysis.confidence,
                            confidence_level=ConfidenceLevel.MEDIUM,
                            verification_status="skipped",
                            search_trace=search_trace
                        ),
                        meta=SearchMeta(
                            request_id=request_id,
                            iterations=0,
                            queries_executed=0,
                            sources_consulted=0,
                            execution_time_ms=execution_time_ms,
                            cache_hit=False
                        )
                    )

            # STEP 2: PLAN
            await emitter.emit(events.planning_search(request_id))

            if state.query_analysis:
                state.search_plan = await self.analyzer.create_search_plan(
                    request.query, state.query_analysis, request.context
                )
            else:
                action = await self.planner.plan(request.query, request.context)
                state.search_plan = SearchPlan(
                    original_query=request.query,
                    decomposed_questions=[request.query],
                    search_phases=[{"phase": "initial", "queries": action.queries}],
                    priority_order=list(range(len(action.queries))),
                    fallback_strategies=["broaden search terms"],
                    estimated_iterations=request.max_iterations,
                    reasoning=action.reasoning
                )

            # Initialize pending queries
            initial_queries = []
            for phase in state.search_plan.search_phases:
                initial_queries.extend(phase.get("queries", []))
            state.add_pending_queries(initial_queries)

            # Set scratchpad mission with decomposed questions
            decomposed_qs = state.search_plan.decomposed_questions
            completion_criteria = {
                f"q{i+1}": f"Find comprehensive information about: {q}"
                for i, q in enumerate(decomposed_qs)
            }
            scratchpad.set_mission(decomposed_qs, completion_criteria)

            await emitter.emit(events.search_planned(
                request_id,
                initial_queries,
                len(state.search_plan.search_phases)
            ))

            search_trace.append({
                "step": "plan",
                "decomposed_questions": state.search_plan.decomposed_questions,
                "phases": len(state.search_plan.search_phases),
                "initial_queries": len(initial_queries),
                "scratchpad_mission_set": True,
                "estimated_iterations": state.search_plan.estimated_iterations,
                "reasoning": state.search_plan.reasoning
            })

            # STEP 3: ADAPTIVE ReAct LOOP
            for iteration in range(request.max_iterations):
                state.iteration = iteration + 1

                # Emit iteration start
                await emitter.emit(SearchEvent(
                    event_type=EventType.ITERATION_START,
                    request_id=request_id,
                    message=f"Iteration {state.iteration}/{request.max_iterations}",
                    iteration=state.iteration,
                    max_iterations=request.max_iterations,
                    sources_count=state.sources_consulted
                ))

                # ACT: Execute pending searches
                if state.pending_queries:
                    queries_to_execute = state.pending_queries[:3]

                    await emitter.emit(events.searching(
                        request_id,
                        queries_to_execute,
                        state.iteration,
                        request.max_iterations
                    ))

                    # Record queries in scratchpad
                    for q in queries_to_execute:
                        scratchpad.record_search(q)

                    # TTL pin cache during web search
                    async with ToolCallContext(request_id, ToolType.WEB_SEARCH, manager=self.ttl_manager):
                        results = await self.searcher.search(queries_to_execute)
                    state.add_results(results)

                    # Record preliminary findings in scratchpad
                    for result in results:
                        question_id = self._match_result_to_question(result, scratchpad)
                        if question_id:
                            scratchpad.add_finding(
                                question_id=question_id,
                                content=result.snippet[:500],
                                source_url=result.url,
                                source_title=result.title,
                                finding_type=FindingType.FACT,
                                confidence=result.relevance_score if hasattr(result, 'relevance_score') else 0.5
                            )

                    for q in queries_to_execute:
                        state.mark_query_executed(q)

                    await emitter.emit(events.search_results(
                        request_id,
                        len(results),
                        len(state.unique_domains)
                    ))

                    # Log scratchpad progress
                    sp_status = scratchpad.get_completion_status()

                    search_trace.append({
                        "step": "search",
                        "iteration": iteration + 1,
                        "queries": queries_to_execute,
                        "results_count": len(results),
                        "total_sources": state.sources_consulted,
                        "unique_domains": len(state.unique_domains),
                        "scratchpad_progress": sp_status['overall']
                    })

                # VERIFY
                if (request.verification_level != VerificationLevel.NONE and
                    state.raw_results and not state.verified_claims):

                    await emitter.emit(events.verifying_claims(
                        request_id,
                        len(state.raw_results)
                    ))

                    combined_text = " ".join(r.snippet for r in state.raw_results[:10])
                    state.claims = await self.verifier.extract_claims(combined_text)

                    if state.claims:
                        state.verified_claims = await self.verifier.verify(
                            state.claims, state.raw_results, request.verification_level
                        )
                        verified_count = sum(1 for v in state.verified_claims if v.verified)

                        await emitter.emit(events.claims_verified(
                            request_id, verified_count, len(state.verified_claims)
                        ))

                        search_trace.append({
                            "step": "verify",
                            "iteration": iteration + 1,
                            "claims_verified": len(state.verified_claims),
                            "verified_count": verified_count
                        })

                # Check termination conditions
                if request.search_mode == SearchMode.FIXED:
                    if iteration >= request.max_iterations - 1:
                        break

                elif request.search_mode == SearchMode.ADAPTIVE:
                    if not state.can_continue_search(request.max_sources):
                        break

                    if state.has_sufficient_sources(request.min_sources):
                        should_continue, reason, new_queries = await self.analyzer.should_continue_search(
                            request.query,
                            [{"title": r.title, "snippet": r.snippet} for r in state.raw_results],
                            state.search_plan,
                            iteration
                        )

                        if not should_continue:
                            state.information_sufficient = True
                            break

                        if new_queries:
                            state.add_pending_queries(new_queries)
                            state.refinement_attempts += 1

                elif request.search_mode == SearchMode.EXHAUSTIVE:
                    if not state.pending_queries and not state.can_continue_search(request.max_sources):
                        state.leads_exhausted = True
                        break

                if not state.pending_queries:
                    state.leads_exhausted = True
                    break

            # STEP 4: INTELLIGENT URL EVALUATION, SCRAPING, AND COVERAGE CHECK
            # This step may loop back to search if content coverage is insufficient
            # OPTIMIZATION: Reduced from 2 to 1 refinement (saves ~30s per search)
            max_scrape_refinements = 1
            scrape_refinement = 0
            scraped_content = []

            while scrape_refinement <= max_scrape_refinements:
                if state.raw_results and len(state.raw_results) > 0:
                    # Convert WebSearchResult objects to dicts for evaluation
                    results_for_eval = [
                        {
                            "url": r.url,
                            "title": r.title,
                            "snippet": r.snippet,
                            "source_domain": r.source_domain
                        }
                        for r in state.raw_results
                    ]

                    # Use LLM to evaluate which URLs are worth scraping
                    await emitter.emit(events.evaluating_urls(request_id, len(results_for_eval)))
                    logger.info(f"[{request_id}] Evaluating {len(results_for_eval)} URLs for relevance...")

                    relevant_urls = await self.analyzer.evaluate_urls_for_scraping(
                        request.query,
                        results_for_eval,
                        max_urls=8
                    )

                    await emitter.emit(events.urls_evaluated(
                        request_id, len(relevant_urls), len(results_for_eval)
                    ))
                    logger.info(f"[{request_id}] {len(relevant_urls)} URLs deemed relevant for scraping")

                    # Log which URLs were selected and why
                    for ru in relevant_urls:
                        logger.info(f"[{request_id}]   - [{ru.get('relevance', 'medium')}] {ru.get('url', '')[:60]}...")
                        if ru.get('reason'):
                            logger.debug(f"[{request_id}]     Reason: {ru.get('reason')}")

                    # Scrape all relevant URLs
                    if relevant_urls:
                        urls_to_scrape = [
                            (ru["url"], ru["title"])
                            for ru in relevant_urls
                            if not ru["url"].lower().endswith('.pdf')
                        ]

                        logger.info(f"[{request_id}] Scraping {len(urls_to_scrape)} relevant sources for full content...")

                        for idx, (url, title) in enumerate(urls_to_scrape, 1):
                            # Skip URLs we've already scraped (check both local list and scratchpad)
                            if any(s.get("url") == url for s in scraped_content) or scratchpad.has_scraped(url):
                                continue
                            try:
                                await emitter.emit(events.scraping_url(request_id, url, idx, len(urls_to_scrape)))

                                # TTL pin cache during web scrape
                                async with ToolCallContext(request_id, ToolType.WEB_SCRAPE, manager=self.ttl_manager):
                                    scraped = await self.scraper.scrape_url(url)
                                if scraped.get("success") and scraped.get("content"):
                                    content_len = len(scraped["content"])
                                    scraped_content.append({
                                        "url": url,
                                        "title": title,
                                        "content": scraped["content"][:12000],
                                        "content_type": scraped.get("content_type", "html")
                                    })
                                    # Record in scratchpad
                                    scratchpad.record_scrape(url)
                                    await emitter.emit(events.url_scraped(
                                        request_id, url, content_len, scraped.get("content_type", "html")
                                    ))
                                    logger.info(f"[{request_id}] Scraped {content_len} chars from {url[:50]}")
                            except Exception as e:
                                logger.warning(f"[{request_id}] Failed to scrape {url}: {e}")

                        logger.info(f"[{request_id}] Successfully scraped {len(scraped_content)} sources")

                # STEP 4b: EVALUATE CONTENT COVERAGE
                if (scraped_content and
                    state.search_plan and
                    state.search_plan.decomposed_questions and
                    scrape_refinement < max_scrape_refinements):

                    logger.info(f"[{request_id}] Evaluating content coverage (refinement {scrape_refinement})...")

                    # Emit progress event for coverage check
                    await emitter.emit(events.progress_update(
                        request_id, 75 + scrape_refinement * 5,
                        f"Evaluating content coverage... (round {scrape_refinement + 1})"
                    ))

                    # TTL pin cache during LLM analysis
                    async with ToolCallContext(request_id, ToolType.OLLAMA_GENERATE, manager=self.ttl_manager):
                        coverage = await self.analyzer.evaluate_content_coverage(
                            query=request.query,
                            decomposed_questions=state.search_plan.decomposed_questions,
                            scraped_content=scraped_content
                        )

                    search_trace.append({
                        "step": "coverage_check",
                        "refinement": scrape_refinement,
                        "coverage_score": coverage.get("coverage_score", 0),
                        "is_sufficient": coverage.get("is_sufficient", True),
                        "gaps": coverage.get("information_gaps", []),
                        "reasoning": coverage.get("reasoning", "")
                    })

                    if coverage.get("is_sufficient", True):
                        logger.info(f"[{request_id}] Content coverage sufficient ({coverage.get('coverage_score', 0):.0%})")
                        break

                    # Content not sufficient - perform additional searches
                    new_queries = coverage.get("suggested_queries", [])
                    if new_queries and state.iteration < request.max_iterations:
                        logger.info(f"[{request_id}] Content coverage insufficient ({coverage.get('coverage_score', 0):.0%})")
                        logger.info(f"[{request_id}] Gaps: {coverage.get('information_gaps', [])}")
                        logger.info(f"[{request_id}] Adding {len(new_queries)} refined queries: {new_queries}")

                        await emitter.emit(events.progress_update(
                            request_id, 78 + scrape_refinement * 5,
                            f"Searching for missing information: {', '.join(new_queries[:2])}..."
                        ))

                        # Add new queries and perform additional search iteration
                        state.add_pending_queries(new_queries)
                        state.refinement_attempts += 1

                        # Execute the new searches
                        for iteration in range(min(2, request.max_iterations - state.iteration)):
                            if not state.pending_queries:
                                break

                            state.iteration += 1
                            queries_to_execute = state.pending_queries[:3]

                            await emitter.emit(events.searching(
                                request_id,
                                queries_to_execute,
                                state.iteration,
                                request.max_iterations
                            ))

                            # TTL pin cache during web search
                            async with ToolCallContext(request_id, ToolType.WEB_SEARCH, manager=self.ttl_manager):
                                results = await self.searcher.search(queries_to_execute)
                            state.add_results(results)

                            for q in queries_to_execute:
                                state.mark_query_executed(q)

                            await emitter.emit(events.search_results(
                                request_id,
                                len(results),
                                len(state.unique_domains)
                            ))

                            search_trace.append({
                                "step": "coverage_refinement_search",
                                "refinement": scrape_refinement,
                                "queries": queries_to_execute,
                                "results_count": len(results),
                                "total_sources": state.sources_consulted
                            })

                        scrape_refinement += 1
                        continue  # Loop back to scrape new URLs

                # No more refinements needed or possible
                break

            # STEP 5: SYNTHESIZE with scraped content
            await emitter.emit(events.synthesizing(request_id, state.sources_consulted))

            # Include scratchpad context for the synthesizer
            scratchpad_context = scratchpad.to_context_for_agent("synthesizer")
            synthesis_context = request.context.copy() if request.context else {}
            synthesis_context["scratchpad_summary"] = scratchpad_context

            # Determine if we should use a thinking model for synthesis
            thinking_model = None
            if state.query_analysis and state.query_analysis.requires_thinking_model:
                thinking_model = DEFAULT_THINKING_MODEL
                logger.info(f"[{request_id}] Using THINKING MODEL ({thinking_model}) for complex reasoning")

            # TTL pin cache during synthesis (prevents eviction during ~30-90s for thinking models)
            async with ToolCallContext(request_id, ToolType.OLLAMA_GENERATE, manager=self.ttl_manager):
                # Use scraped content if available, otherwise fall back to snippets
                synthesis = await self.synthesizer.synthesize_with_content(
                    request.query,
                    state.raw_results,
                    scraped_content,
                    state.verified_claims if request.verification_level != VerificationLevel.NONE else None,
                    synthesis_context,
                    model_override=thinking_model
                )

            # Mark scratchpad as complete
            scratchpad.mark_complete(f"Synthesized {len(state.raw_results)} sources from {len(state.unique_domains)} domains")

            # Calculate intermediate confidence for event
            interim_scraped_count = len(scraped_content) if scraped_content else 0
            interim_confidence = self.verifier.calculate_overall_confidence(
                state.verified_claims,
                source_count=len(state.raw_results),
                unique_domains=len(state.unique_domains),
                synthesis_length=len(synthesis),
                scraped_sources=interim_scraped_count
            )
            await emitter.emit(events.synthesis_complete(
                request_id,
                len(synthesis),
                interim_confidence
            ))

            # Get final scratchpad status
            final_scratchpad_status = scratchpad.get_completion_status()

            search_trace.append({
                "step": "synthesize",
                "iteration": state.iteration,
                "synthesis_length": len(synthesis),
                "total_sources": state.sources_consulted,
                "unique_domains": len(state.unique_domains),
                "scratchpad_status": {
                    "overall_completion": final_scratchpad_status["overall"],
                    "questions_answered": sum(1 for q in scratchpad.questions.values() if q.status == QuestionStatus.ANSWERED),
                    "findings_count": len(scratchpad.findings),
                    "contradictions_unresolved": final_scratchpad_status.get("unresolved_contradictions", 0)
                }
            })

            # Calculate final metrics with multiple signals
            scraped_count = len(scraped_content) if scraped_content else 0
            confidence_score = self.verifier.calculate_overall_confidence(
                state.verified_claims,
                source_count=len(state.raw_results),
                unique_domains=len(state.unique_domains),
                synthesis_length=len(synthesis),
                scraped_sources=scraped_count
            )
            confidence_level = self.synthesizer.determine_confidence_level(
                state.verified_claims, len(state.raw_results)
            )

            if request.verification_level == VerificationLevel.NONE:
                verification_status = "skipped"
            elif state.verified_claims:
                verified_ratio = sum(1 for v in state.verified_claims if v.verified) / len(state.verified_claims)
                verification_status = "verified" if verified_ratio >= 0.7 else "partial"
            else:
                # Even without explicit verification, consider it partial if we have good coverage
                if scraped_count >= 3 and len(state.unique_domains) >= 3:
                    verification_status = "partial"
                else:
                    verification_status = "unverified"

            execution_time_ms = int((time.time() - start_time) * 1000)

            response = SearchResponse(
                success=True,
                data=SearchResultData(
                    synthesized_context=synthesis,
                    sources=[
                        {"title": r.title, "url": r.url, "domain": r.source_domain}
                        for r in state.raw_results[:20]
                    ],
                    search_queries=state.executed_queries,
                    confidence_score=confidence_score,
                    confidence_level=confidence_level,
                    verification_status=verification_status,
                    search_trace=search_trace
                ),
                meta=SearchMeta(
                    request_id=request_id,
                    iterations=state.iteration,
                    queries_executed=len(state.executed_queries),
                    sources_consulted=state.sources_consulted,
                    execution_time_ms=execution_time_ms,
                    cache_hit=False
                )
            )

            # Cache results
            if request.cache_results:
                self._cache[cache_key] = response

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Search with events failed: {e}", exc_info=True)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return SearchResponse(
                success=False,
                data=SearchResultData(
                    synthesized_context="An error occurred during search.",
                    sources=[],
                    search_queries=[request.query],
                    confidence_score=0.0,
                    confidence_level=ConfidenceLevel.UNKNOWN,
                    verification_status="error"
                ),
                meta=SearchMeta(
                    request_id=request_id,
                    execution_time_ms=execution_time_ms
                ),
                errors=[{"type": "search_error", "message": str(e)}]
            )

    async def simple_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute a simple, non-agentic search.

        Returns raw search results without synthesis.
        """
        results = await self.searcher.search([query], max_results_per_query=max_results)

        return [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "source": r.source_domain
            }
            for r in results
        ]

    def _get_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key from request"""
        return f"{request.query}:{request.verification_level}:{request.search_mode}:{request.max_iterations}"

    def _match_result_to_question(self, result, scratchpad: AgenticScratchpad) -> Optional[str]:
        """
        Match a search result to a decomposed question in the scratchpad.
        Uses simple keyword overlap to determine relevance.

        Returns:
            question_id if matched, None otherwise
        """
        if not scratchpad.questions:
            return None

        result_text = f"{result.title} {result.snippet}".lower()
        best_match = None
        best_score = 0

        for q_id, q in scratchpad.questions.items():
            # Simple keyword matching
            question_words = set(q.question_text.lower().split())
            # Remove common words
            question_words -= {"what", "are", "the", "is", "a", "an", "how", "which", "for", "of", "to", "in", "and", "or"}

            if not question_words:
                continue

            # Count matching words
            matches = sum(1 for word in question_words if word in result_text)
            score = matches / len(question_words)

            if score > best_score:
                best_score = score
                best_match = q_id

        # Only match if score is above threshold
        if best_score >= 0.3:
            return best_match

        # Default to first question if no strong match
        return list(scratchpad.questions.keys())[0] if scratchpad.questions else None

    async def _store_in_memory(
        self,
        user_id: str,
        query: str,
        response: SearchResponse
    ):
        """Store search results in memOS memory service"""
        if not self.memory_service:
            return

        try:
            memory_content = {
                "type": "search_result",
                "query": query,
                "synthesis": response.data.synthesized_context[:500],
                "sources_count": len(response.data.sources),
                "confidence": response.data.confidence_score,
                "timestamp": response.meta.timestamp
            }

            # This would call the actual memory service
            # await self.memory_service.store(
            #     user_id=user_id,
            #     content=memory_content,
            #     memory_type="procedural",
            #     privacy_level="minimal"
            # )

            logger.debug(f"Stored search result in memory for user {user_id}")

        except Exception as e:
            logger.warning(f"Failed to store in memory: {e}")

    def clear_cache(self):
        """Clear the search cache"""
        self._cache.clear()
        logger.info("Search cache cleared")

    async def deep_search(
        self,
        query: str,
        max_sources_to_scrape: int = 5,
        include_pdfs: bool = True
    ) -> Dict[str, Any]:
        """
        Execute deep search with content scraping and LLM analysis.

        This goes beyond basic search by:
        1. Finding relevant sources via web search
        2. Scraping full content from top sources (including PDFs)
        3. Using a powerful reasoning model to analyze the content
        4. Extracting specific answers to the question

        Args:
            query: The user's question
            max_sources_to_scrape: Number of sources to scrape (default 5)
            include_pdfs: Whether to download and analyze PDFs

        Returns:
            Dict with answer, key_findings, sources_used, confidence
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]

        logger.info(f"[{request_id}] Starting deep search: {query[:50]}...")

        try:
            # Step 1: Select the best available reasoning model
            best_model = await self.deep_reader.select_best_model()
            self.deep_reader.model = best_model
            logger.info(f"[{request_id}] Using reasoning model: {best_model}")

            # Step 2: Perform initial search to find sources
            logger.info(f"[{request_id}] Searching for relevant sources...")

            # Use analyzer to create good search queries
            analysis = await self.analyzer.analyze(query, None)
            search_queries = analysis.suggested_queries if analysis.suggested_queries else [query]

            # Execute search
            search_results = await self.searcher.search(search_queries[:3], max_results_per_query=5)

            if not search_results:
                return {
                    "success": False,
                    "answer": "No relevant sources found.",
                    "confidence": 0.0,
                    "sources_used": [],
                    "key_findings": [],
                    "limitations": "Search returned no results.",
                    "execution_time_ms": int((time.time() - start_time) * 1000)
                }

            logger.info(f"[{request_id}] Found {len(search_results)} sources")

            # Step 3: Filter URLs for scraping
            urls_to_scrape = []
            for result in search_results[:max_sources_to_scrape * 2]:  # Get more to filter
                url = result.url
                is_pdf = url.lower().endswith('.pdf') or 'pdf' in url.lower()

                if is_pdf and not include_pdfs:
                    continue

                # Prioritize certain domains
                priority_domains = [
                    'fanuc', 'robot-forum.com', 'robodk.com',
                    '.gov', '.edu', 'manual', 'reference', 'guide',
                    'documentation', 'docs.'
                ]
                is_priority = any(pd in url.lower() for pd in priority_domains)

                urls_to_scrape.append((url, result.title, is_priority, is_pdf))

            # Sort by priority (PDFs and priority domains first)
            urls_to_scrape.sort(key=lambda x: (not x[2], not x[3]), reverse=False)
            urls_to_scrape = urls_to_scrape[:max_sources_to_scrape]

            logger.info(f"[{request_id}] Scraping {len(urls_to_scrape)} sources:")
            for url, title, priority, is_pdf in urls_to_scrape:
                logger.info(f"  {'[PDF]' if is_pdf else '[WEB]'} {'[PRIORITY]' if priority else ''} {title[:50]}")

            # Step 4: Scrape content
            urls_only = [u[0] for u in urls_to_scrape]
            scraped_content = await self.scraper.scrape_urls(urls_only, max_concurrent=3)

            successful_scrapes = [s for s in scraped_content if s.get("success")]
            logger.info(f"[{request_id}] Successfully scraped {len(successful_scrapes)}/{len(urls_only)} sources")

            for scrape in scraped_content:
                status = "✓" if scrape.get("success") else "✗"
                content_len = len(scrape.get("content", ""))
                logger.info(f"  {status} [{scrape.get('content_type', '?')}] {content_len:,} chars - {scrape.get('url', '')[:60]}")

            if not successful_scrapes:
                # Fallback to snippets only
                logger.warning(f"[{request_id}] No content scraped, using snippets")
                successful_scrapes = [
                    {
                        "url": r.url,
                        "title": r.title,
                        "content": f"{r.title}\n{r.snippet}",
                        "content_type": "snippet",
                        "success": True
                    }
                    for r in search_results[:max_sources_to_scrape]
                ]

            # Step 4b: Analyze images from PDFs using vision model
            all_images = []
            for scrape in successful_scrapes:
                if scrape.get("images"):
                    for img in scrape.get("images", []):
                        img["source_url"] = scrape.get("url", "")
                        all_images.append(img)

            if all_images:
                logger.info(f"[{request_id}] Analyzing {len(all_images)} images from PDFs with vision model...")
                try:
                    image_analyses = await self.vision_analyzer.analyze_images_batch(
                        images=all_images,
                        question=query
                    )

                    # Append image descriptions to the relevant scraped content
                    image_context = []
                    for analysis in image_analyses:
                        if analysis.get("success") and analysis.get("description"):
                            page_info = f"(page {analysis.get('page')})" if analysis.get("page") else ""
                            image_context.append(
                                f"\n[IMAGE ANALYSIS {page_info}]\n{analysis['description']}"
                            )
                            logger.info(f"  Analyzed image: {len(analysis['description'])} chars")

                    if image_context:
                        # Add image context to the first PDF scrape
                        for scrape in successful_scrapes:
                            if scrape.get("content_type") == "pdf":
                                scrape["content"] += "\n\n--- CHART/IMAGE ANALYSIS ---" + "".join(image_context)
                                logger.info(f"[{request_id}] Added {len(image_context)} image analyses to content")
                                break

                except Exception as e:
                    logger.warning(f"[{request_id}] Image analysis failed: {e}")

            # Step 5: Deep analysis with reasoning model
            logger.info(f"[{request_id}] Analyzing content with {best_model}...")

            analysis_result = await self.deep_reader.analyze_content(
                question=query,
                scraped_content=successful_scrapes,
                max_context_chars=40000  # More context for deep analysis
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            logger.info(f"[{request_id}] Deep search complete in {execution_time_ms}ms")
            logger.info(f"  Confidence: {analysis_result.get('confidence', 0):.0%}")
            logger.info(f"  Key findings: {len(analysis_result.get('key_findings', []))}")

            return {
                "success": True,
                "answer": analysis_result.get("answer", ""),
                "confidence": analysis_result.get("confidence", 0.0),
                "sources_used": analysis_result.get("sources_used", []),
                "key_findings": analysis_result.get("key_findings", []),
                "limitations": analysis_result.get("limitations", ""),
                "model_used": best_model,
                "sources_scraped": len(successful_scrapes),
                "execution_time_ms": execution_time_ms,
                "search_results_count": len(search_results)
            }

        except Exception as e:
            logger.error(f"[{request_id}] Deep search failed: {e}", exc_info=True)
            return {
                "success": False,
                "answer": f"Deep search failed: {str(e)}",
                "confidence": 0.0,
                "sources_used": [],
                "key_findings": [],
                "limitations": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
