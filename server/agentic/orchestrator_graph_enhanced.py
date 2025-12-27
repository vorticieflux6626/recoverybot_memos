"""
Graph-Enhanced Agentic Search Orchestrator

This module extends the base orchestrator with graph-based KV cache optimization:
- Agent Step Graph for workflow-aware cache management (KVFlow-inspired)
- Scratchpad Cache for intermediate result caching (ROG-inspired)
- Proactive prefetching based on transition probabilities
- Prefix-optimized prompts for maximum cache reuse

The integration is designed to minimize latency while maintaining compatibility
with the existing orchestrator interface.
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
from .ttl_cache_manager import get_ttl_cache_manager, ToolType, ToolCallContext

# Graph-based cache imports
from .agent_step_graph import AgentType, get_agent_step_graph
from .scratchpad_cache import get_scratchpad_cache, CachedSubQuery
from .graph_cache_integration import (
    GraphCacheIntegration,
    get_graph_cache_integration,
    map_orchestrator_phase_to_agent_type
)
from .prefix_optimized_prompts import (
    build_scratchpad_context,
    estimate_prefix_reuse,
    get_prompt_registry
)

from .events import EventEmitter, EventType, SearchEvent

logger = logging.getLogger("agentic.orchestrator_graph_enhanced")


class GraphEnhancedOrchestrator:
    """
    Graph-enhanced agentic search orchestrator.

    Extends base orchestrator with:
    1. Agent Step Graph for workflow-aware eviction
    2. Scratchpad Cache for intermediate caching
    3. Proactive prefetching for likely next agents
    4. Mission decomposition reuse
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        mcp_url: str = "http://localhost:7777",
        brave_api_key: Optional[str] = None,
        memory_service: Optional[Any] = None
    ):
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url

        # Initialize standard agents
        self.analyzer = QueryAnalyzer(ollama_url=ollama_url)
        self.planner = PlannerAgent(ollama_url=ollama_url, mcp_url=mcp_url)
        self.searcher = SearcherAgent(brave_api_key=brave_api_key)
        self.verifier = VerifierAgent(ollama_url=ollama_url)
        self.synthesizer = SynthesizerAgent(ollama_url=ollama_url, mcp_url=mcp_url)

        # Content scraper
        self.scraper = ContentScraper()
        self.deep_reader = DeepReader(ollama_url=ollama_url)
        self.vision_analyzer = VisionAnalyzer(ollama_url=ollama_url)

        # Memory service
        self.memory_service = memory_service

        # Standard managers
        self.scratchpad_manager = ScratchpadManager(memory_service=memory_service)
        self.ttl_manager = get_ttl_cache_manager()

        # GRAPH-BASED CACHE INTEGRATION
        self.graph_cache = get_graph_cache_integration(ollama_url)
        self.agent_graph = get_agent_step_graph()
        self.scratchpad_cache = get_scratchpad_cache()
        self.prompt_registry = get_prompt_registry()

        # Search cache
        self._cache: Dict[str, SearchResponse] = {}
        self._cache_ttl = 3600

        # Statistics for graph cache performance
        self.graph_stats = {
            'total_searches': 0,
            'graph_cache_hits': 0,
            'mission_reuse_count': 0,
            'subquery_reuse_count': 0,
            'prefetch_success_count': 0
        }

    async def initialize(self):
        """Initialize orchestrator and graph cache system"""
        mcp_available = await self.planner.check_mcp_available()
        await self.synthesizer.check_mcp_available()

        # Initialize high-priority prompt templates for warming
        templates = self.prompt_registry.get_high_priority_templates(limit=5)
        logger.info(f"Graph-enhanced orchestrator initialized. "
                   f"MCP: {mcp_available}, Priority templates: {len(templates)}")

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute graph-enhanced agentic search.

        This method integrates the Agent Step Graph and Scratchpad Cache
        to optimize KV cache usage throughout the search workflow.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        self.graph_stats['total_searches'] += 1

        logger.info(f"[{request_id}] Starting GRAPH-ENHANCED search: {request.query[:50]}...")

        # Check caches first
        cache_key = self._get_cache_key(request)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.meta.cache_hit = True
            logger.info(f"[{request_id}] In-memory cache hit")
            return cached

        # Check semantic cache
        semantic_result = await self._check_semantic_cache(request, request_id)
        if semantic_result:
            return semantic_result

        try:
            # START GRAPH WORKFLOW
            workflow_context = await self.graph_cache.start_workflow(request_id, request.query)
            logger.info(f"[{request_id}] Graph workflow started")

            # Initialize search state
            state = SearchState(
                query=request.query,
                max_iterations=request.max_iterations,
                search_mode=request.search_mode.value
            )

            # Create scratchpad
            scratchpad = self.scratchpad_manager.create(
                query=request.query,
                request_id=request_id,
                user_id=request.user_id
            )

            search_trace = []

            # Build scratchpad state for graph cache
            scratchpad_state = {
                'mission': request.query,
                'sub_questions': [],
                'findings': [],
                'search_history': []
            }

            # STEP 1: ANALYZE with graph tracking
            if request.analyze_query:
                analysis_result = await self._execute_analyze_phase(
                    request, state, scratchpad_state, search_trace, request_id
                )
                if analysis_result:
                    # Early return for no-search-needed cases
                    await self.graph_cache.end_workflow(request_id, success=True)
                    return analysis_result

            # STEP 2: PLAN with mission decomposition caching
            await self._execute_plan_phase(
                request, state, scratchpad, scratchpad_state, search_trace, request_id
            )

            # STEP 3: ADAPTIVE ReAct LOOP with graph tracking
            await self._execute_search_loop(
                request, state, scratchpad, scratchpad_state, search_trace, request_id
            )

            # STEP 4: URL EVALUATION & SCRAPING
            scraped_content = await self._execute_scrape_phase(
                request, state, scratchpad, scratchpad_state, search_trace, request_id
            )

            # STEP 5: SYNTHESIZE with graph tracking
            response = await self._execute_synthesis_phase(
                request, state, scratchpad, scraped_content,
                scratchpad_state, search_trace, request_id, start_time
            )

            # END GRAPH WORKFLOW
            workflow_stats = await self.graph_cache.end_workflow(request_id, success=True)
            logger.info(f"[{request_id}] Workflow completed: {workflow_stats}")

            # Cache results
            await self._cache_results(request, response, request_id, cache_key)

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Graph-enhanced search failed: {e}")
            await self.graph_cache.end_workflow(request_id, success=False)
            raise

    async def _execute_analyze_phase(
        self, request: SearchRequest, state: SearchState,
        scratchpad_state: Dict, search_trace: List, request_id: str
    ) -> Optional[SearchResponse]:
        """Execute ANALYZE phase with graph cache integration"""

        # Notify graph cache of agent transition
        cached_data = await self.graph_cache.before_agent_call(
            request_id, AgentType.ANALYZER, scratchpad_state
        )

        logger.info(f"[{request_id}] ANALYZE phase (graph-tracked)")
        start = time.time()

        state.query_analysis = await self.analyzer.analyze(
            request.query, request.context
        )

        duration_ms = (time.time() - start) * 1000

        # Record in graph
        await self.graph_cache.after_agent_call(
            request_id, AgentType.ANALYZER,
            {'query_type': state.query_analysis.query_type},
            duration_ms
        )

        search_trace.append({
            "step": "analyze",
            "requires_search": state.query_analysis.requires_search,
            "query_type": state.query_analysis.query_type,
            "complexity": state.query_analysis.estimated_complexity,
            "reasoning": state.query_analysis.search_reasoning,
            "confidence": state.query_analysis.confidence,
            "graph_tracked": True
        })

        # Early return if no search needed
        if not state.query_analysis.requires_search:
            logger.info(f"[{request_id}] Analysis indicates no search needed")

            # Track synthesizer call
            await self.graph_cache.before_agent_call(
                request_id, AgentType.SYNTHESIZER, scratchpad_state
            )

            synthesis = await self.synthesizer.synthesize(
                request.query, [], None, request.context
            )

            await self.graph_cache.after_agent_call(
                request_id, AgentType.SYNTHESIZER,
                {'synthesis_length': len(synthesis)},
                (time.time() - start) * 1000
            )

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
                    execution_time_ms=int(duration_ms),
                    cache_hit=False
                )
            )

        return None

    async def _execute_plan_phase(
        self, request: SearchRequest, state: SearchState,
        scratchpad: AgenticScratchpad, scratchpad_state: Dict,
        search_trace: List, request_id: str
    ):
        """Execute PLAN phase with mission decomposition caching"""

        # Check for cached mission decomposition (ROG-style)
        cached_mission = self.scratchpad_cache.get_mission_decomposition(request.query)
        if cached_mission and cached_mission.success_rate > 0.6:
            logger.info(f"[{request_id}] Using CACHED mission decomposition "
                       f"(success_rate={cached_mission.success_rate:.2f})")
            self.graph_stats['mission_reuse_count'] += 1

            # Build plan from cached decomposition
            state.search_plan = SearchPlan(
                original_query=request.query,
                decomposed_questions=[q.get('question', q) if isinstance(q, dict) else q
                                     for q in cached_mission.sub_questions],
                search_phases=[{"phase": "cached", "queries": [request.query]}],
                priority_order=list(range(len(cached_mission.sub_questions))),
                fallback_strategies=["broaden search terms"],
                estimated_iterations=request.max_iterations,
                reasoning="Using cached decomposition pattern"
            )

            search_trace.append({
                "step": "plan",
                "cache_hit": True,
                "cached_success_rate": cached_mission.success_rate,
                "decomposed_questions": state.search_plan.decomposed_questions
            })
        else:
            # Track planner call in graph
            await self.graph_cache.before_agent_call(
                request_id, AgentType.PLANNER, scratchpad_state
            )

            start = time.time()

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

            duration_ms = (time.time() - start) * 1000

            await self.graph_cache.after_agent_call(
                request_id, AgentType.PLANNER,
                {'decomposed_questions': state.search_plan.decomposed_questions},
                duration_ms
            )

            # Cache the mission decomposition for future queries
            self.scratchpad_cache.cache_mission_decomposition(
                mission=request.query,
                sub_questions=[
                    {'question': q, 'criteria': f'Find info about {q}', 'priority': i+1}
                    for i, q in enumerate(state.search_plan.decomposed_questions)
                ],
                success_rate=0.5  # Initial rate
            )

            search_trace.append({
                "step": "plan",
                "cache_hit": False,
                "decomposed_questions": state.search_plan.decomposed_questions,
                "phases": len(state.search_plan.search_phases),
                "graph_tracked": True
            })

        # Update scratchpad state for graph cache
        scratchpad_state['sub_questions'] = state.search_plan.decomposed_questions

        # Initialize pending queries
        initial_queries = []
        for phase in state.search_plan.search_phases:
            initial_queries.extend(phase.get("queries", []))
        state.add_pending_queries(initial_queries)

        # Set scratchpad mission
        completion_criteria = {
            f"q{i+1}": f"Find comprehensive information about: {q}"
            for i, q in enumerate(state.search_plan.decomposed_questions)
        }
        scratchpad.set_mission(state.search_plan.decomposed_questions, completion_criteria)

    async def _execute_search_loop(
        self, request: SearchRequest, state: SearchState,
        scratchpad: AgenticScratchpad, scratchpad_state: Dict,
        search_trace: List, request_id: str
    ):
        """Execute adaptive ReAct search loop with graph tracking"""

        for iteration in range(request.max_iterations):
            state.iteration = iteration + 1

            logger.info(f"[{request_id}] Iteration {state.iteration}/{state.max_iterations}")

            if not state.pending_queries:
                state.leads_exhausted = True
                break

            # Track searcher in graph
            await self.graph_cache.before_agent_call(
                request_id, AgentType.SEARCHER, scratchpad_state
            )

            start = time.time()
            queries_to_execute = state.pending_queries[:3]

            # Check for cached sub-query results (ROG-style)
            for query in queries_to_execute:
                cached_sq = self.scratchpad_cache.get_subquery(query)
                if cached_sq:
                    logger.info(f"[{request_id}] Sub-query cache hit: {query[:30]}...")
                    self.graph_stats['subquery_reuse_count'] += 1
                    # Add cached result to findings
                    scratchpad_state['findings'].append({
                        'source': 'cached',
                        'summary': cached_sq.answer[:200]
                    })

            # Record searches in scratchpad
            for q in queries_to_execute:
                scratchpad.record_search(q)
                scratchpad_state['search_history'].append(q)

            # Execute search with TTL pinning
            async with ToolCallContext(request_id, ToolType.WEB_SEARCH, manager=self.ttl_manager):
                results = await self.searcher.search(queries_to_execute)
            state.add_results(results)

            duration_ms = (time.time() - start) * 1000

            await self.graph_cache.after_agent_call(
                request_id, AgentType.SEARCHER,
                {'results_count': len(results), 'queries': queries_to_execute},
                duration_ms
            )

            # Record findings in scratchpad (both local and graph cache)
            for result in results:
                question_id = self._match_result_to_question(result, scratchpad)
                if question_id:
                    scratchpad.add_finding(
                        question_id=question_id,
                        content=result.snippet[:500],
                        source_url=result.url,
                        source_title=result.title,
                        finding_type=FindingType.FACT,
                        confidence=getattr(result, 'relevance_score', 0.5)
                    )
                    scratchpad_state['findings'].append({
                        'source': result.url,
                        'summary': result.snippet[:200]
                    })

                # Cache finding for deduplication
                self.scratchpad_cache.cache_finding(
                    content=result.snippet,
                    source_url=result.url,
                    confidence=getattr(result, 'relevance_score', 0.5)
                )

            # Mark queries executed
            for q in queries_to_execute:
                state.mark_query_executed(q)

            search_trace.append({
                "step": "search",
                "iteration": iteration + 1,
                "queries": queries_to_execute,
                "results_count": len(results),
                "graph_tracked": True
            })

            # Verification phase (if enabled)
            if (request.verification_level != VerificationLevel.NONE and
                state.raw_results and not state.verified_claims):

                await self.graph_cache.before_agent_call(
                    request_id, AgentType.VERIFIER, scratchpad_state
                )

                start = time.time()
                combined_text = " ".join(r.snippet for r in state.raw_results[:10])
                state.claims = await self.verifier.extract_claims(combined_text)

                if state.claims:
                    state.verified_claims = await self.verifier.verify(
                        state.claims, state.raw_results, request.verification_level
                    )

                    # Cache verified claims as sub-query results
                    for claim in state.verified_claims:
                        if claim.verified:
                            self.scratchpad_cache.cache_subquery(
                                query=claim.claim,
                                answer=claim.claim,
                                sources=[s for s in claim.supporting_sources],
                                confidence=claim.confidence
                            )

                duration_ms = (time.time() - start) * 1000

                await self.graph_cache.after_agent_call(
                    request_id, AgentType.VERIFIER,
                    {'claims_verified': len(state.verified_claims) if state.verified_claims else 0},
                    duration_ms
                )

            # Check continuation conditions
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
                        state.search_plan, iteration
                    )
                    if not should_continue:
                        state.information_sufficient = True
                        break
                    if new_queries:
                        state.add_pending_queries(new_queries)
                        state.refinement_attempts += 1

    async def _execute_scrape_phase(
        self, request: SearchRequest, state: SearchState,
        scratchpad: AgenticScratchpad, scratchpad_state: Dict,
        search_trace: List, request_id: str
    ) -> List[Dict]:
        """Execute URL evaluation and scraping with graph tracking"""

        scraped_content = []
        max_scrape_refinements = 1
        scrape_refinement = 0

        while scrape_refinement <= max_scrape_refinements:
            if not state.raw_results:
                break

            # Track scraper in graph
            await self.graph_cache.before_agent_call(
                request_id, AgentType.SCRAPER, scratchpad_state
            )

            start = time.time()

            # Evaluate URLs for scraping
            results_for_eval = [
                {"url": r.url, "title": r.title, "snippet": r.snippet, "source_domain": r.source_domain}
                for r in state.raw_results
            ]

            relevant_urls = await self.analyzer.evaluate_urls_for_scraping(
                request.query, results_for_eval, max_urls=8
            )

            if relevant_urls:
                urls_to_scrape = [
                    (ru["url"], ru["title"]) for ru in relevant_urls
                    if not ru["url"].lower().endswith('.pdf')
                ]

                for url, title in urls_to_scrape:
                    # Check if already scraped (via graph cache)
                    existing = self.scratchpad_cache.get_findings_by_source(url)
                    if existing:
                        logger.debug(f"[{request_id}] Skipping already-cached URL: {url}")
                        continue

                    if scratchpad.has_scraped(url):
                        continue

                    try:
                        async with ToolCallContext(request_id, ToolType.WEB_SCRAPE, manager=self.ttl_manager):
                            scraped = await self.scraper.scrape_url(url)
                        if scraped.get("success") and scraped.get("content"):
                            scraped_content.append({
                                "url": url,
                                "title": title,
                                "content": scraped["content"][:15000]
                            })
                            scratchpad.record_scraped_url(url)

                            # Cache scraped content
                            self.scratchpad_cache.cache_finding(
                                content=scraped["content"][:2000],
                                source_url=url,
                                confidence=0.8
                            )
                    except Exception as e:
                        logger.debug(f"[{request_id}] Scrape failed for {url}: {e}")

            duration_ms = (time.time() - start) * 1000

            await self.graph_cache.after_agent_call(
                request_id, AgentType.SCRAPER,
                {'scraped_count': len(scraped_content)},
                duration_ms
            )

            search_trace.append({
                "step": "scrape",
                "refinement": scrape_refinement,
                "urls_evaluated": len(relevant_urls) if relevant_urls else 0,
                "scraped_count": len(scraped_content),
                "graph_tracked": True
            })

            scrape_refinement += 1
            if not relevant_urls:
                break

        return scraped_content

    async def _execute_synthesis_phase(
        self, request: SearchRequest, state: SearchState,
        scratchpad: AgenticScratchpad, scraped_content: List[Dict],
        scratchpad_state: Dict, search_trace: List,
        request_id: str, start_time: float
    ) -> SearchResponse:
        """Execute synthesis with graph tracking"""

        # Track synthesizer in graph
        await self.graph_cache.before_agent_call(
            request_id, AgentType.SYNTHESIZER, scratchpad_state
        )

        synth_start = time.time()

        # Build context with scratchpad
        scratchpad_context = scratchpad.to_context_for_agent("synthesizer")
        synthesis_context = request.context.copy() if request.context else {}
        synthesis_context["scratchpad_summary"] = scratchpad_context

        # Add graph cache context (prefix estimation)
        prefix_estimate = estimate_prefix_reuse('synthesizer', len(scratchpad_state.get('findings', [])))
        synthesis_context["graph_prefix_estimate"] = prefix_estimate

        # Determine thinking model
        thinking_model = None
        if state.query_analysis and state.query_analysis.requires_thinking_model:
            thinking_model = DEFAULT_THINKING_MODEL

        # Execute synthesis with TTL pinning
        async with ToolCallContext(request_id, ToolType.OLLAMA_GENERATE, manager=self.ttl_manager):
            if scraped_content:
                synthesis = await self.synthesizer.synthesize_with_content(
                    request.query, state.raw_results, scraped_content,
                    state.verified_claims if request.verification_level != VerificationLevel.NONE else None,
                    synthesis_context, model_override=thinking_model
                )
            else:
                synthesis = await self.synthesizer.synthesize(
                    request.query, state.raw_results,
                    state.verified_claims if request.verification_level != VerificationLevel.NONE else None,
                    synthesis_context
                )

        duration_ms = (time.time() - synth_start) * 1000

        await self.graph_cache.after_agent_call(
            request_id, AgentType.SYNTHESIZER,
            {'synthesis_length': len(synthesis), 'used_thinking_model': thinking_model is not None},
            duration_ms
        )

        # Mark scratchpad complete
        scratchpad.mark_complete(f"Synthesized {len(state.raw_results)} sources")
        final_status = scratchpad.get_completion_status()

        search_trace.append({
            "step": "synthesize",
            "synthesis_length": len(synthesis),
            "used_thinking_model": thinking_model is not None,
            "graph_tracked": True,
            "scratchpad_status": {
                "overall_completion": final_status["overall"],
                "questions_answered": sum(1 for q in scratchpad.questions.values()
                                         if q.status == QuestionStatus.ANSWERED),
                "findings_count": len(scratchpad.findings)
            }
        })

        # Calculate confidence
        scraped_count = len(scraped_content)
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

        # Determine verification status
        if request.verification_level == VerificationLevel.NONE:
            verification_status = "skipped"
        elif state.verified_claims:
            verified_ratio = sum(1 for v in state.verified_claims if v.verified) / len(state.verified_claims)
            verification_status = "verified" if verified_ratio >= 0.7 else "partial"
        else:
            verification_status = "partial" if scraped_count >= 3 else "unverified"

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Update mission success rate in cache
        success = confidence_score >= 0.5
        self.scratchpad_cache.update_mission_success_rate(request.query, success)

        return SearchResponse(
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

    async def _check_semantic_cache(self, request: SearchRequest, request_id: str) -> Optional[SearchResponse]:
        """Check semantic cache for similar queries"""
        try:
            from core.embedding_service import EmbeddingService
            content_cache = get_content_cache()
            embedding_service = EmbeddingService()

            query_embedding = await embedding_service.generate_embedding(request.query)
            similar_result = content_cache.find_similar_query(query_embedding, similarity_threshold=0.88)

            if similar_result:
                logger.info(f"[{request_id}] Semantic cache hit (similarity={similar_result.get('similarity_score', 0):.3f})")
                self.graph_stats['graph_cache_hits'] += 1

                return SearchResponse(
                    success=similar_result.get("success", True),
                    data=SearchResultData(
                        synthesized_context=similar_result.get("data", {}).get("synthesized_context", ""),
                        sources=similar_result.get("data", {}).get("sources", []),
                        search_queries=similar_result.get("data", {}).get("search_queries", []),
                        search_trace=similar_result.get("data", {}).get("search_trace", []),
                        confidence_score=similar_result.get("data", {}).get("confidence_score", 0.0),
                        confidence_level=ConfidenceLevel(similar_result.get("data", {}).get("confidence_level", "medium"))
                    ),
                    meta=SearchMeta(
                        request_id=request_id,
                        timestamp=datetime.now().isoformat(),
                        execution_time_ms=0,
                        cache_hit=True,
                        semantic_match=True,
                        matched_query=similar_result.get("matched_query", ""),
                        similarity_score=similar_result.get("similarity_score", 0)
                    )
                )
        except Exception as e:
            logger.debug(f"[{request_id}] Semantic cache check failed: {e}")

        return None

    async def _cache_results(self, request: SearchRequest, response: SearchResponse,
                              request_id: str, cache_key: str):
        """Cache successful results"""
        if request.cache_results:
            self._cache[cache_key] = response

            try:
                from core.embedding_service import EmbeddingService
                content_cache = get_content_cache()
                embedding_service = EmbeddingService()

                query_embedding = await embedding_service.generate_embedding(request.query)
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
                content_cache.set_query_result(request.query, cache_data, query_embedding)
            except Exception as e:
                logger.debug(f"[{request_id}] Semantic cache store failed: {e}")

    def _get_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        key_data = f"{request.query}:{request.search_mode}:{request.verification_level}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _match_result_to_question(self, result, scratchpad) -> Optional[str]:
        """Match search result to scratchpad question"""
        best_match = None
        best_score = 0.0

        result_words = set(result.snippet.lower().split())

        for q_id, question in scratchpad.questions.items():
            q_words = set(question.text.lower().split())
            overlap = len(result_words & q_words) / max(len(q_words), 1)
            if overlap > best_score and overlap > 0.1:
                best_score = overlap
                best_match = q_id

        return best_match

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph cache statistics"""
        return {
            'orchestrator_stats': self.graph_stats,
            'graph_cache_stats': self.graph_cache.get_comprehensive_stats(),
            'agent_graph_stats': self.agent_graph.get_graph_stats(),
            'scratchpad_cache_stats': self.scratchpad_cache.get_stats()
        }


# Factory function to create enhanced orchestrator
def create_graph_enhanced_orchestrator(
    ollama_url: str = "http://localhost:11434",
    mcp_url: str = "http://localhost:7777",
    brave_api_key: Optional[str] = None,
    memory_service: Optional[Any] = None
) -> GraphEnhancedOrchestrator:
    """Create a new graph-enhanced orchestrator instance"""
    return GraphEnhancedOrchestrator(
        ollama_url=ollama_url,
        mcp_url=mcp_url,
        brave_api_key=brave_api_key,
        memory_service=memory_service
    )
