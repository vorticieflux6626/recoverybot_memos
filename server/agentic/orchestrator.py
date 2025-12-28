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
from .query_classifier import (
    QueryClassifier, QueryClassification, RecommendedPipeline, QueryCategory
)
from .self_reflection import (
    SelfReflectionAgent, ReflectionResult, SupportLevel, get_self_reflection_agent
)
from .retrieval_evaluator import (
    RetrievalEvaluator, RetrievalEvaluation, RetrievalQuality, CorrectiveAction
)
from .experience_distiller import (
    ExperienceDistiller, get_experience_distiller
)
from .classifier_feedback import (
    ClassifierFeedback, get_classifier_feedback
)
from .context_limits import (
    get_synthesizer_limits,
    get_dynamic_source_allocation,
    get_analyzer_limits,
    SYNTHESIZER_LIMITS,
    THINKING_SYNTHESIZER_LIMITS,
)
from .sufficient_context import (
    SufficientContextClassifier,
    PositionalOptimizer,
    DynamicContextAllocator,
    SufficiencyResult,
    ContextSufficiency,
    get_sufficient_context_classifier,
    get_positional_optimizer,
    get_dynamic_allocator,
)
from . import events
from .events import (
    EventEmitter, EventType, SearchEvent,
    AgentGraphState, get_graph_state, reset_graph_state,
    graph_node_entered, graph_node_completed, graph_state_update
)

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

        # Initialize query classifier (DeepSeek-R1 based)
        self.classifier = QueryClassifier(ollama_url=ollama_url)

        # Initialize agents
        self.analyzer = QueryAnalyzer(ollama_url=ollama_url)
        self.planner = PlannerAgent(ollama_url=ollama_url, mcp_url=mcp_url)
        self.searcher = SearcherAgent(brave_api_key=brave_api_key)
        self.verifier = VerifierAgent(ollama_url=ollama_url)
        self.synthesizer = SynthesizerAgent(ollama_url=ollama_url, mcp_url=mcp_url)
        self.reflector = SelfReflectionAgent(ollama_url=ollama_url)
        self.retrieval_evaluator = RetrievalEvaluator(ollama_url=ollama_url)

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

        # Experience distiller for learning from successful searches
        self.experience_distiller = get_experience_distiller()

        # Classifier feedback for adaptive learning
        self.classifier_feedback = get_classifier_feedback()

        # Search cache (in-memory, cleared on restart)
        self._cache: Dict[str, SearchResponse] = {}
        self._cache_ttl = 3600  # 1 hour

    async def initialize(self):
        """Initialize orchestrator and check service availability"""
        # Check MCP availability for agents that use it
        mcp_available = await self.planner.check_mcp_available()
        await self.synthesizer.check_mcp_available()

        logger.info(f"Orchestrator initialized. MCP available: {mcp_available}")

    async def classify_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryClassification:
        """
        Classify a query to determine optimal processing pipeline.

        Uses DeepSeek-R1 14B with Chain-of-Draft prompting for efficient
        classification with reasoning capabilities.

        Args:
            query: User's query text
            context: Optional context dictionary

        Returns:
            QueryClassification with category, capabilities, and pipeline recommendation
        """
        return await self.classifier.classify(query, context)

    async def route_by_classification(
        self,
        request: SearchRequest,
        classification: QueryClassification
    ) -> SearchResponse:
        """
        Route a request based on its classification.

        Args:
            request: The search request
            classification: Result from classify_query

        Returns:
            SearchResponse from the appropriate pipeline
        """
        logger.info(
            f"Routing query: category={classification.category.value}, "
            f"pipeline={classification.recommended_pipeline.value}"
        )

        if classification.recommended_pipeline == RecommendedPipeline.DIRECT_ANSWER:
            # Skip search, use synthesizer directly
            return await self._direct_answer(request, classification)

        elif classification.recommended_pipeline == RecommendedPipeline.CODE_ASSISTANT:
            # Technical/code mode (same as agentic for now, but can be extended)
            return await self.search(request)

        elif classification.recommended_pipeline == RecommendedPipeline.WEB_SEARCH:
            # Standard web search
            return await self.search(request)

        else:  # AGENTIC_SEARCH
            # Full multi-agent pipeline
            return await self.search(request)

    async def _direct_answer(
        self,
        request: SearchRequest,
        classification: QueryClassification
    ) -> SearchResponse:
        """
        Generate a direct answer without web search.

        Used for queries that don't require external information.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.info(f"[{request_id}] Direct answer mode (no search needed)")

        try:
            # Use synthesizer to generate answer from model knowledge
            synthesis = await self.synthesizer.synthesize(
                request.query,
                [],  # No search results
                None,
                request.context
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return SearchResponse(
                success=True,
                data=SearchResultData(
                    synthesized_context=synthesis,
                    sources=[],
                    search_queries=[],
                    confidence_score=classification.use_thinking_model and 0.8 or 0.7,
                    confidence_level=ConfidenceLevel.MEDIUM,
                    verification_status="model_knowledge",
                    search_trace=[{
                        "step": "classify",
                        "category": classification.category.value,
                        "pipeline": classification.recommended_pipeline.value,
                        "reasoning": classification.reasoning
                    }, {
                        "step": "direct_answer",
                        "reason": "No web search needed per classification"
                    }]
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
        except Exception as e:
            logger.error(f"[{request_id}] Direct answer failed: {e}")
            # Fallback to regular search
            return await self.search(request)

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

                # CRAG: Pre-synthesis retrieval quality evaluation
                if state.raw_results and iteration == 0:
                    # Only evaluate on first iteration (before refinements)
                    try:
                        logger.info(f"[{request_id}] CRAG: Evaluating retrieval quality...")
                        retrieval_eval = await self.retrieval_evaluator.evaluate(
                            query=request.query,
                            search_results=[
                                {"title": r.title, "snippet": r.snippet, "url": r.url}
                                for r in state.raw_results[:10]
                            ],
                            decomposed_questions=list(scratchpad.questions.keys()) if scratchpad.questions else None
                        )

                        logger.info(
                            f"[{request_id}] CRAG result: quality={retrieval_eval.quality.value}, "
                            f"relevance={retrieval_eval.overall_relevance:.2f}, "
                            f"action={retrieval_eval.recommended_action.value}"
                        )

                        search_trace.append({
                            "step": "crag_evaluation",
                            "iteration": iteration + 1,
                            "quality": retrieval_eval.quality.value,
                            "relevance": retrieval_eval.overall_relevance,
                            "coverage": retrieval_eval.query_coverage,
                            "action": retrieval_eval.recommended_action.value
                        })

                        # Take corrective action if needed
                        if retrieval_eval.recommended_action == CorrectiveAction.REFINE_QUERY:
                            # Add refined queries to pending
                            for refined_q in retrieval_eval.refined_queries[:3]:
                                if refined_q not in state.executed_queries:
                                    state.pending_queries.append(refined_q)
                            logger.info(f"[{request_id}] CRAG: Added {len(retrieval_eval.refined_queries)} refined queries")

                        elif retrieval_eval.recommended_action == CorrectiveAction.DECOMPOSE:
                            # Add decomposed questions to scratchpad
                            for sub_q in retrieval_eval.decomposed_questions[:4]:
                                if sub_q not in scratchpad.questions:
                                    scratchpad.add_question(sub_q)
                            logger.info(f"[{request_id}] CRAG: Decomposed into {len(retrieval_eval.decomposed_questions)} sub-questions")

                        elif retrieval_eval.recommended_action == CorrectiveAction.WEB_FALLBACK:
                            # Add fallback queries with different terms
                            for fallback_q in retrieval_eval.refined_queries[:3]:
                                if fallback_q not in state.executed_queries:
                                    state.pending_queries.append(fallback_q)
                            logger.info(f"[{request_id}] CRAG: Web fallback with {len(retrieval_eval.refined_queries)} alternative queries")

                    except Exception as e:
                        logger.warning(f"[{request_id}] CRAG evaluation failed (non-fatal): {e}")

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
            # Use configurable scrape refinements from request (default 3)
            max_scrape_refinements = getattr(request, 'max_scrape_refinements', 3)
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
                        max_urls=request.max_urls_to_scrape
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
                                        "content": scraped["content"][:request.max_content_per_source],
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

            # STEP 4.5: SUFFICIENT CONTEXT CHECK & POSITIONAL OPTIMIZATION
            # Based on Google's "Sufficient Context" research (arXiv:2411.06037)
            # Check if we have enough context to answer the query before synthesis
            sufficiency_result = None
            try:
                if scraped_content and len(scraped_content) >= 2:
                    # Combine scraped content for sufficiency check
                    combined_context = "\n\n---\n\n".join([
                        f"Source: {sc.get('title', 'Unknown')}\n{sc.get('content', '')[:3000]}"
                        for sc in scraped_content[:10]  # Check first 10 sources
                    ])

                    # Run sufficient context classification
                    context_classifier = get_sufficient_context_classifier()
                    sufficiency_result = await context_classifier.classify(
                        question=request.query,
                        context=combined_context
                    )

                    search_trace.append({
                        "step": "sufficient_context_check",
                        "is_sufficient": sufficiency_result.is_sufficient,
                        "confidence": sufficiency_result.confidence,
                        "sufficiency_level": sufficiency_result.sufficiency_level.value,
                        "missing_info": sufficiency_result.missing_information[:3],
                        "recommendation": sufficiency_result.recommendation
                    })

                    logger.info(
                        f"[{request_id}] Sufficient context check: "
                        f"sufficient={sufficiency_result.is_sufficient}, "
                        f"confidence={sufficiency_result.confidence:.2f}, "
                        f"level={sufficiency_result.sufficiency_level.value}"
                    )

                    # If insufficient and we can still refine, add targeted queries
                    if (not sufficiency_result.is_sufficient and
                        sufficiency_result.confidence >= 0.7 and
                        sufficiency_result.missing_information and
                        state.iteration < request.max_iterations - 1):

                        logger.warning(
                            f"[{request_id}] Context INSUFFICIENT: {sufficiency_result.missing_information}"
                        )
                        # Generate targeted queries for missing information
                        missing_queries = [
                            f"{request.query} {missing}"
                            for missing in sufficiency_result.missing_information[:2]
                        ]
                        state.add_pending_queries(missing_queries)
                        logger.info(f"[{request_id}] Added targeted queries for missing info: {missing_queries}")

            except Exception as e:
                logger.warning(f"[{request_id}] Sufficient context check failed (non-fatal): {e}")

            # STEP 4.6: DYNAMIC CONTEXT ALLOCATION
            # Adjust context budget based on current confidence and sufficiency
            dynamic_allocator = get_dynamic_allocator()
            current_confidence = 0.5  # Default mid-point

            # Estimate current confidence from verified claims
            if state.verified_claims:
                verified_ratio = sum(1 for v in state.verified_claims if v.verified) / len(state.verified_claims)
                current_confidence = verified_ratio

            context_budget = dynamic_allocator.calculate_budget(
                current_confidence=current_confidence,
                sufficiency_result=sufficiency_result,
                iteration=state.iteration,
                source_count=len(scraped_content) if scraped_content else len(state.raw_results)
            )

            # Apply dynamic limits to scraped content
            if scraped_content:
                max_per_source = context_budget["max_per_source"]
                max_sources = context_budget["max_sources"]

                # Truncate content per source based on dynamic allocation
                for sc in scraped_content:
                    if len(sc.get("content", "")) > max_per_source:
                        sc["content"] = sc["content"][:max_per_source]

                # Limit number of sources if needed
                if len(scraped_content) > max_sources:
                    logger.info(
                        f"[{request_id}] Dynamic allocation: reducing sources from "
                        f"{len(scraped_content)} to {max_sources} (confidence={current_confidence:.2f})"
                    )
                    scraped_content = scraped_content[:max_sources]

                search_trace.append({
                    "step": "dynamic_context_allocation",
                    "allocation_ratio": context_budget["allocation_ratio"],
                    "max_total_content": context_budget["max_total_content"],
                    "max_per_source": max_per_source,
                    "max_sources": max_sources,
                    "current_confidence": current_confidence
                })

            # STEP 4.7: POSITIONAL OPTIMIZATION (Lost-in-the-Middle Mitigation)
            # Reorder sources to place most relevant at beginning and end
            if scraped_content and len(scraped_content) >= 5:
                try:
                    positional_optimizer = get_positional_optimizer()

                    # Score relevance of each source
                    relevance_scores = await positional_optimizer.score_relevance(
                        request.query,
                        scraped_content[:15]  # Score top 15
                    )

                    # Reorder for optimal attention
                    reordered_content, positional_analysis = positional_optimizer.reorder_for_optimal_attention(
                        scraped_content[:15],
                        relevance_scores
                    )

                    # Replace scraped content with reordered version
                    scraped_content = reordered_content + scraped_content[15:]

                    search_trace.append({
                        "step": "positional_optimization",
                        "reordered_indices": positional_analysis.optimal_reorder,
                        "beginning_relevance": round(positional_analysis.beginning_relevance, 2),
                        "middle_relevance": round(positional_analysis.middle_relevance, 2),
                        "end_relevance": round(positional_analysis.end_relevance, 2),
                        "lost_in_middle_risk": positional_analysis.lost_in_middle_risk
                    })

                    logger.info(
                        f"[{request_id}] Positional optimization: "
                        f"begin={positional_analysis.beginning_relevance:.2f}, "
                        f"middle={positional_analysis.middle_relevance:.2f}, "
                        f"end={positional_analysis.end_relevance:.2f}, "
                        f"risk={positional_analysis.lost_in_middle_risk}"
                    )

                except Exception as e:
                    logger.warning(f"[{request_id}] Positional optimization failed (non-fatal): {e}")

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

            # STEP 6: SELF-RAG REFLECTION - Check synthesis quality before returning
            # Implements ISREL/ISSUP/ISUSE checks and temporal validation
            logger.info(f"[{request_id}] Performing Self-RAG reflection on synthesis...")
            try:
                sources_for_reflection = [
                    {"title": r.title, "snippet": r.snippet, "url": r.url}
                    for r in state.raw_results[:10]
                ]
                reflection_result = await self.reflector.reflect(
                    query=request.query,
                    synthesis=synthesis,
                    sources=sources_for_reflection,
                    scraped_content=scraped_content
                )

                # Log reflection results
                logger.info(
                    f"[{request_id}] Reflection: relevance={reflection_result.relevance_score:.2f}, "
                    f"support={reflection_result.support_level.value}, "
                    f"usefulness={reflection_result.usefulness_score:.2f}, "
                    f"temporal_conflicts={len(reflection_result.temporal_conflicts)}"
                )

                # If reflection finds issues, attempt to refine synthesis
                if reflection_result.needs_refinement:
                    logger.warning(
                        f"[{request_id}] Synthesis needs refinement: "
                        f"suggestions={reflection_result.refinement_suggestions}"
                    )
                    # Only refine if we have temporal conflicts (most serious issue)
                    if reflection_result.temporal_conflicts:
                        refined_synthesis = await self.reflector.refine_synthesis(
                            synthesis,
                            reflection_result,
                            sources_for_reflection
                        )
                        if refined_synthesis != synthesis:
                            logger.info(f"[{request_id}] Synthesis refined to fix temporal issues")
                            synthesis = refined_synthesis

                # Add reflection to search trace
                search_trace.append({
                    "step": "self_reflection",
                    "reflection": reflection_result.to_dict()
                })

            except Exception as e:
                logger.warning(f"[{request_id}] Self-reflection failed (non-fatal): {e}")
                reflection_result = None

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
            base_confidence = self.verifier.calculate_overall_confidence(
                state.verified_claims,
                source_count=len(state.raw_results),
                unique_domains=len(state.unique_domains),
                synthesis_length=len(synthesis),
                scraped_sources=scraped_count
            )

            # Blend base confidence with reflection-based confidence (if available)
            if reflection_result is not None:
                # Use reflection's confidence which accounts for ISREL/ISSUP/ISUSE + temporal
                reflection_confidence = reflection_result.overall_confidence
                # Blend: 60% base verifier confidence, 40% reflection confidence
                confidence_score = (base_confidence * 0.6) + (reflection_confidence * 0.4)
                logger.info(
                    f"[{request_id}] Blended confidence: base={base_confidence:.2f}, "
                    f"reflection={reflection_confidence:.2f}, final={confidence_score:.2f}"
                )
            else:
                confidence_score = base_confidence

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

            # Experience distillation: Capture successful searches for template learning
            if response.success and confidence_score >= 0.75:
                try:
                    # Determine query type from analysis or classification
                    query_type = "research"  # default
                    if state.query_analysis:
                        query_type = state.query_analysis.query_type
                    elif hasattr(state, 'classification') and state.classification:
                        query_type = state.classification.category.value

                    # Get decomposed questions from scratchpad
                    decomposed = list(scratchpad.questions.keys()) if scratchpad else []

                    await self.experience_distiller.capture_experience(
                        query=request.query,
                        response=response,
                        query_type=query_type,
                        decomposed_questions=decomposed
                    )
                    logger.debug(f"[{request_id}] Experience captured for template distillation")
                except Exception as exp_err:
                    logger.debug(f"[{request_id}] Experience capture failed: {exp_err}")

            # Classifier feedback: Record outcome for adaptive learning
            if state.query_analysis:
                try:
                    from .query_classifier import QueryClassification, QueryCategory, RecommendedPipeline, QueryComplexity

                    # Create a pseudo-classification from query analysis
                    pseudo_classification = QueryClassification(
                        category=QueryCategory(state.query_analysis.query_type),
                        capabilities=["web_search"] if state.query_analysis.requires_search else [],
                        complexity=QueryComplexity.MODERATE,  # Default
                        urgency="medium",
                        use_thinking_model=False,
                        recommended_pipeline=RecommendedPipeline.AGENTIC_SEARCH,
                        reasoning=state.query_analysis.search_reasoning or ""
                    )

                    self.classifier_feedback.record_outcome(
                        query=request.query,
                        classification=pseudo_classification,
                        confidence=confidence_score,
                        iteration_count=state.iteration,
                        source_count=len(state.raw_results),
                        execution_time_ms=execution_time_ms,
                        was_successful=response.success
                    )
                    logger.debug(f"[{request_id}] Classifier outcome recorded for feedback loop")
                except Exception as fb_err:
                    logger.debug(f"[{request_id}] Classifier feedback failed: {fb_err}")

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

            # Initialize graph state for visualization
            reset_graph_state()
            graph = get_graph_state()

            # Emit scratchpad initialization event
            await emitter.emit(events.scratchpad_initialized(request_id, [request.query]))

            # STEP 1: ANALYZE
            if request.analyze_query:
                # Graph: Enter analyze node
                await emitter.emit(graph_node_entered(request_id, "analyze", graph))
                await emitter.emit(events.analyzing_query(request_id, request.query))

                analyze_start = time.time()
                state.query_analysis = await self.analyzer.analyze(
                    request.query,
                    request.context
                )
                analyze_ms = int((time.time() - analyze_start) * 1000)

                await emitter.emit(events.query_analyzed(
                    request_id,
                    state.query_analysis.requires_search,
                    state.query_analysis.query_type
                ))
                # Graph: Complete analyze node
                await emitter.emit(graph_node_completed(request_id, "analyze", True, graph, analyze_ms))

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
            # Graph: Enter plan node
            await emitter.emit(graph_node_entered(request_id, "plan", graph))
            await emitter.emit(events.planning_search(request_id))
            plan_start = time.time()

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

            # Emit scratchpad update with decomposed questions
            await emitter.emit(events.scratchpad_initialized(request_id, decomposed_qs))

            plan_ms = int((time.time() - plan_start) * 1000)
            await emitter.emit(events.search_planned(
                request_id,
                initial_queries,
                len(state.search_plan.search_phases)
            ))
            # Graph: Complete plan node
            await emitter.emit(graph_node_completed(request_id, "plan", True, graph, plan_ms))

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

                # Emit detailed iteration start
                await emitter.emit(events.iteration_start_detailed(
                    request_id,
                    state.iteration,
                    request.max_iterations,
                    len(state.pending_queries),
                    state.sources_consulted
                ))

                # ACT: Execute pending searches
                if state.pending_queries:
                    queries_to_execute = state.pending_queries[:3]

                    # Graph: Enter search node (only on first search in this iteration)
                    if state.iteration == 1:
                        await emitter.emit(graph_node_entered(request_id, "search", graph))
                    search_start = time.time()

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
                    search_ms = int((time.time() - search_start) * 1000)

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
                    # Emit graph state update
                    await emitter.emit(graph_state_update(
                        request_id, graph,
                        f"Search: {len(results)} results from {len(state.unique_domains)} domains"
                    ))

                    search_trace.append({
                        "step": "search",
                        "iteration": iteration + 1,
                        "queries": queries_to_execute,
                        "results_count": len(results),
                        "total_sources": state.sources_consulted,
                        "unique_domains": len(state.unique_domains),
                        "scratchpad_progress": sp_status['overall']
                    })

                # CRAG: Pre-synthesis retrieval quality evaluation
                if state.raw_results and iteration == 0:
                    try:
                        # Graph: Enter CRAG node
                        await emitter.emit(graph_node_entered(request_id, "crag", graph))
                        crag_start = time.time()
                        await emitter.emit(events.crag_evaluating(request_id, len(state.raw_results)))

                        retrieval_eval = await self.retrieval_evaluator.evaluate(
                            query=request.query,
                            search_results=[
                                {"title": r.title, "snippet": r.snippet, "url": r.url}
                                for r in state.raw_results[:10]
                            ],
                            decomposed_questions=list(scratchpad.questions.keys()) if scratchpad.questions else None
                        )

                        await emitter.emit(events.crag_evaluation_complete(
                            request_id,
                            retrieval_eval.quality.value,
                            retrieval_eval.overall_relevance,
                            retrieval_eval.recommended_action.value
                        ))

                        logger.info(
                            f"[{request_id}] CRAG result: quality={retrieval_eval.quality.value}, "
                            f"relevance={retrieval_eval.overall_relevance:.2f}, "
                            f"action={retrieval_eval.recommended_action.value}"
                        )

                        search_trace.append({
                            "step": "crag_evaluation",
                            "iteration": iteration + 1,
                            "quality": retrieval_eval.quality.value,
                            "relevance": retrieval_eval.overall_relevance,
                            "coverage": retrieval_eval.query_coverage,
                            "action": retrieval_eval.recommended_action.value
                        })

                        # Take corrective action if needed
                        if retrieval_eval.recommended_action == CorrectiveAction.REFINE_QUERY:
                            refined_queries = retrieval_eval.refined_queries[:3]
                            for refined_q in refined_queries:
                                if refined_q not in state.executed_queries:
                                    state.pending_queries.append(refined_q)
                            await emitter.emit(events.crag_refining(request_id, refined_queries))
                            logger.info(f"[{request_id}] CRAG: Added {len(refined_queries)} refined queries")

                        elif retrieval_eval.recommended_action == CorrectiveAction.DECOMPOSE:
                            for sub_q in retrieval_eval.decomposed_questions[:4]:
                                if sub_q not in scratchpad.questions:
                                    scratchpad.add_question(sub_q)
                            await emitter.emit(events.crag_refining(request_id, retrieval_eval.decomposed_questions[:4]))
                            logger.info(f"[{request_id}] CRAG: Decomposed into {len(retrieval_eval.decomposed_questions)} sub-questions")

                        elif retrieval_eval.recommended_action == CorrectiveAction.WEB_FALLBACK:
                            for fallback_q in retrieval_eval.refined_queries[:3]:
                                if fallback_q not in state.executed_queries:
                                    state.pending_queries.append(fallback_q)
                            await emitter.emit(events.web_search_fallback(request_id, "CRAG", "refined_search", "low relevance"))
                            logger.info(f"[{request_id}] CRAG: Web fallback with {len(retrieval_eval.refined_queries)} alternative queries")

                        # Graph: Complete CRAG node
                        crag_ms = int((time.time() - crag_start) * 1000)
                        await emitter.emit(graph_node_completed(request_id, "crag", True, graph, crag_ms))

                    except Exception as e:
                        logger.warning(f"[{request_id}] CRAG evaluation failed (non-fatal): {e}")
                        # Graph: Complete CRAG node with failure
                        await emitter.emit(graph_node_completed(request_id, "crag", False, graph, 0))

                # VERIFY
                if (request.verification_level != VerificationLevel.NONE and
                    state.raw_results and not state.verified_claims):

                    # Graph: Enter verify node
                    await emitter.emit(graph_node_entered(request_id, "verify", graph))
                    verify_start = time.time()

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

                        verify_ms = int((time.time() - verify_start) * 1000)
                        await emitter.emit(events.claims_verified(
                            request_id, verified_count, len(state.verified_claims)
                        ))
                        # Graph: Complete verify node
                        await emitter.emit(graph_node_completed(request_id, "verify", True, graph, verify_ms))

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
            # Use configurable scrape refinements from request (default 3)
            max_scrape_refinements = getattr(request, 'max_scrape_refinements', 3)
            scrape_refinement = 0
            scraped_content = []

            # Graph: Enter scrape node
            await emitter.emit(graph_node_entered(request_id, "scrape", graph))
            scrape_start = time.time()

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
                        max_urls=request.max_urls_to_scrape
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
                                        "content": scraped["content"][:request.max_content_per_source],
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

                    # Emit coverage evaluation event
                    await emitter.emit(events.coverage_evaluating(request_id))

                    # TTL pin cache during LLM analysis
                    async with ToolCallContext(request_id, ToolType.OLLAMA_GENERATE, manager=self.ttl_manager):
                        coverage = await self.analyzer.evaluate_content_coverage(
                            query=request.query,
                            decomposed_questions=state.search_plan.decomposed_questions,
                            scraped_content=scraped_content
                        )

                    # Emit coverage evaluation result
                    await emitter.emit(events.coverage_evaluated(
                        request_id,
                        coverage.get("coverage_score", 0),
                        coverage.get("is_sufficient", True),
                        coverage.get("information_gaps", [])
                    ))

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

                        # Emit coverage insufficient and refinement events
                        await emitter.emit(events.coverage_insufficient(request_id, coverage.get("information_gaps", [])))
                        await emitter.emit(events.refinement_cycle_start(request_id, scrape_refinement + 1, max_scrape_refinements))
                        await emitter.emit(events.refinement_queries_generated(request_id, new_queries))

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

            # Graph: Complete scrape node
            scrape_ms = int((time.time() - scrape_start) * 1000)
            await emitter.emit(graph_node_completed(request_id, "scrape", len(scraped_content) > 0, graph, scrape_ms))

            # STEP 5: SYNTHESIZE with scraped content
            # Graph: Enter synthesize node
            await emitter.emit(graph_node_entered(request_id, "synthesize", graph))
            synth_start = time.time()
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
            synth_ms = int((time.time() - synth_start) * 1000)
            await emitter.emit(events.synthesis_complete(
                request_id,
                len(synthesis),
                interim_confidence
            ))
            # Graph: Complete synthesize node
            await emitter.emit(graph_node_completed(request_id, "synthesize", len(synthesis) > 0, graph, synth_ms))

            # STEP 6: SELF-RAG REFLECTION - Check synthesis quality before returning
            # Graph: Enter reflect node
            await emitter.emit(graph_node_entered(request_id, "reflect", graph))
            reflect_start = time.time()
            logger.info(f"[{request_id}] Performing Self-RAG reflection on synthesis...")
            reflection_result = None
            try:
                await emitter.emit(events.self_rag_reflecting(request_id, len(synthesis)))

                sources_for_reflection = [
                    {"title": r.title, "snippet": r.snippet, "url": r.url}
                    for r in state.raw_results[:10]
                ]
                reflection_result = await self.reflector.reflect(
                    query=request.query,
                    synthesis=synthesis,
                    sources=sources_for_reflection,
                    scraped_content=scraped_content
                )

                # Emit reflection complete event
                await emitter.emit(events.self_rag_complete(
                    request_id,
                    reflection_result.relevance_score,
                    reflection_result.support_level.value,
                    reflection_result.usefulness_score,
                    len(reflection_result.temporal_conflicts)
                ))

                logger.info(
                    f"[{request_id}] Reflection: relevance={reflection_result.relevance_score:.2f}, "
                    f"support={reflection_result.support_level.value}, "
                    f"usefulness={reflection_result.usefulness_score:.2f}, "
                    f"temporal_conflicts={len(reflection_result.temporal_conflicts)}"
                )

                # If reflection finds issues, attempt to refine synthesis
                if reflection_result.needs_refinement:
                    logger.warning(
                        f"[{request_id}] Synthesis needs refinement: "
                        f"suggestions={reflection_result.refinement_suggestions}"
                    )
                    # Only refine if we have temporal conflicts (most serious issue)
                    if reflection_result.temporal_conflicts:
                        await emitter.emit(events.self_rag_refining(request_id, "temporal conflicts detected"))

                        refined_synthesis = await self.reflector.refine_synthesis(
                            synthesis,
                            reflection_result,
                            sources_for_reflection
                        )
                        if refined_synthesis != synthesis:
                            logger.info(f"[{request_id}] Synthesis refined to fix temporal issues")
                            synthesis = refined_synthesis

                # Add reflection to search trace
                search_trace.append({
                    "step": "self_reflection",
                    "reflection": reflection_result.to_dict()
                })

                # Graph: Complete reflect node (success)
                reflect_ms = int((time.time() - reflect_start) * 1000)
                await emitter.emit(graph_node_completed(request_id, "reflect", True, graph, reflect_ms))

            except Exception as e:
                logger.warning(f"[{request_id}] Self-reflection failed (non-fatal): {e}")
                # Graph: Complete reflect node (failure)
                reflect_ms = int((time.time() - reflect_start) * 1000)
                await emitter.emit(graph_node_completed(request_id, "reflect", False, graph, reflect_ms))

            # Graph: Enter complete node
            await emitter.emit(graph_node_entered(request_id, "complete", graph))

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
            base_confidence = self.verifier.calculate_overall_confidence(
                state.verified_claims,
                source_count=len(state.raw_results),
                unique_domains=len(state.unique_domains),
                synthesis_length=len(synthesis),
                scraped_sources=scraped_count
            )

            # Blend base confidence with reflection-based confidence (if available)
            if reflection_result is not None:
                reflection_confidence = reflection_result.overall_confidence
                confidence_score = (base_confidence * 0.6) + (reflection_confidence * 0.4)
                logger.info(
                    f"[{request_id}] Blended confidence: base={base_confidence:.2f}, "
                    f"reflection={reflection_confidence:.2f}, final={confidence_score:.2f}"
                )
            else:
                confidence_score = base_confidence

            confidence_level = self.synthesizer.determine_confidence_level(
                state.verified_claims, len(state.raw_results)
            )

            # Emit corpus quality assessment event
            await emitter.emit(events.corpus_quality_assessed(
                request_id,
                confidence_score,
                len(state.raw_results),
                len(state.unique_domains),
                state.iteration
            ))

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

            # Experience distillation: Capture successful searches for template learning
            if response.success and confidence_score >= 0.75:
                try:
                    query_type = "research"  # default
                    if state.query_analysis:
                        query_type = state.query_analysis.query_type
                    decomposed = list(scratchpad.questions.keys()) if scratchpad else []

                    await self.experience_distiller.capture_experience(
                        query=request.query,
                        response=response,
                        query_type=query_type,
                        decomposed_questions=decomposed
                    )
                    await emitter.emit(events.experience_captured(request_id, query_type, confidence_score))
                    logger.debug(f"[{request_id}] Experience captured for template distillation")
                except Exception as exp_err:
                    logger.debug(f"[{request_id}] Experience capture failed: {exp_err}")

            # Classifier feedback: Record outcome for adaptive learning
            if state.query_analysis:
                try:
                    from .query_classifier import QueryClassification, QueryCategory, RecommendedPipeline, QueryComplexity

                    pseudo_classification = QueryClassification(
                        category=QueryCategory(state.query_analysis.query_type),
                        capabilities=["web_search"] if state.query_analysis.requires_search else [],
                        complexity=QueryComplexity.MODERATE,
                        urgency="medium",
                        use_thinking_model=False,
                        recommended_pipeline=RecommendedPipeline.AGENTIC_SEARCH,
                        reasoning=state.query_analysis.search_reasoning or ""
                    )

                    outcome = self.classifier_feedback.record_outcome(
                        query=request.query,
                        classification=pseudo_classification,
                        confidence=confidence_score,
                        iteration_count=state.iteration,
                        source_count=len(state.raw_results),
                        execution_time_ms=execution_time_ms,
                        was_successful=response.success
                    )
                    await emitter.emit(events.outcome_recorded(
                        request_id,
                        outcome.predicted_category,
                        outcome.outcome_quality.value,
                        outcome.was_overkill,
                        outcome.was_underkill
                    ))
                    logger.debug(f"[{request_id}] Classifier outcome recorded for feedback loop")
                except Exception as fb_err:
                    logger.debug(f"[{request_id}] Classifier feedback failed: {fb_err}")

            # Graph: Complete the graph - emit final state
            await emitter.emit(graph_node_completed(request_id, "complete", True, graph, execution_time_ms))
            await emitter.emit(graph_state_update(
                request_id, graph,
                f"Complete: {graph.to_line_simple()}"
            ))

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
