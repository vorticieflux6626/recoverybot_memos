"""
Enhanced Agentic Search Orchestrator

DEPRECATED: This module is deprecated. Use UniversalOrchestrator instead.

    from agentic import UniversalOrchestrator, OrchestratorPreset
    orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)

All enhanced features are available in UniversalOrchestrator:
- enable_pre_act_planning=True
- enable_stuck_detection=True
- enable_contradiction_detection=True
- enable_parallel_execution=True

---

Integrates research-backed improvements with the base orchestrator:
1. Pre-Act Pattern - Multi-step planning before acting
2. Self-Reflection Loop - Critique and refinement
3. Stuck State Detection - Loop recovery
4. Parallel Action Execution - Concurrent searches
5. Contradiction Detection - Surface conflicts

This orchestrator extends the base functionality while maintaining
backward compatibility with existing API endpoints.
"""

import asyncio
import warnings
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
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
    SearchPlan
)
from .analyzer import QueryAnalyzer
from .planner import PlannerAgent
from .searcher import SearcherAgent
from .verifier import VerifierAgent
from .synthesizer import SynthesizerAgent, DEFAULT_THINKING_MODEL
from .scraper import ContentScraper, DeepReader
from .scratchpad import AgenticScratchpad, ScratchpadManager, QuestionStatus, FindingType
from .content_cache import get_content_cache
from .ttl_cache_manager import get_ttl_cache_manager, ToolType, ToolCallContext
from .context_limits import (
    get_synthesizer_limits,
    get_dynamic_source_allocation,
    SYNTHESIZER_LIMITS,
    THINKING_SYNTHESIZER_LIMITS,
)
from .enhanced_reasoning import (
    EnhancedReasoningEngine,
    get_enhanced_reasoning,
    PreActPlan,
    PlannedAction,
    ActionType,
    ReflectionResult,
    ContradictionInfo,
    StuckStateMetrics
)
from . import events
from .events import EventEmitter, EventType, SearchEvent

logger = logging.getLogger("agentic.orchestrator_enhanced")


class EnhancedAgenticOrchestrator:
    """
    Enhanced orchestrator with research-backed improvements.

    DEPRECATED: Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED) instead.

    Key improvements over base orchestrator:
    - Pre-Act planning for better action sequences
    - Self-reflection for synthesis quality
    - Stuck state detection and recovery
    - Parallel search execution
    - Contradiction surfacing
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        mcp_url: str = "http://localhost:7777",
        brave_api_key: Optional[str] = None,
        memory_service: Optional[Any] = None,
        enable_reflection: bool = True,
        enable_pre_act: bool = True,
        enable_stuck_detection: bool = True,
        enable_contradiction_detection: bool = True,
        max_reflection_iterations: int = 2
    ):
        warnings.warn(
            "EnhancedAgenticOrchestrator is deprecated. Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url

        # Feature flags
        self.enable_reflection = enable_reflection
        self.enable_pre_act = enable_pre_act
        self.enable_stuck_detection = enable_stuck_detection
        self.enable_contradiction_detection = enable_contradiction_detection
        self.max_reflection_iterations = max_reflection_iterations

        # Initialize base agents
        self.analyzer = QueryAnalyzer(ollama_url=ollama_url)
        self.planner = PlannerAgent(ollama_url=ollama_url, mcp_url=mcp_url)
        self.searcher = SearcherAgent(brave_api_key=brave_api_key)
        self.verifier = VerifierAgent(ollama_url=ollama_url)
        self.synthesizer = SynthesizerAgent(ollama_url=ollama_url, mcp_url=mcp_url)

        # Content scraper
        self.scraper = ContentScraper()
        self.deep_reader = DeepReader(ollama_url=ollama_url)

        # Enhanced reasoning engine
        self.reasoning_engine = get_enhanced_reasoning(ollama_url)

        # Scratchpad manager
        self.scratchpad_manager = ScratchpadManager(memory_service=memory_service)

        # TTL cache manager
        self.ttl_manager = get_ttl_cache_manager()

        # Memory service
        self.memory_service = memory_service

        # Stats
        self.stats = {
            'searches_completed': 0,
            'pre_act_plans_used': 0,
            'reflections_triggered': 0,
            'stuck_recoveries': 0,
            'parallel_batches_executed': 0,
            'contradictions_surfaced': 0,
            'avg_quality_improvement': 0.0
        }

    async def initialize(self):
        """Initialize orchestrator and check service availability"""
        mcp_available = await self.planner.check_mcp_available()
        await self.synthesizer.check_mcp_available()
        logger.info(f"Enhanced orchestrator initialized. MCP available: {mcp_available}")

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute enhanced agentic search with research-backed patterns.

        Flow:
        1. Pre-Act Planning (if enabled)
        2. Parallel Search Execution
        3. Content Scraping
        4. Contradiction Detection
        5. Synthesis
        6. Self-Reflection & Refinement
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.info(f"[{request_id}] Starting ENHANCED agentic search: {request.query[:50]}...")
        logger.info(f"[{request_id}] Features: pre_act={self.enable_pre_act}, "
                   f"reflection={self.enable_reflection}, "
                   f"stuck_detection={self.enable_stuck_detection}")

        # Initialize state tracking
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

        # Stuck state metrics
        stuck_metrics = StuckStateMetrics()

        search_trace = []
        scraped_content = []
        contradictions = []

        try:
            # ================================================================
            # STEP 1: PRE-ACT PLANNING
            # ================================================================
            pre_act_plan = None
            if self.enable_pre_act:
                pre_act_plan = await self.reasoning_engine.create_pre_act_plan(
                    request.query,
                    request.context,
                    max_actions=6
                )
                search_trace.append({
                    "step": "pre_act_plan",
                    "actions_planned": len(pre_act_plan.actions),
                    "confidence": pre_act_plan.confidence,
                    "reasoning": pre_act_plan.reasoning
                })
                self.stats['pre_act_plans_used'] += 1

            # ================================================================
            # STEP 2: QUERY ANALYSIS
            # ================================================================
            if request.analyze_query:
                state.query_analysis = await self.analyzer.analyze(
                    request.query,
                    request.context
                )
                search_trace.append({
                    "step": "analyze",
                    "requires_search": state.query_analysis.requires_search,
                    "query_type": state.query_analysis.query_type,
                    "complexity": state.query_analysis.estimated_complexity
                })

                if not state.query_analysis.requires_search:
                    # Direct synthesis without search
                    synthesis = await self.synthesizer.synthesize(
                        request.query, [], None, request.context
                    )
                    return self._build_response(
                        request_id, start_time, state, synthesis,
                        [], search_trace, 0.6
                    )

            # ================================================================
            # STEP 3: SEARCH PLANNING
            # ================================================================
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

            # Set scratchpad mission
            scratchpad.set_mission(
                state.search_plan.decomposed_questions,
                {f"q{i+1}": f"Find info about: {q}"
                 for i, q in enumerate(state.search_plan.decomposed_questions)}
            )

            search_trace.append({
                "step": "plan",
                "queries": len(initial_queries),
                "decomposed_questions": len(state.search_plan.decomposed_questions)
            })

            # ================================================================
            # STEP 4: PARALLEL SEARCH EXECUTION
            # ================================================================
            for iteration in range(request.max_iterations):
                state.iteration = iteration + 1

                # Check for stuck state
                if self.enable_stuck_detection:
                    is_stuck, stuck_reason = self.reasoning_engine.check_stuck_state(
                        stuck_metrics,
                        current_query=str(state.pending_queries)
                    )
                    if is_stuck:
                        recovery = await self.reasoning_engine.recover_from_stuck(
                            request.query, {"results": state.raw_results}, stuck_reason
                        )
                        if recovery["strategy"] == "ACCEPT":
                            logger.info(f"[{request_id}] Stuck state: accepting current results")
                            break
                        elif recovery.get("new_queries"):
                            state.add_pending_queries(recovery["new_queries"])
                            self.stats['stuck_recoveries'] += 1

                if not state.pending_queries:
                    logger.info(f"[{request_id}] No more queries to execute")
                    break

                # Execute searches in parallel batches
                queries_batch = state.pending_queries[:4]  # Up to 4 parallel
                logger.info(f"[{request_id}] Iteration {iteration+1}: "
                           f"executing {len(queries_batch)} queries in parallel")

                async with ToolCallContext(request_id, ToolType.WEB_SEARCH, manager=self.ttl_manager):
                    # Execute all queries concurrently
                    search_tasks = [
                        self.searcher.search([q]) for q in queries_batch
                    ]
                    results_list = await asyncio.gather(*search_tasks, return_exceptions=True)

                # Process results
                for i, results in enumerate(results_list):
                    if isinstance(results, Exception):
                        logger.warning(f"[{request_id}] Search failed: {results}")
                        continue
                    state.add_results(results)
                    for r in results:
                        scratchpad.add_finding(
                            question_id=f"q{(i % len(state.search_plan.decomposed_questions)) + 1}",
                            content=r.snippet[:500],
                            source_url=r.url,
                            source_title=r.title,
                            finding_type=FindingType.FACT,
                            confidence=0.5
                        )

                # Mark as executed
                for q in queries_batch:
                    state.mark_query_executed(q)

                self.stats['parallel_batches_executed'] += 1

                search_trace.append({
                    "step": "parallel_search",
                    "iteration": iteration + 1,
                    "queries": len(queries_batch),
                    "total_results": len(state.raw_results)
                })

                # Check if sufficient
                if state.has_sufficient_sources(request.min_sources):
                    should_continue, reason, new_queries = await self.analyzer.should_continue_search(
                        request.query,
                        [{"title": r.title, "snippet": r.snippet} for r in state.raw_results],
                        state.search_plan,
                        iteration
                    )
                    if not should_continue:
                        logger.info(f"[{request_id}] Sufficient information gathered")
                        break
                    if new_queries:
                        state.add_pending_queries(new_queries)

            # ================================================================
            # STEP 5: CONTENT SCRAPING
            # ================================================================
            if state.raw_results:
                results_for_eval = [
                    {"url": r.url, "title": r.title, "snippet": r.snippet, "source_domain": r.source_domain}
                    for r in state.raw_results
                ]

                relevant_urls = await self.analyzer.evaluate_urls_for_scraping(
                    request.query, results_for_eval, max_urls=request.max_urls_to_scrape
                )

                for url_info in relevant_urls:
                    url = url_info["url"]
                    if url.lower().endswith('.pdf') or scratchpad.has_scraped(url):
                        continue
                    try:
                        async with ToolCallContext(request_id, ToolType.WEB_SCRAPE, manager=self.ttl_manager):
                            scraped = await self.scraper.scrape_url(url)
                        if scraped.get("success") and scraped.get("content"):
                            scraped_content.append({
                                "url": url,
                                "title": url_info["title"],
                                "content": scraped["content"][:request.max_content_per_source]
                            })
                            scratchpad.record_scrape(url)
                    except Exception as e:
                        logger.warning(f"[{request_id}] Scrape failed for {url}: {e}")

                search_trace.append({
                    "step": "scrape",
                    "sources_scraped": len(scraped_content)
                })

            # ================================================================
            # STEP 6: CONTRADICTION DETECTION
            # ================================================================
            if self.enable_contradiction_detection and len(scraped_content) >= 2:
                key_claims = state.search_plan.decomposed_questions[:5]
                contradictions = await self.reasoning_engine.detect_contradictions(
                    scraped_content, key_claims
                )
                if contradictions:
                    self.stats['contradictions_surfaced'] += len(contradictions)
                    search_trace.append({
                        "step": "contradiction_detection",
                        "contradictions_found": len(contradictions)
                    })

            # ================================================================
            # STEP 7: SYNTHESIS
            # ================================================================
            synthesis = await self._synthesize_with_full_content(
                request.query,
                state.raw_results,
                scraped_content,
                state.verified_claims,
                request.context,
                contradictions
            )

            # ================================================================
            # STEP 8: SELF-REFLECTION & REFINEMENT
            # ================================================================
            final_quality = 0.7
            if self.enable_reflection and synthesis:
                for reflection_iter in range(self.max_reflection_iterations):
                    reflection = await self.reasoning_engine.reflect_on_synthesis(
                        request.query, synthesis, scraped_content, reflection_iter
                    )

                    search_trace.append({
                        "step": "reflection",
                        "iteration": reflection_iter + 1,
                        "quality_score": reflection.quality_score,
                        "issues": reflection.issues,
                        "should_refine": reflection.should_refine
                    })

                    if not reflection.should_refine or reflection.quality_score >= 0.85:
                        final_quality = reflection.quality_score
                        break

                    # Refine synthesis
                    self.stats['reflections_triggered'] += 1
                    synthesis = await self.reasoning_engine.refine_synthesis(
                        request.query, synthesis, reflection, scraped_content
                    )
                    final_quality = reflection.quality_score + 0.1  # Improvement bonus

            # ================================================================
            # STEP 9: CALCULATE CONFIDENCE
            # ================================================================
            confidence_score = self._calculate_enhanced_confidence(
                state, scraped_content, contradictions, final_quality
            )

            self.stats['searches_completed'] += 1

            return self._build_response(
                request_id, start_time, state, synthesis,
                scraped_content, search_trace, confidence_score,
                contradictions=contradictions
            )

        except Exception as e:
            logger.error(f"[{request_id}] Enhanced search failed: {e}")
            import traceback
            traceback.print_exc()
            return SearchResponse(
                success=False,
                data=SearchResultData(
                    synthesized_context=f"Search failed: {str(e)}",
                    sources=[],
                    search_queries=[],
                    confidence_score=0.0,
                    confidence_level=ConfidenceLevel.LOW
                ),
                meta=SearchMeta(
                    request_id=request_id,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    error=str(e)
                )
            )

    async def _synthesize_with_full_content(
        self,
        query: str,
        search_results: List,
        scraped_content: List[Dict],
        verifications: Optional[List] = None,
        context: Optional[Dict] = None,
        contradictions: List[ContradictionInfo] = None
    ) -> str:
        """Synthesize with full scraped content and contradiction awareness"""

        # Build comprehensive source text - use ALL scraped content to maximize context utilization
        # Determine if we're using a thinking model for dynamic limit calculation
        is_thinking_model = hasattr(self, '_use_thinking_model') and self._use_thinking_model

        if scraped_content:
            sources_text = ""
            # Get dynamic limits based on model context window
            limits = THINKING_SYNTHESIZER_LIMITS if is_thinking_model else SYNTHESIZER_LIMITS
            max_context_chars = limits["max_total_content"]
            max_sources = limits["max_urls_to_scrape"]
            chars_per_source_base = limits["max_content_per_source"]

            # Dynamic per-source allocation based on total sources available
            chars_per_source, sources_to_use = get_dynamic_source_allocation(
                len(scraped_content),
                is_thinking_model=is_thinking_model
            )

            logger.info(f"Enhanced synthesis: {max_context_chars} chars budget, {chars_per_source} chars/source for {len(scraped_content)} sources")

            for i, sc in enumerate(scraped_content, 1):
                content_preview = sc.get("content", "")[:chars_per_source]
                sources_text += f"\n\n[Source {i}] {sc.get('title', 'Unknown')} ({sc.get('url', '')[:50]}...)\n{content_preview}"

                # Stop if we're approaching context limit or max sources
                if len(sources_text) > max_context_chars or i >= max_sources:
                    break
        else:
            # Fallback to snippets - use more snippets with dynamic limits
            limits = SYNTHESIZER_LIMITS
            max_snippets = limits.get("max_snippets_if_no_scrape", 15)
            sources_text = "\n".join([
                f"[{i+1}] {r.title}: {r.snippet}"
                for i, r in enumerate(search_results[:max_snippets])
            ])

        # Add contradiction warnings - include all contradictions for complete context
        contradiction_text = ""
        if contradictions:
            contradiction_text = "\n\nNOTE - Conflicting information found:\n"
            for c in contradictions:
                contradiction_text += f"- {c.claim}: Sources disagree. {c.resolution_suggestion}\n"

        prompt = f"""You are a research synthesizer providing accurate, well-structured answers.
Create a comprehensive answer based on the sources provided.

Question: {query}

Sources:
{sources_text}
{contradiction_text}

Instructions:
1. Synthesize information from all sources
2. Cite sources using [Source N] format
3. If sources conflict, present both viewpoints
4. Be direct and solution-focused
5. Provide practical, actionable guidance
6. Acknowledge limitations if information is incomplete

Synthesized Answer:"""

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "qwen3:8b",
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.5,
                            "num_predict": 4096,
                            "num_ctx": 32000
                        }
                    }
                )
                response.raise_for_status()
                return response.json().get("response", "No synthesis generated")
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Synthesis failed: {e}"

    def _calculate_enhanced_confidence(
        self,
        state: SearchState,
        scraped_content: List[Dict],
        contradictions: List[ContradictionInfo],
        quality_score: float
    ) -> float:
        """Calculate confidence with enhanced multi-signal approach"""

        signals = []

        # Source diversity (25%)
        unique_domains = len(state.unique_domains)
        source_score = min(1.0, unique_domains / 5)
        signals.append(("source_diversity", 0.25, source_score))

        # Content depth (25%) - increased threshold to match expanded context window usage
        content_chars = sum(len(sc.get("content", "")) for sc in scraped_content)
        depth_score = min(1.0, content_chars / 50000)
        signals.append(("content_depth", 0.25, depth_score))

        # Contradiction penalty (15%)
        contradiction_penalty = len(contradictions) * 0.1
        consistency_score = max(0.0, 1.0 - contradiction_penalty)
        signals.append(("consistency", 0.15, consistency_score))

        # Reflection quality (20%)
        signals.append(("quality", 0.20, quality_score))

        # Coverage (15%) - increased threshold to match expanded source limits
        coverage_score = min(1.0, state.sources_consulted / 15)
        signals.append(("coverage", 0.15, coverage_score))

        # Calculate weighted score
        confidence = sum(weight * score for _, weight, score in signals)

        logger.debug(f"Confidence signals: {signals}, total={confidence:.2f}")

        return round(confidence, 2)

    def _build_response(
        self,
        request_id: str,
        start_time: float,
        state: SearchState,
        synthesis: str,
        scraped_content: List[Dict],
        search_trace: List[Dict],
        confidence_score: float,
        contradictions: List[ContradictionInfo] = None
    ) -> SearchResponse:
        """Build the final search response"""

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.5:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW

        # Build sources list
        sources = []
        for sc in scraped_content:
            sources.append({
                "url": sc.get("url", ""),
                "title": sc.get("title", ""),
                "domain": sc.get("url", "").split("/")[2] if "/" in sc.get("url", "") else ""
            })

        # Add contradiction info to trace if present
        if contradictions:
            search_trace.append({
                "step": "contradictions_summary",
                "count": len(contradictions),
                "claims": [c.claim for c in contradictions]
            })

        return SearchResponse(
            success=True,
            data=SearchResultData(
                synthesized_context=synthesis,
                sources=sources,
                search_queries=list(state.executed_queries),
                search_trace=search_trace,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                verification_status="enhanced"
            ),
            meta=SearchMeta(
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                execution_time_ms=execution_time_ms,
                iterations=state.iteration,
                queries_executed=len(state.executed_queries),
                sources_consulted=state.sources_consulted,
                cache_hit=False,
                enhanced_features={
                    "pre_act": self.enable_pre_act,
                    "reflection": self.enable_reflection,
                    "contradiction_detection": self.enable_contradiction_detection,
                    "stuck_detection": self.enable_stuck_detection
                }
            )
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            **self.stats,
            "reasoning_engine": self.reasoning_engine.get_stats()
        }


# Import httpx at module level
import httpx


def get_enhanced_orchestrator(
    ollama_url: str = "http://localhost:11434",
    **kwargs
) -> EnhancedAgenticOrchestrator:
    """Factory function to create enhanced orchestrator"""
    return EnhancedAgenticOrchestrator(ollama_url=ollama_url, **kwargs)
