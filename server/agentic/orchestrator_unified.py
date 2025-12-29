"""
Unified Agentic Orchestrator

DEPRECATED: This module is deprecated. Use UniversalOrchestrator instead.

    from agentic import UniversalOrchestrator, OrchestratorPreset
    orchestrator = UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED)

All unified features are available in UniversalOrchestrator with 'enhanced' preset:
- enable_hyde=True
- enable_hybrid_reranking=True
- enable_ragas=True
- enable_entity_tracking=True
- enable_thought_library=True

---

Integrates ALL available agentic features into a single enhanced pipeline:

QUERY UNDERSTANDING:
- QueryClassifier (existing)
- EntityTracker (NEW) - Track entities across sources
- EmbeddingAggregator (NEW) - Route to domain experts

QUERY EXPANSION:
- HyDEExpander (NEW) - Generate hypothetical documents for better retrieval

SEARCH EXECUTION:
- SearcherAgent (existing) - SearXNG/DuckDuckGo/Brave
- DomainCorpus (NEW) - Domain-specific knowledge augmentation

RESULT RE-RANKING:
- BGEM3HybridRetriever (NEW) - Dense+sparse fusion ranking

QUALITY CONTROL:
- RetrievalEvaluator (existing) - CRAG pre-synthesis
- SelfReflectionAgent (existing) - Self-RAG post-synthesis
- RAGASEvaluator (NEW) - Faithfulness/relevancy scoring

LEARNING:
- ThoughtLibrary (NEW) - Reuse successful reasoning patterns
- ExperienceDistiller (existing) - Learn from successes
- ClassifierFeedback (existing) - Adaptive classification

This module is designed as a gateway for all user messages, not just search-specific queries.
"""

import asyncio
import warnings
import logging
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import uuid

from .orchestrator import AgenticOrchestrator
from .models import (
    SearchRequest,
    SearchResponse,
    SearchResultData,
    SearchMeta,
    SearchState,
    WebSearchResult
)
from .events import EventEmitter, EventType, SearchEvent

# NEW Feature Imports
from .hyde import (
    HyDEExpander,
    HyDEMode,
    HyDEResult,
    get_hyde_expander,
    create_hyde_expander
)
from .bge_m3_hybrid import (
    BGEM3HybridRetriever,
    HybridDocument,
    HybridSearchResult,
    RetrievalMode,
    get_hybrid_retriever,
    create_hybrid_retriever
)
from .ragas import (
    RAGASEvaluator,
    RAGASResult,
    get_ragas_evaluator,
    create_ragas_evaluator
)
from .entity_tracker import (
    EntityTracker,
    EntityState,
    create_entity_tracker
)
from .thought_library import (
    ThoughtLibrary,
    ThoughtTemplate,
    TemplateCategory,
    get_thought_library
)
from .embedding_aggregator import (
    EmbeddingAggregator,
    get_embedding_aggregator
)

# Additional Feature Imports (previously unused)
from .metrics import (
    PerformanceMetrics,
    QueryMetrics,
    PhaseTimer,
    get_performance_metrics
)
from .domain_corpus import (
    DomainCorpusManager,
    CorpusRetriever,
    get_corpus_manager,
    initialize_default_corpuses
)
from .reasoning_dag import (
    ReasoningDAG,
    ReasoningNode,
    NodeType,
    NodeStatus,
    create_reasoning_dag
)
from .prefix_optimized_prompts import (
    SYSTEM_PREFIX,
    AGENT_ROLE_PREFIXES,
    build_scratchpad_context,
    estimate_prefix_reuse
)

logger = logging.getLogger("agentic.orchestrator_unified")


class UnifiedOrchestrator:
    """
    Unified orchestrator integrating all agentic features.

    DEPRECATED: Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED) instead.

    This is designed as the main gateway for ALL user messages,
    treating the system as a general AI agent rather than just a search tool.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        mcp_url: str = "http://localhost:7777",
        brave_api_key: Optional[str] = None,
        memory_service: Optional[Any] = None,
        enable_hyde: bool = True,
        enable_hybrid_reranking: bool = True,
        enable_ragas: bool = True,
        enable_entity_tracking: bool = True,
        enable_thought_library: bool = True,
        enable_domain_corpus: bool = True,
        enable_reasoning_dag: bool = False,  # Disabled by default (expensive for simple queries)
        enable_metrics: bool = True,
        db_path: Optional[str] = None
    ):
        warnings.warn(
            "UnifiedOrchestrator is deprecated. Use UniversalOrchestrator(preset=OrchestratorPreset.ENHANCED) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.ollama_url = ollama_url
        self.mcp_url = mcp_url

        # Feature flags for optional components
        self.enable_hyde = enable_hyde
        self.enable_hybrid_reranking = enable_hybrid_reranking
        self.enable_ragas = enable_ragas
        self.enable_entity_tracking = enable_entity_tracking
        self.enable_thought_library = enable_thought_library
        self.enable_domain_corpus = enable_domain_corpus
        self.enable_reasoning_dag = enable_reasoning_dag
        self.enable_metrics = enable_metrics

        # Database path for persistent components
        self.db_path = db_path or "/home/sparkone/sdd/Recovery_Bot/memOS/data"

        # Base orchestrator (existing functionality)
        self.base_orchestrator = AgenticOrchestrator(
            ollama_url=ollama_url,
            mcp_url=mcp_url,
            brave_api_key=brave_api_key,
            memory_service=memory_service
        )

        # NEW: Enhanced components (lazy loaded)
        self._hyde_expander: Optional[HyDEExpander] = None
        self._hybrid_retriever: Optional[BGEM3HybridRetriever] = None
        self._ragas_evaluator: Optional[RAGASEvaluator] = None
        self._entity_tracker: Optional[EntityTracker] = None
        self._thought_library: Optional[ThoughtLibrary] = None
        self._embedding_aggregator: Optional[EmbeddingAggregator] = None

        # Additional feature instances (previously unused)
        self._domain_corpus_manager: Optional[DomainCorpusManager] = None
        self._reasoning_dag: Optional[ReasoningDAG] = None
        self._metrics: Optional[PerformanceMetrics] = None

        # Performance tracking
        self.feature_timings: Dict[str, List[float]] = {
            "hyde": [],
            "hybrid_rerank": [],
            "ragas": [],
            "entity_tracking": [],
            "thought_library": [],
            "domain_corpus": [],
            "reasoning_dag": []
        }

    async def initialize(self):
        """Initialize all components."""
        await self.base_orchestrator.initialize()

        # Initialize enhanced components in parallel
        init_tasks = []

        if self.enable_hyde:
            init_tasks.append(self._init_hyde())

        if self.enable_hybrid_reranking:
            init_tasks.append(self._init_hybrid())

        if self.enable_ragas:
            init_tasks.append(self._init_ragas())

        if self.enable_entity_tracking:
            init_tasks.append(self._init_entity_tracker())

        if self.enable_thought_library:
            init_tasks.append(self._init_thought_library())

        if self.enable_domain_corpus:
            init_tasks.append(self._init_domain_corpus())

        if self.enable_metrics:
            init_tasks.append(self._init_metrics())

        await asyncio.gather(*init_tasks, return_exceptions=True)

        logger.info(
            f"UnifiedOrchestrator initialized with features: "
            f"HyDE={self.enable_hyde}, "
            f"Hybrid={self.enable_hybrid_reranking}, "
            f"RAGAS={self.enable_ragas}, "
            f"EntityTracking={self.enable_entity_tracking}, "
            f"ThoughtLibrary={self.enable_thought_library}, "
            f"DomainCorpus={self.enable_domain_corpus}, "
            f"Metrics={self.enable_metrics}"
        )

    async def _init_hyde(self):
        """Initialize HyDE expander."""
        try:
            self._hyde_expander = await create_hyde_expander(
                ollama_url=self.ollama_url,
                generation_model="gemma3:4b",
                embedding_model="bge-m3"
            )
            logger.info("HyDE expander initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize HyDE: {e}")
            self.enable_hyde = False

    async def _init_hybrid(self):
        """Initialize BGE-M3 hybrid retriever."""
        try:
            self._hybrid_retriever = await create_hybrid_retriever(
                ollama_url=self.ollama_url,
                db_path=f"{self.db_path}/unified_hybrid.db",
                load_existing=True
            )
            logger.info("BGE-M3 hybrid retriever initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid retriever: {e}")
            self.enable_hybrid_reranking = False

    async def _init_ragas(self):
        """Initialize RAGAS evaluator."""
        try:
            self._ragas_evaluator = await create_ragas_evaluator(
                ollama_url=self.ollama_url
            )
            logger.info("RAGAS evaluator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize RAGAS: {e}")
            self.enable_ragas = False

    async def _init_entity_tracker(self):
        """Initialize entity tracker."""
        try:
            self._entity_tracker = await create_entity_tracker(
                ollama_url=self.ollama_url
            )
            logger.info("Entity tracker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize entity tracker: {e}")
            self.enable_entity_tracking = False

    async def _init_thought_library(self):
        """Initialize thought library."""
        try:
            self._thought_library = get_thought_library()
            logger.info(f"Thought library initialized with {len(self._thought_library.templates)} templates")
        except Exception as e:
            logger.warning(f"Failed to initialize thought library: {e}")
            self.enable_thought_library = False

    async def _init_domain_corpus(self):
        """Initialize domain corpus manager for knowledge augmentation."""
        try:
            # Initialize with default domain schemas (FANUC, Raspberry Pi)
            self._domain_corpus_manager = initialize_default_corpuses(
                ollama_url=self.ollama_url
            )
            corpus_count = len(self._domain_corpus_manager.corpuses)
            logger.info(f"Domain corpus manager initialized with {corpus_count} corpuses")
        except Exception as e:
            logger.warning(f"Failed to initialize domain corpus: {e}")
            self.enable_domain_corpus = False

    async def _init_metrics(self):
        """Initialize performance metrics tracking."""
        try:
            self._metrics = get_performance_metrics()
            logger.info("Performance metrics initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize metrics: {e}")
            self.enable_metrics = False

    # =========================================================================
    # DOMAIN CORPUS AUGMENTATION
    # =========================================================================

    async def _search_domain_corpus(
        self,
        query: str,
        entities: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Search domain corpus for relevant knowledge.

        This augments web search with local domain-specific knowledge
        from corpuses like FANUC robotics, Raspberry Pi, etc.
        """
        if not self._domain_corpus_manager or not self.enable_domain_corpus:
            return None

        try:
            # Search across all registered corpuses using cross_domain_query
            results = await self._domain_corpus_manager.cross_domain_query(
                query=query,
                domain_ids=None  # Search all domains
            )

            if results and results.get("results"):
                return {
                    "corpus_results": results["results"],
                    "domains_searched": results.get("domains_queried", []),
                    "corpus_count": len(results["results"])
                }
            return None
        except Exception as e:
            logger.warning(f"Domain corpus search failed: {e}")
            return None

    # =========================================================================
    # MAIN SEARCH PIPELINE
    # =========================================================================

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute enhanced search with all integrated features.

        Pipeline:
        1. Query classification (existing)
        2. Entity extraction (NEW)
        3. HyDE query expansion (NEW)
        4. Web search (existing)
        5. Hybrid re-ranking (NEW)
        6. CRAG evaluation (existing)
        7. Synthesis (existing)
        8. Self-RAG reflection (existing)
        9. RAGAS evaluation (NEW)
        10. Experience distillation (existing)

        Returns:
            SearchResponse with enhanced metadata
        """
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        enhancement_metadata = {}

        logger.info(f"[{request_id}] UnifiedOrchestrator processing: {request.query[:50]}...")

        try:
            # Phase 1: Entity Extraction (optional)
            entities = []
            if self.enable_entity_tracking and self._entity_tracker:
                entity_start = time.time()
                try:
                    entity_state = await self._entity_tracker.extract_entities(request.query)
                    entities = entity_state.entities if entity_state else []
                    enhancement_metadata["entities_extracted"] = len(entities)
                    self.feature_timings["entity_tracking"].append(time.time() - entity_start)
                    logger.debug(f"[{request_id}] Extracted {len(entities)} entities")
                except Exception as e:
                    logger.warning(f"[{request_id}] Entity extraction failed: {e}")

            # Phase 2: HyDE Query Expansion (optional)
            hyde_result: Optional[HyDEResult] = None
            expanded_queries: List[str] = []
            if self.enable_hyde and self._hyde_expander:
                hyde_start = time.time()
                try:
                    hyde_result = await self._hyde_expander.expand(
                        query=request.query,
                        mode=HyDEMode.SINGLE
                    )
                    if hyde_result and hyde_result.hypothetical_documents:
                        # Extract additional search terms from hypothetical
                        expanded_queries = self._extract_queries_from_hyde(
                            request.query,
                            hyde_result.hypothetical_documents[0]
                        )
                        enhancement_metadata["hyde_expansion"] = True
                        enhancement_metadata["expanded_query_count"] = len(expanded_queries)
                    self.feature_timings["hyde"].append(time.time() - hyde_start)
                    logger.debug(f"[{request_id}] HyDE generated {len(expanded_queries)} expanded queries")
                except Exception as e:
                    logger.warning(f"[{request_id}] HyDE expansion failed: {e}")

            # Phase 3: Execute Base Search (with optional expanded queries)
            # Inject expanded queries into the request if available
            enhanced_request = request
            if expanded_queries:
                # Create a modified request with expanded context
                enhanced_request = SearchRequest(
                    query=request.query,
                    user_id=request.user_id,
                    context={
                        **(request.context or {}),
                        "expanded_queries": expanded_queries,
                        "entities": [str(e) for e in entities[:5]]  # Top 5 entities
                    },
                    max_iterations=request.max_iterations,
                    min_sources=request.min_sources,
                    max_sources=request.max_sources,
                    verification_level=request.verification_level,
                    search_mode=request.search_mode
                )

            # Run base orchestrator search
            base_response = await self.base_orchestrator.search(enhanced_request)

            # Phase 4: Hybrid Re-ranking (optional)
            if (self.enable_hybrid_reranking and
                self._hybrid_retriever and
                base_response.success and
                base_response.data and
                base_response.data.sources):

                hybrid_start = time.time()
                try:
                    reranked_sources = await self._apply_hybrid_reranking(
                        request.query,
                        base_response.data.sources,
                        hyde_result
                    )
                    if reranked_sources:
                        base_response.data.sources = reranked_sources
                        enhancement_metadata["hybrid_reranked"] = True
                    self.feature_timings["hybrid_rerank"].append(time.time() - hybrid_start)
                    logger.debug(f"[{request_id}] Hybrid re-ranking complete")
                except Exception as e:
                    logger.warning(f"[{request_id}] Hybrid re-ranking failed: {e}")

            # Phase 5: RAGAS Evaluation (optional)
            if (self.enable_ragas and
                self._ragas_evaluator and
                base_response.success and
                base_response.data and
                base_response.data.synthesized_context):

                ragas_start = time.time()
                try:
                    # Sources are dicts, not objects - use dict access
                    contexts = [
                        s.get("snippet", s.get("title", ""))
                        for s in (base_response.data.sources or [])[:5]
                        if isinstance(s, dict)
                    ]
                    ragas_result = await self._ragas_evaluator.evaluate(
                        question=request.query,
                        answer=base_response.data.synthesized_context[:2000],
                        contexts=contexts
                    )
                    if ragas_result:
                        enhancement_metadata["ragas_faithfulness"] = ragas_result.faithfulness
                        enhancement_metadata["ragas_relevancy"] = ragas_result.answer_relevancy
                        enhancement_metadata["ragas_overall"] = ragas_result.overall_score

                        # Blend RAGAS score with existing confidence
                        if base_response.data.confidence_score:
                            blended = (
                                0.6 * base_response.data.confidence_score +
                                0.4 * ragas_result.overall_score
                            )
                            base_response.data.confidence_score = blended
                            enhancement_metadata["confidence_blended"] = True

                    self.feature_timings["ragas"].append(time.time() - ragas_start)
                    logger.debug(f"[{request_id}] RAGAS evaluation complete: {ragas_result.overall_score:.2f}")
                except Exception as e:
                    logger.warning(f"[{request_id}] RAGAS evaluation failed: {e}")

            # Phase 6: Thought Library Learning (optional)
            if (self.enable_thought_library and
                self._thought_library and
                base_response.success and
                base_response.data and
                base_response.data.confidence_score and
                base_response.data.confidence_score >= 0.8):

                try:
                    # Store successful pattern for future use
                    self._thought_library.record_success(
                        category=TemplateCategory.SYNTHESIS,
                        query=request.query,
                        confidence=base_response.data.confidence_score
                    )
                except Exception as e:
                    logger.debug(f"[{request_id}] Thought library update failed: {e}")

            # Add enhancement metadata to response
            total_time = time.time() - start_time
            if base_response.meta:
                # Preserve existing meta fields and add enhancement metadata
                base_response.meta = SearchMeta(
                    execution_time_ms=int(total_time * 1000),
                    iterations=base_response.meta.iterations,
                    queries_executed=base_response.meta.queries_executed,
                    sources_consulted=base_response.meta.sources_consulted,
                    cache_hit=base_response.meta.cache_hit,
                    semantic_match=base_response.meta.semantic_match,
                    matched_query=base_response.meta.matched_query,
                    similarity_score=base_response.meta.similarity_score,
                    enhancement_metadata=enhancement_metadata if enhancement_metadata else None
                )

            return base_response

        except Exception as e:
            logger.error(f"[{request_id}] UnifiedOrchestrator error: {e}", exc_info=True)
            return SearchResponse(
                success=False,
                data=None,
                meta=SearchMeta(
                    execution_time_ms=int((time.time() - start_time) * 1000)
                ),
                errors=[str(e)]
            )

    def _extract_queries_from_hyde(
        self,
        original_query: str,
        hypothetical: str
    ) -> List[str]:
        """Extract additional search queries from HyDE hypothetical document."""
        # Extract key phrases from hypothetical that aren't in original query
        original_words = set(original_query.lower().split())
        hypothetical_words = hypothetical.lower().split()

        # Find unique phrases (2-4 word chunks)
        new_phrases = []
        for i in range(len(hypothetical_words) - 2):
            phrase = " ".join(hypothetical_words[i:i+3])
            phrase_words = set(phrase.split())

            # Check if phrase adds new information
            if len(phrase_words - original_words) >= 2:
                new_phrases.append(phrase)

        # Deduplicate and limit
        unique_phrases = list(set(new_phrases))[:3]

        return unique_phrases

    async def _apply_hybrid_reranking(
        self,
        query: str,
        sources: List[Dict[str, Any]],
        hyde_result: Optional[HyDEResult] = None
    ) -> List[Dict[str, Any]]:
        """Re-rank sources using BGE-M3 hybrid retrieval.

        Note: sources are dicts with keys like 'url', 'title', 'snippet', 'relevance_score'
        """
        if not sources or not self._hybrid_retriever:
            return sources

        # Clear temporary index
        self._hybrid_retriever.documents.clear()

        # Index sources - sources are dicts, not objects
        for source in sources:
            if not isinstance(source, dict):
                continue
            url = source.get("url", "")
            if not url:
                continue
            doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
            title = source.get("title", "")
            snippet = source.get("snippet", "")
            content = f"{title} {snippet}"
            await self._hybrid_retriever.add_document(
                doc_id=doc_id,
                content=content,
                metadata={"url": url, "domain": source.get("source_domain", "")}
            )

        # Use HyDE embedding if available, otherwise query embedding
        search_query = query
        if hyde_result and hyde_result.hypothetical_documents:
            # Combine query with hypothetical for better matching
            search_query = f"{query} {hyde_result.hypothetical_documents[0][:200]}"

        # Execute hybrid search
        results = await self._hybrid_retriever.search(
            query=search_query,
            top_k=len(sources),
            mode=RetrievalMode.HYBRID
        )

        # Create URL to new score mapping
        # HybridSearchResult uses combined_score, not score
        url_to_score = {}
        for r in results:
            if r.metadata and "url" in r.metadata:
                url_to_score[r.metadata["url"]] = r.combined_score

        # Update source scores - sources are dicts
        for source in sources:
            if not isinstance(source, dict):
                continue
            url = source.get("url", "")
            if url in url_to_score:
                # Blend original score with hybrid score
                hybrid_score = url_to_score[url]
                original_score = source.get("relevance_score", 0.5)
                source["relevance_score"] = (
                    0.4 * original_score +
                    0.6 * hybrid_score
                )

        # Re-sort by updated scores
        sources.sort(key=lambda s: s.get("relevance_score", 0), reverse=True)

        return sources

    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get unified orchestrator statistics."""
        stats = {
            "base_orchestrator": "active",
            "features_enabled": {
                "hyde": self.enable_hyde,
                "hybrid_reranking": self.enable_hybrid_reranking,
                "ragas": self.enable_ragas,
                "entity_tracking": self.enable_entity_tracking,
                "thought_library": self.enable_thought_library,
                "domain_corpus": self.enable_domain_corpus,
                "reasoning_dag": self.enable_reasoning_dag,
                "metrics": self.enable_metrics
            },
            "feature_timings": {}
        }

        # Calculate average timings
        for feature, timings in self.feature_timings.items():
            if timings:
                stats["feature_timings"][feature] = {
                    "count": len(timings),
                    "avg_ms": sum(timings) / len(timings) * 1000,
                    "max_ms": max(timings) * 1000,
                    "min_ms": min(timings) * 1000
                }

        # Add component stats
        if self._thought_library:
            stats["thought_library"] = {
                "templates": len(self._thought_library.templates),
                "categories": list(set(t.category.value for t in self._thought_library.templates.values()))
            }

        if self._hybrid_retriever:
            try:
                hybrid_stats = self._hybrid_retriever.get_stats()
                stats["hybrid_retriever"] = {
                    "documents": hybrid_stats.documents_indexed,
                    "vocabulary_size": hybrid_stats.vocabulary_size
                }
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not get hybrid retriever stats: {e}")

        # Domain corpus stats
        if self._domain_corpus_manager:
            try:
                stats["domain_corpus"] = {
                    "corpuses": list(self._domain_corpus_manager.corpuses.keys()),
                    "corpus_count": len(self._domain_corpus_manager.corpuses)
                }
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not get domain corpus stats: {e}")

        # Metrics stats
        if self._metrics:
            try:
                stats["performance_metrics"] = self._metrics.get_summary()
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not get performance metrics: {e}")

        return stats


# Singleton instance
_unified_orchestrator: Optional[UnifiedOrchestrator] = None


async def get_unified_orchestrator(
    ollama_url: str = "http://localhost:11434",
    **kwargs
) -> UnifiedOrchestrator:
    """Get or create the unified orchestrator instance."""
    global _unified_orchestrator
    if _unified_orchestrator is None:
        _unified_orchestrator = UnifiedOrchestrator(ollama_url=ollama_url, **kwargs)
        await _unified_orchestrator.initialize()
    return _unified_orchestrator


async def create_unified_orchestrator(
    ollama_url: str = "http://localhost:11434",
    **kwargs
) -> UnifiedOrchestrator:
    """Create a new unified orchestrator instance."""
    orchestrator = UnifiedOrchestrator(ollama_url=ollama_url, **kwargs)
    await orchestrator.initialize()
    return orchestrator
