"""
Event System for Agentic Search Progress Notifications

Provides real-time progress updates to clients via Server-Sent Events (SSE).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, Set, Callable, Awaitable, List
from dataclasses import dataclass, field, asdict
import uuid

logger = logging.getLogger("agentic.events")


class EventType(str, Enum):
    """Types of events that can be emitted during agentic search"""

    # Search lifecycle
    SEARCH_STARTED = "search_started"
    SEARCH_COMPLETED = "search_completed"
    SEARCH_FAILED = "search_failed"

    # Analysis phase
    ANALYZING_QUERY = "analyzing_query"
    QUERY_ANALYZED = "query_analyzed"

    # Planning phase
    PLANNING_SEARCH = "planning_search"
    SEARCH_PLANNED = "search_planned"

    # Search execution
    SEARCHING = "searching"
    SEARCH_RESULTS = "search_results"

    # URL evaluation
    EVALUATING_URLS = "evaluating_urls"
    URLS_EVALUATED = "urls_evaluated"

    # Content scraping
    SCRAPING_URL = "scraping_url"
    URL_SCRAPED = "url_scraped"

    # Vision analysis
    ANALYZING_IMAGE = "analyzing_image"
    IMAGE_ANALYZED = "image_analyzed"

    # VL (Vision-Language) Scraping - for JS-heavy pages
    VL_SCRAPING_START = "vl_scraping_start"
    VL_SCRAPING_SCREENSHOT = "vl_scraping_screenshot"
    VL_SCRAPING_EXTRACTING = "vl_scraping_extracting"
    VL_SCRAPING_COMPLETE = "vl_scraping_complete"
    VL_SCRAPING_FAILED = "vl_scraping_failed"

    # Verification
    VERIFYING_CLAIMS = "verifying_claims"
    CLAIMS_VERIFIED = "claims_verified"

    # Synthesis
    SYNTHESIZING = "synthesizing"
    SYNTHESIS_COMPLETE = "synthesis_complete"

    # Model selection
    MODEL_SELECTED = "model_selected"
    MODEL_LOADING = "model_loading"

    # Iteration updates
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"

    # Progress percentage
    PROGRESS_UPDATE = "progress_update"

    # AIME-style dynamic planning events
    PLANNING = "planning"  # Task decomposition/planning phase
    AGENT_START = "agent_start"  # Agent starting execution
    AGENT_COMPLETE = "agent_complete"  # Agent finished execution
    PROGRESS = "progress"  # Task progress update
    COMPLETE = "complete"  # Overall completion
    ERROR = "error"  # Error occurred

    # ========== NEW: Comprehensive Agent Processing Events ==========

    # Query Classification (DeepSeek-R1)
    CLASSIFYING_QUERY = "classifying_query"
    QUERY_CLASSIFIED = "query_classified"

    # CRAG Pre-Retrieval Evaluation
    CRAG_EVALUATING = "crag_evaluating"
    CRAG_EVALUATION_COMPLETE = "crag_evaluation_complete"
    CRAG_REFINING = "crag_refining"  # When corrective action is taken

    # Self-RAG Post-Synthesis Reflection
    SELF_RAG_REFLECTING = "self_rag_reflecting"
    SELF_RAG_COMPLETE = "self_rag_complete"
    SELF_RAG_REFINING = "self_rag_refining"  # When synthesis needs refinement

    # Scratchpad/Blackboard Updates
    SCRATCHPAD_INITIALIZED = "scratchpad_initialized"
    SCRATCHPAD_UPDATED = "scratchpad_updated"
    SCRATCHPAD_FINDING_ADDED = "scratchpad_finding_added"
    SCRATCHPAD_QUESTION_ANSWERED = "scratchpad_question_answered"
    SCRATCHPAD_GAP_DETECTED = "scratchpad_gap_detected"

    # Thought Template (Buffer of Thoughts)
    THOUGHT_TEMPLATE_MATCHED = "thought_template_matched"
    THOUGHT_TEMPLATE_APPLIED = "thought_template_applied"

    # Reasoning Composer (Self-Discover)
    REASONING_STRATEGY_COMPOSED = "reasoning_strategy_composed"

    # Experience Distillation
    EXPERIENCE_CAPTURED = "experience_captured"
    EXPERIENCE_DISTILLING = "experience_distilling"
    TEMPLATE_CREATED = "template_created"

    # Classifier Feedback
    OUTCOME_RECORDED = "outcome_recorded"
    ADAPTIVE_HINT_APPLIED = "adaptive_hint_applied"

    # Coverage Assessment
    COVERAGE_EVALUATING = "coverage_evaluating"
    COVERAGE_EVALUATED = "coverage_evaluated"
    COVERAGE_INSUFFICIENT = "coverage_insufficient"

    # Refinement Cycles
    REFINEMENT_CYCLE_START = "refinement_cycle_start"
    REFINEMENT_CYCLE_COMPLETE = "refinement_cycle_complete"
    REFINEMENT_QUERIES_GENERATED = "refinement_queries_generated"

    # Adaptive Refinement (Phase 2)
    ADAPTIVE_REFINEMENT_START = "adaptive_refinement_start"
    ADAPTIVE_REFINEMENT_DECISION = "adaptive_refinement_decision"
    GAPS_IDENTIFIED = "gaps_identified"
    ANSWER_GRADED = "answer_graded"
    QUERY_DECOMPOSED = "query_decomposed"
    ADAPTIVE_REFINEMENT_COMPLETE = "adaptive_refinement_complete"

    # Reasoning DAG (Graph of Thoughts)
    REASONING_BRANCH_CREATED = "reasoning_branch_created"
    REASONING_NODE_VERIFIED = "reasoning_node_verified"
    REASONING_PATHS_MERGED = "reasoning_paths_merged"

    # Entity Tracking (GSW)
    ENTITIES_EXTRACTED = "entities_extracted"
    ENTITY_RELATION_FOUND = "entity_relation_found"

    # LLM Calls (for debugging)
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_COMPLETE = "llm_call_complete"

    # Quality Assessment
    QUALITY_CHECK_START = "quality_check_start"
    QUALITY_CHECK_COMPLETE = "quality_check_complete"
    CORPUS_QUALITY_ASSESSED = "corpus_quality_assessed"

    # Web Search Specific
    WEB_SEARCH_START = "web_search_start"
    WEB_SEARCH_COMPLETE = "web_search_complete"
    WEB_SEARCH_FALLBACK = "web_search_fallback"  # When switching to fallback API

    # Decision Points (for debugging)
    DECISION_POINT = "decision_point"  # Generic decision logging
    PIPELINE_ROUTED = "pipeline_routed"  # Which pipeline was chosen
    CONTEXT_TRANSFER = "context_transfer"  # Context flow between pipeline stages

    # ========== P2 Observability Events ==========
    # Feature status tracking
    FEATURE_STATUS = "feature_status"  # Feature enabled/skipped with reason
    # Confidence breakdown
    CONFIDENCE_CALCULATED = "confidence_calculated"  # Multi-signal confidence with weights
    # Technician log
    TECHNICIAN_LOG_READY = "technician_log_ready"  # Human-readable diagnostic ready

    # ========== NEW: BGE-M3 Hybrid Retrieval Events ==========
    HYBRID_SEARCH_START = "hybrid_search_start"
    HYBRID_SEARCH_COMPLETE = "hybrid_search_complete"
    HYBRID_INDEXING = "hybrid_indexing"
    HYBRID_INDEX_COMPLETE = "hybrid_index_complete"
    BM25_SEARCH = "bm25_search"
    DENSE_EMBEDDING = "dense_embedding"
    RRF_FUSION = "rrf_fusion"

    # ========== NEW: HyDE Query Expansion Events ==========
    HYDE_GENERATING = "hyde_generating"
    HYDE_HYPOTHETICAL_GENERATED = "hyde_hypothetical_generated"
    HYDE_EMBEDDING = "hyde_embedding"
    HYDE_FUSING = "hyde_fusing"
    HYDE_COMPLETE = "hyde_complete"

    # ========== NEW: RAGAS Evaluation Events ==========
    RAGAS_EVALUATING = "ragas_evaluating"
    RAGAS_CLAIMS_EXTRACTING = "ragas_claims_extracting"
    RAGAS_CLAIMS_EXTRACTED = "ragas_claims_extracted"
    RAGAS_CLAIM_VERIFYING = "ragas_claim_verifying"
    RAGAS_CLAIM_VERIFIED = "ragas_claim_verified"
    RAGAS_EVALUATION_COMPLETE = "ragas_evaluation_complete"

    # ========== NEW: Agent Graph Traversal Visualization ==========
    GRAPH_STATE_UPDATE = "graph_state_update"  # Main event for graph visualization
    GRAPH_NODE_ENTERED = "graph_node_entered"
    GRAPH_NODE_COMPLETED = "graph_node_completed"
    GRAPH_EDGE_TRAVERSED = "graph_edge_traversed"
    GRAPH_BRANCH_CREATED = "graph_branch_created"
    GRAPH_PATHS_MERGED = "graph_paths_merged"

    # ========== NEW: Technical Documentation (PDF API) Events ==========
    TECHNICAL_DOCS_SEARCHING = "technical_docs_searching"
    TECHNICAL_DOCS_FOUND = "technical_docs_found"
    TECHNICAL_DOCS_NOT_AVAILABLE = "technical_docs_not_available"
    TROUBLESHOOT_PATH_QUERYING = "troubleshoot_path_querying"
    TROUBLESHOOT_PATH_FOUND = "troubleshoot_path_found"
    PDF_API_HEALTH_CHECK = "pdf_api_health_check"
    CORPUS_SYNC_START = "corpus_sync_start"
    CORPUS_SYNC_PROGRESS = "corpus_sync_progress"
    CORPUS_SYNC_COMPLETE = "corpus_sync_complete"
    ENTITY_ENRICHMENT_START = "entity_enrichment_start"
    ENTITY_ENRICHMENT_COMPLETE = "entity_enrichment_complete"
    FANUC_PATTERN_DETECTED = "fanuc_pattern_detected"

    # ========== G.6 Agent Coordination Events ==========

    # G.6.1: A-MEM Semantic Memory
    SEMANTIC_MEMORY_STORING = "semantic_memory_storing"
    SEMANTIC_MEMORY_STORED = "semantic_memory_stored"
    SEMANTIC_MEMORY_RETRIEVED = "semantic_memory_retrieved"
    MEMORY_CONNECTION_CREATED = "memory_connection_created"

    # G.6.2: DyLAN Agent Importance Scores
    DYLAN_COMPLEXITY_CLASSIFIED = "dylan_complexity_classified"
    DYLAN_AGENT_SKIPPED = "dylan_agent_skipped"
    DYLAN_CONTRIBUTION_RECORDED = "dylan_contribution_recorded"

    # G.6.4: Information Bottleneck Filtering
    IB_FILTERING_START = "ib_filtering_start"
    IB_FILTERING_COMPLETE = "ib_filtering_complete"

    # G.6.5: Contrastive Retriever Training
    CONTRASTIVE_SESSION_RECORDED = "contrastive_session_recorded"
    CONTRASTIVE_INSIGHT_GENERATED = "contrastive_insight_generated"


@dataclass
class SearchEvent:
    """An event emitted during agentic search"""

    event_type: EventType
    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional event data
    message: Optional[str] = None
    progress_percent: Optional[int] = None
    iteration: Optional[int] = None
    max_iterations: Optional[int] = None

    # Search-specific data
    query: Optional[str] = None
    queries: Optional[list] = None
    results_count: Optional[int] = None
    sources_count: Optional[int] = None
    confidence: Optional[float] = None  # Confidence score (0.0-1.0) for search_completed
    engines: Optional[list] = None  # Search engines used (e.g., ["SearXNG", "DuckDuckGo", "Brave"])

    # Model-specific data
    model_name: Optional[str] = None

    # URL-specific data
    url: Optional[str] = None
    url_index: Optional[int] = None
    url_total: Optional[int] = None
    content_type: Optional[str] = None
    content_length: Optional[int] = None

    # Additional metadata
    data: Optional[Dict[str, Any]] = None

    # Enhanced search marker
    enhanced: bool = False

    # Graph visualization (one-line representation)
    graph_line: Optional[str] = None  # ASCII graph like: "●─○─○─◎" or "[A]→[S]→[V]→[Σ]"

    def to_sse(self) -> str:
        """Convert to Server-Sent Event format"""
        event_data = {
            "event": self.event_type.value,
            "request_id": self.request_id,
            "timestamp": self.timestamp
        }

        # Add non-None fields
        if self.message:
            event_data["message"] = self.message
        if self.progress_percent is not None:
            event_data["progress"] = self.progress_percent
        if self.iteration is not None:
            event_data["iteration"] = self.iteration
        if self.max_iterations is not None:
            event_data["max_iterations"] = self.max_iterations
        if self.query:
            event_data["query"] = self.query
        if self.queries:
            event_data["queries"] = self.queries
        if self.results_count is not None:
            event_data["results_count"] = self.results_count
        if self.sources_count is not None:
            event_data["sources_count"] = self.sources_count
        if self.confidence is not None:
            event_data["confidence"] = self.confidence
        if self.engines:
            event_data["engines"] = self.engines
        if self.model_name:
            event_data["model"] = self.model_name
        if self.url:
            event_data["url"] = self.url
        if self.url_index is not None:
            event_data["url_index"] = self.url_index
        if self.url_total is not None:
            event_data["url_total"] = self.url_total
        if self.content_type:
            event_data["content_type"] = self.content_type
        if self.content_length is not None:
            event_data["content_length"] = self.content_length
        if self.data:
            event_data["data"] = self.data
        if self.enhanced:
            event_data["enhanced"] = self.enhanced
        if self.graph_line:
            event_data["graph_line"] = self.graph_line

        return f"event: {self.event_type.value}\ndata: {json.dumps(event_data)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {"event": self.event_type.value, "request_id": self.request_id, "timestamp": self.timestamp}

        for key, value in asdict(self).items():
            if value is not None and key not in ["event_type", "request_id", "timestamp"]:
                result[key] = value

        return result


class EventEmitter:
    """
    Manages event subscriptions and broadcasts for a search request.

    Each search request gets its own EventEmitter instance that
    clients can subscribe to for real-time updates.
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self._subscribers: Set[asyncio.Queue] = set()
        self._event_history: list = []
        self._max_history = 100
        self._closed = False

    async def emit(self, event: Optional[SearchEvent]):
        """Emit an event to all subscribers. Pass None to signal end of stream."""
        if self._closed:
            return

        # Handle None as end-of-stream signal
        if event is None:
            logger.debug(f"[{self.request_id}] Emitting: END_OF_STREAM (None)")
        else:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            logger.debug(f"[{self.request_id}] Emitting: {event.event_type.value}")

        # Broadcast to all subscribers
        dead_queues = []
        for queue in self._subscribers:
            try:
                await asyncio.wait_for(queue.put(event), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(f"[{self.request_id}] Subscriber queue full, skipping")
            except Exception as e:
                logger.warning(f"[{self.request_id}] Failed to send to subscriber: {e}")
                dead_queues.append(queue)

        # Remove dead subscribers
        for queue in dead_queues:
            self._subscribers.discard(queue)

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to events, returns a queue to read from"""
        queue = asyncio.Queue(maxsize=50)
        self._subscribers.add(queue)
        logger.debug(f"[{self.request_id}] New subscriber, total: {len(self._subscribers)}")
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from events"""
        self._subscribers.discard(queue)
        logger.debug(f"[{self.request_id}] Subscriber removed, remaining: {len(self._subscribers)}")

    def close(self):
        """Close the emitter and notify all subscribers"""
        self._closed = True
        for queue in self._subscribers:
            try:
                queue.put_nowait(None)  # Signal end of stream
            except (asyncio.QueueFull, RuntimeError):
                pass  # Queue full or closed, safe to ignore during shutdown
        self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def get_history(self) -> list:
        """Get event history"""
        return self._event_history.copy()


class EventManager:
    """
    Global manager for all active search events.

    Tracks active searches and their event emitters,
    allowing clients to subscribe to specific searches.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._emitters: Dict[str, EventEmitter] = {}
            cls._instance._cleanup_task = None
        return cls._instance

    def create_emitter(self, request_id: Optional[str] = None) -> EventEmitter:
        """Create a new event emitter for a search request"""
        if request_id is None:
            request_id = str(uuid.uuid4())

        emitter = EventEmitter(request_id)
        self._emitters[request_id] = emitter

        logger.info(f"Created event emitter for request {request_id}")
        return emitter

    def get_emitter(self, request_id: str) -> Optional[EventEmitter]:
        """Get an existing event emitter"""
        return self._emitters.get(request_id)

    def remove_emitter(self, request_id: str):
        """Remove an event emitter"""
        if request_id in self._emitters:
            self._emitters[request_id].close()
            del self._emitters[request_id]
            logger.debug(f"Removed event emitter for request {request_id}")

    @property
    def active_searches(self) -> int:
        return len(self._emitters)

    def get_active_request_ids(self) -> list:
        return list(self._emitters.keys())

    async def cleanup_stale_emitters(self, max_age_seconds: int = 3600):
        """Remove emitters that have been around too long"""
        # Implementation would track creation time and remove old ones
        pass


# Global event manager instance
event_manager = EventManager()


def get_event_manager() -> EventManager:
    """Get the global event manager"""
    return event_manager


# Helper functions for creating common events
def search_started(request_id: str, query: str, max_iterations: int, enhanced: bool = False, **kwargs) -> SearchEvent:
    mode = "ENHANCED search" if enhanced else "search"
    return SearchEvent(
        event_type=EventType.SEARCH_STARTED,
        request_id=request_id,
        message=f"Starting {mode} for: {query[:50]}...",
        query=query,
        max_iterations=max_iterations,
        progress_percent=0,
        enhanced=enhanced
    )


def analyzing_query(request_id: str, query: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.ANALYZING_QUERY,
        request_id=request_id,
        message="Analyzing query to determine search strategy...",
        query=query,
        progress_percent=5
    )


def query_analyzed(request_id: str, requires_search: bool, query_type: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.QUERY_ANALYZED,
        request_id=request_id,
        message=f"Query analyzed: {'search needed' if requires_search else 'no search needed'}",
        progress_percent=10,
        data={"requires_search": requires_search, "query_type": query_type}
    )


def planning_search(request_id: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.PLANNING_SEARCH,
        request_id=request_id,
        message="Creating search plan...",
        progress_percent=15
    )


def search_planned(request_id: str, queries: list, phases: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SEARCH_PLANNED,
        request_id=request_id,
        message=f"Search plan ready: {len(queries)} queries in {phases} phases",
        queries=queries,
        progress_percent=20,
        data={"phases": phases}
    )


def searching(
    request_id: str,
    queries: list,
    iteration: int,
    max_iterations: int,
    engines: Optional[list] = None
) -> SearchEvent:
    """
    Create a searching event.

    Args:
        request_id: Request identifier
        queries: List of search queries
        iteration: Current iteration
        max_iterations: Maximum iterations
        engines: List of search engines being used (e.g., ["SearXNG", "DuckDuckGo", "Brave"])
    """
    progress = 20 + int((iteration / max_iterations) * 40)  # 20-60%
    # Format message with engine info if provided
    engine_str = f" via {', '.join(engines)}" if engines else ""
    return SearchEvent(
        event_type=EventType.SEARCHING,
        request_id=request_id,
        message=f"Searching{engine_str}: {', '.join(queries[:2])}{'...' if len(queries) > 2 else ''}",
        queries=queries,
        iteration=iteration,
        max_iterations=max_iterations,
        progress_percent=progress,
        engines=engines
    )


def search_results(request_id: str, results_count: int, sources_count: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SEARCH_RESULTS,
        request_id=request_id,
        message=f"Found {results_count} results from {sources_count} sources",
        results_count=results_count,
        sources_count=sources_count
    )


def evaluating_urls(request_id: str, total_urls: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.EVALUATING_URLS,
        request_id=request_id,
        message=f"Evaluating {total_urls} URLs for relevance...",
        results_count=total_urls,
        progress_percent=55
    )


def urls_evaluated(request_id: str, relevant_count: int, total_count: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.URLS_EVALUATED,
        request_id=request_id,
        message=f"Selected {relevant_count}/{total_count} relevant sources to scrape",
        results_count=relevant_count,
        sources_count=total_count,
        progress_percent=58,
        data={"relevant": relevant_count, "total": total_count}
    )


def scraping_url(request_id: str, url: str, index: int, total: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SCRAPING_URL,
        request_id=request_id,
        message=f"Scraping source {index}/{total}...",
        url=url,
        url_index=index,
        url_total=total,
        progress_percent=60 + int((index / total) * 10)  # 60-70%
    )


def url_scraped(request_id: str, url: str, content_length: int, content_type: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.URL_SCRAPED,
        request_id=request_id,
        message=f"Scraped {content_length:,} characters",
        url=url,
        content_length=content_length,
        content_type=content_type
    )


def analyzing_image(request_id: str, image_index: int, total_images: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.ANALYZING_IMAGE,
        request_id=request_id,
        message=f"Analyzing image {image_index}/{total_images} with vision model...",
        progress_percent=70 + int((image_index / total_images) * 5)  # 70-75%
    )


# VL (Vision-Language) Scraping Events - for JS-heavy pages
def vl_scraping_start(request_id: str, url: str, reason: str) -> SearchEvent:
    """
    Emitted when VL scraper is invoked for a JS-heavy page.

    Args:
        request_id: Request identifier
        url: URL being scraped
        reason: Why VL scraper was chosen (e.g., "js_heavy_domain", "minimal_content")
    """
    return SearchEvent(
        event_type=EventType.VL_SCRAPING_START,
        request_id=request_id,
        message=f"Starting VL scrape ({reason}): {url[:60]}...",
        url=url,
        progress_percent=62,
        data={"reason": reason}
    )


def vl_scraping_screenshot(request_id: str, url: str, screenshot_count: int) -> SearchEvent:
    """
    Emitted when screenshot capture is in progress.

    Args:
        request_id: Request identifier
        url: URL being captured
        screenshot_count: Number of screenshots taken (for scroll-and-capture)
    """
    return SearchEvent(
        event_type=EventType.VL_SCRAPING_SCREENSHOT,
        request_id=request_id,
        message=f"Captured {screenshot_count} screenshot(s)",
        url=url,
        progress_percent=64,
        data={"screenshot_count": screenshot_count}
    )


def vl_scraping_extracting(request_id: str, url: str, model_name: str) -> SearchEvent:
    """
    Emitted when VL model is extracting content from screenshots.

    Args:
        request_id: Request identifier
        url: URL being processed
        model_name: VL model being used (e.g., "qwen3-vl", "llama3.2-vision")
    """
    return SearchEvent(
        event_type=EventType.VL_SCRAPING_EXTRACTING,
        request_id=request_id,
        message=f"Extracting with {model_name}...",
        url=url,
        model_name=model_name,
        progress_percent=66
    )


def vl_scraping_complete(
    request_id: str,
    url: str,
    content_length: int,
    model_used: str,
    extraction_type: str
) -> SearchEvent:
    """
    Emitted when VL scraping completes successfully.

    Args:
        request_id: Request identifier
        url: URL that was scraped
        content_length: Length of extracted content in characters
        model_used: VL model that performed extraction
        extraction_type: Type of extraction (e.g., "GENERAL_INFO", "CONTACT_INFO")
    """
    return SearchEvent(
        event_type=EventType.VL_SCRAPING_COMPLETE,
        request_id=request_id,
        message=f"VL extracted {content_length:,} chars",
        url=url,
        content_length=content_length,
        model_name=model_used,
        progress_percent=68,
        data={"extraction_type": extraction_type}
    )


def vl_scraping_failed(request_id: str, url: str, error: str) -> SearchEvent:
    """
    Emitted when VL scraping fails.

    Args:
        request_id: Request identifier
        url: URL that failed
        error: Error message
    """
    return SearchEvent(
        event_type=EventType.VL_SCRAPING_FAILED,
        request_id=request_id,
        message=f"VL scrape failed: {error[:80]}",
        url=url,
        progress_percent=68,
        data={"error": error}
    )


def verifying_claims(request_id: str, claims_count: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.VERIFYING_CLAIMS,
        request_id=request_id,
        message=f"Verifying {claims_count} claims...",
        progress_percent=75,
        data={"total": claims_count}  # Required for Android Tool Notification display
    )


def claims_verified(request_id: str, verified_count: int, total_count: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.CLAIMS_VERIFIED,
        request_id=request_id,
        message=f"Verified {verified_count}/{total_count} claims",
        progress_percent=80,
        data={"verified": verified_count, "total": total_count}
    )


def model_selected(request_id: str, model_name: str, task: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.MODEL_SELECTED,
        request_id=request_id,
        message=f"Selected model: {model_name} for {task}",
        model_name=model_name,
        data={"task": task}
    )


def synthesizing(request_id: str, sources_count: int, model: str = "") -> SearchEvent:
    """
    Synthesizing answer event.

    Args:
        request_id: Unique request identifier
        sources_count: Number of sources being synthesized
        model: Model name being used for synthesis (e.g., "qwen3:8b", "deepseek-r1:14b")
    """
    message = f"Synthesizing answer from {sources_count} sources..."
    if model:
        message = f"Synthesizing answer from {sources_count} sources using {model}..."

    return SearchEvent(
        event_type=EventType.SYNTHESIZING,
        request_id=request_id,
        message=message,
        sources_count=sources_count,
        progress_percent=85,
        model_name=model if model else None,
        data={"model": model} if model else {}
    )


def synthesis_complete(request_id: str, answer_length: int, confidence: float) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SYNTHESIS_COMPLETE,
        request_id=request_id,
        message="Answer synthesized",
        progress_percent=95,
        data={"answer_length": answer_length, "confidence": confidence}
    )


def search_completed(
    request_id: str,
    sources_count: int = 0,
    execution_time_ms: int = 0,
    response = None,
    enhanced: bool = False,
    confidence: float = None,
    **kwargs
) -> SearchEvent:
    # Extract from response if provided
    if response:
        sources_count = len(response.data.sources) if response.data and response.data.sources else sources_count
        execution_time_ms = response.meta.execution_time_ms if response.meta else execution_time_ms
        # Extract confidence from response if not explicitly provided
        if confidence is None and response.data and hasattr(response.data, 'confidence'):
            confidence = response.data.confidence

    mode = "ENHANCED search" if enhanced else "Search"

    # Build data dict - MUST include full response for Android client parsing
    event_data = {
        "execution_time_ms": execution_time_ms,
        "enhanced": enhanced
    }

    # Include confidence in data for backwards compatibility
    if confidence is not None:
        event_data["confidence_score"] = confidence

    # Include full response object if provided (Android client expects data.response)
    if response:
        event_data["response"] = response.model_dump()

    return SearchEvent(
        event_type=EventType.SEARCH_COMPLETED,
        request_id=request_id,
        message=f"{mode} complete: {sources_count} sources in {execution_time_ms}ms",
        sources_count=sources_count,
        confidence=confidence,  # Now a top-level field!
        progress_percent=100,
        data=event_data,
        enhanced=enhanced
    )


def search_failed(request_id: str, error: str, enhanced: bool = False, **kwargs) -> SearchEvent:
    mode = "ENHANCED search" if enhanced else "Search"
    return SearchEvent(
        event_type=EventType.SEARCH_FAILED,
        request_id=request_id,
        message=f"{mode} failed: {error}",
        progress_percent=100,
        data={"error": error, "enhanced": enhanced},
        enhanced=enhanced
    )


def progress_update(request_id: str, percent: int, message: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.PROGRESS_UPDATE,
        request_id=request_id,
        message=message,
        progress_percent=percent
    )


# ========== NEW: Comprehensive Agent Processing Event Helpers ==========

def classifying_query(request_id: str, query: str) -> SearchEvent:
    """Query classification started"""
    return SearchEvent(
        event_type=EventType.CLASSIFYING_QUERY,
        request_id=request_id,
        message="Classifying query to determine optimal pipeline...",
        query=query,
        progress_percent=2
    )


def query_classified(
    request_id: str,
    category: str,
    pipeline: str,
    complexity: str,
    capabilities: list
) -> SearchEvent:
    """Query classification completed"""
    return SearchEvent(
        event_type=EventType.QUERY_CLASSIFIED,
        request_id=request_id,
        message=f"Query classified: {category} → {pipeline}",
        progress_percent=4,
        data={
            "category": category,
            "pipeline": pipeline,
            "complexity": complexity,
            "capabilities": capabilities
        }
    )


def crag_evaluating(request_id: str, document_count: int) -> SearchEvent:
    """CRAG pre-retrieval evaluation started"""
    return SearchEvent(
        event_type=EventType.CRAG_EVALUATING,
        request_id=request_id,
        message=f"CRAG: Evaluating quality of {document_count} retrieved documents...",
        results_count=document_count,
        progress_percent=45
    )


def crag_evaluation_complete(
    request_id: str,
    quality: str,
    relevance: float,
    action: str
) -> SearchEvent:
    """CRAG evaluation completed"""
    return SearchEvent(
        event_type=EventType.CRAG_EVALUATION_COMPLETE,
        request_id=request_id,
        message=f"CRAG: Quality={quality}, Relevance={relevance:.2f}, Action={action}",
        progress_percent=48,
        data={
            "quality": quality,
            "relevance": relevance,
            "action": action
        }
    )


def crag_refining(request_id: str, refined_queries: list) -> SearchEvent:
    """CRAG triggered refinement"""
    return SearchEvent(
        event_type=EventType.CRAG_REFINING,
        request_id=request_id,
        message=f"CRAG: Refining with {len(refined_queries)} new queries",
        queries=refined_queries,
        progress_percent=50
    )


def self_rag_reflecting(request_id: str, synthesis_length: int) -> SearchEvent:
    """Self-RAG reflection started"""
    return SearchEvent(
        event_type=EventType.SELF_RAG_REFLECTING,
        request_id=request_id,
        message=f"Self-RAG: Reflecting on synthesis ({synthesis_length:,} chars)...",
        progress_percent=88,
        data={"synthesis_length": synthesis_length}
    )


def self_rag_complete(
    request_id: str,
    relevance: float,
    support_level: str,
    usefulness: float,
    temporal_conflicts: int = 0
) -> SearchEvent:
    """Self-RAG reflection completed"""
    return SearchEvent(
        event_type=EventType.SELF_RAG_COMPLETE,
        request_id=request_id,
        message=f"Self-RAG: Relevance={relevance:.2f}, Support={support_level}, Usefulness={usefulness:.2f}",
        progress_percent=90,
        data={
            "relevance": relevance,
            "support_level": support_level,
            "usefulness": usefulness,
            "temporal_conflicts": temporal_conflicts
        }
    )


def self_rag_refining(request_id: str, reason: str) -> SearchEvent:
    """Self-RAG triggered synthesis refinement"""
    return SearchEvent(
        event_type=EventType.SELF_RAG_REFINING,
        request_id=request_id,
        message=f"Self-RAG: Refining synthesis - {reason}",
        progress_percent=91,
        data={"reason": reason}
    )


def scratchpad_initialized(request_id: str, questions: list) -> SearchEvent:
    """Scratchpad initialized with decomposed questions"""
    return SearchEvent(
        event_type=EventType.SCRATCHPAD_INITIALIZED,
        request_id=request_id,
        message=f"Scratchpad: Initialized with {len(questions)} questions",
        progress_percent=18,
        data={"questions": questions[:5]}  # First 5 for brevity
    )


def scratchpad_updated(request_id: str, findings_count: int, gaps_count: int) -> SearchEvent:
    """Scratchpad state updated"""
    return SearchEvent(
        event_type=EventType.SCRATCHPAD_UPDATED,
        request_id=request_id,
        message=f"Scratchpad: {findings_count} findings, {gaps_count} gaps remaining",
        data={"findings": findings_count, "gaps": gaps_count}
    )


def scratchpad_finding_added(request_id: str, finding_type: str, source: str) -> SearchEvent:
    """New finding added to scratchpad"""
    return SearchEvent(
        event_type=EventType.SCRATCHPAD_FINDING_ADDED,
        request_id=request_id,
        message=f"Scratchpad: Added {finding_type} finding from {source[:30]}...",
        data={"type": finding_type, "source": source}
    )


def scratchpad_question_answered(request_id: str, question: str, confidence: float) -> SearchEvent:
    """Question marked as answered"""
    return SearchEvent(
        event_type=EventType.SCRATCHPAD_QUESTION_ANSWERED,
        request_id=request_id,
        message=f"Scratchpad: Question answered (confidence={confidence:.2f})",
        data={"question": question[:50], "confidence": confidence}
    )


def scratchpad_gap_detected(request_id: str, gap_description: str) -> SearchEvent:
    """Gap detected in coverage"""
    return SearchEvent(
        event_type=EventType.SCRATCHPAD_GAP_DETECTED,
        request_id=request_id,
        message=f"Scratchpad: Gap detected - {gap_description[:60]}...",
        data={"gap": gap_description}
    )


def coverage_evaluating(request_id: str) -> SearchEvent:
    """Coverage evaluation started"""
    return SearchEvent(
        event_type=EventType.COVERAGE_EVALUATING,
        request_id=request_id,
        message="Evaluating corpus coverage against query requirements...",
        progress_percent=72
    )


def coverage_evaluated(
    request_id: str,
    coverage_score: float,
    is_sufficient: bool,
    gaps: list
) -> SearchEvent:
    """Coverage evaluation completed"""
    return SearchEvent(
        event_type=EventType.COVERAGE_EVALUATED,
        request_id=request_id,
        message=f"Coverage: {coverage_score:.1%} - {'Sufficient' if is_sufficient else f'{len(gaps)} gaps found'}",
        progress_percent=74,
        data={
            "coverage_score": coverage_score,
            "is_sufficient": is_sufficient,
            "gaps": gaps[:3]  # First 3 gaps
        }
    )


def coverage_insufficient(request_id: str, gaps: list) -> SearchEvent:
    """Coverage insufficient, need more sources"""
    return SearchEvent(
        event_type=EventType.COVERAGE_INSUFFICIENT,
        request_id=request_id,
        message=f"Coverage insufficient: {len(gaps)} information gaps",
        data={"gaps": gaps}
    )


def refinement_cycle_start(request_id: str, cycle: int, max_cycles: int) -> SearchEvent:
    """Refinement cycle started"""
    return SearchEvent(
        event_type=EventType.REFINEMENT_CYCLE_START,
        request_id=request_id,
        message=f"Starting refinement cycle {cycle}/{max_cycles}",
        iteration=cycle,
        max_iterations=max_cycles,
        progress_percent=75 + (cycle * 5)  # 75-90%
    )


def refinement_cycle_complete(
    request_id: str,
    cycle: int,
    new_sources: int,
    quality_improved: bool
) -> SearchEvent:
    """Refinement cycle completed"""
    return SearchEvent(
        event_type=EventType.REFINEMENT_CYCLE_COMPLETE,
        request_id=request_id,
        message=f"Refinement cycle {cycle} complete: +{new_sources} sources, quality {'improved' if quality_improved else 'unchanged'}",
        data={"cycle": cycle, "new_sources": new_sources, "quality_improved": quality_improved}
    )


def refinement_queries_generated(request_id: str, queries: list) -> SearchEvent:
    """New refinement queries generated"""
    return SearchEvent(
        event_type=EventType.REFINEMENT_QUERIES_GENERATED,
        request_id=request_id,
        message=f"Generated {len(queries)} refinement queries",
        queries=queries,
        data={"count": len(queries)}
    )


# =============================================================================
# Adaptive Refinement Events (Phase 2)
# =============================================================================

def adaptive_refinement_start(
    request_id: str,
    current_confidence: float,
    threshold: float,
    max_attempts: int
) -> SearchEvent:
    """Adaptive refinement loop started"""
    return SearchEvent(
        event_type=EventType.ADAPTIVE_REFINEMENT_START,
        request_id=request_id,
        message=f"Starting adaptive refinement (confidence {current_confidence:.1%} < {threshold:.1%} threshold)",
        data={
            "current_confidence": current_confidence,
            "threshold": threshold,
            "max_attempts": max_attempts
        }
    )


def adaptive_refinement_decision(
    request_id: str,
    decision: str,
    confidence: float,
    iteration: int,
    reason: str = ""
) -> SearchEvent:
    """Adaptive refinement decision made"""
    return SearchEvent(
        event_type=EventType.ADAPTIVE_REFINEMENT_DECISION,
        request_id=request_id,
        message=f"Refinement decision: {decision} (confidence: {confidence:.1%}, iteration: {iteration})",
        data={
            "decision": decision,
            "confidence": confidence,
            "iteration": iteration,
            "reason": reason
        }
    )


def gaps_identified(
    request_id: str,
    gaps: list,
    coverage_score: float
) -> SearchEvent:
    """Information gaps identified in synthesis"""
    return SearchEvent(
        event_type=EventType.GAPS_IDENTIFIED,
        request_id=request_id,
        message=f"Identified {len(gaps)} gaps (coverage: {coverage_score:.1%})",
        data={
            "gaps": gaps[:5],  # Limit to 5 for display
            "coverage_score": coverage_score,
            "gap_count": len(gaps)
        }
    )


def answer_graded(
    request_id: str,
    grade: str,
    score: int,
    gaps: list = None
) -> SearchEvent:
    """Answer quality graded"""
    return SearchEvent(
        event_type=EventType.ANSWER_GRADED,
        request_id=request_id,
        message=f"Answer grade: {grade} ({score}/5)",
        data={
            "grade": grade,
            "score": score,
            "gaps": gaps or []
        }
    )


def query_decomposed(
    request_id: str,
    original_query: str,
    sub_queries: list
) -> SearchEvent:
    """Complex query decomposed into sub-questions"""
    return SearchEvent(
        event_type=EventType.QUERY_DECOMPOSED,
        request_id=request_id,
        message=f"Decomposed query into {len(sub_queries)} sub-questions",
        query=original_query,
        data={
            "sub_queries": sub_queries,
            "count": len(sub_queries)
        }
    )


def adaptive_refinement_complete(
    request_id: str,
    final_decision: str,
    initial_confidence: float,
    final_confidence: float,
    iterations: int,
    total_duration_ms: int
) -> SearchEvent:
    """Adaptive refinement loop completed"""
    improvement = final_confidence - initial_confidence
    return SearchEvent(
        event_type=EventType.ADAPTIVE_REFINEMENT_COMPLETE,
        request_id=request_id,
        message=f"Refinement complete: {final_decision} ({improvement:+.1%} confidence in {iterations} iterations)",
        data={
            "final_decision": final_decision,
            "initial_confidence": initial_confidence,
            "final_confidence": final_confidence,
            "confidence_improvement": improvement,
            "iterations": iterations,
            "total_duration_ms": total_duration_ms
        }
    )


def llm_call_start(
    request_id: str,
    model: str,
    task: str,
    agent_phase: str,
    classification: str,
    input_tokens: int = 0,
    context_window: int = 0,
    prompt_preview: str = ""
) -> SearchEvent:
    """
    LLM API call started - detailed debugging event.

    Args:
        request_id: Unique request identifier
        model: Model name (e.g., "qwen3:8b", "deepseek-r1:14b")
        task: Specific task (e.g., "query_analysis", "synthesis", "verification")
        agent_phase: Pipeline phase (e.g., "PHASE_1_ANALYZE", "PHASE_6_SYNTHESIZE")
        classification: Model classification (e.g., "reasoning", "fast", "vision", "embedding")
        input_tokens: Estimated input token count
        context_window: Model's context window size
        prompt_preview: First 200 chars of prompt for debugging
    """
    return SearchEvent(
        event_type=EventType.LLM_CALL_START,
        request_id=request_id,
        message=f"LLM [{agent_phase}]: {model} ({classification}) starting {task}",
        model_name=model,
        data={
            "task": task,
            "agent_phase": agent_phase,
            "classification": classification,
            "model": model,
            "input_tokens": input_tokens,
            "context_window": context_window,
            "prompt_preview": prompt_preview[:200] if prompt_preview else "",
            "utilization_pct": round((input_tokens / context_window * 100), 1) if context_window > 0 else 0
        }
    )


def llm_call_complete(
    request_id: str,
    model: str,
    task: str,
    agent_phase: str,
    classification: str,
    duration_ms: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    context_window: int = 0,
    output_preview: str = "",
    cache_hit: bool = False,
    thinking_tokens: int = 0
) -> SearchEvent:
    """
    LLM API call completed - detailed debugging event.

    Args:
        request_id: Unique request identifier
        model: Model name used
        task: Specific task completed
        agent_phase: Pipeline phase that made the call
        classification: Model classification used
        duration_ms: Call duration in milliseconds
        input_tokens: Actual input token count
        output_tokens: Output token count
        context_window: Model's context window size
        output_preview: First 300 chars of output for debugging
        cache_hit: Whether KV cache was used
        thinking_tokens: For reasoning models, tokens used in thinking
    """
    tokens_per_sec = round(output_tokens / (duration_ms / 1000), 1) if duration_ms > 0 else 0
    total_tokens = input_tokens + output_tokens + thinking_tokens

    return SearchEvent(
        event_type=EventType.LLM_CALL_COMPLETE,
        request_id=request_id,
        message=f"LLM [{agent_phase}]: {model} completed {task} in {duration_ms}ms ({tokens_per_sec} tok/s)",
        model_name=model,
        data={
            "task": task,
            "agent_phase": agent_phase,
            "classification": classification,
            "model": model,
            "duration_ms": duration_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "context_window": context_window,
            "utilization_pct": round((input_tokens / context_window * 100), 1) if context_window > 0 else 0,
            "output_preview": output_preview[:300] if output_preview else "",
            "cache_hit": cache_hit
        }
    )


def corpus_quality_assessed(
    request_id: str,
    confidence: float,
    sources: int,
    domains: int,
    iterations: int
) -> SearchEvent:
    """Overall corpus quality assessment"""
    return SearchEvent(
        event_type=EventType.CORPUS_QUALITY_ASSESSED,
        request_id=request_id,
        message=f"Corpus quality: confidence={confidence:.2f}, {sources} sources from {domains} domains",
        progress_percent=93,
        data={
            "confidence": confidence,
            "sources": sources,
            "unique_domains": domains,
            "iterations": iterations
        }
    )


def web_search_start(request_id: str, query: str, api: str) -> SearchEvent:
    """Web search API call started"""
    return SearchEvent(
        event_type=EventType.WEB_SEARCH_START,
        request_id=request_id,
        message=f"Web search: Querying {api}...",
        query=query,
        data={"api": api}
    )


def web_search_complete(request_id: str, results_count: int, api: str) -> SearchEvent:
    """Web search API call completed"""
    return SearchEvent(
        event_type=EventType.WEB_SEARCH_COMPLETE,
        request_id=request_id,
        message=f"Web search: {results_count} results from {api}",
        results_count=results_count,
        data={"api": api}
    )


def web_search_fallback(request_id: str, from_api: str, to_api: str, reason: str) -> SearchEvent:
    """Falling back to alternative search API"""
    return SearchEvent(
        event_type=EventType.WEB_SEARCH_FALLBACK,
        request_id=request_id,
        message=f"Web search: Falling back from {from_api} to {to_api} ({reason})",
        data={"from_api": from_api, "to_api": to_api, "reason": reason}
    )


def decision_point(
    request_id: str,
    decision_type: str,
    decision: str,
    reason: str,
    alternatives: list = None
) -> SearchEvent:
    """Log a decision point for debugging"""
    return SearchEvent(
        event_type=EventType.DECISION_POINT,
        request_id=request_id,
        message=f"Decision [{decision_type}]: {decision}",
        data={
            "type": decision_type,
            "decision": decision,
            "reason": reason,
            "alternatives": alternatives or []
        }
    )


def pipeline_routed(request_id: str, pipeline: str, reason: str) -> SearchEvent:
    """Query routed to specific pipeline"""
    return SearchEvent(
        event_type=EventType.PIPELINE_ROUTED,
        request_id=request_id,
        message=f"Pipeline: Routing to {pipeline}",
        progress_percent=5,
        data={"pipeline": pipeline, "reason": reason}
    )


def experience_captured(request_id: str, query_type: str, confidence: float) -> SearchEvent:
    """Experience captured for distillation"""
    return SearchEvent(
        event_type=EventType.EXPERIENCE_CAPTURED,
        request_id=request_id,
        message=f"Experience captured: {query_type} query (confidence={confidence:.2f})",
        data={"query_type": query_type, "confidence": confidence}
    )


# ========== Phase 21: Meta-Buffer & Reasoning Composer Events ==========

def thought_template_matched(
    request_id: str,
    template_id: str,
    similarity: float
) -> SearchEvent:
    """Meta-Buffer found a matching template"""
    return SearchEvent(
        event_type=EventType.THOUGHT_TEMPLATE_MATCHED,
        request_id=request_id,
        message=f"Meta-Buffer: Template matched (id={template_id}, similarity={similarity:.2f})",
        progress_percent=12,
        data={
            "template_id": template_id,
            "similarity": similarity
        }
    )


def thought_template_applied(
    request_id: str,
    template_id: str,
    applied_components: list
) -> SearchEvent:
    """Template applied to guide search"""
    return SearchEvent(
        event_type=EventType.THOUGHT_TEMPLATE_APPLIED,
        request_id=request_id,
        message=f"Meta-Buffer: Template applied ({len(applied_components)} components)",
        progress_percent=15,
        data={
            "template_id": template_id,
            "applied_components": applied_components
        }
    )


def template_created(request_id: str, template_id: str) -> SearchEvent:
    """New template distilled from successful search"""
    return SearchEvent(
        event_type=EventType.TEMPLATE_CREATED,
        request_id=request_id,
        message=f"Meta-Buffer: Template created (id={template_id})",
        progress_percent=98,
        data={"template_id": template_id}
    )


def experience_distilling(request_id: str, experience_count: int) -> SearchEvent:
    """Experience distillation in progress"""
    return SearchEvent(
        event_type=EventType.EXPERIENCE_DISTILLING,
        request_id=request_id,
        message=f"Distilling {experience_count} experiences into templates...",
        progress_percent=97,
        data={"experience_count": experience_count}
    )


def reasoning_strategy_composed(
    request_id: str,
    module_count: int,
    modules: list
) -> SearchEvent:
    """Self-Discover reasoning strategy composed"""
    return SearchEvent(
        event_type=EventType.REASONING_STRATEGY_COMPOSED,
        request_id=request_id,
        message=f"Reasoning Composer: Strategy composed with {module_count} modules",
        progress_percent=13,
        data={
            "module_count": module_count,
            "modules": modules[:5]  # Limit to first 5 for brevity
        }
    )


def reasoning_branch_created(
    request_id: str,
    branch_id: str,
    hypothesis: str,
    depth: int
) -> SearchEvent:
    """Reasoning DAG branch created"""
    return SearchEvent(
        event_type=EventType.REASONING_BRANCH_CREATED,
        request_id=request_id,
        message=f"Reasoning DAG: Branch '{branch_id}' created at depth {depth}",
        progress_percent=14,
        data={
            "branch_id": branch_id,
            "hypothesis": hypothesis[:100] if hypothesis else "",
            "depth": depth
        }
    )


def reasoning_node_verified(
    request_id: str,
    node_id: str,
    is_valid: bool,
    confidence: float
) -> SearchEvent:
    """Reasoning DAG node verified"""
    return SearchEvent(
        event_type=EventType.REASONING_NODE_VERIFIED,
        request_id=request_id,
        message=f"Reasoning DAG: Node '{node_id}' verified (valid={is_valid}, conf={confidence:.2f})",
        progress_percent=15,
        data={
            "node_id": node_id,
            "is_valid": is_valid,
            "confidence": confidence
        }
    )


def reasoning_paths_merged(
    request_id: str,
    path_count: int,
    merged_node_id: str
) -> SearchEvent:
    """Reasoning DAG paths merged"""
    return SearchEvent(
        event_type=EventType.REASONING_PATHS_MERGED,
        request_id=request_id,
        message=f"Reasoning DAG: {path_count} paths merged into '{merged_node_id}'",
        progress_percent=16,
        data={
            "path_count": path_count,
            "merged_node_id": merged_node_id
        }
    )


def entities_extracted(
    request_id: str,
    entity_count: int,
    entity_names: list
) -> SearchEvent:
    """Entities extracted from query"""
    return SearchEvent(
        event_type=EventType.ENTITIES_EXTRACTED,
        request_id=request_id,
        message=f"Extracted {entity_count} entities: {', '.join(entity_names[:3])}",
        progress_percent=11,
        data={
            "entity_count": entity_count,
            "entity_names": entity_names[:5]
        }
    )


def entity_relation_found(
    request_id: str,
    source_entity: str,
    target_entity: str,
    relation_type: str
) -> SearchEvent:
    """Entity relation discovered"""
    return SearchEvent(
        event_type=EventType.ENTITY_RELATION_FOUND,
        request_id=request_id,
        message=f"Relation: {source_entity} --[{relation_type}]--> {target_entity}",
        progress_percent=11,
        data={
            "source_entity": source_entity,
            "target_entity": target_entity,
            "relation_type": relation_type
        }
    )


def outcome_recorded(
    request_id: str,
    category: str,
    quality: str,
    was_overkill: bool,
    was_underkill: bool
) -> SearchEvent:
    """Classifier outcome recorded for feedback"""
    return SearchEvent(
        event_type=EventType.OUTCOME_RECORDED,
        request_id=request_id,
        message=f"Outcome recorded: {category} → {quality}",
        data={
            "category": category,
            "quality": quality,
            "was_overkill": was_overkill,
            "was_underkill": was_underkill
        }
    )


def iteration_start_detailed(
    request_id: str,
    iteration: int,
    max_iterations: int,
    pending_queries: int,
    sources_so_far: int
) -> SearchEvent:
    """Detailed iteration start with context"""
    return SearchEvent(
        event_type=EventType.ITERATION_START,
        request_id=request_id,
        message=f"Iteration {iteration}/{max_iterations}: {pending_queries} queries pending, {sources_so_far} sources collected",
        iteration=iteration,
        max_iterations=max_iterations,
        sources_count=sources_so_far,
        progress_percent=20 + int((iteration / max_iterations) * 60),  # 20-80%
        data={"pending_queries": pending_queries}
    )


def iteration_complete_detailed(
    request_id: str,
    iteration: int,
    results_this_iteration: int,
    total_sources: int,
    continue_reason: str = None
) -> SearchEvent:
    """Detailed iteration completion"""
    return SearchEvent(
        event_type=EventType.ITERATION_COMPLETE,
        request_id=request_id,
        message=f"Iteration {iteration} complete: +{results_this_iteration} results, {total_sources} total sources",
        iteration=iteration,
        sources_count=total_sources,
        data={
            "results_this_iteration": results_this_iteration,
            "continue_reason": continue_reason
        }
    )


# ========== BGE-M3 Hybrid Retrieval Events ==========

def hybrid_search_start(
    request_id: str,
    query: str,
    mode: str = "hybrid"
) -> SearchEvent:
    """Hybrid search started"""
    return SearchEvent(
        event_type=EventType.HYBRID_SEARCH_START,
        request_id=request_id,
        message=f"Hybrid search: Starting {mode} retrieval...",
        query=query,
        data={"mode": mode}
    )


def hybrid_search_complete(
    request_id: str,
    results_count: int,
    dense_count: int,
    sparse_count: int,
    duration_ms: int
) -> SearchEvent:
    """Hybrid search completed"""
    return SearchEvent(
        event_type=EventType.HYBRID_SEARCH_COMPLETE,
        request_id=request_id,
        message=f"Hybrid search: {results_count} results (dense={dense_count}, sparse={sparse_count})",
        results_count=results_count,
        data={
            "dense_count": dense_count,
            "sparse_count": sparse_count,
            "duration_ms": duration_ms
        }
    )


def hybrid_indexing(
    request_id: str,
    doc_count: int,
    current: int = 0
) -> SearchEvent:
    """Hybrid indexing progress"""
    return SearchEvent(
        event_type=EventType.HYBRID_INDEXING,
        request_id=request_id,
        message=f"Hybrid index: Indexing {current}/{doc_count} documents...",
        data={"total": doc_count, "current": current}
    )


def hybrid_index_complete(
    request_id: str,
    indexed_count: int,
    total_docs: int
) -> SearchEvent:
    """Hybrid indexing completed"""
    return SearchEvent(
        event_type=EventType.HYBRID_INDEX_COMPLETE,
        request_id=request_id,
        message=f"Hybrid index: {indexed_count} documents indexed (total: {total_docs})",
        data={"indexed": indexed_count, "total": total_docs}
    )


def bm25_search(
    request_id: str,
    query: str,
    results_count: int,
    duration_ms: int
) -> SearchEvent:
    """BM25 sparse search completed"""
    return SearchEvent(
        event_type=EventType.BM25_SEARCH,
        request_id=request_id,
        message=f"BM25 sparse: {results_count} lexical matches in {duration_ms}ms",
        query=query,
        results_count=results_count,
        data={"duration_ms": duration_ms}
    )


def dense_embedding(
    request_id: str,
    text_length: int,
    dimensions: int,
    duration_ms: int
) -> SearchEvent:
    """Dense embedding generated"""
    return SearchEvent(
        event_type=EventType.DENSE_EMBEDDING,
        request_id=request_id,
        message=f"Dense embedding: {dimensions}D vector from {text_length} chars in {duration_ms}ms",
        data={
            "text_length": text_length,
            "dimensions": dimensions,
            "duration_ms": duration_ms
        }
    )


def rrf_fusion(
    request_id: str,
    dense_count: int,
    sparse_count: int,
    fused_count: int,
    rrf_k: int = 60
) -> SearchEvent:
    """RRF score fusion completed"""
    return SearchEvent(
        event_type=EventType.RRF_FUSION,
        request_id=request_id,
        message=f"RRF fusion: {dense_count}+{sparse_count}→{fused_count} (k={rrf_k})",
        data={
            "dense_count": dense_count,
            "sparse_count": sparse_count,
            "fused_count": fused_count,
            "rrf_k": rrf_k
        }
    )


# ========== HyDE Query Expansion Events ==========

def hyde_generating(
    request_id: str,
    query: str,
    doc_type: str = "answer",
    num_hypotheticals: int = 1
) -> SearchEvent:
    """HyDE hypothetical document generation started"""
    return SearchEvent(
        event_type=EventType.HYDE_GENERATING,
        request_id=request_id,
        message=f"HyDE: Generating {num_hypotheticals} hypothetical {doc_type}(s)...",
        query=query,
        data={"doc_type": doc_type, "num_hypotheticals": num_hypotheticals}
    )


def hyde_hypothetical_generated(
    request_id: str,
    index: int,
    total: int,
    length: int,
    duration_ms: int
) -> SearchEvent:
    """Single hypothetical document generated"""
    return SearchEvent(
        event_type=EventType.HYDE_HYPOTHETICAL_GENERATED,
        request_id=request_id,
        message=f"HyDE: Generated hypothetical {index}/{total} ({length} chars) in {duration_ms}ms",
        data={
            "index": index,
            "total": total,
            "length": length,
            "duration_ms": duration_ms
        }
    )


def hyde_embedding(
    request_id: str,
    text_count: int,
    include_query: bool
) -> SearchEvent:
    """HyDE embedding generation"""
    texts = text_count + (1 if include_query else 0)
    return SearchEvent(
        event_type=EventType.HYDE_EMBEDDING,
        request_id=request_id,
        message=f"HyDE: Embedding {texts} texts (query included: {include_query})",
        data={"text_count": texts, "include_query": include_query}
    )


def hyde_fusing(
    request_id: str,
    embedding_count: int,
    fusion_method: str = "mean"
) -> SearchEvent:
    """HyDE embedding fusion"""
    return SearchEvent(
        event_type=EventType.HYDE_FUSING,
        request_id=request_id,
        message=f"HyDE: Fusing {embedding_count} embeddings via {fusion_method}",
        data={"embedding_count": embedding_count, "fusion_method": fusion_method}
    )


def hyde_complete(
    request_id: str,
    hypotheticals_generated: int,
    gen_time_ms: int,
    emb_time_ms: int,
    dimensions: int
) -> SearchEvent:
    """HyDE expansion completed"""
    return SearchEvent(
        event_type=EventType.HYDE_COMPLETE,
        request_id=request_id,
        message=f"HyDE: Complete - {hypotheticals_generated} docs, {gen_time_ms}ms gen + {emb_time_ms}ms emb",
        data={
            "hypotheticals": hypotheticals_generated,
            "generation_time_ms": gen_time_ms,
            "embedding_time_ms": emb_time_ms,
            "dimensions": dimensions
        }
    )


# ========== RAGAS Evaluation Events ==========

def ragas_evaluating(
    request_id: str,
    metrics: list,
    context_count: int
) -> SearchEvent:
    """RAGAS evaluation started"""
    return SearchEvent(
        event_type=EventType.RAGAS_EVALUATING,
        request_id=request_id,
        message=f"RAGAS: Evaluating {len(metrics)} metrics on {context_count} contexts...",
        data={"metrics": metrics, "context_count": context_count}
    )


def ragas_claims_extracting(
    request_id: str,
    answer_length: int
) -> SearchEvent:
    """RAGAS claim extraction started"""
    return SearchEvent(
        event_type=EventType.RAGAS_CLAIMS_EXTRACTING,
        request_id=request_id,
        message=f"RAGAS: Extracting claims from {answer_length}-char answer...",
        data={"answer_length": answer_length}
    )


def ragas_claims_extracted(
    request_id: str,
    claim_count: int,
    duration_ms: int
) -> SearchEvent:
    """RAGAS claims extracted"""
    return SearchEvent(
        event_type=EventType.RAGAS_CLAIMS_EXTRACTED,
        request_id=request_id,
        message=f"RAGAS: Extracted {claim_count} claims in {duration_ms}ms",
        data={"claim_count": claim_count, "duration_ms": duration_ms}
    )


def ragas_claim_verifying(
    request_id: str,
    claim_index: int,
    total_claims: int,
    claim_preview: str
) -> SearchEvent:
    """RAGAS verifying a single claim"""
    preview = claim_preview[:50] + "..." if len(claim_preview) > 50 else claim_preview
    return SearchEvent(
        event_type=EventType.RAGAS_CLAIM_VERIFYING,
        request_id=request_id,
        message=f"RAGAS: Verifying claim {claim_index}/{total_claims}: {preview}",
        data={"claim_index": claim_index, "total_claims": total_claims}
    )


def ragas_claim_verified(
    request_id: str,
    claim_index: int,
    verdict: str,
    confidence: float
) -> SearchEvent:
    """RAGAS single claim verified"""
    return SearchEvent(
        event_type=EventType.RAGAS_CLAIM_VERIFIED,
        request_id=request_id,
        message=f"RAGAS: Claim {claim_index} → {verdict} ({confidence:.2f})",
        data={
            "claim_index": claim_index,
            "verdict": verdict,
            "confidence": confidence
        }
    )


def ragas_evaluation_complete(
    request_id: str,
    faithfulness: float,
    answer_relevancy: float,
    context_relevancy: float,
    overall_score: float,
    duration_ms: int
) -> SearchEvent:
    """RAGAS evaluation completed"""
    return SearchEvent(
        event_type=EventType.RAGAS_EVALUATION_COMPLETE,
        request_id=request_id,
        message=f"RAGAS: Complete - F={faithfulness:.2f} AR={answer_relevancy:.2f} CR={context_relevancy:.2f} → {overall_score:.2f}",
        data={
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_relevancy": context_relevancy,
            "overall_score": overall_score,
            "duration_ms": duration_ms
        }
    )


# ========== Agent Graph Traversal Visualization ==========

class AgentGraphState:
    """
    Tracks the current state of agent graph traversal for visualization.

    Generates one-line ASCII representations like:
      [A]→[S•]→[V]→[Σ]     (S is active)
      ●─●─◎─○─○              (step 3 of 5 active)
      Analyze→Search→Verify→Synthesize
    """

    # Agent symbols for compact display
    AGENT_SYMBOLS = {
        "classify": "C",
        "analyze": "A",
        "plan": "P",
        "search": "S",
        "scrape": "W",  # Web scrape
        "verify": "V",
        "synthesize": "Σ",
        "reflect": "R",
        "hyde": "H",
        "hybrid": "M",  # Mixed/hybrid
        "ragas": "Q",  # Quality
        "crag": "E",   # Evaluate
        "complete": "✓"
    }

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.current_index: int = -1
        self.branches: List[List[str]] = []  # For multi-path visualization

    def add_node(self, agent: str, status: str = "pending") -> int:
        """Add a node to the graph"""
        node = {
            "agent": agent,
            "symbol": self.AGENT_SYMBOLS.get(agent.lower(), agent[0].upper()),
            "status": status,
            "index": len(self.nodes)
        }
        self.nodes.append(node)
        return node["index"]

    def set_active(self, index: int):
        """Set a node as the currently active one"""
        for i, node in enumerate(self.nodes):
            if i < index:
                node["status"] = "completed"
            elif i == index:
                node["status"] = "active"
                self.current_index = i
            else:
                node["status"] = "pending"

    def complete_node(self, index: int, success: bool = True):
        """Mark a node as completed"""
        if 0 <= index < len(self.nodes):
            self.nodes[index]["status"] = "completed" if success else "failed"

    def to_line_simple(self) -> str:
        """Generate simple one-line representation: [A]→[S•]→[V]→[Σ]"""
        if not self.nodes:
            return ""

        parts = []
        for node in self.nodes:
            symbol = node["symbol"]
            if node["status"] == "active":
                parts.append(f"[{symbol}•]")
            elif node["status"] == "completed":
                parts.append(f"[{symbol}✓]")
            elif node["status"] == "failed":
                parts.append(f"[{symbol}✗]")
            else:
                parts.append(f"[{symbol}]")

        return "→".join(parts)

    def to_line_dots(self) -> str:
        """Generate dot-based representation: ●─●─◎─○─○"""
        if not self.nodes:
            return ""

        symbols = []
        for node in self.nodes:
            if node["status"] == "completed":
                symbols.append("●")
            elif node["status"] == "active":
                symbols.append("◎")
            elif node["status"] == "failed":
                symbols.append("✗")
            else:
                symbols.append("○")

        return "─".join(symbols)

    def to_line_names(self) -> str:
        """Generate name-based representation: Analyze→Search→Verify"""
        if not self.nodes:
            return ""

        parts = []
        for node in self.nodes:
            name = node["agent"].title()
            if node["status"] == "active":
                parts.append(f"*{name}*")
            elif node["status"] == "completed":
                parts.append(name)
            else:
                parts.append(f"({name})")

        return "→".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SSE transmission"""
        return {
            "nodes": self.nodes,
            "current_index": self.current_index,
            "line_simple": self.to_line_simple(),
            "line_dots": self.to_line_dots(),
            "line_names": self.to_line_names()
        }


# Global graph state for easy access
_graph_state: Optional[AgentGraphState] = None


def get_graph_state() -> AgentGraphState:
    """Get or create the global graph state"""
    global _graph_state
    if _graph_state is None:
        _graph_state = AgentGraphState()
    return _graph_state


def reset_graph_state():
    """Reset the global graph state for a new search"""
    global _graph_state
    _graph_state = AgentGraphState()


def graph_state_update(
    request_id: str,
    graph_state: AgentGraphState,
    message: str = ""
) -> SearchEvent:
    """Emit current graph state for visualization"""
    state_dict = graph_state.to_dict()
    return SearchEvent(
        event_type=EventType.GRAPH_STATE_UPDATE,
        request_id=request_id,
        message=message or f"Graph: {state_dict['line_simple']}",
        graph_line=state_dict["line_simple"],
        data=state_dict
    )


def graph_node_entered(
    request_id: str,
    agent: str,
    graph_state: AgentGraphState
) -> SearchEvent:
    """Agent node entered"""
    index = graph_state.add_node(agent, status="active")
    graph_state.set_active(index)

    return SearchEvent(
        event_type=EventType.GRAPH_NODE_ENTERED,
        request_id=request_id,
        message=f"→ {agent.title()}",
        graph_line=graph_state.to_line_simple(),
        data={
            "agent": agent,
            "index": index,
            "graph": graph_state.to_dict()
        }
    )


def graph_node_completed(
    request_id: str,
    agent: str,
    success: bool,
    graph_state: AgentGraphState,
    duration_ms: int = 0
) -> SearchEvent:
    """Agent node completed"""
    # Find and complete the node
    for i, node in enumerate(graph_state.nodes):
        if node["agent"].lower() == agent.lower() and node["status"] == "active":
            graph_state.complete_node(i, success)
            break

    status = "✓" if success else "✗"
    return SearchEvent(
        event_type=EventType.GRAPH_NODE_COMPLETED,
        request_id=request_id,
        message=f"{agent.title()} {status} ({duration_ms}ms)",
        graph_line=graph_state.to_line_simple(),
        data={
            "agent": agent,
            "success": success,
            "duration_ms": duration_ms,
            "graph": graph_state.to_dict()
        }
    )


def graph_edge_traversed(
    request_id: str,
    from_agent: str,
    to_agent: str,
    graph_state: AgentGraphState,
    reason: str = ""
) -> SearchEvent:
    """Edge traversed between agents"""
    return SearchEvent(
        event_type=EventType.GRAPH_EDGE_TRAVERSED,
        request_id=request_id,
        message=f"{from_agent}→{to_agent}" + (f": {reason}" if reason else ""),
        graph_line=graph_state.to_line_simple(),
        data={
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "graph": graph_state.to_dict()
        }
    )


def graph_branch_created(
    request_id: str,
    parent_agent: str,
    branch_agents: List[str],
    graph_state: AgentGraphState
) -> SearchEvent:
    """Branch created for parallel exploration"""
    branch_str = ", ".join(branch_agents)
    return SearchEvent(
        event_type=EventType.GRAPH_BRANCH_CREATED,
        request_id=request_id,
        message=f"{parent_agent} → [{branch_str}]",
        graph_line=graph_state.to_line_simple(),
        data={
            "parent": parent_agent,
            "branches": branch_agents,
            "graph": graph_state.to_dict()
        }
    )


def graph_paths_merged(
    request_id: str,
    merged_agents: List[str],
    target_agent: str,
    graph_state: AgentGraphState
) -> SearchEvent:
    """Multiple paths merged"""
    sources = ", ".join(merged_agents)
    return SearchEvent(
        event_type=EventType.GRAPH_PATHS_MERGED,
        request_id=request_id,
        message=f"[{sources}] → {target_agent}",
        graph_line=graph_state.to_line_simple(),
        data={
            "sources": merged_agents,
            "target": target_agent,
            "graph": graph_state.to_dict()
        }
    )


# ============================================
# Technical Documentation (PDF API) Event Helpers
# ============================================

def technical_docs_searching(
    request_id: str,
    query: str,
    error_codes: Optional[List[str]] = None,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Technical documentation search started"""
    return SearchEvent(
        event_type=EventType.TECHNICAL_DOCS_SEARCHING,
        request_id=request_id,
        message=f"Searching FANUC documentation: {query[:50]}...",
        query=query,
        graph_line=graph_line,
        data={
            "error_codes_detected": error_codes or [],
            "source": "pdf_extraction_tools"
        }
    )


def technical_docs_found(
    request_id: str,
    results_count: int,
    query: str,
    top_result_title: Optional[str] = None,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Technical documentation results found"""
    return SearchEvent(
        event_type=EventType.TECHNICAL_DOCS_FOUND,
        request_id=request_id,
        message=f"Found {results_count} technical documents",
        results_count=results_count,
        query=query,
        graph_line=graph_line,
        data={
            "top_result": top_result_title,
            "source": "pdf_extraction_tools"
        }
    )


def technical_docs_not_available(
    request_id: str,
    reason: str = "PDF API not available",
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Technical documentation search unavailable"""
    return SearchEvent(
        event_type=EventType.TECHNICAL_DOCS_NOT_AVAILABLE,
        request_id=request_id,
        message=f"Technical docs unavailable: {reason}",
        graph_line=graph_line,
        data={
            "reason": reason,
            "fallback": "web_search"
        }
    )


def troubleshoot_path_querying(
    request_id: str,
    error_code: str,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """PathRAG troubleshooting query started"""
    return SearchEvent(
        event_type=EventType.TROUBLESHOOT_PATH_QUERYING,
        request_id=request_id,
        message=f"Querying troubleshooting path for {error_code}",
        graph_line=graph_line,
        data={
            "error_code": error_code,
            "method": "pathrag_traversal"
        }
    )


def troubleshoot_path_found(
    request_id: str,
    error_code: str,
    steps_count: int,
    first_step: Optional[str] = None,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """PathRAG troubleshooting path found"""
    return SearchEvent(
        event_type=EventType.TROUBLESHOOT_PATH_FOUND,
        request_id=request_id,
        message=f"Found {steps_count}-step troubleshooting path for {error_code}",
        graph_line=graph_line,
        data={
            "error_code": error_code,
            "steps_count": steps_count,
            "first_step": first_step
        }
    )


def pdf_api_health_check(
    request_id: str,
    available: bool,
    response_time_ms: Optional[int] = None,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """PDF API health check result"""
    status = "healthy" if available else "unavailable"
    return SearchEvent(
        event_type=EventType.PDF_API_HEALTH_CHECK,
        request_id=request_id,
        message=f"PDF API: {status}",
        graph_line=graph_line,
        data={
            "available": available,
            "response_time_ms": response_time_ms,
            "api_url": "http://localhost:8002"
        }
    )


def corpus_sync_start(
    request_id: str,
    entity_count: int,
    sync_type: str = "full",
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Corpus synchronization with PDF API started"""
    return SearchEvent(
        event_type=EventType.CORPUS_SYNC_START,
        request_id=request_id,
        message=f"Starting {sync_type} corpus sync ({entity_count} entities)",
        graph_line=graph_line,
        data={
            "entity_count": entity_count,
            "sync_type": sync_type
        }
    )


def corpus_sync_progress(
    request_id: str,
    synced: int,
    total: int,
    current_entity: Optional[str] = None,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Corpus synchronization progress update"""
    percent = int((synced / total) * 100) if total > 0 else 0
    return SearchEvent(
        event_type=EventType.CORPUS_SYNC_PROGRESS,
        request_id=request_id,
        message=f"Syncing: {synced}/{total} ({percent}%)",
        progress_percent=percent,
        graph_line=graph_line,
        data={
            "synced": synced,
            "total": total,
            "current_entity": current_entity
        }
    )


def corpus_sync_complete(
    request_id: str,
    synced: int,
    new_entities: int,
    updated_entities: int,
    failed: int = 0,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Corpus synchronization completed"""
    return SearchEvent(
        event_type=EventType.CORPUS_SYNC_COMPLETE,
        request_id=request_id,
        message=f"Corpus sync complete: {synced} synced, {new_entities} new, {updated_entities} updated",
        progress_percent=100,
        graph_line=graph_line,
        data={
            "synced": synced,
            "new": new_entities,
            "updated": updated_entities,
            "failed": failed
        }
    )


def entity_enrichment_start(
    request_id: str,
    error_code: str,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Entity enrichment from PDF API started"""
    return SearchEvent(
        event_type=EventType.ENTITY_ENRICHMENT_START,
        request_id=request_id,
        message=f"Enriching {error_code} from PDF documentation",
        graph_line=graph_line,
        data={
            "error_code": error_code,
            "source": "pdf_extraction_tools"
        }
    )


def entity_enrichment_complete(
    request_id: str,
    error_code: str,
    steps_added: int,
    relations_added: int,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Entity enrichment completed"""
    return SearchEvent(
        event_type=EventType.ENTITY_ENRICHMENT_COMPLETE,
        request_id=request_id,
        message=f"Enriched {error_code}: +{steps_added} steps, +{relations_added} relations",
        graph_line=graph_line,
        data={
            "error_code": error_code,
            "steps_added": steps_added,
            "relations_added": relations_added
        }
    )


def fanuc_pattern_detected(
    request_id: str,
    query: str,
    error_codes: List[str],
    components: Optional[List[str]] = None,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """FANUC-specific patterns detected in query"""
    return SearchEvent(
        event_type=EventType.FANUC_PATTERN_DETECTED,
        request_id=request_id,
        message=f"FANUC patterns detected: {', '.join(error_codes[:3])}",
        query=query,
        graph_line=graph_line,
        data={
            "error_codes": error_codes,
            "components": components or [],
            "is_fanuc_query": True
        }
    )


# ========== G.6 Agent Coordination Event Helpers ==========

# G.6.1: A-MEM Semantic Memory
def semantic_memory_storing(
    request_id: str,
    memory_type: str,
    content_preview: str,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Semantic memory storage starting"""
    return SearchEvent(
        event_type=EventType.SEMANTIC_MEMORY_STORING,
        request_id=request_id,
        message=f"Storing {memory_type} memory...",
        graph_line=graph_line,
        data={
            "memory_type": memory_type,
            "content_preview": content_preview[:100] + "..." if len(content_preview) > 100 else content_preview
        }
    )


def semantic_memory_stored(
    request_id: str,
    memory_id: str,
    memory_type: str,
    connections_count: int = 0,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Semantic memory stored successfully"""
    return SearchEvent(
        event_type=EventType.SEMANTIC_MEMORY_STORED,
        request_id=request_id,
        message=f"Stored {memory_type} with {connections_count} connections",
        graph_line=graph_line,
        data={
            "memory_id": memory_id,
            "memory_type": memory_type,
            "connections_count": connections_count
        }
    )


def semantic_memory_retrieved(
    request_id: str,
    query: str,
    memories_count: int,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Semantic memories retrieved"""
    return SearchEvent(
        event_type=EventType.SEMANTIC_MEMORY_RETRIEVED,
        request_id=request_id,
        message=f"Retrieved {memories_count} relevant memories",
        query=query,
        graph_line=graph_line,
        data={
            "memories_count": memories_count
        }
    )


def memory_connection_created(
    request_id: str,
    source_id: str,
    target_id: str,
    connection_type: str,
    strength: float,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Memory connection created in graph"""
    return SearchEvent(
        event_type=EventType.MEMORY_CONNECTION_CREATED,
        request_id=request_id,
        message=f"Created {connection_type} connection (strength: {strength:.2f})",
        graph_line=graph_line,
        data={
            "source_id": source_id,
            "target_id": target_id,
            "connection_type": connection_type,
            "strength": strength
        }
    )


# G.6.2: DyLAN Agent Importance Scores
def dylan_complexity_classified(
    request_id: str,
    complexity: str,
    confidence: float,
    recommended_agents: List[str],
    graph_line: Optional[str] = None
) -> SearchEvent:
    """DyLAN query complexity classified"""
    return SearchEvent(
        event_type=EventType.DYLAN_COMPLEXITY_CLASSIFIED,
        request_id=request_id,
        message=f"Query complexity: {complexity} (confidence: {confidence:.2f})",
        graph_line=graph_line,
        data={
            "complexity": complexity,
            "confidence": confidence,
            "recommended_agents": recommended_agents
        }
    )


def dylan_agent_skipped(
    request_id: str,
    agent_name: str,
    reason: str,
    importance_score: float,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """DyLAN decision to skip an agent"""
    return SearchEvent(
        event_type=EventType.DYLAN_AGENT_SKIPPED,
        request_id=request_id,
        message=f"Skipping {agent_name}: {reason}",
        graph_line=graph_line,
        data={
            "agent_name": agent_name,
            "reason": reason,
            "importance_score": importance_score
        }
    )


def dylan_contribution_recorded(
    request_id: str,
    agent_name: str,
    quality_delta: float,
    execution_time_ms: int,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """DyLAN agent contribution recorded"""
    return SearchEvent(
        event_type=EventType.DYLAN_CONTRIBUTION_RECORDED,
        request_id=request_id,
        message=f"Recorded {agent_name} contribution (delta: {quality_delta:+.2f})",
        graph_line=graph_line,
        data={
            "agent_name": agent_name,
            "quality_delta": quality_delta,
            "execution_time_ms": execution_time_ms
        }
    )


# G.6.4: Information Bottleneck Filtering
def ib_filtering_start(
    request_id: str,
    passages_count: int,
    filtering_level: str,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Information bottleneck filtering starting"""
    return SearchEvent(
        event_type=EventType.IB_FILTERING_START,
        request_id=request_id,
        message=f"Filtering {passages_count} passages ({filtering_level} level)",
        graph_line=graph_line,
        data={
            "passages_count": passages_count,
            "filtering_level": filtering_level
        }
    )


def ib_filtering_complete(
    request_id: str,
    original_count: int,
    filtered_count: int,
    compression_ratio: float,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Information bottleneck filtering complete"""
    return SearchEvent(
        event_type=EventType.IB_FILTERING_COMPLETE,
        request_id=request_id,
        message=f"Filtered {original_count} → {filtered_count} passages ({compression_ratio:.1%} compression)",
        graph_line=graph_line,
        data={
            "original_count": original_count,
            "filtered_count": filtered_count,
            "compression_ratio": compression_ratio
        }
    )


# G.6.5: Contrastive Retriever Training
def contrastive_session_recorded(
    request_id: str,
    documents_count: int,
    cited_count: int,
    synthesis_confidence: float,
    strategy: str,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Contrastive retrieval session recorded for learning"""
    return SearchEvent(
        event_type=EventType.CONTRASTIVE_SESSION_RECORDED,
        request_id=request_id,
        message=f"Recorded session: {cited_count}/{documents_count} docs cited (conf: {synthesis_confidence:.2f})",
        graph_line=graph_line,
        data={
            "documents_count": documents_count,
            "cited_count": cited_count,
            "synthesis_confidence": synthesis_confidence,
            "strategy": strategy
        }
    )


def contrastive_insight_generated(
    request_id: str,
    insight_type: str,
    description: str,
    impact: str,
    graph_line: Optional[str] = None
) -> SearchEvent:
    """Contrastive retriever generated learning insight"""
    return SearchEvent(
        event_type=EventType.CONTRASTIVE_INSIGHT_GENERATED,
        request_id=request_id,
        message=f"Insight ({insight_type}): {description[:50]}...",
        graph_line=graph_line,
        data={
            "insight_type": insight_type,
            "description": description,
            "impact": impact
        }
    )
