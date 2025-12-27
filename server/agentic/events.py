"""
Event System for Agentic Search Progress Notifications

Provides real-time progress updates to clients via Server-Sent Events (SSE).
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, Set, Callable, Awaitable
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


@dataclass
class SearchEvent:
    """An event emitted during agentic search"""

    event_type: EventType
    request_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

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

    async def emit(self, event: SearchEvent):
        """Emit an event to all subscribers"""
        if self._closed:
            return

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
            except:
                pass
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
def search_started(request_id: str, query: str, max_iterations: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SEARCH_STARTED,
        request_id=request_id,
        message=f"Starting search for: {query[:50]}...",
        query=query,
        max_iterations=max_iterations,
        progress_percent=0
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


def searching(request_id: str, queries: list, iteration: int, max_iterations: int) -> SearchEvent:
    progress = 20 + int((iteration / max_iterations) * 40)  # 20-60%
    return SearchEvent(
        event_type=EventType.SEARCHING,
        request_id=request_id,
        message=f"Searching: {', '.join(queries[:2])}{'...' if len(queries) > 2 else ''}",
        queries=queries,
        iteration=iteration,
        max_iterations=max_iterations,
        progress_percent=progress
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


def verifying_claims(request_id: str, claims_count: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.VERIFYING_CLAIMS,
        request_id=request_id,
        message=f"Verifying {claims_count} claims...",
        progress_percent=75
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


def synthesizing(request_id: str, sources_count: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SYNTHESIZING,
        request_id=request_id,
        message=f"Synthesizing answer from {sources_count} sources...",
        sources_count=sources_count,
        progress_percent=85
    )


def synthesis_complete(request_id: str, answer_length: int, confidence: float) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SYNTHESIS_COMPLETE,
        request_id=request_id,
        message="Answer synthesized",
        progress_percent=95,
        data={"answer_length": answer_length, "confidence": confidence}
    )


def search_completed(request_id: str, sources_count: int, execution_time_ms: int) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SEARCH_COMPLETED,
        request_id=request_id,
        message=f"Search complete: {sources_count} sources in {execution_time_ms}ms",
        sources_count=sources_count,
        progress_percent=100,
        data={"execution_time_ms": execution_time_ms}
    )


def search_failed(request_id: str, error: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.SEARCH_FAILED,
        request_id=request_id,
        message=f"Search failed: {error}",
        progress_percent=100,
        data={"error": error}
    )


def progress_update(request_id: str, percent: int, message: str) -> SearchEvent:
    return SearchEvent(
        event_type=EventType.PROGRESS_UPDATE,
        request_id=request_id,
        message=message,
        progress_percent=percent
    )
