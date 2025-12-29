"""
BaseSearchPipeline - Shared functionality extracted from all orchestrators.

This mixin provides core pipeline methods that all orchestrators use:
- Agent initialization
- Cache management
- State management
- Event emission
- Error handling

By extracting this shared code, we reduce duplication and ensure consistency
across all orchestrator implementations.
"""

import asyncio
import logging
import time
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import uuid

from urllib.parse import urlparse
from .models import (
    SearchRequest,
    SearchResponse,
    SearchResultData,
    SearchMeta,
    SearchState,
    SearchMode,
    ConfidenceLevel,
    WebSearchResult
)
from .analyzer import QueryAnalyzer
from .planner import PlannerAgent
from .searcher import SearcherAgent
from .verifier import VerifierAgent
from .synthesizer import SynthesizerAgent
from .scraper import ContentScraper, DeepReader, VisionAnalyzer
from .scratchpad import AgenticScratchpad, ScratchpadManager
from .content_cache import get_content_cache
from .ttl_cache_manager import get_ttl_cache_manager
from .events import EventEmitter, EventType, SearchEvent

# Lazy settings import to avoid circular dependencies
_settings = None
def _get_settings():
    global _settings
    if _settings is None:
        from config.settings import get_settings
        _settings = get_settings()
    return _settings

logger = logging.getLogger("agentic.base_pipeline")


class BaseSearchPipeline(ABC):
    """
    Abstract base class providing shared pipeline functionality.

    All orchestrators inherit from this to get:
    - Standard agent initialization
    - Cache key generation
    - Semantic cache checking
    - Response building
    - Confidence calculation
    - Event emission helpers
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        mcp_url: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        memory_service: Optional[Any] = None
    ):
        settings = _get_settings()
        self.ollama_url = ollama_url or settings.ollama_base_url
        self.mcp_url = mcp_url or settings.mcp_url

        # Core agents (shared across all orchestrators)
        self.analyzer = QueryAnalyzer(ollama_url=self.ollama_url)
        self.planner = PlannerAgent(ollama_url=self.ollama_url, mcp_url=self.mcp_url)
        self.searcher = SearcherAgent(brave_api_key=brave_api_key)
        self.verifier = VerifierAgent(ollama_url=self.ollama_url)
        self.synthesizer = SynthesizerAgent(ollama_url=self.ollama_url, mcp_url=self.mcp_url)

        # Content processing
        self.scraper = ContentScraper()
        self.deep_reader = DeepReader(ollama_url=self.ollama_url)
        self.vision_analyzer = VisionAnalyzer(ollama_url=self.ollama_url)

        # Memory and caching
        self.memory_service = memory_service
        self.scratchpad_manager = ScratchpadManager(memory_service=memory_service)
        self.ttl_manager = get_ttl_cache_manager()

        # In-memory cache
        self._cache: Dict[str, SearchResponse] = {}
        self._cache_ttl = 3600  # 1 hour

        # Event emitter (set by search methods)
        self.event_emitter: Optional[EventEmitter] = None

    async def initialize(self):
        """Initialize pipeline and check service availability."""
        mcp_available = await self.planner.check_mcp_available()
        await self.synthesizer.check_mcp_available()
        logger.info(f"{self.__class__.__name__} initialized. MCP available: {mcp_available}")
        return mcp_available

    def set_event_emitter(self, emitter: EventEmitter):
        """Set event emitter for SSE streaming."""
        self.event_emitter = emitter

    async def emit_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        request_id: str = "",
        message: str = "",
        graph_line: str = None
    ):
        """Emit an event if emitter is set."""
        if self.event_emitter:
            await self.event_emitter.emit(SearchEvent(
                event_type=event_type,
                request_id=request_id or str(uuid.uuid4())[:8],
                data=data,
                message=message,
                graph_line=graph_line
            ))

    def get_cache_key(self, request: SearchRequest) -> str:
        """Generate a cache key for the request."""
        key_data = f"{request.query}:{request.max_iterations}:{request.search_mode.value}"
        if request.context:
            key_data += f":{hash(str(sorted(request.context.items())))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def check_cache(self, request: SearchRequest) -> Optional[SearchResponse]:
        """Check in-memory cache for existing response."""
        cache_key = self.get_cache_key(request)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.meta.cache_hit = True
            return cached
        return None

    async def check_semantic_cache(
        self,
        request: SearchRequest,
        request_id: str
    ) -> Optional[SearchResponse]:
        """Check semantic cache for similar queries."""
        try:
            content_cache = get_content_cache()
            # First try exact match with get_query_result
            cached = content_cache.get_query_result(request.query)
            if cached:
                logger.info(f"[{request_id}] Exact cache hit for query")
                return self._build_cached_response(cached, request_id)
            # Note: Semantic similarity search via find_similar_query requires
            # pre-computed embeddings, so we skip it here for simplicity
        except Exception as e:
            logger.warning(f"[{request_id}] Semantic cache check failed: {e}")
        return None

    def _build_cached_response(
        self,
        cached: Dict[str, Any],
        request_id: str
    ) -> SearchResponse:
        """Build a response from cached data."""
        return SearchResponse(
            success=True,
            data=SearchResultData(
                synthesized_context=cached.get("synthesis", ""),
                sources=cached.get("sources", []),
                search_queries=cached.get("queries", []),
                confidence_score=cached.get("confidence", 0.5),
                confidence_level=ConfidenceLevel.MEDIUM,
                verification_status="cached"
            ),
            meta=SearchMeta(
                request_id=request_id,
                cache_hit=True,
                semantic_match=True,
                matched_query=cached.get("matched_query"),
                similarity_score=cached.get("similarity_score")
            )
        )

    def store_in_cache(
        self,
        request: SearchRequest,
        response: SearchResponse
    ):
        """Store response in in-memory cache."""
        cache_key = self.get_cache_key(request)
        self._cache[cache_key] = response

    def create_search_state(self, request: SearchRequest) -> SearchState:
        """Create initial search state from request."""
        return SearchState(
            query=request.query,
            max_iterations=request.max_iterations,
            search_mode=request.search_mode.value
        )

    def create_scratchpad(
        self,
        request: SearchRequest,
        request_id: str
    ) -> AgenticScratchpad:
        """Create scratchpad for the search session."""
        return self.scratchpad_manager.create(
            query=request.query,
            request_id=request_id,
            user_id=request.user_id
        )

    def calculate_blended_confidence(
        self,
        verifier_confidence: float,
        reflection_confidence: Optional[float] = None,
        ragas_score: Optional[float] = None,
        source_diversity: Optional[float] = None,
        content_depth: Optional[float] = None
    ) -> float:
        """
        Calculate blended confidence from multiple signals.

        Standard weights (from design docs):
        - Verification: 40%
        - Source diversity: 25%
        - Content depth: 20%
        - Synthesis quality: 15%

        With reflection/RAGAS, adjust weights dynamically.
        """
        # Base: verifier confidence
        confidence = verifier_confidence

        # If reflection provided, blend 60% verifier + 40% reflection
        if reflection_confidence is not None:
            confidence = (verifier_confidence * 0.6) + (reflection_confidence * 0.4)

        # If RAGAS provided, blend with RAGAS score
        if ragas_score is not None:
            confidence = (confidence * 0.7) + (ragas_score * 0.3)

        # Add diversity bonus (up to +0.1)
        if source_diversity is not None:
            confidence += min(source_diversity * 0.1, 0.1)

        # Add depth bonus (up to +0.05)
        if content_depth is not None:
            confidence += min(content_depth * 0.05, 0.05)

        return min(1.0, max(0.0, confidence))

    def calculate_heuristic_confidence(
        self,
        sources: List[Dict[str, Any]],
        synthesis: str,
        query: str,
        max_sources: int = 10
    ) -> float:
        """
        Calculate confidence without requiring evaluation features.

        This is used as a baseline when reflection, RAGAS, and other
        evaluation features are disabled (e.g., minimal preset).

        Based on observable metrics that don't require LLM evaluation:
        - Source coverage (how many sources found)
        - Domain diversity and trust
        - Content depth (synthesis length)
        - Query term coverage

        Returns a confidence score between 0.0 and 1.0.
        """
        # Source coverage (0-0.30): More sources = higher confidence
        source_count = len(sources)
        source_score = min(source_count / max_sources, 1.0) * 0.30

        # Domain diversity and trust (0-0.25)
        trusted_domains = {
            'robot-forum.com', 'fanucamerica.com', 'fanuc.co.jp',
            'stackoverflow.com', 'github.com', 'arxiv.org', 'reddit.com',
            'wikipedia.org', 'docs.python.org', 'developer.mozilla.org',
            'learn.microsoft.com', 'cloud.google.com', 'aws.amazon.com',
            'huggingface.co', 'pytorch.org', 'tensorflow.org',
            'medium.com', 'towardsdatascience.com', 'ieee.org'
        }

        domains = set()
        trusted_count = 0
        for src in sources:
            url = src.get('url', '') or src.get('link', '')
            if url:
                try:
                    netloc = urlparse(url).netloc.lower()
                    # Remove www. prefix
                    if netloc.startswith('www.'):
                        netloc = netloc[4:]
                    domains.add(netloc)
                    # Check if it's a trusted domain
                    if any(trusted in netloc for trusted in trusted_domains):
                        trusted_count += 1
                except (ValueError, AttributeError):
                    pass  # Invalid URL format, skip

        # Domain diversity score (unique domains / total sources)
        diversity = len(domains) / max(source_count, 1)
        # Trust score (trusted sources / total sources)
        trust = trusted_count / max(source_count, 1)
        # Combined diversity score
        diversity_score = (diversity * 0.5 + trust * 0.5) * 0.25

        # Content depth (0-0.25): Longer synthesis suggests more thorough answer
        expected_length = 2000  # chars for a good response
        synthesis_length = len(synthesis) if synthesis else 0
        # Allow up to 1.5x expected to cap bonus
        depth_ratio = min(synthesis_length / expected_length, 1.5) / 1.5
        depth_score = depth_ratio * 0.25

        # Query term coverage (0-0.20): How many query terms appear in synthesis
        if synthesis and query:
            # Simple word-based coverage
            query_terms = set(word.lower() for word in query.split() if len(word) > 2)
            synthesis_lower = synthesis.lower()
            covered = sum(1 for term in query_terms if term in synthesis_lower)
            coverage = covered / max(len(query_terms), 1)
        else:
            coverage = 0.0
        coverage_score = coverage * 0.20

        total = source_score + diversity_score + depth_score + coverage_score

        logger.debug(
            f"Heuristic confidence: sources={source_score:.2f}, "
            f"diversity={diversity_score:.2f}, depth={depth_score:.2f}, "
            f"coverage={coverage_score:.2f}, total={total:.2f}"
        )

        return min(1.0, max(0.0, total))

    def get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Map confidence score to confidence level."""
        if score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.50:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

    def build_response(
        self,
        synthesis: str,
        sources: List[Dict[str, str]],
        queries: List[str],
        confidence: float,
        state: SearchState,
        request_id: str,
        search_trace: List[Dict[str, Any]],
        execution_time_ms: int,
        enhancement_metadata: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """Build a complete search response."""
        return SearchResponse(
            success=True,
            data=SearchResultData(
                synthesized_context=synthesis,
                sources=sources,
                search_queries=queries,
                confidence_score=confidence,
                confidence_level=self.get_confidence_level(confidence),
                verification_status="verified" if confidence >= 0.7 else "partial",
                search_trace=search_trace
            ),
            meta=SearchMeta(
                request_id=request_id,
                iterations=state.iteration,
                queries_executed=len(state.executed_queries),
                sources_consulted=state.sources_consulted,
                execution_time_ms=execution_time_ms,
                enhancement_metadata=enhancement_metadata
            )
        )

    def build_error_response(
        self,
        error_message: str,
        request_id: str,
        execution_time_ms: int = 0
    ) -> SearchResponse:
        """Build an error response."""
        return SearchResponse(
            success=False,
            data=SearchResultData(
                synthesized_context=f"Search failed: {error_message}",
                sources=[],
                search_queries=[],
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                verification_status="failed"
            ),
            meta=SearchMeta(
                request_id=request_id,
                execution_time_ms=execution_time_ms
            ),
            errors=[{"code": "SEARCH_ERROR", "message": error_message}]
        )

    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute search - must be implemented by concrete orchestrators."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics - can be overridden for additional stats."""
        return {
            "cache_size": len(self._cache),
            "ollama_url": self.ollama_url,
            "mcp_url": self.mcp_url
        }
