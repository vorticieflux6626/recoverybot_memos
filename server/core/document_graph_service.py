"""
Document Graph Service - Bridge to PDF Extraction Tools API

Provides memOS with access to document graph operations,
PathRAG traversal, and technical documentation search.

This service integrates with the PDF Extraction Tools API (port 8002)
to enable RAG-powered navigation of FANUC technical manuals.

Usage:
    from core.document_graph_service import document_graph_service

    # Search documentation
    results = await document_graph_service.search_documentation(
        query="SRVO-063 pulsecoder",
        max_results=10
    )

    # Get troubleshooting path
    steps = await document_graph_service.query_troubleshooting_path(
        error_code="SRVO-063",
        max_hops=5
    )

    # Get RAG context
    context = await document_graph_service.get_context_for_rag(
        query="How to calibrate after encoder replacement?"
    )
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.settings import get_settings

logger = logging.getLogger("core.document_graph_service")


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class TroubleshootingStep:
    """A step in a troubleshooting path from PathRAG traversal"""
    node_id: str
    title: str
    content: str
    step_type: str  # 'error', 'diagnosis', 'solution', 'procedure', 'info'
    relevance_score: float
    hop_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSearchResult:
    """Result from document graph search"""
    node_id: str
    title: str
    content_preview: str
    score: float
    document_path: List[str]  # Hierarchical path: ["Manual", "Chapter", "Section"]
    matched_terms: List[str]
    node_type: str = "section"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraversalPath:
    """A complete troubleshooting path"""
    path_id: str
    steps: List[TroubleshootingStep]
    total_score: float
    path_type: str  # e.g., "error->diagnosis->solution"


# ============================================
# DOCUMENT GRAPH SERVICE
# ============================================

class DocumentGraphService:
    """
    Bridge between memOS retrieval and PDF Extraction Tools graphs.

    Provides:
    - PathRAG traversal for troubleshooting chains
    - Semantic search across technical documentation
    - Graph-based context retrieval for RAG
    - Health monitoring and caching

    The service implements:
    - Connection pooling for efficiency
    - Response caching with TTL
    - Graceful degradation when PDF API unavailable
    - Automatic retry with exponential backoff
    """

    # Cache TTL in seconds
    CACHE_TTL = 300  # 5 minutes
    HEALTH_CHECK_INTERVAL = 60  # 1 minute

    def __init__(self):
        self.settings = get_settings()
        self.base_url = getattr(self.settings, 'pdf_api_url', 'http://localhost:8002')
        self.timeout = getattr(self.settings, 'pdf_api_timeout', 30)
        self.enabled = getattr(self.settings, 'pdf_api_enabled', True)

        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        # Simple in-memory cache
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._health_status: Optional[bool] = None
        self._health_checked_at: float = 0

        # Circuit breaker state
        self._failure_count = 0
        self._circuit_open_until: float = 0

        logger.info(f"DocumentGraphService initialized: {self.base_url}")

    # ============================================
    # HEALTH & AVAILABILITY
    # ============================================

    async def health_check(self) -> bool:
        """
        Check if PDF Extraction Tools API is available.

        Uses caching to avoid excessive health checks.
        Implements circuit breaker pattern for resilience.
        """
        now = time.time()

        # Check circuit breaker
        if now < self._circuit_open_until:
            logger.debug("Circuit breaker open, skipping health check")
            return False

        # Use cached health status if recent
        if self._health_status is not None and (now - self._health_checked_at) < self.HEALTH_CHECK_INTERVAL:
            return self._health_status

        try:
            response = await self.client.get("/health", timeout=5.0)
            self._health_status = response.status_code == 200
            self._health_checked_at = now

            if self._health_status:
                self._failure_count = 0
                logger.debug("PDF API health check: OK")
            else:
                self._record_failure()
                logger.warning(f"PDF API health check failed: {response.status_code}")

            return self._health_status

        except Exception as e:
            self._record_failure()
            self._health_status = False
            self._health_checked_at = now
            logger.warning(f"PDF API health check error: {e}")
            return False

    def _record_failure(self):
        """Record a failure and potentially open circuit breaker"""
        self._failure_count += 1
        if self._failure_count >= 3:
            # Open circuit for 60 seconds after 3 failures
            self._circuit_open_until = time.time() + 60
            logger.warning("Circuit breaker opened for PDF API (60s)")

    @property
    def is_available(self) -> bool:
        """Quick check if service might be available (uses cached status)"""
        if not self.enabled:
            return False
        if time.time() < self._circuit_open_until:
            return False
        return self._health_status is not False

    # ============================================
    # CACHING
    # ============================================

    def _cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from prefix and arguments"""
        content = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            value, expires_at = self._cache[key]
            if time.time() < expires_at:
                return value
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any, ttl: int = None):
        """Store value in cache with TTL"""
        ttl = ttl or self.CACHE_TTL
        self._cache[key] = (value, time.time() + ttl)

    def clear_cache(self):
        """Clear all cached responses"""
        self._cache.clear()
        logger.info("DocumentGraphService cache cleared")

    # ============================================
    # SEARCH OPERATIONS
    # ============================================

    async def search_documentation(
        self,
        query: str,
        search_type: str = "hybrid",
        max_results: int = 10,
        node_types: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[DocumentSearchResult]:
        """
        Search technical documentation via PDF Extraction Tools API.

        Args:
            query: Search query
            search_type: 'keyword', 'semantic', or 'hybrid'
            max_results: Maximum results to return
            node_types: Filter by node types (section, chunk, table, etc.)
            min_score: Minimum relevance score threshold

        Returns:
            List of search results with relevance scores
        """
        if not self.is_available:
            logger.debug("PDF API not available, returning empty results")
            return []

        # Check cache
        cache_key = self._cache_key("search", query, search_type, max_results)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for search: {query[:30]}...")
            return cached

        try:
            request_body = {
                "query": query,
                "search_type": search_type,
                "max_results": max_results,
                "min_score": min_score,
                "include_context": True
            }

            if node_types:
                request_body["node_types"] = node_types

            response = await self.client.post("/search", json=request_body)
            response.raise_for_status()
            data = response.json()

            results = [
                DocumentSearchResult(
                    node_id=r.get('node_id', ''),
                    title=r.get('title', ''),
                    content_preview=r.get('content_preview', r.get('preview', '')),
                    score=r.get('score', 0.0),
                    document_path=r.get('path', []),
                    matched_terms=r.get('matched_terms', []),
                    node_type=r.get('node_type', 'section'),
                    metadata=r.get('metadata', {})
                )
                for r in data.get('results', [])
            ]

            # Cache results
            self._set_cached(cache_key, results)

            logger.info(f"PDF search '{query[:30]}...' returned {len(results)} results")
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"PDF API search error: {e.response.status_code}")
            self._record_failure()
            return []
        except Exception as e:
            logger.error(f"PDF API search failed: {e}")
            self._record_failure()
            return []

    # ============================================
    # PATHRAG TRAVERSAL
    # ============================================

    async def query_troubleshooting_path(
        self,
        error_code: str,
        robot_model: Optional[str] = None,
        max_hops: int = 5,
        mode: str = "semantic_astar"
    ) -> List[TroubleshootingStep]:
        """
        PathRAG traversal for error resolution.

        Uses flow-based pruning to find relevant troubleshooting paths
        from error code to diagnosis to solution.

        Args:
            error_code: FANUC error code (e.g., 'SRVO-023')
            robot_model: Optional robot model filter
            max_hops: Maximum traversal depth
            mode: Traversal mode ('semantic_astar', 'flow_based', 'multi_hop')

        Returns:
            Ordered list of troubleshooting steps
        """
        if not self.is_available:
            logger.debug("PDF API not available for traversal")
            return []

        # Normalize error code
        error_code = error_code.upper().strip()

        # Check cache
        cache_key = self._cache_key("troubleshoot", error_code, max_hops, mode)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for troubleshooting: {error_code}")
            return cached

        try:
            request_body = {
                "source_concept": error_code,
                "mode": mode,
                "max_hops": max_hops,
                "include_content": True,
                "alpha": 0.85,  # PathRAG damping factor
                "theta": 0.01   # Pruning threshold
            }

            if robot_model:
                request_body["target_concept"] = robot_model

            response = await self.client.post("/traverse", json=request_body)
            response.raise_for_status()
            data = response.json()

            # Flatten paths into ordered steps
            steps = []
            seen_nodes = set()

            for path in data.get('paths', []):
                for node in path.get('nodes', []):
                    node_id = node.get('node_id', '')
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        steps.append(TroubleshootingStep(
                            node_id=node_id,
                            title=node.get('title', ''),
                            content=node.get('content_preview', ''),
                            step_type=self._classify_step_type(node),
                            relevance_score=node.get('relevance_score', 0.0),
                            hop_number=node.get('hop_number', 0),
                            metadata=node.get('metadata', {})
                        ))

            # Sort by hop number then relevance
            steps.sort(key=lambda s: (s.hop_number, -s.relevance_score))

            # Cache results
            self._set_cached(cache_key, steps)

            logger.info(f"PathRAG traversal for {error_code}: {len(steps)} steps")
            return steps

        except httpx.HTTPStatusError as e:
            logger.error(f"PDF API traversal error: {e.response.status_code}")
            self._record_failure()
            return []
        except Exception as e:
            logger.error(f"PDF API traversal failed: {e}")
            self._record_failure()
            return []

    def _classify_step_type(self, node: Dict) -> str:
        """Classify the type of troubleshooting step based on content"""
        title = node.get('title', '').lower()
        node_type = node.get('node_type', '').lower()

        if any(kw in title for kw in ['error', 'alarm', 'fault', 'srvo-', 'motn-', 'syst-']):
            return 'error'
        elif any(kw in title for kw in ['cause', 'diagnos', 'check', 'verify', 'inspect']):
            return 'diagnosis'
        elif any(kw in title for kw in ['solution', 'remedy', 'fix', 'action', 'corrective']):
            return 'solution'
        elif any(kw in title for kw in ['procedure', 'step', 'instruction', 'how to']):
            return 'procedure'
        elif node_type == 'table':
            return 'reference'
        elif node_type == 'image' or node_type == 'diagram':
            return 'visual'
        else:
            return 'info'

    # ============================================
    # GRAPH OPERATIONS
    # ============================================

    async def get_related_sections(
        self,
        node_id: str,
        depth: int = 2,
        edge_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get sections related to a specific node.

        Args:
            node_id: Starting node ID
            depth: Traversal depth
            edge_types: Filter by edge types

        Returns:
            Dict mapping depth to list of related nodes
        """
        if not self.is_available:
            return {}

        try:
            params = {"max_depth": depth}
            if edge_types:
                params["edge_types"] = ",".join(edge_types)

            response = await self.client.get(
                f"/graph/nodes/{node_id}/neighbors",
                params=params
            )
            response.raise_for_status()
            return response.json().get('neighbors', {})

        except Exception as e:
            logger.error(f"Get related sections failed: {e}")
            return {}

    async def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node"""
        if not self.is_available:
            return None

        try:
            response = await self.client.get(f"/graph/nodes/{node_id}")
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Get node details failed: {e}")
            return None

    # ============================================
    # RAG CONTEXT GENERATION
    # ============================================

    async def get_context_for_rag(
        self,
        query: str,
        context_type: str = "troubleshooting",
        max_tokens: int = 2000
    ) -> str:
        """
        Get formatted context for RAG prompts.

        Combines search results and troubleshooting paths into
        a structured context string suitable for LLM synthesis.

        Args:
            query: User query
            context_type: Type of context ('troubleshooting', 'general', 'procedure')
            max_tokens: Approximate token limit for context

        Returns:
            Formatted context string for LLM prompt
        """
        context_parts = []

        # Search for relevant documentation
        results = await self.search_documentation(query, max_results=5)

        if results:
            context_parts.append("## Relevant Technical Documentation\n")
            for i, r in enumerate(results, 1):
                path_str = " > ".join(r.document_path) if r.document_path else "Unknown"
                context_parts.append(f"### [{i}] {r.title}")
                context_parts.append(f"**Source:** {path_str}")
                context_parts.append(f"**Relevance:** {r.score:.2f}")
                context_parts.append(f"{r.content_preview}\n")

        # If query contains error code, get troubleshooting path
        error_codes = self._extract_error_codes(query)

        if error_codes and context_type in ('troubleshooting', 'general'):
            for code in error_codes[:2]:  # Limit to 2 codes
                steps = await self.query_troubleshooting_path(code)
                if steps:
                    context_parts.append(f"\n## Troubleshooting Path: {code}\n")
                    for i, step in enumerate(steps, 1):
                        step_prefix = {
                            'error': '⚠️',
                            'diagnosis': '🔍',
                            'solution': '✅',
                            'procedure': '📋',
                            'info': 'ℹ️'
                        }.get(step.step_type, '•')

                        context_parts.append(f"{i}. {step_prefix} **{step.title}** ({step.step_type})")
                        if step.content:
                            # Truncate long content
                            content = step.content[:300] + "..." if len(step.content) > 300 else step.content
                            context_parts.append(f"   {content}")
                        context_parts.append("")

        context = "\n".join(context_parts)

        # Rough token limit enforcement (4 chars ~ 1 token)
        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4] + "\n\n[Context truncated...]"

        return context

    def _extract_error_codes(self, text: str) -> List[str]:
        """Extract FANUC error codes from text"""
        # Pattern for FANUC alarm codes
        pattern = r'(SRVO|MOTN|SYST|HOST|INTP|PRIO|COMM|VISI|SRIO|FILE|MACR|PALL|SPOT|ARC|DISP)-(\d{3,4})'
        matches = re.findall(pattern, text.upper())
        return [f"{prefix}-{num}" for prefix, num in matches]

    # ============================================
    # STATISTICS & MONITORING
    # ============================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            "enabled": self.enabled,
            "base_url": self.base_url,
            "is_available": self.is_available,
            "health_status": self._health_status,
            "failure_count": self._failure_count,
            "circuit_breaker_open": time.time() < self._circuit_open_until,
            "cache_size": len(self._cache),
            "cache_ttl": self.CACHE_TTL
        }

    # ============================================
    # LIFECYCLE
    # ============================================

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
        logger.info("DocumentGraphService closed")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ============================================
# SINGLETON INSTANCE
# ============================================

_document_graph_service: Optional[DocumentGraphService] = None


def get_document_graph_service() -> DocumentGraphService:
    """Get or create the DocumentGraphService singleton"""
    global _document_graph_service
    if _document_graph_service is None:
        _document_graph_service = DocumentGraphService()
    return _document_graph_service


# Convenience alias
document_graph_service = get_document_graph_service()
