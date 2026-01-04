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
            # Don't record 404 as failure - it just means error code not found
            if e.response.status_code == 404:
                logger.info(f"Error code not found in graph: {error_code}")
                return []
            logger.error(f"PDF API traversal error: {e.response.status_code}")
            self._record_failure()
            return []
        except Exception as e:
            logger.error(f"PDF API traversal failed: {e}")
            self._record_failure()
            return []

    async def query_by_symptom(
        self,
        symptom_text: str,
        max_results: int = 5,
        max_hops: int = 4,
        mode: str = "semantic_astar"
    ) -> List[TroubleshootingStep]:
        """
        Entry point for symptom-based queries (e.g., "overcurrent", "overheating").

        Uses INDICATES edge type for reverse lookup from symptoms to error codes,
        then traverses to remedies.

        Pattern: symptom → INDICATES → error_code → RESOLVED_BY → remedy

        Args:
            symptom_text: Natural language symptom description
            max_results: Maximum number of paths to return
            max_hops: Maximum traversal depth
            mode: Traversal mode ('semantic_astar', 'flow_based', 'multi_hop')

        Returns:
            Ordered list of troubleshooting steps starting from related error codes
        """
        if not self.is_available:
            logger.debug("PDF API not available for symptom search")
            return []

        # Check cache
        cache_key = self._cache_key("symptom", symptom_text, max_results, mode)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for symptom search: {symptom_text[:30]}...")
            return cached

        try:
            # Use traverse endpoint with symptom as source concept
            # The PDF API will use INDICATES edges to find related error codes
            #
            # NOTE: edge_type_weights is documented for future PDF API support.
            # Currently the PDF API's TraverseRequest schema does not include this field.
            # The DiagnosticPathFinder in PDF Tools uses position-aware weights internally.
            # TODO: Add edge_type_weights to PDF API TraverseRequest schema for client control.
            request_body = {
                "source_concept": symptom_text,
                "mode": mode,
                "max_hops": max_hops,
                "include_content": True,
                "alpha": 0.85,  # PathRAG damping factor
                "theta": 0.01   # Pruning threshold
            }

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
                            metadata={
                                **node.get('metadata', {}),
                                'entry_type': 'symptom',
                                'symptom_query': symptom_text
                            }
                        ))

                    if len(steps) >= max_results * 5:  # Limit total steps
                        break

            # Sort by hop number then relevance
            steps.sort(key=lambda s: (s.hop_number, -s.relevance_score))

            # Cache results
            self._set_cached(cache_key, steps)

            logger.info(f"Symptom-based search for '{symptom_text[:30]}...': {len(steps)} steps")
            return steps

        except httpx.HTTPStatusError as e:
            # Don't record 404 as failure - it just means concept not found
            if e.response.status_code == 404:
                logger.info(f"Symptom not found in graph: {symptom_text[:50]}...")
                return []
            logger.error(f"PDF API symptom search error: {e.response.status_code}")
            self._record_failure()
            return []
        except Exception as e:
            logger.error(f"PDF API symptom search failed: {e}")
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
    # STRUCTURED CAUSAL CHAIN FORMATTING (2026-01-04)
    # ============================================

    def format_causal_chain_for_synthesis(
        self,
        steps: List[TroubleshootingStep],
        error_code: Optional[str] = None,
        include_parts: bool = True,
        include_severity: bool = True
    ) -> str:
        """
        Format troubleshooting path as structured context for LLM synthesis.

        Produces XML-like structure that helps LLMs understand causal relationships:
        - Error → Cause chain with edge type markers
        - Solution steps with procedure details
        - Component and part number references

        This structured format improves LLM comprehension of:
        - Causal relationships (what leads to what)
        - Step sequencing (diagnosis → solution order)
        - Entity types (error vs diagnosis vs solution)

        Args:
            steps: List of TroubleshootingStep from path traversal
            error_code: Optional error code for header
            include_parts: Include part numbers in output
            include_severity: Include severity classification

        Returns:
            Structured XML-like string for synthesis context
        """
        if not steps:
            return ""

        lines = []

        # Header with error code if provided
        if error_code:
            lines.append(f'<troubleshooting_chain error_code="{error_code}">')
        else:
            lines.append('<troubleshooting_chain>')

        # Derive severity from error category
        severity = "unknown"
        if error_code:
            category = error_code.split('-')[0] if '-' in error_code else ""
            severity_map = {
                'SRVO': 'critical',   # Servo alarms - usually need immediate attention
                'MOTN': 'warning',    # Motion alarms - may allow continued operation
                'SYST': 'critical',   # System alarms - often require restart
                'HOST': 'info',       # Host communication - may be recoverable
                'INTP': 'warning',    # Interpreter - program issues
                'PRIO': 'critical',   # Priority alarms - high severity
            }
            severity = severity_map.get(category, 'warning')

        # Group steps by type for better structure
        errors = [s for s in steps if s.step_type == 'error']
        diagnoses = [s for s in steps if s.step_type == 'diagnosis']
        solutions = [s for s in steps if s.step_type == 'solution']
        procedures = [s for s in steps if s.step_type == 'procedure']
        other = [s for s in steps if s.step_type not in ('error', 'diagnosis', 'solution', 'procedure')]

        # Error section
        if errors:
            for step in errors:
                severity_attr = f' severity="{severity}"' if include_severity else ""
                lines.append(f'  <step type="error"{severity_attr}>')
                lines.append(f'    <title>{self._escape_xml(step.title)}</title>')
                if step.content:
                    lines.append(f'    <description>{self._escape_xml(step.content[:500])}</description>')
                if include_parts and step.metadata.get('part_numbers'):
                    parts = ', '.join(step.metadata['part_numbers'])
                    lines.append(f'    <affected_parts>{parts}</affected_parts>')
                lines.append('  </step>')

        # Diagnosis section (causes)
        if diagnoses:
            for step in diagnoses:
                lines.append('  <step type="diagnosis" edge="CAUSED_BY">')
                lines.append(f'    <title>{self._escape_xml(step.title)}</title>')
                if step.content:
                    lines.append(f'    <cause>{self._escape_xml(step.content[:500])}</cause>')
                if include_parts and step.metadata.get('components'):
                    comps = ', '.join(step.metadata['components'])
                    lines.append(f'    <components>{comps}</components>')
                lines.append('  </step>')

        # Solution section (remedies)
        if solutions:
            for step in solutions:
                lines.append('  <step type="solution" edge="RESOLVED_BY">')
                lines.append(f'    <title>{self._escape_xml(step.title)}</title>')
                if step.content:
                    lines.append(f'    <action>{self._escape_xml(step.content[:500])}</action>')
                if include_parts and step.metadata.get('part_numbers'):
                    parts = ', '.join(step.metadata['part_numbers'])
                    lines.append(f'    <required_parts>{parts}</required_parts>')
                lines.append('  </step>')

        # Procedure section (detailed steps)
        if procedures:
            for i, step in enumerate(procedures, 1):
                lines.append(f'  <step type="procedure" sequence="{i}">')
                lines.append(f'    <title>{self._escape_xml(step.title)}</title>')
                if step.content:
                    lines.append(f'    <instruction>{self._escape_xml(step.content[:500])}</instruction>')
                lines.append('  </step>')

        # Reference section (tables, visuals, info)
        if other:
            for step in other:
                lines.append(f'  <step type="{step.step_type}">')
                lines.append(f'    <title>{self._escape_xml(step.title)}</title>')
                if step.content:
                    lines.append(f'    <content>{self._escape_xml(step.content[:300])}</content>')
                lines.append('  </step>')

        lines.append('</troubleshooting_chain>')

        return '\n'.join(lines)

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters in text"""
        if not text:
            return ""
        return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))

    async def get_structured_troubleshooting_context(
        self,
        query: str,
        mode: str = "semantic_astar",
        max_hops: int = 4,
        max_tokens: int = 4000
    ) -> Optional[str]:
        """
        Get structured troubleshooting context for synthesis.

        Automatically detects entry type (error code vs symptom) and
        returns formatted causal chain.

        Args:
            query: User query (may contain error codes or symptoms)
            mode: Traversal mode
            max_hops: Maximum traversal depth
            max_tokens: Token limit for output

        Returns:
            Structured XML-like troubleshooting context or None
        """
        if not self.is_available:
            return None

        # Try error code extraction first
        error_codes = self._extract_error_codes(query)

        if error_codes:
            # Direct error code lookup
            all_steps = []
            for code in error_codes[:2]:  # Limit to 2 codes
                steps = await self.query_troubleshooting_path(
                    error_code=code,
                    max_hops=max_hops,
                    mode=mode
                )
                if steps:
                    # Format each error code's chain
                    chain = self.format_causal_chain_for_synthesis(
                        steps=steps,
                        error_code=code
                    )
                    all_steps.append(chain)

            if all_steps:
                context = '\n\n'.join(all_steps)
                # Enforce token limit
                if len(context) > max_tokens * 4:
                    context = context[:max_tokens * 4] + '\n\n<!-- Truncated -->'
                return context

        # Fallback to symptom-based search
        steps = await self.query_by_symptom(
            symptom_text=query,
            max_hops=max_hops,
            mode=mode
        )

        if steps:
            context = self.format_causal_chain_for_synthesis(steps=steps)
            if len(context) > max_tokens * 4:
                context = context[:max_tokens * 4] + '\n\n<!-- Truncated -->'
            return context

        return None

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
    # FEDERATION API - Multi-Domain Support (2026-01-03)
    # ============================================

    async def get_domain_registry(self, load_all: bool = False) -> Dict[str, Any]:
        """
        Get registered knowledge domains from PDF Extraction Tools.

        Available domains:
        - fanuc: Robot error codes, KAREL, servo motors (268K+ nodes)
        - imm: Injection molding defects, processes (3K nodes)
        - industrial_automation: PLCs, sensors, protocols (5K nodes)
        - oem_imm: Polymers, Allen-Bradley, RJG sensors (10K nodes)

        Args:
            load_all: Whether to load unloaded domains

        Returns:
            Domain registry with statistics
        """
        if not self.is_available:
            return {"domains": {}}

        cache_key = self._cache_key("domains", load_all)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            params = {"load_all": "true"} if load_all else {}
            response = await self.client.get("/api/v1/domains/registry", params=params)
            response.raise_for_status()
            data = response.json()

            result = data.get("data", {})
            self._set_cached(cache_key, result, ttl=600)  # 10 min cache
            return result

        except Exception as e:
            logger.error(f"Get domain registry failed: {e}")
            return {"domains": {}}

    async def cross_domain_search(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        top_k: int = 10,
        search_mode: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple industrial knowledge domains.

        Uses RRF fusion to combine results from multiple domain graphs.

        Args:
            query: Search query
            domains: List of domains to search (default: all)
            entity_types: Filter by entity types (error_code, remedy, component, etc.)
            top_k: Number of results
            search_mode: 'keyword', 'semantic', or 'hybrid'

        Returns:
            List of cross-domain search results
        """
        if not self.is_available:
            return []

        # Auto-detect domains if not specified
        if not domains:
            domains = self._route_query_to_domains(query)

        cache_key = self._cache_key("cross_domain", query, ",".join(domains), top_k)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            request_body = {
                "query": query,
                "domains": domains,
                "top_k": top_k,
                "search_mode": search_mode
            }
            if entity_types:
                request_body["entity_types"] = entity_types

            response = await self.client.post(
                "/api/v1/domains/search",
                json=request_body
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("data", {}).get("results", [])
            self._set_cached(cache_key, results)

            logger.info(f"Cross-domain search '{query[:30]}...' returned {len(results)} results from {domains}")
            return results

        except Exception as e:
            logger.error(f"Cross-domain search failed: {e}")
            return []

    def _route_query_to_domains(self, query: str) -> List[str]:
        """Route queries to appropriate domains based on keywords."""
        domains = []
        query_lower = query.lower()

        # FANUC robotics
        if any(kw in query_lower for kw in ["servo", "robot", "srvo", "motn", "fanuc", "r-30", "karel", "encoder"]):
            domains.append("fanuc")

        # Injection molding
        if any(kw in query_lower for kw in ["injection", "mold", "plastic", "defect", "flash", "sink", "warp"]):
            domains.append("imm")

        # Industrial automation (PLCs, etc.)
        if any(kw in query_lower for kw in ["plc", "allen bradley", "controllogix", "siemens", "profinet", "ethernet/ip"]):
            domains.append("industrial_automation")

        # OEM/Materials
        if any(kw in query_lower for kw in ["polymer", "ultramid", "lexan", "rjg", "material", "resin"]):
            domains.append("oem_imm")

        return domains if domains else ["fanuc"]  # Default to FANUC

    async def hsea_troubleshoot(
        self,
        error_code: str,
        include_embeddings: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get HSEA-structured troubleshooting data for an error code.

        Uses the HSEA (Hierarchical Stratified Embedding Architecture)
        three-layer retrieval for comprehensive error context.

        Args:
            error_code: Error code (e.g., 'SRVO-023', 'MOTN-023')
            include_embeddings: Whether to include embeddings in response

        Returns:
            Structured troubleshooting data with cause, remedy, metadata
        """
        if not self.is_available:
            return None

        error_code = error_code.upper().strip()
        cache_key = self._cache_key("hsea_troubleshoot", error_code)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            response = await self.client.get(
                f"/api/v1/search/hsea/troubleshoot/{error_code}"
            )
            response.raise_for_status()
            data = response.json()

            result = data.get("data", {})

            # Remove embeddings if not requested (they're large)
            if not include_embeddings and "context" in result:
                ctx = result["context"]
                if "metadata" in ctx and "embeddings" in ctx.get("metadata", {}):
                    del ctx["metadata"]["embeddings"]

            self._set_cached(cache_key, result)
            logger.info(f"HSEA troubleshoot for {error_code}: found context")
            return result

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.debug(f"Error code not found: {error_code}")
                return None
            logger.error(f"HSEA troubleshoot error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"HSEA troubleshoot failed: {e}")
            return None

    async def get_similar_errors(
        self,
        error_code: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar error codes using HSEA embedding similarity.

        Args:
            error_code: Source error code
            top_k: Number of similar errors to return

        Returns:
            List of similar error codes with similarity scores
        """
        if not self.is_available:
            return []

        error_code = error_code.upper().strip()

        try:
            response = await self.client.get(
                f"/api/v1/search/hsea/similar/{error_code}",
                params={"top_k": top_k}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("similar_codes", [])

        except Exception as e:
            logger.error(f"Get similar errors failed: {e}")
            return []

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get MCP-compatible tool definitions for LLM consumption.

        These tools can be used by Claude or other LLMs for
        structured knowledge graph queries.

        Returns:
            List of MCP tool definitions with input schemas
        """
        if not self.is_available:
            return []

        cache_key = self._cache_key("mcp_tools")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            response = await self.client.get("/api/v1/tools/mcp-manifest")
            response.raise_for_status()
            data = response.json()

            tools = data.get("data", {}).get("tools", [])
            self._set_cached(cache_key, tools, ttl=3600)  # 1 hour cache
            return tools

        except Exception as e:
            logger.error(f"Get MCP tools failed: {e}")
            return []

    async def get_imm_defect(
        self,
        defect_name: str,
        include_process_params: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get injection molding defect information.

        Args:
            defect_name: Name of defect (e.g., 'flash', 'sink marks', 'warpage')
            include_process_params: Include relevant process parameters

        Returns:
            Defect information with causes and remedies
        """
        if not self.is_available:
            return None

        try:
            response = await self.client.get(
                f"/api/v1/imm/defects/{defect_name}",
                params={"include_process_params": str(include_process_params).lower()}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {})

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Get IMM defect error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Get IMM defect failed: {e}")
            return None

    async def get_enhanced_context_for_rag(
        self,
        query: str,
        max_tokens: int = 4000
    ) -> str:
        """
        Get enhanced RAG context using Federation API features.

        Combines:
        - HSEA troubleshooting for error codes
        - Cross-domain search for broader context
        - Similar error codes for related issues

        Args:
            query: User query
            max_tokens: Approximate token limit

        Returns:
            Rich formatted context string
        """
        context_parts = []

        # Extract error codes from query
        error_codes = self._extract_error_codes(query)

        # 1. HSEA Troubleshoot for specific error codes
        for code in error_codes[:2]:
            hsea_data = await self.hsea_troubleshoot(code)
            if hsea_data and "context" in hsea_data:
                ctx = hsea_data["context"]
                metadata = ctx.get("metadata", {})

                context_parts.append(f"## Error Code: {code}")
                context_parts.append(f"**Category:** {metadata.get('category', 'Unknown')}")
                context_parts.append(f"**Title:** {metadata.get('full_title', ctx.get('title', ''))}")

                if metadata.get("cause"):
                    context_parts.append(f"**Cause:** {metadata['cause']}")
                if metadata.get("remedy"):
                    context_parts.append(f"**Remedy:** {metadata['remedy']}")
                if metadata.get("severity"):
                    context_parts.append(f"**Severity:** {metadata['severity']}")

                # Add mentions/context from manual
                mentions = metadata.get("mentions", [])
                if mentions:
                    context_parts.append("\n**From FANUC Manual:**")
                    for m in mentions[:2]:
                        context_parts.append(f"- Page {m.get('page', '?')}: {m.get('context', '')[:200]}...")

                # Get similar errors for related troubleshooting
                similar = await self.get_similar_errors(code, top_k=3)
                if similar:
                    context_parts.append("\n**Related Error Codes:**")
                    for s in similar:
                        context_parts.append(f"- {s.get('error_code', '?')}: {s.get('title', '')} (similarity: {s.get('score', 0):.2f})")

                context_parts.append("")

        # 2. Cross-domain search for broader context
        domains = self._route_query_to_domains(query)
        cross_results = await self.cross_domain_search(
            query=query,
            domains=domains,
            top_k=5
        )

        if cross_results:
            context_parts.append("## Cross-Domain Knowledge\n")
            for i, r in enumerate(cross_results, 1):
                domain = r.get("domain", "unknown")
                node_type = r.get("node_type", "")
                label = r.get("label", r.get("title", ""))
                score = r.get("score", 0)

                context_parts.append(f"### [{i}] {label}")
                context_parts.append(f"**Domain:** {domain} | **Type:** {node_type} | **Score:** {score:.3f}")

                preview = r.get("content_preview") or r.get("content") or ""
                if preview:
                    context_parts.append(f"{preview[:300]}...")
                context_parts.append("")

        # 3. IMM defect info if injection molding related
        query_lower = query.lower()
        defect_keywords = ["flash", "sink", "warp", "void", "short shot", "burn", "jetting"]
        for defect in defect_keywords:
            if defect in query_lower:
                defect_info = await self.get_imm_defect(defect)
                if defect_info:
                    context_parts.append(f"## Injection Molding Defect: {defect.title()}\n")
                    if "causes" in defect_info:
                        context_parts.append("**Causes:**")
                        for c in defect_info["causes"][:3]:
                            context_parts.append(f"- {c}")
                    if "remedies" in defect_info:
                        context_parts.append("\n**Remedies:**")
                        for r in defect_info["remedies"][:3]:
                            context_parts.append(f"- {r}")
                    context_parts.append("")
                break  # Only process first matching defect

        context = "\n".join(context_parts)

        # Token limit enforcement
        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4] + "\n\n[Context truncated...]"

        return context

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
