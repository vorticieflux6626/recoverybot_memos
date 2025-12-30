"""
Searcher Agent - Web Search Execution

Implements multi-provider web search with fallback:
1. SearXNG (primary, self-hosted, no rate limits)
2. DuckDuckGo HTML (secondary, no API key needed)
3. Brave Search API (tertiary, requires API key)

Includes result scoring and domain filtering for research-relevant content.
Features semantic relevance filtering to prevent irrelevant results from
trusted domains being boosted incorrectly.
"""

import asyncio
import logging
import re
import string
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

import httpx

from .models import WebSearchResult
from .search_metrics import get_search_metrics

# Lazy settings import to avoid circular dependencies
_settings = None
def _get_settings():
    global _settings
    if _settings is None:
        from config.settings import get_settings
        _settings = get_settings()
    return _settings

logger = logging.getLogger("agentic.searcher")


class SearchProvider(ABC):
    """Base class for search providers"""

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """Search for the query and return results. Must be implemented by subclasses."""
        pass


class SearXNGSearchProvider(SearchProvider):
    """
    SearXNG self-hosted metasearch provider.

    Primary provider with no rate limits.
    Aggregates results from Brave, Bing, Startpage, DuckDuckGo, etc.

    NOTE: Google engine disabled due to upstream bug #5286 (Oct 2025).
    Revisit in Jan 2026 for potential fix.

    Supports dynamic engine selection based on query type:
    - Academic: arxiv, semantic_scholar, pubmed, crossref
    - Technical: github, gitlab, stackoverflow, pypi, npm
    - General: brave, bing, startpage, duckduckgo
    - FANUC/Robotics: reddit, brave, electronics/robotics stackexchange
    - Q&A: stackoverflow, superuser, askubuntu, serverfault
    """

    # Cache TTL for availability check (seconds)
    AVAILABILITY_CACHE_TTL = 60  # Re-check every 60 seconds

    # Engine groups for different query types
    ENGINE_GROUPS = {
        # Primary engines: Brave (1.5 weight), Bing (1.2), Startpage (1.1), DDG (1.0)
        # Google disabled due to upstream bug #5286 - revisit Jan 2026
        "general": "brave,bing,startpage,duckduckgo,wikipedia",
        "academic": "arxiv,semantic_scholar,pubmed,crossref,wikipedia",
        "technical": "github,gitlab,stackoverflow,superuser,serverfault,pypi,npm,dockerhub,bing",
        "news": "bing_news,duckduckgo",
        "all": "brave,bing,startpage,duckduckgo,wikipedia,arxiv,semantic_scholar,github,stackoverflow",
        # FANUC/Industrial robotics - Reddit + Stack Exchange for troubleshooting
        "fanuc": "reddit,brave,bing,startpage,arxiv,electronics_stackexchange,robotics_stackexchange",
        "robotics": "reddit,brave,bing,arxiv,github,gitlab,robotics_stackexchange,electronics_stackexchange",
        # Q&A focused
        "qa": "stackoverflow,superuser,askubuntu,serverfault,unix_stackexchange,reddit",
        # Linux/sysadmin
        "linux": "askubuntu,unix_stackexchange,serverfault,arch_linux_wiki,gentoo,reddit,bing",
        # Package/library search
        "packages": "pypi,npm,crates,pkg_go_dev,dockerhub",
        # Injection Molding Machines (IMM) / Euromap
        "imm": "reddit,brave,bing,startpage,wikipedia",
        "euromap": "reddit,brave,bing,startpage,wikipedia",
        "plastics": "reddit,brave,bing,startpage,wikipedia",
    }

    # Patterns to detect academic queries
    ACADEMIC_PATTERNS = [
        r"\bpaper\b", r"\bresearch\b", r"\bstudy\b", r"\bjournal\b",
        r"\barxiv\b", r"\bcitation\b", r"\bpublish", r"\bpeer.?review",
        r"\babstract\b", r"\bsurvey\b", r"\bstate.of.the.art\b",
        r"\bnovel\b.*\bmethod", r"\bproposed\b", r"\bexperiment",
        r"\bbaseline\b", r"\bbenchmark\b", r"\bdataset\b"
    ]

    # Patterns to detect technical queries
    TECHNICAL_PATTERNS = [
        r"\bcode\b", r"\bapi\b", r"\blibrary\b", r"\bframework\b",
        r"\bbug\b", r"\berror\b", r"\bimplement", r"\bfunction\b",
        r"\bclass\b", r"\bmethod\b", r"\bpackage\b", r"\binstall\b",
        r"\bdocker\b", r"\bkubernetes\b", r"\bpython\b", r"\brust\b",
        r"\bjavascript\b", r"\btypescript\b", r"\bgithub\b", r"\bnpm\b",
        r"\bpip\b", r"\bcargo\b", r"\bdependenc"
    ]

    # Patterns to detect FANUC/industrial robotics queries
    FANUC_PATTERNS = [
        # FANUC alarm codes
        r"\bsrvo-\d+", r"\bsyst-\d+", r"\balrm-\d+", r"\bcomm-\d+",
        r"\bintp-\d+", r"\bvisi-\d+", r"\bprio-\d+", r"\bsrio-\d+",
        # Brand and model names
        r"\bfanuc\b", r"\br-30i[ab]\b", r"\br-2000i[abc]\b", r"\bm-\d+i[abc]",
        r"\blr\s*mate\b", r"\barc\s*mate\b", r"\bpaint\s*mate\b",
        # Components and procedures
        r"\bpulsecoder\b", r"\bservo\s+amplifier\b", r"\bteach\s+pendant\b",
        r"\bmastering\b", r"\brcal\b", r"\bcollision\s+detect",
        r"\bkarl\b", r"\bkarl\s+program", r"\btp\s+program",
        # Parameters and system variables
        r"\$mcr\.", r"\$param", r"\$spc_reset", r"\$master",
        # Industrial terms in robotics context
        r"\bovertravel\b", r"\bjoint\s*\d+\b", r"\baxis\s*\d+\b",
        r"\brobot\s+arm\b", r"\bend\s+effector\b", r"\btool\s+center\s+point\b"
    ]

    # Patterns to detect Injection Molding Machine (IMM) queries
    IMM_PATTERNS = [
        # Euromap protocols
        r"\beuromap\s*(6[7-9]|7\d|8\d)", r"\bem\s*6[7-9]", r"\bspi\s+interface",
        # Machine manufacturers
        r"\bkraussmaffei\b", r"\bmilacron\b", r"\bcincinnati\b", r"\bvan\s*dorn\b",
        r"\bsumitomo\b", r"\bdemag\b", r"\bel-exis\b", r"\bintellect\b",
        # Control systems
        r"\bmc[3456]\b", r"\bmosaic\+?\b", r"\bpathfinder\b", r"\bacramatic\b",
        # Machine models
        r"\bvista\s+toggle\b", r"\barrow\b.*\bimm\b", r"\bhawk\s*\d+",
        r"\bmx\s*\d{3,4}\b", r"\bcx\s*\d+\b", r"\bgx\s*\d+\b", r"\bpx\s*\d+\b",
        # Process terms
        r"\binjection\s+mold", r"\bmoulding\s+machine\b", r"\bimm\b",
        r"\bclamp\s+tonnage\b", r"\bshot\s+size\b", r"\bplasticiz",
        r"\bscrew\s+(tip|check\s+ring|recovery)", r"\bbarrel\s+(heater|zone)",
        # Defects
        r"\bshort\s+shot\b", r"\bsink\s+mark\b", r"\bweld\s+line\b",
        r"\bflash\b.*\bmold", r"\bburn\s+mark\b", r"\bwarpage\b", r"\bvoid\b",
        # Scientific molding
        r"\bdecoupled\s+(molding|i+)\b", r"\bcavity\s+pressure\b",
        r"\brjg\b", r"\bedart\b", r"\bcopilot\b",
        # Euromap signals
        r"\bmould\s+(open|close)", r"\brobot\s+ready\b", r"\bcycle\s+start\b",
    ]

    def __init__(self, base_url: Optional[str] = None):
        settings = _get_settings()
        self.base_url = (base_url or settings.searxng_url).rstrip("/")
        self._available = None  # Cache availability check
        self._available_checked_at = 0  # Timestamp of last check
        self._client: Optional[httpx.AsyncClient] = None  # Shared HTTP client

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @property
    def available(self) -> bool:
        """Check if SearXNG is available (cached)"""
        return self._available if self._available is not None else True

    def detect_query_type(self, query: str) -> str:
        """
        Detect query type based on patterns.

        Returns: 'fanuc', 'imm', 'academic', 'technical', or 'general'
        """
        query_lower = query.lower()

        # Count pattern matches for each category
        fanuc_score = sum(
            1 for p in self.FANUC_PATTERNS
            if re.search(p, query_lower, re.IGNORECASE)
        )
        imm_score = sum(
            1 for p in self.IMM_PATTERNS
            if re.search(p, query_lower, re.IGNORECASE)
        )
        academic_score = sum(
            1 for p in self.ACADEMIC_PATTERNS
            if re.search(p, query_lower, re.IGNORECASE)
        )
        technical_score = sum(
            1 for p in self.TECHNICAL_PATTERNS
            if re.search(p, query_lower, re.IGNORECASE)
        )

        # FANUC queries take priority - they're very specific
        if fanuc_score >= 1:
            logger.info(f"Detected FANUC query (score={fanuc_score}): {query[:50]}")
            return "fanuc"

        # IMM/Euromap queries are also very specific
        if imm_score >= 1:
            logger.info(f"Detected IMM query (score={imm_score}): {query[:50]}")
            return "imm"

        # Determine type based on scores
        if academic_score >= 2:
            return "academic"
        elif technical_score >= 2:
            return "technical"
        elif academic_score == 1 and technical_score == 0:
            return "academic"
        elif technical_score == 1 and academic_score == 0:
            return "technical"
        else:
            return "general"

    def get_engines_for_query(self, query: str, query_type: Optional[str] = None) -> str:
        """
        Get appropriate engines for a query, filtering out rate-limited engines.

        Args:
            query: The search query
            query_type: Optional override ('academic', 'technical', 'general', 'all')

        Returns:
            Comma-separated engine list (rate-limited engines excluded)
        """
        if query_type is None:
            query_type = self.detect_query_type(query)

        base_engines = self.ENGINE_GROUPS.get(query_type, self.ENGINE_GROUPS["general"])
        engine_list = [e.strip() for e in base_engines.split(",")]

        # Filter out engines in backoff using metrics
        metrics = get_search_metrics()
        available_engines, skipped = metrics.get_available_engines(engine_list)

        if skipped:
            for engine, reason in skipped.items():
                logger.info(f"Skipping engine {engine}: {reason}")

        if not available_engines:
            # All engines in backoff - use original list (will likely fail but try anyway)
            logger.warning(f"All engines in backoff, using original list: {engine_list}")
            return base_engines

        return ",".join(available_engines)

    async def check_availability(self) -> bool:
        """Check if SearXNG server is responding (with TTL cache)"""
        import time
        current_time = time.time()

        # Use cached result if within TTL and was available
        if self._available is not None:
            cache_age = current_time - self._available_checked_at
            if cache_age < self.AVAILABILITY_CACHE_TTL:
                return self._available
            # If cached as unavailable, re-check more aggressively (every 10s)
            if not self._available and cache_age < 10:
                return self._available

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/search",
                params={"q": "test", "format": "json"},
                timeout=5.0  # Override timeout for availability check
            )
            self._available = response.status_code == 200
            self._available_checked_at = current_time
            if self._available:
                logger.info(f"SearXNG available at {self.base_url}")
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning(f"SearXNG not available: {e}")
            self._available = False
            self._available_checked_at = current_time

        return self._available

    async def search(
        self,
        query: str,
        max_results: int = 10,
        query_type: Optional[str] = None,
        engines: Optional[str] = None
    ) -> List[WebSearchResult]:
        """
        Search using SearXNG with dynamic engine selection.

        Args:
            query: Search query string
            max_results: Maximum number of results
            query_type: Optional type override ('academic', 'technical', 'general', 'all')
            engines: Optional explicit engine list (comma-separated)

        Returns:
            List of WebSearchResult objects
        """
        metrics = get_search_metrics()
        start_time = time.time()

        try:
            # Determine engines to use
            if engines is None:
                engines = self.get_engines_for_query(query, query_type)
                detected_type = query_type or self.detect_query_type(query)
                logger.info(f"Query type detected: {detected_type}, using engines: {engines}")

            client = await self._get_client()
            # SearXNG aggregates multiple pages if we request more results
            # Each page typically has 10 results, so we may need multiple pages
            all_results = []
            seen_urls = set()

            # Request up to 3 pages if needed to get max_results
            pages_needed = min(3, (max_results + 9) // 10)

            for page in range(1, pages_needed + 1):
                response = await client.get(
                    f"{self.base_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "engines": engines,
                        "language": "en-US",
                        "pageno": page
                    }
                )

                if response.status_code != 200:
                    logger.warning(f"SearXNG page {page} failed: {response.status_code}")
                    break

                data = response.json()
                page_results = data.get("results", [])
                logger.debug(f"SearXNG page {page} returned {len(page_results)} results")

                # Process unresponsive engines (rate limits, CAPTCHAs, timeouts)
                unresponsive = data.get("unresponsive_engines", [])
                if unresponsive:
                    metrics.record_unresponsive_engines(unresponsive)
                    logger.info(f"SearXNG unresponsive engines: {unresponsive}")

                # Record successful engines based on results
                engine_result_counts = {}
                for item in page_results:
                    item_engines = item.get("engines", [])
                    for eng in item_engines:
                        engine_result_counts[eng] = engine_result_counts.get(eng, 0) + 1

                for eng, count in engine_result_counts.items():
                    metrics.record_engine_result(eng, success=True, results_count=count)

                for item in page_results:
                    url = item.get("url", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(item)
                        if len(all_results) >= max_results:
                            break

                if len(all_results) >= max_results:
                    break

            # Convert to WebSearchResult objects
            results = []
            for item in all_results[:max_results]:
                url = item.get("url", "")
                domain = urlparse(url).netloc.replace("www.", "")

                # Calculate score based on position and engine count
                engines = item.get("engines", ["searxng"])
                # SearXNG can return scores > 1, cap to 0.9 to leave room for boost
                base_score = min(0.9, item.get("score", 0.7))

                # Boost for multi-engine results (cap total at 1.0)
                if len(engines) > 1:
                    base_score = min(1.0, base_score + 0.05 * len(engines))

                results.append(WebSearchResult(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("content", ""),
                    source_domain=domain,
                    relevance_score=base_score
                ))

            duration_ms = (time.time() - start_time) * 1000
            metrics.record_search(
                "searxng", query, len(results), duration_ms, success=True
            )
            logger.info(f"SearXNG returned {len(results)} results for: {query[:50]}")
            return results

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_search(
                "searxng", query, 0, duration_ms, success=False,
                error=str(type(e).__name__)
            )
            logger.error(f"SearXNG search error: {e}")
            self._available = False
            return []


class BraveSearchProvider(SearchProvider):
    """
    Brave Search API provider.

    Requires BRAVE_API_KEY environment variable.
    Free tier: 2,000 queries/month
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("BRAVE_API_KEY")
        self.available = bool(self.api_key)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10)
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        if not self.available:
            logger.debug("Brave API key not configured")
            return []

        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
            params = {
                "q": query,
                "count": max_results,
                "safesearch": "moderate"
            }

            client = await self._get_client()
            response = await client.get(
                self.BASE_URL,
                headers=headers,
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                for item in data.get("web", {}).get("results", [])[:max_results]:
                    domain = urlparse(item.get("url", "")).netloc
                    results.append(WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                        source_domain=domain,
                        relevance_score=0.8  # Brave results are generally high quality
                    ))

                logger.info(f"Brave search returned {len(results)} results for: {query[:50]}")
                return results
            else:
                logger.warning(f"Brave search failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []


class DuckDuckGoProvider(SearchProvider):
    """
    DuckDuckGo HTML search provider.

    No API key required, but may be rate-limited.
    Uses HTML parsing as DDG doesn't have a public API.
    Features retry logic with exponential backoff on rate limits.
    """

    BASE_URL = "https://html.duckduckgo.com/html/"
    MAX_RETRIES = 2

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10)
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        metrics = get_search_metrics()
        start_time = time.time()

        # Check if we're in backoff period
        available, reason = metrics.is_provider_available("duckduckgo")
        if not available:
            logger.info(f"DuckDuckGo skipped: {reason}")
            return []

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                }
                data = {"q": query}

                client = await self._get_client()
                response = await client.post(
                    self.BASE_URL,
                    headers=headers,
                    data=data,
                    follow_redirects=True
                )

                duration_ms = (time.time() - start_time) * 1000

                if response.status_code == 200:
                    html = response.text
                    results = self._parse_results(html, max_results)
                    metrics.record_search(
                        "duckduckgo", query, len(results), duration_ms, success=True
                    )
                    metrics.reset_rate_limit("duckduckgo")
                    return results

                elif response.status_code == 202:
                    backoff = metrics.record_rate_limit("duckduckgo")
                    if attempt < self.MAX_RETRIES:
                        wait_time = min(backoff, 10)  # Max 10s wait between retries
                        logger.info(f"DuckDuckGo rate limited, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        start_time = time.time()  # Reset timer for next attempt
                        continue
                    metrics.record_search(
                        "duckduckgo", query, 0, duration_ms, success=False,
                        error="rate_limited"
                    )
                    return []

                else:
                    metrics.record_search(
                        "duckduckgo", query, 0, duration_ms, success=False,
                        error=f"http_{response.status_code}"
                    )
                    return []

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                metrics.record_search(
                    "duckduckgo", query, 0, duration_ms, success=False,
                    error=str(type(e).__name__)
                )
                logger.error(f"DuckDuckGo search error: {e}")
                return []

        return []

    def _parse_results(self, html: str, max_results: int) -> List[WebSearchResult]:
        """Parse DuckDuckGo HTML response"""
        results = []

        # Extract result blocks
        result_pattern = r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>'
        matches = re.findall(result_pattern, html, re.DOTALL)

        for url, title, snippet in matches[:max_results]:
            # Clean HTML tags
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()

            # Extract actual URL from DDG redirect
            if '/l/?uddg=' in url:
                url_match = re.search(r'uddg=([^&]+)', url)
                if url_match:
                    from urllib.parse import unquote
                    url = unquote(url_match.group(1))

            domain = urlparse(url).netloc if url.startswith('http') else ''

            if clean_title and clean_snippet:
                results.append(WebSearchResult(
                    title=clean_title,
                    url=url,
                    snippet=clean_snippet,
                    source_domain=domain,
                    relevance_score=0.6
                ))

        logger.info(f"DuckDuckGo returned {len(results)} results")
        return results


class PDFDocumentProvider(SearchProvider):
    """
    PDF Extraction Tools API provider for FANUC technical documentation.

    Searches the local PDF knowledge base via the PDF Extraction Tools API
    running on port 8002. Results are converted to WebSearchResult format
    for seamless integration with the search pipeline.

    Features:
    - PathRAG traversal for troubleshooting paths
    - Entity-aware search (error codes, components, parameters)
    - Circuit breaker pattern for graceful degradation
    - Result caching with configurable TTL
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8002",
        timeout: float = 30.0,
        enabled: bool = True
    ):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.enabled = enabled
        self._client: Optional[httpx.AsyncClient] = None
        self._available: Optional[bool] = None
        self._last_check: float = 0
        self._failure_count: int = 0
        self._max_failures: int = 3

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=10)
            )
        return self._client

    async def is_available(self) -> bool:
        """Check if PDF API is available with caching."""
        if not self.enabled:
            return False

        # Circuit breaker: too many failures
        if self._failure_count >= self._max_failures:
            # Reset after 5 minutes
            if time.time() - self._last_check > 300:
                self._failure_count = 0
            else:
                return False

        # Cache availability for 60 seconds
        if self._available is not None and time.time() - self._last_check < 60:
            return self._available

        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            self._available = response.status_code == 200
            self._last_check = time.time()
            if self._available:
                self._failure_count = 0
            return self._available
        except Exception as e:
            logger.warning(f"PDF API health check failed: {e}")
            self._available = False
            self._last_check = time.time()
            self._failure_count += 1
            return False

    async def search(self, query: str, max_results: int = 10) -> List[WebSearchResult]:
        """
        Search FANUC technical documentation.

        Args:
            query: Search query (may include error codes)
            max_results: Maximum number of results

        Returns:
            List of WebSearchResult from PDF documents
        """
        if not await self.is_available():
            logger.debug("PDF API not available, skipping")
            return []

        try:
            client = await self._get_client()

            # Search the PDF knowledge base
            response = await client.post(
                "/search",
                json={
                    "query": query,
                    "max_results": max_results,
                    "include_context": True
                }
            )

            if response.status_code != 200:
                logger.warning(f"PDF API search failed: {response.status_code}")
                self._failure_count += 1
                return []

            data = response.json()
            results = []

            for doc in data.get("results", []):
                # Convert PDF result to WebSearchResult format
                results.append(WebSearchResult(
                    title=doc.get("title", "FANUC Technical Document"),
                    url=doc.get("source_url", f"pdf://{doc.get('document_id', 'unknown')}"),
                    snippet=doc.get("content", doc.get("snippet", ""))[:500],
                    source_domain="fanuc.pdf.local",
                    relevance_score=doc.get("score", 0.8),
                    # Mark as technical documentation
                    metadata={
                        "source_type": "technical_documentation",
                        "document_type": doc.get("document_type", "manual"),
                        "page_number": doc.get("page_number"),
                        "section": doc.get("section")
                    }
                ))

            logger.info(f"PDF API returned {len(results)} results for: {query[:50]}")
            return results

        except Exception as e:
            logger.error(f"PDF API search error: {e}")
            self._failure_count += 1
            return []

    async def get_troubleshooting_path(
        self,
        error_code: str,
        context: Optional[str] = None
    ) -> List[dict]:
        """
        Get step-by-step troubleshooting path for an error code.

        Uses PathRAG traversal to build resolution steps.

        Args:
            error_code: FANUC error code (e.g., SRVO-063)
            context: Optional context for better path selection

        Returns:
            List of troubleshooting steps with sources
        """
        if not await self.is_available():
            return []

        try:
            client = await self._get_client()

            response = await client.post(
                "/traverse",
                json={
                    "error_code": error_code,
                    "context": context,
                    "max_depth": 5
                }
            )

            if response.status_code != 200:
                logger.warning(f"PDF API traverse failed: {response.status_code}")
                return []

            data = response.json()
            return data.get("steps", [])

        except Exception as e:
            logger.error(f"PDF API traverse error: {e}")
            return []

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class SearcherAgent:
    """
    Main searcher agent that orchestrates web search across providers.

    Features:
    - Multi-provider fallback (SearXNG → DuckDuckGo → Brave)
    - Domain scoring for research-relevant content
    - Result deduplication and scoring
    - Parallel query execution
    """

    # Domains with higher trust for research/technical content
    # Organized by category with scoring tiers
    TRUSTED_DOMAINS = {
        # ===== Academic/Research (Premium) =====
        "arxiv.org",
        "scholar.google.com",
        "semanticscholar.org",
        "researchgate.net",
        "ieee.org",
        "acm.org",
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "wiley.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "jstor.org",
        "ssrn.com",
        "biorxiv.org",
        "medrxiv.org",
        "openreview.net",
        "aclanthology.org",
        "neurips.cc",
        "proceedings.mlr.press",

        # ===== Technical Documentation (Premium) =====
        "docs.python.org",
        "docs.rust-lang.org",
        "docs.oracle.com",
        "docs.microsoft.com",
        "learn.microsoft.com",
        "developer.apple.com",
        "developer.android.com",
        "developer.mozilla.org",
        "devdocs.io",
        "readthedocs.io",
        "readthedocs.org",

        # ===== Code Repositories (Trusted) =====
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "sourceforge.net",
        "codeberg.org",
        "huggingface.co",

        # ===== Q&A / Community (Trusted) =====
        "stackoverflow.com",
        "stackexchange.com",
        "serverfault.com",
        "superuser.com",
        "askubuntu.com",
        "unix.stackexchange.com",
        "cs.stackexchange.com",
        "ai.stackexchange.com",
        "datascience.stackexchange.com",

        # ===== Cloud Providers (Trusted) =====
        "cloud.google.com",
        "aws.amazon.com",
        "docs.aws.amazon.com",
        "azure.microsoft.com",
        "docs.digitalocean.com",
        "docs.cloudflare.com",

        # ===== Framework Documentation (Premium) =====
        "kubernetes.io",
        "docker.com",
        "docs.docker.com",
        "nginx.org",
        "fastapi.tiangolo.com",
        "flask.palletsprojects.com",
        "docs.djangoproject.com",
        "pytorch.org",
        "tensorflow.org",
        "numpy.org",
        "pandas.pydata.org",
        "scikit-learn.org",
        "docs.scipy.org",
        "matplotlib.org",
        "langchain.com",
        "docs.anthropic.com",
        "platform.openai.com",
        "ollama.com",

        # ===== Hardware/Engineering (Trusted) =====
        "electronics.stackexchange.com",
        "hackaday.com",
        "instructables.com",
        "adafruit.com",
        "sparkfun.com",
        "raspberrypi.org",
        "raspberrypi.com",
        "arduino.cc",
        "eevblog.com",
        "allaboutcircuits.com",

        # ===== Injection Molding / Plastics (Premium for IMM) =====
        "ptonline.com",                # Plastics Technology
        "plasticstoday.com",
        "plasticsnews.com",
        "rjginc.com",                  # RJG Scientific Molding
        "traininteractive.com",
        "aim.institute",               # American Injection Molding Institute
        "4spe.org",                    # Society of Plastics Engineers
        "injectionmoldingonline.com",  # IM Forums
        "vitalplastics.com",           # RJG implementation case studies
        "nexeoplastics.com",           # Fill-Pack-Recover optimization
        "elastron.com",                # Defect troubleshooting guides
        "waykenrm.com",                # Molding problems/solutions
        "guanxin-machinery.com",       # Error codes/troubleshooting

        # ===== Machine Manufacturers (Premium for IMM) =====
        "kraussmaffei.com",
        "trainingacademy.kraussmaffei.com",  # MC6 control training
        "press.kraussmaffei.com",      # Brochures and datasheets
        "milacron.com",
        "sumitomo-shi-demag.us",       # Van Dorn/Demag legacy specs
        "sumitomo-shi-demag.eu",
        "euromap.org",                 # Euromap 67/73/77 Standards (Critical)
        "fanuc.eu",                    # FANUC ROBOSHOT specs
        "robot-forum.com",             # FANUC Integration discussions
        "plctalk.net",                 # PLC Programming
        "practicalmachinist.com",
        "eng-tips.com",
        "support.industry.siemens.com", # Siemens PLC troubleshooting

        # ===== Euromap Protocol Resources =====
        "plastech.pl",                 # Euromap 67 mirror
        "astor.com.pl",                # Euromap technical overview
        "zacobria.com",                # Universal Robots Euromap guides
        "machinebuilding.net",         # Robot-IMM safety integration

        # ===== Material Suppliers (Technical Data) =====
        "plastics-rubber.basf.com",    # BASF troubleshooter with photos
        "dupont.com",
        "sabic.com",
        "covestro.com",
        "entecpolymers.com",           # Troubleshooting guide PDFs

        # ===== Mold Components & Standards =====
        "dme.net",                     # Mold component standards
        "hasco.com",                   # 100,000+ standard components

        # ===== Parts, Manuals & Service =====
        "mcspt.com",                   # Cincinnati Milacron manuals
        "industrialmanuals.com",       # Van Dorn, Cincinnati manuals
        "controlrepair.com",           # Control system repair info
        "cincinnatirpt.com",           # 30,000+ Cincinnati parts
        "telarcorp.com",               # Van Dorn parts catalog
        "store.milacron.com",          # OEM parts ordering
        "onlinestore.dpg.com",         # Van Dorn, Demag, Newbury parts
        "wisrepair.com",               # PathFinder controller repair
        "paragontech.com",             # PathFinder 5000
        "acsindustrial.com",           # PathFinder 3000
        "capetronics.com",             # Van Dorn Siemens PathFinder
        "radwell.com",                 # Legacy control parts with warranty
        "acwei.com",                   # Siemens S5 PathFinder parts
        "rochesterindustrialservices.com",  # CAMAC repairs 1-year warranty

        # ===== FANUC Official & Semi-Official (Premium) =====
        "techtransfer.fanucamerica.com",  # 100+ free tutorials (CRITICAL)
        "fanucamerica.com",            # Official FANUC America
        "crc2.frc.com",                # Customer Resource Center
        "fanuc-academy.uk",            # Official UK training
        "content.fanucworld.com",      # Alarm codes list

        # ===== FANUC Third-Party Documentation =====
        "therobotguyllc.com",          # KAREL manual, R-30iA maintenance
        "manuals.plus",                # R-30iB operators manual
        "manualmachine.com",           # R-30iB maintenance manual
        "studylib.net",                # R-30iB Plus, DCS manuals
        "robots.com",                  # Error codes explained
        "diy-robotics.com",            # Mastering procedures, iRVision
        "tristarcnc.com",              # Servo/spindle alarm codes
        "cncspares.com",               # Servo/spindle codes UK
        "mroelectric.com",             # Servo motor alarms
        "okmarts.com",                 # Servo amplifier alarm codes
        "robochallenge.pl",            # iRVision 2D manual
        "docs.mech-mind.net",          # Automatic calibration
        "productivity.com",            # Battery loss recovery

        # ===== FANUC Training Providers =====
        "nrtcautomation.com",          # FANUC, ABB, KUKA training
        "ramtecohio.com",              # RAMTEC certified courses
        "gcodetutor.com",              # Online CNC courses
        "aleksandarhaber.com",         # 250+ free tutorials

        # ===== Industrial Forums (Highest Value) =====
        "control.com",                 # Process control, automation
        "mrplc.com",                   # Multi-vendor PLC support
        "cnczone.com",                 # CNC and control retrofits
        "robotics.stackexchange.com",  # Structured Q&A
        "engineering.stackexchange.com",  # General engineering Q&A
        "linuxcnc.org",                # CNC retrofit community

        # ===== KraussMaffei Third-Party =====
        "pdfcoffee.com",               # MX1600 complete manual
        "opcturkey.com",               # MC4 Ethernet driver manual
        "machineryhost.com",           # MC5 CX full documentation
        "directindustry.com",          # All series catalogs
        "rgbelektronika.eu",           # CX, GX, MX, PX repair
        "kubousek.cz",                 # Authorized training Czech
        "bruys.nl",                    # OEM spare parts Netherlands
        "im-machinery.de",             # Hydraulic parts Germany

        # ===== PLC & Electrical Troubleshooting =====
        "realpars.com",                # PLC digital I/O, Allen-Bradley
        "dosupply.com",                # PLC/VFD troubleshooting flowcharts
        "ladderlogicworld.com",        # 7-part ladder logic tutorial
        "plcacademy.com",              # Comprehensive PLC tutorial
        "solisplc.com",                # Allen-Bradley, large programs
        "library.automationdirect.com",  # Symbols, 4-20mA troubleshooting
        "cdn.automationdirect.com",    # Technical manuals PDFs

        # ===== VFD & Motor Control =====
        "cdn.logic-control.com",       # ABB VFD troubleshooting
        "motioncontroltips.com",       # VFD no-power checks, IGBT
        "vfds.com",                    # VFD motor damage prevention
        "americanindustrialinc.com",   # Overcurrent fault diagnosis

        # ===== Servo & Encoder Resources =====
        "gesrepair.com",               # Servo drive troubleshooting
        "mitchell-electronics.com",    # 8 servo motor tips
        "ato.com",                     # Servo failure analysis
        "gtencoder.com",               # Encoder troubleshooting FAQ
        "encoder.com",                 # Encoder troubleshooting PDF
        "dynapar.com",                 # Encoder signal oscilloscope

        # ===== Sensors & Instrumentation =====
        "temprel.com",                 # Thermocouple troubleshooting
        "peaksensors.com",             # TC and RTD testing
        "controlglobal.com",           # TC noise, grounding
        "en.jumo.pl",                  # 18 thermocouple errors
        "automationforum.co",          # TC troubleshooting checklist
        "fluke.com",                   # 4-20mA, thermal imaging
        "devarinc.com",                # 4-20mA scaling issues
        "seametrics.com",              # Current loop sneak currents
        "instrumentationtools.com",    # 4-20mA DVM methods

        # ===== Industrial Networks =====
        "profinetuniversity.com",      # PROFINET/PROFIBUS troubleshooting
        "us.profinet.com",             # PROFINET fundamentals
        "trend-networks.com",          # Network testing tools
        "industrialautomationco.com",  # EtherNet/IP vs PROFINET

        # ===== Hydraulics =====
        "crossmfg.com",                # Step-by-step hydraulic troubleshooting
        "advancedfluidsystems.com",    # Hydraulic flowcharts PDF
        "machinerylubrication.com",    # 5-step methodology
        "brennaninc.com",              # Systematic approach
        "powermotiontech.com",         # Thermal diagnostics
        "supremeintegratedtechnology.com",  # Pump noise, leaks

        # ===== Vibration & Thermal Analysis =====
        "vi-institute.org",            # ISO 18436 Cat I-IV
        "mobiusinstitute.com",         # Cat I online certification
        "avtreliability.com",          # ISO 18436 Levels 1-3
        "ctconline.com",               # Free accelerometer basics
        "failureprevention.com",       # Cat 1-2 certification

        # ===== Safety & Compliance =====
        "osha.gov",                    # LOTO regulations
        "vectorsolutions.com",         # 6-step LOTO procedure
        "safetyculture.com",           # LOTO training, checklists
        "nfpa.org",                    # Arc flash NFPA 70E

        # ===== Root Cause Analysis =====
        "asq.org",                     # ASQ RCA e-Learning
        "kepner-tregoe.com",           # 65+ year RCA methodology
        "thinkreliability.com",        # Cause Mapping
        "6sigma.us",                   # RCA templates, certification

        # ===== Training Platforms =====
        "toolingu.com",                # Tooling U-SME 600+ courses
        "learn.toolingu.com",          # LMS platform
        "isa.org",                     # ISA automation standards
        "tpctraining.com",             # Electrical troubleshooting
        "bin95.com",                   # Free motor control simulator
        "academy.boschrexroth.com",    # Hands-on hydraulics
        "boschrexroth.com",            # Bosch Rexroth training US
        "yaskawa.com",                 # Drives, motion control
        "interplaylearning.com",       # 500+ hours, 3D sims
        "360training.com",             # 300+ industrial courses

        # ===== Publications (Technical Articles) =====
        "magazines.amiplastics.com",   # Injection World free digital
        "smart-molding.com",           # Industry 4.0, automation
        "moldmakingtechnology.com",    # Mold design, CAD/CAM
        "injectionmoldingdivision.org",  # SPE technical papers
        "reliableplant.com",           # Maintenance, reliability
        "plantengineering.com",        # Plant maintenance
        "controleng.com",              # Automation, control

        # ===== Parts Suppliers (Technical Resources) =====
        "grainger.com",                # Product specs, how-to
        "mscdirect.com",               # 2.2M products, tech support
        "motionindustries.com",        # Bearings, hydraulics expertise

        # ===== Allen-Bradley / Rockwell Automation (Premium) =====
        "literature.rockwellautomation.com",  # Direct PDF manual access
        "rockwellautomation.com",      # 40,000+ technotes
        "compatibility.rockwellautomation.com",  # Firmware, compatibility
        "pesquality.com",              # PowerFlex 40 fault codes
        "precision-elec.com",          # PowerFlex 755 troubleshooting
        "wireless-telemetry.com",      # PowerFlex 753 fault codes PDF
        "blog.acsindustrial.com",      # Repair-focused troubleshooting

        # ===== Siemens Additional =====
        "cache.industry.siemens.com",  # PDF cache for manuals
        "kwoco-plc.com",               # Siemens drive fault codes

        # ===== Industrial Sensors (Banner, Turck, IFM, Omron, SICK) =====
        "bannerengineering.com",       # Sensor technical library
        "info.bannerengineering.com",  # PDF documentation
        "turck.us",                    # FAQ database, wiring diagrams
        "ifm.com",                     # Product documentation finder
        "ia.omron.com",                # Sensor technical guides
        "sick.com",                    # Troubleshooting services
        "balluff.com",                 # NPN vs PNP guides
        "accautomation.ca",            # Sensor wiring tutorials
        "tc-inc.com",                  # Thermocouple/RTD guide
        "piecal.com",                  # Calibrator troubleshooting
        "800loadcel.com",              # Load cell troubleshooting
        "apecusa.com",                 # Load cell problems/solutions
        "mhforce.com",                 # Morehouse load cell guide

        # ===== TCUs / Thermolators =====
        "advantageengineering.com",    # Service manuals, alert codes
        "mokon.com",                   # Support portal, troubleshooting
        "regloplas.com",               # FAQ/troubleshooting
        "regloplasusa.com",            # US troubleshooting
        "sterlco.com",                 # Sterling resources portal
        "diecastmachinery.com",        # TCU troubleshooting guides
        "aecinternet.com",             # AEC temperature controllers

        # ===== Chillers =====
        "trane.com",                   # Industrial chiller manuals
        "star-supply.com",             # Trane alert codes PDF
        "acerrorcode.com",             # Chiller error codes guide
        "generalairproducts.com",      # Controller alarm manual

        # ===== Dryers & Material Handling =====
        "conairgroup.com",             # Dryer/granulator manuals
        "novatec.com",                 # Knowledge center (excellent)
        "wittmann-group.com",          # Download center
        "motan.com",                   # Drying products/technical

        # ===== Hot Runner Systems =====
        "moldmasters.com",             # User manuals, troubleshooting PDF
        "husky.co",                    # Ultra series, UltraShot manuals
        "incoe.com",                   # Interactive troubleshooting
        "yudoeu.com",                  # YUDO user handbook
        "ewikon.com",                  # EWIKON manuals
        "synventive.com",              # Download center

        # ===== Conveyors =====
        "dornerconveyors.com",         # Manuals, service videos
        "hytrol.com",                  # Manuals index
        "cdn.hytrol.com",              # Direct PDF access
        "flexlink.com",                # Technical library
        "mknorthamerica.com",          # Operating manuals
        "accurateindustrial.com",      # Belt tracking guide
        "spantechconveyors.com",       # Common problems guide
        "sparksbelting.com",           # Tracking diagnostics

        # ===== Granulators / Grinders =====
        "cumberlandplastics.com",      # Resources portal
        "rapidgranulator.com",         # Downloads, FAQ
        "granulator-blades.com",       # Common problems FAQ
        "servicesforplastics.com",     # All brand blade specs
        "bladesmachinery.com",         # Parts and service

        # ===== Manual Aggregators =====
        "manualslib.com",              # 9.6M+ PDFs, free access
        "manualzz.com",                # AI-assisted search
        "manualsdir.com",              # Industrial equipment

        # ===== AutomationDirect =====
        "automationdirect.com",        # Main site
        "support.automationdirect.com",  # Troubleshooting guides

        # ===== Reference (Trusted) =====
        "wikipedia.org",
        "wikimedia.org",
        "britannica.com",
        "merriam-webster.com",
        "wolframalpha.com",

        # ===== Standards Bodies (Premium) =====
        "nist.gov",
        "ietf.org",
        "w3.org",
        "iso.org",
        "rfc-editor.org",
        "oasis-open.org",
        "ecma-international.org",

        # ===== Tech News/Analysis (Standard) =====
        "arstechnica.com",
        "wired.com",
        "thenewstack.io",
        "infoq.com",
        "lwn.net",
        "techcrunch.com",
        "theverge.com"
    }

    # Premium domains get highest boost
    PREMIUM_DOMAINS = {
        # Academic
        "arxiv.org", "semanticscholar.org", "openreview.net",
        "aclanthology.org", "neurips.cc", "proceedings.mlr.press",
        # Documentation
        "docs.python.org", "developer.mozilla.org", "kubernetes.io",
        "pytorch.org", "tensorflow.org", "huggingface.co",
        # Standards
        "nist.gov", "ietf.org", "w3.org", "rfc-editor.org", "osha.gov",
        # Allen-Bradley / Rockwell (Premium - 40,000+ technotes)
        "literature.rockwellautomation.com", "rockwellautomation.com",
        "support.industry.siemens.com",
        # Injection Molding / Euromap (Premium for IMM queries)
        "euromap.org", "ptonline.com", "rjginc.com",
        "kraussmaffei.com", "milacron.com", "sumitomo-shi-demag.us",
        "trainingacademy.kraussmaffei.com", "pdfcoffee.com",
        # FANUC Official & High-Value (Premium)
        "fanuc.eu", "robot-forum.com", "techtransfer.fanucamerica.com",
        "fanucamerica.com", "diy-robotics.com", "therobotguyllc.com",
        # Industrial Forums (Highest Value for Troubleshooting)
        "control.com", "cnczone.com", "plctalk.net", "practicalmachinist.com",
        # Plastics Industry Technical
        "plastics-rubber.basf.com", "4spe.org", "aim.institute",
        "injectionmoldingonline.com", "dme.net",
        # Sensors (Premium Manufacturers)
        "bannerengineering.com", "turck.us", "sick.com", "ifm.com",
        # Hot Runner Systems (Premium)
        "moldmasters.com", "husky.co", "incoe.com",
        # Plastics Auxiliary Equipment (Premium)
        "novatec.com", "conairgroup.com", "advantageengineering.com",
        # Manual Aggregators (Premium - 9.6M+ PDFs)
        "manualslib.com", "automationdirect.com",
        # Training & Certification (Premium)
        "toolingu.com", "isa.org", "realpars.com", "vi-institute.org",
        # Publications (Premium Technical Content)
        "reliableplant.com", "controleng.com", "plantengineering.com",
    }

    # Common stopwords to exclude from keyword matching
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "also", "now", "what",
        "which", "who", "whom", "this", "that", "these", "those", "am", "or",
        "and", "but", "if", "because", "until", "while", "about", "against",
        "i", "me", "my", "you", "your", "he", "she", "it", "we", "they"
    }

    # Minimum keyword overlap ratio for domain boost (0.0-1.0)
    MIN_KEYWORD_RELEVANCE = 0.15  # Lowered from 0.3 to allow partial matches

    # Domains to filter out from results (dictionaries, generic reference sites)
    BLOCKED_DOMAINS = {
        "collinsdictionary.com",
        "merriam-webster.com",
        "dictionary.com",
        "thesaurus.com",
        "cambridge.org/dictionary",
        "yourdictionary.com",
        "wordreference.com",
        "vocabulary.com",
        "urbandictionary.com",
    }

    def __init__(self, brave_api_key: Optional[str] = None, searxng_url: Optional[str] = None):
        self.searxng = SearXNGSearchProvider(searxng_url)  # Uses settings if None
        self.brave = BraveSearchProvider(brave_api_key)
        self.duckduckgo = DuckDuckGoProvider()
        self._embedding_model = None  # Lazy-load for semantic similarity

    def _stem_word(self, word: str) -> str:
        """Apply simple stemming to normalize a word to its base form."""
        if word.endswith('ing') and len(word) > 5:
            return word[:-3]  # debugging -> debug
        elif word.endswith('ed') and len(word) > 4:
            return word[:-2]   # fixed -> fix
        elif word.endswith('es') and len(word) > 4:
            return word[:-2]   # fixes -> fix
        elif word.endswith('s') and len(word) > 4:
            return word[:-1]   # alarms -> alarm
        elif word.endswith('ly') and len(word) > 4:
            return word[:-2]   # slowly -> slow
        elif word.endswith('ment') and len(word) > 6:
            return word[:-4]   # replacement -> replac
        return word

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text, excluding stopwords.

        Returns stemmed keywords for consistent matching.
        Preserves hyphenated technical terms (e.g., SRVO-063).
        """
        text = text.lower()

        # Preserve hyphenated terms (like SRVO-063) by extracting them first
        import re
        hyphenated = re.findall(r'[a-z0-9]+-[a-z0-9]+', text)

        # Now remove punctuation for regular word extraction
        text_clean = text.translate(str.maketrans("", "", string.punctuation))

        # Split into words and filter
        words = text_clean.split()
        keywords = set()

        for w in words:
            if w in self.STOPWORDS or len(w) <= 2:
                continue
            # Add both original and stemmed form
            keywords.add(w)
            stem = self._stem_word(w)
            if stem != w and len(stem) > 2:
                keywords.add(stem)

        # Add hyphenated terms (e.g., srvo-063)
        for term in hyphenated:
            keywords.add(term)
            # Also add without hyphen for matching flexibility
            keywords.add(term.replace('-', ''))

        return keywords

    def _calculate_keyword_relevance(
        self,
        query_keywords: Set[str],
        result: WebSearchResult
    ) -> float:
        """
        Calculate keyword overlap between query and result.
        Returns a score from 0.0 to 1.0.
        """
        if not query_keywords:
            return 0.0

        # Extract keywords from title and snippet
        result_text = f"{result.title} {result.snippet}"
        result_keywords = self._extract_keywords(result_text)

        if not result_keywords:
            return 0.0

        # Calculate overlap: how many query keywords appear in result
        overlap = query_keywords.intersection(result_keywords)
        relevance = len(overlap) / len(query_keywords)

        return relevance

    def _is_result_relevant(
        self,
        query_keywords: Set[str],
        result: WebSearchResult
    ) -> Tuple[bool, float]:
        """
        Check if a result is semantically relevant to the query.
        Returns (is_relevant, relevance_score).

        Uses keyword overlap as a fast heuristic. Results must have at least
        MIN_KEYWORD_RELEVANCE overlap to be considered relevant.
        """
        relevance = self._calculate_keyword_relevance(query_keywords, result)
        is_relevant = relevance >= self.MIN_KEYWORD_RELEVANCE

        return is_relevant, relevance

    async def search(
        self,
        queries: List[str],
        max_results_per_query: int = 3,
        query_type: Optional[str] = None
    ) -> List[WebSearchResult]:
        """
        Execute searches for multiple queries.
        Provider priority: SearXNG → DuckDuckGo → Brave

        Args:
            queries: List of search query strings
            max_results_per_query: Max results per query
            query_type: Optional type hint ('academic', 'technical', 'general')
                       If None, auto-detected from first query

        Domain boost is only applied to results that are semantically relevant
        to prevent irrelevant results from trusted domains being over-ranked.
        """
        all_results = []
        seen_urls = set()
        metrics = get_search_metrics()

        # Extract keywords from all queries for relevance checking
        all_query_keywords: Set[str] = set()
        for query in queries[:5]:
            all_query_keywords.update(self._extract_keywords(query))

        logger.debug(f"Query keywords for relevance: {all_query_keywords}")

        # Determine best available provider using intelligent selection
        # Priority: SearXNG (self-hosted, no limits) → DuckDuckGo → Brave
        # Uses metrics to avoid rate-limited providers
        provider = None
        provider_name = None

        # Try SearXNG first (no rate limits when self-hosted)
        if await self.searxng.check_availability():
            provider = self.searxng
            provider_name = "SearXNG"
        else:
            # Fall back using metrics-based selection
            best_provider, reason = metrics.get_best_provider(["duckduckgo", "brave"])
            logger.info(f"Provider selection: {best_provider} ({reason})")

            if best_provider == "duckduckgo":
                provider = self.duckduckgo
                provider_name = "DuckDuckGo"
            elif best_provider == "brave" and self.brave.available:
                provider = self.brave
                provider_name = "Brave"
            else:
                # Fallback to DDG even if rate-limited
                provider = self.duckduckgo
                provider_name = "DuckDuckGo"

        logger.info(f"Using {provider_name} search provider")

        # Auto-detect query type from first query if not specified
        if query_type is None and queries and isinstance(provider, SearXNGSearchProvider):
            query_type = provider.detect_query_type(queries[0])
            logger.info(f"Auto-detected query type: {query_type}")

        # Execute searches in parallel
        tasks = [
            self._search_single(provider, query, max_results_per_query, query_type)
            for query in queries[:5]  # Limit to 5 queries
        ]

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        for results in results_lists:
            if isinstance(results, Exception):
                logger.error(f"Search task failed: {results}")
                continue

            for result in results:
                # Deduplicate by URL
                if result.url not in seen_urls:
                    seen_urls.add(result.url)

                    # Skip blocked domains (dictionaries, etc.)
                    if any(bd in result.source_domain for bd in self.BLOCKED_DOMAINS):
                        logger.debug(f"Blocked off-topic domain: {result.source_domain}")
                        continue

                    # Check semantic relevance before domain boost
                    is_relevant, keyword_relevance = self._is_result_relevant(
                        all_query_keywords, result
                    )

                    # Only apply domain boost if result is semantically relevant
                    is_trusted_domain = any(
                        td in result.source_domain for td in self.TRUSTED_DOMAINS
                    )
                    is_premium_domain = any(
                        pd in result.source_domain for pd in self.PREMIUM_DOMAINS
                    )

                    if is_trusted_domain:
                        if is_relevant:
                            # Premium domains get higher boost (0.25) vs trusted (0.15)
                            boost = 0.25 if is_premium_domain else 0.15
                            result.relevance_score = min(1.0, result.relevance_score + boost)
                            tier = "premium" if is_premium_domain else "trusted"
                            logger.debug(
                                f"Boosted {tier} result: {result.title[:40]}... "
                                f"(keyword_relevance={keyword_relevance:.2f}, boost={boost})"
                            )
                        else:
                            # Penalize irrelevant results from trusted domains
                            result.relevance_score = max(0.1, result.relevance_score - 0.3)
                            logger.debug(
                                f"Penalized irrelevant result: {result.title[:40]}... "
                                f"(keyword_relevance={keyword_relevance:.2f})"
                            )
                    elif is_relevant:
                        # Small boost for relevant non-trusted domain results
                        result.relevance_score = min(1.0, result.relevance_score + 0.1)
                    else:
                        # Heavily penalize irrelevant results from non-trusted domains
                        # These are likely spam or off-topic content
                        if keyword_relevance == 0.0:
                            # Zero keyword overlap = completely irrelevant
                            result.relevance_score = max(0.05, result.relevance_score - 0.5)
                            logger.debug(
                                f"Heavily penalized off-topic result: {result.title[:40]}..."
                            )
                        else:
                            # Some overlap but below threshold
                            result.relevance_score = max(0.1, result.relevance_score - 0.2)

                    all_results.append(result)

        # Sort by relevance score
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Filter out completely irrelevant results (zero keyword overlap)
        # Keep only results with at least minimal relevance
        filtered_results = [
            r for r in all_results
            if self._calculate_keyword_relevance(all_query_keywords, r) > 0
        ]

        # If aggressive filtering removed too many results, fall back to penalized list
        if len(filtered_results) < 3:
            filtered_results = all_results[:max(10, len(all_results))]
            logger.warning("Aggressive filtering removed too many results, using penalized fallback")
        else:
            removed_count = len(all_results) - len(filtered_results)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} off-topic results with zero keyword overlap")

        # Log top results for debugging
        logger.info(f"Top 5 results after relevance filtering:")
        for i, r in enumerate(filtered_results[:5], 1):
            logger.info(f"  {i}. [{r.source_domain}] {r.title[:50]}... (score={r.relevance_score:.2f})")

        return filtered_results

    async def _search_single(
        self,
        provider: SearchProvider,
        query: str,
        max_results: int,
        query_type: Optional[str] = None
    ) -> List[WebSearchResult]:
        """Execute a single search with cascading fallback"""
        # Pass query_type to SearXNG for dynamic engine selection
        if isinstance(provider, SearXNGSearchProvider):
            results = await provider.search(query, max_results, query_type=query_type)
        else:
            results = await provider.search(query, max_results)

        # Cascading fallback: SearXNG → DuckDuckGo → Brave
        if not results:
            if provider == self.searxng:
                logger.info("SearXNG failed, falling back to DuckDuckGo")
                results = await self.duckduckgo.search(query, max_results)

            if not results and self.brave.available:
                logger.info("Falling back to Brave API")
                results = await self.brave.search(query, max_results)

        return results

    def format_results_for_synthesis(self, results: List[WebSearchResult]) -> str:
        """Format search results as text for the synthesizer"""
        if not results:
            return "No web search results available."

        formatted = []
        for i, result in enumerate(results[:10], 1):
            formatted.append(
                f"**[{i}] {result.title}**\n"
                f"Source: {result.source_domain}\n"
                f"{result.snippet}\n"
            )

        return "\n---\n".join(formatted)
