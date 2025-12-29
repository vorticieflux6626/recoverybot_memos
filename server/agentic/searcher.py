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
from typing import List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

import httpx

from .models import WebSearchResult

logger = logging.getLogger("agentic.searcher")


class SearchProvider:
    """Base class for search providers"""

    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        raise NotImplementedError


class SearXNGSearchProvider(SearchProvider):
    """
    SearXNG self-hosted metasearch provider.

    Primary provider with no rate limits.
    Aggregates results from Google, Bing, DuckDuckGo, Brave, etc.
    """

    # Cache TTL for availability check (seconds)
    AVAILABILITY_CACHE_TTL = 60  # Re-check every 60 seconds

    def __init__(self, base_url: str = "http://localhost:8888"):
        self.base_url = base_url.rstrip("/")
        self._available = None  # Cache availability check
        self._available_checked_at = 0  # Timestamp of last check

    @property
    def available(self) -> bool:
        """Check if SearXNG is available (cached)"""
        return self._available if self._available is not None else True

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
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params={"q": "test", "format": "json"}
                )
                self._available = response.status_code == 200
                self._available_checked_at = current_time
                if self._available:
                    logger.info(f"SearXNG available at {self.base_url}")
        except Exception as e:
            logger.warning(f"SearXNG not available: {e}")
            self._available = False
            self._available_checked_at = current_time

        return self._available

    async def search(self, query: str, max_results: int = 10) -> List[WebSearchResult]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                            "engines": "google,bing,duckduckgo,brave,wikipedia",
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

                logger.info(f"SearXNG returned {len(results)} results for: {query[:50]}")
                return results

        except Exception as e:
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

            async with httpx.AsyncClient(timeout=15.0) as client:
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
    """

    BASE_URL = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
            data = {"q": query}

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    self.BASE_URL,
                    headers=headers,
                    data=data,
                    follow_redirects=True
                )

                if response.status_code == 200:
                    html = response.text
                    return self._parse_results(html, max_results)
                elif response.status_code == 202:
                    logger.warning("DuckDuckGo rate limited (202)")
                    return []
                else:
                    logger.warning(f"DuckDuckGo search failed: {response.status_code}")
                    return []

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
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
    TRUSTED_DOMAINS = {
        # Technical documentation
        "docs.python.org", "developer.mozilla.org", "stackoverflow.com",
        "github.com", "gitlab.com", "bitbucket.org",
        # Academic/Research
        "arxiv.org", "scholar.google.com", "researchgate.net",
        "ieee.org", "acm.org", "nature.com", "sciencedirect.com",
        # Official documentation
        "docs.microsoft.com", "cloud.google.com", "aws.amazon.com",
        "kubernetes.io", "docker.com", "nginx.org",
        # Engineering/Hardware
        "electronics.stackexchange.com", "hackaday.com",
        "instructables.com", "adafruit.com", "sparkfun.com",
        # Reference
        "wikipedia.org", "wikimedia.org", "britannica.com",
        # Government/Standards
        "nist.gov", "ietf.org", "w3.org", "iso.org"
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
    MIN_KEYWORD_RELEVANCE = 0.3

    def __init__(self, brave_api_key: Optional[str] = None, searxng_url: str = "http://localhost:8888"):
        self.searxng = SearXNGSearchProvider(searxng_url)
        self.brave = BraveSearchProvider(brave_api_key)
        self.duckduckgo = DuckDuckGoProvider()
        self._embedding_model = None  # Lazy-load for semantic similarity

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text, excluding stopwords.

        Also extracts word stems to handle plural/singular matching.
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Split into words and filter
        words = text.split()
        keywords = set()

        for w in words:
            if w in self.STOPWORDS or len(w) <= 2:
                continue
            keywords.add(w)
            # Add simple stemming for common suffixes
            if w.endswith('ing'):
                keywords.add(w[:-3])  # debugging -> debug
            elif w.endswith('ed'):
                keywords.add(w[:-2])   # fixed -> fix
            elif w.endswith('s') and len(w) > 4:
                keywords.add(w[:-1])   # deadlocks -> deadlock
            elif w.endswith('es') and len(w) > 4:
                keywords.add(w[:-2])   # fixes -> fix

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
        max_results_per_query: int = 3
    ) -> List[WebSearchResult]:
        """
        Execute searches for multiple queries.
        Provider priority: SearXNG → DuckDuckGo → Brave

        Domain boost is only applied to results that are semantically relevant
        to prevent irrelevant results from trusted domains being over-ranked.
        """
        all_results = []
        seen_urls = set()

        # Extract keywords from all queries for relevance checking
        all_query_keywords: Set[str] = set()
        for query in queries[:5]:
            all_query_keywords.update(self._extract_keywords(query))

        logger.debug(f"Query keywords for relevance: {all_query_keywords}")

        # Determine best available provider
        # Priority: SearXNG (self-hosted, no limits) → DuckDuckGo → Brave
        if await self.searxng.check_availability():
            provider = self.searxng
            provider_name = "SearXNG"
        else:
            # DuckDuckGo is always available (no API key needed)
            provider = self.duckduckgo
            provider_name = "DuckDuckGo"

        logger.info(f"Using {provider_name} search provider")

        # Execute searches in parallel
        tasks = [
            self._search_single(provider, query, max_results_per_query)
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

                    # Check semantic relevance before domain boost
                    is_relevant, keyword_relevance = self._is_result_relevant(
                        all_query_keywords, result
                    )

                    # Only apply domain boost if result is semantically relevant
                    is_trusted_domain = any(
                        td in result.source_domain for td in self.TRUSTED_DOMAINS
                    )

                    if is_trusted_domain:
                        if is_relevant:
                            # Full boost for relevant trusted domain results
                            result.relevance_score = min(1.0, result.relevance_score + 0.2)
                            logger.debug(
                                f"Boosted relevant result: {result.title[:40]}... "
                                f"(keyword_relevance={keyword_relevance:.2f})"
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
        max_results: int
    ) -> List[WebSearchResult]:
        """Execute a single search with cascading fallback"""
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
