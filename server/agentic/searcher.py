"""
Searcher Agent - Web Search Execution

Implements multi-provider web search with fallback:
1. Brave Search API (primary, requires API key)
2. DuckDuckGo HTML (fallback, no API key needed)

Includes result scoring and domain filtering for recovery-relevant content.
"""

import asyncio
import logging
import re
from typing import List, Optional
from urllib.parse import quote_plus, urlparse

import httpx

from .models import WebSearchResult

logger = logging.getLogger("agentic.searcher")


class SearchProvider:
    """Base class for search providers"""

    async def search(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        raise NotImplementedError


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
    - Multi-provider fallback (Brave → DuckDuckGo)
    - Domain allowlisting for recovery-relevant content
    - Result deduplication and scoring
    - Parallel query execution
    """

    # Domains with higher trust for recovery/health content
    TRUSTED_DOMAINS = {
        "samhsa.gov", "nida.nih.gov", "cdc.gov", "who.int",
        "mayoclinic.org", "webmd.com", "healthline.com",
        "drugabuse.gov", "recovery.org", "aa.org", "na.org",
        "psychologytoday.com", "nimh.nih.gov", "nih.gov"
    }

    def __init__(self, brave_api_key: Optional[str] = None):
        self.brave = BraveSearchProvider(brave_api_key)
        self.duckduckgo = DuckDuckGoProvider()

    async def search(
        self,
        queries: List[str],
        max_results_per_query: int = 3
    ) -> List[WebSearchResult]:
        """
        Execute searches for multiple queries.
        Uses Brave if available, falls back to DuckDuckGo.
        """
        all_results = []
        seen_urls = set()

        # Try Brave first
        provider = self.brave if self.brave.available else self.duckduckgo
        provider_name = "Brave" if self.brave.available else "DuckDuckGo"
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
                    # Boost score for trusted domains
                    if any(td in result.source_domain for td in self.TRUSTED_DOMAINS):
                        result.relevance_score = min(1.0, result.relevance_score + 0.2)
                    all_results.append(result)

        # Sort by relevance score
        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return all_results

    async def _search_single(
        self,
        provider: SearchProvider,
        query: str,
        max_results: int
    ) -> List[WebSearchResult]:
        """Execute a single search with fallback"""
        results = await provider.search(query, max_results)

        # If primary fails and we have fallback
        if not results and provider == self.brave:
            logger.info("Falling back to DuckDuckGo")
            results = await self.duckduckgo.search(query, max_results)

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
