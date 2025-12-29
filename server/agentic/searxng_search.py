"""
SearXNG Search Integration for Agentic Search

Replaces rate-limited DuckDuckGo API calls with self-hosted SearXNG metasearch.
Provides unlimited search queries with results from multiple engines.

Usage:
    from agentic.searxng_search import SearXNGSearcher, get_searxng_searcher

    searcher = get_searxng_searcher()
    results = await searcher.search(["query 1", "query 2"])
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import sys
import os

# Add searxng directory to path for client import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'searxng'))

try:
    from searxng_client import SearXNGClient, SearchResult as SearXNGResult, get_searxng_client
except ImportError:
    # Fallback: define inline if import fails
    SearXNGClient = None
    SearXNGResult = None

import httpx

# Lazy settings import to avoid circular dependencies
_settings = None
def _get_settings():
    global _settings
    if _settings is None:
        from config.settings import get_settings
        _settings = get_settings()
    return _settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResultItem:
    """Normalized search result for agentic pipeline"""
    title: str
    url: str
    snippet: str
    source_domain: str = ""
    engine: str = "searxng"
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_domain": self.source_domain,
            "engine": self.engine,
            "score": self.score,
            "metadata": self.metadata
        }


class SearXNGSearcher:
    """
    SearXNG-based web searcher for the agentic search pipeline.

    Features:
    - No rate limiting (self-hosted)
    - Multiple search engines (Google, Bing, DuckDuckGo, Brave, etc.)
    - Caching allowed
    - Parallel query execution
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        default_engines: Optional[List[str]] = None,
        max_results_per_query: int = 10
    ):
        """
        Initialize SearXNG searcher.

        Args:
            base_url: SearXNG server URL (defaults to settings.searxng_url)
            timeout: Request timeout in seconds
            default_engines: Engines to use (None = google, bing, duckduckgo)
            max_results_per_query: Maximum results per query
        """
        settings = _get_settings()
        self.base_url = (base_url or settings.searxng_url).rstrip("/")
        self.timeout = timeout
        self.default_engines = default_engines or ["google", "bing", "duckduckgo"]
        self.max_results_per_query = max_results_per_query
        self._client: Optional[httpx.AsyncClient] = None
        self._available = None  # Cache availability check

        self._stats = {
            "total_searches": 0,
            "total_results": 0,
            "failed_searches": 0,
            "fallback_to_duckduckgo": 0
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True
            )
        return self._client

    async def is_available(self) -> bool:
        """Check if SearXNG is available"""
        if self._available is not None:
            return self._available

        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.base_url}/search",
                params={"q": "test", "format": "json"},
                timeout=5.0
            )
            self._available = response.status_code == 200
        except Exception as e:
            logger.warning(f"SearXNG not available: {e}")
            self._available = False

        return self._available

    async def search(
        self,
        queries: List[str],
        engines: Optional[List[str]] = None,
        max_results: Optional[int] = None
    ) -> List[SearchResultItem]:
        """
        Execute search queries and return results.

        Args:
            queries: List of search query strings
            engines: Specific engines to use (overrides default)
            max_results: Maximum total results to return

        Returns:
            List of SearchResultItem objects
        """
        max_results = max_results or (len(queries) * self.max_results_per_query)
        engines = engines or self.default_engines

        # Check availability
        if not await self.is_available():
            logger.warning("SearXNG not available, falling back to DuckDuckGo")
            self._stats["fallback_to_duckduckgo"] += 1
            return await self._fallback_duckduckgo(queries, max_results)

        # Execute queries in parallel
        tasks = [
            self._search_single(query, engines)
            for query in queries
        ]

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine and deduplicate results
        all_results = []
        seen_urls = set()

        for result in results_lists:
            if isinstance(result, Exception):
                logger.warning(f"Query failed: {result}")
                self._stats["failed_searches"] += 1
                continue

            for item in result:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    all_results.append(item)

        self._stats["total_searches"] += len(queries)
        self._stats["total_results"] += len(all_results)

        return all_results[:max_results]

    async def _search_single(
        self,
        query: str,
        engines: List[str]
    ) -> List[SearchResultItem]:
        """Execute a single search query"""
        client = await self._get_client()

        try:
            response = await client.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "engines": ",".join(engines),
                    "language": "en-US"
                }
            )
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("results", [])[:self.max_results_per_query]:
                url = item.get("url", "")
                domain = self._extract_domain(url)

                results.append(SearchResultItem(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("content", ""),
                    source_domain=domain,
                    engine=item.get("engine", "searxng"),
                    score=item.get("score", 0.0),
                    metadata={
                        "category": item.get("category", "general"),
                        "publishedDate": item.get("publishedDate"),
                        "thumbnail": item.get("thumbnail")
                    }
                ))

            logger.debug(f"SearXNG query '{query[:30]}...': {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"SearXNG search failed for '{query[:30]}...': {e}")
            raise

    async def _fallback_duckduckgo(
        self,
        queries: List[str],
        max_results: int
    ) -> List[SearchResultItem]:
        """Fallback to DuckDuckGo if SearXNG is unavailable"""
        try:
            from duckduckgo_search import DDGS

            results = []
            seen_urls = set()

            with DDGS() as ddgs:
                for query in queries:
                    try:
                        for r in ddgs.text(query, max_results=self.max_results_per_query):
                            url = r.get("href", "")
                            if url not in seen_urls:
                                seen_urls.add(url)
                                results.append(SearchResultItem(
                                    title=r.get("title", ""),
                                    url=url,
                                    snippet=r.get("body", ""),
                                    source_domain=self._extract_domain(url),
                                    engine="duckduckgo"
                                ))
                    except Exception as e:
                        logger.warning(f"DuckDuckGo query failed: {e}")

            return results[:max_results]

        except ImportError:
            logger.error("duckduckgo_search not installed for fallback")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo fallback failed: {e}")
            return []

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return ""

    @property
    def stats(self) -> Dict[str, Any]:
        """Get searcher statistics"""
        return self._stats.copy()

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_searxng_searcher: Optional[SearXNGSearcher] = None


def get_searxng_searcher(**kwargs) -> SearXNGSearcher:
    """Get or create the SearXNG searcher singleton"""
    global _searxng_searcher
    if _searxng_searcher is None:
        _searxng_searcher = SearXNGSearcher(**kwargs)
    return _searxng_searcher


async def search_with_searxng(
    queries: List[str],
    max_results: int = 20
) -> List[Dict[str, Any]]:
    """
    Convenience function for quick searches.

    Args:
        queries: Search queries
        max_results: Maximum results

    Returns:
        List of result dictionaries
    """
    searcher = get_searxng_searcher()
    results = await searcher.search(queries, max_results=max_results)
    return [r.to_dict() for r in results]


# Test function
async def main():
    """Test the SearXNG searcher"""
    import json

    searcher = get_searxng_searcher()

    print("Testing SearXNG searcher...")
    print("-" * 50)

    # Check availability
    available = await searcher.is_available()
    print(f"SearXNG available: {available}")

    if available:
        # Test search
        results = await searcher.search(
            ["Python programming", "machine learning"],
            max_results=10
        )

        print(f"\nFound {len(results)} results:")
        for i, r in enumerate(results[:5], 1):
            print(f"\n{i}. [{r.engine}] {r.title}")
            print(f"   {r.url}")
            print(f"   {r.snippet[:80]}...")

        print("\nStats:", json.dumps(searcher.stats, indent=2))
    else:
        print("SearXNG not available - start it with ./start.sh")

    await searcher.close()


if __name__ == "__main__":
    asyncio.run(main())
