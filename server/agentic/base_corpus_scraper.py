"""
Base Corpus Scraper - Unified Foundation for Domain Scrapers

Provides a common base class for domain-specific corpus scrapers:
- PLCCorpusScraper (PLC/Automation)
- RJGCorpusScraper (Scientific Molding)
- Future domain scrapers

Features:
- Unified rate limiting via aiometer
- Redis caching with circuit breaker fallback
- Standardized User-Agent management
- Metrics and monitoring hooks
- Common HTML content extraction

Phase 3 of scraping consolidation (2026-01).

Usage:
    class MyDomainScraper(BaseCorpusScraper):
        def create_schema(self) -> DomainSchema:
            return DomainSchema(...)

        def get_seed_urls(self) -> List[Dict[str, str]]:
            return [...]

        def get_article_urls(self) -> List[Dict[str, str]]:
            return [...]

    scraper = MyDomainScraper()
    await scraper.build_corpus()
"""

import asyncio
import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx

from .rate_limiter import get_rate_limiter, UnifiedRateLimiter, RateLimitedClient
from .redis_cache_service import get_redis_cache_service, RedisCacheService
from .user_agent_config import get_user_agent, UserAgents
from .domain_corpus import DomainCorpus, DomainSchema, CorpusBuilder
from .llm_config import get_llm_config
from .proxy_manager import get_proxy_manager

logger = logging.getLogger("agentic.base_corpus_scraper")


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class ScrapeResult:
    """Result of scraping a single URL."""
    url: str
    success: bool
    title: str = ""
    content: str = ""
    content_preview: str = ""
    word_count: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    from_cache: bool = False
    duration_ms: float = 0.0


@dataclass
class ScraperMetrics:
    """Metrics for scraper operations."""
    urls_scraped: int = 0
    urls_cached: int = 0
    urls_failed: int = 0
    total_content_bytes: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    total_duration_ms: float = 0.0
    rate_limit_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    by_source_type: Dict[str, int] = field(default_factory=dict)
    by_status_code: Dict[int, int] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def record_scrape(
        self,
        success: bool,
        from_cache: bool,
        source_type: str,
        status_code: Optional[int],
        content_bytes: int,
        entities: int,
        relations: int,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """Record a scrape operation."""
        if success:
            if from_cache:
                self.urls_cached += 1
                self.cache_hits += 1
            else:
                self.urls_scraped += 1
                self.cache_misses += 1
        else:
            self.urls_failed += 1
            if error:
                self.errors.append({
                    "error": error,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                # Keep only last 100 errors
                if len(self.errors) > 100:
                    self.errors = self.errors[-100:]

        self.total_content_bytes += content_bytes
        self.entities_extracted += entities
        self.relations_extracted += relations
        self.total_duration_ms += duration_ms

        # Track by source type
        self.by_source_type[source_type] = self.by_source_type.get(source_type, 0) + 1

        # Track by status code
        if status_code:
            self.by_status_code[status_code] = self.by_status_code.get(status_code, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        total = self.urls_scraped + self.urls_cached + self.urls_failed
        return {
            "urls_scraped": self.urls_scraped,
            "urls_cached": self.urls_cached,
            "urls_failed": self.urls_failed,
            "total_urls": total,
            "success_rate": (self.urls_scraped + self.urls_cached) / total if total > 0 else 0,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "total_content_bytes": self.total_content_bytes,
            "total_content_mb": round(self.total_content_bytes / (1024 * 1024), 2),
            "entities_extracted": self.entities_extracted,
            "relations_extracted": self.relations_extracted,
            "avg_duration_ms": self.total_duration_ms / total if total > 0 else 0,
            "rate_limit_hits": self.rate_limit_hits,
            "by_source_type": self.by_source_type,
            "by_status_code": self.by_status_code,
            "recent_errors": self.errors[-10:] if self.errors else []
        }


# ============================================
# BASE CORPUS SCRAPER
# ============================================

class BaseCorpusScraper(ABC):
    """
    Abstract base class for domain-specific corpus scrapers.

    Subclasses must implement:
    - create_schema(): Return the DomainSchema for this corpus
    - get_seed_urls(): Return list of seed URLs to scrape
    - get_article_urls(): Return list of specific article URLs
    - get_user_agent(): Return User-Agent string for this scraper

    Optional overrides:
    - get_extraction_model(): Return model name for entity extraction
    - filter_url(): Custom URL filtering logic
    - transform_content(): Custom content transformation
    """

    def __init__(
        self,
        domain_id: str,
        db_path: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        rate_limit_delay: float = 2.0,
        max_retries: int = 3,
        extraction_model: Optional[str] = None,
        use_rate_limiter: bool = True,
        use_redis_cache: bool = True,
        cache_ttl_seconds: int = 3600
    ):
        """
        Initialize the corpus scraper.

        Args:
            domain_id: Unique identifier for this domain (e.g., "plc_automation")
            db_path: Path to SQLite database for corpus storage
            ollama_url: URL for Ollama API (entity extraction)
            rate_limit_delay: Fallback delay between requests (if rate limiter disabled)
            max_retries: Max retry attempts per URL
            extraction_model: Model for entity extraction (defaults to config)
            use_rate_limiter: Whether to use unified rate limiter
            use_redis_cache: Whether to use Redis cache service
            cache_ttl_seconds: Cache TTL for content
        """
        self.domain_id = domain_id
        self.ollama_url = ollama_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.cache_ttl_seconds = cache_ttl_seconds

        # Set up database path
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "cache" / f"{domain_id}_corpus.db")
        self.db_path = db_path

        # Get extraction model from config if not provided
        extraction_model = extraction_model or self.get_extraction_model()
        self._extraction_model = extraction_model

        # Create schema and corpus
        self.schema = self.create_schema()
        self.corpus = DomainCorpus(
            schema=self.schema,
            db_path=db_path,
            ollama_url=ollama_url
        )
        self.builder = CorpusBuilder(
            corpus=self.corpus,
            extraction_model=extraction_model
        )

        # Rate limiter (unified)
        self._use_rate_limiter = use_rate_limiter
        self._rate_limiter: Optional[UnifiedRateLimiter] = None
        if use_rate_limiter:
            self._rate_limiter = get_rate_limiter()

        # Redis cache (unified)
        self._use_redis_cache = use_redis_cache
        self._redis_cache: Optional[RedisCacheService] = None
        if use_redis_cache:
            self._redis_cache = get_redis_cache_service()

        # Legacy content cache fallback
        from .content_cache import get_content_cache
        self._content_cache = get_content_cache()

        # Tracking
        self._scraped_urls: Set[str] = set()
        self._metrics = ScraperMetrics()

        # Expose entity/relation types for inspection
        self.entity_types = [e.entity_type for e in self.schema.entity_types]
        self.relation_types = [r.relation_type for r in self.schema.relationships]

        logger.info(
            f"BaseCorpusScraper initialized: domain={domain_id}, "
            f"rate_limiter={use_rate_limiter}, redis_cache={use_redis_cache}"
        )

    # ============================================
    # ABSTRACT METHODS - Must be implemented
    # ============================================

    @abstractmethod
    def create_schema(self) -> DomainSchema:
        """
        Create the DomainSchema for this corpus.

        Returns:
            DomainSchema with entity types and relationships
        """
        pass

    @abstractmethod
    def get_seed_urls(self) -> List[Dict[str, str]]:
        """
        Return list of seed URLs to scrape.

        Each dict should have:
        - url: The URL to scrape
        - source_type: Type of source (e.g., "knowledge_base", "forum")
        - priority: "high", "medium", or "low"

        Returns:
            List of URL info dicts
        """
        pass

    @abstractmethod
    def get_article_urls(self) -> List[Dict[str, str]]:
        """
        Return list of specific article URLs to scrape.

        Each dict should have:
        - url: The URL to scrape
        - source_type: Type of source
        - title: Article title

        Returns:
            List of URL info dicts
        """
        pass

    @abstractmethod
    def get_user_agent(self) -> str:
        """
        Return the User-Agent string for this scraper.

        Returns:
            User-Agent string
        """
        pass

    # ============================================
    # OPTIONAL OVERRIDES
    # ============================================

    def get_extraction_model(self) -> str:
        """
        Get the model to use for entity extraction.

        Override to customize based on domain.
        Default: Uses LLM config's entity_extractor model.
        """
        llm_config = get_llm_config()
        return llm_config.utility.entity_extractor.model

    def filter_url(self, url: str, url_info: Dict[str, str]) -> bool:
        """
        Filter URLs before scraping.

        Override to add domain-specific filtering logic.

        Args:
            url: The URL to check
            url_info: URL metadata dict

        Returns:
            True if URL should be scraped, False to skip
        """
        return True

    def transform_content(self, content: str, url: str) -> str:
        """
        Transform extracted content before storage.

        Override to add domain-specific content processing.

        Args:
            content: Raw extracted content
            url: Source URL

        Returns:
            Transformed content
        """
        return content

    def extract_metadata(self, url: str, html: str) -> Dict[str, Any]:
        """
        Extract domain-specific metadata from HTML.

        Override to extract custom metadata.

        Args:
            url: Source URL
            html: Raw HTML content

        Returns:
            Metadata dict
        """
        return {}

    # ============================================
    # CORE SCRAPING LOGIC
    # ============================================

    async def _ensure_cache_connected(self):
        """Ensure Redis cache is connected."""
        if self._redis_cache and not self._redis_cache._connected:
            await self._redis_cache.connect()

    async def scrape_url(
        self,
        url: str,
        source_type: str = "article",
        title: str = "",
        extract_entities: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ScrapeResult:
        """
        Scrape a single URL and add to corpus.

        Args:
            url: URL to scrape
            source_type: Type of source (article, manual, forum, etc.)
            title: Optional title override
            extract_entities: Whether to run entity extraction
            metadata: Additional metadata to store

        Returns:
            ScrapeResult with extraction statistics
        """
        start_time = time.time()
        metadata = metadata or {}

        # Ensure cache is connected
        await self._ensure_cache_connected()

        # Check cache first
        cached = await self._get_from_cache(url)
        if cached and cached.get("success"):
            logger.debug(f"Cache hit for {url}")
            duration_ms = (time.time() - start_time) * 1000

            # Still add to corpus if not already there
            if not self.corpus.has_content(cached["content"]):
                result = await self.builder.add_document(
                    content=cached["content"],
                    source_url=url,
                    source_type=source_type,
                    title=cached.get("title", title),
                    extract_entities=extract_entities,
                    metadata=metadata
                )
                entities = result.get("entities", 0)
                relations = result.get("relations", 0)
            else:
                entities = 0
                relations = 0

            self._metrics.record_scrape(
                success=True,
                from_cache=True,
                source_type=source_type,
                status_code=None,
                content_bytes=len(cached["content"]),
                entities=entities,
                relations=relations,
                duration_ms=duration_ms
            )

            return ScrapeResult(
                url=url,
                success=True,
                title=cached.get("title", title),
                content=cached["content"],
                content_preview=cached["content"][:500] + "..." if len(cached["content"]) > 500 else cached["content"],
                word_count=len(cached["content"].split()),
                entities_extracted=entities,
                relations_extracted=relations,
                metadata=metadata,
                from_cache=True,
                duration_ms=duration_ms
            )

        # Scrape URL with retries
        for attempt in range(self.max_retries):
            try:
                # Use rate limiter if available
                if self._rate_limiter:
                    async with RateLimitedClient(self._rate_limiter) as client:
                        fetch_result = await client.get(url)

                        if not fetch_result.success:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                                continue

                            duration_ms = (time.time() - start_time) * 1000
                            self._metrics.record_scrape(
                                success=False,
                                from_cache=False,
                                source_type=source_type,
                                status_code=fetch_result.status_code,
                                content_bytes=0,
                                entities=0,
                                relations=0,
                                duration_ms=duration_ms,
                                error=fetch_result.error
                            )

                            return ScrapeResult(
                                url=url,
                                success=False,
                                metadata=metadata,
                                error=fetch_result.error,
                                duration_ms=duration_ms
                            )

                        html = fetch_result.content.decode("utf-8", errors="replace")
                        status_code = fetch_result.status_code
                else:
                    # Fallback to direct httpx (with optional proxy support)
                    proxy_manager = get_proxy_manager()
                    proxy_config = None
                    proxy_url = None

                    # Get proxy if available
                    if proxy_manager.has_proxies():
                        proxy_url = await proxy_manager.get_proxy()
                        proxy_config = proxy_manager.get_proxy_config(proxy_url)

                    async with httpx.AsyncClient(
                        timeout=30.0,
                        follow_redirects=True,
                        proxy=proxy_config,
                        headers={
                            "User-Agent": self.get_user_agent(),
                            "Accept": "text/html,application/xhtml+xml"
                        }
                    ) as client:
                        response = await client.get(url)
                        status_code = response.status_code

                        # Report proxy result for health tracking
                        if proxy_manager.has_proxies() and proxy_url:
                            await proxy_manager.report_result(
                                proxy_url,
                                success=(response.status_code == 200),
                                latency_ms=(time.time() - start_time) * 1000
                            )

                        if response.status_code != 200:
                            if attempt < self.max_retries - 1:
                                await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                                continue

                            duration_ms = (time.time() - start_time) * 1000
                            self._metrics.record_scrape(
                                success=False,
                                from_cache=False,
                                source_type=source_type,
                                status_code=status_code,
                                content_bytes=0,
                                entities=0,
                                relations=0,
                                duration_ms=duration_ms,
                                error=f"HTTP {status_code}"
                            )

                            return ScrapeResult(
                                url=url,
                                success=False,
                                metadata=metadata,
                                error=f"HTTP {status_code}",
                                duration_ms=duration_ms
                            )

                        html = response.text

                # Extract content
                extracted_title, content = self._extract_content(html)

                # Apply domain-specific transformation
                content = self.transform_content(content, url)

                # Extract domain-specific metadata
                extracted_metadata = self.extract_metadata(url, html)
                metadata.update(extracted_metadata)

                if not content or len(content) < 100:
                    duration_ms = (time.time() - start_time) * 1000
                    self._metrics.record_scrape(
                        success=False,
                        from_cache=False,
                        source_type=source_type,
                        status_code=status_code,
                        content_bytes=0,
                        entities=0,
                        relations=0,
                        duration_ms=duration_ms,
                        error="No meaningful content extracted"
                    )

                    return ScrapeResult(
                        url=url,
                        success=False,
                        metadata=metadata,
                        error="No meaningful content extracted",
                        duration_ms=duration_ms
                    )

                # Cache the content
                await self._set_cache(url, extracted_title or title, content)

                # Add to corpus
                result = await self.builder.add_document(
                    content=content,
                    source_url=url,
                    source_type=source_type,
                    title=extracted_title or title,
                    extract_entities=extract_entities,
                    metadata=metadata
                )

                entities = result.get("entities", 0)
                relations = result.get("relations", 0)
                self._scraped_urls.add(url)

                duration_ms = (time.time() - start_time) * 1000
                self._metrics.record_scrape(
                    success=True,
                    from_cache=False,
                    source_type=source_type,
                    status_code=status_code,
                    content_bytes=len(content),
                    entities=entities,
                    relations=relations,
                    duration_ms=duration_ms
                )

                # Rate limiting (if not using unified rate limiter)
                if not self._rate_limiter:
                    await asyncio.sleep(self.rate_limit_delay)

                return ScrapeResult(
                    url=url,
                    success=True,
                    title=extracted_title or title,
                    content=content,
                    content_preview=content[:500] + "..." if len(content) > 500 else content,
                    word_count=len(content.split()),
                    entities_extracted=entities,
                    relations_extracted=relations,
                    metadata=metadata,
                    from_cache=False,
                    duration_ms=duration_ms
                )

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.rate_limit_delay * (attempt + 1))
                    continue

                duration_ms = (time.time() - start_time) * 1000
                self._metrics.record_scrape(
                    success=False,
                    from_cache=False,
                    source_type=source_type,
                    status_code=None,
                    content_bytes=0,
                    entities=0,
                    relations=0,
                    duration_ms=duration_ms,
                    error="Timeout"
                )

                return ScrapeResult(
                    url=url,
                    success=False,
                    metadata=metadata,
                    error="Timeout",
                    duration_ms=duration_ms
                )

            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                duration_ms = (time.time() - start_time) * 1000
                self._metrics.record_scrape(
                    success=False,
                    from_cache=False,
                    source_type=source_type,
                    status_code=None,
                    content_bytes=0,
                    entities=0,
                    relations=0,
                    duration_ms=duration_ms,
                    error=str(e)
                )

                return ScrapeResult(
                    url=url,
                    success=False,
                    metadata=metadata,
                    error=str(e),
                    duration_ms=duration_ms
                )

        # Should not reach here, but just in case
        duration_ms = (time.time() - start_time) * 1000
        return ScrapeResult(
            url=url,
            success=False,
            metadata=metadata,
            error="Max retries exceeded",
            duration_ms=duration_ms
        )

    def _extract_content(self, html: str) -> Tuple[str, str]:
        """
        Extract title and main content from HTML.

        Simple extraction without heavy dependencies.

        Args:
            html: Raw HTML content

        Returns:
            Tuple of (title, content)
        """
        # Extract title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        # Remove scripts and styles
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Try to find main content areas
        content = ""

        # Look for article content
        article_match = re.search(
            r"<article[^>]*>(.*?)</article>",
            html, flags=re.DOTALL | re.IGNORECASE
        )
        if article_match:
            content = article_match.group(1)
        else:
            # Look for main content div
            main_match = re.search(
                r'<(main|div)[^>]*class="[^"]*content[^"]*"[^>]*>(.*?)</\1>',
                html, flags=re.DOTALL | re.IGNORECASE
            )
            if main_match:
                content = main_match.group(2)
            else:
                # Look for post content (forums)
                post_match = re.search(
                    r'<div[^>]*class="[^"]*(?:post|message|thread)[^"]*"[^>]*>(.*?)</div>',
                    html, flags=re.DOTALL | re.IGNORECASE
                )
                if post_match:
                    content = post_match.group(1)
                else:
                    # Fall back to body
                    body_match = re.search(
                        r"<body[^>]*>(.*?)</body>",
                        html, flags=re.DOTALL | re.IGNORECASE
                    )
                    if body_match:
                        content = body_match.group(1)

        # Remove remaining HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content).strip()

        # Decode HTML entities
        content = content.replace("&nbsp;", " ")
        content = content.replace("&amp;", "&")
        content = content.replace("&lt;", "<")
        content = content.replace("&gt;", ">")
        content = content.replace("&quot;", '"')
        content = content.replace("&#39;", "'")

        return title, content

    # ============================================
    # CACHE OPERATIONS
    # ============================================

    async def _get_from_cache(self, url: str) -> Optional[Dict[str, Any]]:
        """Get content from cache (Redis or legacy)."""
        # Try Redis cache first
        if self._redis_cache:
            try:
                result = await self._redis_cache.get_content(url)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Redis cache get error: {e}")

        # Fall back to legacy content cache
        cached = self._content_cache.get_content(url)
        if cached and cached.get("success"):
            return cached

        return None

    async def _set_cache(self, url: str, title: str, content: str):
        """Set content in cache (Redis and legacy)."""
        # Set in Redis cache
        if self._redis_cache:
            try:
                await self._redis_cache.set_content(
                    url=url,
                    title=title,
                    content=content,
                    content_type="html",
                    success=True,
                    ttl_override=self.cache_ttl_seconds
                )
            except Exception as e:
                logger.debug(f"Redis cache set error: {e}")

        # Also set in legacy cache for compatibility
        self._content_cache.set_content(
            url=url,
            title=title,
            content=content,
            content_type="html",
            success=True
        )

    # ============================================
    # CORPUS BUILDING
    # ============================================

    async def build_corpus(
        self,
        seed_urls: bool = True,
        article_urls: bool = True,
        max_urls: Optional[int] = None,
        url_filter: Optional[Callable[[Dict[str, str]], bool]] = None
    ) -> Dict[str, Any]:
        """
        Build the corpus from configured URLs.

        Args:
            seed_urls: Whether to scrape seed URLs
            article_urls: Whether to scrape specific article URLs
            max_urls: Maximum number of URLs to scrape
            url_filter: Optional filter function for URLs

        Returns:
            Build statistics
        """
        urls_to_scrape = []

        if seed_urls:
            urls_to_scrape.extend(self.get_seed_urls())

        if article_urls:
            urls_to_scrape.extend(self.get_article_urls())

        # Apply custom filter if provided
        if url_filter:
            urls_to_scrape = [u for u in urls_to_scrape if url_filter(u)]

        # Apply base class filter
        urls_to_scrape = [
            u for u in urls_to_scrape
            if self.filter_url(u["url"], u)
        ]

        if max_urls:
            urls_to_scrape = urls_to_scrape[:max_urls]

        results = []
        for url_info in urls_to_scrape:
            url = url_info["url"]
            source_type = url_info.get("source_type", "article")
            title = url_info.get("title", "")
            metadata = {k: v for k, v in url_info.items() if k not in ["url", "source_type", "title"]}

            logger.info(f"Scraping: {url}")
            result = await self.scrape_url(url, source_type, title, metadata=metadata)
            results.append(result)

            if result.success:
                logger.info(
                    f"  ✓ {result.word_count} words, "
                    f"{result.entities_extracted} entities "
                    f"({result.duration_ms:.0f}ms)"
                )
            else:
                logger.warning(f"  ✗ {result.error}")

        return {
            "domain_id": self.domain_id,
            "urls_attempted": len(urls_to_scrape),
            "urls_successful": sum(1 for r in results if r.success),
            "urls_failed": sum(1 for r in results if not r.success),
            "urls_from_cache": sum(1 for r in results if r.from_cache),
            "total_entities": self._metrics.entities_extracted,
            "total_relations": self._metrics.relations_extracted,
            "metrics": self._metrics.to_dict(),
            "corpus_stats": self.corpus.get_stats(),
            "results": [
                {
                    "url": r.url,
                    "success": r.success,
                    "title": r.title,
                    "word_count": r.word_count,
                    "entities": r.entities_extracted,
                    "from_cache": r.from_cache,
                    "duration_ms": r.duration_ms,
                    "error": r.error
                }
                for r in results
            ]
        }

    async def add_manual_content(
        self,
        content: str,
        source: str,
        source_type: str = "manual",
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add manually provided content to corpus.

        Useful for adding content from PDFs, internal docs, etc.

        Args:
            content: Text content to add
            source: Source identifier (filename, URL, etc.)
            source_type: Type of source
            title: Optional title
            metadata: Additional metadata

        Returns:
            Extraction statistics
        """
        result = await self.builder.add_document(
            content=content,
            source_url=source,
            source_type=source_type,
            title=title or f"[{self.domain_id.upper()}] Manual Content",
            extract_entities=True,
            metadata=metadata or {}
        )

        self._metrics.record_scrape(
            success=True,
            from_cache=False,
            source_type=source_type,
            status_code=None,
            content_bytes=len(content),
            entities=result.get("entities", 0),
            relations=result.get("relations", 0),
            duration_ms=0
        )

        return result

    # ============================================
    # QUERY AND STATS
    # ============================================

    async def query(
        self,
        query: str,
        top_k: int = 5,
        include_relations: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the corpus.

        Args:
            query: Search query
            top_k: Number of results to return
            include_relations: Whether to include related entities
            filters: Optional filters for results

        Returns:
            Query results with entities and context
        """
        # Generate query embedding
        query_embedding = await self.corpus.generate_embedding(query)

        if not query_embedding:
            return {"error": "Failed to generate query embedding", "results": []}

        # Search entities by embedding similarity
        results = []
        for entity in self.corpus.entities.values():
            # Apply filters if provided
            if filters:
                skip = False
                for key, value in filters.items():
                    if entity.attributes.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            if entity.embedding:
                similarity = self._cosine_similarity(query_embedding, entity.embedding)
                results.append({
                    "entity": entity,
                    "similarity": similarity
                })

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:top_k]

        # Build response
        response_entities = []
        for r in results:
            entity = r["entity"]
            entity_dict = {
                "id": entity.id,
                "type": entity.entity_type,
                "name": entity.name,
                "description": entity.description,
                "similarity": r["similarity"],
                "attributes": entity.attributes
            }

            if include_relations:
                relations = self.corpus.get_relations_for_entity(entity.id)
                entity_dict["related"] = [
                    {
                        "entity": rel_entity.name,
                        "type": rel_entity.entity_type,
                        "relation": relation.relation_type
                    }
                    for rel_entity, relation in relations[:5]
                ]

            response_entities.append(entity_dict)

        return {
            "query": query,
            "domain_id": self.domain_id,
            "filters": filters,
            "results": response_entities,
            "count": len(response_entities),
            "total_entities_searched": len(self.corpus.entities)
        }

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))

    def get_stats(self) -> Dict[str, Any]:
        """Get scraper and corpus statistics."""
        return {
            "domain_id": self.domain_id,
            "scraper_metrics": self._metrics.to_dict(),
            "corpus_stats": self.corpus.get_stats(),
            "scraped_urls": list(self._scraped_urls),
            "rate_limiter_stats": self._rate_limiter.get_stats() if self._rate_limiter else None,
        }

    def get_metrics(self) -> ScraperMetrics:
        """Get the metrics object directly."""
        return self._metrics

    def reset_metrics(self):
        """Reset scraper metrics."""
        self._metrics = ScraperMetrics()
