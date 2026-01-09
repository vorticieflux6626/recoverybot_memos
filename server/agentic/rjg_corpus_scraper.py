"""
RJG Scientific Molding Corpus Scraper

Builds a domain-specific knowledge corpus from RJG Inc. and related
scientific molding resources. Integrates with:
- DomainCorpus for persistent storage
- ContentCache for deduplication
- RJG Schema for entity extraction

Sources:
- rjginc.com/resources (public articles, case studies)
- Plastics Technology (ptonline.com)
- Priamus cavity pressure documentation
- Public injection molding troubleshooting guides

Usage:
    from agentic.rjg_corpus_scraper import RJGCorpusScraper

    scraper = RJGCorpusScraper()
    await scraper.build_corpus()

    # Or scrape specific URL
    result = await scraper.scrape_url("https://rjginc.com/resources/some-article")

Author: Claude Code
Date: December 2025
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx

from .content_cache import get_content_cache
from .llm_config import get_llm_config
from .domain_corpus import (
    DomainCorpus,
    DomainSchema,
    DomainEntityDef,
    DomainRelationDef,
    CorpusBuilder,
    TroubleshootingEntityType,
    TroubleshootingRelationType,
)
from .schemas.rjg_schema import (
    RJG_SCHEMA,
    RJGEntityType,
    RJG_DEFECT_PATTERNS,
    RJG_PROCESS_VARIABLE_PATTERNS,
    RJG_TECHNIQUE_PATTERNS,
    is_scientific_molding_query,
)

logger = logging.getLogger("agentic.rjg_scraper")


# ============================================
# RJG DOMAIN SCHEMA FOR CORPUS
# ============================================

def create_rjg_corpus_schema() -> DomainSchema:
    """
    Create DomainSchema for RJG scientific molding corpus.

    This translates the RJG_SCHEMA into the DomainCorpus format.
    """
    entity_types = [
        DomainEntityDef(
            entity_type="process_variable",
            description="Measurable injection molding process parameters",
            extraction_patterns=[p for p, _ in RJG_PROCESS_VARIABLE_PATTERNS],
            examples=["melt temperature", "injection velocity", "pack pressure", "cushion"],
            attributes=["unit", "typical_range", "effect_on_quality"]
        ),
        DomainEntityDef(
            entity_type="defect",
            description="Injection molding part defects",
            extraction_patterns=[p for p, _ in RJG_DEFECT_PATTERNS],
            examples=["short shot", "flash", "sink mark", "weld line", "burn mark"],
            attributes=["phase_related", "severity", "visual_indicators"]
        ),
        DomainEntityDef(
            entity_type="technique",
            description="Scientific molding techniques and methods",
            extraction_patterns=RJG_TECHNIQUE_PATTERNS,
            examples=["Decoupled III", "DOE", "viscosity curve", "gate seal study"],
            attributes=["complexity", "equipment_required", "benefits"]
        ),
        DomainEntityDef(
            entity_type="sensor",
            description="Cavity pressure and process monitoring sensors",
            extraction_patterns=[
                r"cavity\s*pressure\s*(sensor|transducer)?",
                r"piezoelectric",
                r"strain\s*gauge",
            ],
            examples=["cavity pressure sensor", "piezoelectric transducer", "post-mounted sensor"],
            attributes=["sensor_type", "measurement_range", "mounting"]
        ),
        DomainEntityDef(
            entity_type="product",
            description="RJG products and systems",
            extraction_patterns=[r"\beDART\b", r"\bCoPilot\b", r"\bHUB\b", r"\bMAX\b"],
            examples=["eDART", "CoPilot", "HUB", "MAX", "Lynx"],
            attributes=["function", "compatibility"]
        ),
        DomainEntityDef(
            entity_type="quality_metric",
            description="Quality measurement and statistics",
            extraction_patterns=[
                r"CPK|Cpk|process\s*capability",
                r"SPC",
                r"DOE|design\s*of\s*experiments",
            ],
            examples=["Cpk", "SPC chart", "DOE", "OEE"],
            attributes=["target_value", "calculation_method"]
        ),
        DomainEntityDef(
            entity_type="material",
            description="Plastic materials and their properties",
            extraction_patterns=[
                r"(ABS|PC|PP|PE|PA|POM|PEEK|PPS|PBT)",
                r"glass[-\s]?filled",
                r"(amorphous|crystalline|semi-crystalline)",
            ],
            examples=["ABS", "PC/ABS blend", "glass-filled PA66", "PEEK"],
            attributes=["type", "melt_temp_range", "shrinkage"]
        ),
    ]

    relationships = [
        DomainRelationDef(
            relation_type="causes",
            source_types=["process_variable"],
            target_types=["defect"],
            description="Process condition causes defect"
        ),
        DomainRelationDef(
            relation_type="prevents",
            source_types=["technique"],
            target_types=["defect"],
            description="Technique prevents defect"
        ),
        DomainRelationDef(
            relation_type="measures",
            source_types=["sensor"],
            target_types=["process_variable"],
            description="Sensor measures process variable"
        ),
        DomainRelationDef(
            relation_type="optimizes",
            source_types=["technique"],
            target_types=["process_variable"],
            description="Technique optimizes process variable"
        ),
        DomainRelationDef(
            relation_type="indicates",
            source_types=["defect"],
            target_types=["process_variable"],
            description="Defect indicates process issue"
        ),
        DomainRelationDef(
            relation_type="used_by",
            source_types=["sensor"],
            target_types=["product"],
            description="Sensor used by RJG product"
        ),
        DomainRelationDef(
            relation_type="affects",
            source_types=["material"],
            target_types=["process_variable"],
            description="Material property affects process settings"
        ),
    ]

    return DomainSchema(
        domain_id="rjg_scientific_molding",
        domain_name="RJG Scientific Molding",
        description="Scientific injection molding methodology, cavity pressure monitoring, and defect troubleshooting",
        entity_types=entity_types,
        relationships=relationships,
        extraction_hints={
            "defect_patterns": [p for p, _ in RJG_DEFECT_PATTERNS],
            "process_patterns": [p for p, _ in RJG_PROCESS_VARIABLE_PATTERNS],
            "technique_patterns": RJG_TECHNIQUE_PATTERNS,
        },
        priority_patterns=[
            "cavity pressure", "scientific molding", "decoupled",
            "fill pack hold", "eDART", "viscosity curve"
        ]
    )


# ============================================
# RJG SEED URLs - PUBLIC RESOURCES
# ============================================

RJG_SEED_URLS: List[Dict[str, str]] = [
    # RJG Resources (public)
    {
        "url": "https://rjginc.com/resources/",
        "source_type": "resource_hub",
        "priority": "high"
    },
    # Plastics Technology - Scientific Molding Articles
    {
        "url": "https://www.ptonline.com/knowledgecenter/Injection-Molding",
        "source_type": "technical_article",
        "priority": "high"
    },
    # Plastics Today - Troubleshooting
    {
        "url": "https://www.plasticstoday.com/injection-molding",
        "source_type": "technical_article",
        "priority": "medium"
    },
]

# Specific high-value article URLs to scrape
RJG_ARTICLE_URLS: List[Dict[str, str]] = [
    # Scientific Molding Methodology
    {
        "url": "https://www.ptonline.com/articles/what-is-scientific-molding",
        "source_type": "methodology",
        "title": "What is Scientific Molding"
    },
    {
        "url": "https://www.ptonline.com/articles/the-decoupled-molding-process",
        "source_type": "methodology",
        "title": "Decoupled Molding Process"
    },
    # Defect Troubleshooting
    {
        "url": "https://www.ptonline.com/articles/troubleshooting-short-shots",
        "source_type": "troubleshooting",
        "title": "Troubleshooting Short Shots"
    },
    {
        "url": "https://www.ptonline.com/articles/troubleshooting-flash",
        "source_type": "troubleshooting",
        "title": "Troubleshooting Flash"
    },
    {
        "url": "https://www.ptonline.com/articles/troubleshooting-sink-marks",
        "source_type": "troubleshooting",
        "title": "Troubleshooting Sink Marks"
    },
    # Cavity Pressure
    {
        "url": "https://www.ptonline.com/articles/cavity-pressure-basics",
        "source_type": "cavity_pressure",
        "title": "Cavity Pressure Basics"
    },
    # Process Studies
    {
        "url": "https://www.ptonline.com/articles/gate-seal-study",
        "source_type": "process_study",
        "title": "Gate Seal Study"
    },
    {
        "url": "https://www.ptonline.com/articles/viscosity-curve-basics",
        "source_type": "process_study",
        "title": "Viscosity Curve Basics"
    },
]


# ============================================
# RJG CORPUS SCRAPER
# ============================================

@dataclass
class ScrapeResult:
    """Result of scraping a single URL"""
    url: str
    success: bool
    title: str = ""
    content: str = ""
    word_count: int = 0
    entities_extracted: int = 0
    relations_extracted: int = 0
    error: Optional[str] = None
    from_cache: bool = False


class RJGCorpusScraper:
    """
    Scraper for building RJG scientific molding knowledge corpus.

    Features:
    - Rate-limited scraping with respectful delays
    - Content caching to avoid re-scraping
    - Entity extraction via LLM
    - Integration with DomainCorpus
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        rate_limit_delay: float = 2.0,
        max_retries: int = 3,
        extraction_model: Optional[str] = None
    ):
        """
        Initialize the RJG corpus scraper.

        Args:
            db_path: Path to SQLite database for corpus storage
            ollama_url: URL for Ollama API (entity extraction)
            rate_limit_delay: Seconds between requests
            max_retries: Max retry attempts per URL
            extraction_model: Model for entity extraction (defaults to config)
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "cache" / "rjg_corpus.db")

        # Load model from central config if not provided
        # RJG is IMM-related, so use imm_extractor config
        llm_config = get_llm_config()
        extraction_model = extraction_model or llm_config.corpus.imm_extractor.model

        self.schema = create_rjg_corpus_schema()
        self.corpus = DomainCorpus(
            schema=self.schema,
            db_path=db_path,
            ollama_url=ollama_url
        )
        self.builder = CorpusBuilder(
            corpus=self.corpus,
            extraction_model=extraction_model
        )
        self.content_cache = get_content_cache()
        self.ollama_url = ollama_url
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

        self._scraped_urls: Set[str] = set()
        self._stats = {
            "urls_scraped": 0,
            "urls_cached": 0,
            "urls_failed": 0,
            "total_content_bytes": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
        }

    async def scrape_url(
        self,
        url: str,
        source_type: str = "article",
        title: str = "",
        extract_entities: bool = True
    ) -> ScrapeResult:
        """
        Scrape a single URL and add to corpus.

        Args:
            url: URL to scrape
            source_type: Type of source (article, manual, forum, etc.)
            title: Optional title override
            extract_entities: Whether to run entity extraction

        Returns:
            ScrapeResult with extraction statistics
        """
        # Check cache first
        cached = self.content_cache.get_content(url)
        if cached and cached.get("success"):
            logger.info(f"Cache hit for {url}")
            self._stats["urls_cached"] += 1

            # Still add to corpus if not already there
            if not self.corpus.has_content(cached["content"]):
                result = await self.builder.add_document(
                    content=cached["content"],
                    source_url=url,
                    source_type=source_type,
                    title=cached.get("title", title),
                    extract_entities=extract_entities
                )
                return ScrapeResult(
                    url=url,
                    success=True,
                    title=cached.get("title", title),
                    content=cached["content"][:500] + "...",
                    word_count=len(cached["content"].split()),
                    entities_extracted=result.get("entities", 0),
                    relations_extracted=result.get("relations", 0),
                    from_cache=True
                )
            return ScrapeResult(
                url=url,
                success=True,
                title=cached.get("title", title),
                word_count=len(cached["content"].split()),
                from_cache=True
            )

        # Scrape URL
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=30.0,
                    follow_redirects=True,
                    headers={
                        "User-Agent": "RJG-Corpus-Builder/1.0 (Educational/Research)",
                        "Accept": "text/html,application/xhtml+xml"
                    }
                ) as client:
                    response = await client.get(url)

                    if response.status_code != 200:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.rate_limit_delay * 2)
                            continue
                        return ScrapeResult(
                            url=url,
                            success=False,
                            error=f"HTTP {response.status_code}"
                        )

                    # Extract content
                    html = response.text
                    extracted_title, content = self._extract_content(html)

                    if not content or len(content) < 100:
                        return ScrapeResult(
                            url=url,
                            success=False,
                            error="No meaningful content extracted"
                        )

                    # Cache the content
                    self.content_cache.set_content(
                        url=url,
                        title=extracted_title or title,
                        content=content,
                        content_type="html",
                        success=True
                    )

                    # Add to corpus
                    result = await self.builder.add_document(
                        content=content,
                        source_url=url,
                        source_type=source_type,
                        title=extracted_title or title,
                        extract_entities=extract_entities
                    )

                    self._stats["urls_scraped"] += 1
                    self._stats["total_content_bytes"] += len(content)
                    self._stats["entities_extracted"] += result.get("entities", 0)
                    self._stats["relations_extracted"] += result.get("relations", 0)
                    self._scraped_urls.add(url)

                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)

                    return ScrapeResult(
                        url=url,
                        success=True,
                        title=extracted_title or title,
                        content=content[:500] + "...",
                        word_count=len(content.split()),
                        entities_extracted=result.get("entities", 0),
                        relations_extracted=result.get("relations", 0),
                        from_cache=False
                    )

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.rate_limit_delay * 2)
                    continue
                return ScrapeResult(url=url, success=False, error="Timeout")
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                return ScrapeResult(url=url, success=False, error=str(e))

        self._stats["urls_failed"] += 1
        return ScrapeResult(url=url, success=False, error="Max retries exceeded")

    def _extract_content(self, html: str) -> tuple[str, str]:
        """
        Extract title and main content from HTML.

        Simple extraction without heavy dependencies.
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

    async def build_corpus(
        self,
        seed_urls: bool = True,
        article_urls: bool = True,
        max_urls: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build the RJG corpus from configured URLs.

        Args:
            seed_urls: Whether to scrape seed URLs
            article_urls: Whether to scrape specific article URLs
            max_urls: Maximum number of URLs to scrape

        Returns:
            Build statistics
        """
        urls_to_scrape = []

        if seed_urls:
            urls_to_scrape.extend(RJG_SEED_URLS)

        if article_urls:
            urls_to_scrape.extend(RJG_ARTICLE_URLS)

        if max_urls:
            urls_to_scrape = urls_to_scrape[:max_urls]

        results = []
        for url_info in urls_to_scrape:
            url = url_info["url"]
            source_type = url_info.get("source_type", "article")
            title = url_info.get("title", "")

            logger.info(f"Scraping: {url}")
            result = await self.scrape_url(url, source_type, title)
            results.append(result)

            if result.success:
                logger.info(f"  ✓ {result.word_count} words, {result.entities_extracted} entities")
            else:
                logger.warning(f"  ✗ {result.error}")

        return {
            "urls_attempted": len(urls_to_scrape),
            "urls_successful": sum(1 for r in results if r.success),
            "urls_failed": sum(1 for r in results if not r.success),
            "urls_from_cache": sum(1 for r in results if r.from_cache),
            "total_entities": self._stats["entities_extracted"],
            "total_relations": self._stats["relations_extracted"],
            "corpus_stats": self.corpus.get_stats(),
            "results": [
                {
                    "url": r.url,
                    "success": r.success,
                    "title": r.title,
                    "word_count": r.word_count,
                    "entities": r.entities_extracted,
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
        title: str = ""
    ) -> Dict[str, Any]:
        """
        Add manually provided content to corpus.

        Useful for adding content from PDFs, internal docs, etc.

        Args:
            content: Text content to add
            source: Source identifier (filename, URL, etc.)
            source_type: Type of source
            title: Optional title

        Returns:
            Extraction statistics
        """
        return await self.builder.add_document(
            content=content,
            source_url=source,
            source_type=source_type,
            title=title,
            extract_entities=True
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get scraper and corpus statistics"""
        return {
            "scraper_stats": self._stats,
            "corpus_stats": self.corpus.get_stats(),
            "scraped_urls": list(self._scraped_urls)
        }

    async def query(
        self,
        query: str,
        top_k: int = 5,
        include_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RJG corpus.

        Args:
            query: Search query
            top_k: Number of results to return
            include_relations: Whether to include related entities

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
            "results": response_entities,
            "total_entities_searched": len(self.corpus.entities)
        }

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))


# ============================================
# FACTORY FUNCTION
# ============================================

_scraper_instance: Optional[RJGCorpusScraper] = None


def get_rjg_scraper() -> RJGCorpusScraper:
    """Get singleton RJG corpus scraper instance"""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = RJGCorpusScraper()
    return _scraper_instance


# ============================================
# CLI INTERFACE
# ============================================

async def main():
    """CLI interface for building RJG corpus"""
    import argparse

    parser = argparse.ArgumentParser(description="Build RJG Scientific Molding Corpus")
    parser.add_argument("--url", help="Scrape specific URL")
    parser.add_argument("--build", action="store_true", help="Build corpus from seed URLs")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    parser.add_argument("--query", help="Query the corpus")
    parser.add_argument("--max-urls", type=int, help="Maximum URLs to scrape")

    args = parser.parse_args()

    scraper = RJGCorpusScraper()

    if args.url:
        result = await scraper.scrape_url(args.url)
        print(f"Scraped: {result.url}")
        print(f"  Success: {result.success}")
        print(f"  Title: {result.title}")
        print(f"  Words: {result.word_count}")
        print(f"  Entities: {result.entities_extracted}")
        if result.error:
            print(f"  Error: {result.error}")

    elif args.build:
        print("Building RJG corpus...")
        result = await scraper.build_corpus(max_urls=args.max_urls)
        print(f"\nBuild Complete:")
        print(f"  URLs Attempted: {result['urls_attempted']}")
        print(f"  URLs Successful: {result['urls_successful']}")
        print(f"  URLs from Cache: {result['urls_from_cache']}")
        print(f"  Total Entities: {result['total_entities']}")
        print(f"  Total Relations: {result['total_relations']}")

    elif args.query:
        result = await scraper.query(args.query)
        print(f"Query: {args.query}")
        print(f"Results: {len(result['results'])}")
        for r in result['results']:
            print(f"  - {r['name']} ({r['type']}): {r['similarity']:.3f}")
            if r.get('related'):
                for rel in r['related'][:2]:
                    print(f"      → {rel['relation']}: {rel['entity']}")

    elif args.stats:
        stats = scraper.get_stats()
        print("RJG Corpus Statistics:")
        print(f"  Domain: {stats['corpus_stats']['domain_name']}")
        print(f"  Total Entities: {stats['corpus_stats']['total_entities']}")
        print(f"  Total Relations: {stats['corpus_stats']['total_relations']}")
        print(f"  Total Documents: {stats['corpus_stats']['total_documents']}")
        print(f"  Entity Types:")
        for etype, count in stats['corpus_stats'].get('entity_types', {}).items():
            print(f"    - {etype}: {count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
