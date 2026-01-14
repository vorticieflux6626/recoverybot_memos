"""
PLC/Automation Corpus Scraper

Builds a domain-specific knowledge corpus from PLC resources:
- Allen-Bradley/Rockwell Automation
- Siemens (TIA Portal, S7)
- AutomationDirect (Click, Do-more)

Sources:
- Official documentation portals
- PLCTalk forums (plctalk.net)
- Technical training sites (realpars, plcacademy)
- Manufacturer knowledge bases

Usage:
    from agentic.plc_corpus_scraper import PLCCorpusScraper

    scraper = PLCCorpusScraper()
    await scraper.build_corpus()

    # Or scrape specific URL
    result = await scraper.scrape_url("https://www.plctalk.net/threads/...")

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
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx

from .content_cache import get_content_cache
from .llm_config import get_llm_config
from .user_agent_config import UserAgents
from .domain_corpus import (
    DomainCorpus,
    DomainSchema,
    DomainEntityDef,
    DomainRelationDef,
    CorpusBuilder,
)
from .schemas.plc_schema import (
    PLC_SCHEMA,
    PLCEntityType,
    AB_FAULT_PATTERNS,
    AB_MODULE_PATTERNS,
    SIEMENS_FAULT_PATTERNS,
    SIEMENS_MODULE_PATTERNS,
    PROTOCOL_PATTERNS,
    is_plc_query,
    detect_plc_manufacturer,
)

logger = logging.getLogger("agentic.plc_scraper")


# ============================================
# PLC DOMAIN SCHEMA FOR CORPUS
# ============================================

def create_plc_corpus_schema() -> DomainSchema:
    """
    Create DomainSchema for PLC/Automation corpus.

    This translates PLC_SCHEMA into the DomainCorpus format.
    """
    # Get fault patterns
    all_fault_patterns = [p for p, _ in AB_FAULT_PATTERNS + SIEMENS_FAULT_PATTERNS]
    all_module_patterns = [p for p, _ in AB_MODULE_PATTERNS + SIEMENS_MODULE_PATTERNS]
    all_protocol_patterns = [p for p, _ in PROTOCOL_PATTERNS]

    entity_types = [
        DomainEntityDef(
            entity_type="fault_code",
            description="PLC fault and alarm codes",
            extraction_patterns=all_fault_patterns[:10],  # Limit for performance
            examples=["Type 4 Code 20", "F00001", "Major Fault 3:16", "Kinetix Fault 12"],
            attributes=["manufacturer", "severity", "remedy", "category"]
        ),
        DomainEntityDef(
            entity_type="module",
            description="PLC hardware modules (CPUs, I/O, drives)",
            extraction_patterns=all_module_patterns[:10],
            examples=["1756-L83E", "CPU 1516", "PowerFlex 525", "6ES7 516-3AN02"],
            attributes=["manufacturer", "series", "function", "part_number"]
        ),
        DomainEntityDef(
            entity_type="instruction",
            description="PLC programming instructions",
            extraction_patterns=[
                r"\b(XIC|XIO|OTE|OTL|OTU|TON|TOF|RTO|CTU|CTD|MOV|COP|ADD|SUB|MUL|DIV|JSR|RET|MSG)\b"
            ],
            examples=["XIC", "OTE", "TON", "MSG", "MOV"],
            attributes=["category", "usage", "parameters"]
        ),
        DomainEntityDef(
            entity_type="protocol",
            description="Industrial communication protocols",
            extraction_patterns=all_protocol_patterns[:8],
            examples=["EtherNet/IP", "Profinet", "Modbus TCP", "DeviceNet"],
            attributes=["speed", "topology", "manufacturer"]
        ),
        DomainEntityDef(
            entity_type="address",
            description="PLC memory addresses and tags",
            extraction_patterns=[
                r"\b[NIOBSTCRD]\d+[:/]\d+(?:/\d+)?\b",  # PLC-5/SLC
                r"\b[IMQPTCD]B?\d+(?:\.\d+)?\b",  # Siemens
                r"\bDB\d+\.DB[XWDB]\d+\b",  # Siemens DB
            ],
            examples=["N7:0", "B3:0/0", "MW100", "DB10.DBX0.0"],
            attributes=["type", "data_type", "scope"]
        ),
        DomainEntityDef(
            entity_type="procedure",
            description="Troubleshooting and maintenance procedures",
            extraction_patterns=[
                r"(step\s*\d+|procedure|troubleshoot|how\s*to|check\s*the)",
            ],
            examples=["Reset fault", "Download program", "Configure I/O", "Go online"],
            attributes=["manufacturer", "complexity", "tools_required"]
        ),
    ]

    relationships = [
        DomainRelationDef(
            relation_type="causes",
            source_types=["fault_code"],
            target_types=["fault_code", "module"],
            description="Fault causes another issue"
        ),
        DomainRelationDef(
            relation_type="resolves",
            source_types=["procedure"],
            target_types=["fault_code"],
            description="Procedure resolves fault"
        ),
        DomainRelationDef(
            relation_type="communicates_with",
            source_types=["module", "protocol"],
            target_types=["module"],
            description="Communication path"
        ),
        DomainRelationDef(
            relation_type="addresses",
            source_types=["instruction"],
            target_types=["address"],
            description="Instruction uses address"
        ),
        DomainRelationDef(
            relation_type="compatible_with",
            source_types=["module"],
            target_types=["module", "protocol"],
            description="Hardware/protocol compatibility"
        ),
    ]

    return DomainSchema(
        domain_id="plc_automation",
        domain_name="PLC/Automation Systems",
        description="Industrial PLC troubleshooting for Allen-Bradley, Siemens, and AutomationDirect",
        entity_types=entity_types,
        relationships=relationships,
        extraction_hints={
            "fault_patterns": all_fault_patterns[:10],
            "module_patterns": all_module_patterns[:10],
            "protocol_patterns": all_protocol_patterns[:5],
        },
        priority_patterns=[
            "fault", "alarm", "error", "troubleshoot", "1756", "6ES7",
            "controllogix", "s7-1500", "ethernet/ip", "profinet", "plc"
        ]
    )


# ============================================
# PLC SEED URLs - PUBLIC RESOURCES
# ============================================

PLC_SEED_URLS: List[Dict[str, str]] = [
    # Allen-Bradley / Rockwell
    {
        "url": "https://rockwellautomation.custhelp.com/",
        "source_type": "knowledge_base",
        "manufacturer": "allen_bradley",
        "priority": "high"
    },
    # Siemens Support
    {
        "url": "https://support.industry.siemens.com/cs/ww/en/",
        "source_type": "knowledge_base",
        "manufacturer": "siemens",
        "priority": "high"
    },
    # AutomationDirect
    {
        "url": "https://library.automationdirect.com/",
        "source_type": "technical_library",
        "manufacturer": "automationdirect",
        "priority": "high"
    },
    # PLC Forums
    {
        "url": "https://www.plctalk.net/",
        "source_type": "forum",
        "manufacturer": "multi",
        "priority": "high"
    },
    # Training Sites
    {
        "url": "https://www.realpars.com/",
        "source_type": "training",
        "manufacturer": "multi",
        "priority": "medium"
    },
    {
        "url": "https://www.plcacademy.com/",
        "source_type": "training",
        "manufacturer": "multi",
        "priority": "medium"
    },
]

# Specific high-value article URLs to scrape
PLC_ARTICLE_URLS: List[Dict[str, str]] = [
    # Allen-Bradley Troubleshooting
    {
        "url": "https://www.plctalk.net/threads/controllogix-major-faults-explained.130852/",
        "source_type": "forum",
        "title": "ControlLogix Major Faults Explained",
        "manufacturer": "allen_bradley"
    },
    {
        "url": "https://www.realpars.com/blog/plc-troubleshooting",
        "source_type": "training",
        "title": "PLC Troubleshooting Guide",
        "manufacturer": "multi"
    },
    {
        "url": "https://www.plcacademy.com/allen-bradley-plc-fault-codes/",
        "source_type": "training",
        "title": "Allen-Bradley PLC Fault Codes",
        "manufacturer": "allen_bradley"
    },
    # Siemens
    {
        "url": "https://www.plctalk.net/threads/siemens-s7-1500-cpu-stops-unexpectedly.148293/",
        "source_type": "forum",
        "title": "Siemens S7-1500 CPU Stops",
        "manufacturer": "siemens"
    },
    {
        "url": "https://www.realpars.com/blog/siemens-plc-programming",
        "source_type": "training",
        "title": "Siemens PLC Programming",
        "manufacturer": "siemens"
    },
    # Communication
    {
        "url": "https://www.realpars.com/blog/ethernet-ip-vs-profinet",
        "source_type": "training",
        "title": "EtherNet/IP vs Profinet",
        "manufacturer": "multi"
    },
    {
        "url": "https://www.plcacademy.com/modbus-rtu-simply-explained/",
        "source_type": "training",
        "title": "Modbus RTU Explained",
        "manufacturer": "multi"
    },
    # HMI
    {
        "url": "https://www.plctalk.net/threads/factorytalk-view-alarm-setup.155421/",
        "source_type": "forum",
        "title": "FactoryTalk View Alarm Setup",
        "manufacturer": "allen_bradley"
    },
    {
        "url": "https://www.realpars.com/blog/hmi-programming",
        "source_type": "training",
        "title": "HMI Programming Basics",
        "manufacturer": "multi"
    },
    # VFDs and Drives
    {
        "url": "https://www.plctalk.net/threads/powerflex-525-fault-codes.162384/",
        "source_type": "forum",
        "title": "PowerFlex 525 Fault Codes",
        "manufacturer": "allen_bradley"
    },
    {
        "url": "https://www.realpars.com/blog/vfd-troubleshooting",
        "source_type": "training",
        "title": "VFD Troubleshooting",
        "manufacturer": "multi"
    },
]


# ============================================
# PLC CORPUS SCRAPER
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
    manufacturer: str = ""
    error: Optional[str] = None
    from_cache: bool = False


class PLCCorpusScraper:
    """
    Scraper for building PLC/Automation knowledge corpus.

    Features:
    - Rate-limited scraping with respectful delays
    - Content caching to avoid re-scraping
    - Entity extraction via LLM
    - Manufacturer-aware indexing
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
        Initialize the PLC corpus scraper.

        Args:
            db_path: Path to SQLite database for corpus storage
            ollama_url: URL for Ollama API (entity extraction)
            rate_limit_delay: Seconds between requests
            max_retries: Max retry attempts per URL
            extraction_model: Model for entity extraction (defaults to config)
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "cache" / "plc_corpus.db")

        # Load model from central config if not provided
        llm_config = get_llm_config()
        extraction_model = extraction_model or llm_config.corpus.plc_extractor.model

        self.db_path = db_path
        self.schema = create_plc_corpus_schema()
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

        # Expose entity/relation types for inspection
        self.entity_types = [e.entity_type for e in self.schema.entity_types]
        self.relation_types = [r.relation_type for r in self.schema.relationships]

        self._scraped_urls: Set[str] = set()
        self._stats = {
            "urls_scraped": 0,
            "urls_cached": 0,
            "urls_failed": 0,
            "total_content_bytes": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "by_manufacturer": {
                "allen_bradley": 0,
                "siemens": 0,
                "automationdirect": 0,
                "multi": 0,
            }
        }

    async def scrape_url(
        self,
        url: str,
        source_type: str = "article",
        title: str = "",
        manufacturer: str = "",
        extract_entities: bool = True
    ) -> ScrapeResult:
        """
        Scrape a single URL and add to corpus.

        Args:
            url: URL to scrape
            source_type: Type of source (article, manual, forum, etc.)
            title: Optional title override
            manufacturer: Manufacturer (allen_bradley, siemens, automationdirect)
            extract_entities: Whether to run entity extraction

        Returns:
            ScrapeResult with extraction statistics
        """
        # Auto-detect manufacturer if not provided
        if not manufacturer:
            manufacturer = detect_plc_manufacturer(url) or "multi"

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
                    extract_entities=extract_entities,
                    metadata={"manufacturer": manufacturer}
                )
                return ScrapeResult(
                    url=url,
                    success=True,
                    title=cached.get("title", title),
                    content=cached["content"][:500] + "...",
                    word_count=len(cached["content"].split()),
                    entities_extracted=result.get("entities", 0),
                    relations_extracted=result.get("relations", 0),
                    manufacturer=manufacturer,
                    from_cache=True
                )
            return ScrapeResult(
                url=url,
                success=True,
                title=cached.get("title", title),
                word_count=len(cached["content"].split()),
                manufacturer=manufacturer,
                from_cache=True
            )

        # Scrape URL
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=30.0,
                    follow_redirects=True,
                    headers={
                        "User-Agent": UserAgents.PLC_CORPUS_BUILDER,
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
                            manufacturer=manufacturer,
                            error=f"HTTP {response.status_code}"
                        )

                    # Extract content
                    html = response.text
                    extracted_title, content = self._extract_content(html)

                    if not content or len(content) < 100:
                        return ScrapeResult(
                            url=url,
                            success=False,
                            manufacturer=manufacturer,
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
                        extract_entities=extract_entities,
                        metadata={"manufacturer": manufacturer}
                    )

                    self._stats["urls_scraped"] += 1
                    self._stats["total_content_bytes"] += len(content)
                    self._stats["entities_extracted"] += result.get("entities", 0)
                    self._stats["relations_extracted"] += result.get("relations", 0)
                    self._stats["by_manufacturer"][manufacturer] = \
                        self._stats["by_manufacturer"].get(manufacturer, 0) + 1
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
                        manufacturer=manufacturer,
                        from_cache=False
                    )

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.rate_limit_delay * 2)
                    continue
                return ScrapeResult(
                    url=url,
                    success=False,
                    manufacturer=manufacturer,
                    error="Timeout"
                )
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                return ScrapeResult(
                    url=url,
                    success=False,
                    manufacturer=manufacturer,
                    error=str(e)
                )

        self._stats["urls_failed"] += 1
        return ScrapeResult(
            url=url,
            success=False,
            manufacturer=manufacturer,
            error="Max retries exceeded"
        )

    def _extract_content(self, html: str) -> Tuple[str, str]:
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

    async def build_corpus(
        self,
        seed_urls: bool = True,
        article_urls: bool = True,
        max_urls: Optional[int] = None,
        manufacturer_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build the PLC corpus from configured URLs.

        Args:
            seed_urls: Whether to scrape seed URLs
            article_urls: Whether to scrape specific article URLs
            max_urls: Maximum number of URLs to scrape
            manufacturer_filter: Only scrape URLs for specific manufacturer

        Returns:
            Build statistics
        """
        urls_to_scrape = []

        if seed_urls:
            for url_info in PLC_SEED_URLS:
                if manufacturer_filter and url_info.get("manufacturer") != manufacturer_filter:
                    continue
                urls_to_scrape.append(url_info)

        if article_urls:
            for url_info in PLC_ARTICLE_URLS:
                if manufacturer_filter and url_info.get("manufacturer") != manufacturer_filter:
                    continue
                urls_to_scrape.append(url_info)

        if max_urls:
            urls_to_scrape = urls_to_scrape[:max_urls]

        results = []
        for url_info in urls_to_scrape:
            url = url_info["url"]
            source_type = url_info.get("source_type", "article")
            title = url_info.get("title", "")
            manufacturer = url_info.get("manufacturer", "")

            logger.info(f"Scraping: {url}")
            result = await self.scrape_url(url, source_type, title, manufacturer)
            results.append(result)

            if result.success:
                logger.info(f"  OK {result.word_count} words, {result.entities_extracted} entities [{result.manufacturer}]")
            else:
                logger.warning(f"  FAILED {result.error}")

        return {
            "urls_attempted": len(urls_to_scrape),
            "urls_successful": sum(1 for r in results if r.success),
            "urls_failed": sum(1 for r in results if not r.success),
            "urls_from_cache": sum(1 for r in results if r.from_cache),
            "total_entities": self._stats["entities_extracted"],
            "total_relations": self._stats["relations_extracted"],
            "by_manufacturer": self._stats["by_manufacturer"],
            "corpus_stats": self.corpus.get_stats(),
            "results": [
                {
                    "url": r.url,
                    "success": r.success,
                    "title": r.title,
                    "word_count": r.word_count,
                    "entities": r.entities_extracted,
                    "manufacturer": r.manufacturer,
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
        manufacturer: str = ""
    ) -> Dict[str, Any]:
        """
        Add manually provided content to corpus.

        Useful for adding content from PDFs, internal docs, etc.

        Args:
            content: Text content to add
            source: Source identifier (filename, URL, etc.)
            source_type: Type of source
            title: Optional title
            manufacturer: Manufacturer (allen_bradley, siemens, automationdirect)

        Returns:
            Extraction statistics
        """
        if not manufacturer:
            manufacturer = detect_plc_manufacturer(content) or "multi"

        # Add manufacturer prefix to title for filtering
        full_title = f"[{manufacturer.upper()}] {title}" if title else f"[{manufacturer.upper()}] Manual Content"

        result = await self.builder.add_document(
            content=content,
            source_url=source,
            source_type=source_type,
            title=full_title,
            extract_entities=True
        )
        result["manufacturer"] = manufacturer
        return result

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
        manufacturer_filter: Optional[str] = None,
        include_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Query the PLC corpus.

        Args:
            query: Search query
            top_k: Number of results to return
            manufacturer_filter: Filter by manufacturer
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
            # Filter by manufacturer if specified
            if manufacturer_filter:
                entity_mfr = entity.attributes.get("manufacturer", "")
                if entity_mfr and entity_mfr != manufacturer_filter:
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
            "manufacturer_filter": manufacturer_filter,
            "results": response_entities,
            "count": len(response_entities),
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

_scraper_instance: Optional[PLCCorpusScraper] = None


def get_plc_scraper() -> PLCCorpusScraper:
    """Get singleton PLC corpus scraper instance"""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = PLCCorpusScraper()
    return _scraper_instance


# ============================================
# CLI INTERFACE
# ============================================

async def main():
    """CLI interface for building PLC corpus"""
    import argparse

    parser = argparse.ArgumentParser(description="Build PLC/Automation Corpus")
    parser.add_argument("--url", help="Scrape specific URL")
    parser.add_argument("--build", action="store_true", help="Build corpus from seed URLs")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    parser.add_argument("--query", help="Query the corpus")
    parser.add_argument("--manufacturer", choices=["allen_bradley", "siemens", "automationdirect"],
                        help="Filter by manufacturer")
    parser.add_argument("--max-urls", type=int, help="Maximum URLs to scrape")

    args = parser.parse_args()

    scraper = PLCCorpusScraper()

    if args.url:
        result = await scraper.scrape_url(args.url, manufacturer=args.manufacturer or "")
        print(f"Scraped: {result.url}")
        print(f"  Success: {result.success}")
        print(f"  Title: {result.title}")
        print(f"  Words: {result.word_count}")
        print(f"  Entities: {result.entities_extracted}")
        print(f"  Manufacturer: {result.manufacturer}")
        if result.error:
            print(f"  Error: {result.error}")

    elif args.build:
        print("Building PLC corpus...")
        result = await scraper.build_corpus(
            max_urls=args.max_urls,
            manufacturer_filter=args.manufacturer
        )
        print(f"\nBuild Complete:")
        print(f"  URLs Attempted: {result['urls_attempted']}")
        print(f"  URLs Successful: {result['urls_successful']}")
        print(f"  URLs from Cache: {result['urls_from_cache']}")
        print(f"  Total Entities: {result['total_entities']}")
        print(f"  Total Relations: {result['total_relations']}")
        print(f"  By Manufacturer: {result['by_manufacturer']}")

    elif args.query:
        result = await scraper.query(args.query, manufacturer_filter=args.manufacturer)
        print(f"Query: {args.query}")
        if args.manufacturer:
            print(f"Manufacturer Filter: {args.manufacturer}")
        print(f"Results: {len(result['results'])}")
        for r in result['results']:
            print(f"  - {r['name']} ({r['type']}): {r['similarity']:.3f}")
            if r.get('related'):
                for rel in r['related'][:2]:
                    print(f"      -> {rel['relation']}: {rel['entity']}")

    elif args.stats:
        stats = scraper.get_stats()
        print("PLC Corpus Statistics:")
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
