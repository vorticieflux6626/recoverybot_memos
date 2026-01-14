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

Refactored to use BaseCorpusScraper (Phase 3 consolidation).

Author: Claude Code
Date: December 2025 (Refactored January 2026)
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_corpus_scraper import BaseCorpusScraper, ScrapeResult
from .llm_config import get_llm_config
from .user_agent_config import UserAgents
from .domain_corpus import (
    DomainSchema,
    DomainEntityDef,
    DomainRelationDef,
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

class PLCCorpusScraper(BaseCorpusScraper):
    """
    Scraper for building PLC/Automation knowledge corpus.

    Features:
    - Manufacturer-aware indexing (Allen-Bradley, Siemens, AutomationDirect)
    - Rate-limited scraping with respectful delays
    - Content caching via Redis with fallback
    - Entity extraction via LLM
    - Integration with DomainCorpus

    Inherits from BaseCorpusScraper for:
    - Unified rate limiting (aiometer)
    - Redis caching with circuit breaker
    - Standardized metrics and monitoring
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
            rate_limit_delay: Seconds between requests (fallback)
            max_retries: Max retry attempts per URL
            extraction_model: Model for entity extraction (defaults to config)
        """
        # Initialize base class
        super().__init__(
            domain_id="plc_automation",
            db_path=db_path,
            ollama_url=ollama_url,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
            extraction_model=extraction_model,
            use_rate_limiter=True,
            use_redis_cache=True
        )

        # PLC-specific tracking
        self._manufacturer_stats = {
            "allen_bradley": 0,
            "siemens": 0,
            "automationdirect": 0,
            "multi": 0,
        }

    # ============================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================

    def create_schema(self) -> DomainSchema:
        """Create DomainSchema for PLC/Automation corpus."""
        # Get fault patterns
        all_fault_patterns = [p for p, _ in AB_FAULT_PATTERNS + SIEMENS_FAULT_PATTERNS]
        all_module_patterns = [p for p, _ in AB_MODULE_PATTERNS + SIEMENS_MODULE_PATTERNS]
        all_protocol_patterns = [p for p, _ in PROTOCOL_PATTERNS]

        entity_types = [
            DomainEntityDef(
                entity_type="fault_code",
                description="PLC fault and alarm codes",
                extraction_patterns=all_fault_patterns[:10],
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

    def get_seed_urls(self) -> List[Dict[str, str]]:
        """Return PLC seed URLs."""
        return PLC_SEED_URLS

    def get_article_urls(self) -> List[Dict[str, str]]:
        """Return PLC article URLs."""
        return PLC_ARTICLE_URLS

    def get_user_agent(self) -> str:
        """Return User-Agent for PLC scraper."""
        return UserAgents.PLC_CORPUS_BUILDER

    def get_extraction_model(self) -> str:
        """Get extraction model from LLM config."""
        llm_config = get_llm_config()
        return llm_config.corpus.plc_extractor.model

    # ============================================
    # PLC-SPECIFIC OVERRIDES
    # ============================================

    def filter_url(self, url: str, url_info: Dict[str, str]) -> bool:
        """Filter URLs based on manufacturer if specified."""
        # Could add manufacturer filtering here if needed
        return True

    def extract_metadata(self, url: str, html: str) -> Dict[str, Any]:
        """Extract PLC-specific metadata including manufacturer."""
        metadata = {}

        # Auto-detect manufacturer from URL and content
        manufacturer = detect_plc_manufacturer(url) or detect_plc_manufacturer(html[:5000])
        if manufacturer:
            metadata["manufacturer"] = manufacturer

        return metadata

    def transform_content(self, content: str, url: str) -> str:
        """Transform content - add manufacturer prefix if detectable."""
        return content

    # ============================================
    # PLC-SPECIFIC METHODS
    # ============================================

    async def scrape_url(
        self,
        url: str,
        source_type: str = "article",
        title: str = "",
        manufacturer: str = "",
        extract_entities: bool = True
    ) -> ScrapeResult:
        """
        Scrape a URL with manufacturer awareness.

        Args:
            url: URL to scrape
            source_type: Type of source
            title: Optional title override
            manufacturer: Manufacturer (allen_bradley, siemens, automationdirect)
            extract_entities: Whether to run entity extraction

        Returns:
            ScrapeResult with extraction statistics
        """
        # Auto-detect manufacturer if not provided
        if not manufacturer:
            manufacturer = detect_plc_manufacturer(url) or "multi"

        # Build metadata with manufacturer
        metadata = {"manufacturer": manufacturer}

        # Call parent scrape_url
        result = await super().scrape_url(
            url=url,
            source_type=source_type,
            title=title,
            extract_entities=extract_entities,
            metadata=metadata
        )

        # Track manufacturer stats
        if result.success:
            self._manufacturer_stats[manufacturer] = \
                self._manufacturer_stats.get(manufacturer, 0) + 1

        # Add manufacturer to result metadata
        result.metadata["manufacturer"] = manufacturer

        return result

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
        # Create filter function for manufacturer
        def mfr_filter(url_info: Dict[str, str]) -> bool:
            if manufacturer_filter:
                return url_info.get("manufacturer") == manufacturer_filter
            return True

        # Call parent build_corpus with filter
        result = await super().build_corpus(
            seed_urls=seed_urls,
            article_urls=article_urls,
            max_urls=max_urls,
            url_filter=mfr_filter
        )

        # Add manufacturer-specific stats
        result["by_manufacturer"] = self._manufacturer_stats.copy()

        return result

    async def query(
        self,
        query: str,
        top_k: int = 5,
        manufacturer_filter: Optional[str] = None,
        include_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Query the PLC corpus with manufacturer filtering.

        Args:
            query: Search query
            top_k: Number of results to return
            manufacturer_filter: Filter by manufacturer
            include_relations: Whether to include related entities

        Returns:
            Query results with entities and context
        """
        filters = {}
        if manufacturer_filter:
            filters["manufacturer"] = manufacturer_filter

        result = await super().query(
            query=query,
            top_k=top_k,
            include_relations=include_relations,
            filters=filters if filters else None
        )

        result["manufacturer_filter"] = manufacturer_filter
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get scraper stats including manufacturer breakdown."""
        stats = super().get_stats()
        stats["by_manufacturer"] = self._manufacturer_stats.copy()
        return stats


# ============================================
# FACTORY FUNCTION
# ============================================

_scraper_instance: Optional[PLCCorpusScraper] = None


def get_plc_scraper() -> PLCCorpusScraper:
    """Get singleton PLC corpus scraper instance."""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = PLCCorpusScraper()
    return _scraper_instance


# ============================================
# CLI INTERFACE
# ============================================

async def main():
    """CLI interface for building PLC corpus."""
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
        print(f"  Manufacturer: {result.metadata.get('manufacturer', 'unknown')}")
        print(f"  Duration: {result.duration_ms:.0f}ms")
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
        print(f"  By Manufacturer: {stats['by_manufacturer']}")
        print(f"  Scraper Metrics:")
        metrics = stats['scraper_metrics']
        print(f"    URLs Scraped: {metrics['urls_scraped']}")
        print(f"    Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
        print(f"    Success Rate: {metrics['success_rate']:.1%}")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
