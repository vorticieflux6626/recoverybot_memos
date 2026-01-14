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

Refactored to use BaseCorpusScraper (Phase 3 consolidation).

Author: Claude Code
Date: December 2025 (Refactored January 2026)
"""

import asyncio
import logging
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

class RJGCorpusScraper(BaseCorpusScraper):
    """
    Scraper for building RJG scientific molding knowledge corpus.

    Features:
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
        Initialize the RJG corpus scraper.

        Args:
            db_path: Path to SQLite database for corpus storage
            ollama_url: URL for Ollama API (entity extraction)
            rate_limit_delay: Seconds between requests (fallback)
            max_retries: Max retry attempts per URL
            extraction_model: Model for entity extraction (defaults to config)
        """
        # Initialize base class
        super().__init__(
            domain_id="rjg_scientific_molding",
            db_path=db_path,
            ollama_url=ollama_url,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
            extraction_model=extraction_model,
            use_rate_limiter=True,
            use_redis_cache=True
        )

    # ============================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================

    def create_schema(self) -> DomainSchema:
        """Create DomainSchema for RJG scientific molding corpus."""
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

    def get_seed_urls(self) -> List[Dict[str, str]]:
        """Return RJG seed URLs."""
        return RJG_SEED_URLS

    def get_article_urls(self) -> List[Dict[str, str]]:
        """Return RJG article URLs."""
        return RJG_ARTICLE_URLS

    def get_user_agent(self) -> str:
        """Return User-Agent for RJG scraper."""
        return UserAgents.RJG_CORPUS_BUILDER

    def get_extraction_model(self) -> str:
        """Get extraction model from LLM config (IMM extractor for RJG)."""
        llm_config = get_llm_config()
        return llm_config.corpus.imm_extractor.model


# ============================================
# FACTORY FUNCTION
# ============================================

_scraper_instance: Optional[RJGCorpusScraper] = None


def get_rjg_scraper() -> RJGCorpusScraper:
    """Get singleton RJG corpus scraper instance."""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = RJGCorpusScraper()
    return _scraper_instance


# ============================================
# CLI INTERFACE
# ============================================

async def main():
    """CLI interface for building RJG corpus."""
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
        print(f"  Duration: {result.duration_ms:.0f}ms")
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
        print(f"  Scraper Metrics:")
        metrics = stats['scraper_metrics']
        print(f"    URLs Scraped: {metrics['urls_scraped']}")
        print(f"    Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
        print(f"    Success Rate: {metrics['success_rate']:.1%}")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
