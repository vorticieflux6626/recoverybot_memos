#!/usr/bin/env python3
"""
IMM Corpus Ingestion Script

Ingests content from the URL sources defined in IMM_URL_SOURCES into the
IMM domain corpus for injection molding machine troubleshooting.

Usage:
    # Dry run - show what would be ingested
    python scripts/ingest_imm_corpus.py --dry-run

    # Ingest priority URLs (critical + high priority)
    python scripts/ingest_imm_corpus.py --priority

    # Ingest specific source
    python scripts/ingest_imm_corpus.py --source euromap_standards

    # Ingest all sources
    python scripts/ingest_imm_corpus.py --all

    # Show corpus stats
    python scripts/ingest_imm_corpus.py --stats
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agentic.imm_corpus_builder import get_imm_builder, IMMCorpusBuilder
from agentic.schemas.imm_schema import IMM_URL_SOURCES, get_priority_urls

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("imm_ingestion")


async def ingest_source(builder: IMMCorpusBuilder, source_name: str, source_config: dict, delay: float = 1.0) -> dict:
    """Ingest all URLs from a single source"""
    results = {
        "source": source_name,
        "total": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0
    }

    urls = source_config.get("urls", [])
    content_type = source_config.get("content_type", "web")

    logger.info(f"Ingesting {len(urls)} URLs from {source_name} ({content_type})")

    for url in urls:
        results["total"] += 1
        try:
            result = await builder.ingest_url(
                url=url,
                source_type=content_type
            )

            status = result.get("status", "unknown")
            if status == "indexed":
                results["success"] += 1
                logger.info(f"  ✓ Indexed: {url[:60]}... ({result.get('entities', 0)} entities)")
            elif status == "skipped":
                results["skipped"] += 1
                logger.info(f"  - Skipped: {url[:60]}... ({result.get('reason', '')})")
            else:
                results["failed"] += 1
                logger.warning(f"  ✗ Failed: {url[:60]}... ({result.get('error', '')})")

        except Exception as e:
            results["failed"] += 1
            logger.error(f"  ✗ Error: {url[:60]}... ({e})")

        if delay > 0:
            await asyncio.sleep(delay)

    return results


async def ingest_priority_urls(builder: IMMCorpusBuilder, delay: float = 1.0) -> dict:
    """Ingest only high-priority URLs"""
    priority_urls = get_priority_urls()
    results = {
        "total": len(priority_urls),
        "success": 0,
        "failed": 0,
        "skipped": 0
    }

    logger.info(f"Ingesting {len(priority_urls)} priority URLs")

    for url in priority_urls:
        try:
            result = await builder.ingest_url(url=url)

            status = result.get("status", "unknown")
            if status == "indexed":
                results["success"] += 1
                logger.info(f"  ✓ Indexed: {url[:60]}...")
            elif status == "skipped":
                results["skipped"] += 1
            else:
                results["failed"] += 1
                logger.warning(f"  ✗ Failed: {url[:60]}...")

        except Exception as e:
            results["failed"] += 1
            logger.error(f"  ✗ Error: {url[:60]}... ({e})")

        if delay > 0:
            await asyncio.sleep(delay)

    return results


async def ingest_all_sources(builder: IMMCorpusBuilder, delay: float = 1.0) -> dict:
    """Ingest all sources"""
    all_results = []

    for source_name, source_config in IMM_URL_SOURCES.items():
        results = await ingest_source(builder, source_name, source_config, delay)
        all_results.append(results)

    # Aggregate
    total = sum(r["total"] for r in all_results)
    success = sum(r["success"] for r in all_results)
    failed = sum(r["failed"] for r in all_results)
    skipped = sum(r["skipped"] for r in all_results)

    return {
        "sources": len(all_results),
        "total": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "details": all_results
    }


def show_dry_run():
    """Show what would be ingested without actually doing it"""
    print("\n=== IMM Corpus Ingestion - Dry Run ===\n")

    total_urls = 0
    for source_name, source_config in IMM_URL_SOURCES.items():
        priority = source_config.get("priority", "medium")
        content_type = source_config.get("content_type", "web")
        urls = source_config.get("urls", [])

        print(f"\n[{source_name}] ({priority} priority, {content_type})")
        for url in urls:
            print(f"  - {url}")
            total_urls += 1

    print(f"\n\nTotal URLs to ingest: {total_urls}")
    print("\nPriority URLs (critical + high):")
    priority_urls = get_priority_urls()
    for url in priority_urls:
        print(f"  - {url}")
    print(f"\nPriority URL count: {len(priority_urls)}")


def show_stats(builder: IMMCorpusBuilder):
    """Show corpus statistics"""
    stats = builder.get_stats()

    print("\n=== IMM Corpus Statistics ===\n")

    print("Builder Stats:")
    for key, value in stats["builder_stats"].items():
        print(f"  {key}: {value}")

    print("\nCorpus Stats:")
    for key, value in stats["corpus_stats"].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


async def main():
    parser = argparse.ArgumentParser(description="IMM Corpus Ingestion Script")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested")
    parser.add_argument("--priority", action="store_true", help="Ingest only priority URLs")
    parser.add_argument("--source", type=str, help="Ingest specific source by name")
    parser.add_argument("--all", action="store_true", help="Ingest all sources")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama URL")

    args = parser.parse_args()

    if args.dry_run:
        show_dry_run()
        return

    # Initialize builder
    builder = get_imm_builder(ollama_url=args.ollama_url)

    if args.stats:
        show_stats(builder)
        return

    if args.priority:
        results = await ingest_priority_urls(builder, delay=args.delay)
        print(f"\n=== Priority Ingestion Complete ===")
        print(f"Total: {results['total']}, Success: {results['success']}, Failed: {results['failed']}, Skipped: {results['skipped']}")
        return

    if args.source:
        if args.source not in IMM_URL_SOURCES:
            print(f"Error: Unknown source '{args.source}'")
            print(f"Available sources: {', '.join(IMM_URL_SOURCES.keys())}")
            return
        results = await ingest_source(builder, args.source, IMM_URL_SOURCES[args.source], delay=args.delay)
        print(f"\n=== Source Ingestion Complete: {args.source} ===")
        print(f"Total: {results['total']}, Success: {results['success']}, Failed: {results['failed']}, Skipped: {results['skipped']}")
        return

    if args.all:
        results = await ingest_all_sources(builder, delay=args.delay)
        print(f"\n=== Full Ingestion Complete ===")
        print(f"Sources: {results['sources']}, Total URLs: {results['total']}")
        print(f"Success: {results['success']}, Failed: {results['failed']}, Skipped: {results['skipped']}")
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
