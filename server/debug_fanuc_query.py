#!/usr/bin/env python3
"""
Debug script to trace FANUC query processing through the pipeline.
Identifies where dictionary URLs and other irrelevant results come from.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentic.analyzer import QueryAnalyzer
from agentic.searcher import SearcherAgent, SearXNGSearchProvider
from agentic.fanuc_corpus_builder import FANUCCorpusBuilder

FANUC_QUERY = "SRVO-063 RCAL alarm axis 2 after encoder replacement"

async def main():
    print("=" * 70)
    print("FANUC Query Debug Trace")
    print("=" * 70)
    print(f"\nOriginal Query: {FANUC_QUERY}")

    # Step 1: Query type detection
    print("\n" + "=" * 70)
    print("STEP 1: Query Type Detection")
    print("=" * 70)
    searxng_provider = SearXNGSearchProvider()
    query_type = searxng_provider.detect_query_type(FANUC_QUERY)
    print(f"Detected query type: {query_type}")
    print(f"Engines for this query: {searxng_provider.get_engines_for_query(FANUC_QUERY)}")

    # Step 2: Pattern extraction
    print("\n" + "=" * 70)
    print("STEP 2: FANUC Pattern Extraction")
    print("=" * 70)
    builder = FANUCCorpusBuilder()
    patterns = builder.extract_fanuc_patterns(FANUC_QUERY)
    print(f"Extracted patterns: {patterns}")

    # Step 3: Analyzer decomposition
    print("\n" + "=" * 70)
    print("STEP 3: Query Analyzer Decomposition")
    print("=" * 70)
    analyzer = QueryAnalyzer()

    plan = None
    try:
        analysis = await analyzer.analyze(FANUC_QUERY)
        print(f"Query Type: {analysis.query_type}")
        print(f"Key Topics: {analysis.key_topics}")
        print(f"Suggested Queries: {analysis.suggested_queries}")
        print(f"Complexity: {analysis.estimated_complexity}")

        # Create search plan
        plan = await analyzer.create_search_plan(FANUC_QUERY, analysis)
        print(f"\nDecomposed Questions: {plan.decomposed_questions}")
        print(f"Search Phases: {plan.search_phases}")
    except Exception as e:
        print(f"Analyzer error: {e}")
        import traceback
        traceback.print_exc()

    # Step 4: Raw search results
    print("\n" + "=" * 70)
    print("STEP 4: Raw Search Results (before filtering)")
    print("=" * 70)

    searcher = SearcherAgent()

    # Search with just the original query first
    print(f"\nSearching for: {FANUC_QUERY}")
    results = await searcher.search([FANUC_QUERY], max_results_per_query=15)

    print(f"\nTotal results: {len(results)}")
    print("\nAll results:")
    for i, r in enumerate(results):
        relevance = f"score={r.relevance_score:.2f}" if hasattr(r, 'relevance_score') else "no_score"
        print(f"  {i+1}. [{r.source_domain}] {r.title[:50]}...")
        print(f"      URL: {r.url}")
        print(f"      {relevance}")

    # Check for dictionary URLs
    print("\n" + "=" * 70)
    print("STEP 5: Checking for dictionary/off-topic URLs")
    print("=" * 70)

    off_topic_patterns = [
        "dictionary", "thesaurus", "definitions", "merriam-webster",
        "collinsdictionary", "cambridge", "oxford", "yourdictionary"
    ]

    off_topic_results = []
    for r in results:
        url_lower = r.url.lower()
        if any(pattern in url_lower for pattern in off_topic_patterns):
            off_topic_results.append(r)

    if off_topic_results:
        print(f"\nFound {len(off_topic_results)} dictionary/off-topic URLs:")
        for r in off_topic_results:
            print(f"  - {r.url}")
            print(f"    Title: {r.title}")
    else:
        print("\nNo dictionary URLs found in results.")

    # Step 6: Test with decomposed queries
    if 'plan' in dir() and plan.decomposed_questions:
        print("\n" + "=" * 70)
        print("STEP 6: Search with Decomposed Queries")
        print("=" * 70)

        for q in plan.decomposed_questions[:3]:
            print(f"\n--- Query: {q} ---")
            q_results = await searcher.search([q], max_results_per_query=5)
            print(f"Results: {len(q_results)}")
            for r in q_results[:3]:
                print(f"  - [{r.source_domain}] {r.title[:50]}...")
                print(f"    URL: {r.url}")

    print("\n" + "=" * 70)
    print("Debug trace complete")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
