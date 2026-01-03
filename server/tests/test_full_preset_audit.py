#!/usr/bin/env python3
"""
Full Preset Audit - RESEARCH and FULL preset testing with observability analysis.

Runs comprehensive tests and captures observability data for effectiveness audit.

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python tests/test_full_preset_audit.py research
    python tests/test_full_preset_audit.py full

Created: 2026-01-02
"""

import asyncio
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("preset_audit")

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, '/home/sparkone/sdd/Recovery_Bot/memOS/server')

# Industrial troubleshooting queries for comprehensive testing
AUDIT_QUERIES = [
    {
        "id": "fanuc-collision",
        "query": "FANUC R-30iB SRVO-023 collision detection alarm troubleshooting steps",
        "expected_terms": ["srvo-023", "collision", "detection", "fanuc"],
        "complexity": "expert"
    },
    {
        "id": "allen-bradley-fault",
        "query": "Allen-Bradley 1756-L71 ControlLogix major fault code 16#0001 diagnosis",
        "expected_terms": ["1756-l71", "controllogix", "fault", "allen-bradley"],
        "complexity": "expert"
    },
]


async def run_audit_test(preset_name: str, query_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single audit test with full observability capture."""
    from agentic import UniversalOrchestrator
    from agentic.models import SearchRequest
    from agentic.orchestrator_universal import OrchestratorPreset

    preset = OrchestratorPreset(preset_name)
    request_id = f"audit-{preset_name}-{query_info['id']}-{int(time.time())}"

    logger.info(f"\n{'='*70}")
    logger.info(f"AUDIT: {query_info['id']} with {preset_name.upper()} preset")
    logger.info(f"Query: {query_info['query'][:80]}...")
    logger.info(f"Request ID: {request_id}")
    logger.info(f"{'='*70}")

    result = {
        "request_id": request_id,
        "preset": preset_name,
        "query_id": query_info["id"],
        "query": query_info["query"],
        "start_time": datetime.now().isoformat(),
        "success": False,
        "duration_s": 0,
        "synthesis": "",
        "confidence": 0,
        "source_count": 0,
        "citations": 0,
        "on_topic_terms": [],
        "term_coverage": 0,
        "observability": {},
        "errors": [],
        "warnings": []
    }

    try:
        orchestrator = UniversalOrchestrator(preset=preset)

        # Log enabled features
        from dataclasses import fields
        enabled_features = [f.name for f in fields(orchestrator.config)
                          if getattr(orchestrator.config, f.name) is True]
        logger.info(f"Enabled features ({len(enabled_features)}): {enabled_features[:10]}...")

        request = SearchRequest(
            query=query_info["query"],
            max_results=10,
            enable_scraping=True,
            enable_thinking_model=query_info["complexity"] == "expert",
            request_id=request_id
        )

        start = time.time()
        search_result = await orchestrator.search(request)
        duration = time.time() - start

        # Debug: Show what type was returned
        logger.info(f"DEBUG: search_result type = {type(search_result).__name__}")
        logger.info(f"DEBUG: search_result attrs = {dir(search_result)[:15]}...")
        if hasattr(search_result, 'data'):
            logger.info(f"DEBUG: search_result.data type = {type(search_result.data).__name__}")
            if search_result.data is not None:
                logger.info(f"DEBUG: search_result.data attrs = {dir(search_result.data)[:15]}...")
                if hasattr(search_result.data, 'synthesized_context'):
                    synth_len = len(search_result.data.synthesized_context) if search_result.data.synthesized_context else 0
                    logger.info(f"DEBUG: synthesized_context length = {synth_len}")

        # Extract synthesis - SearchResponse has data.synthesized_context
        synthesis = ""
        if hasattr(search_result, 'data') and search_result.data is not None:
            if hasattr(search_result.data, 'synthesized_context'):
                synthesis = search_result.data.synthesized_context or ""
        elif hasattr(search_result, 'synthesized_context'):
            synthesis = search_result.synthesized_context or ""

        # Extract confidence - SearchResponse has data.confidence_score
        confidence = 0
        if hasattr(search_result, 'data') and search_result.data is not None:
            if hasattr(search_result.data, 'confidence_score'):
                confidence = search_result.data.confidence_score or 0
        elif hasattr(search_result, 'confidence_score'):
            confidence = search_result.confidence_score or 0

        # Extract sources - SearchResponse has data.sources
        sources = []
        if hasattr(search_result, 'data') and search_result.data is not None:
            if hasattr(search_result.data, 'sources'):
                sources = search_result.data.sources or []
        elif hasattr(search_result, 'sources'):
            sources = search_result.sources or []

        # Check on-topic terms
        synthesis_lower = synthesis.lower()
        expected_terms = query_info.get("expected_terms", [])
        found_terms = [t for t in expected_terms if t in synthesis_lower]
        term_coverage = len(found_terms) / len(expected_terms) if expected_terms else 0

        # Count citations - match both [Source X] and (Source X) formats
        import re
        citations = re.findall(r'[\[\(]Source \d+[\]\)]|[\[\(]Domain Knowledge[\]\)]', synthesis)

        result.update({
            "success": True,
            "duration_s": round(duration, 1),
            "synthesis": synthesis,
            "synthesis_length": len(synthesis),
            "confidence": round(confidence * 100) if confidence else 0,
            "source_count": len(sources),
            "citations": len(citations),
            "on_topic_terms": found_terms,
            "term_coverage": round(term_coverage * 100),
            "end_time": datetime.now().isoformat()
        })

        # Try to get observability data
        try:
            from agentic.observability import get_observability_dashboard
            dashboard = get_observability_dashboard()
            if dashboard:
                obs_data = await dashboard.get_request_summary(request_id)
                result["observability"] = obs_data or {}
        except Exception as e:
            logger.warning(f"Could not get observability data: {e}")

        # Try to get performance metrics
        try:
            from agentic.performance_metrics import get_performance_metrics
            metrics = get_performance_metrics()
            if metrics:
                result["performance_metrics"] = {
                    "context_utilization": metrics.get_context_utilization(request_id),
                    "tool_latencies": metrics.get_tool_latencies(request_id)
                }
        except Exception as e:
            logger.warning(f"Could not get performance metrics: {e}")

        # Log results
        logger.info(f"\n--- RESULTS ---")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Confidence: {result['confidence']}%")
        logger.info(f"Sources: {result['source_count']}")
        logger.info(f"Citations: {result['citations']}")
        logger.info(f"Term coverage: {result['term_coverage']}% ({found_terms})")
        logger.info(f"Synthesis length: {result['synthesis_length']} chars")

        # Quality assessment
        quality_issues = []
        if result['confidence'] < 60:
            quality_issues.append(f"LOW CONFIDENCE: {result['confidence']}%")
        if result['term_coverage'] < 50:
            quality_issues.append(f"LOW TERM COVERAGE: {result['term_coverage']}%")
        if result['citations'] < 3:
            quality_issues.append(f"FEW CITATIONS: {result['citations']}")
        if duration > 600:
            quality_issues.append(f"VERY SLOW: {duration:.0f}s")

        if quality_issues:
            logger.warning(f"Quality issues: {quality_issues}")
            result["warnings"] = quality_issues
        else:
            logger.info("✅ Quality checks passed")

        # Show synthesis preview
        logger.info(f"\n--- SYNTHESIS PREVIEW ---")
        preview = synthesis[:1000] + "..." if len(synthesis) > 1000 else synthesis
        print(preview)

    except Exception as e:
        result["errors"].append(str(e))
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    return result


async def run_preset_audit(preset_name: str):
    """Run full audit for a preset."""
    logger.info(f"\n{'#'*70}")
    logger.info(f"# STARTING {preset_name.upper()} PRESET AUDIT")
    logger.info(f"# Time: {datetime.now().isoformat()}")
    logger.info(f"# Queries: {len(AUDIT_QUERIES)}")
    logger.info(f"{'#'*70}\n")

    results = []

    for i, query_info in enumerate(AUDIT_QUERIES, 1):
        logger.info(f"\n[QUERY {i}/{len(AUDIT_QUERIES)}]")
        result = await run_audit_test(preset_name, query_info)
        results.append(result)

        # Brief pause between queries
        if i < len(AUDIT_QUERIES):
            logger.info("\nWaiting 5s before next query...")
            await asyncio.sleep(5)

    # Generate summary
    logger.info(f"\n{'='*70}")
    logger.info(f"AUDIT SUMMARY: {preset_name.upper()}")
    logger.info(f"{'='*70}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal queries: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_duration = sum(r["duration_s"] for r in successful) / len(successful)
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_term_coverage = sum(r["term_coverage"] for r in successful) / len(successful)
        avg_citations = sum(r["citations"] for r in successful) / len(successful)

        print(f"\n--- Aggregate Metrics ---")
        print(f"Avg Duration: {avg_duration:.0f}s")
        print(f"Avg Confidence: {avg_confidence:.0f}%")
        print(f"Avg Term Coverage: {avg_term_coverage:.0f}%")
        print(f"Avg Citations: {avg_citations:.1f}")

        print(f"\n--- Per-Query Results ---")
        for r in successful:
            status = "✅" if not r.get("warnings") else "⚠️"
            print(f"  {status} {r['query_id']}: conf={r['confidence']}%, "
                  f"terms={r['term_coverage']}%, cites={r['citations']}, "
                  f"time={r['duration_s']}s")
            if r.get("warnings"):
                for w in r["warnings"]:
                    print(f"      ⚠️ {w}")

    if failed:
        print(f"\n--- Failed Queries ---")
        for r in failed:
            print(f"  ❌ {r['query_id']}: {r.get('errors', ['Unknown error'])}")

    # Save detailed results
    output_file = f"/tmp/preset_audit_{preset_name}_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        # Remove large synthesis from JSON (keep preview)
        for r in results:
            if r.get("synthesis"):
                r["synthesis_preview"] = r["synthesis"][:500]
                del r["synthesis"]
        json.dump({
            "preset": preset_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "avg_duration_s": avg_duration if successful else 0,
                "avg_confidence": avg_confidence if successful else 0,
                "avg_term_coverage": avg_term_coverage if successful else 0
            },
            "results": results
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

    return results


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_full_preset_audit.py <preset>")
        print("Available: research, full")
        sys.exit(1)

    preset = sys.argv[1].lower()
    if preset not in ["research", "full"]:
        print(f"Invalid preset: {preset}")
        print("Available: research, full")
        sys.exit(1)

    await run_preset_audit(preset)


if __name__ == "__main__":
    asyncio.run(main())
