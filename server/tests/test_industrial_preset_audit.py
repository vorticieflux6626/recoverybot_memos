#!/usr/bin/env python3
"""
Industrial Systems Preset Audit - Complex integrated troubleshooting queries.

Tests RESEARCH and FULL presets with real-world industrial automation scenarios
involving FANUC robotics, Allen-Bradley PLCs, Siemens controllers, servo systems,
and integrated manufacturing systems.

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python tests/test_industrial_preset_audit.py research
    python tests/test_industrial_preset_audit.py full
    python tests/test_industrial_preset_audit.py both

Created: 2026-01-02
"""

import asyncio
import sys
import time
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("industrial_audit")

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, '/home/sparkone/sdd/Recovery_Bot/memOS/server')

# Complex industrial troubleshooting queries
INDUSTRIAL_QUERIES = [
    # FANUC Robotics - Servo/Motion Errors
    {
        "id": "fanuc-srvo-023",
        "query": "FANUC R-30iB controller SRVO-023 chain 1 abnormal collision detection troubleshooting procedure",
        "expected_terms": ["srvo-023", "collision", "chain", "fanuc", "r-30ib"],
        "complexity": "expert",
        "category": "fanuc_servo"
    },
    {
        "id": "fanuc-motn-023",
        "query": "FANUC robot MOTN-023 position deviation error during arc welding operation diagnosis",
        "expected_terms": ["motn-023", "position", "deviation", "welding", "fanuc"],
        "complexity": "expert",
        "category": "fanuc_motion"
    },

    # Allen-Bradley ControlLogix - Fault Codes
    {
        "id": "ab-major-fault",
        "query": "Allen-Bradley 1756-L71 ControlLogix major recoverable fault type 4 code 16#0042 motion group fault troubleshooting",
        "expected_terms": ["1756-l71", "controllogix", "fault", "motion", "allen-bradley"],
        "complexity": "expert",
        "category": "allen_bradley"
    },
    {
        "id": "ab-ethernet-io",
        "query": "Allen-Bradley 1756-EN2T EtherNet/IP module connection timeout to remote I/O rack diagnosis",
        "expected_terms": ["1756-en2t", "ethernet", "timeout", "connection", "allen-bradley"],
        "complexity": "expert",
        "category": "allen_bradley"
    },

    # Siemens S7 - Communication/Network
    {
        "id": "siemens-profinet",
        "query": "Siemens S7-1500 PROFINET IO device not accessible diagnostic buffer error analysis",
        "expected_terms": ["s7-1500", "profinet", "diagnostic", "siemens"],
        "complexity": "expert",
        "category": "siemens"
    },

    # Integrated Systems - Multi-vendor
    {
        "id": "fanuc-plc-integration",
        "query": "FANUC robot integration with Allen-Bradley PLC via EtherNet/IP adapter communication setup and handshaking",
        "expected_terms": ["fanuc", "allen-bradley", "ethernet/ip", "integration", "communication"],
        "complexity": "expert",
        "category": "integration"
    },

    # Servo Drive Systems
    {
        "id": "servo-overcurrent",
        "query": "Yaskawa Sigma-7 servo drive A.710 overcurrent alarm during acceleration troubleshooting steps",
        "expected_terms": ["yaskawa", "sigma-7", "overcurrent", "acceleration", "a.710"],
        "complexity": "expert",
        "category": "servo_drives"
    },

    # Vision Systems
    {
        "id": "fanuc-irobot-vision",
        "query": "FANUC iRVision 2D camera calibration offset error in pick and place application",
        "expected_terms": ["irvision", "calibration", "offset", "camera", "fanuc"],
        "complexity": "expert",
        "category": "vision"
    },
]


async def run_single_query(preset_name: str, query_info: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single audit query with full observability."""
    from agentic import UniversalOrchestrator
    from agentic.models import SearchRequest
    from agentic.orchestrator_universal import OrchestratorPreset

    preset = OrchestratorPreset(preset_name)
    request_id = f"audit-{preset_name}-{query_info['id']}-{int(time.time())}"

    logger.info(f"\n{'='*70}")
    logger.info(f"QUERY: {query_info['id']} | PRESET: {preset_name.upper()}")
    logger.info(f"Category: {query_info['category']}")
    logger.info(f"Query: {query_info['query'][:80]}...")
    logger.info(f"{'='*70}")

    result = {
        "request_id": request_id,
        "preset": preset_name,
        "query_id": query_info["id"],
        "query": query_info["query"],
        "category": query_info["category"],
        "start_time": datetime.now().isoformat(),
        "success": False,
        "duration_s": 0,
        "synthesis": "",
        "synthesis_length": 0,
        "confidence": 0,
        "source_count": 0,
        "citations": 0,
        "citation_list": [],
        "on_topic_terms": [],
        "term_coverage": 0,
        "quality_score": 0,
        "errors": [],
        "warnings": []
    }

    try:
        orchestrator = UniversalOrchestrator(preset=preset)

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

        # Extract synthesis
        synthesis = ""
        if hasattr(search_result, 'data') and search_result.data is not None:
            if hasattr(search_result.data, 'synthesized_context'):
                synthesis = search_result.data.synthesized_context or ""
        elif hasattr(search_result, 'synthesized_context'):
            synthesis = search_result.synthesized_context or ""

        # Extract confidence
        confidence = 0
        if hasattr(search_result, 'data') and search_result.data is not None:
            if hasattr(search_result.data, 'confidence_score'):
                confidence = search_result.data.confidence_score or 0
        elif hasattr(search_result, 'confidence_score'):
            confidence = search_result.confidence_score or 0

        # Extract sources
        sources = []
        if hasattr(search_result, 'data') and search_result.data is not None:
            if hasattr(search_result.data, 'sources'):
                sources = search_result.data.sources or []
        elif hasattr(search_result, 'sources'):
            sources = search_result.sources or []

        # Count citations - match [Source X] and (Source X) formats
        citations = re.findall(r'\[Source \d+\]|\(Source \d+\)|\[Domain Knowledge\]|\(Domain Knowledge\)', synthesis)
        unique_citations = list(set(citations))

        # Check term coverage
        synthesis_lower = synthesis.lower()
        expected_terms = query_info.get("expected_terms", [])
        found_terms = [t for t in expected_terms if t in synthesis_lower]
        term_coverage = len(found_terms) / len(expected_terms) if expected_terms else 0

        # Calculate quality score (weighted)
        # 40% confidence, 30% term coverage, 20% citation density, 10% length
        citation_density = min(1.0, len(unique_citations) / 5)  # Target 5+ unique citations
        length_score = min(1.0, len(synthesis) / 2000)  # Target 2000+ chars
        quality_score = (
            0.40 * (confidence if confidence <= 1 else confidence / 100) +
            0.30 * term_coverage +
            0.20 * citation_density +
            0.10 * length_score
        )

        result.update({
            "success": True,
            "duration_s": round(duration, 1),
            "synthesis": synthesis,
            "synthesis_length": len(synthesis),
            "confidence": round(confidence * 100) if confidence <= 1 else round(confidence),
            "source_count": len(sources),
            "citations": len(citations),
            "citation_list": unique_citations,
            "on_topic_terms": found_terms,
            "term_coverage": round(term_coverage * 100),
            "quality_score": round(quality_score * 100),
            "end_time": datetime.now().isoformat()
        })

        # Quality assessment
        quality_issues = []
        if result['confidence'] < 60:
            quality_issues.append(f"LOW CONFIDENCE: {result['confidence']}%")
        if result['term_coverage'] < 50:
            quality_issues.append(f"LOW TERM COVERAGE: {result['term_coverage']}%")
        if len(unique_citations) < 3:
            quality_issues.append(f"FEW CITATIONS: {len(unique_citations)}")
        if len(synthesis) < 500:
            quality_issues.append(f"SHORT RESPONSE: {len(synthesis)} chars")
        if duration > 300:
            quality_issues.append(f"SLOW: {duration:.0f}s")

        if quality_issues:
            result["warnings"] = quality_issues

        # Log results
        logger.info(f"\n--- RESULTS ---")
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Confidence: {result['confidence']}%")
        logger.info(f"Quality Score: {result['quality_score']}%")
        logger.info(f"Sources: {result['source_count']}")
        logger.info(f"Citations: {len(citations)} ({len(unique_citations)} unique)")
        logger.info(f"Term coverage: {result['term_coverage']}% ({found_terms})")
        logger.info(f"Synthesis length: {result['synthesis_length']} chars")

        if quality_issues:
            logger.warning(f"Quality issues: {quality_issues}")
        else:
            logger.info("Quality checks PASSED")

        # Show synthesis preview
        logger.info(f"\n--- SYNTHESIS PREVIEW (first 1200 chars) ---")
        preview = synthesis[:1200] + "..." if len(synthesis) > 1200 else synthesis
        print(preview)

    except Exception as e:
        result["errors"].append(str(e))
        logger.error(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    return result


async def run_preset_audit(preset_name: str, queries: List[Dict] = None):
    """Run full audit for a preset."""
    queries = queries or INDUSTRIAL_QUERIES

    logger.info(f"\n{'#'*70}")
    logger.info(f"# INDUSTRIAL SYSTEMS AUDIT: {preset_name.upper()} PRESET")
    logger.info(f"# Time: {datetime.now().isoformat()}")
    logger.info(f"# Queries: {len(queries)}")
    logger.info(f"{'#'*70}\n")

    results = []

    for i, query_info in enumerate(queries, 1):
        logger.info(f"\n[QUERY {i}/{len(queries)}]")
        result = await run_single_query(preset_name, query_info)
        results.append(result)

        # Brief pause between queries to avoid overwhelming the system
        if i < len(queries):
            logger.info("\nWaiting 10s before next query...")
            await asyncio.sleep(10)

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
        avg_quality = sum(r["quality_score"] for r in successful) / len(successful)

        print(f"\n--- Aggregate Metrics ---")
        print(f"Avg Duration: {avg_duration:.0f}s")
        print(f"Avg Confidence: {avg_confidence:.0f}%")
        print(f"Avg Term Coverage: {avg_term_coverage:.0f}%")
        print(f"Avg Citations: {avg_citations:.1f}")
        print(f"Avg Quality Score: {avg_quality:.0f}%")

        # Per-category breakdown
        categories = {}
        for r in successful:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        print(f"\n--- Per-Category Results ---")
        for cat, cat_results in categories.items():
            cat_avg_quality = sum(r["quality_score"] for r in cat_results) / len(cat_results)
            cat_avg_conf = sum(r["confidence"] for r in cat_results) / len(cat_results)
            print(f"  {cat}: quality={cat_avg_quality:.0f}%, conf={cat_avg_conf:.0f}% ({len(cat_results)} queries)")

        print(f"\n--- Per-Query Results ---")
        for r in successful:
            status = "PASS" if not r.get("warnings") else "WARN"
            emoji = "✅" if status == "PASS" else "⚠️"
            print(f"  {emoji} {r['query_id']}: quality={r['quality_score']}%, "
                  f"conf={r['confidence']}%, terms={r['term_coverage']}%, "
                  f"cites={r['citations']}, time={r['duration_s']}s")
            if r.get("warnings"):
                for w in r["warnings"]:
                    print(f"      ⚠️ {w}")

    if failed:
        print(f"\n--- Failed Queries ---")
        for r in failed:
            print(f"  ❌ {r['query_id']}: {r.get('errors', ['Unknown error'])}")

    # Save detailed results
    output_file = f"/tmp/industrial_audit_{preset_name}_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        # Remove large synthesis from JSON (keep preview)
        for r in results:
            if r.get("synthesis"):
                r["synthesis_preview"] = r["synthesis"][:800]
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
                "avg_term_coverage": avg_term_coverage if successful else 0,
                "avg_quality_score": avg_quality if successful else 0
            },
            "results": results
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

    return results


async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_industrial_preset_audit.py <preset>")
        print("Available: research, full, both")
        print("\nQueries will test:")
        for q in INDUSTRIAL_QUERIES:
            print(f"  - [{q['category']}] {q['id']}: {q['query'][:60]}...")
        sys.exit(1)

    preset = sys.argv[1].lower()

    if preset == "both":
        # Run both presets sequentially
        print("\n" + "="*70)
        print("RUNNING RESEARCH PRESET FIRST")
        print("="*70)
        await run_preset_audit("research")

        print("\n" + "="*70)
        print("WAITING 30s BEFORE FULL PRESET")
        print("="*70)
        await asyncio.sleep(30)

        print("\n" + "="*70)
        print("RUNNING FULL PRESET")
        print("="*70)
        await run_preset_audit("full")
    elif preset in ["research", "full"]:
        await run_preset_audit(preset)
    else:
        print(f"Invalid preset: {preset}")
        print("Available: research, full, both")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
