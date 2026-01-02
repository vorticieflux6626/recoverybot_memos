#!/usr/bin/env python3
"""
Pipeline Effectiveness Audit - Real Query Testing

Runs fresh industrial troubleshooting queries through the REAL agentic pipeline
(not simulation) to collect observability data for effectiveness analysis.

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python tests/test_pipeline_effectiveness_audit.py

Created: 2026-01-02
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("pipeline_audit")

# Add parent to path
sys.path.insert(0, '/home/sparkone/sdd/Recovery_Bot/memOS/server')

# Fresh industrial queries - designed to avoid cache hits
# Using specific part numbers, unique error combinations, and different contexts
AUDIT_QUERIES = [
    {
        "id": "allen-bradley-plc-1",
        "query": "Allen-Bradley 1756-L71 ControlLogix PLC showing major fault code 16#0001 on power-up after battery replacement. I/O tree won't come online.",
        "expected_domains": ["rockwellautomation.com", "plctalk.net", "stackoverflow.com"],
        "complexity": "high",
        "category": "plc_troubleshooting"
    },
    {
        "id": "siemens-drive-servo-1",
        "query": "SINAMICS S120 drive fault F07011 encoder 1 hardware fault on axis 2 of cartesian robot. Heidenhain ERN 1387 encoder. Safe torque off engaging unexpectedly.",
        "expected_domains": ["siemens.com", "support.industry.siemens.com"],
        "complexity": "expert",
        "category": "servo_drive_troubleshooting"
    },
    {
        "id": "hot-runner-controller-1",
        "query": "Mold-Masters TempMaster M2 zone 14 showing intermittent open thermocouple error E-12 but heater tests good at 25 ohms. Wiring checks passed. 500°F setpoint for PA66.",
        "expected_domains": ["moldmasters.com", "plasticsnews.com"],
        "complexity": "medium",
        "category": "injection_mold_hot_runner"
    },
    {
        "id": "fanuc-arc-welding-1",
        "query": "FANUC R-30iB Plus ARC Mate 100iD showing WELD-022 arc sensing deviation alarm during root pass on stainless steel pipe. Lincoln PowerWave S350 power source, 0.035 wire.",
        "expected_domains": ["fanuc.com", "lincolnelectric.com", "weldingweb.com"],
        "complexity": "expert",
        "category": "welding_robot_troubleshooting"
    },
    {
        "id": "kuka-collision-1",
        "query": "KUKA KR 210 R2700 prime with KR C4 controller stopped with collision detection triggered on axis A2 during palletizing. $TORQ_DIFF[2] exceeded 15%. No physical contact observed.",
        "expected_domains": ["kuka.com", "robot-forum.com"],
        "complexity": "high",
        "category": "robot_collision_detection"
    }
]


async def run_query_through_pipeline(query_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single query through the real UniversalOrchestrator pipeline.
    """
    from agentic import UniversalOrchestrator
    from agentic.models import SearchRequest

    request_id = f"audit-{query_info['id']}-{int(time.time())}"
    query = query_info["query"]

    logger.info(f"\n{'='*80}")
    logger.info(f"QUERY: {query_info['id']}")
    logger.info(f"Category: {query_info['category']} | Complexity: {query_info['complexity']}")
    logger.info(f"{'='*80}")
    logger.info(f"Query: {query[:100]}...")

    start_time = time.time()

    try:
        # Create orchestrator with ENHANCED preset for thorough evaluation
        orchestrator = UniversalOrchestrator(preset="enhanced")

        # Create search request
        request = SearchRequest(
            query=query,
            max_results=10,
            enable_scraping=True,
            enable_thinking_model=query_info["complexity"] in ["expert", "high"],
            request_id=request_id
        )

        # Run the pipeline
        logger.info("Starting pipeline execution...")
        result = await orchestrator.search(request)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract key metrics from result
        synthesis = result.synthesis if hasattr(result, 'synthesis') else str(result)
        confidence = result.confidence if hasattr(result, 'confidence') else 0.0
        sources = result.sources if hasattr(result, 'sources') else []

        # Analyze response quality
        quality_metrics = analyze_response_quality(
            query=query,
            synthesis=synthesis,
            expected_domains=query_info.get("expected_domains", []),
            sources=sources
        )

        logger.info(f"\n--- Results for {request_id} ---")
        logger.info(f"Duration: {duration_ms}ms")
        logger.info(f"Confidence: {confidence:.0%}")
        logger.info(f"Sources: {len(sources)}")
        logger.info(f"Synthesis length: {len(synthesis)} chars")
        logger.info(f"Quality Score: {quality_metrics['overall_score']:.0%}")

        # Print synthesis preview
        logger.info(f"\n--- Synthesis Preview ---")
        print(synthesis[:1000] + "..." if len(synthesis) > 1000 else synthesis)

        return {
            "request_id": request_id,
            "query_id": query_info["id"],
            "category": query_info["category"],
            "complexity": query_info["complexity"],
            "success": True,
            "duration_ms": duration_ms,
            "confidence": confidence,
            "source_count": len(sources),
            "synthesis_length": len(synthesis),
            "quality_metrics": quality_metrics,
            "synthesis_preview": synthesis[:500]
        }

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "request_id": request_id,
            "query_id": query_info["id"],
            "success": False,
            "error": str(e),
            "duration_ms": duration_ms
        }


def analyze_response_quality(
    query: str,
    synthesis: str,
    expected_domains: List[str],
    sources: List[Any]
) -> Dict[str, Any]:
    """
    Analyze the quality of the pipeline response.
    """
    import re

    # Check for citations
    citation_pattern = r'\[Source \d+\]|\[Domain Knowledge\]'
    citations = re.findall(citation_pattern, synthesis)
    has_citations = len(citations) > 0
    citation_count = len(citations)

    # Check for technical term coverage
    query_lower = query.lower()
    synthesis_lower = synthesis.lower()

    # Extract key technical terms from query
    technical_patterns = [
        r'[A-Z]+-\d+[A-Za-z]*',  # Error codes like SRVO-023, F07011
        r'\d+[A-Z]-[A-Z]+\d*',   # Part numbers like 1756-L71
        r'[A-Z]\d+(?:\s|$)',     # Axis names like A2, J1
        r'\d+(?:°[FC]|F|C)',     # Temperatures like 500°F
    ]

    query_terms = set()
    for pattern in technical_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        query_terms.update(m.lower() for m in matches)

    # Also extract key words
    key_words = [w for w in query.split() if len(w) > 4 and not w.lower() in
                 ['showing', 'during', 'after', 'before', 'with', 'from', 'that', 'this']]
    query_terms.update(w.lower() for w in key_words[:10])

    # Check term coverage
    terms_found = sum(1 for term in query_terms if term in synthesis_lower)
    term_coverage = terms_found / len(query_terms) if query_terms else 0

    # Check for structured response elements
    has_numbered_steps = bool(re.search(r'\d+\.\s+\w', synthesis))
    has_headers = bool(re.search(r'(?:^|\n)#+\s+\w|(?:^|\n)\*\*[^*]+\*\*:', synthesis))
    has_bullet_points = bool(re.search(r'(?:^|\n)\s*[-*•]\s+\w', synthesis))

    # Check response completeness
    is_short = len(synthesis) < 200
    is_too_generic = any(phrase in synthesis_lower for phrase in [
        "i cannot", "i don't have", "please provide more", "need more information"
    ])

    # Calculate overall score
    scores = {
        "has_citations": 0.25 if has_citations else 0,
        "term_coverage": term_coverage * 0.25,
        "structured": 0.15 if (has_numbered_steps or has_bullet_points) else 0,
        "completeness": 0.20 if not is_short and not is_too_generic else 0,
        "depth": min(0.15, len(synthesis) / 3000 * 0.15)  # Up to 0.15 for length
    }

    overall_score = sum(scores.values())

    return {
        "overall_score": overall_score,
        "citation_count": citation_count,
        "has_citations": has_citations,
        "term_coverage": term_coverage,
        "terms_found": terms_found,
        "terms_expected": len(query_terms),
        "has_structured_format": has_numbered_steps or has_bullet_points,
        "response_length": len(synthesis),
        "is_complete": not is_short and not is_too_generic,
        "component_scores": scores
    }


async def main():
    """Run the pipeline effectiveness audit."""
    logger.info("\n" + "="*80)
    logger.info("PIPELINE EFFECTIVENESS AUDIT")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*80 + "\n")

    results = []

    # Run each query
    for i, query_info in enumerate(AUDIT_QUERIES, 1):
        logger.info(f"\n[AUDIT {i}/{len(AUDIT_QUERIES)}]")
        result = await run_query_through_pipeline(query_info)
        results.append(result)

        # Brief pause between queries to avoid overwhelming the system
        if i < len(AUDIT_QUERIES):
            logger.info("Waiting 5s before next query...")
            await asyncio.sleep(5)

    # Generate audit summary
    logger.info("\n" + "="*80)
    logger.info("AUDIT SUMMARY")
    logger.info("="*80 + "\n")

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"Total Queries: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_duration = sum(r["duration_ms"] for r in successful) / len(successful)
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_quality = sum(r["quality_metrics"]["overall_score"] for r in successful) / len(successful)
        avg_citations = sum(r["quality_metrics"]["citation_count"] for r in successful) / len(successful)
        avg_term_coverage = sum(r["quality_metrics"]["term_coverage"] for r in successful) / len(successful)

        print(f"\n--- Aggregate Metrics ---")
        print(f"Avg Duration: {avg_duration:.0f}ms")
        print(f"Avg Confidence: {avg_confidence:.0%}")
        print(f"Avg Quality Score: {avg_quality:.0%}")
        print(f"Avg Citations: {avg_citations:.1f}")
        print(f"Avg Term Coverage: {avg_term_coverage:.0%}")

        print(f"\n--- Per-Query Results ---")
        for r in successful:
            qm = r["quality_metrics"]
            print(f"  {r['query_id']}: quality={qm['overall_score']:.0%}, "
                  f"citations={qm['citation_count']}, "
                  f"terms={qm['terms_found']}/{qm['terms_expected']}, "
                  f"duration={r['duration_ms']}ms")

    if failed:
        print(f"\n--- Failed Queries ---")
        for r in failed:
            print(f"  {r['query_id']}: {r.get('error', 'Unknown error')}")

    # Save detailed results to JSON
    output_file = f"/tmp/pipeline_audit_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "avg_duration_ms": avg_duration if successful else 0,
                "avg_confidence": avg_confidence if successful else 0,
                "avg_quality_score": avg_quality if successful else 0
            },
            "results": results
        }, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
