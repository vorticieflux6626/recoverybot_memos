#!/usr/bin/env python3
"""
Observability Analysis Test Suite

Tests queries across multiple domains and analyzes pipeline behavior
through the observability layer.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import httpx

# Test queries across different domains
TEST_QUERIES = [
    {
        "id": 1,
        "name": "FANUC Servo Error",
        "query": "FANUC robot showing SRVO-023 stop error excess during injection molding part extraction. J4 axis stops mid-cycle.",
        "domain": "fanuc",
        "expected_terms": ["SRVO-023", "servo", "J4", "encoder", "position"]
    },
    {
        "id": 2,
        "name": "Allen-Bradley PLC Fault",
        "query": "Allen-Bradley ControlLogix 1756-L72 showing major fault during EtherNet/IP communication with servo drives. How to use GSV instruction to read fault codes?",
        "domain": "industrial_automation",
        "expected_terms": ["1756", "major fault", "GSV", "EtherNet/IP", "servo"]
    },
    {
        "id": 3,
        "name": "Injection Molding Defect",
        "query": "Injection molded parts showing sink marks near ribs. Using ABS material with hot runner system. What process adjustments needed?",
        "domain": "imm",
        "expected_terms": ["sink", "ABS", "hot runner", "process", "pressure"]
    }
]


async def run_query(query_data: Dict, preset: str = "enhanced", timeout: int = 300) -> Dict[str, Any]:
    """Run a single query and capture observability data."""

    start_time = time.time()
    result = {
        "query_id": query_data["id"],
        "query_name": query_data["name"],
        "domain": query_data["domain"],
        "preset": preset,
        "start_time": datetime.now().isoformat(),
        "success": False,
        "duration_seconds": 0,
        "pipeline_stages": [],
        "metrics": {}
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(
                "http://localhost:8001/api/v1/search/universal",
                json={
                    "query": query_data["query"],
                    "preset": preset,
                    "max_iterations": 3
                }
            )

            duration = time.time() - start_time
            result["duration_seconds"] = duration

            if response.status_code == 200:
                data = response.json()
                result["success"] = data.get("success", False)

                # Extract key metrics
                resp_data = data.get("data", {})
                result["metrics"] = {
                    "confidence_score": resp_data.get("confidence_score"),
                    "synthesis_length": len(resp_data.get("synthesized_context", "")),
                    "sources_count": len(resp_data.get("sources", [])),
                    "iterations": resp_data.get("iterations", 0),
                    "search_enhanced": resp_data.get("search_enhanced", False)
                }

                # Extract pipeline trace
                trace = resp_data.get("trace", [])
                if trace:
                    result["pipeline_stages"] = trace

                # Check for expected terms
                synthesis = resp_data.get("synthesized_context", "").lower()
                found_terms = [t for t in query_data["expected_terms"] if t.lower() in synthesis]
                result["metrics"]["term_coverage"] = len(found_terms) / len(query_data["expected_terms"])
                result["metrics"]["found_terms"] = found_terms

                # Store synthesis preview
                result["synthesis_preview"] = resp_data.get("synthesized_context", "")[:500]

            else:
                result["error"] = f"HTTP {response.status_code}"

    except httpx.TimeoutException:
        result["error"] = f"Timeout after {timeout}s"
        result["duration_seconds"] = timeout
    except Exception as e:
        result["error"] = str(e)
        result["duration_seconds"] = time.time() - start_time

    return result


async def get_observability_metrics() -> Dict[str, Any]:
    """Fetch observability metrics from the server."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("http://localhost:8001/api/v1/search/metrics")
            if response.status_code == 200:
                return response.json().get("data", {})
    except Exception as e:
        print(f"Warning: Could not fetch metrics: {e}")
    return {}


def print_stage_analysis(stages: List[Dict]) -> None:
    """Print analysis of pipeline stages."""
    print("\n  Pipeline Trace:")
    print("  " + "-" * 60)

    for stage in stages:
        step = stage.get("step", "unknown")
        duration = stage.get("duration_ms", 0)

        # Format based on stage type
        if step == "analyze":
            print(f"  [A] Analysis: query_type={stage.get('query_type')}, "
                  f"complexity={stage.get('complexity')}, {duration}ms")
        elif step == "search":
            print(f"  [S] Search: {stage.get('queries_executed')} queries, "
                  f"{stage.get('results_found')} results, {duration}ms")
        elif step == "crag_evaluation":
            quality = stage.get("quality", "?")
            action = stage.get("action", "?")
            bypass = stage.get("crag_bypass", "")
            bypass_str = f" (bypass: {bypass})" if bypass else ""
            print(f"  [C] CRAG: quality={quality}, action={action}{bypass_str}")
        elif step == "scrape":
            print(f"  [W] Scrape: {stage.get('content_scraped')}/{stage.get('urls_attempted')} "
                  f"URLs, {duration}ms")
        elif step == "verify":
            print(f"  [V] Verify: {stage.get('verified_count')}/{stage.get('claims_checked')} "
                  f"claims, confidence={stage.get('confidence', 0):.2f}")
        elif step == "synthesize":
            util = stage.get("context_utilization", 0) * 100
            print(f"  [Σ] Synthesize: {stage.get('synthesis_length')} chars, "
                  f"context_util={util:.1f}%, {duration}ms")
        elif step == "pre_act_plan":
            print(f"  [P] Pre-Act: {stage.get('actions_planned')} actions planned")
        elif step == "multi_agent":
            print(f"  [M] Multi-Agent: {stage.get('perspectives')} perspectives")
        else:
            print(f"  [?] {step}: {stage}")


async def main():
    """Run observability analysis tests."""

    print("=" * 70)
    print("OBSERVABILITY ANALYSIS TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Get baseline metrics
    print("\n📊 Fetching baseline metrics...")
    baseline_metrics = await get_observability_metrics()

    results = []
    preset = "enhanced"

    for query_data in TEST_QUERIES:
        print(f"\n{'=' * 70}")
        print(f"Query {query_data['id']}: {query_data['name']}")
        print(f"Domain: {query_data['domain']} | Preset: {preset}")
        print(f"{'=' * 70}")
        print(f"\nQuery: {query_data['query'][:100]}...")
        print("\n⏳ Running query...")

        result = await run_query(query_data, preset=preset, timeout=480)
        results.append(result)

        # Print results
        if result["success"]:
            metrics = result["metrics"]
            print(f"\n✓ Completed in {result['duration_seconds']:.1f}s")
            print(f"  Confidence: {metrics.get('confidence_score', 0)*100:.1f}%")
            print(f"  Synthesis: {metrics.get('synthesis_length', 0)} chars")
            print(f"  Sources: {metrics.get('sources_count', 0)}")
            print(f"  Term Coverage: {metrics.get('term_coverage', 0)*100:.0f}% ({metrics.get('found_terms', [])})")

            # Print pipeline trace
            if result.get("pipeline_stages"):
                print_stage_analysis(result["pipeline_stages"])

            # Print synthesis preview
            if result.get("synthesis_preview"):
                print(f"\n  Synthesis Preview:")
                print(f"  {result['synthesis_preview'][:300]}...")
        else:
            print(f"\n✗ Failed: {result.get('error', 'Unknown error')}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r["success"]]

    if successful:
        avg_duration = sum(r["duration_seconds"] for r in successful) / len(successful)
        avg_confidence = sum(r["metrics"].get("confidence_score", 0) or 0 for r in successful) / len(successful)
        avg_coverage = sum(r["metrics"].get("term_coverage", 0) for r in successful) / len(successful)

        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        print(f"Avg Duration: {avg_duration:.1f}s")
        print(f"Avg Confidence: {avg_confidence*100:.1f}%")
        print(f"Avg Term Coverage: {avg_coverage*100:.0f}%")

    # Save results
    output_file = "/tmp/observability_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
