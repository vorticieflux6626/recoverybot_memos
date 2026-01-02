#!/usr/bin/env python3
"""
FANUC 20-Query Test Suite for Directive Propagation Audit

Tests query classification accuracy, thinking model routing,
engine selection, and iteration limits across 5 tiers.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional, List
import httpx


@dataclass
class QueryTest:
    id: str
    tier: str
    query: str
    expects_thinking: bool
    min_confidence: float
    max_time_seconds: float


@dataclass
class TestResult:
    query_id: str
    tier: str
    query: str
    success: bool
    confidence: float
    used_thinking: bool
    expected_thinking: bool
    thinking_correct: bool
    response_time: float
    max_time: float
    time_ok: bool
    query_type: Optional[str] = None
    complexity: Optional[str] = None
    error: Optional[str] = None


# Define the 20-query test suite
# Time limits doubled (2025-01-02) to accommodate full agentic pipeline with web scraping
TEST_QUERIES = [
    # Tier 1: Knowledge (NO thinking model) - Simple lookups
    QueryTest("K1", "Knowledge", "What does FANUC SRVO-063 alarm mean?", False, 0.70, 120),
    QueryTest("K2", "Knowledge", "FANUC R-30iB J1 motor part number", False, 0.70, 120),
    QueryTest("K3", "Knowledge", "What is the $PARAM_GROUP[1].$PAYLOAD meaning?", False, 0.70, 120),
    QueryTest("K4", "Knowledge", "FANUC DCS Safe Position definition", False, 0.70, 120),

    # Tier 2: Diagnostic (YES thinking model) - Root cause analysis
    QueryTest("D1", "Diagnostic", "Robot intermittently loses J1 encoder position after warm-up. What's causing this?", True, 0.65, 300),
    QueryTest("D2", "Diagnostic", "SRVO-023 appearing only during fast J2 movements but not slow. Root cause?", True, 0.65, 300),
    QueryTest("D3", "Diagnostic", "Why does my FANUC robot drift after a warm restart but not cold start?", True, 0.65, 300),
    QueryTest("D4", "Diagnostic", "Intermittent MOTN-063 during palletizing cycle, happens more in humidity", True, 0.65, 300),

    # Tier 3: Procedural (MAYBE thinking model) - Step-by-step guides
    QueryTest("P1", "Procedural", "Step-by-step zero mastering procedure for FANUC M-20iD after J3 motor replacement", False, 0.60, 240),
    QueryTest("P2", "Procedural", "How to calibrate iRVision 2D camera for bin picking application?", False, 0.60, 240),
    QueryTest("P3", "Procedural", "Complete DCS configuration for collaborative robot safe zone setup", False, 0.60, 240),
    QueryTest("P4", "Procedural", "Backup and restore procedure for R-30iB controller with USB", False, 0.60, 240),

    # Tier 4: Expert (YES thinking model) - Complex comparisons
    QueryTest("E1", "Expert", "Compare FANUC R-2000iC vs M-900iB for automotive spot welding: reach, payload, cycle time", True, 0.55, 420),
    QueryTest("E2", "Expert", "Should I upgrade from R-30iA to R-30iB for existing KAREL programs? Compatibility issues?", True, 0.55, 420),
    QueryTest("E3", "Expert", "FANUC vs KUKA for high-precision assembly: encoder resolution, repeatability, servo tuning", True, 0.55, 420),
    QueryTest("E4", "Expert", "Troubleshoot intermittent actuation failures in press-tending application with DCS safety", True, 0.55, 420),

    # Tier 5: Multi-Domain (YES thinking model) - System integration
    QueryTest("M1", "Multi-Domain", "Design conveyor-to-robot handoff with FANUC iRVision, EtherNet/IP, and OPC-UA logging", True, 0.50, 600),
    QueryTest("M2", "Multi-Domain", "Retrofit legacy FANUC ARC Mate 100iC with modern collision detection and force control", True, 0.50, 600),
    QueryTest("M3", "Multi-Domain", "Implement dual-robot coordinated motion for large part handling with shared safe zones", True, 0.50, 600),
    QueryTest("M4", "Multi-Domain", "Troubleshoot FANUC SCARA vs articulated robot choice for PCB assembly with 0.02mm accuracy", True, 0.50, 600),
]


async def test_query(client: httpx.AsyncClient, query: QueryTest, preset: str = "balanced") -> TestResult:
    """Run a single query test and capture metrics."""
    start_time = time.time()

    try:
        # Call the gateway endpoint with SSE streaming
        response = await client.post(
            "http://localhost:8001/api/v1/search/gateway/stream",
            json={
                "query": query.query,
                "preset": preset,
            },
            timeout=query.max_time_seconds + 60  # Allow some buffer
        )

        elapsed = time.time() - start_time

        if response.status_code != 200:
            return TestResult(
                query_id=query.id,
                tier=query.tier,
                query=query.query,
                success=False,
                confidence=0.0,
                used_thinking=False,
                expected_thinking=query.expects_thinking,
                thinking_correct=False,
                response_time=elapsed,
                max_time=query.max_time_seconds,
                time_ok=False,
                error=f"HTTP {response.status_code}"
            )

        # Parse SSE response
        text = response.text
        confidence = 0.0
        used_thinking = False
        query_type = None
        complexity = None

        for line in text.split('\n'):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])

                    # Check for thinking model in synthesis
                    if data.get('event') == 'synthesizing':
                        model = data.get('data', {}).get('model', '')
                        if 'deepseek' in model.lower() or 'r1' in model.lower():
                            used_thinking = True

                    # Get query analysis info
                    if data.get('event') == 'query_analyzed':
                        analysis = data.get('data', {})
                        query_type = analysis.get('query_type')
                        complexity = analysis.get('complexity')
                        if analysis.get('requires_thinking_model'):
                            used_thinking = True

                    # Get final confidence
                    if data.get('event') == 'search_completed':
                        # Try both field names for compatibility
                        confidence = data.get('data', {}).get('confidence_score',
                                     data.get('data', {}).get('confidence', 0.0))

                except json.JSONDecodeError:
                    continue

        # Determine success
        thinking_correct = (used_thinking == query.expects_thinking) or query.tier == "Procedural"
        time_ok = elapsed <= query.max_time_seconds
        confidence_ok = confidence >= query.min_confidence

        return TestResult(
            query_id=query.id,
            tier=query.tier,
            query=query.query,
            success=thinking_correct and time_ok and confidence_ok,
            confidence=confidence,
            used_thinking=used_thinking,
            expected_thinking=query.expects_thinking,
            thinking_correct=thinking_correct,
            response_time=elapsed,
            max_time=query.max_time_seconds,
            time_ok=time_ok,
            query_type=query_type,
            complexity=complexity,
        )

    except httpx.TimeoutException:
        return TestResult(
            query_id=query.id,
            tier=query.tier,
            query=query.query,
            success=False,
            confidence=0.0,
            used_thinking=False,
            expected_thinking=query.expects_thinking,
            thinking_correct=False,
            response_time=time.time() - start_time,
            max_time=query.max_time_seconds,
            time_ok=False,
            error="Timeout"
        )
    except Exception as e:
        return TestResult(
            query_id=query.id,
            tier=query.tier,
            query=query.query,
            success=False,
            confidence=0.0,
            used_thinking=False,
            expected_thinking=query.expects_thinking,
            thinking_correct=False,
            response_time=time.time() - start_time,
            max_time=query.max_time_seconds,
            time_ok=False,
            error=str(e)
        )


async def run_tier(client: httpx.AsyncClient, tier: str, preset: str = "balanced") -> List[TestResult]:
    """Run all queries for a specific tier."""
    queries = [q for q in TEST_QUERIES if q.tier == tier]
    results = []

    print(f"\n{'='*60}")
    print(f"  TIER: {tier} ({len(queries)} queries)")
    print(f"{'='*60}")

    for query in queries:
        print(f"\n  [{query.id}] {query.query[:60]}...")
        result = await test_query(client, query, preset)
        results.append(result)

        # Print result
        status = "✅ PASS" if result.success else "❌ FAIL"
        thinking = "🔴 THINKING" if result.used_thinking else "🟢 FAST"
        expected = "expected" if result.thinking_correct else "MISMATCH"

        print(f"        {status} | {thinking} ({expected}) | "
              f"Conf: {result.confidence:.1%} | Time: {result.response_time:.1f}s")

        if result.error:
            print(f"        Error: {result.error}")
        if result.query_type:
            print(f"        Type: {result.query_type} | Complexity: {result.complexity}")

    return results


async def run_full_suite(preset: str = "balanced"):
    """Run the complete 20-query test suite."""
    print("\n" + "="*60)
    print("  FANUC 20-Query Directive Propagation Test Suite")
    print(f"  Preset: {preset}")
    print("="*60)

    async with httpx.AsyncClient() as client:
        # Check server health
        try:
            health = await client.get("http://localhost:8001/api/v1/system/health", timeout=5)
            if health.status_code != 200:
                print("❌ Server not healthy. Start the server first.")
                return
        except:
            print("❌ Cannot connect to server at localhost:8001. Start the server first.")
            return

        all_results = []
        tiers = ["Knowledge", "Diagnostic", "Procedural", "Expert", "Multi-Domain"]

        for tier in tiers:
            results = await run_tier(client, tier, preset)
            all_results.extend(results)

        # Print summary
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)

        total_pass = sum(1 for r in all_results if r.success)
        total = len(all_results)

        # Per-tier summary
        for tier in tiers:
            tier_results = [r for r in all_results if r.tier == tier]
            tier_pass = sum(1 for r in tier_results if r.success)
            tier_total = len(tier_results)
            thinking_correct = sum(1 for r in tier_results if r.thinking_correct)

            print(f"  {tier:15} | {tier_pass}/{tier_total} pass | "
                  f"Thinking: {thinking_correct}/{tier_total} correct")

        print("-"*60)
        print(f"  TOTAL: {total_pass}/{total} ({total_pass/total*100:.1f}%)")

        # Thinking model accuracy
        thinking_expected = [r for r in all_results if r.expected_thinking]
        thinking_correct = sum(1 for r in thinking_expected if r.used_thinking)
        fast_expected = [r for r in all_results if not r.expected_thinking and r.tier != "Procedural"]
        fast_correct = sum(1 for r in fast_expected if not r.used_thinking)

        print(f"\n  Thinking Model Accuracy:")
        print(f"    - Should think, did think: {thinking_correct}/{len(thinking_expected)}")
        print(f"    - Should be fast, was fast: {fast_correct}/{len(fast_expected)}")

        # Save results
        results_file = "/tmp/fanuc_20_query_results.json"
        with open(results_file, 'w') as f:
            json.dump([{
                "query_id": r.query_id,
                "tier": r.tier,
                "query": r.query,
                "success": r.success,
                "confidence": r.confidence,
                "used_thinking": r.used_thinking,
                "expected_thinking": r.expected_thinking,
                "thinking_correct": r.thinking_correct,
                "response_time": r.response_time,
                "query_type": r.query_type,
                "complexity": r.complexity,
                "error": r.error,
            } for r in all_results], f, indent=2)

        print(f"\n  Results saved to: {results_file}")

        return all_results


async def run_quick_test(num_queries: int = 4, preset: str = "balanced"):
    """Run a quick subset of queries for validation."""
    print(f"\n  Quick Test: {num_queries} queries (1 per tier K, D, P, E)")

    quick_queries = [
        TEST_QUERIES[0],   # K1
        TEST_QUERIES[4],   # D1
        TEST_QUERIES[8],   # P1
        TEST_QUERIES[12],  # E1
    ][:num_queries]

    async with httpx.AsyncClient() as client:
        results = []
        for query in quick_queries:
            print(f"\n  [{query.id}] {query.query[:50]}...")
            result = await test_query(client, query, preset)
            results.append(result)

            status = "✅" if result.success else "❌"
            thinking = "THINKING" if result.used_thinking else "FAST"
            print(f"        {status} | {thinking} | Conf: {result.confidence:.1%} | {result.response_time:.1f}s")

        passed = sum(1 for r in results if r.success)
        print(f"\n  Quick Test: {passed}/{len(results)} passed")
        return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(run_quick_test())
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        preset = sys.argv[2] if len(sys.argv) > 2 else "balanced"
        asyncio.run(run_full_suite(preset))
    else:
        print("Usage:")
        print("  python test_fanuc_20_query_suite.py quick       # Run 4 queries")
        print("  python test_fanuc_20_query_suite.py full        # Run all 20 queries")
        print("  python test_fanuc_20_query_suite.py full research  # Run with research preset")
