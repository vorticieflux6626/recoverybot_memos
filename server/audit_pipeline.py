#!/usr/bin/env python3
"""
Pipeline Audit Script - Comprehensive testing of the agentic search pipeline.
Tests multiple presets, captures timing, SSE events, and response quality.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Optional

BASE_URL = "http://localhost:8001"

# Test queries spanning different complexity levels
TEST_QUERIES = [
    {
        "query": "What is the capital of France?",
        "preset": "minimal",
        "expected_type": "factual",
        "description": "Simple factual query - should be fast"
    },
    {
        "query": "FANUC SRVO-063 alarm causes and solutions",
        "preset": "balanced",
        "expected_type": "technical",
        "description": "Technical troubleshooting - domain-specific"
    },
    {
        "query": "Compare RAG architectures: naive vs advanced retrieval techniques in 2024",
        "preset": "research",
        "expected_type": "research",
        "description": "Research query - multi-step reasoning"
    },
]


async def test_gateway_stream(query: str, preset: str, timeout: int = 180) -> dict:
    """Test the gateway streaming endpoint and capture all events."""
    results = {
        "query": query,
        "preset": preset,
        "events": [],
        "errors": [],
        "timing": {},
        "response": None,
        "sources": [],
        "confidence": None,
        "graph_line": None,
    }

    start_time = time.time()

    payload = {
        "query": query,
        "preset": preset,
        "stream": True,
        "max_iterations": 5
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/search/gateway/stream",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                results["timing"]["first_byte"] = time.time() - start_time

                if response.status != 200:
                    results["errors"].append(f"HTTP {response.status}")
                    return results

                buffer = ""
                async for chunk in response.content:
                    text = chunk.decode('utf-8')
                    buffer += text

                    # Parse SSE events (multi-line format: "event: type\ndata: {json}\n\n")
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        # Handle multi-line SSE format
                        lines = event_str.strip().split('\n')
                        data_line = None
                        for line in lines:
                            if line.startswith("data: "):
                                data_line = line[6:]  # Extract JSON after "data: "
                                break
                        if data_line:
                            try:
                                event_data = json.loads(data_line)
                                event_record = {
                                    "type": event_data.get("event", "unknown"),
                                    "time": time.time() - start_time,
                                    "message": event_data.get("message", ""),
                                }

                                # Capture specific data
                                if "graph_line" in event_data:
                                    results["graph_line"] = event_data["graph_line"]
                                    event_record["graph_line"] = event_data["graph_line"]

                                if "data" in event_data:
                                    data = event_data["data"]
                                    if isinstance(data, dict):
                                        if "confidence" in data:
                                            results["confidence"] = data["confidence"]
                                        if "synthesis" in data:
                                            results["response"] = data["synthesis"][:500] + "..." if len(data.get("synthesis", "")) > 500 else data.get("synthesis")
                                        if "sources" in data:
                                            results["sources"] = data["sources"]

                                results["events"].append(event_record)

                            except json.JSONDecodeError:
                                pass

    except asyncio.TimeoutError:
        results["errors"].append(f"Timeout after {timeout}s")
    except Exception as e:
        results["errors"].append(str(e))

    results["timing"]["total"] = time.time() - start_time
    return results


async def test_universal_search(query: str, preset: str, timeout: int = 180) -> dict:
    """Test the universal search endpoint (non-streaming)."""
    results = {
        "query": query,
        "preset": preset,
        "errors": [],
        "timing": {},
        "response": None,
        "sources": [],
        "confidence": None,
    }

    start_time = time.time()

    payload = {
        "query": query,
        "preset": preset,
        "max_iterations": 5
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/search/universal",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                results["timing"]["response_time"] = time.time() - start_time

                if response.status != 200:
                    results["errors"].append(f"HTTP {response.status}")
                    text = await response.text()
                    results["errors"].append(text[:500])
                    return results

                data = await response.json()

                if data.get("success"):
                    resp_data = data.get("data", {})
                    results["response"] = resp_data.get("synthesis", resp_data.get("response", ""))[:500]
                    results["sources"] = resp_data.get("sources", [])
                    results["confidence"] = resp_data.get("confidence")
                else:
                    results["errors"].append(data.get("errors", []))

    except asyncio.TimeoutError:
        results["errors"].append(f"Timeout after {timeout}s")
    except Exception as e:
        results["errors"].append(str(e))

    results["timing"]["total"] = time.time() - start_time
    return results


def print_results(results: dict, test_info: dict):
    """Print formatted test results."""
    print("\n" + "="*80)
    print(f"TEST: {test_info['description']}")
    print(f"Query: {results['query']}")
    print(f"Preset: {results['preset']}")
    print("="*80)

    if results.get("errors"):
        print(f"\n❌ ERRORS: {results['errors']}")

    print(f"\n⏱️  TIMING:")
    for key, val in results.get("timing", {}).items():
        print(f"   {key}: {val:.2f}s")

    if results.get("events"):
        print(f"\n📊 SSE EVENTS ({len(results['events'])} total):")
        event_types = {}
        for e in results["events"]:
            event_types[e["type"]] = event_types.get(e["type"], 0) + 1
        for etype, count in sorted(event_types.items()):
            print(f"   {etype}: {count}")

        # Show last few events
        print("\n   Last 5 events:")
        for e in results["events"][-5:]:
            graph = f" [{e.get('graph_line', '')}]" if e.get('graph_line') else ""
            print(f"   - {e['time']:.1f}s: {e['type']}{graph}")

    if results.get("graph_line"):
        print(f"\n🔀 FINAL GRAPH: {results['graph_line']}")

    if results.get("confidence"):
        print(f"\n📈 CONFIDENCE: {results['confidence']:.2%}")

    if results.get("sources"):
        print(f"\n📚 SOURCES ({len(results['sources'])} found):")
        for i, src in enumerate(results["sources"][:5]):
            if isinstance(src, dict):
                print(f"   {i+1}. {src.get('title', src.get('url', 'Unknown'))[:60]}")
            else:
                print(f"   {i+1}. {str(src)[:60]}")

    if results.get("response"):
        print(f"\n📝 RESPONSE PREVIEW:")
        print(f"   {results['response'][:300]}...")

    return not results.get("errors")


async def run_audit():
    """Run full pipeline audit."""
    print("\n" + "="*80)
    print("🔍 AGENTIC SEARCH PIPELINE AUDIT")
    print(f"   Started: {datetime.now().isoformat()}")
    print("="*80)

    all_results = []
    passed = 0
    failed = 0

    for test_info in TEST_QUERIES:
        print(f"\n\n🧪 Testing: {test_info['description']}...")

        # Test streaming endpoint
        print("\n--- Testing Gateway Stream ---")
        stream_results = await test_gateway_stream(
            test_info["query"],
            test_info["preset"],
            timeout=300
        )
        stream_success = print_results(stream_results, test_info)

        all_results.append({
            "test": test_info,
            "endpoint": "gateway/stream",
            "results": stream_results,
            "success": stream_success
        })

        if stream_success:
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n\n" + "="*80)
    print("📊 AUDIT SUMMARY")
    print("="*80)
    print(f"   Total tests: {len(all_results)}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")

    # Identify issues
    print("\n🔧 IDENTIFIED ISSUES:")
    issues = []
    for r in all_results:
        if r["results"].get("errors"):
            issues.append(f"  - {r['test']['description']}: {r['results']['errors']}")
        if r["results"].get("timing", {}).get("total", 0) > 120:
            issues.append(f"  - {r['test']['description']}: Slow response ({r['results']['timing']['total']:.0f}s)")
        if not r["results"].get("sources") and r["test"]["preset"] != "minimal":
            issues.append(f"  - {r['test']['description']}: No sources returned")
        if r["results"].get("confidence") and r["results"]["confidence"] < 0.5:
            issues.append(f"  - {r['test']['description']}: Low confidence ({r['results']['confidence']:.0%})")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("   No critical issues found!")

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_audit())
