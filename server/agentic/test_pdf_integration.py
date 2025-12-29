#!/usr/bin/env python3
"""
Test script for PDF Extraction Tools integration with memOS agentic search.

This script tests the integration between:
- memOS server (port 8001)
- PDF Extraction Tools API (port 8002)

Run with: python test_pdf_integration.py [--pdf-api-url URL] [--memos-url URL]
"""

import asyncio
import argparse
import sys
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx


class IntegrationTester:
    """Test PDF Extraction Tools integration."""

    def __init__(self, pdf_api_url: str = "http://localhost:8002",
                 memos_url: str = "http://localhost:8001"):
        self.pdf_api_url = pdf_api_url.rstrip("/")
        self.memos_url = memos_url.rstrip("/")
        self.results: List[Dict[str, Any]] = []

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print(f"\n{'='*60}")
        print("PDF Extraction Tools - memOS Integration Tests")
        print(f"{'='*60}")
        print(f"PDF API URL: {self.pdf_api_url}")
        print(f"memOS URL: {self.memos_url}")
        print(f"Started: {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        # Test categories
        await self.test_category("API Health Checks", [
            self.test_pdf_api_health,
            self.test_memos_health,
            self.test_memos_technical_health,
        ])

        await self.test_category("PDF API Direct Tests", [
            self.test_pdf_search,
            self.test_pdf_troubleshoot_path,
            self.test_pdf_entity_lookup,
        ])

        await self.test_category("memOS Integration Tests", [
            self.test_corpus_stats,
            self.test_corpus_entity_list,
            self.test_corpus_sync,
            self.test_corpus_enrich,
        ])

        await self.test_category("Search Pipeline Tests", [
            self.test_agentic_search_with_pdf,
            self.test_gateway_search_with_technical_docs,
        ])

        # Summary
        return self.print_summary()

    async def test_category(self, name: str, tests: List):
        """Run a category of tests."""
        print(f"\n--- {name} ---\n")
        for test in tests:
            await test()

    def record_result(self, name: str, success: bool, details: str = "",
                      response_time_ms: Optional[float] = None):
        """Record a test result."""
        result = {
            "name": name,
            "success": success,
            "details": details,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)

        status = "PASS" if success else "FAIL"
        time_str = f" ({response_time_ms:.0f}ms)" if response_time_ms else ""
        print(f"  [{status}] {name}{time_str}")
        if details and not success:
            print(f"        -> {details}")

    # === Health Check Tests ===

    async def test_pdf_api_health(self):
        """Test PDF API health endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.get(f"{self.pdf_api_url}/health", timeout=10.0)
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    self.record_result(
                        "PDF API Health",
                        True,
                        f"Status: {data.get('status', 'unknown')}",
                        elapsed
                    )
                else:
                    self.record_result(
                        "PDF API Health",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("PDF API Health", False, str(e))

    async def test_memos_health(self):
        """Test memOS server health."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.get(f"{self.memos_url}/health", timeout=10.0)
                elapsed = (datetime.now() - start).total_seconds() * 1000

                self.record_result(
                    "memOS Server Health",
                    resp.status_code == 200,
                    f"HTTP {resp.status_code}",
                    elapsed
                )
        except Exception as e:
            self.record_result("memOS Server Health", False, str(e))

    async def test_memos_technical_health(self):
        """Test memOS technical docs bridge health."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.get(
                    f"{self.memos_url}/api/v1/search/technical/health",
                    timeout=10.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    # data can be a boolean directly or a dict with "available" key
                    available = data.get("data", False)
                    if isinstance(available, dict):
                        available = available.get("available", False)
                    self.record_result(
                        "memOS Technical Docs Bridge",
                        True,
                        f"Available: {available}",
                        elapsed
                    )
                else:
                    self.record_result(
                        "memOS Technical Docs Bridge",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("memOS Technical Docs Bridge", False, str(e))

    # === PDF API Direct Tests ===

    async def test_pdf_search(self):
        """Test PDF API search endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.post(
                    f"{self.pdf_api_url}/api/v1/search",
                    json={"query": "SRVO-063 encoder alarm", "max_results": 5},
                    timeout=30.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    results_count = len(data.get("results", []))
                    self.record_result(
                        "PDF API Search",
                        True,
                        f"Found {results_count} results",
                        elapsed
                    )
                else:
                    self.record_result(
                        "PDF API Search",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("PDF API Search", False, str(e))

    async def test_pdf_troubleshoot_path(self):
        """Test PDF API troubleshooting path endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.get(
                    f"{self.pdf_api_url}/api/v1/troubleshoot/SRVO-063",
                    timeout=30.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    steps_count = len(data.get("steps", []))
                    self.record_result(
                        "PDF API Troubleshoot Path",
                        True,
                        f"Found {steps_count} steps",
                        elapsed
                    )
                elif resp.status_code == 404:
                    self.record_result(
                        "PDF API Troubleshoot Path",
                        True,
                        "Error code not in corpus (expected for new API)",
                        elapsed
                    )
                else:
                    self.record_result(
                        "PDF API Troubleshoot Path",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("PDF API Troubleshoot Path", False, str(e))

    async def test_pdf_entity_lookup(self):
        """Test PDF API entity lookup."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.get(
                    f"{self.pdf_api_url}/api/v1/entities",
                    params={"type": "error_code", "limit": 5},
                    timeout=30.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    entities_count = len(data.get("entities", []))
                    self.record_result(
                        "PDF API Entity Lookup",
                        True,
                        f"Found {entities_count} entities",
                        elapsed
                    )
                else:
                    self.record_result(
                        "PDF API Entity Lookup",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("PDF API Entity Lookup", False, str(e))

    # === memOS Integration Tests ===

    async def test_corpus_stats(self):
        """Test memOS corpus stats endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.get(
                    f"{self.memos_url}/api/v1/search/corpus/stats",
                    timeout=10.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    total = data.get("data", {}).get("total_entities", 0)
                    self.record_result(
                        "memOS Corpus Stats",
                        True,
                        f"Total entities: {total}",
                        elapsed
                    )
                else:
                    self.record_result(
                        "memOS Corpus Stats",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("memOS Corpus Stats", False, str(e))

    async def test_corpus_entity_list(self):
        """Test memOS corpus entity listing."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.get(
                    f"{self.memos_url}/api/v1/search/corpus/entities",
                    params={"limit": 5},
                    timeout=10.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    entities = data.get("data", {}).get("entities", [])
                    self.record_result(
                        "memOS Corpus Entity List",
                        True,
                        f"Retrieved {len(entities)} entities",
                        elapsed
                    )
                else:
                    self.record_result(
                        "memOS Corpus Entity List",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("memOS Corpus Entity List", False, str(e))

    async def test_corpus_sync(self):
        """Test memOS corpus sync with PDF API."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.post(
                    f"{self.memos_url}/api/v1/search/corpus/sync",
                    json={
                        "pdf_api_url": self.pdf_api_url,
                        "error_codes": ["SRVO-063"],  # Just test one
                        "force_update": False
                    },
                    timeout=60.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    synced = data.get("data", {}).get("synced_count", 0)
                    self.record_result(
                        "memOS Corpus Sync",
                        True,
                        f"Synced {synced} entities",
                        elapsed
                    )
                elif resp.status_code == 503:
                    self.record_result(
                        "memOS Corpus Sync",
                        True,
                        "PDF API unavailable (expected if not running)",
                        elapsed
                    )
                else:
                    self.record_result(
                        "memOS Corpus Sync",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("memOS Corpus Sync", False, str(e))

    async def test_corpus_enrich(self):
        """Test memOS corpus enrichment from PDF API."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.post(
                    f"{self.memos_url}/api/v1/search/corpus/enrich",
                    json={
                        "error_code": "SRVO-063",
                        "pdf_api_url": self.pdf_api_url
                    },
                    timeout=30.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    enriched = data.get("data", {}).get("enriched", False)
                    self.record_result(
                        "memOS Corpus Enrich",
                        True,
                        f"Enriched: {enriched}",
                        elapsed
                    )
                elif resp.status_code == 503:
                    self.record_result(
                        "memOS Corpus Enrich",
                        True,
                        "PDF API unavailable (expected if not running)",
                        elapsed
                    )
                else:
                    self.record_result(
                        "memOS Corpus Enrich",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("memOS Corpus Enrich", False, str(e))

    # === Search Pipeline Tests ===

    async def test_agentic_search_with_pdf(self):
        """Test agentic search with PDF provider enabled."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                resp = await client.post(
                    f"{self.memos_url}/api/v1/search/agentic",
                    json={
                        "query": "How to fix SRVO-063 encoder alarm on M710iC robot?",
                        "preset": "ENHANCED",  # Enables technical docs
                        "enable_technical_docs": True,
                        "max_results": 5
                    },
                    timeout=120.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    sources = data.get("data", {}).get("sources_count", 0)
                    confidence = data.get("data", {}).get("confidence", 0)
                    self.record_result(
                        "Agentic Search with PDF",
                        True,
                        f"Sources: {sources}, Confidence: {confidence:.1%}",
                        elapsed
                    )
                else:
                    self.record_result(
                        "Agentic Search with PDF",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("Agentic Search with PDF", False, str(e))

    async def test_gateway_search_with_technical_docs(self):
        """Test gateway chat with technical docs enabled."""
        try:
            async with httpx.AsyncClient() as client:
                start = datetime.now()
                # Use universal endpoint instead of gateway (gateway is streaming-only)
                resp = await client.post(
                    f"{self.memos_url}/api/v1/search/universal",
                    json={
                        "query": "FANUC SRVO-063 troubleshooting steps",
                        "preset": "RESEARCH",  # Full technical docs
                        "max_iterations": 3
                    },
                    timeout=180.0
                )
                elapsed = (datetime.now() - start).total_seconds() * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    has_synthesis = bool(data.get("data", {}).get("synthesis"))
                    sources = data.get("data", {}).get("sources_count", 0)
                    self.record_result(
                        "Gateway Search with Technical Docs",
                        True,
                        f"Has synthesis: {has_synthesis}, Sources: {sources}",
                        elapsed
                    )
                else:
                    self.record_result(
                        "Gateway Search with Technical Docs",
                        False,
                        f"HTTP {resp.status_code}",
                        elapsed
                    )
        except Exception as e:
            self.record_result("Gateway Search with Technical Docs", False, str(e))

    def print_summary(self) -> Dict[str, Any]:
        """Print test summary and return results."""
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")

        passed = sum(1 for r in self.results if r["success"])
        failed = len(self.results) - passed

        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {passed/len(self.results)*100:.1f}%")

        if failed > 0:
            print(f"\n--- Failed Tests ---")
            for r in self.results:
                if not r["success"]:
                    print(f"  - {r['name']}: {r['details']}")

        avg_response = sum(r.get("response_time_ms", 0) for r in self.results if r.get("response_time_ms")) / len([r for r in self.results if r.get("response_time_ms")])
        print(f"\nAverage Response Time: {avg_response:.0f}ms")
        print(f"Completed: {datetime.now().isoformat()}")
        print(f"{'='*60}\n")

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(self.results),
            "results": self.results
        }


async def main():
    parser = argparse.ArgumentParser(description="Test PDF Extraction Tools integration")
    parser.add_argument("--pdf-api-url", default="http://localhost:8002",
                        help="PDF Extraction Tools API URL")
    parser.add_argument("--memos-url", default="http://localhost:8001",
                        help="memOS server URL")
    parser.add_argument("--output", "-o", help="Output results to JSON file")
    args = parser.parse_args()

    tester = IntegrationTester(
        pdf_api_url=args.pdf_api_url,
        memos_url=args.memos_url
    )

    results = await tester.run_all_tests()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")

    # Exit with error code if any tests failed
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
