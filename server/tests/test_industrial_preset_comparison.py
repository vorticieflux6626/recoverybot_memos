#!/usr/bin/env python3
"""
Industrial Automation Preset Comparison Test Suite

Runs complex industrial automation queries through BALANCED, ENHANCED, RESEARCH, and FULL presets
with comprehensive observability analysis and duration tracking.

Usage:
    python test_industrial_preset_comparison.py [--query N] [--preset PRESET]

    --query N: Run only query N (1-5)
    --preset PRESET: Run only specified preset
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


# Complex industrial automation test queries (research-based)
INDUSTRIAL_QUERIES = [
    {
        "id": 1,
        "name": "FANUC Injection Molding Servo Alarms",
        "query": """FANUC R-30iB Plus controller showing intermittent SRVO-023 (Stop error excess) and
MOTN-023 (In singularity) alarms during part extraction from Arburg injection molding machine.
Robot stops mid-cycle when gripper approaches sprue picker position. Suspected issues with J4 axis
servo or interference from hot runner controller. How to diagnose servo following error vs
electrical noise from mold heaters? Need step-by-step troubleshooting procedure.""",
        "expected_terms": ["SRVO-023", "MOTN-023", "servo", "encoder", "following error", "singularity", "position"],
        "domain": "fanuc_robotics"
    },
    {
        "id": 2,
        "name": "Allen-Bradley ControlLogix Major Fault Recovery",
        "query": """Allen-Bradley 1756-L72 ControlLogix PLC experiencing T01:C62 major fault during
communication with Rockwell Kinetix 5500 servo drives over EtherNet/IP. Fault occurs when
executing coordinated motion with 4 axes. Studio 5000 shows fault in motion group. How to
configure Controller Fault Handler to recover gracefully and log diagnostic data? Need to
understand GSV/SSV objects for fault code extraction.""",
        "expected_terms": ["1756", "major fault", "T01", "Controller Fault Handler", "GSV", "SSV", "motion"],
        "domain": "allen_bradley"
    },
    {
        "id": 3,
        "name": "Siemens S7-1500 PROFINET Servo Integration",
        "query": """Siemens S7-1517 TF with PROFINET IRT connection to Sinamics S120 multi-axis servo
system dropping communication cyclically. OB82 diagnostic interrupt triggered with 16#F015
hardware fault. TIA Portal shows "Station not available" for drive CU320-2 PN. How to use
DeviceStates and ModuleStates instructions to diagnose PROFINET telegram loss? Need to configure
proper watchdog and implement automatic reconnection logic.""",
        "expected_terms": ["S7-1500", "PROFINET", "OB82", "diagnostic", "telegram", "DeviceStates", "Sinamics"],
        "domain": "siemens_automation"
    },
    {
        "id": 4,
        "name": "Hot Runner Controller Integration with Robot",
        "query": """Mold-Masters TempMaster M2 hot runner controller causing interference with FANUC
LR Mate 200iD/7L robot end-of-arm tooling signals. Thermocouple noise spikes when heater zones
cycle, corrupting vacuum sensor and gripper position feedback. Robot throws SRVO-062 (BZAL alarm)
and loses absolute encoder reference. How to properly ground and shield analog signals? Need
EMC compliance solution for 48-zone hot runner in tight cell layout.""",
        "expected_terms": ["hot runner", "EMC", "shielding", "grounding", "thermocouple", "SRVO-062", "encoder"],
        "domain": "industrial_integration"
    },
    {
        "id": 5,
        "name": "Multi-Robot Cell Synchronization Faults",
        "query": """Dual FANUC M-20iA robots in automotive welding cell losing synchronization during
coordinated spot welding sequence. Robot 1 (master) and Robot 2 (slave) connected via RSI
(Robot Server Interface) over DeviceNet. Intermittent SYST-211 (Group axis not ready) and
MOTN-063 (Configuration mismatch) errors when executing timed wait points. Cycle time
increasing from 45s to 52s with frequent recoveries. How to diagnose RSI communication
latency and configure proper motion group synchronization?""",
        "expected_terms": ["synchronization", "RSI", "DeviceNet", "SYST-211", "MOTN-063", "coordinated", "motion group"],
        "domain": "multi_robot"
    }
]

PRESETS = ["balanced", "enhanced", "research", "full"]

# Timeout configuration (in seconds)
PRESET_TIMEOUTS = {
    "balanced": 300,    # 5 minutes
    "enhanced": 480,    # 8 minutes
    "research": 720,    # 12 minutes
    "full": 900         # 15 minutes
}


@dataclass
class TestResult:
    """Result of a single preset test"""
    query_id: int
    query_name: str
    preset: str
    success: bool
    duration_seconds: float
    confidence_score: Optional[float] = None
    synthesis_length: int = 0
    sources_count: int = 0
    domain_knowledge_cited: bool = False
    expected_terms_found: List[str] = field(default_factory=list)
    term_coverage_pct: float = 0.0
    search_trace: List[Dict] = field(default_factory=list)
    error_message: Optional[str] = None
    crag_quality: Optional[str] = None
    crag_bypass: Optional[str] = None
    verified_claims: int = 0
    total_claims: int = 0
    verification_confidence: float = 0.0


@dataclass
class QueryResults:
    """All results for a single query across presets"""
    query_id: int
    query_name: str
    query_text: str
    results: Dict[str, TestResult] = field(default_factory=dict)

    def best_preset(self) -> Optional[str]:
        """Return preset with best composite score"""
        scores = {}
        for preset, result in self.results.items():
            if result.success:
                # Composite: confidence + term coverage + domain knowledge bonus
                score = (
                    (result.confidence_score or 0) * 0.4 +
                    result.term_coverage_pct * 0.4 +
                    (0.2 if result.domain_knowledge_cited else 0)
                )
                scores[preset] = score
        return max(scores, key=scores.get) if scores else None


class IndustrialPresetTester:
    """Run industrial queries through multiple presets with observability analysis"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results: List[QueryResults] = []
        self.output_dir = Path("/tmp/industrial_preset_tests")
        self.output_dir.mkdir(exist_ok=True)

    async def run_single_test(
        self,
        query: Dict,
        preset: str,
        timeout: int
    ) -> TestResult:
        """Run a single query with a specific preset"""
        query_id = query["id"]
        query_name = query["name"]
        query_text = query["query"]
        expected_terms = query["expected_terms"]

        print(f"\n{'='*60}")
        print(f"Query {query_id}: {query_name}")
        print(f"Preset: {preset.upper()}")
        print(f"Timeout: {timeout}s ({timeout/60:.1f} min)")
        print(f"{'='*60}")

        start_time = time.time()

        request_payload = {
            "query": query_text,
            "preset": preset,
            "max_iterations": 3 if preset in ["balanced", "enhanced"] else 5,
            "stream": False
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/v1/search/universal",
                    json=request_payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    duration = time.time() - start_time

                    if response.status != 200:
                        return TestResult(
                            query_id=query_id,
                            query_name=query_name,
                            preset=preset,
                            success=False,
                            duration_seconds=duration,
                            error_message=f"HTTP {response.status}"
                        )

                    data = await response.json()

                    if not data.get("success"):
                        return TestResult(
                            query_id=query_id,
                            query_name=query_name,
                            preset=preset,
                            success=False,
                            duration_seconds=duration,
                            error_message=str(data.get("errors", "Unknown error"))
                        )

                    result_data = data.get("data", {})
                    synthesis = result_data.get("synthesized_context", "")
                    sources = result_data.get("sources", [])
                    search_trace = result_data.get("search_trace", [])

                    # Check for expected terms
                    synthesis_lower = synthesis.lower()
                    found_terms = [t for t in expected_terms if t.lower() in synthesis_lower]
                    term_coverage = len(found_terms) / len(expected_terms) * 100 if expected_terms else 0

                    # Check for domain knowledge citation
                    domain_knowledge_cited = "[Domain Knowledge]" in synthesis

                    # Extract CRAG info from trace
                    crag_quality = None
                    crag_bypass = None
                    verified_claims = 0
                    total_claims = 0
                    verification_confidence = 0.0

                    for step in search_trace:
                        if step.get("step") == "crag_evaluation":
                            crag_quality = step.get("quality")
                            crag_bypass = step.get("crag_bypass")
                        if step.get("step") == "verify":
                            verified_claims = step.get("verified_count", 0)
                            total_claims = step.get("claims_checked", 0)
                            verification_confidence = step.get("confidence", 0)

                    result = TestResult(
                        query_id=query_id,
                        query_name=query_name,
                        preset=preset,
                        success=True,
                        duration_seconds=duration,
                        confidence_score=result_data.get("confidence_score"),
                        synthesis_length=len(synthesis),
                        sources_count=len(sources),
                        domain_knowledge_cited=domain_knowledge_cited,
                        expected_terms_found=found_terms,
                        term_coverage_pct=term_coverage,
                        search_trace=search_trace,
                        crag_quality=crag_quality,
                        crag_bypass=crag_bypass,
                        verified_claims=verified_claims,
                        total_claims=total_claims,
                        verification_confidence=verification_confidence
                    )

                    # Save synthesis for review
                    synthesis_file = self.output_dir / f"query{query_id}_{preset}_synthesis.txt"
                    synthesis_file.write_text(synthesis)

                    # Print summary
                    print(f"\n✓ Completed in {duration:.1f}s ({duration/60:.2f} min)")
                    print(f"  Confidence: {result.confidence_score:.2%}" if result.confidence_score else "  Confidence: N/A")
                    print(f"  Synthesis: {result.synthesis_length} chars")
                    print(f"  Sources: {result.sources_count}")
                    print(f"  Domain Knowledge cited: {result.domain_knowledge_cited}")
                    print(f"  Term coverage: {result.term_coverage_pct:.1f}% ({len(found_terms)}/{len(expected_terms)})")
                    print(f"  CRAG: quality={crag_quality}, bypass={crag_bypass}")
                    print(f"  Verification: {verified_claims}/{total_claims} claims, confidence={verification_confidence:.2f}")

                    return result

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            print(f"\n✗ TIMEOUT after {duration:.1f}s")
            return TestResult(
                query_id=query_id,
                query_name=query_name,
                preset=preset,
                success=False,
                duration_seconds=duration,
                error_message=f"Timeout after {timeout}s"
            )
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n✗ ERROR: {e}")
            return TestResult(
                query_id=query_id,
                query_name=query_name,
                preset=preset,
                success=False,
                duration_seconds=duration,
                error_message=str(e)
            )

    async def run_query_all_presets(
        self,
        query: Dict,
        presets: Optional[List[str]] = None
    ) -> QueryResults:
        """Run a single query through all specified presets"""
        presets = presets or PRESETS
        query_results = QueryResults(
            query_id=query["id"],
            query_name=query["name"],
            query_text=query["query"]
        )

        for preset in presets:
            timeout = PRESET_TIMEOUTS.get(preset, 600)
            result = await self.run_single_test(query, preset, timeout)
            query_results.results[preset] = result

            # Brief pause between presets to let system recover
            if preset != presets[-1]:
                print(f"\n⏳ Waiting 10s before next preset...")
                await asyncio.sleep(10)

        return query_results

    def print_comparison_table(self, query_results: QueryResults):
        """Print comparison table for a query across presets"""
        print(f"\n{'='*80}")
        print(f"COMPARISON: {query_results.query_name}")
        print(f"{'='*80}")

        # Header
        print(f"\n{'Preset':<12} {'Status':<8} {'Duration':<12} {'Confidence':<12} {'Terms':<10} {'DomainKB':<10} {'Sources':<8}")
        print("-" * 80)

        for preset in PRESETS:
            if preset in query_results.results:
                r = query_results.results[preset]
                status = "✓" if r.success else "✗"
                duration = f"{r.duration_seconds:.1f}s"
                confidence = f"{r.confidence_score:.1%}" if r.confidence_score else "N/A"
                terms = f"{r.term_coverage_pct:.0f}%"
                domain_kb = "Yes" if r.domain_knowledge_cited else "No"
                sources = str(r.sources_count)

                print(f"{preset.upper():<12} {status:<8} {duration:<12} {confidence:<12} {terms:<10} {domain_kb:<10} {sources:<8}")

        best = query_results.best_preset()
        if best:
            print(f"\n🏆 Best preset: {best.upper()}")

    def generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        report = []
        report.append("# Industrial Automation Preset Comparison Report")
        report.append(f"\n**Generated**: {datetime.now().isoformat()}")
        report.append(f"\n**Queries tested**: {len(self.results)}")
        report.append(f"**Presets compared**: {', '.join(p.upper() for p in PRESETS)}")

        # Executive summary
        report.append("\n## Executive Summary\n")

        total_tests = sum(len(qr.results) for qr in self.results)
        successful = sum(1 for qr in self.results for r in qr.results.values() if r.success)
        report.append(f"- **Total tests**: {total_tests}")
        report.append(f"- **Successful**: {successful} ({successful/total_tests*100:.1f}%)")

        # Aggregate by preset
        preset_stats = {p: {"success": 0, "total": 0, "avg_duration": 0, "avg_confidence": 0, "domain_kb_count": 0} for p in PRESETS}
        for qr in self.results:
            for preset, result in qr.results.items():
                preset_stats[preset]["total"] += 1
                if result.success:
                    preset_stats[preset]["success"] += 1
                    preset_stats[preset]["avg_duration"] += result.duration_seconds
                    preset_stats[preset]["avg_confidence"] += result.confidence_score or 0
                    if result.domain_knowledge_cited:
                        preset_stats[preset]["domain_kb_count"] += 1

        report.append("\n### Preset Performance Summary\n")
        report.append("| Preset | Success Rate | Avg Duration | Avg Confidence | Domain KB Usage |")
        report.append("|--------|--------------|--------------|----------------|-----------------|")

        for preset in PRESETS:
            stats = preset_stats[preset]
            if stats["success"] > 0:
                avg_dur = stats["avg_duration"] / stats["success"]
                avg_conf = stats["avg_confidence"] / stats["success"]
                dk_rate = stats["domain_kb_count"] / stats["success"] * 100
            else:
                avg_dur = avg_conf = dk_rate = 0

            success_rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
            report.append(f"| {preset.upper()} | {success_rate:.0f}% | {avg_dur:.1f}s | {avg_conf:.1%} | {dk_rate:.0f}% |")

        # Detailed results per query
        report.append("\n## Detailed Results\n")

        for qr in self.results:
            report.append(f"\n### Query {qr.query_id}: {qr.query_name}\n")
            report.append(f"**Query**: {qr.query_text[:200]}...\n")

            report.append("\n| Preset | Status | Duration | Confidence | Term Coverage | Domain KB | CRAG | Verification |")
            report.append("|--------|--------|----------|------------|---------------|-----------|------|--------------|")

            for preset in PRESETS:
                if preset in qr.results:
                    r = qr.results[preset]
                    status = "✓" if r.success else "✗"
                    duration = f"{r.duration_seconds:.1f}s"
                    confidence = f"{r.confidence_score:.1%}" if r.confidence_score else "N/A"
                    terms = f"{r.term_coverage_pct:.0f}%"
                    domain_kb = "✓" if r.domain_knowledge_cited else "✗"
                    crag = f"{r.crag_quality or 'N/A'}"
                    verify = f"{r.verified_claims}/{r.total_claims}"

                    report.append(f"| {preset.upper()} | {status} | {duration} | {confidence} | {terms} | {domain_kb} | {crag} | {verify} |")

            best = qr.best_preset()
            if best:
                report.append(f"\n**Best preset**: {best.upper()}")

        return "\n".join(report)

    async def run_all(
        self,
        query_ids: Optional[List[int]] = None,
        presets: Optional[List[str]] = None
    ):
        """Run all specified tests"""
        queries = INDUSTRIAL_QUERIES
        if query_ids:
            queries = [q for q in queries if q["id"] in query_ids]

        print(f"\n{'#'*80}")
        print(f"# INDUSTRIAL AUTOMATION PRESET COMPARISON TEST")
        print(f"# Queries: {len(queries)}")
        print(f"# Presets: {presets or PRESETS}")
        print(f"# Started: {datetime.now().isoformat()}")
        print(f"{'#'*80}")

        for query in queries:
            query_results = await self.run_query_all_presets(query, presets)
            self.results.append(query_results)
            self.print_comparison_table(query_results)

            # Save intermediate results
            self.save_results()

            # Pause between queries
            if query != queries[-1]:
                print(f"\n⏳ Waiting 30s before next query...")
                await asyncio.sleep(30)

        # Generate final report
        report = self.generate_report()
        report_file = self.output_dir / "comparison_report.md"
        report_file.write_text(report)
        print(f"\n📄 Report saved to: {report_file}")

        print("\n" + "="*80)
        print("TEST COMPLETE")
        print("="*80)

    def save_results(self):
        """Save current results to JSON"""
        results_file = self.output_dir / "results.json"

        # Convert to serializable format
        data = []
        for qr in self.results:
            qr_dict = {
                "query_id": qr.query_id,
                "query_name": qr.query_name,
                "results": {p: asdict(r) for p, r in qr.results.items()}
            }
            data.append(qr_dict)

        results_file.write_text(json.dumps(data, indent=2))


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Industrial Preset Comparison Tests")
    parser.add_argument("--query", type=int, help="Run only specific query (1-5)")
    parser.add_argument("--preset", type=str, help="Run only specific preset")
    args = parser.parse_args()

    query_ids = [args.query] if args.query else None
    presets = [args.preset] if args.preset else None

    tester = IndustrialPresetTester()
    await tester.run_all(query_ids=query_ids, presets=presets)


if __name__ == "__main__":
    asyncio.run(main())
