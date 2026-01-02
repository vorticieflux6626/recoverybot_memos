#!/usr/bin/env python3
"""
Diverse Agentic Search Test Suite
=================================

Tests the dynamic adaptability of the agentic search pipeline using
randomized query selection from multiple domain categories.

Based on research into agentic RAG architecture best practices:
- Context Engineering (2025): Six-layer context stack model
- ReAct Framework: Reasoning + Acting with observation loops
- ACE Framework: Agentic Context Engineering with 14.8% improvement over baseline
- Multi-Agent Orchestration: Specialized agents with dynamic routing

Test Categories:
- K: Knowledge queries (factual lookups)
- D: Diagnostic queries (troubleshooting)
- P: Procedural queries (step-by-step guidance)
- E: Expert/Comparative queries (complex analysis)
- M: Multi-domain queries (cross-domain synthesis)

Usage:
    python tests/test_agentic_diverse_queries.py [--count N] [--category CAT]
"""

import asyncio
import aiohttp
import json
import random
import time
import argparse
import textwrap
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ============================================================================
# DIVERSE QUERY BANK - Multiple queries per category to avoid cache hits
# ============================================================================

QUERY_BANK = {
    # K: Knowledge queries - factual lookups about specific topics
    "K": [
        ("What does FANUC SRVO-063 alarm mean?", ["overcurrent", "motor", "servo", "alarm"]),
        ("What is the purpose of FANUC DCS safety function?", ["safety", "dcs", "fence", "speed"]),
        ("Explain FANUC R-30iB controller architecture", ["controller", "cpu", "memory", "axis"]),
        ("What are FANUC pulsecoder types and differences?", ["encoder", "pulsecoder", "absolute", "incremental"]),
        ("What does Allen-Bradley fault code F0070 indicate?", ["fault", "drive", "undervoltage", "bus"]),
        ("Explain Siemens S7-1500 CPU memory partitioning", ["memory", "load", "work", "retain"]),
        ("What is the function of FANUC RCAL data?", ["rcal", "calibration", "encoder", "mastering"]),
        ("Define FANUC soft float vs hard float operation", ["soft", "float", "compliance", "force"]),
    ],

    # D: Diagnostic queries - troubleshooting and problem-solving
    "D": [
        ("Robot intermittently loses J1 encoder position after power cycle", ["encoder", "battery", "backup", "mastering"]),
        ("FANUC robot vibrates excessively during high-speed moves", ["vibration", "acceleration", "gain", "servo"]),
        ("Allen-Bradley PLC shows intermittent I/O module faults", ["module", "backplane", "communication", "rack"]),
        ("Servo motor overheating during continuous operation", ["motor", "thermal", "cooling", "current"]),
        ("Robot TCP position drifts over repeated cycles", ["drift", "calibration", "wear", "encoder"]),
        ("Siemens drive trips on F30001 motor overtemperature", ["temperature", "motor", "cooling", "current"]),
        ("FANUC iRVision camera image quality degraded", ["camera", "lens", "lighting", "calibration"]),
        ("Robot arm exhibits backlash in J2 axis", ["backlash", "gear", "reducer", "wear"]),
    ],

    # P: Procedural queries - step-by-step guidance
    "P": [
        ("Step-by-step zero mastering procedure for FANUC M-710iC", ["mastering", "fixture", "calibration", "step"]),
        ("How to backup and restore FANUC robot programs", ["backup", "restore", "usb", "memory"]),
        ("Procedure for replacing FANUC servo amplifier", ["amplifier", "replace", "parameter", "calibration"]),
        ("How to configure FANUC EtherNet/IP scanner", ["ethernet", "scanner", "ip", "connection"]),
        ("Steps to tune FANUC servo gains for heavy payload", ["gain", "tune", "payload", "parameter"]),
        ("Procedure for FANUC collision detection setup", ["collision", "detection", "threshold", "sensitivity"]),
        ("How to set up Allen-Bradley Safe Torque Off (STO)", ["sto", "safety", "hardwired", "function"]),
        ("Steps to configure Siemens ProfiNet IO communication", ["profinet", "io", "device", "configure"]),
    ],

    # E: Expert/Comparative queries - complex analysis requiring synthesis
    "E": [
        ("Compare FANUC R-2000iC vs M-900iB for automotive spot welding", ["payload", "reach", "speed", "cycle"]),
        ("Analyze trade-offs between FANUC and KUKA for arc welding cells", ["fanuc", "kuka", "arc", "welding"]),
        ("Evaluate robot vs cobot for collaborative assembly application", ["cobot", "collaborative", "safety", "payload"]),
        ("Compare Allen-Bradley vs Siemens for high-speed packaging PLC", ["allen-bradley", "siemens", "speed", "io"]),
        ("Analyze predictive maintenance strategies for robot fleets", ["predictive", "maintenance", "condition", "monitoring"]),
        ("Compare vision-guided vs fixed tooling for bin picking", ["vision", "bin", "picking", "cycle"]),
        ("Evaluate edge computing vs cloud for robot analytics", ["edge", "cloud", "latency", "analytics"]),
        ("Compare different robot offline programming software options", ["offline", "programming", "simulation", "path"]),
    ],

    # M: Multi-domain queries - cross-domain synthesis
    "M": [
        ("Integrate FANUC robot with Allen-Bradley safety PLC via CIP Safety", ["fanuc", "allen-bradley", "cip", "safety"]),
        ("Design robot cell with Siemens PLC coordination and FANUC handling", ["siemens", "fanuc", "coordination", "cell"]),
        ("Implement machine vision quality inspection with robot handling", ["vision", "quality", "robot", "inspection"]),
        ("Configure hot runner temperature control integrated with IMM cycle", ["hot runner", "temperature", "imm", "cycle"]),
        ("Design palletizing cell with multiple conveyors and robot coordination", ["palletizing", "conveyor", "robot", "coordination"]),
        ("Integrate FANUC robot with CNC machine for automated part loading", ["fanuc", "cnc", "machine", "loading"]),
        ("Implement safety-rated monitored stop for human-robot interaction", ["safety", "monitored", "stop", "interaction"]),
        ("Design flexible manufacturing cell with quick-change tooling", ["flexible", "manufacturing", "quick", "tooling"]),
    ],
}

# Query type descriptions for reporting
CATEGORY_DESCRIPTIONS = {
    "K": "Knowledge (Factual Lookup)",
    "D": "Diagnostic (Troubleshooting)",
    "P": "Procedural (Step-by-Step)",
    "E": "Expert (Comparative Analysis)",
    "M": "Multi-domain (Cross-domain Synthesis)",
}

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_answer_quality(answer: str, expected_terms: List[str], query: str) -> Dict[str, Any]:
    """
    Evaluate answer quality based on multiple signals.

    Based on research:
    - Term coverage: Are expected concepts mentioned?
    - Structure quality: Is the answer well-organized?
    - Source attribution: Are sources cited?
    - Depth indicators: Is the answer comprehensive?
    """
    answer_lower = answer.lower()

    # Term coverage
    found_terms = [t for t in expected_terms if t.lower() in answer_lower]
    term_coverage = len(found_terms) / len(expected_terms) if expected_terms else 0

    # Structure quality
    has_headers = any(x in answer for x in ['###', '**', '## '])
    has_lists = any(x in answer for x in ['- ', '1.', '* ', '• '])
    has_sections = answer.count('---') >= 1 or answer.count('\n\n') >= 3
    structure_score = (int(has_headers) + int(has_lists) + int(has_sections)) / 3

    # Source attribution
    source_refs = answer.count('[Source') + answer.count('(Source')
    has_sources = source_refs > 0

    # Depth indicators
    word_count = len(answer.split())
    sentence_count = answer.count('.') + answer.count('!') + answer.count('?')
    depth_score = min(1.0, word_count / 300)  # Normalize to 300 words as "deep"

    # Query term overlap
    query_terms = set(query.lower().split())
    answer_terms = set(answer_lower.split())
    query_overlap = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0

    # Actionable content (for procedural queries)
    action_words = ['step', 'first', 'then', 'next', 'finally', 'ensure', 'verify', 'check']
    actionable_count = sum(1 for w in action_words if w in answer_lower)

    return {
        "term_coverage": term_coverage,
        "found_terms": found_terms,
        "missing_terms": [t for t in expected_terms if t.lower() not in answer_lower],
        "structure_score": structure_score,
        "has_headers": has_headers,
        "has_lists": has_lists,
        "has_sources": has_sources,
        "source_count": source_refs,
        "depth_score": depth_score,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "query_overlap": query_overlap,
        "actionable_count": actionable_count,
    }

def calculate_composite_score(eval_result: Dict[str, Any], category: str) -> float:
    """
    Calculate composite quality score with category-specific weighting.

    Different query types have different quality requirements:
    - K (Knowledge): High term coverage, sources important
    - D (Diagnostic): Term coverage, actionable steps
    - P (Procedural): Structure, actionable, depth
    - E (Expert): Depth, structure, sources
    - M (Multi-domain): All factors balanced
    """
    weights = {
        "K": {"term_coverage": 0.35, "structure_score": 0.15, "depth_score": 0.20, "has_sources": 0.30},
        "D": {"term_coverage": 0.30, "structure_score": 0.20, "actionable_count": 0.25, "depth_score": 0.25},
        "P": {"structure_score": 0.30, "actionable_count": 0.30, "depth_score": 0.25, "term_coverage": 0.15},
        "E": {"depth_score": 0.30, "structure_score": 0.25, "term_coverage": 0.25, "has_sources": 0.20},
        "M": {"term_coverage": 0.25, "structure_score": 0.25, "depth_score": 0.25, "has_sources": 0.25},
    }

    w = weights.get(category, weights["M"])

    # Normalize actionable_count to 0-1 (4+ action words = 1.0)
    actionable_norm = min(1.0, eval_result.get("actionable_count", 0) / 4)

    score = (
        w.get("term_coverage", 0) * eval_result.get("term_coverage", 0) +
        w.get("structure_score", 0) * eval_result.get("structure_score", 0) +
        w.get("depth_score", 0) * eval_result.get("depth_score", 0) +
        w.get("has_sources", 0) * (1.0 if eval_result.get("has_sources", False) else 0.0) +
        w.get("actionable_count", 0) * actionable_norm
    )

    return score

# ============================================================================
# TEST EXECUTION
# ============================================================================

async def run_single_query(
    session: aiohttp.ClientSession,
    query: str,
    expected_terms: List[str],
    category: str,
    query_id: str,
    use_fresh: bool = True
) -> Dict[str, Any]:
    """Execute a single query and evaluate the response."""

    start_time = time.time()

    # Add timestamp to query to bypass cache if use_fresh=True
    test_query = query
    if use_fresh:
        timestamp = int(time.time() * 1000) % 100000
        test_query = f"{query} [test-{timestamp}]"

    try:
        async with session.post(
            'http://localhost:8001/api/v1/search/universal',
            json={
                'query': test_query,
                'preset': 'research',
                'force_thinking_model': True,
                'max_iterations': 5,
                'max_sources': 15
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            data = await resp.json()

            if data.get('success'):
                d = data.get('data', {})
                answer = d.get('synthesized_context') or d.get('answer', '')
                confidence = d.get('confidence_score', 0)
                sources = d.get('sources', [])
                trace = d.get('search_trace', [{}])
                meta = data.get('meta', {})

                # Evaluate quality
                eval_result = evaluate_answer_quality(answer, expected_terms, query)
                composite_score = calculate_composite_score(eval_result, category)

                return {
                    'success': True,
                    'query_id': query_id,
                    'category': category,
                    'query': query,  # Original query
                    'confidence': confidence,
                    'source_count': len(sources),
                    'pipeline': trace[0].get('step', 'unknown') if trace else 'unknown',
                    'cache_hit': meta.get('cache_hit', False),
                    'execution_time_ms': meta.get('execution_time_ms', 0),
                    'evaluation': eval_result,
                    'composite_score': composite_score,
                    'answer_preview': answer[:500] if answer else '',
                    'answer_length': len(answer),
                }
            else:
                return {
                    'success': False,
                    'query_id': query_id,
                    'category': category,
                    'query': query,
                    'error': str(data.get('errors', 'Unknown error'))
                }
    except asyncio.TimeoutError:
        return {
            'success': False,
            'query_id': query_id,
            'category': category,
            'query': query,
            'error': 'Timeout (300s)'
        }
    except Exception as e:
        return {
            'success': False,
            'query_id': query_id,
            'category': category,
            'query': query,
            'error': str(e)
        }

async def run_diverse_test(
    count_per_category: int = 2,
    categories: List[str] = None,
    use_fresh: bool = True
) -> List[Dict[str, Any]]:
    """Run diverse test suite with random query selection."""

    if categories is None:
        categories = list(QUERY_BANK.keys())

    # Select random queries from each category
    selected_queries = []
    for cat in categories:
        queries = QUERY_BANK.get(cat, [])
        if queries:
            selected = random.sample(queries, min(count_per_category, len(queries)))
            for i, (query, terms) in enumerate(selected):
                selected_queries.append({
                    'query': query,
                    'terms': terms,
                    'category': cat,
                    'id': f"{cat}{i+1}"
                })

    # Shuffle to avoid predictable ordering
    random.shuffle(selected_queries)

    print(f"\n{'='*70}")
    print(f"DIVERSE AGENTIC SEARCH TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Categories: {', '.join(categories)}")
    print(f"Queries per category: {count_per_category}")
    print(f"Total queries: {len(selected_queries)}")
    print(f"Fresh queries (bypass cache): {use_fresh}")
    print(f"{'='*70}\n")

    results = []
    async with aiohttp.ClientSession() as session:
        for i, q in enumerate(selected_queries, 1):
            print(f"\n[{i}/{len(selected_queries)}] {q['id']}: {q['query'][:60]}...")

            result = await run_single_query(
                session,
                q['query'],
                q['terms'],
                q['category'],
                q['id'],
                use_fresh
            )
            results.append(result)

            if result['success']:
                eval_r = result['evaluation']
                print(f"  ✓ Pipeline: {result['pipeline']}")
                print(f"  ✓ Confidence: {result['confidence']*100:.0f}%")
                print(f"  ✓ Term Coverage: {eval_r['term_coverage']*100:.0f}% ({len(eval_r['found_terms'])}/{len(q['terms'])})")
                print(f"  ✓ Composite Score: {result['composite_score']*100:.0f}%")
                print(f"  ✓ Answer: {result['answer_length']} chars")
                print(f"  ✓ Time: {result['execution_time_ms']/1000:.1f}s")
            else:
                print(f"  ✗ ERROR: {result.get('error', 'Unknown')}")

    return results

def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print comprehensive test summary."""

    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")

    # Overall stats
    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    print(f"\nOverall: {len(successful)}/{len(results)} successful ({len(successful)/len(results)*100:.0f}%)")

    if failed:
        print(f"\nFailed queries:")
        for r in failed:
            print(f"  - {r['query_id']}: {r.get('error', 'Unknown error')}")

    if not successful:
        return

    # Per-category breakdown
    print(f"\n{'Category':<12} {'Success':<10} {'Conf%':<8} {'Terms%':<8} {'Score%':<8} {'Avg Time'}")
    print("-" * 70)

    for cat in sorted(set(r['category'] for r in results)):
        cat_results = [r for r in successful if r['category'] == cat]
        if not cat_results:
            continue

        cat_all = [r for r in results if r['category'] == cat]
        success_rate = len(cat_results) / len(cat_all)
        avg_conf = sum(r['confidence'] for r in cat_results) / len(cat_results)
        avg_terms = sum(r['evaluation']['term_coverage'] for r in cat_results) / len(cat_results)
        avg_score = sum(r['composite_score'] for r in cat_results) / len(cat_results)
        avg_time = sum(r['execution_time_ms'] for r in cat_results) / len(cat_results)

        print(f"{cat} ({CATEGORY_DESCRIPTIONS[cat][:20]})")
        print(f"  {' '*10} {success_rate*100:>5.0f}%     {avg_conf*100:>5.0f}%    {avg_terms*100:>5.0f}%    {avg_score*100:>5.0f}%    {avg_time/1000:>5.1f}s")

    # Overall averages
    print("-" * 70)
    overall_conf = sum(r['confidence'] for r in successful) / len(successful)
    overall_terms = sum(r['evaluation']['term_coverage'] for r in successful) / len(successful)
    overall_score = sum(r['composite_score'] for r in successful) / len(successful)
    overall_time = sum(r['execution_time_ms'] for r in successful) / len(successful)

    print(f"{'OVERALL':<12} {len(successful)/len(results)*100:>5.0f}%     {overall_conf*100:>5.0f}%    {overall_terms*100:>5.0f}%    {overall_score*100:>5.0f}%    {overall_time/1000:>5.1f}s")

    # Pipeline distribution
    print(f"\nPipeline Distribution:")
    pipelines = {}
    for r in successful:
        p = r.get('pipeline', 'unknown')
        pipelines[p] = pipelines.get(p, 0) + 1
    for p, count in sorted(pipelines.items(), key=lambda x: -x[1]):
        print(f"  {p}: {count} ({count/len(successful)*100:.0f}%)")

    # Cache hit analysis
    cache_hits = sum(1 for r in successful if r.get('cache_hit'))
    print(f"\nCache Analysis:")
    print(f"  Cache hits: {cache_hits}/{len(successful)} ({cache_hits/len(successful)*100:.0f}%)")

    # Quality insights
    print(f"\nQuality Insights:")
    low_conf = [r for r in successful if r['confidence'] < 0.5]
    low_terms = [r for r in successful if r['evaluation']['term_coverage'] < 0.5]
    no_sources = [r for r in successful if not r['evaluation']['has_sources']]

    print(f"  Low confidence (<50%): {len(low_conf)}")
    print(f"  Low term coverage (<50%): {len(low_terms)}")
    print(f"  Missing source citations: {len(no_sources)}")

    if low_conf:
        print(f"\n  Low confidence queries:")
        for r in low_conf[:3]:
            print(f"    - {r['query_id']}: {r['confidence']*100:.0f}%")

def main():
    parser = argparse.ArgumentParser(description='Diverse Agentic Search Test Suite')
    parser.add_argument('--count', '-n', type=int, default=2,
                        help='Number of queries per category (default: 2)')
    parser.add_argument('--category', '-c', type=str, default=None,
                        help='Specific category to test (K/D/P/E/M), default: all')
    parser.add_argument('--cached', action='store_true',
                        help='Allow cached results (default: fresh queries)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for detailed results')

    args = parser.parse_args()

    categories = None
    if args.category:
        categories = [c.upper() for c in args.category.split(',')]
        invalid = [c for c in categories if c not in QUERY_BANK]
        if invalid:
            print(f"Invalid categories: {invalid}")
            print(f"Valid categories: {list(QUERY_BANK.keys())}")
            return

    results = asyncio.run(run_diverse_test(
        count_per_category=args.count,
        categories=categories,
        use_fresh=not args.cached
    ))

    print_summary(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'count_per_category': args.count,
                    'categories': categories or list(QUERY_BANK.keys()),
                    'use_fresh': not args.cached
                },
                'results': results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == '__main__':
    main()
