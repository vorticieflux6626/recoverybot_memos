#!/usr/bin/env python3
"""
Answer Effectiveness Test Suite
================================

Tests the effectiveness of the agentic pipeline's answers against the questions asked.
Measures multiple dimensions of answer quality.

Metrics:
1. Question-Answer Alignment: Does the answer address the question?
2. Term Coverage: Are expected domain terms present?
3. Citation Accuracy: Are claims properly cited?
4. Confidence Calibration: Does confidence match actual quality?
5. Completeness: Are all aspects of the question covered?

Usage:
    python tests/test_answer_effectiveness.py [--category CAT] [--verbose]
"""

import asyncio
import aiohttp
import json
import time
import argparse
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


# ============================================================================
# EFFECTIVENESS METRICS
# ============================================================================

@dataclass
class EffectivenessMetrics:
    """Comprehensive effectiveness metrics for an answer."""

    # Core metrics (0-1)
    question_alignment: float      # Semantic alignment Q→A
    term_coverage: float           # Expected terms found
    citation_accuracy: float       # Claims with valid citations
    confidence_calibration: float  # Confidence vs actual quality
    completeness: float            # Question aspects covered

    # Derived
    effectiveness_score: float     # Weighted composite

    # Details
    found_terms: List[str]
    missing_terms: List[str]
    citation_count: int
    question_aspects_covered: List[str]
    question_aspects_missing: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'question_alignment': self.question_alignment,
            'term_coverage': self.term_coverage,
            'citation_accuracy': self.citation_accuracy,
            'confidence_calibration': self.confidence_calibration,
            'completeness': self.completeness,
            'effectiveness_score': self.effectiveness_score,
            'found_terms': self.found_terms,
            'missing_terms': self.missing_terms,
            'citation_count': self.citation_count,
            'aspects_covered': self.question_aspects_covered,
            'aspects_missing': self.question_aspects_missing,
        }


# ============================================================================
# TEST QUERY BANK
# ============================================================================

# Each test case: (question, expected_terms, question_aspects, quality_threshold)
EFFECTIVENESS_TESTS: Dict[str, List[Tuple[str, List[str], List[str], float]]] = {

    "factual": [
        (
            "What does FANUC SRVO-063 alarm mean?",
            ["overcurrent", "servo", "motor", "alarm", "axis"],
            ["error_meaning", "cause", "affected_component"],
            0.7
        ),
        (
            "What is the payload capacity of FANUC M-20iD/25?",
            ["25kg", "payload", "robot", "capacity"],
            ["specification", "value", "model"],
            0.8
        ),
    ],

    "diagnostic": [
        (
            "Why does the robot lose position after power cycle?",
            ["encoder", "battery", "backup", "position", "mastering"],
            ["cause", "symptoms", "affected_components", "solution"],
            0.6
        ),
        (
            "What causes intermittent SRVO alarms during operation?",
            ["cable", "connection", "servo", "intermittent", "wiring"],
            ["possible_causes", "diagnostic_steps", "verification"],
            0.6
        ),
    ],

    "procedural": [
        (
            "How do I perform zero mastering on a FANUC robot?",
            ["mastering", "zero", "position", "jog", "fixture"],
            ["prerequisites", "steps", "verification", "warnings"],
            0.7
        ),
        (
            "What is the procedure to backup robot programs?",
            ["backup", "program", "teach", "pendant", "USB", "save"],
            ["steps", "file_types", "storage"],
            0.7
        ),
    ],

    "comparative": [
        (
            "Compare FANUC R-30iA vs R-30iB controller features",
            ["R-30iA", "R-30iB", "controller", "features", "performance"],
            ["differences", "improvements", "compatibility"],
            0.6
        ),
    ],

    "troubleshooting": [
        (
            "Robot servo motor is overheating, what should I check?",
            ["motor", "overheating", "temperature", "load", "cooling", "duty"],
            ["diagnosis", "causes", "solutions", "prevention"],
            0.6
        ),
    ],
}


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def extract_citations(text: str) -> List[str]:
    """Extract [Source N] citations from text."""
    pattern = r'\[Source\s*(\d+)\]'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(set(matches))


def count_question_aspects_in_answer(answer: str, aspects: List[str]) -> Tuple[List[str], List[str]]:
    """Check which question aspects are addressed in the answer."""
    answer_lower = answer.lower()

    # Aspect keyword mapping
    aspect_keywords = {
        "error_meaning": ["means", "indicates", "refers to", "definition", "occurs when"],
        "cause": ["cause", "because", "due to", "result of", "when"],
        "affected_component": ["motor", "servo", "axis", "encoder", "amplifier"],
        "symptoms": ["symptom", "observe", "notice", "appears", "shows"],
        "solution": ["fix", "resolve", "solution", "repair", "replace", "adjust"],
        "diagnostic_steps": ["check", "verify", "inspect", "test", "measure"],
        "verification": ["confirm", "verify", "ensure", "validate"],
        "prerequisites": ["before", "first", "ensure", "required", "need"],
        "steps": ["step", "then", "next", "after", "1.", "2.", "3."],
        "warnings": ["warning", "caution", "danger", "do not", "important"],
        "differences": ["difference", "unlike", "compared to", "whereas", "vs"],
        "improvements": ["improve", "better", "enhanced", "upgraded", "new"],
        "compatibility": ["compatible", "support", "work with"],
        "specification": ["spec", "rated", "maximum", "minimum", "kg", "mm"],
        "value": ["25", "kg", "mm", "degrees", "%"],
        "model": ["model", "M-20", "R-30", "FANUC"],
        "possible_causes": ["possible", "might", "could", "may"],
        "diagnosis": ["diagnose", "identify", "determine"],
        "causes": ["cause", "reason", "why"],
        "solutions": ["solution", "fix", "resolve"],
        "prevention": ["prevent", "avoid", "maintain"],
        "file_types": ["file", ".tp", ".ls", "image", "backup"],
        "storage": ["usb", "storage", "save", "memory"],
    }

    covered = []
    missing = []

    for aspect in aspects:
        keywords = aspect_keywords.get(aspect, [aspect])
        if any(kw.lower() in answer_lower for kw in keywords):
            covered.append(aspect)
        else:
            missing.append(aspect)

    return covered, missing


def calculate_effectiveness(
    answer: str,
    confidence: float,
    expected_terms: List[str],
    question_aspects: List[str],
    source_count: int = 0
) -> EffectivenessMetrics:
    """Calculate comprehensive effectiveness metrics."""

    answer_lower = answer.lower()

    # 1. Term Coverage
    found_terms = [t for t in expected_terms if t.lower() in answer_lower]
    missing_terms = [t for t in expected_terms if t.lower() not in answer_lower]
    term_coverage = len(found_terms) / len(expected_terms) if expected_terms else 1.0

    # 2. Citation Accuracy
    citations = extract_citations(answer)
    citation_count = len(citations)
    # Expect at least 1 citation per 200 chars of substantial content
    expected_citations = max(1, len(answer) // 300)
    citation_accuracy = min(1.0, citation_count / expected_citations) if answer else 0.0

    # 3. Completeness (question aspects covered)
    covered, missing = count_question_aspects_in_answer(answer, question_aspects)
    completeness = len(covered) / len(question_aspects) if question_aspects else 1.0

    # 4. Question Alignment (heuristic based on structure and length)
    # Better answers tend to be structured and substantive
    has_structure = any(marker in answer for marker in ["**", "1.", "2.", "-", "###"])
    is_substantive = len(answer) > 200
    is_not_generic = "I don't have" not in answer and "No information" not in answer.lower()
    alignment_score = 0.0
    if is_substantive:
        alignment_score += 0.4
    if has_structure:
        alignment_score += 0.3
    if is_not_generic:
        alignment_score += 0.3
    question_alignment = alignment_score

    # 5. Confidence Calibration
    # Compare reported confidence to actual quality indicators
    actual_quality = (term_coverage + completeness + citation_accuracy) / 3
    calibration_diff = abs(confidence - actual_quality)
    confidence_calibration = max(0, 1.0 - calibration_diff)

    # Weighted composite score
    # Weights: alignment 0.15, terms 0.25, citations 0.15, calibration 0.10, completeness 0.35
    effectiveness_score = (
        0.15 * question_alignment +
        0.25 * term_coverage +
        0.15 * citation_accuracy +
        0.10 * confidence_calibration +
        0.35 * completeness
    )

    return EffectivenessMetrics(
        question_alignment=question_alignment,
        term_coverage=term_coverage,
        citation_accuracy=citation_accuracy,
        confidence_calibration=confidence_calibration,
        completeness=completeness,
        effectiveness_score=effectiveness_score,
        found_terms=found_terms,
        missing_terms=missing_terms,
        citation_count=citation_count,
        question_aspects_covered=covered,
        question_aspects_missing=missing,
    )


# ============================================================================
# TEST EXECUTION
# ============================================================================

async def run_effectiveness_test(
    session: aiohttp.ClientSession,
    question: str,
    expected_terms: List[str],
    question_aspects: List[str],
    threshold: float,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run a single effectiveness test."""

    # Add timestamp to avoid cache
    timestamp = int(time.time() * 1000) % 100000
    test_query = f"{question} [effectiveness-test-{timestamp}]"

    start_time = time.time()

    try:
        async with session.post(
            'http://localhost:8001/api/v1/search/universal',
            json={
                'query': test_query,
                'preset': 'research',
                'max_iterations': 5,
                'max_sources': 15
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            data = await resp.json()
            elapsed = time.time() - start_time

            if data.get('success'):
                d = data.get('data', {})
                answer = d.get('synthesized_context') or d.get('answer', '')
                confidence = d.get('confidence_score', 0)
                sources = d.get('sources', [])

                # Calculate effectiveness
                metrics = calculate_effectiveness(
                    answer=answer,
                    confidence=confidence,
                    expected_terms=expected_terms,
                    question_aspects=question_aspects,
                    source_count=len(sources)
                )

                passed = metrics.effectiveness_score >= threshold

                if verbose:
                    status = "PASS" if passed else "FAIL"
                    print(f"\n  [{status}] Effectiveness: {metrics.effectiveness_score*100:.0f}% (threshold: {threshold*100:.0f}%)")
                    print(f"    - Question Alignment: {metrics.question_alignment*100:.0f}%")
                    print(f"    - Term Coverage: {metrics.term_coverage*100:.0f}% ({len(metrics.found_terms)}/{len(expected_terms)})")
                    print(f"    - Citation Accuracy: {metrics.citation_accuracy*100:.0f}% ({metrics.citation_count} citations)")
                    print(f"    - Completeness: {metrics.completeness*100:.0f}% ({len(metrics.question_aspects_covered)}/{len(question_aspects)})")
                    print(f"    - Confidence Calibration: {metrics.confidence_calibration*100:.0f}%")
                    if metrics.missing_terms:
                        print(f"    - Missing Terms: {metrics.missing_terms}")
                    if metrics.question_aspects_missing:
                        print(f"    - Missing Aspects: {metrics.question_aspects_missing}")

                return {
                    'success': True,
                    'passed': passed,
                    'question': question,
                    'metrics': metrics.to_dict(),
                    'confidence': confidence,
                    'source_count': len(sources),
                    'answer_length': len(answer),
                    'execution_time': elapsed,
                    'threshold': threshold,
                }
            else:
                if verbose:
                    print(f"\n  [ERROR] API Error: {data.get('errors')}")
                return {
                    'success': False,
                    'passed': False,
                    'question': question,
                    'error': str(data.get('errors')),
                }

    except asyncio.TimeoutError:
        if verbose:
            print(f"\n  [ERROR] Timeout after 300s")
        return {
            'success': False,
            'passed': False,
            'question': question,
            'error': 'Timeout (300s)',
        }
    except Exception as e:
        if verbose:
            print(f"\n  [ERROR] {e}")
        return {
            'success': False,
            'passed': False,
            'question': question,
            'error': str(e),
        }


async def run_effectiveness_suite(
    categories: Optional[List[str]] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run the full effectiveness test suite."""

    if categories is None:
        categories = list(EFFECTIVENESS_TESTS.keys())

    print(f"\n{'='*70}")
    print(f"ANSWER EFFECTIVENESS TEST SUITE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Categories: {', '.join(categories)}")
    print(f"{'='*70}")

    results = []

    async with aiohttp.ClientSession() as session:
        test_num = 0
        total_tests = sum(len(EFFECTIVENESS_TESTS.get(c, [])) for c in categories)

        for category in categories:
            tests = EFFECTIVENESS_TESTS.get(category, [])
            if not tests:
                continue

            print(f"\n## Category: {category.upper()}")

            for question, terms, aspects, threshold in tests:
                test_num += 1
                print(f"\n[{test_num}/{total_tests}] {question[:60]}...")

                result = await run_effectiveness_test(
                    session=session,
                    question=question,
                    expected_terms=terms,
                    question_aspects=aspects,
                    threshold=threshold,
                    verbose=verbose
                )
                result['category'] = category
                results.append(result)

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print test summary."""

    print(f"\n{'='*70}")
    print("EFFECTIVENESS TEST SUMMARY")
    print(f"{'='*70}")

    successful = [r for r in results if r.get('success')]
    passed = [r for r in results if r.get('passed')]
    failed = [r for r in successful if not r.get('passed')]

    print(f"\nTotal Tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Passed: {len(passed)}/{len(successful)} ({len(passed)/len(successful)*100:.0f}%)" if successful else "N/A")

    if successful:
        # Aggregate metrics
        avg_metrics = {}
        metrics_keys = ['question_alignment', 'term_coverage', 'citation_accuracy',
                       'confidence_calibration', 'completeness', 'effectiveness_score']
        for key in metrics_keys:
            values = [r['metrics'][key] for r in successful if 'metrics' in r]
            avg_metrics[key] = sum(values) / len(values) if values else 0

        print(f"\nAverage Metrics:")
        print(f"  Effectiveness Score: {avg_metrics['effectiveness_score']*100:.0f}%")
        print(f"  Question Alignment:  {avg_metrics['question_alignment']*100:.0f}%")
        print(f"  Term Coverage:       {avg_metrics['term_coverage']*100:.0f}%")
        print(f"  Citation Accuracy:   {avg_metrics['citation_accuracy']*100:.0f}%")
        print(f"  Completeness:        {avg_metrics['completeness']*100:.0f}%")
        print(f"  Conf. Calibration:   {avg_metrics['confidence_calibration']*100:.0f}%")

    if failed:
        print(f"\nFailed Tests ({len(failed)}):")
        for r in failed:
            eff = r.get('metrics', {}).get('effectiveness_score', 0)
            thr = r.get('threshold', 0)
            print(f"  - {r['question'][:50]}... ({eff*100:.0f}% < {thr*100:.0f}%)")

    # Per-category breakdown
    categories = set(r.get('category') for r in results)
    print(f"\nPer-Category Results:")
    print(f"{'Category':<15} {'Pass':<8} {'Eff%':<8} {'Terms%':<8} {'Cite%':<8}")
    print("-" * 50)

    for cat in sorted(categories):
        cat_results = [r for r in successful if r.get('category') == cat]
        if cat_results:
            cat_passed = len([r for r in cat_results if r.get('passed')])
            avg_eff = sum(r['metrics']['effectiveness_score'] for r in cat_results) / len(cat_results)
            avg_terms = sum(r['metrics']['term_coverage'] for r in cat_results) / len(cat_results)
            avg_cite = sum(r['metrics']['citation_accuracy'] for r in cat_results) / len(cat_results)
            print(f"{cat:<15} {cat_passed}/{len(cat_results):<5} {avg_eff*100:>5.0f}%   {avg_terms*100:>5.0f}%   {avg_cite*100:>5.0f}%")


def main():
    parser = argparse.ArgumentParser(description='Answer Effectiveness Test Suite')
    parser.add_argument('--category', '-c', type=str, default=None,
                        help='Specific category to test')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    categories = [args.category] if args.category else None

    results = asyncio.run(run_effectiveness_suite(
        categories=categories,
        verbose=not args.quiet
    ))

    print_summary(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
