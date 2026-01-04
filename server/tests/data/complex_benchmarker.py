#!/usr/bin/env python3
"""
Complex Synthesis Benchmarker - Test models on expert-crafted contexts.

Uses expected analysis points for accurate scoring.
"""

import asyncio
import aiohttp
import json
import sqlite3
import time
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# Top models to compare
MODELS_TO_TEST = [
    "ministral-3:3b",
    "ministral-3:8b",
    "ministral-3:14b",
    "qwen3:8b",
    "deepseek-r1:14b-qwen-distill-q8_0",
]


class SynthesisDB:
    """Database interface."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "synthesis_contexts.db"
        self.db_path = str(db_path)

    def get_complex_contexts(self) -> list:
        """Get all complex_ prefixed contexts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT * FROM synthesis_contexts
            WHERE context_id LIKE 'complex_%'
            ORDER BY context_id
        """)
        rows = cur.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def store_benchmark(self, result: dict) -> bool:
        """Store benchmark result."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO synthesis_benchmarks
                (context_id, model, synthesis_output, duration_ms, ttfs_ms, output_tokens,
                 thinking_tokens, vram_before_mb, vram_after_mb, topic_coverage_score,
                 factual_density_score, coherence_score, overall_score, success,
                 error_message, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result['context_id'],
                result['model'],
                result.get('synthesis_output'),
                result.get('duration_ms'),
                result.get('ttfs_ms'),
                result.get('output_tokens'),
                result.get('thinking_tokens'),
                result.get('vram_before_mb'),
                result.get('vram_after_mb'),
                result.get('topic_coverage_score'),
                result.get('factual_density_score'),
                result.get('coherence_score'),
                result.get('overall_score'),
                1 if result.get('success') else 0,
                result.get('error_message'),
                result['timestamp']
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing benchmark: {e}")
            return False
        finally:
            conn.close()

    def get_complex_rankings(self) -> list:
        """Get rankings for complex contexts only."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT
                model,
                COUNT(*) as test_count,
                ROUND(AVG(duration_ms), 0) as avg_duration_ms,
                ROUND(AVG(ttfs_ms), 0) as avg_ttfs_ms,
                ROUND(AVG(output_tokens), 0) as avg_tokens,
                ROUND(AVG(topic_coverage_score), 3) as avg_analysis_score,
                ROUND(AVG(factual_density_score), 3) as avg_factual_score,
                ROUND(AVG(coherence_score), 3) as avg_coherence_score,
                ROUND(AVG(overall_score), 3) as avg_overall_score,
                SUM(success) as successes
            FROM synthesis_benchmarks
            WHERE context_id LIKE 'complex_%'
            GROUP BY model
            ORDER BY avg_overall_score DESC
        """)
        rows = cur.fetchall()
        conn.close()

        return [dict(row) for row in rows]


def get_vram_mb() -> int:
    """Get current VRAM usage in MB."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except:
        return 0


def unload_all_models():
    """Unload all Ollama models."""
    try:
        result = subprocess.run(
            ['curl', '-s', 'http://localhost:11434/api/ps'],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        for m in data.get('models', []):
            name = m.get('name', '')
            if name:
                subprocess.run([
                    'curl', '-s', '-X', 'POST', 'http://localhost:11434/api/generate',
                    '-d', json.dumps({'model': name, 'keep_alive': 0})
                ], capture_output=True)
    except:
        pass


def calculate_analysis_coverage(output: str, expected_points: list) -> tuple:
    """
    Calculate how many expected analysis points are covered.
    Returns (score, matched_points, missed_points)
    """
    if not expected_points:
        return 0.5, [], []

    output_lower = output.lower()
    matched = []
    missed = []

    for point in expected_points:
        # Check if key concepts from the point are present
        point_lower = point.lower()
        key_terms = [t.strip() for t in point_lower.split() if len(t.strip()) > 3]

        # Count how many key terms appear
        matches = sum(1 for term in key_terms if term in output_lower)
        coverage = matches / len(key_terms) if key_terms else 0

        if coverage >= 0.5:  # At least half the terms present
            matched.append(point)
        else:
            missed.append(point)

    score = len(matched) / len(expected_points)
    return score, matched, missed


def calculate_technical_depth(output: str) -> float:
    """Calculate technical depth of response."""
    indicators = 0

    # Technical patterns
    patterns = [
        r'\d+\s*(psi|bar|°[CF]|Hz|kHz|mm|µs|ms|cSt)',  # Measurements with units
        r'[A-Z]{2,}-\d+',  # Error codes
        r'\$[A-Z_]+\[',  # System variables
        r'step\s+\d+',  # Procedure steps
        r'cause[sd]?\s*:',  # Cause analysis
        r'solution\s*:',  # Solutions
        r'check\s+(the|for|if)',  # Diagnostic actions
        r'(increase|decrease|adjust|configure|set)',  # Action verbs
        r'because|therefore|due to|results in',  # Causal language
        r'(first|second|third|finally|then)',  # Sequential reasoning
    ]

    for pattern in patterns:
        matches = len(re.findall(pattern, output, re.IGNORECASE))
        indicators += min(matches, 3)  # Cap each pattern contribution

    # Normalize to 0-1 (expect 15-30 indicators in good response)
    return min(1.0, indicators / 20.0)


def calculate_reasoning_quality(output: str) -> float:
    """Evaluate reasoning chain quality."""
    score = 0.5  # Base

    # Look for reasoning indicators
    if re.search(r'root cause|underlying|fundamental', output, re.I):
        score += 0.1
    if re.search(r'led to|caused|resulted in|triggered', output, re.I):
        score += 0.1
    if re.search(r'recommend|suggest|should|must', output, re.I):
        score += 0.1
    if re.search(r'verify|confirm|check|test', output, re.I):
        score += 0.1
    if re.search(r'(if|when).*then', output, re.I):
        score += 0.1

    # Penalize if too short
    if len(output) < 500:
        score -= 0.2
    elif len(output) > 1500:
        score += 0.1

    return min(1.0, max(0.0, score))


async def synthesize_with_model(model: str, query: str, context: str) -> dict:
    """Run synthesis with a model."""
    vram_before = get_vram_mb()
    start_time = time.time()
    ttfs = None

    prompt = f"""You are an expert industrial automation engineer performing root cause analysis.

SCENARIO:
{query}

TECHNICAL REFERENCE:
{context}

Provide a thorough analysis that:
1. Identifies the root cause of the problem
2. Explains the causal chain of events
3. References specific technical details from the context
4. Recommends corrective actions with priority

ANALYSIS:"""

    async with aiohttp.ClientSession() as session:
        try:
            chunks = []
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"num_predict": 2048, "temperature": 0.7}
                },
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            if 'response' in data:
                                if ttfs is None:
                                    ttfs = (time.time() - start_time) * 1000
                                chunks.append(data['response'])
                        except:
                            pass

            duration_ms = (time.time() - start_time) * 1000
            output = ''.join(chunks)

            return {
                "success": True,
                "output": output,
                "duration_ms": duration_ms,
                "ttfs_ms": ttfs,
                "output_tokens": len(output.split()),
                "vram_before_mb": vram_before,
                "vram_after_mb": get_vram_mb()
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def benchmark_model(model: str, context: dict, unload: bool = True) -> dict:
    """Benchmark a model on a context."""
    if unload:
        unload_all_models()
        await asyncio.sleep(2)

    # Parse expected analysis points
    expected_points = []
    try:
        expected_points = json.loads(context.get('expected_topics', '[]'))
    except:
        pass

    result = await synthesize_with_model(
        model=model,
        query=context['query'],
        context=context['retrieved_context']
    )

    if not result['success']:
        return {
            'context_id': context['context_id'],
            'model': model,
            'success': False,
            'error_message': result.get('error'),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    output = result['output']

    # Calculate scores
    analysis_score, matched, missed = calculate_analysis_coverage(output, expected_points)
    technical_score = calculate_technical_depth(output)
    reasoning_score = calculate_reasoning_quality(output)

    # Weighted overall (analysis coverage is most important for these tests)
    overall = (
        analysis_score * 0.50 +
        technical_score * 0.30 +
        reasoning_score * 0.20
    )

    return {
        'context_id': context['context_id'],
        'model': model,
        'synthesis_output': output[:15000],
        'duration_ms': result['duration_ms'],
        'ttfs_ms': result.get('ttfs_ms'),
        'output_tokens': result['output_tokens'],
        'thinking_tokens': 0,
        'vram_before_mb': result['vram_before_mb'],
        'vram_after_mb': result['vram_after_mb'],
        'topic_coverage_score': analysis_score,  # Repurposed for analysis coverage
        'factual_density_score': technical_score,
        'coherence_score': reasoning_score,
        'overall_score': overall,
        'success': True,
        'matched_points': matched,
        'missed_points': missed,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


def print_rankings(db: SynthesisDB):
    """Print current complex context rankings."""
    rankings = db.get_complex_rankings()

    if not rankings:
        print("\nNo complex benchmark results yet.")
        return

    print(f"\n{'='*110}")
    print("COMPLEX SYNTHESIS RANKINGS (Expert-Crafted Contexts)")
    print(f"{'='*110}")
    print(f"{'Rank':<5} {'Model':<35} {'Duration':<12} {'TTFS':<10} {'Tokens':<8} {'Analysis':<10} {'Technical':<10} {'Reasoning':<10} {'Overall':<8}")
    print("-" * 110)

    for i, r in enumerate(rankings, 1):
        print(f"{i:<5} {r['model']:<35} {r['avg_duration_ms']:.0f}ms{'':<5} "
              f"{r['avg_ttfs_ms'] or 0:.0f}ms{'':<4} "
              f"{r['avg_tokens']:.0f}{'':<4} "
              f"{r['avg_analysis_score']:.3f}{'':<5} "
              f"{r['avg_factual_score']:.3f}{'':<5} "
              f"{r['avg_coherence_score']:.3f}{'':<5} "
              f"{r['avg_overall_score']:.3f}")


async def run_benchmarks(models: list = None, max_contexts: int = 6):
    """Run benchmarks on complex contexts."""
    db = SynthesisDB()

    if models is None:
        models = MODELS_TO_TEST

    contexts = db.get_complex_contexts()[:max_contexts]

    print(f"Benchmarking {len(models)} models on {len(contexts)} complex contexts")

    for ctx in contexts:
        # Parse reasoning type from domains field
        domains = json.loads(ctx.get('domains', '[]'))
        reasoning_type = domains[0] if domains else 'unknown'

        print(f"\n{'='*80}")
        print(f"Context: {ctx['context_id']}")
        print(f"Reasoning Type: {reasoning_type}")
        print(f"Query: {ctx['query'][:70]}...")
        print(f"{'='*80}")

        for model in models:
            print(f"\n--- {model} ---")

            result = await benchmark_model(model, ctx)

            if result['success']:
                db.store_benchmark(result)
                print(f"  Duration: {result['duration_ms']:.0f}ms | TTFS: {result.get('ttfs_ms', 0):.0f}ms")
                print(f"  Tokens: {result['output_tokens']}")
                print(f"  Analysis Coverage: {result['topic_coverage_score']:.1%} ({len(result.get('matched_points', []))}/{len(result.get('matched_points', [])) + len(result.get('missed_points', []))} points)")
                print(f"  Technical Depth: {result['factual_density_score']:.2f}")
                print(f"  Reasoning Quality: {result['coherence_score']:.2f}")
                print(f"  Overall: {result['overall_score']:.3f}")

                if result.get('missed_points'):
                    print(f"  Missed: {result['missed_points'][:2]}...")
            else:
                print(f"  FAILED: {result.get('error_message')}")

            await asyncio.sleep(2)

    print_rankings(db)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Complex synthesis benchmarker")
    parser.add_argument("--models", type=str, help="Comma-separated models")
    parser.add_argument("--max-contexts", type=int, default=6, help="Max contexts")
    parser.add_argument("--rankings", action="store_true", help="Show rankings only")
    args = parser.parse_args()

    db = SynthesisDB()

    if args.rankings:
        print_rankings(db)
        return

    models = MODELS_TO_TEST
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    await run_benchmarks(models=models, max_contexts=args.max_contexts)


if __name__ == "__main__":
    asyncio.run(main())
