#!/usr/bin/env python3
"""
Synthesis Model Benchmarker - Test model effectiveness on stored contexts.

Benchmarks different models on synthesis tasks using stored contexts,
measuring quality metrics like topic coverage, factual density, and coherence.
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
from typing import Optional

# Models to benchmark (in order of expected capability)
SYNTHESIS_MODELS = [
    # Fast models
    "gemma3:4b",
    "qwen3:8b",
    "ministral-3:3b",

    # Medium models
    "gemma3:12b",

    # Thinking models
    "cogito:8b",
    "deepseek-r1:14b-qwen-distill-q8_0",
]


class SynthesisContextDB:
    """Database for synthesis contexts and benchmarks."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "synthesis_contexts.db"
        self.db_path = str(db_path)

    def get_all_contexts(self) -> list:
        """Get all stored contexts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM synthesis_contexts ORDER BY created_at DESC")
        rows = cur.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def store_benchmark(self, result: dict) -> bool:
        """Store a synthesis benchmark result."""
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

    def get_rankings(self) -> list:
        """Get benchmark rankings by model."""
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
                ROUND(AVG(topic_coverage_score), 2) as avg_topic_score,
                ROUND(AVG(factual_density_score), 2) as avg_factual_score,
                ROUND(AVG(coherence_score), 2) as avg_coherence_score,
                ROUND(AVG(overall_score), 2) as avg_overall_score,
                SUM(success) as successes
            FROM synthesis_benchmarks
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
        # Get running models
        result = subprocess.run(
            ['curl', '-s', 'http://localhost:11434/api/ps'],
            capture_output=True, text=True
        )
        data = json.loads(result.stdout)
        models = data.get('models', [])

        for m in models:
            name = m.get('name', '')
            if name:
                subprocess.run([
                    'curl', '-s', '-X', 'POST', 'http://localhost:11434/api/generate',
                    '-d', json.dumps({'model': name, 'keep_alive': 0})
                ], capture_output=True)
    except:
        pass


def count_thinking_tokens(text: str) -> int:
    """Count tokens in thinking sections."""
    patterns = [
        r'<think>(.*?)</think>',
        r'<thinking>(.*?)</thinking>',
        r'<\|thinking\|>(.*?)<\|/thinking\|>',
    ]

    total = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            # Rough token count: words / 0.75
            total += int(len(match.split()) / 0.75)

    return total


def calculate_topic_coverage(output: str, expected_topics: list) -> float:
    """Calculate what percentage of expected topics are covered."""
    if not expected_topics:
        return 0.5  # Neutral if no expected topics

    output_lower = output.lower()
    covered = sum(1 for topic in expected_topics if topic.lower() in output_lower)
    return covered / len(expected_topics)


def calculate_factual_density(output: str) -> float:
    """Estimate factual density (specific numbers, terms, procedures)."""
    # Count specific patterns that indicate factual content
    patterns = [
        r'\d+\.?\d*\s*(mm|cm|°C|°F|Hz|kHz|MHz|GB|MB|KB|ms|sec|min|V|A|W|ohm|psi|bar)',  # Measurements
        r'\b[A-Z]{2,}-\d+',  # Error codes like SRVO-023
        r'\bstep\s+\d+',  # Procedure steps
        r'\bparameter\s+\d+',  # Parameters
        r'\bport\s+\d+',  # Ports
        r'\baddress\s+\d+',  # Addresses
        r'\bfault\s+(code\s+)?\d+',  # Fault codes
    ]

    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, output, re.IGNORECASE))

    # Normalize to 0-1 scale (expect 5-20 facts in good response)
    return min(1.0, count / 10.0)


def calculate_coherence(output: str) -> float:
    """Estimate coherence based on structure and flow."""
    score = 0.5  # Base score

    # Check for structured elements
    if re.search(r'^\d+\.', output, re.MULTILINE):  # Numbered lists
        score += 0.1
    if re.search(r'^[-*•]', output, re.MULTILINE):  # Bullet points
        score += 0.1
    if re.search(r'^#{1,3}\s', output, re.MULTILINE):  # Headers
        score += 0.1
    if 'first' in output.lower() or 'then' in output.lower() or 'next' in output.lower():
        score += 0.1  # Sequential language
    if len(output) > 500:  # Sufficient length
        score += 0.1

    return min(1.0, score)


async def synthesize_with_model(
    model: str,
    query: str,
    context: str,
    max_tokens: int = 2048
) -> dict:
    """Run synthesis with a specific model."""

    vram_before = get_vram_mb()
    start_time = time.time()
    ttfs = None

    prompt = f"""You are an expert industrial automation engineer. Based on the following context, provide a comprehensive answer to the query.

QUERY: {query}

CONTEXT:
{context[:30000]}

Provide a detailed, practical response that:
1. Directly addresses the diagnostic question
2. Lists specific troubleshooting steps
3. Identifies likely root causes based on the symptoms
4. Recommends corrective actions

RESPONSE:"""

    async with aiohttp.ClientSession() as session:
        try:
            output_chunks = []

            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                    }
                },
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status != 200:
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}"
                    }

                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            if 'response' in data:
                                if ttfs is None:
                                    ttfs = (time.time() - start_time) * 1000
                                output_chunks.append(data['response'])
                        except:
                            pass

            duration_ms = (time.time() - start_time) * 1000
            vram_after = get_vram_mb()
            output = ''.join(output_chunks)

            return {
                "success": True,
                "output": output,
                "duration_ms": duration_ms,
                "ttfs_ms": ttfs,
                "output_tokens": len(output.split()),
                "thinking_tokens": count_thinking_tokens(output),
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout after 300s"}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def benchmark_model_on_context(
    model: str,
    context: dict,
    unload_first: bool = True
) -> dict:
    """Benchmark a model on a specific context."""

    if unload_first:
        unload_all_models()
        await asyncio.sleep(2)

    # Parse expected topics
    expected_topics = []
    try:
        expected_topics = json.loads(context.get('expected_topics', '[]'))
    except:
        pass

    # Run synthesis
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

    # Calculate quality scores
    topic_score = calculate_topic_coverage(output, expected_topics)
    factual_score = calculate_factual_density(output)
    coherence_score = calculate_coherence(output)

    # Overall score: weighted average
    overall_score = (
        topic_score * 0.4 +
        factual_score * 0.3 +
        coherence_score * 0.3
    )

    return {
        'context_id': context['context_id'],
        'model': model,
        'synthesis_output': output[:10000],  # Cap storage
        'duration_ms': result['duration_ms'],
        'ttfs_ms': result.get('ttfs_ms'),
        'output_tokens': result['output_tokens'],
        'thinking_tokens': result['thinking_tokens'],
        'vram_before_mb': result['vram_before_mb'],
        'vram_after_mb': result['vram_after_mb'],
        'topic_coverage_score': topic_score,
        'factual_density_score': factual_score,
        'coherence_score': coherence_score,
        'overall_score': overall_score,
        'success': True,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


def print_rankings(db: SynthesisContextDB):
    """Print current rankings."""
    rankings = db.get_rankings()

    if not rankings:
        print("\nNo benchmark results yet.")
        return

    print(f"\n{'='*100}")
    print("SYNTHESIS MODEL RANKINGS")
    print(f"{'='*100}")
    print(f"{'Rank':<5} {'Model':<40} {'Duration':<12} {'TTFS':<10} {'Tokens':<8} {'Topic':<8} {'Fact':<8} {'Coher':<8} {'Overall':<8}")
    print("-" * 100)

    for i, r in enumerate(rankings, 1):
        print(f"{i:<5} {r['model']:<40} {r['avg_duration_ms']:.0f}ms{'':<5} "
              f"{r['avg_ttfs_ms'] or 0:.0f}ms{'':<4} "
              f"{r['avg_tokens']:.0f}{'':<4} "
              f"{r['avg_topic_score']:.2f}{'':<4} "
              f"{r['avg_factual_score']:.2f}{'':<4} "
              f"{r['avg_coherence_score']:.2f}{'':<4} "
              f"{r['avg_overall_score']:.2f}")


async def run_benchmark(
    models: list = None,
    contexts: list = None,
    max_contexts: int = 3,
    unload_between: bool = True
):
    """Run benchmark on models and contexts."""

    db = SynthesisContextDB()

    if models is None:
        models = SYNTHESIS_MODELS

    if contexts is None:
        contexts = db.get_all_contexts()[:max_contexts]

    print(f"Benchmarking {len(models)} models on {len(contexts)} contexts")

    for ctx in contexts:
        print(f"\n{'='*80}")
        print(f"Context: {ctx['context_id']}")
        print(f"Query: {ctx['query'][:60]}...")
        print(f"Context size: {len(ctx['retrieved_context'])} chars")
        print(f"{'='*80}")

        for model in models:
            print(f"\n--- Testing: {model} ---")

            result = await benchmark_model_on_context(
                model=model,
                context=ctx,
                unload_first=unload_between
            )

            if result['success']:
                db.store_benchmark(result)
                print(f"  Duration: {result['duration_ms']:.0f}ms")
                print(f"  TTFS: {result.get('ttfs_ms', 0):.0f}ms")
                print(f"  Tokens: {result['output_tokens']}")
                print(f"  Topic: {result['topic_coverage_score']:.2f}")
                print(f"  Factual: {result['factual_density_score']:.2f}")
                print(f"  Coherence: {result['coherence_score']:.2f}")
                print(f"  Overall: {result['overall_score']:.2f}")
            else:
                print(f"  FAILED: {result.get('error_message')}")

            await asyncio.sleep(2)

    print_rankings(db)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Synthesis model benchmarker")
    parser.add_argument("--models", type=str, help="Comma-separated list of models")
    parser.add_argument("--context", type=str, help="Specific context ID")
    parser.add_argument("--max-contexts", type=int, default=3, help="Max contexts to test")
    parser.add_argument("--no-unload", action="store_true", help="Don't unload between tests")
    parser.add_argument("--rankings", action="store_true", help="Show current rankings")
    args = parser.parse_args()

    db = SynthesisContextDB()

    if args.rankings:
        print_rankings(db)
        return

    models = SYNTHESIS_MODELS
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    contexts = None
    if args.context:
        all_contexts = db.get_all_contexts()
        contexts = [c for c in all_contexts if c['context_id'] == args.context]
        if not contexts:
            print(f"Context not found: {args.context}")
            return

    await run_benchmark(
        models=models,
        contexts=contexts,
        max_contexts=args.max_contexts,
        unload_between=not args.no_unload
    )


if __name__ == "__main__":
    asyncio.run(main())
