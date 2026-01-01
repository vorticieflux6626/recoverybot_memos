#!/usr/bin/env python3
"""
SGLang vs Ollama Benchmark Script
G.8.5: Evaluate speculative decoding for 2-5x throughput improvement

This script compares:
- Ollama (current serving framework)
- SGLang (high-performance alternative with speculative decoding)

Usage:
    python scripts/benchmark_sglang_vs_ollama.py --ollama-only
    python scripts/benchmark_sglang_vs_ollama.py --sglang-only
    python scripts/benchmark_sglang_vs_ollama.py --compare
"""

import asyncio
import time
import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Optional
import httpx


@dataclass
class BenchmarkResult:
    """Result from a single inference benchmark."""
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float  # Time to first token
    total_time_ms: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run."""
    provider: str
    model: str
    num_requests: int
    avg_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    avg_tokens_per_second: float
    p50_tokens_per_second: float
    total_tokens: int
    success_rate: float


# Benchmark prompts - varying complexity
BENCHMARK_PROMPTS = [
    # Short factual
    "What is the capital of France?",

    # Medium technical
    "Explain the SRVO-063 BZAL alarm on FANUC robots and how to fix it.",

    # Long procedure
    """Provide a step-by-step procedure for performing zero mastering on a FANUC robot
    after replacing the encoder on axis J2. Include all safety precautions and
    the specific menu navigation required.""",

    # Complex reasoning
    """Compare and contrast the collision detection methods available on FANUC robots:
    1. ACAL (Advanced Collision Avoidance Logic)
    2. External force sensing
    3. Torque monitoring

    Discuss the advantages, disadvantages, and appropriate use cases for each.""",

    # Code generation
    """Write a KAREL program for a FANUC robot that:
    1. Moves to a home position
    2. Picks up a part from a conveyor
    3. Places it in a fixture
    4. Returns to home
    Include error handling for gripper failures.""",
]


async def benchmark_ollama(
    prompt: str,
    model: str = "qwen3:8b",
    base_url: str = "http://localhost:11434"
) -> BenchmarkResult:
    """Benchmark a single request to Ollama."""

    start_time = time.perf_counter()
    ttft = None
    completion_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_ctx": 8192,
                        "temperature": 0.7,
                    }
                },
            )
            response.raise_for_status()

            full_response = ""
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if ttft is None and data.get("response"):
                        ttft = (time.perf_counter() - start_time) * 1000

                    if data.get("response"):
                        full_response += data["response"]
                        completion_tokens += 1

                    if data.get("done"):
                        break

            total_time = (time.perf_counter() - start_time) * 1000

            # Estimate prompt tokens (rough approximation)
            prompt_tokens = len(prompt.split()) * 1.3

            return BenchmarkResult(
                provider="ollama",
                model=model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=int(prompt_tokens) + completion_tokens,
                ttft_ms=ttft or total_time,
                total_time_ms=total_time,
                tokens_per_second=completion_tokens / (total_time / 1000) if total_time > 0 else 0,
                success=True,
            )

    except Exception as e:
        total_time = (time.perf_counter() - start_time) * 1000
        return BenchmarkResult(
            provider="ollama",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ttft_ms=0,
            total_time_ms=total_time,
            tokens_per_second=0,
            success=False,
            error=str(e),
        )


async def benchmark_sglang(
    prompt: str,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    base_url: str = "http://localhost:30000"
) -> BenchmarkResult:
    """Benchmark a single request to SGLang server."""

    start_time = time.perf_counter()
    ttft = None
    completion_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # SGLang uses OpenAI-compatible API
            response = await client.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "stream": True,
                },
            )
            response.raise_for_status()

            full_response = ""
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    data = json.loads(data_str)
                    if ttft is None:
                        ttft = (time.perf_counter() - start_time) * 1000

                    if data.get("choices"):
                        text = data["choices"][0].get("text", "")
                        full_response += text
                        completion_tokens += len(text.split())

            total_time = (time.perf_counter() - start_time) * 1000
            prompt_tokens = len(prompt.split()) * 1.3

            return BenchmarkResult(
                provider="sglang",
                model=model,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=int(prompt_tokens) + completion_tokens,
                ttft_ms=ttft or total_time,
                total_time_ms=total_time,
                tokens_per_second=completion_tokens / (total_time / 1000) if total_time > 0 else 0,
                success=True,
            )

    except Exception as e:
        total_time = (time.perf_counter() - start_time) * 1000
        return BenchmarkResult(
            provider="sglang",
            model=model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ttft_ms=0,
            total_time_ms=total_time,
            tokens_per_second=0,
            success=False,
            error=str(e),
        )


def calculate_summary(results: list[BenchmarkResult], provider: str, model: str) -> BenchmarkSummary:
    """Calculate summary statistics from benchmark results."""

    successful = [r for r in results if r.success]

    if not successful:
        return BenchmarkSummary(
            provider=provider,
            model=model,
            num_requests=len(results),
            avg_ttft_ms=0,
            p50_ttft_ms=0,
            p95_ttft_ms=0,
            avg_tokens_per_second=0,
            p50_tokens_per_second=0,
            total_tokens=0,
            success_rate=0,
        )

    ttfts = [r.ttft_ms for r in successful]
    tps = [r.tokens_per_second for r in successful]

    return BenchmarkSummary(
        provider=provider,
        model=model,
        num_requests=len(results),
        avg_ttft_ms=statistics.mean(ttfts),
        p50_ttft_ms=statistics.median(ttfts),
        p95_ttft_ms=sorted(ttfts)[int(len(ttfts) * 0.95)] if len(ttfts) > 1 else ttfts[0],
        avg_tokens_per_second=statistics.mean(tps),
        p50_tokens_per_second=statistics.median(tps),
        total_tokens=sum(r.total_tokens for r in successful),
        success_rate=len(successful) / len(results),
    )


async def run_ollama_benchmark(num_runs: int = 3) -> BenchmarkSummary:
    """Run full Ollama benchmark."""

    print("\n" + "="*60)
    print("OLLAMA BENCHMARK")
    print("="*60)

    # Check if Ollama is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        return None

    print(f"✅ Ollama is running")
    print(f"Running {len(BENCHMARK_PROMPTS)} prompts x {num_runs} runs = {len(BENCHMARK_PROMPTS) * num_runs} requests\n")

    results = []
    for run in range(num_runs):
        print(f"--- Run {run + 1}/{num_runs} ---")
        for i, prompt in enumerate(BENCHMARK_PROMPTS):
            result = await benchmark_ollama(prompt)
            results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  Prompt {i+1}: {status} TTFT={result.ttft_ms:.0f}ms, {result.tokens_per_second:.1f} tok/s")

    summary = calculate_summary(results, "ollama", "qwen3:8b")

    print(f"\n📊 OLLAMA SUMMARY:")
    print(f"   Success rate: {summary.success_rate:.1%}")
    print(f"   Avg TTFT: {summary.avg_ttft_ms:.0f}ms")
    print(f"   P50 TTFT: {summary.p50_ttft_ms:.0f}ms")
    print(f"   P95 TTFT: {summary.p95_ttft_ms:.0f}ms")
    print(f"   Avg tokens/sec: {summary.avg_tokens_per_second:.1f}")
    print(f"   Total tokens: {summary.total_tokens}")

    return summary


async def run_sglang_benchmark(num_runs: int = 3) -> BenchmarkSummary:
    """Run full SGLang benchmark."""

    print("\n" + "="*60)
    print("SGLANG BENCHMARK")
    print("="*60)

    # Check if SGLang is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:30000/health")
            response.raise_for_status()
    except Exception as e:
        print(f"❌ SGLang server not available: {e}")
        print("\nTo start SGLang server:")
        print("  python -m sglang.launch_server --model meta-llama/Llama-3.1-8B-Instruct --port 30000")
        return None

    print(f"✅ SGLang is running")
    print(f"Running {len(BENCHMARK_PROMPTS)} prompts x {num_runs} runs = {len(BENCHMARK_PROMPTS) * num_runs} requests\n")

    results = []
    for run in range(num_runs):
        print(f"--- Run {run + 1}/{num_runs} ---")
        for i, prompt in enumerate(BENCHMARK_PROMPTS):
            result = await benchmark_sglang(prompt)
            results.append(result)
            status = "✅" if result.success else "❌"
            print(f"  Prompt {i+1}: {status} TTFT={result.ttft_ms:.0f}ms, {result.tokens_per_second:.1f} tok/s")

    summary = calculate_summary(results, "sglang", "Llama-3.1-8B-Instruct")

    print(f"\n📊 SGLANG SUMMARY:")
    print(f"   Success rate: {summary.success_rate:.1%}")
    print(f"   Avg TTFT: {summary.avg_ttft_ms:.0f}ms")
    print(f"   P50 TTFT: {summary.p50_ttft_ms:.0f}ms")
    print(f"   P95 TTFT: {summary.p95_ttft_ms:.0f}ms")
    print(f"   Avg tokens/sec: {summary.avg_tokens_per_second:.1f}")
    print(f"   Total tokens: {summary.total_tokens}")

    return summary


def compare_results(ollama: BenchmarkSummary, sglang: BenchmarkSummary):
    """Compare and print results from both providers."""

    print("\n" + "="*60)
    print("COMPARISON: OLLAMA vs SGLANG")
    print("="*60)

    if not ollama or not sglang:
        print("⚠️  Cannot compare - one or both benchmarks failed")
        return

    ttft_improvement = (ollama.avg_ttft_ms - sglang.avg_ttft_ms) / ollama.avg_ttft_ms * 100
    tps_improvement = (sglang.avg_tokens_per_second - ollama.avg_tokens_per_second) / ollama.avg_tokens_per_second * 100

    print(f"\n{'Metric':<25} {'Ollama':<15} {'SGLang':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Avg TTFT (ms)':<25} {ollama.avg_ttft_ms:<15.0f} {sglang.avg_ttft_ms:<15.0f} {ttft_improvement:+.1f}%")
    print(f"{'P50 TTFT (ms)':<25} {ollama.p50_ttft_ms:<15.0f} {sglang.p50_ttft_ms:<15.0f}")
    print(f"{'P95 TTFT (ms)':<25} {ollama.p95_ttft_ms:<15.0f} {sglang.p95_ttft_ms:<15.0f}")
    print(f"{'Avg tokens/sec':<25} {ollama.avg_tokens_per_second:<15.1f} {sglang.avg_tokens_per_second:<15.1f} {tps_improvement:+.1f}%")
    print(f"{'Success rate':<25} {ollama.success_rate:<15.1%} {sglang.success_rate:<15.1%}")

    print("\n📈 VERDICT:")
    if tps_improvement > 50:
        print(f"   SGLang provides {tps_improvement:.0f}% higher throughput - SIGNIFICANT IMPROVEMENT")
    elif tps_improvement > 20:
        print(f"   SGLang provides {tps_improvement:.0f}% higher throughput - MODERATE IMPROVEMENT")
    elif tps_improvement > 0:
        print(f"   SGLang provides {tps_improvement:.0f}% higher throughput - MARGINAL IMPROVEMENT")
    else:
        print(f"   Ollama performs better by {-tps_improvement:.0f}% - STICK WITH OLLAMA")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark SGLang vs Ollama")
    parser.add_argument("--ollama-only", action="store_true", help="Only benchmark Ollama")
    parser.add_argument("--sglang-only", action="store_true", help="Only benchmark SGLang")
    parser.add_argument("--compare", action="store_true", help="Benchmark both and compare")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--output", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    if not any([args.ollama_only, args.sglang_only, args.compare]):
        args.ollama_only = True  # Default to Ollama only

    ollama_summary = None
    sglang_summary = None

    if args.ollama_only or args.compare:
        ollama_summary = await run_ollama_benchmark(args.runs)

    if args.sglang_only or args.compare:
        sglang_summary = await run_sglang_benchmark(args.runs)

    if args.compare:
        compare_results(ollama_summary, sglang_summary)

    if args.output:
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ollama": asdict(ollama_summary) if ollama_summary else None,
            "sglang": asdict(sglang_summary) if sglang_summary else None,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📁 Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
