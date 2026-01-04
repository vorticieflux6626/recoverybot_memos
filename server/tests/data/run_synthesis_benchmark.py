#!/usr/bin/env python3
"""
Synthesis Benchmark Runner - Test models systematically for synthesis quality.

Usage:
    python run_synthesis_benchmark.py --models qwen3:8b,gemma3:4b --context fanuc_servo
    python run_synthesis_benchmark.py --tier small --context all
    python run_synthesis_benchmark.py --model deepseek-r1:14b-qwen-distill-q8_0 --context plc_network
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.data.model_benchmarks import ModelBenchmark, SYNTHESIS_CONTEXTS


# Model tiers by VRAM requirements
MODEL_TIERS = {
    "tiny": [  # <3GB VRAM
        "gemma3:1b",
        "llama3.2:1b-instruct-fp16",
        "nemotron-mini:4b",
    ],
    "small": [  # 3-6GB VRAM
        "gemma3:4b",
        "qwen3:8b",
        "llama3.2:3b-instruct-fp16",
        "ministral-3:3b",
        "olmo-3:7b",
    ],
    "medium": [  # 6-12GB VRAM
        "gemma3:12b",
        "ministral-3:8b",
        "qwen3:8b",
    ],
    "large": [  # 12-20GB VRAM
        "deepseek-r1:14b-qwen-distill-q8_0",
        "deepseek-r1:8b-0528-qwen3-fp16",
        "qwen3:30b-a3b-instruct-2507-q4_K_M",
        "devstral-small-2:24b",
    ],
    "xlarge": [  # >20GB VRAM
        "olmo-3.1:32b",
        "nemotron-3-nano:30b",
        "devstral-small-2:24b-instruct-2512-q8_0",
    ]
}

# Recommended models for synthesis benchmarking
SYNTHESIS_MODELS = [
    # Tier 1: Small/Fast (baseline)
    "gemma3:4b",
    "qwen3:8b",
    "ministral-3:3b",

    # Tier 2: Medium (balance)
    "gemma3:12b",
    "olmo-3:7b",

    # Tier 3: Large/Quality (current default)
    "deepseek-r1:14b-qwen-distill-q8_0",

    # Tier 4: XL (if VRAM allows)
    # "qwen3:30b-a3b-instruct-2507-q4_K_M",
]


async def run_benchmark(
    models: list[str],
    context_ids: list[str],
    unload_between: bool = True,
    verbose: bool = True
):
    """Run benchmarks for specified models and contexts."""
    bench = ModelBenchmark()

    results = []

    for ctx_id in context_ids:
        ctx = bench.get_test_context(ctx_id)
        if not ctx:
            print(f"Warning: Context '{ctx_id}' not found, skipping")
            continue

        print(f"\n{'='*80}")
        print(f"CONTEXT: {ctx['name']} ({ctx_id})")
        print(f"Domain: {ctx['domain']} | Difficulty: {ctx['difficulty']}")
        print(f"Query: {ctx['query'][:80]}...")
        print(f"{'='*80}")

        for model in models:
            print(f"\n--- Testing: {model} ---")

            try:
                result = await bench.benchmark_synthesis(
                    model=model,
                    context=ctx["context"],
                    query=ctx["query"],
                    test_context_id=ctx_id,
                    unload_first=unload_between
                )

                results.append({
                    "model": model,
                    "context": ctx_id,
                    "result": result
                })

                # Print result summary
                status = "SUCCESS" if result.success else "FAILED"
                print(f"  Status: {status}")
                print(f"  TTFS: {result.ttfs_ms:.0f}ms" if result.ttfs_ms else "  TTFS: N/A")
                print(f"  Duration: {result.total_duration_ms:.0f}ms ({result.total_duration_ms/1000:.1f}s)")
                print(f"  VRAM: {result.vram_before_mb}MB -> {result.vram_after_mb}MB (+{result.vram_after_mb - result.vram_before_mb}MB)")
                print(f"  Tokens: {result.input_tokens} in / {result.output_tokens} out")

                if verbose and result.output_preview:
                    print(f"\n  Output preview:")
                    for line in result.output_preview[:300].split('\n')[:5]:
                        print(f"    {line[:75]}")

                if result.error_message:
                    print(f"  Error: {result.error_message}")

            except Exception as e:
                print(f"  ERROR: {e}")

            # Small delay between models
            await asyncio.sleep(2)

    # Print final rankings
    print("\n")
    bench.print_rankings("synthesis")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Run synthesis model benchmarks")
    parser.add_argument("--models", type=str, help="Comma-separated list of models")
    parser.add_argument("--model", type=str, help="Single model to test")
    parser.add_argument("--tier", type=str, choices=MODEL_TIERS.keys(), help="Model tier to test")
    parser.add_argument("--context", type=str, default="fanuc_servo", help="Context ID or 'all'")
    parser.add_argument("--no-unload", action="store_true", help="Don't unload models between tests")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--init", action="store_true", help="Just initialize test contexts")
    args = parser.parse_args()

    bench = ModelBenchmark()

    # Initialize test contexts
    for ctx_id, ctx_data in SYNTHESIS_CONTEXTS.items():
        bench.add_test_context(
            context_id=ctx_id,
            name=ctx_data["name"],
            context=ctx_data["context"],
            query=ctx_data["query"],
            domain=ctx_data["domain"],
            expected_keywords=ctx_data.get("expected_keywords"),
            difficulty=ctx_data["difficulty"]
        )

    if args.init:
        print("Test contexts initialized:")
        for ctx_id in SYNTHESIS_CONTEXTS:
            print(f"  - {ctx_id}")
        return

    # Determine models to test
    if args.model:
        models = [args.model]
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.tier:
        models = MODEL_TIERS[args.tier]
    else:
        models = SYNTHESIS_MODELS[:3]  # Default to first 3

    # Determine contexts
    if args.context == "all":
        contexts = list(SYNTHESIS_CONTEXTS.keys())
    else:
        contexts = [args.context]

    print(f"Models to test: {models}")
    print(f"Contexts: {contexts}")
    print()

    await run_benchmark(
        models=models,
        context_ids=contexts,
        unload_between=not args.no_unload,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    asyncio.run(main())
