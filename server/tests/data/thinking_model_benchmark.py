#!/usr/bin/env python3
"""
Thinking Model Benchmark Framework

Systematically tests reasoning/thinking models across:
- Parameter sizes (1.5B → 671B)
- Quantization levels (q4_K_M, q8_0, fp16)
- Base architectures (Qwen, Llama, Qwen3)

Integrates with model_specs.py and model_benchmarks.py.
"""

import asyncio
import sqlite3
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import httpx

# Import from sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agentic.model_specs import OLLAMA_MODEL_SPECS, get_reasoning_models

# ============== THINKING MODEL CATALOG ==============

THINKING_MODELS = {
    # DeepSeek-R1 variants by size and quantization
    "deepseek-r1": {
        "1.5b": {
            "q4_K_M": {"tag": "deepseek-r1:1.5b-qwen-distill-q4_K_M", "size_gb": 1.1, "vram_gb": 2},
            "q8_0": {"tag": "deepseek-r1:1.5b-qwen-distill-q8_0", "size_gb": 1.9, "vram_gb": 3},
            "fp16": {"tag": "deepseek-r1:1.5b-qwen-distill-fp16", "size_gb": 3.6, "vram_gb": 5},
        },
        "7b": {
            "q4_K_M": {"tag": "deepseek-r1:7b-qwen-distill-q4_K_M", "size_gb": 4.7, "vram_gb": 6},
            "q8_0": {"tag": "deepseek-r1:7b-qwen-distill-q8_0", "size_gb": 8.1, "vram_gb": 10},
            "fp16": {"tag": "deepseek-r1:7b-qwen-distill-fp16", "size_gb": 15, "vram_gb": 18},
        },
        "8b-llama": {
            "q4_K_M": {"tag": "deepseek-r1:8b-llama-distill-q4_K_M", "size_gb": 4.9, "vram_gb": 6},
            "q8_0": {"tag": "deepseek-r1:8b-llama-distill-q8_0", "size_gb": 8.5, "vram_gb": 10},
            "fp16": {"tag": "deepseek-r1:8b-llama-distill-fp16", "size_gb": 16, "vram_gb": 18},
        },
        "8b-qwen3": {
            "q4_K_M": {"tag": "deepseek-r1:8b-0528-qwen3-q4_K_M", "size_gb": 5.2, "vram_gb": 7},
            "q8_0": {"tag": "deepseek-r1:8b-0528-qwen3-q8_0", "size_gb": 8.9, "vram_gb": 11},
            "fp16": {"tag": "deepseek-r1:8b-0528-qwen3-fp16", "size_gb": 16, "vram_gb": 18},
        },
        "14b": {
            "q4_K_M": {"tag": "deepseek-r1:14b-qwen-distill-q4_K_M", "size_gb": 9.0, "vram_gb": 11},
            "q8_0": {"tag": "deepseek-r1:14b-qwen-distill-q8_0", "size_gb": 16, "vram_gb": 18},
            "fp16": {"tag": "deepseek-r1:14b-qwen-distill-fp16", "size_gb": 30, "vram_gb": 34},
        },
        "32b": {
            "q4_K_M": {"tag": "deepseek-r1:32b-qwen-distill-q4_K_M", "size_gb": 20, "vram_gb": 22},
            "q8_0": {"tag": "deepseek-r1:32b-qwen-distill-q8_0", "size_gb": 35, "vram_gb": 38},
            "fp16": {"tag": "deepseek-r1:32b-qwen-distill-fp16", "size_gb": 66, "vram_gb": 70},
        },
        "70b-llama": {
            "q4_K_M": {"tag": "deepseek-r1:70b-llama-distill-q4_K_M", "size_gb": 43, "vram_gb": 48},
            "q8_0": {"tag": "deepseek-r1:70b-llama-distill-q8_0", "size_gb": 75, "vram_gb": 80},
        },
    },

    # QwQ variants
    "qwq": {
        "32b": {
            "q4_K_M": {"tag": "qwq:32b", "size_gb": 19, "vram_gb": 22},
        },
    },

    # Phi-4 Reasoning
    "phi4-reasoning": {
        "14b": {
            "q4_K_M": {"tag": "phi4-reasoning:14b", "size_gb": 11, "vram_gb": 13},
        },
    },

    # OpenThinker
    "openthinker": {
        "7b": {
            "q4_K_M": {"tag": "openthinker:7b", "size_gb": 4.7, "vram_gb": 6},
        },
        "32b": {
            "q4_K_M": {"tag": "openthinker:32b", "size_gb": 19, "vram_gb": 22},
        },
    },

    # Cogito
    "cogito": {
        "8b": {
            "q4_K_M": {"tag": "cogito:8b", "size_gb": 4.9, "vram_gb": 6},
        },
        "32b": {
            "q4_K_M": {"tag": "cogito:32b", "size_gb": 19, "vram_gb": 22},
        },
    },

    # Qwen3-VL Thinking (vision + reasoning)
    "qwen3-vl-thinking": {
        "4b": {
            "bf16": {"tag": "qwen3-vl:4b-thinking-bf16", "size_gb": 8.9, "vram_gb": 11},
        },
        "8b": {
            "bf16": {"tag": "qwen3-vl:8b-thinking-bf16", "size_gb": 17, "vram_gb": 19},
        },
    },
}

# Models that fit in 24GB VRAM (TITAN RTX)
MODELS_24GB = [
    # Tier 1: Small (< 8GB VRAM)
    "deepseek-r1:1.5b-qwen-distill-q4_K_M",
    "deepseek-r1:1.5b-qwen-distill-q8_0",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "deepseek-r1:8b",  # Llama default q4
    "openthinker:7b",
    "cogito:8b",

    # Tier 2: Medium (8-12GB VRAM)
    "deepseek-r1:7b-qwen-distill-q8_0",
    "deepseek-r1:8b-0528-qwen3-q8_0",
    "deepseek-r1:14b-qwen-distill-q4_K_M",
    "phi4-reasoning:14b",
    "qwen3-vl:4b-thinking-bf16",

    # Tier 3: Large (12-20GB VRAM)
    "deepseek-r1:8b-0528-qwen3-fp16",
    "deepseek-r1:14b-qwen-distill-q8_0",

    # Tier 4: XL (20-24GB VRAM)
    "deepseek-r1:32b",  # q4_K_M default
    "qwq:32b",
    "openthinker:32b",
    "cogito:32b",
]


class ThinkingModelBenchmark:
    """Benchmark framework for thinking/reasoning models."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "model_benchmarks.db"
        self.db_path = Path(db_path)
        self.ollama_url = "http://localhost:11434"
        self._init_db()

    def _init_db(self):
        """Extend benchmark database with thinking-specific tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS thinking_benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    parameter_size TEXT NOT NULL,
                    quantization TEXT NOT NULL,
                    model_tag TEXT NOT NULL,

                    -- Timing metrics
                    ttfs_ms REAL,
                    thinking_time_ms REAL,
                    total_duration_ms REAL,

                    -- Resource metrics
                    vram_before_mb INTEGER,
                    vram_after_mb INTEGER,
                    vram_peak_mb INTEGER,
                    model_load_time_ms REAL,

                    -- Token metrics
                    input_tokens INTEGER,
                    thinking_tokens INTEGER,
                    output_tokens INTEGER,
                    tokens_per_second REAL,

                    -- Quality metrics
                    reasoning_depth_score REAL,
                    answer_correctness REAL,
                    thinking_coherence REAL,

                    -- Test info
                    test_context_id TEXT,
                    query TEXT,
                    output_preview TEXT,
                    thinking_preview TEXT,
                    success INTEGER DEFAULT 1,
                    error_message TEXT,
                    notes TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_thinking_family ON thinking_benchmarks(model_family);
                CREATE INDEX IF NOT EXISTS idx_thinking_size ON thinking_benchmarks(parameter_size);
                CREATE INDEX IF NOT EXISTS idx_thinking_quant ON thinking_benchmarks(quantization);
            """)

    def get_gpu_stats(self) -> Dict[str, int]:
        """Get current GPU memory stats."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "used_mb": int(parts[0]),
                    "total_mb": int(parts[1]),
                    "free_mb": int(parts[2])
                }
        except Exception:
            pass
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0}

    def get_local_models(self) -> List[str]:
        """Get list of locally available models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                return [line.split()[0] for line in lines if line]
        except Exception:
            pass
        return []

    def check_model_available(self, model_tag: str) -> bool:
        """Check if model is available locally."""
        local_models = self.get_local_models()
        # Handle partial matches (e.g., "deepseek-r1:8b" matches "deepseek-r1:8b")
        return any(model_tag in m or m.startswith(model_tag.split(':')[0]) for m in local_models)

    async def unload_all_models(self):
        """Unload all models for clean VRAM measurement."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/ps")
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("models", []):
                        await client.post(
                            f"{self.ollama_url}/api/generate",
                            json={"model": model["name"], "keep_alive": 0}
                        )
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Warning: Could not unload models: {e}")

    def parse_thinking_output(self, text: str) -> tuple[str, str]:
        """
        Parse thinking model output to separate thinking from answer.
        Returns (thinking_text, answer_text)
        """
        # DeepSeek-R1 uses <think>...</think> tags
        import re
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            return thinking, answer

        # Some models use ### Thinking or similar headers
        if '### Thinking' in text or '## Thinking' in text:
            parts = re.split(r'###?\s*(?:Answer|Response|Conclusion)', text, maxsplit=1)
            if len(parts) == 2:
                thinking = parts[0].replace('### Thinking', '').replace('## Thinking', '').strip()
                return thinking, parts[1].strip()

        # No clear separation - assume all is answer
        return "", text

    async def benchmark_thinking_model(
        self,
        model_tag: str,
        context: str,
        query: str,
        model_family: str = "unknown",
        parameter_size: str = "unknown",
        quantization: str = "unknown",
        test_context_id: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.6,
        unload_first: bool = True
    ) -> Dict[str, Any]:
        """Benchmark a thinking model."""

        if unload_first:
            await self.unload_all_models()
            await asyncio.sleep(3)

        vram_before = self.get_gpu_stats()["used_mb"]

        # Synthesis prompt for thinking models
        prompt = f"""You are an expert technical analyst. Carefully reason through this problem step by step.

## Context
{context}

## Query
{query}

## Instructions
1. Think through the problem systematically
2. Consider multiple possible causes
3. Evaluate evidence from the context
4. Provide a clear, actionable answer

Begin your analysis:"""

        start_time = time.time()
        model_load_start = start_time
        ttfs_ms = None
        thinking_start = None
        output_text = ""
        error_msg = None
        success = False
        input_tokens = None
        output_tokens = None

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                first_token_time = None

                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model_tag,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if first_token_time is None and data.get("response"):
                                    first_token_time = time.time()
                                    ttfs_ms = (first_token_time - start_time) * 1000

                                output_text += data.get("response", "")

                                if data.get("done"):
                                    input_tokens = data.get("prompt_eval_count")
                                    output_tokens = data.get("eval_count")
                            except json.JSONDecodeError:
                                continue

                success = len(output_text) > 50

        except Exception as e:
            error_msg = str(e)
            success = False

        total_duration_ms = (time.time() - start_time) * 1000
        vram_after = self.get_gpu_stats()["used_mb"]

        # Parse thinking vs answer
        thinking_text, answer_text = self.parse_thinking_output(output_text)
        thinking_tokens = len(thinking_text.split()) if thinking_text else 0

        # Calculate tokens per second
        tps = output_tokens / (total_duration_ms / 1000) if output_tokens and total_duration_ms > 0 else 0

        result = {
            "model_tag": model_tag,
            "model_family": model_family,
            "parameter_size": parameter_size,
            "quantization": quantization,
            "success": success,
            "ttfs_ms": ttfs_ms,
            "total_duration_ms": total_duration_ms,
            "vram_before_mb": vram_before,
            "vram_after_mb": vram_after,
            "vram_delta_mb": vram_after - vram_before,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "tokens_per_second": tps,
            "thinking_preview": thinking_text[:500] if thinking_text else "",
            "output_preview": answer_text[:500] if answer_text else output_text[:500],
            "error_message": error_msg
        }

        # Store in database
        self._store_thinking_result(result, test_context_id, query)

        return result

    def _store_thinking_result(self, result: Dict, test_context_id: Optional[str], query: str):
        """Store thinking benchmark result."""
        timestamp = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO thinking_benchmarks (
                    timestamp, model_family, parameter_size, quantization, model_tag,
                    ttfs_ms, total_duration_ms, vram_before_mb, vram_after_mb, vram_peak_mb,
                    input_tokens, thinking_tokens, output_tokens, tokens_per_second,
                    test_context_id, query, output_preview, thinking_preview,
                    success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, result["model_family"], result["parameter_size"],
                result["quantization"], result["model_tag"],
                result.get("ttfs_ms"), result["total_duration_ms"],
                result["vram_before_mb"], result["vram_after_mb"],
                max(result["vram_before_mb"], result["vram_after_mb"]),
                result.get("input_tokens"), result.get("thinking_tokens"),
                result.get("output_tokens"), result.get("tokens_per_second"),
                test_context_id, query, result.get("output_preview"),
                result.get("thinking_preview"),
                1 if result["success"] else 0, result.get("error_message")
            ))

    def get_rankings(self, by: str = "duration") -> List[Dict]:
        """Get model rankings."""
        order_col = {
            "duration": "AVG(total_duration_ms)",
            "ttfs": "AVG(ttfs_ms)",
            "vram": "AVG(vram_after_mb - vram_before_mb)",
            "tps": "AVG(tokens_per_second) DESC"
        }.get(by, "AVG(total_duration_ms)")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT
                    model_family,
                    parameter_size,
                    quantization,
                    model_tag,
                    COUNT(*) as runs,
                    AVG(ttfs_ms) as avg_ttfs_ms,
                    AVG(total_duration_ms) as avg_duration_ms,
                    AVG(vram_after_mb - vram_before_mb) as avg_vram_delta,
                    AVG(output_tokens) as avg_output_tokens,
                    AVG(thinking_tokens) as avg_thinking_tokens,
                    AVG(tokens_per_second) as avg_tps,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM thinking_benchmarks
                GROUP BY model_tag
                ORDER BY {order_col}
            """)
            return [dict(row) for row in cursor.fetchall()]

    def print_rankings(self):
        """Print formatted rankings."""
        rankings = self.get_rankings()

        print(f"\n{'='*120}")
        print("THINKING MODEL RANKINGS")
        print(f"{'='*120}")
        print(f"{'Rank':<5} {'Model':<45} {'Duration':<12} {'TTFS':<10} {'VRAM Δ':<10} {'TPS':<8} {'Think':<8} {'Success':<8}")
        print(f"{'-'*120}")

        for i, r in enumerate(rankings, 1):
            duration = f"{r['avg_duration_ms']:.0f}ms" if r['avg_duration_ms'] else "N/A"
            ttfs = f"{r['avg_ttfs_ms']:.0f}ms" if r['avg_ttfs_ms'] else "N/A"
            vram = f"{r['avg_vram_delta']:.0f}MB" if r['avg_vram_delta'] else "N/A"
            tps = f"{r['avg_tps']:.1f}" if r['avg_tps'] else "N/A"
            think = f"{r['avg_thinking_tokens']:.0f}" if r['avg_thinking_tokens'] else "0"
            success = f"{r['success_rate']:.0f}%"

            print(f"{i:<5} {r['model_tag']:<45} {duration:<12} {ttfs:<10} {vram:<10} {tps:<8} {think:<8} {success:<8}")


# Test context for thinking models (harder than regular synthesis)
THINKING_TEST_CONTEXT = {
    "id": "multi_fault_diagnosis",
    "name": "Multi-System Fault Diagnosis",
    "domain": "industrial_automation",
    "difficulty": "expert",
    "context": """## Incident Report

A manufacturing line experienced a cascading failure affecting multiple systems:

1. **PLC System**: Allen-Bradley ControlLogix 1756-L73 showing intermittent Major Fault 1:13
2. **Robot**: FANUC R-30iB displaying SRVO-023 on J2 axis during part placement
3. **Vision**: Cognex camera reporting "Image Acquisition Timeout" errors
4. **Network**: EtherNet/IP traffic showing packet loss >5%

Timeline:
- 14:32 - Power monitoring detected 15% voltage sag lasting 200ms
- 14:33 - PLC faulted, robot stopped mid-cycle
- 14:35 - Vision system began reporting timeouts
- 14:40 - Maintenance attempted reset, symptoms returned within 10 minutes

Environment: 95°F ambient, high humidity day, plant AC struggling

## Available Data

PLC Diagnostics:
- EN2T module status: Link OK, but CIP connections dropping
- Power supply output: 23.8V (spec: 24V ±5%)
- Backplane current: 2.1A (spec: max 4A)

Robot Diagnostics:
- J2 motor temp: 78°C (warning at 80°C)
- Disturbance torque: 45% at fault (normal: <30%)
- Encoder battery: 2.9V (replace at 2.8V)

Network Analysis:
- Multicast storm detected from IP 192.168.1.45
- Switch port counters show CRC errors on ports 3, 7, 12
""",
    "query": "Given the cascading failures, voltage sag, and environmental conditions, what is the most likely root cause? Develop a systematic diagnostic approach prioritizing the most probable failure modes.",
    "expected_reasoning": [
        "voltage_sag_impact",
        "thermal_stress",
        "network_multicast_storm",
        "power_supply_marginal",
        "systematic_approach"
    ]
}


async def run_thinking_benchmark(models: List[str], verbose: bool = True):
    """Run benchmark on specified thinking models."""
    bench = ThinkingModelBenchmark()

    ctx = THINKING_TEST_CONTEXT
    results = []

    for model_tag in models:
        # Parse model info from tag
        parts = model_tag.split(':')
        family = parts[0]
        variant = parts[1] if len(parts) > 1 else "default"

        # Determine quantization and size from variant
        size = "unknown"
        quant = "q4_K_M"
        if 'fp16' in variant:
            quant = "fp16"
        elif 'q8' in variant:
            quant = "q8_0"
        elif 'bf16' in variant:
            quant = "bf16"

        for s in ['1.5b', '7b', '8b', '14b', '32b', '70b']:
            if s in variant:
                size = s
                break

        print(f"\n{'='*80}")
        print(f"Testing: {model_tag}")
        print(f"Family: {family} | Size: {size} | Quant: {quant}")
        print(f"{'='*80}")

        try:
            result = await bench.benchmark_thinking_model(
                model_tag=model_tag,
                context=ctx["context"],
                query=ctx["query"],
                model_family=family,
                parameter_size=size,
                quantization=quant,
                test_context_id=ctx["id"]
            )

            results.append(result)

            # Print result
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"  Status: {status}")
            print(f"  TTFS: {result['ttfs_ms']:.0f}ms" if result['ttfs_ms'] else "  TTFS: N/A")
            print(f"  Duration: {result['total_duration_ms']:.0f}ms ({result['total_duration_ms']/1000:.1f}s)")
            print(f"  VRAM: {result['vram_before_mb']}MB -> {result['vram_after_mb']}MB (+{result['vram_delta_mb']}MB)")
            print(f"  Tokens: {result.get('input_tokens', 'N/A')} in / {result.get('output_tokens', 'N/A')} out")
            print(f"  TPS: {result.get('tokens_per_second', 0):.1f}")
            print(f"  Thinking tokens: {result.get('thinking_tokens', 0)}")

            if verbose and result.get('output_preview'):
                print(f"\n  Output preview:")
                for line in result['output_preview'][:300].split('\n')[:4]:
                    print(f"    {line[:75]}")

        except Exception as e:
            print(f"  ERROR: {e}")

        await asyncio.sleep(2)

    # Print final rankings
    print("\n")
    bench.print_rankings()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thinking Model Benchmark")
    parser.add_argument("--models", type=str, help="Comma-separated model tags")
    parser.add_argument("--tier", type=str, choices=["small", "medium", "large", "all"],
                       help="Test models by VRAM tier")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.list:
        print("Available thinking models for 24GB VRAM:")
        for m in MODELS_24GB:
            print(f"  - {m}")
        sys.exit(0)

    # Determine models to test
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.tier:
        if args.tier == "small":
            models = MODELS_24GB[:6]
        elif args.tier == "medium":
            models = MODELS_24GB[6:11]
        elif args.tier == "large":
            models = MODELS_24GB[11:]
        else:
            models = MODELS_24GB
    else:
        # Default: test a few representative models
        models = [
            "deepseek-r1:8b",
            "deepseek-r1:14b-qwen-distill-q8_0",
        ]

    asyncio.run(run_thinking_benchmark(models, verbose=not args.quiet))
