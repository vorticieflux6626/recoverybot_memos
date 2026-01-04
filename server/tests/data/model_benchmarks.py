#!/usr/bin/env python3
"""
Model Benchmarking Framework - Systematic testing of LLM models for agentic operations.

Tracks: TTFS, VRAM usage, total duration, context sizes, output quality.

Usage:
    from tests.data.model_benchmarks import ModelBenchmark

    bench = ModelBenchmark()

    # Run synthesis benchmark
    result = await bench.benchmark_synthesis(
        model="qwen3:8b",
        context="...",
        query="..."
    )

    # Get model rankings
    rankings = bench.get_synthesis_rankings()
"""

import asyncio
import sqlite3
import json
import time
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import httpx


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    model: str
    operation: str  # synthesis, analysis, verification, etc.
    success: bool
    ttfs_ms: Optional[float]  # Time to first token (streaming)
    total_duration_ms: float
    vram_before_mb: int
    vram_after_mb: int
    vram_peak_mb: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    context_length: int
    output_preview: str
    output_quality_score: Optional[float]  # 0-1 manual or auto score
    error_message: Optional[str]
    model_size_gb: float
    quantization: str
    parameter_count: str
    notes: Optional[str]


class ModelBenchmark:
    """Benchmark framework for LLM model comparison."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent / "model_benchmarks.db"
        self.db_path = Path(db_path)
        self.ollama_url = "http://localhost:11434"
        self._init_db()

    def _init_db(self):
        """Initialize benchmark database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    ttfs_ms REAL,
                    total_duration_ms REAL NOT NULL,
                    vram_before_mb INTEGER,
                    vram_after_mb INTEGER,
                    vram_peak_mb INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    context_length INTEGER,
                    output_preview TEXT,
                    output_full TEXT,
                    output_quality_score REAL,
                    error_message TEXT,
                    model_size_gb REAL,
                    quantization TEXT,
                    parameter_count TEXT,
                    notes TEXT,
                    test_context_id TEXT,
                    query TEXT
                );

                CREATE TABLE IF NOT EXISTS test_contexts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT,
                    context TEXT NOT NULL,
                    query TEXT NOT NULL,
                    expected_keywords TEXT,
                    difficulty TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS model_rankings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    updated_at TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    model TEXT NOT NULL,
                    avg_duration_ms REAL,
                    avg_ttfs_ms REAL,
                    avg_vram_mb REAL,
                    success_rate REAL,
                    avg_quality_score REAL,
                    run_count INTEGER,
                    efficiency_score REAL,
                    rank INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_runs_model ON benchmark_runs(model);
                CREATE INDEX IF NOT EXISTS idx_runs_operation ON benchmark_runs(operation);
                CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON benchmark_runs(timestamp);
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

    def parse_model_info(self, model_name: str) -> Tuple[str, str, float]:
        """Parse model name to extract parameter count, quantization, size estimate."""
        # Extract parameter count
        param_match = re.search(r'(\d+\.?\d*)b', model_name.lower())
        param_count = param_match.group(1) + "B" if param_match else "unknown"

        # Extract quantization
        quant_patterns = ['q8_0', 'q4_k_m', 'q4_0', 'q5_k_m', 'q6_k', 'fp16', 'bf16', 'f16']
        quantization = "default"
        for q in quant_patterns:
            if q in model_name.lower():
                quantization = q.upper()
                break

        # Estimate size (rough)
        size_gb = 0.0
        if param_match:
            params = float(param_match.group(1))
            if 'fp16' in model_name.lower() or 'bf16' in model_name.lower():
                size_gb = params * 2  # 2 bytes per param
            elif 'q8' in model_name.lower():
                size_gb = params * 1  # 1 byte per param
            elif 'q4' in model_name.lower():
                size_gb = params * 0.5  # 0.5 bytes per param
            else:
                size_gb = params * 0.6  # default estimate

        return param_count, quantization, size_gb

    async def unload_all_models(self):
        """Unload all models from Ollama to get clean VRAM reading."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get loaded models
                resp = await client.get(f"{self.ollama_url}/api/ps")
                if resp.status_code == 200:
                    data = resp.json()
                    for model in data.get("models", []):
                        # Generate with keep_alive=0 to unload
                        await client.post(
                            f"{self.ollama_url}/api/generate",
                            json={"model": model["name"], "keep_alive": 0}
                        )
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Warning: Could not unload models: {e}")

    async def benchmark_synthesis(
        self,
        model: str,
        context: str,
        query: str,
        test_context_id: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.6,
        unload_first: bool = True
    ) -> BenchmarkResult:
        """Benchmark a model for synthesis operation."""

        # Optionally unload models first for clean measurement
        if unload_first:
            await self.unload_all_models()
            await asyncio.sleep(3)

        vram_before = self.get_gpu_stats()["used_mb"]

        # Build synthesis prompt
        prompt = f"""You are an expert technical synthesizer. Based on the following context, provide a comprehensive answer to the query.

## Context
{context}

## Query
{query}

## Instructions
- Synthesize information from the context to answer the query
- Be specific and technical
- Cite sources when available
- If information is incomplete, acknowledge limitations

## Response"""

        param_count, quantization, size_gb = self.parse_model_info(model)

        start_time = time.time()
        ttfs_ms = None
        output_text = ""
        error_msg = None
        success = False
        input_tokens = None
        output_tokens = None

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                # Use streaming to capture TTFS
                first_token_time = None

                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
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
                                    # Extract token counts from final response
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
        vram_peak = max(vram_before, vram_after)  # Approximation

        result = BenchmarkResult(
            model=model,
            operation="synthesis",
            success=success,
            ttfs_ms=ttfs_ms,
            total_duration_ms=total_duration_ms,
            vram_before_mb=vram_before,
            vram_after_mb=vram_after,
            vram_peak_mb=vram_peak,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_length=len(prompt),
            output_preview=output_text[:500] if output_text else "",
            output_quality_score=None,  # To be scored manually or by LLM-as-judge
            error_message=error_msg,
            model_size_gb=size_gb,
            quantization=quantization,
            parameter_count=param_count,
            notes=None
        )

        # Store in database
        self._store_result(result, test_context_id, query, output_text)

        return result

    def _store_result(
        self,
        result: BenchmarkResult,
        test_context_id: Optional[str],
        query: str,
        full_output: str
    ):
        """Store benchmark result in database."""
        timestamp = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO benchmark_runs (
                    timestamp, model, operation, success, ttfs_ms, total_duration_ms,
                    vram_before_mb, vram_after_mb, vram_peak_mb, input_tokens, output_tokens,
                    context_length, output_preview, output_full, output_quality_score,
                    error_message, model_size_gb, quantization, parameter_count, notes,
                    test_context_id, query
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, result.model, result.operation, 1 if result.success else 0,
                result.ttfs_ms, result.total_duration_ms, result.vram_before_mb,
                result.vram_after_mb, result.vram_peak_mb, result.input_tokens,
                result.output_tokens, result.context_length, result.output_preview,
                full_output, result.output_quality_score, result.error_message,
                result.model_size_gb, result.quantization, result.parameter_count,
                result.notes, test_context_id, query
            ))

    def add_test_context(
        self,
        context_id: str,
        name: str,
        context: str,
        query: str,
        domain: Optional[str] = None,
        expected_keywords: Optional[List[str]] = None,
        difficulty: str = "medium"
    ):
        """Add a reusable test context."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO test_contexts
                (id, name, domain, context, query, expected_keywords, difficulty, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                context_id, name, domain, context, query,
                json.dumps(expected_keywords) if expected_keywords else None,
                difficulty, datetime.utcnow().isoformat()
            ))

    def get_test_context(self, context_id: str) -> Optional[Dict]:
        """Get a test context by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM test_contexts WHERE id = ?", (context_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get("expected_keywords"):
                    result["expected_keywords"] = json.loads(result["expected_keywords"])
                return result
        return None

    def get_benchmark_results(
        self,
        model: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query benchmark results."""
        conditions = []
        params = []

        if model:
            conditions.append("model = ?")
            params.append(model)
        if operation:
            conditions.append("operation = ?")
            params.append(operation)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(f"""
                SELECT * FROM benchmark_runs
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params)
            return [dict(row) for row in cursor.fetchall()]

    def calculate_rankings(self, operation: str = "synthesis") -> List[Dict]:
        """Calculate model rankings based on benchmark data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    model,
                    COUNT(*) as run_count,
                    AVG(total_duration_ms) as avg_duration_ms,
                    AVG(ttfs_ms) as avg_ttfs_ms,
                    AVG(vram_after_mb - vram_before_mb) as avg_vram_delta_mb,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(output_quality_score) as avg_quality_score,
                    AVG(model_size_gb) as model_size_gb,
                    quantization,
                    parameter_count
                FROM benchmark_runs
                WHERE operation = ?
                GROUP BY model
                HAVING run_count >= 1
                ORDER BY avg_duration_ms ASC
            """, (operation,))

            results = []
            for i, row in enumerate(cursor.fetchall(), 1):
                r = dict(row)
                # Calculate efficiency score: quality / (duration * vram)
                duration = r["avg_duration_ms"] or 1
                vram = abs(r["avg_vram_delta_mb"] or 1000)
                quality = r["avg_quality_score"] or 0.5
                r["efficiency_score"] = (quality * 1000000) / (duration * vram) if vram > 0 else 0
                r["rank"] = i
                results.append(r)

            return results

    def print_rankings(self, operation: str = "synthesis"):
        """Print formatted rankings table."""
        rankings = self.calculate_rankings(operation)

        print(f"\n{'='*100}")
        print(f"MODEL RANKINGS FOR: {operation.upper()}")
        print(f"{'='*100}")
        print(f"{'Rank':<5} {'Model':<40} {'Duration':<12} {'TTFS':<10} {'VRAM Δ':<10} {'Success':<8} {'Quality':<8}")
        print(f"{'-'*100}")

        for r in rankings:
            duration = f"{r['avg_duration_ms']:.0f}ms" if r['avg_duration_ms'] else "N/A"
            ttfs = f"{r['avg_ttfs_ms']:.0f}ms" if r['avg_ttfs_ms'] else "N/A"
            vram = f"{r['avg_vram_delta_mb']:.0f}MB" if r['avg_vram_delta_mb'] else "N/A"
            success = f"{r['success_rate']*100:.0f}%" if r['success_rate'] else "N/A"
            quality = f"{r['avg_quality_score']:.2f}" if r['avg_quality_score'] else "N/A"

            print(f"{r['rank']:<5} {r['model']:<40} {duration:<12} {ttfs:<10} {vram:<10} {success:<8} {quality:<8}")


# Synthesis test contexts
SYNTHESIS_CONTEXTS = {
    "fanuc_servo": {
        "name": "FANUC Servo Alarm Troubleshooting",
        "domain": "industrial_robotics",
        "difficulty": "hard",
        "context": """## Retrieved Sources

### Source 1: FANUC Alarm Code Reference (fanucamerica.com)
SRVO-023: Servo motor overload alarm. This alarm indicates excessive current draw on a servo motor axis. Common causes include:
- Mechanical binding or obstruction in the axis
- Worn or damaged gearbox
- Incorrect servo amplifier parameters
- Motor insulation breakdown
- Excessive payload or acceleration settings

### Source 2: Industrial Robotics Forum (robotics.stackexchange.com)
User reported SRVO-023 on J2 axis during welding cycle. Resolution involved:
1. Check motor temperature with thermal camera
2. Measure motor resistance (should be balanced across phases)
3. Inspect gearbox for metal particles in oil
4. Review torque monitoring data in controller logs
5. Compare current draw at various speeds

### Source 3: FANUC R-30iB Maintenance Manual
For servo overload alarms:
- Access MENU > SYSTEM > MOTION > SERVO LOG
- Review disturbance torque values (normal: <30%)
- Check servo amplifier error history
- Perform motor insulation test (megger test)
- Verify brake release operation

### Source 4: Welding Robot Best Practices (aws.org)
Arc welding robots experience higher J2 loads due to:
- Torch cable routing creating drag
- Weld spatter accumulation
- Thermal expansion during long cycles
Recommended: Reduce speed to 70% for initial diagnostics.""",
        "query": "FANUC R-30iB SRVO-023 alarm on J2 axis during arc welding at 85% speed. How do I diagnose mechanical binding vs servo amplifier failure?",
        "expected_keywords": ["torque", "gearbox", "temperature", "resistance", "megger", "servo log", "disturbance"]
    },

    "plc_network": {
        "name": "PLC Network Communication Failure",
        "domain": "industrial_automation",
        "difficulty": "hard",
        "context": """## Retrieved Sources

### Source 1: Rockwell Automation Knowledgebase
Major Fault Type 01 Code 13: I/O Chassis Communication Failure
- Indicates loss of communication with remote I/O rack
- Can be triggered by: power interruption, cable fault, module failure
- Check EN2T module status LEDs (MS, NS, OK)
- Review RSLinx connection log for timeout events

### Source 2: ControlLogix Troubleshooting Guide (literature.rockwellautomation.com)
For intermittent I/O faults after power events:
1. Check capacitor health on power supplies (ESR test)
2. Verify EtherNet/IP switch port negotiation (should be 100Mbps Full)
3. Inspect backplane connections for oxidation
4. Review 1756-IF16 analog input diagnostics (underrange/overrange)
5. Check for ground loops between racks

### Source 3: PLCTalk Forum Discussion
Similar issue after power dip - turned out to be:
- EN2T firmware incompatibility after partial update
- Brown-out caused module to reset but retain corrupted settings
- Solution: Full firmware reflash and reconfiguration

### Source 4: Allen-Bradley 1756-IF16 Manual
Analog input faults can cascade to chassis communication:
- Check input range configuration vs actual signals
- Verify 4-20mA loop power is stable
- Input impedance: 249Ω internal
- Common mode voltage limit: ±10V""",
        "query": "ControlLogix 1756-L73 with fault 1:13 after power dip. EN2T and IF16 cards in rack. How to isolate the root cause?",
        "expected_keywords": ["EN2T", "firmware", "backplane", "power supply", "LED", "RSLinx", "EtherNet/IP"]
    },

    "imm_heater": {
        "name": "Injection Molding Heater Oscillation",
        "domain": "plastics_processing",
        "difficulty": "medium",
        "context": """## Retrieved Sources

### Source 1: Plastics Technology Magazine
Temperature oscillation in barrel heaters typically caused by:
- SSR (Solid State Relay) degradation - partial conduction
- Contactor chatter from voltage fluctuations
- Heater band hot spots from poor surface contact
- Thermocouple placement or grounding issues

### Source 2: Engel Machine Manual
Zone temperature PID tuning recommendations:
- Proportional band: 2-5% of setpoint
- Integral time: 20-60 seconds
- Derivative: typically 0 for heaters
If oscillation persists after tuning, check hardware.

### Source 3: Injection Molding Forum
User experienced ±15°C oscillation - root cause was:
- SCR output partially failed (firing asymmetrically)
- Diagnosed by measuring current with clamp meter
- Both half-cycles should be equal

### Source 4: Heater Band Diagnostics Guide
Resistance testing:
- Ceramic bands: 1-20Ω typical
- Compare zones - should be within 10%
- Insulation resistance: >1MΩ at 500VDC
- Visual: check for discoloration/hot spots""",
        "query": "Engel Victory 500 barrel zone 3 oscillating ±15°C around 245°C setpoint. PID tuned, thermocouple replaced. Is this SSR, contactor, or heater band?",
        "expected_keywords": ["SSR", "SCR", "clamp meter", "resistance", "asymmetric", "current", "half-cycle"]
    }
}


if __name__ == "__main__":
    import asyncio

    async def main():
        bench = ModelBenchmark()

        # Add test contexts
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
            print(f"Added test context: {ctx_id}")

        print("\nTest contexts loaded. Run benchmarks with:")
        print("  bench = ModelBenchmark()")
        print("  ctx = bench.get_test_context('fanuc_servo')")
        print("  result = await bench.benchmark_synthesis('qwen3:8b', ctx['context'], ctx['query'])")

    asyncio.run(main())
