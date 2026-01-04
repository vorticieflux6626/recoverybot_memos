#!/usr/bin/env python3
"""
Synthesis Context Builder - Generate and store contexts for model benchmarking.

Creates cross-domain reasoning queries, runs them through agentic presets,
and stores the resulting contexts for synthesis model comparison.
"""

import asyncio
import aiohttp
import json
import sqlite3
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

# Cross-domain reasoning queries requiring synthesis of multiple knowledge areas
CROSS_DOMAIN_QUERIES = {
    "plc_robot_integration": {
        "query": "Allen-Bradley ControlLogix 1756-L73 communicating with FANUC R-30iB controller via EtherNet/IP. The robot reports SRVO-023 servo overload on J2 axis but only during specific PLC-triggered motion sequences. How do I diagnose whether this is a PLC timing issue, robot parameter mismatch, or mechanical binding?",
        "domains": ["allen_bradley", "fanuc_robotics", "industrial_networking"],
        "difficulty": "expert",
        "expected_topics": ["EtherNet/IP", "SRVO alarms", "motion synchronization", "PLC scan time"]
    },
    "imm_hotrunner_thermal": {
        "query": "Husky HyPET injection molding system with Mold-Masters hot runner showing zone 3 temperature instability. The barrel zones are stable but hot runner oscillates +/-8°C. Could this be interaction between the IMM barrel heater PID loops and hot runner controller, or a thermocouple placement issue?",
        "domains": ["injection_molding", "hot_runner", "thermal_control", "pid_tuning"],
        "difficulty": "expert",
        "expected_topics": ["thermal coupling", "PID interaction", "thermocouple response", "zone isolation"]
    },
    "conveyor_plc_vfd": {
        "query": "Dorner 2200 conveyor system controlled by Siemens S7-1200 PLC via Profinet to SEW Movitrac VFD. Conveyor speed drifts during accumulation mode causing product collisions. The VFD shows no faults but actual speed deviates from setpoint. Is this a Profinet communication latency issue or VFD parameter tuning?",
        "domains": ["siemens_plc", "conveyor_systems", "vfd_drives", "profinet"],
        "difficulty": "advanced",
        "expected_topics": ["Profinet cycle time", "VFD ramp parameters", "accumulation logic", "encoder feedback"]
    },
    "robot_vision_calibration": {
        "query": "FANUC M-20iA with iRVision 2D camera for pick-and-place. Vision calibration passes but picking accuracy degrades over 8-hour shift by 2-3mm. Robot repeatability test passes. Temperature in cell rises 15°C during production. How to determine if this is thermal expansion of the camera mount, robot base, or lighting intensity drift?",
        "domains": ["fanuc_robotics", "machine_vision", "thermal_effects", "calibration"],
        "difficulty": "expert",
        "expected_topics": ["thermal compensation", "vision recalibration", "lighting consistency", "fixture expansion"]
    },
    "granulator_chiller_interaction": {
        "query": "Conair beside-the-press granulator feeding regrind to Wittmann gravimetric blender. Chiller for mold cooling cycles on/off causing voltage dips that affect granulator motor startup. This causes inconsistent regrind particle size. How to isolate whether this is electrical supply issue, motor soft-start failure, or blade wear from voltage fluctuations?",
        "domains": ["granulators", "chillers", "electrical_systems", "material_handling"],
        "difficulty": "advanced",
        "expected_topics": ["voltage sag", "soft starter", "blade sharpness", "electrical isolation"]
    },
    "dryer_hopper_dewpoint": {
        "query": "Novatec desiccant dryer showing -40°F dewpoint at dryer outlet but moisture analyzer at hopper shows -20°F. Material is hygroscopic PET requiring -30°F minimum. 50ft insulated hose connects dryer to hopper. Is the moisture pickup in the hose, hopper lid seal, or is the dryer dewpoint sensor drifting?",
        "domains": ["material_dryers", "hopper_systems", "moisture_control", "sensor_calibration"],
        "difficulty": "advanced",
        "expected_topics": ["dewpoint measurement", "hose insulation", "desiccant regeneration", "sensor drift"]
    },
    "multiaxis_cnc_vibration": {
        "query": "Mazak 5-axis machining center with FANUC 31i-B5 control showing chatter marks on finish passes in titanium. Vibration analysis shows 1.2kHz resonance. Tool is new carbide with correct geometry. Spindle bearings replaced 6 months ago. Could this be fixture rigidity, spindle balance, or adaptive control parameter issue?",
        "domains": ["cnc_machining", "fanuc_cnc", "vibration_analysis", "tooling"],
        "difficulty": "expert",
        "expected_topics": ["chatter frequency", "spindle dynamics", "fixture modes", "cutting parameters"]
    },
    "scara_encoder_drift": {
        "query": "Epson SCARA robot with incremental encoders showing position drift after 4000 cycles. Homing routine corrects it but production requires continuous operation. Battery backup is new. Similar issue on axis 1 and 2 but not 3 and 4. What diagnostic approach differentiates encoder cable noise, reducer backlash, or encoder disc contamination?",
        "domains": ["scara_robots", "encoder_systems", "motion_control", "mechanical_wear"],
        "difficulty": "advanced",
        "expected_topics": ["encoder signals", "cable shielding", "backlash measurement", "encoder cleaning"]
    }
}

# Presets to test
PRESETS_TO_TEST = ["balanced", "enhanced", "research"]


@dataclass
class SynthesisContext:
    """A stored synthesis context for benchmarking."""
    context_id: str
    query: str
    domains: list
    difficulty: str
    preset_used: str
    retrieved_context: str  # The context retrieved by the pipeline
    source_count: int
    source_urls: list
    retrieval_duration_ms: float
    timestamp: str
    expected_topics: list


class SynthesisContextDB:
    """Database for storing synthesis contexts."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent / "synthesis_contexts.db"
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Synthesis contexts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS synthesis_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT UNIQUE NOT NULL,
                query TEXT NOT NULL,
                domains TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                preset_used TEXT NOT NULL,
                retrieved_context TEXT NOT NULL,
                source_count INTEGER,
                source_urls TEXT,
                retrieval_duration_ms REAL,
                timestamp TEXT NOT NULL,
                expected_topics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Synthesis benchmark results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS synthesis_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                model TEXT NOT NULL,
                synthesis_output TEXT,
                duration_ms REAL,
                ttfs_ms REAL,
                output_tokens INTEGER,
                thinking_tokens INTEGER,
                vram_before_mb INTEGER,
                vram_after_mb INTEGER,
                topic_coverage_score REAL,
                factual_density_score REAL,
                coherence_score REAL,
                overall_score REAL,
                success INTEGER,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (context_id) REFERENCES synthesis_contexts(context_id)
            )
        """)

        conn.commit()
        conn.close()

    def store_context(self, ctx: SynthesisContext) -> bool:
        """Store a synthesis context."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT OR REPLACE INTO synthesis_contexts
                (context_id, query, domains, difficulty, preset_used, retrieved_context,
                 source_count, source_urls, retrieval_duration_ms, timestamp, expected_topics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ctx.context_id,
                ctx.query,
                json.dumps(ctx.domains),
                ctx.difficulty,
                ctx.preset_used,
                ctx.retrieved_context,
                ctx.source_count,
                json.dumps(ctx.source_urls),
                ctx.retrieval_duration_ms,
                ctx.timestamp,
                json.dumps(ctx.expected_topics)
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing context: {e}")
            return False
        finally:
            conn.close()

    def get_context(self, context_id: str) -> Optional[dict]:
        """Get a synthesis context by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM synthesis_contexts WHERE context_id = ?", (context_id,))
        row = cur.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_all_contexts(self) -> list:
        """Get all stored contexts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM synthesis_contexts ORDER BY created_at DESC")
        rows = cur.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def store_benchmark_result(self, result: dict) -> bool:
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

    def get_benchmark_rankings(self) -> list:
        """Get benchmark rankings by model."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT
                model,
                COUNT(*) as test_count,
                AVG(duration_ms) as avg_duration_ms,
                AVG(ttfs_ms) as avg_ttfs_ms,
                AVG(output_tokens) as avg_tokens,
                AVG(topic_coverage_score) as avg_topic_score,
                AVG(overall_score) as avg_overall_score,
                SUM(success) as successes
            FROM synthesis_benchmarks
            GROUP BY model
            ORDER BY avg_overall_score DESC
        """)
        rows = cur.fetchall()
        conn.close()

        return [dict(row) for row in rows]


async def fetch_synthesis_context(
    query: str,
    preset: str,
    server_url: str = "http://localhost:8001"
) -> dict:
    """Fetch synthesis context from the agentic search pipeline."""

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        try:
            # Use the universal search endpoint
            async with session.post(
                f"{server_url}/api/v1/search/universal",
                json={
                    "query": query,
                    "preset": preset,
                    "max_results": 15,
                    "include_sources": True
                },
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    duration_ms = (time.time() - start_time) * 1000

                    # Extract context from response
                    result = data.get('data', data)

                    return {
                        "success": True,
                        "context": result.get('synthesized_context', result.get('context', '')),
                        "sources": result.get('sources', []),
                        "source_count": len(result.get('sources', [])),
                        "duration_ms": duration_ms
                    }
                else:
                    error_text = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {error_text[:200]}"
                    }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Timeout after 300s"}
        except Exception as e:
            return {"success": False, "error": str(e)}


async def build_synthesis_contexts(
    queries: dict = None,
    presets: list = None,
    server_url: str = "http://localhost:8001"
) -> list:
    """Build synthesis contexts by running queries through presets."""

    if queries is None:
        queries = CROSS_DOMAIN_QUERIES
    if presets is None:
        presets = PRESETS_TO_TEST

    db = SynthesisContextDB()
    contexts_created = []

    for query_id, query_data in queries.items():
        for preset in presets:
            context_id = f"{query_id}_{preset}"

            print(f"\n{'='*80}")
            print(f"Building context: {context_id}")
            print(f"Query: {query_data['query'][:80]}...")
            print(f"Preset: {preset}")
            print(f"{'='*80}")

            result = await fetch_synthesis_context(
                query=query_data['query'],
                preset=preset,
                server_url=server_url
            )

            if result['success']:
                ctx = SynthesisContext(
                    context_id=context_id,
                    query=query_data['query'],
                    domains=query_data['domains'],
                    difficulty=query_data['difficulty'],
                    preset_used=preset,
                    retrieved_context=result['context'],
                    source_count=result['source_count'],
                    source_urls=[s.get('url', '') for s in result.get('sources', [])],
                    retrieval_duration_ms=result['duration_ms'],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    expected_topics=query_data.get('expected_topics', [])
                )

                if db.store_context(ctx):
                    contexts_created.append(context_id)
                    print(f"  ✓ Stored: {len(result['context'])} chars, {result['source_count']} sources")
                    print(f"  Duration: {result['duration_ms']:.0f}ms")
                else:
                    print(f"  ✗ Failed to store context")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

            # Brief delay between requests
            await asyncio.sleep(2)

    return contexts_created


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Build synthesis contexts for benchmarking")
    parser.add_argument("--query", type=str, help="Specific query ID to build")
    parser.add_argument("--preset", type=str, help="Specific preset to use")
    parser.add_argument("--list", action="store_true", help="List available queries")
    parser.add_argument("--show-contexts", action="store_true", help="Show stored contexts")
    parser.add_argument("--server", type=str, default="http://localhost:8001", help="Server URL")
    args = parser.parse_args()

    if args.list:
        print("Available cross-domain queries:")
        for qid, qdata in CROSS_DOMAIN_QUERIES.items():
            print(f"\n  {qid}:")
            print(f"    Domains: {', '.join(qdata['domains'])}")
            print(f"    Difficulty: {qdata['difficulty']}")
            print(f"    Query: {qdata['query'][:100]}...")
        return

    if args.show_contexts:
        db = SynthesisContextDB()
        contexts = db.get_all_contexts()
        print(f"\nStored contexts: {len(contexts)}")
        for ctx in contexts:
            print(f"\n  {ctx['context_id']}:")
            print(f"    Preset: {ctx['preset_used']}")
            print(f"    Context length: {len(ctx['retrieved_context'])} chars")
            print(f"    Sources: {ctx['source_count']}")
        return

    # Build contexts
    queries = CROSS_DOMAIN_QUERIES
    presets = PRESETS_TO_TEST

    if args.query:
        if args.query not in queries:
            print(f"Unknown query: {args.query}")
            return
        queries = {args.query: queries[args.query]}

    if args.preset:
        presets = [args.preset]

    print(f"Building synthesis contexts...")
    print(f"Queries: {list(queries.keys())}")
    print(f"Presets: {presets}")

    created = await build_synthesis_contexts(
        queries=queries,
        presets=presets,
        server_url=args.server
    )

    print(f"\n{'='*80}")
    print(f"Created {len(created)} contexts:")
    for ctx_id in created:
        print(f"  - {ctx_id}")


if __name__ == "__main__":
    asyncio.run(main())
