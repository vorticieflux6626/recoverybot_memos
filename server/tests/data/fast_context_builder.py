#!/usr/bin/env python3
"""
Fast Context Builder - Extract search contexts quickly for model benchmarking.

Bypasses full synthesis pipeline to collect raw search contexts faster.
"""

import asyncio
import aiohttp
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

# Cross-domain reasoning queries
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
    }
}


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

    def store_context(self, context_data: dict) -> bool:
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
                context_data['context_id'],
                context_data['query'],
                json.dumps(context_data['domains']),
                context_data['difficulty'],
                context_data['preset_used'],
                context_data['retrieved_context'],
                context_data['source_count'],
                json.dumps(context_data['source_urls']),
                context_data['retrieval_duration_ms'],
                context_data['timestamp'],
                json.dumps(context_data.get('expected_topics', []))
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing context: {e}")
            return False
        finally:
            conn.close()

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


async def search_web(query: str, max_results: int = 10) -> list:
    """Quick web search via SearXNG."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "http://localhost:8888/search",
                params={
                    "q": query,
                    "format": "json",
                    "engines": "brave,bing,reddit",
                    "max_results": max_results
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('results', [])
        except Exception as e:
            print(f"SearXNG search failed: {e}")
    return []


async def scrape_content(url: str, max_chars: int = 8000) -> str:
    """Scrape content from URL."""
    async with aiohttp.ClientSession() as session:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
                allow_redirects=True
            ) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    # Basic HTML stripping
                    import re
                    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text[:max_chars]
        except Exception as e:
            pass
    return ""


async def build_context(query_id: str, query_data: dict, preset: str = "balanced") -> dict:
    """Build synthesis context for a query."""

    print(f"\n{'='*60}")
    print(f"Building: {query_id} ({preset})")
    print(f"Query: {query_data['query'][:60]}...")
    print(f"{'='*60}")

    start_time = time.time()

    # Search for results
    print("  Searching...")
    results = await search_web(query_data['query'], max_results=12)

    if not results:
        print("  No search results!")
        return None

    print(f"  Found {len(results)} results")

    # Scrape top results
    print("  Scraping content...")
    sources = []
    source_urls = []

    for r in results[:8]:  # Top 8 results
        url = r.get('url', '')
        title = r.get('title', '')

        if not url:
            continue

        content = await scrape_content(url)
        if content and len(content) > 200:
            sources.append({
                'url': url,
                'title': title,
                'content': content
            })
            source_urls.append(url)
            print(f"    ✓ {title[:50]}... ({len(content)} chars)")

        if len(sources) >= 6:
            break

    if not sources:
        print("  No content scraped!")
        return None

    # Build context string
    context_parts = []
    for s in sources:
        context_parts.append(f"### {s['title']}\nSource: {s['url']}\n\n{s['content']}\n")

    retrieved_context = "\n---\n".join(context_parts)
    duration_ms = (time.time() - start_time) * 1000

    print(f"  Context: {len(retrieved_context)} chars from {len(sources)} sources")
    print(f"  Duration: {duration_ms:.0f}ms")

    return {
        'context_id': f"{query_id}_{preset}",
        'query': query_data['query'],
        'domains': query_data['domains'],
        'difficulty': query_data['difficulty'],
        'preset_used': preset,
        'retrieved_context': retrieved_context,
        'source_count': len(sources),
        'source_urls': source_urls,
        'retrieval_duration_ms': duration_ms,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'expected_topics': query_data.get('expected_topics', [])
    }


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fast context builder for benchmarking")
    parser.add_argument("--query", type=str, help="Specific query ID")
    parser.add_argument("--preset", type=str, default="balanced", help="Preset name")
    parser.add_argument("--all", action="store_true", help="Build all contexts")
    parser.add_argument("--list", action="store_true", help="List queries")
    parser.add_argument("--show", action="store_true", help="Show stored contexts")
    args = parser.parse_args()

    db = SynthesisContextDB()

    if args.list:
        print("Available queries:")
        for qid, qdata in CROSS_DOMAIN_QUERIES.items():
            print(f"\n  {qid}:")
            print(f"    Domains: {', '.join(qdata['domains'])}")
            print(f"    Difficulty: {qdata['difficulty']}")
        return

    if args.show:
        contexts = db.get_all_contexts()
        print(f"\nStored contexts: {len(contexts)}")
        for ctx in contexts:
            print(f"\n  {ctx['context_id']}:")
            print(f"    Sources: {ctx['source_count']}")
            print(f"    Context: {len(ctx['retrieved_context'])} chars")
        return

    if args.all:
        queries = CROSS_DOMAIN_QUERIES
    elif args.query:
        if args.query not in CROSS_DOMAIN_QUERIES:
            print(f"Unknown query: {args.query}")
            return
        queries = {args.query: CROSS_DOMAIN_QUERIES[args.query]}
    else:
        # Build first 3 queries
        queries = dict(list(CROSS_DOMAIN_QUERIES.items())[:3])

    created = 0
    for qid, qdata in queries.items():
        ctx = await build_context(qid, qdata, args.preset)
        if ctx and db.store_context(ctx):
            created += 1
            print(f"  ✓ Stored: {ctx['context_id']}")

    print(f"\n{'='*60}")
    print(f"Created {created} contexts")


if __name__ == "__main__":
    asyncio.run(main())
