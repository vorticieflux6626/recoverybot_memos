#!/usr/bin/env python3
"""
Preset Timeout Test Suite

Tests each orchestrator preset to measure execution time and validate timeout settings.

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python tests/test_preset_timeouts.py [preset]

    # Test specific preset
    python tests/test_preset_timeouts.py minimal
    python tests/test_preset_timeouts.py balanced

    # Test all presets
    python tests/test_preset_timeouts.py all

Created: 2026-01-02
"""

import asyncio
import sys
import time
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("preset_test")

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add parent to path
sys.path.insert(0, '/home/sparkone/sdd/Recovery_Bot/memOS/server')

# Test query - industrial troubleshooting for realistic timing
TEST_QUERY = "FANUC R-30iB SRVO-023 collision detection alarm troubleshooting"

# Expected timeout ranges per preset (in seconds)
# These are EXPECTED execution times, not hard limits
EXPECTED_TIMES = {
    "minimal": (20, 90),      # 20-90 seconds
    "balanced": (60, 180),    # 1-3 minutes
    "enhanced": (120, 300),   # 2-5 minutes
    "research": (180, 600),   # 3-10 minutes
    "full": (300, 900),       # 5-15 minutes
}


async def test_preset(preset_name: str) -> Dict[str, Any]:
    """
    Test a single preset and measure execution time.

    Returns dict with timing and quality metrics.
    """
    from agentic import UniversalOrchestrator
    from agentic.models import SearchRequest
    from agentic.orchestrator_universal import OrchestratorPreset

    # Get preset enum
    preset = OrchestratorPreset(preset_name)
    expected_min, expected_max = EXPECTED_TIMES.get(preset_name, (60, 300))

    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING PRESET: {preset_name.upper()}")
    logger.info(f"Expected time: {expected_min}-{expected_max}s")
    logger.info(f"{'='*60}")

    result = {
        "preset": preset_name,
        "success": False,
        "duration_s": 0,
        "within_expected": False,
        "synthesis_length": 0,
        "confidence": 0,
        "citations": 0,
        "on_topic_terms": [],
        "error": None
    }

    try:
        orchestrator = UniversalOrchestrator(preset=preset)

        request = SearchRequest(
            query=TEST_QUERY,
            max_results=5,
            enable_scraping=False,
            request_id=f"preset-test-{preset_name}"
        )

        logger.info(f"Starting search...")
        start = time.time()
        search_result = await orchestrator.search(request)
        duration = time.time() - start

        # Extract results
        synthesis = ""
        if hasattr(search_result, 'synthesized_context'):
            synthesis = search_result.synthesized_context or ""
        elif hasattr(search_result, 'synthesis'):
            synthesis = search_result.synthesis or ""
        else:
            synthesis = str(search_result)[:2000]

        confidence = 0
        if hasattr(search_result, 'confidence_score'):
            confidence = search_result.confidence_score or 0
        elif hasattr(search_result, 'confidence'):
            confidence = search_result.confidence or 0

        # Check on-topic terms
        synthesis_lower = synthesis.lower()
        on_topic_terms = ['srvo-023', 'collision', 'detection', 'r-30ib', 'fanuc']
        found_terms = [t for t in on_topic_terms if t in synthesis_lower]

        # Count citations
        import re
        citations = re.findall(r'\[Source \d+\]|\[Domain Knowledge\]', synthesis)

        result.update({
            "success": True,
            "duration_s": round(duration, 1),
            "within_expected": expected_min <= duration <= expected_max,
            "synthesis_length": len(synthesis),
            "confidence": round(confidence * 100) if confidence else 0,
            "citations": len(citations),
            "on_topic_terms": found_terms
        })

        # Log results
        logger.info(f"\n--- Results for {preset_name.upper()} ---")
        logger.info(f"Duration: {duration:.1f}s (expected: {expected_min}-{expected_max}s)")
        logger.info(f"Within expected: {'✅ YES' if result['within_expected'] else '❌ NO'}")
        logger.info(f"Confidence: {result['confidence']}%")
        logger.info(f"Synthesis: {result['synthesis_length']} chars")
        logger.info(f"Citations: {result['citations']}")
        logger.info(f"On-topic terms: {found_terms}")

        if duration > expected_max:
            logger.warning(f"⚠️  SLOWER than expected by {duration - expected_max:.1f}s")
        elif duration < expected_min:
            logger.info(f"⚡ FASTER than expected by {expected_min - duration:.1f}s")

        # Show synthesis preview
        logger.info(f"\n--- Synthesis Preview ---")
        preview = synthesis[:600] + "..." if len(synthesis) > 600 else synthesis
        print(preview)

    except asyncio.TimeoutError:
        result["error"] = "TIMEOUT"
        logger.error(f"❌ TIMEOUT after expected max time")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    return result


async def test_all_presets():
    """Test all presets sequentially."""
    results = []

    for preset_name in ["minimal", "balanced", "enhanced", "research", "full"]:
        result = await test_preset(preset_name)
        results.append(result)

        # Brief pause between tests
        if preset_name != "full":
            logger.info("\nWaiting 10s before next preset...")
            await asyncio.sleep(10)

    # Summary
    print("\n" + "="*70)
    print("PRESET TIMEOUT TEST SUMMARY")
    print("="*70)
    print(f"{'Preset':<12} {'Duration':>10} {'Expected':>15} {'Status':>10} {'Confidence':>12}")
    print("-"*70)

    for r in results:
        expected = EXPECTED_TIMES.get(r['preset'], (0, 0))
        expected_str = f"{expected[0]}-{expected[1]}s"
        status = "✅ OK" if r['success'] and r['within_expected'] else "⚠️ SLOW" if r['success'] else "❌ FAIL"
        conf = f"{r['confidence']}%" if r['success'] else "N/A"
        dur = f"{r['duration_s']}s" if r['success'] else "N/A"
        print(f"{r['preset']:<12} {dur:>10} {expected_str:>15} {status:>10} {conf:>12}")

    print("="*70)

    # Recommendations
    print("\nRECOMMENDATIONS:")
    for r in results:
        if r['success'] and not r['within_expected']:
            expected_max = EXPECTED_TIMES[r['preset']][1]
            if r['duration_s'] > expected_max:
                recommended = int(r['duration_s'] * 1.3)  # 30% buffer
                print(f"  - {r['preset']}: Increase timeout to {recommended}s (current max: {expected_max}s)")

    return results


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        preset = sys.argv[1].lower()
        if preset == "all":
            await test_all_presets()
        elif preset in EXPECTED_TIMES:
            await test_preset(preset)
        else:
            print(f"Unknown preset: {preset}")
            print(f"Available: {', '.join(EXPECTED_TIMES.keys())}, all")
            sys.exit(1)
    else:
        print("Usage: python test_preset_timeouts.py [preset|all]")
        print(f"Available presets: {', '.join(EXPECTED_TIMES.keys())}")
        print("\nRunning MINIMAL preset as default...")
        await test_preset("minimal")


if __name__ == "__main__":
    asyncio.run(main())
