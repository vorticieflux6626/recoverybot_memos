#!/usr/bin/env python3
"""
Quick comparison of models for URL evaluation task.

Tests ministral-3:3b vs qwen3:8b on the evaluate_urls_for_scraping task.
"""

import asyncio
import time
import json
import httpx
from typing import List, Dict, Any

OLLAMA_URL = "http://localhost:11434"

# Sample search results to evaluate
SAMPLE_RESULTS = [
    {
        "title": "FANUC SRVO-023 Servo Overload - Troubleshooting Guide",
        "url": "https://fanucamerica.com/support/alarms/srvo-023",
        "snippet": "SRVO-023 indicates excessive current draw on a servo motor. Common causes include mechanical binding, worn gearbox, or incorrect amplifier parameters.",
        "source_domain": "fanucamerica.com"
    },
    {
        "title": "Buy Servo Motors - Industrial Supply Co",
        "url": "https://industrialsupply.com/servo-motors",
        "snippet": "Shop our wide selection of servo motors. Free shipping on orders over $500. Best prices guaranteed.",
        "source_domain": "industrialsupply.com"
    },
    {
        "title": "SRVO-023 Fix - Robotics Forum Discussion",
        "url": "https://robotics.stackexchange.com/questions/12345/srvo-023-j2-axis",
        "snippet": "User reported SRVO-023 on J2 axis during welding. Resolution involved checking motor temperature and measuring resistance across phases.",
        "source_domain": "robotics.stackexchange.com"
    },
    {
        "title": "FANUC R-30iB Controller Manual PDF",
        "url": "https://fanuc.co.jp/manuals/r30ib-maintenance.pdf",
        "snippet": "Complete maintenance manual for FANUC R-30iB robot controller. Includes alarm codes, servo diagnostics, and calibration procedures.",
        "source_domain": "fanuc.co.jp"
    },
    {
        "title": "What is a Servo Motor? - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Servo_motor",
        "snippet": "A servomotor is a rotary actuator that allows for precise control of angular position, velocity and acceleration.",
        "source_domain": "en.wikipedia.org"
    },
    {
        "title": "Login Required - TechSupport Portal",
        "url": "https://techsupport.example.com/login",
        "snippet": "Please login to access technical documentation and support resources.",
        "source_domain": "techsupport.example.com"
    },
    {
        "title": "Servo Amplifier Error Codes - Allen Bradley",
        "url": "https://literature.rockwellautomation.com/servo-errors",
        "snippet": "Reference guide for Kinetix servo amplifier fault codes. Includes diagnostic steps and corrective actions.",
        "source_domain": "literature.rockwellautomation.com"
    },
    {
        "title": "Amazon.com: Servo Motor Parts",
        "url": "https://amazon.com/servo-motor-parts/dp/B08XYZ",
        "snippet": "Replacement servo motor parts and accessories. Prime shipping available.",
        "source_domain": "amazon.com"
    }
]

QUERY = "FANUC R-30iB SRVO-023 alarm on J2 axis - how to diagnose?"


def build_prompt(query: str, results: List[Dict]) -> str:
    """Build the URL evaluation prompt."""
    results_summary = []
    for i, result in enumerate(results):
        title = result.get("title", "Unknown")
        url = result.get("url", "")
        snippet = result.get("snippet", "")[:200]
        domain = result.get("source_domain", result.get("domain", ""))

        results_summary.append(
            f"{i+1}. **{title}**\n"
            f"   URL: {url}\n"
            f"   Domain: {domain}\n"
            f"   Snippet: {snippet}"
        )

    results_text = "\n\n".join(results_summary)

    return f"""Evaluate which of these search results are worth scraping to answer the user's question.

USER'S QUESTION: {query}

SEARCH RESULTS:
{results_text}

For each result, determine if it's likely to contain USEFUL INFORMATION to answer the question.

Consider:
- Does the title/snippet suggest relevant content?
- Is it from a credible/authoritative source for this topic?
- Will scraping this page likely provide actionable information?
- Avoid: generic landing pages, login walls, paywalls, irrelevant topics

Return a JSON array of objects for URLs WORTH SCRAPING (only include relevant ones):
[
  {{"index": 1, "url": "...", "relevance": "high/medium", "reason": "why this is relevant"}},
  {{"index": 3, "url": "...", "relevance": "high/medium", "reason": "why this is relevant"}}
]

Only include URLs with HIGH or MEDIUM relevance. Skip low relevance or irrelevant results.
Return ONLY the JSON array, no other text. /no_think"""


async def test_model(model: str, prompt: str, runs: int = 3) -> Dict[str, Any]:
    """Test a model on URL evaluation."""
    results = []

    for i in range(runs):
        start = time.time()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2048
                    }
                }
            )
            data = response.json()

        duration = (time.time() - start) * 1000
        output = data.get("response", "")

        # Try to parse JSON
        try:
            import re
            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                valid_json = True
                num_urls = len(parsed)
            else:
                valid_json = False
                num_urls = 0
        except:
            valid_json = False
            num_urls = 0

        results.append({
            "run": i + 1,
            "duration_ms": duration,
            "valid_json": valid_json,
            "num_urls_selected": num_urls,
            "output_length": len(output),
            "output_preview": output[:300]
        })

        print(f"  Run {i+1}: {duration:.0f}ms, valid={valid_json}, urls={num_urls}")

    # Calculate averages
    avg_duration = sum(r["duration_ms"] for r in results) / len(results)
    success_rate = sum(1 for r in results if r["valid_json"]) / len(results)
    avg_urls = sum(r["num_urls_selected"] for r in results) / len(results)

    return {
        "model": model,
        "avg_duration_ms": avg_duration,
        "success_rate": success_rate,
        "avg_urls_selected": avg_urls,
        "runs": results
    }


async def main():
    """Run comparison."""
    models = ["ministral-3:3b", "qwen3:8b"]
    prompt = build_prompt(QUERY, SAMPLE_RESULTS)

    print(f"Prompt length: {len(prompt)} chars")
    print(f"Testing URL evaluation with {len(SAMPLE_RESULTS)} search results")
    print(f"Query: {QUERY}\n")

    # Expected: Should select indices 1, 3, 4 (FANUC docs, forum, manual)
    # Should skip: 2 (shopping), 5 (generic Wikipedia), 6 (login wall), 7 (Allen-Bradley - different brand), 8 (Amazon)
    print("Expected relevant URLs: 1, 3, 4 (possibly 7)")
    print("Expected irrelevant: 2, 5, 6, 8\n")

    all_results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing: {model}")
        print(f"{'='*60}")

        result = await test_model(model, prompt, runs=3)
        all_results.append(result)

        print(f"\nSummary for {model}:")
        print(f"  Avg duration: {result['avg_duration_ms']:.0f}ms")
        print(f"  Success rate: {result['success_rate']*100:.0f}%")
        print(f"  Avg URLs selected: {result['avg_urls_selected']:.1f}")

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Duration':<12} {'Success':<10} {'URLs':<10}")
    print(f"{'-'*52}")
    for r in all_results:
        print(f"{r['model']:<20} {r['avg_duration_ms']:.0f}ms{'':<5} {r['success_rate']*100:.0f}%{'':<6} {r['avg_urls_selected']:.1f}")

    # Speed comparison
    if len(all_results) >= 2:
        speedup = all_results[1]["avg_duration_ms"] / all_results[0]["avg_duration_ms"]
        print(f"\n{all_results[0]['model']} is {speedup:.1f}x faster than {all_results[1]['model']}")

    # Show last output from each model for quality comparison
    print(f"\n{'='*60}")
    print("LAST OUTPUT SAMPLES (for quality check)")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n--- {r['model']} ---")
        print(r["runs"][-1]["output_preview"])


if __name__ == "__main__":
    asyncio.run(main())
