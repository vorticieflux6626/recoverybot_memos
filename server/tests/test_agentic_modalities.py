#!/usr/bin/env python3
"""
Agentic Modality Test Suite
===========================

Tests each agentic "thought" modality individually to audit implementation
and integration quality.

Modalities tested:
1. Query Understanding (query_analysis, entity_tracking)
2. Query Expansion (hyde, query_tree, flare_retrieval)
3. Search Optimization (hybrid_reranking, cross_encoder)
4. Pre-Synthesis Quality (crag_evaluation, context_curation)
5. Synthesis Enhancement (thought_library, reasoning_dag, reasoning_composer)
6. Post-Synthesis Quality (self_reflection, ragas, verification)
7. Iteration Control (adaptive_refinement, entropy_halting)
8. Planning (dynamic_planning, pre_act_planning)
9. Memory Systems (semantic_memory, semantic_cache, memory_tiers)
10. Domain Knowledge (domain_corpus, technical_docs)

Usage:
    python tests/test_agentic_modalities.py [--modality NAME]
"""

import asyncio
import aiohttp
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# ============================================================================
# MODALITY DEFINITIONS
# ============================================================================

@dataclass
class ModalityConfig:
    """Configuration for testing a specific modality."""
    name: str
    description: str
    feature_flags: Dict[str, bool]
    test_query: str
    expected_behaviors: List[str]
    category: str

# Base minimal config - only essential features
MINIMAL_CONFIG = {
    "enable_query_analysis": True,
    "enable_verification": False,
    "enable_scratchpad": True,
    "enable_self_reflection": False,
    "enable_crag_evaluation": False,
    "enable_experience_distillation": False,
    "enable_classifier_feedback": False,
    "enable_adaptive_refinement": False,
    "enable_content_cache": False,
    "enable_semantic_cache": False,
    "enable_hyde": False,
    "enable_hybrid_reranking": False,
    "enable_ragas": False,
    "enable_entropy_halting": False,
    "enable_flare_retrieval": False,
    "enable_query_tree": False,
    "enable_semantic_memory": False,
    "enable_meta_buffer": False,
    "enable_reasoning_composer": False,
    "enable_entity_tracking": False,
    "enable_thought_library": False,
    "enable_reasoning_dag": False,
    "enable_domain_corpus": False,
    "enable_technical_docs": False,
    "enable_dynamic_planning": False,
    "enable_pre_act_planning": False,
    "enable_actor_factory": False,
    "enable_dylan_agent_skipping": False,
    "enable_information_bottleneck": False,
    "enable_contrastive_learning": False,
    "enable_context_curation": False,
    "enable_cross_encoder": False,
}

# Define modalities with their specific feature flags
MODALITIES: Dict[str, ModalityConfig] = {
    # ========== 1. QUERY UNDERSTANDING ==========
    "query_analysis": ModalityConfig(
        name="Query Analysis",
        description="LLM-based query understanding with topic extraction and complexity estimation",
        feature_flags={**MINIMAL_CONFIG, "enable_query_analysis": True},
        test_query="What are the main differences between FANUC R-30iA and R-30iB controllers?",
        expected_behaviors=["key_topics extracted", "complexity estimated", "requires_search determined"],
        category="Query Understanding"
    ),

    "entity_tracking": ModalityConfig(
        name="GSW Entity Tracking",
        description="Extract entities (actors, roles, states) from content for context compression",
        feature_flags={**MINIMAL_CONFIG, "enable_entity_tracking": True},
        test_query="The FANUC M-710iC robot with R-30iB controller uses αiS servo motors for J1-J3 axes",
        expected_behaviors=["entities extracted", "relations identified", "context compressed"],
        category="Query Understanding"
    ),

    # ========== 2. QUERY EXPANSION ==========
    "hyde": ModalityConfig(
        name="HyDE Query Expansion",
        description="Generate hypothetical documents to bridge query-document semantic gap",
        feature_flags={**MINIMAL_CONFIG, "enable_hyde": True},
        test_query="How to troubleshoot intermittent encoder faults?",
        expected_behaviors=["hypothetical document generated", "embedding created", "better retrieval"],
        category="Query Expansion"
    ),

    "query_tree": ModalityConfig(
        name="RQ-RAG Query Tree",
        description="Tree-based query decomposition for parallel exploration",
        feature_flags={**MINIMAL_CONFIG, "enable_query_tree": True, "enable_crag_evaluation": True},
        test_query="Compare servo tuning approaches for high-speed vs precision applications",
        expected_behaviors=["query variations generated", "parallel retrieval", "aggregated results"],
        category="Query Expansion"
    ),

    "flare_retrieval": ModalityConfig(
        name="FLARE Active Retrieval",
        description="Forward-looking retrieval triggered by synthesis uncertainty",
        feature_flags={**MINIMAL_CONFIG, "enable_flare_retrieval": True},
        test_query="What are the specific steps for FANUC collision guard setup with DCS?",
        expected_behaviors=["uncertainty detected", "proactive retrieval", "synthesis augmented"],
        category="Query Expansion"
    ),

    # ========== 3. SEARCH OPTIMIZATION ==========
    "hybrid_reranking": ModalityConfig(
        name="BGE-M3 Hybrid Reranking",
        description="Dense + sparse fusion with reciprocal rank fusion",
        feature_flags={**MINIMAL_CONFIG, "enable_hybrid_reranking": True},
        test_query="SRVO-062 alarm troubleshooting procedure",
        expected_behaviors=["dense scores", "sparse BM25 scores", "RRF fusion"],
        category="Search Optimization"
    ),

    "cross_encoder": ModalityConfig(
        name="Cross-Encoder Reranking",
        description="LLM-based cross-encoder for high-precision reranking (+28% NDCG)",
        feature_flags={**MINIMAL_CONFIG, "enable_cross_encoder": True},
        test_query="Best practices for robot TCP calibration accuracy",
        expected_behaviors=["cross-encoder scores", "reranked results", "improved precision"],
        category="Search Optimization"
    ),

    # ========== 4. PRE-SYNTHESIS QUALITY ==========
    "crag_evaluation": ModalityConfig(
        name="CRAG Retrieval Evaluation",
        description="Pre-synthesis quality check with corrective actions",
        feature_flags={**MINIMAL_CONFIG, "enable_crag_evaluation": True},
        test_query="What causes MOTN-023 alarm and how to resolve it?",
        expected_behaviors=["quality assessed", "corrective action", "refined queries if needed"],
        category="Pre-Synthesis Quality"
    ),

    "context_curation": ModalityConfig(
        name="DIG Context Curation",
        description="Document Information Gain scoring for context filtering",
        feature_flags={**MINIMAL_CONFIG, "enable_context_curation": True},
        test_query="Comprehensive guide to robot path optimization strategies",
        expected_behaviors=["DIG scores", "redundancy removed", "coverage tracked"],
        category="Pre-Synthesis Quality"
    ),

    "information_bottleneck": ModalityConfig(
        name="Information Bottleneck Filtering",
        description="IB theory-based noise filtering for cleaner context",
        feature_flags={**MINIMAL_CONFIG, "enable_information_bottleneck": True},
        test_query="Key parameters affecting robot repeatability and accuracy",
        expected_behaviors=["IB scores", "noise filtered", "essential content retained"],
        category="Pre-Synthesis Quality"
    ),

    # ========== 5. SYNTHESIS ENHANCEMENT ==========
    "thought_library": ModalityConfig(
        name="Buffer of Thoughts",
        description="Reusable reasoning templates from successful searches",
        feature_flags={**MINIMAL_CONFIG, "enable_thought_library": True},
        test_query="Analyze root causes of servo motor overheating",
        expected_behaviors=["template retrieved", "reasoning structured", "pattern applied"],
        category="Synthesis Enhancement"
    ),

    "reasoning_dag": ModalityConfig(
        name="Graph of Thoughts DAG",
        description="Multi-path reasoning with branching and aggregation",
        feature_flags={**MINIMAL_CONFIG, "enable_reasoning_dag": True},
        test_query="Evaluate trade-offs between different robot safety systems",
        expected_behaviors=["DAG created", "paths explored", "conclusions aggregated"],
        category="Synthesis Enhancement"
    ),

    "reasoning_composer": ModalityConfig(
        name="Self-Discover Reasoning",
        description="Compose task-specific reasoning from atomic modules",
        feature_flags={**MINIMAL_CONFIG, "enable_reasoning_composer": True},
        test_query="Design a systematic approach to diagnose intermittent faults",
        expected_behaviors=["modules selected", "strategy composed", "reasoning adapted"],
        category="Synthesis Enhancement"
    ),

    "meta_buffer": ModalityConfig(
        name="Meta-Buffer Templates",
        description="Cross-session template persistence and retrieval",
        feature_flags={**MINIMAL_CONFIG, "enable_meta_buffer": True},
        test_query="Standard methodology for servo gain tuning",
        expected_behaviors=["template searched", "past success leveraged", "template applied"],
        category="Synthesis Enhancement"
    ),

    # ========== 6. POST-SYNTHESIS QUALITY ==========
    "self_reflection": ModalityConfig(
        name="Self-RAG Reflection",
        description="ISREL/ISSUP/ISUSE evaluation with temporal validation",
        feature_flags={**MINIMAL_CONFIG, "enable_self_reflection": True},
        test_query="When was the FANUC R-30iB Plus controller released and what are its key features?",
        expected_behaviors=["relevance scored", "support checked", "temporal validated"],
        category="Post-Synthesis Quality"
    ),

    "ragas": ModalityConfig(
        name="RAGAS Evaluation",
        description="Reference-free RAG quality assessment",
        feature_flags={**MINIMAL_CONFIG, "enable_ragas": True},
        test_query="Explain the purpose and function of FANUC DCS software options",
        expected_behaviors=["faithfulness scored", "relevancy scored", "context precision"],
        category="Post-Synthesis Quality"
    ),

    "verification": ModalityConfig(
        name="Claim Verification",
        description="Cross-check claims against sources",
        feature_flags={**MINIMAL_CONFIG, "enable_verification": True},
        test_query="What is the maximum payload capacity of FANUC M-20iD/25?",
        expected_behaviors=["claims extracted", "verification attempted", "confidence adjusted"],
        category="Post-Synthesis Quality"
    ),

    # ========== 7. ITERATION CONTROL ==========
    "adaptive_refinement": ModalityConfig(
        name="Adaptive Refinement Loop",
        description="Gap detection and iterative improvement",
        feature_flags={**MINIMAL_CONFIG, "enable_adaptive_refinement": True, "enable_gap_detection": True},
        test_query="Complete troubleshooting guide for SYST-011 system alarm",
        expected_behaviors=["gaps identified", "refinement queries", "confidence improved"],
        category="Iteration Control"
    ),

    "entropy_halting": ModalityConfig(
        name="UALA Entropy Halting",
        description="Entropy-based confident halting decision",
        feature_flags={**MINIMAL_CONFIG, "enable_entropy_halting": True},
        test_query="What is the standard mastering procedure for FANUC robots?",
        expected_behaviors=["entropy monitored", "halt decision", "iteration saved"],
        category="Iteration Control"
    ),

    "iteration_bandit": ModalityConfig(
        name="UCB Iteration Bandit",
        description="Multi-armed bandit for action selection",
        feature_flags={**MINIMAL_CONFIG, "enable_iteration_bandit": True},
        test_query="How to optimize cycle time for pick and place operations?",
        expected_behaviors=["UCB scores", "action selected", "reward recorded"],
        category="Iteration Control"
    ),

    # ========== 8. PLANNING ==========
    "dynamic_planning": ModalityConfig(
        name="AIME Dynamic Planning",
        description="Dual strategic/tactical planning with task decomposition",
        feature_flags={**MINIMAL_CONFIG, "enable_dynamic_planning": True, "enable_progress_tracking": True},
        test_query="Design a complete commissioning plan for a new robot cell",
        expected_behaviors=["task hierarchy", "tactical actions", "progress tracked"],
        category="Planning"
    ),

    "pre_act_planning": ModalityConfig(
        name="Pre-Act Planning",
        description="Multi-step execution planning before acting",
        feature_flags={**MINIMAL_CONFIG, "enable_pre_act_planning": True},
        test_query="Steps to migrate robot programs from R-30iA to R-30iB controller",
        expected_behaviors=["plan created", "steps sequenced", "parallel identified"],
        category="Planning"
    ),

    # ========== 9. MEMORY SYSTEMS ==========
    "semantic_memory": ModalityConfig(
        name="A-MEM Semantic Memory",
        description="Zettelkasten-style memory with auto-connections",
        feature_flags={**MINIMAL_CONFIG, "enable_semantic_memory": True},
        test_query="What are common causes of robot position drift over time?",
        expected_behaviors=["memory searched", "connections traversed", "context enriched"],
        category="Memory Systems"
    ),

    "semantic_cache": ModalityConfig(
        name="Semantic Query Cache",
        description="Embedding-based cache for similar queries",
        feature_flags={**MINIMAL_CONFIG, "enable_semantic_cache": True, "enable_content_cache": True},
        test_query="How to reset FANUC servo alarms?",
        expected_behaviors=["cache checked", "similarity scored", "hit/miss logged"],
        category="Memory Systems"
    ),

    # ========== 10. DOMAIN KNOWLEDGE ==========
    "domain_corpus": ModalityConfig(
        name="Domain Corpus",
        description="Domain-specific knowledge graph retrieval",
        feature_flags={**MINIMAL_CONFIG, "enable_domain_corpus": True, "enable_embedding_aggregator": True},
        test_query="FANUC SRVO-068 DTERR alarm causes and solutions",
        expected_behaviors=["corpus queried", "entities retrieved", "graph traversed"],
        category="Domain Knowledge"
    ),

    "technical_docs": ModalityConfig(
        name="Technical Documentation",
        description="PDF extraction tools integration for FANUC manuals",
        feature_flags={**MINIMAL_CONFIG, "enable_technical_docs": True},
        test_query="What does the FANUC maintenance manual say about brake inspection?",
        expected_behaviors=["PDF API called", "context retrieved", "source cited"],
        category="Domain Knowledge"
    ),

    # ========== 11. MULTI-AGENT ==========
    "dylan_agent_skipping": ModalityConfig(
        name="DyLAN Agent Skipping",
        description="Query complexity-based agent skipping",
        feature_flags={**MINIMAL_CONFIG, "enable_dylan_agent_skipping": True, "enable_self_reflection": True},
        test_query="What color is the FANUC teach pendant?",
        expected_behaviors=["complexity classified", "skip decision", "agents skipped"],
        category="Multi-Agent"
    ),

    "contrastive_learning": ModalityConfig(
        name="Contrastive Retriever",
        description="Trial-and-feedback learning from cited sources",
        feature_flags={**MINIMAL_CONFIG, "enable_contrastive_learning": True},
        test_query="Best approaches for robot workcell layout design",
        expected_behaviors=["session recorded", "citations tracked", "weights adjusted"],
        category="Multi-Agent"
    ),
}

# ============================================================================
# TEST EXECUTION
# ============================================================================

def get_preset_for_modality(modality_name: str) -> str:
    """Determine the best preset that enables the modality feature."""
    # Map modalities to the preset that enables them
    PRESET_MAP = {
        # These are in minimal/balanced
        "query_analysis": "minimal",
        "verification": "balanced",
        "semantic_cache": "balanced",
        # These are in enhanced
        "crag_evaluation": "enhanced",
        "self_reflection": "enhanced",
        "context_curation": "enhanced",
        "technical_docs": "enhanced",
        # These are in research
        "hyde": "research",
        "hybrid_reranking": "research",
        "entity_tracking": "research",
        "thought_library": "research",
        "reasoning_dag": "research",
        "dynamic_planning": "research",
        "meta_buffer": "research",
        "reasoning_composer": "research",
        "adaptive_refinement": "research",
        "entropy_halting": "research",
        "iteration_bandit": "research",
        "semantic_memory": "research",
        "flare_retrieval": "research",
        "query_tree": "research",
        "information_bottleneck": "research",
        "contrastive_learning": "research",
        "dylan_agent_skipping": "research",
        # These are in full
        "ragas": "full",
        "cross_encoder": "full",
        "pre_act_planning": "full",
        "domain_corpus": "full",
        "actor_factory": "full",
    }
    return PRESET_MAP.get(modality_name, "research")

async def test_modality(
    session: aiohttp.ClientSession,
    modality: ModalityConfig,
    verbose: bool = True
) -> Dict[str, Any]:
    """Test a single modality and return results."""

    start_time = time.time()
    timestamp = int(time.time() * 1000) % 100000

    # Get preset that enables this modality
    preset = get_preset_for_modality(modality.name.lower().replace(" ", "_").replace("-", "_"))

    # Build request - API supports limited feature overrides
    request_body = {
        'query': f"{modality.test_query} [modality-test-{timestamp}]",
        'preset': preset,
        'max_iterations': 3,
        'max_sources': 10,
    }

    # Add supported feature overrides
    feature_key = None
    for k, v in modality.feature_flags.items():
        if v and k.startswith("enable_"):
            feature_key = k
            # Only these overrides are supported by the API
            if k in ['enable_hyde', 'enable_hybrid_reranking', 'enable_ragas',
                     'enable_entity_tracking', 'enable_thought_library',
                     'enable_pre_act_planning', 'enable_parallel_execution']:
                request_body[k] = v

    if verbose:
        print(f"\n{'='*70}")
        print(f"MODALITY: {modality.name}")
        print(f"Category: {modality.category}")
        print(f"Preset: {preset}")
        print(f"Description: {modality.description}")
        print(f"{'='*70}")
        print(f"Test Query: {modality.test_query[:60]}...")
        print(f"Expected: {', '.join(modality.expected_behaviors)}")
        print("-"*70)

    try:
        async with session.post(
            'http://localhost:8001/api/v1/search/universal',
            json=request_body,
            timeout=aiohttp.ClientTimeout(total=180)
        ) as resp:
            data = await resp.json()
            elapsed = time.time() - start_time

            if data.get('success'):
                d = data.get('data', {})
                meta = data.get('meta', {})

                answer = d.get('synthesized_context') or d.get('answer', '')
                confidence = d.get('confidence_score', 0)
                sources = d.get('sources', [])
                trace = d.get('search_trace', [])
                enhancement = meta.get('enhancement_metadata', {})
                features_used = enhancement.get('features_used', [])

                # Check if expected features were activated
                feature_key = list(modality.feature_flags.keys())[0]
                for k, v in modality.feature_flags.items():
                    if v and k != "enable_query_analysis" and k != "enable_scratchpad":
                        feature_key = k.replace("enable_", "")
                        break

                feature_activated = feature_key in features_used or any(
                    feature_key.replace("_", "") in f.replace("_", "")
                    for f in features_used
                )

                result = {
                    'success': True,
                    'modality': modality.name,
                    'category': modality.category,
                    'confidence': confidence,
                    'source_count': len(sources),
                    'answer_length': len(answer),
                    'execution_time_s': elapsed,
                    'features_used': features_used,
                    'feature_activated': feature_activated,
                    'trace_steps': len(trace),
                    'answer_preview': answer[:300] if answer else '',
                }

                if verbose:
                    status = "✓" if feature_activated else "⚠"
                    print(f"{status} Feature Activated: {feature_activated}")
                    print(f"✓ Features Used: {features_used}")
                    print(f"✓ Confidence: {confidence*100:.0f}%")
                    print(f"✓ Sources: {len(sources)}")
                    print(f"✓ Answer Length: {len(answer)} chars")
                    print(f"✓ Time: {elapsed:.1f}s")
                    if not feature_activated:
                        print(f"  NOTE: Expected feature '{feature_key}' not in features_used")

                return result
            else:
                if verbose:
                    print(f"✗ ERROR: {data.get('errors')}")
                return {
                    'success': False,
                    'modality': modality.name,
                    'category': modality.category,
                    'error': str(data.get('errors', 'Unknown'))
                }

    except asyncio.TimeoutError:
        if verbose:
            print(f"✗ TIMEOUT after 180s")
        return {
            'success': False,
            'modality': modality.name,
            'category': modality.category,
            'error': 'Timeout (180s)'
        }
    except Exception as e:
        if verbose:
            print(f"✗ EXCEPTION: {e}")
        return {
            'success': False,
            'modality': modality.name,
            'category': modality.category,
            'error': str(e)
        }

async def run_all_modality_tests(
    modalities: Optional[List[str]] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run tests for specified modalities or all."""

    if modalities is None:
        modalities = list(MODALITIES.keys())

    print(f"\n{'='*70}")
    print(f"AGENTIC MODALITY TEST SUITE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"Testing {len(modalities)} modalities")

    results = []
    async with aiohttp.ClientSession() as session:
        for i, mod_name in enumerate(modalities, 1):
            if mod_name not in MODALITIES:
                print(f"\n[{i}/{len(modalities)}] Unknown modality: {mod_name}")
                continue

            modality = MODALITIES[mod_name]
            print(f"\n[{i}/{len(modalities)}] Testing: {modality.name}")

            result = await test_modality(session, modality, verbose)
            results.append(result)

    return results

def print_modality_summary(results: List[Dict[str, Any]]) -> None:
    """Print comprehensive modality test summary."""

    print(f"\n{'='*70}")
    print("MODALITY TEST SUMMARY")
    print(f"{'='*70}")

    # Group by category
    categories = {}
    for r in results:
        cat = r.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    # Print by category
    for cat, cat_results in sorted(categories.items()):
        print(f"\n{cat}:")
        print("-" * 50)

        for r in cat_results:
            name = r.get('modality', 'Unknown')
            if r.get('success'):
                activated = "✓" if r.get('feature_activated') else "⚠"
                conf = r.get('confidence', 0) * 100
                time_s = r.get('execution_time_s', 0)
                print(f"  {activated} {name:<30} {conf:>4.0f}% conf  {time_s:>5.1f}s")
            else:
                print(f"  ✗ {name:<30} ERROR: {r.get('error', 'Unknown')[:30]}")

    # Overall stats
    successful = [r for r in results if r.get('success')]
    activated = [r for r in successful if r.get('feature_activated')]

    print(f"\n{'='*70}")
    print(f"TOTALS:")
    print(f"  Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.0f}%)")
    if len(successful) > 0:
        print(f"  Features Activated: {len(activated)}/{len(successful)} ({len(activated)/len(successful)*100:.0f}% of successful)")

    if successful:
        avg_conf = sum(r['confidence'] for r in successful) / len(successful)
        avg_time = sum(r['execution_time_s'] for r in successful) / len(successful)
        print(f"  Avg Confidence: {avg_conf*100:.0f}%")
        print(f"  Avg Time: {avg_time:.1f}s")

def list_modalities():
    """List all available modalities."""
    print("\nAvailable Modalities:")
    print("=" * 70)

    categories = {}
    for name, mod in MODALITIES.items():
        if mod.category not in categories:
            categories[mod.category] = []
        categories[mod.category].append((name, mod))

    for cat, mods in sorted(categories.items()):
        print(f"\n{cat}:")
        for name, mod in mods:
            print(f"  {name:<25} - {mod.description[:45]}")

def main():
    parser = argparse.ArgumentParser(description='Agentic Modality Test Suite')
    parser.add_argument('--modality', '-m', type=str, default=None,
                        help='Specific modality to test (comma-separated for multiple)')
    parser.add_argument('--category', '-c', type=str, default=None,
                        help='Test all modalities in a category')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available modalities')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for detailed results')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if args.list:
        list_modalities()
        return

    modalities = None
    if args.modality:
        modalities = [m.strip() for m in args.modality.split(',')]
    elif args.category:
        modalities = [name for name, mod in MODALITIES.items()
                     if mod.category.lower() == args.category.lower()]
        if not modalities:
            print(f"No modalities found for category: {args.category}")
            print("Available categories:", set(m.category for m in MODALITIES.values()))
            return

    results = asyncio.run(run_all_modality_tests(
        modalities=modalities,
        verbose=not args.quiet
    ))

    print_modality_summary(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

if __name__ == '__main__':
    main()
