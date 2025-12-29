#!/usr/bin/env python3
"""
Test RESEARCH Preset Features: Dynamic Planning + Graph Cache

This script specifically tests the Layer 4 features that are now included
in the RESEARCH preset:
- enable_dynamic_planning
- enable_progress_tracking
- enable_graph_cache
- enable_prefetching

Usage:
    python test_research_preset.py
    python test_research_preset.py --full  # Run with actual LLM calls
"""

import asyncio
import sys
import os
import logging
import time
import json

sys.path.insert(0, '/home/sparkone/sdd/Recovery_Bot/memOS/server')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_research_preset")

# Test results tracking
results = {
    'passed': [],
    'failed': [],
    'skipped': []
}


def log_result(test_name: str, passed: bool, message: str = ""):
    """Log test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if message:
        print(f"       {message}")
    if passed:
        results['passed'].append(test_name)
    else:
        results['failed'].append(test_name)


async def test_1_preset_config():
    """Test 1: Verify RESEARCH preset has expected features"""
    print("\n" + "=" * 70)
    print("TEST 1: RESEARCH Preset Configuration")
    print("=" * 70)

    try:
        from agentic import OrchestratorPreset, PRESET_CONFIGS

        research_config = PRESET_CONFIGS[OrchestratorPreset.RESEARCH]

        # Check Layer 4 features
        checks = {
            'enable_dynamic_planning': research_config.enable_dynamic_planning,
            'enable_progress_tracking': research_config.enable_progress_tracking,
            'enable_graph_cache': research_config.enable_graph_cache,
            'enable_prefetching': research_config.enable_prefetching,
        }

        print(f"\nRESEARCH Preset Layer 4 Features:")
        all_enabled = True
        for feature, enabled in checks.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {feature}: {enabled}")
            if not enabled:
                all_enabled = False

        # Count total enabled features
        enabled_count = sum(1 for attr in dir(research_config)
                          if attr.startswith('enable_') and getattr(research_config, attr))
        print(f"\n  Total enabled features: {enabled_count}")

        log_result("RESEARCH preset Layer 4 features", all_enabled)
        return all_enabled

    except Exception as e:
        log_result("RESEARCH preset Layer 4 features", False, str(e))
        return False


async def test_2_full_preset_config():
    """Test 2: Verify FULL preset has all features including multi-agent"""
    print("\n" + "=" * 70)
    print("TEST 2: FULL Preset Configuration")
    print("=" * 70)

    try:
        from agentic import OrchestratorPreset, PRESET_CONFIGS

        full_config = PRESET_CONFIGS[OrchestratorPreset.FULL]

        # Check Layer 4 features
        checks = {
            'enable_dynamic_planning': full_config.enable_dynamic_planning,
            'enable_progress_tracking': full_config.enable_progress_tracking,
            'enable_graph_cache': full_config.enable_graph_cache,
            'enable_prefetching': full_config.enable_prefetching,
            'enable_actor_factory': full_config.enable_actor_factory,
            'enable_multi_agent': full_config.enable_multi_agent,
        }

        print(f"\nFULL Preset Layer 4 Features:")
        all_enabled = True
        for feature, enabled in checks.items():
            status = "✅" if enabled else "❌"
            print(f"  {status} {feature}: {enabled}")
            if not enabled:
                all_enabled = False

        # Count total enabled features
        enabled_count = sum(1 for attr in dir(full_config)
                          if attr.startswith('enable_') and getattr(full_config, attr))
        print(f"\n  Total enabled features: {enabled_count}")

        log_result("FULL preset Layer 4 features", all_enabled)
        return all_enabled

    except Exception as e:
        log_result("FULL preset Layer 4 features", False, str(e))
        return False


async def test_3_dynamic_planner_import():
    """Test 3: Verify DynamicPlanner can be imported and instantiated"""
    print("\n" + "=" * 70)
    print("TEST 3: DynamicPlanner Import & Instantiation")
    print("=" * 70)

    try:
        from agentic.dynamic_planner import DynamicPlanner, TaskNode, TaskStatus

        planner = DynamicPlanner()
        print(f"\n  DynamicPlanner created successfully")
        print(f"  Ollama URL: {planner.ollama_url}")
        print(f"  Planning Model: {planner.planning_model}")

        log_result("DynamicPlanner import & instantiation", True)
        return True

    except Exception as e:
        log_result("DynamicPlanner import & instantiation", False, str(e))
        return False


async def test_4_graph_cache_import():
    """Test 4: Verify GraphCacheIntegration can be imported and instantiated"""
    print("\n" + "=" * 70)
    print("TEST 4: GraphCacheIntegration Import & Instantiation")
    print("=" * 70)

    try:
        from agentic.graph_cache_integration import (
            GraphCacheIntegration,
            get_graph_cache_integration,
            WorkflowContext
        )
        from agentic.agent_step_graph import AgentStepGraph, AgentType
        from agentic.scratchpad_cache import ScratchpadCache

        # Create instance
        graph_cache = get_graph_cache_integration()
        print(f"\n  GraphCacheIntegration created successfully")
        print(f"  Active workflows: {len(graph_cache.active_workflows)}")
        print(f"  Stats: {json.dumps(graph_cache.stats, indent=4)}")

        # Test workflow lifecycle
        workflow_id = "test_workflow_1"
        await graph_cache.start_workflow(workflow_id, "Test query")
        print(f"\n  Started workflow: {workflow_id}")

        # Check if workflow is tracked
        if workflow_id in graph_cache.active_workflows:
            print(f"  Workflow tracked in active_workflows: ✅")

        # End workflow
        end_stats = await graph_cache.end_workflow(workflow_id, success=True)
        print(f"  Ended workflow: {end_stats}")

        log_result("GraphCacheIntegration import & instantiation", True)
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("GraphCacheIntegration import & instantiation", False, str(e))
        return False


async def test_5_orchestrator_initialization():
    """Test 5: Verify UniversalOrchestrator initializes with RESEARCH preset"""
    print("\n" + "=" * 70)
    print("TEST 5: UniversalOrchestrator Initialization (RESEARCH)")
    print("=" * 70)

    try:
        from agentic import UniversalOrchestrator, OrchestratorPreset

        orchestrator = UniversalOrchestrator(
            preset=OrchestratorPreset.RESEARCH,
            db_path="/home/sparkone/sdd/Recovery_Bot/memOS/data"
        )

        print(f"\n  UniversalOrchestrator created with RESEARCH preset")
        print(f"  Dynamic planning enabled: {orchestrator.config.enable_dynamic_planning}")
        print(f"  Progress tracking enabled: {orchestrator.config.enable_progress_tracking}")
        print(f"  Graph cache enabled: {orchestrator.config.enable_graph_cache}")
        print(f"  Prefetching enabled: {orchestrator.config.enable_prefetching}")

        # Initialize
        await orchestrator.initialize()
        print(f"\n  Orchestrator initialized successfully")

        # Check internal components
        if orchestrator.config.enable_dynamic_planning:
            planner = orchestrator._get_dynamic_planner()
            print(f"  DynamicPlanner accessible: ✅")

        if orchestrator.config.enable_graph_cache:
            graph_cache = orchestrator._get_graph_cache()
            print(f"  GraphCacheIntegration accessible: ✅")

        log_result("UniversalOrchestrator RESEARCH initialization", True)
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("UniversalOrchestrator RESEARCH initialization", False, str(e))
        return False


async def test_6_dynamic_planning_execution(run_full: bool = False):
    """Test 6: Execute a search with dynamic planning enabled"""
    print("\n" + "=" * 70)
    print("TEST 6: Dynamic Planning Execution")
    print("=" * 70)

    if not run_full:
        print("\n  Skipping LLM execution (use --full to run)")
        results['skipped'].append("Dynamic planning execution")
        return True

    try:
        from agentic.dynamic_planner import DynamicPlanner

        planner = DynamicPlanner()
        goal = "Compare Python vs Go for building microservices"

        print(f"\n  Goal: {goal}")
        print(f"  Running initial decomposition...")

        start = time.time()
        output = await planner.initial_decomposition(goal)
        duration = time.time() - start

        print(f"\n  Completed in {duration:.1f}s")
        print(f"  Tasks created: {planner.get_stats()['tasks_created']}")
        print(f"  Task hierarchy:")
        print(planner.render_markdown())

        if output.tactical:
            print(f"\n  Next action: {output.tactical.action_type.value}")
            print(f"  Description: {output.tactical.description}")

        success = output is not None and planner.get_stats()['tasks_created'] > 0
        log_result("Dynamic planning execution", success)
        return success

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Dynamic planning execution", False, str(e))
        return False


async def test_7_graph_cache_workflow(run_full: bool = False):
    """Test 7: Execute graph cache workflow tracking"""
    print("\n" + "=" * 70)
    print("TEST 7: Graph Cache Workflow")
    print("=" * 70)

    try:
        from agentic.graph_cache_integration import get_graph_cache_integration
        from agentic.agent_step_graph import AgentType

        graph_cache = get_graph_cache_integration()
        workflow_id = "test_workflow_full"

        # Start workflow
        await graph_cache.start_workflow(workflow_id, "Test research query")
        print(f"\n  Started workflow: {workflow_id}")

        # Simulate agent transitions
        agents = [AgentType.ANALYZER, AgentType.PLANNER, AgentType.SEARCHER]
        scratchpad_state = {'mission': 'test', 'findings': []}

        for agent in agents:
            cached = await graph_cache.before_agent_call(workflow_id, agent, scratchpad_state)
            print(f"  Before {agent.value}: cache_hit={cached is not None}")

            # Simulate some processing
            await asyncio.sleep(0.1)

            await graph_cache.after_agent_call(
                workflow_id, agent,
                {'result': f'{agent.value}_output'},
                duration_ms=100.0
            )
            print(f"  After {agent.value}: recorded")

        # End workflow
        end_stats = await graph_cache.end_workflow(workflow_id, success=True)
        print(f"\n  Workflow stats: {json.dumps(end_stats, indent=4)}")

        # Check graph stats
        graph_stats = graph_cache.get_comprehensive_stats()
        print(f"\n  Graph cache stats:")
        print(f"    Workflows completed: {graph_stats.get('workflows_completed', 0)}")
        print(f"    Cache hits: {graph_stats.get('total_cache_hits', 0)}")

        log_result("Graph cache workflow", True)
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Graph cache workflow", False, str(e))
        return False


async def test_8_full_research_search(run_full: bool = False):
    """Test 8: Full search with RESEARCH preset"""
    print("\n" + "=" * 70)
    print("TEST 8: Full RESEARCH Preset Search")
    print("=" * 70)

    if not run_full:
        print("\n  Skipping full search (use --full to run)")
        results['skipped'].append("Full RESEARCH preset search")
        return True

    try:
        from agentic import UniversalOrchestrator, OrchestratorPreset
        from agentic.models import SearchRequest

        orchestrator = UniversalOrchestrator(
            preset=OrchestratorPreset.RESEARCH,
            db_path="/home/sparkone/sdd/Recovery_Bot/memOS/data"
        )
        await orchestrator.initialize()

        request = SearchRequest(
            query="What are the key differences between FastAPI and Flask?",
            max_iterations=3
        )

        print(f"\n  Query: {request.query}")
        print(f"  Preset: RESEARCH")
        print(f"  Max iterations: {request.max_iterations}")
        print(f"\n  Executing search...")

        start = time.time()
        response = await orchestrator.search(request)
        duration = time.time() - start

        print(f"\n  Completed in {duration:.1f}s")
        print(f"  Success: {response.success}")
        print(f"  Confidence: {response.data.confidence if response.data else 'N/A'}")

        # Check if features were used
        if response.meta:
            features_used = response.meta.additional_info.get('features_used', [])
            print(f"\n  Features used:")
            for feature in features_used:
                print(f"    - {feature}")

            dynamic_planning_used = 'dynamic_planning' in features_used
            print(f"\n  Dynamic planning activated: {'✅' if dynamic_planning_used else '❌'}")

        log_result("Full RESEARCH preset search", response.success)
        return response.success

    except Exception as e:
        import traceback
        traceback.print_exc()
        log_result("Full RESEARCH preset search", False, str(e))
        return False


async def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("RESEARCH PRESET FEATURE TESTS")
    print("Testing: Dynamic Planning + Graph Cache")
    print("=" * 70)

    run_full = '--full' in sys.argv

    if run_full:
        print("\n🔧 Running in FULL mode (includes LLM calls)")
    else:
        print("\n⚡ Running in QUICK mode (no LLM calls)")
        print("   Use --full for complete testing")

    # Run tests
    await test_1_preset_config()
    await test_2_full_preset_config()
    await test_3_dynamic_planner_import()
    await test_4_graph_cache_import()
    await test_5_orchestrator_initialization()
    await test_6_dynamic_planning_execution(run_full)
    await test_7_graph_cache_workflow(run_full)
    await test_8_full_research_search(run_full)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"\n  ✅ Passed: {len(results['passed'])}")
    print(f"  ❌ Failed: {len(results['failed'])}")
    print(f"  ⏭️  Skipped: {len(results['skipped'])}")

    if results['failed']:
        print(f"\n  Failed tests:")
        for test in results['failed']:
            print(f"    - {test}")

    print("\n" + "=" * 70)

    return len(results['failed']) == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
