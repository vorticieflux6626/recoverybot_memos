#!/usr/bin/env python3
"""
Observability Integration Test - Industrial Troubleshooting Queries

Tests the full observability stack (P0-P3) with real integrated systems
troubleshooting queries:
1. Plastic injection mold + FANUC interface issues
2. End effector assembly + FANUC troubles

This validates:
- Decision logging (P0)
- Context flow tracking (P0)
- LLM call logging (P1)
- Scratchpad observation (P1)
- Technician log generation (P1)
- Confidence breakdown (P2)
- Dashboard aggregation (P3)

Usage:
    cd /home/sparkone/sdd/Recovery_Bot/memOS/server
    source venv/bin/activate
    python tests/test_observability_integrated_systems.py

Created: 2026-01-02
"""

import asyncio
import logging
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("test_observability")

# Add parent to path
sys.path.insert(0, '/home/sparkone/sdd/Recovery_Bot/memOS/server')

# Import observability modules
from agentic import (
    # P0: Decision and Context
    DecisionLogger, get_decision_logger, DecisionType, AgentName,
    ContextFlowTracker, get_context_tracker, PipelineStage, ContextType,
    # P1: LLM, Scratchpad, Technician Log
    LLMCallLogger, get_llm_logger,
    ScratchpadObserver, get_scratchpad_observer, ScratchpadOperation,
    TechnicianLogBuilder, get_log_builder, store_technician_log, get_technician_log,
    # P2: Confidence
    ConfidenceLogger, get_confidence_logger,
    # P3: Dashboard
    ObservabilityDashboard, ObservabilityAggregator, get_observability_dashboard,
)


# Industrial troubleshooting queries
INTEGRATED_SYSTEMS_QUERIES = [
    {
        "id": "injection-mold-fanuc-1",
        "query": "FANUC robot SRVO-023 servo alarm during injection mold part extraction - robot stops mid-cycle when removing parts from Husky HyPET mold",
        "category": "injection_mold_fanuc",
        "complexity": "high",
        "systems": ["FANUC robot", "Husky injection mold", "servo system"],
    },
    {
        "id": "end-effector-fanuc-1",
        "query": "FANUC M-20iA end effector assembly gripper not closing fully - SRVO-062 collision detection triggers during part pick on assembly line",
        "category": "end_effector_fanuc",
        "complexity": "high",
        "systems": ["FANUC M-20iA", "pneumatic gripper", "collision detection"],
    },
    {
        "id": "injection-mold-fanuc-2",
        "query": "Communication timeout between FANUC R-30iB controller and Arburg injection molding machine - Euromap 67 interface errors",
        "category": "injection_mold_fanuc",
        "complexity": "medium",
        "systems": ["FANUC R-30iB", "Arburg IMM", "Euromap 67"],
    },
    {
        "id": "end-effector-fanuc-2",
        "query": "FANUC LR Mate 200iD vacuum end effector losing suction during assembly - MOTN-063 position error after pick",
        "category": "end_effector_fanuc",
        "complexity": "high",
        "systems": ["FANUC LR Mate", "vacuum gripper", "motion control"],
    },
]


async def simulate_pipeline_with_observability(query_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate running a query through the pipeline with full observability.

    In production, this would call UniversalOrchestrator.search_with_events().
    Here we simulate the pipeline stages to demonstrate observability integration.
    """
    request_id = f"obs-{query_info['id']}-{uuid.uuid4().hex[:6]}"
    query = query_info["query"]
    start_time = time.time()

    logger.info(f"\n{'='*80}")
    logger.info(f"REQUEST: {request_id}")
    logger.info(f"QUERY: {query[:80]}...")
    logger.info(f"{'='*80}\n")

    # Initialize all observability components
    decision_logger = get_decision_logger(request_id)
    context_tracker = get_context_tracker(request_id)
    llm_logger = get_llm_logger(request_id, verbose=False)
    scratchpad_observer = get_scratchpad_observer(request_id)
    confidence_logger = get_confidence_logger(request_id)
    tech_log_builder = get_log_builder(request_id)

    # Initialize aggregator
    aggregator = ObservabilityAggregator(request_id, query, preset="enhanced")

    try:
        # ============================================================
        # STAGE 1: Query Classification (P0 Decision Logging)
        # ============================================================
        logger.info("[Stage 1] Query Classification")

        await decision_logger.log_decision(
            agent_name=AgentName.QUERY_CLASSIFIER.value,
            decision_type=DecisionType.CLASSIFICATION.value,
            decision_made="agentic_search",
            reasoning=f"Complex integrated systems query involving {', '.join(query_info['systems'])}",
            confidence=0.92,
            alternatives=["direct_answer", "web_search_only"],
            metadata={"complexity": query_info["complexity"], "category": query_info["category"]}
        )

        context_tracker.record_transfer(
            source=PipelineStage.INPUT.value,
            target=PipelineStage.ANALYZER.value,
            content=query,
            context_type=ContextType.QUERY.value
        )

        tech_log_builder.set_query(query[:100])
        tech_log_builder.add_reasoning_step(f"Identified as {query_info['category']} troubleshooting query")

        await asyncio.sleep(0.1)  # Simulate processing

        # ============================================================
        # STAGE 2: Query Analysis (P1 LLM Logging)
        # ============================================================
        logger.info("[Stage 2] Query Analysis")

        analysis_prompt = f"Analyze this industrial troubleshooting query: {query}"
        llm_call = llm_logger.track_call_sync(
            agent_name="analyzer",
            operation="analysis",
            model="qwen3:8b",
            prompt=analysis_prompt,
            prompt_template="query_analysis"
        )

        # Simulate LLM response
        await asyncio.sleep(0.2)
        analysis_response = f"Analysis: {query_info['systems']} integration issue. Error codes indicate servo/motion problems."
        llm_logger.finalize_call_sync(llm_call, response=analysis_response, parse_success=True)

        context_tracker.record_transfer(
            source=PipelineStage.ANALYZER.value,
            target=PipelineStage.PLANNER.value,
            content=analysis_response,
            context_type=ContextType.QUERY_ANALYSIS.value
        )

        await decision_logger.log_decision(
            agent_name=AgentName.ANALYZER.value,
            decision_type=DecisionType.EVALUATION.value,
            decision_made=f"complexity={query_info['complexity']}",
            reasoning="Multiple integrated systems with error codes",
            confidence=0.88,
            metadata={"systems_detected": query_info["systems"]}
        )

        tech_log_builder.add_reasoning_step(f"Detected systems: {', '.join(query_info['systems'])}")

        # ============================================================
        # STAGE 3: Web Search (P1 Scratchpad Observation)
        # ============================================================
        logger.info("[Stage 3] Web Search + Scratchpad")

        await scratchpad_observer.on_question_added(
            agent="planner",
            question_id="q1",
            question_text=f"What causes {query_info['systems'][0]} issues?",
            priority="high"
        )

        # Simulate search
        await asyncio.sleep(0.3)

        # Add findings to scratchpad
        await scratchpad_observer.on_finding_added(
            agent="searcher",
            finding_id="f1",
            content_preview=f"FANUC servo alarms often relate to encoder or cable issues...",
            confidence=0.85,
            source_count=3
        )

        await scratchpad_observer.on_finding_added(
            agent="searcher",
            finding_id="f2",
            content_preview=f"Integration timing between robot and peripheral equipment is critical...",
            confidence=0.78,
            source_count=2
        )

        context_tracker.record_transfer(
            source=PipelineStage.SEARCHER.value,
            target=PipelineStage.VERIFIER.value,
            content=["finding_1", "finding_2"],
            context_type=ContextType.SEARCH_RESULTS.value
        )

        tech_log_builder.add_source(
            title="FANUC Servo Troubleshooting Guide",
            url="https://fanuc.com/servo-guide",
            relevance=0.92,
            source_type="documentation",
            trust_level="official"
        )
        tech_log_builder.add_source(
            title="Industrial Robot Integration Forum",
            url="https://robotforum.com/thread/12345",
            relevance=0.75,
            source_type="forum",
            trust_level="community"
        )

        # ============================================================
        # STAGE 4: Verification (P0/P1)
        # ============================================================
        logger.info("[Stage 4] Claim Verification")

        await decision_logger.log_decision(
            agent_name=AgentName.VERIFIER.value,
            decision_type=DecisionType.EVALUATION.value,
            decision_made="claims_verified",
            reasoning="Cross-referenced with FANUC documentation",
            confidence=0.82,
            metadata={"verified_claims": 2, "unverified_claims": 0}
        )

        await scratchpad_observer.on_question_answered(
            agent="verifier",
            question_id="q1",
            question_text=f"What causes {query_info['systems'][0]} issues?",
            old_status="open",
            new_status="answered"
        )

        tech_log_builder.add_reasoning_step("Verified findings against official documentation")

        # ============================================================
        # STAGE 5: Synthesis (P1 LLM Logging)
        # ============================================================
        logger.info("[Stage 5] Answer Synthesis")

        synthesis_prompt = f"Synthesize answer for: {query}\nFindings: [f1, f2]"
        synth_call = llm_logger.track_call_sync(
            agent_name="synthesizer",
            operation="synthesis",
            model="qwen3:8b",
            prompt=synthesis_prompt,
            prompt_template="synthesis_with_content"
        )

        await asyncio.sleep(0.4)
        synthesis_response = f"""
Based on your query about {query_info['systems'][0]} issues:

1. **Root Cause Analysis**: The {query_info['systems'][0]} alarm indicates...
2. **Troubleshooting Steps**:
   - Check encoder connections
   - Verify servo motor parameters
   - Review integration timing with {query_info['systems'][1] if len(query_info['systems']) > 1 else 'peripheral'}
3. **Safety Precaution**: Always engage E-stop before physical inspection

This is a common issue when integrating {query_info['category'].replace('_', ' ')} systems.
"""
        llm_logger.finalize_call_sync(synth_call, response=synthesis_response, parse_success=True)

        context_tracker.record_transfer(
            source=PipelineStage.SYNTHESIZER.value,
            target="output",
            content=synthesis_response,
            context_type=ContextType.SYNTHESIS.value
        )

        tech_log_builder.add_reasoning_step("Synthesized answer from verified findings")
        tech_log_builder.set_recommendation(
            "Check encoder connections and verify servo parameters. "
            "Review integration timing if issue persists."
        )
        tech_log_builder.add_safety_warning("Engage E-stop before any physical inspection")
        tech_log_builder.add_prerequisite("Robot in manual mode")
        tech_log_builder.add_prerequisite("Mold/peripheral in safe position")

        # ============================================================
        # STAGE 6: Confidence Calculation (P2)
        # ============================================================
        logger.info("[Stage 6] Confidence Calculation")

        confidence_breakdown = confidence_logger.calculate_and_log_sync(
            verification_score=0.82,
            source_diversity_score=0.70,
            content_depth_score=0.75,
            synthesis_quality_score=0.80
        )

        tech_log_builder.set_confidence(
            confidence_breakdown.final_confidence,
            [(s.signal, s.raw_score) for s in confidence_breakdown.signals]
        )

        # Add uncertainty declarations
        tech_log_builder.add_uncertainty("Model-specific variations may apply")
        tech_log_builder.add_uncertainty("Controller firmware version differences")
        tech_log_builder.add_info_needed("Specific robot model and controller version")
        tech_log_builder.add_info_needed("Exact error timing in cycle")

        # Extract error codes
        for system in query_info['systems']:
            if 'FANUC' in system.upper():
                tech_log_builder.add_equipment(system)

        # Check for error codes in query
        import re
        error_codes = re.findall(r'[A-Z]+-\d{3}', query)
        for code in error_codes:
            tech_log_builder.add_error_code(code)

        # ============================================================
        # STAGE 7: Build Technician Log (P1)
        # ============================================================
        logger.info("[Stage 7] Building Technician Log")

        tech_log = tech_log_builder.build()
        store_technician_log(tech_log)

        # ============================================================
        # STAGE 8: Aggregate Observability (P3)
        # ============================================================
        logger.info("[Stage 8] Aggregating Observability")

        duration_ms = int((time.time() - start_time) * 1000)

        aggregator.add_decisions(decision_logger.decisions)
        aggregator.add_context_flow(context_tracker.get_flow_summary())
        aggregator.add_llm_calls(llm_logger.get_call_summary())
        aggregator.add_scratchpad(scratchpad_observer.get_change_summary())
        aggregator.add_confidence(confidence_breakdown)
        aggregator.add_technician_log(tech_log.to_markdown())

        # Track features
        aggregator.add_feature_status("crag_evaluation", True)
        aggregator.add_feature_status("self_reflection", True)
        aggregator.add_feature_status("graph_cache", False, "preset=enhanced")

        obs = aggregator.finalize(success=True, duration_ms=duration_ms)

        # Store in dashboard
        dashboard = get_observability_dashboard()
        dashboard.store_request(obs)

        # ============================================================
        # OUTPUT RESULTS
        # ============================================================
        logger.info(f"\n{'='*80}")
        logger.info("OBSERVABILITY SUMMARY")
        logger.info(f"{'='*80}")

        print(f"\n--- Technician Log ({request_id}) ---")
        print(tech_log.to_markdown())

        print(f"\n--- Observability Summary ---")
        print(f"Request ID: {request_id}")
        print(f"Duration: {duration_ms}ms")
        print(f"Confidence: {confidence_breakdown.final_confidence:.0%} ({confidence_breakdown.confidence_level})")
        print(f"Decisions logged: {len(decision_logger.decisions)}")
        print(f"Context transfers: {len(context_tracker.transfers)}")
        print(f"LLM calls: {len(llm_logger.calls)}")
        print(f"Scratchpad changes: {len(scratchpad_observer.changes)}")

        return {
            "request_id": request_id,
            "success": True,
            "duration_ms": duration_ms,
            "confidence": confidence_breakdown.final_confidence,
            "technician_log": tech_log.to_markdown(),
            "observability": obs.to_dict()
        }

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        duration_ms = int((time.time() - start_time) * 1000)
        obs = aggregator.finalize(success=False, duration_ms=duration_ms, error=str(e))
        dashboard = get_observability_dashboard()
        dashboard.store_request(obs)
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
            "duration_ms": duration_ms
        }


async def main():
    """Run all integrated systems tests."""
    logger.info("\n" + "="*80)
    logger.info("OBSERVABILITY INTEGRATION TEST - INDUSTRIAL TROUBLESHOOTING")
    logger.info("="*80 + "\n")

    results = []

    # Run each query
    for i, query_info in enumerate(INTEGRATED_SYSTEMS_QUERIES, 1):
        logger.info(f"\n[TEST {i}/{len(INTEGRATED_SYSTEMS_QUERIES)}] {query_info['id']}")
        result = await simulate_pipeline_with_observability(query_info)
        results.append(result)
        await asyncio.sleep(0.5)  # Brief pause between tests

    # ============================================================
    # DASHBOARD STATISTICS
    # ============================================================
    logger.info("\n" + "="*80)
    logger.info("DASHBOARD AGGREGATE STATISTICS")
    logger.info("="*80 + "\n")

    dashboard = get_observability_dashboard()
    stats = dashboard.get_stats(last_hours=1)

    print(f"\n--- Dashboard Stats (Last Hour) ---")
    print(f"Total Requests: {stats.total_requests}")
    print(f"Successful: {stats.successful_requests}")
    print(f"Failed: {stats.failed_requests}")
    print(f"Avg Duration: {stats.avg_duration_ms:.1f}ms")
    print(f"Avg Confidence: {stats.avg_confidence:.0%}")
    print(f"Confidence Distribution: {stats.confidence_distribution}")
    print(f"Agent Activity: {stats.agent_call_counts}")

    # Test technician audit retrieval
    if results:
        first_request_id = results[0]["request_id"]
        audit = dashboard.get_technician_audit(first_request_id)
        if audit:
            print(f"\n--- Sample Technician Audit ---")
            print(audit[:2000] + "..." if len(audit) > 2000 else audit)

    # Summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY: {successful}/{len(results)} queries processed successfully")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
